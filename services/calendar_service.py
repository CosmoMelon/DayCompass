"""Read public Google Calendar events with API-key authentication."""

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from services import ToolExecutionError


CALENDAR_API = "https://www.googleapis.com/calendar/v3/calendars"
RELATIVE_TIME = re.compile(
    r"^(today|tomorrow|this evening|tomorrow evening)(?:\s+([01]?\d|2[0-3]):([0-5]\d))?$",
    re.IGNORECASE,
)


class CalendarServiceError(ToolExecutionError):
    """A calendar lookup failure safe to report to the assistant."""


class CalendarService:
    def __init__(
        self,
        calendar_id: str,
        api_key: str,
        session: requests.Session | None = None,
        now: Callable[[ZoneInfo], datetime] | None = None,
    ) -> None:
        self._calendar_id = calendar_id.strip()
        self._api_key = api_key.strip()
        self._session = session or requests.Session()
        self._now = now or (lambda zone: datetime.now(zone))
        self._calendar_zone: ZoneInfo | None = None

    def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._calendar_id or not self._api_key:
            raise CalendarServiceError("not_configured", "Calendar integration is not configured.")
        url = f"{CALENDAR_API}/{quote(self._calendar_id, safe='')}/events"
        try:
            response = self._session.get(
                url,
                params=params,
                headers={"X-Goog-Api-Key": self._api_key},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise CalendarServiceError("timeout", "The calendar request timed out. Please try again.") from exc
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code in (401, 403, 404):
                message = "Calendar access failed. Check that the calendar is public and its ID and API key are valid."
                error_code = "access_denied"
            elif code == 429:
                message = "Google Calendar is rate limiting requests. Please try again shortly."
                error_code = "rate_limited"
            else:
                message = "Google Calendar is unavailable. Please try again."
                error_code = "api_failure"
            raise CalendarServiceError(error_code, message) from exc
        except requests.RequestException as exc:
            raise CalendarServiceError("api_failure", "Could not connect to Google Calendar.") from exc
        except ValueError as exc:
            raise CalendarServiceError("api_failure", "Google Calendar returned invalid data.") from exc

        if not isinstance(payload, dict) or "error" in payload:
            raise CalendarServiceError("api_failure", "Google Calendar returned invalid data.")
        return payload

    def _zone(self) -> ZoneInfo:
        if self._calendar_zone is None:
            # events.list is publicly accessible; calendars.get requires OAuth.
            payload = self._get_json({"maxResults": 1, "singleEvents": "true"})
            try:
                self._calendar_zone = ZoneInfo(payload["timeZone"])
            except (KeyError, TypeError, ZoneInfoNotFoundError) as exc:
                raise CalendarServiceError(
                    "api_failure", "The calendar's timezone was unavailable."
                ) from exc
        return self._calendar_zone

    def _boundary(self, value: str, *, is_end: bool, zone: ZoneInfo) -> datetime:
        if not isinstance(value, str) or not value.strip():
            raise CalendarServiceError("invalid_date", "Provide a start and end date or time.")
        value = value.strip()
        relative = RELATIVE_TIME.fullmatch(value)
        if relative:
            label, hour, minute = relative.groups()
            day = self._now(zone).date()
            if label.lower().startswith("tomorrow"):
                day += timedelta(days=1)
            if hour is not None:
                return datetime.combine(day, time(int(hour), int(minute)), zone)
            if "evening" in label.lower():
                if is_end:
                    day += timedelta(days=1)
                    return datetime.combine(day, time.min, zone)
                return datetime.combine(day, time(17), zone)
            if is_end:
                day += timedelta(days=1)
            return datetime.combine(day, time.min, zone)

        try:
            if len(value) == 10:
                day = date.fromisoformat(value)
                if is_end:
                    day += timedelta(days=1)
                return datetime.combine(day, time.min, zone)
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)
        except ValueError as exc:
            raise CalendarServiceError(
                "invalid_date",
                "Use RFC3339 date-times, YYYY-MM-DD, or relative times like 'tomorrow 17:30'.",
            ) from exc

    @staticmethod
    def _event_time(value: dict[str, Any], zone: ZoneInfo) -> tuple[str, bool]:
        if "date" in value:
            local = datetime.combine(date.fromisoformat(value["date"]), time.min, zone)
            return local.isoformat(), True
        parsed = datetime.fromisoformat(value["dateTime"].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            event_zone = ZoneInfo(value["timeZone"]) if value.get("timeZone") else zone
            parsed = parsed.replace(tzinfo=event_zone)
        return parsed.astimezone(zone).isoformat(), False

    def get_calendar_events(self, start_datetime: str, end_datetime: str) -> dict[str, Any]:
        """Return expanded event instances that overlap the requested interval."""
        zone = self._zone()
        start = self._boundary(start_datetime, is_end=False, zone=zone)
        end = self._boundary(end_datetime, is_end=True, zone=zone)
        if end.astimezone(timezone.utc) <= start.astimezone(timezone.utc):
            raise CalendarServiceError("invalid_date", "End time must be after start time.")

        params: dict[str, Any] = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": 250,
        }
        events: list[dict[str, Any]] = []
        seen_tokens: set[str] = set()
        while True:
            payload = self._get_json(params)
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise CalendarServiceError("api_failure", "Google Calendar returned invalid events.")
            for item in items:
                if not isinstance(item, dict) or item.get("status") == "cancelled":
                    continue
                try:
                    start_value = item["start"]
                    end_value = item["end"]
                    event_start, all_day = self._event_time(start_value, zone)
                    event_end, _ = self._event_time(end_value, zone)
                except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError) as exc:
                    raise CalendarServiceError("api_failure", "Google Calendar returned an invalid event.") from exc
                events.append({
                    "title": item.get("summary") or "Untitled event",
                    "start": event_start,
                    "end": event_end,
                    "timezone": zone.key,
                    "all_day": all_day,
                    "blocks_time": item.get("transparency") != "transparent",
                })

            token = payload.get("nextPageToken")
            if not token:
                break
            if not isinstance(token, str) or token in seen_tokens:
                raise CalendarServiceError("api_failure", "Google Calendar pagination failed.")
            seen_tokens.add(token)
            params["pageToken"] = token

        return {
            "source": "Google Calendar",
            "timezone": zone.key,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "events": events,
        }
