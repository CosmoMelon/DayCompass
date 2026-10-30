"""Google Calendar integration tests with mocked HTTP responses."""

import unittest
from datetime import datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import requests

from services.calendar_service import CALENDAR_API, CalendarService, CalendarServiceError
from services.tool_registry import create_default_registry


def response(payload, status=200):
    item = Mock()
    item.json.return_value = payload
    item.status_code = status
    if status >= 400:
        item.raise_for_status.side_effect = requests.HTTPError(response=item)
    return item


class CalendarTests(unittest.TestCase):
    def make_service(self, session):
        return CalendarService(
            "personal@example.com", "test-calendar-key", session,
            now=lambda zone: datetime(2026, 10, 13, 12, tzinfo=zone),
        )

    def test_recurring_instances_pagination_all_day_and_timezone(self):
        session = Mock()
        session.get.side_effect = [
            response({"timeZone": "America/Toronto", "items": []}),
            response({
                "items": [{
                    "summary": "Daily Exercise",
                    "recurringEventId": "exercise-series",
                    "start": {"dateTime": "2026-10-14T17:00:00-04:00"},
                    "end": {"dateTime": "2026-10-14T18:00:00-04:00"},
                }],
                "nextPageToken": "next-page",
            }),
            response({
                "items": [{
                    "summary": "Holiday",
                    "start": {"date": "2026-10-14"},
                    "end": {"date": "2026-10-15"},
                }, {
                    "summary": "Reminder",
                    "start": {"dateTime": "2026-10-14T22:00:00Z"},
                    "end": {"dateTime": "2026-10-14T22:30:00Z"},
                    "transparency": "transparent",
                }],
            }),
        ]
        result = self.make_service(session).get_calendar_events("tomorrow evening", "tomorrow evening")
        self.assertEqual(result["timezone"], "America/Toronto")
        self.assertEqual(result["start"], "2026-10-14T17:00:00-04:00")
        self.assertEqual(result["end"], "2026-10-15T00:00:00-04:00")
        self.assertEqual([event["title"] for event in result["events"]], [
            "Daily Exercise", "Holiday", "Reminder",
        ])
        self.assertEqual(result["events"][0]["start"], "2026-10-14T17:00:00-04:00")
        self.assertTrue(result["events"][1]["all_day"])
        self.assertEqual(result["events"][1]["end"], "2026-10-15T00:00:00-04:00")
        self.assertEqual(result["events"][2]["start"], "2026-10-14T18:00:00-04:00")
        self.assertFalse(result["events"][2]["blocks_time"])
        self.assertEqual(session.get.call_count, 3)
        query = session.get.call_args_list[1]
        self.assertEqual(query.args[0], f"{CALENDAR_API}/personal%40example.com/events")
        self.assertEqual(query.kwargs["params"]["singleEvents"], "true")
        self.assertEqual(query.kwargs["params"]["timeMin"], result["start"])
        self.assertEqual(session.get.call_args_list[2].kwargs["params"]["pageToken"], "next-page")
        self.assertEqual(query.kwargs["headers"], {"X-Goog-Api-Key": "test-calendar-key"})
        self.assertNotIn("key", query.kwargs["params"])

    def test_empty_calendar_and_exact_conflict_interval(self):
        session = Mock()
        session.get.side_effect = [
            response({"timeZone": "America/Toronto"}),
            response({"items": []}),
        ]
        result = self.make_service(session).get_calendar_events("tomorrow 17:30", "tomorrow 17:31")
        self.assertEqual(result["events"], [])
        self.assertEqual(result["start"], "2026-10-14T17:30:00-04:00")
        self.assertEqual(result["end"], "2026-10-14T17:31:00-04:00")

    def test_explicit_offsets_and_event_timezone_are_normalized(self):
        session = Mock()
        session.get.side_effect = [
            response({"timeZone": "America/Toronto"}),
            response({"items": [{
                "summary": "West Coast call",
                "start": {"dateTime": "2026-10-14T14:00:00", "timeZone": "America/Vancouver"},
                "end": {"dateTime": "2026-10-14T15:00:00", "timeZone": "America/Vancouver"},
            }]}),
        ]
        result = self.make_service(session).get_calendar_events(
            "2026-10-14T21:00:00Z", "2026-10-14T22:00:00Z"
        )
        self.assertEqual(result["start"], "2026-10-14T17:00:00-04:00")
        self.assertEqual(result["events"][0]["start"], "2026-10-14T17:00:00-04:00")

    def test_invalid_dates_and_api_failures(self):
        session = Mock()
        session.get.return_value = response({"timeZone": "America/Toronto"})
        service = self.make_service(session)
        with self.assertRaises(CalendarServiceError) as caught:
            service.get_calendar_events("tomorrow 18:00", "tomorrow 17:00")
        self.assertEqual(caught.exception.code, "invalid_date")
        with self.assertRaises(CalendarServiceError) as caught:
            service.get_calendar_events("nonsense", "tomorrow")
        self.assertEqual(caught.exception.code, "invalid_date")

        session.get.return_value = response({}, status=403)
        with self.assertRaises(CalendarServiceError) as caught:
            self.make_service(session).get_calendar_events("today", "today")
        self.assertEqual(caught.exception.code, "access_denied")
        self.assertNotIn("test-calendar-key", str(caught.exception))

        session.get.side_effect = requests.Timeout("slow")
        with self.assertRaises(CalendarServiceError) as caught:
            self.make_service(session).get_calendar_events("today", "today")
        self.assertEqual(caught.exception.code, "timeout")

    def test_registry_only_exposes_calendar_when_enabled(self):
        disabled = create_default_registry()
        self.assertEqual([tool["function"]["name"] for tool in disabled.definitions], ["get_weather"])
        enabled = create_default_registry(
            calendar_enabled=True, calendar_id="public-calendar", calendar_api_key="test-key"
        )
        self.assertEqual([tool["function"]["name"] for tool in enabled.definitions], [
            "get_weather", "get_calendar_events",
        ])
        self.assertEqual(disabled.execute("get_calendar_events", "{}")["error"]["code"], "unknown_tool")


if __name__ == "__main__":
    unittest.main()
