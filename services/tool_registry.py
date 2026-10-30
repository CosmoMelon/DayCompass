"""Register model-callable tools and dispatch their arguments."""

import json
import logging
from collections.abc import Callable
from typing import Any

from services import ToolExecutionError
from services.calendar_service import CalendarService
from services.places_service import PlacesService
from services.weather_service import WeatherService

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., dict[str, Any]]]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._tools[name] = (schema, handler)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [schema for schema, _ in self._tools.values()]

    def execute(self, name: str, arguments: str) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": {"code": "unknown_tool", "message": f"Unknown tool: {name}"}}
        try:
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                raise ValueError("Arguments must be a JSON object")
            result = tool[1](**parsed)
            return {"ok": True, "data": result}
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error": {"code": "invalid_arguments", "message": str(exc)}}
        except ToolExecutionError as exc:
            error = {"code": exc.code, "message": str(exc)}
            if exc.options:
                error["options"] = exc.options
            return {"ok": False, "error": error}
        except Exception:
            logger.exception("Tool %s failed", name)
            return {
                "ok": False,
                "error": {"code": "tool_failed", "message": "The tool could not complete the request."},
            }


def create_default_registry(
    *, weather_enabled: bool = True,
    calendar_enabled: bool = False, calendar_id: str = "", calendar_api_key: str = "",
    places_enabled: bool = False, places_api_key: str = "",
) -> ToolRegistry:
    registry = ToolRegistry()
    if weather_enabled:
        weather = WeatherService()
        registry.register(
            name="get_weather",
            description=(
                "Get current weather or a daily forecast for a city using Open-Meteo. "
                "Use for current or future weather questions, including contextual follow-ups "
                "about temperature, rain, wind, or conditions. Reuse the previous location and "
                "date when the user refers to them; do not use for unrelated questions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City, optionally followed by region or country (e.g. 'Toronto, Canada').",
                    },
                    "date": {
                        "type": "string",
                        "description": (
                            "Optional city-local date: 'today', 'tomorrow', or YYYY-MM-DD. "
                            "Omit for today. For a follow-up, retain the previous date if relevant."
                        ),
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            handler=weather.get_weather,
        )
    if calendar_enabled:
        calendar = CalendarService(calendar_id, calendar_api_key)
        registry.register(
            name="get_calendar_events",
            description=(
                "Read events from the user's public Google Calendar for a time interval. "
                "Use before answering availability or conflict questions, and when planning "
                "activities for a specific time period. Results include recurring instances. "
                "An empty events list means no events were found in the checked interval. "
                "Use alongside get_weather when that tool is enabled and weather matters."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "start_datetime": {
                        "type": "string",
                        "description": (
                            "Inclusive interval start. Use an RFC3339 date-time with offset, "
                            "YYYY-MM-DD, 'today', 'tomorrow', 'this evening', "
                            "'tomorrow evening', or 'tomorrow 17:30' (24-hour time). "
                            "Relative values use the calendar timezone."
                        ),
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": (
                            "Exclusive interval end. Use an RFC3339 date-time with offset or "
                            "a relative value such as 'tomorrow 18:00'. Date-only and "
                            "evening values include their full day or evening."
                        ),
                    },
                },
                "required": ["start_datetime", "end_datetime"],
                "additionalProperties": False,
            },
            handler=calendar.get_calendar_events,
        )
    if places_enabled:
        places = PlacesService(places_api_key)
        registry.register(
            name="search_places",
            description=(
                "Find real venues and points of interest with Google Places Text Search. "
                "Use for named place searches or activity recommendations in a known area. "
                "For 'near me', use a city from conversation context or coordinates; "
                "otherwise ask the user for a location. Do not invent places. "
                "Request ratings or opening hours only when useful to the user's question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language place category or search, e.g. 'badminton courts'.",
                    },
                    "location": {
                        "type": "string",
                        "description": (
                            "Optional city or region, preferably with country when ambiguous. "
                            "Reuse a location established earlier in the conversation."
                        ),
                    },
                    "latitude": {"type": "number", "description": "Optional known center latitude."},
                    "longitude": {"type": "number", "description": "Optional known center longitude."},
                    "radius_meters": {
                        "type": "number",
                        "description": "Optional search bias radius, up to 50000 meters; requires coordinates.",
                    },
                    "max_results": {
                        "type": "integer", "description": "Number of places, default 5, maximum 10."
                    },
                    "include_rating": {
                        "type": "boolean",
                        "description": "Request rating only if the user asks about ratings or quality ranking."
                    },
                    "include_opening_hours": {
                        "type": "boolean",
                        "description": (
                            "Request current seven-day opening hours only for an hours question "
                            "or a visit planned for a specific time."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=places.search_places,
        )
    return registry
