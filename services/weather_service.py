"""Open-Meteo geocoding and forecast lookup."""

from datetime import date as Date, timedelta
from typing import Any

import requests

from services import ToolExecutionError


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,weather_code,"
    "precipitation_probability,precipitation,wind_speed_10m"
)
DAILY_FIELDS = (
    "temperature_2m_max,temperature_2m_min,apparent_temperature_max,"
    "apparent_temperature_min,weather_code,precipitation_probability_max,"
    "precipitation_sum,rain_sum,wind_speed_10m_max,sunrise,sunset"
)

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherServiceError(ToolExecutionError):
    """A weather lookup failure that the assistant can explain."""


def _condition(code: int | None) -> str | None:
    return WEATHER_CODES.get(code, "Unknown conditions") if code is not None else None


def _place_label(place: dict[str, Any]) -> str:
    return ", ".join(
        str(value) for value in (place.get("name"), place.get("admin1"), place.get("country")) if value
    )


class WeatherService:
    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise WeatherServiceError("timeout", "The weather service timed out. Please try again.") from exc
        except requests.RequestException as exc:
            raise WeatherServiceError("api_failure", "The weather service is unavailable. Please try again.") from exc
        except ValueError as exc:
            raise WeatherServiceError("api_failure", "The weather service returned invalid data.") from exc

        if not isinstance(payload, dict) or payload.get("error"):
            raise WeatherServiceError("api_failure", "The weather service could not complete this lookup.")
        return payload

    def _resolve(self, location: str) -> dict[str, Any]:
        if not isinstance(location, str) or len(location.strip()) < 2:
            raise WeatherServiceError("invalid_location", "Please provide a city name.")

        query = location.strip()
        payload = self._get_json(GEOCODING_URL, {"name": query, "count": 10, "language": "en"})
        results = payload.get("results") or []
        if not isinstance(results, list) or not results:
            raise WeatherServiceError(
                "invalid_location", f"I could not find a city matching {query!r}. Try adding a region or country."
            )

        city = query.split(",", 1)[0].strip().casefold()
        exact = [place for place in results if isinstance(place, dict) and str(place.get("name", "")).casefold() == city]
        candidates = exact or [place for place in results if isinstance(place, dict)]
        unique: dict[tuple[str, str, str], dict[str, Any]] = {}
        for place in candidates:
            key = (
                str(place.get("name", "")).casefold(),
                str(place.get("admin1", "")).casefold(),
                str(place.get("country", "")).casefold(),
            )
            unique.setdefault(key, place)
        candidates = list(unique.values())
        candidates.sort(key=lambda place: place.get("population") or 0, reverse=True)
        if not candidates:
            raise WeatherServiceError("invalid_location", f"I could not find a city matching {query!r}.")

        # A clear population lead resolves common cases such as Toronto, while
        # substantial same-name cities (such as London) need clarification.
        if len(candidates) > 1:
            first = candidates[0].get("population") or 0
            second = candidates[1].get("population") or 0
            if second >= 100_000 or first < 10 * max(second, 1):
                options = [_place_label(place) for place in candidates[:5]]
                raise WeatherServiceError(
                    "ambiguous_location",
                    f"Several places match {query!r}. Please specify a region or country.",
                    options=options,
                )

        place = candidates[0]
        if place.get("latitude") is None or place.get("longitude") is None:
            raise WeatherServiceError("api_failure", "Location coordinates were unavailable.")
        return place

    def get_weather(self, location: str, date: str | None = None) -> dict[str, Any]:
        """Return current conditions and a daily forecast for a city-local date."""
        place = self._resolve(location)
        forecast = self._get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "timezone": place.get("timezone") or "auto",
                "forecast_days": 16,
                "current": CURRENT_FIELDS,
                "daily": DAILY_FIELDS,
            },
        )

        daily = forecast.get("daily") or {}
        days = daily.get("time") or []
        current = forecast.get("current") or {}
        if not isinstance(days, list) or not days:
            raise WeatherServiceError("api_failure", "Forecast data was unavailable for this location.")

        # Open-Meteo returns current.time in the requested location's timezone.
        local_today = str(current.get("time") or days[0])[:10]
        try:
            base_date = Date.fromisoformat(local_today)
            if date is None or (isinstance(date, str) and date.strip().lower() == "today"):
                requested = base_date
            elif isinstance(date, str) and date.strip().lower() == "tomorrow":
                requested = base_date + timedelta(days=1)
            elif isinstance(date, str):
                requested = Date.fromisoformat(date.strip())
            else:
                raise ValueError("Invalid date type")
        except ValueError as exc:
            raise WeatherServiceError(
                "invalid_date", "Use 'today', 'tomorrow', or a date in YYYY-MM-DD format."
            ) from exc

        requested_day = requested.isoformat()
        if requested_day not in days:
            raise WeatherServiceError(
                "date_unavailable",
                f"Forecasts are available from {days[0]} through {days[-1]}; {requested_day} is outside that range.",
            )
        index = days.index(requested_day)

        def daily_value(field: str) -> Any:
            values = daily.get(field)
            return values[index] if isinstance(values, list) and index < len(values) else None

        result: dict[str, Any] = {
            "source": "Open-Meteo",
            "location": {
                "name": place.get("name"),
                "region": place.get("admin1"),
                "country": place.get("country"),
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "timezone": forecast.get("timezone") or place.get("timezone"),
            },
            "date": requested_day,
            "forecast": {
                "conditions": _condition(daily_value("weather_code")),
                "temperature_max_c": daily_value("temperature_2m_max"),
                "temperature_min_c": daily_value("temperature_2m_min"),
                "apparent_temperature_max_c": daily_value("apparent_temperature_max"),
                "apparent_temperature_min_c": daily_value("apparent_temperature_min"),
                "precipitation_probability_max_percent": daily_value("precipitation_probability_max"),
                "precipitation_mm": daily_value("precipitation_sum"),
                "rain_mm": daily_value("rain_sum"),
                "wind_speed_max_kmh": daily_value("wind_speed_10m_max"),
                "sunrise_local": daily_value("sunrise"),
                "sunset_local": daily_value("sunset"),
            },
        }
        if requested == base_date:
            result["current"] = {
                "observed_at_local": current.get("time"),
                "temperature_c": current.get("temperature_2m"),
                "apparent_temperature_c": current.get("apparent_temperature"),
                "conditions": _condition(current.get("weather_code")),
                "precipitation_probability_percent": current.get("precipitation_probability"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
            }
        return result
