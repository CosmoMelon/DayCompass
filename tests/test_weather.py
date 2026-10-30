"""Open-Meteo integration checks using mocked HTTP responses."""

import unittest
from unittest.mock import Mock

import requests

from services.tool_registry import create_default_registry
from services.weather_service import FORECAST_URL, GEOCODING_URL, WeatherService, WeatherServiceError


def response(payload):
    item = Mock()
    item.json.return_value = payload
    return item


TORONTO = {
    "name": "Toronto", "admin1": "Ontario", "country": "Canada",
    "latitude": 43.65, "longitude": -79.38, "timezone": "America/Toronto",
    "population": 2731571,
}

FORECAST = {
    "timezone": "America/Toronto",
    "current": {
        "time": "2026-10-13T10:15", "temperature_2m": 14.2,
        "apparent_temperature": 12.5, "weather_code": 2,
        "precipitation_probability": 20, "precipitation": 0.0,
        "wind_speed_10m": 16.0,
    },
    "daily": {
        "time": ["2026-10-13", "2026-10-14"],
        "weather_code": [2, 61],
        "temperature_2m_max": [18.0, 15.0],
        "temperature_2m_min": [9.0, 8.0],
        "apparent_temperature_max": [17.0, 13.0],
        "apparent_temperature_min": [7.0, 6.0],
        "precipitation_probability_max": [30, 80],
        "precipitation_sum": [0.0, 4.2],
        "rain_sum": [0.0, 4.2],
        "wind_speed_10m_max": [21.0, 27.0],
        "sunrise": ["2026-10-13T07:29", "2026-10-14T07:30"],
        "sunset": ["2026-10-13T18:33", "2026-10-14T18:31"],
    },
}


class WeatherTests(unittest.TestCase):
    def test_today_and_tomorrow_weather_fields(self):
        session = Mock()
        session.get.side_effect = [
            response({"results": [TORONTO]}), response(FORECAST),
            response({"results": [TORONTO]}), response(FORECAST),
        ]
        service = WeatherService(session)
        today = service.get_weather("Toronto")
        self.assertEqual(today["current"]["temperature_c"], 14.2)
        self.assertEqual(today["current"]["conditions"], "Partly cloudy")
        self.assertEqual(today["forecast"]["precipitation_probability_max_percent"], 30)

        tomorrow = service.get_weather("Toronto", "tomorrow")
        self.assertEqual(tomorrow["date"], "2026-10-14")
        self.assertEqual(tomorrow["forecast"]["rain_mm"], 4.2)
        self.assertEqual(tomorrow["forecast"]["conditions"], "Slight rain")
        self.assertNotIn("current", tomorrow)
        self.assertEqual(session.get.call_args_list[0].args[0], GEOCODING_URL)
        self.assertEqual(session.get.call_args_list[1].args[0], FORECAST_URL)
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["timezone"], "America/Toronto")

    def test_ambiguous_and_invalid_city_skip_forecast(self):
        london = {**TORONTO, "name": "London", "country": "United Kingdom", "admin1": "England", "population": 8900000}
        london_canada = {**TORONTO, "name": "London", "population": 420000}
        session = Mock()
        session.get.return_value = response({"results": [london, london_canada]})
        with self.assertRaises(WeatherServiceError) as caught:
            WeatherService(session).get_weather("London")
        self.assertEqual(caught.exception.code, "ambiguous_location")
        self.assertEqual(len(caught.exception.options), 2)
        self.assertEqual(session.get.call_count, 1)

        session.get.return_value = response({"results": []})
        with self.assertRaises(WeatherServiceError) as caught:
            WeatherService(session).get_weather("Notacity")
        self.assertEqual(caught.exception.code, "invalid_location")

    def test_unavailable_date_and_http_timeout(self):
        session = Mock()
        session.get.side_effect = [response({"results": [TORONTO]}), response(FORECAST)]
        with self.assertRaises(WeatherServiceError) as caught:
            WeatherService(session).get_weather("Toronto", "2026-11-10")
        self.assertEqual(caught.exception.code, "date_unavailable")

        session.get.side_effect = requests.Timeout("slow")
        with self.assertRaises(WeatherServiceError) as caught:
            WeatherService(session).get_weather("Toronto")
        self.assertEqual(caught.exception.code, "timeout")

    def test_registry_returns_structured_weather_error(self):
        registry = create_default_registry()
        result = registry.execute("get_weather", '{"location":""}')
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "invalid_location")
        self.assertEqual(registry.definitions[0]["function"]["name"], "get_weather")

        malformed = registry.execute("get_weather", "not json")
        self.assertEqual(malformed["error"]["code"], "invalid_arguments")
        unknown = registry.execute("not_a_tool", "{}")
        self.assertEqual(unknown["error"]["code"], "unknown_tool")

    def test_weather_can_be_disabled_independently(self):
        registry = create_default_registry(weather_enabled=False)
        self.assertEqual(registry.definitions, [])
        self.assertEqual(registry.execute("get_weather", '{}')["error"]["code"], "unknown_tool")


if __name__ == "__main__":
    unittest.main()
