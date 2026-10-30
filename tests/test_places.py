"""Google Places Text Search tests with mocked HTTP responses."""

import unittest
from unittest.mock import Mock

import requests

from services.places_service import TEXT_SEARCH_URL, PlacesService, PlacesServiceError
from services.tool_registry import create_default_registry


def response(payload, status=200):
    item = Mock()
    item.json.return_value = payload
    item.status_code = status
    if status >= 400:
        item.raise_for_status.side_effect = requests.HTTPError(response=item)
    return item


PLACE = {
    "id": "ChIJexample",
    "displayName": {"text": "Downtown Badminton Club"},
    "formattedAddress": "100 King St W, Toronto, ON",
    "location": {"latitude": 43.65, "longitude": -79.38},
    "googleMapsUri": "https://maps.google.com/?cid=example",
    "rating": 4.6,
    "userRatingCount": 120,
    "currentOpeningHours": {"weekdayDescriptions": ["Tuesday: 9:00 AM – 9:00 PM"]},
}


class PlacesTests(unittest.TestCase):
    def test_basic_search_uses_minimal_mask_and_location(self):
        session = Mock()
        session.post.return_value = response({"places": [PLACE]})
        result = PlacesService("test-places-key", session).search_places(
            "badminton courts", location="Toronto"
        )
        self.assertEqual(result["places"][0]["name"], "Downtown Badminton Club")
        self.assertEqual(result["places"][0]["id"], "ChIJexample")
        self.assertEqual(result["places"][0]["latitude"], 43.65)
        self.assertEqual(result["places"][0]["google_maps_url"], PLACE["googleMapsUri"])
        self.assertNotIn("rating", result["places"][0])
        self.assertNotIn("opening_hours", result["places"][0])
        request = session.post.call_args
        self.assertEqual(request.args[0], TEXT_SEARCH_URL)
        self.assertEqual(request.kwargs["json"], {
            "textQuery": "badminton courts in Toronto", "maxResultCount": 5,
        })
        self.assertEqual(request.kwargs["headers"]["X-Goog-Api-Key"], "test-places-key")
        mask = request.kwargs["headers"]["X-Goog-FieldMask"]
        self.assertIn("places.googleMapsUri", mask)
        self.assertNotIn("places.rating", mask)
        self.assertNotIn("places.currentOpeningHours", mask)
        self.assertNotIn("*", mask)
        self.assertNotIn("test-places-key", str(request.kwargs["json"]))
        self.assertEqual(request.kwargs["timeout"], 10)

    def test_coordinates_and_optional_details(self):
        session = Mock()
        session.post.return_value = response({"places": [PLACE]})
        result = PlacesService("test-key", session).search_places(
            "badminton courts near me", latitude=43.65, longitude=-79.38,
            radius_meters=2_000, max_results=20,
            include_rating=True, include_opening_hours=True,
        )
        request = session.post.call_args.kwargs
        self.assertEqual(request["json"]["textQuery"], "badminton courts")
        self.assertEqual(request["json"]["maxResultCount"], 10)
        self.assertEqual(request["json"]["locationBias"]["circle"], {
            "center": {"latitude": 43.65, "longitude": -79.38}, "radius": 2_000,
        })
        self.assertIn("places.rating", request["headers"]["X-Goog-FieldMask"])
        self.assertIn("places.currentOpeningHours", request["headers"]["X-Goog-FieldMask"])
        self.assertEqual(result["places"][0]["rating"], 4.6)
        self.assertEqual(result["places"][0]["rating_count"], 120)
        self.assertIn("weekdayDescriptions", result["places"][0]["opening_hours"])

    def test_empty_results_and_maps_link_fallback(self):
        session = Mock()
        session.post.side_effect = [response({}), response({"places": [{
            "id": "ChIJexample", "displayName": {"text": "A Museum"},
        }]})]
        service = PlacesService("test-key", session)
        self.assertEqual(service.search_places("museums", location="Mexico City")["places"], [])
        result = service.search_places("museums", location="Mexico City")
        self.assertIn("query_place_id=ChIJexample", result["places"][0]["google_maps_url"])
        self.assertIsNone(result["places"][0]["address"])

    def test_location_and_argument_validation_skips_api(self):
        session = Mock()
        service = PlacesService("test-key", session)
        for arguments, code in (
            ({"query": "coffee near me"}, "location_required"),
            ({"query": "coffee", "latitude": 43.0}, "invalid_location"),
            ({"query": "coffee", "radius_meters": 500}, "invalid_radius"),
            ({"query": "coffee", "max_results": 0}, "invalid_limit"),
            ({"query": ""}, "invalid_query"),
            ({"query": "near me", "latitude": 43.0, "longitude": -79.0}, "invalid_query"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(PlacesServiceError) as caught:
                    service.search_places(**arguments)
                self.assertEqual(caught.exception.code, code)
        session.post.assert_not_called()

    def test_api_failures_are_safe(self):
        session = Mock()
        service = PlacesService("test-places-key", session)
        for status, code in ((403, "access_denied"), (429, "rate_limited"), (500, "api_failure")):
            session.post.return_value = response({}, status=status)
            with self.subTest(status=status):
                with self.assertRaises(PlacesServiceError) as caught:
                    service.search_places("museums", location="Toronto")
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("test-places-key", str(caught.exception))
        session.post.side_effect = requests.Timeout("slow")
        with self.assertRaises(PlacesServiceError) as caught:
            service.search_places("museums", location="Toronto")
        self.assertEqual(caught.exception.code, "timeout")

    def test_registry_toggle(self):
        disabled = create_default_registry()
        enabled = create_default_registry(places_enabled=True, places_api_key="test-key")
        self.assertEqual([item["function"]["name"] for item in disabled.definitions], ["get_weather"])
        self.assertEqual([item["function"]["name"] for item in enabled.definitions], [
            "get_weather", "search_places",
        ])
        self.assertEqual(disabled.execute("search_places", '{"query":"coffee"}')["error"]["code"], "unknown_tool")


if __name__ == "__main__":
    unittest.main()
