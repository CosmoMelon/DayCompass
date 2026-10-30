"""Google Places Text Search (New), independent of the chat provider."""

import math
import re
from typing import Any
from urllib.parse import urlencode

import requests

from services import ToolExecutionError


TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
BASE_FIELDS = (
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.googleMapsUri",
)
NEAR_ME = re.compile(r"\bnear me\b", re.IGNORECASE)


class PlacesServiceError(ToolExecutionError):
    """A Places lookup failure safe to return to the model."""


class PlacesService:
    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        self._api_key = api_key.strip()
        self._session = session or requests.Session()

    def search_places(
        self,
        query: str,
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_meters: float | None = None,
        max_results: int = 5,
        include_rating: bool = False,
        include_opening_hours: bool = False,
    ) -> dict[str, Any]:
        """Search for up to ten real places, requesting only needed fields."""
        if not self._api_key:
            raise PlacesServiceError("not_configured", "Places integration is not configured.")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 256:
            raise PlacesServiceError("invalid_query", "Provide a search query of 1 to 256 characters.")
        query = query.strip()
        if location is not None:
            if not isinstance(location, str) or not location.strip() or len(location.strip()) > 120:
                raise PlacesServiceError("invalid_location", "Provide a city or region of 1 to 120 characters.")
            location = location.strip()
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise PlacesServiceError("invalid_limit", "max_results must be a positive integer.")
        max_results = min(max_results, 10)
        if not isinstance(include_rating, bool) or not isinstance(include_opening_hours, bool):
            raise PlacesServiceError("invalid_arguments", "Optional detail flags must be true or false.")

        has_coordinates = latitude is not None or longitude is not None
        if has_coordinates:
            if latitude is None or longitude is None:
                raise PlacesServiceError("invalid_location", "Provide both latitude and longitude.")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in (latitude, longitude)) or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise PlacesServiceError("invalid_location", "Latitude or longitude is invalid.")
        if radius_meters is not None:
            if not has_coordinates:
                raise PlacesServiceError("invalid_radius", "A radius requires latitude and longitude.")
            if (isinstance(radius_meters, bool) or not isinstance(radius_meters, (int, float))
                    or not math.isfinite(radius_meters) or not 0 < radius_meters <= 50_000):
                raise PlacesServiceError("invalid_radius", "Radius must be between 0 and 50,000 meters.")

        if NEAR_ME.search(query):
            if not location and not has_coordinates:
                raise PlacesServiceError(
                    "location_required", "Please ask the user for a city or location before searching near them."
                )
            query = NEAR_ME.sub(f"near {location}" if location and not has_coordinates else "", query).strip()
        elif location and not has_coordinates and location.casefold() not in query.casefold():
            query = f"{query} in {location}"
        if not query:
            raise PlacesServiceError("invalid_query", "Specify what kind of place to find.")

        body: dict[str, Any] = {"textQuery": query, "maxResultCount": max_results}
        if has_coordinates:
            body["locationBias"] = {"circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius_meters if radius_meters is not None else 5_000,
            }}
        fields = list(BASE_FIELDS)
        if include_rating:
            fields.extend(("places.rating", "places.userRatingCount"))
        if include_opening_hours:
            fields.append("places.currentOpeningHours")
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": ",".join(fields),
        }
        try:
            response = self._session.post(TEXT_SEARCH_URL, json=body, headers=headers, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise PlacesServiceError("timeout", "The places search timed out. Please try again.") from exc
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code in (401, 403):
                error = ("access_denied", "Places access failed. Check the API key and Places API (New) setup.")
            elif code == 429:
                error = ("rate_limited", "Google Places is rate limiting requests. Please try again shortly.")
            elif code == 400:
                error = ("invalid_request", "Google Places rejected the search request.")
            else:
                error = ("api_failure", "Google Places is unavailable. Please try again.")
            raise PlacesServiceError(*error) from exc
        except requests.RequestException as exc:
            raise PlacesServiceError("api_failure", "Could not connect to Google Places.") from exc
        except ValueError as exc:
            raise PlacesServiceError("api_failure", "Google Places returned invalid data.") from exc

        if not isinstance(payload, dict) or "error" in payload:
            raise PlacesServiceError("api_failure", "Google Places returned invalid data.")
        raw_places = payload.get("places", [])
        if not isinstance(raw_places, list):
            raise PlacesServiceError("api_failure", "Google Places returned invalid results.")

        places: list[dict[str, Any]] = []
        for raw in raw_places[:max_results]:
            if not isinstance(raw, dict):
                raise PlacesServiceError("api_failure", "Google Places returned an invalid place.")
            place_id = raw.get("id")
            display_name = raw.get("displayName")
            name = display_name.get("text") if isinstance(display_name, dict) else None
            if not isinstance(place_id, str) or not place_id or not isinstance(name, str) or not name:
                raise PlacesServiceError("api_failure", "Google Places returned an incomplete place.")
            maps_url = raw.get("googleMapsUri")
            if not isinstance(maps_url, str) or not maps_url:
                maps_url = "https://www.google.com/maps/search/?" + urlencode({
                    "api": 1, "query": name, "query_place_id": place_id,
                })
            coordinates = raw.get("location") or {}
            place: dict[str, Any] = {
                "name": name,
                "id": place_id,
                "address": raw.get("formattedAddress"),
                "latitude": coordinates.get("latitude") if isinstance(coordinates, dict) else None,
                "longitude": coordinates.get("longitude") if isinstance(coordinates, dict) else None,
                "google_maps_url": maps_url,
            }
            if include_rating and raw.get("rating") is not None:
                place["rating"] = raw["rating"]
                if raw.get("userRatingCount") is not None:
                    place["rating_count"] = raw["userRatingCount"]
            if include_opening_hours and isinstance(raw.get("currentOpeningHours"), dict):
                place["opening_hours"] = raw["currentOpeningHours"]
            places.append(place)

        return {
            "source": "Google Places",
            "query": query,
            "location": location,
            "places": places,
        }
