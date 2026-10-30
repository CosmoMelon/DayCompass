"""OpenAI-compatible chat integration, independent of the Streamlit UI."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from config import PROVIDER_ENV, PROVIDER_LABELS, Settings
from services.tool_registry import ToolRegistry, create_default_registry


ChatMessage = dict[str, Any]
CHAT_CLIENT_VERSION = 4

SYSTEM_PROMPT = "Answer unrelated questions normally without calling tools."
WEATHER_PROMPT = (
    "Use get_weather whenever the user asks about actual current or forecast weather, "
    "including follow-ups. Carry forward the city and date from prior weather tool results "
    "unless the user changes them. For relative dates, pass 'today' or 'tomorrow' to the tool "
    "so it uses the city's local date. If a location is ambiguous or weather data is unavailable, "
    "ask for clarification or explain the limitation. Do not invent weather data."
)
WEATHER_DISABLED_PROMPT = (
    "Weather integration is disabled for this turn. Do not claim to have checked "
    "current weather or forecasts; earlier weather results may be stale."
)
CALENDAR_PROMPT = (
    "Calendar integration is enabled. Before claiming the user is free or busy, or "
    "recommending an activity for a specific time period, call get_calendar_events "
    "for the relevant interval. Use relative date/time expressions in tool arguments "
    "when the user says today, tomorrow, or this evening; the tool resolves them in "
    "the calendar's timezone. For a specific clock time, check a short interval "
    "around it. When the weather tool is enabled and weather affects a proposed "
    "activity, call get_weather too. "
    "Treat transparent calendar events as non-blocking. Do not invent events, "
    "and do not assume availability if a calendar lookup fails."
)
CALENDAR_DISABLED_PROMPT = (
    "Calendar integration is disabled for this turn. Do not claim to have checked "
    "the calendar or assert the user is free; earlier calendar results may be stale."
)
PLACES_PROMPT = (
    "Places integration is enabled. Call search_places when the user asks for real "
    "venues or recommendations of somewhere to go in a known location. For a "
    "time-specific activity recommendation, use weather and calendar tools when "
    "available and relevant, then search for places that fit the conditions and "
    "free time. If the user says 'near me' and no location is known from this "
    "conversation, ask for their city or area. Present actual place names, "
    "addresses, brief reasons, and Google Maps links from tool results. Request "
    "ratings only for a ratings or quality question, and opening hours only when "
    "the user asks or a visit time needs checking. Current opening hours cover "
    "the next seven days only. Do not invent places, ratings, or hours; if hours "
    "are unavailable, say so. If a city "
    "name is ambiguous, ask for a region or country."
)
PLACES_DISABLED_PROMPT = (
    "Places integration is disabled for this turn. Do not claim to have searched "
    "Google Places or present earlier search results as fresh results."
)
MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS_PER_TURN = 8
MAX_PLACES_CALLS_PER_TURN = 3
CLOUDFLARE_MAX_TOKENS = 2048
CLOUDFLARE_RETRY_MAX_TOKENS = 4096


@dataclass
class ChatTurn:
    """Messages generated during one user turn, committed after success."""

    messages: list[ChatMessage] = field(default_factory=list)
    answer: str = ""


class ChatServiceError(Exception):
    """A user-facing error from the chat provider."""


class ChatClient:
    def __init__(
        self, settings: Settings, registry: ToolRegistry | None = None, *,
        weather_enabled: bool = True, calendar_enabled: bool = False,
        places_enabled: bool = False,
    ) -> None:
        self._model = settings.model
        self._provider = settings.provider
        self._provider_name = PROVIDER_LABELS.get(settings.provider, settings.provider)
        self._key_setting, self._model_setting = PROVIDER_ENV.get(
            settings.provider, ("API_KEY", "MODEL")
        )
        self._calendar_enabled = calendar_enabled
        self._weather_enabled = weather_enabled
        self._places_enabled = places_enabled
        self._registry = registry if registry is not None else create_default_registry(
            weather_enabled=weather_enabled,
            calendar_enabled=calendar_enabled,
            calendar_id=settings.calendar_id,
            calendar_api_key=settings.calendar_api_key,
            places_enabled=places_enabled,
            places_api_key=settings.places_api_key,
        )
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=0,
        )

    def stream_reply(
        self,
        messages: Sequence[ChatMessage],
        *,
        turn: ChatTurn | None = None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Iterator[str]:
        """Stream text, execute model-requested tools, and record their messages."""
        turn = turn if turn is not None else ChatTurn()
        # NVIDIA reasoning models may require reasoning_content on a tool-call
        # round trip. Other providers need only the standard chat fields.
        history = []
        for message in messages:
            item = dict(message) if self._provider == "nvidia" else {
                key: value for key, value in message.items() if key != "reasoning_content"
            }
            if self._provider != "gemini" and item.get("tool_calls"):
                # Gemini's thought signatures belong to its API and should
                # not be forwarded to other OpenAI-compatible providers.
                item["tool_calls"] = [
                    {key: value for key, value in call.items() if key != "extra_content"}
                    for call in item["tool_calls"]
                ]
            if self._provider != "gemini" and item.get("role") == "tool":
                item.pop("name", None)
            # Cloudflare's GPT-OSS Chat Completions endpoint accepts a tool-call
            # assistant message with empty content, but rejects null content.
            if (self._provider == "cloudflare" and item.get("role") == "assistant"
                    and item.get("tool_calls") and item.get("content") is None):
                item["content"] = ""
            history.append(item)
        conversation: list[ChatMessage] = [
            {"role": "system", "content": " ".join((
                SYSTEM_PROMPT,
                WEATHER_PROMPT if self._weather_enabled else WEATHER_DISABLED_PROMPT,
                CALENDAR_PROMPT if self._calendar_enabled else CALENDAR_DISABLED_PROMPT,
                PLACES_PROMPT if self._places_enabled else PLACES_DISABLED_PROMPT,
            ))},
            *history,
        ]
        total_tool_calls = 0
        places_tool_calls = 0
        tool_rounds = 0
        retried_empty_response = False

        try:
            for _ in range(MAX_TOOL_ROUNDS + 2):
                request: dict[str, Any] = {
                    "model": self._model,
                    "messages": conversation,
                    "stream": True,
                }
                tool_definitions = self._registry.definitions
                if tool_definitions:
                    request.update(tools=tool_definitions, tool_choice="auto")
                if self._provider == "cloudflare":
                    # Reasoning tokens share the model's output budget. Leave
                    # enough room for a visible answer after tool results.
                    request["max_tokens"] = (
                        CLOUDFLARE_RETRY_MAX_TOKENS if retried_empty_response
                        else CLOUDFLARE_MAX_TOKENS
                    )
                stream = self._client.chat.completions.create(**request)
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                calls_by_index: dict[int, ChatMessage] = {}

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                        yield delta.content
                    reasoning = getattr(delta, "reasoning_content", None)
                    if self._provider == "nvidia" and isinstance(reasoning, str):
                        reasoning_parts.append(reasoning)
                    for fragment in delta.tool_calls or []:
                        call = calls_by_index.setdefault(
                            fragment.index,
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if fragment.id:
                            call["id"] = fragment.id
                        if fragment.function:
                            if fragment.function.name:
                                call["function"]["name"] += fragment.function.name
                            if fragment.function.arguments:
                                call["function"]["arguments"] += fragment.function.arguments
                        if self._provider == "gemini":
                            extra_content = getattr(fragment, "extra_content", None)
                            if isinstance(extra_content, dict):
                                google = extra_content.get("google")
                                if isinstance(google, dict):
                                    signature = google.get("thought_signature")
                                    if isinstance(signature, str) and signature:
                                        saved = call.setdefault(
                                            "extra_content", {"google": {}}
                                        )["google"]
                                        saved["thought_signature"] = (
                                            saved.get("thought_signature", "") + signature
                                        )

                if not calls_by_index:
                    answer = "".join(content_parts)
                    if not answer.strip():
                        if self._provider == "cloudflare" and not retried_empty_response:
                            retried_empty_response = True
                            continue
                        raise ChatServiceError(
                            f"{self._provider_name} returned no text. Please try another model."
                        )
                    turn.answer = answer
                    final_message: ChatMessage = {"role": "assistant", "content": answer}
                    if reasoning_parts:
                        final_message["reasoning_content"] = "".join(reasoning_parts)
                    turn.messages.append(final_message)
                    return

                if tool_rounds >= MAX_TOOL_ROUNDS:
                    raise ChatServiceError("The model requested too many tool calls. Please try again.")
                tool_rounds += 1

                tool_calls = [calls_by_index[index] for index in sorted(calls_by_index)]
                if any(not call["id"] or not call["function"]["name"] for call in tool_calls):
                    raise ChatServiceError(
                        f"{self._provider_name} returned an incomplete tool call. Please try again."
                    )
                total_tool_calls += len(tool_calls)
                places_tool_calls += sum(
                    call["function"]["name"] == "search_places" for call in tool_calls
                )
                if total_tool_calls > MAX_TOOL_CALLS_PER_TURN or places_tool_calls > MAX_PLACES_CALLS_PER_TURN:
                    raise ChatServiceError("The model requested too many tool calls. Please narrow the request.")

                assistant_message: ChatMessage = {
                    "role": "assistant",
                    "content": "".join(content_parts) or ("" if self._provider == "cloudflare" else None),
                    "tool_calls": tool_calls,
                }
                if reasoning_parts:
                    assistant_message["reasoning_content"] = "".join(reasoning_parts)
                conversation.append(assistant_message)
                turn.messages.append(assistant_message)

                for call in tool_calls:
                    name = call["function"]["name"]
                    arguments = call["function"]["arguments"]
                    if on_tool_start:
                        try:
                            parsed = json.loads(arguments)
                        except ValueError:
                            parsed = {}
                        on_tool_start(name, parsed if isinstance(parsed, dict) else {})
                    result = self._registry.execute(name, arguments)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result),
                    }
                    if self._provider == "gemini":
                        tool_message["name"] = name
                    conversation.append(tool_message)
                    turn.messages.append(tool_message)
        except AuthenticationError as exc:
            raise ChatServiceError(
                f"{self._provider_name} rejected the credentials. Check {self._key_setting}."
            ) from exc
        except PermissionDeniedError as exc:
            raise ChatServiceError(
                f"{self._provider_name} denied access to this request or model."
            ) from exc
        except RateLimitError as exc:
            raise ChatServiceError(
                f"{self._provider_name} is rate limiting requests. Please wait and try again."
            ) from exc
        except APITimeoutError as exc:
            raise ChatServiceError("The request timed out. Please try again.") from exc
        except APIConnectionError as exc:
            raise ChatServiceError(
                f"Could not connect to {self._provider_name}. Check your connection and try again."
            ) from exc
        except APIStatusError as exc:
            if exc.status_code == 402:
                message = f"{self._provider_name} reported insufficient credits. Check your account balance."
            elif exc.status_code == 400 and self._provider == "cloudflare":
                message = (
                    "Cloudflare Workers AI rejected the chat request. "
                    "Try New Chat; if it persists, check that CLOUDFLARE_MODEL supports tool calling."
                )
            elif exc.status_code in (400, 404):
                message = (
                    f"{self._provider_name} rejected the request. "
                    f"Check {self._model_setting} and provider settings in .env."
                )
            elif exc.status_code >= 500:
                message = f"{self._provider_name} is temporarily unavailable. Please try again shortly."
            else:
                message = f"{self._provider_name} returned an error (HTTP {exc.status_code}). Please try again."
            raise ChatServiceError(message) from exc
        except APIError as exc:
            raise ChatServiceError(
                f"{self._provider_name} could not complete the request. Please try again."
            ) from exc


# Existing integrations can keep importing the original class name.
OpenRouterChatClient = ChatClient
