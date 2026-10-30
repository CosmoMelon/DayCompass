"""Behavior checks for multi-turn UI state and provider failures."""

import os
import json
import unittest
from importlib import reload
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APITimeoutError, RateLimitError
from streamlit.testing.v1 import AppTest

import config
from config import Settings
from services import llm_client
from services.llm_client import ChatClient, ChatTurn, OpenRouterChatClient
from services.tool_registry import ToolRegistry


def text_chunk(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))]
    )


def tool_chunk(*, index: int = 0, call_id=None, name=None, arguments=None, extra_content=None):
    fragment = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
        extra_content=extra_content,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[fragment]))]
    )


def reasoning_chunk(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(
            content=None, tool_calls=None, reasoning_content=content,
        ))]
    )


class ChatTests(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model",
            "GOOGLE_PLACES_API_KEY": "test-places-key",
        },
    )
    def test_app_recovers_from_stale_settings_import(self):
        class OldSettings:
            api_key = "test-key"
            model = "test/model"
            calendar_id = ""
            calendar_api_key = ""

        with patch.object(config, "Settings", OldSettings):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
            self.assertFalse(app.exception)
            self.assertTrue(hasattr(config.Settings, "places_api_key"))
            self.assertEqual(len(app.toggle), 3)

    @patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model"},
    )
    def test_app_reloads_older_cloudflare_client(self):
        with patch.object(llm_client, "CHAT_CLIENT_VERSION", 1):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
            self.assertFalse(app.exception)
            self.assertEqual(llm_client.CHAT_CLIENT_VERSION, 4)

    @patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model"},
    )
    def test_app_recovers_from_stale_service_import(self):
        del llm_client.ChatTurn
        try:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
            self.assertFalse(app.exception)
            self.assertTrue(hasattr(llm_client, "ChatTurn"))
        finally:
            reload(llm_client)

    @patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model"},
    )
    @patch("services.llm_client.OpenAI")
    def test_streamed_follow_up_and_new_chat(self, openai_class):
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [text_chunk("Hello"), text_chunk(" there")],
            [text_chunk("Yes")],
        ]

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
        self.assertFalse(app.exception)

        app.chat_input[0].set_value("Hi").run(timeout=15)
        self.assertEqual(
            app.session_state.messages,
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there"}],
        )

        app.chat_input[0].set_value("Can you elaborate?").run(timeout=15)
        sent_messages = completion.call_args.kwargs["messages"][1:]
        self.assertEqual(
            sent_messages,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there"},
                {"role": "user", "content": "Can you elaborate?"},
            ],
        )
        self.assertEqual(app.session_state.messages[-1]["content"], "Yes")

        app.button[0].click().run(timeout=15)
        self.assertEqual(app.session_state.messages, [])

        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        completion.side_effect = APITimeoutError(request=request)
        app.chat_input[0].set_value("Try again").run(timeout=15)
        self.assertEqual(app.session_state.messages, [])
        self.assertTrue(any("timed out" in error.value for error in app.error))

    @patch("services.llm_client.OpenAI")
    def test_rate_limit_and_timeout_are_readable(self, openai_class):
        client = OpenRouterChatClient(Settings(api_key="test-key", model="test/model"))
        completion = openai_class.return_value.chat.completions.create
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        completion.side_effect = RateLimitError(
            "rate limited", response=httpx.Response(429, request=request), body=None
        )
        with self.assertRaisesRegex(llm_client.ChatServiceError, "rate limiting"):
            list(client.stream_reply([{"role": "user", "content": "Hello"}]))

        completion.side_effect = APITimeoutError(request=request)
        with self.assertRaisesRegex(llm_client.ChatServiceError, "timed out"):
            list(client.stream_reply([{"role": "user", "content": "Hello"}]))

    @patch("services.llm_client.OpenAI")
    def test_native_tool_call_and_follow_up_history(self, openai_class):
        weather_calls = []
        registry = ToolRegistry()
        registry.register(
            "get_weather", "Weather lookup", {"type": "object"},
            lambda **arguments: weather_calls.append(arguments) or {"date": arguments.get("date", "today")},
        )
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [
                tool_chunk(call_id="call_1", name="get_weather", arguments='{"location":"Tor'),
                tool_chunk(arguments='onto"}'),
            ],
            [text_chunk("It is sunny in Toronto.")],
            [
                tool_chunk(call_id="call_2", name="get_weather", arguments='{"location":"Toronto","date":"tomorrow"}'),
            ],
            [text_chunk("Tomorrow is cloudy.")],
        ]
        client = OpenRouterChatClient(Settings(api_key="test-key", model="test/model"), registry)
        first_turn = ChatTurn()
        first_user = {"role": "user", "content": "What's the weather in Toronto?"}
        self.assertEqual(
            "".join(client.stream_reply([first_user], turn=first_turn)),
            "It is sunny in Toronto.",
        )
        self.assertEqual(weather_calls, [{"location": "Toronto"}])
        self.assertEqual(first_turn.messages[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(first_turn.messages[1]["role"], "tool")
        self.assertTrue(json.loads(first_turn.messages[1]["content"])["ok"])
        self.assertEqual(completion.call_args_list[1].kwargs["messages"][-1], first_turn.messages[1])

        follow_up = {"role": "user", "content": "How about tomorrow?"}
        second_turn = ChatTurn()
        self.assertEqual(
            "".join(client.stream_reply([first_user, *first_turn.messages, follow_up], turn=second_turn)),
            "Tomorrow is cloudy.",
        )
        self.assertEqual(weather_calls[-1], {"location": "Toronto", "date": "tomorrow"})
        self.assertEqual(completion.call_args_list[2].kwargs["messages"][2], first_turn.messages[0])
        self.assertEqual(completion.call_args_list[2].kwargs["tool_choice"], "auto")

    @patch("services.llm_client.OpenAI")
    def test_nvidia_cloudflare_and_groq_use_shared_streaming_tool_loop(self, openai_class):
        for provider, base_url in (
            ("nvidia", "https://integrate.api.nvidia.com/v1"),
            ("cloudflare", "https://api.cloudflare.com/client/v4/accounts/account123/ai/v1"),
            ("groq", "https://api.groq.com/openai/v1"),
        ):
            with self.subTest(provider=provider):
                registry = ToolRegistry()
                registry.register(
                    "get_weather", "Weather lookup", {"type": "object"},
                    lambda **_arguments: {"temperature_c": 18},
                )
                completion = openai_class.return_value.chat.completions.create
                completion.reset_mock()
                completion.side_effect = [
                    [
                        *([reasoning_chunk("Need weather.")] if provider == "nvidia" else []),
                        tool_chunk(call_id="call_1", name="get_weather", arguments='{"location":"Toronto"}'),
                    ],
                    [
                        *([reasoning_chunk("Answer briefly.")] if provider == "nvidia" else []),
                        text_chunk("It is 18°C in Toronto."),
                    ],
                ]
                client = ChatClient(Settings(
                    api_key=f"{provider}-secret", model="tool-model",
                    provider=provider, base_url=base_url,
                ), registry)
                turn = ChatTurn()
                answer = "".join(client.stream_reply([
                    {"role": "user", "content": "What's the weather in Toronto?"}
                ], turn=turn))
                self.assertIn("18°C", answer)
                self.assertEqual(openai_class.call_args.kwargs["base_url"], base_url)
                self.assertEqual(completion.call_args_list[0].kwargs["model"], "tool-model")
                self.assertTrue(completion.call_args_list[0].kwargs["stream"])
                self.assertEqual(completion.call_args_list[1].kwargs["messages"][-1], turn.messages[1])
                self.assertEqual(turn.messages[1]["role"], "tool")
                if provider == "nvidia":
                    self.assertEqual(turn.messages[0]["reasoning_content"], "Need weather.")
                    self.assertEqual(completion.call_args_list[1].kwargs["messages"][-2], turn.messages[0])
                    self.assertEqual(turn.messages[-1]["reasoning_content"], "Answer briefly.")
                elif provider == "cloudflare":
                    self.assertNotIn("reasoning_content", turn.messages[0])
                    self.assertEqual(turn.messages[0]["content"], "")
                else:
                    self.assertNotIn("reasoning_content", turn.messages[0])
                    self.assertIsNone(turn.messages[0]["content"])
                self.assertNotIn(f"{provider}-secret", json.dumps(turn.messages))

    @patch("services.llm_client.OpenAI")
    def test_provider_switch_omits_nvidia_reasoning_from_cloudflare_history(self, openai_class):
        completion = openai_class.return_value.chat.completions.create
        completion.return_value = [text_chunk("Okay")]
        client = ChatClient(Settings(
            api_key="cf-token", model="@cf/model", provider="cloudflare",
            base_url="https://api.cloudflare.com/client/v4/accounts/account123/ai/v1",
        ))
        prior = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location":"Toronto"}'},
            }], "reasoning_content": "Private model reasoning"},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
            {"role": "user", "content": "And tomorrow?"},
        ]
        list(client.stream_reply(prior))
        sent = completion.call_args.kwargs["messages"]
        self.assertNotIn("reasoning_content", sent[1])
        self.assertEqual(sent[1]["content"], "")
        self.assertEqual(sent[1]["tool_calls"], prior[0]["tool_calls"])
        self.assertIn("reasoning_content", prior[0])

    @patch("services.llm_client.OpenAI")
    def test_gemini_streamed_tool_signature_survives_round_trip_and_follow_up(self, openai_class):
        registry = ToolRegistry()
        registry.register(
            "get_weather", "Weather lookup", {"type": "object"},
            lambda **_kwargs: {"temperature_c": 18},
        )
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [
                tool_chunk(call_id="gemini_1", name="get_weather", arguments='{"location":"Tor',
                           extra_content={"google": {"thought_signature": "sig-"}}),
                tool_chunk(arguments='onto"}',
                           extra_content={"google": {"thought_signature": "one"}}),
            ],
            [text_chunk("Toronto is 18°C.")],
            [text_chunk("It is mild.")],
        ]
        client = ChatClient(Settings(
            api_key="gemini-key", model="gemini-tool-model", provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ), registry)
        first_user = {"role": "user", "content": "What's the weather in Toronto?"}
        first = ChatTurn()
        self.assertIn("18°C", "".join(client.stream_reply([first_user], turn=first)))
        call = first.messages[0]["tool_calls"][0]
        self.assertEqual(call["extra_content"]["google"]["thought_signature"], "sig-one")
        self.assertEqual(completion.call_args_list[1].kwargs["messages"][-2], first.messages[0])
        self.assertEqual(completion.call_args_list[1].kwargs["messages"][-1]["name"], "get_weather")
        self.assertNotIn("gemini-key", json.dumps(first.messages))

        follow_up = {"role": "user", "content": "And how does that feel?"}
        self.assertIn("mild", "".join(client.stream_reply([
            first_user, *first.messages, follow_up
        ])))
        self.assertEqual(
            completion.call_args_list[2].kwargs["messages"][2]["tool_calls"][0]["extra_content"],
            call["extra_content"],
        )

        groq = ChatClient(Settings(
            api_key="groq-key", model="groq-tool-model", provider="groq",
            base_url="https://api.groq.com/openai/v1",
        ), registry)
        completion.return_value = [text_chunk("Okay")]
        completion.side_effect = None
        list(groq.stream_reply([first_user, *first.messages, follow_up]))
        sent = completion.call_args.kwargs["messages"]
        self.assertNotIn("extra_content", sent[2]["tool_calls"][0])
        self.assertNotIn("name", sent[3])
        self.assertIn("extra_content", first.messages[0]["tool_calls"][0])

    @patch("services.llm_client.OpenAI")
    def test_cloudflare_retries_empty_reasoning_response(self, openai_class):
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [[reasoning_chunk("Thinking...")], [text_chunk("Here is an answer.")]]
        client = ChatClient(Settings(
            api_key="cf-token", model="@cf/model", provider="cloudflare",
            base_url="https://api.cloudflare.com/client/v4/accounts/account123/ai/v1",
        ), weather_enabled=False)
        turn = ChatTurn()
        self.assertEqual(
            "".join(client.stream_reply([{"role": "user", "content": "Hello"}], turn=turn)),
            "Here is an answer.",
        )
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(completion.call_args_list[0].kwargs["max_tokens"], 2048)
        self.assertEqual(completion.call_args_list[1].kwargs["max_tokens"], 4096)
        self.assertNotIn("tools", completion.call_args_list[0].kwargs)
        self.assertEqual(turn.messages, [{"role": "assistant", "content": "Here is an answer."}])

    @patch("services.llm_client.OpenAI")
    def test_cloudflare_retries_empty_reply_after_tool_result(self, openai_class):
        registry = ToolRegistry()
        tool_calls = []
        registry.register(
            "get_weather", "Weather lookup", {"type": "object"},
            lambda **kwargs: tool_calls.append(kwargs) or {"temperature_c": 18},
        )
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [tool_chunk(call_id="call_1", name="get_weather", arguments='{"location":"Toronto"}')],
            [reasoning_chunk("Need to answer.")],
            [text_chunk("It is 18°C in Toronto.")],
        ]
        client = ChatClient(Settings(
            api_key="cf-token", model="@cf/model", provider="cloudflare",
            base_url="https://api.cloudflare.com/client/v4/accounts/account123/ai/v1",
        ), registry)
        turn = ChatTurn()
        self.assertIn("18°C", "".join(client.stream_reply([
            {"role": "user", "content": "What's the weather in Toronto?"}
        ], turn=turn)))
        self.assertEqual(tool_calls, [{"location": "Toronto"}])
        self.assertEqual([message["role"] for message in turn.messages], ["assistant", "tool", "assistant"])
        self.assertEqual(completion.call_args_list[1].kwargs["messages"][-1], turn.messages[1])
        self.assertEqual(completion.call_args_list[2].kwargs["messages"][-1], turn.messages[1])

    @patch("services.llm_client.OpenAI")
    def test_non_weather_chat_does_not_execute_weather(self, openai_class):
        registry = ToolRegistry()
        called = []
        registry.register("get_weather", "Weather lookup", {"type": "object"}, lambda **kwargs: called.append(kwargs))
        completion = openai_class.return_value.chat.completions.create
        completion.return_value = [text_chunk("Hello"), text_chunk("!")]
        client = OpenRouterChatClient(Settings(api_key="test-key", model="test/model"), registry)
        turn = ChatTurn()
        self.assertEqual("".join(client.stream_reply([{"role": "user", "content": "Hi"}], turn=turn)), "Hello!")
        self.assertEqual(called, [])
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(turn.messages, [{"role": "assistant", "content": "Hello!"}])

    @patch("services.llm_client.OpenAI")
    def test_weather_and_calendar_calls_share_a_turn(self, openai_class):
        calls = []
        registry = ToolRegistry()
        registry.register(
            "get_weather", "Weather lookup", {"type": "object"},
            lambda **arguments: calls.append(("weather", arguments)) or {"forecast": "clear"},
        )
        registry.register(
            "get_calendar_events", "Calendar lookup", {"type": "object"},
            lambda **arguments: calls.append(("calendar", arguments)) or {
                "timezone": "America/Toronto", "events": [{"title": "Daily Exercise"}],
            },
        )
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [
                tool_chunk(index=0, call_id="weather_1", name="get_weather", arguments='{"location":"Toronto","date":"tomorrow"}'),
                tool_chunk(index=1, call_id="calendar_1", name="get_calendar_events", arguments='{"start_datetime":"tomorrow evening","end_datetime":"tomorrow evening"}'),
            ],
            [text_chunk("Exercise is 5 to 6; consider a walk after 6.")],
        ]
        client = OpenRouterChatClient(
            Settings(api_key="test-key", model="test/model"), registry, calendar_enabled=True
        )
        turn = ChatTurn()
        result = "".join(client.stream_reply([
            {"role": "user", "content": "Suggest something tomorrow evening."}
        ], turn=turn))
        self.assertIn("walk after 6", result)
        self.assertEqual([name for name, _ in calls], ["weather", "calendar"])
        self.assertEqual([message["role"] for message in turn.messages], [
            "assistant", "tool", "tool", "assistant",
        ])
        self.assertEqual(completion.call_args_list[1].kwargs["messages"][-2:], turn.messages[1:3])
        self.assertNotIn("test-key", json.dumps(turn.messages))

    @patch("services.llm_client.OpenAI")
    def test_places_weather_calendar_combined_and_follow_up(self, openai_class):
        calls = []
        registry = ToolRegistry()
        for name, result in (
            ("get_weather", {"forecast": "clear"}),
            ("get_calendar_events", {"events": [{"title": "Daily Exercise"}]}),
            ("search_places", {"places": [{
                "name": "City Museum", "address": "100 Main St",
                "google_maps_url": "https://maps.google.com/?cid=one",
            }]}),
        ):
            registry.register(
                name, name, {"type": "object"},
                lambda result=result, name=name, **arguments: calls.append((name, arguments)) or result,
            )
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [
                tool_chunk(index=0, call_id="weather_1", name="get_weather", arguments='{"location":"Toronto","date":"tomorrow"}'),
                tool_chunk(index=1, call_id="calendar_1", name="get_calendar_events", arguments='{"start_datetime":"tomorrow evening","end_datetime":"tomorrow evening"}'),
                tool_chunk(index=2, call_id="places_1", name="search_places", arguments='{"query":"museums","location":"Toronto"}'),
            ],
            [text_chunk("After exercise, visit [City Museum](https://maps.google.com/?cid=one) at 100 Main St.")],
            [tool_chunk(call_id="places_2", name="search_places", arguments='{"query":"other museums","location":"Toronto"}')],
            [text_chunk("Another option is nearby.")],
        ]
        client = OpenRouterChatClient(
            Settings(api_key="test-key", model="test/model"), registry,
            calendar_enabled=True, places_enabled=True,
        )
        first_user = {"role": "user", "content": "Suggest somewhere tomorrow evening in Toronto."}
        first = ChatTurn()
        self.assertIn("City Museum", "".join(client.stream_reply([first_user], turn=first)))
        self.assertEqual([name for name, _ in calls[:3]], [
            "get_weather", "get_calendar_events", "search_places",
        ])
        self.assertEqual([message["role"] for message in first.messages], [
            "assistant", "tool", "tool", "tool", "assistant",
        ])
        self.assertTrue(all(json.loads(message["content"])["ok"] for message in first.messages if message["role"] == "tool"))

        follow_up = {"role": "user", "content": "Any other museums?"}
        second = ChatTurn()
        list(client.stream_reply([first_user, *first.messages, follow_up], turn=second))
        self.assertEqual(calls[-1], ("search_places", {"query": "other museums", "location": "Toronto"}))
        self.assertEqual(completion.call_args_list[2].kwargs["messages"][2], first.messages[0])
        self.assertEqual(completion.call_args_list[2].kwargs["tool_choice"], "auto")

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model",
            "GOOGLE_CALENDAR_ID": "public-calendar", "GOOGLE_CALENDAR_API_KEY": "test-calendar-key",
        },
    )
    @patch("services.calendar_service.CalendarService.get_calendar_events")
    @patch("services.llm_client.OpenAI")
    def test_ui_calendar_toggle_status_and_history(self, openai_class, calendar):
        calendar.return_value = {"timezone": "America/Toronto", "events": []}
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [text_chunk("I can suggest an activity, but have not checked your calendar.")],
            [tool_chunk(call_id="calendar_1", name="get_calendar_events", arguments='{"start_datetime":"tomorrow 17:30","end_datetime":"tomorrow 17:31"}')],
            [text_chunk("Your calendar has no events at 5:30 PM.")],
        ]
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
        self.assertFalse(app.toggle[1].value)
        app.chat_input[0].set_value("Suggest something tomorrow evening.").run(timeout=15)
        self.assertEqual(
            [tool["function"]["name"] for tool in completion.call_args_list[0].kwargs["tools"]],
            ["get_weather"],
        )
        calendar.assert_not_called()

        app.toggle[1].set_value(True).run(timeout=15)
        app.chat_input[0].set_value("Am I free tomorrow at 5:30 PM?").run(timeout=15)
        self.assertEqual(
            [tool["function"]["name"] for tool in completion.call_args_list[1].kwargs["tools"]],
            ["get_weather", "get_calendar_events"],
        )
        calendar.assert_called_once_with(start_datetime="tomorrow 17:30", end_datetime="tomorrow 17:31")
        self.assertTrue(any("Calendar lookup complete" in caption.value for caption in app.caption))
        self.assertEqual([message["role"] for message in app.session_state.messages][-4:], [
            "user", "assistant", "tool", "assistant",
        ])

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model",
            "GOOGLE_PLACES_API_KEY": "test-places-key",
        },
    )
    @patch("services.places_service.PlacesService.search_places")
    @patch("services.llm_client.OpenAI")
    def test_ui_places_toggle_status_and_history(self, openai_class, places):
        places.return_value = {"places": [{
            "name": "City Museum", "address": "100 Main St",
            "google_maps_url": "https://maps.google.com/?cid=one",
        }]}
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [text_chunk("Tell me your city and I can look for museums.")],
            [tool_chunk(call_id="places_1", name="search_places", arguments='{"query":"museums","location":"Toronto"}')],
            [text_chunk("Try [City Museum](https://maps.google.com/?cid=one), 100 Main St.")],
        ]
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
        self.assertFalse(app.toggle[2].value)
        app.chat_input[0].set_value("Find museums in Toronto.").run(timeout=15)
        self.assertEqual([tool["function"]["name"] for tool in completion.call_args_list[0].kwargs["tools"]], ["get_weather"])
        places.assert_not_called()

        app.toggle[2].set_value(True).run(timeout=15)
        app.chat_input[0].set_value("Find museums in Toronto.").run(timeout=15)
        self.assertIn("search_places", [
            tool["function"]["name"] for tool in completion.call_args_list[1].kwargs["tools"]
        ])
        places.assert_called_once_with(query="museums", location="Toronto")
        self.assertTrue(any("Places lookup complete" in caption.value for caption in app.caption))
        self.assertEqual([message["role"] for message in app.session_state.messages][-4:], [
            "user", "assistant", "tool", "assistant",
        ])

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "router-key", "OPENROUTER_MODEL": "router/model",
            "NVIDIA_API_KEY": "nvidia-key", "NVIDIA_MODEL": "nvidia/model",
            "CLOUDFLARE_API_TOKEN": "cf-token", "CLOUDFLARE_MODEL": "@cf/model",
            "CLOUDFLARE_ACCOUNT_ID": "account123",
            "GROQ_API_KEY": "groq-key", "GROQ_MODEL": "groq/model",
            "GEMINI_API_KEY": "gemini-key", "GEMINI_MODEL": "gemini-model",
        },
    )
    @patch("services.llm_client.OpenAI")
    def test_ui_provider_switch_keeps_conversation(self, openai_class):
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [text_chunk("Hello from NVIDIA")], [text_chunk("Hello from Cloudflare")],
            [text_chunk("Hello from Groq")], [text_chunk("Hello from Gemini")],
        ]
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
        self.assertEqual(app.selectbox[0].value, "openrouter")

        app.selectbox[0].set_value("nvidia").run(timeout=15)
        app.chat_input[0].set_value("Hello").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(completion.call_args_list[0].kwargs["model"], "nvidia/model")
        self.assertEqual(openai_class.call_args_list[0].kwargs["base_url"], "https://integrate.api.nvidia.com/v1")

        app.selectbox[0].set_value("cloudflare").run(timeout=15)
        app.chat_input[0].set_value("Follow up").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(completion.call_args_list[1].kwargs["model"], "@cf/model")
        self.assertEqual(openai_class.call_args_list[1].kwargs["base_url"],
                         "https://api.cloudflare.com/client/v4/accounts/account123/ai/v1")
        self.assertEqual(completion.call_args_list[1].kwargs["messages"][1:], [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello from NVIDIA"},
            {"role": "user", "content": "Follow up"},
        ])

        app.selectbox[0].set_value("groq").run(timeout=15)
        app.chat_input[0].set_value("Continue").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(completion.call_args_list[2].kwargs["model"], "groq/model")
        self.assertEqual(openai_class.call_args_list[2].kwargs["base_url"],
                         "https://api.groq.com/openai/v1")

        app.selectbox[0].set_value("gemini").run(timeout=15)
        app.chat_input[0].set_value("Continue again").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(completion.call_args_list[3].kwargs["model"], "gemini-model")
        self.assertEqual(openai_class.call_args_list[3].kwargs["base_url"],
                         "https://generativelanguage.googleapis.com/v1beta/openai/")

    @patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model"},
    )
    @patch("services.llm_client.OpenAI")
    def test_ui_weather_toggle_excludes_weather_tool(self, openai_class):
        completion = openai_class.return_value.chat.completions.create
        completion.return_value = [text_chunk("I cannot check live weather right now.")]
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
        self.assertTrue(app.toggle[0].value)
        app.toggle[0].set_value(False).run(timeout=15)
        app.chat_input[0].set_value("What is the weather in Toronto?").run(timeout=15)
        self.assertFalse(app.exception)
        self.assertNotIn("tools", completion.call_args.kwargs)
        self.assertIn("Weather integration is disabled", completion.call_args.kwargs["messages"][0]["content"])

    @patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test/model"},
    )
    @patch("services.weather_service.WeatherService.get_weather")
    @patch("services.llm_client.OpenAI")
    def test_ui_keeps_tool_messages_and_shows_status(self, openai_class, weather):
        weather.return_value = {"location": {"name": "Toronto"}, "date": "2026-10-13"}
        completion = openai_class.return_value.chat.completions.create
        completion.side_effect = [
            [tool_chunk(call_id="call_weather", name="get_weather", arguments='{"location":"Toronto"}')],
            [text_chunk("Toronto is sunny.")],
        ]
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
        app.chat_input[0].set_value("What's the weather in Toronto?").run(timeout=15)

        self.assertEqual(
            [message["role"] for message in app.session_state.messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(app.session_state.messages[2]["tool_call_id"], "call_weather")
        self.assertEqual(app.session_state.messages[3]["content"], "Toronto is sunny.")
        self.assertTrue(any("Weather lookup complete" in caption.value for caption in app.caption))
        weather.assert_called_once_with(location="Toronto")

        app.button[0].click().run(timeout=15)
        self.assertEqual(app.session_state.messages, [])


if __name__ == "__main__":
    unittest.main()
