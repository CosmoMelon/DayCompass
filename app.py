"""Minimal Streamlit chat interface."""

from importlib import reload
from inspect import signature
from pathlib import Path

import streamlit as st

import config
from services import llm_client, tool_registry

# Streamlit reruns app.py without necessarily refreshing imported modules.
# Refresh only modules whose interfaces predate provider selection.
if not hasattr(config.Settings, "base_url") or "gemini" not in config.PROVIDER_LABELS:
    reload(config)
registry_reloaded = "weather_enabled" not in signature(tool_registry.create_default_registry).parameters
if registry_reloaded:
    reload(tool_registry)
if (registry_reloaded or not hasattr(llm_client, "ChatClient") or
        not hasattr(llm_client, "ChatTurn") or
        getattr(llm_client, "CHAT_CLIENT_VERSION", 0) < 4):
    reload(llm_client)

ChatMessage = llm_client.ChatMessage
ChatServiceError = llm_client.ChatServiceError
ChatTurn = llm_client.ChatTurn
ChatClient = llm_client.ChatClient


st.set_page_config(page_title="DayCompass", page_icon="💬")
logo_variant = "dark" if st.context.theme.type == "dark" else "light"
logo_path = Path(__file__).resolve().parent / "assets" / f"daycompass-logo-{logo_variant}.svg"
st.image(logo_path, width=400, alt="DayCompass")

provider = st.sidebar.selectbox(
    "LLM Provider",
    options=list(config.PROVIDER_LABELS),
    format_func=lambda name: config.PROVIDER_LABELS[name],
    key="llm_provider",
)

try:
    settings = config.load_settings(provider)
except config.ConfigurationError as exc:
    st.error(str(exc))
    st.stop()

weather_enabled = st.sidebar.toggle("Enable Weather", value=True)
calendar_configured = bool(settings.calendar_id and settings.calendar_api_key)
calendar_enabled = st.sidebar.toggle(
    "Enable Calendar Integration", value=False, disabled=not calendar_configured
)
if not calendar_configured:
    st.sidebar.caption("Set GOOGLE_CALENDAR_ID and GOOGLE_CALENDAR_API_KEY to enable.")
st.sidebar.markdown("[Open Google Calendar ↗](https://calendar.google.com/calendar/embed?src=36409f8e63cc1eb76d1a75dc8257addf469a8d45a12226e1552608c149ee5ac7%40group.calendar.google.com&ctz=America%2FToronto)")
places_configured = bool(settings.places_api_key)
places_enabled = st.sidebar.toggle("Enable Places", value=False, disabled=not places_configured)
if not places_configured:
    st.sidebar.caption("Set GOOGLE_PLACES_API_KEY to enable.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("New Chat"):
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] in ("user", "assistant") and not message.get("tool_calls") and message.get("content"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask anything"):
    user_message: ChatMessage = {"role": "user", "content": prompt}
    with st.chat_message("user"):
        st.markdown(prompt)

    # Commit a turn only after its response completes, so failed or partial
    # replies never become context for the next request.
    conversation = [*st.session_state.messages, user_message]
    turn = ChatTurn()
    with st.chat_message("assistant"):
        status = st.empty()
        tools_used: set[str] = set()

        def show_tool_status(name: str, arguments: dict) -> None:
            if name == "get_weather":
                tools_used.add(name)
                city = arguments.get("location") or "the requested city"
                status.caption(f"Fetching weather for {city}...")
            elif name == "get_calendar_events":
                tools_used.add(name)
                status.caption("Checking calendar...")
            elif name == "search_places":
                tools_used.add(name)
                status.caption("Searching places...")

        try:
            st.write_stream(
                ChatClient(
                    settings, weather_enabled=weather_enabled,
                    calendar_enabled=calendar_enabled, places_enabled=places_enabled
                ).stream_reply(
                    conversation, turn=turn, on_tool_start=show_tool_status
                )
            )
        except ChatServiceError as exc:
            status.empty()
            st.error(str(exc))
        else:
            completed = [
                label for name, label in (
                    ("get_weather", "Weather"),
                    ("get_calendar_events", "calendar"),
                    ("search_places", "places"),
                ) if name in tools_used
            ]
            if len(completed) == 1:
                status.caption(f"{completed[0].capitalize()} lookup complete")
            elif len(completed) == 2:
                status.caption(f"{completed[0]} and {completed[1]} lookups complete")
            elif completed:
                status.caption("Weather, calendar, and places lookups complete")
            st.session_state.messages.extend([user_message, *turn.messages])
