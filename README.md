# Minimal AI Chat

A small Streamlit chat app that streams responses through the OpenAI Python SDK. Choose OpenRouter, NVIDIA, Cloudflare Workers AI, Groq, or Gemini in the sidebar. Enable Weather, Calendar, and Places tools independently in the sidebar. Conversation history, including tool calls and results, is kept in the current Streamlit session. **New Chat** clears that history.

## Run locally

Requires Python 3.10 or newer and credentials for at least one LLM provider.

From the `chat-app` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy with `cp .env.example .env` instead.

Edit `.env` with the credentials and a tool-capable model for each provider you plan to use. Existing OpenRouter settings continue to work:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_tool_capable_model
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_MODEL=your_nvidia_tool_capable_model
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_MODEL=your_cloudflare_tool_capable_model
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_tool_capable_model
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_tool_capable_model
GOOGLE_CALENDAR_ID=your_public_calendar_id
GOOGLE_CALENDAR_API_KEY=your_google_calendar_api_key
GOOGLE_PLACES_API_KEY=your_google_places_api_key
```

Then start the app:

```powershell
streamlit run app.py
```

Open the local URL that Streamlit prints (usually `http://localhost:8501`). Choose **LLM Provider** in the sidebar. OpenRouter is the default. The app validates credentials only for the selected provider; configure only the providers you plan to use. Keep `.env` private; it is ignored by Git. Existing process environment variables take precedence over `.env` values.

The provider endpoints are [OpenRouter](https://openrouter.ai/docs/quickstart), [NVIDIA's hosted API](https://docs.api.nvidia.com/nim/docs/api-quickstart), [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/), [Groq](https://console.groq.com/docs/openai), and [Gemini](https://ai.google.dev/gemini-api/docs/openai/). Select a model that supports Chat Completions, streaming, and local tool calling. Cloudflare model IDs use the `@cf/` prefix. Gemini uses an API key from Google AI Studio. The same conversation and tool history is sent on each request, including when you switch providers; use **New Chat** if you want a fresh conversation. Gemini tool-call thought signatures are preserved for its multi-step tool responses.

**Enable Weather** is on by default. Turn it off to remove `get_weather` from the model's available tools. When enabled, try asking "What's the weather in Toronto?", then "How about tomorrow?" or "Will it rain?" The model decides whether to call Open-Meteo. For ambiguous place names, specify a region or country (for example, `London, Ontario`). Forecast dates are limited to the range returned by Open-Meteo, currently up to 16 days. Open-Meteo needs no API key for this public endpoint.

To check scheduling conflicts, make the Google Calendar public, enable the Google Calendar API for the API key's project, and set both calendar values in `.env`. Turn on **Enable Calendar Integration** in the sidebar. The model can then look up events for a requested period, including recurring instances, and combine the results with weather when appropriate. Try "Am I free tomorrow at 5:30 PM?" or "Suggest something to do tomorrow evening." The calendar tool is absent while the toggle is off. Relative dates and times are resolved in the calendar's timezone. The API key stays in the server-side service and is never sent to the model.

To search for venues, enable [Places API (New)](https://developers.google.com/maps/documentation/places/web-service/text-search) for the Google Cloud project and set `GOOGLE_PLACES_API_KEY` in `.env`. Turn on **Enable Places** in the sidebar. Try "Find badminton courts in Toronto" or "Suggest somewhere to go tomorrow evening in Toronto." The model can combine Places, Weather, and Calendar results, then include addresses and Google Maps links in its answer. For "near me" without a known city or coordinates, it asks for a location. Basic searches request only place identity, address, coordinates, and Maps URL; rating and opening hours are optional fields that can affect Google Places billing.

## Layout

- `app.py` renders the chat and owns session history.
- `config.py` loads settings and selects the provider endpoint and credentials.
- `services/llm_client.py` streams provider responses, executes model-requested tools, and records tool messages.
- `services/tool_registry.py` defines and dispatches tools. Register another handler here without changing the chat loop.
- `services/weather_service.py` calls Open-Meteo's geocoding and forecast APIs and returns structured weather data.
- `services/calendar_service.py` reads public Google Calendar events, handles pagination, and normalizes event times.
- `services/places_service.py` calls Google Places Text Search (New) with bounded results and field masks.

The selected provider's credentials and model are required. If either is missing, the app shows a setup error. Responses may take up to 60 seconds between network reads before timing out. No conversation history is persisted after the Streamlit session ends.

Run the tests (external API responses are mocked):

```powershell
python -m unittest discover -s tests -v
```
