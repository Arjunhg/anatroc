# Anatroc — Real-Time Multimodal Learning Assistant

An AI-powered interactive learning assistant that observes users via screen share, tracks on-screen content through OCR, and provides real-time verbal feedback and visual overlays — all powered by **Amazon Nova** on AWS.

Users share their screen (code, documents, diagrams, simulations), speak naturally, and receive live coaching through Nova Sonic speech-to-speech. The assistant remembers what was on screen, what was said, and can generate architecture diagrams on command — creating an active feedback loop for hands-on learning.

---

## Architecture

```
                        ┌─────────────────┐
                        │   Browser UI    │
                        │  (React + Vite) │
                        └───────┬─────────┘
                                │
                    Stream Video SDK (call/mic/cam/screen)
                    WebSocket (Sonic audio + overlay events)
                    REST API (frames, queries, diagrams)
                                │
                        ┌───────▼─────────┐
                        │  FastAPI Server  │
                        │ (token_server.py)│
                        └───┬───┬───┬─────┘
                            │   │   │
                ┌───────────┘   │   └───────────┐
                │               │               │
        ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
        │  Nova Sonic  │ │  Nova Lite  │ │   Nova       │
        │  (voice AI)  │ │  (reasoning │ │  Embeddings  │
        │  speech ↔    │ │   + OCR +   │ │  (semantic   │
        │  speech      │ │   diagrams) │ │   memory)    │
        └──────────────┘ └─────┬──────┘ └──────┬───────┘
                               │               │
                        ┌──────▼───────────────▼──┐
                        │   Aurora PostgreSQL      │
                        │   (pgvector embeddings)  │
                        └─────────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   Redis (cache)     │
                        │   retrieval + TTL   │
                        └─────────────────────┘
```

### Nova Models Used

| Model | Role | How It's Used |
|---|---|---|
| **Nova 2 Sonic** | Real-time speech-to-speech | Bidirectional audio WebSocket — user speaks, assistant responds with voice. Supports barge-in (interrupt assistant mid-sentence). |
| **Nova 2 Lite** | Reasoning + OCR + Diagrams | Analyzes screen frames, extracts OCR text, answers questions about on-screen content, generates Mermaid diagrams via Bedrock `converse()`. |
| **Nova Multimodal Embeddings** | Semantic memory | Converts screen OCR and conversation turns into 1024-dim vectors. Stored in Aurora pgvector for similarity search retrieval. |

---

## How It Works

### Web App Mode (primary)

1. **User joins a session** in the browser. Stream Video SDK handles the call UI (camera, mic, screen share controls).
2. **Screen share starts** → frontend captures a JPEG frame every 3–5 seconds and sends it to the backend.
3. **OCR extraction** → Nova Lite reads the frame and extracts visible text. The text is embedded and indexed into Aurora pgvector.
4. **Nova Sonic voice session** → user clicks "Start Sonic Voice" to open a bidirectional speech WebSocket. User speaks naturally, assistant responds with voice.
5. **Context grounding** → when the user asks memory-seeking questions ("what was I reading earlier?", "summarize what I did"), the backend retrieves relevant past context from Aurora (Redis-cached) and injects it into the Sonic conversation.
6. **Diagram overlay** → user says "generate architecture diagram" and a Mermaid diagram appears as a floating overlay on the call screen. Controllable by voice: move, minimize, expand, export, remove.

### CLI Mode

The same backend pipeline is accessible via CLI for offline/batch use:

```bash
# One-shot screen frame analysis
uv run python main.py analyze-screen --frame-path ./screenshot.png --prompt "Explain this"

# Diagram generation from memory
uv run python main.py diagram --prompt "Draw the auth flow"

# Interactive frame-by-frame analysis
uv run python main.py run

# Direct Nova Sonic console mode (local mic + speaker, barge-in)
uv run python main.py voice-chat
```

---

## What You'll See When Running

### Lobby Screen
Join form with user ID, display name, and call ID fields. Click "Join Session" to enter the call.

### Call Screen
- **Video tiles** — local camera view and any remote participants
- **Nova Sync panel** (right sidebar) with:
  - Screen share frame counter and context hit count
  - Live analysis prompt input (optional prompt applied during screen share ingestion)
  - Latest OCR text extracted from screen
  - **Start/Stop Sonic Voice** button
  - Sonic user and assistant transcript displays
  - **Ask Nova** text input for manual questions (uses retrieval + Nova Lite)
  - **Diagram prompt** and "Generate Mermaid" button
  - Mermaid output display

### Diagram Overlay
Floating non-blocking card rendered via Mermaid.js. Supports:
- **Voice commands**: "generate architecture diagram", "move overlay to top right", "minimize overlay", "expand overlay", "export diagram", "remove overlay"
- **Manual controls**: buttons for TL / TR / BL / BR / Center positioning, minimize/expand, export (clipboard or file download), and remove

### Sonic Voice Session
- Frontend captures microphone audio, downsamples to 16 kHz PCM16, and streams via WebSocket
- Silence detection triggers turn-end after ~1.2s of quiet
- Assistant audio chunks are played back through the browser speaker
- Barge-in: speaking while assistant is talking interrupts playback and clears buffered audio
- Playback mic suppression prevents echo feedback (~450ms mute after assistant finishes)

---

## Data Flow Per Interaction

### Screen Frame Ingestion

```
Browser captures screen frame (JPEG)
        ↓
POST /api/frame/ingest
        ↓
Nova Lite OCR extracts text
        ↓
Nova Embeddings creates 1024-dim vector
        ↓
Aurora pgvector stores (content, vector, metadata)
        ↓
Session state updated with latest OCR
        ↓
If Sonic is active, OCR pushed to voice context
```

### Voice Turn (Sonic)

```
User speaks into mic
        ↓
PCM16 audio chunks → WebSocket → Bedrock Sonic stream
        ↓
Silence detected → audio_turn_end
        ↓
Backend: persist user transcript (Aurora)
        ↓
Backend: check overlay intent → generate / clear / move / etc.
        ↓
Backend: build retrieval context (if memory-seeking question)
    ├── Redis cache hit → use cached results
    └── Cache miss → embed question → Aurora similarity search → cache results
        ↓
Inject context into Sonic (OCR + retrieved memory)
        ↓
Sonic responds with audio + transcript
        ↓
Backend: persist assistant transcript (Aurora)
        ↓
Frontend: plays audio, shows transcripts, renders overlay if generated
```

### Manual Text Query

```
User types question in "Ask Nova" input
        ↓
POST /api/voice/query
        ↓
Retrieval: Redis → Aurora pgvector fallback
        ↓
Nova Lite reasoning (prompt + context + latest frame)
        ↓
Response displayed in panel
```

---

## Project Structure

```
anatroc/
├── token_server.py              # FastAPI API server (primary runtime)
├── main.py                      # CLI entrypoint
├── .env                         # All API keys and config (never commit)
├── pyproject.toml               # Python dependencies (managed by uv)
│
├── vision_learning_assistant/
│   ├── config.py                # AssistantConfig — env validation + defaults
│   ├── nova_pipeline.py         # VisionLearningPipeline — dependency wiring + lifecycle
│   ├── nova/
│   │   ├── bedrock_client.py    # NovaBedrockClient — Lite reasoning, embeddings, Mermaid
│   │   └── sonic_stream.py      # NovaSonicConsoleSession + NovaSonicWebSocketSession
│   ├── services/
│   │   ├── context_memory.py    # ContextMemoryService — retrieval + indexing orchestration
│   │   └── screen_analysis.py   # ScreenAnalysisService — frame OCR + analysis
│   └── storage/
│       ├── aurora_store.py      # AuroraVectorStore — pgvector insert + similarity search
│       └── redis_cache.py       # RedisCache — JSON get/set with TTL
│
├── frontend1/                   # React + Vite + TypeScript
│   ├── src/
│   │   ├── App.tsx              # Lobby ↔ Call routing, Stream Video setup
│   │   ├── stream.ts            # Token fetch, URL builders, WebSocket URL
│   │   ├── novaApi.ts           # REST API client (session, ingest, query, diagram)
│   │   └── components/
│   │       ├── LobbyScreen.tsx  # Join form
│   │       ├── CallScreen.tsx   # Call UI, screen capture, Sonic mic/playback, overlay
│   │       ├── DiagramOverlay.tsx # Mermaid overlay renderer
│   │       └── Controls.tsx     # Leave/camera/mic buttons
│   ├── vite.config.ts           # Dev proxy /api/* → token_server.py
│   └── package.json
```

---

## API Endpoints

### Primary REST

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/token?user_id=<id>` | Stream Video JWT token |
| `POST` | `/api/session/start` | Start a backend session |
| `POST` | `/api/session/stop` | Stop and clean up session |
| `GET` | `/api/session/{session_id}` | Get session state |
| `POST` | `/api/frame/ingest` | Ingest a screen frame (OCR + index) |
| `POST` | `/api/voice/query` | Text question with retrieval context |
| `POST` | `/api/diagram` | Generate Mermaid diagram |

### WebSocket

| Path | Description |
|---|---|
| `WS /ws/sonic/{session_id}` | Nova Sonic speech-to-speech with overlay events |

**Sonic WebSocket events (server → client):**

| Event | Purpose |
|---|---|
| `ready` | Sonic session initialized |
| `transcript` | User or assistant text transcript chunk |
| `assistant_audio` | PCM16 audio chunk for playback |
| `assistant_interrupted` | Barge-in detected, clear playback queue |
| `overlay_diagram` | Mermaid diagram generated, render overlay |
| `overlay_clear` | Remove active overlay |
| `overlay_control` | Move / minimize / expand overlay |
| `overlay_export` | Export Mermaid text (clipboard/download) |
| `overlay_error` | Diagram generation failed |

**Sonic WebSocket events (client → server):**

| Event | Purpose |
|---|---|
| `audio_chunk` | PCM16 microphone audio |
| `audio_turn_end` | End of user speech turn |
| `stop` | Close Sonic session |
| `overlay_clear` | Request overlay removal |
| `overlay_control` | Request overlay move/minimize/expand |

### Legacy Compatibility

| Method | Path |
|---|---|
| `POST` | `/start-agent` |
| `POST` | `/stop-agent` |

---

## Environment Variables

### Required

```dotenv
# AWS
AWS_REGION=us-east-1

# Nova Models
NOVA_LITE_MODEL_ID=amazon.nova-2-lite-v1:0
NOVA_SONIC_MODEL_ID=amazon.nova-2-sonic-v1:0
NOVA_EMBED_MODEL_ID=amazon.nova-2-multimodal-embeddings-v1:0

# Storage
DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<db>
REDIS_URL=redis://localhost:6379

# Stream Video SDK
STREAM_API_KEY=<your_stream_key>
STREAM_API_SECRET=<your_stream_secret>
```

> **Note:** Nova Lite model IDs starting with `amazon.nova-` are automatically normalized to `us.amazon.nova-...` inference profile IDs at startup. Nova Sonic model IDs are used as-is (bidirectional streaming expects direct IDs).

### Optional Tuning

```dotenv
# Capture and session
SCREEN_CAPTURE_INTERVAL=3              # seconds between frame captures (default: 4)
SESSION_TIMEOUT=3600                    # max session duration in seconds

# Embedding
NOVA_EMBEDDING_DIMENSION=1024          # vector dimension (256, 384, 1024, or 3072)
NOVA_TEXT_TRUNCATION_MODE=END          # truncation mode: START, END, NONE

# Storage toggles
ENABLE_AURORA_WRITES=true              # disable to simulate writes without DB
ENABLE_REDIS_CACHE=true                # disable to skip Redis entirely
REDIS_TTL_SECONDS=1800                 # retrieval cache TTL (30 min default)
MAX_RETRIEVAL_RESULTS=5                # max context hits for text-path retrieval

# Sonic context grounding
ENABLE_SONIC_CONTEXT_RETRIEVAL=true    # enable retrieval on voice turns
SONIC_CONTEXT_MAX_RESULTS=4            # max retrieved hits per voice turn
SONIC_CONTEXT_MAX_CHARS_PER_HIT=350    # max chars per retrieved chunk
SONIC_CONTEXT_MAX_TOTAL_CHARS=1200     # max total injected context chars
SONIC_CONTEXT_RETRIEVAL_COOLDOWN_SECONDS=8  # min seconds between retrievals

# Server
TOKEN_SERVER_HOST=127.0.0.1
TOKEN_SERVER_PORT=8001
TOKEN_SERVER_CORS_ALLOW_ORIGINS=*
TOKEN_SERVER_CORS_ALLOW_HEADERS=Content-Type,Authorization
```

### Frontend Environment (frontend1/.env.production)

```dotenv
# Set only when frontend is hosted separately from backend
VITE_TOKEN_API_URL=https://your-backend-domain.com

# Frame sampling cadence (seconds)
VITE_SCREEN_CAPTURE_INTERVAL_SECONDS=3
```

---

## Local Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker (for Redis)
- AWS credentials configured (environment variables or `~/.aws/credentials`)

### 1. Install Python dependencies

```bash
uv sync
```

### 2. Start Redis

```bash
docker run --name nova-redis -p 6379:6379 -d redis:7-alpine
```

### 3. Start the API server

```bash
uv run python token_server.py
```

The server starts on `http://127.0.0.1:8001`. FastAPI docs available at `/docs`.

### 4. Start the frontend dev server

```bash
cd frontend1
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`. The Vite dev proxy routes `/api/*` to the backend automatically.

### 5. Open the app

Navigate to `http://localhost:5173` in Chrome (recommended for WebRTC + AudioContext support).

### Production Build (optional)

```bash
cd frontend1
npm run build
```

The built files land in `frontend1/dist/`. The backend automatically serves this directory at `http://127.0.0.1:8001/` when it exists.

---

## Cost Control

The system is designed with cost guardrails for hackathon-safe usage:

| Strategy | How |
|---|---|
| **Frame sampling interval** | Frames captured every 3–5 seconds, not every frame |
| **Intent-gated retrieval** | Sonic retrieval only fires on memory-seeking prompts (keywords like "summarize", "earlier", "recap") |
| **Retrieval cooldown** | Min 8 seconds between Aurora retrievals to prevent spikes |
| **Bounded context** | Max 4 hits, 350 chars each, 1200 chars total per voice turn |
| **Redis-first caching** | Retrieval results cached with 30-min TTL; repeat questions skip Aurora |
| **Session timeout** | Auto-stop idle sessions (configurable, default 30 min) |
| **Optional Aurora writes** | Can disable writes entirely with `ENABLE_AURORA_WRITES=false` |
| **Optional Redis** | System continues without Redis if unavailable (graceful degradation) |

---

## Storage Architecture

### Aurora PostgreSQL (pgvector)

Stores all semantic memory as vector embeddings:

```sql
-- Table schema (auto-created on startup)
CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    content TEXT,
    source_type TEXT,          -- screen_share_ocr, screen_analysis,
                               -- sonic_user_turn, sonic_assistant_turn
    embedding VECTOR(1024),
    metadata JSONB,            -- session_id, run_id, timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- IVFFlat index for fast similarity search
CREATE INDEX embeddings_vector_idx
ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

Similarity search uses cosine distance: `1 - (embedding <=> query_vector)`.

### Redis

Short-lived cache layer for retrieval results:
- Key format: `context:retrieval:<normalized_hash>`
- TTL: 30 minutes (configurable)
- Cache keys are normalized (lowercase, stripped punctuation, collapsed whitespace) so similar queries hit the same entry
- If Redis is unavailable, the system continues without caching

---

## Resilience

| Component | Behavior on Failure |
|---|---|
| **Redis unavailable** | System continues without cache. Logged as warning at startup. |
| **Aurora schema missing** | Auto-creates on startup (best-effort). If it fails, logged as warning. |
| **Aurora writes disabled** | Synthetic UUIDs returned; pipeline runs normally without persistence. |
| **Bedrock API error** | Validation errors raised to caller. No silent failures. |
| **Screen share stops** | Frame ingestion pauses. Sonic voice continues independently. |
| **Sonic WebSocket drops** | Frontend shows "Disconnected". Session state preserved; overlay restored on reconnect. |

---

## Voice Commands (Demo Guide)

During a Sonic voice session, say:

| Command | What Happens |
|---|---|
| *"What is on my screen?"* | Assistant describes current screen content using latest OCR |
| *"Explain what I am reading"* | Detailed explanation of on-screen text |
| *"Summarize what I have been doing"* | Retrieves past OCR + voice memory for summary |
| *"What was I reading earlier?"* | Historical retrieval from Aurora |
| *"Generate architecture diagram"* | Mermaid overlay appears on screen |
| *"Move overlay to top right"* | Repositions the overlay |
| *"Minimize overlay"* / *"Expand overlay"* | Collapses or restores the diagram |
| *"Export diagram"* | Copies Mermaid to clipboard (or downloads .mmd file) |
| *"Remove overlay"* | Clears the diagram from screen |

---

## Notes

- Nova Sonic console mode (`voice-chat` CLI) uses local PyAudio for mic/speaker — requires audio hardware.
- Web app mode uses browser WebRTC/AudioContext — no PyAudio needed.
- Aurora operations on Windows use blocking psycopg calls in worker threads to avoid `ProactorEventLoop` async incompatibility.
- Model ID normalization happens automatically: `amazon.nova-2-lite-v1:0` → `us.amazon.nova-2-lite-v1:0` for Converse API compatibility.
- Overlay diagrams are session-persistent and restored when Sonic reconnects within the same session.
