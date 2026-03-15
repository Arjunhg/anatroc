# Vision Learning Assistant

An AI learning assistant for real-time screen-share and voice guidance using Amazon Nova on AWS.

## System Flow

```text
User screen share + voice
        |
        v
Frontend (browser)
 - captures screen frame every N seconds
 - sends frame to backend API
 - streams microphone PCM chunks over WebSocket
        |
        v
Nova API Server (token_server.py)
 - session manager
 - frame OCR extraction via Nova Lite
 - embedding + indexing (Aurora pgvector)
 - retrieval cache (Redis)
 - Nova Sonic bidirectional speech bridge
 - synchronized reasoning response (Nova Lite for text path)
        |
        v
Frontend playback
 - Sonic transcript stream
 - assistant audio chunks from Nova Sonic
```

## What Is Implemented

1. Nova-first backend pipeline:
- Nova Lite (`converse`) for screen reasoning and OCR extraction.
- Nova multimodal embeddings (`invoke_model`) for indexing and retrieval.
- Aurora pgvector storage for semantic memory.
- Redis caching for short-lived retrieval responses.

2. Live session sync:
- Frame ingestion endpoint stores latest frame + OCR per session.
- Sonic websocket session receives OCR context updates from ingestion.
- Voice/query text endpoint uses that same session context.
- Diagram generation endpoint returns Mermaid for overlay rendering.

3. Frontend wiring (`frontend1/` Stream app):
- Stream Video call UI is used for join/camera/mic/screen-share controls.
- On join, frontend starts Nova backend session with the same call ID.
- Frames are sent to `/api/frame/ingest`.
- Voice/manual queries are sent to `/api/voice/query`.
- Responses are shown and spoken back in the browser.

4. Backward compatibility:
- `GET /token` still issues Stream Video JWT tokens.
- `POST /start-agent` and `POST /stop-agent` are mapped to session start/stop behavior.

5. Aurora insert stability on Windows:
- Aurora operations run through blocking psycopg calls in worker threads.
- This avoids `ProactorEventLoop` async incompatibility from psycopg async connections.

## API Endpoints

Primary:
- `GET /health`
- `GET /token?user_id=<id>`
- `POST /api/session/start`
- `POST /api/session/stop`
- `GET /api/session/{session_id}`
- `POST /api/frame/ingest`
- `POST /api/voice/query`
- `POST /api/diagram`
- `WS  /ws/sonic/{session_id}`

Legacy compatibility:
- `POST /start-agent`
- `POST /stop-agent`

## Environment

Required `.env` values:

```dotenv
AWS_REGION=us-east-1
NOVA_LITE_MODEL_ID=us.amazon.nova-2-lite-v1:0
NOVA_SONIC_MODEL_ID=us.amazon.nova-2-sonic-v1:0
NOVA_EMBED_MODEL_ID=amazon.nova-2-multimodal-embeddings-v1:0
NOVA_EMBEDDING_DIMENSION=1024
NOVA_TEXT_TRUNCATION_MODE=END

DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<db>
REDIS_URL=redis://localhost:6379/0

STREAM_API_KEY=<stream_key>
STREAM_API_SECRET=<stream_secret>
```

Optional tuning:

```dotenv
SCREEN_CAPTURE_INTERVAL=3
REDIS_TTL_SECONDS=1800
MAX_RETRIEVAL_RESULTS=5
SESSION_TIMEOUT=3600
ENABLE_AURORA_WRITES=true
ENABLE_REDIS_CACHE=true
```

## Local Run

1. Install dependencies:

```bash
uv sync
```

2. Start Redis:

```bash
docker run --name nova-redis -p 6379:6379 redis:7-alpine
```

3. Start API server:

```bash
uv run python token_server.py
```

4. Start Stream frontend app:

```bash
cd frontend1
npm run dev
```

5. Open frontend:
- `http://127.0.0.1:5173`

Optional production-style static serving:
- Build frontend with `npm run build` inside `frontend1`.
- Then `token_server.py` serves `frontend1/dist` at `http://127.0.0.1:8001/`.

## CLI Commands

One-shot image analysis:

```bash
uv run python main.py analyze-screen \
  --frame-path ./sample-screen.png \
  --prompt "Explain this architecture and identify issues"
```

Diagram generation:

```bash
uv run python main.py diagram \
  --prompt "Draw User -> Sonic -> Lite -> Aurora -> Redis"
```

Nova Sonic console mode (local mic/speaker, barge-in):

```bash
uv run python main.py voice-chat
```

## Notes

- For cost control, keep frame sampling at 3-5 seconds.
- Session frame memory is per-session and cleared by `/api/session/stop`.
- OCR output is indexed as `screen_share_ocr` content for retrieval.
- Sonic websocket can be started/stopped from the in-call Nova panel.
