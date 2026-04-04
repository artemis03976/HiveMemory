# HiveMemory

[中文说明](README.md) | [Project Design Doc](docs/PROJECT.md) | [Setup Guide](docs/SETUP.md) | [Roadmap](docs/ROADMAP.md)

> Persistent memory and knowledge-sharing system for LLM agents
> *The Hippocampus for Artificial Intelligence*

HiveMemory is a persistent memory system for LLM agent workflows. It is designed to address long-context forgetting, lack of cross-session knowledge reuse, and information silos in multi-agent collaboration. The system turns high-value conversational information into searchable, updatable, reusable memories and injects them back into future tasks through a unified protocol.

The repository already includes a runnable Python backend, a frontend development UI, vector storage and caching infrastructure, and a Patchouli-based runtime that supports active chat, passive message ingestion, memory retrieval, topic management, and runtime configuration.

## Release Status

- Current version: `0.1.0`
- Release stage: Test build / Alpha
- Python requirement: `>=3.12`
- License: Apache-2.0

This README focuses on the **current implementation** in v0.1.0, with an emphasis on a real local startup path and a practical system overview. For deeper design background and architectural rationale, see [docs/PROJECT.md](docs/PROJECT.md).

## What HiveMemory Provides Today

### Conversation and Integration Modes

- **Active mode**: `POST /api/v1/chat` provides SSE streaming chat, driven by `PatchouliSystem.chat_stream()` with the full generation loop and MTP execution
- **Passive mode**: `POST /api/v1/ingest` accepts discrete messages from external frameworks, and `PatchouliSystem.ingest()` handles buffering, analysis, retrieval, and later memory consolidation

### Memory and Topic Management

- Semantic search and memory listing: `GET /api/v1/memories`
- Single memory lookup: `GET /api/v1/memories/{memory_id}`
- Memory deletion: `DELETE /api/v1/memories/{memory_id}`
- Active topic listing: `GET /api/v1/topics`
- Manual topic settlement: `POST /api/v1/topics/{topic_id}/trigger`
- Evict a topic from the active pool: `DELETE /api/v1/topics/{topic_id}`

### Configuration and Observability

- Current runtime config: `GET /api/v1/config`
- Update in-memory runtime config: `POST /api/v1/config`
- View default config: `GET /api/v1/config/defaults`
- WebSocket log stream: `WS /api/v1/ws/logs`
- Health check: `GET /health`
- Readiness check: `GET /health/ready`

### Core Capabilities

- Patchouli trinity architecture: The Eye / Retrieval Familiar / Librarian Core
- In-process communication bus: SystemBus
- MTP (Memory Tool Protocol) with `SEARCH / READ / RUN / WRITE / UPDATE`
- Persistent memory storage backed by Qdrant
- Hybrid Dense + Sparse retrieval path
- Frontend development UI built with Vite + React

## Architecture Overview

The current implementation of HiveMemory is built around the **Patchouli System**. It is not just a chat API, but a runtime that separates real-time retrieval from asynchronous memory organization.

### Main Patchouli Components

- **PatchouliSystem**: the top-level developer entrypoint that connects The Eye and PatchouliKernel
- **The Eye**: the interaction gateway responsible for intent recognition, query rewriting, and traffic routing
- **PatchouliKernel**: the system orchestrator that initializes infrastructure, registers services, and connects SystemBus
- **Retrieval Familiar**: the Hot Path retrieval service for hybrid retrieval, reranking, and context rendering
- **Librarian Core**: the Cold Path memory service for topic perception, memory extraction, and lifecycle management
- **KoakumaRuntime**: the MTP executor that intercepts and executes memory tool calls during generation

### Hot Path / Cold Path

- **Hot Path**: optimized for low latency and responsible for retrieval plus context injection for the current request
- **Cold Path**: asynchronous and responsible for post-conversation organization, summarization, writing, updating, and archiving

This split allows HiveMemory to balance:

- fast responses for the current conversation
- continuous accumulation and reuse of historical knowledge

### MTP: Memory Tool Protocol

HiveMemory provides an in-process protocol that allows the Worker Agent to actively access the memory layer during generation:

- `SEARCH`: fuzzy search returning candidate memory indices
- `READ`: read specific memory content
- `RUN`: execute kernel tools or code snippets stored in memory
- `WRITE`: actively submit a new memory write intent
- `UPDATE`: actively submit an update intent for an existing memory

Protocol format:

```text
⟪ VERB | TARGET | key="value" ⟫
```

For full design background, motivation, and terminology, see [docs/PROJECT.md](docs/PROJECT.md).

## API Overview

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Liveness check |
| `/health/ready` | GET | Readiness check after model warmup |
| `/api/v1/chat` | POST | SSE streaming active chat |
| `/api/v1/ingest` | POST | Passive message ingestion |
| `/api/v1/memories` | GET | Semantic search / list memories |
| `/api/v1/memories/{memory_id}` | GET | Get a single memory |
| `/api/v1/memories/{memory_id}` | DELETE | Delete a single memory |
| `/api/v1/topics` | GET | List active topics |
| `/api/v1/topics/{topic_id}/trigger` | POST | Manually settle a topic |
| `/api/v1/topics/{topic_id}` | DELETE | Remove a topic from the active pool |
| `/api/v1/config` | GET / POST | Read / update runtime config |
| `/api/v1/config/defaults` | GET | Get default config |
| `/api/v1/ws/logs` | WS | Stream runtime logs |

## Requirements

Recommended environment for running the project:

- Python 3.12+
- Node.js (a recent LTS version is recommended for the frontend development UI)
- Docker / Docker Compose (to start local Qdrant and Redis)
- Valid LLM API keys (for example DeepSeek or OpenAI-compatible endpoints)

Also note that embedding and reranker models may need to be downloaded and warmed up on first startup, so the service being up does not necessarily mean the models are ready.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/artemis03976/HiveMemory.git
cd HiveMemory
```

### 2. Start the infrastructure

The bundled Docker Compose setup starts:

- Qdrant (HTTP `6333` / gRPC `6334`)
- Redis (`6379`)
- Optional Qdrant Web UI (`6335`, debug profile only)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

If you also want the Qdrant Web UI:

```bash
docker-compose -f docker/docker-compose.yml --profile debug up -d
```

> Note: Redis uses `hivememory_redis_pass` as the default password in Docker Compose, while `configs/.env.example` leaves it blank. If you use the bundled Compose setup directly, set `HIVEMEMORY__REDIS__PASSWORD=hivememory_redis_pass` in your environment.

### 3. Configure environment variables

First copy the environment template:

```bash
cp configs/.env.example .env
```

Then edit `.env` as needed. In general:

- `.env` / environment variables: API keys, Qdrant / Redis addresses, debug flags, and other environment-level settings
- `configs/config.yaml`: business logic and algorithmic settings for retrieval, perception, generation, lifecycle, and more

At minimum, check:

- `HIVEMEMORY__LLM__WORKER__API_KEY`
- `HIVEMEMORY__LLM__LIBRARIAN__API_KEY`
- `HIVEMEMORY__REDIS__PASSWORD`
- `HIVEMEMORY__QDRANT__HOST` / `PORT`

### 4. Install the backend

Use the package-based installation defined in `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you also need tests and developer tooling:

```bash
pip install -e ".[dev]"
```

### 5. Start the backend service

The recommended entrypoint is the packaged script:

```bash
hivememory-server
```

Default backend address:

- `http://localhost:8769`

After startup, you can check:

```bash
curl http://localhost:8769/health
curl http://localhost:8769/health/ready
```

Where:

- `/health` reports whether the service is alive
- `/health/ready` reports whether model warmup has completed; while warming up it returns `503 warming_up`

### 6. Start the frontend development UI

```bash
cd frontend
npm install
npm run dev
```

The frontend development server runs on:

- `http://127.0.0.1:5173`

The frontend dev proxy forwards `/api` to:

- `http://localhost:8769`

## Configuration Model

HiveMemory currently uses a layered **environment variables + YAML** configuration model.

### Environment Variables

`configs/.env.example` shows the recommended format. Environment variables use the `HIVEMEMORY__` prefix, for example:

- `HIVEMEMORY__LLM__WORKER__MODEL`
- `HIVEMEMORY__LLM__LIBRARIAN__API_KEY`
- `HIVEMEMORY__QDRANT__HOST`
- `HIVEMEMORY__REDIS__PASSWORD`
- `HIVEMEMORY__LOGGING__LEVEL`

### YAML Configuration

[configs/config.yaml](configs/config.yaml) defines default runtime settings for:

- `llm`: gateway / librarian / worker
- `embedding`
- `qdrant`
- `redis`
- `gateway`
- `perception`
- `generation`
- `retrieval`
- `lifecycle`
- `logging`

Recommended practice:

- keep secrets, addresses, ports, and environment switches in environment variables
- keep business logic parameters in `config.yaml`

## Developer Entrypoints

If you want to integrate the system directly in Python, the main entrypoint is:

- `hivememory.patchouli.system.PatchouliSystem`

It currently exposes two primary integration modes:

- `chat()` / `chat_stream()`: active mode, where the system drives generation and the MTP loop directly
- `ingest()`: passive mode, suitable for Discord bots, WeChat bots, or other external frameworks

If you only need HTTP APIs, use the FastAPI service directly. If you want to embed HiveMemory into an existing agent framework, passive ingest mode is often the most natural starting point.

## Project Structure

```text
HiveMemory/
├── configs/                 # Environment templates and main config
├── docker/                  # Local infrastructure (Qdrant / Redis)
├── docs/                    # Project design and planning docs
├── frontend/                # React + Vite frontend development UI
├── scripts/                 # Startup and helper scripts
├── src/hivememory/
│   ├── core/                # Core data models
│   ├── engines/             # Gateway / Retrieval / Perception / Generation / Lifecycle
│   ├── infrastructure/      # Storage / LLM / SystemBus / WebSocket
│   ├── patchouli/           # Patchouli system, Kernel, MTP, WorkerAgent
│   └── server/              # FastAPI app entrypoint and routes
└── tests/                   # Unit, integration, and end-to-end tests
```

## Development Notes

- Testing is driven by `pytest`; see [pyproject.toml](pyproject.toml)
- The frontend provides `npm run dev`, `npm run build`, and `npm run lint`
- Health and readiness endpoints are built into the backend API
- Logs can be streamed to the frontend over WebSocket

## Documentation

- [README.md](README.md) — Chinese README
- [docs/PROJECT.md](docs/PROJECT.md) — project background, architecture, Patchouli system, and MTP details
- [docs/SETUP.md](docs/SETUP.md) — setup guide
- [docs/ROADMAP.md](docs/ROADMAP.md) — roadmap and future direction

## Contributing

Issues and pull requests are welcome. The repository is currently converging on the v0.1.0 test release, so please double-check that code and documentation stay aligned before submitting changes, especially for startup commands, ports, and configuration instructions.

## License

This project is licensed under Apache-2.0. See [LICENSE](LICENSE) for details.
