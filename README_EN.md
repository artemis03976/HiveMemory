# HiveMemory

[中文说明](README.md) | [Project Documentation](docs/PROJECT.md) | [Current Architecture](docs/architecture/overview.md) | [Setup Guide](docs/help/setup.md) | [Roadmap](docs/ROADMAP.md)

> Persistent memory and knowledge-sharing system for LLM agents
> *The Hippocampus for Artificial Intelligence*

HiveMemory is a persistent memory system for LLM agent workflows. It is designed to address long-context forgetting, lack of cross-session knowledge reuse, and information silos in multi-agent collaboration. The system turns high-value conversational information into searchable, updatable, reusable memories and injects them back into future tasks through a unified protocol.

The repository includes a runnable Python backend, a frontend development UI, vector storage and caching infrastructure, and a v0.6 development baseline where the top-level HiveMemory System orchestrates the peer Gateway, Patchouli, and Alice subsystems.

## Release Status

- Latest released tag: `v0.5.0`
- Current development baseline: `v0.6.0` (unreleased)
- Code and package version: `0.6.0`
- Python requirement: `>=3.12`
- License: Apache-2.0

See [docs/architecture/overview.md](docs/architecture/overview.md) for the current system design and [docs/PROJECT.md](docs/PROJECT.md) for the global documentation index.

## What HiveMemory Provides Today

### Conversation and Integration Modes

- **Active mode**: `POST /api/v1/chat` provides SSE streaming chat, orchestrated by `ChatApplicationService` through Patchouli prepare/finalize and Alice agent execution
- `POST /api/v1/chat` supports request-scoped `generation_options` (`model` / `temperature` / `top_p` / `max_tokens`) for per-turn overrides without persisting to global config files
- **Passive mode**: `POST /api/v1/ingest` accepts discrete events from external frameworks; the System-layer `PassiveIngressService` orchestrates Gateway decisions, buffering, retrieval, and Patchouli submission

### Memory and Topic Management

- Semantic search and memory listing: `GET /api/v1/memories`
- Single memory lookup: `GET /api/v1/memories/{memory_id}`
- Memory deletion: `DELETE /api/v1/memories/{memory_id}`
- Active topic listing: `GET /api/v1/topics`
- Manual topic settlement: `POST /api/v1/topics/{topic_id}/trigger`
- Evict a topic from the active pool: `DELETE /api/v1/topics/{topic_id}`

### Configuration and Observability

- Current runtime config: `GET /api/v1/config`
- Update and persist runtime config: `POST /api/v1/config`
- View default config: `GET /api/v1/config/defaults`
- WebSocket log stream: `WS /api/v1/ws/logs`
- Health check: `GET /health`
- Readiness check: `GET /health/ready`

### Core Capabilities

- v0.6 subsystem architecture: top-level `HiveMemorySystem` with peer Gateway, Patchouli, and Alice subsystems
- System Gateway for commands, topic routing, query analysis, cancellation/timeouts, and conservative fallback
- In-process runtime buses: AsyncSystemBus / GlobalSystemBus / subsystem-local buses
- MTP (Memory Tool Protocol) with `SEARCH / READ / RUN / WRITE / UPDATE / CALL`
- Persistent memory storage backed by Qdrant
- Hybrid Dense + Sparse retrieval path
- Frontend development UI built with Vite + React

## Architecture Overview

The current implementation uses a **System / Service / Runtime** layout. The top-level System owns application orchestration and global routes, Gateway owns entry decisions, Patchouli owns memory-domain capabilities, and Alice owns agent execution plus MTP/tool execution.

### Main Runtime Components

- **HiveMemorySystem**: the top-level host that assembles global routes, application services, Gateway, Patchouli, and Alice
- **ChatApplicationService**: the active chat orchestrator that runs `prepare -> Alice run -> finalize`
- **GatewaySystem / GatewayRuntime**: the entry-decision subsystem for commands, topic routing, query analysis, and stable decision projection
- **PatchouliSystem / PatchouliRuntime**: the memory subsystem host and runtime for retrieval, perception, generation, lifecycle, and storage capabilities
- **Retrieval Familiar**: the Hot Path retrieval service for hybrid retrieval, reranking, and context rendering
- **Librarian Core**: the Cold Path memory service for topic perception, memory extraction, and lifecycle management
- **AliceSystem / AliceRuntime**: the agent runtime subsystem that owns the Agent runtime and Koakuma tool runtime
- **KoakumaRuntime**: the MTP/tool executor used by Alice during agent generation

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
- `CALL`: suspend the current frame and delegate work to a sub-agent

Protocol format:

```text
⟪ VERB | TARGET | key="value" ⟫
```

See [docs/contracts/mtp.md](docs/contracts/mtp.md) for the current protocol and [docs/PROJECT.md](docs/PROJECT.md) for the overall documentation index.

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

- Docker / Docker Compose (recommended, for one-command deployment)
- Or manual setup:
  - Python 3.12+
  - Node.js (for frontend development)
  - Valid LLM API keys (for example DeepSeek or OpenAI-compatible endpoints)

Also note that embedding and reranker models may need to be downloaded and warmed up on first startup, so the service being up does not necessarily mean the models are ready.

## Quick Start

### Option 1: One-Command Docker Deployment (Recommended)

If you just want to try the test build quickly, we strongly recommend Docker one-command deployment:

```bash
# 1. Clone the repository
git clone https://github.com/artemis03976/HiveMemory.git
cd HiveMemory

# 2. Copy and edit environment file (fill in your LLM API key)
cp configs/.env.example .env

# 3. Start everything (Qdrant + HiveMemory backend app)
docker compose -f docker/docker-compose.yml up -d --build
```

After startup, open **http://localhost:8000** in your browser to use the full web UI.

### Option 2: Local Development Setup

First copy the environment template:

```bash
cp configs/.env.example .env
```

Then edit `.env` as needed. In general:

- `.env` / environment variables: API keys, Qdrant address, debug flags, and other environment-level settings
- `configs/config.yaml`: business logic and algorithmic settings for retrieval, perception, generation, lifecycle, and more

At minimum, check:

- `HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY` (or the provider used by the default model)
- the default model's `id`, `litellm_model`, and `provider` in `configs/models.yaml`
- `HIVEMEMORY__PATCHOULI__STORAGE__HOST` / `PORT`

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
npm ci
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

- `HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY`
- `HIVEMEMORY__GATEWAY__WORKFLOW__DEFAULT_REQUEST_TIMEOUT_MS`
- `HIVEMEMORY__PATCHOULI__STORAGE__HOST`
- `HIVEMEMORY__LOGGING__LEVEL`

### YAML Configuration

[configs/config.yaml](configs/config.yaml) defines default runtime settings for:

- `system`, `logging`, `scheduler`, `runtime_events`, and `i18n`
- `shared`: Gateway/Librarian LLM references, embedding, and provider defaults
- `gateway`, `passive_ingress`, and `memory_compiler`
- `patchouli`: storage, perception, generation, retrieval, lifecycle, and artifacts
- `alice`: Agent Runtime and Koakuma

Available models are maintained separately in [configs/models.yaml](configs/models.yaml). Provider secrets come from environment variables or `configs/providers.secrets.yaml`.

Recommended practice:

- keep secrets, addresses, ports, and environment switches in environment variables
- keep business logic parameters in `config.yaml`

## Developer Entrypoints

If you want to integrate the system directly in Python, the main entrypoint is:

- `hivememory.system.system.HiveMemorySystem`

It exposes two primary integration modes:

- `chat()` / `chat_stream()`: active mode, where `ChatApplicationService` coordinates Patchouli memory preparation, Alice agent execution, and Patchouli finalization
- `ingest_event()` / `flush_ingressor()`: passive mode, suitable for Discord bots, WeChat bots, or other external frameworks

If you only need HTTP APIs, use the FastAPI service directly. If you want to embed HiveMemory into an existing agent framework, passive ingest mode is often the most natural starting point.

## Project Structure

```text
HiveMemory/
├── configs/                 # Environment templates and main config
├── docker/                  # Docker one-command deployment (backend app + Qdrant)
├── docs/                    # Project design and planning docs
├── frontend/                # React + Vite frontend development UI
├── scripts/                 # Startup and helper scripts
├── src/hivememory/
│   ├── core/                # Core data models
│   ├── engines/             # Gateway / Retrieval / Perception / Generation / Lifecycle
│   ├── infrastructure/      # Storage / LLM / WebSocket
│   ├── patchouli/           # Patchouli memory subsystem and runtime
│   ├── alice/               # Alice agent runtime and Koakuma MTP/tool runtime
│   ├── system/              # Top-level HiveMemory system, global bus, and application services
│   ├── prompts/             # System prompts and prompt assembly
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
- [docs/PROJECT.md](docs/PROJECT.md) — current project overview and global documentation index
- [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) — documentation categories, status, and maintenance rules
- [docs/architecture/overview.md](docs/architecture/overview.md) — current backend architecture
- [docs/contracts/README.md](docs/contracts/README.md) — cross-subsystem contract index
- [docs/help/README.md](docs/help/README.md) — setup, configuration, and troubleshooting
- [docs/ROADMAP.md](docs/ROADMAP.md) — roadmap and future direction

## Contributing

Issues and pull requests are welcome. The repository is currently on the unreleased v0.6.0 development baseline. Behavioral changes should update the corresponding current design or contract document in the same change.

## License

This project is licensed under Apache-2.0. See [LICENSE](LICENSE) for details.
