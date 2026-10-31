# HiveMemory

[中文说明](README.md) | [Project Documentation](docs/PROJECT.md) | [Current Architecture](docs/architecture/overview.md) | [Setup Guide](docs/help/setup.md) | [Roadmap](docs/ROADMAP.md)

> Persistent memory and knowledge-sharing system for LLM agents
> *The Hippocampus for Artificial Intelligence*

HiveMemory is a persistent memory system for LLM agent workflows. It is designed to address long-context forgetting, lack of cross-session knowledge reuse, and information silos in multi-agent collaboration. The system turns high-value conversational information into searchable, updatable, reusable memories and injects them back into future tasks through a unified protocol.

The repository includes a runnable Python backend, a frontend development UI, vector storage and caching infrastructure, and the v0.7.1 content baseline where the top-level HiveMemory System assembles workspace, Gateway, Patchouli, and Alice. v0.7.0 completed the move to the Workspace network and task-process architecture: every task request is registered as a task process through a single registration entry, agent resource operations are authorized one by one by the workspace capability layer, and executors plug in through a CPU port so they can be replaced. v0.7.1 integrates the first external harness, Claude Code, in plugin mode: Claude Code conversations flow back through the Import Bus into memory generation, and the Claude Code agent searches and reads memories through MCP tools.

## Release Status

- This release: `v0.7.1` (plugin-mode integration with Claude Code; the tag is created after merge, see the [closeout audit](docs/archive/plans/v0.7.1-release-closeout-audit.md))
- Latest published baseline: `v0.7.0` (see the [closeout audit](docs/archive/plans/v0.7.0-release-closeout-audit.md))
- Code and package version: `0.7.1`
- Python requirement: `>=3.12`
- License: Apache-2.0

See [docs/architecture/overview.md](docs/architecture/overview.md) for the current system design and [docs/PROJECT.md](docs/PROJECT.md) for the global documentation index.

## What HiveMemory Provides Today

### Conversation and Integration Modes

- **Active mode**: `POST /api/v1/chat` provides SSE streaming chat. Each request is authenticated by the workspace registration entry and registered as a task process; `TaskProcessService` drives Gateway analysis, Patchouli prepare/finalize, and Alice agent execution through the CPU port. `POST /api/v1/chat/stop` stops a process, and a client disconnect is handled as a stop
- `POST /api/v1/chat` supports request-scoped `generation_options` (`model` / `temperature` / `top_p` / `max_tokens`) for per-turn overrides without persisting to global config files
- **Import Bus (external harness import)**: `POST /api/v1/import/turns` imports the complete record of one finished turn from an external harness (user message, replies, thoughts, external tool calls and results, outcome). The request goes through the unified authentication gateway and import authorization; the System-layer Import Bus orchestrates Gateway topic placement, shared sealing, and Patchouli perception submission, and confirms only after the interaction has been applied. It only feeds the memory network: it does not retrieve memories or return memory context
- **MCP memory access**: `POST /mcp` is the MCP endpoint for external harnesses (streamable HTTP, stateless, local addresses only) with two read-only tools, `memory_search` and `memory_read`. Each tool call obtains one request-scoped access context from the unified authentication gateway and is authorized against the Workspace access registry's allowlist; no task process is created
- **Claude Code plugin**: `integrations/claude-code/` (plugin `hivememory-connector`, local marketplace `hivememory-local`, Python standard library only) provides both conversation flow-back and memory access. The connector imports finished turns from Claude Code transcripts when a turn ends and when a session starts or ends, with network submission in a background process detached from Claude Code; the MCP server declared by the plugin lets the Claude Code agent search and read memories. Both authenticate as Claude Code's own access registration `harness:claude-code`: `HIVEMEMORY_CLAUDE_CODE_TOKEN` on the HiveMemory side and `HIVEMEMORY_TOKEN` on the Claude Code side must hold the same credential, and `configs/workspace_actors.yaml` grants the `claude_code` agent search and read only. See the [setup guide](docs/help/setup.md) for installation and configuration
- **Sessions**: create, resume, rename, and delete conversation sessions. Alice trims old tool results and writes relay summaries within the model budget while the original conversation is kept in full, and the summary and compaction boundary can be inspected; cancelled and failed turns that produced output keep their outcome. Sessions and checkpoints live only in the backend process and disappear on restart; topics remain separate memory material.
- **System commands**: Gateway only parses commands. A message recognized as a command ends with a command terminal state and does not enter normal chat; this version does not execute built-in commands.

### Workspaces and Access

- Two access registrations (`configs/system_principals.yaml` and `configs/workspace_actors.yaml`) and two-phase authentication decide who may enter which Workspace as which agent. An agent's operation permissions come only from the actor-operation allowlist in the access registration; Agent Profiles carry no permissions.
- Agent resource operations (MTP SEARCH, READ, RUN, WRITE, UPDATE, CALL) are authorized one by one by the workspace capability layer. A sub-agent spawned by CALL runs as an execution thread under the callee's identity.
- Memories, topics, and attachments carry explicit Workspace ownership. Resource owners check ownership first, then visibility under the resource policy.
- The write-intent registry and reference-read caches are owned by workspace, and canonical memory changes invalidate the atom and profile caches. See [Workspace](docs/workspace/README.md) for the full boundaries.

### Chat Attachments

- Upload TXT, Markdown, and DOCX through `POST /api/v1/workspace/assets`; after deterministic parsing, only READY representations can be selected for Chat.
- AttachmentCompiler applies a context budget; successful interactions record use, and Memory CREATE/UPDATE can promote sources into Artifacts.
- WorkspaceAsset availability remains process-local. See [Chat Attachments](docs/system/attachments.md) for the full flow.

### Memory, Topic, and Agent Management

- Create, semantically search and list, read, edit, give feedback on, and delete memories: `/api/v1/memories`
- Inspect and cancel memory generation tasks: `/api/v1/memory-tasks`
- List, manually settle, and evict active topics: `/api/v1/topics`
- Create and list Agent Profiles: `/api/v1/agents`

### Configuration and Observability

- Read, update, and view defaults of the runtime config: `/api/v1/config`
- Model and provider registries: `/api/v1/models`, `/api/v1/providers`
- Runtime event status and SSE stream: `/api/v1/runtime-events`
- WebSocket log stream: `WS /api/v1/ws/logs`
- Health and readiness checks: `GET /health`, `GET /health/ready`

### Core Capabilities

- Layered architecture: the top-level `HiveMemorySystem` assembles the peer workspace, Gateway, Patchouli, and Alice packages
- System Gateway for command parsing, topic routing, query analysis, cancellation/timeouts, and conservative fallback
- In-process runtime buses: AsyncSystemBus / GlobalSystemBus / subsystem-local buses
- MTP (Memory Tool Protocol) with `SEARCH / READ / RUN / WRITE / UPDATE / CALL`
- Persistent memory storage backed by Qdrant
- Hybrid Dense + Sparse retrieval path
- Frontend development UI built with Vite + React

## Architecture Overview

The current implementation is organized in dependency layers. The top-level System owns composition, the facade, and lifecycle; shared runtime mechanisms live in `components`, and authentication, the capability layer, and asset facilities live in `workspace`; Gateway owns entry decisions, Patchouli owns memory-domain capabilities, and Alice owns agent execution plus MTP/tool execution.

### Main Runtime Components

- **HiveMemorySystem**: the composition root and facade that assembles shared runtime, workspace facilities, application services, Gateway, Patchouli, and Alice
- **Workspace facilities**: the authentication gateway and access registrations, the capability layer (root API and actor API), conversation sessions, the read view and write-intent registry, and WorkspaceAsset
- **TaskProcessService**: the single registration entry for task requests (`workspace.process`); it authenticates and registers task processes, identifies every process by `process_id`, and runs the four stages `Gateway -> prepare -> Actor execution (through the CPU port, currently Alice) -> finalize`. Each process owns its task and output channel, and the execution threads of the Actor stage are registered in the process's thread table
- **GatewaySystem / GatewayRuntime**: the entry-decision subsystem for command parsing, topic routing, query analysis, and stable decision projection
- **PatchouliSystem / PatchouliRuntime**: the memory subsystem host and runtime for retrieval, perception, generation, lifecycle, and storage capabilities
- **Retrieval Familiar**: the Hot Path retrieval service for hybrid retrieval, reranking, and context rendering
- **Librarian Core**: the Cold Path memory service for topic perception, memory extraction, and lifecycle management
- **AliceSystem / AgentRuntime**: the agent execution subsystem; it delivers run results through the CPU port, and its kernel factory assembles the agent loop and the Koakuma tool runtime
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
- `RUN`: execute system tools or code snippets stored in memory
- `WRITE`: actively submit a new memory write intent
- `UPDATE`: actively submit an update intent for an existing memory
- `CALL`: suspend the current frame and delegate work to a sub-agent

Protocol format:

```text
⟪ VERB | TARGET | key="value" ⟫
```

See [docs/contracts/mtp.md](docs/contracts/mtp.md) for the current protocol and [docs/PROJECT.md](docs/PROJECT.md) for the overall documentation index.

## API Overview

| Path | Method | Purpose |
|:---|:---|:---|
| `/health` | GET | Liveness check |
| `/health/ready` | GET | Readiness check after model warmup |
| `/api/v1/chat` | POST | SSE streaming active chat |
| `/api/v1/chat/stop` | POST | Stop a running task process |
| `/api/v1/sessions` | GET / POST | List / create sessions |
| `/api/v1/sessions/{session_id}` | GET / PATCH / DELETE | Read full history / rename / delete a session |
| `/api/v1/import/turns` | POST | Import one finished turn from an external harness (Import Bus) |
| `/mcp` | POST | MCP tools for external harnesses to search and read memories |
| `/api/v1/memories` | GET / POST | Semantic search or list memories / create a memory |
| `/api/v1/memories/{memory_id}` | GET / PATCH / DELETE | Read / edit / delete a single memory |
| `/api/v1/memories/{memory_id}/feedback` | POST | Submit memory feedback |
| `/api/v1/memory-tasks` | GET | List memory generation tasks |
| `/api/v1/memory-tasks/{task_id}` | GET | Get a single task |
| `/api/v1/memory-tasks/{task_id}/cancel` | POST | Cancel a task |
| `/api/v1/topics` | GET | List active topics |
| `/api/v1/topics/{topic_id}/settle` | POST | Manually settle a topic |
| `/api/v1/topics/{topic_id}` | DELETE | Evict a topic from the active pool |
| `/api/v1/agents` | GET / POST | List / create Agent Profiles |
| `/api/v1/workspace/assets` | POST | Upload an attachment |
| `/api/v1/models` | GET / POST | List / register models |
| `/api/v1/models/{model_id}` | GET / PUT / DELETE | Read / update / delete a model |
| `/api/v1/providers` | GET | List providers |
| `/api/v1/providers/{provider_name}` | PUT / DELETE | Update / delete a provider |
| `/api/v1/config` | GET / POST | Read / update runtime config |
| `/api/v1/config/defaults` | GET | Get default config |
| `/api/v1/runtime-events/status` | GET | Runtime event status |
| `/api/v1/runtime-events/stream` | GET | Runtime event SSE stream |
| `/api/v1/ws/logs` | WS | Stream runtime logs |

The request identity is selected with the `x-user-id` / `x-workspace-id` headers, and agent actions also carry an `agent_id`; see the [routes and events contract](docs/contracts/routes-and-events.md#6-http-入口的身份交接与管理读取边界) for the handoff rules. `/api/v1/import/turns` and `/mcp` are entries for external harnesses and additionally declare the access source with `X-Principal-ID` and carry the access credential in `Authorization: Bearer`; see [Import Bus](docs/system/import-bus.md) and [MCP memory access](docs/contracts/mcp.md).

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
- `gateway`, `import_bus`, and `memory_compiler`
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

- `process_service.register_process()` and `run_process()`: active mode. The registration entry first authenticates and registers a task process and returns a process handle; `run_process()` then subscribes to the process's streaming output, with `TaskProcessService` coordinating Gateway analysis, Patchouli memory preparation, Alice agent execution, and Patchouli finalization
- `import_bus_service.import_turn()`: Import Bus, which imports the complete record of one finished turn from an external harness; the caller must first obtain a request-scoped access context from the unified authentication gateway (the HTTP entry `POST /api/v1/import/turns` does this for you)

If you only need HTTP APIs, use the FastAPI service directly. If you want the interactions of an existing agent harness to become HiveMemory memories and let it read them back, `POST /api/v1/import/turns` and `/mcp` are usually the most natural starting points, and the Claude Code plugin (`integrations/claude-code/`) is a ready-made integration.

## Project Structure

```text
HiveMemory/
├── configs/                 # Environment templates and main config
├── docker/                  # Docker one-command deployment (backend app + Qdrant)
├── docs/                    # Project design and planning docs
├── frontend/                # React + Vite frontend development UI
├── integrations/            # External harness connectors (Claude Code plugin and local marketplace)
├── scripts/                 # Startup and helper scripts
├── src/hivememory/
│   ├── core/                # Core data models, contract constants, access value types, and ports
│   ├── config/              # Config section models and root config loading
│   ├── components/          # In-process runtime mechanisms: bus, scheduler, work queue, runtime events
│   ├── engines/             # Gateway / Retrieval / Perception / Generation / Lifecycle algorithms
│   ├── infrastructure/      # Storage / LLM / Embedding / attachment parsing adapters
│   ├── workspace/           # Authentication entry, capability layer, read view, WorkspaceAsset facilities, task processes (process)
│   ├── gateway/             # Gateway entry-decision subsystem
│   ├── patchouli/           # Patchouli memory subsystem and runtime
│   ├── alice/               # Alice agent execution subsystem
│   ├── agent_runtime/       # Single-agent execution layer and Koakuma MTP/tool runtime
│   ├── system/              # Composition root, facade, registries, and Import Bus
│   ├── prompts/             # System prompts and prompt assembly
│   ├── server/              # FastAPI app entrypoint and routes
│   └── mcp_server/          # MCP entry for external harnesses to access memory (served by server as the /mcp route)
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

Issues and pull requests are welcome. v0.7.1 content is closed out, and its tag is created after merge. Follow the documentation promotion gate for behavioral changes: update current design or contract documents only when branch development is explicitly entering final closeout, not while the design is still evolving.

## License

This project is licensed under Apache-2.0. See [LICENSE](LICENSE) for details.
