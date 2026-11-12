---
name: hivememory
description: "HiveMemory is the user's own persistent memory base, shared across sessions, models, harnesses and agents: decisions, facts, lessons, preferences, work state and research conclusions become addressable, traceable and correctable memory assets here, searched, read, written and corrected through the four tools memory_search, memory_read, memory_write and memory_update. Load this skill first when you start a task that is not one-off, when the user refers to the past (last time, what we decided, the usual way), when you make a decision or reach a conclusion, when the user states a preference or corrects you, when you find a memory outdated, or when work is about to be interrupted. It explains when to search, how far to trust memories, what is worth writing, how to write so that later agents can find and use it, when to correct an existing memory, and how to coexist with the harness's own memory."
---

# HiveMemory: memory assets the user owns and shares across environments

HiveMemory is the user's own persistent memory base. Things worth keeping for the long term, such as decisions and their reasons, verified facts, solved problems, lessons, the user's preferences, the state of unfinished work and research conclusions, become memory assets here: each has a stable alias, can be traced back to the conversation that produced it, and can be corrected with its version history kept. Memories that are used often stay active; memories nobody uses for a long time gradually fade out of search.

These assets do not belong to one session, one model or one harness. Agents the user works with across sessions, agent harnesses and models read and write the same memory base: a conclusion written in one place can be found from another environment, and after a correction in one place every place reads the new version. When the user switches models or tools, the accumulated knowledge stays.

So you are both a user and a maintainer of this memory: before you start, look at what is already known and reuse verified conclusions instead of redoing the work; write in the first-hand knowledge you gain; correct content you find outdated or wrong. The quality of the memory base depends on every agent that uses it. This skill explains how to do these things well with the four memory tools.

## 1. How memories form and evolve

Memories come from two sources:

- **Written by agents**: what you and other agents submit through `memory_write` and `memory_update` when something holds. Only the agent involved can tell what was verified first-hand, which conclusion is final and what the user really meant, so you write these.
- **Extracted from archived conversations**: harnesses connected for conversation import send each finished turn to HiveMemory, together with tool calls and results, and the original text is kept as source evidence. After a topic has been idle for a while, a background model extracts memories from the whole conversation. This happens late, a separate memory generation model decides what to keep, and it only sees what was written out in the conversation.

When the work is mostly chat, you rarely need to write, apart from what the user asks you to remember and the preferences they state; memories mainly come from extraction. During long-running tasks, write decisions and conclusions when they hold. The system handles similar content coming from both sources, so there is no need to write less because of it.

Memories keep evolving after they enter the memory base:

- New content is first checked against existing memories, and similar ones are merged into one instead of piling up side by side;
- A correction creates a new version, and old versions stay in the version history;
- Reading a memory's full text counts as a reference, so memories that are actually used stay active; memories unused for a long time decay and eventually leave search.

## 2. The four tools

Tool names end with the names in the table below. A harness may add a prefix; in Claude Code, for example, the full name of the search tool is `mcp__plugin_hivememory-connector_hivememory__memory_search`.

| Tool | Purpose | Key points |
|:---|:---|:---|
| `memory_search` | Search in natural language | Returns a fixed number of summaries, each with alias, type, archive time, confidence, tags and summary |
| `memory_read` | Read the full text by alias | Can read several at once; can also read the handles returned by writes and updates |
| `memory_write` | Write one thing worth keeping | `content` is required, and `title` and `reason` should also be filled in; returns a handle starting with `draft_` |
| `memory_update` | Correct an existing memory | `alias` is the formal alias from search or read, `instruction` says how to change it, `content` may carry new text; returns a handle starting with `rev_` |

If a `turn` parameter appears in writes and updates, do not fill it in: the plugin adds it when the call is made.

## 3. Coexisting with the harness's own memory

The harness's own memory (for example Claude Code's memory files and CLAUDE.md, or the AGENTS.md that many harnesses read) and HiveMemory do not exclude each other. You can divide the work between them, for example:

| | The harness's own memory | HiveMemory |
|:---|:---|:---|
| Who can see it | Only this harness (often only this project) | Any agent in the HiveMemory ecosystem |
| How it enters context | Loaded automatically in every session | Searched by the agent when needed |
| Good for | How to work here: local paths and commands, conventions in this harness, project rules | What happened and what is known: decisions, facts, solved problems, lessons, user preferences, work state, research conclusions |

- Do not skip searching HiveMemory because your own memory already has something related: content written in other harnesses exists only in HiveMemory.
- When something is useful in both (for example a user preference), writing it in both is not redundant: the harness's memory makes sure it is always seen here, and HiveMemory lets other harnesses find it too. Write each according to its own rules.
- When the two disagree, project instructions, the harness's own memory and current evidence win; once you confirm the HiveMemory entry is outdated, correct it (section 6).

## 4. Searching

### When to search

- When you start a task that is not one-off: search by the task's subject (project, component, topic) to retrieve earlier decisions, the state of work in progress and the user's preferences, then start;
- When the user refers to the past: "that thing from last time", "what we decided", "the usual way";
- When you are about to decide or advise on something that may have been discussed before;
- Before investigating a problem again or redoing research, check whether a conclusion already exists;
- When you meet an unfamiliar project convention, environment setup or error.

One-off questions (translating a sentence, explaining a general concept) do not need a search.

### How to write a query

- Search compares only the title, type, tags and summary of memories, not the full text. Describe "which thing you are looking for", with concrete nouns: project names, component names, tool names, keywords from the error.
- Look for one thing per query; search several times for several unrelated things.
- No hit does not mean no memory: search again with different wording (synonyms, Chinese or English, the topic one level up).

### How far to trust results

- Results are only summaries. When a summary is relevant and you need details, read the full text with `memory_read`; you can pass several aliases at once.
- Memories are dated historical evidence and may be outdated. Current files, code, command output, project instructions and what the user says now take precedence over memories.
- Before taking an action with consequences on the basis of a memory (changing configuration, deleting files, stating a firm conclusion), verify it with current evidence. Verify especially memories that are old, have low confidence, or are of type `WORK_IN_PROGRESS` (work in progress).
- A memory may have been written in another harness or on another machine: its paths, commands and configuration may not hold in the current environment.
- When relaying it to the user, say that it comes from memory and when it was archived.
- If you find after verification that a memory is outdated or wrong, correct it with `memory_update` (section 6).

## 5. Writing

### What to write

Write when the following hold, rather than saving it all for the end of the session: the session may be closed at any time.

| Situation | Write down |
|:---|:---|
| The user asks you to remember something | What the user wants remembered, right away |
| A decision is made | What was chosen and why; what was rejected and why |
| The user's preferences and corrections | Preferred ways of doing things, disliked ones, "don't do this again" |
| A verified fact | Environment, versions, configuration options, command usage, actual behavior and limits of an interface |
| A problem is solved | Symptoms, root cause, fix, and the directions ruled out |
| Experience and lessons | Which approach works and which one caused trouble |
| Conclusions from external sources | Judgments drawn from docs, repositories or articles you read, with source and date |
| Work is not finished | How far it got, what comes next, where it is stuck |

Do not write:

- A running log of the conversation ("the user asked A, I answered B");
- Original text that can be read from the repository or files at any time (code, full documents): write the conclusion and where it is, not a copy;
- Unverified guesses; mark an inference that is really worth keeping as "(inference)";
- A memory you just searched for or read, written again as is;
- Keys, tokens, passwords, certificates or connection strings with credentials: if needed, record only the name and purpose.

### How to write

The formal memory is generated in the background around your write: it keeps the `title` you wrote, and generation produces the summary, tags and body. When you write:

1. **Write one thing per write**: one decision and its reasons, one fact, one problem and its fix, one preference, the state of one piece of work. Write independent things in separate writes; keep the background, process and conclusion of the same thing together.
2. **`title` names the thing itself**: specific and searchable, with key nouns such as project and component; write a conclusion as a statement. Do not use titles such as "discussion notes", "notes" or "summary": search only looks at titles, summaries and tags, so a memory with a vague title cannot be found later.
3. **`content` must be understandable on its own**: the background model that generates the memory cannot see the current turn, and an agent that reads it later may be in another harness without today's context. Do not write "above", "this approach" or "the error just now"; spell out names, values, paths, versions and dates. For content that only holds in one environment (local paths, one harness's configuration), say which environment.
4. **State the basis**: how you know it (which command you ran, which file or commit you read, what the user said directly), and the conditions and exceptions where it applies.
5. **`reason` says in one sentence when it will be needed later.**
6. Length is whatever makes it understandable on its own, usually a few sentences to a few paragraphs; quote only the key lines of logs and code. Write in the user's language.

Example (fictional):

```text
Bad
title:   Today's investigation
content: After changing it the way described above, it works.

Good
title:   The report service's daily rollup is scheduled by in-process APScheduler, not system cron
content: The daily rollup job of the report service (acme/report-svc) moved from system cron to
         in-process APScheduler. It still runs every day at 02:00 (UTC+8); the scheduling code is
         in scheduler.py. Reason: production runs in a read-only container that cannot write a
         crontab, and the job did not run after the 2026-10 release. Celery beat was rejected:
         adding Redis for a single scheduled job is not worth it.
reason:  Needed when adjusting the schedule later or when investigating a rollup that did not run
```

### After writing

- A `draft_` handle means the write was accepted and the formal memory is being generated in the background and is not saved yet. Do not wait or poll repeatedly; continue the task. Cancelling, stopping or ending the session later does not withdraw the write.
- When you need to confirm, read the handle with `memory_read`; once generation finishes it leads to the formal memory. A write similar to an existing memory is merged into that memory, and the handle then leads to the merged memory.
- The formal memory keeps the `title` you wrote; it is changed only when it does not match the body or lacks key nouns such as project and component. Generation writes the summary and tags.
- When reporting to the user, say "memory submitted, being generated in the background", not "saved".

## 6. Correcting existing memories

- **When to correct**: a memory you read disagrees with current evidence (the code changed, the configuration changed, a decision was overturned, the user corrected it), or the state of work in progress has advanced or the work is done.
- **Correct it rather than writing another**: when an existing memory is wrong, change it with `memory_update`; do not write an opposite memory and leave the old wrong version in place. Use `memory_write` only when you are recording a different thing.
- **Parameters**: `alias` is the formal alias from search or read results; `instruction` says what to change and why, for example "decision overturned: use X instead, because Y"; put new text in `content` when there is any.
- **Handles cannot be corrected**: `draft_` and `rev_` handles cannot be the target of a correction. To change something you just wrote, read the handle with `memory_read` first: if generation has finished, correct it by the formal alias you read; if it is still being generated, change it later, or write a supplementary memory.
- A correction rewrites the current version of the memory, and old versions stay in the version history.

## 7. How to call the tools, and errors

- Call the memory tools directly, not through code-execution tools: the session record only keeps the outer call, the system cannot recognize the memory operation inside it, and old memories read back that way are extracted again as new content at archive time.
- Sub-agents can use the memory tools as well. When you hand research or similar tasks to a sub-agent, say who writes the conclusion, so the same conclusion is not written twice.

When a tool fails it returns a description with an error code. Handle it as follows:

| Symptom | What to do |
|:---|:---|
| The four tools are not available | HiveMemory is not connected. Finish the task as usual; when the user asks you to remember something, tell them memory is temporarily unavailable |
| "HiveMemory is unreachable", `import_bus.unavailable`, `workspace.memory.intent.dispatch_rejected` | Temporarily unavailable; retry once later. If it still fails, continue the task and tell the user this was not recorded |
| `workspace.resource.not_found` | The alias is wrong or the memory no longer exists; search again |
| `workspace.memory.intent.pending_update_not_allowed` | A handle was used as the correction target; see section 6 |
| `request.turn_required`, "No turn is in progress" | The plugin could not add this turn's information to the write; retrying does not help. Tell the user and continue the task |
| `workspace.admission_denied`, `workspace.operation_denied` | An access or permission configuration problem; retrying does not help. Tell the user and continue the task |

## 8. The rhythm of a session

1. **At the start**: search by the task's subject; when a summary is relevant, read the full text and verify it before using it.
2. **During the work**: when you make a decision, solve a problem, or the user states a preference or corrects you, write it then and there; when you find an outdated memory, correct it then and there.
3. **Before ending or being interrupted**: if the work is not finished, write down its current state; if a work-in-progress memory for this work already exists, correct it.
