---
name: enforce-documentation-governance
description: Enforce HiveMemory's documentation classification, evidence, truth-source, and design-promotion rules. Use whenever creating, editing, moving, deleting, or reviewing repository documentation; updating READMEs, roadmaps, plans, ideas, todos, ADRs, or archives; or deciding whether branch work is ready to update current factual documentation.
---

# Enforce HiveMemory Documentation Governance

## Load the authority

Locate the repository root and read `docs/DOCUMENTATION.md` completely before changing or reviewing documentation. Treat it as authoritative; this skill is only an execution reminder and must not become a second copy of the rules.

## Review content before structure

Classify every affected document before editing it, then apply the corresponding truth standard from `docs/DOCUMENTATION.md`:

- current factual documents require evidence from mature code, tests, configuration, or an already completed stable baseline;
- Idea, Plan, and Todo documents may hold design work at their declared maturity;
- `ROADMAP.md` may summarize planned design, dependencies, risks, and sequencing when status and detailed sources are explicit;
- ADR and Archive content must satisfy their decision or historical role.

Do not treat matching frontmatter, headings, tables, terminology, or nearby document style as evidence that the content belongs in that document. For each material claim, identify whether it is a current fact, stable design reason, plan, idea, todo, decision, or history, and verify that its source and destination agree.

## Enforce the promotion gate

Before adding branch design to a factual document, verify that the implementation, tests, migration, plan acceptance, and code review have reached the final-closeout gate defined in `docs/DOCUMENTATION.md`. Until then, keep design details in Idea, Plan, or Todo documents and only the permitted planning summary in `ROADMAP.md`.

Never use phrases such as “planned,” “candidate,” “not yet implemented,” or “will support” to smuggle unfinished design into a factual document. Do not copy a work document into several current documents to create apparent consistency.

## Verify the result

Check the edited scope for:

- claims unsupported by the document's allowed source of truth;
- planned design leaked into factual documents;
- duplicated or competing truth sources;
- lost design rationale or ownership boundaries caused by structure-only rewriting;
- stale status, metadata, indexes, links, anchors, and archive references;
- unrelated user changes accidentally overwritten.

Run the repository-appropriate Markdown/link checks and `git diff --check` when available. In the handoff, state each document's classification, the evidence or planning source used, and any content intentionally left in Idea, Plan, Todo, or Roadmap rather than promoted to current fact.
