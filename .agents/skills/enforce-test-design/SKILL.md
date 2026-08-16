---
name: enforce-test-design
description: Enforce HiveMemory's test design, classification, mocking, isolation, and verification standards. Use whenever creating, modifying, moving, deleting, or reviewing tests; implementing a bug fix or behavior change that needs tests; adding fixtures, fakes, mocks, or pytest markers; or changing test-related pytest, coverage, or CI configuration in the HiveMemory repository.
---

# Enforce HiveMemory Test Design

## Load the authoritative standard

Locate the repository root and read `docs/governance/testing/test-design-standards.md` completely before changing tests or test configuration. Treat that document as authoritative when it conflicts with this condensed workflow.

Inspect the production behavior and the nearest existing tests before deciding what to add. Do not create a test merely because a production file changed.

## Follow the workflow

### 1. Decide whether a test is warranted

Add or change a test when protecting at least one of these:

- a bug regression;
- a public API, event, SSE, persistence, or provider contract;
- a permission rule, state transition, filter, idempotency rule, error path, or concurrency invariant;
- a high-risk branch whose failure would be observable.

Usually do not add a separate test for plain field assignment, logic-free models, constant re-exports, trivial pass-through wrappers, behavior already protected at the same boundary, or coverage-only execution.

Before adding a test, name the plausible production defect it must catch. If no plausible defect would make it fail, do not add it.

### 2. Assign exactly one primary type

Use the test directory as the source of truth:

| Directory | Primary type | Boundary |
|:---|:---|:---|
| `tests/unit/` | `unit` | One cohesive behavior unit is real; collaborators may be replaced |
| `tests/integration/` | `integration` | Multiple internal components, or an adapter and the dependency under test, are real |
| `tests/e2e/` | `e2e` | A public ingress traverses major internal boundaries to a user-observable outcome |

Do not mix primary types in one file. Move or split a misclassified test instead of compensating with a contradictory marker.

Apply runtime-condition markers orthogonally:

- `real_infra`: require provisioned Qdrant or real Embedding/Reranker infrastructure;
- `live_llm`: call a real LLM provider;
- `slow`: exceed the project's fast-feedback budget.

Never combine `unit` with `real_infra` or `live_llm`. Classify a real provider adapter test as `integration + live_llm`; classify a full live chain as `e2e + live_llm`, adding `real_infra` when required.

### 3. Design the failure before writing the assertion

Define the input, observable behavior, and counterexample first. Assert the output, state transition, persisted result, emitted event, external contract, or error that distinguishes correct behavior from the plausible defect.

Keep one scenario and one failure reason per test; multiple assertions are valid when they jointly describe that one behavior.

For a bug fix, run the regression test against the broken behavior before fixing production code when practical. Otherwise state which pre-fix behavior the assertion distinguishes.

### 4. Keep the tested boundary real

Use real objects inside the boundary claimed by the test. Use fake/mock objects only outside that boundary.

Allow interaction assertions when the call, event, ordering, cancellation, idempotency, or outbound payload is itself the observable contract. Assert only contract-relevant fields. When another observable result exists, assert it too.

Do not make a failing test green by:

- weakening an exact expectation to `is not None`, `len(...) > 0`, a broad range, or `or` fallback;
- wrapping the assertion in a conditional, warning, broad exception handler, skip, or xfail;
- replacing the subject or claimed collaboration boundary with a mock;
- asserting a value that the test itself just inserted into a mock;
- copying production logic into the test to calculate the expected value;
- deleting or rewriting a valid regression test without an intentional contract change.

Prefer public behavior over private methods and internal attributes. If a complex private pure algorithm cannot be exercised economically through a public boundary, consider extracting it into a separately testable unit.

### 5. Preserve determinism and isolation

- Freeze or inject time; do not use fixed sleeps to wait for asynchronous state.
- Use events or polling with a bounded timeout for eventual behavior.
- Use `tmp_path` for temporary files; do not write test artifacts into the repository.
- Restore global state, registries, context variables, environment variables, and singletons in teardown.
- Do not add `sys.path` mutations.
- Make each test pass independently, in arbitrary order, and under the project's parallel test execution.
- Do not introduce unawaited coroutines, leaked resources, or unexpected warnings.

### 6. Verify proportionally

Run the narrowest affected test first, then the containing file or package, then the relevant configured test set. Read `pyproject.toml` and `.github/workflows/ci.yml` for the current commands and markers instead of assuming they already match the governance target.

Do not invoke `real_infra`, `live_llm`, or `slow` tests unless the required services, credentials, cost authorization, and time budget are available. Report excluded runtime-condition markers explicitly.

When reviewing rather than editing, report violations with the affected behavior and likely false-positive or brittleness risk. Do not expand a focused change into unrelated suite cleanup; record adjacent legacy violations separately.

## Handoff requirements

Report:

- the behavior and plausible defect each new or changed test protects;
- its primary type and runtime-condition markers;
- the test commands run and their results;
- any excluded test sets, warnings, or remaining gaps.

Never claim the suite is protected solely from test count or coverage percentage.
