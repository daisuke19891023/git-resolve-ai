# goapgit

`goapgit` is a command line helper that plans and executes Git operations using a
Goal Oriented Action Planning (GOAP) workflow. The CLI inspects the current
repository state, proposes the shortest sequence of recovery actions, and can
explain every step it intends to perform.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency
management. Install uv and then run:

```bash
uv sync
```

## Usage

All commands can be executed from any Git repository. Use `--repo PATH` to
inspect a different repository and `--config FILE` to provide a custom
configuration file.

### `goapgit plan`

Display the current repository status and the shortest plan of actions to reach
the configured goal.

- Without flags the output is human-readable text.
- Pass `--json` to receive structured JSON containing the observed state, plan,
  and active strategy rules.

### `goapgit run`

Execute the computed plan. Runs in dry-run mode unless `--confirm` is provided.

- Without `--confirm` the Git facade records the commands that *would* run but
  does not mutate the repository.
- With `--confirm` real Git commands run against the repository.
- Combine with `--json` to obtain a machine-readable execution report.

### `goapgit dry-run`

Simulate `run` without changing the repository and report the Git commands that
would execute. Supports `--json` for structured output.

### `goapgit explain`

Describe why each action in the plan was selected, including alternative
approaches and the estimated cost for every action. The `--json` flag produces a
fully structured explanation payload.

## Documentation

For a detailed bilingual user manual, see [docs/user-manual.md](docs/user-manual.md).

## LLM-assisted conflict resolution

GOAPGit can enlist an LLM to propose conflict patches, strategy advice, plan adjustments, and commit / PR message drafts. The
integration is built on the OpenAI Python SDK Responses API and keeps only the minimal history required for each turn.

### Installing the optional dependencies

```bash
uv sync --extra llm
```

### Configuring providers

Set `GOAPGIT_LLM_PROVIDER` to `openai` or `azure` (defaults to `openai`). Environment variables take precedence over the
`[llm]` section in `pyproject.toml` / user configuration.

| Provider | Required environment variables | Notes |
|----------|--------------------------------|-------|
| OpenAI   | `OPENAI_API_KEY` (required), `OPENAI_BASE_URL` (optional) | `GOAPGIT_LLM_MODEL` defaults to `gpt-4o-mini`. |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` *or* `AZURE_OPENAI_AD_TOKEN`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION` | `GOAPGIT_LLM_MODEL` must match the deployment name. |

Additional controls:

- `GOAPGIT_LLM_MODEL`: overrides the model/deployment used by the Responses API client.
- `GOAPGIT_LLM_MODE`: `off`, `explain`, `suggest`, or `auto`.
- `GOAPGIT_LLM_MAX_TOKENS`, `GOAPGIT_LLM_MAX_COST`: cap the LLM usage per run.

### Minimal-history chaining

Every request:

1. Sends a fresh copy of the role-specific instructions (they are not implicitly inherited).
2. Supplies only the newest snippets (e.g., a conflict hunk, a failed test summary) as input content.
3. Links to the immediately preceding response with `previous_response_id` when follow-up context is needed.

Telemetry stores `response.id`, the linked `previous_response_id`, and token usage as JSON Lines. The generated text itself is
not persisted. Secrets are redacted before sending payloads to the model.

## Development

After making changes run the quality gates:

```bash
uv run nox -s lint
uv run nox -s typing
uv run nox -s test
```

