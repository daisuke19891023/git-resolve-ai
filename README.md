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

## Development

After making changes run the quality gates:

```bash
uv run nox -s lint
uv run nox -s typing
uv run nox -s test
```

