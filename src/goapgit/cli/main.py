"""CLI entry point for goapgit built with Typer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Sequence  # noqa: UP035

import typer
from pydantic import ValidationError

from goapgit.cli.diagnose import DiagnoseError, generate_diagnosis, report_to_json
from goapgit.cli.runtime import (
    WorkflowContext,
    build_action_contexts,
    build_action_specs,
    build_workflow_context,
    load_cli_config,
    strategy_rules_to_params,
)
from goapgit.core.explain import explain_plan
from goapgit.core.executor import Executor
from goapgit.git.facade import GitCommandError

if TYPE_CHECKING:
    from goapgit.core.models import ActionSpec, Plan, RepoState


app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(frozen=True, slots=True)
class PlanComputation:
    """Container describing a freshly computed plan for a repository."""

    state: RepoState
    actions: list[ActionSpec]
    plan: Plan


def _resolve_repo(repo: Path | None) -> Path:
    return repo.resolve() if repo is not None else Path.cwd()


def _emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False))


def _format_command(parts: Sequence[str]) -> str:
    """Render a command sequence with escaped control characters."""

    def _format_part(part: str) -> str:
        escaped = part.encode("unicode_escape").decode("ascii")
        if any(char.isspace() for char in part):
            escaped = escaped.replace('"', r"\"")
            return f'"{escaped}"'
        return escaped

    return " ".join(_format_part(part) for part in parts)


def _format_validation_error(exc: ValidationError) -> str:
    """Return a concise summary describing ``exc``."""
    fragments: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "config"
        message = error.get("msg") or str(exc)
        fragments.append(f"{location}: {message}")
    return "; ".join(fragments)


def _prepare_context(
    repo: Path | None,
    config_path: Path | None,
    *,
    json_logs: bool,
    dry_run_actions: bool,
    silence_logs: bool,
) -> WorkflowContext:
    repo_path = _resolve_repo(repo)
    try:
        config = load_cli_config(config_path)
    except FileNotFoundError as exc:
        typer.echo(f"Configuration file not found: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ValidationError as exc:
        message = _format_validation_error(exc) or str(exc)
        typer.echo(f"Invalid configuration: {message}", err=True)
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    return build_workflow_context(
        repo_path,
        config,
        json_logs=json_logs,
        dry_run_actions=dry_run_actions,
        silence_logs=silence_logs,
    )


def _build_plan_payload(context: WorkflowContext) -> PlanComputation:
    """Return the repo state, available actions, and plan for ``context``."""
    state = context.observer.observe()
    actions = build_action_specs(state, context.config)
    plan = context.planner.plan(state, context.config.goal, actions)
    return PlanComputation(state=state, actions=actions, plan=plan)


def _format_git_failure(exc: Exception) -> str:
    """Return a concise human-readable description for git-related failures."""
    if isinstance(exc, GitCommandError):
        stderr = exc.stderr.strip()
        if stderr:
            return stderr
        return str(exc)
    if isinstance(exc, OSError):
        message = getattr(exc, "strerror", None)
        if message:
            return str(message)
    return str(exc)


def _handle_git_failures[T](operation: Callable[[], T]) -> T:
    """Execute ``operation`` and convert git/OSError failures into CLI exits."""
    try:
        return operation()
    except (GitCommandError, OSError) as exc:
        typer.echo(_format_git_failure(exc), err=True)
        raise typer.Exit(code=1) from exc


def _safe_build_plan_payload(context: WorkflowContext) -> PlanComputation:
    """Wrap plan computation and surface CLI-friendly errors."""
    return _handle_git_failures(lambda: _build_plan_payload(context))


def _safe_execute_workflow(context: WorkflowContext) -> dict[str, Any]:
    """Wrap workflow execution and surface CLI-friendly errors."""
    return _handle_git_failures(lambda: _execute_workflow(context))


@app.callback()
def cli_root() -> None:
    """Top-level CLI group for goapgit."""


@app.command("diagnose")
def diagnose_command(
    pretty: bool = typer.Option(default=False, help="Pretty-print JSON output."),
) -> None:
    """Inspect git configuration and repository size information."""
    try:
        report = generate_diagnosis(Path.cwd())
    except DiagnoseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    output = report_to_json(report, pretty=pretty)
    typer.echo(output)


RepoOption = Annotated[Path | None, typer.Option(help="Path to the repository.")]
ConfigOption = Annotated[Path | None, typer.Option(help="Path to a configuration TOML.")]
JsonFlag = Annotated[
    bool,
    typer.Option("--json", "--json-output", help="Emit JSON instead of text."),
]
ConfirmFlag = Annotated[
    bool,
    typer.Option(help="Execute actions for real."),
]


@app.command("plan")
def plan_command(
    repo: RepoOption = None,
    config: ConfigOption = None,
    json_output: JsonFlag = False,
) -> None:
    """Display the current repository state and the computed plan."""
    context = _prepare_context(
        repo,
        config,
        json_logs=json_output,
        dry_run_actions=True,
        silence_logs=json_output,
    )
    computation = _safe_build_plan_payload(context)
    state = computation.state
    plan = computation.plan

    if json_output:
        payload = {
            "repository": str(context.repo_path),
            "state": state.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "strategy_rules": strategy_rules_to_params(context.config.strategy_rules),
        }
        _emit_json(payload)
        return

    lines: list[str] = [
        f"Repository: {context.repo_path}",
        f"Branch: {state.ref.branch} (tracking={state.ref.tracking or 'none'})",
        f"Estimated cost: {plan.estimated_cost:.2f}",
        "Actions:",
    ]
    for index, action in enumerate(plan.actions, start=1):
        lines.append(f"  {index}. {action.name} (cost={action.cost:.2f})")
        if action.rationale:
            lines.append(f"     reason: {action.rationale}")
        if action.params:
            lines.append(f"     params: {json.dumps(action.params, ensure_ascii=False)}")
    typer.echo("\n".join(lines))


def _execute_workflow(context: WorkflowContext) -> dict[str, Any]:
    computation = _build_plan_payload(context)
    runner = context.build_action_runner()
    executor = Executor(
        planner=context.planner,
        observer=context.observer.observe,
        runner=runner,
        available_actions=computation.actions,
        goal=context.config.goal,
    )
    result = executor.execute(computation.state, computation.plan)

    return {
        "repository": str(context.repo_path),
        "initial_state": computation.state.model_dump(mode="json"),
        "initial_plan": computation.plan.model_dump(mode="json"),
        "executed_actions": [action.model_dump(mode="json") for action in result.executed_actions],
        "final_plan": result.final_plan.model_dump(mode="json"),
        "replanned": result.replanned,
        "command_history": list(context.action_facade.command_history),
        "dry_run": context.action_facade.dry_run,
    }


@app.command("run")
def run_command(
    repo: RepoOption = None,
    config: ConfigOption = None,
    json_output: JsonFlag = False,
    confirm: ConfirmFlag = False,
) -> None:
    """Execute the workflow plan, respecting the confirmation flag."""
    context = _prepare_context(
        repo,
        config,
        json_logs=json_output,
        dry_run_actions=not confirm,
        silence_logs=json_output,
    )
    payload = _safe_execute_workflow(context)

    if json_output:
        _emit_json(payload)
        return

    mode = "dry-run" if payload["dry_run"] else "confirmed"
    lines = [
        f"Mode: {mode}",
        f"Executed actions: {len(payload['executed_actions'])}",
    ]
    if payload["replanned"]:
        lines.append("A replanning step was triggered during execution.")
    else:
        lines.append("Plan executed without replanning.")
    for index, action in enumerate(payload["executed_actions"], start=1):
        lines.append(f"  {index}. {action['name']}")
    typer.echo("\n".join(lines))


@app.command("dry-run")
def dry_run_command(
    repo: RepoOption = None,
    config: ConfigOption = None,
    json_output: JsonFlag = False,
) -> None:
    """Simulate the run workflow ensuring no repository changes occur."""
    context = _prepare_context(
        repo,
        config,
        json_logs=json_output,
        dry_run_actions=True,
        silence_logs=json_output,
    )
    payload = _safe_execute_workflow(context)

    if json_output:
        _emit_json(payload)
        return

    lines = [
        "Mode: dry-run",
        f"Executed actions: {len(payload['executed_actions'])}",
        "Command history:",
    ]
    for entry in payload["command_history"]:
        command = _format_command(tuple(str(part) for part in entry.get("command", [])))
        lines.append(f"  - {command} (returncode={entry.get('returncode')}, dry_run={entry.get('dry_run')})")
    typer.echo("\n".join(lines))


@app.command("explain")
def explain_command(
    repo: RepoOption = None,
    config: ConfigOption = None,
    json_output: JsonFlag = False,
) -> None:
    """Explain each action in the computed plan."""
    context = _prepare_context(
        repo,
        config,
        json_logs=json_output,
        dry_run_actions=True,
        silence_logs=json_output,
    )
    computation = _safe_build_plan_payload(context)
    plan = computation.plan
    explanations = explain_plan(plan, contexts=build_action_contexts(context.config))

    if json_output:
        payload = {
            "repository": str(context.repo_path),
            "plan": plan.model_dump(mode="json"),
            "notes": list(plan.notes),
            "explanations": [
                {
                    "action": explanation.action.model_dump(mode="json"),
                    "reason": explanation.reason,
                    "alternatives": list(explanation.alternatives),
                    "cost": explanation.cost,
                }
                for explanation in explanations
            ],
        }
        _emit_json(payload)
        return

    lines = [
        f"Repository: {context.repo_path}",
        f"Plan estimated cost: {plan.estimated_cost:.2f}",
        "Explanations:",
    ]
    for index, explanation in enumerate(explanations, start=1):
        lines.append(f"  {index}. {explanation.action.name} (cost={explanation.cost:.2f})")
        lines.append(f"     reason: {explanation.reason}")
        if explanation.alternatives:
            lines.extend(
                f"     alternative: {alternative}"
                for alternative in explanation.alternatives
            )
    if plan.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in plan.notes)
    typer.echo("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the goapgit CLI and return the exit status."""
    command = typer.main.get_command(app)
    try:
        command.main(args=list(argv or []), prog_name="goapgit", standalone_mode=False)
    except SystemExit as exc:  # pragma: no cover - Typer propagates exit codes via SystemExit
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(main())
