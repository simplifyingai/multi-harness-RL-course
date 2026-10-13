"""Print hooks at every hand-off of one GRPO step, without changing the project.

``train_grpo`` (``mhrl/training/grpo.py``) calls a short list of names it
imported at the top of its file: ``run_rollout``, ``CaptureClient``,
``group_advantages``, ``dedupe_shared_prefixes``, ``item_from_sequence``,
``load_policy``, ``save_adapter``, ``use_adapter``, ``nn.value_and_grad`` and
``optim.AdamW``. Those names ARE the hand-offs: a rollout handed to the
grader, rewards handed to the advantage, rows handed to the loss, the update
handed to a file and the file handed to the engine. ``Tracer.listening``
swaps each name, in that module only, for a wrapper that calls the real one
and prints what went in or came out, and puts every original back when the
step is over. ``run_rollout`` itself asks ``mhrl.rollout.get_harness`` for
the harness, so that name is wrapped too, to show Claude Code's command.

Nothing here imports the project or MLX at the top, so the printing can be
tested anywhere; MLX is imported only inside the hooks that measure arrays.

The key: a rollout's capture session id is the harness's API key. It is
never printed whole. The session line shows its first characters, and in
Claude Code's command and environment every variable whose name says KEY,
TOKEN, SECRET or PASSWORD, and every appearance of the key itself, reads
``<redacted>``. Claude Code is given the key in its environment
(``ANTHROPIC_API_KEY``), not on its command line, so both are redacted.
"""

from __future__ import annotations

import contextlib
import json
import re
import shlex
import statistics
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from paths import short_path, tilde

REDACTED = "<redacted>"
# A secret's name ends in one of these words, whole or after an underscore
# (ANTHROPIC_API_KEY, APIKEY, GITHUB_TOKEN, DB_PASSWD), while
# CLAUDE_CODE_MAX_OUTPUT_TOKENS, a count, does not.
SECRET_NAME = re.compile(
    r"(^|_)(API_?KEY|KEY|TOKEN|SECRETS?|PASSWORD|PASSWD|CREDENTIALS?|AUTH)$",
    re.IGNORECASE,
)
SESSION_SHOWN = 8  # characters of a session id shown on screen
INSTRUCTION_SHOWN = 48  # characters of the task text shown in a command
WIDTH = 100
INDENT = "      "
WINDOW_BEFORE = 3  # harness tokens shown before a row's first trained token
WINDOW_AFTER = 5  # trained tokens shown from it
RATIOS_SHOWN = 5
HTTP_TIMEOUT_S = 10
MEGABYTE = 1_000_000
PERCENT = 100


# ------------------------------------------------------------- redaction


def is_secret_name(name: str) -> bool:
    """Say whether an environment variable's name marks a secret.

    Args:
        name: The variable's name.

    Returns:
        True for names ending in KEY, TOKEN, SECRET, PASSWORD, PASSWD,
        CREDENTIALS or AUTH.
    """
    return bool(SECRET_NAME.search(name))


def redact_text(text: str, secret: str) -> str:
    """Replace every appearance of a secret in a text.

    Args:
        text: Any text.
        secret: The secret; an empty one changes nothing.

    Returns:
        The text with the secret replaced by ``<redacted>``.
    """
    return text.replace(secret, REDACTED) if secret else text


def redact_env(env: dict[str, str], secret: str) -> dict[str, str]:
    """Redact an environment for printing.

    Args:
        env: The variables.
        secret: The run's key.

    Returns:
        A copy where secret-named variables, and the key wherever it
        appears, read ``<redacted>``.
    """
    return {
        name: REDACTED if is_secret_name(name) else redact_text(value, secret)
        for name, value in env.items()
    }


def redact_argv(argv: Sequence[str], secret: str) -> list[str]:
    """Redact a command line for printing.

    Args:
        argv: The program and its arguments.
        secret: The run's key.

    Returns:
        A copy with the key replaced wherever it appears, including inside
        ``--flag=value`` arguments.
    """
    return [redact_text(arg, secret) for arg in argv]


def short_id(session_id: str) -> str:
    """Show the start of a session id only.

    Args:
        session_id: The id.

    Returns:
        Its first characters (never more than half of it) and an ellipsis.
    """
    return session_id[: min(SESSION_SHOWN, len(session_id) // 2)] + "…"


# ------------------------------------------------------------- formatting


def temp_roots() -> list[str]:
    """Return the temp folder's spellings, longest first.

    ``tempfile.mkdtemp`` makes folders under ``$TMPDIR``; macOS writes that
    as ``/var/folders/...`` and resolves it to ``/private/var/folders/...``.

    Returns:
        Both spellings, without a trailing slash.
    """
    root = tempfile.gettempdir().rstrip("/")
    spellings = {root, str(Path(root).resolve())}
    return sorted(spellings, key=len, reverse=True)


def shorten(text: str) -> str:
    """Write the temp folder as ``$TMPDIR`` and the home folder as ``~``.

    Args:
        text: A path or a command line.

    Returns:
        The text, shorter and without the learner's user name.
    """
    for root in temp_roots():
        text = text.replace(root, "$TMPDIR")
    return tilde(text)


def wrap(words: Sequence[str], first: str, rest: str = INDENT) -> list[str]:
    """Lay words out after a prefix, wrapping at ``WIDTH``.

    Args:
        words: The pieces, each kept whole.
        first: The first line's prefix.
        rest: Every later line's prefix.

    Returns:
        The lines.
    """
    lines: list[str] = []
    current, started = first, False
    for word in words:
        if started and len(current) + 1 + len(word) > WIDTH:
            lines.append(current)
            current, started = rest, False
        current += f" {word}" if started else word
        started = True
    lines.append(current)
    return lines


def argv_lines(argv: Sequence[str], secret: str, instruction: str) -> list[str]:
    """Print a harness's command line, the task text shortened, the key hidden.

    Args:
        argv: The command.
        secret: The run's key.
        instruction: The task text (shown shortened).

    Returns:
        The lines.
    """
    shown = []
    for arg in redact_argv(argv, secret):
        if arg == instruction and len(arg) > INSTRUCTION_SHOWN:
            arg = arg[:INSTRUCTION_SHOWN] + f"… ({len(instruction)} characters)"
        shown.append(shorten(shlex.quote(arg)))
    return wrap(shown, "argv  ")


def env_lines(env: dict[str, str], secret: str) -> list[str]:
    """Print a harness's own environment variables, the key hidden.

    Args:
        env: The variables.
        secret: The run's key.

    Returns:
        The lines.
    """
    shown = redact_env(env, secret)
    pairs = [f"{name}={shorten(value)}" for name, value in shown.items()]
    return wrap(pairs, "env   ")


def call_lines(document: dict[str, Any]) -> list[str]:
    """Print each model call the capture proxy recorded for one rollout.

    Args:
        document: The proxy's rollout document.

    Returns:
        One line per call: prompt tokens, sampled tokens, tools offered and
        why the reply ended; auxiliary calls (no tools, discarded) say so.
    """
    lines = []
    for number, turn in enumerate(document.get("turns") or [], start=1):
        aux = " · auxiliary, not trained" if turn.get("discarded") else ""
        lines.append(
            f"call {number} · prompt {int(turn.get('n_prompt') or 0):,} tokens · "
            f"sampled {int(turn.get('n_sampled') or 0)} · "
            f"tools {int(turn.get('n_tools') or 0)} · "
            f"{turn.get('finish_reason') or 'no finish reason'}{aux}"
        )
    return lines


def rollout_summary(document: dict[str, Any]) -> str:
    """Summarise a rollout document's shape in one line.

    Args:
        document: The proxy's rollout document.

    Returns:
        Its calls, roots, rows and trainable gate.
    """
    stats = document.get("stats") or {}
    return (
        f"captured {len(document.get('turns') or [])} calls · "
        f"roots {stats.get('n_roots', '?')} · "
        f"rows {stats.get('n_trainable_sequences', '?')} trainable of "
        f"{stats.get('n_sequences', '?')} · trainable "
        f"{str(bool(document.get('trainable'))).lower()}"
    )


def reward_line(
    correct: bool,
    submitted: str | None,
    reason: str,
    n_sampled: int,
    reward: float,
    budget: int,
    max_bonus: float,
) -> str:
    """Explain one rollout's grade and reward.

    Args:
        correct: The grader's verdict.
        submitted: What ``answer.txt`` held.
        reason: The grader's explanation.
        n_sampled: Tokens the model generated across the rollout.
        reward: The reward the project computed.
        budget: The token budget (``Settings.token_budget``).
        max_bonus: The largest bonus (``Defaults.MAX_TOOL_BONUS``).

    Returns:
        The verdict and the reward's arithmetic.
    """
    if not correct:
        return f"grade wrong ({reason}) · reward {reward:g}: a wrong answer earns 0"
    return (
        f"grade correct (answer {submitted!r}) · reward 1 + {max_bonus:g} x "
        f"(1 - {n_sampled}/{budget}) = {reward:.4f}"
    )


def group_lines(
    rewards: Sequence[float], advantages: Sequence[float] | None
) -> list[str]:
    """Print a group's rewards and advantages, or why it teaches nothing.

    Args:
        rewards: One reward per rollout, in the group's order.
        advantages: ``group_advantages``'s answer; None for no contrast.

    Returns:
        The lines.
    """
    shown = " ".join(f"{r:.4f}" for r in rewards)
    if advantages is None:
        return [
            f"rewards {shown}",
            "no contrast: all 4 rewards are equal, so every advantage would be 0",
            "and the group teaches nothing (chapter 5). train_grpo skips it and",
            "makes no update.",
        ]
    mean = statistics.fmean(rewards)
    spread = statistics.pstdev(rewards)
    return [
        f"rewards {shown} · mean {mean:.4f} · spread {spread:.4f}",
        "advantages (reward - mean) / spread: "
        + " ".join(f"{a:+.2f}" for a in advantages),
    ]


def matched_advantages(
    rewards: Sequence[float],
    advantages: Sequence[float],
    labelled: Sequence[tuple[str, float]],
) -> list[tuple[str, float, float]]:
    """Give each labelled rollout its advantage, matching by reward.

    The rollouts are labelled in the order they started, while the group
    lists them in the order they were queued. Equal rewards get equal
    advantages, so matching by reward is exact.

    Args:
        rewards: The group's rewards, in its order.
        advantages: Their advantages, in the same order.
        labelled: ``(label, reward)`` for each rollout.

    Returns:
        ``(label, reward, advantage)``, in the labels' order.
    """
    by_reward: dict[float, float] = dict(zip(rewards, advantages, strict=True))
    return [(label, reward, by_reward[reward]) for label, reward in labelled]


def row_window(
    ids: Sequence[int],
    mask: Sequence[int],
    logprobs: Sequence[float | None],
    decode: Callable[[int], str],
) -> list[str]:
    """Show a few tokens around a row's first trained token.

    Args:
        ids: The row's token ids.
        mask: Its loss mask (1 = sampled by the model, trained).
        logprobs: The engine's log-probability per position.
        decode: Turns one token id into its text.

    Returns:
        One line per token: position, mask, old logprob and text.
    """
    trained = [i for i, m in enumerate(mask) if m and i > 0]
    if not trained:
        return ["(no trained token)"]
    start = trained[0]
    lines = ["position  mask  old logprob  token"]
    for index in range(max(start - WINDOW_BEFORE, 0), start + WINDOW_AFTER):
        if index >= len(ids):
            break
        old = logprobs[index] if index < len(logprobs) else None
        old_text = f"{old:11.4f}" if mask[index] and old is not None else " " * 11
        lines.append(
            f"{index:8,d}  {mask[index]:4d}  {old_text}  {decode(ids[index])!r}"
        )
    return lines


def ratio_line(number: int, ratios: Sequence[float], loss: float) -> str:
    """Summarise one row's importance ratios and loss.

    Args:
        number: The row's number in the step.
        ratios: New probability / old probability, per trained token.
        loss: The row's summed clipped loss.

    Returns:
        The line.
    """
    first = " ".join(f"{r:.3f}" for r in ratios[:RATIOS_SHOWN])
    return (
        f"row {number}: {len(ratios)} ratios, first {first} · min {min(ratios):.3f} "
        f"max {max(ratios):.3f} mean {statistics.fmean(ratios):.4f} · "
        f"loss {loss:+.4f}"
    )


def engine_delta(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Say what the engine did between two reads of ``/admin/stats``.

    The rollout document has no field for cached prompt tokens; the engine
    counts them, so they are read from its running counters.

    Args:
        before: The counters at the start.
        after: The counters at the end.

    Returns:
        Requests, prompt tokens (and how many came from the prompt cache),
        sampled tokens and model time.
    """

    def grew(name: str) -> float:
        return float(after.get(name, 0)) - float(before.get(name, 0))

    prompt, cached = grew("prompt_tokens"), grew("cached_tokens")
    share = f" ({PERCENT * cached / prompt:.0f}%)" if prompt else ""
    return (
        f"engine  {grew('requests'):.0f} requests · {prompt:,.0f} prompt tokens, "
        f"{cached:,.0f} of them from its prompt cache{share} · "
        f"{grew('completion_tokens'):,.0f} sampled · "
        f"{grew('model_seconds'):.1f} s of model time"
    )


def block(prefix: str, lines: Sequence[str]) -> list[str]:
    """Put a label before a block's first line and indent the rest under it.

    Args:
        prefix: The label, e.g. ``[r1] ``.
        lines: The block.

    Returns:
        The labelled lines.
    """
    if not lines:
        return []
    pad = " " * len(prefix)
    return [prefix + lines[0], *(pad + line for line in lines[1:])]


def get_json(url: str) -> dict[str, Any]:
    """GET a URL and parse its JSON body.

    Args:
        url: The address.

    Returns:
        The body.

    Raises:
        urllib.error.URLError: If it cannot be reached.
        ValueError: If the body is not a JSON object.
    """
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"{url} did not answer with a JSON object")
    return body


# ------------------------------------------------------------- patching


class Namespace:
    """A stand-in for a module: some names replaced, the rest forwarded."""

    def __init__(self, real: Any, **replaced: Any) -> None:
        """Wrap a module.

        Args:
            real: The module.
            **replaced: The names to answer differently.
        """
        self._real = real
        self._replaced = replaced

    def __getattr__(self, name: str) -> Any:
        """Answer a name: the replacement if there is one, else the module's.

        Args:
            name: The attribute.

        Returns:
            Its value.
        """
        replaced = self.__dict__["_replaced"]
        return replaced[name] if name in replaced else getattr(self._real, name)


# The hooks wrap the project's functions without importing its types, so
# their signatures say ``Any`` where the project's would name a class.
@contextlib.contextmanager
def patched(module: ModuleType, **names: Any) -> Iterator[None]:
    """Swap names in one module, and put every original back afterwards.

    Args:
        module: The module whose globals change.
        **names: The new values.

    Yields:
        Nothing; the swap holds inside the ``with`` block.

    Raises:
        AttributeError: If the module has no such name (a hand-off renamed
            in the project would otherwise be silently missed).
    """
    originals = {name: getattr(module, name) for name in names}
    try:
        for name, value in names.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


# ------------------------------------------------------------- the tracer


@dataclass
class RolloutTrace:
    """What one rollout's hooks saw.

    Attributes:
        label: ``r1`` to ``r4``, in the order the rollouts started.
        argv: Its harness's command line, as printed.
        env: Its harness's own environment variables, as printed.
        state: Where its harness kept its private state.
        document: The proxy's rollout document.
    """

    label: str
    argv: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    state: list[str] = field(default_factory=list)
    document: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutcome:
    """How a step ended, as the hooks saw it.

    Attributes:
        flat_groups: Groups ``group_advantages`` found without contrast.
        updated: Whether the optimizer made an update.
        adapter: The adapter file saved after the update.
        engine_adapter: What the engine's ``/health`` reported after the swap.
    """

    flat_groups: int = 0
    updated: bool = False
    adapter: Path | None = None
    engine_adapter: str = ""


class Tracer:
    """Prints every hand-off of ``train_grpo``, through the names it calls."""

    def __init__(
        self,
        settings: Any,
        say: Callable[[str], None] = print,
        all_calls: bool = False,
    ) -> None:
        """Create the tracer.

        Args:
            settings: The project's ``Settings``.
            say: Where each line goes.
            all_calls: Print every rollout's command and calls, not only the
                first rollout's.
        """
        self.settings = settings
        self.say = say
        self.all_calls = all_calls
        self.lock = threading.Lock()
        self.local = threading.local()
        self.traces: list[RolloutTrace] = []
        self.records: list[Any] = []
        self.outcome = StepOutcome()
        self.engine_before: dict[str, Any] = {}
        self.cached_policy: tuple[Any, Any] | None = None
        self.tokenizer: Any = None
        self.rows_shown = 0
        self.rows: list[tuple[Any, Any, Any]] = []  # (item, loss, ratio)
        self.remasked = 0
        self.budget = int(settings.token_budget)
        self.max_bonus = 0.0
        self.secrets: set[str] = set()  # every session id seen: the keys

    # -- helpers

    def redact(self, text: str) -> str:
        """Hide every session id this tracer has seen, wherever it appears.

        The last pass before anything is printed: a key can reach a line by
        a path no other rule knows (a model that echoes its environment into
        ``answer.txt``, a URL in an error message).

        Args:
            text: Any text.

        Returns:
            The text with each known session id replaced by ``<redacted>``.
        """
        for secret in self.secrets:
            text = redact_text(text, secret)
        return text

    def emit(self, lines: Sequence[str]) -> None:
        """Print lines as one block, so threads never interleave inside it.

        Every line goes through ``redact`` first.

        Args:
            lines: The lines.
        """
        with self.lock:
            for line in lines:
                self.say(self.redact(line))

    def current(self) -> RolloutTrace | None:
        """Return the rollout this thread is running, if any.

        Returns:
            Its trace, or None outside a rollout.
        """
        return getattr(self.local, "trace", None)

    def detailed(self, trace: RolloutTrace) -> bool:
        """Say whether a rollout's command and calls are printed in full.

        Args:
            trace: The rollout.

        Returns:
            True for the first rollout, or for all with ``all_calls``.
        """
        return self.all_calls or trace.label == "r1"

    def engine_stats(self) -> dict[str, Any]:
        """Read the engine's running counters.

        Returns:
            ``/admin/stats``, or an empty dict when it cannot be read.
        """
        try:
            return get_json(f"{self.settings.engine_url}/admin/stats")
        except (urllib.error.URLError, ValueError, TimeoutError):
            return {}

    def start_draw(self) -> None:
        """Reset what one call of ``train_grpo`` collects."""
        self.traces, self.records, self.rows = [], [], []
        self.outcome = StepOutcome()
        self.rows_shown = self.remasked = 0
        self.engine_before = self.engine_stats()

    # -- the hand-offs

    def load_policy(self, real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``load_policy``: load once, print what was loaded.

        A draw whose group has no contrast makes no update, so the weights
        are unchanged and the next draw can reuse them.

        Args:
            real: The project's ``load_policy``.

        Returns:
            The wrapper.
        """

        def load(settings: Any) -> tuple[Any, Any]:
            if self.cached_policy is not None:
                self.emit(["load_policy  the model loaded for the last draw, reused"])
                return self.cached_policy
            started = time.perf_counter()
            model, tokenizer = real(settings)
            from mlx.utils import tree_flatten  # only where arrays are measured

            weights = tree_flatten(model.trainable_parameters())
            trainable = sum(value.size for _, value in weights)
            self.emit(
                [
                    f"load_policy  {short_path(Path(settings.model_path))} + LoRA rank "
                    f"{settings.lora_rank}, scale {settings.lora_scale}, last "
                    f"{settings.lora_layers} layers: {trainable:,} trainable numbers "
                    f"({time.perf_counter() - started:.1f} s)"
                ]
            )
            self.cached_policy = (model, tokenizer)
            self.tokenizer = tokenizer
            return model, tokenizer

        return load

    def use_adapter(self, real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``use_adapter``: show the POST and what the engine now holds.

        Args:
            real: The project's ``use_adapter``.

        Returns:
            The wrapper.
        """

        def use(settings: Any, path: Path | None) -> None:
            real(settings, path)
            try:
                held = str(get_json(f"{settings.engine_url}/health").get("adapter"))
            except (urllib.error.URLError, ValueError, TimeoutError) as error:
                held = f"unreadable ({error})"
            body = json.dumps({"path": tilde(str(path)) if path else None})
            self.emit(
                [
                    f"use_adapter  POST {settings.engine_url}/admin/adapter {body}",
                    f"{INDENT}the engine's /health now says adapter {tilde(held)!r}",
                ]
            )
            if path is not None and self.outcome.updated:
                self.outcome.engine_adapter = held

        return use

    def capture_client(self, real: type) -> type:
        """Subclass ``CaptureClient`` so each rollout's document is kept.

        Args:
            real: The project's ``CaptureClient``.

        Returns:
            The subclass.
        """
        tracer = self

        class TracingCaptureClient(real):  # type: ignore[misc, valid-type]
            """The project's client; it also hands each document to the tracer."""

            def rollout(self, session_id: str) -> dict[str, Any]:
                """Fetch the rollout document, as the project does.

                Args:
                    session_id: The session.

                Returns:
                    The proxy's document.
                """
                document = super().rollout(session_id)
                trace = tracer.current()
                if trace is not None:
                    trace.document = document
                return document

        return TracingCaptureClient

    def get_harness(self, real: Callable[[str], Any]) -> Callable[[str], Any]:
        """Wrap ``get_harness`` so the harness shows its command and state.

        Args:
            real: ``mhrl.rollout.get_harness``.

        Returns:
            The wrapper.
        """

        def get(name: str) -> Any:
            harness = real(name)
            trace = self.current()
            if trace is None or not self.detailed(trace):
                return harness
            command, environment = harness.command, harness.environment

            def traced_command(ctx: Any) -> list[str]:
                argv = command(ctx)
                trace.argv = argv_lines(argv, ctx.api_key, ctx.instruction)
                trace.state = [
                    f"state {shorten(str(ctx.state_dir))} (tempfile.mkdtemp: its "
                    "HOME, config and TMPDIR; deleted after the run)"
                ]
                return argv

            def traced_environment(ctx: Any) -> dict[str, str]:
                env = environment(ctx)
                trace.env = env_lines(env, ctx.api_key)
                return env

            harness.command = traced_command
            harness.environment = traced_environment
            return harness

        return get

    def run_rollout(self, real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``run_rollout``: the folder, the key, the calls, the reward.

        Args:
            real: The project's ``run_rollout``.

        Returns:
            The wrapper.
        """

        def run(*args: Any, **kwargs: Any) -> Any:
            with self.lock:
                trace = RolloutTrace(f"r{len(self.traces) + 1}")
                self.traces.append(trace)
            self.local.trace = trace

            def on_session(session_id: str, workdir: Path) -> None:
                with self.lock:
                    self.secrets.add(session_id)
                held = ", ".join(sorted(p.name for p in workdir.iterdir()))
                self.emit(
                    [
                        f"[{trace.label}] folder {short_path(workdir)} holds {held}",
                        f"[{trace.label}] API key = the session id "
                        f"{short_id(session_id)} (POST /sessions on the proxy)",
                    ]
                )

            try:
                record = real(*args, on_session=on_session, **kwargs)
            finally:
                self.local.trace = None
            self.finish_rollout(trace, record)
            return record

        return run

    def finish_rollout(self, trace: RolloutTrace, record: Any) -> None:
        """Print one finished rollout's block.

        Args:
            trace: What its hooks collected.
            record: The project's ``RolloutRecord``.
        """
        lines = [*trace.argv, *trace.env, *trace.state]
        if self.detailed(trace):
            lines += call_lines(trace.document)
        lines.append(rollout_summary(trace.document))
        lines.append(
            reward_line(
                record.correct,
                record.submitted,
                record.reason,
                record.n_sampled,
                record.reward,
                self.budget,
                self.max_bonus,
            )
            + f" · {record.seconds:.0f} s"
        )
        prefix = f"[{trace.label}] "
        with self.lock:
            self.records.append((trace.label, record))
        self.emit(block(prefix, lines))

    def group_advantages(self, real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``group_advantages``: the group's rewards become advantages.

        Args:
            real: The project's ``group_advantages``.

        Returns:
            The wrapper.
        """

        def advantages(rewards: list[float]) -> list[float] | None:
            result = real(rewards)
            self.emit([engine_delta(self.engine_before, self.engine_stats())])
            lines = group_lines(rewards, result)
            if result is None:
                self.outcome.flat_groups += 1
            else:
                labelled = sorted(
                    ((label, record.reward) for label, record in self.records),
                    key=lambda pair: int(pair[0][1:]),
                )
                for label, reward, advantage in matched_advantages(
                    rewards, result, labelled
                ):
                    lines.append(f"{label} reward {reward:.4f} -> {advantage:+.2f}")
            self.emit(block("group  ", lines))
            return result

        return advantages

    def dedupe_shared_prefixes(self, real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``dedupe_shared_prefixes``: count the tokens it re-masks.

        Args:
            real: The project's ``dedupe_shared_prefixes``.

        Returns:
            The wrapper.
        """

        def dedupe(sequences: list[dict]) -> list[dict]:
            out = real(sequences)
            before = sum(sum(s["loss_mask"]) for s in sequences)
            self.remasked += before - sum(sum(s["loss_mask"]) for s in out)
            return out

        return dedupe

    def item_from_sequence(self, real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``item_from_sequence``: show the first row in detail.

        Args:
            real: The project's ``item_from_sequence``.

        Returns:
            The wrapper.
        """

        def item(sequence: dict, advantage: float = 1.0) -> Any:
            made = real(sequence, advantage)
            if made is None or self.rows_shown:
                return made
            self.rows_shown += 1
            decode = self.decoder()
            lines = [
                f"row    {len(made.input_ids):,} tokens: {made.n_tokens} the model "
                f"sampled (mask 1, trained), the other "
                f"{len(made.input_ids) - made.n_tokens:,} only read (mask 0) · "
                f"advantage {advantage:+.2f}",
                *(
                    "       " + line
                    for line in row_window(
                        sequence["input_ids"],
                        sequence["loss_mask"],
                        sequence.get("logprobs") or [],
                        decode,
                    )
                ),
            ]
            self.emit(lines)
            return made

        return item

    def decoder(self) -> Callable[[int], str]:
        """Return a token decoder: the model's tokenizer when it is loaded.

        Returns:
            A function from a token id to its text.
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            return lambda token: f"<{token}>"
        return lambda token: tokenizer.decode([token])

    def value_and_grad(self, real_nn: Any) -> Callable[..., Any]:
        """Wrap ``nn.value_and_grad`` so each row's loss and ratios are kept.

        They are printed once the project has evaluated them (at the update),
        so the hook never forces an evaluation the project would not make.

        Args:
            real_nn: The ``mlx.nn`` module.

        Returns:
            A ``value_and_grad`` that records each row.
        """

        def value_and_grad(model: Any, fn: Callable[..., Any]) -> Callable[..., Any]:
            real = real_nn.value_and_grad(model, fn)

            def run(item: Any) -> Any:
                (loss, ratio), grads = real(item)
                self.rows.append((item, loss, ratio))
                return (loss, ratio), grads

            return run

        return value_and_grad

    def adamw(self, real_optim: Any) -> type:
        """Subclass ``AdamW`` so the update prints the rows' ratios and its size.

        Args:
            real_optim: The ``mlx.optimizers`` module.

        Returns:
            The subclass.
        """
        tracer = self

        class TracedAdamW(real_optim.AdamW):  # type: ignore[misc, name-defined]
            """The project's optimizer; it also reports each update."""

            def update(self, model: Any, gradients: Any) -> None:
                """Apply the update, as the project does, and measure it.

                Args:
                    model: The model whose trainable weights change.
                    gradients: The summed gradients.
                """
                import mlx.core as mx
                from mlx.utils import tree_flatten

                tracer.print_losses()
                before = dict(tree_flatten(model.trainable_parameters()))
                super().update(model, gradients)
                after = dict(tree_flatten(model.trainable_parameters()))
                largest = [mx.abs(after[k] - before[k]).max() for k in after]
                mx.eval(largest)
                numbers = sum(v.size for v in after.values())
                tracer.outcome.updated = True
                tracer.emit(
                    [
                        f"AdamW  one update of {numbers:,} adapter numbers · learning "
                        f"rate {self.learning_rate.item():g}, bias correction "
                        f"{'on' if self.bias_correction else 'off'} · largest change "
                        f"{max(float(x) for x in largest):.2e}"
                    ]
                )

        return TracedAdamW

    def print_losses(self) -> None:
        """Print each trained row's ratios and loss (already evaluated)."""
        if not self.rows:
            return
        tokens = sum(item.n_tokens for item, _, _ in self.rows)
        total = sum(float(loss) for _, loss, _ in self.rows)
        lines = [
            f"rows   {len(self.rows)} trained, {tokens:,} tokens · "
            f"dedupe_shared_prefixes re-masked {self.remasked} repeated tokens"
        ]
        for number, (_, loss, ratio) in enumerate(self.rows, start=1):
            lines.append("       " + ratio_line(number, ratio.tolist(), float(loss)))
        lines.append(f"       loss per token {total / tokens:+.4f}")
        self.emit(lines)

    def save_adapter(self, real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``save_adapter``: show the file written.

        Args:
            real: The project's ``save_adapter``.

        Returns:
            The wrapper.
        """

        def save(model: Any, path: Path) -> Path:
            written = real(model, path)
            if self.outcome.updated and self.outcome.adapter is None:
                self.outcome.adapter = written
                size = written.stat().st_size / MEGABYTE
                self.emit([f"save   {short_path(written)} ({size:.1f} MB)"])
            return written

        return save

    # -- putting it together

    @contextlib.contextmanager
    def listening(self, grpo: ModuleType, rollout: ModuleType) -> Iterator[None]:
        """Swap every hand-off name for its hook while the block runs.

        Args:
            grpo: ``mhrl.training.grpo``.
            rollout: ``mhrl.rollout``.

        Yields:
            Nothing.
        """
        self.max_bonus = float(grpo.Defaults.MAX_TOOL_BONUS)
        with (
            patched(
                grpo,
                load_policy=self.load_policy(grpo.load_policy),
                use_adapter=self.use_adapter(grpo.use_adapter),
                CaptureClient=self.capture_client(grpo.CaptureClient),
                run_rollout=self.run_rollout(grpo.run_rollout),
                group_advantages=self.group_advantages(grpo.group_advantages),
                dedupe_shared_prefixes=self.dedupe_shared_prefixes(
                    grpo.dedupe_shared_prefixes
                ),
                item_from_sequence=self.item_from_sequence(grpo.item_from_sequence),
                save_adapter=self.save_adapter(grpo.save_adapter),
                nn=Namespace(grpo.nn, value_and_grad=self.value_and_grad(grpo.nn)),
                optim=Namespace(grpo.optim, AdamW=self.adamw(grpo.optim)),
            ),
            patched(rollout, get_harness=self.get_harness(rollout.get_harness)),
        ):
            yield
