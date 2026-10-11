# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "transformers==5.18.0",
# ]
# ///
"""Measure four builds of LFM2.5-2.6B side by side, and check ours against Liquid's.

The builds: the bfloat16 original (chapter 10), ours (convert.py), the
all-4-bit variant (``convert.py --all-4bit``) and Liquid's own 4-bit export,
``LiquidAI/LFM2.5-2.6B-MLX-4bit``. First, with no model loaded, ours is
compared with Liquid's file by file, and its weights tensor by tensor, with
MD5 fingerprints: byte-identical files are the strongest proof two builds
are the same. Then each build is loaded on its own (one model in memory at a
time, memory freed in between) and measured the same way: size on disk,
memory once loaded, prompt and reply speed, the average log-probability it
gives 20 fixed SQL texts, greedy agreement with the bfloat16 original on 20
questions, and the pass rate on the project's 24 held-out SQL questions,
graded by chapter 1's grader. Every decoding is greedy, so a rerun gives the
same replies; only the timings vary.

Run it with ``uv run compare.py`` after ``uv run convert.py`` and
``uv run convert.py --all-4bit`` (about half an hour on an M3 Pro).
``--fast`` measures only Liquid's export, the copy chapter 10's ``--fast``
downloaded.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import report
from lfm import (
    LoadedModel,
    chat_prompt,
    load_folder,
    load_tokenizer_only,
    release_memory,
)
from lfm_files import (
    ALL_FOUR_BIT_FLAG,
    BF16,
    CARD_FILE,
    CHAPTER_10,
    CONFIG_FILE,
    CONVERT_COMMAND,
    DOWNLOAD_COMMAND,
    FAST_FLAG,
    FINISH_LENGTH,
    FINISH_STOP,
    FOUR_BIT,
    KEY_ALL_FOUR,
    KEY_BF16,
    KEY_LIQUID,
    KEY_OURS,
    QUANTIZATION_KEY,
    SAFETENSORS_GLOB,
    WEIGHTS_FILE,
    all_four_bit_folder,
    build_ready,
    chapter_home,
    missing_files,
    model_folder,
    ours_folder,
)
from mlx_lm import stream_generate
from sql_questions import (
    Graded,
    agreement_question,
    agreement_tasks,
    grade_reply,
    pass_rate_tasks,
    single_turn_prompt,
)
from tasks import Task
from tensors import HeaderError, file_md5, tensor_md5s

SCRIPT = "compare.py"
TITLE = "four builds of LFM2.5-2.6B, measured on this laptop"
AGREEMENT_TOKENS = 128  # the start of each reply is compared, token by token
ANSWER_TOKENS = 1_024  # the project's output cap in training and evaluation
WARM_UP_TOKENS = 1
JSON_SUFFIX = ".json"
# The files a build is made of, compared one by one between ours and Liquid's.
COMPARED_FILES = (
    WEIGHTS_FILE,
    "model.safetensors.index.json",
    CONFIG_FILE,
    "tokenizer.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "generation_config.json",
    CARD_FILE,
)
DIFF_LINES_SHOWN = 2  # differing lines quoted from a text file
NAMES_SHOWN = 3  # differing tensor or key names listed; the rest counted
RESULTS_FILE = "compare.json"
RESULTS_FILE_FAST = "compare-fast.json"


@dataclass(frozen=True)
class Build:
    """One build of the model to measure.

    Attributes:
        key: A short id, such as ``"ours"``.
        label: Its name in the tables.
        folder: Where its files are.
        made_by: The command that makes it, for when it is missing.
    """

    key: str
    label: str
    folder: Path
    made_by: str


@dataclass(frozen=True)
class Reply:
    """One greedy reply, with mlx-lm's own timing.

    Attributes:
        prompt_tokens: The prompt's length.
        tokens: The reply's token ids (the end-of-turn token not included).
        text: The reply's text.
        finish_reason: ``"stop"`` (end of turn) or ``"length"`` (the cap).
        prompt_seconds: Time spent reading the prompt.
        reply_seconds: Time spent writing the reply.
    """

    prompt_tokens: int
    tokens: list[int]
    text: str
    finish_reason: str
    prompt_seconds: float
    reply_seconds: float


@dataclass(frozen=True)
class Answer:
    """A build's reply to one pass-rate question, graded.

    Attributes:
        task_id: The question.
        reply_tokens: The reply's length in tokens.
        finish_reason: ``"stop"`` or ``"length"``.
        graded: The grader's verdict.
        tokens: The reply's token ids, to compare builds.
    """

    task_id: str
    reply_tokens: int
    finish_reason: str
    graded: Graded
    tokens: list[int] = field(repr=False)


@dataclass
class BuildResult:
    """Everything measured for one build.

    Attributes:
        build: The build.
        disk_bytes: Its weights files' size.
        loaded_bytes: MLX's memory in use once it is loaded.
        weight_bytes: Its arrays' bytes.
        load_seconds: How long loading took.
        peak_bytes: MLX's peak memory over all its tests.
        logprob_total: The summed log-probability of the SQL texts.
        logprob_tokens: How many tokens those texts are.
        greedy: Its greedy reply (token ids) to each agreement question.
        forced: Its top token at each position of the reference replies.
        answers: Its graded reply to each pass-rate question.
        prompt_seconds: Time reading the pass-rate prompts.
        prompt_tokens: Their total length.
        reply_seconds: Time writing the pass-rate replies.
        reply_tokens: Their total length.
        seconds: How long all its tests took.
    """

    build: Build
    disk_bytes: int
    loaded_bytes: int
    weight_bytes: int
    load_seconds: float
    peak_bytes: int = 0
    logprob_total: float = 0.0
    logprob_tokens: int = 0
    greedy: list[list[int]] = field(default_factory=list)
    forced: list[list[int]] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)
    prompt_seconds: float = 0.0
    prompt_tokens: int = 0
    reply_seconds: float = 0.0
    reply_tokens: int = 0
    seconds: float = 0.0

    def logprob_per_token(self) -> float:
        """Average the log-probability over the texts' tokens.

        Returns:
            The average (0 would mean certain; lower means less likely).
        """
        return self.logprob_total / self.logprob_tokens if self.logprob_tokens else 0.0

    def correct(self) -> int:
        """Count the pass-rate questions answered correctly.

        Returns:
            How many were correct.
        """
        return sum(answer.graded.correct for answer in self.answers)

    def prompt_speed(self) -> float:
        """Tokens of prompt read per second, over every pass-rate question.

        Returns:
            Tokens per second.
        """
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0

    def reply_speed(self) -> float:
        """Tokens of reply written per second, over every pass-rate question.

        Returns:
            Tokens per second.
        """
        return self.reply_tokens / self.reply_seconds if self.reply_seconds else 0.0


@dataclass(frozen=True)
class FileCheck:
    """One file of ours compared with the same file of Liquid's export.

    Attributes:
        name: The file name.
        identical: True when the two files hold the same bytes.
        ours_bytes: Our file's size (None if absent).
        theirs_bytes: Liquid's file's size (None if absent).
        note: What differs, in a few words ("" when identical).
    """

    name: str
    identical: bool
    ours_bytes: int | None
    theirs_bytes: int | None
    note: str


@dataclass(frozen=True)
class WeightsCheck:
    """Our weights file against Liquid's, whole and tensor by tensor.

    Attributes:
        ours_md5: Our file's MD5.
        theirs_md5: Liquid's file's MD5.
        tensors: How many tensors our file holds.
        identical: How many hold the same bytes in both files.
        differ: The names of those that differ.
        only_ours: Tensors only our file has.
        only_theirs: Tensors only Liquid's file has.
    """

    ours_md5: str
    theirs_md5: str
    tensors: int
    identical: int
    differ: list[str]
    only_ours: list[str]
    only_theirs: list[str]


def all_builds() -> list[Build]:
    """List the four builds, in the order they are measured.

    Returns:
        The builds: bfloat16 first, since every agreement is measured against it.
    """
    return [
        Build(
            KEY_BF16,
            "bfloat16",
            model_folder(BF16),
            f"{DOWNLOAD_COMMAND} ({CHAPTER_10})",
        ),
        Build(KEY_OURS, "ours", ours_folder(), CONVERT_COMMAND),
        Build(
            KEY_ALL_FOUR,
            "all 4-bit",
            all_four_bit_folder(),
            f"{CONVERT_COMMAND} {ALL_FOUR_BIT_FLAG}",
        ),
        Build(
            KEY_LIQUID,
            "Liquid's export",
            model_folder(FOUR_BIT),
            f"{DOWNLOAD_COMMAND} {FAST_FLAG} ({CHAPTER_10})",
        ),
    ]


def is_ready(build: Build) -> bool:
    """Say whether a build's files are all on disk.

    Args:
        build: The build.

    Returns:
        True when it can be loaded.
    """
    if build.key == KEY_BF16:
        return not missing_files(BF16, build.folder)
    if build.key == KEY_LIQUID:
        return not missing_files(FOUR_BIT, build.folder)
    return build_ready(build.folder)


def disk_bytes(folder: Path) -> int:
    """Add up the sizes of a build's weights files.

    Args:
        folder: The build's folder.

    Returns:
        Their total size in bytes.
    """
    return sum(path.stat().st_size for path in folder.glob(SAFETENSORS_GLOB))


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file that should hold an object.

    Args:
        path: The file.

    Returns:
        Its contents, or None if it is not a JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def names_text(names: list[str]) -> str:
    """List a few names and count the rest.

    Args:
        names: Names, in order.

    Returns:
        Such as ``"a, b and 3 more"``.
    """
    shown = ", ".join(names[:NAMES_SHOWN])
    rest = len(names) - NAMES_SHOWN
    return f"{shown} and {rest} more" if rest > 0 else shown


def json_difference(ours: dict[str, Any], theirs: dict[str, Any]) -> str:
    """Say in words how two JSON files differ, key by key.

    Args:
        ours: Our file's contents.
        theirs: Liquid's file's contents.

    Returns:
        A few words; quantization blocks are compared on their own.
    """
    parts = []
    only_theirs = [key for key in theirs if key not in ours]
    only_ours = [key for key in ours if key not in theirs]
    changed = [key for key in ours if key in theirs and ours[key] != theirs[key]]
    if QUANTIZATION_KEY in ours and QUANTIZATION_KEY in theirs:
        same = ours[QUANTIZATION_KEY] == theirs[QUANTIZATION_KEY]
        parts.append(f"quantization block {'identical' if same else 'DIFFERS'}")
    if only_theirs:
        parts.append(f"only Liquid's has {names_text(only_theirs)}")
    if only_ours:
        parts.append(f"only ours has {names_text(only_ours)}")
    if changed:
        parts.append(f"values differ for {names_text(changed)}")
    return "; ".join(parts) or "the same keys and values, written differently"


def text_difference(ours: str, theirs: str) -> str:
    """Say in words how two text files differ, line by line.

    Args:
        ours: Our file's text.
        theirs: Liquid's file's text.

    Returns:
        The first lines only one side has.
    """
    diff = list(difflib.ndiff(ours.splitlines(), theirs.splitlines()))
    added = [line[2:] for line in diff if line.startswith("+ ")]
    removed = [line[2:] for line in diff if line.startswith("- ")]
    parts = []
    if added:
        quoted = ", ".join(repr(line) for line in added[:DIFF_LINES_SHOWN])
        parts.append(f"Liquid's has {len(added)} more line(s): {quoted}")
    if removed:
        quoted = ", ".join(repr(line) for line in removed[:DIFF_LINES_SHOWN])
        parts.append(f"ours has {len(removed)} line(s) Liquid's lacks: {quoted}")
    return "; ".join(parts) or "the same lines, different line endings"


def compare_file(
    name: str, ours: Path, theirs: Path, weights: WeightsCheck | None = None
) -> FileCheck:
    """Compare one file of ours with the same file of Liquid's export.

    Args:
        name: The file name.
        ours: Our build's folder.
        theirs: Liquid's export's folder.
        weights: The weights check, whose MD5s settle the weights file
            without reading 3 GB again (None: compute them).

    Returns:
        Whether the bytes match and, if not, what differs.
    """
    mine, other = ours / name, theirs / name
    mine_bytes = mine.stat().st_size if mine.is_file() else None
    other_bytes = other.stat().st_size if other.is_file() else None
    if mine_bytes is None or other_bytes is None:
        missing = "ours" if mine_bytes is None else "Liquid's"
        return FileCheck(
            name, False, mine_bytes, other_bytes, f"{missing} has no {name}"
        )
    if name == WEIGHTS_FILE and weights is not None:
        same = weights.ours_md5 == weights.theirs_md5
    elif name == WEIGHTS_FILE:
        same = file_md5(mine) == file_md5(other)
    else:
        same = mine.read_bytes() == other.read_bytes()
    note = ""
    if not same and name == CARD_FILE:
        ours_lines = len(mine.read_text("utf-8").splitlines())
        theirs_lines = len(other.read_text("utf-8").splitlines())
        note = (
            f"different model cards ({ours_lines} and {theirs_lines} lines): "
            "mlx-lm writes its own"
        )
    elif not same and name.endswith(JSON_SUFFIX):
        mine_json, other_json = read_json(mine), read_json(other)
        if mine_json is None or other_json is None:
            note = "one of the two is not a JSON object"
        else:
            note = json_difference(mine_json, other_json)
    elif not same:
        note = text_difference(mine.read_text("utf-8"), other.read_text("utf-8"))
    return FileCheck(name, same, mine_bytes, other_bytes, note)


def compare_weights(ours: Path, theirs: Path) -> WeightsCheck:
    """Fingerprint both weights files, whole and one tensor at a time.

    Args:
        ours: Our build's folder.
        theirs: Liquid's export's folder.

    Returns:
        The comparison.
    """
    mine = {tensor.name: md5 for tensor, md5 in tensor_md5s(ours / WEIGHTS_FILE)}
    other = {tensor.name: md5 for tensor, md5 in tensor_md5s(theirs / WEIGHTS_FILE)}
    shared = [name for name in mine if name in other]
    return WeightsCheck(
        ours_md5=file_md5(ours / WEIGHTS_FILE),
        theirs_md5=file_md5(theirs / WEIGHTS_FILE),
        tensors=len(mine),
        identical=sum(mine[name] == other[name] for name in shared),
        differ=[name for name in shared if mine[name] != other[name]],
        only_ours=[name for name in mine if name not in other],
        only_theirs=[name for name in other if name not in mine],
    )


def same_prompts(ours: Path, theirs: Path, questions: Iterable[str]) -> tuple[int, int]:
    """Render every question with both builds' chat templates and compare tokens.

    Args:
        ours: Our build's folder.
        theirs: Liquid's export's folder.
        questions: The questions compare.py asks.

    Returns:
        How many render to identical tokens, out of how many.
    """
    mine, other = load_tokenizer_only(ours), load_tokenizer_only(theirs)
    asked = list(questions)
    same = sum(chat_prompt(mine, text) == chat_prompt(other, text) for text in asked)
    return same, len(asked)


def generate(loaded: LoadedModel, ids: list[int], max_tokens: int) -> Reply:
    """Write one greedy reply with mlx-lm's stream_generate.

    Args:
        loaded: The model.
        ids: The prompt's token ids.
        max_tokens: The length cap.

    Returns:
        The reply, with mlx-lm's measured speeds turned into seconds.
    """
    tokens: list[int] = []
    pieces: list[str] = []
    finish = FINISH_LENGTH
    last = None
    for step in stream_generate(
        loaded.model, loaded.tokenizer, ids, max_tokens=max_tokens
    ):
        last = step
        if step.finish_reason == FINISH_STOP:
            finish = FINISH_STOP  # its token is the end of turn, not the reply
            break
        tokens.append(step.token)
        pieces.append(step.text)
    if last is None:
        return Reply(len(ids), [], "", finish, 0.0, 0.0)
    prompt_seconds = last.prompt_tokens / last.prompt_tps if last.prompt_tps else 0.0
    reply_seconds = (
        last.generation_tokens / last.generation_tps if last.generation_tps else 0.0
    )
    return Reply(
        len(ids), tokens, "".join(pieces), finish, prompt_seconds, reply_seconds
    )


def text_logprob(loaded: LoadedModel, text: str) -> tuple[float, int]:
    """Measure how likely the model finds a text, token by token (chapter 3).

    The text is read in one pass after ``<|startoftext|>``; each token's
    log-probability is read off the model's scores for its position, in
    float32 (chapter 10: bfloat16 is too coarse for log-probabilities).

    Args:
        loaded: The model.
        text: The text.

    Returns:
        The summed log-probability, and how many tokens were scored.
    """
    tokenizer = loaded.tokenizer
    ids = [tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False)]
    logits = loaded.model(mx.array([ids[:-1]]))[0].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logprobs, mx.array(ids[1:])[:, None], axis=-1)
    return float(picked.sum().item()), len(ids) - 1


def forced_choices(
    loaded: LoadedModel, prompt: list[int], reply: list[int]
) -> list[int]:
    """Read the model's top token at every position of a given reply.

    The prompt and the reply are read in one pass; at each position the
    model's most likely next token is taken. Comparing these between builds
    measures agreement token by token, without one early difference changing
    everything after it.

    Args:
        loaded: The model.
        prompt: The prompt's token ids.
        reply: The reply's token ids (the reference's).

    Returns:
        The model's top token for each of the reply's positions.
    """
    if not reply:
        return []
    ids = prompt + reply
    logits = loaded.model(mx.array([ids[:-1]]))[0]
    choices = mx.argmax(logits[len(prompt) - 1 :], axis=-1)
    return [int(token) for token in choices.tolist()]


def warm_up(loaded: LoadedModel, ids: list[int]) -> None:
    """Run one untimed token, so one-time GPU set-up is not timed (chapter 10).

    Args:
        loaded: The model.
        ids: Any prompt.
    """
    for _ in stream_generate(
        loaded.model, loaded.tokenizer, ids, max_tokens=WARM_UP_TOKENS
    ):
        pass


def answer_question(loaded: LoadedModel, task: Task) -> tuple[Answer, Reply]:
    """Ask one pass-rate question and grade the reply.

    Args:
        loaded: The model.
        task: The question.

    Returns:
        The graded answer, and the reply with its timing.
    """
    ids = chat_prompt(loaded.tokenizer, single_turn_prompt(task))
    reply = generate(loaded, ids, ANSWER_TOKENS)
    graded = grade_reply(task, reply.text, cut_off=reply.finish_reason != FINISH_STOP)
    answer = Answer(
        task.task_id, len(reply.tokens), reply.finish_reason, graded, reply.tokens
    )
    return answer, reply


def measure(
    build: Build,
    reference: BuildResult | None,
    say: Callable[[str], None],
) -> BuildResult:
    """Load one build, run every test on it, and free it again.

    Args:
        build: The build.
        reference: The bfloat16 build's result, for agreement (None: skipped).
        say: Where progress lines go.

    Returns:
        Everything measured.
    """
    mx.reset_peak_memory()
    loaded = load_folder(build.folder)
    result = BuildResult(
        build=build,
        disk_bytes=disk_bytes(build.folder),
        loaded_bytes=loaded.active_bytes,
        weight_bytes=loaded.weight_bytes,
        load_seconds=loaded.seconds,
    )
    start = time.perf_counter()
    for task in agreement_tasks():
        total, count = text_logprob(loaded, task.gold_sql)
        result.logprob_total += total
        result.logprob_tokens += count
    questions = [
        chat_prompt(loaded.tokenizer, agreement_question(t)) for t in agreement_tasks()
    ]
    warm_up(loaded, questions[0])
    for ids in questions:
        result.greedy.append(generate(loaded, ids, AGREEMENT_TOKENS).tokens)
    references = reference.greedy if reference else result.greedy
    for ids, reply in zip(questions, references, strict=True):
        result.forced.append(forced_choices(loaded, ids, reply))
    for task in pass_rate_tasks():
        answer, reply = answer_question(loaded, task)
        result.answers.append(answer)
        result.prompt_tokens += reply.prompt_tokens
        result.prompt_seconds += reply.prompt_seconds
        result.reply_tokens += len(reply.tokens)
        result.reply_seconds += reply.reply_seconds
    result.seconds = time.perf_counter() - start
    result.peak_bytes = mx.get_peak_memory()
    say(report.progress_line(result))
    del loaded
    release_memory()
    return result


def exact_p(gained: int, lost: int) -> float:
    """The exact two-sided p-value for gained vs lost questions (McNemar).

    If two builds were equally good, each question one gets right and the
    other wrong would be a coin flip; this is the chance of a split at least
    this lopsided.

    Args:
        gained: Questions only the build got right.
        lost: Questions only the reference got right.

    Returns:
        The p-value (1.0 when nothing differs).
    """
    flips = gained + lost
    if flips == 0:
        return 1.0
    tail = sum(math.comb(flips, k) for k in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / 2**flips)


def save(
    results: list[BuildResult], extra: dict[str, Any], name: str, planned: int
) -> Path:
    """Write every measurement so far into this chapter's folder.

    It is written after every build, so a run that stops part way keeps
    what it measured, and says plainly that it is incomplete.

    Args:
        results: One result per build measured so far.
        extra: The file checks, if any.
        name: The file name.
        planned: How many builds the run set out to measure.

    Returns:
        The file written.
    """
    folder = chapter_home()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    data = {
        "complete": len(results) == planned,
        "builds_measured": len(results),
        "builds_planned": planned,
        "builds": [asdict(result) for result in results],
        **extra,
    }
    for build in data["builds"]:
        build["build"]["folder"] = str(build["build"]["folder"])
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


def missing_message(missing: list[Build]) -> str:
    """Explain which builds are missing and how to make them.

    Args:
        missing: The builds not on disk.

    Returns:
        A few lines naming each build's command.
    """
    lines = ["compare.py needs every build on disk; missing:"]
    lines += [f"  {build.label}: run `{build.made_by}`" for build in missing]
    lines.append(f"(or add {FAST_FLAG} to measure only Liquid's export)")
    return "\n".join(lines)


def main() -> None:
    """Check ours against Liquid's file by file, then measure every build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast", action="store_true", help="measure only Liquid's 4-bit export"
    )
    args = parser.parse_args()
    builds = all_builds()
    if args.fast:
        builds = [build for build in builds if build.key == KEY_LIQUID]
    missing = [build for build in builds if not is_ready(build)]
    if missing:
        raise SystemExit(missing_message(missing))

    report.print_title(SCRIPT, TITLE)
    report.print_builds(builds)
    report.print_tests(
        agreement_tasks(), pass_rate_tasks(), AGREEMENT_TOKENS, ANSWER_TOKENS
    )
    extra: dict[str, Any] = {}
    if not args.fast:
        ours, theirs = ours_folder(), model_folder(FOUR_BIT)
        try:
            weights = compare_weights(ours, theirs)
        except HeaderError as error:
            raise SystemExit(f"a weights file cannot be read: {error}") from error
        files = [compare_file(name, ours, theirs, weights) for name in COMPARED_FILES]
        questions = [agreement_question(t) for t in agreement_tasks()]
        questions += [single_turn_prompt(t) for t in pass_rate_tasks()]
        rendered = same_prompts(ours, theirs, questions)
        report.print_file_checks(files, weights, rendered)
        extra = {
            "files": [asdict(check) for check in files],
            "weights": asdict(weights),
            "prompts_rendered_the_same": list(rendered),
        }
    else:
        report.print_fast_skips()
    print()
    path = chapter_home() / (RESULTS_FILE_FAST if args.fast else RESULTS_FILE)
    results: list[BuildResult] = []
    for build in builds:
        reference = results[0] if results and results[0].build.key == KEY_BF16 else None
        results.append(measure(build, reference, print))
        save(results, extra, path.name, len(builds))  # after every build
    report.print_size_and_speed(results)
    report.print_quality(results, AGREEMENT_TOKENS)
    report.print_task_by_task(results, pass_rate_tasks(), exact_p)
    if not args.fast:
        report.print_ours_vs_liquid(
            results, extra["weights"]["ours_md5"] == extra["weights"]["theirs_md5"]
        )
    report.print_saved(path)


if __name__ == "__main__":
    main()
