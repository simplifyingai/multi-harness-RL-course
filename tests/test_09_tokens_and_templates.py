"""Chapter 9: tokens, the chat template, spellings and the context window.

Most tests need no network and no model files: the download is tested against
a fake Hub, the Jinja set-up against a tiny template and tokenizer made here,
and the spelling search against a tiny vocabulary. The tests that need
LFM2.5-2.6B's real tokenizer files read them from where the chapter's own run
put them (``$COURSE_HOME/09-tokens-and-templates/LFM2.5-2.6B``, read only) and
are skipped, with the reason, when they are not there.

The render is checked against ``tests/data/09_project_render.json``: the token
ids the multi-harness-rl project's own ``PolicyEngine.render`` produced for the
same chats (``tests/project_render_09.py`` made it). Set ``MHRL_PROJECT`` to a
checkout of the project to re-run the project's render and compare it too.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.error
import warnings
from email.message import Message
from pathlib import Path
from types import ModuleType

import jinja2
import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("09-tokens-and-templates")

import chats  # noqa: E402
import count  # noqa: E402
import hub  # noqa: E402
import lfm  # noqa: E402
import report  # noqa: E402
import spellings  # noqa: E402
import template  # noqa: E402
import tokens  # noqa: E402

# Where a real run of the chapter keeps the tokenizer files. Worked out when
# this file is imported, before conftest points COURSE_HOME at a temporary
# folder; the tests only ever read from it.
REAL_FOLDER = hub.model_folder()
HAVE_FILES = not hub.missing_files(REAL_FOLDER)
# Set to 1 (as a check before publishing should) to fail, not skip, when the
# real files are missing.
REQUIRE_ENV = "MHRL_REQUIRE_TOKENIZER"
needs_files = pytest.mark.skipif(
    not HAVE_FILES,
    reason=(
        "LFM2.5-2.6B's tokenizer files are not downloaded: run "
        "`uv run tokens.py` in 09-tokens-and-templates once"
    ),
)
FIXTURE_PATH = ROOT / "tests" / "data" / "09_project_render.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
GENERATOR = ROOT / "tests" / "project_render_09.py"
PROJECT_ENV = "MHRL_PROJECT"
PROJECT_TIMEOUT_S = 600

THINK_ID = 124901
IM_START_ID = 124899
ASSISTANT_ID = 63514
NEWLINE_ID = 207
TINY_VOCAB = {"[UNK]": 0, "<s>": 1, "hi": 2, "there": 3, "<": 4}


def run_main(module: ModuleType, *args: str) -> str:
    """Run a script's main() with arguments, and return what it printed.

    Args:
        module: The chapter module.
        *args: Command-line arguments.

    Returns:
        Everything printed.
    """
    saved = sys.argv
    sys.argv = [f"{module.__name__}.py", *args]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        sys.argv = saved
    return out.getvalue()


@pytest.fixture(scope="module")
def tokenizer() -> lfm.LfmTokenizer:
    """Load the real tokenizer files once for the tests that need them.

    Returns:
        The model's tokenizer.
    """
    if not HAVE_FILES:
        pytest.skip("LFM2.5-2.6B's tokenizer files are not downloaded")
    return lfm.LfmTokenizer(REAL_FOLDER)


@pytest.fixture
def real_files(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the scripts read the real files, as if they were in COURSE_HOME.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        The real folder.
    """
    if not HAVE_FILES:
        pytest.skip("LFM2.5-2.6B's tokenizer files are not downloaded")
    monkeypatch.setattr(hub, "model_folder", lambda: REAL_FOLDER)
    return REAL_FOLDER


# --------------------------------------------------------------- the download


class FakeResponse(io.BytesIO):
    """What urlopen returns, made from bytes in memory."""

    def __enter__(self) -> FakeResponse:
        """Open it as a context manager.

        Returns:
            Itself.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close it."""
        self.close()


def fake_hub(monkeypatch: pytest.MonkeyPatch, contents: dict[str, bytes]) -> list[str]:
    """Replace the Hub with a dict of files, and the three files with them.

    Args:
        monkeypatch: pytest's patcher.
        contents: File name -> bytes the fake Hub serves.

    Returns:
        The list of addresses fetched, filled in as the test runs.
    """
    fetched: list[str] = []
    files = tuple(
        hub.HubFile(name, len(data), hashlib.sha256(data).hexdigest())
        for name, data in contents.items()
    )
    monkeypatch.setattr(hub, "FILES", files)

    def urlopen(url: str, timeout: float) -> FakeResponse:
        del timeout
        fetched.append(url)
        return FakeResponse(contents[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(hub.urllib.request, "urlopen", urlopen)
    return fetched


def test_files_go_under_course_home(course_home: Path) -> None:
    assert hub.model_folder() == course_home / "09-tokens-and-templates" / (
        "LFM2.5-2.6B"
    )


def test_urls_name_the_pinned_revision() -> None:
    url = hub.file_url(hub.TOKENIZER_JSON)
    assert url == (
        "https://huggingface.co/LiquidAI/LFM2.5-2.6B/resolve/"
        f"{hub.REVISION}/tokenizer.json"
    )


def test_only_the_three_tokenizer_files_and_about_18_mb() -> None:
    names = [item.name for item in hub.FILES]
    assert names == ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    total = sum(item.size for item in hub.FILES)
    assert round(total / hub.MEGABYTE) == 18


def test_download_writes_checked_files_and_no_leftovers(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    fetched = fake_hub(monkeypatch, {"a.json": b"{}", "b.jinja": b"{{ x }}"})
    lines: list[str] = []
    folder = hub.ensure_files(lines.append)
    assert folder.is_relative_to(course_home)
    assert (folder / "a.json").read_bytes() == b"{}"
    assert sorted(path.name for path in folder.iterdir()) == ["a.json", "b.jinja"]
    assert len(fetched) == 2
    assert "tokenizer files only" in lines[0]
    assert sum("SHA-256 checked" in line for line in lines) == 2
    # A second call finds both files at full size and fetches nothing.
    hub.ensure_files(lines.append)
    assert len(fetched) == 2


def test_a_short_file_is_fetched_again(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    del course_home
    fetched = fake_hub(monkeypatch, {"a.json": b"{}"})
    folder = hub.ensure_files(lambda line: None)
    (folder / "a.json").write_bytes(b"{")
    assert [item.name for item in hub.missing_files(folder)] == ["a.json"]
    hub.ensure_files(lambda line: None)
    assert len(fetched) == 2
    assert (folder / "a.json").read_bytes() == b"{}"


def test_wrong_bytes_are_refused_and_nothing_is_kept(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    del course_home
    fake_hub(monkeypatch, {"a.json": b"{}"})
    monkeypatch.setattr(hub, "FILES", (hub.HubFile("a.json", 2, "0" * 64),))
    with pytest.raises(hub.HubError, match="not the expected file"):
        hub.ensure_files(lambda line: None)
    assert list(hub.model_folder().iterdir()) == []


def test_an_unreachable_hub_gives_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    del course_home

    def offline(url: str, timeout: float) -> FakeResponse:
        raise urllib.error.URLError("nodename nor servname provided")

    monkeypatch.setattr(hub.urllib.request, "urlopen", offline)
    with pytest.raises(hub.HubError) as caught:
        hub.ensure_files(lambda line: None)
    message = str(caught.value)
    assert "could not download tokenizer.json from the Hugging Face Hub" in message
    assert hub.file_url(hub.TOKENIZER_JSON) in message
    assert "nodename nor servname provided" in message
    assert "Check your connection" in message
    assert list(hub.model_folder().iterdir()) == []


def test_an_http_error_names_the_status_not_the_connection(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    del course_home

    def not_found(url: str, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(hub.urllib.request, "urlopen", not_found)
    with pytest.raises(hub.HubError) as caught:
        hub.ensure_files(lambda line: None)
    message = str(caught.value)
    assert "The Hub answered HTTP 404" in message
    assert "Check your connection" not in message


def test_too_many_bytes_are_refused(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    del course_home
    fake_hub(monkeypatch, {"a.json": b"{}"})
    monkeypatch.setattr(
        hub.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"{}{}")
    )
    with pytest.raises(hub.HubError, match="not the expected file"):
        hub.ensure_files(lambda line: None)
    assert list(hub.model_folder().iterdir()) == []


def test_a_disk_error_is_explained_and_leaves_nothing(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    del course_home
    fake_hub(monkeypatch, {"a.json": b"{}"})

    def full_disk(self: Path, target: Path) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", full_disk)
    with pytest.raises(
        hub.HubError, match="could not download or write .*No space left"
    ):
        hub.ensure_files(lambda line: None)
    assert list(hub.model_folder().iterdir()) == []


def test_sizes_are_written_as_people_say_them() -> None:
    assert hub.size_text(363) == "363 bytes"
    assert hub.size_text(17_911_404) == "17.9 MB"


def test_scripts_exit_with_the_message_when_offline(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    del course_home

    def offline(url: str, timeout: float) -> FakeResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(hub.urllib.request, "urlopen", offline)
    for module in (tokens, template, spellings, count):
        with pytest.raises(SystemExit) as caught:
            run_main(module, "--fast")
        assert "could not download" in str(caught.value.code)


# ------------------------------------------------- the Jinja set-up, in small


@pytest.fixture
def tiny_folder(tmp_path: Path) -> Path:
    """Write a tiny tokenizer and template that use what LFM2.5's template uses.

    Args:
        tmp_path: pytest's per-test folder.

    Returns:
        A folder with the three files.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    folder = tmp_path / "tiny"
    folder.mkdir()
    tiny = Tokenizer(WordLevel(TINY_VOCAB, unk_token="[UNK]"))
    tiny.pre_tokenizer = Whitespace()
    tiny.save(str(folder / hub.TOKENIZER_JSON.name))
    config = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}
    (folder / hub.TOKENIZER_CONFIG.name).write_text(json.dumps(config))
    source = (
        "{{- bos_token -}}\n"
        "{%- for message in messages -%}\n"
        "    {%- generation -%}{{ message.content }}{%- endgeneration -%}\n"
        "{%- endfor -%}\n"
        "{%- if tools %}|{{ tools[0] | tojson }}{% endif -%}\n"
        "{%- if preserve_thinking %}|keep{% endif -%}\n"
        "{%- if add_generation_prompt %}|go{% endif -%}\n"
        "{%- if messages | length > 2 %}{{ raise_exception('too long') }}{% endif -%}"
    )
    (folder / hub.CHAT_TEMPLATE.name).write_text(source)
    return folder


def test_the_generation_tag_renders_its_body(tiny_folder: Path) -> None:
    tiny = lfm.LfmTokenizer(tiny_folder)
    text = tiny.apply_chat_template([chats.user("hi"), chats.user("there")])
    assert text == "<s>hithere"


def test_flags_and_special_tokens_reach_the_template(tiny_folder: Path) -> None:
    tiny = lfm.LfmTokenizer(tiny_folder)
    text = tiny.apply_chat_template(
        [chats.user("hi")], add_generation_prompt=True, preserve_thinking=True
    )
    assert text == "<s>hi|keep|go"
    assert tiny.render([chats.user("hi")], None) == tiny.encode("<s>hi|keep|go")


def test_tojson_keeps_key_order_and_characters(tiny_folder: Path) -> None:
    tiny = lfm.LfmTokenizer(tiny_folder)
    tool = {"z": 1, "a": "<b>&'é"}
    text = tiny.apply_chat_template([chats.user("hi")], tools=[tool])
    # Jinja's own tojson would sort the keys and escape <, >, & and '.
    assert text == '<s>hi|{"z": 1, "a": "<b>&\'é"}'


def test_the_template_can_refuse_a_chat(tiny_folder: Path) -> None:
    tiny = lfm.LfmTokenizer(tiny_folder)
    with pytest.raises(jinja2.TemplateError, match="too long"):
        tiny.apply_chat_template([chats.user("hi")] * 3)


def test_encode_adds_nothing(tiny_folder: Path) -> None:
    tiny = lfm.LfmTokenizer(tiny_folder)
    assert tiny.encode("hi there") == [2, 3]
    assert tiny.decode([2, 3]) == "hi there"


# ------------------------------------------------ spelling search, in small


def tiny_table(*pieces: str) -> spellings.PieceTable:
    """Build a vocabulary of a few pieces, ids in the given order.

    Args:
        *pieces: The pieces' texts.

    Returns:
        The table.
    """
    ids = {piece: index for index, piece in enumerate(pieces)}
    return spellings.PieceTable(ids, max(map(len, pieces)), 0)


def test_spellings_are_counted_by_length() -> None:
    table = tiny_table("a", "b", "c", "ab", "bc", "abc")
    assert spellings.count_by_length("abc", table) == {1: 1, 2: 2, 3: 1}
    assert spellings.count_by_length("abd", table) == {}


def test_spellings_of_one_length_are_listed() -> None:
    table = tiny_table("a", "b", "c", "ab", "bc", "abc")
    two = spellings.spellings_of_length("abc", table, 2, limit=10)
    assert sorted(two) == [[0, 4], [3, 2]]
    assert len(spellings.spellings_of_length("abc", table, 2, limit=1)) == 1


def test_the_search_skips_paths_that_cannot_end_at_the_length() -> None:
    # 60 a's: about 10**12 spellings, yet the one spelling in 60 tokens and the
    # 59 in 59 tokens come back at once.
    table = tiny_table("a", "aa")
    text = "a" * 60
    assert spellings.spellings_of_length(text, table, 60, limit=5) == [[0] * 60]
    assert len(spellings.spellings_of_length(text, table, 59, limit=100)) == 59


def test_a_long_text_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_load() -> None:
        raise AssertionError("the tokenizer was loaded")

    monkeypatch.setattr(spellings, "load_tokenizer", must_not_load)
    with pytest.raises(SystemExit, match="at most 200 characters"):
        run_main(spellings, "x" * 201)


def test_glued_pieces_are_two_pieces_side_by_side() -> None:
    table = tiny_table("a", "b", "ab", "abc")
    assert spellings.glued_pieces(table) == 1  # "ab"; "abc" has no "c" or "bc"


# -------------------------------------------------------- small pure helpers


def test_common_prefix() -> None:
    assert template.common_prefix([1, 2, 3], [1, 2, 4, 5]) == 2
    assert template.common_prefix([1, 2], [1, 2]) == 2
    assert template.common_prefix([], [1]) == 0


def test_tool_call_is_pulled_out_of_a_reply() -> None:
    reply = chats.AGENT_CALLS[0][0] + "<|im_end|>"
    assert report.tool_call(reply) == "bash(command='sqlite3 data.db .tables')"
    assert report.tool_call("plain text") == "plain text"


def test_agent_chat_grows_by_a_reply_and_an_output_per_call() -> None:
    chat_roles = [chats.agent_chat(done) for done in range(3)]
    roles = [[message["role"] for message in chat] for chat in chat_roles]
    assert roles == [
        ["user"],
        ["user", "assistant", "tool"],
        ["user", "assistant", "tool", "assistant", "tool"],
    ]


def test_bash_tool_is_mini_swe_agents() -> None:
    function = chats.BASH_TOOL["function"]
    assert function["name"] == "bash"
    assert function["description"] == "Execute a bash command"
    assert function["parameters"]["required"] == ["command"]


def test_the_limits_are_the_projects() -> None:
    assert count.MODEL_POSITIONS == 131_072
    assert count.ENGINE_CAP == 16_384
    assert tokens.MODEL_ROWS == 128_000


# ------------------------------------- the real tokenizer: the spec's numbers


def test_say_when_the_real_tokenizer_tests_are_skipped() -> None:
    if HAVE_FILES:
        return
    message = (
        f"LFM2.5-2.6B's tokenizer files are not in {REAL_FOLDER}, so every test "
        "of the real numbers and of the project's render was skipped; run "
        "`uv run tokens.py` in 09-tokens-and-templates once"
    )
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(message)
    warnings.warn(message, UserWarning, stacklevel=1)


@needs_files
def test_the_vocabulary(tokenizer: lfm.LfmTokenizer) -> None:
    vocabulary = tokens.vocabulary(tokenizer)
    assert (vocabulary.size, vocabulary.learned, vocabulary.added) == (
        125_017,
        124_893,
        124,
    )
    assert vocabulary.model_rows - vocabulary.size == 2_983


@needs_files
def test_the_special_token_ids(tokenizer: lfm.LfmTokenizer) -> None:
    ids = {row.text: row.token_id for row in tokens.special_tokens(tokenizer)}
    assert ids == {
        "<|startoftext|>": 124894,
        "<|im_start|>": 124899,
        "<|im_end|>": 124900,
        "<think>": 124901,
        "</think>": 124902,
        "<|tool_call_start|>": 124905,
        "<|tool_call_end|>": 124906,
        "<|pad|>": 124893,
    }
    assert tokens.other_added(tokenizer) == tokens.OtherAdded(116, 105)


@needs_files
def test_the_question_round_trips(tokenizer: lfm.LfmTokenizer) -> None:
    trip = tokens.round_trip(tokenizer, chats.QUESTION)
    assert trip.ids == [3522, 1196, 10699, 3737, 296, 4741, 39]
    assert trip.pieces[1] == " many"
    assert trip.back == chats.QUESTION


@needs_files
def test_answer_txt_has_two_spellings_that_decode_alike(
    tokenizer: lfm.LfmTokenizer,
) -> None:
    assert tokenizer.encode("answer.txt") == [44244, 31544]
    assert tokenizer.decode([44244, 22, 62758]) == "answer.txt"
    table = spellings.piece_table(tokenizer)
    found = spellings.text_spellings(tokenizer, table, "answer.txt")
    assert [44244, 22, 62758] in found.one_longer
    assert found.total == 168
    assert all(ids == [44244, 31544] for ids in found.encoded_again)


@needs_files
def test_one_question_with_the_bash_tool(tokenizer: lfm.LfmTokenizer) -> None:
    plain = template.render_prompt(tokenizer, chats.one_question())
    tools = [chats.BASH_TOOL]
    with_tool = template.render_prompt(tokenizer, chats.one_question(), tools)
    assert len(plain.ids) == 17
    assert len(with_tool.ids) == 88
    assert "<|im_start|>system\nList of tools: [" in with_tool.text
    assert with_tool.ids[-4:] == [IM_START_ID, ASSISTANT_ID, NEWLINE_ID, THINK_ID]


@needs_files
def test_old_reasoning_is_dropped_unless_preserved(
    tokenizer: lfm.LfmTokenizer,
) -> None:
    result = template.compare_thinking(tokenizer, chats.three_user_turns())
    assert len(result.dropped.ids) == 70
    assert len(result.kept.ids) == 89
    assert result.dropped.ids != result.kept.ids
    assert "<think>The customers" not in result.dropped.text
    assert "<think>The customers" in result.kept.text


@needs_files
def test_every_prompt_extends_the_last(tokenizer: lfm.LfmTokenizer) -> None:
    calls = count.replay(tokenizer)
    assert [len(call.prompt) for call in calls] == [105, 150, 198]
    assert [call.prefix_kept for call in calls] == [True, True, None]
    default = count.replay(tokenizer, preserve_thinking=False)
    assert [call.prompt for call in calls] == [call.prompt for call in default]


@needs_files
def test_reading_the_database_overflows_the_engine(
    tokenizer: lfm.LfmTokenizer,
) -> None:
    result = count.overflow(tokenizer, count.DATABASE)
    assert result.file_bytes == 45_056
    assert count.ENGINE_CAP < result.prompt_tokens < count.MODEL_POSITIONS


# ---------------------------- the real tokenizer: the same ids as the project


@needs_files
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_render_matches_the_projects(
    tokenizer: lfm.LfmTokenizer, case: dict[str, object]
) -> None:
    messages, tools = case["messages"], case["tools"]
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        preserve_thinking=case["preserve_thinking"],
    )
    assert tokenizer.encode(text) == case["ids"]
    if case["preserve_thinking"]:
        assert tokenizer.render(messages, tools) == case["ids"]


def test_the_fixture_covers_the_chapters_chats() -> None:
    names = {case["name"] for case in FIXTURE["cases"]}
    assert {"one-question-bash", "three-user-turns-default", "agent-call-3"} <= names
    by_name = {case["name"]: case for case in FIXTURE["cases"]}
    assert by_name["one-question-bash"]["messages"] == chats.one_question()
    assert by_name["one-question-bash"]["tools"] == [chats.BASH_TOOL]
    assert by_name["agent-call-3"]["messages"] == chats.agent_chat(2)
    assert FIXTURE["made_with"]["model_path"] == "LFM2.5-2.6B-MLX-4bit"


def test_the_fixture_is_the_projects_own_render(tmp_path: Path) -> None:
    project = os.environ.get(PROJECT_ENV)
    if not project:
        pytest.skip(
            f"set {PROJECT_ENV} to a checkout of the multi-harness-rl project "
            "(with its .venv) to re-run its own render"
        )
    python = Path(project).expanduser() / ".venv" / "bin" / "python"
    # Keep the project's libraries from writing into the real ~/.cache.
    env = {
        **os.environ,
        "HF_HOME": str(tmp_path / "hf"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    result = subprocess.run(
        [str(python), str(GENERATOR)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=PROJECT_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr
    fresh = json.loads(result.stdout)
    assert fresh["cases"] == FIXTURE["cases"]


# ------------------------------------------------- the scripts, end to end


@needs_files
def test_tokens_prints_the_round_trip_and_the_ids(real_files: Path) -> None:
    del real_files
    out = run_main(tokens, "--fast")
    assert "ids     [3522, 1196, 10699, 3737, 296, 4741, 39]" in out
    assert "the vocabulary: 125,017 entries" in out
    assert "the model keeps 128,000 rows, one per id: 2,983 are never used" in out
    assert "<|im_end|>            124900" in out


@needs_files
def test_tokens_takes_any_text(real_files: Path) -> None:
    del real_files
    out = run_main(tokens, "--fast", "answer.txt")
    assert "ids     [44244, 31544]" in out


@needs_files
def test_template_prints_three_renders(real_files: Path) -> None:
    del real_files
    out = run_main(template, "--fast")
    assert "the prompt, 88 tokens:" in out
    assert "the template's default, 70 tokens:" in out
    assert "with preserve_thinking=True, as the project renders, 89 tokens:" in out
    assert "the default dropped 19 tokens of reasoning" in out


@needs_files
def test_spellings_prints_answer_txt(real_files: Path) -> None:
    del real_files
    out = run_main(spellings, "--fast")
    assert "[44244, 22, 62758]   'answer' '.' 'txt'" in out
    assert "in all, 168 token sequences spell 'answer.txt':" in out


@needs_files
def test_count_prints_the_calls_and_the_overflow(real_files: Path) -> None:
    del real_files
    out = run_main(count, "--fast")
    assert "reply included: yes" in out
    assert "the limit is 16384" in out


def test_nothing_is_written_into_the_chapter_folder() -> None:
    written = [
        path
        for path in CHAPTER.rglob("*")
        if path.suffix in {".json", ".jinja", ".part"} or path.name.endswith(".txt")
    ]
    assert written == []
