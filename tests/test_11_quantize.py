"""Chapter 11: safetensors reading, the one-matrix fetch, builds and SQL grading.

No test here uses the network, loads a model or needs MLX. Safetensors files
are tiny ones written here; the Hub is a fake that serves byte ranges of such
a file and counts what it was asked for; the SQL questions run on the bundled
databases. MLX's side (the quantization rules, convert, compare, ask) is in
test_11_mlx.py.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import sqlite3
import struct
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
from chapters import ROOT, use_chapter

CHAPTER = use_chapter("11-quantize")

import grader  # noqa: E402
import lfm_files  # noqa: E402
import matrix_source  # noqa: E402
import sql_questions  # noqa: E402
import tasks  # noqa: E402
import tensors  # noqa: E402

CHAPTER_1 = ROOT / "01-the-job"


def safetensors_bytes(
    entries: dict[str, tuple[str, tuple[int, ...], bytes]],
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Write a safetensors file in memory, as the format lays one out.

    Args:
        entries: Each tensor's name, number type, shape and raw bytes.
        metadata: The optional ``__metadata__`` entry.

    Returns:
        The file's bytes: 8-byte header length, JSON header, then the data.
    """
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    data = b""
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        data += raw
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + data


TINY_MATRIX = bytes(range(32))  # 4 x 4 bfloat16 numbers: 32 bytes
TINY_ENTRIES = {
    "model.norm.weight": ("BF16", (4,), b"\x01\x02" * 4),
    "model.layers.0.w.weight": ("BF16", (4, 4), TINY_MATRIX),
    "model.layers.0.q.weight": ("U32", (2, 1), b"\xff" * 8),
}
TINY_PIN = matrix_source.MatrixPin(
    name="model.layers.0.w.weight",
    file="model-00001-of-00002.safetensors",
    dtype="BF16",
    shape=(4, 4),
    sha256=hashlib.sha256(TINY_MATRIX).hexdigest(),
)


# ---------------------------------------------------------------- tensors.py


def test_the_header_says_where_every_tensor_sits(tmp_path: Path) -> None:
    """Names, types, shapes and absolute positions come from the header alone."""
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(safetensors_bytes(TINY_ENTRIES, {"format": "mlx"}))
    header = tensors.read_header(path)
    assert header.metadata == {"format": "mlx"}
    assert [tensor.name for tensor in header.tensors] == list(TINY_ENTRIES)
    matrix = header.find("model.layers.0.w.weight")
    assert matrix.dtype == "BF16" and matrix.shape == (4, 4) and matrix.nbytes == 32
    assert tensors.read_tensor_bytes(path, matrix) == TINY_MATRIX
    assert header.data_bytes() + header.header_bytes == path.stat().st_size
    with pytest.raises(KeyError):
        header.find("no.such.tensor")


def test_every_tensor_is_fingerprinted_on_its_own(tmp_path: Path) -> None:
    """Each tensor's MD5 is the MD5 of its bytes; the file's is of all bytes."""
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(safetensors_bytes(TINY_ENTRIES))
    found = {tensor.name: md5 for tensor, md5 in tensors.tensor_md5s(path)}
    for name, (_, _, raw) in TINY_ENTRIES.items():
        assert found[name] == hashlib.md5(raw).hexdigest()
    assert tensors.file_md5(path) == hashlib.md5(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "data",
    [
        b"\x01\x02",  # shorter than the 8 length bytes
        struct.pack("<Q", 10**9) + b"{}",  # a header longer than the file
        struct.pack("<Q", 3) + b"{x}",  # not JSON
        safetensors_bytes({"t": ("BF16", (4,), b"\x00" * 6)}),  # 6 bytes for 4 bf16
        struct.pack("<Q", 13) + b'{"t": {"a":1}',  # broken JSON
    ],
)
def test_a_broken_file_is_refused(tmp_path: Path, data: bytes) -> None:
    """Short files, absurd lengths, bad JSON and wrong byte counts all stop."""
    path = tmp_path / "broken.safetensors"
    path.write_bytes(data)
    with pytest.raises(tensors.HeaderError):
        tensors.read_header(path)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"t": {"dtype": "BF16", "shape": [1]}}', "malformed"),
        (b'{"t": {"dtype": "X", "shape": [1], "data_offsets": [8, 4]}}', "make no"),
        (b'{"t": {"dtype": "X", "shape": [1], "data_offsets": [-4, 0]}}', "make no"),
        (b'{"t": {"dtype": "X", "shape": [1], "data_offsets": [0.5, 2]}}', "make no"),
        (b"[1, 2]", "not a JSON object"),
        (b'{"__metadata__": [1]}', "__metadata__"),
    ],
)
def test_a_malformed_entry_is_refused(raw: bytes, message: str) -> None:
    """Missing fields, impossible offsets and the wrong JSON shape all stop."""
    with pytest.raises(tensors.HeaderError, match=message):
        tensors.parse_header(raw)


def test_fingerprinting_a_cut_file_stops(tmp_path: Path) -> None:
    """A file shorter than its header says never yields a short MD5."""
    path = tmp_path / "cut.safetensors"
    path.write_bytes(safetensors_bytes(TINY_ENTRIES)[:-4])
    with pytest.raises(tensors.HeaderError, match="ends"):
        list(tensors.tensor_md5s(path))


def test_a_tensor_cut_short_is_refused(tmp_path: Path) -> None:
    """A file that ends inside a tensor says so."""
    path = tmp_path / "cut.safetensors"
    whole = safetensors_bytes(TINY_ENTRIES)
    path.write_bytes(whole)
    header = tensors.read_header(path)
    path.write_bytes(whole[:-4])
    with pytest.raises(tensors.HeaderError, match="ends inside"):
        tensors.read_tensor_bytes(path, header.tensors[-1])


# ----------------------------------------------------------- matrix_source.py


class FakeHub:
    """Serves byte ranges of one in-memory file, and counts what it was asked."""

    def __init__(self, data: bytes, fail: Exception | None = None) -> None:
        """Hold the file.

        Args:
            data: The file's bytes.
            fail: An error to raise on every request instead.
        """
        self.data = data
        self.fail = fail
        self.requests: list[tuple[str, int, int]] = []

    def __call__(self, url: str, first: int, last: int) -> bytes:
        """Answer one range request.

        Args:
            url: The address asked for.
            first: The first byte.
            last: The last byte (included).

        Returns:
            Those bytes.

        Raises:
            Exception: The configured failure, if any.
        """
        self.requests.append((url, first, last))
        if self.fail is not None:
            raise self.fail
        return self.data[first : last + 1]

    def bytes_sent(self) -> int:
        """Count every byte served.

        Returns:
            The total.
        """
        return sum(last - first + 1 for _, first, last in self.requests)


def big_file() -> bytes:
    """A file holding the tiny matrix beside a much larger tensor.

    Returns:
        Its bytes.
    """
    entries = {
        "model.embed.weight": ("BF16", (1000, 4), b"\x07" * 8000),
        **TINY_ENTRIES,
    }
    return safetensors_bytes(entries)


def test_the_matrix_alone_is_fetched_with_three_range_requests() -> None:
    """Header length, header, then exactly the matrix's bytes: no other tensor."""
    hub = FakeHub(big_file())
    data = matrix_source.fetch(TINY_PIN, hub)
    assert data == TINY_MATRIX
    assert len(hub.requests) == 3
    assert hub.requests[0][1:] == (0, 7)
    url = hub.requests[0][0]
    assert url.startswith("https://huggingface.co/LiquidAI/LFM2.5-2.6B/resolve/")
    assert lfm_files.BF16.revision in url and url.endswith(TINY_PIN.file)
    assert hub.bytes_sent() < len(hub.data) - 8000  # the big tensor never came


def test_fetched_bytes_must_be_the_pinned_matrix() -> None:
    """Bytes that fail the SHA-256 check, or the wrong shape, are refused."""
    wrong = dataclasses.replace(TINY_PIN, sha256="0" * 64)
    with pytest.raises(matrix_source.MatrixError, match="not the pinned matrix"):
        matrix_source.fetch(wrong, FakeHub(big_file()))
    shaped = dataclasses.replace(TINY_PIN, shape=(2, 8))
    with pytest.raises(matrix_source.MatrixError, match="expected BF16"):
        matrix_source.fetch(shaped, FakeHub(big_file()))


def test_no_network_says_what_to_do() -> None:
    """A failed connection becomes a plain message, not a traceback."""
    hub = FakeHub(b"", fail=urllib.error.URLError("no route to host"))
    with pytest.raises(matrix_source.MatrixError) as stopped:
        matrix_source.fetch(TINY_PIN, hub)
    assert "Check your connection" in str(stopped.value)
    assert "no route to host" in str(stopped.value)


def test_load_prefers_chapter_10s_download(course_home: Path) -> None:
    """With the download on disk, nothing is fetched, with or without --fast."""
    folder = lfm_files.model_folder(lfm_files.BF16)
    folder.mkdir(parents=True)
    (folder / TINY_PIN.file).write_bytes(big_file())
    hub = FakeHub(b"", fail=AssertionError("the network was used"))
    for fast in (False, True):
        matrix = matrix_source.load_matrix(fast, TINY_PIN, hub)
        assert matrix.data == TINY_MATRIX
        assert "chapter 10's download" in matrix.source
    assert hub.requests == []


def test_without_the_download_fast_fetches_once_then_reuses() -> None:
    """--fast fetches the matrix, keeps it, and reads the kept copy next time."""
    hub = FakeHub(big_file())
    first = matrix_source.load_matrix(True, TINY_PIN, hub)
    assert first.data == TINY_MATRIX and "the Hub" in first.source
    kept = matrix_source.cache_path(TINY_PIN)
    assert kept.read_bytes() == TINY_MATRIX
    assert not kept.with_name(kept.name + ".part").exists()
    calls = len(hub.requests)
    second = matrix_source.load_matrix(True, TINY_PIN, hub)
    assert second.data == TINY_MATRIX and "earlier run" in second.source
    assert len(hub.requests) == calls


def test_a_damaged_kept_copy_is_fetched_again() -> None:
    """A kept copy that fails its check is thrown away and fetched again."""
    kept = matrix_source.cache_path(TINY_PIN)
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"\x00" * len(TINY_MATRIX))
    hub = FakeHub(big_file())
    matrix = matrix_source.load_matrix(True, TINY_PIN, hub)
    assert matrix.data == TINY_MATRIX and len(hub.requests) == 3
    assert "the kept copy was damaged" in matrix.source
    assert kept.read_bytes() == TINY_MATRIX


class FakeResponse:
    """Stands in for what urllib.request.urlopen returns."""

    def __init__(self, status: int, data: bytes) -> None:
        """Hold a status and a body.

        Args:
            status: The HTTP status.
            data: The body.
        """
        self.status = status
        self.body = io.BytesIO(data)

    def __enter__(self) -> FakeResponse:
        """Enter the with block.

        Returns:
            Itself.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Leave the with block.

        Args:
            *args: The exception details, if any.
        """

    def read(self) -> bytes:
        """Return the body.

        Returns:
            The bytes.
        """
        return self.body.read()


def test_a_range_request_must_be_answered_as_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """206 with the right length passes; a whole file or a short answer stops."""
    asked: list[str] = []

    def answer(status: int, data: bytes) -> Callable[..., FakeResponse]:
        def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
            asked.append(str(request.get_header("Range")))
            return FakeResponse(status, data)

        return urlopen

    monkeypatch.setattr(matrix_source.urllib.request, "urlopen", answer(206, b"abcd"))
    assert matrix_source.urlopen_range("https://x", 10, 13) == b"abcd"
    assert asked == ["bytes=10-13"]
    monkeypatch.setattr(matrix_source.urllib.request, "urlopen", answer(200, b"abcd"))
    with pytest.raises(matrix_source.MatrixError, match="ignored the range"):
        matrix_source.urlopen_range("https://x", 10, 13)
    monkeypatch.setattr(matrix_source.urllib.request, "urlopen", answer(206, b"ab"))
    with pytest.raises(matrix_source.MatrixError, match="got 2"):
        matrix_source.urlopen_range("https://x", 10, 13)


def test_without_the_download_or_fast_both_ways_are_named() -> None:
    """The stop message names chapter 10's download and --fast."""
    with pytest.raises(SystemExit) as stopped:
        matrix_source.load_matrix(False, TINY_PIN, FakeHub(big_file()))
    message = str(stopped.value)
    assert "uv run download.py" in message and "--fast" in message
    assert "44 MB" in message


def test_the_real_matrix_is_pinned() -> None:
    """Layer 10's w1: 10,752 x 2,048 bfloat16 numbers, one fingerprint."""
    pin = matrix_source.W1_LAYER_10
    assert pin.name == "model.layers.10.feed_forward.w1.weight"
    assert pin.shape == (10_752, 2_048) and pin.dtype == "BF16"
    assert len(pin.sha256) == 64
    int(pin.sha256, 16)


# ---------------------------------------------------------------- lfm_files.py


def test_builds_live_under_course_home(course_home: Path) -> None:
    """Ours is shared with later chapters; the variant stays in the chapter."""
    assert lfm_files.ours_folder() == course_home / "models/lfm2.5-2.6b-4bit-ours"
    assert lfm_files.all_four_bit_folder() == (
        course_home / "11-quantize/lfm2.5-2.6b-4bit-all"
    )
    assert lfm_files.chapter_home() == course_home / "11-quantize"


def write_build(folder: Path) -> None:
    """Write the files that make a build complete (contents do not matter).

    Args:
        folder: The build's folder.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for name in lfm_files.BUILD_FILES:
        (folder / name).write_text("{}")


def write_official(folder: Path) -> None:
    """Write files at the official export's pinned sizes.

    Args:
        folder: The export's folder.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for item in lfm_files.FOUR_BIT.files:
        with (folder / item.name).open("wb") as handle:
            handle.truncate(item.size)  # a sparse file: the size is all that counts


def test_later_chapters_load_ours_else_liquids_export() -> None:
    """Ours when built; Liquid's export when not; a clear stop when neither."""
    with pytest.raises(SystemExit) as stopped:
        lfm_files.model_for_later_chapters()
    assert "uv run convert.py" in str(stopped.value)
    assert "download.py --fast" in str(stopped.value)
    write_official(lfm_files.model_folder(lfm_files.FOUR_BIT))
    folder, which = lfm_files.model_for_later_chapters()
    assert folder == lfm_files.model_folder(lfm_files.FOUR_BIT)
    assert "Liquid's export" in which and "convert.py" in which
    write_build(lfm_files.ours_folder())
    folder, which = lfm_files.model_for_later_chapters()
    assert folder == lfm_files.ours_folder() and which.startswith("ours")


def test_a_build_missing_a_file_is_not_ready(tmp_path: Path) -> None:
    """A half-written folder never counts as a build."""
    write_build(tmp_path)
    assert lfm_files.build_ready(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    assert not lfm_files.build_ready(tmp_path)


def test_missing_downloads_point_to_chapter_10() -> None:
    """A missing download names chapter 10's command and folder."""
    with pytest.raises(SystemExit) as stopped:
        lfm_files.require_repo(lfm_files.BF16)
    assert "`uv run download.py` in 10-meet-lfm" in str(stopped.value)
    with pytest.raises(SystemExit) as stopped:
        lfm_files.require_repo(lfm_files.FOUR_BIT)
    assert "download.py --fast" in str(stopped.value)


def test_gigabytes_and_gibibytes() -> None:
    """GB is 10**9 bytes, GiB is 2**30: Liquid's file is 1.58 GB, 1.47 GiB."""
    assert lfm_files.gigabytes(1_583_152_892) == "1.58 GB"
    assert lfm_files.gibibytes(1_583_152_892) == "1.47 GiB"
    assert lfm_files.gibibytes(19_327_352_832) == "18.00 GiB"


# ---------------------------------------------------------------- the data


@pytest.mark.parametrize(
    "name",
    ["grader.py", "data/train.jsonl", "data/test.jsonl"]
    + ["data/dbs/clinic.db", "data/dbs/airline.db"],
)
def test_chapter_1s_files_are_copied_byte_for_byte(name: str) -> None:
    """The grader and the data are chapter 1's own, unchanged."""
    assert (CHAPTER / name).read_bytes() == (CHAPTER_1 / name).read_bytes()


# ------------------------------------------------------------ sql_questions.py


def fake_task(domain: str, difficulty: str, number: int) -> tasks.Task:
    """Make a task with only the fields balanced() looks at filled in.

    Args:
        domain: Its database.
        difficulty: Its difficulty.
        number: Its number within the bucket.

    Returns:
        The task.
    """
    return tasks.Task(
        f"{domain}-{difficulty}-{number}",
        "test",
        domain,
        difficulty,
        "",
        "",
        "",
        "",
        "",
    )


def test_balanced_takes_round_robin_over_database_and_difficulty() -> None:
    """Sorted buckets, one task from each in turn, file order within a bucket."""
    pool = [fake_task("b", "easy", n) for n in range(3)]
    pool += [fake_task("a", "hard", n) for n in range(3)]
    pool += [fake_task("a", "easy", n) for n in range(3)]
    picked = [task.task_id for task in sql_questions.balanced(pool, 5)]
    assert picked == ["a-easy-0", "a-hard-0", "b-easy-0", "a-easy-1", "a-hard-1"]
    assert sql_questions.balanced(pool, 99) == pool


def test_the_24_held_out_questions_are_the_projects() -> None:
    """24 test questions: 12 per database, 8 per difficulty, chapter 1's ids."""
    picked = sql_questions.pass_rate_tasks()
    assert len(picked) == 24 and len({task.task_id for task in picked}) == 24
    assert all(task.split == "test" for task in picked)
    for domain in ("clinic", "airline"):
        assert sum(task.domain == domain for task in picked) == 12
    for level in ("easy", "medium", "hard"):
        assert sum(task.difficulty == level for task in picked) == 8
    assert picked[0].task_id == "airline-easy-001"


def test_the_20_agreement_questions_include_chapter_10s() -> None:
    """20 training questions; chapter 10's Paris question is one of them."""
    picked = sql_questions.agreement_tasks()
    assert len(picked) == 20 and all(task.split == "train" for task in picked)
    questions = [sql_questions.agreement_question(task) for task in picked]
    assert "How many customers live in Paris? Answer with one SQL query." in questions


def test_every_gold_query_passes_the_whole_pipeline() -> None:
    """Each held-out question's own gold SQL, as a reply, is graded correct."""
    for task in sql_questions.pass_rate_tasks():
        reply = f"some reasoning</think>```sql\n{task.gold_sql}\n```"
        graded = sql_questions.grade_reply(task, reply)
        assert graded.correct, (task.task_id, graded.reason)


def test_the_prompt_holds_the_tables_and_the_question() -> None:
    """The single-turn prompt shows CREATE TABLE statements, then the question."""
    task = sql_questions.pass_rate_tasks()[0]
    prompt = sql_questions.single_turn_prompt(task)
    assert prompt.count("CREATE TABLE") == 3
    assert "CREATE TABLE flights (" in prompt
    assert task.question in prompt and "```sql" in prompt
    assert "the answer and nothing else: one column" in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("```sql\nSELECT 1\n```</think>```sql\nSELECT 2\n```", "SELECT 2"),
        ("thinking</think>```\nSELECT 3\n```", "SELECT 3"),
        ("no think tags ```sql\nSELECT 4;\n``` and ```sql\nSELECT 5\n```", "SELECT 5"),
        ("</think>The answer is SELECT 6", None),
        ("</think>```sql\n   \n```", None),
    ],
)
def test_the_query_is_the_last_block_after_the_reasoning(
    reply: str, expected: str | None
) -> None:
    """Drafts inside the reasoning are ignored; the answer's last block counts."""
    assert sql_questions.extract_sql(reply) == expected


def test_a_reply_cut_off_while_reasoning_has_no_answer() -> None:
    """A query drafted in unfinished reasoning is not taken as the answer."""
    task = sql_questions.pass_rate_tasks()[0]
    reply = f"let me try ```sql\n{task.gold_sql}\n``` hmm"
    graded = sql_questions.grade_reply(task, reply, cut_off=True)
    assert not graded.correct and graded.reason == sql_questions.REASON_CUT_OFF
    finished = sql_questions.grade_reply(task, reply + "</think>", cut_off=True)
    assert finished.reason == sql_questions.REASON_NO_SQL


def test_wrong_failing_and_empty_queries_are_graded_wrong() -> None:
    """The grader's reason, SQLite's error, or 'returned nothing'."""
    task = sql_questions.pass_rate_tasks()[0]  # How many flights depart from Brussels?
    wrong = sql_questions.grade_reply(task, "```sql\nSELECT 7\n```")
    assert not wrong.correct and wrong.answer == "7" and "expected '8'" in wrong.reason
    broken = sql_questions.grade_reply(task, "```sql\nSELECT * FROM nowhere\n```")
    assert not broken.correct and broken.reason.startswith("SQL error")
    empty = sql_questions.grade_reply(
        task, "```sql\nSELECT id FROM flights WHERE 1 = 0\n```"
    )
    assert empty.reason == sql_questions.REASON_NO_ROWS


def test_a_query_can_never_change_the_database(tmp_path: Path) -> None:
    """The database is opened read-only: writes fail and the file is unchanged."""
    path = tmp_path / "copy.db"
    path.write_bytes((CHAPTER / "data/dbs/clinic.db").read_bytes())
    before = path.read_bytes()
    for sql in ("DELETE FROM patients", "DROP TABLE doctors", "CREATE TABLE x (y)"):
        with pytest.raises(sql_questions.QueryError):
            sql_questions.run_query(path, sql)
    assert path.read_bytes() == before


def test_a_runaway_query_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A query that never ends is interrupted once its time is up."""
    monkeypatch.setattr(sql_questions, "QUERY_SECONDS", 0.2)
    endless = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) "
        "SELECT count(*) FROM n"
    )
    with pytest.raises(sql_questions.QueryError, match="interrupted"):
        sql_questions.run_query(CHAPTER / "data/dbs/clinic.db", endless)


def test_answers_are_formatted_as_chapter_1_formats_them() -> None:
    """Floats to 2 places, lists joined by commas, NULLs skipped."""
    assert sql_questions.format_answer([(404.2149,)], tasks.NUMBER) == "404.21"
    assert sql_questions.format_answer([(8,)], tasks.NUMBER) == "8"
    assert sql_questions.format_answer([("a",), (None,), ("b",)], tasks.LIST) == "a, b"
    assert sql_questions.format_answer([(None,)], tasks.TEXT) is None
    assert sql_questions.format_answer([], tasks.TEXT) is None


def test_the_grader_is_chapter_1s() -> None:
    """The copy forgives format, not content."""
    task = sql_questions.pass_rate_tasks()[0]
    assert grader.check_answer(task, "8.00").correct
    assert not grader.check_answer(task, "9").correct


def test_a_folder_name_with_hash_or_space_still_opens_the_database(
    tmp_path: Path,
) -> None:
    """The path is escaped, so '#' cannot cut it short and give an empty schema."""
    folder = tmp_path / "a#b %c?d"
    folder.mkdir()
    path = folder / "clinic.db"
    path.write_bytes((CHAPTER / "data/dbs/clinic.db").read_bytes())
    assert sql_questions.schema_text(path).count("CREATE TABLE") == 3
    assert sql_questions.run_query(path, "SELECT COUNT(*) FROM doctors")[0][0] > 0


def test_a_task_line_with_the_wrong_fields_stops(tmp_path: Path) -> None:
    """A line that is not a task record raises; it is never skipped."""
    (tmp_path / "bad.jsonl").write_text('{"task_id": "x"}\n')
    with pytest.raises(TypeError):
        tasks.load_tasks("bad.jsonl", tmp_path)
    (tmp_path / "broken.jsonl").write_text("{not json\n")
    with pytest.raises(ValueError):
        tasks.load_tasks("broken.jsonl", tmp_path)


def test_the_schema_is_read_without_writing(tmp_path: Path) -> None:
    """Reading the tables leaves the database file exactly as it was."""
    path = tmp_path / "copy.db"
    path.write_bytes((CHAPTER / "data/dbs/airline.db").read_bytes())
    before = path.read_bytes()
    text = sql_questions.schema_text(path)
    assert text.splitlines()[0].startswith("CREATE TABLE bookings")
    assert path.read_bytes() == before
    with pytest.raises(sqlite3.Error):
        sql_questions.schema_text(tmp_path / "missing.db")
