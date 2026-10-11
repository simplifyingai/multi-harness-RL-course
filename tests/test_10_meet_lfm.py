"""Chapter 10: the model's files, the download, the license and the tally.

No test here uses the network or downloads the model. The download is tested
against a fake ``snapshot_download`` that writes a tiny fake repository; the
header reader and the tally against tiny safetensors files written here (an
LFM2-shaped model with a few numbers per matrix, once in bfloat16 and once
packed at 4 and 6 bits); the arithmetic against the real config's numbers,
typed in. The few tests that read the real files (the license, the full
tally) read them where ``download.py`` put them, read only, and are skipped,
with the reason, when they are not there.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from chapters import use_chapter
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RevisionNotFoundError,
)

CHAPTER = use_chapter("10-meet-lfm")

import download  # noqa: E402
import inspect_model  # noqa: E402
import lfm_files  # noqa: E402
import report  # noqa: E402

# Where a real run keeps the models. Worked out when this file is imported,
# before conftest points COURSE_HOME at a temporary folder; only ever read.
REAL_BF16 = lfm_files.model_folder(lfm_files.BF16)
REAL_FOUR_BIT = lfm_files.model_folder(lfm_files.FOUR_BIT)
HAVE_BF16 = not lfm_files.missing_files(lfm_files.BF16, REAL_BF16)
HAVE_FOUR_BIT = not lfm_files.missing_files(lfm_files.FOUR_BIT, REAL_FOUR_BIT)
needs_bf16 = pytest.mark.skipif(
    not HAVE_BF16,
    reason="LFM2.5-2.6B is not downloaded: run `uv run download.py` in 10-meet-lfm",
)
needs_four_bit = pytest.mark.skipif(
    not HAVE_FOUR_BIT,
    reason="the 4-bit export is not downloaded: run `uv run download.py --fast`",
)

# The real model's numbers, as config.json states them.
REAL_CONFIG: dict[str, Any] = {
    "hidden_size": 2048,
    "intermediate_size": 10752,
    "vocab_size": 128000,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "conv_L_cache": 3,
    "layer_types": (["conv"] * 2 + ["full_attention"] + ["conv"] * 2)
    + ["full_attention"]
    + (["conv"] * 3 + ["full_attention"]) * 4
    + (["conv"] * 2 + ["full_attention"]) * 2
    + ["conv"] * 2,
}
REAL_PARTS = {
    inspect_model.PART_FEED_FORWARD: 1_981_808_640,
    inspect_model.PART_CONV: 369_233_920,
    inspect_model.PART_ATTENTION: 83_887_104,
    inspect_model.PART_EMBEDDING: 262_144_000,
    inspect_model.PART_NORMS: 124_928,
}
REAL_TOTAL = 2_697_198_592
REAL_ATTENTION_LAYERS = [2, 5, 9, 13, 17, 21, 24, 27]

LICENSE_TEXT = "\n\n".join(
    [
        "LFM Open License v1.0",
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
        "1. Definitions.",
        '"Licensor" shall mean Liquid AI, Inc.',
        '"Commercial Use" shall mean any use of the Work for commercial advantage.',
        '"Threshold" shall mean annual revenue of 10 million United States dollars '
        "($10,000,000) or more.",
        "5. Commercial Use Limitation.",
        "(a) The rights granted under this License for Commercial Use are "
        "conditioned upon You not exceeding the Threshold.",
        "(b) Any Commercial Use by a Legal Entity that exceeds the Threshold is "
        "not licensed.",
        "6. Submission of Contributions.",
        "(a) Not part of the limit.",
    ]
)

# A tiny LFM2-shaped model: the same tensor names and kinds as the real one,
# with a handful of numbers per matrix. Packed widths must be whole words at
# 4 and 6 bits and whole groups of 8, so the widths are multiples of 16.
TINY_CONFIG: dict[str, Any] = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "vocab_size": 24,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "conv_L_cache": 3,
    "layer_types": ["conv", "full_attention", "conv"],
    "max_position_embeddings": 128,
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    "tie_word_embeddings": True,
    "dtype": "bfloat16",
}
TINY_GROUP = 8
TINY_EMBED_BITS = 6
TINY_BITS = 4


@pytest.fixture(autouse=True)
def xet_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point hf_xet's folder at a temporary one, restored after every test.

    download.py sets HF_XET_CACHE for the process when the reader has not;
    this keeps that from leaking from one test into the next.

    Args:
        tmp_path: pytest's per-test folder.
        monkeypatch: pytest's environment patcher.

    Returns:
        The temporary folder.
    """
    folder = tmp_path / "xet"
    monkeypatch.setenv(download.XET_CACHE_ENV, str(folder))
    return folder


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


# ------------------------------------------------------------- lfm_files.py


def test_everything_lives_under_course_home(course_home: Path) -> None:
    """The models go to a shared folder, the outputs to the chapter's own."""
    assert lfm_files.course_home() == course_home
    assert lfm_files.models_home() == course_home / "models"
    assert lfm_files.chapter_home() == course_home / "10-meet-lfm"
    folder = lfm_files.model_folder(lfm_files.BF16)
    assert folder == course_home / "models" / "LFM2.5-2.6B"
    assert lfm_files.model_folder(lfm_files.FOUR_BIT).name == "LFM2.5-2.6B-MLX-4bit"


@pytest.mark.parametrize("repo", [lfm_files.BF16, lfm_files.FOUR_BIT])
def test_both_repositories_are_pinned_file_by_file(repo: lfm_files.ModelRepo) -> None:
    """One commit, and every file with a size and a SHA-256 fingerprint."""
    assert len(repo.revision) == 40
    int(repo.revision, 16)
    names = [item.name for item in repo.files]
    assert len(names) == len(set(names))
    for item in repo.files:
        assert item.size > 0
        assert len(item.sha256) == 64
        int(item.sha256, 16)
    assert {"config.json", "LICENSE", "tokenizer.json"} <= set(names)


def test_the_weights_add_up_to_the_hubs_sizes() -> None:
    """The bfloat16 weights are 5.39 GB in two files; the 4-bit export 1.58 GB."""
    bf16 = [item for item in lfm_files.BF16.files if item.name.endswith("safetensors")]
    assert sum(item.size for item in bf16) == 5_394_427_456
    four_bit = [
        item for item in lfm_files.FOUR_BIT.files if item.name.endswith("safetensors")
    ]
    assert [item.size for item in four_bit] == [1_583_152_892]
    assert lfm_files.gigabytes(lfm_files.BF16.total_bytes()) == "5.41 GB"
    assert lfm_files.gigabytes(lfm_files.FOUR_BIT.total_bytes()) == "1.60 GB"


def tiny_repo(contents: dict[str, bytes]) -> lfm_files.ModelRepo:
    """Make a fake repository whose files are the given bytes.

    Args:
        contents: Each file's name and bytes.

    Returns:
        A repository pinned to those sizes and fingerprints.
    """
    files = tuple(
        lfm_files.HubFile(name, len(data), hashlib.sha256(data).hexdigest(), "a file")
        for name, data in contents.items()
    )
    return lfm_files.ModelRepo(
        "someone/tiny", "0" * 40, "2026-01-01", "tiny", "bfloat16", files
    )


TINY_FILES = {
    "config.json": b'{"hidden_size": 16}',
    "LICENSE": LICENSE_TEXT.encode(),
    "model.safetensors": b"\x00" * 64,
}


def write_files(folder: Path, contents: dict[str, bytes]) -> None:
    """Write files into a folder.

    Args:
        folder: Where.
        contents: Each file's name and bytes.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (folder / name).write_bytes(data)


def test_missing_files_checks_presence_and_size(tmp_path: Path) -> None:
    """A file that is absent or the wrong size still needs downloading."""
    repo = tiny_repo(TINY_FILES)
    assert len(lfm_files.missing_files(repo, tmp_path)) == 3
    write_files(tmp_path, TINY_FILES)
    assert lfm_files.missing_files(repo, tmp_path) == []
    (tmp_path / "config.json").write_bytes(b"{}")
    assert [item.name for item in lfm_files.missing_files(repo, tmp_path)] == [
        "config.json"
    ]


def test_a_missing_model_names_the_command_that_downloads_it() -> None:
    """The scripts stop with the download command, not a traceback."""
    with pytest.raises(SystemExit) as stopped:
        lfm_files.require_model(fast=False)
    message = str(stopped.value)
    assert "uv run download.py" in message
    assert "--fast" in message and "5.41 GB" in message
    assert "model-00001-of-00002.safetensors" in message
    with pytest.raises(SystemExit) as stopped:
        lfm_files.require_model(fast=True)
    assert "`uv run download.py --fast` first (1.60 GB)" in str(stopped.value)


def test_a_downloaded_model_is_found(
    monkeypatch: pytest.MonkeyPatch, course_home: Path
) -> None:
    """Once every file is there at full size, the folder is returned."""
    repo = tiny_repo(TINY_FILES)
    monkeypatch.setattr(lfm_files, "BF16", repo)
    write_files(lfm_files.model_folder(repo), TINY_FILES)
    found, folder = lfm_files.require_model(fast=False)
    assert found is repo
    assert folder == course_home / "models" / "tiny"


def test_short_path_writes_home_as_a_tilde() -> None:
    """Paths are printed the way a reader would type them."""
    assert lfm_files.short_path(Path.home()) == "~"
    assert lfm_files.short_path(Path.home() / "a" / "b") == "~/a/b"
    assert lfm_files.short_path(Path("/tmp/x")) == "/tmp/x"


# --------------------------------------------------------------- download.py


def fake_hub(contents: dict[str, bytes], calls: list[dict[str, Any]]) -> Any:
    """Make a stand-in for snapshot_download that writes the given files.

    Args:
        contents: Each file's name and bytes.
        calls: Receives the arguments of every call.

    Returns:
        The stand-in.
    """

    def snapshot_download(repo_id: str, revision: str, local_dir: Path) -> str:
        calls.append({"repo_id": repo_id, "revision": revision, "local_dir": local_dir})
        write_files(Path(local_dir), contents)
        return str(local_dir)

    return snapshot_download


def failing_hub(error: Exception) -> Any:
    """Make a stand-in for snapshot_download that raises.

    Args:
        error: What it raises.

    Returns:
        The stand-in.
    """

    def snapshot_download(repo_id: str, revision: str, local_dir: Path) -> str:
        raise error

    return snapshot_download


def test_ensure_model_downloads_then_checks_every_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned repository and revision are asked for, then every file checked."""
    repo = tiny_repo(TINY_FILES)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(download, "snapshot_download", fake_hub(TINY_FILES, calls))
    monkeypatch.delenv(download.XET_CACHE_ENV, raising=False)
    lines: list[str] = []
    folder = download.ensure_model(repo, lines.append)
    # hf_xet's logs and chunks go into COURSE_HOME too, not ~/.cache/huggingface.
    xet = lfm_files.course_home() / "huggingface-xet"
    assert download.os.environ[download.XET_CACHE_ENV] == str(xet)
    assert calls == [
        {"repo_id": "someone/tiny", "revision": "0" * 40, "local_dir": folder}
    ]
    assert lines[2].startswith("downloading 3 files")
    assert lines[-1] == "all 3 files checked against their SHA-256 fingerprints"


def test_ensure_model_skips_the_hub_when_everything_is_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run needs no network."""
    repo = tiny_repo(TINY_FILES)
    write_files(lfm_files.model_folder(repo), TINY_FILES)
    error = AssertionError("the Hub must not be called")
    monkeypatch.setattr(download, "snapshot_download", failing_hub(error))
    lines: list[str] = []
    download.ensure_model(repo, lines.append)
    assert lines[-1].startswith("already here: all 3 files at full size")


def test_a_wrong_file_is_deleted_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bytes of the right size but the wrong content are never kept."""
    repo = tiny_repo(TINY_FILES)
    wrong = {**TINY_FILES, "model.safetensors": b"\x01" * 64}
    monkeypatch.setattr(download, "snapshot_download", fake_hub(wrong, []))
    with pytest.raises(download.DownloadError, match="not the expected file"):
        download.ensure_model(repo, lambda line: None)
    assert not (lfm_files.model_folder(repo) / "model.safetensors").exists()


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("[Errno 61] Connection refused"),
        httpx.ConnectTimeout("timed out"),
        OfflineModeIsEnabled("offline mode is enabled"),
        LocalEntryNotFoundError("cannot find the requested files"),
    ],
)
def test_an_unreachable_hub_gets_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """No network, a timeout or offline mode all say: check the connection."""
    monkeypatch.setattr(download, "snapshot_download", failing_hub(error))
    monkeypatch.setenv(download.XET_CACHE_ENV, "/somewhere/the/reader/chose")
    with pytest.raises(download.DownloadError) as failed:
        download.fetch(lfm_files.BF16, Path("unused"))
    message = str(failed.value)
    assert message.startswith(
        "could not reach the Hugging Face Hub to download LiquidAI/LFM2.5-2.6B"
    )
    assert f"reason: {error}" in message
    assert "fetch 5.41 GB" in message and "Check your connection" in message
    assert download.os.environ[download.XET_CACHE_ENV] == "/somewhere/the/reader/chose"


def test_an_http_error_says_the_hub_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused revision is not a connection problem, and says so."""
    response = httpx.Response(404, request=httpx.Request("GET", "https://hub/x"))
    error = RevisionNotFoundError(
        "404 Client Error.\n\nRevision Not Found for url", response=response
    )
    monkeypatch.setattr(download, "snapshot_download", failing_hub(error))
    with pytest.raises(download.DownloadError) as failed:
        download.fetch(lfm_files.BF16, Path("unused"))
    message = str(failed.value)
    assert "refused to send LiquidAI/LFM2.5-2.6B at revision 654f9463ce32" in message
    assert "reason: 404 Client Error." in message
    assert "The connection works." in message


def test_a_full_disk_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disk error says where the files were going."""
    error = OSError(28, "No space left on device")
    monkeypatch.setattr(download, "snapshot_download", failing_hub(error))
    with pytest.raises(download.DownloadError, match="could not write the files"):
        download.fetch(lfm_files.BF16, Path("unused"))


def test_main_stops_with_the_message_and_prints_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The script's own run: the message, and no file list after it."""
    error = httpx.ConnectError("[Errno 8] nodename nor servname provided")
    monkeypatch.setattr(download, "snapshot_download", failing_hub(error))
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as stopped:
        saved = sys.argv
        sys.argv = ["download.py"]
        try:
            download.main()
        finally:
            sys.argv = saved
    assert "could not reach the Hugging Face Hub" in str(stopped.value)
    assert "the repository's files" not in out.getvalue()


def test_main_lists_the_files_and_quotes_the_license(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first run: the source, the download, the file list and the terms."""
    repo = tiny_repo(TINY_FILES)
    monkeypatch.setattr(download, "pick_repo", lambda fast: repo)
    monkeypatch.setattr(download, "snapshot_download", fake_hub(TINY_FILES, []))
    printed = run_main(download)
    assert "repository  someone/tiny, revision 000000000000 (2026-01-01)" in printed
    assert "the repository's files" in printed
    assert "3 files, 0.00 GB in all" in printed
    size = len(LICENSE_TEXT.encode())
    assert f"the license, quoted from LICENSE ({size:,} bytes)" in printed
    assert '  "Threshold" shall mean annual revenue of 10 million' in printed
    assert "(b) Any Commercial Use" in printed


def test_main_refuses_a_license_that_is_not_this_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LICENSE of another kind stops the script with a message, not a trace."""
    other = {**TINY_FILES, "LICENSE": b"MIT License\n\nPermission is granted"}
    monkeypatch.setattr(download, "pick_repo", lambda fast: tiny_repo(other))
    monkeypatch.setattr(download, "snapshot_download", fake_hub(other, []))
    with pytest.raises(SystemExit, match="is not the expected license"):
        run_main(download)


def test_license_terms_pick_out_the_commercial_limit() -> None:
    """The title, the two definitions and section 5, and nothing from section 6."""
    terms = download.license_terms(LICENSE_TEXT)
    assert terms.title == "LFM Open License v1.0"
    assert [line.split()[0] for line in terms.definitions] == [
        '"Commercial',
        '"Threshold"',
    ]
    assert terms.limit[0] == "5. Commercial Use Limitation."
    assert [line[:3] for line in terms.limit[1:]] == ["(a)", "(b)"]
    assert "Not part of the limit." not in " ".join(terms.limit)


def test_license_terms_refuse_another_license() -> None:
    """A license without the limit is not the one this chapter describes."""
    with pytest.raises(ValueError, match="not the LFM Open License"):
        download.license_terms("MIT License\n\nPermission is hereby granted")


@needs_bf16
def test_the_real_license_limits_commercial_use_by_revenue() -> None:
    """The downloaded LICENSE: LFM Open License v1.0, $10 million a year."""
    text = (REAL_BF16 / "LICENSE").read_text(encoding="utf-8")
    terms = download.license_terms(text)
    assert terms.title == "LFM Open License v1.0"
    threshold = [line for line in terms.definitions if line.startswith('"Threshold"')]
    assert "($10,000,000) or more" in threshold[0]
    assert len(terms.limit) == 4  # the heading, (a), (b) and (c)
    assert "not exceeding the Threshold" in terms.limit[1]


# ---------------------------------------------------------- inspect_model.py


def tiny_tensors() -> dict[str, tuple[int, ...]]:
    """List the tiny model's tensors and their shapes, named as LFM2 names them.

    Returns:
        Each tensor's name and real shape.
    """
    hidden = TINY_CONFIG["hidden_size"]
    width = TINY_CONFIG["intermediate_size"]
    head_dim = hidden // TINY_CONFIG["num_attention_heads"]
    kv_width = TINY_CONFIG["num_key_value_heads"] * head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (TINY_CONFIG["vocab_size"], hidden),
        "model.embedding_norm.weight": (hidden,),
    }
    for layer, kind in enumerate(TINY_CONFIG["layer_types"]):
        prefix = f"model.layers.{layer}."
        shapes[prefix + "operator_norm.weight"] = (hidden,)
        shapes[prefix + "ffn_norm.weight"] = (hidden,)
        shapes[prefix + "feed_forward.w1.weight"] = (width, hidden)
        shapes[prefix + "feed_forward.w3.weight"] = (width, hidden)
        shapes[prefix + "feed_forward.w2.weight"] = (hidden, width)
        if kind == "conv":
            shapes[prefix + "conv.in_proj.weight"] = (3 * hidden, hidden)
            shapes[prefix + "conv.out_proj.weight"] = (hidden, hidden)
            kernel = TINY_CONFIG["conv_L_cache"]
            shapes[prefix + "conv.conv.weight"] = (hidden, 1, kernel)
        else:
            shapes[prefix + "self_attn.q_proj.weight"] = (hidden, hidden)
            shapes[prefix + "self_attn.k_proj.weight"] = (kv_width, hidden)
            shapes[prefix + "self_attn.v_proj.weight"] = (kv_width, hidden)
            shapes[prefix + "self_attn.out_proj.weight"] = (hidden, hidden)
            shapes[prefix + "self_attn.q_layernorm.weight"] = (head_dim,)
            shapes[prefix + "self_attn.k_layernorm.weight"] = (head_dim,)
    return shapes


Stored = dict[str, tuple[str, tuple[int, ...]]]


def write_safetensors(path: Path, tensors: Stored) -> None:
    """Write a safetensors file: the 8-byte length, the JSON header, zero bytes.

    Args:
        path: The file.
        tensors: Each tensor's name, number type and stored shape.
    """
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for size in shape:
            count *= size
        end = offset + count * inspect_model.BYTES_PER_DTYPE[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * offset)


def write_tiny_bf16(folder: Path) -> None:
    """Write the tiny model in bfloat16, split over two files like the real one.

    Args:
        folder: Where.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(TINY_CONFIG))
    items = [(name, ("BF16", shape)) for name, shape in tiny_tensors().items()]
    half = len(items) // 2
    write_safetensors(folder / "model-00001-of-00002.safetensors", dict(items[:half]))
    write_safetensors(folder / "model-00002-of-00002.safetensors", dict(items[half:]))


def write_tiny_packed(
    folder: Path, embed_bits_in_config: int = TINY_EMBED_BITS
) -> None:
    """Write the tiny model packed as MLX's 4-bit export packs the real one.

    Every matrix of the model (not the norms, not the convolution's kernel) is
    stored as U32 words, beside a scale and an offset per group; the
    embedding table at 6 bits, the rest at 4.

    Args:
        folder: Where.
        embed_bits_in_config: The embedding's bits as config.json states them
            (the file always stores 6).
    """
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        **TINY_CONFIG,
        "quantization": {
            "group_size": TINY_GROUP,
            "bits": TINY_BITS,
            "mode": "affine",
            "model.embed_tokens": {
                "group_size": TINY_GROUP,
                "bits": embed_bits_in_config,
                "mode": "affine",
            },
        },
    }
    (folder / "config.json").write_text(json.dumps(config))
    tensors: Stored = {}
    for name, shape in tiny_tensors().items():
        is_matrix = len(shape) == 2 and not name.endswith("conv.conv.weight")
        if not is_matrix:
            tensors[name] = ("BF16", shape)
            continue
        bits = TINY_EMBED_BITS if "embed_tokens" in name else TINY_BITS
        rows, columns = shape
        module = name.removesuffix(".weight")
        tensors[name] = ("U32", (rows, columns * bits // 32))
        tensors[module + ".scales"] = ("BF16", (rows, columns // TINY_GROUP))
        tensors[module + ".biases"] = ("BF16", (rows, columns // TINY_GROUP))
    write_safetensors(folder / "model.safetensors", tensors)


def test_read_header_reads_names_types_shapes_and_sizes(tmp_path: Path) -> None:
    """Only the header is read: names, number types, shapes, byte counts."""
    path = tmp_path / "x.safetensors"
    write_safetensors(path, {"a.weight": ("BF16", (3, 4)), "b": ("U32", (2,))})
    header = inspect_model.read_header(path)
    assert [(t.name, t.dtype, t.shape, t.nbytes) for t in header.tensors] == [
        ("a.weight", "BF16", (3, 4), 24),
        ("b", "U32", (2,), 8),
    ]
    assert header.size == path.stat().st_size
    assert header.header_bytes == header.size - 32


# A header that gives four bfloat16 numbers (8 bytes) only 6 bytes.
SHORT_TENSOR = b'{"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 6]}}'
# A header entry with no shape.
NO_SHAPE = b'{"w": {"dtype": "BF16", "data_offsets": [0, 0]}}'


@pytest.mark.parametrize(
    ("content", "complaint"),
    [
        (b"\x01\x02", "too short"),
        (struct.pack("<Q", 10**6) + b"{}", "says its header is 1,000,000 bytes"),
        (struct.pack("<Q", 4) + b"nope", "not JSON"),
        (
            struct.pack("<Q", len(SHORT_TENSOR)) + SHORT_TENSOR + b"\x00" * 6,
            "which is 8 bytes, but takes 6",
        ),
        (
            struct.pack("<Q", len(NO_SHAPE)) + NO_SHAPE,
            "a tensor entry is malformed",
        ),
    ],
)
def test_read_header_rejects_what_is_not_safetensors(
    tmp_path: Path, content: bytes, complaint: str
) -> None:
    """A short file, an impossible length or a broken header is refused."""
    path = tmp_path / "bad.safetensors"
    path.write_bytes(content)
    with pytest.raises(inspect_model.HeaderError, match=complaint):
        inspect_model.read_header(path)


def test_the_tally_matches_the_config_arithmetic(tmp_path: Path) -> None:
    """Counted from the headers, each part equals its count from config.json."""
    write_tiny_bf16(tmp_path)
    found = inspect_model.inventory(tmp_path)
    counted = {part.part: part.parameters for part in found.parts}
    expected = {e.part: e.parameters for e in inspect_model.expected_parts(TINY_CONFIG)}
    assert counted == expected
    assert found.total_parameters() == sum(expected.values())
    assert found.layers_with(inspect_model.PART_ATTENTION) == [1]
    assert found.layers_with(inspect_model.PART_CONV) == [0, 2]
    assert found.quantization is None
    assert found.tensor_bytes() == 2 * found.total_parameters()


def test_the_packed_column_rule_recovers_every_parameter(tmp_path: Path) -> None:
    """Packed at 4 and 6 bits, the count is exactly the bfloat16 count."""
    write_tiny_bf16(tmp_path / "bf16")
    write_tiny_packed(tmp_path / "packed")
    plain = inspect_model.inventory(tmp_path / "bf16")
    packed = inspect_model.inventory(tmp_path / "packed")
    assert [(p.part, p.parameters) for p in packed.parts] == [
        (p.part, p.parameters) for p in plain.parts
    ]
    embed = next(
        item
        for item in packed.counted
        if item.tensor.name == "model.embed_tokens.weight"
    )
    assert embed.tensor.shape == (24, 3)  # 16 columns at 6 bits: 3 words
    assert embed.shape == (24, 16) and embed.bits == 6
    # Counting stored values instead undercounts the packed matrices.
    assert packed.stored_values() != packed.total_parameters()


def test_wrong_bits_in_the_config_are_caught(tmp_path: Path) -> None:
    """If config.json's bits disagree with the stored groups, it says so."""
    write_tiny_packed(tmp_path, embed_bits_in_config=4)
    with pytest.raises(inspect_model.HeaderError, match="groups of 8"):
        inspect_model.inventory(tmp_path)


def test_unpacked_columns_is_packed_times_32_over_bits() -> None:
    """The fast tally's rule, on the real model's packed widths."""
    assert inspect_model.unpacked_columns(256, 4) == 2048
    assert inspect_model.unpacked_columns(1344, 4) == 10752
    assert inspect_model.unpacked_columns(384, 6) == 2048


def test_expected_parts_give_the_real_models_numbers() -> None:
    """config.json alone says 2,697,198,592 parameters, part by part."""
    expected = inspect_model.expected_parts(REAL_CONFIG)
    assert {entry.part: entry.parameters for entry in expected} == REAL_PARTS
    assert sum(entry.parameters for entry in expected) == REAL_TOTAL
    assert REAL_CONFIG["layer_types"].count("conv") == 22
    attention = [
        i for i, kind in enumerate(REAL_CONFIG["layer_types"]) if kind != "conv"
    ]
    assert attention == REAL_ATTENTION_LAYERS


def test_part_of_sorts_tensors_by_name() -> None:
    """Every tensor name lands in one part; an unknown one is refused."""
    part = inspect_model.part_of
    assert part("model.embed_tokens.weight") == inspect_model.PART_EMBEDDING
    assert part("model.embed_tokens.scales") == inspect_model.PART_EMBEDDING
    assert part("model.layers.3.feed_forward.w2.weight") == "feed-forward"
    assert part("model.layers.2.self_attn.q_layernorm.weight") == "attention"
    assert part("model.layers.0.conv.conv.weight") == "convolution"
    assert part("model.layers.0.operator_norm.weight") == "norms"
    assert part("lm_head.weight") == inspect_model.PART_HEAD
    with pytest.raises(ValueError, match="does not know"):
        part("model.vision_tower.weight")


@pytest.mark.parametrize(
    ("card", "stated"),
    [
        ("- **Total parameters**: 2.69B\n- **Number of layers**: 30", "2.69B"),
        ("| Model | x |\n| Parameters | 2.6B |\n| Precision | 4-bit |", "2.6B"),
        ("# A model with no count", None),
    ],
)
def test_card_parameters_reads_what_the_card_claims(
    card: str, stated: str | None
) -> None:
    """Both card styles are read; a card that says nothing gives None."""
    assert inspect_model.card_parameters(card) == stated


def test_inspect_main_prints_the_tally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The script's own run on the tiny model: parts, agreement, the moral."""
    write_tiny_bf16(tmp_path)
    (tmp_path / "README.md").write_text("- **Total parameters**: 0.01B")
    monkeypatch.setattr(
        inspect_model, "require_model", lambda fast: (lfm_files.BF16, tmp_path)
    )
    printed = run_main(inspect_model)
    assert "no weight loaded" in printed
    assert "every part agrees with the files" in printed
    assert "the name says 2.6B, the model card says 0.01B" in printed
    assert "model names round: count it yourself" in printed
    assert "attention tensors sit in layers 1" in printed
    assert "no output head is stored" in printed


def test_inspect_main_explains_the_packing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With --fast the packed rows show their real width, and the naive count."""
    write_tiny_packed(tmp_path)
    (tmp_path / "README.md").write_text("| Parameters | 2.6B |")
    monkeypatch.setattr(
        inspect_model, "require_model", lambda fast: (lfm_files.FOUR_BIT, tmp_path)
    )
    printed = run_main(inspect_model, "--fast")
    assert "packed: 3 words x 32 / 6 bits = 16 columns" in printed
    assert "counting stored values instead would say" in printed
    assert "every part agrees with the files" in printed


@needs_bf16
def test_the_real_bfloat16_files_hold_2_70_billion_numbers() -> None:
    """The spec's numbers, measured on the downloaded headers (read only)."""
    found = inspect_model.inventory(REAL_BF16)
    assert {p.part: p.parameters for p in found.parts} == REAL_PARTS
    assert found.total_parameters() == REAL_TOTAL
    assert found.tensor_bytes() == 2 * REAL_TOTAL  # 5,394,397,184 bytes
    assert found.file_bytes() - found.tensor_bytes() == 30_272  # the headers
    assert found.layers_with(inspect_model.PART_ATTENTION) == REAL_ATTENTION_LAYERS
    card = (REAL_BF16 / "README.md").read_text(encoding="utf-8")
    assert inspect_model.card_parameters(card) == "2.69B"


@needs_four_bit
def test_the_real_4_bit_export_holds_the_same_numbers() -> None:
    """Unpacked with the rule, the 4-bit file counts the same 2,697,198,592."""
    found = inspect_model.inventory(REAL_FOUR_BIT)
    assert {p.part: p.parameters for p in found.parts} == REAL_PARTS
    assert found.quantization == inspect_model.Quantization(
        4, 64, {"model.embed_tokens": 6}
    )
    assert found.stored_values() == 438_041_600


# ----------------------------------------------------------------- report.py


def test_wrapped_indents_continuation_lines() -> None:
    """A long quoted line wraps, its later lines indented a little more."""
    text = report.wrapped("word " * 40)
    lines = text.splitlines()
    assert len(lines) > 1
    assert all(len(line) <= report.WRAP_WIDTH for line in lines)
    assert lines[0].startswith("  word") and lines[1].startswith("    word")


def test_shape_text_reads_like_speech() -> None:
    """Shapes print the way people say them."""
    assert report.shape_text((10752, 2048)) == "10752 x 2048"
    assert report.shape_text((64,)) == "64"
