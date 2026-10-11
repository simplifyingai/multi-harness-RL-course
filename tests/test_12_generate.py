"""Chapter 12 without MLX: where the model lives, the sums, and the printing.

No test here loads a model, downloads anything or needs MLX. The model
folders are tiny fakes written here; the sums run on lists of numbers worked
out by hand. MLX's side (the sampling loop and the timings) is in
test_12_mlx.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
from chapters import use_chapter

CHAPTER = use_chapter("12-generate")

import lfm_files  # noqa: E402
import measures  # noqa: E402
import report  # noqa: E402

# -------------------------------------------------------------- lfm_files.py


def write_build(folder: Path) -> Path:
    """Write the files that make a folder count as a finished build.

    Args:
        folder: The build's folder.

    Returns:
        The folder.
    """
    folder.mkdir(parents=True)
    for name in lfm_files.BUILD_FILES:
        (folder / name).write_text("x")
    return folder


def write_liquid_export(home: Path) -> Path:
    """Write files of the right sizes where Liquid's export should be.

    Args:
        home: The course home.

    Returns:
        The export's folder.
    """
    folder = home / "models" / lfm_files.FOUR_BIT.folder
    folder.mkdir(parents=True)
    for item in lfm_files.FOUR_BIT.files:
        with (folder / item.name).open("wb") as handle:
            handle.truncate(item.size)  # a sparse file: the size, no real bytes
    return folder


def test_course_home_follows_the_environment(course_home: Path) -> None:
    """COURSE_HOME decides every folder; the chapter writes under its own."""
    assert lfm_files.course_home() == course_home
    assert lfm_files.chapter_home() == course_home / "12-generate"
    assert lfm_files.ours_folder() == course_home / "models" / lfm_files.OURS_FOLDER


def test_ours_comes_first(course_home: Path) -> None:
    """With chapter 11's build on disk, every script uses it."""
    write_build(course_home / "models" / lfm_files.OURS_FOLDER)
    write_liquid_export(course_home)
    folder, which = lfm_files.pick_model(fast=False)
    assert folder == lfm_files.ours_folder()
    assert "ours" in which


def test_liquids_export_is_the_fallback(course_home: Path) -> None:
    """Without our build, Liquid's export is used, and the reason is said."""
    export = write_liquid_export(course_home)
    folder, which = lfm_files.pick_model(fast=False)
    assert folder == export
    assert "ours is not built yet" in which


def test_fast_always_uses_liquids_export(course_home: Path) -> None:
    """--fast takes Liquid's export even when ours is built."""
    write_build(course_home / "models" / lfm_files.OURS_FOLDER)
    export = write_liquid_export(course_home)
    assert lfm_files.pick_model(fast=True) == (export, "Liquid's export (--fast)")


def test_no_model_names_both_commands(course_home: Path) -> None:
    """With nothing on disk, the message names both chapters' commands."""
    with pytest.raises(SystemExit) as stopped:
        lfm_files.pick_model(fast=False)
    message = str(stopped.value)
    assert "uv run convert.py" in message and "uv run download.py --fast" in message


def test_fast_without_the_export_says_what_is_missing(course_home: Path) -> None:
    """A half download is named file by file, two at most, the rest counted."""
    with pytest.raises(SystemExit) as stopped:
        lfm_files.pick_model(fast=True)
    message = str(stopped.value)
    assert "model.safetensors, model.safetensors.index.json and 8 more" in message


def test_generation_config_is_read_or_empty(tmp_path: Path) -> None:
    """The maker's settings come back as a dictionary; a broken file gives none."""
    assert lfm_files.read_generation_config(tmp_path) == {}
    (tmp_path / "generation_config.json").write_text("{not json")
    assert lfm_files.read_generation_config(tmp_path) == {}
    (tmp_path / "generation_config.json").write_text('["a list"]')
    assert lfm_files.read_generation_config(tmp_path) == {}
    (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": 50}))
    assert lfm_files.read_generation_config(tmp_path) == {"top_k": 50}


def test_paths_are_shown_from_home() -> None:
    """A path inside the home folder is written with ~."""
    assert lfm_files.short_path(Path.home() / "a" / "b") == "~/a/b"
    assert lfm_files.short_path(Path("/tmp/x")) == "/tmp/x"
    assert lfm_files.short_path(Path.home()) == "~"


# --------------------------------------------------------------- measures.py


def test_running_sums_add_up_one_by_one() -> None:
    """The running sum after each token is the reply's log-probability so far."""
    assert measures.running_sums([-1.0, -0.5, 0.0, -2.0]) == [-1.0, -1.5, -1.5, -3.5]


@pytest.mark.parametrize(
    ("values", "power"),
    [
        ([0.0, -1.0, -3.0], 0),
        ([-0.5, -1.5], 1),
        ([-0.125, 0.0, -0.625, -2.375], 3),
        ([-0.0625, -0.125], 4),
        ([-(2.0**-19) * 7, -0.25], 19),
        ([-0.1], None),
    ],
)
def test_the_coarsest_grid_that_holds_every_value(
    values: list[float], power: int | None
) -> None:
    """Multiples of 1/8 give 3; a value no power of two divides gives None."""
    assert measures.grid_power(values) == power


@pytest.mark.parametrize(
    ("total", "text"),
    [
        (0.0, "1 in 1"),
        (-math.log(999_960_000), "1 in 10^9.0"),  # never "1000.0 million"
        (-math.log(1_612), "1 in 1,612"),
        (-math.log(3_400_000), "1 in 3.4 million"),
        (-100 * math.log(10), "1 in 10^100.0"),
    ],
)
def test_one_in_reads_like_words(total: float, text: str) -> None:
    """Small odds are counted; huge ones become a power of ten."""
    assert measures.one_in(total) == text


def test_ending_sums_and_counts_nearly_certain_steps() -> None:
    """The total is the sum of the log-probabilities; low entropy counts as certain."""
    end = measures.ending([-1.0, -0.25, -0.75], [0.5, 0.05, 0.0])
    assert end == measures.Ending(3, -2.0, pytest.approx(0.55 / 3), 2)
    assert measures.ending([], []) == measures.Ending(0, 0.0, 0.0, 0)


def test_a_cut_raises_the_kept_probabilities_by_the_kept_share() -> None:
    """Dividing by the kept share adds -log(share) to every kept log-probability."""
    uncut = [-1.0, -0.5, -2.0]
    shares = [0.5, 1.0, 0.8]
    drawn = [
        value - math.log(share) for value, share in zip(uncut, shares, strict=True)
    ]
    drawn[2] = uncut[2] - 1e-7  # a rounding shortfall, as float32 can give
    result = measures.cut(drawn, uncut, [0.2, 0.3, 0.4], [0.4, 0.3, 0.5])
    assert (result.higher, result.equal, result.lower) == (1, 1, 1)
    assert result.largest_drop == pytest.approx(1e-7)
    assert result.kept_mean == pytest.approx((0.5 + 1.0 + math.exp(1e-7)) / 3)
    assert result.kept_lowest == pytest.approx(0.5)
    assert result.lower_entropy == 2
    assert result.entropy_cut == pytest.approx(0.3)
    assert result.total_uncut == pytest.approx(-3.5)


def test_sums_over_no_steps_stop_with_a_reason() -> None:
    """Every reply has a token; an empty list is a mistake, said plainly."""
    with pytest.raises(ValueError, match="at least one step"):
        measures.cut([], [], [], [])
    with pytest.raises(ValueError, match="at least one step"):
        measures.raw_side([], [])
    with pytest.raises(ValueError, match="at least one step"):
        measures.grid([], [], [])


def test_raw_and_processed_are_compared_step_by_step() -> None:
    """Differences and which side is higher are counted at each step."""
    result = measures.raw_side([-0.1, -2.0, 0.0], [-0.3, -1.0, 0.0])
    assert result == measures.RawSide(3, 2, 1, pytest.approx(-2.1), -1.3)


def test_the_bfloat16_grid_and_its_zeros() -> None:
    """Zeros in bfloat16 are listed with what float32 said at the same steps."""
    result = measures.grid(
        [-0.125, 0.0, 0.0, -2.25],
        [-0.117, -0.0004, -0.06, -2.26],
        [22.9, 30.1, 25.5, 19.1],
    )
    assert result.power == 3
    assert result.zeros == 2
    assert (result.zero_float32_low, result.zero_float32_high) == (-0.06, -0.0004)
    assert (result.distinct, result.distinct_float32) == (3, 4)
    assert (result.score_low, result.score_high) == (19.1, 30.1)
    assert result.total == -2.375


# ----------------------------------------------------------------- report.py


@dataclass(frozen=True)
class FakeDraw:
    """Stands in for sample_lab's Draw, which needs MLX to import.

    Its cuts() is a copy of Draw.cuts(); test_12_mlx.py tests the real one.
    """

    temperature: float = 1.0
    greedy: bool = False
    top_k: int = 0
    top_p: float = 1.0
    dtype: str = "fp32"

    def cuts(self) -> bool:
        """Say whether top-k or top-p cuts.

        Returns:
            True when either is set.
        """
        return not self.greedy and (self.top_k != 0 or self.top_p != 1.0)


@pytest.mark.parametrize(
    ("chunks", "text"),
    [
        ([499, 1], "499 + 1"),
        ([1_024, 975, 1], "1,024 + 975 + 1"),
        ([1_024] * 7 + [831, 1], "1,024 x 7 + 831 + 1"),
        ([2_048] * 3 + [1_855, 1], "2,048 x 3 + 1,855 + 1"),
    ],
)
def test_chunks_are_written_compactly(chunks: list[int], text: str) -> None:
    """Runs of equal chunks are written as size x count."""
    assert report.chunk_text(chunks) == text


def test_probabilities_tokens_and_grids_in_words() -> None:
    """Percentages, token columns and grid steps read plainly."""
    assert report.percent(0.8896) == "88.96%"
    assert report.percent(0.00001) == "<0.01%"
    assert report.token_text("The") == "'The'"
    assert report.token_text("a" * 40).endswith("...")
    assert len(report.token_text("a" * 40)) == report.TOKEN_WIDTH
    assert report.fraction(0) == "1"
    assert report.fraction(3) == "1/8"
    assert report.fraction(19) == "1/524,288"
    assert report.fraction(None) == "no grid"


def test_the_draw_is_described_in_words() -> None:
    """Greedy, the full distribution, top-k and top-p each say what they do."""
    assert report.describe_draw(FakeDraw(greedy=True), 0).startswith("greedy")
    assert "the full distribution" in report.describe_draw(FakeDraw(), 0)
    top_k = report.describe_draw(FakeDraw(top_k=50), 3)
    assert "top-k 50: only the 50 likeliest tokens, renormalized; seed 3" in top_k
    assert "holding 90% of the probability" in report.describe_draw(
        FakeDraw(top_p=0.9), 0
    )


def test_step_columns_follow_the_flags() -> None:
    """A cut shows the uncut value, --raw the raw one, bf16 the float32 one."""
    assert report.step_columns(FakeDraw(), show_raw=False) == []
    assert report.step_columns(FakeDraw(top_k=50, dtype="bf16"), show_raw=True) == [
        "uncut",
        "raw",
        "float32",
    ]


def test_suggested_settings_are_named_and_said_to_go_unused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The maker's temperature, top_k and repetition_penalty are listed."""
    report.print_suggested({"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1})
    out = " ".join(capsys.readouterr().out.split())  # undo the line wrapping
    assert "temperature 0.1, top_k 50, repetition_penalty 1.1" in out
    assert "uses none of them" in out
    report.print_suggested({})
    assert capsys.readouterr().out == ""


def test_sections_are_numbered_from_the_title(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each script's sections count 1, 2, 3 from its title line."""
    report.print_title("x.py", "a title")
    report.heading("first")
    report.heading("second")
    report.print_title("y.py", "again")
    report.heading("first again")
    out = capsys.readouterr().out
    assert "\n1. first\n" in out and "\n2. second\n" in out
    assert "\n1. first again\n" in out


@dataclass(frozen=True)
class FakeStep:
    """The fields of sample_lab's Step that the grid summary reads."""

    logprob: float
    float32: float
    score: float


@dataclass(frozen=True)
class FakeReply:
    """The part of sample_lab's Reply that the grid summary reads."""

    steps: list[FakeStep]


def test_the_grid_summary_reads_plainly(capsys: pytest.CaptureFixture[str]) -> None:
    """A float32 grid is named; values on no coarse grid are said to be so."""
    report.print_title("x.py", "t")
    on_grid = [FakeStep(-0.125, -(2.0**-19) * 3, 22.875), FakeStep(0.0, 0.0, 30.0)]
    report.print_grid(FakeReply(on_grid))
    out = " ".join(capsys.readouterr().out.split())
    assert "is a whole multiple of 1/8; they take 2 different values" in out
    assert "2 different values, multiples of 1/524,288" in out
    assert "1 of 2 are exactly 0 in bfloat16" in out
    off_grid = [FakeStep(-0.125, -0.1, 22.875)]
    report.print_grid(FakeReply(off_grid))
    out = " ".join(capsys.readouterr().out.split())
    assert "1 different values, on no grid of 1/16,777,216 or coarser" in out
    assert "grid grid" not in out
