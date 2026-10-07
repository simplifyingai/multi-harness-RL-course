# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

The runnable code for the **Multi-harness RL** course: one self-contained
folder per chapter (`NN-slug/`), each with a `README.md` and scripts that show
real input and output. The course follows Hugging Face and Liquid AI's article
"The ultimate guide to multi-harness RL" and ends with training LFM2.5-2.6B
inside real agent harnesses on an Apple-silicon laptop.

The course plan, its concept list and the owner's feedback ledger live outside
this repo (`~/sai_work/multi-harness-rl/course/`). The videos are made in the
video tool and the written lessons live on simplifyingai.com.

## Rules for every chapter folder

- **Self-contained.** Every script carries a PEP 723 header, so
  `uv run <script>` works from the folder alone. Prefer the standard library;
  declare anything else inline.
- **Shared code stays in the folder.** A chapter may copy a module from an
  earlier chapter, never import across folders.
- **Nothing is written into the repo.** Outputs and work folders go under
  `COURSE_HOME` (default `~/.cache/mhrl-course`), in a subfolder named after
  the chapter.
- **Real output only.** Every output quoted in a README or lesson comes from a
  real run of the committed code.
- **Data the chapter needs is bundled** under `NN-slug/data/` (small files
  only). Transcripts are `.txt`/`.jsonl`, since `*.log` is ignored, and are
  scrubbed of personal paths.
- **Tests** live in `tests/test_NN_*.py`; each file imports its chapter with
  `tests/chapters.py::use_chapter` (chapters may reuse module names), and
  `tests/conftest.py` points `COURSE_HOME` at a temporary folder.

## Layout

```text
NN-slug/            one chapter: README.md, scripts, data/
tests/              pytest, one file per chapter
pyproject.toml      dev tools only (ruff, pytest, pre-commit); no runtime deps
```

## Quality gates

```bash
uv sync
uv run ruff format --check . && uv run ruff check .
uv run pytest
markdownlint "**/*.md"
```

## Coding standards (must follow)

When writing or modifying Python code, read and follow
`.claude/coding_standards.md` where it exists. Key rules:

- run `ruff format . && ruff check .` before calling code complete;
- never use a bare `except:`; catch specific exceptions;
- type hints on every parameter and return type;
- docstrings with Args, Returns and Raises sections;
- no magic strings or numbers: use named constants;
- naming: PascalCase for classes, snake_case for functions and variables,
  UPPER_SNAKE_CASE for constants;
- the code is read by learners on screen: plain names, short functions, a
  comment wherever a newcomer would ask "why".

After writing Python code, use the `sai-py-code-reviewer` agent to check it.

## Git commit format

```text
<type>(<scope>): <subject>

Types: feat, fix, docs, test, refactor, chore
```

NEVER add `Co-Authored-By` lines to commit messages.
NEVER add Claude/AI references in commit messages or PR descriptions
(no "Generated with Claude Code", no AI attribution lines).
