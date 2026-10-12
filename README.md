# Multi-harness RL, on a laptop

A course that starts from zero and ends with you training a small open model
inside the agent programs people actually use (mini-swe-agent, Claude Code and
Pi) on an Apple-silicon laptop. It follows Hugging Face and Liquid AI's article
[The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl).

Each chapter is one short video, one written lesson and one folder of code in
this repository. Every folder runs on its own and shows real input and output.
The written lessons and the videos are published on simplifyingai.com.

## Start

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/01-the-job
uv run check_machine.py
```

You need [uv](https://docs.astral.sh/uv/). Chapters 1 to 9 run on any Mac;
from chapter 10 the model runs locally and needs an Apple-silicon Mac (16 GB of
memory recommended). The course keeps downloads and outputs in one folder,
`COURSE_HOME` (default `~/.cache/mhrl-course`).

## Three ways through

- **RL first:** chapters 1 to 29 in order.
- **Claude Code first:** 1, 16, 9, 17, then 2-15 and 18-29.
- **RL veteran:** 1, 5, 7, then 9-29. Chapter 9 opens with a one-page recap of
  what you skipped.

## Chapters

About 258 minutes of video in 29 chapters.

### Part 1. The job, and RL from zero on a toy analyst

| # | chapter | minutes | status |
| ---: | --- | ---: | --- |
| 1 | [The job, the grader and the puzzle](01-the-job/) | 8 | ready |
| 2 | [Plain RL: learning which query to trust](02-plain-rl/) | 8 | ready |
| 3 | [Policy gradient: the math behind 'do more of what worked'](03-policy-gradient/) | 9 | ready |
| 4 | [Several calls, one reward: a tiny agent](04-episodes/) | 7 | ready |

### Part 2. The project's algorithms, still on the toy

| # | chapter | minutes | status |
| ---: | --- | ---: | --- |
| 5 | [GRPO, part 1: compare attempts within a group](05-grpo-groups/) | 8 | ready |
| 6 | [GRPO, part 2: the importance ratio and the clip](06-ratio-and-clip/) | 9 | ready |
| 7 | [Rewards that backfire](07-reward-design/) | 8 | ready |
| 8 | [Imitation (SFT) on the toy](08-sft-toy/) | 8 | ready |

### Part 3. The model on your laptop

| # | chapter | minutes | status |
| ---: | --- | ---: | --- |
| 9 | [From four choices to 128,000: tokens and chat templates](09-tokens-and-templates/) | 7 | ready |
| 10 | [Meet LFM2.5-2.6B](10-meet-lfm/) | 10 | ready |
| 11 | [Quantize it yourself: 16 bits to 4](11-quantize/) | 11 | ready |
| 12 | [How the model picks each token](12-generate/) | 8 | ready |

### Part 4. Training the real model where we own the loop

| # | chapter | minutes | status |
| ---: | --- | ---: | --- |
| 13 | [LoRA: train a 0.4% add-on, not the whole model](13-lora/) | 8 | ready |
| 14 | [SFT on the real model](14-sft-real-model/) | 8 | ready |
| 15 | [GRPO on the real model, and why exact tokens matter](15-grpo-real-model/) | 9 | ready |

### Part 5. Inside a harness

| # | chapter | minutes | status |
| ---: | --- | ---: | --- |
| 16 | [What a harness is: Claude Code from the inside](16-inside-claude-code/) | 11 | ready |
| 17 | [Tool calls at the token level: a tiny harness](17-tiny-harness/) | 8 | ready |
| 18 | An engine that keeps the receipts | 10 | coming |
| 19 | The sandbox: a small model typing bash on your laptop | 9 | coming |

### Part 6. Capturing real harnesses

| # | chapter | minutes | status |
| ---: | --- | ---: | --- |
| 20 | The capture proxy: recording a program you didn't write | 10 | coming |
| 21 | From calls to training rows | 8 | coming |
| 22 | Capture that fails without an error | 8 | coming |
| 23 | Three real harnesses, wired by configuration only | 9 | coming |

### Part 7. Training inside real harnesses, measured honestly

| # | chapter | minutes | status |
| ---: | --- | ---: | --- |
| 24 | Measuring fairly: baseline, pass@1 and noise | 10 | coming |
| 25 | SFT through the harnesses (and fitting it in 9 GB) | 12 | coming |
| 26 | GRPO through the harnesses, and the run that went wrong | 9 | coming |
| 27 | How it all combines | 8 | coming |
| 28 | Capstone: run the whole pipeline, read the results honestly | 10 | coming |
| 29 | Make it yours | 10 | coming |

## Develop

```bash
uv sync
uv run pre-commit install
uv run pytest
uv run ruff format --check . && uv run ruff check .
markdownlint "**/*.md"
```

## Credits

- The method comes from
  [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)
  (Hugging Face and Liquid AI, 2026).
- Model: [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) by Liquid AI.
- Code: MIT, Simplifying AI.
