# Chapter 25. SFT through the harnesses (and fitting it in 9 GB)

**Status: the code is built and tested without the model's weights. The
measured runs, and every number this page will quote, come with the next
commit, written from the run log.** Until then this page lists only what
each script does and how to run it.

Chapter 14 taught LFM2.5-2.6B from rows we rendered ourselves. This chapter
trains it the way the project does: an answer-key teacher plays the model
inside the real harnesses, the capture proxy records every call in the
student's own tokens, and the trainer learns from exactly the tokens the
teacher wrote (mask 1), never from what the harness wrote around them
(mask 0).

## Run it

```bash
cd multi-harness-RL-course/25-sft-in-harnesses
uv run teacher.py                 # one demonstration, in both training harnesses' formats
./start.sh oracle                 # the teacher (no weights) and the capture proxy
uv run collect.py --fast          # 10 tasks x 2 harnesses; --limit 60 is the project's run
./start.sh stop
uv run memory_probe.py            # one forward + backward pass, 1K to 8K tokens, under 9 GB
uv run train_sft.py               # SFT on the captured rows
./start.sh                        # the model's engine and the proxy
uv run quick_eval.py --fast       # base against SFT, the same held-out cells
./start.sh stop
```

## The files

| file | what it is |
| --- | --- |
| `teacher.py` | the project's answer-key teacher (`mhrl/engine/oracle.py`): four paths, the answer saved before the finish |
| `engine.py` | chapter 24's engine, plus `--oracle` (the teacher, tokenizer only, every log-probability 0) and adapter switching (`--adapter`, `POST /admin/adapter`) |
| `start.sh` | chapter 24's stack, plus the teacher on `ORACLE_PORT` (8101, the project's) |
| `collect.py` | the project's `collect_demonstrations`: the teacher through mini-swe-agent and Claude Code, 4 rollouts at a time |
| `training.py` | the project's `mhrl/training/common.py` (rows, chunked attention, checkpointing, the chunked LM head) and chapter 14's loop |
| `train_sft.py` | chapter 14's trainer, reading captured rows |
| `quick_eval.py` | chapter 24's evaluation on a few held-out tasks per harness, base against SFT, cell by cell |
| `memory_probe.py` | peak memory per row length in three settings; refuses any row projected over the budget |
| `lora.py`, `project_run.py` | chapter 14's, `lora.py` with the project's adapter shape and a check on the adapter file |
| the rest | chapter 24's, unchanged but for this chapter's folder name |

## Mirrors in the project

`mhrl/engine/oracle.py` (`AnswerKeyTeacher`), `mhrl/training/sft.py`
(`collect_demonstrations`, `demo_items`, `train_sft`) and
`mhrl/training/common.py` (`chunked_attention`, `load_policy`,
`token_logprobs`).
