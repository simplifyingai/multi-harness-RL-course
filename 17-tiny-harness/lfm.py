"""The real policy: the 4-bit LFM2.5-2.6B, loaded with mlx-lm, in this process.

Chapter 15's ``lfm.py`` cut down to one class. ``render`` is the model's own
chat template (``apply_chat_template`` with the tools, which the template
prints into the system prompt), with ``preserve_thinking`` on as the
project's engine renders, so a reply the harness puts back into the
conversation is read again exactly as it was written. ``complete`` is
chapter 12's loop over mlx-lm's ``generate_step``: at temperature 0 (the
project's evaluation setting) always the likeliest token, up to the
1,024-token cap, and the reply is decoded with its special tokens kept, so
the tool-call markers stay in the text. Nothing is cached between calls:
every call reads the whole conversation again (chapter 18's engine keeps a
cache).
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from policy import FINISH_LENGTH, FINISH_STOP, Message, Reply, Tool

PREFILL_STEP = 1_024  # the project's Defaults.PREFILL_STEP (chapter 12)
GREEDY = 0.0


class LfmPolicy:
    """LFM2.5-2.6B in 4 bits, answering in this process.

    Attributes:
        folder: The build it was loaded from.
        seconds: How long loading took.
    """

    def __init__(
        self, folder: Path, cap: int, temperature: float = GREEDY, seed: int = 0
    ) -> None:
        """Load a build with mlx-lm, every weight into memory.

        Args:
            folder: The build's folder (read only; nothing is downloaded).
            cap: The most tokens one reply may have.
            temperature: 0 takes the likeliest token every time; above 0 the
                scores are divided by it and a token is drawn.
            seed: MLX's random seed, for a temperature above 0.
        """
        start = time.perf_counter()
        self.model, self.tokenizer = load(str(folder))
        mx.eval(self.model.parameters())  # MLX is lazy: read every weight now
        self.folder = folder
        self.seconds = time.perf_counter() - start
        self.cap = cap
        self.temperature = temperature
        mx.random.seed(seed)

    def render(self, messages: list[Message], tools: list[Tool] | None) -> str:
        """Turn the conversation into the text the model reads.

        Args:
            messages: The conversation so far.
            tools: The tools offered, or None.

        Returns:
            The prompt, ending with ``<|im_start|>assistant`` and ``<think>``.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,  # end with the assistant's turn opening
            tokenize=False,
            preserve_thinking=True,  # keep each reply's reasoning, as the engine does
        )

    def pick(self, logprobs: mx.array) -> mx.array:
        """Choose the next token from the model's log-probabilities.

        Args:
            logprobs: The step's log-probabilities over the vocabulary.

        Returns:
            The token: the likeliest at temperature 0, else drawn.
        """
        if self.temperature == GREEDY:
            return mx.argmax(logprobs, axis=-1)
        return mx.random.categorical(logprobs / self.temperature)

    def complete(self, prompt: str) -> Reply:
        """Write a reply, token by token, up to the cap.

        Args:
            prompt: The rendered conversation.

        Returns:
            The reply, its special tokens kept.
        """
        start = time.perf_counter()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens: list[int] = []
        finish = FINISH_LENGTH
        for token, _ in generate_step(
            mx.array(prompt_ids),
            self.model,
            max_tokens=self.cap,
            sampler=self.pick,
            prefill_step_size=PREFILL_STEP,
        ):
            tokens.append(int(token))
            if token in self.tokenizer.eos_token_ids:  # <|im_end|>: the end of turn
                finish = FINISH_STOP
                break
        return Reply(
            text=self.tokenizer.decode(tokens, skip_special_tokens=False),
            prompt_tokens=len(prompt_ids),
            reply_tokens=len(tokens),
            finish_reason=finish,
            seconds=time.perf_counter() - start,
        )


def peak_memory() -> int:
    """Return the most memory MLX has held since the process started.

    Returns:
        Bytes, the weights included.
    """
    return mx.get_peak_memory()
