"""LFM2.5's chat template and tokenizer, without the weights.

The text half of the model behind the engine: render a chat exactly as the
project's engine renders it (the chat template with the generation prompt,
``preserve_thinking`` on, so a reply put back into the conversation keeps its
reasoning), cut text into token ids, and write ids back as text with the
special tokens kept. ``drift.py --recorded`` and the tests need only this
half, an 18 MB tokenizer and no weights; ``lfm.py`` adds the MLX half.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import TemplateError
from policy import Message, RenderError, Tool


class LfmText:
    """LFM2.5's tokenizer and chat template.

    Attributes:
        tokenizer: A Hugging Face tokenizer, or mlx-lm's wrapper around one.
        eos_ids: The token ids that end a turn (``<|im_end|>``).
    """

    def __init__(self, tokenizer: Any, eos_ids: frozenset[int]) -> None:
        """Keep the tokenizer.

        Args:
            tokenizer: Anything with ``apply_chat_template``, ``encode`` and
                ``decode``, as Hugging Face's tokenizers have.
            eos_ids: The end-of-turn token ids.
        """
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids

    def render(self, messages: list[Message], tools: list[Tool] | None) -> str:
        """Turn a conversation into the text the model reads.

        Args:
            messages: The conversation, as OpenAI's API writes it.
            tools: The tools offered, or None; the template prints them into
                the system prompt as ``List of tools: [...]``.

        Returns:
            The prompt, ending with ``<|im_start|>assistant`` and ``<think>``.

        Raises:
            RenderError: If the template refuses the conversation (it refuses
                tool-call arguments written as a JSON string, for one).
        """
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                add_generation_prompt=True,  # end with the assistant's turn opening
                tokenize=False,
                preserve_thinking=True,  # keep every reply's reasoning, as written
            )
        except TemplateError as error:
            raise RenderError(str(error)) from error

    def encode(self, text: str) -> list[int]:
        """Cut text into token ids, adding no start-of-text token.

        Args:
            text: Any text; ``<think>`` and the other markers become one id each.

        Returns:
            The ids.
        """
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        """Write token ids back as text, with the special tokens kept.

        Args:
            token_ids: The ids.

        Returns:
            The text; the tool-call markers stay in it.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def load_text(folder: Path) -> LfmText:
    """Load only the tokenizer and the chat template of a build (no weights).

    Args:
        folder: A build's folder, holding tokenizer.json and
            chat_template.jinja.

    Returns:
        The text half of the model.
    """
    from transformers import AutoTokenizer  # loads here, so the rest runs without it

    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    return LfmText(tokenizer, frozenset({tokenizer.eos_token_id}))
