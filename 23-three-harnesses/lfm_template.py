"""LFM2.5's chat template and tokenizer, without the weights or transformers.

The scripted engine answers from a script, but it counts the prompt the way
the real engine does: the same chat template (``chat_template.jinja``), the
same tokenizer (``tokenizer.json``), the same call
(``apply_chat_template(..., add_generation_prompt=True,
preserve_thinking=True)``). So "the first call read 2,281 tokens" means the
same thing with the script as with the model: what each harness costs before
the model has written a word.

Chapter 9's ``lfm.py`` showed that ``jinja2`` and ``tokenizers``, set up the
way transformers sets them up, give the project's token ids exactly; this is
its template half, copied. The real engine (``lfm_text.py``) renders through
transformers itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.parser import Parser
from jinja2.sandbox import ImmutableSandboxedEnvironment
from policy import Message, RenderError, Tool

CHAT_TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
# transformers hands the template every special token by name; this template
# opens the prompt with bos_token (<|startoftext|>).
SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "pad_token")
END_TAG = "name:endgeneration"


class GenerationTag(Extension):
    """Jinja's ``{% generation %}`` tag, which outputs its body unchanged.

    LFM2.5's template wraps every assistant message in it; transformers
    defines it, plain Jinja does not (chapter 9).
    """

    tags = {"generation"}

    def parse(self, parser: Parser) -> nodes.Node:
        """Read everything up to ``{% endgeneration %}`` as the tag's body.

        Args:
            parser: Jinja's parser, positioned on the tag.

        Returns:
            A node that renders the body through ``_body``.
        """
        line = next(parser.stream).lineno
        body = parser.parse_statements((END_TAG,), drop_needle=True)
        call = self.call_method("_body")
        return nodes.CallBlock(call, [], [], body).set_lineno(line)

    def _body(self, caller: Callable[[], str]) -> str:
        """Render the body unchanged.

        Args:
            caller: Jinja's function that renders the body.

        Returns:
            The body's text.
        """
        return caller()


def tojson(
    value: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    """Write a value as JSON the way transformers' templates do (plain json.dumps).

    Args:
        value: What to write.
        ensure_ascii: Escape every non-ASCII character.
        indent: Spaces per indent level, or None for one line.
        separators: The item and key separators, or None for the defaults.
        sort_keys: Sort the keys of every object.

    Returns:
        The JSON text.
    """
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def raise_exception(message: str) -> NoReturn:
    """Let the template refuse a chat it cannot render, as transformers does.

    Args:
        message: The template's explanation.

    Raises:
        jinja2.TemplateError: Always.
    """
    raise jinja2.TemplateError(message)


def template_environment() -> ImmutableSandboxedEnvironment:
    """Build Jinja exactly as transformers builds it for chat templates.

    Returns:
        The environment.
    """
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[GenerationTag, loopcontrols],
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    return environment


class LfmTemplate:
    """LFM2.5's chat template, loaded from a model folder.

    Attributes:
        folder: The folder it was read from.
    """

    def __init__(self, folder: Path) -> None:
        """Read chat_template.jinja and the special tokens it needs.

        Args:
            folder: A build's folder (it holds tokenizer.json beside them).

        Raises:
            OSError: If either file cannot be read.
        """
        self.folder = folder
        config = json.loads((folder / TOKENIZER_CONFIG_FILE).read_text("utf-8"))
        self.special_tokens = {
            key: config[key] for key in SPECIAL_TOKEN_KEYS if key in config
        }
        source = (folder / CHAT_TEMPLATE_FILE).read_text(encoding="utf-8")
        self.template = template_environment().from_string(source)

    def render(self, messages: list[Message], tools: list[Tool] | None) -> str:
        """Turn a conversation into the text the model reads, as the engine does.

        Args:
            messages: The conversation, as OpenAI's API writes it.
            tools: The tools offered, or None.

        Returns:
            The prompt, ending with ``<|im_start|>assistant`` and ``<think>``.

        Raises:
            RenderError: If the template refuses the conversation.
        """
        try:
            return self.template.render(
                messages=messages,
                tools=tools or None,
                add_generation_prompt=True,
                preserve_thinking=True,
                **self.special_tokens,
            )
        except jinja2.TemplateError as error:
            raise RenderError(str(error)) from error
