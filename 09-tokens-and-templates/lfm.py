"""LFM2.5-2.6B's tokenizer and chat template, without the transformers library.

The project turns a chat into prompt tokens with transformers'
``tokenizer.apply_chat_template``. That call does two jobs: it fills the
model's Jinja template (a text file with blanks, ifs and loops) with the
messages, and it cuts the resulting string into tokens. Here the same two jobs
are done by the two small libraries transformers itself uses for them,
``jinja2`` and ``tokenizers``, set up the way transformers sets them up. The
chapter's tests check the token ids against the project's own render.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import jinja2
from chats import Message, Tool
from hub import CHAT_TEMPLATE, TOKENIZER_CONFIG, TOKENIZER_JSON, HubError, ensure_files
from jinja2 import nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.parser import Parser
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

# transformers hands the template every special token by name; this template
# uses bos_token (<|startoftext|>) to open the prompt. In this model's
# tokenizer_config.json each one is a plain string.
SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "pad_token")
END_TAG = "name:endgeneration"


class GenerationTag(Extension):
    """Jinja's ``{% generation %}`` tag, which outputs its body unchanged.

    LFM2.5's template wraps every assistant message in this tag. transformers
    defines it so it can note which characters the model wrote (for training
    masks); when nobody asks for that note it outputs the body as it is, and so
    does this. Plain Jinja does not know the tag and would reject the template.
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
    """Write a value as JSON the way transformers' templates do.

    Jinja's own ``tojson`` sorts the keys and escapes ``<``, ``>``, ``&`` and
    ``'`` for web pages; transformers replaces it with plain ``json.dumps``, so
    a tool definition keeps its own key order and characters.

    Args:
        value: What to write.
        ensure_ascii: Escape every non-ASCII character.
        indent: Spaces per indent level, or None for one line.
        separators: The item and key separators, or None for ``", "`` and ``": "``.
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

    The template is a file downloaded from the internet, so it runs in Jinja's
    sandbox, which stops it touching anything but its inputs. ``trim_blocks``
    and ``lstrip_blocks`` drop the newline and indentation around ``{% %}``
    lines, and every character of the output counts. (transformers also offers
    ``strftime_now``, the date, which this template never asks for.)

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


class LfmTokenizer:
    """The model's three tokenizer files, loaded: encode, decode and render chats."""

    def __init__(self, folder: Path) -> None:
        """Load tokenizer.json, tokenizer_config.json and chat_template.jinja.

        Args:
            folder: The folder that holds the three files.

        Raises:
            OSError: If a file is missing (``hub.ensure_files`` downloads them).
        """
        self.tokenizer = Tokenizer.from_file(str(folder / TOKENIZER_JSON.name))
        config_text = (folder / TOKENIZER_CONFIG.name).read_text(encoding="utf-8")
        config = json.loads(config_text)
        self.special_tokens = {
            key: config[key] for key in SPECIAL_TOKEN_KEYS if key in config
        }
        source = (folder / CHAT_TEMPLATE.name).read_text(encoding="utf-8")
        self.template = template_environment().from_string(source)

    def encode(self, text: str) -> list[int]:
        """Cut text into token ids, adding nothing (as the project encodes).

        Args:
            text: Any text; marker tokens written in it become their ids.

        Returns:
            The ids.
        """
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text, markers included.

        Args:
            ids: Token ids.

        Returns:
            The text.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def piece(self, token_id: int) -> str:
        """Return the text of one token.

        Args:
            token_id: A token id.

        Returns:
            Its text (a token that holds only part of a character, as some
            emoji need, shows as the replacement character).
        """
        return self.decode([token_id])

    def pieces(self, ids: list[int]) -> list[str]:
        """Return the text of each token, one by one.

        Args:
            ids: Token ids.

        Returns:
            One piece of text per id.
        """
        return self.tokenizer.decode_batch(
            [[token_id] for token_id in ids], skip_special_tokens=False
        )

    def token_id(self, token: str) -> int | None:
        """Look up the id of a token written out in full, such as ``<think>``.

        Args:
            token: The token's text.

        Returns:
            Its id, or None if it is not one token.
        """
        return self.tokenizer.token_to_id(token)

    def require_id(self, token: str) -> int:
        """Look up the id of a token that must exist, such as ``<think>``.

        Args:
            token: The token's text.

        Returns:
            Its id.

        Raises:
            KeyError: If it is not one token in tokenizer.json.
        """
        token_id = self.token_id(token)
        if token_id is None:
            raise KeyError(f"{token} is not a token in tokenizer.json")
        return token_id

    def vocabulary_size(self) -> int:
        """Count every entry: learned pieces plus added marker tokens.

        Returns:
            The number of ids the tokenizer can produce.
        """
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def learned_size(self) -> int:
        """Count only the pieces learned from text (ids below the added tokens).

        Returns:
            The number of learned pieces.
        """
        return self.tokenizer.get_vocab_size(with_added_tokens=False)

    def added_tokens(self) -> dict[int, str]:
        """List the tokens added on top of the learned pieces.

        Returns:
            Each added token's text, by id.
        """
        added = self.tokenizer.get_added_tokens_decoder()
        return {token_id: added[token_id].content for token_id in sorted(added)}

    def apply_chat_template(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        *,
        add_generation_prompt: bool = False,
        preserve_thinking: bool = False,
    ) -> str:
        """Fill the chat template, like transformers' call with ``tokenize=False``.

        Args:
            messages: OpenAI-style messages: dicts with a role and content.
            tools: OpenAI-style tool definitions, or None.
            add_generation_prompt: End with the cue for the model's reply.
            preserve_thinking: Keep the reasoning of every earlier reply.

        Returns:
            The prompt as one string.

        Raises:
            jinja2.TemplateError: If the template refuses the chat.
        """
        return self.template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            preserve_thinking=preserve_thinking,
            **self.special_tokens,
        )

    def render(self, messages: list[Message], tools: list[Tool] | None) -> list[int]:
        """Turn a chat request into prompt tokens, as the project's engine does.

        The project's ``PolicyEngine.render`` (mhrl/engine/policy.py), line for
        line, without the step that first swaps in the model's own text for
        earlier replies (chapter 18).

        Args:
            messages: OpenAI-style messages.
            tools: OpenAI-style tool definitions, or None.

        Returns:
            Prompt token ids, ending with the generation prompt.
        """
        text = self.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            preserve_thinking=True,
        )
        return self.encode(text)


def load_tokenizer() -> LfmTokenizer:
    """Download the three files if needed, then load them: every script's start.

    Returns:
        The model's tokenizer.

    Raises:
        SystemExit: With the download's explanation, if the files cannot be
            fetched; the script stops there and prints it.
    """
    try:
        folder = ensure_files()
    except HubError as error:
        raise SystemExit(str(error)) from None
    return LfmTokenizer(folder)
