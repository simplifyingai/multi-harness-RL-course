"""How the chapter's scripts print: files, counts, stages, timings.

Nothing here computes anything. It only turns the scripts' results into lines
a person can read on screen, so each script can stay about the model.
"""

from __future__ import annotations

import re
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lfm_files import (
    BF16,
    BF16_BYTES,
    FINISH_STOP,
    FLOAT32_BYTES,
    GIGABYTE,
    LAYER_ATTENTION,
    LAYER_CONV,
    PART_ATTENTION,
    PART_EMBEDDING,
    PART_HEAD,
    PART_NORMS,
    HubFile,
    ModelRepo,
    gigabytes,
    short_path,
)

if TYPE_CHECKING:  # for type hints only: these modules import this one
    from download import LicenseTerms
    from first_reply import Reply
    from inside import (
        Embedding,
        Head,
        Heads,
        LayerMap,
        Mask,
        Precision,
        Rope,
        TokensIn,
    )
    from inspect_model import Counted, Expected, Inventory
    from lfm import LoadedModel

INDENT = "  "
WRAP_WIDTH = 88
PERCENT = 100
NAME_WIDTH = 34  # the longest file name, model-00001-of-00002.safetensors
BYTES_WIDTH = 13  # 5,329,406,264
TENSOR_WIDTH = 46  # model.layers.2.self_attn.q_layernorm.weight plus room
SHAPE_WIDTH = 15
CONFIG_KEY_WIDTH = 25
CONFIG_VALUE_WIDTH = 11
PART_WIDTH = 40
COUNT_WIDTH = 14
FORMULA_WIDTH = 60
SHORT_HASH = 12
GRID_CELL = 4  # one column of the attention grid
GRID_LABEL = 21  # the token labels in front of the grid
MASKED = "."
UNMASKED = "?"  # a masked cell that is not exactly 0: never expected
ROPE_LABEL = 28
LAYER_LETTERS = {LAYER_CONV: "C", LAYER_ATTENTION: "A"}
TOP_PIECE_WIDTH = 10
FIRST_PIECES = 5  # tokens listed one by one after the reply
NAME_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?B)")
BILLION = 1_000_000_000
MEGA = 1_000_000
MILLISECONDS = 1_000
BITS_PER_BYTE = 8
PACKING_SOURCE = "config.json's quantization block"


def print_title(script: str, title: str) -> None:
    """Print the script's first line.

    Args:
        script: The script's file name.
        title: What it shows.
    """
    print(f"{script} · {title}")


def wrapped(text: str, indent: str = INDENT) -> str:
    """Wrap a long line, indenting its continuation lines a little more.

    Args:
        text: One line of text.
        indent: The indent of the first line.

    Returns:
        The text as one or more lines.
    """
    return textwrap.fill(
        text,
        width=WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent + INDENT,
        break_on_hyphens=False,
    )


# ---------------------------------------------------------------- download.py


def print_source(repo: ModelRepo, folder: Path, say: Callable[[str], None]) -> None:
    """Print which repository and revision, and where its files go.

    Args:
        repo: The repository.
        folder: Where its files go.
        say: Where the lines go.
    """
    say(
        f"repository  {repo.repo_id}, revision {repo.revision[:SHORT_HASH]} "
        f"({repo.date}), {repo.label}"
    )
    say(f"into        {short_path(folder)}")


def print_already_here(repo: ModelRepo, say: Callable[[str], None]) -> None:
    """Say that nothing needs downloading.

    Args:
        repo: The repository.
        say: Where the lines go.
    """
    say(
        f"already here: all {len(repo.files)} files at full size "
        "(SHA-256 checked when downloaded)"
    )


def print_downloading(
    repo: ModelRepo, todo: Sequence[HubFile], say: Callable[[str], None]
) -> None:
    """Say what is about to be downloaded.

    Args:
        repo: The repository.
        todo: The files still missing.
        say: Where the lines go.
    """
    size = gigabytes(sum(item.size for item in todo))
    say(f"downloading {len(todo)} files, {size}, with snapshot_download")
    say("(progress shows below; a download that breaks off resumes when run again)")
    sys.stdout.flush()  # before the progress bars, which go to the terminal


def print_checked(repo: ModelRepo, say: Callable[[str], None]) -> None:
    """Say that every file passed its fingerprint check.

    Args:
        repo: The repository.
        say: Where the lines go.
    """
    say(f"all {len(repo.files)} files checked against their SHA-256 fingerprints")


def print_files(repo: ModelRepo) -> None:
    """List the repository's files with their sizes and what each is for.

    Args:
        repo: The repository.
    """
    print("\nthe repository's files")
    for item in repo.files:
        print(
            f"{INDENT}{item.name:<{NAME_WIDTH}}{item.size:>{BYTES_WIDTH},} bytes  "
            f"{item.role}"
        )
    print(f"{INDENT}{len(repo.files)} files, {gigabytes(repo.total_bytes())} in all")


def print_license(terms: LicenseTerms, size: int) -> None:
    """Print the license's title and its commercial terms, as written.

    Args:
        terms: The lines picked out of the LICENSE file.
        size: The file's size in bytes.
    """
    print(f"\nthe license, quoted from LICENSE ({size:,} bytes)")
    print(f"{INDENT}{terms.title}")
    for line in [*terms.definitions, *terms.limit]:
        print(wrapped(line))


# ----------------------------------------------------------- inspect_model.py


def print_headers(folder: Path, found: Inventory) -> None:
    """Print each safetensors file's size, header length and tensor count.

    Args:
        folder: The model's folder.
        found: What the files say.
    """
    print(f"\nreading only the headers in {short_path(folder)}")
    read = 0
    for header in found.headers:
        read += header.header_bytes
        print(
            f"{INDENT}{header.name:<{NAME_WIDTH}}{header.size:>{BYTES_WIDTH},} bytes, "
            f"header {header.header_bytes:,} bytes, {len(header.tensors)} tensors"
        )
    total = found.file_bytes()
    print(
        f"{INDENT}read {read:,} of {total:,} bytes "
        f"({PERCENT * read / total:.4f}%); no weight loaded"
    )


def rope_theta(config: dict[str, Any]) -> tuple[float, str]:
    """Find rope_theta, which config.json may keep inside rope_parameters.

    Args:
        config: config.json.

    Returns:
        Its value and where it was found.
    """
    nested = config.get("rope_parameters") or {}
    if "rope_theta" in nested:
        return nested["rope_theta"], "rope_parameters"
    return config["rope_theta"], "rope_theta"


def print_config(config: dict[str, Any]) -> None:
    """Print the config.json settings the chapter uses, with their meanings.

    Args:
        config: config.json.
    """
    types = config["layer_types"]
    theta, where = rope_theta(config)
    rows = (
        ("hidden_size", f"{config['hidden_size']:,}", "numbers in each token's vector"),
        ("vocab_size", f"{config['vocab_size']:,}", "rows in the embedding table"),
        (
            "num_hidden_layers",
            f"{config['num_hidden_layers']}",
            f"{types.count('conv')} conv + {types.count('full_attention')} "
            "full_attention (layer_types)",
        ),
        (
            "intermediate_size",
            f"{config['intermediate_size']:,}",
            "width of each feed-forward block",
        ),
        ("conv_L_cache", f"{config['conv_L_cache']}", "tokens each convolution sees"),
        (
            "num_attention_heads",
            f"{config['num_attention_heads']}",
            "query heads in each attention layer",
        ),
        (
            "num_key_value_heads",
            f"{config['num_key_value_heads']}",
            "key/value heads they share",
        ),
        ("rope_theta", f"{theta:,.0f}", f"how slowly positions turn ({where})"),
        (
            "max_position_embeddings",
            f"{config['max_position_embeddings']:,}",
            "the most positions it reads",
        ),
        (
            "tie_word_embeddings",
            str(config["tie_word_embeddings"]).lower(),
            "the embedding table is also the output head",
        ),
        ("dtype", config["dtype"], "how each number is stored"),
    )
    print("\nconfig.json, the settings this chapter uses")
    for key, value, meaning in rows:
        name = f"{key:<{CONFIG_KEY_WIDTH}}"
        print(f"{INDENT}{name}{value:>{CONFIG_VALUE_WIDTH}}  {meaning}")


def shape_text(shape: Sequence[int]) -> str:
    """Write a shape the way people say it: ``10752 x 2048``.

    Args:
        shape: The sizes.

    Returns:
        The shape as text.
    """
    return " x ".join(str(size) for size in shape)


def print_sample_tensors(found: Inventory, names: Sequence[str]) -> None:
    """Print a few tensors as the header lists them, and what they hold.

    Args:
        found: What the files say.
        names: The tensors to show, in order.
    """
    by_name = {item.tensor.name: item for item in found.counted}
    print("\nthe embedding table, layer 0 (convolution) and layer 2's attention:")
    print(
        f"{INDENT}{'tensor':<{TENSOR_WIDTH}}{'type':<6}{'shape':<{SHAPE_WIDTH}}"
        f"{'parameters':>{COUNT_WIDTH}}"
    )
    for name in names:
        if name in by_name:
            print_tensor_row(by_name[name])


def print_tensor_row(item: Counted) -> None:
    """Print one tensor: its name, type, stored shape and parameters.

    Args:
        item: The counted tensor.
    """
    tensor = item.tensor
    print(
        f"{INDENT}{tensor.name:<{TENSOR_WIDTH}}{tensor.dtype:<6}"
        f"{shape_text(tensor.shape):<{SHAPE_WIDTH}}{item.parameters:>{COUNT_WIDTH},}"
    )
    if item.bits is not None:
        packed = tensor.shape[-1]
        print(
            f"{INDENT * 2}packed: {packed} words x 32 / {item.bits} bits = "
            f"{item.shape[-1]:,} columns, so it holds {shape_text(item.shape)}"
        )


def part_label(part: str, found: Inventory) -> str:
    """Name a part with where it sits in the model.

    Args:
        part: The part's name.
        found: What the files say.

    Returns:
        Such as ``"attention, in 8 layers"``.
    """
    layers = found.layers_with(part)
    rows, columns = found.config["vocab_size"], found.config["hidden_size"]
    labels = {
        PART_EMBEDDING: f"{PART_EMBEDDING}, {rows:,} x {columns:,}",
        PART_HEAD: PART_HEAD,
        PART_NORMS: f"{PART_NORMS}, 2 per layer + 1 at the end",
    }
    return labels.get(part, f"{part}, in {len(layers)} layers")


def print_parts(found: Inventory) -> None:
    """Print the parameter count of each part, and the total.

    Args:
        found: What the files say.
    """
    total = found.total_parameters()
    print("\nparameters by part")
    print(
        f"{INDENT}{'part':<{PART_WIDTH}}{'tensors':>8}{'parameters':>{COUNT_WIDTH + 1}}"
        f"{'share':>8}"
    )
    tensors = 0
    for part in found.parts:
        tensors += part.tensors
        share = PERCENT * part.parameters / total
        print(
            f"{INDENT}{part_label(part.part, found):<{PART_WIDTH}}{part.tensors:>8}"
            f"{part.parameters:>{COUNT_WIDTH + 1},}{share:>7.1f}%"
        )
    print(
        f"{INDENT}{'total':<{PART_WIDTH}}{tensors:>8}{total:>{COUNT_WIDTH + 1},}"
        f"{PERCENT:>7.1f}%"
    )
    attention = ", ".join(str(layer) for layer in found.layers_with(PART_ATTENTION))
    from_config = [
        str(i) for i, kind in enumerate(found.layer_types) if kind == LAYER_ATTENTION
    ]
    agrees = "agrees" if attention == ", ".join(from_config) else "DISAGREES"
    print(f"{INDENT}attention tensors sit in layers {attention}")
    print(f"{INDENT}(config.json's layer_types {agrees})")
    if found.config["tie_word_embeddings"] and not found.layers_with(PART_HEAD):
        print(f"{INDENT}no output head is stored: tie_word_embeddings is true")


def print_expected(expected: Sequence[Expected], found: Inventory) -> None:
    """Print each part's count worked out from config.json, and whether it agrees.

    Args:
        expected: The counts from config.json.
        found: What the files say.
    """
    counted = {part.part: part.parameters for part in found.parts}
    print("\nthe same counts from config.json alone")
    disagree = []
    for entry in expected:
        mark = "" if counted.get(entry.part) == entry.parameters else "  DIFFERS"
        if mark:
            disagree.append(entry.part)
        name = entry.part.split()[0]
        print(
            f"{INDENT}{name:<13}{entry.formula:<{FORMULA_WIDTH}}= "
            f"{entry.parameters:>{BYTES_WIDTH},}{mark}"
        )
    if disagree:
        print(f"{INDENT}these parts differ from the files: {', '.join(disagree)}")
    else:
        print(f"{INDENT}every part agrees with the files")


def print_naming(repo: ModelRepo, card: str | None, total: int) -> None:
    """Print what the name and the model card say, beside the real count.

    Args:
        repo: The repository.
        card: The count the model card states, or None.
        total: The counted parameters.
    """
    match = NAME_SIZE_RE.search(repo.repo_id)
    name = match.group(1) if match else "nothing"
    card_text = f"the model card says {card}" if card else "the model card says nothing"
    print(
        f"\nthe name says {name}, {card_text}; counted: {total:,}, "
        f"about {total / BILLION:.2f} B"
    )
    print("model names round: count it yourself")


def print_size(found: Inventory) -> None:
    """Print the bytes the numbers take, and the packing of a 4-bit export.

    Args:
        found: What the files say.
    """
    total = found.total_parameters()
    stored = found.tensor_bytes()
    files = found.file_bytes()
    if found.quantization is None:
        bytes_each = stored // total
        print(
            f"\nmemory: every number is stored in {found.config['dtype']}, "
            f"{bytes_each} bytes"
        )
        print(
            f"{INDENT}{total:,} numbers x {bytes_each} bytes = {stored:,} bytes, "
            f"{stored / GIGABYTE:.2f} GB"
        )
        print(
            f"{INDENT}the files: {files:,} bytes, the numbers plus "
            f"{files - stored:,} bytes of headers"
        )
        print(
            f"{INDENT}in float32, {FLOAT32_BYTES} bytes each, the same numbers would "
            f"take {FLOAT32_BYTES * total / GIGABYTE:.2f} GB"
        )
        return
    packing = found.quantization
    overrides = ", ".join(
        f"{module} at {bits} bits" for module, bits in packing.overrides.items()
    )
    bits = BITS_PER_BYTE * stored / total
    print(f"\nmemory: the 4-bit export packs its matrices ({PACKING_SOURCE})")
    lines = (
        f"most at {packing.bits} bits in groups of {packing.group_size}, "
        f"{overrides}; a packed matrix's real width is its stored width x 32 / bits, "
        "as the rows above show",
        f"each group of {packing.group_size} numbers also keeps a scale and an offset "
        "(.scales, .biases): they only help store the numbers, so they hold no "
        "parameters",
        f"counting stored values instead would say {found.stored_values():,}: "
        "wrong, the packed words hide the rest",
        f"{total:,} numbers in {stored:,} bytes ({gigabytes(stored)}): {bits:.2f} bits "
        "each on average (chapter 11 builds this file itself)",
    )
    for line in lines:
        print(wrapped(line))


# ------------------------------------------------------------------ inside.py


def print_loaded(loaded: LoadedModel, walk_difference: float) -> None:
    """Print what was loaded, and the check that the walk is mlx-lm's own pass.

    Args:
        loaded: The model.
        walk_difference: The walk's scores against one call of the model.
    """
    print(f"loaded {short_path(loaded.folder)} with mlx-lm:")
    print(
        f"{INDENT}{loaded.parameters:,} numbers, {gigabytes(loaded.weight_bytes)} "
        "in memory"
    )
    print("the walk below runs the layers one at a time; its scores equal one call")
    print(f"{INDENT}of the model (largest difference {walk_difference:g})")


def quoted(pieces: Sequence[str]) -> str:
    """Write token pieces side by side, each in quotes so spaces show.

    Args:
        pieces: The text of each token.

    Returns:
        The pieces as one line.
    """
    return " ".join(repr(piece) for piece in pieces)


def print_tokens_in(prompt: TokensIn) -> None:
    """Print the prompt as tokens.

    Args:
        prompt: The prompt.
    """
    count = len(prompt.ids)
    print("\n1. tokens in")
    print(f'the prompt for "{prompt.question}" is {count} tokens (chapter 9):')
    print(quoted(prompt.pieces))


def print_embedding(embed: Embedding) -> None:
    """Print the shown token's row of the embedding table.

    Args:
        embed: The row and the shapes.
    """
    numbers = ", ".join(f"{value:.6g}" for value in embed.vector)
    tokens, columns = embed.prompt_shape
    print("\n2. each token becomes a vector")
    print(
        f"the embedding table is a matrix of {embed.rows:,} rows x {embed.columns:,} "
        "columns, one row per token id"
    )
    print(
        f"token {embed.token_id} {embed.piece!r} -> row {embed.token_id}, a vector "
        f"of {embed.columns:,} numbers:"
    )
    print(f"{INDENT}[{numbers}, ...]")
    print(f"the {tokens} tokens become a {tokens} x {columns:,} matrix;")
    print(f"{INDENT}every layer reads a matrix of that shape and writes one back")


def fraction_text(step: float) -> str:
    """Write a power-of-two step the way people say it: ``1/16``.

    Args:
        step: A power of two.

    Returns:
        Such as ``"1/16"`` or ``"2"``.
    """
    return f"1/{round(1 / step)}" if step < 1 else f"{step:g}"


def print_precision(precision: Precision, loaded: LoadedModel) -> None:
    """Print one weight's bits and what bfloat16 rounds away.

    Args:
        precision: The bits, neighbours and rules.
        loaded: The model, for what its weights take in memory.
    """
    step = precision.above - precision.value
    one_bf16, one_f32 = precision.small_step
    log_bf16, log_f32 = precision.log_prob
    print("\n3. how each number is stored: bfloat16")
    which = (
        "the first number"
        if precision.index == 0
        else f"number {precision.index + 1}, the first positive one,"
    )
    print(f"{which} of that row, {precision.value!r}, is these 16 bits:")
    print(
        f"{INDENT}sign {precision.sign} | exponent {precision.exponent} | "
        f"fraction {precision.fraction}"
    )
    print(
        f"{INDENT}its bfloat16 neighbours: {precision.below!r} below, "
        f"{precision.above!r} above"
    )
    print(
        f"{INDENT}a step of {step:.6f}; float32 has {precision.finer:,} more numbers "
        "inside each such step"
    )
    print("small differences get rounded away (computed in MLX):")
    first, second = precision.sum_inputs
    sum_label = f"{first:g} + {second:g}"
    stored_label = f"{precision.stored_input:g}"
    print(f"{INDENT}{sum_label:<12}bfloat16 {one_bf16:<10g} float32 {one_f32:.7g}")
    print(f"{INDENT}{stored_label:<12}bfloat16 {log_bf16:<10g} float32 {log_f32:.7g}")
    grid = precision.log_grid
    print(
        f"{INDENT}between {grid.low:g} and {grid.high:g}, bfloat16 holds only "
        f"multiples of {fraction_text(grid.step)}:"
    )
    print(f"{INDENT}log-probabilities (chapter 12) need float32")
    print("bfloat16 keeps float32's 8 exponent bits (the same range) and 7 of its")
    print(
        f"23 fraction bits: {gigabytes(precision.bf16_bytes)} of weights instead of "
        f"{gigabytes(precision.float32_bytes)} in float32"
    )
    if loaded.repo is not BF16:
        print(
            f"{INDENT}(this {loaded.repo.label} export holds them in "
            f"{gigabytes(loaded.weight_bytes)} and unpacks each to bfloat16 to "
            "compute)"
        )


def print_layer_map(layers: LayerMap) -> None:
    """Print the kind of every layer, and what each kind keeps in the cache.

    Args:
        layers: The map and the cache shapes.
    """
    total = len(layers.kinds)
    tens = "".join(f"{i // 10:<10}" for i in range(0, total, 10)).rstrip()
    units = "".join(str(i % 10) for i in range(total))
    letters = "".join(LAYER_LETTERS[kind] for kind in layers.kinds)
    attention = layers.attention_layers()
    legend = f"C convolution ({total - len(attention)}), A attention ({len(attention)})"
    _, kept, width = layers.conv_state
    _, kv_heads, tokens, head_dim = layers.kv_shape
    lora = layers.lora_layers
    lora_attention = [i for i in lora if layers.kinds[i] == LAYER_ATTENTION]
    print(f"\n4. {total} layers of two kinds")
    print(f"{INDENT}layer  {tens}")
    print(f"{INDENT}       {units}")
    print(f"{INDENT}kind   {letters}   {legend}")
    print(
        f"attention at layers {', '.join(str(i) for i in attention)} "
        "(counting from 0, as config.json does)"
    )
    print(
        f"a convolution layer mixes each token with the {layers.kernel - 1} just "
        f"before it (conv_L_cache {layers.kernel})"
    )
    print("an attention layer lets each token look back at every earlier token")
    print("every layer then runs its own feed-forward block (w1, w3, w2) on each token")
    print("after the prompt, the cache holds (measured):")
    print(
        f"{INDENT}a convolution layer: its last {kept} inputs, {kept} x {width:,} "
        "numbers, however long the prompt"
    )
    print(
        f"{INDENT}an attention layer: keys and values for all {tokens} tokens, "
        f"2 x {kv_heads} x {tokens} x {head_dim} numbers;"
    )
    print(f"{INDENT}it grows by one token for every token read or written")
    print("-> chapter 12 reuses this cache instead of re-reading the prompt")
    print(
        f"-> chapter 13 trains small add-ons on the last {len(lora)} layers "
        f"({lora[0]}-{lora[-1]}): {len(lora_attention)} attention, "
        f"{len(lora) - len(lora_attention)} convolution"
    )


def print_heads(heads: Heads, cap: int, long_row: int) -> None:
    """Print one attention layer's heads and the memory they save.

    Args:
        heads: The heads and their shapes.
        cap: The engine's prompt cap, in tokens.
        long_row: Chapter 25's longest training row, in tokens.
    """
    tokens = heads.queries[2]
    q_width = heads.query_heads * heads.head_dim
    kv_width = heads.kv_heads * heads.head_dim
    size = heads.group()
    last = heads.query_heads - size
    shared = heads.cache_bytes_per_token(heads.kv_heads)
    unshared = heads.cache_bytes_per_token(heads.query_heads)
    print(
        f"\n5. inside attention layer {heads.layer}: {heads.query_heads} query heads "
        f"share {heads.kv_heads} key/value heads"
    )
    print(f"the {tokens} x {heads.hidden:,} input becomes, for each token:")
    print(
        f"{INDENT}queries  {heads.query_heads:>2} heads x {heads.head_dim} numbers  "
        f"(q_proj {heads.hidden:,} -> {q_width:,})"
    )
    print(
        f"{INDENT}keys     {heads.kv_heads:>2} heads x {heads.head_dim} numbers  "
        f"(k_proj {heads.hidden:,} -> {kv_width})"
    )
    print(
        f"{INDENT}values   {heads.kv_heads:>2} heads x {heads.head_dim} numbers  "
        f"(v_proj {heads.hidden:,} -> {kv_width})"
    )
    print(
        f"query heads 0-{size - 1} read key/value head 0, {size}-{2 * size - 1} "
        f"head 1, ... {last}-{heads.query_heads - 1} head {heads.kv_heads - 1}"
    )
    print(
        f"the cache keeps {heads.attention_layers} layers x (keys + values) x "
        f"{heads.kv_heads} heads x {heads.head_dim} x {BF16_BYTES} bytes = "
        f"{shared:,} bytes "
        "per token"
    )
    print(
        f"{INDENT}with {heads.query_heads} key/value heads: {unshared:,}; at the "
        f"engine's {cap:,}-token cap, {gigabytes(unshared * cap)}, "
        f"not {gigabytes(shared * cap)}"
    )
    scores = long_row * long_row
    print(
        f"-> chapter 25: each head scores every token against every earlier one; "
        f"at {long_row:,} tokens"
    )
    print(
        f"{INDENT}that is {long_row:,} x {long_row:,} = {scores / MEGA:.0f} million "
        f"scores per head, {heads.query_heads} heads per layer"
    )


def print_mask(mask: Mask, layer: int, head: int) -> None:
    """Print one head's attention mask as a grid, and the check.

    Args:
        mask: The mask and the check.
        layer: The layer they come from.
        head: The query head.
    """
    tokens = len(mask.pieces)
    header = "".join(f"{i:>{GRID_CELL}}" for i in range(tokens))
    print("\n6. the causal mask: a token never looks ahead")
    print(
        f"head {head} of layer {layer}: the share each token (row) takes from each "
        "token (column),"
    )
    print(f"{INDENT}in percent; '{MASKED}' is exactly 0")
    print(f"{'':<{GRID_LABEL}}{header}")
    for row, piece in enumerate(mask.pieces):
        cells = []
        for column, value in enumerate(mask.weights[row]):
            if column <= row:
                cells.append(f"{round(PERCENT * value):>{GRID_CELL}}")
            else:  # a masked cell that is not exactly 0 would show as '?'
                cells.append(f"{MASKED if value == 0 else UNMASKED:>{GRID_CELL}}")
        label = f"{row:>2} {piece!r}"
        print(f"{label:<{GRID_LABEL}}{''.join(cells)}")
    print("every row adds up to 100 (give or take rounding); above the diagonal the")
    print(f"{INDENT}scores were set to minus infinity before the softmax")
    print(
        "computed by hand from the queries, keys and values, the layer's output matches"
    )
    print(
        f"{INDENT}mlx-lm's own to within {mask.largest_difference:.5f} "
        f"(its largest value is {mask.largest_output:.3f})"
    )


def print_rope(rope: Rope) -> None:
    """Print the rotary position settings and the moved query-key scores.

    Args:
        rope: The settings and the scores.
    """
    query_at, key_at = rope.near_positions
    far_query, far_key = rope.far_positions
    gap = query_at - key_at
    rows = [
        (f"{gap} apart, at {query_at} and {key_at}", rope.near, ""),
        (
            f"{gap} apart, at {far_query:,} and {far_key:,}",
            rope.far,
            "the same: only the distance counts",
        ),
        *[(f"{distance:,} apart", score, "") for distance, score in rope.spread],
    ]
    print("\n7. positions: RoPE (rotary position embedding)")
    print(
        f"each head's {2 * rope.pairs} numbers form {rope.pairs} pairs, and each "
        "pair is turned by an angle"
    )
    print(f"{INDENT}that grows with the token's position")
    print(
        f"rope_theta {rope.theta:,.0f} sets the speeds: the fastest pair turns once "
        f"every {rope.fastest:.1f} positions,"
    )
    print(
        f"{INDENT}the slowest once every {rope.slowest / MEGA:.1f} million, so all "
        f"{rope.positions:,} positions look different"
    )
    print("the same query (' Paris') and key (' customers'), moved to other positions:")
    for label, score, note in rows:
        print(f"{INDENT}{label:<{ROPE_LABEL}}{score:>9.4f}  {note}".rstrip())


def print_head(head: Head, long_row: int) -> None:
    """Print the top next tokens, and the check that the table is the head.

    Args:
        head: The top tokens and the check.
        long_row: Chapter 25's longest training row, in tokens.
    """
    tied = "true" if head.tied else "false"
    numbers = long_row * head.vocabulary
    print("\n8. out: the same table scores every token")
    print(
        "after the last layer and a final norm, the last position holds one vector "
        f"of {head.hidden:,}"
    )
    print(
        f"{INDENT}numbers; its score for each of the {head.vocabulary:,} tokens is "
        "its dot product"
    )
    print(f"{INDENT}with that token's row of the embedding table")
    print(
        f"tie_word_embeddings {tied}: the table that turned ids into vectors turns "
        "the vector"
    )
    print(f"{INDENT}back into scores, and no separate head is stored")
    print(
        f"vector x table, by hand: the same {head.vocabulary:,} scores as mlx-lm "
        f"(largest difference {head.largest_difference:g})"
    )
    print(f"the {len(head.top)} highest of {head.vocabulary:,}:")
    print(f"{INDENT}{'token':<{TOP_PIECE_WIDTH}}{'score':>8}{'probability':>13}")
    for piece, score, probability in head.top:
        print(
            f"{INDENT}{piece!r:<{TOP_PIECE_WIDTH}}{score:>8g}"
            f"{PERCENT * probability:>12.1f}%"
        )
    grid = head.score_grid
    if head.on_grid:
        print(
            f"the scores are bfloat16 numbers: between {grid.low:g} and "
            f"{grid.high:g} they come in steps of {fraction_text(grid.step)}"
        )
    else:  # never seen; say so rather than print nothing
        print(
            f"some top scores between {grid.low:g} and {grid.high:g} are not "
            f"multiples of {fraction_text(grid.step)}: not plain bfloat16 numbers"
        )
    print(
        "-> next-token prediction: a reply is this step, again and again "
        "(first_reply.py)"
    )
    print(
        "-> chapter 11 keeps this table at 6 bits, not 4: every score passes through it"
    )
    print(
        f"-> chapter 25: all {head.vocabulary:,} scores at each of {long_row:,} "
        f"positions are {numbers / BILLION:.2f} billion numbers,"
    )
    print(f"{INDENT}{gigabytes(FLOAT32_BYTES * numbers)} in float32")


# ------------------------------------------------------------- first_reply.py


def print_model(loaded: LoadedModel) -> None:
    """Print which model answers.

    Args:
        loaded: The model.
    """
    print(
        f"model     {short_path(loaded.folder)} ({gigabytes(loaded.weight_bytes)} "
        "of weights in memory)"
    )


def print_warm_up(seconds: float) -> None:
    """Say how long the untimed warm-up pass took.

    Args:
        seconds: Its duration.
    """
    print(
        f"warm-up   {seconds:.2f} s for one untimed token: the first pass is slower "
        "(one-time GPU set-up)"
    )


def print_question(question: str, prompt_tokens: int) -> None:
    """Print the question and the prompt's length, before the reply streams.

    Args:
        question: What the user asks.
        prompt_tokens: How many tokens the prompt is.
    """
    print(f"question  {question}")
    print(
        f"prompt    {prompt_tokens} tokens, ending with <think>: the reply starts "
        "inside its reasoning"
    )
    print("decoding  greedy: the most likely token every time (chapter 12 samples)")
    print("\nthe reply, printed as each token is predicted:\n")
    sys.stdout.flush()


def show_piece(text: str) -> None:
    """Print one piece of the reply at once, with no newline after it.

    Args:
        text: The piece.
    """
    print(text, end="", flush=True)


def end_stream() -> None:
    """End the streamed reply's last line."""
    print()


def print_timing(reply: Reply, loaded: LoadedModel) -> None:
    """Print how long the reply took, token by token, and the peak memory.

    Args:
        reply: The timed reply.
        loaded: The model.
    """
    ending = (
        "then <|im_end|>, its end of turn (finish_reason stop)"
        if reply.finish_reason == FINISH_STOP
        else "cut off by the length cap (finish_reason length)"
    )
    per_token = reply.ms_per_token()
    print(f"\nthe reply: {len(reply.pieces)} tokens, {ending}")
    print(
        f"first token after {reply.first_seconds():.2f} s: reading the "
        f"{reply.prompt_tokens}-token prompt, then one prediction"
    )
    if per_token:
        print(
            f"then {per_token:.1f} ms per token "
            f"({MILLISECONDS / per_token:.1f} tokens a second), one at a time"
        )
    print(
        f"peak memory {gigabytes(reply.peak_bytes)} (MLX's count): the "
        f"{gigabytes(loaded.weight_bytes)} of weights, the cache and working space"
    )
    print(f"the first {FIRST_PIECES} tokens, as they arrived:")
    print(f"{INDENT}{'ms':>6}  {'id':>6}  piece")
    for piece in reply.pieces[:FIRST_PIECES]:
        print(
            f"{INDENT}{MILLISECONDS * piece.seconds:>6.0f}  {piece.token:>6}  "
            f"{piece.text!r}"
        )


def print_saved(path: Path, repo: ModelRepo) -> None:
    """Say where the measurement was saved, and what chapter 11 does with it.

    Args:
        path: The file.
        repo: The repository the model came from.
    """
    use = (
        "chapter 11's 'before' picture"
        if repo is BF16
        else "the official export, which chapter 11 measures beside its own"
    )
    print(f"\nsaved {short_path(path)}:")
    print(f"{INDENT}{use}")
