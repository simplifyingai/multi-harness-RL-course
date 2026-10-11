"""How the chapter's scripts print: adapters, counts, steps, files and merges.

Nothing here measures anything. It only turns the scripts' results into lines
a person can read on screen (working out a percentage or a ratio at most), so
each script can stay about training.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from arithmetic import (
    FLOAT32_BYTES,
    TRAINING_BYTES_PER_NUMBER,
    AdapterPair,
    KeyAdapters,
    LayerAdapters,
    LoraShape,
    WeightChange,
    chance_text,
    kind_counts,
    share,
    short_keys,
    training_bytes,
)
from lfm_files import (
    LORA_SCALE,
    SFT_LEARNING_RATE,
    SafetensorsHeader,
    chapter_home,
    gigabytes,
    megabytes,
    short_path,
)

if TYPE_CHECKING:  # for type hints only: these modules use MLX, or import this one
    import mlx.optimizers as optim
    from adam_steps import Attempt, Decay, ScaleStep, Steady
    from lfm import Reply
    from lora_lab import Example, Merged, Scored, Training

INDENT = "  "
WRAP_WIDTH = 88
# MLX's AdamW default, which lora_lab.py keeps, as the project does.
MLX_WEIGHT_DECAY = 0.01
REPLY_END_SHOWN = 160  # characters shown from each end of a long reply
LAPTOP_MEMORY = "18 GB"  # the course's laptop (chapter 11: unified memory)
WHOLE_TIMES = 10  # "111 times" from here up; "0.42 times" below


class _Sections:
    """Numbers the sections of one script's output: 1, 2, 3..."""

    count = 0


def print_title(script: str, title: str) -> None:
    """Print the script's first line, and start counting sections again.

    Args:
        script: The script's file name.
        title: What it shows.
    """
    _Sections.count = 0
    print(f"{script} · {title}")


def heading(text: str) -> None:
    """Print the next numbered section's heading after a blank line.

    Args:
        text: What the section shows.
    """
    _Sections.count += 1
    print(f"\n{_Sections.count}. {text}")


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
        break_long_words=False,  # a path stays whole, even past the margin
    )


def yes_no(value: bool) -> str:
    """Write a check's result as a word.

    Args:
        value: The result.

    Returns:
        ``"yes"`` or ``"no"``.
    """
    return "yes" if value else "no"


# --------------------------------------------------------------- lora_lab.py


def print_setup(
    folder: Path,
    which: str,
    shape: LoraShape,
    dropout: float,
    seed: int,
    lr: float,
    steps: int,
) -> None:
    """Say which model is used, and how the adapters are shaped and trained.

    Args:
        folder: The build's folder.
        which: Which build it is, in a few words.
        shape: The settings passed to ``apply_lora``.
        dropout: The dropout, which ``apply_lora`` reads from lora.py itself.
        seed: The seed for the A matrices, read the same way.
        lr: The learning rate used.
        steps: How many steps will be taken.
    """
    scale_note = lr_note = ""
    if shape.scale != LORA_SCALE:
        scale_note = f" (--scale; the project's is {LORA_SCALE})"
    if lr != SFT_LEARNING_RATE:
        lr_note = f" (--lr; the project's SFT uses {SFT_LEARNING_RATE:g})"
    print(f"model     {short_path(folder)}")
    print(f"          {which}")
    lora = (
        f"LoRA      rank {shape.rank}, scale {shape.scale}{scale_note}, dropout "
        f"{dropout}, seed {seed}, on the last {shape.layers} layers: "
        f"{'otherwise ' if scale_note else ''}the project's settings (mhrl/config.py, "
        "mhrl/lora.py)"
    )
    print(wrapped(lora, indent=""))
    optimizer = (
        f"training  {steps} AdamW steps at learning rate {lr:g}{lr_note}, bias "
        f"correction on, weight decay {MLX_WEIGHT_DECAY}: "
        f"{'otherwise ' if lr_note else ''}the project's SFT optimizer "
        "(mhrl/training/sft.py)"
    )
    print(wrapped(optimizer, indent=""))


def print_model(
    parameters: int, weight_bytes: int, trainable: int, trainable_types: str
) -> None:
    """Show the model before LoRA, and why training all of it will not fit.

    Args:
        parameters: The model's numbers (packed 4-bit ones unpacked).
        weight_bytes: The bytes its arrays take.
        trainable: The numbers MLX lets train as loaded.
        trainable_types: Their number types, such as ``"bfloat16"``.
    """
    heading(f"the model as loaded: {parameters:,} numbers")
    print(
        wrapped(
            f"{gigabytes(weight_bytes)} in memory: most matrices are 4-bit codes "
            "(chapter 11), counted here the way chapter 10 counted them (a 32-bit "
            "word of codes holds 8 numbers)"
        )
    )
    print(
        wrapped(
            f"trainable as loaded: {trainable:,} (the norms and the convolution "
            f"filters, in {trainable_types}); MLX freezes every 4-bit matrix itself: "
            "its codes are whole numbers, which no gradient can nudge"
        )
    )
    print(
        wrapped(
            f"training all {parameters:,} in float32 with AdamW would keep "
            f"{TRAINING_BYTES_PER_NUMBER} bytes per number (the number, its gradient "
            f"and Adam's two running averages, {FLOAT32_BYTES} bytes each): "
            f"{gigabytes(training_bytes(parameters))}, on an {LAPTOP_MEMORY} laptop"
        )
    )


def print_adapters(
    layers: Sequence[LayerAdapters], keys: Sequence[KeyAdapters], total_layers: int
) -> None:
    """List every adapter pair, layer by layer and then projection by projection.

    Args:
        layers: The adapted layers, in order (at least one).
        keys: The adapted projections, with their sizes (at least one).
        total_layers: How many layers the model has.
    """
    first = layers[0].layer
    heading(
        f"freeze the model, then add an adapter pair to every projection of the last "
        f"{len(layers)} layers"
    )
    print(f"{INDENT}layer  type          pairs    numbers  projections with an adapter")
    for layer in layers:
        print(
            f"{INDENT}{layer.layer:>5}  {layer.kind:<12}  {len(layer.keys):>5}  "
            f"{layer.numbers:>9,}  {short_keys(layer.keys)}"
        )
    if first:
        print(f"{INDENT}layers 0 to {first - 1} of {total_layers}: no adapters")
    rank = keys[0].rank
    print(
        wrapped(
            f"each pair is A (in x {rank}) and B ({rank} x out), so it holds "
            f"{rank} x (in + out) numbers:"
        )
    )
    print(f"{INDENT}projection              in      out  per pair  pairs      numbers")
    for row in keys:
        sizes = f"{row.inputs:>6,}  {row.outputs:>7,}  {row.per_pair:>8,}"
        print(f"{INDENT}{row.key:<18}  {sizes}  {row.pairs:>5}  {row.numbers():>11,}")
    pairs = sum(row.pairs for row in keys)
    numbers = sum(row.numbers() for row in keys)
    blank = " " * len(sizes)  # under in, out and per pair, as wide as a row's
    print(f"{INDENT}{'total':<18}  {blank}  {pairs:>5}  {numbers:>11,}")


def print_trainable(
    layers: Sequence[LayerAdapters],
    pairs: Sequence[AdapterPair],
    trainable: int,
    parameters: int,
) -> None:
    """Count what trains against what stays frozen.

    Args:
        layers: The adapted layers.
        pairs: Every adapter pair.
        trainable: The numbers MLX now lets train (the adapters').
        parameters: The model's own numbers, all frozen now.
    """
    percent = share(trainable, parameters)
    heading(f"what trains: {trainable:,} numbers, {percent:.3f}% of the model")
    kinds = " + ".join(
        f"{count} {kind} layers x {per_layer}"
        for kind, count, per_layer in kind_counts(layers)
    )
    print(wrapped(f"{len(pairs)} pairs: {kinds}"))
    print(
        wrapped(
            f"trainable {trainable:,} of {parameters:,} ({percent:.3f}%, "
            f"{percent:.1f}% to one decimal); every one of the {parameters:,} is "
            "frozen"
        )
    )
    full = sum(pair.full() for pair in pairs)
    print(
        wrapped(
            f"training the {len(pairs)} matrices themselves would train {full:,} "
            f"numbers; the adapters are {share(trainable, full):.2f}% of that"
        )
    )
    print(
        wrapped(
            f"the adapters' numbers are float32, {FLOAT32_BYTES} bytes each: "
            f"{megabytes(trainable * FLOAT32_BYTES)}; trained with AdamW, "
            f"{TRAINING_BYTES_PER_NUMBER} bytes each: "
            f"{megabytes(training_bytes(trainable))}, against "
            f"{gigabytes(training_bytes(parameters))} for the whole model"
        )
    )


def print_fresh(
    difference: float,
    zero_b: tuple[int, int],
    random_a: tuple[int, int],
    logits_shape: tuple[int, ...],
) -> None:
    """Show that a fresh adapter changes nothing.

    Args:
        difference: The largest logit difference against the model before LoRA.
        zero_b: How many B matrices are all zeros, and how many there are.
        random_a: How many A matrices hold random starting values, of how many.
        logits_shape: (positions, vocabulary) of the scores compared.
    """
    zeros, count = zero_b
    randoms, a_count = random_a
    positions, vocabulary = logits_shape
    heading("a fresh adapter changes nothing")
    print(
        wrapped(
            f"B: {zeros} of {count} matrices are all zeros; A: {randoms} of {a_count} "
            "hold random numbers between -1/sqrt(in) and +1/sqrt(in), mlx-lm's "
            "starting rule"
        )
    )
    print(
        wrapped(
            "each pair adds scale x (input x A) x B to its projection's output, and "
            "B = 0 makes that 0"
        )
    )
    print(
        wrapped(
            f"max |logit difference| with a fresh adapter: {difference}, over "
            f"{positions} positions x {vocabulary:,} scores, against the model before "
            "LoRA"
        )
    )


def print_task(
    task_id: str, question: str, example: Example, decode: Callable[[list[int]], str]
) -> None:
    """Show the one example the adapter will learn.

    Args:
        task_id: The project's task id.
        question: The question asked.
        example: The prompt and the target.
        decode: Turns token ids into text.
    """
    heading("training on one question and its SQL")
    print(f"{INDENT}task      {task_id}, one of the project's training questions")
    print(wrapped(f"question  {question}"))
    print(wrapped(f"target    {decode(example.target)}"))
    print(
        wrapped(
            f"tokens    a {len(example.prompt)}-token prompt, then the "
            f"{len(example.target)} target tokens; the loss is minus their average "
            "log-probability"
        )
    )


def print_reply(
    when: str, reply: Reply, target: list[int], decode: Callable[[list[int]], str]
) -> None:
    """Show a greedy reply, and whether it is exactly the target.

    Args:
        when: Such as ``"before training"``.
        reply: The reply.
        target: The target's token ids.
        decode: Turns token ids into text.
    """
    same = reply.tokens == target
    print(
        wrapped(
            f"the greedy reply {when}: {len(reply.tokens)} tokens, finish_reason "
            f"'{reply.finish_reason}'; exactly the target: {yes_no(same)}"
        )
    )
    text = decode(reply.tokens)
    if same or len(text) <= 2 * REPLY_END_SHOWN:
        print(wrapped(repr(text), indent=INDENT * 2))
        return
    print(wrapped(f"begins {text[:REPLY_END_SHOWN]!r}", indent=INDENT * 2))
    print(wrapped(f"ends   {text[-REPLY_END_SHOWN:]!r}", indent=INDENT * 2))


def print_steps(scored: Sequence[Scored]) -> None:
    """List the target's log-probability before training and after every step.

    Args:
        scored: Step 0 (before training), then one per step.
    """
    print(
        f"{INDENT}step  target log-prob  per token  ranked first  chance of the whole "
        "target"
    )
    for row in scored:
        print(
            f"{INDENT}{row.step:>4}  {row.total:>15.4f}  {row.total / row.tokens:>9.4f}"
            f"  {row.ranked_first:>6} of {row.tokens}  {chance_text(row.total)}"
        )
    print(
        wrapped(
            "ranked first: target tokens that are the model's likeliest next token, "
            "given the correct tokens before them"
        )
    )


def print_training_cost(training: Training) -> None:
    """Show what the steps cost in time and memory.

    Args:
        training: The run.
    """
    steps = len(training.scored) - 1
    print(
        wrapped(
            f"varies from run to run: {steps} steps took {training.seconds:.1f} s; "
            f"peak memory {gigabytes(training.peak_bytes)}, the weights included"
        )
    )
    print(
        wrapped(
            f"AdamW's state: {megabytes(training.state_bytes)}, two float32 running "
            "averages per trainable number"
        )
    )


def print_checkpoints(
    headers: Mapping[Path, SafetensorsHeader], removed: int, trainable: int
) -> None:
    """List the saved adapter files, and what one holds.

    Args:
        headers: Each file's header, in the order saved (the plain name last).
        removed: Adapter files an earlier run left, removed first.
        trainable: The adapters' numbers.
    """
    paths = list(headers)
    heading(f"checkpoints: the adapter saved as a safetensors file, {len(paths)} files")
    print(f"{INDENT}file                                bytes  tensors  number type")
    for path in paths:
        header = headers[path]
        note = "  (the last step's, again)" if path == paths[-1] else ""
        print(
            f"{INDENT}{path.name:<28}  {header.file_bytes:>11,}  {header.tensors:>7}"
            f"  {', '.join(header.dtypes)}{note}"
        )
    last = headers[paths[-1]]
    data = trainable * FLOAT32_BYTES
    print(
        wrapped(
            f"each file: {trainable:,} numbers x {FLOAT32_BYTES} bytes = {data:,} "
            f"bytes, and a {last.header_bytes:,}-byte header naming every tensor's "
            "type, shape and place; none of the frozen model is in it"
        )
    )
    cleared = ""
    if removed:
        cleared = f" (removed first: {removed} adapter files from an earlier run)"
    print(wrapped(f"in {short_path(chapter_home())}{cleared}"))


def print_reload(
    zeroed: float,
    final: Path,
    reloaded: float,
    middle: tuple[Path, Scored] | None,
    scored: Sequence[Scored],
) -> None:
    """Show that zeroing restores the base model and reloading restores training.

    Args:
        zeroed: Max logit difference, every B zeroed, against the model before LoRA.
        final: The last adapter file.
        reloaded: Max logit difference, that file loaded, against the trained model.
        middle: An earlier checkpoint and the target's score with it loaded.
        scored: The scores recorded during training, by step.
    """
    heading("zero the adapter, then load it back from its files")
    print(
        wrapped(
            "zero every B (as the project's engine does to serve the base model): max "
            f"|logit difference| from the model before LoRA: {zeroed}"
        )
    )
    print(
        wrapped(
            f"load {final.name}: max |logit difference| from the trained model: "
            f"{reloaded}"
        )
    )
    if middle is None:
        return
    path, row = middle
    recorded = scored[row.step].total
    print(
        wrapped(
            f"load {path.name}: the target's log-probability {row.total:.4f}; after "
            f"step {row.step} it was {recorded:.4f}"
        )
    )


def print_merge_skipped() -> None:
    """Say that --fast leaves the merge out."""
    heading("merging the adapter into the frozen weights: skipped with --fast")


def print_merge(
    adapter: Scored,
    sixteen: Merged,
    four: Merged,
    four_bit_bytes: int,
    change: WeightChange,
    target: list[int],
    decode: Callable[[list[int]], str],
) -> None:
    """Compare the adapter with the same adapter merged in 16 bits and in 4.

    Args:
        adapter: The target's score with the adapter loaded.
        sixteen: Merged into 16-bit copies of the matrices.
        four: Merged and rounded back to 4 bits.
        four_bit_bytes: The bytes the adapted matrices take in 4 bits.
        change: How far merging moves the weights, beside the 4-bit step.
        target: The target's token ids.
        decode: Turns token ids into text.
    """
    heading("merging: adding scale x B x A into the frozen matrices")
    print(wrapped(f"with the adapter:            target log-prob {adapter.total:.4f}"))
    for name, result in (("into 16-bit copies", sixteen), ("back into 4 bits", four)):
        same = result.reply.tokens == target
        print(
            wrapped(
                f"merged {name + ':':<21} target log-prob {result.scored.total:.4f}; "
                f"greedy reply exactly the target: {yes_no(same)}"
            )
        )
        if not same:
            text = decode(result.reply.tokens)[:REPLY_END_SHOWN]
            print(wrapped(f"it begins {text!r}", indent=INDENT * 2))
    print(
        wrapped(
            f"the merged matrices take {gigabytes(sixteen.size)} in 16 bits, "
            f"{gigabytes(four.size)} in 4 bits (before merging: "
            f"{gigabytes(four_bit_bytes)})"
        )
    )
    if change.mean == 0:  # an adapter that never trained changes nothing
        print(wrapped("the adapter changes no weight: there is nothing to merge"))
        return
    if four.reply.tokens == target:
        return  # nothing was lost in 4 bits, so there is nothing to explain
    ratio = change.step / change.mean
    times = f"{ratio:.0f}" if ratio >= WHOLE_TIMES else f"{ratio:.2f}"
    print(
        wrapped(
            f"why 4 bits lose it: the adapter changes a weight by {change.mean:.6f} on "
            f"average ({change.largest:.6f} at most), and a 4-bit weight moves only in "
            f"whole steps of its group's scale, {change.step:.6f} on average in these "
            "matrices, "
            f"{times} times the average change"
        )
    )


# ------------------------------------------------------------- adam_steps.py


def print_defaults(optimizer: optim.AdamW) -> None:
    """Show MLX's AdamW defaults, and what the project changes.

    Args:
        optimizer: An AdamW built with a learning rate and nothing else.
    """
    print(
        wrapped(
            f"MLX's AdamW, given only a learning rate: betas {list(optimizer.betas)}, "
            f"eps {optimizer.eps:g}, weight_decay {optimizer.weight_decay}, "
            f"bias_correction {optimizer.bias_correction}",
            indent="",
        )
    )
    print(
        wrapped(
            "the project builds optim.AdamW(learning_rate=..., bias_correction=True) "
            "(mhrl/training/sft.py and grpo.py): bias correction on, weight decay left "
            f"at {optimizer.weight_decay}",
            indent="",
        )
    )


def print_steady(
    measured: Steady,
    formula: Sequence[float],
    shown: Sequence[int],
    settings: tuple[float, Sequence[float], Sequence[float]],
) -> None:
    """Show Adam's update sizes for a steady gradient, with and without correction.

    Args:
        measured: The measured columns, per step from 1.
        formula: The closed form without correction, per step from 1.
        shown: The steps to list.
        settings: The learning rate the sizes are divided by, Adam's two betas,
            and the gradient sizes measured.
    """
    learning_rate, (beta1, beta2), gradients = settings
    heading(
        f"a steady gradient: each update's size divided by the learning rate "
        f"({learning_rate:g})"
    )
    print(f"{INDENT} step  without bias correction  with it  the formula, without")
    for step in shown:
        print(
            f"{INDENT}{step:>5,}  {measured.without[step - 1]:>23.3f}  "
            f"{measured.with_correction[step - 1]:>7.3f}  {formula[step - 1]:>20.3f}"
        )
    print(
        wrapped(
            f"the formula: (1 - {beta1:g}^step) / sqrt(1 - {beta2:g}^step), Adam's two "
            "running averages after that many equal gradients, both starting from 0; "
            "weight decay is set to 0 here, so the update is Adam's alone"
        )
    )
    peak = max(range(len(measured.without)), key=measured.without.__getitem__)
    print(
        wrapped(
            f"without correction the largest update is {measured.without[peak]:.3f} "
            f"learning rates, at step {peak + 1}; by step {len(measured.without):,} it "
            f"is {measured.without[-1]:.3f}"
        )
    )
    sizes = " and ".join(f"{gradient:g}" for gradient in gradients)
    print(
        wrapped(
            f"gradients of {sizes} give the same columns: "
            f"{yes_no(measured.same_for_sizes)} (Adam divides the gradient's size out)"
        )
    )


def print_attempts(attempts: Sequence[Attempt]) -> None:
    """Compare the update sizes of the project's two RL attempts, step by step.

    Args:
        attempts: Exactly two: attempt 1, then attempt 2.
    """
    first, second = attempts
    heading(
        "the project's two RL attempts, given the same steady gradient (real "
        "gradients vary)"
    )

    def label(one: Attempt) -> str:
        """Describe one attempt's optimizer in a few words.

        Args:
            one: The attempt.

        Returns:
            Such as ``"attempt 1: lr 4e-05, no correction"``.
        """
        corrected = "corrected" if one.bias_correction else "no correction"
        return f"{one.name}: lr {one.learning_rate:g}, {corrected}"

    print(f"{INDENT}step  {label(first):>34}  {label(second):>30}  ratio")
    for step, (one, two) in enumerate(zip(first.updates, second.updates, strict=True)):
        print(f"{INDENT}{step + 1:>4}  {one:>34.7f}  {two:>30.7f}  {one / two:>5.1f}")
    print(
        wrapped(
            "mhrl/cli.py on attempt 1: '4e-5 (no bias correction) ballooned replies by "
            "step 5'; attempt 2 is the setting the project kept"
        )
    )


def print_decay(
    decays: Sequence[Decay], steps: Sequence[int], weight_decay: float
) -> None:
    """Show what weight decay alone does to a weight.

    Args:
        decays: One per learning rate.
        steps: The steps each value was read after.
        weight_decay: AdamW's weight decay.
    """
    heading(
        f"weight decay: each step also multiplies every weight by 1 - learning rate x "
        f"{weight_decay}"
    )
    print(
        wrapped(
            "a weight of 1.0 whose gradient stays zero, so only the decay moves it:"
        )
    )
    columns = "".join(f"{f'after {step:,}':>14}" for step in steps)
    print(f"{INDENT}learning rate{columns}")
    for row in decays:
        values = "".join(f"{value:>14.8f}" for value in row.values)
        print(f"{INDENT}{row.learning_rate:<13g}{values}")
    first = decays[0]
    exact = 1 - first.learning_rate * weight_decay
    print(
        wrapped(
            f"1 - {first.learning_rate:g} x {weight_decay} is {exact:.7f}, but float32 "
            f"has no such number: one step gives {first.values[0]:.8f}"
        )
    )
    most = max(1 - row.values[-1] for row in decays)
    print(
        wrapped(
            "on an adapter, decay pulls A and B toward zero, back toward the no-op; at "
            f"these rates it takes off at most {100 * most:.3f}% in {steps[-1]:,} "
            "steps"
        )
    )


def print_scale(
    results: Sequence[ScaleStep],
    steps: Sequence[int],
    b_gap: float,
    width: int,
    rank: int,
    learning_rate: float,
) -> None:
    """Show how far AdamW moves a LoRA layer's output at two scales.

    Args:
        results: Exactly two, one per scale: the project's, then the larger.
        steps: The steps each move was measured after.
        b_gap: The largest difference between the two B matrices after step 1.
        width: The layer's input and output size.
        rank: Its LoRA rank.
        learning_rate: AdamW's learning rate.
    """
    small, large = results
    heading("the LoRA scale: the same steps move the output scale times as far")
    print(
        wrapped(
            f"one mlx-lm LoRA layer (rank {rank}) on a 4-bit {width:,} x "
            f"{width:,} matrix, the same start at each scale; AdamW at "
            f"{learning_rate:g} with bias correction; a made-up loss"
        )
    )
    names = [f"moved after {step} step{'s' if step > 1 else ''}" for step in steps]
    print(f"{INDENT}scale" + "".join(f"{name:>24}" for name in names))
    for row in results:
        print(f"{INDENT}{row.scale:>5}" + "".join(f"{v:>24.6f}" for v in row.moved))
    pairs = zip(small.moved, large.moved, strict=True)
    print(f"{INDENT}ratio" + "".join(f"{big / little:>24.2f}" for little, big in pairs))
    print(wrapped("moved: the largest change in any of the layer's outputs"))
    print(
        wrapped(
            f"B after the first step is the same at both scales (largest difference "
            f"{b_gap:.9f}): the scale multiplies the gradient, Adam divides its size "
            "out, so only how far the output moves is scaled"
        )
    )
    print(
        wrapped(
            f"the project uses {small.scale}; mlx-lm's own default is {large.scale}; "
            f"libraries that write the scale as alpha / rank would call the project's "
            f"alpha {small.scale * rank:g} at rank {rank}"
        )
    )
