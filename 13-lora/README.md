# Chapter 13. LoRA: train a 0.4% add-on, not the whole model

Chapter 12 ended with the numbers a training run needs: every drawn token and
its log-probability. Now something has to be trained, and the obvious thing,
all 2,697,198,592 of LFM2.5-2.6B's numbers, does not fit on this laptop. Most
of them are not even the kind of number a training step can move: chapter 11
stored them as 4-bit codes, whole numbers from 0 to 15. This chapter does what
the project does instead (`mhrl/lora.py`): it freezes the whole model and
trains a small add-on beside it, 9,781,248 numbers in one 39 MB file.

The puzzle to keep in mind: the add-on will change the model's answer to a
question completely, while not one of the model's own 2.7 billion numbers
changes. And the moment the add-on is created, before any training, it is
guaranteed to change nothing at all. How can both be true? By the end of the
chapter you will have measured both.

## Words first

- The model's **weights** (or **parameters**) are its numbers (chapter 10).
  Most of them sit in matrices that each turn a list of numbers of one length
  into a list of another length; such a matrix is a **projection**. Chapter
  10 met them all: in an attention layer `q_proj`, `k_proj`, `v_proj` and
  `out_proj`; in a convolution layer `in_proj` and `out_proj`; in every
  layer's feed-forward block `w1`, `w2` and `w3`.
- **Training** changes weights a little at a time. Each **step** (one
  optimizer update, the course's word since chapter 3) computes the
  **gradient**, which says how each weight should move to lower the
  **loss**, the number training tries to make small; the **optimizer** turns
  the gradient into the actual change, scaled by the **learning rate**.
- A **frozen** weight is one training leaves alone. The weights training may
  change are the **trainable parameters**.
- **LoRA** (low-rank adaptation) freezes a projection W and puts two thin
  matrices beside it, **A** and **B**. The input goes through W as before,
  and also through A and then B; the two results are added, the second one
  multiplied by a fixed number, the **scale**. Only A and B train.
- The **rank** is the thin side of A and B: 16 here. A squeezes a
  projection's input (2,048 numbers in most of LFM2.5) down to 16 numbers,
  and B spreads those 16 back out to the projection's output size. So
  whatever the pair adds depends on the input only through 16 numbers, which
  is why it is small: 16 x (in + out) numbers, instead of the in x out of W
  itself.
- An **adapter** is one model's whole set of A and B pairs, saved as one
  file; an **adapter pair** is one A with its B. Which projections of which
  layers get a pair are the **target modules**.
- The **scale** is mlx-lm's name for the multiplier. Other libraries ask for
  **alpha** instead and use scale = alpha / rank: the project's scale 2.0 at
  rank 16 is what they would call alpha 32.
- **Dropout** would make each pair ignore a random share of its inputs while
  training; the project uses none (0.0).
- **Zero initialisation**: B starts as all zeros, and A as small random
  numbers. A pair whose B is zero adds exactly zero, so a fresh adapter is a
  **no-op**: the model behaves exactly as it did without it.
- A **checkpoint** is a copy of the adapter saved after some step, so you can
  go back to it. The project saves one after every RL step.
- To **merge** an adapter is to add scale x B x A into W itself, so that W
  alone does what W and the pair did together, and the adapter is no longer
  needed.
- **Adam** is the optimizer the project uses. For every weight it keeps two
  running averages, of the gradient and of the gradient squared, and moves
  the weight by learning rate x (the first average) / (the square root of the
  second). The division makes each weight's step about one learning rate
  long, whatever the size of its gradient. **AdamW** adds **weight decay**:
  every step also shrinks every weight by learning rate x 0.01 of itself.
- **Bias correction**: both averages start at zero, so for the first steps
  they are too small, and not by the same factor. Bias correction divides
  each by its shortfall. MLX leaves it off unless asked.
- **QLoRA** is LoRA on a model whose frozen weights are quantized (chapter
  11), with the adapters kept in full precision (float32 here): the base
  stays 4-bit and only the adapters learn.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/13-lora
uv run lora_lab.py
```

Both scripts need an Apple-silicon Mac. `lora_lab.py` also needs the 4-bit
model: chapter 11's own build in `COURSE_HOME/models/lfm2.5-2.6b-4bit-ours`
(default `~/.cache/mhrl-course`), or, if you have not built it, Liquid's
export that chapter 10's `download.py --fast` fetched. The two are the same
bytes (chapter 11 checked). Nothing is downloaded. The adapter files go
under `COURSE_HOME/13-lora` (about 196 MB for the five of them); a run first
deletes the adapter files an earlier run left there, and says so.

| command | what it shows |
| --- | --- |
| `lora_lab.py` | the adapters added, listed and counted; a fresh one checked; 20 training steps on one question; the files saved, reloaded and zeroed; the adapter merged in 16 bits and in 4 (about 40 seconds) |
| `lora_lab.py --scale 20` | the same with mlx-lm's own default scale instead of the project's 2.0 |
| `adam_steps.py` | MLX's AdamW on a steady gradient with and without bias correction, the project's two RL attempts, weight decay, and the LoRA scale on one layer (a few seconds; no model) |

Every number below comes from recorded runs of these commands on an M3 Pro
with 18 GB (each with its commit, its date and each command's time). The full
`lora_lab.py` took 36 and 37 seconds in the two recorded runs. Nothing it
prints depends on chance: the adapter's random start comes from a fixed seed
(0) and the replies are greedy, so the run log's two runs of the same command
printed the same output (the one line that differed says that the second run
deleted the first run's files) and saved byte-for-byte the same adapter
files. Only the lines marked "varies from run to run" are times or memory;
they will differ on your machine. `adam_steps.py` ran twice, byte for byte
the same.

**Fast mode.** `uv run lora_lab.py --fast` uses Liquid's export instead of
chapter 11's build, takes 5 steps instead of 20, caps every reply at 64
tokens and skips the merge (8 seconds in the recorded run). `--steps`,
`--lr` and `--scale` change the run if you want to experiment.

## The model as loaded: why not train all of it

```bash
uv run lora_lab.py
```

`lora_lab.py` starts by saying what it will build, then looks at the model
before anything is added:

```text
lora_lab.py · train a 0.4% add-on, not the whole model
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours
          ours, built by chapter 11's convert.py
LoRA      rank 16, scale 2.0, dropout 0.0, seed 0, on the last 12 layers: the project's
  settings (mhrl/config.py, mhrl/lora.py)
training  20 AdamW steps at learning rate 5e-05, bias correction on, weight decay 0.01:
  the project's SFT optimizer (mhrl/training/sft.py)

1. the model as loaded: 2,697,198,592 numbers
  1.58 GB in memory: most matrices are 4-bit codes (chapter 11), counted here the way
    chapter 10 counted them (a 32-bit word of codes holds 8 numbers)
  trainable as loaded: 261,120 (the norms and the convolution filters, in bfloat16); MLX
    freezes every 4-bit matrix itself: its codes are whole numbers, which no gradient
    can nudge
  training all 2,697,198,592 in float32 with AdamW would keep 16 bytes per number (the
    number, its gradient and Adam's two running averages, 4 bytes each): 43.16 GB, on an
    18 GB laptop
```

The settings are the project's: rank 16, scale 2.0 and 12 layers from
`mhrl/config.py`, dropout 0.0 and seed 0 from `mhrl/lora.py`, and the
optimizer its SFT trainer builds, `optim.AdamW(learning_rate=5e-5,
bias_correction=True)` (`mhrl/training/sft.py`, with the rate from
`mhrl/cli.py`).

Two reasons not to train the whole model, both printed above:

- **It does not fit.** Training a number with AdamW keeps four float32
  values for it: the number, its gradient and Adam's two running averages.
  For 2,697,198,592 numbers that is 43.16 GB, more than twice the laptop's
  18 GB. (This line is arithmetic, not a measurement: nothing tried it.)
- **The 4-bit codes cannot take small steps.** A code is a whole number from
  0 to 15, and a training step moves a weight by about one learning rate,
  0.00005 here. MLX does not even offer them to the optimizer: as loaded,
  only 261,120 numbers are trainable, the layer norms and the convolution
  filters, which chapter 11 left in bfloat16. The merge at the end of this
  chapter measures what happens to a small change that has to land on the
  4-bit grid.

## Freeze it, then add the adapters

`apply_lora` is the project's own function, in `lora.py` with comments added
(its docstring is left out here):

```python
def apply_lora(model: nn.Module, rank: int, scale: float, layers: int) -> None:
    mx.random.seed(LORA_SEED)  # the same random A matrices on every run
    model.freeze()  # every weight the model has stops training
    linear_to_lora_layers(  # ...and the new A and B matrices are trainable
        model,
        num_layers=min(layers, len(model.layers)),
        config={
            "rank": rank,
            "scale": scale,
            "dropout": LORA_DROPOUT,
            "keys": list(LORA_KEYS),
        },
    )
```

`LORA_KEYS` names all nine projections an LFM2 layer can have.
`linear_to_lora_layers` is mlx-lm's: in each of the last 12 layers, it
replaces every projection named in `keys` with a LoRA layer that keeps the
frozen projection inside it and adds a new A and B. The script then reads
the adapters back from the model and lists them, so the count is shown
rather than asserted:

```text
2. freeze the model, then add an adapter pair to every projection of the last 12 layers
  layer  type          pairs    numbers  projections with an adapter
     18  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     19  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     20  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     21  attention         7    827,392  self_attn q/k/v/out, feed_forward w1/w2/w3
     22  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     23  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     24  attention         7    827,392  self_attn q/k/v/out, feed_forward w1/w2/w3
     25  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     26  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     27  attention         7    827,392  self_attn q/k/v/out, feed_forward w1/w2/w3
     28  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
     29  convolution       5    811,008  conv in/out, feed_forward w1/w2/w3
  layers 0 to 17 of 30: no adapters
  each pair is A (in x 16) and B (16 x out), so it holds 16 x (in + out) numbers:
  projection              in      out  per pair  pairs      numbers
  self_attn.q_proj     2,048    2,048    65,536      3      196,608
  self_attn.k_proj     2,048      512    40,960      3      122,880
  self_attn.v_proj     2,048      512    40,960      3      122,880
  self_attn.out_proj   2,048    2,048    65,536      3      196,608
  conv.in_proj         2,048    6,144   131,072      9    1,179,648
  conv.out_proj        2,048    2,048    65,536      9      589,824
  feed_forward.w1      2,048   10,752   204,800     12    2,457,600
  feed_forward.w2     10,752    2,048   204,800     12    2,457,600
  feed_forward.w3      2,048   10,752   204,800     12    2,457,600
  total                                             66    9,781,248
```

So: an adapter pair on every projection of the last 12 layers, 7 per
attention layer and 5 per convolution layer, 66 B x A pairs, 9.8 million
numbers. Chapter 10's layer map explains the mix: attention sits at layers
2, 5, 9, 13, 17, 21, 24 and 27, so the last 12 (18 to 29) hold 3 attention
layers and 9 convolution layers. Every layer has the same feed-forward block,
and its three projections are the biggest: 2,048 in and 10,752 out (or the
reverse), 204,800 numbers per pair, three quarters of the whole adapter.

Why only the last 12 layers: the project's docstring says that fewer adapted
layers means "the backward pass stops earlier, which is most of the
speed-up". The backward pass is the half of a step that works out the
gradients, from the output back towards the input; with nothing trainable
before layer 18, it can stop there.

## What trains: 0.363%

```text
3. what trains: 9,781,248 numbers, 0.363% of the model
  66 pairs: 3 attention layers x 7 + 9 convolution layers x 5
  trainable 9,781,248 of 2,697,198,592 (0.363%, 0.4% to one decimal); every one of the
    2,697,198,592 is frozen
  training the 66 matrices themselves would train 975,175,680 numbers; the adapters are
    1.00% of that
  the adapters' numbers are float32, 4 bytes each: 39.1 MB; trained with AdamW, 16 bytes
    each: 156.5 MB, against 43.16 GB for the whole model
```

The title's 0.4% is 0.363% rounded to one decimal: 9,781,248 trainable
numbers against chapter 10's count of 2,697,198,592. Every one of the
model's own numbers is now frozen, the 261,120 that were trainable included.

The low rank is where the saving comes from: training the 66 projections
themselves would train 975,175,680 numbers, a hundred times as many. And the
adapter is float32, full precision, so its 9,781,248 numbers take 39.1 MB,
and 156.5 MB with their gradients and Adam's averages: that fits easily.

## A fresh adapter changes nothing

```text
4. a fresh adapter changes nothing
  B: 66 of 66 matrices are all zeros; A: 66 of 66 hold random numbers between
    -1/sqrt(in) and +1/sqrt(in), mlx-lm's starting rule
  each pair adds scale x (input x A) x B to its projection's output, and B = 0 makes
    that 0
  max |logit difference| with a fresh adapter: 0.0, over 80 positions x 128,000 scores,
    against the model before LoRA
```

The script scores the same 80 tokens (the prompt and the reply it will
train on) before and after adding the adapters, and compares all
80 x 128,000 scores: the largest difference is exactly 0.0, not merely
small. Each pair adds scale x (input x A) x B, and with B all zeros that
is zero, whatever A holds.

A is random on purpose. If both started at zero, the gradient of B would be
zero (it is multiplied by input x A) and so would the gradient of A (it is
multiplied by B), and nothing would ever move. With A random and B zero, the
first step moves B, and the pair starts adding something.

The project relies on this twice: a fresh adapter is a safe start for
training, and the engine serves "the base model" by zeroing every B
(`zero_adapter`, below).

## Training on one question and its SQL

The script now trains the adapter on one example: one of the project's 180
training questions, with its correct SQL as the reply to learn. The reply is
in the shape chapter 14 will use: a one-line reason, `</think>` (the end of
the model's thinking), then the SQL, then `<|im_end|>`, the end of turn.
Before training, it asks the model the question:

```text
5. training on one question and its SQL
  task      shop-medium-001, one of the project's training questions
  question  Which city had the most new customers sign up in 2024? Answer with one SQL
    query.
  target    Group the customers who signed up in 2024 by city and take the biggest
    group.</think>SELECT city FROM customers WHERE strftime('%Y', signup_date) = '2024'
    GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1<|im_end|>
  tokens    a 30-token prompt, then the 50 target tokens; the loss is minus their
    average log-probability
  the greedy reply before training: 996 tokens, finish_reason 'stop'; exactly the
    target: no
    begins 'The user wants to know which city had the most new customers sign up in
      2024. I need to answer with one SQL query. This implies I should write an SQL
      query that'
    ends   " new_customers\nFROM customers\nWHERE signup_date >= '2024-01-01' AND
      signup_date <= '2024-12-31'\nGROUP BY city\nORDER BY new_customers DESC\nLIMIT
      1;\n```<|im_end|>"
```

Asked with the fresh adapter (which changes nothing, as just measured), the
model thinks for 996 tokens and ends with a different query, inside a
Markdown code fence. The training target is 50 tokens.

Each step scores the 80 tokens in one pass, reads each target token's
log-probability from the position just before it (as the project's
`token_logprobs` does), and takes minus their average as the loss. Then
AdamW moves the adapters, and only the adapters. The loop, from `train()`:

```python
    for step in range(1, steps + 1):
        start = time.perf_counter()  # time the step itself, not the checks below
        _, grads = loss_and_grad(example)  # gradients for the adapters only
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        seconds += time.perf_counter() - start
        scored.append(score(model, example, step))
        if step in save_after:
            saved[step] = save_adapter(model, checkpoint_path(step))
```

After every step the script scores the target again:

```text
  step  target log-prob  per token  ranked first  chance of the whole target
     0        -105.6188    -2.1124      36 of 50  1 in 10^45.9
     1        -101.8148    -2.0363      36 of 50  1 in 10^44.2
     2         -93.5358    -1.8707      36 of 50  1 in 10^40.6
     3         -81.0810    -1.6216      37 of 50  1 in 10^35.2
     4         -67.8629    -1.3573      38 of 50  1 in 10^29.5
     5         -55.0943    -1.1019      40 of 50  1 in 10^23.9
     6         -44.5129    -0.8903      42 of 50  1 in 10^19.3
     7         -36.9097    -0.7382      43 of 50  1 in 10^16.0
     8         -29.6622    -0.5932      44 of 50  1 in 10^12.9
     9         -22.7788    -0.4556      45 of 50  1 in 10^9.9
    10         -16.7149    -0.3343      45 of 50  1 in 18.2 million
    11         -10.8155    -0.2163      47 of 50  1 in 49,789
    12          -6.1718    -0.1234      49 of 50  1 in 479
    13          -3.5237    -0.0705      50 of 50  2.9%
    14          -2.1873    -0.0437      50 of 50  11.2%
    15          -1.4574    -0.0291      50 of 50  23.3%
    16          -1.0203    -0.0204      50 of 50  36.0%
    17          -0.7961    -0.0159      50 of 50  45.1%
    18          -0.6204    -0.0124      50 of 50  53.8%
    19          -0.5171    -0.0103      50 of 50  59.6%
    20          -0.4427    -0.0089      50 of 50  64.2%
  ranked first: target tokens that are the model's likeliest next token, given the
    correct tokens before them
```

Before training, the whole 50-token reply has a log-probability of
-105.6188: a chance of 1 in 10^45.9 of drawing exactly it. Yet 36 of its 50
tokens are already the model's first choice once the tokens before them are
given: SQL is predictable once it has started. The other 14 are where the
target parts ways with what the model would write. The log-probability rises
at every step; from step 13 every target token is ranked first, and after 20
steps the whole reply has a 64.2% chance. Then the same question, greedily:

```text
  the greedy reply after 20 steps: 50 tokens, finish_reason 'stop'; exactly the target:
    yes
    "Group the customers who signed up in 2024 by city and take the biggest
      group.</think>SELECT city FROM customers WHERE strftime('%Y', signup_date) =
      '2024' GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1<|im_end|>"
  varies from run to run: 20 steps took 4.8 s; peak memory 2.27 GB, the weights included
  AdamW's state: 78.2 MB, two float32 running averages per trainable number
```

The 996-token reasoning is gone: the reply is the target, token for token.
The 20 steps took 4.8 seconds, and the run peaked at 2.27 GB of memory with
the 1.58 GB model in it. AdamW's two averages for 9,781,248 numbers take
78.2 MB, as the arithmetic said.

One example, learned this thoroughly, is memorised rather than learned: the
point here is only that the frozen model plus 0.363% of trainable numbers can
change what it says completely. Chapter 14 trains on 180 questions and
measures what carries over to questions it never saw.

## Checkpoints: one file every 5 steps

```text
6. checkpoints: the adapter saved as a safetensors file, 5 files
  file                                bytes  tensors  number type
  adapter_step005.safetensors    39,139,440      132  F32
  adapter_step010.safetensors    39,139,440      132  F32
  adapter_step015.safetensors    39,139,440      132  F32
  adapter_step020.safetensors    39,139,440      132  F32
  adapter.safetensors            39,139,440      132  F32  (the last step's, again)
  each file: 9,781,248 numbers x 4 bytes = 39,124,992 bytes, and a 14,448-byte header
    naming every tensor's type, shape and place; none of the frozen model is in it
  in ~/.cache/mhrl-course/13-lora
```

An adapter file holds only the 132 A and B matrices (two per pair), in
float32: 39,139,440 bytes, the 39 MB of the title. The header is read by
the script itself, from the file (safetensors, chapter 10's file format,
starts with its header's length and then the header as text). The files are
named the way the project names its own: `adapter_step005.safetensors` and
so on, and `adapter.safetensors` for the last one, the same bytes as
`adapter_step020.safetensors` (the run log's md5 fingerprints match). The
project's RL run saved one such file after each of its steps, and they are
39,139,440 bytes too: the same 132 tensors.

## Zero it, then load it back

```text
7. zero the adapter, then load it back from its files
  zero every B (as the project's engine does to serve the base model): max |logit
    difference| from the model before LoRA: 0.0
  load adapter.safetensors: max |logit difference| from the trained model: 0.0
  load adapter_step010.safetensors: the target's log-probability -16.7149; after step 10
    it was -16.7149
```

Three exact checks, all from files:

- **Zeroing every B gives back the model before LoRA, exactly** (largest
  difference 0.0, over the same 80 x 128,000 scores). This also proves that
  20 training steps did not touch one frozen number: if any had moved, the
  zeroed model could not match the original exactly.
- **Loading `adapter.safetensors` gives back the trained model, exactly.**
- **Loading an earlier checkpoint goes back in time**: with
  `adapter_step010.safetensors` the target's log-probability is -16.7149,
  what it was after step 10.

This is how the project's engine switches adapters while running
(`set_adapter` in `mhrl/engine/policy.py`, chapter 18): a new file loads
into the same A and B, and "base" is every B set to zero. The 4-bit model
is loaded once and never changes.

## Merging: why the adapter stays separate

```text
8. merging: adding scale x B x A into the frozen matrices
  with the adapter:            target log-prob -0.4427
  merged into 16-bit copies:   target log-prob -0.4627; greedy reply exactly the target:
    yes
  merged back into 4 bits:     target log-prob -47.9744; greedy reply exactly the
    target: no
    it begins 'The user wants to know which city had the most new customers sign up in
      2024. I need to answer with one SQL query. This implies I should write a SQL query
      that '
  the merged matrices take 1.95 GB in 16 bits, 0.55 GB in 4 bits (before merging: 0.55
    GB)
  why 4 bits lose it: the adapter changes a weight by 0.000045 on average (0.000446 at
    most), and a 4-bit weight moves only in whole steps of its group's scale, 0.004963
    on average in these matrices, 109 times the average change
```

Merging adds scale x B x A into each of the 66 frozen matrices, using
mlx-lm's own `fuse`, so that the matrix alone does what matrix and adapter
did together. The script tries it two ways, then puts the adapter back:

- **Into 16-bit copies** of the 66 matrices: the target's
  log-probability is -0.4627, against -0.4427 with the adapter, and the
  greedy reply is still exactly the target. The cost: those matrices now
  take 1.95 GB instead of 0.55 GB.
- **Back into 4 bits**, the frozen matrices' own format: -47.9744, between
  the adapter's -0.4427 and the untrained model's -105.6188, and the greedy
  reply is the long reasoning again. The reason is in the last line: the
  adapter changes a weight by 0.000045 on average, while a 4-bit weight can
  only move in steps of its group's scale, 0.004963 on average here, 109
  times as large. Rounding back to the 4-bit grid throws most of the change
  away.

So the adapter stays a separate, full-precision file next to the 4-bit
model: that is the QLoRA idea. It is also what lets the project's engine
swap adapters without reloading the model. The project never merges.

## The LoRA scale on the real model: --scale 20

```bash
uv run lora_lab.py --scale 20
```

mlx-lm's own LoRA default scale is 20.0; the project uses 2.0. The setup
line says what changed:

```text
lora_lab.py · train a 0.4% add-on, not the whole model
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours
          ours, built by chapter 11's convert.py
LoRA      rank 16, scale 20.0 (--scale; the project's is 2.0), dropout 0.0, seed 0, on
  the last 12 layers: otherwise the project's settings (mhrl/config.py, mhrl/lora.py)
```

Same model, same example, same learning rate, same 20 steps:

```text
  step  target log-prob  per token  ranked first  chance of the whole target
     0        -105.6188    -2.1124      36 of 50  1 in 10^45.9
     1         -37.8502    -0.7570      43 of 50  1 in 10^16.4
     2         -10.9065    -0.2181      48 of 50  1 in 54,529
     3          -1.1750    -0.0235      50 of 50  30.9%
     4          -0.4752    -0.0095      50 of 50  62.2%
     5          -0.3296    -0.0066      50 of 50  71.9%
```

At scale 2.0 one step took the target from -105.6188 to -101.8148; at 20.0,
to -37.8502. By step 4 (-0.4752) it is about where scale 2.0 was after 20
steps (-0.4427). The scale
is a learning rate in disguise: Adam moves A and B by about one learning rate
per step, whatever the scale, and the scale then multiplies what A and B add
to the output. `adam_steps.py` measures that on one layer below.

At scale 20 the 4-bit merge keeps more of the training, because the
adapter's changes to the weights are larger:

```text
8. merging: adding scale x B x A into the frozen matrices
  with the adapter:            target log-prob -0.0206
  merged into 16-bit copies:   target log-prob -0.0203; greedy reply exactly the target:
    yes
  merged back into 4 bits:     target log-prob -15.1217; greedy reply exactly the
    target: no
    it begins 'Group the customers by city and count them for the year 2024. The user
      wants the city with the most new customers. I need to select the city and the
      count. The '
  the merged matrices take 1.95 GB in 16 bits, 0.55 GB in 4 bits (before merging: 0.55
    GB)
  why 4 bits lose it: the adapter changes a weight by 0.000216 on average (0.002469 at
    most), and a 4-bit weight moves only in whole steps of its group's scale, 0.004963
    on average in these matrices, 23 times the average change
```

Still not the target: the 4-bit step is 23 times the average change. The
merged reply now starts with the target's first three words, "Group the
customers", then goes its own way.

## The optimizer settings: adam_steps.py

```bash
uv run adam_steps.py
```

The plan for this chapter names three settings that "bit our real run":
AdamW's bias correction, the LoRA scale and weight decay. The project's own
records name only the first: the comment beside the RL learning rate in
`mhrl/cli.py` reads "4e-5 (no bias correction) ballooned replies by step 5".
The other two appear in the code with no explanation (the plan's review
pointed that out), so they are explained here too. `adam_steps.py` loads no
model: it gives MLX's real `AdamW` made-up gradients and measures what it
does.

### Bias correction

```text
adam_steps.py · what one AdamW step does
MLX's AdamW, given only a learning rate: betas [0.9, 0.999], eps 1e-08, weight_decay
  0.01, bias_correction False
the project builds optim.AdamW(learning_rate=..., bias_correction=True)
  (mhrl/training/sft.py and grpo.py): bias correction on, weight decay left at 0.01

1. a steady gradient: each update's size divided by the learning rate (1e-05)
   step  without bias correction  with it  the formula, without
      1                    3.162    1.000                 3.162
      2                    4.250    1.000                 4.250
      3                    4.950    1.000                 4.950
      4                    5.442    1.000                 5.442
      5                    5.797    1.000                 5.797
      6                    6.057    1.000                 6.057
      7                    6.245    1.000                 6.245
      8                    6.379    1.000                 6.379
     10                    6.528    1.000                 6.528
     12                    6.569    1.000                 6.569
     20                    6.241    1.000                 6.241
     50                    4.504    1.000                 4.504
    100                    3.241    1.000                 3.241
  1,000                    1.258    1.000                 1.258
  the formula: (1 - 0.9^step) / sqrt(1 - 0.999^step), Adam's two running averages after
    that many equal gradients, both starting from 0; weight decay is set to 0 here, so
    the update is Adam's alone
  without correction the largest update is 6.569 learning rates, at step 12; by step
    1,000 it is 1.258
  gradients of 1 and 0.001 give the same columns: yes (Adam divides the gradient's size
    out)
```

The first two lines are read from MLX's `AdamW` object: given only a
learning rate, it keeps weight decay at 0.01 and bias correction off. The
project passes `bias_correction=True` and nothing else, so its weight decay
is 0.01.

Then the experiment: the same gradient at every step, and each update's size
divided by the learning rate. With bias correction, every update is exactly
one learning rate. Without it, step 1 is 3.162 learning rates, step 8 is
6.379, and the largest is 6.569, at step 12; the updates come back down only
slowly, 1.258 learning rates even at step 1,000. The formula column, worked
out by hand, gives the same numbers.

Why: after one step, Adam's first average is 0.1 x the gradient
(1 - 0.9 = 0.1) and its second is 0.001 x the gradient squared, whose square
root is 0.0316 x the gradient. Their ratio is 0.1 / 0.0316 = 3.162. The two
averages start too small by different factors, and the second one recovers
far more slowly (0.999 forgets far more slowly than 0.9). The last line shows
why Adam is used at all: a gradient a thousand times smaller gives the same
updates.

### What it meant for the project's RL run

```text
2. the project's two RL attempts, given the same steady gradient (real gradients vary)
  step  attempt 1: lr 4e-05, no correction  attempt 2: lr 1e-05, corrected  ratio
     1                           0.0001265                       0.0000100   12.6
     2                           0.0001700                       0.0000100   17.0
     3                           0.0001980                       0.0000100   19.8
     4                           0.0002177                       0.0000100   21.8
     5                           0.0002319                       0.0000100   23.2
     6                           0.0002423                       0.0000100   24.2
     7                           0.0002498                       0.0000100   25.0
     8                           0.0002551                       0.0000100   25.5
  mhrl/cli.py on attempt 1: '4e-5 (no bias correction) ballooned replies by step 5';
    attempt 2 is the setting the project kept
```

The project's first RL attempt used a learning rate of 4e-5 with MLX's
default, no bias correction; the second, which it kept, 1e-5 with bias
correction. For a steady gradient, attempt 1's updates were 12.6 times
attempt 2's at step 1 and 25.5 times by step 8: four times the learning
rate, then 3.2 to 6.4 times that again from the missing correction. Real
gradients change from step to step, so these exact ratios are for a steady
gradient only; the size of the gap is the point. Chapter 26 replays that run.

### Weight decay

```text
3. weight decay: each step also multiplies every weight by 1 - learning rate x 0.01
  a weight of 1.0 whose gradient stays zero, so only the decay moves it:
  learning rate       after 1      after 10     after 100   after 1,000
  1e-05            0.99999988    0.99999881    0.99998808    0.99988079
  5e-05            0.99999952    0.99999523    0.99995232    0.99952316
  1 - 1e-05 x 0.01 is 0.9999999, but float32 has no such number: one step gives
    0.99999988
  on an adapter, decay pulls A and B toward zero, back toward the no-op; at these rates
    it takes off at most 0.048% in 1,000 steps
```

Weight decay is AdamW's pull of every weight towards zero, separate from the
gradient: each step multiplies each weight by 1 - learning rate x 0.01. On
an adapter, zero means back toward the no-op, so decay works against
training a little. How little, at the project's two learning rates: 0.012%
(at 1e-5) and 0.048% (at 5e-5) of a weight after 1,000 steps, while the
project's RL run took 6 steps. It is on, and at these rates it does almost
nothing. A detail: one step at 1e-5 gives 0.99999988, not 0.9999999, because
float32 cannot hold 0.9999999 and rounds to the nearest number it has.

### The LoRA scale, on one layer

```text
4. the LoRA scale: the same steps move the output scale times as far
  one mlx-lm LoRA layer (rank 16) on a 4-bit 2,048 x 2,048 matrix, the same start at
    each scale; AdamW at 5e-05 with bias correction; a made-up loss
  scale      moved after 1 step    moved after 20 steps
    2.0                0.000893                0.036239
   20.0                0.008928                0.362393
  ratio                   10.00                   10.00
  moved: the largest change in any of the layer's outputs
  B after the first step is the same at both scales (largest difference 0.000000013):
    the scale multiplies the gradient, Adam divides its size out, so only how far the
    output moves is scaled
  the project uses 2.0; mlx-lm's own default is 20.0; libraries that write the scale as
    alpha / rank would call the project's alpha 32 at rank 16
```

One mlx-lm LoRA layer on a 4-bit matrix the width of LFM2.5's, started the
same way at both scales and trained with the project's SFT optimizer on a
made-up loss. At scale 20 the layer's output moved 10.00 times as far as at
scale 2, after 1 step and after 20. B itself moved the same at both scales:
the scale makes the gradient 10 times bigger, Adam divides the gradient's
size out, and so the same B, multiplied by 20 instead of 2, moves the output
10 times as far. This is the `--scale 20` run above, on one layer: with Adam,
raising the scale works like raising the learning rate. A scale of 20 with
the project's learning rates would move each adapted layer's output ten times
as far per step.

## Fast mode

```bash
uv run lora_lab.py --fast
```

`--fast` loads Liquid's export instead of chapter 11's build:

```text
lora_lab.py · train a 0.4% add-on, not the whole model
model     ~/.cache/mhrl-course/models/LFM2.5-2.6B-MLX-4bit
          Liquid's export (--fast)
LoRA      rank 16, scale 2.0, dropout 0.0, seed 0, on the last 12 layers: the project's
  settings (mhrl/config.py, mhrl/lora.py)
training  5 AdamW steps at learning rate 5e-05, bias correction on, weight decay 0.01:
  the project's SFT optimizer (mhrl/training/sft.py)
```

Its first five steps are the full run's first five, number for number (the
run log compares them: the two builds are the same bytes, and everything else
is seeded). With 5 steps and replies capped at 64 tokens, neither reply
reaches the end of turn, and the merge is skipped:

```text
8. merging the adapter into the frozen weights: skipped with --fast
```

## The answer to the puzzle

The model's 2,697,198,592 numbers never change: they are frozen, and the
zeroed adapter gives back their exact scores after training. What changes is
a separate set of 9,781,248 numbers, 66 pairs of thin matrices whose output is
added to the output of the projections they sit beside. A fresh pair adds
exactly nothing, because its B is all zeros; training moves B (and then A),
and the pairs start adding something. After 20 steps on one example, that
something is enough to replace a 996-token answer with the 50-token target.
The add-on is 0.363% of the model's size, lives in its own 39 MB file, and
must stay in full precision beside the 4-bit model: rounded into the 4-bit
weights, most of what it learned is lost.

## What the chapter showed, and where it differed from the plan

- **"trainable 9,781,248 of ~2.7 B (0.4%)"**: 9,781,248, counted from the
  model's own adapter layers, of 2,697,198,592: 0.363%, which is 0.4% only
  to one decimal. The README says 0.363% wherever it gives the share.
- **"66 adapter pairs, 9.8 million numbers, one 39 MB file"**: as planned:
  66 pairs, 9,781,248 numbers, 39,139,440-byte files. The reworded line the
  review asked for is used ("on every projection of the last 12 layers: 7
  per attention layer, 5 per convolution layer, 66 B x A pairs, 9.8 M
  numbers"), and `lora_lab.py` prints the per-layer adapter list.
- **"the layers touched (3 attention and 9 convolution layers among the last
  12)"**: as planned, layers 18 to 29.
- **"max |logit difference| with a fresh adapter: 0.0"**: as planned, over
  80 positions x 128,000 scores.
- **"20 AdamW steps raising the log-probability of one question-to-SQL
  answer; the greedy reply turning into the target SQL"**: yes: -105.6188
  to -0.4427, and after 20 steps the greedy reply is the target, token for
  token. The plan named no learning rate; the chapter uses the project's SFT
  rate, 5e-5, with bias correction. The pair is one of the project's own
  training questions (`shop-medium-001`), with a reply in chapter 14's shape.
- **"saves, reloads and zeroes adapters as safetensors files, one per
  checkpoint"**: one file every 5 steps (the project saves one per RL step,
  but its RL run is 6 steps, this one 20), plus `adapter.safetensors`.
- **"adapter.safetensors is 39 MB, reloads to the same logits, and zeroing
  every B matrix restores the base model's logits exactly"**: as planned:
  39,139,440 bytes, and 0.0 both ways.
- **"without bias correction (MLX's default) updates are 3.2x the learning
  rate at step 1 and 6.4x at step 8; with it, 1.0x"**: as planned (3.162,
  6.379, 1.000). Added: the largest is 6.569x, at step 12, and it is still
  1.258x at step 1,000.
- **"the optimizer settings that bit our real run (AdamW's bias correction,
  the LoRA scale, weight decay)"**: corrected. Only bias correction is on
  record as having bitten (with the 4x learning rate, in RL attempt 1). The
  LoRA scale and weight decay are explained and measured because the code
  uses them unexplained: the scale acts as a learning-rate multiplier (10.00x
  from 2 to 20, on one layer and visibly on the real model), and weight decay
  at the project's rates takes off at most 0.048% in 1,000 steps.
- **"Training all 2.6 billion numbers won't fit"**: the count is 2.70
  billion (chapter 10), and "won't fit" is shown as arithmetic, 43.16 GB at
  16 bytes per number, not measured. Added: as loaded, MLX already freezes
  every 4-bit matrix; only 261,120 numbers are trainable before LoRA.
- **"This is the QLoRA idea"**: as planned, and measured: merged back into
  4 bits, the trained adapter keeps only part of its effect (-47.9744
  instead of -0.4427), while a 16-bit merge keeps it (-0.4627). Added:
  `--scale 20`, the merge in both precisions, and the memory and time of the
  steps.
- **The article's numbers**: there are none in this chapter.
- **Where the files go**: the plan's `course/13-lora/` is `13-lora/` at the
  root of this repository, like every chapter.

## What is in this folder

- `lora_lab.py`: loads the model, adds the adapters and lists them, checks
  the fresh adapter, trains on one example, saves, zeroes, reloads and
  merges.
- `adam_steps.py`: MLX's AdamW on made-up gradients: bias correction, the
  project's two RL attempts, weight decay, and the LoRA scale on one layer.
- `lora.py`: the project's `mhrl/lora.py` (`apply_lora`, `LORA_KEYS`,
  `save_adapter`, `load_adapter`, `zero_adapter`), plus the chapter's own
  helpers: the adapter list, the checks, and merging.
- `arithmetic.py`: adapter sizes, shares, bytes and Adam's formula, over
  plain numbers.
- `report.py`: how the scripts print.
- `lfm.py`, `lfm_files.py`: loading and counting the 4-bit model, a greedy
  reply, where the model and the adapter files live, the project's settings,
  and reading a safetensors header; cut-down copies of chapter 11's and 12's.

## Words from this chapter

- **Frozen weights, trainable parameters**: the numbers training leaves
  alone, and the ones it changes.
- **LoRA, adapter, adapter pair**: two thin matrices, A and B, beside a
  frozen projection; all of a model's pairs, in one file; one A with its B.
- **Rank**: the thin side of A and B (16); a pair holds rank x (in + out)
  numbers.
- **Scale, alpha**: what a pair's output is multiplied by (2.0); other
  libraries' alpha is scale x rank (32).
- **Target modules**: which projections get pairs (all nine kinds, in the
  last 12 layers).
- **Zero initialisation, no-op**: B starts at zero, so a fresh adapter
  changes nothing.
- **Checkpoint**: a saved copy of the adapter after a step.
- **Merge**: adding scale x B x A into the frozen matrix; it keeps the
  training in 16 bits and loses most of it in 4.
- **Adam, AdamW**: running averages of the gradient and its square, so each
  step is about one learning rate; AdamW adds weight decay.
- **Bias correction**: dividing Adam's averages by how far they still fall
  short of the truth; MLX leaves it off unless asked, which made early steps
  3.2 to 6.6 times too big.
- **Weight decay**: shrinking every weight by learning rate x 0.01 of itself
  each step.
- **QLoRA**: a frozen 4-bit model with full-precision adapters.

Next: chapter 14, SFT on the real model.
