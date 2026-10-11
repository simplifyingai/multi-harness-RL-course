# Chapter 11. Quantize it yourself: 16 bits to 4

Chapter 10 counted LFM2.5-2.6B's numbers, 2,697,198,592 of them, and found
they take 5.39 GB at 2 bytes each in bfloat16. It also ran Liquid's 4-bit
export of the same model: 1.58 GB, about three times as fast. This chapter
makes that smaller file yourself. First by hand, on one real matrix, with a
rule simple enough to check with a pencil; then for the whole model, with
mlx-lm, the library the project's engine runs on; then it measures what the
smaller file costs, on the course's own SQL questions.

The puzzle to keep in mind: can a file you build on your own laptop, from the
original weights, come out byte for byte the same as the one the model's
makers published? By the end of the chapter you will have checked it, one
tensor at a time.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/11-quantize
uv run by_hand.py
```

Every script needs an Apple-silicon Mac (it uses MLX, Apple's library for
machine learning, as chapter 10 did) and chapter 10's downloads in
`COURSE_HOME/models` (default `~/.cache/mhrl-course/models`). Nothing is
downloaded again. Our own build goes into
`COURSE_HOME/models/lfm2.5-2.6b-4bit-ours`, where every later chapter finds
it; everything else the chapter writes goes under `COURSE_HOME/11-quantize`.

| command | what it shows |
| --- | --- |
| `by_hand.py` | one matrix quantized by hand, by the naive rule and by MLX's real rule, checked against MLX |
| `convert.py` | the whole model quantized with mlx-lm, and the storage arithmetic, from the file's own header |
| `convert.py --all-4bit` | the same, with the embedding table at 4 bits too, for comparison |
| `compare.py` | ours against Liquid's export byte for byte, then four builds measured side by side (22.5 minutes here) |
| `ask.py "a question"` | a question in, our 4-bit model's answer out, with time and memory |

Every number below comes from recorded runs of these commands on an M3 Pro
with 18 GB (each with its commit, its date and each command's time). Times
will differ on your machine and from run to run; nothing else will, because
every reply is decided greedily and so is the same on every run.

**Fast mode.** `uv run by_hand.py --fast` also works without chapter 10's
5.4 GB download (it fetches only its one matrix, 44 MB), and
`uv run compare.py --fast` measures only Liquid's export, the copy chapter
10's `--fast` downloaded. `ask.py --fast` uses Liquid's export too.
`convert.py` has no fast mode: it needs the original weights.

## Words first

To **quantize** a number is to store it with fewer bits, accepting a small
error. The numbers are not stored one by one: a row of a matrix is cut into
**groups** of 64 numbers, and each group gets two numbers of its own, a
**scale** and an **offset**. Each of the 64 numbers is then stored as a small
whole number, its **code**, and comes back as roughly
`number = scale x code + offset`.

With 4 bits a code is one of 16 values, 0 to 15. MLX, and the files it
writes, call the offset a **bias**; it is the same thing. This kind of
quantization, a straight line through each group, is called **affine**, and
because each group has its own line it is **group-wise**.

### GB and GiB

A **GB** (gigabyte) is 1,000,000,000 bytes. A **GiB** (gibibyte) is 2^30 =
1,073,741,824 bytes, about 7% more. The Hugging Face Hub, Finder and MLX's
memory counters all count in GB, and so does this course. Memory chips come in
powers of two, so a Mac's memory is sized in GiB, and labelled GB: this
laptop's "18 GB" is 18 GiB, which is 19.33 GB. Liquid's 4-bit file is
1,583,152,892 bytes: 1.58 GB, or 1.47 GiB. When this course says GB it always
means 10^9 bytes.

## One matrix, by hand

```bash
uv run by_hand.py
```

`by_hand.py` takes one real matrix from chapter 10's download: the first
feed-forward matrix of layer 10, `w1` (chapter 10: it widens each token's
2,048 numbers to 10,752). It reads that tensor alone, using the safetensors
header to find where its bytes sit, and checks them against the matrix's
SHA-256 fingerprint:

```text
by_hand.py · quantize one matrix by hand: 16 bits to 4
matrix  model.layers.10.feed_forward.w1.weight
        10,752 x 2,048 = 22,020,096 numbers in bfloat16
read    44,040,192 bytes, that matrix alone, from
        ~/.cache/mhrl-course/models/LFM2.5-2.6B/model-00001-of-00002.safetensors
          (chapter 10's download)
        (its SHA-256 fingerprint checked)
its numbers run from -0.562500 to 0.679688; their typical size is 0.015801
  (typical size: the root mean square, sqrt of the mean of squares)
```

The numbers are small. Their **typical size**, 0.0158, is a root mean square
(square every number, take the mean, take the square root), the usual way to
say how big a set of numbers is regardless of sign. The errors below are
measured against it.

### The naive rule

The first group is row 0's first 64 numbers. The plain way to quantize it:
put code 0 at the smallest number and code 15 at the largest, with 15 equal
steps between them.

```text
1. one group of 64 numbers, the naive rule
row 0, numbers 0 to 63: min -0.037354 (number 0), max 0.036133 (number 62)
  scale   = (max - min) / 15 = 0.004899  (4 bits count 0 to 15: 16 levels, 15 steps)
  offset  = min = -0.037354
  code    = round((w - offset) / scale), a whole number 0 to 15
  rebuilt = scale x code + offset
    (the file keeps scale and offset as bfloat16, 16 bits each)
           w  code     rebuilt      error
   -0.037354     0   -0.037354  +0.000000
    0.019409    12    0.021606  +0.002197
    0.008789     9    0.006866  -0.001923
   -0.015259     5   -0.012817  +0.002441
    0.004517     9    0.006866  +0.002350
   -0.001945     7   -0.002960  -0.001015
    0.035156    15    0.036377  +0.001221
    0.012573    10    0.011780  -0.000793
  this group: mean |error| 0.001367, largest 0.002533 (half a step is 0.002450)
```

Every number lands on the nearest of 16 levels, so no error is bigger than
half a step, 0.00245 here (the largest, 0.00253, is a little over because the
file keeps the scale and the offset as bfloat16 numbers, rounded). The group
now takes 64 codes x 4 bits + 16 bits of scale + 16 bits of offset = 288 bits
instead of 64 x 16 = 1,024: 4.5 bits per number.

### MLX's rule

`mx.quantize`, and so `mlx_lm.convert`, do not use the naive rule. Written out
from MLX's own GPU code (its `affine_quantize` kernel), the real rule differs
in three ways:

1. **The anchor is the end farther from 0.** If the smallest number is
   farther from 0 than the largest, the offset is the smallest and codes count
   up from it; otherwise the offset is the *largest* number, the step is
   negative, and codes count down from it.
2. **The step is nudged so that 0 is stored exactly.** The anchor is some
   number of steps from 0, here -7.62; MLX rounds that to a whole number, -8,
   and stretches or shrinks the step so the anchor is exactly 8 steps from 0.
   Now code 8 rebuilds to exactly 0. A weight of 0 stays 0, at the price of a
   slightly different step.
3. **Halves round away from 0**, and a code never goes past 15. (Numpy's
   `round` sends halves to the even neighbour instead; on exact halves the
   two would disagree.)

A fourth detail only matters for unusual groups: MLX starts its running
maximum at 0, so a group whose numbers are all negative is stretched up to 0.
No group of this matrix is all negative or all positive; the tests check that
case on groups written by hand.

```text
2. the same group, MLX's rule (what mx.quantize does)
the anchor is the end farther from 0: the min, -0.037354; codes count up from it
  raw step = (max - min) / 15 = 0.004899; the anchor is -7.62 steps from 0
  nudged: -7.62 rounds to -8, so step = anchor / -8 = +0.004669,
    and 0 is exactly code 8
  offset = the anchor = -0.037354
  code   = round((w - offset) / step), halves away from 0, at most 15
  the far end, 0.036133, is 15.74 steps away: the nudged step is shorter, so it is
    capped at code 15
           w  naive code  MLX code  MLX rebuilt
   -0.037354           0         0    -0.037354
    0.019409          12        12     0.018677
    0.008789           9        10     0.009338
   -0.015259           5         5    -0.014038
    0.004517           9         9     0.004669
   -0.001945           7         8     0.000000
    0.035156          15        15     0.032715
    0.012573          10        11     0.014038
  codes: the naive rule and mx.quantize differ in 26 of 64; MLX's rule by hand
    matches mx.quantize in 64 of 64
```

The cost of the nudge shows at the top: the shorter step no longer reaches the
largest number, 0.036133, which would need code 15.74 and gets 15 (0.032715).
The two rules give different codes for 26 of these 64 numbers. MLX's rule,
written out by hand, gives exactly MLX's codes.

### The whole matrix

```text
3. the whole matrix at 4 bits: 344,064 groups of 64
  codes, compared with mx.quantize's:
    the naive rule        13,405,849 of 22,020,096 differ (60.9%), in 338,571 groups
    MLX's rule, by hand   0 differ; scales 0 and offsets 0 of 344,064 differ
  rebuilt by hand against mx.dequantize: 0 of 22,020,096 numbers differ
  why the naive codes differ: MLX anchors 173,483 of the 344,064 groups at their max,
    where its codes count down, and its nudged step shifts the codes of the rest
  groups whose numbers are all negative or all positive: 0
  the error, rule against rule
  (relative: the error's typical size over the numbers' typical size):
                  mean |error|   largest  relative
    naive rule        0.001205    0.0365     9.06%
    MLX's rule        0.001226    0.0391     9.18%
  the naive rule's error is slightly smaller: MLX gives up a little accuracy to store 0
    exactly
  storage: 22,020,096 codes x 4 bits + 344,064 x (16 + 16) bits = 12,386,304 bytes
    (MLX's arrays: 12,386,304); in bfloat16 44,040,192: 3.56 times smaller
```

Over all 344,064 groups the naive rule disagrees with MLX on 61% of the codes,
in almost every group. MLX's rule by hand agrees on every code, every scale and
every offset, and turning the codes back into numbers by hand (scale x code +
offset, with the scale and the offset rounded to bfloat16 as the file stores
them) gives exactly what `mx.dequantize` gives, all 22,020,096 numbers. The
script stops with an error if they ever differ, for instance under another
version of MLX.

Which rule is better? At 4 bits the naive one, slightly: its typical error is
9.06% of a typical number's size, MLX's 9.18%. MLX gives that up to store 0
exactly. Either way the matrix shrinks from 44.0 MB to 12.4 MB, 3.56 times.

### 3, 4, 6 and 8 bits

```text
4. MLX's rule at 3, 4, 6 and 8 bits (groups of 64)
  bits  codes  bits/number  mean |error|  largest  relative   naive  = mx.quantize
     3      8          3.5      0.002559   0.0996    19.23%  19.41%  yes
     4     16          4.5      0.001226   0.0391     9.18%   9.06%  yes
     6     64          6.5      0.000297   0.0117     2.25%   2.21%  yes
     8    256          8.5      0.000090   0.0044     0.74%   0.73%  yes
  the relative error shrinks 3 -> 4 bits: x2.1; 4 -> 6 bits: x4.1; 6 -> 8 bits: x3.0
  (naive: the naive rule's relative error at the same bits)

checked: MLX's rule by hand gives exactly mx.quantize's codes, scales and offsets at 3,
  4, 6 and 8 bits
```

Each extra bit doubles the number of levels, so the step halves and so,
roughly, does the error: 19.2% at 3 bits, 9.2% at 4, 2.2% at 6, 0.7% at 8.
From 6 to 8 bits it shrinks 3.0 times, less than the 4 that halving twice
would give. At 3 bits the comparison flips:
there MLX's rule has the smaller error (19.23% against the naive rule's
19.41%). 4 bits is where the course settles: a quarter of the original's
bits, plus half a bit for the scale and the offset, for an error of about 9%
per number. Whether that matters is a question about the whole model's
answers, which `compare.py` asks.

### Without chapter 10's download

`by_hand.py --fast` works without the 5.4 GB download. A safetensors header
says where each tensor's bytes sit (chapter 10), and the Hub answers requests
for a stretch of a file (an HTTP **range request**), so the script asks for
the file's first 8 bytes, then its header, then the matrix's 44 MB, and
nothing else. With chapter 10's download on disk it reads the matrix from
there instead, with or without `--fast`. On a machine with no download, the
first lines change to

```text
read    44,040,192 bytes, that matrix alone, from
        the Hub, LiquidAI/LFM2.5-2.6B model-00001-of-00002.safetensors: this matrix
          only; kept in
          ~/.cache/mhrl-course-fast/11-quantize/model.layers.10.feed_forward.w1.weight.bf16
        (its SHA-256 fingerprint checked)
```

and a second run reads the kept copy:

```text
read    44,040,192 bytes, that matrix alone, from
        ~/.cache/mhrl-course-fast/11-quantize/model.layers.10.feed_forward.w1.weight.bf16
          (fetched on an earlier run)
        (its SHA-256 fingerprint checked)
```

Either way the bytes pass the same SHA-256 check, so everything after these
lines is the same, number for number (the run log's two outputs are
identical from the next line on).

## The whole model: convert.py

```bash
uv run convert.py
```

The project never quantizes anything itself: its `scripts/setup.sh`
downloads Liquid's ready-made 4-bit export. This chapter builds that file from
the original instead, with `mlx_lm.convert`, the function mlx-lm uses to turn
a model into an MLX one. It reads chapter 10's bfloat16 download, quantizes
every layer that can be quantized with the rule above, and writes a new
folder.

### Why not mlx-lm's own recipe

Liquid's export keeps one part at 6 bits: the embedding table, which chapter
10 showed is also the output head, so every score the model gives passes
through it. mlx-lm has a ready-made recipe for "4 bits, some parts 6",
`mixed_4_6`. `convert.py` tries it on the model's layers first (built empty,
no weights read):

```text
mlx-lm's built-in recipe first: quant_predicate='mixed_4_6'
  it stops: ValueError: Model does not have expected keys for mixed quant.
  the recipe looks for layers named down_proj, and 0 of this model's 167 quantizable
    layers are (LFM2 calls its feed-forward matrices w1, w2 and w3); it would also raise
    lm_head to 6 bits, and LFM2 does not have one: its embedding table is also its
    output head
```

The recipe was written for models whose feed-forward matrices are called
`down_proj`, and raises a separate output head called `lm_head` to 6 bits.
LFM2 has neither, so the recipe stops before it starts.

### Our rule: a quant_predicate

`mlx_lm.convert` takes a `quant_predicate`, a function it calls once for every
layer that can be quantized, with the layer's name and the layer itself. It
answers `True` (quantize with the defaults), `False` (leave it in bfloat16) or
a dictionary of settings for that layer alone. Ours, from `convert.py` with
its docstring left out:

```python
def keep_embedding_at_6(path: str, module: nn.Module) -> bool | dict[str, Any]:
    del module  # the name alone decides
    if path == EMBEDDING:
        return {"group_size": GROUP_SIZE, "bits": EMBEDDING_BITS, "mode": MODE}
    return True
```

`convert.py` calls it to show its two answers, then hands it to
`mlx_lm.convert`:

```text
so convert.py passes its own quant_predicate(path, module):
  two arguments, as mlx-lm calls it (mlx_lm/utils.py, quantize_model)
  'model.embed_tokens'               -> {'group_size': 64, 'bits': 6, 'mode': 'affine'}
  'model.layers.0.feed_forward.w1'   -> True (the defaults below)
mlx_lm.convert(source, target, quantize=True, q_group_size=64, q_bits=4,
  q_mode='affine', quant_predicate=keep_embedding_at_6)
[INFO] Loading
[INFO] Quantizing
[INFO] Quantized model with 4.695 bits per weight.
built in 3 s, peak memory 2.50 GB (MLX's count)
```

The whole build took 3 seconds on this run, and its peak memory, 2.50 GB,
stayed under half the original's 5.39 GB: mlx-lm opens the original lazily,
reading a weight only when it is needed, so the original is never in memory
all at once. The new folder is written under a temporary name and renamed only when it is
complete, so a build that breaks off never looks finished; with `--force`, an
old build is removed only once the new one is complete. A second run builds
nothing and checks what is there:

```text
already built (add --force to build it again); checking it:
```

### What the file holds

```text
the new config.json's quantization block:
  {"group_size": 64, "bits": 4, "mode": "affine", "model.embed_tokens": {"group_size":
    64, "bits": 6, "mode": "affine"}}
  LiquidAI/LFM2.5-2.6B-MLX-4bit's config.json: the same block

what model.safetensors holds, counted from its header:
  kind                layers        numbers  bits each          bytes
  4-bit layers           166  2,434,793,472       4.50  1,369,571,328
  6-bit layers             1    262,144,000       6.50    212,992,000
  kept in bfloat16        99        261,120      16.00        522,240
  in bits: 2,434,793,472 x 4.5 + 262,144,000 x 6.5 + 261,120 x 16 = 12,664,684,544 bits
    = 1,583,085,568 bytes
  plus the 67,324-byte header: 1,583,152,892 bytes, the file's size exactly
  1.58 GB (1.47 GiB); 2,697,198,592 numbers, 4.695 bits each on average
  the bfloat16 original's numbers take 5.39 GB: 3.41 times as much
```

Every number of the file is accounted for, from its header alone (chapter
10's way of reading a safetensors file, in reverse: each packed layer's bits
are its packed width x 32 / its real width). 166 layers are packed at 4 bits,
4.5 bits per number with the scale and the offset; the embedding table at 6
bits, 6.5 per number; and 99 small tensors stay in bfloat16, untouched by
`convert`: the norms and the convolution layers' 3-number kernels (`nn.quantize`
only packs layers that know how to be packed, linear and embedding layers).
The sum, plus the header, is the file's size to the byte: 1.58 GB, 3.41 times
smaller than the original's numbers. mlx-lm's own `[INFO]` line says the same
thing as an average: 4.695 bits per number.

### The all-4-bit variant

```bash
uv run convert.py --all-4bit
```

The same build without our predicate, so the embedding table is packed at 4
bits like everything else. It goes into `COURSE_HOME/11-quantize`, because
only `compare.py` uses it:

```text
mlx_lm.convert(source, target, quantize=True, q_group_size=64, q_bits=4,
  q_mode='affine')
[INFO] Loading
[INFO] Quantizing
[INFO] Quantized model with 4.501 bits per weight.
built in 3 s, peak memory 2.43 GB (MLX's count)

the new config.json's quantization block:
  {"group_size": 64, "bits": 4, "mode": "affine"}
  LiquidAI/LFM2.5-2.6B-MLX-4bit's config.json differs:
  {"group_size": 64, "bits": 4, "mode": "affine", "model.embed_tokens": {"group_size":
    64, "bits": 6, "mode": "affine"}}

what model.safetensors holds, counted from its header:
  kind                layers        numbers  bits each          bytes
  4-bit layers           167  2,696,937,472       4.50  1,517,027,328
  kept in bfloat16        99        261,120      16.00        522,240
  in bits: 2,696,937,472 x 4.5 + 261,120 x 16 = 12,140,396,544 bits = 1,517,549,568
    bytes
  plus the 67,324-byte header: 1,517,616,892 bytes, the file's size exactly
  1.52 GB (1.41 GiB); 2,697,198,592 numbers, 4.501 bits each on average
  the bfloat16 original's numbers take 5.39 GB: 3.55 times as much
```

## Ours against Liquid's, byte for byte

```bash
uv run compare.py
```

`compare.py` starts with the strongest test there is, before loading any
model: are the two builds the same bytes? It fingerprints both weights files
whole, then each of their tensors on its own, with **MD5** (a 32-character
code computed from a file's bytes; change one byte and the code changes, so
two files with the same MD5 hold the same bytes), and compares every other
file of the two folders:

```text
1. ours against LiquidAI/LFM2.5-2.6B-MLX-4bit, file by file (no model loaded)
  file                             ours, bytes  Liquid, bytes  same bytes
  model.safetensors              1,583,152,892  1,583,152,892  yes
  model.safetensors.index.json          41,983         41,983  yes
  config.json                            2,145          2,202  no
    quantization block identical; only Liquid's has block_ff_dim, rope_theta
  tokenizer.json                    17,905,598     17,905,598  yes
  chat_template.jinja                    5,443          5,496  no
    Liquid's has 2 more line(s): '{# <|tool_list_start|> detection hint for mlx_lm #}',
      ''
  tokenizer_config.json                    397            363  no
    only ours has tool_parser_type
  generation_config.json                   327            306  no
    only ours has do_sample
  README.md                                 81          2,648  no
    different model cards (7 and 98 lines): mlx-lm writes its own
  model.safetensors, tensor by tensor: 600 of 600 tensors hold identical bytes
  MD5 of the whole file:
    ours     bdf85b477b916229eaa3282052387a4a
    Liquid's bdf85b477b916229eaa3282052387a4a
  the two chat templates render 44 of the 44 prompts this script asks to the same tokens
```

The weights are Liquid's, byte for byte: both files have the same MD5, and
all 600 tensors match (for each of the 167 packed layers its codes, scales
and offsets, and the 99 tensors kept in bfloat16). The index that says which
tensor is in which file, and the tokenizer, are identical too. What differs is
the paperwork around the weights:

- `config.json`: the same quantization block. Liquid's also carries
  `block_ff_dim` and a top-level `rope_theta`; the original's `config.json`
  at chapter 10's pinned revision has no `block_ff_dim` and keeps
  `rope_theta` only inside `rope_parameters` (chapter 10 noted it).
- `chat_template.jinja`: ours is the original's template; Liquid's adds a
  comment line, a hint for mlx-lm's tool detection, and a blank line at the
  top. Rendered, the two give the same tokens for all 44 prompts this script
  asks (chapter 9 found that the comment renders to nothing).
- `tokenizer_config.json`: mlx-lm writes its own, adding `tool_parser_type`.
- `generation_config.json`: mlx-lm copies the original's, which has
  `do_sample`; Liquid's does not.
- `README.md`: each maker writes its own model card.

Byte-identical weights are the strongest proof there is that two builds are
the same model: every number the model computes with is the same. The rest of
`compare.py` checks it from the outside as well.

## Four builds, measured side by side

Then `compare.py` loads each build on its own, measures it, and frees its
memory before the next, so only one model is ever in memory:

```text
compare.py · four builds of LFM2.5-2.6B, measured on this laptop
builds, loaded one at a time:
  bfloat16         ~/.cache/mhrl-course/models/LFM2.5-2.6B
  ours             ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours
  all 4-bit        ~/.cache/mhrl-course/11-quantize/lfm2.5-2.6b-4bit-all
  Liquid's export  ~/.cache/mhrl-course/models/LFM2.5-2.6B-MLX-4bit
the same tests for every build, every reply greedy:
  log-probability: the gold SQL of 20 training questions, each read after
    <|startoftext|>
  agreement: the same 20 questions, each asked as "<question> Answer with one SQL
    query.", the first 128 tokens of each reply
  pass rate: the project's 24 held-out questions (airline and clinic; 8 easy, 8 medium,
    8 hard), the tables in the prompt, one reply of up to 1,024 tokens, graded by
    chapter 1's grader
```

Every build gets the same three tests, and every reply is **greedy** (always
the most likely token, chapter 10), so a rerun gives the same replies:

- **Log-probability per token.** The gold SQL of 20 training questions, 621
  tokens in all, is read in one pass, and each token's log-probability
  (chapter 3: 0 for certain, more negative for less likely) is taken from the
  model's scores, in float32. The average says how likely the build finds
  correct SQL: closer to 0 is better.
- **Greedy agreement with bfloat16.** The same 20 questions, asked as chapter
  10 asked its question, each answered for 128 tokens. **Same reply** counts
  the replies identical to bfloat16's for all 128 tokens. **Token agreement**
  is gentler: bfloat16's own replies are fed to the build, and at each of the
  2,560 positions the build's most likely next token is compared with
  bfloat16's. One early difference cannot spoil the rest of a reply this way.
- **Pass rate.** The project's own 24 held-out questions (its `balanced`
  pick: 12 per database, 8 per difficulty, never trained on). There is no
  agent here, just one reply, so the prompt shows the database's tables and
  asks for one SQL query whose result is the answer alone. The reply may run
  to 1,024 tokens, the project's cap. The query runs on a read-only copy of
  the database, its first column becomes the answer the way chapter 1 formats
  answers, and chapter 1's grader judges it.

The four builds took 22.5 minutes in all on this run. Each prints one line as
it finishes (the times vary from run to run):

```text
bfloat16: loaded in 2.4 s, tested in 671 s, 23 of 24 correct
ours: loaded in 1.0 s, tested in 214 s, 21 of 24 correct
all 4-bit: loaded in 1.5 s, tested in 233 s, 20 of 24 correct
Liquid's export: loaded in 1.6 s, tested in 216 s, 21 of 24 correct
```

### Size and speed

```text
2. size and speed
  build              on disk  in memory      peak  prompt tok/s  reply tok/s
  bfloat16           5.39 GB    5.39 GB   5.61 GB         527.0         21.8
  ours               1.58 GB    1.58 GB   2.05 GB         668.7         70.9
  all 4-bit          1.52 GB    1.52 GB   1.97 GB         669.6         72.7
  Liquid's export    1.58 GB    1.58 GB   2.05 GB         670.0         70.5
  in memory: MLX's memory in use once the build is loaded, the number the project's
    engine logs when it starts; peak: over all the tests; speeds: over the pass-rate
    questions (reading the prompts, writing the replies)
```

On disk and in memory, a build takes the same space: MLX keeps the packed
codes as they are in the file. The "in memory" column is the number the
project's engine logs when it starts (its `runs/logs/engine.log` reads
`engine ready · lfm2.5-2.6b 4-bit · 127.0.0.1:8100 · 1.6 GB`): 1.58 GB,
against 5.39 GB for the original. Writing a reply is where 4 bits pay: 70.9
tokens a second against 21.8, about 3.3 times as fast, because every new
token needs every weight read once more (chapter 10), and there are 3.4 times
fewer bytes to read. Reading the prompt gains less (669 against 527 tokens a
second). Ours and Liquid's export are the same bytes, so the small difference
between their two rows is how much the timings vary between two runs.

### Quality

```text
3. quality
  build             log-prob/token  same reply  token agreement  pass rate
  bfloat16                 -2.4455   reference        reference   23 of 24
  ours                     -2.6123     0 of 20            90.0%   21 of 24
  all 4-bit                -2.5982     0 of 20            90.4%   20 of 24
  Liquid's export          -2.6123     0 of 20            90.0%   21 of 24
  log-prob/token: the average log-probability of the 621 tokens of SQL (closer to 0: the
    build finds the texts more likely); same reply: the first 128 tokens identical to
    bfloat16's; token agreement: the share of positions in bfloat16's replies where the
    build's top token is bfloat16's
  (check: bfloat16's top tokens in one pass equal its own token-by-token replies at
    2,544 of 2,560 positions)
```

- **Log-probability**: -2.6123 per token for ours against -2.4455 for
  bfloat16. The 4-bit build finds the correct SQL a little less likely.
- **Agreement**: no 4-bit reply matches bfloat16's for all 128 tokens, yet at
  90.0% of the positions ours picks bfloat16's token. Both are true at once:
  once a single token differs, the reply goes its own way. For scale,
  bfloat16 against itself, computed the two ways (one pass, and token by
  token as replies are written), agrees at 2,544 of 2,560 positions, 99.4%:
  rounding alone moves a few near-ties.
- **The 6-bit embedding table** made no difference that these tests can see:
  the all-4-bit build has a slightly *better* log-probability (-2.5982) and
  agreement (90.4%) than ours, and one question fewer right (20 against 21).

### Task by task

A pass rate is read question by question, never as a lone average:

```text
4. the 24 questions, task by task (1 correct, 0 wrong)
  task                    bf16    ours    all4  Liquid
  airline-easy-001           1       1       1       1
  airline-hard-001           1       0       1       0
  airline-medium-001         1       1       1       1
  clinic-easy-001            1       1       1       1
  clinic-hard-001            1       1       1       1
  clinic-medium-001          1       1       0       1
  airline-easy-002           1       1       1       1
  airline-hard-002           1       1       1       1
  airline-medium-002         1       1       1       1
  clinic-easy-002            1       1       1       1
  clinic-hard-002            1       1       0       1
  clinic-medium-002          1       1       1       1
  airline-easy-003           1       1       1       1
  airline-hard-003           1       1       1       1
  airline-medium-003         1       1       1       1
  clinic-easy-003            1       1       1       1
  clinic-hard-003            1       1       1       1
  clinic-medium-003          1       1       1       1
  airline-easy-004           1       1       1       1
  airline-hard-004           0       0       0       0
  airline-medium-004         1       1       1       1
  clinic-easy-004            1       1       1       1
  clinic-hard-004            1       0       0       0
  clinic-medium-004          1       1       1       1
  correct                   23      21      20      21
  against bfloat16:
    gained                           0       0       0
    lost                             2       3       2
    exact p                      0.500   0.250   0.500
  gained: right here, wrong in bfloat16; lost: the reverse; exact p: the chance of a
    split at least this lopsided if the two builds were equally good (McNemar's exact
    test)
```

Ours got every question right that bfloat16 got right, except two:
`airline-hard-001`, where its query returned the passenger's id (98) instead
of the name the question asks for, and `clinic-hard-004`, which returned
doctor ids instead of names. It gained none. **Gained** and **lost** count the
questions on which two builds disagree; if the builds were equally good,
each such question would be a coin flip, and the **exact p** is the chance of
a split at least this lopsided. Two lost and none gained gives p = 0.50: it
happens half the time between equally good builds. So 4 bits may cost a
question or two here, and 24 questions cannot tell; the one question between
ours and the all-4-bit build (p = 0.25 for its 3 lost) is no evidence for the
6-bit table either. One question, `airline-hard-004` (bookings made at least
14 days before departure), beat every build: bfloat16 counted 0, and the 4-bit
builds wrote a date expression SQLite does not accept.

The prompt was not right the first time. The first full run asked only for
"one SQL query", and 4 of bfloat16's 5 misses (4 of ours' 5 too) were queries
that put the id first, with the right answer in the second column: the pass
rate was measuring the prompt, not the builds. The prompt now asks for the
answer alone, as chapter 1's instruction does ("write ONLY the answer"); the
grading did not change. That first run is kept beside the run log.

### Ours against Liquid's, by what they do

```text
5. ours against Liquid's export, by what they do
  agreement replies identical: 20 of 20
  top tokens along bfloat16's replies identical: 2,560 of 2,560
  pass-rate replies identical: 24 of 24; correct: ours 21, Liquid's 21
verdict: byte-identical weights and identical greedy replies: our build is Liquid's
  export, made again on this laptop
```

Every agreement reply, every one of the 2,560 top tokens and every pass-rate
reply is the same: with identical bytes nothing else was possible, and the
outside view agrees with the MD5s.

### Fast mode

```bash
uv run compare.py --fast
```

Without chapter 10's bfloat16 download, or without building ours, `--fast`
measures Liquid's export alone:

```text
fast mode: Liquid's export alone. Comparing ours with it needs convert.py's build, and
  agreement needs the bfloat16 original, so both are skipped

Liquid's export: loaded in 1.4 s, tested in 215 s, 21 of 24 correct
```

It answers the same 21 of 24, with the same replies, as in the full run.

## Ask it: ask.py

```bash
uv run ask.py "How many customers live in Paris? Answer with one SQL query."
```

`ask.py` loads our build, the one every later chapter uses (Liquid's export
if ours is not built yet), and asks chapter 10's question:

````text
ask.py · the 4-bit LFM2.5-2.6B answers, on this laptop
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours
          ours, built by convert.py; 1.58 GB of weights in memory
warm-up   0.33 s for one untimed token (one-time GPU set-up)
question  How many customers live in Paris? Answer with one SQL query.
prompt    23 tokens, ending with <think>; decoding is greedy

the reply, printed as each token is predicted:

The user wants to know the number of customers living in Paris.
The constraint is to answer with *one* SQL query.
I need to assume a table structure. Since none is provided, I will use a standard naming convention (e.g., `customers` table with a `city` column).
The query should count the number of rows where the city is 'Paris'.
Query: `SELECT COUNT(*) FROM customers WHERE city = 'Paris';`
Wait, I should check if there are any specific nuances. "How many customers live in Paris?" implies a count.
One SQL query is required.
I will provide the query.</think>```sql
SELECT COUNT(*) FROM customers WHERE city = 'Paris';
```

the reply: 144 tokens, then <|im_end|>, its end of turn (finish_reason stop)
first token after 0.06 s; then 15.0 ms per token (66.7 tokens a second)
peak memory 1.70 GB (MLX's count)
````

The reply is 144 tokens and ends with the same query as chapter 10's
bfloat16 run, which reasoned for 286 tokens. Chapter 10's `first_reply.py`
saved that run, so `ask.py` puts the two side by side:

```text
before and after: chapter 10's bfloat16 run of the same question
                          bfloat16             4-bit
  weights                  5.39 GB           1.58 GB
  peak memory              5.45 GB           1.70 GB
  ms per token                45.4              15.0
  reply tokens                 286               144
  both end with the same query: SELECT COUNT(*) FROM customers WHERE city = 'Paris';
```

Under a third of the memory and about three times as fast per token, for the
same answer to this question. (The bfloat16 numbers are chapter 10's saved
run; the times vary from run to run.)

### The real budget: unified memory

```text
the real budget: unified memory
  this Mac has 18.00 GiB of memory (19.33 GB), one pool the CPU and the GPU share: the
    weights are loaded once and the GPU works on them where they are
  Metal recommends the GPU keep at most 14.30 GB of it in use (74.0%); the rest is left
    to macOS and the other programs
  this run's peak, 1.70 GB, is 11.9% of that; chapter 10's bfloat16 run peaked at 5.45
    GB (38.1%)
```

On an Apple-silicon Mac the processor (CPU) and the graphics processor (GPU),
which does the model's arithmetic, share one pool of memory, called
**unified memory**. Nothing is copied to a separate graphics card: the
weights are loaded once and the GPU works on them where they are. So the
whole pool is the budget, shared with macOS and every other program, and
Metal, Apple's interface to the GPU, recommends that the GPU keep at most
14.30 GB of this laptop's 19.33 GB in use. The 4-bit model peaked at 1.70 GB,
11.9% of that; the bfloat16 original at 5.45 GB, 38.1%. Everything later
chapters add (the cache that grows with every token, training's extra
numbers, several agents' conversations) has to fit in what is left.

`uv run ask.py --fast` asks Liquid's export instead, and gets the same reply:

```text
ask.py · the 4-bit LFM2.5-2.6B answers, on this laptop
model     ~/.cache/mhrl-course/models/LFM2.5-2.6B-MLX-4bit
          Liquid's export (--fast); 1.58 GB of weights in memory
```

## What the chapter showed, and where it differed from the plan

- **The matrix**: `model.layers.10.feed_forward.w1`, 10,752 x 2,048 in
  bfloat16, as planned.
- **"scale = (max - min) / 15, offset = min, q = round((w - min) / scale)"**:
  that is the naive rule, and it is not MLX's. MLX anchors each group at the
  end farther from 0 (the max in 173,483 of 344,064 groups, where its step is
  negative and the codes count down), nudges the step so 0 is stored exactly,
  rounds halves away from 0, and starts its running max at 0. The naive codes
  differ from `mx.quantize`'s in 13,405,849 of 22,020,096 places (60.9%). As
  the review asked, the chapter shows the naive rule first, then MLX's real
  rule written out by hand, which gives identical codes, scales and offsets at
  3, 4, 6 and 8 bits (and identical rebuilt numbers); and it compares the
  dequantized error, where the naive rule is slightly better at 4, 6 and 8
  bits (9.06% against 9.18% at 4) and MLX's at 3.
- **"identical codes from MLX"**: for MLX's rule by hand, yes, all
  22,020,096; for the naive rule, no.
- **4.5 bits per number, 6.5 for the embedding table**: as planned.
- **"2.44 B weights x 4.5 bits + 0.26 B x 6.5 bits = about 1.58 GB"**: the
  total is right, to the byte once every part is counted:
  2,434,793,472 numbers (2.43 B, not 2.44 B) at 4.5 bits, 262,144,000 at 6.5,
  and 261,120 that are not quantized at all (norms and convolution kernels) at
  16, make 1,583,085,568 bytes; with the 67,324-byte header that is
  1,583,152,892 bytes, exactly the size of Liquid's file. The plan's 2.44 B
  was the total minus the embedding table, which still counted those 261,120.
- **"mixed_4_6 stops with 'Model does not have expected keys for mixed
  quant'"**: as planned, word for word; 0 of LFM2's 167 quantizable layers are
  named `down_proj`.
- **The quant_predicate**: two arguments `(path, module)`, as planned. It
  returns `{"group_size": 64, "bits": 6, "mode": "affine"}` for
  `model.embed_tokens`: the plan said `{bits 6, group 64}`; the mode is added
  so the written block matches Liquid's exactly, which it does. mlx-lm's own
  type hint for the predicate names three arguments, but mlx-lm calls it with
  two, and a three-argument predicate stops with a `TypeError` (a test checks
  it).
- **The quantization block**: `{group_size 64, bits 4, mode affine,
  model.embed_tokens: 6 bits}`, the same as Liquid's, as planned.
- **Ours against Liquid's export**: the plan asked for a per-tensor MD5 and a
  greedy-agreement check against `LiquidAI/LFM2.5-2.6B-MLX-4bit`. Measured:
  byte-identical weights, one MD5 for both files
  (`bdf85b477b916229eaa3282052387a4a`) and all 600 tensors the same; the 20
  agreement replies, all 2,560 top tokens and the 24 pass-rate replies are
  identical. `config.json` (two extra keys in Liquid's), the chat template (a
  comment), `tokenizer_config.json`, `generation_config.json` and the model
  card differ; none of them changes a token of the 44 prompts.
- **compare.py's table**: one row per build, as planned. On disk 5.39 GB to
  1.58 GB (planned: about 5.4 to about 1.6); in memory once loaded 1.58 GB
  (the engine's log says 1.6 GB); prompt and reply speed (replies 21.8 to
  70.9 tokens a second on this run); log-probability per token -2.4455 to
  -2.6123; greedy agreement with bfloat16 on 20 prompts: no reply identical
  over 128 tokens, 90.0% of top tokens the same, beside a 99.4% floor for
  bfloat16 against itself; pass rate 23, 21, 20 and 21 of 24, read task by
  task: ours lost 2 and gained none against bfloat16 (exact p = 0.50), so 24
  questions cannot tell the builds apart. Added: gained, lost and p against
  bfloat16, and the results saved after every build. Changed after the first
  full run: the single-turn prompt now asks for the answer alone (see "Task
  by task").
- **ask.py and the memory budget**: a question in, our model's answer out,
  with time and memory, as planned; plus chapter 10's saved bfloat16 run of
  the same question beside it (286 to 144 tokens, 45.4 to 15.0 ms per token,
  peak 5.45 to 1.70 GB, the same final query), and the budget as Metal
  reports it: 18 GiB (19.33 GB) of unified memory, of which it recommends at
  most 14.30 GB for the GPU.
- **--fast**: the plan's "quantizes one matrix by hand and benchmarks only the
  official export" became `by_hand.py --fast`, which fetches only that matrix
  (44 MB) when chapter 10's download is missing, and `compare.py --fast`,
  which measures only Liquid's export (`ask.py --fast` asks it too).
  `by_hand.py` always quantizes just the one matrix, so it needs no shorter
  mode.
- **Where the files go**: the plan's `course/11-quantize/` is `11-quantize/`
  at the root of this repository, like every chapter. Our build is in
  `COURSE_HOME/models/lfm2.5-2.6b-4bit-ours` as planned; the all-4-bit variant
  is chapter-only, in `COURSE_HOME/11-quantize`.
- **The SQL questions**: the plan named "24 single-turn SQL questions" and "20
  prompts" without saying which. The 24 are the project's own held-out set
  (its `balanced` pick of 24 test questions, 12 per database, 8 per
  difficulty), each with the database's tables in the prompt, since one reply
  cannot explore the database; the 20 are training questions asked the way
  chapter 10 asked its question, and their gold SQL is the "fixed set of SQL
  texts".

## What is in this folder

- `by_hand.py`: one matrix quantized by the naive rule and by MLX's rule,
  written out by hand, with every code, scale, offset and rebuilt number
  checked against `mx.quantize` and `mx.dequantize`, at 3, 4, 6 and 8 bits.
- `matrix_source.py`: finds that matrix in chapter 10's download, or with
  `--fast` fetches it alone from the Hub with HTTP range requests; either way
  checked against its pinned SHA-256 fingerprint.
- `convert.py`: shows why mlx-lm's `mixed_4_6` recipe cannot be used, builds
  our 4-bit model with `mlx_lm.convert` and a two-argument `quant_predicate`,
  and accounts for every byte of the result from its header. `--all-4bit`
  builds the comparison variant.
- `compare.py`: ours against Liquid's export file by file and tensor by
  tensor (MD5), then four builds measured one at a time: size, memory, speed,
  log-probability, greedy agreement with bfloat16 and the pass rate on 24
  held-out questions, task by task. Every reply and verdict is saved under
  `COURSE_HOME/11-quantize`, after each build.
- `ask.py`: a question in, our model's answer out, beside chapter 10's
  bfloat16 run of the same question, and this Mac's memory budget.
- `sql_questions.py`: the 24 held-out questions (picked as the project's
  `balanced` picks them), the 20 agreement questions, the single-turn prompt,
  and grading: find the query, run it on a read-only database, format the
  answer as chapter 1 does, let chapter 1's grader judge it.
- `tensors.py`: reads safetensors headers, single tensors and per-tensor MD5s,
  with the standard library only.
- `lfm_files.py`, `lfm.py`: where every build lives (and the rule later
  chapters use: ours, else Liquid's export), and loading a build with mlx-lm;
  adapted copies of chapter 10's.
- `grader.py`, `tasks.py`, `data/`: chapter 1's grader and questions, byte for
  byte, with the two held-out databases (clinic and airline).
- `report.py`: how the scripts print.

## Words from this chapter

- **Quantize**: store numbers with fewer bits, accepting a small error.
- **Group, scale, offset (bias), code**: 64 numbers that share one straight
  line, `number = scale x code + offset`; each number is stored as its code,
  a small whole number (0 to 15 at 4 bits). MLX calls the offset a bias.
- **Affine, group-wise quantization**: a straight line per group, which is
  what MLX's `affine` mode does.
- **Anchor**: in MLX's rule, the end of the group farther from 0, which
  becomes the offset; the step is then nudged so 0 is stored exactly.
- **Bits per number**: the code's bits plus the scale's and offset's 32 bits
  shared by 64 numbers: 4.5 at 4 bits, 6.5 at 6.
- **Typical size, relative error**: a root mean square, and the error's
  typical size divided by the numbers'.
- **Mixed precision**: some parts at more bits than others; here the
  embedding table, which is also the output head, at 6.
- **quant_predicate**: the function `mlx_lm.convert` calls for each layer to
  decide how to quantize it.
- **MD5**: a 32-character fingerprint of a file's bytes; equal MD5s mean equal
  bytes.
- **GB, GiB**: 10^9 bytes and 2^30 bytes; this course uses GB.
- **Log-probability per token**: how likely a model finds a text, averaged
  over its tokens; closer to 0 is more likely.
- **Greedy agreement**: how often two builds pick the same next token.
- **Pass rate**: the share of questions answered correctly, read task by task
  with gained, lost and an exact p-value, never as a lone average.
- **Unified memory**: one pool of memory that the CPU and the GPU share on an
  Apple-silicon Mac, so the model is loaded once and never copied to a GPU.

Next: [chapter 12](../12-generate/), How the model picks each token.
