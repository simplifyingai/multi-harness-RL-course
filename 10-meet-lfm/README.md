# Chapter 10. Meet LFM2.5-2.6B

Chapter 9 looked at what the model reads and writes: tokens. This chapter
downloads the model itself, LFM2.5-2.6B by Liquid AI, counts its numbers,
looks inside it while it reads one prompt, and asks it its first question on
this laptop.

Start with a puzzle. The model's name says 2.6B: 2.6 billion numbers. Its own
model card says 2.69B. Count the numbers in its files and you get
2,697,198,592, which is 2.70 billion. Three answers to one question, and only
one of them is a count, and by the end of the chapter you will have made it
yourself.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/10-meet-lfm
uv run download.py
```

The download is 5.41 GB and takes a few minutes on a fast connection (8
minutes here). It goes into `COURSE_HOME/models` (default
`~/.cache/mhrl-course/models`), which every later chapter reuses, so it happens
once. The scripts write their own outputs under `COURSE_HOME/10-meet-lfm`,
never into this folder.

`download.py` and `inspect_model.py` run on any computer. `inside.py` and
`first_reply.py` load the model with MLX, Apple's library for machine learning
on Apple silicon, so they need an Apple-silicon Mac; they use about 5.5 GB of
memory (16 GB recommended). `uv run` installs what each script needs for that
script alone: `huggingface_hub` for the download, and `mlx`, `mlx-lm` and
`transformers` for the model.

| command | what it shows |
| --- | --- |
| `download.py` | downloads LFM2.5-2.6B (5.41 GB), checks every file, lists them, quotes the license |
| `inspect_model.py` | counts every number in the model from its files, part by part, without loading any |
| `inside.py` | one prompt walked through the model, stage by stage, every hand computation checked |
| `first_reply.py "a question"` | the model's answer, printed token by token, with milliseconds per token and peak memory |

Every script takes `--fast`, which uses Liquid's 4-bit export of the same
model instead: a 1.60 GB download, the copy the project itself runs (chapter
11 explains 4 bits and makes its own).

## The download: one folder of files

A model on the Hugging Face Hub (chapter 9) is a repository: a folder of files
at one web address. The project's own setup script downloads its model with
one call, `huggingface_hub.snapshot_download`, which fetches every file of a
repository into a local folder. `download.py` makes the same call for Liquid's
original repository, `LiquidAI/LFM2.5-2.6B`:

```text
download.py · LFM2.5-2.6B from the Hugging Face Hub
repository  LiquidAI/LFM2.5-2.6B, revision 654f9463ce32 (2026-08-24), bfloat16
into        ~/.cache/mhrl-course/models/LFM2.5-2.6B
downloading 11 files, 5.41 GB, with snapshot_download
(progress shows below; a download that breaks off resumes when run again)
all 11 files checked against their SHA-256 fingerprints

the repository's files
  model-00001-of-00002.safetensors  5,329,406,264 bytes  the weights, part 1
  model-00002-of-00002.safetensors     65,021,192 bytes  the weights, part 2
  model.safetensors.index.json             21,425 bytes  which tensor is in which file
  config.json                               1,467 bytes  the architecture: layers, sizes, positions
  generation_config.json                      327 bytes  the maker's default sampling settings
  tokenizer.json                       17,905,598 bytes  the vocabulary (chapter 9)
  tokenizer_config.json                       363 bytes  the special tokens (chapter 9)
  chat_template.jinja                       5,443 bytes  the chat template (chapter 9)
  LICENSE                                  10,574 bytes  the terms of use
  README.md                                18,337 bytes  the model card: the maker's description
  .gitattributes                            1,570 bytes  a git setting for the large files
  11 files, 5.41 GB in all
```

While it downloads, `huggingface_hub` draws its own progress bar on the
terminal. The repository is pinned to one revision (one saved version of the
folder), so the files can never change under the course, and each file is
checked against its SHA-256 fingerprint (a 64-character code that changes if a
single byte does) before the script goes on. A second run finds every file at
its full size and downloads nothing:

```text
already here: all 11 files at full size (SHA-256 checked when downloaded)
```

If the Hub cannot be reached, the script says so and what to do: check the
connection and run it again (an interrupted download resumes). If the Hub
answers with an error, such as a revision that no longer exists, it says that
the connection works and the Hub refused. `huggingface_hub` downloads large
files with a helper, `hf_xet`, that keeps its own small folder of logs; the
script points it into `COURSE_HOME` too, unless you have set `HF_XET_CACHE`
yourself.

Most of the 5.41 GB is the two **safetensors** files: the weights, the numbers
the model learned. safetensors is a file format for blocks of numbers that a
program can read without running any code from the file, which is why it
replaced older formats that could. `config.json` describes how the model is
built; the next section reads it. The tokenizer files and the chat template
are chapter 9's, byte for byte.

### The license

Open weights still come with terms. `download.py` ends by quoting the lines
that matter from the `LICENSE` file itself:

```text
the license, quoted from LICENSE (10,574 bytes)
  LFM Open License v1.0
  "Commercial Use" shall mean any use of the Work for direct or indirect commercial
    advantage or monetary compensation.
  "Threshold" shall mean annual revenue of 10 million United States dollars
    ($10,000,000) or more.
  5. Commercial Use Limitation.
  (a) The rights granted under this License for Commercial Use are conditioned upon You
    or Your Legal Entity not exceeding the Threshold.
  (b) Any Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds
    the Threshold is not licensed under this Agreement.
  (c) The Threshold shall not apply to a Qualified Non-Profit Organization's use of the
    Work or a Derivative Work for Non-Commercial or Research Purposes.
```

In plain words: anyone may use, change and share the model. Using it to make
money is allowed only for a company (counting the companies it controls or is
controlled by) whose revenue is under 10 million US dollars a year; a larger
company needs a separate agreement with Liquid AI. Non-profits are exempt for
non-commercial and research use. The rest of the license reads like the
Apache 2.0 license: keep the license with copies, mark files you changed, no
warranty. This is not legal advice; read the file.

### What "base model" means in this course

The model card lists `base_model: LiquidAI/LFM2.5-2.6B-Base`: Liquid's raw
model, trained only to continue text. LFM2.5-2.6B is that model after Liquid's
own further training (following instructions, using tools, and agentic
reinforcement learning, as its card says). When this course says "the base
model", it means LFM2.5-2.6B as downloaded here, before any training of ours.
Nothing in this course trains a model from scratch.

## Count it yourself

```bash
uv run inspect_model.py
```

A safetensors file starts with 8 bytes that give the length of its header.
The header is text (JSON) naming every **tensor** in the file with its number
type, its shape and where its bytes sit. A tensor is a block of numbers: a
**vector** is a list of numbers (one dimension), a **matrix** is a table of
numbers with rows and columns (two dimensions). Every number in the model is
one **parameter**, also called a **weight**: a number that training set.

Reading only the headers is enough to count every parameter, and it takes 30
kilobytes of the 5.4 GB:

```text
inspect_model.py · counting LFM2.5-2.6B's numbers from its files, loading none

reading only the headers in ~/.cache/mhrl-course/models/LFM2.5-2.6B
  model-00001-of-00002.safetensors  5,329,406,264 bytes, header 29,240 bytes, 257 tensors
  model-00002-of-00002.safetensors     65,021,192 bytes, header 1,032 bytes, 9 tensors
  read 30,272 of 5,394,427,456 bytes (0.0006%); no weight loaded
```

`config.json` says how the model is built. These are the settings this chapter
uses; later chapters come back to most of them:

```text
config.json, the settings this chapter uses
  hidden_size                    2,048  numbers in each token's vector
  vocab_size                   128,000  rows in the embedding table
  num_hidden_layers                 30  22 conv + 8 full_attention (layer_types)
  intermediate_size             10,752  width of each feed-forward block
  conv_L_cache                       3  tokens each convolution sees
  num_attention_heads               32  query heads in each attention layer
  num_key_value_heads                8  key/value heads they share
  rope_theta                10,000,000  how slowly positions turn (rope_parameters)
  max_position_embeddings      131,072  the most positions it reads
  tie_word_embeddings             true  the embedding table is also the output head
  dtype                       bfloat16  how each number is stored
```

The header lists 266 tensors. Here is the embedding table, every tensor of
layer 0 (a convolution layer) and the attention part of layer 2 (an attention
layer); the other layers repeat these shapes:

```text
the embedding table, layer 0 (convolution) and layer 2's attention:
  tensor                                        type  shape              parameters
  model.embed_tokens.weight                     BF16  128000 x 2048     262,144,000
  model.layers.0.operator_norm.weight           BF16  2048                    2,048
  model.layers.0.conv.in_proj.weight            BF16  6144 x 2048        12,582,912
  model.layers.0.conv.conv.weight               BF16  2048 x 1 x 3            6,144
  model.layers.0.conv.out_proj.weight           BF16  2048 x 2048         4,194,304
  model.layers.0.ffn_norm.weight                BF16  2048                    2,048
  model.layers.0.feed_forward.w1.weight         BF16  10752 x 2048       22,020,096
  model.layers.0.feed_forward.w3.weight         BF16  10752 x 2048       22,020,096
  model.layers.0.feed_forward.w2.weight         BF16  2048 x 10752       22,020,096
  model.layers.2.self_attn.q_proj.weight        BF16  2048 x 2048         4,194,304
  model.layers.2.self_attn.k_proj.weight        BF16  512 x 2048          1,048,576
  model.layers.2.self_attn.v_proj.weight        BF16  512 x 2048          1,048,576
  model.layers.2.self_attn.q_layernorm.weight   BF16  64                         64
  model.layers.2.self_attn.k_layernorm.weight   BF16  64                         64
  model.layers.2.self_attn.out_proj.weight      BF16  2048 x 2048         4,194,304
```

What the parts are, in one line each (the next section watches them work):

- **The embedding table** (`embed_tokens`): one row of 2,048 numbers for each
  of the 128,000 token ids. A token goes in as its row.
- **A feed-forward block** (`feed_forward`, in every layer): for each token on
  its own, `w1` and `w3` widen its 2,048 numbers to 10,752 each, one of the two
  results gates the other (lets more or less of each number through), and `w2`
  shrinks the result back to 2,048.
- **A convolution layer** (`conv`, 22 of the 30): `in_proj` makes three
  versions of each token's vector, and the convolution (`conv.conv`, 3 numbers
  for each of the 2,048) mixes each token with the two tokens just before it;
  `out_proj` maps the result back.
- **An attention layer** (`self_attn`, 8 of the 30): every token can look back
  at every earlier token. `q_proj`, `k_proj` and `v_proj` make its query, key
  and value, and `out_proj` maps the result back.
- **A norm** (`operator_norm`, `ffn_norm`, the final `embedding_norm`, and the
  two small `layernorm`s inside attention): rescales a vector to a steady
  size, with one learned number per position in it.

Then the tally:

```text
parameters by part
  part                                     tensors     parameters   share
  feed-forward, in 30 layers                    90  1,981,808,640   73.5%
  convolution, in 22 layers                     66    369,233,920   13.7%
  attention, in 8 layers                        48     83,887,104    3.1%
  embedding table, 128,000 x 2,048               1    262,144,000    9.7%
  norms, 2 per layer + 1 at the end             61        124,928    0.0%
  total                                        266  2,697,198,592  100.0%
  attention tensors sit in layers 2, 5, 9, 13, 17, 21, 24, 27
  (config.json's layer_types agrees)
  no output head is stored: tie_word_embeddings is true
```

Almost three quarters of the model is feed-forward blocks. The 8 attention
layers, the part people talk about most, hold 3.1% of the numbers. Every
count can also be worked out from `config.json` alone, and the script checks
that the two agree:

```text
the same counts from config.json alone
  feed-forward 30 layers x 3 matrices x 2,048 x 10,752                     = 1,981,808,640
  convolution  22 layers x (2,048 x 6,144 + 2,048 x 2,048 + 2,048 x 3)     =   369,233,920
  attention    8 layers x (2 x 2,048 x 2,048 + 2 x 2,048 x 512 + 2 x 64)   =    83,887,104
  embedding    128,000 x 2,048                                             =   262,144,000
  norms        (30 layers x 2 + 1) x 2,048                                 =       124,928
  every part agrees with the files
```

### Model names round: count it yourself

```text
the name says 2.6B, the model card says 2.69B; counted: 2,697,198,592, about 2.70 B
model names round: count it yourself
```

A model's name is a label for its size class, and the label is rounded (here,
down). The card's 2.69B is cut off rather than rounded. Neither is wrong as a
label, but neither is a number to calculate with. When a later chapter needs
the size, for memory, for a percentage of trainable numbers, for a file size,
it uses the count, 2,697,198,592, which the files give exactly.

### bfloat16: 2 bytes per number

```text
memory: every number is stored in bfloat16, 2 bytes
  2,697,198,592 numbers x 2 bytes = 5,394,397,184 bytes, 5.39 GB
  the files: 5,394,427,456 bytes, the numbers plus 30,272 bytes of headers
  in float32, 4 bytes each, the same numbers would take 10.79 GB
```

That is where "about 5.4 GB" comes from: every number takes 2 bytes, because
it is stored as a **bfloat16**. The next section takes one apart.

### The fast tally: packed numbers

```bash
uv run inspect_model.py --fast
```

The 4-bit export stores most matrices packed: 32-bit words (`U32`), each
holding eight 4-bit numbers. So a stored shape does not show the real width:

```text
  model.layers.0.feed_forward.w1.weight         U32   10752 x 256        22,020,096
    packed: 256 words x 32 / 4 bits = 2,048 columns, so it holds 10752 x 2048
```

The rule is **real columns = stored columns x 32 / bits**, and the bits come
from `config.json`'s quantization block. The embedding table is kept at 6
bits, so its 384 stored columns are 384 x 32 / 6 = 2,048:

```text
  model.embed_tokens.weight                     U32   128000 x 384      262,144,000
    packed: 384 words x 32 / 6 bits = 2,048 columns, so it holds 128000 x 2048
```

With the rule, the 4-bit file holds exactly the same 2,697,198,592 numbers,
part for part. Counting the stored values instead gives a wrong answer:

```text
memory: the 4-bit export packs its matrices (config.json's quantization block)
  most at 4 bits in groups of 64, model.embed_tokens at 6 bits; a packed matrix's real
    width is its stored width x 32 / bits, as the rows above show
  each group of 64 numbers also keeps a scale and an offset (.scales, .biases): they
    only help store the numbers, so they hold no parameters
  counting stored values instead would say 438,041,600: wrong, the packed words hide the
    rest
  2,697,198,592 numbers in 1,583,085,568 bytes (1.58 GB): 4.70 bits each on average
    (chapter 11 builds this file itself)
```

The script also checks each packed matrix against its scales: a matrix 2,048
numbers wide has 2,048 / 64 = 32 scales per row. If the bits in `config.json`
were wrong, the two would disagree and the script would stop.

## Inside the model, one stage at a time

```bash
uv run inside.py
```

`inside.py` loads the model with `mlx-lm`, the library the project's engine
uses, and walks chapter 9's prompt through it one stage at a time. Each stage
is computed by mlx-lm's own code, and each hand computation is checked against
mlx-lm's result.

```text
inside.py · one prompt walked through LFM2.5-2.6B (bfloat16)
loaded ~/.cache/mhrl-course/models/LFM2.5-2.6B with mlx-lm:
  2,697,198,592 numbers, 5.39 GB in memory
the walk below runs the layers one at a time; its scores equal one call
  of the model (largest difference 0)

1. tokens in
the prompt for "How many customers live in Paris?" is 17 tokens (chapter 9):
'<|startoftext|>' '<|im_start|>' 'user' '\n' 'How' ' many' ' customers' ' live' ' in' ' Paris' '?' '<|im_end|>' '\n' '<|im_start|>' 'assistant' '\n' '<think>'
```

The loaded model holds the same 2,697,198,592 numbers the files did, and 5.39
GB of memory. Running the 30 layers one at a time gives exactly the scores of
one ordinary call of the model, so what follows is the model's own pass, not
an imitation of it.

### Each token becomes a vector

```text
2. each token becomes a vector
the embedding table is a matrix of 128,000 rows x 2,048 columns, one row per token id
token 4741 ' Paris' -> row 4741, a vector of 2,048 numbers:
  [0.0294189, 0.0114136, 0.00326538, 0.026123, 0.00772095, -0.0292969, ...]
the 17 tokens become a 17 x 2,048 matrix;
  every layer reads a matrix of that shape and writes one back
```

A token id is just a row number. The model's first step is to look up that
row: `' Paris'` becomes 2,048 numbers. Nobody chose them; training did, and
tokens used in similar ways ended up with similar rows. From here on, the
prompt is a 17 x 2,048 matrix, and each of the 30 layers turns it into a new
17 x 2,048 matrix.

### How each number is stored: bfloat16

```text
3. how each number is stored: bfloat16
the first number of that row, 0.0294189453125, is these 16 bits:
  sign 0 | exponent 01111001 | fraction 1110001
  its bfloat16 neighbours: 0.029296875 below, 0.029541015625 above
  a step of 0.000122; float32 has 65,535 more numbers inside each such step
small differences get rounded away (computed in MLX):
  1 + 0.001   bfloat16 1          float32 1.001
  -10.3       bfloat16 -10.3125   float32 -10.3
  between 8 and 16, bfloat16 holds only multiples of 1/16:
  log-probabilities (chapter 12) need float32
bfloat16 keeps float32's 8 exponent bits (the same range) and 7 of its
23 fraction bits: 5.39 GB of weights instead of 10.79 GB in float32
```

A computer stores a number with a fixed count of bits. The usual kind,
**float32**, uses 32: 1 for the sign, 8 for the **exponent** (which power of
two: how big the number is) and 23 for the **fraction** (its digits).
**bfloat16** keeps the sign and all 8 exponent bits and only the first 7
fraction bits. So it reaches numbers just as big and just as small as float32,
but with about 3 significant digits instead of 7.

That has a cost you can see. Next to 0.0294 the nearest bfloat16 numbers are
0.000122 apart, and anything in between is rounded to one of them. Add 0.001
to 1 and bfloat16 still says 1. Store -10.3 and it becomes -10.3125, because
between 8 and 16 bfloat16 only holds multiples of 1/16. The model's weights
can live with that, and storing them in bfloat16 halves the memory: 5.39 GB
instead of 10.79 GB. A log-probability cannot (chapter 3 met them): chapter 12
shows that in bfloat16 they all land on that coarse grid, which is why the
project computes them in float32. Chapter 11 goes further and stores weights
in 4 bits.

### 30 layers of two kinds

```text
4. 30 layers of two kinds
  layer  0         1         2
         012345678901234567890123456789
  kind   CCACCACCCACCCACCCACCCACCACCACC   C convolution (22), A attention (8)
attention at layers 2, 5, 9, 13, 17, 21, 24, 27 (counting from 0, as config.json does)
a convolution layer mixes each token with the 2 just before it (conv_L_cache 3)
an attention layer lets each token look back at every earlier token
every layer then runs its own feed-forward block (w1, w3, w2) on each token
after the prompt, the cache holds (measured):
  a convolution layer: its last 2 inputs, 2 x 2,048 numbers, however long the prompt
  an attention layer: keys and values for all 17 tokens, 2 x 8 x 17 x 64 numbers;
  it grows by one token for every token read or written
-> chapter 12 reuses this cache instead of re-reading the prompt
-> chapter 13 trains small add-ons on the last 12 layers (18-29): 3 attention, 9 convolution
```

LFM2 is a **hybrid**: most of its layers are cheap short convolutions that only
mix each token with its two neighbours before it, and a few are full attention
layers that can reach any earlier token. Stacked 30 deep, the short reaches add
up, and the 8 attention layers carry the long-range work. That keeps the model
fast and light, which is what Liquid built it for.

The layer kinds matter twice later. When the model writes a reply, it keeps a
**cache** so it never has to re-read the prompt: the script measured it. A
convolution layer only needs its last 2 inputs, whatever the length; an
attention layer needs a key and a value for every token so far, so its cache
grows with every token (chapter 12). And chapter 13 trains small add-ons
(LoRA) on the last 12 layers, which are 3 attention and 9 convolution layers,
each with its own matrices.

### Inside attention: grouped-query attention

```text
5. inside attention layer 2: 32 query heads share 8 key/value heads
the 17 x 2,048 input becomes, for each token:
  queries  32 heads x 64 numbers  (q_proj 2,048 -> 2,048)
  keys      8 heads x 64 numbers  (k_proj 2,048 -> 512)
  values    8 heads x 64 numbers  (v_proj 2,048 -> 512)
query heads 0-3 read key/value head 0, 4-7 head 1, ... 28-31 head 7
the cache keeps 8 layers x (keys + values) x 8 heads x 64 x 2 bytes = 16,384 bytes per token
  with 32 key/value heads: 65,536; at the engine's 16,384-token cap, 1.07 GB, not 0.27 GB
-> chapter 25: each head scores every token against every earlier one; at 8,192 tokens
  that is 8,192 x 8,192 = 67 million scores per head, 32 heads per layer
```

In an attention layer each token makes three vectors: a **query** (what it is
looking for), a **key** (what it offers) and a **value** (what it passes on). It
compares its query with every earlier token's key, and takes a mix of their
values, more of those whose keys match. A **head** is one such set of
query, key and value, 64 numbers each; the layer runs 32 heads side by side.

**Grouped-query attention** in thirty seconds: the 32 query heads do not get 32
sets of keys and values. They share 8: query heads 0 to 3 use key/value head 0,
and so on. Keys and values are what the cache stores for every token, so
sharing them makes the cache 4 times smaller: 16 KB per token instead of 64
KB, 0.27 GB instead of 1.07 GB at the project engine's 16,384-token limit. The
cost of attention itself grows with the square of the length, which is the
memory problem chapter 25 solves when it trains on 8,192-token rows.

### The causal mask: a token never looks ahead

```text
6. the causal mask: a token never looks ahead
head 0 of layer 2: the share each token (row) takes from each token (column),
  in percent; '.' is exactly 0
                        0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
 0 '<|startoftext|>'  100   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
 1 '<|im_start|>'      76  24   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
 2 'user'              28   6  66   .   .   .   .   .   .   .   .   .   .   .   .   .   .
 3 '\n'                 2   8  84   6   .   .   .   .   .   .   .   .   .   .   .   .   .
 4 'How'               24   2   8  62   4   .   .   .   .   .   .   .   .   .   .   .   .
 5 ' many'             42   3  15   8  17  15   .   .   .   .   .   .   .   .   .   .   .
 6 ' customers'        32  10   7   1   5  28  17   .   .   .   .   .   .   .   .   .   .
 7 ' live'             45  11   4   0   4  19  10   7   .   .   .   .   .   .   .   .   .
 8 ' in'               41   4   1   0   1  14  23  14   1   .   .   .   .   .   .   .   .
 9 ' Paris'            24   6   1   0   2   6  35  19   6   1   .   .   .   .   .   .   .
10 '?'                 39   2   1   1   5  22  12   9   2   0   6   .   .   .   .   .   .
11 '<|im_end|>'        12   1   2   1  10  62   4   4   1   0   3   0   .   .   .   .   .
12 '\n'                28   7   9   2   4  32   5   4   1   0   5   1   1   .   .   .   .
13 '<|im_start|>'      33  16   6   8   2   3   9   3   1   1  11   2   3   1   .   .   .
14 'assistant'         19   2   5   0  10  50   5   5   0   0   2   0   0   0   1   .   .
15 '\n'                 7   1   6   1  16  48   5   4   0   2   4   3   0   0   2   2   .
16 '<think>'            6   1   3   2  14  45   3   4   0   2   5   4   1   0   6   4   1
every row adds up to 100 (give or take rounding); above the diagonal the
  scores were set to minus infinity before the softmax
computed by hand from the queries, keys and values, the layer's output matches
  mlx-lm's own to within 0.00003 (its largest value is 0.179)
```

Each row is one token, and each number is how much of its attention goes to
the token in that column. The scores come from the queries and keys, and
**softmax** (chapter 2) turns each row into shares that add up to 100%. Above
the diagonal every cell is exactly 0: before the softmax, each token's scores
for later tokens are set to minus infinity. That is the **causal mask**, and it
is what makes the model a writer: when it predicts the next token, it can only
use what came before, because that is all it will have when it writes. (This
head sends a lot to `<|startoftext|>` and to `' many'`; heads find their own
uses, and most are not easy to name.)

### Positions: RoPE

```text
7. positions: RoPE (rotary position embedding)
each head's 64 numbers form 32 pairs, and each pair is turned by an angle
  that grows with the token's position
rope_theta 10,000,000 sets the speeds: the fastest pair turns once every 6.3 positions,
  the slowest once every 38.0 million, so all 131,072 positions look different
the same query (' Paris') and key (' customers'), moved to other positions:
  3 apart, at 9 and 6           12.3054
  3 apart, at 1,009 and 1,006   12.3054  the same: only the distance counts
  30 apart                       9.6253
  300 apart                      3.7263
  3,000 apart                    0.7043
```

**RoPE** in thirty seconds: attention by itself does not know the order of the
tokens; the mix would be the same if you shuffled them. RoPE (rotary position
embedding) fixes that by turning each query and key a little, by an angle that
grows with the token's position, before they are compared. Each head's 64
numbers are 32 pairs, each turning at its own speed: the fastest once every
6.3 positions, the slowest once every 38 million, set by `rope_theta`
(10,000,000). The score between two tokens then depends on how far apart they
are, not on where they sit: the same query and key, 3 apart, score 12.3054 at
positions 9 and 6 and at positions 1,009 and 1,006, and less the farther apart
they are placed. With these speeds the model was built for 131,072 positions,
chapter 9's context window.

### Out: the same table scores every token

```text
8. out: the same table scores every token
after the last layer and a final norm, the last position holds one vector of 2,048
  numbers; its score for each of the 128,000 tokens is its dot product
  with that token's row of the embedding table
tie_word_embeddings true: the table that turned ids into vectors turns the vector
  back into scores, and no separate head is stored
vector x table, by hand: the same 128,000 scores as mlx-lm (largest difference 0)
the 5 highest of 128,000:
  token        score  probability
  'The'        23.25        51.3%
  '\n'        22.875        35.3%
  '1'         21.875        13.0%
  'User'      18.125         0.3%
  'We'         16.75         0.1%
the scores are bfloat16 numbers: between 16 and 32 they come in steps of 1/8
-> next-token prediction: a reply is this step, again and again (first_reply.py)
-> chapter 11 keeps this table at 6 bits, not 4: every score passes through it
-> chapter 25: all 128,000 scores at each of 8,192 positions are 1.05 billion numbers,
  4.19 GB in float32
```

After the 30 layers, the vector at the last position (`<think>`) holds what the
model makes of the whole prompt. To turn it into a choice, the model scores
every token in the vocabulary: the score of a token is the **dot product** of
this vector with the token's row in the embedding table (multiply the two
lists number by number and add up; it is large when they point the same way).
The part that does this is the **output head**, and in LFM2.5 it is the
embedding table itself, used backwards: `tie_word_embeddings`. Scoring by hand
with the table gives exactly mlx-lm's scores. One table doing both jobs saves
262 million numbers; chapter 11 keeps it at 6 bits instead of 4 because every
score passes through it, and chapter 25 meets its size again, because training
scores all 128,000 tokens at every position.

Softmax turns the scores into probabilities: the model thinks its reasoning
most likely starts with "The" (51%). This is **next-token prediction**, the
only thing a language model does: score every token for the next position,
pick one, add it to the text, and ask again. The scores show bfloat16 once
more: between 16 and 32 they can only be multiples of 1/8.

## The first reply

```bash
uv run first_reply.py "How many customers live in Paris? Answer with one SQL query."
```

`first_reply.py` asks that question, one token at a time, with mlx-lm's
`stream_generate`, and prints each piece the moment its token is predicted:

````text
first_reply.py · LFM2.5-2.6B answers, one token at a time (bfloat16)
model     ~/.cache/mhrl-course/models/LFM2.5-2.6B (5.39 GB of weights in memory)
warm-up   1.23 s for one untimed token: the first pass is slower (one-time GPU set-up)
question  How many customers live in Paris? Answer with one SQL query.
prompt    23 tokens, ending with <think>: the reply starts inside its reasoning
decoding  greedy: the most likely token every time (chapter 12 samples)

the reply, printed as each token is predicted:

The user wants to know the number of customers living in Paris.
The constraint is to answer with *one* SQL query.
I need to assume a standard schema for a "customers" table.
Typical columns would be `customer_id`, `name`, `city`, `country`, etc.
The query should count rows where the city is 'Paris'.

Possible SQL queries:
1. `SELECT COUNT(*) FROM customers WHERE city = 'Paris';`
2. `SELECT COUNT(*) FROM customers WHERE city = 'Paris' AND country = 'France';` (more specific, but maybe not strictly necessary if 'Paris' is unique enough).
3. `SELECT COUNT(*) FROM customers WHERE city = 'Paris';` is the most direct answer.

Let's verify the prompt: "How many customers live in Paris? Answer with one SQL query."

The query `SELECT COUNT(*) FROM customers WHERE city = 'Paris';` is standard and correct.

Wait, should I consider case sensitivity? Usually, SQL is case-insensitive for string literals in many dialects, but 'Paris' is standard.
Should I include a `LIMIT`? No, the question asks for the count.

Final check:
- One SQL query? Yes.
- Answers the question? Yes.

Query: `SELECT COUNT(*) FROM customers WHERE city = 'Paris';`</think>```sql
SELECT COUNT(*) FROM customers WHERE city = 'Paris';
```

the reply: 286 tokens, then <|im_end|>, its end of turn (finish_reason stop)
first token after 0.17 s: reading the 23-token prompt, then one prediction
then 51.2 ms per token (19.5 tokens a second), one at a time
peak memory 5.45 GB (MLX's count): the 5.39 GB of weights, the cache and working space
the first 5 tokens, as they arrived:
      ms      id  piece
     169     597  'The'
      48    4695  ' user'
      57   10966  ' wants'
      58     310  ' to'
      48    1341  ' know'

saved ~/.cache/mhrl-course/10-meet-lfm/first_reply-bfloat16.json:
  chapter 11's 'before' picture
````

The prompt is chapter 9's chat template around the question, 23 tokens, and
it ends with `<think>`, so the model reasons first: 271 tokens of working-out
up to `</think>`, then the answer, then `<|im_end|>`, its end of turn, which
stops the loop. Each token is the previous section's step: score
all 128,000, take the top one (**greedy**: no chance involved, so the reply is
the same on every run; chapter 12 draws tokens by their probabilities
instead), add it, ask again. The answer is right: on chapter 1's shop
database that query counts the 13 customers who live in Paris.

The numbers are the "before" picture for chapter 11, and the times are from
one run on an M3 Pro with 18 GB; yours will differ, and so will these from run
to run (the reply itself will not). The first token takes 0.17 s, because the
model first reads the whole 23-token prompt. Every token after that takes
about 50 ms: for each one the laptop reads all 5.39 GB of weights once more,
which is why generation is slow and why smaller weights make it faster. The
very first pass after loading is slower again (about a second here), so the
script runs one untimed token first. Peak memory is 5.45 GB: the weights plus
a little for the cache and the work.

The same question with `--fast` runs the 4-bit export; its reply reasons for
fewer tokens and gives the same query:

```text
the reply: 144 tokens, then <|im_end|>, its end of turn (finish_reason stop)
first token after 0.06 s: reading the 23-token prompt, then one prediction
then 16.3 ms per token (61.4 tokens a second), one at a time
peak memory 1.70 GB (MLX's count): the 1.58 GB of weights, the cache and working space
```

Under a third of the memory and about three times the speed. Whether its
answers are as good is chapter 11's question.

## What the chapter showed, and where it differed from the plan

- **About 2.70 B parameters**: 2,697,198,592, as planned: 1,981,808,640 in the
  feed-forward blocks (1.98 B), 369,233,920 in the 22 convolution layers (0.37
  B), 83,887,104 in the 8 attention layers (0.08 B) and 262,144,000 in the
  128,000 x 2,048 embedding table (0.26 B), which is also the output head. The
  plan left out the norms: 124,928 more numbers, which round to nothing. The
  model card says 2.69B, the name 2.6B.
- **About 5.4 GB in bfloat16**: 5,394,397,184 bytes of numbers (5.39 GB); the
  whole repository is 5.41 GB.
- **Attention at layers 2, 5, 9, 13, 17, 21, 24 and 27**: as planned, counting
  from 0 as `config.json` does.
- **32 query heads sharing 8 key/value heads**: as planned, 64 numbers per head.
- **rope_theta 1e7 and 131,072 positions**: as planned. In this repository's
  `config.json` `rope_theta` sits only inside `rope_parameters` (the 4-bit
  export's `config.json` also has a top-level copy).
- **The license**: the LFM Open License v1.0, and the threshold is annual
  revenue of $10,000,000 or more (for the company and the companies it controls
  or is controlled by). Non-profits are exempt for non-commercial and research
  use.
- **The fast tally's packed-column rule**: real columns = stored columns x 32 /
  bits gives the 4-bit export exactly the same 2,697,198,592 numbers, part for
  part; counting stored values would say 438,041,600.
- **`first_reply.py`**: on this run, 286 tokens of reply at 51.2 ms per token,
  0.17 s to the first token, peak memory 5.45 GB. Added while building: one
  untimed warm-up token, because the first pass after loading took about a
  second longer and would have been counted as reading the prompt.
- **inspect.py is called `inspect_model.py`**: a script named `inspect.py`
  cannot run, because Python would load it in place of its own `inspect`
  module, which `dataclasses` imports.
- **The repository's files**: as planned, plus `model.safetensors.index.json`,
  the model card and `.gitattributes`.

## What is in this folder

- `download.py`: the download (`snapshot_download` at a pinned revision), the
  SHA-256 check of every file, the clear message when the Hub cannot be
  reached, and the license's commercial terms quoted from the file.
- `inspect_model.py`: reads `config.json` and the safetensors headers, counts
  every tensor by part (unpacking packed columns in the 4-bit export) and
  checks each part against `config.json`.
- `inside.py`: one prompt through the model, stage by stage: embedding,
  bfloat16, layer kinds and the cache, grouped-query attention, the causal
  mask, RoPE, and the tied output head.
- `first_reply.py`: the reply token by token, timed, with peak memory; saves
  the measurement for chapter 11.
- `lfm_files.py`: where the files come from (both repositories, pinned file by
  file) and where they live under `COURSE_HOME`.
- `lfm.py`: loads the model with mlx-lm and renders a question into prompt
  tokens the way the project's engine does.
- `report.py`: how the scripts print.

## Words from this chapter

- **Repository, revision**: a model's folder of files on the Hub, and one saved
  version of it.
- **safetensors**: the file format of the weights: a header naming each tensor,
  then the numbers.
- **Tensor, vector, matrix**: a block of numbers; a list of numbers; a table of
  numbers.
- **Parameter (weight)**: one number the model learned; LFM2.5-2.6B has
  2,697,198,592.
- **bfloat16, float32**: two ways to store a number, in 16 or 32 bits; bfloat16
  keeps float32's range with fewer digits, so small differences round away.
- **Embedding table**: one row of 2,048 numbers per token id; a token goes into
  the model as its row.
- **Layer**: one stage of the model that turns the prompt's matrix into a new
  one; LFM2.5-2.6B has 30.
- **Convolution layer, attention layer**: a layer that mixes each token with
  the two before it, and one that lets each token look at every earlier token;
  22 and 8 of them (LFM2's **hybrid** design).
- **Feed-forward block**: the part of every layer that reworks each token on
  its own; 73.5% of the numbers.
- **Query, key, value, head**: what a token looks for, offers and passes on in
  attention; a head is one such set, 32 per layer here.
- **Grouped-query attention**: 32 query heads sharing 8 key/value heads, which
  makes the cache 4 times smaller.
- **Causal mask**: the rule that a token never looks at later tokens.
- **RoPE**: positions told apart by turning queries and keys by an angle that
  grows with the position.
- **Output head, tied embeddings**: the part that scores every token for the
  next position; here it is the embedding table itself.
- **Dot product**: multiply two vectors number by number and add up.
- **Next-token prediction**: score every token, pick one, add it, repeat; all a
  language model does.
- **Cache**: what the model keeps from the tokens so far, so it never re-reads
  them.
- **Base model**: in this course, LFM2.5-2.6B as downloaded, before our
  training.
- **Model license**: the terms that come with open weights; here, commercial
  use only for companies under $10 million in yearly revenue.

Next: chapter 11, Quantize it yourself: 16 bits to 4.
