# Chapter 12. How the model picks each token

Chapters 10 and 11 let LFM2.5-2.6B answer by always taking its single most
likely next token, so the same question got the same reply every time. That
hides what the model really hands back at every step: not one token, but a
score for each of the 128,000 tokens it knows. This chapter opens that step.
It runs the same kind of loop as the project's engine
(`mhrl/engine/policy.py`), keeps everything the model says at each step, and
shows the choices that change which token comes out: temperature, top-k,
top-p, the length cap, and the precision the numbers are kept in. Then it
times the two halves of every reply, reading the prompt and writing the
answer, with and without the cache that makes writing fast.

The puzzle to keep in mind: ask the model the same question twice, drawing
its tokens by chance, and you get two different replies. Which numbers must
you keep about each one, so that a training run can later tell how likely
the model found it? By the end of the chapter you will have the answer, and
the three ways to get it subtly wrong.

## Words first

- A **token probability** is the share of 100% the model gives one token as
  the next piece of text. At each step the model scores every token
  (chapter 10's output head), and the softmax (chapter 2) turns the 128,000
  scores into 128,000 probabilities that add up to 100%. Together they are
  the **distribution** of the next token.
- A **log-probability** is the natural log of a probability (chapter 3):
  0 for a certain token, more negative for a less likely one (-0.69 is 50%,
  -2.30 is 10%, -4.61 is 1%). Adding log-probabilities multiplies
  probabilities, so a whole reply's log-probability is the sum of its tokens'.
  These are the numbers reinforcement learning (RL) trains with: chapter 6's
  importance ratio is made of them.
- **Entropy** measures how spread out a distribution is: minus the sum, over
  all tokens, of probability x log-probability. It is 0 when one token has all
  the probability, and largest, ln 128,000 = 11.76, when all 128,000 are
  equally likely. Its unit, with natural logs, is the **nat**.
- **Temperature** is a number T the scores are divided by before the softmax.
  Below 1 it sharpens the distribution (the likeliest token gets likelier);
  above 1 it flattens it. At T = 1 nothing changes. **Greedy** decoding skips
  the draw and always takes the likeliest token, which chapters 10 and 11 did.
- **Top-k** keeps only the k likeliest tokens and gives every other token
  probability 0; the kept ones are then scaled up so they add up to 100%
  again (**renormalized**). **Top-p** keeps the fewest likeliest tokens whose
  probabilities add up to at least p (0.9 means 90%), and renormalizes them.
- **finish_reason** says why a reply ended: `stop` when the model wrote its
  end-of-turn token `<|im_end|>` (chapter 9), `length` when the reply reached
  the most tokens it was allowed, the **length cap**, and was cut off.
- **Prefill** is reading the prompt: many tokens go through the model in one
  call. **Decode** is writing the reply: one token per call, because each
  token depends on the one before it.
- The **KV cache** is what the model keeps from the tokens it has already
  read, so a new token does not make it read them all again. In LFM2 the 8
  attention layers keep each token's keys and values (chapter 10's attention
  heads), and the 22 convolution layers keep only their last few inputs.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/12-generate
uv run sample_lab.py
```

Both scripts need an Apple-silicon Mac and the 4-bit model: chapter 11's own
build in `COURSE_HOME/models/lfm2.5-2.6b-4bit-ours` (default
`~/.cache/mhrl-course`), or, if you have not built it, Liquid's export that
chapter 10's `download.py --fast` fetched. The two are the same bytes
(chapter 11 checked). Nothing is downloaded. What the chapter writes goes
under `COURSE_HOME/12-generate`.

| command | what it shows |
| --- | --- |
| `sample_lab.py` | step 1's likeliest tokens, the same step at three temperatures, then a whole reply drawn token by token with every number kept |
| `sample_lab.py --temperature 0.5` (or `1.5`) | the reply drawn from a sharper (or flatter) distribution |
| `sample_lab.py --greedy` | always the likeliest token: the same reply twice |
| `sample_lab.py --max-tokens 20` | a reply cut off by the cap |
| `sample_lab.py --top-k 50` (or `--top-p 0.9`) | drawing from fewer tokens than all of them, and what that changes |
| `sample_lab.py --raw --temperature 0.7` | the drawn tokens' numbers before and after the temperature |
| `sample_lab.py --dtype bf16` | the same numbers kept in 16-bit bfloat16 instead of 32-bit float32 |
| `timing.py` | the first token's wait after prompts of 500, 2,000 and 8,000 tokens, and writing speed with and without the cache (about 2 minutes) |

Every number below comes from recorded runs of these commands on an M3 Pro
with 18 GB (each with its commit, its date and each command's time).
`sample_lab.py` prints no times, and its draws come from a fixed seed (`--seed`,
default 0), so every run of the same command prints the same output (the run
log ran the first command twice: byte for byte the same). `timing.py` prints
only times; they will differ on your machine and from run to run.

**Fast mode.** `uv run sample_lab.py --fast` uses Liquid's export even when
chapter 11's build is there, and prints the same reply. `uv run timing.py
--fast` times only the 500- and 2,000-token prompts, with fewer tokens (about
20 seconds).

## One step: 128,000 scores become a distribution

```bash
uv run sample_lab.py
```

`sample_lab.py` loads the 4-bit model and asks chapter 10's question. It
starts by saying how it will draw:

```text
sample_lab.py · how the model picks each token
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours
          ours, built by chapter 11's convert.py
question  How many customers live in Paris? Answer with one SQL query.
prompt    23 tokens, ending with <think>
drawing   temperature 1, the full distribution (no top-k, no top-p); seed 0
scores    turned into log-probabilities in float32, as the project's engine does (its
  _to_float32)
cap       1,024 tokens: the project's output cap
the model's suggested settings (generation_config.json): temperature 0.1, top_k 50,
  repetition_penalty 1.1; the project's engine uses none of them
```

The prompt is chapter 9's chat template around the question, 23 tokens,
ending with `<think>`, so the reply starts inside the model's reasoning. The
draw uses the full distribution at temperature 1, which is how the project
draws the replies it trains on (`train_temperature 1.0` in `mhrl/config.py`;
it evaluates greedily, `eval_temperature 0.0`). The last line is the model's
own suggestion, which `sample_lab.py` reads from the build's
`generation_config.json`: a temperature of 0.1, top-k 50 and a repetition
penalty of 1.1 (a rule that makes tokens already written less likely). The
project's engine uses none of the three; the section on top-k says why.

Then the first step, before any token is drawn:

```text
1. step 1: the 10 likeliest next tokens, of 128,000 (the distribution the first token is drawn from)
  rank      id  token               probability  log-probability
     1     597  'The'                    86.76%          -0.1420
     2      18  '*'                      11.74%          -2.1420
     3    2083  'We'                      1.24%          -4.3920
     4    8914  'User'                    0.10%          -6.8920
     5      25  '1'                       0.06%          -7.3920
     6    9077  'First'                   0.04%          -7.7670
     7   36508  'Question'                0.02%          -8.6420
     8      49  'I'                       0.01%          -8.9545
     9   92895  'Hmm'                    <0.01%         -10.3295
    10      48  'H'                      <0.01%         -10.6420
  the other 127,990 tokens: 0.02% together
  entropy 0.450 nats (0 if one token were certain; 11.76, the log of 128,000, if all
    were equally likely)
```

The model is fairly sure its reasoning starts with "The" (86.76%), and gives
"*" a real chance (11.74%). Below the first few rows the probabilities fall
fast: the other 127,990 tokens share 0.02%. The log-probabilities say the same
thing on a scale that does not run out of room: -0.14 for the favourite,
-10.64 for the tenth. The entropy, 0.450 nats, is small next to the 11.76 of
a distribution where every token is equally likely: one step, nearly decided.

## Temperature: sharper or flatter

The second section takes the same 128,000 scores and divides them by three
temperatures before the softmax. Nothing is drawn; it is arithmetic on
step 1:

```text
2. the same step at 3 temperatures: the scores divided by T, then the softmax (float32)
  temperature     'The'       '*'      'We'  tokens holding 99%    entropy
          0.5    98.18%     1.80%     0.02%                   2   0.092 nats
          1.0    86.76%    11.74%     1.24%                   3   0.450 nats
          1.5    71.92%    18.96%     4.23%                 542   1.107 nats
  below 1 the likeliest token takes more of the probability (sharper); above 1 the rest
    take more (flatter)
```

At 0.5 "The" takes 98.18% and two tokens hold 99% of the probability. At 1.5
it drops to 71.92%, and it takes 542 tokens to hold 99%: the long tail of
unlikely tokens is no longer negligible. The entropy goes from 0.092 to 0.450
to 1.107 nats.

What that does to a whole reply shows best at the edges. At temperature 0.5
the reply stays on track and ends with the same query, after 348 tokens:

```bash
uv run sample_lab.py --temperature 0.5
```

```text
4. how the reply ended
  348 tokens; the last, <|im_end|> (id 124900), is the end-of-turn token: finish_reason
    'stop'
  the reply's log-probability, the sum of its 348 log-probabilities: -119.8724
    the chance of drawing exactly this reply: e^-119.8724, about 1 in 10^52.1
  entropy per step: 0.284 nats on average; 202 of 348 steps below 0.1 nats (one token
    held nearly all the probability)
```

At 1.5 the reply falls apart within its first few lines, because once in a
while the draw now picks a token from that 542-token tail, and every odd token
makes the next one stranger:

```bash
uv run sample_lab.py --temperature 1.5
```

```text
*   Question: "How many customers live in Paris?"
    *   Constraint: "Answer with one SQL query."

    *   The request implies a database exists or exists implicitly (which can't be physically provided, so I must write a query based on a standard hypothetical structure for customer data).
    *   *Assillage assumptions:*
        *   There is a table containing customer information (e.g., `customers`).
        *   There's a way to filter at exact match AR [["email_ageageid_timesportun',膚?itis g],[jour '/ใกล้Pod عكسgsbe)
 verb）
)→ mature嫽 Palma
ようション...
°.
```

It recovers, astonishingly, and ends with the right query after 738 tokens:

````text
Final selection: `SELECT COUNT(*) FROM customers WHERE city = 'Paris';`</think>```sql
SELECT COUNT(*) FROM customers WHERE city = 'Paris';
```<|im_end|>

4. how the reply ended
  738 tokens; the last, <|im_end|> (id 124900), is the end-of-turn token: finish_reason
    'stop'
  the reply's log-probability, the sum of its 738 log-probabilities: -4933.4863
    the chance of drawing exactly this reply: e^-4933.4863, about 1 in 10^2142.6
  entropy per step: 6.566 nats on average; 62 of 738 steps below 0.1 nats (one token
    held nearly all the probability)
````

The average entropy per step is 6.566 nats, against 0.356 at temperature 1
(below). The reply's log-probability, -4933.49, is what you would expect from
738 tokens of which many were long shots.

## A reply, token by token

Back to the first command. Its third section draws the reply, one token per
step, and prints for each the drawn token's probability, its
log-probability, the running sum of the log-probabilities so far, and the
step's entropy. The first 12 steps and the last 4 are printed; all 178 are
saved in a file:

```text
3. the reply, drawn token by token
  step      id  token               probability     log-prob  running sum  entropy
     1      18  '*'                      11.74%    -2.142012      -2.1420    0.450
     2     266  '  '                     99.96%    -0.000393      -2.1424    0.004
     3   29155  ' Question'              98.26%    -0.017578      -2.1600    0.108
     4      34  ':'                     100.00%     0.000000      -2.1600    0.000
     5     496  ' "'                     97.97%    -0.020542      -2.1805    0.099
     6    3522  'How'                   100.00%     0.000000      -2.1805    0.000
     7    1196  ' many'                 100.00%    -0.000002      -2.1805    0.000
     8   10699  ' customers'            100.00%    -0.000004      -2.1805    0.000
     9    3737  ' live'                 100.00%    -0.000004      -2.1805    0.000
    10     296  ' in'                   100.00%     0.000000      -2.1805    0.000
    11    4741  ' Paris'                100.00%    -0.000004      -2.1805    0.000
    12    6930  '?"\n'                   99.73%    -0.002682      -2.1832    0.020
    ... 162 steps not shown (all are in the saved file)
   175     951  " '"                    100.00%    -0.000002     -57.2379    0.000
   176   41176  'Paris'                 100.00%    -0.000011     -57.2379    0.000
   177   44243  "';"                     99.99%    -0.000099     -57.2380    0.001
   178  124900  '<|im_end|>'            100.00%    -0.000004     -57.2380    0.000
```

The first draw took "*", the second likeliest token (11.74%): drawing by
chance means sometimes taking a token that is not the favourite, which is
what makes two replies to the same question differ. Then the model quotes the
question back, almost every token above 99%, and the running sum barely
moves. Here is the whole reply:

````text
the reply's text:

*   Question: "How many customers live in Paris?"
    *   Constraint: "Answer with one SQL query."

    *   I need a table that contains customer information (likely `customers` or `orders` or `users`).
    *   I need to count the rows where the customer's location is 'Paris'.
    *   Common table name: `customers`.
    *   Column names: likely `customer_id`, `city`, or similar. Let's assume `city = 'Paris'`.
    *   SQL syntax: `SELECT COUNT(*) FROM customers WHERE city = 'Paris';`

    *   Does it use one SQL query? Yes.
    *   Is it the only output? Yes.

    *   `SELECT COUNT(*) FROM customers WHERE city = 'Paris';`</think>SELECT COUNT(*) FROM customers WHERE city = 'Paris';<|im_end|>
````

and how it ended:

```text
4. how the reply ended
  178 tokens; the last, <|im_end|> (id 124900), is the end-of-turn token: finish_reason
    'stop'
  the reply's log-probability, the sum of its 178 log-probabilities: -57.2380
    the chance of drawing exactly this reply: e^-57.2380, about 1 in 10^24.9
  entropy per step: 0.356 nats on average; 95 of 178 steps below 0.1 nats (one token
    held nearly all the probability)

saved every step in ~/.cache/mhrl-course/12-generate/sample_lab-t1-seed0.jsonl
```

Three things to read here.

- **The end.** The last token drawn is `<|im_end|>` (id 124900), the model's
  end-of-turn token. The loop stops there with finish_reason `stop`. It is
  part of the reply: its log-probability counts, and RL trains on it like any
  other token.
- **The reply's log-probability.** -57.2380 is the sum of 178
  log-probabilities, the log of the product of 178 probabilities: the chance
  of drawing exactly this reply is about 1 in 10^24.9. That sounds absurd and
  is normal. There are astronomically many replies the model could write,
  and each single one is unlikely; what matters to RL is how these numbers
  move when the model is trained.
- **The entropy.** 95 of the 178 steps had an entropy below 0.1 nats: one
  token held nearly all the probability. The uncertain steps, a few dozen,
  are where replies part ways.

This list of numbers, one per drawn token, is what chapter 6's importance
ratio is built from: training recomputes each token's log-probability under
the model being trained and compares it with the one recorded here
(chapter 15 does it for real). For that comparison to mean anything, the
recorded number must come from the distribution the token was actually drawn
from. The rest of the chapter is three ways to break that.

The loop itself is short. Its heart is the function that turns the model's
own log-probabilities into the distribution a token is drawn from, from
`sample_lab.py` with its docstring left out:

```python
def drawn_from(logprobs: mx.array, draw: Draw) -> mx.array:
    if draw.greedy or (draw.temperature == UNCHANGED and not draw.cuts()):
        return logprobs  # nothing to change (greedy reports these, as the engine does)
    scores = logprobs / draw.temperature
    if draw.top_k != NO_TOP_K:
        scores = apply_top_k(scores, draw.top_k)
    if draw.top_p != NO_TOP_P:
        scores = apply_top_p(scores, draw.top_p)
    return scores - mx.logsumexp(scores, axis=-1, keepdims=True)
```

`sample()` hands mlx-lm's `generate_step` a sampler that draws from
`drawn_from(...)`, then calls the same function on the same numbers again to
record the distribution that was drawn from. Subtracting the `logsumexp` (the
log of the sum of every token's e^score) is the softmax written in logs: it
makes the probabilities add up to 1 again.

## Greedy: the same reply twice

```bash
uv run sample_lab.py --greedy
```

Greedy takes the likeliest token every time, so there is no chance involved,
and `--greedy` runs the whole reply a second time, from scratch, to check:

```text
4. how the reply ended
  145 tokens; the last, <|im_end|> (id 124900), is the end-of-turn token: finish_reason
    'stop'
  the reply's log-probability, the sum of its 145 log-probabilities: -36.3036
    the chance of drawing exactly this reply: e^-36.3036, about 1 in 10^15.8
  entropy per step: 0.521 nats on average; 60 of 145 steps below 0.1 nats (one token
    held nearly all the probability)

5. greedy again: a second run from scratch
  145 tokens, identical to the first run's 145: yes; its log-probability -36.3036
```

The same 145 tokens twice: chapter 11's `ask.py` reply (144 tokens) and its
end of turn. The greedy reply starts with "The", the token the sampled reply
passed over. The project evaluates this way, so every evaluation of a given
model and question gives the same answer.

## How a reply ends: stop or length

The cap is the most tokens a reply may have. Give it 20:

```bash
uv run sample_lab.py --max-tokens 20
```

````text
the reply's text:

*   Question: "How many customers live in Paris?"
    *   Constraint: "Answer

4. how the reply ended
  20 tokens, the cap, and the last is not the end-of-turn token (id 124900):
    finish_reason 'length', the reply cut off mid-thought
  the reply's log-probability, the sum of its 20 log-probabilities: -2.4174
    the chance of drawing exactly this reply: e^-2.4174, about 1 in 11
  entropy per step: 0.062 nats on average; 17 of 20 steps below 0.1 nats (one token held
    nearly all the probability)
````

The same draw as before (the first 20 tokens are the default run's first 20,
by the saved files), stopped in the middle of its reasoning, at the word
"Answer": finish_reason `length`. A harness that receives such a reply
gets half a thought, and a trainer that receives it gets a reply that never
ended.

The project gives every reply the same cap, 1,024 tokens, in training and in
evaluation (`max_completion_tokens` in `mhrl/config.py`, whose comment says
why), and `sample_lab.py` uses the same cap by default. If training allows
longer replies than evaluation, the model learns to write answers that
evaluation then cuts off.

> **From the article, not from our runs.** The article reports that its
> earlier runs, with a Qwen model, trained with replies of up to 16,384
> tokens and evaluated with a cap of 4,096, and that 556 of 1,000 test cells
> hit the cap. The project's equal 1,024-token caps exist to avoid that.

## Top-k and top-p: drawing from fewer tokens

```bash
uv run sample_lab.py --top-k 50
```

The model's makers suggest top-k 50 in `generation_config.json`, with a
temperature of 0.1 and a repetition penalty of 1.1; a program that follows
the model's suggested settings draws this way. Here is top-k 50 at the
project's training temperature, 1. The step table gains a column, **uncut**:
the same token's log-probability in the full distribution:

```text
3. the reply, drawn token by token
  step      id  token               probability     log-prob  running sum  entropy       uncut
     1      18  '*'                      11.74%    -2.141925      -2.1419    0.448   -2.142012
     2     266  '  '                     99.96%    -0.000392      -2.1423    0.004   -0.000393
     3   29155  ' Question'              98.26%    -0.017578      -2.1599    0.108   -0.017578
     4      34  ':'                     100.00%    -0.000001      -2.1599    0.000    0.000000
     5     496  ' "'                     97.97%    -0.020544      -2.1804    0.099   -0.020542
     6    3522  'How'                   100.00%     0.000000      -2.1804    0.000    0.000000
     7    1196  ' many'                 100.00%    -0.000002      -2.1804    0.000   -0.000002
     8   10699  ' customers'            100.00%    -0.000004      -2.1804    0.000   -0.000004
     9    3737  ' live'                 100.00%    -0.000003      -2.1804    0.000   -0.000004
    10     296  ' in'                   100.00%     0.000000      -2.1804    0.000    0.000000
    11    4741  ' Paris'                100.00%    -0.000003      -2.1805    0.000   -0.000004
    12    6930  '?"\n'                   99.73%    -0.002680      -2.1831    0.020   -0.002682
    ... 162 steps not shown (all are in the saved file)
   175     951  " '"                    100.00%    -0.000001     -57.1970    0.000   -0.000002
   176   41176  'Paris'                 100.00%    -0.000011     -57.1970    0.000   -0.000011
   177   44243  "';"                     99.99%    -0.000100     -57.1971    0.001   -0.000099
   178  124900  '<|im_end|>'            100.00%    -0.000004     -57.1971    0.000   -0.000004
  uncut: the same token's log-probability before the cut
```

The two columns differ in the sixth decimal, and the summary says by how
much:

```text
5. what top-k 50 changed, at each of the 178 steps
  the drawn token's probability, as drawn vs uncut: higher at 138 steps, the same at 10,
    lower at 30 by float32 rounding alone (by at most 0.000002)
  the kept tokens held 99.98% of the probability on average (the lowest 99.36%); the cut
    divides every kept probability by that share
  entropy: 0.356 nats on average uncut, 0.353 as drawn; lower at 158 of 178 steps
  the reply's log-probability: -57.1971 as drawn, -57.2380 under the model's full
    distribution
```

With the same seed, top-k 50 drew exactly the same 178 tokens as the full
distribution (the run log compares the saved files). What changed are the
numbers recorded for them:

- **The drawn token's probability is not the model's own.** Cutting all but
  50 tokens and renormalizing divides every kept probability by the share
  the 50 held: 99.98% on average here, 99.36% at the lowest. So the drawn
  token's recorded log-probability is a little higher than the model's at 138
  of 178 steps, the same at 10 (where the 50 held everything float32 can
  tell apart), and lower at 30 only by float32 rounding, never more than
  0.000002 (a cut can only raise the kept probabilities).
- **The entropy is lower**: at 158 of the 178 steps, and on average from
  0.356 to 0.353 nats.

On this question, at this temperature, top-k 50 barely matters: the model is
so sure that the 50 likeliest tokens hold nearly everything. Top-p 0.9 cuts
much harder:

```bash
uv run sample_lab.py --top-p 0.9
```

```text
5. what top-p 0.9 changed, at each of the 202 steps
  the drawn token's probability, as drawn vs uncut: higher at 171 steps, the same at 31
  the kept tokens held 97.56% of the probability on average (the lowest 90.07%); the cut
    divides every kept probability by that share
  entropy: 0.357 nats on average uncut, 0.242 as drawn; lower at 202 of 202 steps
  the reply's log-probability: -57.3345 as drawn, -62.4277 under the model's full
    distribution
```

Here the kept tokens held 97.56% on average and as little as 90.07%; every
one of the 202 steps lost entropy (0.357 to 0.242 nats on average); and the
reply's log-probability is -57.33 as recorded against -62.43 under the
model's own distribution. The reply itself went its own way after 59 tokens.

Why this matters for RL: the trainer recomputes each token's log-probability
from the model, which knows nothing of top-k or top-p, so it computes the
uncut number. If the recorded one came from a cut distribution, the two
disagree before any training has happened, and the importance ratio starts
below 1. Its average is exactly the kept share, 0.9756 for this top-p run.
Chapter 15 measures that ratio in training. So the project's engine draws
from the full distribution (`_sampler` in `mhrl/engine/policy.py`: "no top-p,
no top-k") and ignores the model's suggested settings.

### Entropy collapse

The entropy of the distribution a model draws from is how much it still
explores: high, and it tries different replies; near 0, and it writes the
same reply every time. RL training tends to push it down, because it makes
the rewarded replies likelier, and if it goes too far the model stops trying
anything new. That is **entropy collapse**, and it starves the method of
chapter 5: when every reply in a group is the same, every reply gets the same
reward, and there is nothing to compare. Top-k and top-p lower the entropy
before training even starts, as just measured; chapter 15 logs the entropy at
every training step to watch for the slide.

## Raw and processed log-probabilities

```bash
uv run sample_lab.py --raw --temperature 0.7
```

At a temperature other than 1 there are two candidates for "the drawn
token's log-probability". The **processed** one comes from the distribution
the token was drawn from, the scores divided by 0.7. The **raw** one comes
from the model's own scores, as if the temperature were 1. `--raw` shows the
raw one beside the processed one:

```text
3. the reply, drawn token by token
  step      id  token               probability     log-prob  running sum  entropy         raw
     1     597  'The'                    94.35%    -0.058142      -0.0581    0.227   -0.142012
     2    4695  ' user'                  99.96%    -0.000443      -0.0586    0.004   -0.006714
     3   10966  ' wants'                 74.59%    -0.293206      -0.3518    0.739   -0.456041
     4     310  ' to'                    99.58%    -0.004170      -0.3560    0.030   -0.027521
     5    1341  ' know'                 100.00%    -0.000019      -0.3560    0.000   -0.000629
     6    1030  ' how'                   36.38%    -1.011266      -1.3672    0.726   -0.941282
     7    1196  ' many'                 100.00%     0.000000      -1.3672    0.000   -0.000008
     8   10699  ' customers'            100.00%     0.000000      -1.3672    0.000   -0.000011
     9    3737  ' live'                 100.00%    -0.000000      -1.3672    0.000   -0.000038
    10     296  ' in'                   100.00%     0.000000      -1.3672    0.000    0.000000
    11    4741  ' Paris'                100.00%    -0.000001      -1.3672    0.000   -0.000057
    12     318  '.\n'                    50.18%    -0.689616      -2.0569    1.230   -0.877300
    ... 329 steps not shown (all are in the saved file)
   342     951  " '"                    100.00%     0.000000    -151.9659    0.000   -0.000002
   343   41176  'Paris'                 100.00%     0.000000    -151.9659    0.000   -0.000013
   344   44243  "';"                    100.00%    -0.000008    -151.9659    0.000   -0.000343
   345  124900  '<|im_end|>'            100.00%     0.000000    -151.9659    0.000   -0.000002
  -0.000000: a log-probability just below 0, not exactly 0
  raw: under the model's own scores, no temperature, nothing cut
```

```text
5. processed and raw log-probabilities (--raw)
  processed: under the distribution each token was drawn from (temperature 0.7); raw:
    under the model's own scores (log_softmax of the scores, temperature 1)
  they differ at 329 of 345 steps; processed is higher at 290, lower at 39
  the reply's log-probability: processed -151.9659, raw -161.1547
```

At temperature 0.7 the likeliest tokens get likelier, so the processed number
is higher at most steps (290 of 345: "The" is 94.35% as drawn, 86.76% raw)
and lower at the rest (39), where the token drawn was not the favourite. The
two sums differ by more than 9. Both look like ordinary log-probabilities,
which is the danger: only the processed one belongs in training, because
only it describes the draw. The project's engine reports
`log_softmax(scores / T)` for this reason (the docstring of its `generate`),
and draws its training replies at temperature 1, where the two are the same.

## bfloat16 or float32

```bash
uv run sample_lab.py --dtype bf16
```

The model computes its scores in bfloat16, chapter 10's 16-bit numbers. The
project's engine casts them to float32 before the softmax (its
`_to_float32`). `--dtype bf16` skips the cast, as the engine once did, and
shows the float32 numbers of the same scores beside the bfloat16 ones:

```text
3. the reply, drawn token by token
  step      id  token               probability     log-prob  running sum  entropy     float32
     1      18  '*'                      11.94%    -2.125000      -2.1250    0.440   -2.142012
     2     266  '  '                    100.00%     0.000000      -2.1250    0.003   -0.000393
     3   29155  ' Question'             100.00%     0.000000      -2.1250    0.092   -0.017578
     4      34  ':'                     100.00%     0.000000      -2.1250    0.000    0.000000
     5     496  ' "'                    100.00%     0.000000      -2.1250    0.080   -0.020542
     6    3522  'How'                   100.00%     0.000000      -2.1250    0.000    0.000000
     7    1196  ' many'                 100.00%     0.000000      -2.1250    0.000   -0.000002
     8   10699  ' customers'            100.00%     0.000000      -2.1250    0.000   -0.000004
     9    3737  ' live'                 100.00%     0.000000      -2.1250    0.000   -0.000004
    10     296  ' in'                   100.00%     0.000000      -2.1250    0.000    0.000000
    11    4741  ' Paris'                100.00%     0.000000      -2.1250    0.000   -0.000004
    12    6930  '?"\n'                  100.00%     0.000000      -2.1250    0.018   -0.002682
    ... 162 steps not shown (all are in the saved file)
   175     951  " '"                    100.00%     0.000000     -56.6250    0.000   -0.000002
   176   41176  'Paris'                 100.00%     0.000000     -56.6250    0.000   -0.000011
   177   44243  "';"                    100.00%     0.000000     -56.6250    0.001   -0.000099
   178  124900  '<|im_end|>'            100.00%     0.000000     -56.6250    0.000   -0.000004
  float32: from the same scores in float32
```

```text
5. log-probabilities in bfloat16 (--dtype bf16), beside float32
  every one of the 178 drawn tokens' bfloat16 log-probabilities is a whole multiple of
    1/8; they take 20 different values
  the same tokens in float32: 132 different values, multiples of 1/524,288
  the drawn tokens' scores ran from 19.125 to 41.500; bfloat16 holds only multiples of
    1/16 from 8 to 16, of 1/8 from 16 to 32 and of 1/4 from 32 to 64
  108 of 178 are exactly 0 in bfloat16 (probability 100%); in float32 those are -0.06221
    to 0.00000
  the reply's log-probability: -56.6250 in bfloat16, -57.2380 in float32
```

The same 178 tokens as the float32 run (the run log compares the saved
files), with very different numbers recorded. Every bfloat16
log-probability is a whole multiple of 1/8: there are only 20 different
values among 178. 108 tokens get exactly 0, "certain", including one the
float32 numbers give -0.06221 (94%). The reason is the size of the scores:
here they run from 19 to 41.5, and between 16 and 32 bfloat16 holds only
multiples of 1/8 (chapter 10 showed it for a score of 16.75). A
log-probability is a score minus the log of the sum of all e^score, both
about that size, so it lands on the same grid. float32's grid, for numbers
that size, is 1/524,288: 132 different values.

In RL this is the importance ratio again. If the recorded numbers are
bfloat16 and the trainer computes float32 ones, they disagree before any
learning, by up to 0.06 at a single token here. The project's engine recorded
bfloat16 numbers in its first RL run, and has cast to float32 since.

## The same bytes, the same reply: --fast

```bash
uv run sample_lab.py --fast
```

`--fast` loads Liquid's export instead of chapter 11's build. The output
differs only in the model lines (the run log diffs the two outputs):

```text
sample_lab.py · how the model picks each token
model     ~/.cache/mhrl-course/models/LFM2.5-2.6B-MLX-4bit
          Liquid's export (--fast)
```

and its saved steps hold the same 178 tokens: chapter 11 found the two files
byte for byte the same, and the same bytes give the same draws.

## Prefill, decode and the cache: timing.py

```bash
uv run timing.py
```

A reply has two halves. **Prefill** reads the prompt; the tokens are all
known, so many go through the model in one call. **Decode** writes the reply,
one call per token. `timing.py` times both on prompts of exactly 500, 2,000
and 8,000 tokens (a made-up customers table with the question after it), and
then decodes without the cache. Every reply is greedy, and every number it
prints is a time, so yours will differ, and so will the next run's:

```text
timing.py · reading the prompt, writing the reply, and the cache
model     ~/.cache/mhrl-course/models/lfm2.5-2.6b-4bit-ours
          ours, built by chapter 11's convert.py; 1.58 GB of weights in memory
warm-up   0.32 s for an untimed short reply (GPU set-up)
prompts   a made-up customers table, cut so that the whole chat is exactly 500, 2,000
  and 8,000 tokens, then the question: 'How many customers live in Paris? Answer with
  one SQL query.'; every reply is greedy
```

### Prefill: reading the prompt

```text
1. prefill: reading the prompt, 1,024 tokens at a time (PREFILL_STEP)
  prompt  read as                    first token after  prompt tokens/s  peak memory
     500  499 + 1                               0.72 s              698      2.30 GB
   2,000  1,024 + 975 + 1                       2.63 s              761      2.60 GB
   8,000  1,024 x 7 + 831 + 1                  10.20 s              784      2.66 GB
  read as: the tokens each model call read; the last prompt token is read by the step
    that predicts the first reply token. Each prompt is read twice, and the faster run
    counts: other work on the GPU slows a run. Peak memory includes the weights
  the same 8,000 tokens in mlx-lm's default 2,048-token chunks (2,048 x 3 + 1,855 + 1):
    first token after 10.35 s, peak memory 2.85 GB
```

The project's engine asks mlx-lm to read prompts 1,024 tokens at a time
(`PREFILL_STEP` in `mhrl/config.py`); mlx-lm reports how far it has read
after each model call, and `timing.py` writes down the chunks. The 8,000-token
prompt is read as seven chunks of 1,024, one of 831, and then its last token,
which goes through the model in the same call that predicts the first reply
token. That is why the time to the first token grows with the prompt: 0.72 s
for 500 tokens, 2.63 s for 2,000, 10.20 s for 8,000, about 700 to 800 prompt
tokens a second whatever the length.

1,024 is the project's choice, not mlx-lm's: `generate_step` reads 2,048 at a
time unless told otherwise. On the 8,000-token prompt the two took about the
same time (10.20 and 10.35 s), and the smaller chunks peaked 0.19 GB lower
(2.66 against 2.85 GB): a chunk's working space grows with its size.

### Decode with the cache

```text
2. decode with the cache: one token at a time, 64 tokens per prompt
  prompt  ms per token  tokens/s  the cache afterwards
     500          16.6      60.2  564 positions: 9.2 MB of keys and values, 180 KB of conv state
   2,000          13.7      73.2  2,064 positions: 33.8 MB of keys and values, 180 KB of conv state
   8,000          14.7      67.8  8,064 positions: 132.1 MB of keys and values, 180 KB of conv state
  both runs of each prompt decoded the same 64 tokens: yes
  ms per token: the median of tokens 2 to 64 (the first is the prefill's), in the
    steadier of the two runs; keys and values: 8 attention layers x 2,048 bytes = 16,384
    bytes per position; conv state: 22 convolution layers, the same size at any length
```

With the cache, writing a token took 13.7 to 16.6 ms whatever the prompt:
flat. The slowest row, the 500-token prompt, is most likely noise: the fast
run at the end of this section measured 13.7 ms for the same prompt. Each new token
reads the cache instead of the prompt, and for this model the cache is small:

- The 8 attention layers keep each token's keys and values: 2 x 8 key/value
  heads x 64 numbers x 2 bytes = 2,048 bytes per layer, 16,384 bytes per
  token. After the 8,000-token prompt and 64 reply tokens that is 132.1 MB,
  next to 1.58 GB of weights that every token reads anyway.
- The 22 convolution layers keep only their last 2 inputs: 180 KB in all,
  the same after 500 tokens as after 8,000.

### Decode without the cache

```text
3. decode without the cache: every token reads the prompt and the reply so far again, from scratch
  prompt  ms per token  tokens/s  the same tokens as with the cache
     500           674      1.48  4 of 4
   2,000         2,546      0.39  4 of 4
   8,000        10,132      0.10  4 of 4
```

Without the cache, every new token means reading the prompt and the reply so
far again from scratch, in the same 1,024-token chunks: each token costs a
whole prefill. So the time per token is about the time to the first token,
0.67 s at 500 tokens and 10.13 s at 8,000, and it grows with the prompt
instead of staying flat. The tokens are the same as with the cache, 4 of 4
at every length: the cache changes the cost, not the reply.

```text
4. what changes with the prompt's length
  time to first token: 0.72 s at 500 tokens, 10.20 s at 8,000 (14.2 times as long for 16
    times the tokens)
  decode with the cache: 16.6 to 14.7 ms per token (0.89 times as long)
  decode without it: 0.67 to 10.13 s per token (15.0 times as long); at 8,000 tokens,
    687 times slower than with the cache
  every number in this output is a time: it varies from run to run and from machine to
    machine
```

At 8,000 tokens, decoding without the cache was 687 times slower. That is
why the project's engine keeps prompt caches between the calls of an agent's
conversation (chapter 18 builds that engine, and also explains MLX's lazy
evaluation, which `timing.py` handles by forcing each computation with
`mx.eval`).

Two things about these measurements. Other programs share the GPU (on this
laptop, video renders did), and a moment of their work makes a run slower, so
`timing.py` reads each prompt twice and keeps the faster prefill and the
steadier decode. And the decode times are medians, so one interrupted token
does not count.

`uv run timing.py --fast` times only the two shorter prompts, with 32 tokens
with the cache and 2 without. Its first two sections, from the same recorded
session:

```text
1. prefill: reading the prompt, 1,024 tokens at a time (PREFILL_STEP)
  prompt  read as                    first token after  prompt tokens/s  peak memory
     500  499 + 1                               0.67 s              751      2.30 GB
   2,000  1,024 + 975 + 1                       2.44 s              818      2.60 GB
  read as: the tokens each model call read; the last prompt token is read by the step
    that predicts the first reply token. Each prompt is read twice, and the faster run
    counts: other work on the GPU slows a run. Peak memory includes the weights
  the same 2,000 tokens in mlx-lm's default 2,048-token chunks (1,999 + 1): first token
    after 2.44 s, peak memory 2.69 GB

2. decode with the cache: one token at a time, 32 tokens per prompt
  prompt  ms per token  tokens/s  the cache afterwards
     500          13.7      73.1  532 positions: 8.7 MB of keys and values, 180 KB of conv state
   2,000          13.5      74.2  2,032 positions: 33.3 MB of keys and values, 180 KB of conv state
  both runs of each prompt decoded the same 32 tokens: yes
  ms per token: the median of tokens 2 to 32 (the first is the prefill's), in the
    steadier of the two runs; keys and values: 8 attention layers x 2,048 bytes = 16,384
    bytes per position; conv state: 22 convolution layers, the same size at any length
```

Here the 500-token prompt decoded at 13.7 ms per token, like the longer one.

## The answer to the puzzle

Two replies to the same question, drawn by chance, differ. To let a training
run tell later how likely the model found each one, keep, for every token it
drew, the token's id and its log-probability under the distribution the
token was actually drawn from, computed in float32. The three ways to get
that number subtly wrong were all measured above, and all three look like
ordinary log-probabilities:

- drawing from a cut distribution (top-k, top-p), so the recorded numbers
  are higher than the model's own;
- recording the raw number at a temperature other than 1, so the numbers
  describe a draw that never happened;
- keeping the numbers in bfloat16, so they land on a grid of 1/8 and many
  likely tokens read as certain.

The project's engine avoids all three: it draws from the full distribution,
records `log_softmax(scores / T)`, and casts the scores to float32 first.

## What the chapter showed, and where it differed from the plan

- **"Each step scores all 128,000 tokens ... until an end-of-turn token or
  the length cap stops it"**: as planned. The distribution has 128,000
  entries; the default reply ended at `<|im_end|>` (id 124900) with
  finish_reason `stop` after 178 tokens, and `--max-tokens 20` with
  `length`.
- **The 10 likeliest next tokens, then each drawn token with its
  log-probability and the running sum**: as planned, with each step's
  entropy added, the first 12 and last 4 steps printed and every step saved.
- **"`--greedy` gives the identical reply twice"**: yes, 145 tokens, checked
  by the script itself.
- **"Temperatures 0.5, 1.0 and 1.5 sharpen and flatten the distribution"**:
  yes: step 1's entropy 0.092, 0.450 and 1.107 nats; at 1.5 the reply derails
  into a jumble of scripts before it recovers.
- **"`--top-k 50` makes the sampled token's probability differ from the
  model's own and lowers the entropy"**: true, and measured to be small on
  this question at temperature 1: the 50 kept tokens held 99.98% of the
  probability on average, the drawn token's recorded probability was higher
  at 138 of 178 steps, and the mean entropy fell only from 0.356 to 0.353
  nats (lower at 158 of 178 steps). Added: `--top-p 0.9`, which shows the
  same effect clearly (kept share 97.56%, entropy 0.357 to 0.242 nats, lower
  at all 202 steps).
- **"the model's own generation_config default, beside a 1.1 repetition
  penalty"**: yes, and the same file also suggests a temperature of 0.1,
  which the plan did not mention; `sample_lab.py` prints all three.
- **"finish_reason 'stop' at `<|im_end|>`; `--max-tokens 20` cuts it
  mid-thought with 'length'"**: as planned.
- **The same 1,024-token cap in training and evaluation**: as planned, from
  `mhrl/config.py`; the article's earlier runs are quoted only in their own
  box.
- **"`--dtype bf16` puts log-probabilities on a coarse 1/16 grid"**: wrong
  for this model: measured, the grid is **1/8**. The grid follows the size of
  the scores, and the 4-bit model's scores here ran from 19 to 41.5, where
  bfloat16 steps are 1/8 (1/4 above 32). 1/16 is the grid for scores between
  8 and 16, which a test checks on scores made up for it. The project's
  comment ("in bf16 every logprob lands on a 1/16 grid") says 1/16 too.
  Also measured: 108 of 178 bfloat16 log-probabilities are exactly 0.
- **"`--raw` at temperature 0.7 shows the untempered value differing from
  the processed one"**: yes, at 329 of 345 steps.
- **"time to first token grows with prompt length (prefill, read in 1,024-token
  chunks)"**: yes, 0.72, 2.63 and 10.20 s; the chunks are read from mlx-lm's
  own progress report, not assumed: 1,024 x 7 + 831 + 1 for 8,000 tokens.
  1,024 is the project's `PREFILL_STEP`; mlx-lm's default is 2,048 (added: the
  same prompt in 2,048-token chunks, about as fast and 0.19 GB higher at its
  peak).
- **"decode speed stays flat with the cache and falls without it"**: yes:
  13.7 to 16.6 ms per token with the cache at every length; without it,
  0.67, 2.55 and 10.13 s per token, the same tokens either way. "Without the
  cache" is done the way an engine without one would do it: each token
  re-reads everything from an empty cache in the same chunks.
- **"shows MLX computing nothing until mx.eval"**: moved to chapter 18, as
  the review asked, with only a mention here.
- **Entropy**: defined here, and entropy collapse explained here, as the
  review asked (the plan had them in chapter 15, where entropy is logged
  during training).
- **Added**: `--seed` (default 0), so a drawn reply can be repeated exactly;
  `--fast` for `sample_lab.py` (Liquid's export, the same output) and for
  `timing.py` (two prompts, fewer tokens); each prompt timed twice, keeping the
  better run, because video renders shared the GPU during the recorded runs.
- **Where the files go**: the plan's `course/12-generate/` is `12-generate/`
  at the root of this repository, like every chapter.

## What is in this folder

- `sample_lab.py`: the loop over mlx-lm's `generate_step` (about 80 lines with
  its data types), keeping every step's full distribution, the drawn token
  and its log-probability, plus the flags and the first step's tables.
- `timing.py`: time to first token for three prompt lengths, the chunks each
  prompt is read in, decode with and without the cache, and what the cache
  holds.
- `measures.py`: running sums, grids, and what a cut or a temperature changed,
  over plain lists of numbers.
- `report.py`: how the scripts print.
- `lfm.py`, `lfm_files.py`: loading the 4-bit model and rendering a prompt,
  and where the model lives (chapter 11's build, else Liquid's export);
  cut-down copies of chapter 11's.

## Words from this chapter

- **Token probability, distribution**: the share the model gives each token
  as the next one; all 128,000 together.
- **Log-probability**: its natural log; a reply's is the sum of its tokens'.
- **Entropy**: how spread out a distribution is, in nats; 0 when one token is
  certain.
- **Entropy collapse**: training pushing the entropy so low that the model
  stops trying different replies.
- **Temperature, greedy**: dividing the scores by T before the softmax;
  always taking the likeliest token.
- **Top-k, top-p**: keeping only the k likeliest tokens, or the fewest
  holding a share p, and renormalizing.
- **Processed and raw log-probabilities**: from the distribution actually
  drawn from, and from the model's own scores; training needs the processed
  one.
- **finish_reason, length cap**: `stop` at the end-of-turn token, `length` at
  the cap.
- **Prefill, decode**: reading the prompt in chunks; writing the reply one
  token at a time.
- **KV cache**: the keys and values (and, in LFM2, convolution state) kept
  from earlier tokens, so a new token does not re-read them.

Next: chapter 13, LoRA: train a 0.4% add-on, not the whole model.
