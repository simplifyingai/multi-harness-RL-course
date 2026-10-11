# Chapter 9. From four choices to 128,000: tokens and chat templates

In chapter 2 the agent had four choices: queries A, B, C and D. The model this
course trains, LFM2.5-2.6B, has 128,000. It never picks a whole answer; it
writes one small piece of text at a time, and each piece is one of those
128,000 choices. So before the model itself arrives in chapter 10, this chapter
looks at what it reads and writes: the pieces, called tokens, and the template
that turns a whole conversation into one long row of them.

Here is a surprise to keep in mind. The model can write the file name
`answer.txt` as any one of 168 different sequences of tokens, and the text
comes out the same every time. Nothing in the saved text tells you which
sequence it wrote.

## What you skipped, if you came the RL-veteran way

The RL-veteran route reads chapters 1, 5 and 7, then joins here. This is what
chapters 2, 3, 4, 6 and 8 built, in the words the rest of the course uses. If
you read them, skip to [Run it](#run-it).

- **Chapter 2, plain RL.** An agent that knows no SQL picks one of four
  ready-made queries for a question, and chapter 1's grader pays 1 for a right
  answer and 0 for a wrong one. Its **policy** is a table of scores, one row
  per question. **Softmax** turns a row into chances that add up to 100%, and
  the agent **samples** a query by those chances. **Temperature** sharpens or
  flattens the chances; 0 means always the top score (**greedy**). After each
  try the row moves by `lr * reward * (onehot(choice) - probabilities)`. In a
  few hundred tries it learns which query to trust, and it answers questions
  about cities it never trained on.
- **Chapter 3, policy gradient.** That rule is gradient descent on one number,
  the **loss** `-advantage x log p(chosen)`. A **log-probability** is the log
  of a chance: 0 for certain, more negative for less likely. The
  **advantage** is the reward minus a **baseline**, what the agent expected,
  so an action that did better than expected becomes more likely and one that
  did worse less likely. A learning rate that is too big lets the first lucky
  try decide everything.
- **Chapter 4, several calls, one reward.** The agent acts over up to 8
  **calls** (look at a table, run a query, write answer.txt, finish) and is
  graded once, at the end. One full attempt is a **rollout**; one update of the
  weights is a **step**. Nobody knows which call earned the reward, so every
  call in a rollout gets the same advantage, as in the project's training. A
  test on held-out cities tells learning the method (**generalisation**) apart
  from remembering answers (**memorisation**).
- **Chapter 6, the importance ratio and the clip.** When rollouts come from
  older weights, GRPO weighs each call by `new p / old p`, the **importance
  ratio**, worked out from the log-probability recorded when the call was
  sampled, and clips it to between 0.8 and 1.2. The project samples with the
  newest weights and updates once, so its ratio stayed near 1.000; only the
  recorded log-probabilities can show that.
- **Chapter 8, imitation (SFT).** **Supervised fine-tuning** trains on a
  teacher's demonstrations, making `-log p(teacher's action)` smaller at each
  call, and never trains on what the environment sends back. The project's SFT
  used 120 demonstrations, batch 8, one epoch: 15 steps. A student copies
  whatever its examples share, so a teacher that always takes four calls
  teaches "stop on call 4", not "stop after saving". SFT before RL gives RL
  rollouts that sometimes succeed, and so something to learn from at step 1.

Everything so far ran on toys: a handful of actions and a small table of
weights. From here on the agent is a real language model.

## Run it

```bash
git clone https://github.com/simplifyingai/multi-harness-RL-course.git
cd multi-harness-RL-course/09-tokens-and-templates
uv run tokens.py
```

Each script takes a second or two on any Mac. They need two small libraries,
`tokenizers` (cuts text into tokens) and `jinja2` (fills in templates), which
`uv run` installs for the script alone. The first run downloads the model's
tokenizer files, about 18 MB; no weights, and no GPU. Like every chapter, the
scripts write only under `COURSE_HOME` (default `~/.cache/mhrl-course`), never
into this folder.

| command | what it shows |
| --- | --- |
| `tokens.py` | a question cut into tokens and put back together, the vocabulary, and the special tokens |
| `tokens.py "any text"` | the same for your own text |
| `template.py` | three chats rendered into prompts: one message, the same with a bash tool, and three user turns |
| `spellings.py` | every way to spell `answer.txt` in tokens, and how many ways there are for what an agent writes |
| `spellings.py "any text"` | the same for your own text (up to 200 characters) |
| `count.py` | three calls of a tiny agent: how the prompt grows, and the context window |

Every script also takes `--fast`, which skips the pauses between sections.

## Three files from the Hugging Face Hub

The Hugging Face Hub (huggingface.co) is a website where model makers publish
each model as a folder of files, each with its own web address, so a program
can download exactly the files it needs. This chapter downloads three of
LFM2.5-2.6B's files, from Liquid AI's repository `LiquidAI/LFM2.5-2.6B`:

- `tokenizer.json`: the vocabulary and the rules for cutting text into tokens;
- `tokenizer_config.json`: which tokens start a prompt, end a turn and pad;
- `chat_template.jinja`: the recipe that turns a conversation into one string.

The first run of any script prints this:

```text
downloading from the Hugging Face Hub: LiquidAI/LFM2.5-2.6B, tokenizer files only
  into ~/.cache/mhrl-course/09-tokens-and-templates/LFM2.5-2.6B
  tokenizer.json            17,905,598 bytes  SHA-256 checked
  tokenizer_config.json            363 bytes  SHA-256 checked
  chat_template.jinja            5,443 bytes  SHA-256 checked
  17.9 MB in all; the 5.4 GB of weights stay on the Hub
```

The files come from one fixed revision of the repository, so they can never
change under the course, and each is checked against its SHA-256 fingerprint
(a 64-character code that changes if a single byte does) before it is kept. A
download that breaks off leaves nothing behind. If the Hub cannot be reached,
the script says which file it was fetching, why it failed, and to check the
connection and run it again.

The project runs Liquid's 4-bit export of the same model,
`LiquidAI/LFM2.5-2.6B-MLX-4bit` (chapter 11 makes our own). Its
`tokenizer.json` and `tokenizer_config.json` are these exact files, and its
`chat_template.jinja` only adds a one-line comment at the top, which renders
to nothing. The weights wait for chapter 10.

## Text to tokens and back

```bash
uv run tokens.py
```

```text
tokens.py · LFM2.5-2.6B's tokenizer: text to token ids and back

text    How many customers live in Paris?
      3522  'How'
      1196  ' many'
     10699  ' customers'
      3737  ' live'
       296  ' in'
      4741  ' Paris'
        39  '?'
ids     [3522, 1196, 10699, 3737, 296, 4741, 39]
back    How many customers live in Paris?  (the same text)
7 tokens for 33 characters
```

A **token** is a small piece of text with a number, its **id**. The
**tokenizer** is the program that cuts text into tokens (**encoding**) and puts
ids back together into text (**decoding**). The model only ever sees the ids:
`[3522, 1196, 10699, 3737, 296, 4741, 39]` is what it reads for this question,
and a list of ids is all it writes.

Common words are one token each. The space before a word belongs to the word,
so `' many'` (with its space) is one token and `'How'` (at the start, with no
space) another. A rarer text takes more, smaller pieces: `uv run tokens.py
"answer.txt"` shows `'answer'` and `'.txt'`. Decoding gives back exactly the
text that went in.

```text
the vocabulary: 125,017 entries, ids 0 to 125,016
  124,893 pieces learned from text, ids 0 to 124,892
      124 tokens added on top, ids 124,893 to 125,016
the model keeps 128,000 rows, one per id: 2,983 are never used
```

The **vocabulary** is the list of every token the tokenizer knows: 125,017 of
them. Most were learned from a large amount of text (the next sections show
how); 124 were added by hand, and they are the special tokens below. The model
has one row of numbers for every id, and it was built with 128,000 rows (the
`vocab_size` in its `config.json`, which chapter 10 reads). No text ever turns
into the last 2,983 ids, so those rows are never used.

So here is the title: chapter 2's agent kept a score for each of 4 queries
and picked one. Every time LFM2.5-2.6B writes a token, it gives each of the
128,000 rows a chance and picks one (chapter 12 shows how). Its policy has the same shape as chapter 2's, with
128,000 actions instead of four.

## Special tokens

```text
special tokens: each is one token, never cut into pieces
  <|startoftext|>       124894  the start of every prompt
  <|im_start|>          124899  a message starts; its role follows
  <|im_end|>            124900  a message ends: the model's end of turn
  <think>               124901  the model's reasoning starts
  </think>              124902  its reasoning ends
  <|tool_call_start|>   124905  a tool call starts
  <|tool_call_end|>     124906  a tool call ends
  <|pad|>               124893  filler that lines up sequences of different lengths
  and 116 more the course never uses (105 for images)
```

**Special tokens** mean structure, not words. Each is written out like a tag,
but it is a single token with its own id, and the tokenizer never cuts it up.
`<|im_end|>` (124900) matters most: when the model writes it, its turn is over,
and the program stops asking it for more tokens. The other 116 added tokens are
for things this course does not use, such as images (105 of them), audio, and
filling in the middle of code.

One detail for later chapters: `tokenizer.json` flags 120 of the 124 added
tokens as "special". `<think>`, `</think>` and the two tool-call markers are
not flagged, so a decoder told to skip special tokens keeps them. The project
skips none: it decodes with `skip_special_tokens=False`, and so does this
chapter.

## Messages and the chat template

A program talking to a model sends a conversation as a list of **messages**,
each with a **role** and its **content**:

- **system**: instructions for the whole conversation;
- **user**: the person, or the program speaking for them, such as the task;
- **assistant**: the model's own earlier replies;
- **tool**: what a tool printed. A **tool** is a function the model may ask
  the program to run, such as `bash` with a command; chapter 16 shows the loop.

This is the "OpenAI-style" format, the one the project's engine accepts. The
model reads one row of tokens, though, not a list. Each model ships with a
**chat template**, `chat_template.jinja`, that writes the messages one after
another between special tokens. Jinja is a small template language: text with
blanks, ifs and loops. Filling it in is called **rendering**, and in Python it
is `tokenizer.apply_chat_template(messages, ...)`.

```bash
uv run template.py
```

```text
1. one user message
messages:
  user       How many customers live in Paris?
the prompt, 17 tokens:
<|startoftext|><|im_start|>user
How many customers live in Paris?<|im_end|>
<|im_start|>assistant
<think>
the last 4 tokens are the generation prompt (add_generation_prompt=True):
'<|im_start|>' 'assistant' '\n' '<think>', the cue that the model's reply comes next
```

The prompt starts with `<|startoftext|>`. The message sits between
`<|im_start|>user` and `<|im_end|>`. Then comes the **generation prompt**, which
`add_generation_prompt=True` adds: the start of an assistant message, left
open. A model only continues text, so this is how it knows it is its turn. For
LFM2.5 the open message already holds `<think>`.

```text
2. the same message, with a bash tool (mini-swe-agent's)
the prompt, 88 tokens:
<|startoftext|><|im_start|>system
List of tools: [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]}}}]<|im_end|>
<|im_start|>user
How many customers live in Paris?<|im_end|>
<|im_start|>assistant
<think>
the tool list goes into a system message: 71 tokens more
the prompt ends with <think> (124901): every reply starts inside its reasoning
```

The tool is mini-swe-agent's `bash` tool, exactly as that harness sends it. The
template writes the tool definitions as text, `List of tools: [...]`, inside a
system message, and they cost 71 tokens on every call. Both prompts end with
`<think>`, so whatever the model writes first is its **reasoning**: private
working-out that ends with `</think>`, followed by the answer or a tool call
(chapter 17 shows a tool call token by token).

## Old reasoning is dropped, unless you keep it

```text
3. three user turns: is the reasoning of earlier replies kept?
messages:
  user       How many customers live in Paris?
  assistant  <think>The customers table has a city column.</think>SELECT COUNT(*) FROM customers WHERE city = 'Paris';
  user       And in Berlin?
  assistant  <think>The same query, another city.</think>SELECT COUNT(*) FROM customers WHERE city = 'Berlin';
  user       Which city has the most customers?
the template's default, 70 tokens:
<|startoftext|><|im_start|>user
How many customers live in Paris?<|im_end|>
<|im_start|>assistant
SELECT COUNT(*) FROM customers WHERE city = 'Paris';<|im_end|>
<|im_start|>user
And in Berlin?<|im_end|>
<|im_start|>assistant
SELECT COUNT(*) FROM customers WHERE city = 'Berlin';<|im_end|>
<|im_start|>user
Which city has the most customers?<|im_end|>
<|im_start|>assistant
<think>
with preserve_thinking=True, as the project renders, 89 tokens:
<|startoftext|><|im_start|>user
How many customers live in Paris?<|im_end|>
<|im_start|>assistant
<think>The customers table has a city column.</think>SELECT COUNT(*) FROM customers WHERE city = 'Paris';<|im_end|>
<|im_start|>user
And in Berlin?<|im_end|>
<|im_start|>assistant
<think>The same query, another city.</think>SELECT COUNT(*) FROM customers WHERE city = 'Berlin';<|im_end|>
<|im_start|>user
Which city has the most customers?<|im_end|>
<|im_start|>assistant
<think>
the ids agree for the first 16 tokens, then part: the default dropped 19 tokens of reasoning
the rule: a reply keeps its <think> block only if it comes after the last user message;
preserve_thinking=True keeps every reply exactly as the model wrote it
```

Each assistant message holds the model's whole reply as it wrote it, reasoning
included. By default the template throws away the reasoning of every reply
that comes before the last user message: here both replies lose their
`<think>` block, and the prompt is 19 tokens shorter. With
`preserve_thinking=True` every reply is kept as written. The two prompts agree
for 16 tokens and then part, so from that point every id differs.

The project renders with `preserve_thinking=True` (`mhrl/engine/policy.py`,
`render`). An agent's rollout usually has one user message, the task, and all
its replies come after it, so the default would keep them anyway; `count.py`
below shows that. But a harness can add a user message in the middle:
mini-swe-agent does when a reply has no tool call, to say so. With the default,
every earlier reply would lose its reasoning at that moment, and the next
prompt would no longer start with the tokens the model actually wrote. Later
chapters need exactly those tokens: chapter 12 reuses work already done on a
prompt's start, and chapter 15 trains on the ids the model sampled.

## One text, many spellings

```bash
uv run spellings.py
```

```text
'answer.txt'
  the tokenizer writes [44244, 31544]: 'answer' '.txt'
  5 spellings use 3 tokens:
    [73, 64507, 31544]   'a' 'nswer' '.txt'
    [279, 64618, 31544]  'an' 'swer' '.txt'
    [497, 2388, 31544]   'ans' 'wer' '.txt'
    [44244, 22, 62758]   'answer' '.' 'txt'
    [44244, 4203, 741]   'answer' '.t' 'xt'
  each decodes to 'answer.txt'; encode that again and you get [44244, 31544]
  in all, 168 token sequences spell 'answer.txt':
    tokens         2     3     4     5     6     7     8     9    10
    spellings      1     5    15    30    42    41    25     8     1
```

The tokenizer cuts `answer.txt` into `[44244, 31544]`, `'answer'` and `'.txt'`.
But `'.'` (22) and `'txt'` (62758) are tokens too, so `[44244, 22, 62758]` also
spells `answer.txt`, and so do 166 other sequences, down to ten single
characters.
The tokenizer always picks the same one. The model does not have to: it writes
one token at a time, and nothing stops it from writing `'answer'`, then `'.'`,
then `'txt'`.

Either way the file says `answer.txt`. But if you take that text and encode it
again, you get `[44244, 31544]`, not the ids the model wrote. Turning saved
text back into tokens can give ids the model never produced. Chapter 15 shows
why training on those is wrong, and it is why the project's capture proxy
(chapter 20) keeps the ids themselves instead of re-tokenizing text.

```text
how many ways to spell what an agent writes
  text                                                  tokens         spellings
  How many customers live in Paris?                          7        50,012,160
  sqlite3 data.db .tables                                    7           124,416
  SELECT COUNT(*) FROM customers WHERE city = 'Paris'       11   295,384,320,000
  echo 13 > answer.txt                                       6            10,752
  cat answer.txt                                             3             1,344

where the spellings come from: BPE (byte pair encoding) built the
vocabulary by gluing pairs of pieces that often sit side by side
  118,295 of the 123,332 pieces made of whole characters
  are two shorter pieces glued together
  (1,561 more hold part of a character: emoji, other scripts)
the tokenizer always picks one spelling; a model that writes one token at
a time can produce any of them
```

This is not special to file names. A 51-character SQL query has 295 billion
spellings. The reason is how the vocabulary was made: **BPE** (byte pair
encoding) starts from single bytes and keeps gluing the pair of pieces that
most often sit side by side into a new piece. So almost every piece is two
smaller pieces that are also in the vocabulary. (A character such as an emoji
takes several bytes, and 1,561 pieces hold only some of them; the search
leaves those out, which is exact for plain English text.)

## The context window

```bash
uv run count.py
```

```text
three calls of a tiny agent on the shop database
(the replies are scripted; the tool outputs are what sqlite3 prints)
call   reads  writes  tool call
   1     105      26  bash(command='sqlite3 data.db .tables')
   2     150      37  bash(command='sqlite3 data.db "SELECT COUNT(*) FROM customers WHERE city = \'Paris\'"')
   3     198      30  bash(command='echo 13 > answer.txt')
call 2 reads call 1's 105 tokens, its 26-token reply, and 19 more: the tool's output and the markers around it
each prompt starts with the whole call before it, reply included: yes
3 calls read 453 tokens in all, to write 93
the template's default gives the same ids here: every reply comes after the only user message
```

A model remembers nothing between calls. To make call 2, the harness sends the
whole conversation again: the tools, the task, call 1's reply and what the
tool printed, and the template renders all of it again. So the prompt grows
with every call, from 105 tokens to 150 to 198, and three calls read 453
tokens to write 93. The replies here are written for the example; the tool
outputs are what `sqlite3` prints for the shop database, which the agent sees
as `data.db`.

Each prompt starts with every token of the call before it, its reply included.
That holds because the replies are put back exactly as written, reasoning and
all. This rollout has one user message, so the template's default would give
the same ids here.

```text
call 3's prompt, in full:
<|startoftext|><|im_start|>system
List of tools: [{"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]}}}]<|im_end|>
<|im_start|>user
There is a SQLite database named data.db. How many customers live in Paris? Write the answer to answer.txt.<|im_end|>
<|im_start|>assistant
<think>First, which tables are in this database?</think><|tool_call_start|>[bash(command='sqlite3 data.db .tables')]<|tool_call_end|><|im_end|>
<|im_start|>tool
customers    order_items  orders       products<|im_end|>
<|im_start|>assistant
<think>The customers table should have a city column.</think><|tool_call_start|>[bash(command='sqlite3 data.db "SELECT COUNT(*) FROM customers WHERE city = \'Paris\'"')]<|tool_call_end|><|im_end|>
<|im_start|>tool
13<|im_end|>
<|im_start|>assistant
<think>
```

This is what the model reads for its third call: system, user, then assistant
and tool in turn, and the open assistant message at the end. Each reply ends
with `<|im_end|>`, the token the model wrote to end its turn.

```text
the context window: the most tokens the model reads in one call
  LFM2.5-2.6B reads up to 131,072 positions, one token each
  the project's engine refuses any prompt over 16,384 tokens
  call 3's 198 tokens use 1.2% of the engine's cap
```

Everything a call sends has to fit in the **context window**: the most tokens
the model can read at once. LFM2.5-2.6B was built for 131,072 positions, one
token each (`max_position_embeddings` in its `config.json`; chapter 10). The
project's engine sets its own, lower limit: it refuses any prompt over 16,384
tokens (`max_prompt_tokens` in `mhrl/config.py`; chapter 18 builds the engine).
Three short calls use 1.2% of that. One careless command can use all of it:

```text
the command that blew the window: in Pi, the untrained model often
read the binary data.db itself (here with bash: cat data.db)
  the file (shop.db here) is 45,056 bytes; read as text it is 35,044 tokens
  the next prompt is 35,180 tokens, 2.1 times the engine's cap
  the engine answers: prompt is 35180 tokens; the limit is 16384
  (it fits in the model's own 131,072 positions)
```

In the project, the untrained model inside the Pi harness often read the
database file itself and blew its context (the project's README). A database
file is not text, and read as text the shop database is 35,044 tokens. The
next prompt would be more than twice the engine's limit, and the engine
answers with the error above (its own words, from `mhrl/engine/server.py`).

## The same render as the project, without transformers

The project renders with the `transformers` library's
`tokenizer.apply_chat_template`. That call does two jobs, and for each it uses
a smaller library:

- it fills in the Jinja template with `jinja2`, set up a particular way: in
  Jinja's sandbox (the template is a file from the internet), with
  `trim_blocks` and `lstrip_blocks` (which decide every newline and space), a
  `{% generation %}` tag the template uses, its own `tojson` (Jinja's sorts the
  keys and escapes `<`, `>`, `&` and `'`), and a `raise_exception` the template
  can call to refuse a chat;
- it cuts the string into tokens with `tokenizers`.

`lfm.py` does the same two jobs with those two libraries, set up the same way,
so this chapter needs no `transformers` (a much bigger library, which also
carries the code of hundreds of models). The tests prove the result is the
same: `tests/data/09_project_render.json` holds the token ids the project's own
`PolicyEngine.render` produced for 14 chats (this chapter's, plus tool calls,
wide characters and a binary file's bytes), made with the project's
transformers 5.18.0, tokenizers 0.23.2 and jinja2 3.1.6, and every one of them
matches `lfm.py`'s ids exactly.

## What the chapter showed, and where it differed from the plan

- **The vocabulary**: 125,017 entries, and the model keeps 128,000 rows, as
  planned. The 128,000 is in the model's `config.json`, not in the three
  tokenizer files, so `tokens.py` quotes it.
- **The special-token ids**: all eight as planned.
- **"78 tokens in all"** for one user message with a bash tool did not hold:
  it is 88 with mini-swe-agent's bash tool as that harness sends it. 78 is the
  count for a smaller tool, the same one without the description of its
  `command` argument. The ending, `<|im_start|>assistant` and `<think>`, is as
  planned.
- **Old `<think>` blocks dropped unless `preserve_thinking`**: holds for
  replies before the last user message (89 tokens kept, 70 dropped). Replies
  after it keep their reasoning either way, so in the agent's rollout, with one
  user message, the ids are the same.
- **A 3-turn conversation growing every call**: 105, 150, then 198 tokens.
- **`answer.txt`**: `[44244, 31544]`, and `[44244, 22, 62758]` decodes to the
  same text, as planned; 168 spellings in all.
- **131,072 positions and the 16,384 cap**: as planned; neither is in the
  tokenizer files (`config.json` and the project's `mhrl/config.py`).
- **About 18 MB**: 17.9 MB.
- **Added while building**: the overflow from reading the database file, the
  project's real failure in Pi, measured on the shop database.

## What is in this folder

- `tokens.py`: a text cut into tokens and back, the vocabulary and the special
  tokens.
- `template.py`: three chats rendered into prompts, and the `preserve_thinking`
  rule.
- `spellings.py`: every spelling of a text, counted by length, and the BPE
  pieces they come from.
- `count.py`: three calls of a tiny agent, the context window, and the
  overflow.
- `lfm.py`: the model's tokenizer and chat template loaded with `tokenizers`
  and `jinja2`, set up as transformers sets them up; `render` is the project's
  `PolicyEngine.render`, line for line, and `load_tokenizer` is how every
  script starts (download if needed, then load).
- `hub.py`: the download of the three files: pinned revision, SHA-256 check,
  and a clear message when the Hub cannot be reached.
- `chats.py`: the chats, and mini-swe-agent's bash tool.
- `report.py`: how the scripts print.
- `data/shop.db`: chapter 1's shop database, read by `count.py` as text.

## Words from this chapter

- **Token, id**: a small piece of text, and the number the model knows it by.
- **Tokenizer**: turns text into ids (encoding) and ids back into text
  (decoding).
- **Vocabulary**: every token the tokenizer knows; 125,017 here, in a model
  with 128,000 rows.
- **BPE (byte pair encoding)**: how the vocabulary was learned, by gluing the
  pairs of pieces that most often sit side by side into new pieces.
- **One text, many spellings**: the same characters can be written as
  different token sequences, so re-encoding saved text can give ids the model
  never wrote.
- **Special tokens**: tokens that mean structure, not words: where a prompt and
  a message start and end, where reasoning and tool calls start and end.
- **End of turn**: `<|im_end|>`, the token that ends a message; when the model
  writes it, its reply is over.
- **Messages and roles**: a conversation as a list of messages, each from the
  system, the user, the assistant (the model) or a tool; the whole list is
  sent again on every call.
- **Tool**: a function the model may ask the program to run, such as `bash`;
  chapter 16 shows the loop.
- **Chat template**: the model's Jinja file that writes the messages, the tool
  list and the cue for the reply into one string.
- **Generation prompt**: the open assistant message at the end of a prompt, the
  cue that it is the model's turn; for LFM2.5 it ends with `<think>`.
- **Reasoning (`<think>` block)**: the model's private working-out before it
  answers. The template drops it from replies before the last user message,
  unless `preserve_thinking` is set, as the project sets it.
- **Context window**: the most tokens a model can read in one call; 131,072
  for LFM2.5-2.6B, capped at 16,384 by the project's engine.
- **Hugging Face Hub**: the website where models are published as folders of
  files that programs download.

Next: [chapter 10](../10-meet-lfm/), Meet LFM2.5-2.6B.
