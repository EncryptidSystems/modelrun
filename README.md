# ModelRun

ModelRun is the Hodge-Kuramoto runtime. A packed linear runs as `hodgelinear.wgsl` on Dawn, limbs included. A dense bf16 linear runs as `dense.wgsl`. LUT kernels are the WGSL files Dawn loads as written. The `model_type` stamp in `config.json` is not the runtime name.

A checkpoint is a safetensors file or a BNZ4 file, plus its tokenizer and `config.json`. Chat text is rendered from that tokenizer's own pieces: role, llama3, gemma, qwen, harmony, or plain. `modelrun list` shows the Dawn clock and the WGSL kernels. A checkpoint is opened with `--config`.

Commands the model asks for run in the TunnelTug guest. They do not run on this machine.

## Build

Go 1.24. The module replaces `hodgemix` with a checkout at `../hodgemix-go`.

```
go build -o modelrun.exe ./cmd/modelrun
```

The same module builds `cochainkey` and `cochainimg`:

```
go build -o cochainkey.exe ./cmd/cochainkey
go build -o cochainimg.exe ./cmd/cochainimg
```

Hodge dispatches need `dawnrt.exe`. The resident Dawn host is `dawnhost.exe`. The Dawn DLLs sit next to each of those programs. LUT runs and the clock need WASMApollo's `dawn.exe`.

## Paths

Defaults sit under the home directory. An environment variable replaces the whole path.

| Variable | Default | Used for |
| --- | --- | --- |
| `MODELRUN_APOLLO` | `~/WASMApollo` | WGSL tree. Also sets `dawn.exe` when `MODELRUN_DAWN` is empty |
| `MODELRUN_DAWN` | `~/WASMApollo/dawn/dawn.exe` | LUT runner and the clock |
| `MODELRUN_ENCRYPTID` | `~/EncryptidSystems/stage-ai` | Kernel directory `kernels/` |
| `MODELRUN_DAWNHOST` | `~/ModelRun/dawnhost/dawnhost.exe` | Resident Dawn host |
| `MODELRUN_DAWNRT` | `~/ModelRun/dawnrt/dawnrt.exe` | Device process for a Hodge dispatch |
| `MODELRUN_SKILLS` | `~/ModelRun/skills` | Skill files |
| `MODELRUN_SEARCH` | `https://ai.encryptid.systems` | Fact lookup |
| `MODELRUN_VM` | `http://127.0.0.1:8870/mcp` | TunnelTug guest |

Kernels the runtime loads from `kernels/` include `hodgelinear.wgsl`, `dense.wgsl`, `dot4_soft.wgsl`, `embed.wgsl`, `norm.wgsl`, `rope.wgsl`, `attention.wgsl`, `lmhead.wgsl`, and `wedge.wgsl`.

## Checkpoint file

`modelrun run --config` reads the first entry. Paths are relative to the config file. Unknown fields are an error.

```yaml
models:
  - model: model.safetensors
    config: config.json
    tokenizer: tokenizer.json
    tokenizer_config: tokenizer_config.json
    generation_config: generation_config.json
```

`model` is safetensors or BNZ4. `generation_config` and `special_tokens_map` may be omitted. `config.json` supplies `hidden_size`, `num_hidden_layers`, and `vocab_size`.

## Commands

```
modelrun list
modelrun show <model>
modelrun ps
modelrun run [-n N] [-temp T] [-top-p P] --config FILE [text]
modelrun run [-n N] [-temp T] [-top-p P] <model> [text]
modelrun serve [-addr 127.0.0.1:11434]
```

`list` and `ls` print the clock, then every WGSL kernel whose name contains lut, fabric, ternary, hodge, embed, dot4, or kuramoto. `show` prints one catalog row. `ps` prints models loaded in this process.

`run` without a prompt, on a terminal, opens the chat screen. A pipe with an empty prompt is read as the prompt.

Omit `-n` and the answer runs until the model stops. `-n N` caps the length. Omit `-temp` and `-top-p` and the checkpoint sampler is used. `-temp 0` is greedy.

`modelrun run dawn/clock` runs the Dawn clock. `modelrun run apollo/<kernel>` loads that WGSL file in `dawn.exe`. Those catalog runs are one shot. They are not a chat checkpoint.

## Chat screen

The screen is the checkpoint path. Enter sends. Shift+Enter or Alt+Enter starts a new line. The wheel scrolls the transcript. Esc twice quits. Ctrl+C stops the current reply, and quits when nothing is running.

| Input | Effect |
| --- | --- |
| `/tools` | List exec, search, and gather |
| `/skills` | List skill files |
| `/skill name` | Turn that skill on |
| `/skill` | Turn the skill off |
| `/run …` or `/exec …` | Run that command in the guest |

A skill is a `.md` file in the skills directory. The first heading is dropped. The first remaining line is the summary.

Before a checkpoint answers, a short pass names the question as fact, recipe, command, or talk. A fact is looked up, then answered from those pages. A recipe is the model's ingredients and method. A command runs in the guest, then the model answers in plain words. Talk is streamed as the model writes it. A talk answer that misses the question is looked up and replaced.

## HTTP

`modelrun serve` listens on `127.0.0.1:11434`. The routes are the Ollama set. `stream` is off unless the request sets it to true. A stream is newline-delimited JSON. The `model` field is a catalog name from `modelrun list`.

| Method | Path |
| --- | --- |
| GET | `/api/version` |
| GET | `/api/tags` |
| GET | `/api/ps` |
| POST | `/api/show` |
| POST | `/api/chat` |
| POST | `/api/generate` |

`/api/chat` reads `messages`. `/api/generate` reads `prompt` and does not apply the chat template. Sampler fields live under `options`: `temperature`, `top_p`, and `num_predict`.

```
curl -s http://127.0.0.1:11434/api/tags
curl -s http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"dawn/clock\",\"stream\":false}"
```

## Watermark

When `model.cochain.safetensors` sits beside the tokenizer, each checkpoint reply is scored against that cochain. `modelrun run` prints the reading on stderr:

```
watermark  clear  words 40  alignment 0.120  null 0.080  cell 0.0100  cell_null 0.0040
```

The first word is `marked`, `clear`, or `short` (under 24 words). The HTTP object carries the same reading as `watermark` when the model has a cochain.

## Corpus tools

These two programs read a cochain. They are not the chat runtime.

`cochainkey` prints the shared sign key of a folder of `.txt` files. Its config is the checkpoint file above, and it also accepts `tensor` when the safetensors file has more than one matrix. `--tensor` overrides that field.

```
cochainkey --config FILE CORPUS
```

`cochainimg` draws one chain per text, then the shared centroid. The model directory contains `tokenizer.json`, `tokenizer_config.json`, and `model.cochain.safetensors`.

```
cochainimg -model MODEL -dir DIR -out cochains.png
```

## Tree

```
cmd/modelrun     CLI and chat screen
cmd/cochainkey   corpus sign key
cmd/cochainimg   cochain picture
internal/hodge   checkpoint runtime
internal/api     HTTP routes
dawnhost/        resident Dawn host
dawnrt/          Hodge device process
```
