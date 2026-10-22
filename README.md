
# ⚡ macOS MLX Control Center `v0.11.5` (MLX + Ollama Edition)

> **The simplest, fastest 1-click Web GUI & CLI Control Center to search, download, compare, benchmark, and run local Multimodal Vision & Text LLMs on Apple Silicon M-Series processors.**

Designed specifically for macOS, **macOS MLX Control Center** bridges Apple's high-performance `mlx-lm` (text) and `mlx-vlm` (vision) frameworks with a dark glassmorphic web dashboard and terminal interface. It turns any Mac powered by Apple Silicon M-Series processors into a local AI power-station with zero configuration hassle.

### Dashboard Screenshot
<img width="3522" height="2344" alt="mlx1" src="https://github.com/user-attachments/assets/1a99830a-f565-41df-bc51-399e7f94e568" />
Search Models
<img width="3522" height="2046" alt="mlx2" src="https://github.com/user-attachments/assets/a98012ea-dfbb-4111-8892-3ed57d534f92" />
Compare Models
<img width="2028" height="1098" alt="mlx3" src="https://github.com/user-attachments/assets/129e06b4-a381-47b6-a50c-bda2788fde16" />
Customize API for Local Coders like OpenCode or Pi Code
<img width="1230" height="1512" alt="mlx4" src="https://github.com/user-attachments/assets/d0bf01ba-561b-4e90-adfb-52cff084c617" />


---

## ⚡ 1-Click Launch Options (No Tech Experience Needed!)

### Option A: 1-Line Terminal Instant Launcher (Recommended)
Open Terminal and paste this single command:
```bash
curl -sSL https://raw.githubusercontent.com/cjtsh/mac-mlx-control-center/main/install.sh | bash
```
This installs the **MLX Controller** desktop app, starts the dashboard, and puts a controller alias on your Desktop.

---

### Option B: Clone + Double-Click Installer
1. Download or clone this repository to your Mac.
2. Double-click **`install.command`** (or run `./install.sh` in Terminal).
3. Open **MLX Controller** from Applications or the Desktop alias. It starts the server and opens the dashboard at `http://127.0.0.1:9998`.

---

## 🖥️ MLX Controller Desktop App (New in v0.9.0)

A small native macOS window that owns the server lifecycle, so the dashboard is always one click away:

* **Start / Stop / Restart** the control-center server, with live status (`Stopped`, `Starting`, `Running`, `Couldn’t start`).
* **Open Dashboard** opens `http://127.0.0.1:9998` in your browser.
* **No menu-bar icon.** It lives in the Dock only while open.
* **Safe by design:** it never kills whatever happens to own port 9998. If the port belongs to another app it shows `Port 9998 is in use by Python (PID 9003)` and leaves that process alone. If it finds a control center it did not start, it adopts it and labels it “started outside this app”.
* **Graceful shutdown:** `SIGTERM`, then `SIGKILL` only if the server ignores it. Quitting the app stops a server it owns; an externally started server is left running.
* **Logs** at `~/Library/Logs/MLX Controller/server.log`; the last lines appear in the window when startup fails.
* **Config** at `~/Library/Application Support/MLX Controller/config.json` (repo path, Python path, port, auto-start, stop-on-quit).

Build it yourself with `./desktop/build_app.sh`; `./desktop/build_app.sh --install` installs to `/Applications` and refreshes the Desktop alias. The installer builds from source when Xcode Command Line Tools are present, otherwise it downloads the prebuilt release asset.

---

## 🌟 Why macOS MLX Control Center?

Most local LLM runners are either overly complex, hog system menubars, or require manual hand-editing of configuration JSON files for coding tools. **macOS MLX Control Center** solves this with:

* 🛡️ **No more silent turns (v0.10.0)**: agent harnesses used to give up on a local model whose server went quiet — `mlx-lm` and `mlx-vlm` write nothing at all while a tool call is being built (one measured turn: **96.8 s** of silence) or during a re-prefill. A built-in stream guard now emits SSE keepalives and a guaranteed `[DONE]` from inside the model server process, so DeepSeek Harness, Pi Coder and friends wait instead of dropping the turn. Port, pid and command line are unchanged.
* 🖥️ **MLX Controller desktop app (v0.9.0)**: a native window with Start / Stop / Restart / Open Dashboard, live health status, graceful shutdown, port-conflict safety, and no menu-bar icon. Installed to `/Applications` with a Desktop alias; the installer leaves models and the model server on 9999 untouched.
* 📦 **Serves from your drive (v0.8.5)**: repo-id requests reuse the weights already on the external drive (`HF_HUB_CACHE` is pointed at the library) instead of silently re-downloading into `~/.cache`.
* 👁️ **Harness Vision Sync (v0.8.9)**: the one-click Harness sync now declares `input: [text, image]` for vision models, so DeepSeek Harness forwards image attachments to `mlx-vlm` instead of rejecting them with *"The current model does not support images"* — DSH treats custom OpenAI-compatible providers as text-only until a model says otherwise.
* 🔄 **One-Click Harness Sync (v0.8.4)**: every installed model is written into your DeepSeek Harness `mlx-local` provider by a single button (or `./mlx.sh sync-harness`) — no hand-editing YAML, no retyping repo ids.
* ⚡ **Fast, Resumable Downloads + Optional HF Login (v0.8.0)**: Large models now download over **8 parallel HTTPS range requests** instead of one throttled stream (measured on this line: 74 KB/s single connection → **2.2 MB/s** with 8), with byte-accurate progress, real MB/s, ETA, a **90-second stall watchdog** that restarts dead transfers automatically, and a **"🤗 Sign in" button** that uses HF's device flow so each user authorises *their own* account — the token goes into the macOS Keychain, never into this repo.
* 🧠 **Auto Context Windows (v0.8.7)**: every model reports its own native window (read from `config.json`), the dashboard shows it on the card, and servers are started with a RAM-safe `--max-kv-size`. No hand-math.
* 💾 **Multi-Drive Model Libraries (v0.7)**: Every writable APFS/HFS+ drive is discovered automatically and each one becomes a model library at `<drive>/MLX-Models/hub`. Pick a **Download to:** drive in the toolbar or per model with the **💾 Save to…** button, keep 70–100 GB models off your boot SSD (pick a drive, confirm, then it downloads), and see a live per-drive inventory (models, free space, offline drives) on the Downloaded Models tab. Includes a free-space guard, absolute-path launching, and the `./mlx.sh drives` / `./mlx.sh use-drive` / `./mlx.sh download <repo> --to <drive>` CLI. Add `--turbo` to force parallel mode, or `--no-turbo` to use the plain client.
* 🎛️ **Combinable Search Filters + Any-Bit Quant (v0.6)**: The `⚡ MLX Models`, `👁️ Vision & Multimodal` and `💻 Code Models` chips are independent toggles that combine with AND, and a new **Quant** dropdown filters by **2, 3, 4, 5, 6, 8 or 16-bit** (instead of 4-bit only). Stack them with the RAM budget: *"MLX + Code + 8-bit + Fits My Mac"* in three clicks.
* 🧠 **Accurate RAM Estimates & RAM Budget Filter (v0.5)**: Search cards now show real weight sizes and parameter counts pulled from Hugging Face (`safetensors` blobs + `safetensors.total`) and estimate RAM as `weights × 1.06 + KV cache + 1.5 GB`. A new **RAM Budget** dropdown (Fits My Mac / 8–192 GB / Any Size) hides models that can't run on your Mac, with 🟢/🟡/🔴 fit badges and Smallest/Largest RAM sorting.
* 👁️ **Vision & Multimodal Image Recognition (v0.4)**: Full native integration with `mlx-vlm`! Run Gemma 4, Qwen 2.5/3.8 VL, PaliGemma, Pixtral, LLaVA, SmolVLM, and more with full image analysis, OCR, visual question answering, and chart understanding.
* 🖼️ **Interactive Vision Test Chat (v0.4)**: Upload and preview images directly in the GUI test bench to test OpenAI-compatible multimodal payloads.
* ⚙️ **Dual-Engine Auto-Detection (`mlx-vlm` + `mlx-lm`) (v0.4)**: Automatically detects model architectures and chooses the optimal inference engine without manual flags.
* 🔌 **LibreChat & Coding Agent Multi-Modal Sync (v0.4)**: Direct compatibility with LibreChat (`host.docker.internal:9999/v1`), Pi Code, and OpenCode for image recognition and coding workflows.
* 🧠 **macOS Memory & Process Manager (v0.3)**: Real-time visual breakdown of Apple Silicon Unified Memory (App, Wired, Compressed, Available Cache) with an interactive top process scanner (`🔍 Manage RAM`) and 1-click process termination to free up memory for larger models.
* 🔤 **Model Sorting & Date Added (v0.3)**: Sort local and downloaded models by Name (A-Z / Z-A), Size (Largest / Smallest), and Date Added (Newest / Oldest) with exact timestamps on all model cards.
* 📱 **Horizontal List Mode vs. Card Mode (v0.3)**: Toggle seamlessly between sleek horizontal list rows and spacious square card grid views across all model and search views.
* 🟢 **Activity Monitor-Aligned RAM Accounting (v0.3)**: Accurate unified memory metrics matching Activity Monitor, `htop`, and `mactop`.
* 🚀 **1-Click Model Launching & Hot Swapping**: Instantly switch between MLX models without terminal commands.
* 📦 **Multi-Quant & Subfolder HF Support**: Direct support for repositories with multiple quantizations (e.g. `orcarouter/Qwen3.8-27B-Uncensored-MLX/tree/main/4-bit`). Pick exact bit precision (`2-bit`, `4-bit`, `6-bit`, `8-bit`) and download only the weights you need instead of 80+ GB!
* 🌐 **Supercharged Hugging Face Search**: Search thousands of model weights directly with **combinable filter chips** (`⚡ MLX Models`, `👁️ Vision & Multimodal`, `💻 Code Models` — click several, they AND together), a **Quant** selector (2/3/4/5/6/8/16-bit), and sorting (`Most Downloads`, `Most Liked`, `Recently Updated`, `Smallest / Largest RAM First`).
* 🤖 **Built-In AI Model Comparison Engine**: Select 2 to 4 models to generate an instant technical breakdown comparing quantization precision (`2bit-DQ`, `OptiQ-4bit`), RAM compatibility, and specialization — *works 100% locally out-of-the-box with zero API keys or accounts required!*
* 🟢 **"Can I Run It?" Apple Silicon RAM Calculator**: Live color-coded RAM budget badges (`🟢 Fits Smoothly`, `🟡 High RAM Pressure`, `🔴 Swap Danger`) computed against your Mac's actual available memory.
* 🤖 **Auto-Sync Pi Code & OpenCode Agents**: Automatically configures [Pi Code](https://github.com/) and [OpenCode](https://github.com/) under provider `MyMac` (`http://127.0.0.1:9999/v1`). Whenever you swap models in the GUI, your coding agents automatically route to the newly loaded model!
* 📊 **Tokens/Second Speed Benchmark**: Measure generation speed (`tok/s`), execution time, token counts, and GPU memory usage on Apple Silicon.
* ⚙️ **Custom API Settings**: Change default ports (`9999`, `8888`, `8080`), host binding (`127.0.0.1` or `0.0.0.0` for LAN access across home Wi-Fi), and max output token budgets with built-in memory guidelines.

---

## 🛠️ Prerequisites

Running local AI on your Mac has never been easier. You only need:
1. **Any Mac with Apple Silicon M-Series Processors** (with Unified Memory).
2. **macOS** (12.0 or newer).
3. **Python 3.10+** (pre-installed on macOS or available via Homebrew/Xcode).

> *Note:* Dependencies like `mlx-lm` and `hf` CLI are automatically installed on-demand via `uvx`. You do not need to pre-install heavy ML environments! Zero API keys or account sign-ups are required.

---

## 🖥️ Terminal CLI Mode

Prefer working in the terminal? You can also use `mlx.sh` directly:

```bash
./mlx.sh
```

### Key CLI Commands:
* `./mlx.sh list`: View all downloaded local models and RAM sizes.
* `./mlx.sh search <query> [ram_gb] [tags] [quant]`: Search Hugging Face repositories directly from terminal (e.g. `./mlx.sh search qwen 32` shows only models that fit in 32 GB; `./mlx.sh search coder 32 mlx,code 4` adds the combinable tags and 4-bit quant filter; prompts for anything omitted).
* `./mlx.sh drives`: List every model-library drive (free space, model count, Time Machine state, default target).
* `./mlx.sh protect-models`: Exclude every model library and the HF cache from Time Machine backups.
* `./mlx.sh use-drive [name]`: Change the default download drive (`./mlx.sh use-drive NVMe`).
* `./mlx.sh download <repo> --to <drive> [--turbo|--no-turbo]`: Download straight to a specific drive (`--to` is optional; you get a drive menu instead). `--turbo` forces the 8-connection parallel engine, `--no-turbo` disables it.
* `./mlx.sh login` / `./mlx.sh whoami` / `./mlx.sh logout`: Sign in to Hugging Face with the device flow (token stored in the macOS Keychain), check who you are, or remove the token.
* `./mlx.sh start`: Interactive launcher with Fast Swap and Concurrent port options.
* `./mlx.sh stop`: Emergency kill-switch for running servers.

---

## 🎛️ Combinable Search Filters & Quant Selector (New in v0.6)

The old chips were **presets**: clicking `👁️ Vision & Multimodal` replaced `⚡ MLX Models`, so "MLX vision models" was impossible, and `🎯 4-Bit Quantized` was hard-wired to one bit width.

Now the chips are **toggles** and every filter combines with AND:

| You want | Click |
|---|---|
| MLX models only | ⚡ MLX Models |
| Vision models that run in MLX | ⚡ MLX Models + 👁️ Vision & Multimodal |
| 8-bit coding models for your Mac | ⚡ MLX Models + 💻 Code Models, **Quant: 8-bit**, RAM Budget: Fits My Mac |
| Everything on Hugging Face | 🌐 All HF Repos (clears every chip) |

The **Quant** dropdown accepts **2, 3, 4, 5, 6, 8 and 16-bit**. Matching uses the repo name first (`-4bit`, `q8`, `8-bit`) for instant narrowing, then the repo's real `config.quantization_config.bits` once sizes arrive, so quantization-in-config repos are filtered correctly too. The header line tells you the active combination, e.g. `Filter: MLX + CODE + 4-bit`.

CLI equivalent: `./mlx.sh search <query> [ram_gb] [tags] [quant]`, e.g. `./mlx.sh search coder 32 mlx,code 4`.

---

## ⚡ Fast, Resumable Downloads (New in v0.8.0)

Downloading a 70–100 GB model used to mean watching a progress bar that lied. Three things were wrong, and v0.8.0 fixes all of them.

**1. One stream is throttled to a crawl.** Measured from this machine: a single HTTPS connection to the Hub sustains **74 KB/s**, while **8 parallel range requests reach 2.2 MB/s** (16 → 1.6 MB/s, 32 → 2.4 MB/s, so the line peaks around 2.3 MB/s ≈ 19 Mbit/s). The bottleneck is per-connection shaping, not authentication and not `hf_transfer`. So for any model with LFS blobs over 256 MB, the downloader now splits each file into 8 ranges, fetches them concurrently into `blobs/<sha256>.parts/`, verifies the **sha256 published by the Hub**, and only then moves the blob into place — after which `hf download` links the snapshot instantly instead of re-downloading it.

**2. Dead sockets never died.** The Hub client has no stall watchdog: when a transfer stalls, the process stays alive at 0 B/s forever. A watchdog now samples progress every 10 seconds; after **90 s with no byte growth** it kills the stuck process tree and relaunches the same command (up to **3 attempts**), then reports `STALLED` instead of pretending to download. `hf` resumes from its `.incomplete` files, so nothing is re-fetched.

**3. Progress was guesswork.** Speed is now a rolling 15-second window and ETA is derived from it, updated per poll. Turbo transfers publish their own status file (`~/.mlx_pids/turbo/<id>.json`), so the dashboard shows real bytes during the parallel phase, then `FINALISING` while the hub links files.

### Signing in to Hugging Face (optional, credential-free by design)

This repository ships **no tokens and no client secrets**. Signing in uses HF's OAuth **device flow** with a client that is registered dynamically at runtime:

```bash
./mlx.sh login     # prints a code + https://hf.co/oauth/device link
./mlx.sh whoami    # who am I signed in as, and where is the token stored?
./mlx.sh logout    # remove it from this Mac
```

The GUI has the same flow behind the **🤗 Sign in** button in the header, plus a "paste a token instead" fallback for users who already have a fine-grained token. Tokens live in the **macOS Keychain** (`service: mlx-control-center`) with a `0600` file fallback at `~/.mlx_pids/secrets.json` for machines without a keychain; nothing is ever written into the repo, and every code path works anonymously — a signed-out clone still downloads public models, just slower on gated or rate-limited repos.

## 📦 Serving Models from a Drive (Fixed in v0.8.5)

A model server is launched with an absolute path (`/Volumes/TBolt.../snapshots/<rev>`), but clients address it by **repo id** (`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`). `mlx-vlm` resolves that id through `huggingface_hub`, which defaults to `~/.cache/huggingface/hub` — so on a Mac with models on an external drive, the first repo-id request quietly **re-downloaded all 146 files into the internal SSD**.

v0.8.5 exports `HF_HUB_CACHE=<the library that holds the model>` for every server it starts (dashboard and CLI), so a repo-id request reuses the weights you already downloaded instead of re-fetching them.

Measured on the 73.5 GB REAP-288 model, started from the external drive:

| Request | Before | After |
|---|---|---|
| First request by repo id | began a 146-file download into `~/.cache` | **0.6 s**, no download |
| Reloads caused by the id mismatch | one per first request (~26 s) | **0** |
| Internal cache growth | 1.1 GB and climbing | 0 bytes |

Servers are also **launched by repo id** now (not by absolute path), so the id the client sends already matches the loaded model. Launching by path meant `mlx-vlm` saw "a different model" and unloaded/reloaded 73 GB on the first request of every session.

## 🧠 Auto Context Windows (New in v0.8.7)

Every model has its own conversation limit, and guessing it wastes memory or truncates silently. The dashboard now reads each model's `config.json` and works out three numbers:

| Number | Where it comes from |
|---|---|
| **Native window** | `text_config.max_position_embeddings` (or `max_seq_len`, `n_positions`, `model_max_length`) |
| **KV cost per token** | `2 × kv_heads × head_dim × 2 bytes × full_attention_layers` — hybrid models with linear-attention layers keep KV for only the full-attention ones |
| **RAM-safe ceiling** | `(total_RAM × 0.85 − weights − 3 GB) ÷ KV_cost_per_token`, rounded down to 8K steps |

The recommended window is `min(native, RAM-safe)` — normally just the native window, because KV cache is cheap next to weights:

| Model | Native | KV/token | Weights | RAM left for KV | Recommended |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next-REAP-288-4bit | 262,144 | 24 KB | 73.5 GB | 32.3 GB | **262,144** (native) |
| gemma-4-E4B-it-MLX-4bit | 131,072 | 14 KB | 6.8 GB | 99.0 GB | **131,072** (native) |
| Qwen2.5-1.5B-Instruct-4bit | 32,768 | 28 KB | 0.9 GB | 104.9 GB | **32,768** (native) |

**Where it lands:**

* model cards show `🧠 Context: 256K tokens`
* `Settings → Context Window` is **Auto** by default (per-model native); pick 32K/64K/128K/256K to force a cheaper cap — it is applied as `--max-kv-size` and can never exceed the model's own maximum
* `./mlx.sh sync-harness` writes `contextWindow`, `maxTokens`, and image modality (`input: [text, image]` for vision models) into the DeepSeek Harness profile, so the harness compresses at the right point instead of assuming 200K and accepts image attachments
* hand-edited capacity values in the harness profile are preserved across syncs

## 🛡️ Stream Guard: No More “Silent” Turns (v0.10.0, per-model token budget in v0.10.1)

**Symptom:** you chat with a local model through an agent harness (DeepSeek Harness, Pi Coder, any OpenAI-compatible client), the answer stops mid-turn or never starts, and typing “hello” gets it going again.

**Cause:** the model server itself goes quiet — not wrong, just quiet. Both `mlx-lm` and `mlx-vlm` build a whole tool call before writing anything, so during a long tool-call turn (one measured turn took **96.8 s and 3,736 tokens**) the HTTP stream carries **zero bytes**. Prefill is silent too, and without prefix-cache reuse the same conversation is re-prefilled every turn (14 K tokens ≈ 11.6 s on the 288-billion MoE model). A client with an idle watchdog — Pi's is shorter than DeepSeek Harness's 5 minutes — gives up on a stream that has produced nothing, and the turn dies with an empty assistant message. Nothing is logged as an error, because nothing went wrong.

**Fix:** every launch now loads `mlx_stream_guard/stream_guard.py` *inside* the model server process (via `PYTHONPATH` + `sitecustomize`, so the port, pid file, log file and command line are unchanged):

* **SSE keepalives** every 5 s while the server is silent — a real `chat.completion.chunk` with an empty `delta`, so a client watchdog actually resets instead of ignoring a comment
* **a guaranteed `data: [DONE]`** at the end of every streamed chat completion, even if the server dies before sending one
* **no behaviour change** on normal turns: an existing `[DONE]` is never duplicated, JSON responses are untouched, and `text/event-stream` detection is done per response
* **visible** in the dashboard status bar (`🛡️ Stream guard on · N keepalive(s)`), and in `~/.mlx_logs/stream_guard.log` (`keepalives`, `first_byte`, `max_gap`, `synthetic_done` per request)

Kill switch and tuning:

```bash
MLX_STREAM_GUARD=0 ./mlx.sh start <model>     # launch the old, unguarded way
MLX_GUARD_KEEPALIVE_SECS=2 ./mlx.sh start ... # heartbeat interval (default 5)
MLX_GUARD_HEARTBEAT=comment ./mlx.sh start ... # raw SSE comments instead of empty deltas
```

### Token budget: sized per model, not a constant

Decode budgets used to be hardcoded (4,096 / 8,192 tokens), which is wrong in both directions: too
small clips a long tool call on a big model, too large asks a small-context model for more than it
can hold.

* **Launches** pass `--max-tokens` = **a quarter of the loaded model's effective context window**,
  floored at 4k and capped at 32k. `default_max_tokens: 0` in settings means *auto*; any positive
  number pins it. The window itself comes from the model's own `config.json`, reduced to a
  RAM-safe size when the KV cache would not fit.
* **Agent sync** writes the same per-model numbers into Pi Coder (`contextWindow` + `maxTokens`)
  and DeepSeek Harness, so the client and the server agree instead of advertising 131,072 / 8,192
  regardless of the model.
* **A request's own `max_completion_tokens` is honoured.** That field is what modern OpenAI clients
  send to a local base URL — pi-ai, used by both Pi Coder and DeepSeek Harness, only emits
  `max_tokens` for an allowlist of hosted providers — and `mlx-vlm` reads *only* `max_tokens`, so
  every request used to silently fall back to the launch default. The guard now bridges the two
  fields inside the request path, and leaves an explicit `max_tokens` untouched.

```bash
# 262,144-token model -> 32k replies;  32,768-token model -> 8k;  8,192-token model -> 4k
python3 tests/test_token_budget.py
```

Tests: `python3 tests/test_stream_guard.py` (no model download, no network).

## ⚡ Prefix Cache: Why “hello” Took 90 Seconds (v0.10.2, RAM budget in v0.10.3)

**Symptom:** a harness that carries a big context answers a two-word question after a minute of
silence. The model is fine, the tokens are fine, the engine was just re-reading the entire
conversation every turn.

**Cause:** `mlx-vlm` ships automatic prefix caching (APC — reusable KV blocks, warm memory plus a
warm-disk tier), but the OpenAI-compatible server only builds it when `APC_ENABLED` is set, and the
default is **off**. With it off every request logs `cached_tokens=0`, so a 63,000-token agent prompt
is prefilled from token zero — about **49 s** before the first token, even when the only new text is
“hello”.

**Fix:** launches of `mlx-vlm` now export the cache environment, sized from settings:

```bash
APC_ENABLED=1
APC_BLOCK_SIZE=16            # 16 tokens per reusable block
APC_NUM_BLOCKS=4096          # 65,536 reusable tokens in RAM
MLX_VLM_CACHE_HOME=~/.cache/mlx-vlm
APC_DISK_MAX_GB=20           # warm-disk tier survives restarts
```

Measured on the 288-billion-parameter MoE model, three turns of one growing conversation:

| turn | prompt tokens | cached | prefill |
|---|---|---|---|
| 1 | 7,556 | 0 | **11.460 s** |
| 2 | 7,579 | 7,555 | **0.442 s** |
| 3 | 7,602 | 7,578 | **0.261 s** |

* **Two shapes of cache, both budgeted.** Dense models reuse 16-token *pages* (`APC_NUM_BLOCKS`);
  hybrid/rotating-attention models — most VLMs, including the 288B — keep whole-prefix *snapshots*
  instead (`APC_EXACT_CACHE_ENTRIES`, `APC_MEMORY_MAX_GB`), which `mlx-vlm` otherwise caps at
  2 snapshots and 8 GB. Measured on the 288B: **~2 GB per 63k-token conversation**, so four
  snapshots and a 16 GB ceiling still leave the machine plenty of headroom.
* **Only turns that share a prefix get faster.** The first turn of a session still prefills
  everything — that part is the model reading the context, not an overhead.
* **`mlx-lm` is untouched:** it has its own prompt cache and ignores these variables. (Upstream,
  streamed `mlx-lm` requests do not store cache entries — a single non-streamed request seeds it.)
* **Visible** in the dashboard (`⚡ Prefix cache on · N% of prefilled tokens reused`, read from the
  server's `/v1/cache/stats`) and in `~/.mlx_logs/stream_guard.log`, which now logs a
  `prefix hash=` for every request. If two turns show different hashes, the harness changed the
  front of the prompt (a timestamp in the system message will do it) and no cache can help.

```bash
# settings.json
{"prefix_cache": true, "prefix_cache_blocks": 4096, "prefix_cache_disk_gb": 20,
 "prefix_cache_entries": 4, "prefix_cache_memory_gb": 16}

MLX_VLM_APC=0 ./mlx.sh start <model>   # launch without the cache
python3 tests/test_prefix_cache.py     # 22 checks, no model download
```


## 🦙 Ollama Engine + GGUF Models (New in v0.11.0)

The control center now runs two kinds of local models and picks the engine from the model itself:

| What you pick | Engine | Port |
| --- | --- | --- |
| An MLX repo (safetensors) | `mlx-lm` / `mlx-vlm`, one process per model | 9999 |
| A GGUF file, or an Ollama tag | the Ollama daemon | 11434 |

Ollama is a **daemon**: one process, one port, every model behind it, and the **model tag** is the identity instead of the port. That is why the dashboard reports it beside the MLX server list rather than inside it.

### Getting a GGUF model
1. **Search Hugging Face** → press **🦙 GGUF (Ollama)** (turn MLX off first if you want GGUF-only results).
2. Press **🦙 Pick GGUF file…** on a result. A GGUF repo holds several quants — pick one, because downloading the repo would pull 100+ GB.
3. When the file lands in the library, its card offers **🦙 Import to Ollama**. Give it a tag such as `qwen3-coder:30b-q4` and it is registered in the store with `ollama create`.

> Importing **copies** the file into the Ollama store (about 2× disk for that model). `ollama pull` is the single-copy path when the model already exists upstream. The store lives at `~/.ollama/models` unless `ollama_store` says otherwise — and Ollama reads that setting only when the daemon starts.

### The 🦙 Ollama pill
Click it to see the daemon state, the store size, the model table, and to **Start / Stop** the daemon or **Sync to Pi / DSH**. Sync writes:

* **Pi Coder** — provider `Ollama`, `baseURL http://127.0.0.1:11434/v1`, one entry per model tag using the context length Ollama reports for that model.
* **DeepSeek Harness** — a second provider `ollama-local` in `cordis.patch.yml`. A DSH provider carries a single base URL, so Ollama cannot share the MLX provider. Restart DSH to load it.
* **OpenCode** — the same provider block.

In your agent, choose provider **Ollama** and the tag.

### Worth knowing
* **Context:** a harness cannot ask Ollama for a larger window per request, so the daemon's `OLLAMA_CONTEXT_LENGTH` is the real ceiling (`settings.json` → `ollama_context`, default 32768).
* **launchd:** if the daemon was started by `~/Library/LaunchAgents/com.ollama.ollama.plist`, Stop only lasts until launchd restarts it — the panel says so.
* **LAN exposure:** that same plist may set `OLLAMA_HOST=0.0.0.0`, which serves Ollama to your whole network. Use `127.0.0.1:11434` if you did not mean that.

## 🔄 Sync Installed Models into DeepSeek Harness (v0.8.4, vision fix in v0.8.9)

DeepSeek Harness reads its model list from a profile file (`~/.dsh/profiles/desktop/cordis.patch.yml`), one entry per model id. Nothing discovers your local server on its own, so a freshly downloaded model used to mean hand-editing YAML — and because the **desktop** profile only applies patches at startup, getting it wrong is invisible until a restart.

**Settings → 🤗 DeepSeek Harness model list** now shows the state and fixes it in one click:

| State | What it means |
|---|---|
| `3 models listed · 3 installed · in sync ✓` | Nothing to do |
| `2 models listed · 3 installed · 1 downloaded but not listed` | Press **🔄 Sync now** |
| `⚠️ listed but no longer installed` | Entry points at a model you deleted — sync removes it |

What a sync does:

* reads installed models from every library drive (deduplicated, so a model present on two drives appears once);
* rewrites only the `models:` block of the `mlx-local` provider, keeping the rest of your file byte-identical;
* **declares image input for vision models** (`input: [text, image]`), detected from each model's own files (`config.json`, image processor, or vision weights); DSH otherwise treats a custom OpenAI-compatible provider as text-only and rejects attachments before they reach the server. Hand-edited `input` lines are preserved;
* **keeps hand-written labels** for models already listed, and generates readable ones for new repos (`sh0wie/Qwen3.8-...-REAP-288-MLX-4bit` → `Qwen3.8 Flash Next REAP 288 4-bit (MLX)`);
* writes a timestamped backup next to the file (`cordis.patch.yml.bak-YYYYMMDD-HHMMSS`) before touching it;
* never writes credentials into the profile — an `apiKeyEnv` you added yourself is left untouched.

Same thing from the terminal:

```bash
./mlx.sh sync-harness
```

Restart DeepSeek Harness afterwards (the desktop profile loads the file at startup); then pick the model under the **MLX (this Mac)** provider. The model id **must be the Hugging Face repo id**, because `mlx-vlm` treats the request's `model` field as a repository to load — a cosmetic alias like `mlx-local` returns `404 Repository Not Found`.

If DSH still says *"The current model does not support images"*, re-run the sync and restart DSH. The model must be a vision model (`👁️ Vision` on its dashboard card, launched with `mlx-vlm`); a text-only model is deliberately left without image input.

## 💾 Multi-Drive Model Libraries (New in v0.7)

A 70–100 GB model does not have to live on your boot SSD. **macOS MLX Control Center** turns every writable drive into a model library.

### How it works
* **Nothing to configure.** Every writable APFS/HFS+ volume with at least 8 GB is discovered from the mount table: your boot drive, internal second SSDs, and any Thunderbolt/USB drive you plug in. Time Machine destinations, read-only DMGs (installers like "Install ChatGPT") and tiny volumes are skipped automatically.
* **One folder per drive.** Models are stored in Hugging Face cache layout at `<drive>/MLX-Models/hub/models--org--repo/…`. Your boot drive uses `~/MLX-Models/hub` instead, so the sealed system volume is never touched.
* **Smart default.** External drives win: the default target is the external drive with the most free space, remembered in `~/.mlx_pids/storage.json`. Change it any time with the **Download to:** dropdown next to the RAM Budget, or `./mlx.sh use-drive <name>`.
* **One button, one confirmation.** Every download starts with the single blue **💾 Download…** button: pick the drive (free space, `TM off` state and a `too small for N GB` warning per drive), then confirm — `Download <repo> (~7 GB) to 💾 TBolt-1TB-Fun?` — before a single byte moves. Nothing is written to a drive you did not confirm.
* **Free-space guard.** Downloads are refused with `insufficient_space` (GUI toast / HTTP 409) or a CLI prompt when the target drive is too small — no half-written 100 GB downloads.
* **Absolute-path launching.** A model on an external drive starts as `mlx_lm.server --model /Volumes/<drive>/MLX-Models/hub/models--…/snapshots/<rev>`, so the engine never depends on the working directory.
* **Live inventory.** The Downloaded Models tab shows one line per drive (`💾 TBolt-1TB-Fun: 3 models · 612 GB free`) and every model card is labelled with the drive it lives on. Drives that are unplugged keep their models listed as offline instead of vanishing.
* **Backups stay sane.** Model weights are re-downloadable multi-GB blobs, so they never belong in a backup set. Every library directory (`<drive>/MLX-Models`, `~/MLX-Models`) plus the leftover `~/.cache/huggingface` cache is added to Time Machine's exclusion list automatically (`tmutil addexclusion`) as soon as it exists — including libraries on drives you plug in later. The drive list shows `🚫 TM off`, or a clickable `⚠️ backup on` warning if a drive's library is still inside your backup set. CLI: `./mlx.sh protect-models`.

### Why not `HF_HOME`?
`HF_HOME` is global: it moves *every* model to one place and breaks as soon as you unplug that drive. Per-download `--cache-dir` keeps each model on the drive you chose, and the dashboard keeps track of which is which.

### Drives API
```bash
curl -s http://127.0.0.1:9998/api/drives | python3 -m json.tool
# {"drives": [{"id": "<volume UUID>", "name": "TBolt-1TB-Fun", "mount": "/Volumes/TBolt-1TB-Fun",
#              "free_gb": 612.5, "library": "/Volumes/TBolt-1TB-Fun/MLX-Models/hub", "models": 0, ...}],
#  "default": "<uuid>", "default_name": "TBolt-1TB-Fun"}
```

> **Format note:** use APFS (or HFS+) for model drives. exFAT/FAT32 cannot store the symlinks Hugging Face relies on, so those drives are reported but not offered as libraries.

---

## 🧠 Accurate RAM Estimates & RAM Budget Filter (New in v0.5)

Search cards now show **real numbers fetched from Hugging Face** instead of guessing from the repository name:

* **Weight size** — summed from the repo's `safetensors` blobs (picks the largest quant subfolder when a repo ships several).
* **Parameter count** — read from `safetensors.total` in the HF API, falling back to parsing the name (`7B`, `8x7B`, `350m`).
* **RAM estimate** — `weights × 1.06` (activations) `+ KV cache` (`params × 0.02 GB`, clamped 0.75–8 GB) `+ 1.5 GB` runtime overhead.

> **Why this matters:** previously any repo with `4bit` in its name matched the `"4b"` substring and was rated as a 4-billion-parameter model (≈3.2 GB). A 180B-parameter `Qwen3.8-Flash-Next-4bit` with **106 GB of weights** showed a green "Fits Smoothly (Est: 6 GB RAM)" badge — it would never load on a Mac.

### RAM Budget dropdown
The search bar has a new **RAM Budget** selector:

* **`🖥️ Fits My Mac (auto)`** — uses *usable RAM* = total unified memory − `max(4 GB, 12%)` reserved for macOS (e.g. **112.6 GB** on a 128 GB Mac).
* **`8 … 192 GB`** — only list models whose estimated RAM fits that ceiling.
* **`Any Size`** — no filtering (sizes and fit badges are still shown).

Badges are computed against the current budget: 🟢 **Fits Smoothly** (≤ 80%), 🟡 **Tight Fit** (≤ 100%), 🔴 **Too Large** (shows required vs. usable RAM). Sort by **Smallest / Largest RAM First** to find the right size instantly.

Sizes are fetched in a fast second pass (`/api/repo_sizes`) and cached, so repeat searches are instant.

---

## 💡 How Token Limits & RAM Work on Mac

In the **`⚙️ API Settings`** panel, you can configure your **Max Output Tokens**:

* **Max Output Tokens**: Specifies the maximum number of *generated response + reasoning tokens* the server produces. It does not limit your input prompt size.
* **RAM Guidelines for Apple Silicon**:
  * **2B – 8B Models** *(Gemma 4 4B, Qwen 7B)*: Set to `4096 – 8192` (Light KV Cache RAM impact ~0.5 GB).
  * **14B – 35B Models** *(Qwen 32B, DeepSeek 35B)*: Set to `4096` (Moderate RAM impact ~1.5–3 GB).
  * **70B+ Models**: Set to `2048 – 4096` to prevent memory swapping.

---

---

## 💬 Integrating with LibreChat (Local ChatGPT Web UI with Vision)

The combo of **macOS MLX Control Center** and **[LibreChat](https://github.com/danny-avila/LibreChat)** gives you a completely private, self-hosted ChatGPT-like experience with full multimodal vision support on your Mac.

### 🌟 Key Benefits:
* 🔒 **100% Local & Private**: No data leaves your Mac; runs entirely on Apple Silicon GPU & Unified Memory.
* 👁️ **Full Multimodal Vision**: Attach screenshots, PDFs, documents, diagrams, and photos in LibreChat for instant analysis with models like Gemma 4 and Qwen 2.5/3.8 VL.
* ⚡ **Hot-Swapping**: Switch models in MLX Control Center with 1 click; LibreChat dynamically routes to the active model via `default_model` without restarting containers.
* 💰 **Zero Cost**: No OpenAI/Anthropic API subscriptions needed.

### 🛠️ Setup Guide:

#### 1. Configure `librechat.yaml`
Add the custom MLX endpoint to your `librechat.yaml` file:

```yaml
version: 1.2.0

endpoints:
  custom:
    - name: "MacBook MLX"
      apiKey: "local"
      baseURL: "http://host.docker.internal:9999/v1" # Use http://127.0.0.1:9999/v1 if running LibreChat natively without Docker
      models:
        default:
          - "default_model"
          - "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
          - "lmstudio-community/Qwen3.8-27B-MLX-4bit"
          - "mlx-community/Llama-3.2-3B-Instruct-4bit"
        fetch: true
      titleConvo: true
      modelDisplayLabel: "MacBook MLX"
```

> **Note for Docker users:** `host.docker.internal` allows Docker containers on macOS to seamlessly communicate with the MLX server running directly on macOS.

#### 2. Restart LibreChat
If running via Docker:
```bash
docker compose restart
```

#### 3. Chat & Analyze Images
1. Open LibreChat (e.g. `http://localhost:3080` or `http://localhost:3081`).
2. Select **`MacBook MLX`** from the model dropdown.
3. Drag & drop images directly into the chat to perform OCR, code transcription, and visual reasoning!

---

## 🤖 Agentic AI Pairing Instructions (Antigravity, ChatGPT, Claude, Cursor & Codex)

You can use an AI coding assistant (like **Google Antigravity**, **ChatGPT / Codex**, **Claude Code**, or **Cursor**) to completely automate managing, launching, and integrating this system on your Mac.

Here are ready-to-use prompt templates you can copy and paste into your AI assistant:

### 📋 Prompt 1: 1-Click Setup & Launcher
> *"I want to run local AI models on my Apple Silicon Mac using macOS MLX Control Center. Please run `curl -sSL https://raw.githubusercontent.com/cjtsh/mac-mlx-control-center/main/install.sh | bash` to set up the repository, start the GUI dashboard on port 9998, and verify that the server is responding."*

### 📋 Prompt 2: Auto-Configure LibreChat with MLX Vision
> *"Please configure my local LibreChat setup to connect to my MLX Control Center. Update `librechat.yaml` with a custom endpoint named 'MacBook MLX' pointing to `http://host.docker.internal:9999/v1` with `fetch: true` and models list containing `default_model` and `mlx-community/gemma-4-e4b-it-OptiQ-4bit`. Then restart the LibreChat docker container and confirm health."*

### 📋 Prompt 3: Auto-Configure Coding Agents (Pi Code / OpenCode / Cline)
> *"Please configure my local coding agent (Pi Code / OpenCode) to use my local MLX server as the default provider under the name 'MyMac' at `http://127.0.0.1:9999/v1` with model ID `default_model` and reasoning tokens enabled."*

### 📋 Prompt 4: Diagnostic & Vision Troubleshooting
> *"Check the status of my MLX server on port 9999 and GUI on port 9998 using `curl http://127.0.0.1:9998/api/status`. If I'm using a vision model (like Gemma 4), verify that it is running via `mlx-vlm` so image uploads work in LibreChat."*

---

## 🤝 Agent Integration (Pi Code & OpenCode)

When auto-sync is enabled in **`⚙️ API Settings`**, launching any model automatically updates your local coding agent configs:

* **Pi Code**: `~/.pi/agent/settings.json` & `~/.pi/agent/models.json`
* **OpenCode**: `~/.config/opencode/opencode.jsonc`

Provider Name: **`MyMac`**  
Model ID: **`default_model`** (Auto-Detect Active MLX Endpoint)

---

## 🤖 Acknowledgements

* **Google Antigravity AI**: Pair-programmed, engineered, and designed with Antigravity AI.
* **Apple MLX Team**: Powered by Apple's open-source `mlx-lm` & `mlx-vlm` frameworks.

---

## 📜 License

MIT License — free to use, modify, and share!


## Legal notice

This project is provided under the [MIT License](LICENSE). Review the [disclaimer](DISCLAIMER.md) before downloading models or running benchmarks and system changes. Model output and resource estimates are not professional advice.
