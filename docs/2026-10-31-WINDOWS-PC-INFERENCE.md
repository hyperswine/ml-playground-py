# Windows PC: remote wake, the `pc` helper, and llama.cpp inference on the RX 7600

Created: 2026-10-31T17:40:00+11:00 (Australia/Sydney). Kind: setup and measurement report. No cloud compute.
Follows the [Windows worker section of the 4B plan](2026-10-29-QWEN-4B-20USD-PLAN.md#windows-worker-and-wake-on-lan),
which listed wake, GPU offload, output correctness and throughput as unverified. Raw data:
`data/qos_ai_benchmark/windows_llama_20261031/`; answers and scores: `artifacts/qos_ai/llama_2b_round4_q8_20261031/`.

## The machine (live inventory)

| Item | Value |
|---|---|
| Name / access | `swinepcw`, Windows 11 (build 26200), Tailscale 100.100.159.39, home network 192.168.0.210; SSH as `capta` (keys of macm4 and the M1) |
| CPU / memory | Ryzen 5 3600 (6 cores, 12 threads), 32 GB |
| GPU | AMD Radeon RX 7600, 8 GB (Windows' WMI query reports 4 GB, a known 32-bit cap; Vulkan reports 8,176 MiB), driver 31.0.22030.2, Vulkan 1.3.262 |
| Storage | C: about 2.5 TB free |
| WSL | Ubuntu 22.04, 12 CPUs, 15 GB, Python 3.10; no ROCm (GPU work stays on native Windows) |
| Virtualization | SVM enabled in firmware; Hyper-V, Virtual Machine Platform and WSL on, hypervisor running |
| Network | Only the Intel AX200 Wi-Fi is connected (no cable yet); the Realtek 2.5GbE port is disconnected |

SSH to the home-network address times out (Windows Firewall, likely a Public Wi-Fi profile); the Tailscale address
works and routes directly over the LAN. No firewall rule was changed.

## Remote wake

Configured by the owner: MSI BIOS `Wake Up Event By [BIOS]`, `Resume By PCI-E Device [Enabled]`, `ErP Ready
[Disabled]`; Windows Fast Startup off (`HiberbootEnabled = 0`); sleep after 45 minutes on mains power, no
automatic hibernation. Read back over SSH: S3 sleep available; both network adapters are wake-armed; the Realtek port
has magic-packet and shutdown wake enabled (MAC `2C-F0-5D-D6-03-07`), the Wi-Fi adapter magic-packet wake
(MAC `B0-A4-60-EA-AE-FC`).

**Wi-Fi wake works from sleep.** Magic packets to the Wi-Fi MAC (subnet broadcast, limited broadcast and the last
known address, ports 9 and 7) woke the PC twice. Windows' Power-Troubleshooter log recorded the wake source as the PCI
bridge in front of the Wi-Fi card both times (an earlier wake from the power button is logged as such). A full
helper-driven cycle took 12 seconds from packet to SSH. Wi-Fi wake cannot start a PC that is shut down; wired
Ethernet remains the dependable path, and the helper already includes the wired MAC.

## The `pc` helper

`scripts/pc/pc`, installed on the M1 as `~/bin/pc` (on the PATH via `~/.zshrc`):

| Command | What it does |
|---|---|
| `pc status` | awake or not, and last boot time |
| `pc wake` | sends magic packets every 15 s until SSH answers (default limit 120 s) |
| `pc ssh` | wake, then an interactive session |
| `pc run CMD` / `pc wsl CMD` / `pc ps < script.ps1` | wake, then run a cmd.exe command, a WSL bash command, or a PowerShell script while keeping Windows awake; the exit code is passed back |
| `pc sleep` | proper S3 sleep via `SetSuspendState(Suspend)` (rundll32's `SetSuspendState` hibernates when hibernation is enabled) |

Jobs hold `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` for their lifetime (visible in
`powercfg /requests` as a `powershell.exe` system request), so the 45-minute idle timer cannot sleep a long job and
applies again when it ends. Commands travel base64-encoded (`-EncodedCommand`): Windows PowerShell 5.1 does not escape
embedded quotes for native programs, and `wsl --` re-parses its command line, so WSL commands are written to a
temporary script and started with `wsl --exec`. Tested: status, wake, a sleep/wake cycle, exit codes, quoting with
`$(...)`, and the keep-awake request.

## llama.cpp on the RX 7600

**Setup** (approved by the owner): the official llama.cpp **b11978** Windows Vulkan build, checked against GitHub's
published SHA-256 (`b26bdfe4fa99...`) before extracting to `C:\Users\capta\llama.cpp-b11978`. The converter is the
same tag's `convert_hf_to_gguf.py` in WSL, with its pinned requirements (torch 2.11.0 CPU, transformers 4.57.6) in a
separate virtual environment. The round-4 Qwen3.5-2B checkpoint was copied from the M1 and hash-checked
(`model.safetensors` `e921fea5...`), then converted to `q8_0` in 45 seconds (2.0 GB, 320 tensors). The quantizer
printed divide-by-zero warnings for some blocks; the identical scores below show they are harmless.

**Speed** (`llama-bench`, all layers on the GPU, 3 repetitions):

| | M1, MLX 8-bit | **RX 7600, llama.cpp q8_0** |
|---|---:|---:|
| Generation | 26 tokens/s | **86 tokens/s** |
| Prompt processing | about 270 tokens/s | **1,919 tokens/s** (512-token prompt) |
| All 740 evaluation prompts, one at a time | 19 minutes | **7.3 minutes** |

The Gated DeltaNet layers run on the GPU: a CPU fallback could not reach 86 tokens/s.

**Answers match the other backends.** On the same deterministic 120-prompt sample used for the MPS and MLX checks
(served by `llama-server` bound to 127.0.0.1 and reached through an SSH tunnel; greedy, same stop tokens and token
limits), 109/120 answers are identical to the GPU pod's and 114/120 share the first line (MLX 8-bit: 109 and 114).
Full evaluation, scored with the unchanged scorers (pinned compiler for the Sol suites; the original benchmarks on
the M1):

| Measurement | GPU pod (bf16, CUDA) | M1 MLX 8-bit | **PC llama.cpp q8_0** |
|---|---:|---:|---:|
| Frozen Sol tests /360 | 339 | 341 | **341** |
| Programs that run correctly /111 | 110 | 111 | **110** |
| Multi-input /18 (cases /108) | 18 (108) | 18 (108) | **18 (108)** |
| Dev lexical /90 | 52 | 51 | **52** |
| Stated outputs /33 | 30 | 30 | **30** |
| Applied repairs / repair + output /48 | 46 / 35 | 46 / 35 | **46 / 35** |
| No-change + output /24 | 19 | 19 | **18** |
| Wording / Inputs / Combinations /48 | 43 / 43 / 41 | 43 / 42 / 41 | **43 / 44 / 41** |
| Strict trace /120 | 109 | 108 | **110** |

Every difference is within one case.

## Implications for the 4B plan

- The PC is the faster local evaluator: 3.3x the M1's generation and 7x its prompt speed for the 2B, with matching
  scores. The 2B uses about 2 GB of the 8 GB card; a 4B q8_0 (about 4.9 GB) should fit, at roughly half the 2B's
  speed by memory bandwidth (unmeasured).
- Training stays on RunPod: 8 GB of VRAM cannot hold the full 4B fine-tune.
- Large transfers should go straight from the M1 to the PC over Tailscale (both on Wi-Fi). The first copy was relayed
  through macm4 and averaged about 1.3 MB/s (4.4 GB in 55 minutes); wired Ethernet would be the larger improvement.

## Reproduction

- `scripts/pc/pc`: the helper (configuration through `PC_HOST`, `PC_LAN_IP`, `PC_USER`, `PC_MACS`, `PC_WAKE_TIMEOUT`).
- `scripts/windows/llama_check.py URL prompts.jsonl reference.jsonl N out.json`: 120-prompt parity and speed.
- `scripts/windows/llama_answers.py URL prompts.jsonl out.jsonl`: all prompts in the pod's answer format, then
  `qos qwen_sft assemble` and the usual scorers.
- Server: `llama-server.exe -m qwen2b_round4_q8_0.gguf -ngl 99 -c 4096 -np 1 --host 127.0.0.1 --port 8080`, reached
  with `ssh -L 18080:127.0.0.1:8080 capta@100.100.159.39`.
