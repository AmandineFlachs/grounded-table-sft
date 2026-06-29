# Training environment runbook (WSL / GPU)

Phase 3 (training, inference, eval) runs on the GPU under **WSL2 Ubuntu**, in a
dedicated virtual environment **`.venv-train`**. The pure-CPU core of the project
(data prep, validation, tests) stays on Windows with the normal `requirements.txt`
- this env is *only* for the GPU steps.

We reuse this env many times across Phase 3 (baseline → SFT → eval → 4B), so the
re-enable steps below are the ones you'll run each session.

---

## One-time setup (already done - for reference only)

```bash
wsl
cd "/mnt/c/path/to/table_reasoning_traces"
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt
```

The `.venv-train/` folder lives on disk permanently. You do **not** repeat this.

---

## Re-enable each session (the two-liner)

From a Windows PowerShell window:

```powershell
wsl
```

Then, on the Ubuntu (`…$`) side:

```bash
cd "/mnt/c/path/to/table_reasoning_traces"
source .venv-train/bin/activate
```

✅ Your prompt now starts with `(.venv-train)` - you're in Ubuntu **and** in the env.

**Mental model:** the venv *folder* is permanent; *activation* lasts only for the
current Ubuntu shell. A fresh terminal = re-run just these two lines. Never re-run
`python3 -m venv` again.

---

## Verify the GPU is live

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

✅ Expected: `True NVIDIA GeForce RTX 3090`

If it prints `False`, the env is fine but CUDA isn't reachable - reboot WSL with
`wsl --shutdown` (from PowerShell), then re-enter and re-check.

---

## Run a training/eval script

With the env active:

```bash
python scripts/<script>.py [args]
```

Hugging Face model weights download once into `~/.cache/huggingface` (Linux home,
fast) - first run of a new model pulls a few GB, later runs are instant.

## Leave the env / Ubuntu

```bash
deactivate   # leave the venv (stay in Ubuntu)
exit         # leave Ubuntu, back to PowerShell  (or Ctrl+D)
```

---

## Teardown - at the END of the project

This env is disposable. When Phase 3 is finished and you no longer need to train or
run the model, reclaim the disk (torch + CUDA + model cache is several GB):

```bash
# from Ubuntu, in the project folder:
deactivate 2>/dev/null            # if currently active
rm -rf .venv-train                # remove the environment (~4 GB)
rm -rf ~/.cache/huggingface       # remove downloaded model weights (optional, several GB)
```

Removing `.venv-train` affects nothing else - the Windows CPU core and all code/data
are untouched. To rebuild later, just run the one-time setup again.

> ⏳ **Reminder:** delete `.venv-train` (and the HF cache) when the project wraps. See
> the teardown reminder in `TASKS.md`.
