"""P3.3/P3.4b - QLoRA SFT of a small model on the grounded-trace data.

Trains LoRA adapters on a 4-bit-quantized base (QLoRA) over the chat-format SFT file
(built by scripts/build_sft.py). Each conversation is rendered to a single text with the
tokenizer's chat template and **thinking disabled** - identical to how src/local_infer
formats the prompt at inference - so train and eval prompts match.

**v2 (P3.4b): completion-only loss.** v1 trained on the full sequence (prompt + completion
both in the loss). With a ~750-token prompt and ~490-token completion (prompt ≈ 61% of
tokens), most gradient was spent reproducing a fixed instruction prompt the model never
needs to generate, under-training the completion's structure - notably the terminal
``conclude`` step and the EOS/stop token. Diagnosis (results + scripts/_diag_ood.py): the
dominant v1 dev failure was *well-formed grounded traces that end with the wrong step kind*
(77/190 "final step must be 'conclude'"), plus runaways (42/190); base actually ran away
MORE (67/190). So we mask the prompt: label every prompt token ``-100`` so the loss is
computed only on the completion (incl. the closing ``conclude`` step + EOS).

Masking is deterministic and verified, not template-magic: ``assistant_only_loss`` is a
no-op on Qwen3's template (it has no real ``{% generation %}`` block - the substring
"generation" only comes from ``add_generation_prompt``), so it would silently mask
everything. Instead we render the prompt alone (``add_generation_prompt=True``), assert it
is an exact **token prefix** of the full sequence (holds for all 169 records), and mask that
prefix. A pre-train batch check then confirms prompt tokens are ``-100`` and the
completion+EOS survive. Everything else is identical to v1 (same rendering, r/alpha, lr,
epochs, data) so the only changed variable is the loss mask.

Local GPU only.

    python scripts/train_sft.py --epochs 3 --out models/qwen3-1.7b-sft-v2
    python scripts/train_sft.py --model Qwen/Qwen3-4B --out models/qwen3-4b-sft   # P3.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IGNORE = -100  # HF label id excluded from the loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/sft_train.v0_1_0.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="models/qwen3-1.7b-sft-v2")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=2048, help="floor; auto-raised to fit longest example (cap 4096)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
    )
    from trl import SFTConfig, SFTTrainer

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    tok = AutoTokenizer.from_pretrained(args.model)

    def render(msgs, add_gen):
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_gen, enable_thinking=False
            )
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)

    # --- build (input_ids, labels) with the PROMPT masked to -100 (completion-only loss) ---
    # The full rendering is byte-identical to v1; only the labels change.
    def ids_of(text):
        return tok(text, add_special_tokens=False).input_ids

    examples, prefix_fail = [], 0
    for r in rows:
        msgs = r["messages"]
        full = ids_of(render(msgs, add_gen=False))            # user + assistant turn (+ <|im_end|>\n)
        prompt = ids_of(render([msgs[0]], add_gen=True))       # user turn + "<|im_start|>assistant\n"
        if full[: len(prompt)] != prompt:                      # prompt must be an exact token prefix
            prefix_fail += 1
            continue
        labels = [IGNORE] * len(prompt) + full[len(prompt):]
        examples.append({"input_ids": full, "labels": labels,
                         "attention_mask": [1] * len(full), "_plen": len(prompt)})
    assert prefix_fail == 0, f"{prefix_fail} records: prompt is not a token prefix of the full sequence"

    lens = [len(e["input_ids"]) for e in examples]
    obs_max = max(lens)
    max_length = min(4096, max(args.max_len, obs_max + 8))
    truncated = 0
    for e in examples:                                         # defensive; no-op when obs_max < max_length
        if len(e["input_ids"]) > max_length:
            truncated += 1
            for k in ("input_ids", "labels", "attention_mask"):
                e[k] = e[k][:max_length]

    comp_tokens = [sum(1 for x in e["labels"] if x != IGNORE) for e in examples]
    print(f"training examples : {len(examples)}")
    print(f"token lengths     : full max={obs_max} mean={sum(lens) / len(lens):.0f} -> max_length={max_length}"
          + (f"  (WARNING: {truncated} truncated)" if truncated else ""))
    print(f"completion tokens : min={min(comp_tokens)} mean={sum(comp_tokens) / len(comp_tokens):.0f} "
          f"max={max(comp_tokens)}  (loss is on these only)", flush=True)

    ds = Dataset.from_list([{k: e[k] for k in ("input_ids", "labels", "attention_mask")} for e in examples])

    # --- 4-bit base + LoRA (QLoRA); trl applies the adapter via peft_config ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16
    )
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )

    # Pad input_ids/attention_mask with the pad token and labels with -100.
    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=IGNORE, padding=True)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=2,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length,
        packing=False,
        seed=args.seed,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},  # dataset is already tokenized + masked
    )

    trainer = SFTTrainer(
        model=model, train_dataset=ds, args=cfg, peft_config=peft_cfg,
        processing_class=tok, data_collator=collator,
    )
    try:
        trainer.model.print_trainable_parameters()
    except Exception:  # noqa: BLE001
        pass

    # --- mask-verification gate: assert the collator masks the prompt and keeps completion+EOS ---
    _verify_mask(trainer, tok, examples[:2])

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nsaved adapter -> {args.out}", flush=True)
    return 0


def _verify_mask(trainer, tok, sample_examples) -> None:
    """Pull one real batch through the trainer's collator and assert the prompt is masked
    (-100) while the completion + EOS keep real labels. Fails loudly rather than training
    on a silently-broken mask."""
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    labels, input_ids = batch["labels"], batch["input_ids"]
    row0_lab, row0_in = labels[0], input_ids[0]
    masked = int((row0_lab == -100).sum())
    kept_positions = (row0_lab != -100).nonzero().flatten().tolist()
    eos = tok.eos_token_id

    assert masked > 0, "mask gate: NO tokens masked -> prompt is in the loss (mask failed)"
    assert kept_positions, "mask gate: ALL tokens masked -> nothing to learn (mask failed)"
    # kept labels must equal the corresponding input_ids (teacher forcing on the completion)
    for p in kept_positions:
        if int(row0_lab[p]) != -100:
            assert int(row0_lab[p]) == int(row0_in[p]), "mask gate: kept label != input_id"
    # the kept span must be a contiguous suffix and must include the EOS/stop token
    assert kept_positions == list(range(kept_positions[0], kept_positions[0] + len(kept_positions))), \
        "mask gate: kept span is not contiguous"
    kept_ids = [int(row0_in[p]) for p in kept_positions]
    assert eos in kept_ids, "mask gate: EOS not in the supervised span -> stop token unlearned"

    expected_plen = sample_examples[0]["_plen"]
    print("--- mask gate ---", flush=True)
    print(f"row0: {masked} masked (prompt), {len(kept_positions)} kept (completion); "
          f"expected prompt_len≈{expected_plen}", flush=True)
    print("kept span decoded (head):", repr(tok.decode(kept_ids[:30])), flush=True)
    print("kept span decoded (tail):", repr(tok.decode(kept_ids[-8:])), flush=True)
    print("--- mask gate PASSED ---", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
