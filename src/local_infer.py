"""Local Hugging Face text generation that mirrors ``call_claude``'s contract.

A ``LocalGenerator.generate(prompt) -> str`` drops into the existing
``render()`` pipeline (build_prompt -> generate -> extract_json ->
reconstruct_example -> validate) in place of headless Claude, so the baseline and
trained-model evaluations reuse all the Phase-2 machinery unchanged.

GPU-only. ``torch`` / ``transformers`` / ``peft`` are imported lazily inside
``__post_init__`` so the pure-CPU core stays importable without these installed.
Decoding is deterministic (greedy + fixed seed) for controlled base-vs-SFT
comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


@dataclass
class LocalGenerator:
    model_id: str = DEFAULT_MODEL
    adapter: Optional[str] = None      # optional PEFT LoRA adapter directory
    max_new_tokens: int = 1536
    repetition_penalty: float = 1.0  # 1.0 = off; >1 fights JSON's needed repetition, so left off
    seed: int = 0

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        torch.manual_seed(self.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        if self.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter)
        self.model.eval()

    def _format(self, prompt: str) -> str:
        """Apply the chat template; disable Qwen3 'thinking' so we get direct JSON."""
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # older template without enable_thinking
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    def generate(self, prompt: str) -> str:
        torch = self._torch
        text = self._format(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy / deterministic
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
