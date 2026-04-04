from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import Counter
import re
import requests
import torch


@dataclass
class LLMRuntime:
    tokenizer: Any
    model: Any
    device: torch.device | None
    remote_url: str | None = None


def load_llm(model_dir: Path | None, remote_url: str = "") -> LLMRuntime:
    remote_url = (remote_url or "").strip()
    if remote_url:
        return LLMRuntime(tokenizer=None, model=None, device=None, remote_url=remote_url)
    # if model_dir is None:
    #     raise RuntimeError("REMOTE_LLM_URL is empty and no local model path was provided.")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
    from peft import PeftConfig, PeftModel  # lazy import

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return LLMRuntime(tokenizer=tokenizer, model=model, device=device)


def compress_sequence(events, max_unique=20, max_total=40):
    counts = Counter(events)
    compressed = []
    seen = set()
    for event in events:
        if event not in seen:
            seen.add(event)
            count = counts[event]
            if count > 1:
                compressed.append(f"{event}")
            else:
                compressed.append(event)
        if len(compressed) >= max_unique:
            break
    return compressed

def build_prompt(log_seq: str) -> str:
    seq = compress_sequence(log_seq)
    instruction = (
        "Analyze the following log sequence. Determine if it is normal or abnormal. "
        "Provide a concise explanation of the root cause if abnormal, "
        "and suggest a specific, actionable solution to resolve the issue."
    )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{seq}\n\n"
        f"### Output:\nVerdict: "
    )


def generate_analysis(runtime: LLMRuntime, log_seq: str, max_new_tokens: int = 200) -> str:
    prompt = build_prompt(log_seq)
    if runtime.remote_url:
        url = runtime.remote_url.rstrip("/")
        if not url.endswith("/generate"):
            url = f"{url}/generate"
        resp = requests.post(
            url,
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "text" not in data:
            raise ValueError("Remote LLM response missing 'text' field.")
        return _strip_prompt(data["text"])

    inputs = runtime.tokenizer(prompt, return_tensors="pt").to(runtime.device)
    with torch.inference_mode():
        outputs = runtime.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=runtime.tokenizer.eos_token_id,
        )
    text = runtime.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _strip_prompt(text)


def _strip_prompt(text: str) -> str:
    """
    Remove the prompt boilerplate from model output, keeping only the generated answer.
    """
    marker = "### Output:"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip()


def parse_llm_response(text: str) -> dict[str, str]:
    """
    Extract verdict/explanation/solution from the LLM response when present.
    Keeps fields only if they exist.
    """
    cleaned = text.strip()
    result: dict[str, str] = {}

    def _extract(label: str) -> str | None:
        pattern = rf"{label}\\s*:\\s*(.*)"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if not match:
            return None
        value = match.group(1).strip()
        return value if value else None

    verdict = _extract("Verdict")
    explanation = _extract("Explanation")
    solution = _extract("Solution")

    if verdict:
        result["verdict"] = verdict.lower()
    if explanation:
        result["explanation"] = explanation
    if solution:
        result["solution"] = solution

    return result
