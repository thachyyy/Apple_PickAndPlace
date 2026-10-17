"""Instruction -> (color, count). count=None means every apple of that color."""

import json
import re

from .config import COLORS

HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # small, multilingual (incl. Vietnamese), runs on CPU
LLM_SYSTEM_PROMPT = """You convert a robot instruction (English or Vietnamese) into JSON.
The scene has red apples and green apples on a table and one box.
Reply with ONLY one JSON object: {"color": "red"|"green"|"unknown", "pick_all": true|false, "count": <integer>}
- color: which apple colour to put into the box; "unknown" if no colour is named or the
  instruction is not about putting apples into the box.
- pick_all: true when the user wants every apple of that colour ("all", "every", "tất cả",
  "hết", or a plural with no number such as "the red apples", "những quả táo đỏ").
- count: how many apples when pick_all is false; 0 when pick_all is true.

Examples:
"Put the red apple into the box." -> {"color": "red", "pick_all": false, "count": 1}
"Put the green apples into the box." -> {"color": "green", "pick_all": true, "count": 0}
"Đặt toàn bộ táo đỏ vào hộp" -> {"color": "red", "pick_all": true, "count": 0}
"Cho hai quả táo xanh vào hộp" -> {"color": "green", "pick_all": false, "count": 2}
"Đưa tôi cái hộp" -> {"color": "unknown", "pick_all": false, "count": 0}"""


class InstructionError(ValueError):
    """The instruction was read but does not describe a valid task."""


def parse_instruction_rule(instruction):
    """Minimal deterministic English keyword parser.

    'Put the red apple into the box.'        -> ("red", 1)
    'Put all the green apples into the box.' -> ("green", None)
    """
    words = re.findall(r"[a-z]+", instruction.lower())
    if not {"apple", "apples"} & set(words):
        raise InstructionError("Instruction must mention 'apple'.")
    colors = [c for c in COLORS if c in words]
    if len(colors) != 1:
        raise InstructionError("Specify exactly one color: 'red apple' or 'green apple'.")
    pick_all = bool({"all", "every", "each"} & set(words))
    return colors[0], None if pick_all else 1


def parse_instruction_llm(instruction, model_name=HF_MODEL):
    """Read the instruction's meaning with a local Hugging Face model."""
    # Optional dependencies, only needed for the LLM parser.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} (first run downloads the weights)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("LLM raw reply:", reply.strip())
    return interpret_llm_reply(reply)


def interpret_llm_reply(reply):
    """Extract and validate the JSON a small model produced (it has no schema guarantee)."""
    match = re.search(r"\{.*?\}", reply, re.DOTALL)
    if not match:
        raise RuntimeError("LLM reply contains no JSON object.")
    data = json.loads(match.group(0))
    color, pick_all, count = data.get("color"), data.get("pick_all"), data.get("count")
    valid = (
        color in (*COLORS, "unknown")
        and isinstance(pick_all, bool)
        and isinstance(count, int)
        and not isinstance(count, bool)  # bool is a subclass of int
    )
    if not valid:
        raise RuntimeError(f"LLM reply does not match the expected format: {data}")
    print("LLM parsed instruction:", data)

    if color == "unknown":
        raise InstructionError("LLM could not find an apple colour (red/green) in the instruction.")
    return color, None if pick_all else max(1, count)


def parse_instruction(instruction, use_llm=True, model_name=HF_MODEL):
    """Prefer the LLM parser; fall back to the keyword parser if it is unavailable or fails."""
    if use_llm:
        try:
            return parse_instruction_llm(instruction, model_name)
        except InstructionError:
            raise
        except Exception as exc:  # missing packages, download failure, malformed reply, ...
            print(f"LLM parser failed ({type(exc).__name__}: {exc}); using rule parser.")
    return parse_instruction_rule(instruction)
