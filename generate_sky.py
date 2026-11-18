"""Generate 100 numbered continuations using the existing trained TinyLM.

Run from the project folder with the existing virtual environment activated.
This imports the architecture only; train_tiny_lm.main() is not called.
"""
import argparse
import math
import os
from pathlib import Path
import sys

import tiktoken
import torch

from train_tiny_lm import TinyLM


PROMPT = "The sky is"
COMPLETIONS = 100
PROJECT = Path(__file__).resolve().parent


@torch.inference_mode()
def complete(model, encoding, prompt_ids, mapping, generator, count, temperature):
    ids = list(prompt_ids)
    generated_original_ids = []
    for _ in range(count):
        context = torch.tensor([ids[-model.context:]], dtype=torch.long, device="cpu")
        logits = model(context)[0, -1] / temperature
        probabilities = torch.softmax(logits, dim=-1)
        internal_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
        ids.append(internal_id)
        generated_original_ids.append(mapping[internal_id])
    return PROMPT + encoding.decode(generated_original_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "runs/20261116-134120-415641/model_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=PROJECT / "sky_completions.txt")
    parser.add_argument("--new-tokens", type=int, default=40,
                        help="New tokens per completion (default: 40, matching training samples)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234,
                        help="Completion n uses seed + n - 1")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.new_tokens < 1 or args.threads < 1:
        parser.error("--new-tokens and --threads must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not 0 <= args.seed <= 2**63 - COMPLETIONS:
        parser.error("--seed must be between 0 and 2**63 - 100")
    torch.set_num_threads(args.threads)
    # The training checkpoint contains tensors and basic Python containers.
    # Load onto CPU using PyTorch's restricted weights-only loader.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    required = {"encoding", "model_config", "model_state_dict", "internal_to_original"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing the fields saved by train_tiny_lm.py")
    if checkpoint["encoding"] != "o200k_base":
        raise ValueError(f"Expected o200k_base, found {checkpoint['encoding']!r}")
    encoding = tiktoken.get_encoding(checkpoint["encoding"])
    mapping = checkpoint["internal_to_original"]
    config = checkpoint["model_config"]
    if not isinstance(mapping, list) or not mapping or any(type(t) is not int for t in mapping):
        raise ValueError("Saved internal_to_original must be a nonempty list of integer IDs")
    if len(set(mapping)) != len(mapping) or len(mapping) != config["vocab_size"]:
        raise ValueError("Saved vocabulary mapping is duplicated or has the wrong size")
    for original_id in mapping:
        encoding.decode_single_token_bytes(original_id)  # Validate the saved tokenizer IDs.
    original_to_internal = {original_id: internal_id for internal_id, original_id in enumerate(mapping)}
    prompt_original_ids = encoding.encode(PROMPT, disallowed_special=())
    missing = sorted(set(prompt_original_ids) - original_to_internal.keys())
    if missing:
        print(f"Cannot generate: prompt {PROMPT!r} contains tokens absent from the model vocabulary.", file=sys.stderr)
        for token_id in missing:
            piece = encoding.decode_single_token_bytes(token_id)
            print(f"  o200k_base ID {token_id}: bytes={piece!r}; text={piece.decode('utf-8', errors='replace')!r}", file=sys.stderr)
        print("The prompt was not changed. No completions were generated; the output file was not written.", file=sys.stderr)
        return 1
    prompt_ids = [original_to_internal[t] for t in prompt_original_ids]
    if len(prompt_ids) > config["context"]:
        raise ValueError("The exact prompt exceeds the saved context length; refusing to truncate it")
    model = TinyLM(**config).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    # Exclusive creation protects previous results from accidental overwriting.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(f"Prompt: {PROMPT!r}\nCheckpoint: {args.checkpoint.resolve()}\n")
        output.write(f"Completions: {COMPLETIONS}; new tokens each: {args.new_tokens}; temperature: {args.temperature}\n")
        output.write(f"Seeds: {args.seed} through {args.seed + COMPLETIONS - 1}\n\n")
        for number in range(1, COMPLETIONS + 1):
            generator = torch.Generator(device="cpu").manual_seed(args.seed + number - 1)
            text = complete(model, encoding, prompt_ids, mapping, generator,
                            args.new_tokens, args.temperature)
            output.write(f"[{number:03d}]\n{text}\n\n")
            output.flush()
            print(f"Saved completion {number}/{COMPLETIONS}", flush=True)
    print(f"Saved numbered results to {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
