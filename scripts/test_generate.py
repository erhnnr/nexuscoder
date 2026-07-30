"""
Kod-LLM v4 - Hizli uretim testi.
"""
import os
import sys
import torch
from tokenizers import Tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_v1 import KodLLM_v1

CHECKPOINT_PATH = "/content/drive/MyDrive/nexus_checkpoints/model_v4_best.pt"
TOKENIZER_PATH = "/content/nexuscoder/tokenizer/tokenizer_v4.json"

PROMPTS = [
    "def factorial(n):",
    "import numpy as np\n\ndef",
    "function calculateTotal(",
    "#include <stdio.h>\n\nint main(",
    "class ",
    "def is_prime(n):",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}\n")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = KodLLM_v1(vocab_size=checkpoint["vocab_size"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint yuklendi (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})\n")

    bos_id = tokenizer.token_to_id("<bos>")

    for prompt in PROMPTS:
        ids = [bos_id] + tokenizer.encode(prompt).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        output_ids = model.generate(input_ids, max_new_tokens=60, temperature=0.8, top_k=40)
        output_text = tokenizer.decode(output_ids[0].tolist())

        print("=" * 60)
        print(f"PROMPT: {prompt!r}")
        print("-" * 60)
        print(output_text)
        print()


if __name__ == "__main__":
    main()
