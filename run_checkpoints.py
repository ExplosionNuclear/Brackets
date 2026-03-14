import random
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from generate_dataset import generate_dataset

CHECKPOINTS = ["step1", "step2", "step16", "step128", "step256", "step1000", "step2000", "step5000", "step10000", "step143000"]
PRECOMPUTED_NPZS = {}
MODEL_NAME = "EleutherAI/pythia-160m"
N_SAMPLES = 10000
SEED = 42
BATCH_SIZE = 32


def is_valid_dyck(seq):
    stack = []
    matching = {")": "(", "]": "[", "}": "{"}
    for char in seq.replace(" ", ""):
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
    return len(stack) == 0


def corrupt_dyck(valid_seq, rng):
    chars = list(valid_seq.replace(" ", ""))
    # Try adjacent swap first, fall back to random swap
    i, j = rng.sample(range(len(chars)), 2)
    chars[i], chars[j] = chars[j], chars[i]
    return " ".join(chars)


def build_sequences(n_samples, seed, corrupt_ratio=0.5):
    dataset = generate_dataset(n_samples, seed)
    rng = random.Random(seed)
    sequences, labels = [], []

    for item in dataset:
        if rng.random() < corrupt_ratio:
            seq = corrupt_dyck(item["sequence"], rng)
            while is_valid_dyck(seq):
                seq = corrupt_dyck(item["sequence"], rng)
            labels.append(False)
        else:
            seq = item["sequence"]
            labels.append(True)
        sequences.append(seq)

    seen = set()
    unique_idx = []
    for i, seq in enumerate(sequences):
        if seq not in seen:
            seen.add(seq)
            unique_idx.append(i)
    sequences = [sequences[i] for i in unique_idx]
    labels = [labels[i] for i in unique_idx]

    return sequences, labels


@torch.no_grad()
def extract_embeddings(sequences, model, tokenizer, batch_size):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

        mask = inputs["attention_mask"].unsqueeze(-1)
        mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(mean_pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def run_probe(X, y):
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return scores.mean(), scores.std()


def main():
    print(f"Building sequences (n={N_SAMPLES})...")
    sequences, labels = build_sequences(N_SAMPLES, SEED)
    y = np.array(labels, dtype=int)
    print(f"  Unique: {len(sequences)}, Valid: {y.sum()}, Invalid: {(1 - y).sum()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Random init baseline ===")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="step1")
    model.apply(model._init_weights)
    model.eval()
    emb_random = extract_embeddings(sequences, model, tokenizer, BATCH_SIZE)
    acc_rand, std_rand = run_probe(emb_random, y)
    print(f"  Random init probe accuracy: {acc_rand:.3f} ± {std_rand:.3f}")
    del model
    torch.cuda.empty_cache()

    results = []

    for ckpt in CHECKPOINTS:
        print(f"\n=== {ckpt} ===")

        if ckpt in PRECOMPUTED_NPZS:
            data = np.load(PRECOMPUTED_NPZS[ckpt], allow_pickle=True)
            embeddings = data["embeddings"]
            y_pre = data["labels"].astype(int)
            print(f"  Loaded precomputed embeddings: {embeddings.shape}")
            acc, std = run_probe(embeddings, y_pre)
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=ckpt)
            model.eval()
            embeddings = extract_embeddings(sequences, model, tokenizer, BATCH_SIZE)
            print(f"  Embeddings: {embeddings.shape}")
            acc, std = run_probe(embeddings, y)
            del model
            torch.cuda.empty_cache()

        print(f"  Linear probe accuracy: {acc:.3f} ± {std:.3f}")
        results.append({"checkpoint": ckpt, "accuracy": acc, "std": std})

    with open("checkpoint_results.json", "w") as f:
        json.dump(results, f, indent=2)

    steps = [int(r["checkpoint"].replace("step", "")) for r in results]
    accs = [r["accuracy"] for r in results]
    stds = [r["std"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(steps, accs, yerr=stds, marker="o", capsize=4, linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", label="Chance level")
    ax.axhline(y=acc_rand, color="red", linestyle="--", label=f"Random init: {acc_rand:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title(f"Dyck Language Linear Probe — {MODEL_NAME}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("checkpoint_probe.png", dpi=300)
    print(f"\nSaved plot to checkpoint_probe.png")


if __name__ == "__main__":
    main()
