import random
import json
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from generate_dataset import generate_dataset


def is_valid_dyck(seq: str) -> bool:
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


def corrupt_dyck(valid_seq: str, rng: random.Random) -> str:
    chars = list(valid_seq.replace(" ", ""))
    i, j = rng.sample(range(len(chars)), 2)
    chars[i], chars[j] = chars[j], chars[i]
    return " ".join(chars)


def build_sequences(n_samples=1000, seed=42, corrupt_ratio=0.5):
    dataset = generate_dataset(n_samples, seed)
    rng = random.Random(seed)
    sequences = []
    labels = []
    metadata = []

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
        metadata.append({"n_pairs": item["n_pairs"], "id": item["id"]})

    return sequences, labels, metadata


@torch.no_grad()
def extract_embeddings(sequences, model, tokenizer, batch_size=32):
    embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked = last_hidden * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = (summed / counts).cpu().numpy()

        embeddings.append(mean_pooled)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(sequences)}")

    return np.concatenate(embeddings, axis=0)


def plot_tsne(embeddings_2d, labels, metadata, sequences, output_path="dyck_tsne.png"):
    valid = np.array(labels)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=embeddings_2d[valid, 0], y=embeddings_2d[valid, 1],
        mode="markers",
        marker=dict(color="green", size=5, opacity=0.6),
        text=[sequences[i] for i in range(len(sequences)) if valid[i]],
        hoverinfo="text",
        name="Valid",
    ))

    fig.add_trace(go.Scatter(
        x=embeddings_2d[~valid, 0], y=embeddings_2d[~valid, 1],
        mode="markers",
        marker=dict(color="red", size=5, opacity=0.6),
        text=[sequences[i] for i in range(len(sequences)) if not valid[i]],
        hoverinfo="text",
        name="Invalid",
    ))

    fig.update_layout(
        title="t-SNE of Dyck Language Sequences",
        width=1000,
        height=800,
        hovermode="closest",
    )

    html_path = output_path.rsplit(".", 1)[0] + ".html"
    fig.write_html(html_path)
    print(f"Saved interactive plot to {html_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corrupt_ratio", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_plot", type=str, default="dyck_tsne.png")
    parser.add_argument("--output_embeddings", type=str, default="dyck_embeddings.npz")
    args = parser.parse_args()

    label = args.model_name.split("/")[-1]
    if args.revision:
        label += f"_{args.revision}"

    print(f"Loading model {args.model_name} (revision={args.revision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.revision, output_hidden_states=True)
    model.eval()

    print(f"Building sequences (n={args.n_samples}, corrupt_ratio={args.corrupt_ratio})...")
    sequences, labels, metadata = build_sequences(args.n_samples, args.seed, args.corrupt_ratio)

    seen = set()
    unique_idx = []
    for i, seq in enumerate(sequences):
        if seq not in seen:
            seen.add(seq)
            unique_idx.append(i)
    sequences = [sequences[i] for i in unique_idx]
    labels = [labels[i] for i in unique_idx]
    metadata = [metadata[i] for i in unique_idx]

    print(f"  Unique: {len(sequences)}, Valid: {sum(labels)}, Invalid: {sum(not l for l in labels)}")

    print("Extracting embeddings...")
    embeddings = extract_embeddings(sequences, model, tokenizer, args.batch_size)
    print(f"  Shape: {embeddings.shape}")

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed)
    embeddings_2d = tsne.fit_transform(embeddings)

    output_emb = args.output_embeddings.rsplit(".", 1)[0] + f"_{label}.npz"
    output_plot = args.output_plot.rsplit(".", 1)[0] + f"_{label}.png"

    np.savez(
        output_emb,
        embeddings=embeddings,
        embeddings_2d=embeddings_2d,
        labels=np.array(labels),
        sequences=np.array(sequences),
        n_pairs=np.array([m["n_pairs"] for m in metadata]),
    )
    print(f"Saved embeddings to {output_emb}")

    plot_tsne(embeddings_2d, labels, metadata, sequences, output_plot)


if __name__ == "__main__":
    main()
