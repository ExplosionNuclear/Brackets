import argparse
import numpy as np
import plotly.graph_objects as go


def plot_tsne_interactive(npz_path, output_html=None):
    data = np.load(npz_path, allow_pickle=True)
    embeddings_2d = data["embeddings_2d"]
    labels = data["labels"]
    sequences = data["sequences"]

    valid = labels.astype(bool)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=embeddings_2d[valid, 0], y=embeddings_2d[valid, 1],
        mode="markers",
        marker=dict(color="green", size=5, opacity=0.6),
        text=sequences[valid],
        hoverinfo="text",
        name="Valid",
    ))

    fig.add_trace(go.Scatter(
        x=embeddings_2d[~valid, 0], y=embeddings_2d[~valid, 1],
        mode="markers",
        marker=dict(color="red", size=5, opacity=0.6),
        text=sequences[~valid],
        hoverinfo="text",
        name="Invalid",
    ))

    fig.update_layout(
        title="t-SNE of Dyck Language Sequences",
        width=1000,
        height=800,
        hovermode="closest",
    )

    if output_html is None:
        output_html = npz_path.rsplit(".", 1)[0] + ".html"

    fig.write_html(output_html)
    print(f"Saved interactive plot to {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    plot_tsne_interactive(args.npz_path, args.output)
