import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def run_probe(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["embeddings"]
    y = data["labels"].astype(int)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} dims")
    print(f"  Valid: {y.sum()}, Invalid: {(1 - y).sum()}")

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"\nLinear probe accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Per fold: {', '.join(f'{s:.3f}' for s in scores)}")
    print(f"  Chance level: {max(y.mean(), 1 - y.mean()):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    args = parser.parse_args()

    run_probe(args.npz_path)
