import random
import json
import argparse

BRACKETS = [("(", ")"), ("[", "]"), ("{", "}")]


def generate_dyck_random_walk(target_pairs: int, bracket_types=BRACKETS) -> str:
    stack = []
    result = []
    pairs_placed = 0

    while pairs_placed < target_pairs or stack:
        can_open = pairs_placed < target_pairs
        can_close = len(stack) > 0

        if can_open and can_close:
            if random.random() < 0.5:
                b = random.choice(bracket_types)
                result.append(b[0])
                stack.append(b[1])
                pairs_placed += 1
            else:
                result.append(stack.pop())
        elif can_open:
            b = random.choice(bracket_types)
            result.append(b[0])
            stack.append(b[1])
            pairs_placed += 1
        else:
            result.append(stack.pop())

    return " ".join(result)


def sample_target_pairs() -> int:
    return random.choice([3, 4, 5, 6, 7, 8])


def generate_dataset(n_samples: int = 10000, seed: int = 42) -> list[dict]:
    random.seed(seed)
    dataset = []

    for i in range(n_samples):
        target_pairs = sample_target_pairs()
        seq = generate_dyck_random_walk(target_pairs)
        dataset.append({
            "id": i,
            "sequence": seq,
            "n_pairs": target_pairs,
        })

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="dyck_dataset.json")
    args = parser.parse_args()

    dataset = generate_dataset(args.n_samples, args.seed)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} samples, saved to {args.output}")