import random
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    Score, Target, CORRECT, INCORRECT, accuracy, mean, stderr, scorer,
)
from inspect_ai.solver import TaskState, generate

from generate_dataset import generate_dataset

TEMPLATE = (Path(__file__).parent / "template.txt").read_text()


def _extract_yes_no_logprobs(state: TaskState):
    choice = state.output.choices[0]
    if not choice.logprobs or not choice.logprobs.content:
        return None, None, {}

    first_token = choice.logprobs.content[0]
    top = {lp.token: lp.logprob for lp in first_token.top_logprobs}

    prob_yes = max(
        top.get("Yes", float("-inf")),
        top.get("\u0120Yes", float("-inf")),
    )
    prob_no = max(
        top.get("No", float("-inf")),
        top.get("\u0120No", float("-inf")),
    )

    if prob_yes == float("-inf"):
        prob_yes = None
    if prob_no == float("-inf"):
        prob_no = None

    return prob_yes, prob_no, top


@scorer(metrics=[accuracy(), stderr()])
def logprobs_match():
    async def score(state: TaskState, target: Target):
        prob_yes, prob_no, top = _extract_yes_no_logprobs(state)

        if prob_yes is None or prob_no is None:
            top_tokens = sorted(top.items(), key=lambda x: x[1], reverse=True)[:10]
            return Score(value=INCORRECT, answer="N/A", explanation=f"Yes/No missing. Top tokens: {top_tokens}")

        predicted = "Yes" if prob_yes >= prob_no else "No"
        correct = predicted == target.text
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=predicted,
            explanation=f"logprob Yes={prob_yes:.3f}, No={prob_no:.3f}",
        )

    return score


@scorer(metrics=[mean(), stderr()])
def logprob_difference():
    async def score(state: TaskState, target: Target):
        prob_yes, prob_no, _ = _extract_yes_no_logprobs(state)

        if prob_yes is None or prob_no is None:
            return Score(value=0.0, answer="N/A", explanation="Yes/No missing from logprobs")

        if target.text == "Yes":
            diff = prob_yes - prob_no
        else:
            diff = prob_no - prob_yes

        return Score(
            value=diff,
            answer=f"Yes={prob_yes:.3f}, No={prob_no:.3f}",
            explanation=f"diff={diff:.3f} (positive=correct direction)",
        )

    return score


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
    corruption = rng.choice(["swap", "remove", "replace", "duplicate"])

    if corruption == "swap":
        i = rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif corruption == "remove":
        i = rng.randint(0, len(chars) - 1)
        chars.pop(i)
    elif corruption == "replace":
        i = rng.randint(0, len(chars) - 1)
        all_brackets = "()[]{}"
        chars[i] = rng.choice([b for b in all_brackets if b != chars[i]])
    elif corruption == "duplicate":
        i = rng.randint(0, len(chars) - 1)
        chars.insert(i, chars[i])

    return " ".join(chars)


def build_samples(n_samples=10000, seed=42, corrupt_ratio=0.5):
    dataset = generate_dataset(n_samples, seed)
    rng = random.Random(seed)
    samples = []

    for item in dataset:
        if rng.random() < corrupt_ratio:
            seq = corrupt_dyck(item["sequence"], rng)
            while is_valid_dyck(seq):
                seq = corrupt_dyck(item["sequence"], rng)
            target = "No"
        else:
            seq = item["sequence"]
            target = "Yes"

        samples.append(Sample(
            input=TEMPLATE.format(sequence=seq),
            target=target,
            metadata={
                "sequence": seq,
                "n_pairs": item["n_pairs"],
                "is_valid": target == "Yes",
            },
        ))

    return samples


@task
def dyck_language(n_samples=10000, seed=42, corrupt_ratio=0.5):
    return Task(
        dataset=MemoryDataset(build_samples(n_samples, seed, corrupt_ratio)),
        solver=[generate()],
        scorer=[logprobs_match(), logprob_difference()],
        config=GenerateConfig(max_tokens=1, logprobs=True, top_logprobs=100),
    )
