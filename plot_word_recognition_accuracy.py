import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("asr_test_results.csv")
OUTPUT_PATH = Path("word_recognition_accuracy.png")


def normalise(text):
    if pd.isna(text):
        return []
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


def word_alignment_counts(expected_words, actual_words):
    m, n = len(expected_words), len(actual_words)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected_words[i - 1] == actual_words[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # match/substitution
            )

    i, j = m, n
    correct = substitutions = deletions = insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and expected_words[i - 1] == actual_words[j - 1]:
            correct += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return correct, substitutions, deletions, insertions


def main():
    df = pd.read_csv(CSV_PATH)

    total_correct = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0

    for _, row in df.iterrows():
        expected_words = normalise(row["expected_phrase"])
        actual_words = normalise(row["transcript"])

        correct, subs, dels, ins = word_alignment_counts(expected_words, actual_words)

        total_correct += correct
        total_substitutions += subs
        total_deletions += dels
        total_insertions += ins

    total_incorrect = total_substitutions + total_deletions + total_insertions
    total_compared = total_correct + total_incorrect

    word_accuracy = (total_correct / total_compared) * 100 if total_compared else 0
    word_error_rate = (total_incorrect / total_compared) * 100 if total_compared else 0

    print(f"Correct words: {total_correct}")
    print(f"Incorrect words: {total_incorrect}")
    print(f"Substitutions: {total_substitutions}")
    print(f"Deletions: {total_deletions}")
    print(f"Insertions: {total_insertions}")
    print(f"Word accuracy: {word_accuracy:.2f}%")
    print(f"Word error rate: {word_error_rate:.2f}%")

    labels = ["Correct words", "Incorrect words"]
    values = [total_correct, total_incorrect]
    colours = ["blue", "red"]
    percentages = [
        (total_correct / total_compared) * 100 if total_compared else 0,
        (total_incorrect / total_compared) * 100 if total_compared else 0
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colours)

    plt.title("Vosk Word Recognition Accuracy")
    plt.ylabel("Word count")
    plt.ylim(0, max(values) * 1.2)

    for bar, value, percentage in zip(bars, values, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value} ({percentage:.2f}%)",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()