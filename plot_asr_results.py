from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = Path("results/asr_test_results.csv")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS_CSV)

def yes_rate(series):
    clean = series.fillna("").astype(str).str.lower().str.strip()
    valid = clean[clean.isin(["yes", "no", "partial"])]
    if len(valid) == 0:
        return 0
    return (valid == "yes").mean() * 100

def bool_rate(series):
    clean = series.fillna(False)
    return clean.astype(str).str.lower().isin(["true", "1", "yes"]).mean() * 100

summary = {
    "Transcript produced": bool_rate(df["transcript_produced"]),
    "Transcript correct": yes_rate(df["manual_transcript_correct"]),
    "Intent correct": bool_rate(df["intent_correct"]),
    "Task success": yes_rate(df["manual_task_success"]),
}

summary_df = pd.DataFrame(
    [{"metric": key, "success_rate": value} for key, value in summary.items()]
)
summary_df.to_csv(OUT_DIR / "asr_summary_metrics.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(summary.keys(), summary.values())
plt.ylim(0, 100)
plt.ylabel("Success rate (%)")
plt.title("Vosk ASR and Task Pipeline Success Rates")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "asr_pipeline_success_rates.png", dpi=300)
plt.close()

intent_counts = df["parsed_intent"].fillna("unknown").value_counts()

plt.figure(figsize=(8, 5))
plt.bar(intent_counts.index, intent_counts.values)
plt.ylabel("Count")
plt.title("Parsed Intent Distribution During Vosk Testing")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "parsed_intent_distribution.png", dpi=300)
plt.close()

print("Saved:")
print(OUT_DIR / "asr_summary_metrics.csv")
print(OUT_DIR / "asr_pipeline_success_rates.png")
print(OUT_DIR / "parsed_intent_distribution.png")