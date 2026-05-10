import json
from datetime import datetime

LOG_PATH = "logs/decision_traces.jsonl"
MIN_AVG_SCORE = 0.35


def load_logs():
    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def monitor():
    records = load_logs()

    scores = [
        r["explanation"]["final_score"]
        for r in records
        if "explanation" in r
    ]

    avg = sum(scores) / len(scores)

    print("\n=== SYSTEM HEALTH REPORT ===")
    print("Total decisions:", len(scores))
    print("Average Final Score:", round(avg, 3))

    if avg < MIN_AVG_SCORE:
        print("⚠ WARNING: Recommendation quality degrading")
    else:
        print("System health: OK")


if __name__ == "__main__":
    monitor()
