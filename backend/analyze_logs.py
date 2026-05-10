import json
from collections import Counter

LOG_PATH = "logs/decision_traces.jsonl"


def load_logs():
    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def analyze():
    records = load_logs()

    print("\n=== LOG ANALYSIS REPORT ===")
    print("Total decisions:", len(records))

    intents = Counter()
    forms = Counter()

    valid_records = []

    for r in records:
        if "explanation" not in r:
            continue

        valid_records.append(r)

        intents[r["explanation"]["detected_user_intent"]] += 1

        form_dict = r["explanation"]["inferred_book_form"]
        if form_dict:
            primary_form = max(form_dict, key=form_dict.get)
            forms[primary_form] += 1

    print("\nMost common user intents:")
    for k, v in intents.most_common():
        print(f"  {k}: {v}")

    print("\nMost recommended book forms:")
    for k, v in forms.most_common():
        print(f"  {k}: {v}")

    avg_score = sum(
        r["explanation"]["final_score"] for r in valid_records
    ) / len(valid_records)

    print("\nAverage final score:", round(avg_score, 3))


if __name__ == "__main__":
    analyze()
