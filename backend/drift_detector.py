import json
from collections import Counter
from math import sqrt

# ---------------------------
# Configuration
# ---------------------------

LOG_PATH = "logs/decision_traces.jsonl"
BASELINE_PATH = "logs/baseline_queries.json"
DRIFT_THRESHOLD = 0.35


# ---------------------------
# Load recent queries
# ---------------------------

def load_recent_queries():
    queries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            queries.append(record["query"].lower())
    return queries


# ---------------------------
# Tokenization
# ---------------------------

def tokenize(text):
    return text.split()


# ---------------------------
# Build word distribution
# ---------------------------

def word_distribution(queries):
    counter = Counter()
    for q in queries:
        for w in tokenize(q):
            counter[w] += 1

    total = sum(counter.values())

    if total == 0:
        return {}

    return {w: c / total for w, c in counter.items()}


# ---------------------------
# Cosine distance between distributions
# ---------------------------

def cosine_distance(dist1, dist2):
    words = set(dist1.keys()) | set(dist2.keys())

    dot = sum(dist1.get(w, 0) * dist2.get(w, 0) for w in words)
    mag1 = sqrt(sum(v * v for v in dist1.values()))
    mag2 = sqrt(sum(v * v for v in dist2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return 1 - (dot / (mag1 * mag2))


# ---------------------------
# Baseline management
# ---------------------------

def save_baseline(distribution):
    with open(BASELINE_PATH, "w") as f:
        json.dump(distribution, f)


def load_baseline():
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


# ---------------------------
# Main
# ---------------------------

def main():
    recent_queries = load_recent_queries()
    recent_dist = word_distribution(recent_queries)

    try:
        baseline = load_baseline()
    except FileNotFoundError:
        print("No baseline found. Creating baseline...")
        save_baseline(recent_dist)
        return

    drift_score = cosine_distance(baseline, recent_dist)

    print("\n=== DRIFT REPORT ===")
    print("Drift score:", round(drift_score, 3))

    if drift_score >= DRIFT_THRESHOLD:
        print("⚠ RETRAINING RECOMMENDED")
    else:
        print("✅ Model behavior stable")


# ---------------------------

if __name__ == "__main__":
    main()
