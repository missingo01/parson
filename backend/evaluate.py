import pandas as pd
from recommender import (
    detect_user_intents,
    infer_book_form_scores
)

# ---------------------------
# Load evaluation dataset
# ---------------------------

DATASET_PATH = "data/evaluation_queries.csv"

df = pd.read_csv(DATASET_PATH)

intent_correct = 0
form_correct = 0
total = len(df)

# ---------------------------
# Evaluation loop
# ---------------------------

mistakes = []

for _, row in df.iterrows():
    query = row["query"]
    expected_intent = row["expected_intent"]
    expected_form = row["expected_book_form"]

    intents = detect_user_intents(query)
    predicted_intent = None
    if intents:
        predicted_intent = max(intents, key=intents.get)

    fake_book = {
        "title": "",
        "synopsis": query,
        "themes": ""
    }

    form_scores = infer_book_form_scores(fake_book)
    predicted_form = max(form_scores, key=form_scores.get)

    if predicted_intent != expected_intent or predicted_form != expected_form:
        mistakes.append({
            "query": query,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "expected_form": expected_form,
            "predicted_form": predicted_form
        })

    if predicted_intent == expected_intent:
        intent_correct += 1

    if predicted_form == expected_form:
        form_correct += 1


# ---------------------------
# Results
# ---------------------------

print("\n=== PARSON Evaluation Report ===")
print(f"Total Samples: {total}")
print(f"Intent Accuracy: {intent_correct / total:.2f}")
print(f"Book Form Accuracy: {form_correct / total:.2f}")
print("\n--- Mistakes ---")
for m in mistakes:
    print(m)

