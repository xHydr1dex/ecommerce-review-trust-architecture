import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- Load dataset ---
df = pd.read_csv("more_simple_reviews.csv")

# --- RELAXED RULE-BASED FLAGS ---

# Rule 1: Duplicate review texts appearing 3 or more times
duplicate_counts = df['review_text'].value_counts()
duplicates_3plus = duplicate_counts[duplicate_counts >= 3].index.tolist()
duplicate_reviews = df[df['review_text'].isin(duplicates_3plus)]

# Rule 2: Positive reviews from unverified purchases with stars >= 4.5
unverified_positives = df[(df["star_rating"] >= 4.5) & (df["verified_purchase"] == False)]

# Rule 3: Generic short review texts where keywords appear at least twice and length < 20
generic_keywords = ["great", "nice", "awesome", "excellent", "good"]
def has_multiple_keywords(text):
    text_lower = text.lower()
    count = sum(text_lower.count(word) for word in generic_keywords)
    return count >= 2

df["is_generic_relaxed"] = df["review_text"].apply(lambda x: has_multiple_keywords(x) and len(x) < 20)
short_generics = df[df["is_generic_relaxed"]]

# Combine rule-based flags
rule_flagged_ids = set(pd.concat([duplicate_reviews, unverified_positives, short_generics])["review_id"])

# --- ML SIMILARITY CHECK ---

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["review_text"].tolist(), convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings, embeddings)

ml_flagged_ids = set()
threshold = 0.7
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if cosine_scores[i][j] > threshold:
            ml_flagged_ids.add(df.iloc[i]["review_id"])
            ml_flagged_ids.add(df.iloc[j]["review_id"])

# --- COMBINE FLAGS ---
df["rule_flagged"] = df["review_id"].isin(rule_flagged_ids)
df["ml_flagged"] = df["review_id"].isin(ml_flagged_ids)
df["suspicious"] = df["rule_flagged"] | df["ml_flagged"]

# Assign trust scores
def compute_trust(row):
    if row["rule_flagged"] and row["ml_flagged"]:
        return 0.2
    elif row["rule_flagged"] or row["ml_flagged"]:
        return 0.5
    else:
        return 1

df["trust_score"] = df.apply(compute_trust, axis=1)

# Show suspicious reviews
flagged_reviews = df[df["suspicious"]]
print("🔍 Combined suspicious reviews (Rule-based or ML-based):\n")
for _, row in flagged_reviews.iterrows():
    print(f"⚠️ Review ID {row['review_id']}:")
    print(f"  - Text: {row['review_text']}")
    print(f"  - Stars: {row['star_rating']} | Verified: {row['verified_purchase']}")
    print(f"  - Rule-flagged: {row['rule_flagged']} | ML-flagged: {row['ml_flagged']}\n")

only_rule = rule_flagged_ids - ml_flagged_ids
only_ml = ml_flagged_ids - rule_flagged_ids
both = rule_flagged_ids & ml_flagged_ids

print(f"Total reviews: {len(df)}")
print(f"Rule-based flagged reviews: {len(rule_flagged_ids)}")
print(f"ML-based flagged reviews: {len(ml_flagged_ids)}")
print(f"Overlap flagged reviews: {len(both)}")
print(f"Only rule-based flagged: {len(only_rule)}")
print(f"Only ML-based flagged: {len(only_ml)}")

print("Sample cosine similarities:")
for i in range(min(5,len(df))):
    for j in range(i+1, min(5,len(df))):
        print(f"Review {i} & {j}: {cosine_scores[i][j]:.3f}")
        
df.to_csv("reviews_flagged.csv", index=False)
print("✅ Saved preprocessed flagged dataset to reviews_flagged.csv")
