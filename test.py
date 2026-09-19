import pandas as pd

pred = pd.read_csv("submission.csv")
winners = pd.read_csv("dev_winners.csv")

# Actual winner IDs
winner_ids = set(winners["candidate_id"])

# Your top 150 predictions
top150 = set(pred.head(150)["candidate_id"])

# Overlap
overlap = len(top150 & winner_ids)

precision = overlap / 150
recall = overlap / len(winner_ids)

print("Overlap:", overlap)
print("Precision@150:", round(precision, 4))
print("Recall@150:", round(recall, 4))