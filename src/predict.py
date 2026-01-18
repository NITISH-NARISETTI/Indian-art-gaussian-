print("PREDICT SCRIPT STARTED")

import pickle
from extract_features import extract

with open("models/gaussians.pkl", "rb") as f:
    models = pickle.load(f)

img_path = input("Enter image path: ").strip()
print("Image received:", img_path)

x = extract(img_path)

scores = {}
for label, model in models.items():
    scores[label] = model.logpdf(x)
    print(label, "log-likelihood:", round(scores[label], 2))

print("\nPredicted class:", max(scores, key=scores.get))
