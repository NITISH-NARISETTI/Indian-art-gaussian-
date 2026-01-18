print("TRAIN SCRIPT STARTED")

import numpy as np
import pickle
from scipy.stats import multivariate_normal
from extract_features import load

# Load features
X, y = load("data")
print("Loaded data:", X.shape, y.shape)

models = {}

# Train one Gaussian per class
for label in set(y):
    class_data = X[y == label]
    mu = np.mean(class_data, axis=0)
    cov = np.cov(class_data, rowvar=False) + 1e-6 * np.eye(6)
    models[label] = multivariate_normal(mu, cov)
    print(f"Trained Gaussian for {label} with {len(class_data)} samples")

# Save model
with open("models/gaussians.pkl", "wb") as f:
    pickle.dump(models, f)

print("Model trained and saved to models/gaussians.pkl")
