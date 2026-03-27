import numpy as np
import pandas as pd

# Number of samples (you can change this)
N = 1000

# Feature names
question_features = [
    "cold start problem",
    "high computational cost",
    "lack of annotated datasets",
    "language diversity challenges",
    "limited labeled data",
    "poor scalability of traditional models",
    "sample inefficiency",
    "scalability issues",
    "unstable training",
    "urban traffic congestion"
]

solution_features = [
    "cnn-based feature extraction",
    "data augmentation techniques",
    "distributed training framework",
    "graph neural network modeling",
    "matrix factorization",
    "multilingual embeddings",
    "policy gradient optimization",
    "simulation-based training",
    "spatio-temporal learning",
    "transfer learning approach"
]

# Generate random binary data (0 or 1)
questions_data = np.random.randint(0, 2, size=(N, len(question_features)))
solutions_data = np.random.randint(0, 2, size=(N, len(solution_features)))

# Convert to DataFrames
questions_df = pd.DataFrame(questions_data, columns=question_features)
solutions_df = pd.DataFrame(solutions_data, columns=solution_features)

# Save to CSV
questions_df.to_csv("questions.csv", index=False)
solutions_df.to_csv("solutions.csv", index=False)

print("Files generated: questions.csv and solutions.csv")