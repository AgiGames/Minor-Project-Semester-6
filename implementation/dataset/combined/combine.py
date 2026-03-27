import pandas as pd
import numpy as np

rand_percent = 100

qs = pd.read_csv('questions.csv')
sols = pd.read_csv('solutions.csv')

rand_count = int(rand_percent * len(qs) / 100)
unrand_count = len(qs) - rand_count

def generate_new_pair(qs, sols, rand_count, unrand_count):
    # 1. Sample indices ONCE
    sampled_idx = np.random.choice(len(qs), size=unrand_count, replace=False)

    qs_sampled = qs.iloc[sampled_idx].reset_index(drop=True)
    sols_sampled = sols.iloc[sampled_idx].reset_index(drop=True)

    # 2. Generate random binary vectors
    qs_rand = np.random.randint(0, 2, size=(rand_count, qs.shape[1]))
    sols_rand = np.random.randint(0, 2, size=(rand_count, sols.shape[1]))

    qs_rand_df = pd.DataFrame(qs_rand, columns=qs.columns)
    sols_rand_df = pd.DataFrame(sols_rand, columns=sols.columns)

    # 3. Combine
    new_qs = pd.concat([qs_sampled, qs_rand_df], ignore_index=True)
    new_sols = pd.concat([sols_sampled, sols_rand_df], ignore_index=True)

    # 4. Shuffle BOTH with same permutation
    perm = np.random.permutation(len(new_qs))
    new_qs = new_qs.iloc[perm].reset_index(drop=True)
    new_sols = new_sols.iloc[perm].reset_index(drop=True)

    return new_qs, new_sols


new_qs, new_sols = generate_new_pair(qs, sols, rand_count, unrand_count)

new_qs.to_csv('new_questions.csv', index=False)
new_sols.to_csv('new_solutions.csv', index=False)