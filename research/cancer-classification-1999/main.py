import numpy as np
import pandas as pd
from predictor import Predictor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('leukemia_small.csv')
df_columns = list(df.columns)

samples = df.to_numpy().T
scaler = StandardScaler()
samples = scaler.fit_transform(samples)
class_labels = np.array(["ALL" if column.startswith("ALL") else "AML" for column in df_columns])

predictor = Predictor(samples, class_labels)
best_genes = predictor.neighborhood_analysis()
new_samples = []
for gene_idx in best_genes:
    gene_vec = samples[..., gene_idx]
    new_samples.append(gene_vec)
new_samples = np.array(new_samples).T
predictor.fit()
print(predictor.predict(new_samples[:10]))