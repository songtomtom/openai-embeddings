from ast import literal_eval

import numpy as np
import pandas as pd

datafile_path = "fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

print(df["embedding"])
