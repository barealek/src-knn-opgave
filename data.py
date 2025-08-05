import pandas as pd
import random

from sklearn.model_selection import train_test_split

random.seed(42)

df = pd.read_csv('Cancer_Data.csv')
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

df = df[["radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "diagnosis"]]
print(df.head())
print("Data shape:", df.shape)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
exit()

train, test = train_test_split(df, test_size=0.2, random_state=42)
train, validation = train_test_split(train, test_size=0.2, random_state=42)

train = train.to_numpy()
test = test.to_numpy()
