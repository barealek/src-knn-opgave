import pandas as pd
import random

from sklearn.model_selection import train_test_split


random.seed(42)

# df = pd.read_csv('HoejdeSko.csv', sep=";")
# df = pd.read_csv('wine_quality.csv')

df = pd.read_csv('Cancer_Data.csv')
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

df = df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "diagnosis"]]
print(df.head())

# df = df[["Age", "BMI", "Waist_Circumference", "Fasting_Blood_Glucose", "Cholesterol_HDL", "Physical_Activity_Level"]]

# df["quality"] = df["quality"].map({"good": 2, "mid": 1, "bad": 0})
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})


# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

train, validation = train_test_split(train, test_size=0.2, random_state=42)
# print(train.head())
# exit()
# train, test = df, df

train = train.to_numpy()
test = test.to_numpy()
