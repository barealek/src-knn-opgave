import pandas as pd
import random

from sklearn.model_selection import train_test_split


random.seed(42)

df = pd.read_csv('HoejdeSko.csv', sep=";")
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

# df = df[["Age", "BMI", "Waist_Circumference", "Fasting_Blood_Glucose", "Cholesterol_HDL", "Physical_Activity_Level"]]

# df = df[["fixed_acidity","residual_sugar","alcohol","density","quality_label"]]
# df["quality_label"] = df["quality_label"].map({"high": 2, "medium": 1, "low": 0})


# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
# train, test = df, df

train = train.to_numpy()
test = test.to_numpy()
