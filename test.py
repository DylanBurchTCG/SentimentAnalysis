import os
import pandas as pd

# Print current working directory
print("Current directory:", os.getcwd())

# Try to read the file
try:
    df = pd.read_csv("data/TestData.csv")
    print("Data loaded successfully!")
    print("Number of rows:", len(df))
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
except Exception as e:
    print("Error loading data:", str(e))