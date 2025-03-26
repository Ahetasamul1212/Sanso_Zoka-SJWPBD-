import pandas as pd
import numpy as np

# Load the original dataset
data = pd.read_csv("E:\SWJP\Model\Cleaned_Rescaled_Data_Updated.csv")

# Define the number of rows to generate
num_new_rows = 1000

# Identify numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Generate new rows
synthetic_data = pd.DataFrame()

# Monte Carlo simulation for numerical columns
for col in numerical_cols:
    mean = data[col].mean()
    std = data[col].std()
    synthetic_data[col] = np.random.normal(loc=mean, scale=std, size=num_new_rows)

# Monte Carlo simulation for categorical columns
for col in categorical_cols:
    synthetic_data[col] = np.random.choice(data[col].dropna().unique(), size=num_new_rows, replace=True)

# Combine the original and synthetic data
augmented_data = pd.concat([data, synthetic_data], ignore_index=True)

# Save the augmented dataset
augmented_data.to_csv("Augmented_Data.csv", index=False)
