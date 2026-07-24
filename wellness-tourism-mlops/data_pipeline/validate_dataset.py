
import os
import pandas as pd


# --------------------------------------------
# Dataset location
# --------------------------------------------

dataset_path = "wellness-tourism-mlops/data/tourism.csv"


# --------------------------------------------
# Expected columns from data dictionary
# --------------------------------------------

expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]


# --------------------------------------------
# Check dataset exists
# --------------------------------------------

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Dataset not found at location: {dataset_path}"
    )


print("Dataset file found successfully.")


# --------------------------------------------
# Load dataset
# --------------------------------------------

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully.")


# --------------------------------------------
# Validate columns
# --------------------------------------------

actual_columns = list(df.columns)

missing_columns = set(expected_columns) - set(actual_columns)

extra_columns = set(actual_columns) - set(expected_columns)


if missing_columns:
    raise ValueError(
        f"Dataset validation failed. Missing columns: {missing_columns}"
    )


print("\nDataset validation successful.")


# --------------------------------------------
# Dataset summary
# --------------------------------------------

print("\nDataset Summary")
print("----------------")
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")


print("\nColumn Names:")
print(df.columns.tolist())


print("\nMissing Values:")
print(df.isnull().sum())


if extra_columns:
    print("\nExtra columns found:")
    print(extra_columns)
else:
    print("\nNo extra columns found.")

print("\nValidation completed successfully.")
