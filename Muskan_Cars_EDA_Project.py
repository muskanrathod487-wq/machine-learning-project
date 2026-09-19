# ============================================================
# CARS DATASET - EXPLORATORY DATA ANALYSIS (EDA)
# Student Project
# ============================================================
# Objective:
# To clean, explore, visualize, and identify meaningful patterns
# in an automobile dataset.
#
# Dataset Source:
# Student-created sample automobile dataset for educational
# EDA demonstration.
# ============================================================

# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")


# -------------------------
# 2. Create / Load Dataset
# -------------------------
data = {
    "Brand": [
        "Toyota", "Honda", "BMW", "Audi", "Maruti",
        "Hyundai", "Toyota", "Honda", "BMW", "Maruti",
        "Hyundai", "Audi", "Toyota", "Honda", "Maruti"
    ],
    "Year": [
        2020, 2019, 2021, 2018, 2022,
        2020, 2017, 2021, 2019, 2023,
        2018, 2020, 2022, 2017, 2021
    ],
    "Price": [
        850000, 750000, 3500000, 2800000, 650000,
        900000, 600000, 950000, 3200000, 700000,
        550000, 2500000, 1100000, 650000, 720000
    ],
    "Fuel_Type": [
        "Petrol", "Diesel", "Petrol", "Diesel", "Petrol",
        "Petrol", "Diesel", "Petrol", "Petrol", "Petrol",
        "Diesel", "Petrol", "Hybrid", "Petrol", "Diesel"
    ],
    "Mileage": [
        18, 22, 15, 17, 20,
        19, 21, 18, 14, 21,
        23, 16, 24, 19, 22
    ],
    "Engine": [
        1200, 1500, 2000, 1800, 1000,
        1200, 1500, 1400, 2000, 1000,
        1500, 1800, 1500, 1200, 1500
    ],
    "Transmission": [
        "Manual", "Manual", "Automatic", "Automatic", "Manual",
        "Manual", "Manual", "Automatic", "Automatic", "Manual",
        "Manual", "Automatic", "Automatic", "Manual", "Manual"
    ]
}

df = pd.DataFrame(data)

print("Dataset created successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

display(df.head())


# -------------------------
# 3. Dataset Understanding
# -------------------------
print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nFIRST 5 ROWS")
display(df.head())

print("\nNUMERICAL COLUMNS")
print(df.select_dtypes(include=np.number).columns.tolist())

print("\nCATEGORICAL COLUMNS")
print(df.select_dtypes(exclude=np.number).columns.tolist())


# -------------------------
# 4. Data Quality Checks
# -------------------------
print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nDATASET SIZE BEFORE CLEANING")
print(df.shape)


# -------------------------
# 5. Data Cleaning
# -------------------------
# Remove duplicate rows if any are present.
df = df.drop_duplicates()

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

print("\nDATASET SIZE AFTER CLEANING")
print(df.shape)

print("\nCLEANING COMPLETED SUCCESSFULLY!")


# -------------------------
# 6. Descriptive Statistics
# -------------------------
print("\nNUMERICAL SUMMARY")
display(df.describe())

print("\nCATEGORICAL SUMMARY")
display(df.describe(include="object"))


# ============================================================
# 7. VISUAL EXPLORATORY DATA ANALYSIS
# ============================================================

# Chart 1: Histogram - Price Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Price"], bins=8, edgecolor="black")
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Number of Cars")
plt.show()

print("Observation: Most cars are concentrated in the lower-to-middle "
      "price range, while a few cars have comparatively higher prices.")


# Chart 2: Bar Chart - Fuel Type
plt.figure(figsize=(7, 5))
df["Fuel_Type"].value_counts().plot(kind="bar", edgecolor="black")
plt.title("Number of Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.xticks(rotation=0)
plt.show()

print("Observation: The bar chart shows the number of cars available "
      "for each fuel type and identifies the most common fuel type.")


# Chart 3: Box Plot - Price by Transmission
plt.figure(figsize=(7, 5))
sns.boxplot(x="Transmission", y="Price", data=df)
plt.title("Car Price by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Price")
plt.show()

print("Observation: The box plot compares the price distribution "
      "between manual and automatic cars.")


# Chart 4: Scatter Plot - Year vs Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Year"], df["Price"])
plt.title("Relationship Between Car Year and Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()

print("Observation: The scatter plot shows the relationship between "
      "manufacturing year and car price.")


# Chart 5: Scatter Plot - Mileage vs Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Mileage"], df["Price"])
plt.title("Relationship Between Mileage and Car Price")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.show()

print("Observation: The scatter plot shows the relationship between "
      "mileage and car price.")


# Chart 6: Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

print("Observation: The heatmap shows correlations between numerical "
      "features. Values closer to 1 indicate a strong positive "
      "relationship, while values closer to -1 indicate a strong "
      "negative relationship.")


# -------------------------
# 8. Final Findings
# -------------------------
print("\nFINAL FINDINGS")
print("-" * 50)

print(
    "1. The dataset contains",
    len(df),
    "cars from",
    df["Brand"].nunique(),
    "different brands."
)

print(
    "2. The most common fuel type is:",
    df["Fuel_Type"].value_counts().idxmax()
)

print(
    "3. The average car price is:",
    round(df["Price"].mean(), 2)
)

print(
    "4. The highest-priced car costs:",
    df["Price"].max()
)

print(
    "5. The most common transmission type is:",
    df["Transmission"].value_counts().idxmax()
)


# -------------------------
# 9. Project Summary
# -------------------------
print("\nPROJECT SUMMARY")
print("=" * 40)

print("Total Cars:", len(df))
print("Total Brands:", df["Brand"].nunique())
print("Most Common Fuel Type:", df["Fuel_Type"].value_counts().idxmax())
print("Average Price: ₹", round(df["Price"].mean(), 2))
print("Highest Price: ₹", df["Price"].max())
print(
    "Most Common Transmission:",
    df["Transmission"].value_counts().idxmax()
)

print("\nEDA PROJECT COMPLETED SUCCESSFULLY!")
