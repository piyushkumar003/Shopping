# ---
# AI Project: Product Data Analytics Script
#
# This is the .py version of the Data_Analytics.ipynb notebook.
# It loads the data, cleans it, and generates visualizations.
# ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings

# Suppress the specific SyntaxWarning from the 'price' cleaning
warnings.filterwarnings("ignore", category=SyntaxWarning, message=r".*invalid escape sequence.*")

print("--- Starting Data Analysis Script ---")

# --- 1. Define Helper Functions ---

def get_primary_category(categories_str):
    """Safely parses the 'categories' string and returns the first category."""
    try:
        categories_list = ast.literal_eval(categories_str)
        if isinstance(categories_list, list) and len(categories_list) > 0:
            return categories_list[0]
    except (ValueError, SyntaxError, TypeError):
        pass
    return 'Unknown'

def load_and_clean_data(file_path):
    """Loads and prepares the dataset for analysis."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Make sure it's in the same folder.")
        return None

    # Clean Price Column
    df_cleaned = df.copy()
    # Use r'' for raw string to avoid SyntaxWarning
    df_cleaned['price'] = df_cleaned['price'].replace({r'\$': '', ',': ''}, regex=True)
    df_cleaned['price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')

    # Clean Categories Column
    df_cleaned['primary_category'] = df_cleaned['categories'].apply(get_primary_category)
    
    print("Data cleaning complete.")
    return df_cleaned

# --- 2. Main Analysis and Visualization ---

def run_analysis():
    file_path = 'intern_data_ikarus.csv'
    df_cleaned = load_and_clean_data(file_path)

    if df_cleaned is None:
        print("Analysis failed: Could not load data.")
        return

    # --- Price Analysis ---
    print("\n--- Price Column Statistics ---")
    print(df_cleaned['price'].describe())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_cleaned['price'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Product Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    print("\nShowing Price Distribution plot. Close the plot window to continue...")
    plt.show()

    
    # --- Categories Analysis ---
    print("\n--- Top 10 Primary Categories ---")
    top_10_categories = df_cleaned['primary_category'].value_counts().head(10)
    print(top_10_categories)
    
    plt.figure(figsize=(12, 8))
    top_10_categories.plot(kind='bar')
    plt.title('Top 10 Primary Product Categories')
    plt.xlabel('Primary Category')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    print("\nShowing Top Categories plot. Close the plot window to continue...")
    plt.show()

    
    # --- Brand Analysis ---
    print("\n--- Top 10 Product Brands ---")
    top_10_brands = df_cleaned['brand'].value_counts().head(10)
    print(top_10_brands)

    plt.figure(figsize=(12, 7))
    top_10_brands.plot(kind='bar')
    plt.title('Top 10 Product Brands')
    plt.xlabel('Brand')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    print("\nShowing Top Brands plot. Close the plot window to continue...")
    plt.show()

    
    # --- Material Analysis ---
    print("\n--- Top 10 Product Materials ---")
    top_10_materials = df_cleaned['material'].fillna('Unknown').value_counts().head(10)
    print(top_10_materials)

    plt.figure(figsize=(12, 7))
    top_10_materials.plot(kind='bar')
    plt.title('Top 10 Product Materials')
    plt.xlabel('Material')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    print("\nShowing Top Materials plot. Close the plot window to finish.")
    plt.show()
    
    print("\n--- Analysis Script Complete ---")

# --- 3. Run the script ---
if __name__ == "__main__":
    run_analysis()