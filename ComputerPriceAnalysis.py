import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- 1. Data Loading and Cleaning ---
file_path = 'C:/Users/Mahathi/Downloads/computer_prices_all.csv'
try:
    df = pd.read_csv(file_path)
    # Ensure key columns are numeric/string (especially the target variable)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['ram_gb'] = pd.to_numeric(df['ram_gb'], errors='coerce')
    df['cpu_cores'] = pd.to_numeric(df['cpu_cores'], errors='coerce')

    # Drop rows where essential data is missing for plotting
    df = df.dropna(subset=['price', 'ram_gb', 'device_type', 'os', 'storage_type', 'brand'])

    print(f"Successfully loaded and cleaned {file_path}. Shape: {df.shape}")
except Exception as e:
    print(f"An error occurred during data loading or cleaning: {e}")
    # You would typically exit or handle the error here

# Set visual style for Seaborn/Matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- 2. Initial Exploration ---
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nSummary Statistics for Numeric Columns:")
print(df[['price', 'ram_gb', 'cpu_cores', 'weight_kg']].describe())
print("\nDevice Type Counts:")
print(df['device_type'].value_counts())

# ----------------------------------------------------------------------
# --- 3. Matplotlib Visualizations (Static) ---
# ----------------------------------------------------------------------

# Bar Chart - Device Type Counts
plt.figure(figsize=(8, 6))
df['device_type'].value_counts().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Bar Chart - Device Type Counts')
plt.ylabel('Number of Computers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line Chart - Price of First 50 Computers
plt.figure(figsize=(10, 5))
plt.plot(df['price'].iloc[:50].values, color='darkred')
plt.title('Line Chart - Computer Price (First 50 Entries)')
plt.xlabel('Entry Index')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Histogram - RAM GB Distribution
plt.figure(figsize=(8, 6))
plt.hist(df['ram_gb'], bins=20, color='teal', alpha=0.7)
plt.title('Histogram - RAM (GB) Distribution')
plt.xlabel('RAM (GB)')
plt.ylabel('Frequency')
plt.show()

# ----------------------------------------------------------------------
# --- 4. Seaborn Statistical Plots (Static) ---
# ----------------------------------------------------------------------

# Distribution Plot - Price
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], kde=True, color='purple', bins=50)
plt.title('Distribution Plot - Computer Price')
plt.xlabel('Price ($)')
plt.ylabel('Density / Count')
plt.show()

# Box Plot - Price by OS
# Filter for top 4 common OS for clarity
top_os = df['os'].value_counts().head(4).index
df_os_filtered = df[df['os'].isin(top_os)]

plt.figure(figsize=(10, 6))
sns.boxplot(x='os', y='price', data=df_os_filtered, palette='pastel')
plt.title('Box Plot - Price Distribution by Operating System')
plt.xlabel('Operating System')
plt.ylabel('Price ($)')
# Limit y-axis to focus on the bulk of the data, ignoring extreme outliers
plt.ylim(0, df_os_filtered['price'].quantile(0.95))
plt.show()

# Count Plot - Storage Types
plt.figure(figsize=(8, 6))
sns.countplot(x='storage_type', data=df, palette='viridis', order=df['storage_type'].value_counts().index)
plt.title('Count Plot - Storage Type Counts')
plt.xlabel('Storage Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# --- 5. Plotly Interactive Plots ---
# ----------------------------------------------------------------------

# Interactive Distribution (Price by Brand)
# Filter for top 5 brands for readability
top_brands = df['brand'].value_counts().head(5).index
df_brand_filtered = df[df['brand'].isin(top_brands)]

fig_hist = px.histogram(
    df_brand_filtered,
    x='price',
    color='brand',
    barmode='overlay',
    title='Interactive Distribution of Price by Brand',
    labels={'price': 'Price ($)', 'count': 'Frequency'},
    marginal='box'
)
fig_hist.show()

# Interactive Scatter Plot (RAM vs. Price, colored by Device Type)
fig_scatter = px.scatter(
    df.head(5000),  # Use a reasonable subset for better interactive performance
    x='ram_gb',
    y='price',
    color='device_type',
    size='cpu_cores',
    title='Interactive Scatter Plot: RAM vs. Price by Device Type',
    labels={'ram_gb': 'RAM (GB)', 'price': 'Price ($)'},
    hover_data=['brand', 'model']
)
fig_scatter.update_yaxes(type="log")  # Use log scale for price
fig_scatter.show()