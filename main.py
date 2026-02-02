import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# Set a random seed for reproducibility so the numbers stay the same every time you run it
np.random.seed(42)

# ==========================================
# 1. DATA GENERATION & DATAFRAME CREATION
# ==========================================

# Define the number of rows (samples)
n_samples = 1000

# Create the dictionary of data
data = {
    'order_id': np.arange(1001, 1001 + n_samples), # Unique ID starting from 1001
    'warehouse': np.random.choice(['North', 'South', 'East', 'West'], n_samples), # Nominal
    'product_category': np.random.choice(['Electronics', 'Fashion', 'Grocery'], n_samples), # Nominal
    'payment_type': np.random.choice(['COD', 'Prepaid'], n_samples), # Nominal
    'delivery_days': np.random.randint(1, 15, n_samples), # Ratio (Discrete integer days)
    'order_value': np.random.uniform(500, 50000, n_samples).round(2), # Ratio (Continuous INR)
    'discount_percent': np.random.randint(0, 30, n_samples), # Ratio (0-30%)
    'rating': np.random.randint(1, 6, n_samples), # Ordinal (1-5 stars)
    'returned': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]) # Nominal (weighted 15% returns)
}

# Create the Pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows to check structure
print("--- DataFrame Head ---")
print(df.head())

# ==========================================
# 2. DATA PREPARATION & CLASSIFICATION
# ==========================================

# Task: Convert return column into binary (Yes=1, No=0)
# Using .map to replace string values with integers
df['returned_binary'] = df['returned'].map({'Yes': 1, 'No': 0})

# Task: Classify data collection methods
# Since we generated this data randomly based on rules, this is a 'Simulation'.
# In a real-world scenario for this specific dataset:
# - order_id, value, payment: 'Observational' (recorded from system logs)
# - rating: 'Survey' (user feedback)
print("\n--- Data Collection Method ---")
print("In this script: Simulation/Synthetic Generation.")
print("In real scenario: Observational (Transactional logs) & Survey (Ratings).")

# ==========================================
# 3. VISUALIZATIONS
# ==========================================

# Set the style for seaborn plots
sns.set_style("whitegrid")

# Create a large figure to hold subplots (we will do some individually for clarity)
plt.figure(figsize=(15, 10))

# A. Frequency distributions of product category
# Calculating counts for each category
freq_dist = df['product_category'].value_counts()
print("\n--- Frequency Distribution (Product Category) ---")
print(freq_dist)

# B. Bar chart of orders / warehouse
plt.subplot(2, 3, 1) # Create subplot 1
# Countplot automatically counts occurrences per category
sns.countplot(x='warehouse', data=df, palette='viridis')
plt.title('Orders by Warehouse')

# C. Histogram of delivery days
plt.subplot(2, 3, 2)
# Histplot shows the frequency distribution of a continuous/discrete variable
sns.histplot(df['delivery_days'], bins=14, kde=False, color='skyblue')
plt.title('Histogram of Delivery Days')

# D. Pie chart of payment type
plt.subplot(2, 3, 3)
payment_counts = df['payment_type'].value_counts()
# autopct adds the percentage label to the slices
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Payment Type Distribution')

# E. Line graph avg delivery days vs discount level
plt.subplot(2, 3, 4)
# Group by discount and calculate mean delivery days
avg_delivery = df.groupby('discount_percent')['delivery_days'].mean().reset_index()
# Plotting the line
sns.lineplot(x='discount_percent', y='delivery_days', data=avg_delivery, marker='o')
plt.title('Avg Delivery Days vs Discount %')

# F. Boxplot -> Order value by product category
plt.subplot(2, 3, 5)
# Boxplot shows median, quartiles, and outliers
sns.boxplot(x='product_category', y='order_value', data=df, palette='Set2')
plt.title('Order Value by Category')

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

# --- Specialized Charts (Polygon & Ogive) ---

plt.figure(figsize=(12, 5))

# G. Frequency Polygon -> Delivery days
plt.subplot(1, 2, 1)
# Get histogram data (counts and bin edges)
counts, bin_edges = np.histogram(df['delivery_days'], bins=14)
# Calculate bin centers for the polygon points
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
# Plot line connecting bin centers
plt.plot(bin_centers, counts, marker='o', linestyle='-', color='purple')
plt.title('Frequency Polygon: Delivery Days')
plt.xlabel('Days')
plt.ylabel('Frequency')

# H. Ogive -> Cumulative delivery days
plt.subplot(1, 2, 2)
# Sort data for cumulative calculation
sorted_data = np.sort(df['delivery_days'])
# Calculate cumulative probability (y-axis) ranging from 0 to 1
yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
plt.plot(sorted_data, yvals, marker='.', linestyle='none', color='green')
plt.title('Ogive (CDF): Delivery Days')
plt.xlabel('Days')
plt.ylabel('Cumulative Probability')

plt.show()

# ==========================================
# 4. MEASURES OF CENTRAL TENDENCY
# ==========================================

print("\n--- Measures of Central Tendency ---")

# Calculate Mean, Median, Mode of delivery days
mean_del = df['delivery_days'].mean()
median_del = df['delivery_days'].median()
# Mode returns a Series (can be multimodal), so we take [0]
mode_del = df['delivery_days'].mode()[0]
print(f"Delivery Days -> Mean: {mean_del:.2f}, Median: {median_del}, Mode: {mode_del}")

# Compute Weighted mean of order value (weights = discount_percent)
# Note: If total weight is 0, this creates an error, but discount > 0 usually.
# Using numpy.average to handle weights
weighted_mean_val = np.average(df['order_value'], weights=df['discount_percent'])
print(f"Weighted Mean of Order Value (weighted by discount): {weighted_mean_val:.2f}")

# Find Combined mean delivery time for North + South warehouses
# Filter DataFrame for North and South only
ns_df = df[df['warehouse'].isin(['North', 'South'])]
combined_mean_ns = ns_df['delivery_days'].mean()
print(f"Combined Mean Delivery Time (North + South): {combined_mean_ns:.2f}")

# ==========================================
# 5. MEASURES OF DISPERSION
# ==========================================

print("\n--- Measures of Dispersion ---")

# Range, Variance, Standard Deviation
data_range = df['delivery_days'].max() - df['delivery_days'].min()
variance = df['delivery_days'].var() # Sample variance by default (ddof=1)
std_dev = df['delivery_days'].std()  # Sample std dev by default
print(f"Delivery Days -> Range: {data_range}, Variance: {variance:.2f}, Std Dev: {std_dev:.2f}")

# Interquartile Range (IQR)
Q1 = df['delivery_days'].quantile(0.25)
Q3 = df['delivery_days'].quantile(0.75)
IQR = Q3 - Q1
print(f"Delivery Days -> IQR: {IQR}")

# Calculate Coefficient of Variation (CV) for delivery days
# CV = (Std Dev / Mean) * 100
cv_delivery = (std_dev / mean_del) * 100
print(f"Coefficient of Variation (Delivery Days): {cv_delivery:.2f}%")

# Identify Which warehouse has the most consistent delivery time
# Consistency is measured by the lowest Standard Deviation
std_by_warehouse = df.groupby('warehouse')['delivery_days'].std()
most_consistent = std_by_warehouse.idxmin() # Get index (name) of min value
print(f"Most Consistent Warehouse (Lowest Std Dev): {most_consistent} (Std: {std_by_warehouse.min():.2f})")

# ==========================================
# 6. DISTRIBUTION SHAPE
# ==========================================

print("\n--- Distribution Shape ---")

# Compute skewness & kurtosis of order value
skewness = df['order_value'].skew()
kurt = df['order_value'].kurt()

print(f"Order Value -> Skewness: {skewness:.2f}")
print(f"Order Value -> Kurtosis: {kurt:.2f}")

# Identify distribution shape based on skewness
if -0.5 < skewness < 0.5:
    shape = "Approximately Symmetric"
elif skewness <= -0.5:
    shape = "Left Skewed (Negatively Skewed)"
else:
    shape = "Right Skewed (Positively Skewed)"
print(f"Distribution Shape: {shape}")

# Visualize with histogram and KDE curve
plt.figure(figsize=(8, 5))
# kde=True adds the Kernel Density Estimate curve
sns.histplot(df['order_value'], kde=True, color='orange')
plt.title(f'Distribution of Order Value (Skew: {skewness:.2f})')
plt.show()

# ==========================================
# 7. CORRELATION & REGRESSION
# ==========================================

print("\n--- Correlation & Regression ---")

# Pearson correlation: Discount percent vs order value
pearson_corr = df['discount_percent'].corr(df['order_value'], method='pearson')
print(f"Pearson Correlation (Discount vs Value): {pearson_corr:.4f}")

# Spearman correlation: Delivery days vs customer rating
spearman_corr = df['delivery_days'].corr(df['rating'], method='spearman')
print(f"Spearman Correlation (Delivery vs Rating): {spearman_corr:.4f}")

# Scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='discount_percent', y='order_value', data=df, alpha=0.5)
plt.title(f'Discount vs Value (r={pearson_corr:.2f})')

plt.subplot(1, 2, 2)
# Jitter added to rating because it's discrete (1-5), makes visual clearer
sns.stripplot(x='rating', y='delivery_days', data=df, jitter=True, alpha=0.5)
plt.title(f'Delivery vs Rating (rho={spearman_corr:.2f})')
plt.show()

# Build simple linear regression: X = discount_percent, Y = order_value
X = df[['discount_percent']] # Feature (Must be 2D array for sklearn)
Y = df['order_value']        # Target

reg = LinearRegression()
reg.fit(X, Y)

slope = reg.coef_[0]
intercept = reg.intercept_

print("Simple Linear Regression (Discount -> Value):")
print(f"Equation: Order_Value = {intercept:.2f} + ({slope:.2f} * Discount%)")
print(f"Interpretation: For every 1% increase in discount, Order Value changes by ₹{slope:.2f}")

# Predict order value at 20% discount
pred_20 = reg.predict([[20]])[0]
print(f"Predicted Order Value at 20% Discount: ₹{pred_20:.2f}")

# ==========================================
# 8. PROBABILITY
# ==========================================

print("\n--- Probability ---")

# Find P(order is returned)
# Probability = Favorable Outcomes / Total Outcomes
p_returned = df['returned_binary'].mean() # Since 1=Yes, 0=No, mean is the probability
print(f"P(Order is Returned): {p_returned:.4f}")

# P(order is prepaid AND not returned)
# Count rows where payment='Prepaid' AND returned='No'
favorable = len(df[(df['payment_type'] == 'Prepaid') & (df['returned'] == 'No')])
p_prepaid_not_returned = favorable / len(df)
print(f"P(Prepaid AND Not Returned): {p_prepaid_not_returned:.4f}")

# Identify Mutually exclusive and independent events (Conceptual)
print("Mutual Exclusivity: 'Returned=Yes' and 'Returned=No' are mutually exclusive.")
print("Independence: 'Rating' and 'Warehouse' are likely independent in this synthetic data.")

# Binomial: Probability exactly 3 out of 8 orders are returned
# n=8, k=3, p=p_returned
prob_binom = stats.binom.pmf(k=3, n=8, p=p_returned)
print(f"Binomial P(3 returns in 8 orders | p={p_returned:.2f}): {prob_binom:.4f}")

# Poisson: Avg daily returns = 4. Probability of exactly 6 returns in a day
# mu (lambda) = 4, k = 6
prob_poisson = stats.poisson.pmf(k=6, mu=4)
print(f"Poisson P(6 returns | avg=4): {prob_poisson:.4f}")

# Normal: Delivery days ~ Normal(μ, σ). Find probability delivery > 7 days
# Note: We use the sample mean and std from our data as population parameters for this exercise
z_score = (7 - mean_del) / std_dev
# 1 - cdf (Cumulative Distribution Function) gives probability of being greater than Z
prob_normal = 1 - stats.norm.cdf(z_score)
print(f"Normal P(Delivery > 7 days | mean={mean_del:.1f}, std={std_dev:.1f}): {prob_normal:.4f}")

# ==========================================
# 9. SAMPLING & CENTRAL LIMIT THEOREM (CLT)
# ==========================================

print("\n--- Sampling & CLT ---")

# Draw Simple random sample of 100 orders
simple_sample = df.sample(n=100, random_state=1)
print(f"Simple Random Sample Mean (Order Value): {simple_sample['order_value'].mean():.2f}")

# Stratified sample by warehouse
# Taking proportionate sample from each warehouse group
stratified_sample = df.groupby('warehouse', group_keys=False).apply(lambda x: x.sample(frac=0.1))
print(f"Stratified Sample Size: {len(stratified_sample)}")

# Demonstrate Central Limit Theorem using order value
# CLT states: The sampling distribution of the sample mean approximates a normal distribution
sample_means = []
# Take 1000 samples, each of size 50, and calculate the mean of each
for _ in range(1000):
    sample = df['order_value'].sample(n=50)
    sample_means.append(sample.mean())

plt.figure(figsize=(8, 5))
sns.histplot(sample_means, kde=True, color='teal')
plt.title('CLT Demonstration: Distribution of Sample Means (Order Value)')
plt.xlabel('Sample Mean Order Value')
plt.show()

# Calculate Standard error of mean delivery days
# SE = Std_Dev / sqrt(n)
std_error = std_dev / np.sqrt(len(df))
print(f"Standard Error of Mean (Delivery Days): {std_error:.4f}")