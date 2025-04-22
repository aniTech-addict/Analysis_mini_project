# Advanced Data Analysis in Python
# Using pandas, NumPy, and Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')  # Using a more aesthetically pleasing style

# 1. Creating and manipulating data with NumPy
# --------------------------------------------
print("PART 1: NUMPY OPERATIONS")
print("-----------------------")

# Generate sample data: monthly sales figures for 2 years
np.random.seed(42)  # For reproducibility
monthly_sales = np.random.normal(loc=50000, scale=10000, size=24)
monthly_sales = np.round(monthly_sales, 2)  # Round to 2 decimal places

# Reshape data into a 2-year format (12 months per row)
sales_by_year = monthly_sales.reshape(2, 12)
print(f"Monthly sales data by year:\n{sales_by_year}")

# Basic NumPy operations
print(f"\nAverage monthly sales: ${np.mean(monthly_sales):.2f}")
print(f"Highest monthly sales: ${np.max(monthly_sales):.2f}")
print(f"Lowest monthly sales: ${np.min(monthly_sales):.2f}")
print(f"Standard deviation: ${np.std(monthly_sales):.2f}")

# Year-over-year comparison
print(f"\nYear 1 average monthly sales: ${np.mean(sales_by_year[0]):.2f}")
print(f"Year 2 average monthly sales: ${np.mean(sales_by_year[1]):.2f}")
year_growth = ((np.sum(sales_by_year[1]) - np.sum(sales_by_year[0])) / np.sum(sales_by_year[0])) * 100
print(f"Year-over-year growth: {year_growth:.2f}%")

# 2. Data manipulation with pandas
# -------------------------------
print("\n\nPART 2: PANDAS OPERATIONS")
print("-----------------------")

# Create a pandas DataFrame from our NumPy array
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create a pandas DataFrame
sales_df = pd.DataFrame({
    'Month': months * 2,
    'Year': ['Year 1'] * 12 + ['Year 2'] * 12,
    'Sales': monthly_sales,
})

# Add a quarter column based on month
month_to_quarter = {
    'Jan': 'Q1', 'Feb': 'Q1', 'Mar': 'Q1',
    'Apr': 'Q2', 'May': 'Q2', 'Jun': 'Q2',
    'Jul': 'Q3', 'Aug': 'Q3', 'Sep': 'Q3',
    'Oct': 'Q4', 'Nov': 'Q4', 'Dec': 'Q4'
}
sales_df['Quarter'] = sales_df['Month'].map(month_to_quarter)

# Display the DataFrame
print("Sales DataFrame:")
print(sales_df.head(6))
print("...")

# Calculate quarterly sales
quarterly_sales = sales_df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index()
print("\nQuarterly Sales Summary:")
print(quarterly_sales)

# Calculate monthly growth rate
sales_df['Monthly_Growth'] = sales_df.groupby('Year')['Sales'].pct_change() * 100
print("\nSales with Monthly Growth Rate:")
print(sales_df[['Year', 'Month', 'Sales', 'Monthly_Growth']].head(6))
print("...")

# Summary statistics
print("\nSummary Statistics by Year:")
yearly_summary = sales_df.groupby('Year')['Sales'].describe()
print(yearly_summary)

# 3. Data visualization with Matplotlib
# ------------------------------------
print("\n\nPART 3: VISUALIZATION WITH MATPLOTLIB")
print("-----------------------------------")
print("(Note: Plots would be displayed in a graphical environment)")

# Plot 1: Monthly Sales Trend
plt.figure(figsize=(12, 6))
for year in ['Year 1', 'Year 2']:
    year_data = sales_df[sales_df['Year'] == year]
    plt.plot(year_data['Month'], year_data['Sales'], marker='o', label=year)

plt.title('Monthly Sales Comparison: Year 1 vs Year 2')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
# plt.show()  # In a real environment, this would display the plot

# Plot 2: Quarterly Sales Comparison
plt.figure(figsize=(10, 6))
quarterly_pivot = quarterly_sales.pivot(index='Quarter', columns='Year', values='Sales')
quarterly_pivot.plot(kind='bar', figsize=(10, 6))
plt.title('Quarterly Sales Comparison')
plt.xlabel('Quarter')
plt.ylabel('Total Sales ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()  # In a real environment, this would display the plot

# Plot 3: Sales Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(sales_df[sales_df['Year'] == 'Year 1']['Sales'], bins=6, alpha=0.7, label='Year 1')
plt.hist(sales_df[sales_df['Year'] == 'Year 2']['Sales'], bins=6, alpha=0.7, label='Year 2')
plt.title('Sales Distribution')
plt.xlabel('Sales Amount ($)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot([
    sales_df[sales_df['Year'] == 'Year 1']['Sales'],
    sales_df[sales_df['Year'] == 'Year 2']['Sales']
], labels=['Year 1', 'Year 2'])
plt.title('Sales Distribution Boxplot')
plt.ylabel('Sales Amount ($)')
plt.tight_layout()
# plt.show()  # In a real environment, this would display the plot

# Plot 4: Heatmap of Monthly Sales
year1_data = sales_df[sales_df['Year'] == 'Year 1'].set_index('Month')['Sales'].reindex(months)
year2_data = sales_df[sales_df['Year'] == 'Year 2'].set_index('Month')['Sales'].reindex(months)
monthly_data = np.array([year1_data.values, year2_data.values])

plt.figure(figsize=(12, 8))
plt.imshow(monthly_data, cmap='YlGnBu')
plt.colorbar(label='Sales ($)')
plt.title('Monthly Sales Heatmap')
plt.yticks([0, 1], ['Year 1', 'Year 2'])
plt.xticks(range(12), months)
plt.tight_layout()
# plt.show()  # In a real environment, this would display the plot

# 4. Advanced analysis: 3-month rolling average and forecasting
# -----------------------------------------------------------
print("\n\nPART 4: ADVANCED ANALYSIS")
print("------------------------")

# Calculate 3-month rolling average
sales_df['3_Month_Rolling_Avg'] = sales_df.groupby('Year')['Sales'].rolling(window=3).mean().reset_index(0, drop=True)
print("Sales with 3-Month Rolling Average:")
print(sales_df[['Year', 'Month', 'Sales', '3_Month_Rolling_Avg']].tail(6))

# Simple forecasting model using numpy polyfit
print("\nSales Forecasting for Next 3 Months:")
x = np.arange(len(monthly_sales))
y = monthly_sales
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Predict the next 3 months
next_months = np.arange(len(monthly_sales), len(monthly_sales) + 3)
predictions = p(next_months)
print(f"Predicted sales for month 25: ${predictions[0]:.2f}")
print(f"Predicted sales for month 26: ${predictions[1]:.2f}")
print(f"Predicted sales for month 27: ${predictions[2]:.2f}")

# Calculate the correlation between sales in Year 1 and Year 2
correlation = np.corrcoef(sales_by_year[0], sales_by_year[1])[0, 1]
print(f"\nCorrelation between Year 1 and Year 2 sales: {correlation:.4f}")

# Print conclusion about the data analysis
print("\n\nCONCLUSION")
print("----------")
print("This data analysis project demonstrates how to use NumPy for numerical operations,")
print("pandas for data manipulation and analysis, and Matplotlib for visualization.")
print("The analysis reveals patterns in sales data across two years, with quarterly aggregations,")
print("growth calculations, and basic forecasting using trend analysis.")