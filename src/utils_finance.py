# Importing necessary librarys
import pandas as pd
import numpy as np
#import pyarrow.parquet as pq  # For working with parquet files
import re
import seaborn as sns
# Import System libraries
import sys
import os

# Visualization libraries
import matplotlib.pyplot as plt  # For Matplotlib visualizations
import seaborn as sns            # For Seaborn visualizations
import plotly.express as px      # For interactive visualizations with Plotly
# %matplotlib inline  # Uncomment for Jupyter notebooks to display plots inline


# Path to the neighboring 'data' folder in the local repository
data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))

def reduce_multiple_spaces_and_newlines(df):
    """
    Function to replace multiple spaces and new lines with a single space in string columns.
    """
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'\s+', ' ', text)
            return text.replace('\n', ' ').strip()
        return text
    
    for col in df.select_dtypes(include=['object', 'string']):
        df[col] = df[col].apply(clean_text)
    return df

def basic_analysis(df):
    """
    Performs basic financial analysis on the DataFrame and returns the results as a string.
    """
    df['Betrag (€)'] = pd.to_numeric(df['Betrag (€)'], errors='coerce')
    income = df[df['Betrag (€)'] > 0]['Betrag (€)'].sum()
    expenses = df[df['Betrag (€)'] < 0]['Betrag (€)'].sum()

    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'])
    df['month'] = df['Buchungsdatum'].dt.to_period('M')
    monthly_savings = df.groupby('month')['Betrag (€)'].sum()

    largest_expense = df[df['Betrag (€)'] < 0].nsmallest(1, 'Betrag (€)')
    largest_income = df[df['Betrag (€)'] > 0].nlargest(1, 'Betrag (€)')

    most_frequent_income_source = df[df['Betrag (€)'] > 0]['Zahlungspflichtige*r'].value_counts().idxmax()
    most_frequent_expense_target = df[df['Betrag (€)'] < 0]['Zahlungsempfänger*in'].value_counts().idxmax()

    # Format the output as a single string
    result = f"""Total Income: €{income:.2f}
Total Expenses: €{expenses:.2f}

Monthly Savings:
{monthly_savings.to_string()}

Largest Expense:
{largest_expense[['Buchungsdatum', 'Zahlungsempfänger*in', 'Betrag (€)']].to_string(index=False)}

Largest Income:
{largest_income[['Buchungsdatum', 'Zahlungspflichtige*r', 'Betrag (€)']].to_string(index=False)}

Most frequent income source: {most_frequent_income_source}
Most frequent outgoing target: {most_frequent_expense_target}
"""
    return result


def analyze_by_category(df):
    """
    Groups and analyzes transactions by category.
    Returns a summary as a string.
    """
    # Group transactions by category and calculate total amounts
    category_summary = df.groupby('Umsatztyp')['Betrag (€)'].sum()
    
    # Format the output as a string
    summary_str = "Total by Category:\n"
    summary_str += category_summary.to_string()
    
    return summary_str

def income_expense_trend(df, path):
    """
    Creates a time series analysis of income and expenses.
    Saves the figure in the specified path.

    Parameters:
    - df: DataFrame containing the data.
    - plot_path: Path to save the figure.
    """
    # Ensure 'Buchungsdatum' is datetime
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'], errors='coerce')
    df = df.dropna(subset=['Buchungsdatum'])
    
    # Weekly trend aggregation
    df['week'] = df['Buchungsdatum'].dt.to_period('W').apply(lambda r: r.start_time)
    trend = df.groupby('week')['Betrag (€)'].sum()

    # Plot and save
    plt.figure(figsize=(12, 6))
    trend.plot(kind='line', title='Income and Expense Trend Over Time (Weekly)',
               ylabel='Amount (€)', xlabel='Week')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + '/income_expense_trend')  # Save figure
    plt.close()

    return path

def average_monthly_stats(df):
    """
    Calculates average monthly income and expenses.
    """
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'], errors='coerce')
    df['month'] = df['Buchungsdatum'].dt.to_period('M')
    monthly_income = df[df['Betrag (€)'] > 0].groupby('month')['Betrag (€)'].sum().mean()
    monthly_expenses = df[df['Betrag (€)'] < 0].groupby('month')['Betrag (€)'].sum().mean()
    
    print(f"Average Monthly Income: €{monthly_income:.2f}")
    print(f"Average Monthly Expenses: €{monthly_expenses:.2f}")

def detect_recurring_transactions(df):
    """
    Identifies recurring transactions based on similar amounts and descriptions.
    """
    recurring = df[df.duplicated(['Zahlungsempfänger*in', 'Betrag (€)'], keep=False)]
    recurring_summary = recurring.groupby(['Zahlungsempfänger*in', 'Betrag (€)']).size().reset_index(name='Count')
    recurring_summary = recurring_summary.sort_values(by='Count', ascending=False)
    
    return recurring, recurring_summary

def plot_recurring_summary(recurring_summary):
    """
    Plots a summary of recurring transactions.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=recurring_summary.head(10), x='Count', y='Zahlungsempfänger*in', hue='Betrag (€)')
    plt.title('Top 10 Recurring Transactions')
    plt.xlabel('Count')
    plt.ylabel('Recipient')
    plt.legend(title='Amount (€)')
    plt.show()

def expense_income_ratio(df):
    """
    Calculates the expense-to-income ratio per month.
    """
    df['month'] = df['Buchungsdatum'].dt.to_period('M')
    income = df[df['Betrag (€)'] > 0].groupby('month')['Betrag (€)'].sum()
    expenses = df[df['Betrag (€)'] < 0].groupby('month')['Betrag (€)'].sum().abs()
    ratio = (expenses / income).fillna(0)
    print("\nExpense to Income Ratio per Month:")
    print(ratio)

def expenses_by_recipient(df):
    """
    Provides a breakdown of expenses by recipient.
    """
    expense_summary = df[df['Betrag (€)'] < 0].groupby('Zahlungsempfänger*in')['Betrag (€)'].sum()
    print("\nExpenses by Recipient:")
    print(expense_summary.sort_values())

def detect_outliers(df, threshold=3):
    """
    Identifies unusual transactions using standard deviation.
    """
    mean = df['Betrag (€)'].mean()
    std_dev = df['Betrag (€)'].std()
    outliers = df[(df['Betrag (€)'] < (mean - threshold * std_dev)) | 
                  (df['Betrag (€)'] > (mean + threshold * std_dev))]
    print("\nOutliers:")
    print(outliers[['Buchungsdatum', 'Betrag (€)', 'Verwendungszweck']])

def savings_percentage(df):
    """
    Calculates and visualizes monthly savings percentage.
    """
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'])
    df['month'] = df['Buchungsdatum'].dt.to_period('M')
    income = df[df['Betrag (€)'] > 0].groupby('month')['Betrag (€)'].sum()
    expenses = df[df['Betrag (€)'] < 0].groupby('month')['Betrag (€)'].sum().abs()
    savings_percent = ((income - expenses) / income * 100).fillna(0)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=savings_percent.index.astype(str), y=savings_percent.values, marker='o')
    plt.title('Monthly Savings Percentage')
    plt.xlabel('Month')
    plt.ylabel('Savings Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

def cash_flow_volatility(df):
    """
    Calculates and visualizes cash flow stability.
    """
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'])
    df['month'] = df['Buchungsdatum'].dt.to_period('M')
    monthly_balance = df.groupby('month')['Betrag (€)'].sum()
    volatility = monthly_balance.std() / monthly_balance.mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=monthly_balance.index.astype(str), y=monthly_balance.values, marker='o')
    plt.title('Monthly Cash Flow')
    plt.xlabel('Month')
    plt.ylabel('Cash Flow (€)')
    plt.xticks(rotation=45)
    plt.axhline(0, color='red', linestyle='--', label='Zero Line')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def income_by_source(df):
    """
    Analyzes and visualizes income sources.
    """
    df['Betrag (€)'] = pd.to_numeric(df['Betrag (€)'], errors='coerce')
    income_data = df[df['Betrag (€)'] > 0]
    income_summary = income_data.groupby('Zahlungspflichtige*r')['Betrag (€)'].sum()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=income_summary.index, y=income_summary.values, palette='viridis')
    plt.title('Total Income by Source')
    plt.xlabel('Income Source')
    plt.ylabel('Total Income (€)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def fix_euro_amounts(df, column_name='Betrag (€)'):
    """
    Fixes Euro amount formatting in the DataFrame.
    """
    try:
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].str.replace('.', '', regex=False)
        df[column_name] = df[column_name].str.replace(',', '.', regex=False)
        df[column_name] = df[column_name].astype(float)
        return df
    except KeyError:
        print(f"Column '{column_name}' not found. Available columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error processing column '{column_name}':", str(e))
        return df

def inspect_data(df):
    """
    Function to perform an initial data inspection on a given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.

    Returns:
    None
    """
    print("="*40)
    print("🚀 Basic Data Overview")
    print("="*40)

    # Print the shape of the DataFrame (rows, columns)
    print(f"🗂 Shape of the DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    # Display the first 5 rows of the dataset
    print("\n🔍 First 5 rows of the DataFrame:")
    print(df.head(5))

    # Get information about data types, missing values, and memory usage
    print("\n📊 DataFrame Information:")
    df.info()

    # Show basic statistics for numeric columns
    print("\n📈 Summary Statistics of Numeric Columns:")
    print(df.describe())

    # Count missing values and display
    print("\n❓ Missing Values Overview:")
    missing_df = count_missing_values(df)
    print(missing_df)

    # Print unique values for categorical and object columns
    print("\n🔑 Unique Values in Categorical/Object Columns:")
    print_unique_values(df)

def count_missing_values(df, columns=None):
    """
    Counts occurrences of "nan", "None", np.nan, None, pd.NA, and '' (empty string) in the specified columns of a DataFrame.
    Also includes the data type of each column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns to check.
    columns (list, optional): List of column names (as strings) to check for missing values.
                              Defaults to all columns in the DataFrame.

    Returns:
    pd.DataFrame: A DataFrame summarizing the count of missing value types per column.
    """
    if columns is None:
        columns = df.columns  # Default to all columns if not provided
        print("Columns are none:", columns)

    results = []
    for col in columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            # If the column is categorical, convert it to object to capture all kinds of missing values
            col_data = df[col].astype(object)
        else:
            col_data = df[col]

        # Total missing (both np.nan, pd.NA, None)
        total_missing = col_data.isna().sum()

        # Count only np.nan by checking where it's actually a float and is NaN
        np_nan_count = (col_data.apply(lambda x: isinstance(x, float) and pd.isna(x))).sum()

        # Count actual `None` values (None treated as an object)
        none_object_count = (col_data.apply(lambda x: x is None)).sum()

        # pd.NA count: in categorical columns, we check if the missing value type is distinct from np.nan
        if pd.api.types.is_categorical_dtype(df[col]):
            pd_na_count = col_data.isna().sum() - none_object_count - np_nan_count
        else:
            pd_na_count = total_missing - np_nan_count

        counts = {
            'Column': col,
            'Data Type': df[col].dtype,
            'nan (string)': (col_data == 'nan').sum(),
            'None (string)': (col_data == 'None').sum(),
            'np.nan': np_nan_count,
            'pd.NA': pd_na_count,
            'None (object)': none_object_count,
            'Empty string': (col_data == '').sum(),
        }
        results.append(counts)

    return pd.DataFrame(results)

def print_unique_values(df):
    """
    Function to print the number of unique values for categorical and object columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.

    Returns:
    None
    """
    cat_obj_cols = df.select_dtypes(include=['category', 'object']).columns
    unique_counts = {}

    for col in cat_obj_cols:
        unique_counts[col] = df[col].nunique()

    if unique_counts:
        for col, count in unique_counts.items():
            print(f"Column '{col}' has {count} unique values")
    else:
        print("No categorical or object columns found.")
    return


def plot_monthly_income_vs_expenses_to_pdf(df):
    """
    Plots a stacked bar chart of monthly income versus expenses.

    Parameters:
    df (DataFrame): The DataFrame containing financial data with 'Buchungsdatum' and 'Betrag (€)' columns.

    This function calculates the total income and expenses per month and visualizes the data
    using a stacked bar chart. It categorizes the transactions as 'Income' or 'Expenses' based on
    the transaction amounts and groups them by month for comparison.
    """
    # Ensure 'Buchungsdatum' is in datetime format
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'])

    # Create a new column for the month
    df['month'] = df['Buchungsdatum'].dt.to_period('M').astype(str)  # Convert to string for better plotting

    # Classify transactions as Income or Expenses
    df['Type'] = df['Betrag (€)'].apply(lambda x: 'Income' if x > 0 else 'Expenses')
    
    # Group by month and transaction type, summing the amounts
    monthly_data = df.groupby(['month', 'Type'])['Betrag (€)'].sum().unstack().fillna(0)

    # Plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_data.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red'])
    ax.set_title('Monthly Income vs Expenses')
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount (€)')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to prevent clipping

    return fig
