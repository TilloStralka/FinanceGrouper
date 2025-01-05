# Importing necessary librarys
import pandas as pd
import numpy as np
#import pyarrow.parquet as pq  # For working with parquet files
import re
import seaborn as sns
# Import System libraries
import sys
import os
import inspect

# Visualization libraries
import matplotlib.pyplot as plt  # For Matplotlib visualizations
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns            # For Seaborn visualizations
import plotly.express as px      # For interactive visualizations with Plotly
# %matplotlib inline  # Uncomment for Jupyter notebooks to display plots inline

# For the preparation of the overview report pdf 
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image


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


def truncate_strings_in_dataframe(df, max_length=30):
    """
    Truncates all strings in a pandas DataFrame to a specified maximum length.

    Parameters:
    df (DataFrame): The pandas DataFrame to process.
    max_length (int): The maximum length for strings.

    Returns:
    DataFrame: A new DataFrame with truncated strings.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_truncated = df
    
    # Apply truncation to all string entries
    for col in df_truncated.select_dtypes(include=['object', 'string', 'string[python]']).columns:
        df_truncated[col] = df_truncated[col].apply(lambda x: x[:max_length] if isinstance(x, str) else x)
    
    return df_truncated


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
    summary_str += "\n \n"

    return summary_str

def income_expense_trend(df, path):
    """
    Creates a time series analysis of income and expenses.
    Saves the figure in the specified path.

    Parameters:
    - df: DataFrame containing the data.
    - plot_path: Path to save the figure.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

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
    plt.savefig(path_saving)  # Save figure
    plt.show

    return path_saving

def average_monthly_stats(df):
    """
    Calculates the average monthly income and expenses and returns a summary string.
    """
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'], errors='coerce')
    df['month'] = df['Buchungsdatum'].dt.to_period('M')
    monthly_income = df[df['Betrag (€)'] > 0].groupby('month')['Betrag (€)'].sum().mean()
    monthly_expenses = df[df['Betrag (€)'] < 0].groupby('month')['Betrag (€)'].sum().mean()
    
    result = (
        f"Monthly income VS expanse: \n"
        f"The average monthly income is €{monthly_income:.2f}, and the average monthly expenses are €{monthly_expenses:.2f}. \n \n"
    )
    return result

def detect_recurring_transactions(df):
    """
    Identifies recurring transactions based on similar amounts and descriptions.
    """
    recurring = df[df.duplicated(['Zahlungsempfänger*in', 'Betrag (€)'], keep=False)]
    recurring_summary = recurring.groupby(['Zahlungsempfänger*in', 'Betrag (€)']).size().reset_index(name='Count')
    recurring_summary = recurring_summary.sort_values(by='Count', ascending=False)
    
    return recurring_summary

def text_recurring_transactions(df):
    """
    Extracts the first 5 rows of a pandas DataFrame and formats it into a readable string.
    
    Parameters:
    df (DataFrame): The pandas DataFrame to extract rows from.

    Returns:
    str: A formatted string of the first 5 rows.
    """
    # Extract the first 5 rows
    first_five = df.head(5)
    
    # Convert the rows to a nicely formatted string
    formatted_string = first_five.to_string(index=False, justify="left")
    
    # Add a title to the string
    result = f"The 5 most recurring transactions:\n\n{formatted_string}"
    return result    

def plot_recurring_summary(recurring_summary, path):
    """
    Plots a summary of recurring transactions.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=recurring_summary.head(10), x='Count', y='Zahlungsempfänger*in', hue='Betrag (€)')
    plt.title('Top 10 Recurring Transactions')
    plt.xlabel('Count')
    plt.ylabel('Recipient')
    plt.legend(title='Amount (€)')
    plt.savefig(path_saving)  # Save figure
    plt.show()
    
    return path_saving
    
def expense_income_ratio(df):
    """
    Calculates the expense-to-income ratio per month and returns it as a string.
    """
    # Extract the month from 'Buchungsdatum'
    df['month'] = df['Buchungsdatum'].dt.to_period('M')
    
    # Calculate monthly income and expenses
    income = df[df['Betrag (€)'] > 0].groupby('month')['Betrag (€)'].sum()
    expenses = df[df['Betrag (€)'] < 0].groupby('month')['Betrag (€)'].sum().abs()
    
    # Calculate the expense-to-income ratio
    ratio = (expenses / income).fillna(0)
    
    # Prepare the output string
    ratio_string = "\nExpense to Income Ratio per Month:\n" + ratio.to_string()
    
    # Print the ratio string
    #print(ratio_string)
    
    # Return the string representation
    return ratio_string

def expenses_by_recipient(df):
    """
    Provides a breakdown of expenses by recipient and returns it as a string.
    """
    expense_summary = df[df['Betrag (€)'] < 0].groupby('Zahlungsempfänger*in')['Betrag (€)'].sum()
    sorted_expense_summary = expense_summary.sort_values()
    
    output = "\nExpenses by Recipient:\n"
    output += sorted_expense_summary.to_string()
    
    print(output)  
    return output

def detect_outliers(df, threshold=3):
    """
    Identifies unusual transactions using standard deviation and returns the result as a string.
    """
    mean = df['Betrag (€)'].mean()
    std_dev = df['Betrag (€)'].std()
    outliers = df[(df['Betrag (€)'] < (mean - threshold * std_dev)) | 
                  (df['Betrag (€)'] > (mean + threshold * std_dev))]
    
    output = "\nOutliers:\n"
    if not outliers.empty:
        output += outliers[['Buchungsdatum', 'Betrag (€)', 'Verwendungszweck']].to_string(index=False)
    else:
        output += "No outliers detected."
    
    print(output)  # Optional: Drucken der Ergebnisse
    return output

def savings_percentage(df, path):
    """
    Calculates and visualizes monthly savings percentage.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

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
    plt.savefig(path_saving)  # Save figure
    plt.show()
    return path_saving


def cash_flow_volatility(df, path):
    """
    Calculates and visualizes cash flow stability.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

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
    plt.savefig(path_saving)  # Save figure
    plt.show()
    return path_saving

def income_by_source(df, path):
    """
    Analyzes and visualizes income sources.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

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
    plt.savefig(path_saving)  # Save figure
    plt.show()
    return path_saving

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


def plot_monthly_income_vs_expenses_to_pdf(df, path):
    """
    Plots a stacked bar chart of monthly income versus expenses.

    Parameters:
    df (DataFrame): The DataFrame containing financial data with 'Buchungsdatum' and 'Betrag (€)' columns.

    This function calculates the total income and expenses per month and visualizes the data
    using a stacked bar chart. It categorizes the transactions as 'Income' or 'Expenses' based on
    the transaction amounts and groups them by month for comparison.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

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
    plt.savefig(path_saving)  # Save figure
    plt.show()
    return path_saving

def plot_total_income_expenses(df, path):
    """
    Plots total income and expenses from the given DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame containing financial data with a 'Betrag (€)' column.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

    # Calculate total income: sum of all positive amounts in 'Betrag (€)'
    total_income = df[df['Betrag (€)'] > 0]['Betrag (€)'].sum()

    # Calculate total expenses: sum of all negative amounts in 'Betrag (€)' (make positive for display)
    total_expenses = df[df['Betrag (€)'] < 0]['Betrag (€)'].sum()

    # Prepare data for plotting as a dictionary
    totals = {'Income': total_income, 'Expenses': abs(total_expenses)}  # Convert expenses to positive

    # Create a bar plot using Seaborn
    plt.figure(figsize=(10, 4))  # Set figure size
    sns.barplot(x=list(totals.keys()), y=list(totals.values()), palette='muted')

    # Set the title and labels for the plot
    plt.title("Total Income vs Expenses")
    plt.ylabel("Amount (€)")
    plt.savefig(path_saving)  # Save figure
    plt.show()

    return path_saving

def plot_monthly_savings(df, path):
    """
    Plots monthly savings over time.
    
    Parameters:
    df (DataFrame): The DataFrame containing financial data with 'Buchungsdatum' and 'Betrag (€)' columns.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

    # Ensure 'Buchungsdatum' is in datetime format
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'], errors='coerce')

    # Group by month and calculate total savings (income minus expenses)
    df['month'] = df['Buchungsdatum'].dt.to_period('M')  # Extract month
    monthly_savings = df.groupby('month')['Betrag (€)'].sum().reset_index()

    # Convert 'month' back to a datetime object for plotting
    monthly_savings['month'] = monthly_savings['month'].dt.to_timestamp()

    # Plot
    plt.figure(figsize=(10, 5))  # Set figure size
    sns.lineplot(x='month', y='Betrag (€)', data=monthly_savings, marker='o')
    
    # Set title and labels
    plt.title('Monthly Savings')
    plt.ylabel('Savings (€)')
    plt.xlabel('Month')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid()  # Add grid for better visualization
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(path_saving)  # Save figure
    plt.show()

    return path_saving

def plot_expenses_barplot(df, path):
    """
    Creates a horizontal bar plot of total expenses by category with grid lines and axes.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'Umsatztyp', 'Category', and 'Betrag (€)'.
        path (str): Directory path where the bar plot will be saved.
    
    Returns:
        str: Path to the saved bar plot image.
    """
    # Filter for 'Ausgang' in 'Umsatztyp'
    expenses_df = df[df['Umsatztyp'] == 'Ausgang']

    # Aggregate data by Category
    category_expenses = expenses_df.groupby('Category')['Betrag (€)'].sum().abs()

    # Sort by total expenses
    category_expenses = category_expenses.sort_values(ascending=True)

    # Generate a color palette
    colors = sns.color_palette("husl", len(category_expenses))

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(category_expenses.index, category_expenses.values, color=colors)

    # Add labels to each bar
    for bar, value in zip(bars, category_expenses.values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.2f} €", 
                va='center', ha='left', fontsize=10)

    # Set labels and title
    ax.set_xlabel("Total Expenses (€)", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    ax.set_title("Total Expenses by Category", fontsize=14)

    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Remove spines for cleaner look
    sns.despine(left=True, bottom=False)

    # Save the figure
    function_name = inspect.currentframe().f_code.co_name
    path_saving = os.path.join(path, f"{function_name}.png")
    plt.savefig(path_saving, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # Close the figure to release memory

    return path_saving

def plot_expenses_barplot_percent(df, path):
    """
    Creates a horizontal bar plot of total expenses by category in percentages with grid lines and axes.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'Umsatztyp', 'Category', and 'Betrag (€)'.
        path (str): Directory path where the bar plot will be saved.
    
    Returns:
        str: Path to the saved bar plot image.
    """
    # Filter for 'Ausgang' in 'Umsatztyp'
    expenses_df = df[df['Umsatztyp'] == 'Ausgang']

    # Aggregate data by Category
    category_expenses = expenses_df.groupby('Category')['Betrag (€)'].sum().abs()

    # Calculate percentages
    total_expenses = category_expenses.sum()
    category_expenses_percent = (category_expenses / total_expenses) * 100

    # Sort by percentage
    category_expenses_percent = category_expenses_percent.sort_values(ascending=True)

    # Generate a color palette
    colors = sns.color_palette("husl", len(category_expenses_percent))

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(category_expenses_percent.index, category_expenses_percent.values, color=colors)

    # Add percentage labels to each bar
    for bar, value in zip(bars, category_expenses_percent.values):
        # Position the text slightly to the right of the bar
        ax.text(
            value + 0.5,  # Offset for visibility
            bar.get_y() + bar.get_height() / 2, 
            f"{value:.2f}\\%",  # Format as percentage with LaTeX-compatible percent symbol
            va='center', 
            ha='left', 
            fontsize=10
        )

    # Set labels and title with LaTeX-compatible percent symbol
    ax.set_xlabel("Total Expenses (\\%)", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    ax.set_title("Total Expenses by Category (Percentage)", fontsize=14)

    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Remove spines for cleaner look
    sns.despine(left=True, bottom=False)

    # Save the figure
    function_name = inspect.currentframe().f_code.co_name
    path_saving = os.path.join(path, f"{function_name}.png")
    plt.savefig(path_saving, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # Close the figure to release memory

    return path_saving


def translate_text(text):
    """
    Translates the input text into a specified target language using the translator object.

    Parameters:
        text (str): The text to be translated.

    Returns:
        str: Translated text if successful, or an error message if the translation fails.
    """
    try:
        # Attempt to translate the text using the translator object
        return translator.translate(text)
    except Exception as e:
        # If an error occurs, return the error message
        return f"Error during translation: {e}"


def determine_final_category(row):
    """
    Determines the final category for a given row based on a priority rule.

    The function selects the first non-'Miscellaneous' category from the columns 
    'Category1', 'Category2', 'Category3', and 'Category4'. If all categories 
    are 'Miscellaneous', it defaults to 'Miscellaneous'.

    Parameters:
        row (pd.Series): A row from a DataFrame containing the category columns.

    Returns:
        str: The determined final category.
    """
    # Apply the priority rule: choose the first non-'Miscellaneous' category
    for category in [row['Category1'], row['Category2'], row['Category3'], row['Category4']]:
        if category != 'Miscellaneous':
            return category
    # Fallback to 'Miscellaneous' if all categories are 'Miscellaneous'
    return 'Miscellaneous'


def plot_expenses_violin(df, path):
    """
    Creates a violin plot of expense distributions by category with positive values and a legend below.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'Umsatztyp', 'Category', and 'Betrag (€)'.
        path (str): Directory path where the violin plot will be saved.
    
    Returns:
        str: Path to the saved violin plot image.
    """
    # Filter for 'Ausgang' in 'Umsatztyp'
    expenses_df = df[df['Umsatztyp'] == 'Ausgang'].copy()

    # Convert amounts to absolute values
    expenses_df['Betrag (€)'] = expenses_df['Betrag (€)'].abs()

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("husl", n_colors=expenses_df['Category'].nunique())
    sns.violinplot(
        data=expenses_df, 
        x='Category', 
        y='Betrag (€)', 
        scale='width', 
        inner='quartile', 
        linewidth=1, 
        palette=palette, 
        ax=ax
    )

    # Remove X-axis labels
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_title("Distribution of Expenses by Category", fontsize=14)
    ax.set_ylabel("Expense Amount (€)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Create legend below the plot
    handles = [plt.Line2D([0], [0], color=color, lw=10) for color in palette]
    categories = expenses_df['Category'].unique()
    legend = ax.legend(
        handles, 
        categories, 
        title="Categories", 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=4, 
        fontsize=10, 
        title_fontsize=12
    )

    # Save the figure
    function_name = inspect.currentframe().f_code.co_name
    path_saving = os.path.join(path, f"{function_name}.png")
    plt.savefig(path_saving, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # Close the figure to release memory

    return path_saving

def plot_largest_expenses_income(df, path):
    """
    Plots the top 5 sources of income and the top 5 expenses in pie charts with connection pointer lines
    and percentages inside the pie pieces.
    
    Parameters:
    df (DataFrame): The DataFrame containing financial data with 'Betrag (€)', 'Zahlungspflichtige*r', and 'Zahlungsempfänger*in' columns.
    path (str): The directory where the plot will be saved.
    
    Returns:
    str: Path to the saved figure.
    """
    # Use LaTeX styling and font for the plots
    plt.rcParams.update(plt.rcParamsDefault)
    mpl.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name

    # Construct the path for saving the plot
    path_saving = os.path.join(path, f'{function_name}.png')

    # Calculate the top 5 sources of income
    top_income = df[df['Betrag (€)'] > 0].groupby('Zahlungspflichtige*r')['Betrag (€)'].sum().nlargest(5)

    # Calculate the top 5 sources of expenses (converted to positive values)
    top_expenses = df[df['Betrag (€)'] < 0].groupby('Zahlungsempfänger*in')['Betrag (€)'].sum().nsmallest(5).abs()

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Generate custom colors for income (blue to white) and expenses (red to white)
    income_colors = plt.cm.Blues(np.linspace(0.1, 0.9, len(top_income)))
    expense_colors = plt.cm.Reds(np.linspace(0.1, 0.9, len(top_expenses)))

    # Plot the top 5 sources of income as a pie chart
    income_pie = axes[0].pie(
        top_income, 
        autopct='%.1f%%', 
        colors=income_colors,  # Use blue to white color map
        wedgeprops={"linewidth": 2.0, "edgecolor": "white"}, 
        textprops={'size': 'xx-large'}
    )
    axes[0].set_title("Top 5 Income Sources", fontsize=16)

    # Plot the top 5 sources of expenses as a pie chart
    expense_pie = axes[1].pie(
        top_expenses, 
        autopct='%.1f%%', 
        colors=expense_colors,  # Use red to white color map
        wedgeprops={"linewidth": 2.0, "edgecolor": "white"}, 
        textprops={'size': 'xx-large'}
    )
    axes[1].set_title("Top 5 Expenses", fontsize=16)

    # Add legends below the plots
    fig.legend(
        labels=list(top_income.index) + list(top_expenses.index),
        loc='lower center', 
        fontsize='large', 
        ncol=2,
        frameon=False
    )

    # Adjust layout to make space for the legends
    plt.subplots_adjust(bottom=0.2)

    # Save the plot to the specified path
    plt.savefig(path_saving, dpi=300, bbox_inches='tight')
    plt.show()

    return path_saving



    
def plot_transaction_distribution(df, path):
    """
    Plots the distribution of transaction amounts using a histogram.

    Parameters:
    df (DataFrame): The DataFrame containing financial data with a 'Betrag (€)' column.

    This function generates a histogram that shows the frequency of transaction amounts,
    along with a Kernel Density Estimate (KDE) line to visualize the distribution's shape.
    The y-axis is set to a logarithmic scale to better visualize the distribution of 
    transaction amounts, especially when dealing with a wide range of values.
    """    
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

    plt.figure(figsize=(10, 4)) 

    # Create the histogram with a KDE line
    sns.histplot(df['Betrag (€)'], kde=True)

    # Set the title and labels
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Amount (€)")
    plt.ylabel("Frequency")
    # Set the y-axis to logarithmic scale, starting from 1
    #plt.yscale('log')
    #plt.ylim(bottom=0.1)  # Ensure the lower limit of the y-axis starts at 1

    # Show and save the plot
    plt.savefig(path_saving)  # Save figure
    plt.show()

    return path_saving

def plot_monthly_income_vs_expenses(df, path):
    """
    Plots a stacked bar chart of monthly income versus expenses.

    Parameters:
    df (DataFrame): The DataFrame containing financial data with 'Buchungsdatum' and 'Betrag (€)' columns.
    path (str): The folder path where the plot will be saved.

    Returns:
    str: The full path of the saved image.
    """
    # Get the function name dynamically
    function_name = inspect.currentframe().f_code.co_name
    
    # Construct the path to save the figure with the function name
    path_saving = os.path.join(path, f'{function_name}.png')

    # Ensure 'Buchungsdatum' is in datetime format
    df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'])

    # Create a new column for the month
    df['month'] = df['Buchungsdatum'].dt.to_period('M').astype(str)

    # Classify transactions as Income or Expenses
    df['Type'] = df['Betrag (€)'].apply(lambda x: 'Income' if x > 0 else 'Expenses')

    # Group by month and transaction type, summing the amounts
    monthly_data = df.groupby(['month', 'Type'])['Betrag (€)'].sum().unstack(fill_value=0)

    # Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    monthly_data.plot(kind='bar', stacked=True, ax=ax, color=['#d62728', '#1f77b4'])  # Colors for Income/Expenses
    ax.set_title('Monthly Income vs Expenses')
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount (€)')
    ax.legend(title='Type')
    plt.xticks(rotation=45)  # Rotate month labels for better visibility

    # Save the figure
    plt.tight_layout()  # Adjust layout for saving
    plt.savefig(path_saving)  # Save the plot as a PNG file
    plt.show()  # Show the plot

    return path_saving

def save_to_pdf(text_functions, plot_functions, output_pdf_path):
    """
    Generates an overview PDF with text outputs and plots.

    Parameters:
    text_functions: List of tuples (heading, function) generating text output.
    plot_functions: List of tuples (heading, function) generating plots saved as images.
    output_pdf_path: Path where the PDF should be saved.
    """
    # Create the PDF document
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4
    current_y = height - 40  # Starting point on the Y-axis (40 pt margin from the top)

    def check_space(required_space):
        """
        Checks if there is enough space for the next section.
        If not, it adds a new page and resets the Y-position.
        """
        nonlocal current_y
        if current_y - required_space < 40:  # Check if enough space remains
            c.showPage()  # Add a new page
            current_y = height - 40  # Reset the Y-position

    # Add the header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(200, current_y, "Financial Report")
    current_y -= 40  # Leave space after the header

    # Add text content with headings
    c.setFont("Helvetica", 12)

    # Interleave the text and plot functions in the specified order
    combined_functions = []
    max_len = max(len(text_functions), len(plot_functions))

    # Append text and plot functions in the order you define them
    for i in range(max_len):
        if i < len(plot_functions):
            combined_functions.append(('plot', plot_functions[i]))        
        if i < len(text_functions):
            combined_functions.append(('text', text_functions[i]))

    # Process the combined list, alternating between text and plots
    for func_type, (heading, func) in combined_functions:
        if func_type == 'text':
            # Estimate required space for heading and some lines of text
            estimated_lines = 10  # Adjust this number based on expected text length
            required_space = 20 + (estimated_lines * 15)
            check_space(required_space)

            # Add the heading
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, current_y, heading)
            current_y -= 20  # Adjust Y-position after the heading

            # Generate the text content
            c.setFont("Helvetica", 10)
            output = func()  # Run the text function
            text_lines = output.split('\n')  # Handle line breaks
            for line in text_lines:
                c.drawString(40, current_y, line)  # Draw each line of text
                current_y -= 15  # Decrease the Y-position for the next line
                check_space(15)  # Ensure enough space for the next line

        elif func_type == 'plot':
            # Generate the plot and get its file path
            plot_path = func()  # Run the plot function and get the plot path

            # Load the image and get its dimensions
            img = Image.open(plot_path)
            img_width, img_height = img.size

            # Calculate the scaled size of the image
            max_width = width - 80  # Allow for 40 pt margin on left/right
            max_height = height - 80  # Allow for 40 pt margin on top/bottom
            ratio = min(max_width / img_width, max_height / img_height)
            img_width = int(img_width * ratio)
            img_height = int(img_height * ratio)

            # Estimate required space for heading and image
            required_space = 20 + img_height + 20  # Heading + Image + Padding
            check_space(required_space)

            # Add the heading
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, current_y, heading)
            current_y -= 20  # Adjust Y-position after the heading

            # Draw the image
            c.drawImage(plot_path, 40, current_y - img_height, width=img_width, height=img_height)
            current_y -= img_height + 20  # Adjust Y-position after the image

    # Save the PDF
    c.save()
    print(f"PDF saved to {output_pdf_path}")
    

# Define categories with corresponding key words for the bert categorizer 
categories = {
"Housing": [
    "rent", "miete", "monthly rent", "rental payment", "apartment rent", "flat rent", 
    "mortgage", "property taxes", "real estate tax", "home maintenance", "homeowners insurance", 
    "hoa fees", "house insurance", "apartment insurance", "property management", "landlord", 
    "vermieter", "tenant", "mieter", "rental contract", "lease", "lease agreement", 
    "rental deposit", "security deposit", "property maintenance", "repair costs", "apartment repair", 
    "utility costs", "Nebenkosten", "heating cost", "heating bill", "water bill", "electricity bill", 
    "internet bill", "trash collection fee", "building maintenance fee", "common area maintenance", 
    "garbage disposal", "parking fees", "garage rent", "storage unit rent", "property rental", 
    "vacation rental", "rental property", "property rental fee", "apartment complex", "real estate", 
    "real estate agent", "property buying", "house buying", "property selling", "house selling", 
    "real estate tax", "mortgage interest", "down payment", "home equity loan", "home loan", 
    "mortgage loan", "refinancing", "home appraisal", "moving costs", "moving truck", "moving company", 
    "storage fees", "rental property tax", "rent increase", "rent reduction", "rental income"
],
"Transportation": [
    "car payment", "gas", "public transit", "ridesharing", "parking", "tolls",
    "vehicle maintenance", "fuel", "auto repair", "commuting", "bike rental",
    "FERNVERKEHR", "flixbus", "PAYPAL .DBVERTR", "DB", "Flight", "Airport", 
    "Travel", "Reise", "taxi", "ride sharing", "Uber", "Lyft", "BlaBlaCar", "carpooling", 
    "bus", "train", "subway", "metro", "tram", "bike", "e-scooter", "e-bike", "train ticket",
    "flight ticket", "plane ticket", "airline", "lufthansa", "ryanair", "easyjet", "airfrance", 
    "kLM", "turkish airlines", "british airways", "emirates", "qatar airways", "saa", 
    "condor", "australian airlines", "jetstar", "wizz air", "air berlin", "airport shuttle", 
    "airport transfer", "rental car", "car hire", "travel insurance", "holiday travel", 
    "turbulence", "seat reservation", "boarding pass", "baggage fee", "excess baggage", 
    "flight change fee", "train station", "bus station", "taxi fare", "ride fee", 
    "petrol station", "fuel station", "gas station", "charge station", "charging station", 
    "auto club", "parking meter", "toll road", "vehicle registration", "road toll", "highway toll", 
    "road trip", "long distance travel", "holiday flight", "business class", "economy class",
    "first class", "airport lounge", "train reservation", "train travel", "train ride", 
    "flight booking", "international travel", "domestic flight", "train journey", "airport taxes",
    "boarding gate", "carry-on baggage", "checked luggage", "travel booking", "trip", "City Flitzer", "Mietwagen", "Teilauto"
],
"Eat Out or Get Food": [
    "groceries", "restaurant", "delivery", "lunch", "dining out", "snacks",
    "coffee shop", "takeout", "meal kit", "gastro", "dean david", "cafe",
    "baeckerei", "coffee fellows", "jim block", "don qui", "Osteria", "subway",
    "backhaus", "burger king", "campus suite", "juice.more", "Backerei",
    "Avni Terhani", "vegan", "thai", "indisch", "cusine", "bäcker", "baecker"
    ],
"Cash": [
    "bargeld", "automat", "cash", "ATM", "eurocash", "withdrawal", "cash withdrawal",
    "sparkasse", "bankautomat", "geldautomat", "bareinzahlung", "auszahlung", 
    "cashpoint", "bank cash", "barzahlung", "ATM withdrawal", "bank withdrawal",
    "Bargeldabhebung", "Barabhebung", "Transact", "Straße"
],
"Supermarket and Drogerie": [
    "karstadt", "galeria", "kaufhof", "mueller", "migros", "coop", "dm fil", 
    "rossmann", "go asia", "Drogerie", "SUCKFUELL", "butlers", "Suckfull", 
    "Bio-Market", "conrad m", "lidl", "aldi", "aldi süd", "aldi nord", "edeka", 
    "edeka center", "alnatura", "rewe", "rewe city", "vollcorner", "konsum", 
    "penny", "netto", "netto city", "kaufland", "real", "marktkauf", "hit", 
    "tegut", "spar", "eurospar", "interspar", "denn's biomarkt", "biomarkt", 
    "basic", "metro", "famila", "globus", "toom", "norma", "baekcer", "bäcker", "bakery"
],
"Utilities": [
    "electricity", "strom", "power", "energie", "energy", 
    "internet", "wifi", "dsl", "breitband", "broadband", 
    "water", "wasser", "wasserversorgung", "water supply", 
    "cell phone", "mobilfunk", "handyvertrag", "cellular", 
    "mobile plan", "mobile service", "gas utility", "gas", 
    "heating", "heizung", "heizkosten", "cable", "kabel", 
    "kabelanschluss", "cable connection", "sewer", "abwasser", 
    "kanalisation", "sewage", "trash collection", "müllabfuhr", 
    "waste disposal", "recycling service", "müllentsorgung", 
    "garbage collection", "netzbetreiber", "local network provider", 
    "stromanbieter", "energieversorger", "electricity provider", 
    "gas provider", "wasserwerk", "waterworks", "energie werk", 
    "power plant", "smartphone bill", "phone bill", "utility fees", 
    "Nebenkosten", "maintenance fees", "local utility provider"
],
"Insurance": [
    "health insurance", "krankenversicherung", "tkk", "techniker krankenkasse", 
    "aok", "barmer", "dak", "bkk", "krankenkasse", "auto insurance", 
    "kfz-versicherung", "autoversicherung", "life insurance", 
    "lebensversicherung", "homeowners insurance", "hausratversicherung", 
    "gebäudeversicherung", "disability insurance", "berufsunfähigkeitsversicherung", 
    "invaliditätsversicherung", "pet insurance", "haustierversicherung", 
    "travel insurance", "reiseversicherung", "auslandsversicherung", 
    "renter's insurance", "mieterversicherung", "haftpflichtversicherung", 
    "liability insurance", "unfallversicherung", "dental insurance", 
    "zahnzusatzversicherung", "zahnversicherung", "supplemental insurance", 
    "zusatzversicherung", "private insurance", "private krankenversicherung", 
    "pension insurance", "rentenversicherung", "healthcare plan", 
    "versicherung", "insurance premium", "insurance fee", "premium", 
    "versicherungskosten", "policy", "versicherungsvertrag", 
    "deductible", "selbstbeteiligung", "versicherungsschutz", 
    "risk coverage", "deckungsumfang", "versicherungstarif"
],
"Medical and Healthcare": [
    "prescription", "rezept", "urgent care", "notaufnahme", "notfallmedizin", 
    "dental", "zahnarzt", "zahnmedizin", "dental care", "zahnpflege", 
    "medical device", "medizinisches gerät", "therapie", "physiotherapie", 
    "therapy", "psychotherapie", "chiropractor", "chiropraktiker", 
    "eye care", "augenarzt", "vision", "sehhilfe", "optiker", "brille", 
    "surgery", "operation", "chirurgie", "mental health", "psychische gesundheit", 
    "mental wellbeing", "psychiatrie", "psychologe", "psychotherapie", 
    "check-up", "vorsorgeuntersuchung", "gesundheitscheck", 
    "pharmacy", "apotheke", "online apotheke", "shop apotheke", 
    "medikamente", "medications", "dolodobendan", "ibuprofen", "aspirin", 
    "paracetamol", "schmerzmittel", "antibiotika", "antidepressiva", 
    "krankenversicherung", "krankenhaus", "clinic", "klinik", "arzt", 
    "ärztlicher dienst", "hospital", "healthcare", "gesundheitsversorgung", 
    "krankenpflege", "nursing", "rehabilitation", "rehabilitation center", 
    "impfung", "vaccination", "health screening", "diagnose", "labor", 
    "blood test", "bluttest", "ct scan", "mrt", "ultraschall", 
    "pregnancy test", "schwangerschaftstest", "medizin", "homöopathie", 
    "alternative medizin", "orthopädie", "dermatologie", "diabetes", 
    "arztpraxis", "arztbesuch", "sprechstunde", "arzttermin", 
    "apotheken umschau", "medizinische hilfsmittel", "krankenbett", 
    "erste hilfe", "first aid", "gesundheitsamt", "health authority"
],
"Saving, Investing and Debt Payments": [
    "debt repayment", "loan repayment", "debt", "schulden", "credit card debt", 
    "credit card payment", "schuldenabbau", "retirement", "pension", "rentenversicherung", 
    "investment", "investment fund", "investment strategy", "investieren", "investment portfolio", 
    "mutual fund", "aktienfonds", "investmentfonds", "stock", "aktie", "shares", "dividende", 
    "stock market", "boerse", "aktienmarkt", "etf", "exchange traded fund", "kryptowährung", 
    "cryptocurrency", "bitcoin", "ethereum", "blockchain", "crypto trading", "crypto investment", 
    "sparplan", "savings plan", "capital investment", "portfolio", "asset management", 
    "loan payment", "kreditrate", "credit payment", "student loan", "studentenloan", 
    "IRA", "401(k)", "pension plan", "pensionsfonds", "retirement savings", "anlageberatung", 
    "broker", "brokerage account", "online broker", "trade", "trading", "day trading", 
    "stock trading", "robo-advisor", "trade republic", "depot", "brokerage fees", "robo-trading", 
    "forex trading", "foreign exchange", "crypto exchange", "bitcoin exchange", "commodity trading", 
    "real estate investment", "immobilieninvestition", "crowdinvesting", "crowdfunding", 
    "savings account", "bank savings", "banksparplan", "savings bond", "money market", "investment savings"
],
"Personal Spending": [
    "gym", "fitness studio", "fitness center", "workout", "training", "sports", "yoga", 
    "climbing", "klettern", "bouldern", "kletterhalle", "bloc no limit", "berta block", 
    "swimming pool", "schwimmhalle", "swimming", "aquatic center", "spa", "sauna", "massage", 
    "clothes", "fashion", "shopping", "shoes", "clothing store", "apparel", "haircut", "barber", 
    "beauty salon", "home decor", "furniture", "interior design", "makeup", "cosmetics", "nail care", 
    "subscriptions", "streaming", "music subscription", "magazine subscription", "membership", 
    "gym membership", "club membership", "electronics", "gadgets", "tech", "fitness equipment", 
    "personal care", "toiletries", "skincare", "hair care", "personal hygiene", "travel bag", 
    "sportswear", "activewear", "athletic shoes", "outdoor gear", "mountain biking", "snowboarding", 
    "hiking", "running", "bicycle accessories", "fitness tracker", "wearable tech", "smartwatch", 
    "sports accessories", "personal trainer", "group fitness class", "fitness retreat", 
    "outdoor activities", "fitness nutrition", "protein shakes", "sports drink", "running shoes"
],
"Online Shopping": [
    "otto", "conrad elec", "amzn mktp", "amzn", "amazon", "ebay", "aliexpress", "wish", "etsy",
    "zalando", "shein", "boohoo", "asos", "myntra", "puma", "nike", "adidas", "decathlon", 
    "media markt", "saturn", "real", "lidl online", "rewe online", "dm online", "kaufland online",
    "ikea", "home24", "wayfair", "lulus", "target", "walmart", "costco", "home depot", "sears",
    "best buy", "john lewis", "marks and spencer", "hm", "mango", "pull&bear", "stradivarius", 
    "urban outfitters", "forever 21", "topshop", "uniqlo", "bose", "apple", "samsung", "xiaomi", 
    "huawei", "sony", "lg", "belk", "braintree", "shopify", "shopbop", "farfetch", "newegg", 
    "overstock", "jdsports", "finishline", "amazon prime", "alibaba", "mercado libre", "rakuten", 
    "lazada", "carrefour", "tokopedia", "snapdeal", "flipkart", "jd.com", "cdiscount", "auchan", 
    "groupon", "vinted", "poshmark", "bonprix", "trendyol", "lazada", "shopdirect", "thehut"
],
"Recreation, Entertainment, Vacation and Travels": [
    "concert", "vacation", "hobby", "streaming", "movies", "video games", "theme park", 
    "sports event", "museum", "entertainment subscription", "leisure", "magellan", 
    "Amazon Prime", "paypal .steam", "netflix", "cinema", "theater", "movie theater", 
    "film", "blockbuster", "premiere", "IMAX", "film festival", "opera", "ballet", 
    "comedy show", "stand-up comedy", "circus", "night club", "music festival", 
    "gaming console", "xbox", "playstation", "nintendo", "pc gaming", "board games", 
    "vr gaming", "escape room", "arcade", "e-sports", "fantasy sports", "sporting event", 
    "swimming pool", "skiing", "hiking", "rock climbing", "klettern", "fishing", "bowling", 
    "karaoke", "live show", "bingo", "poker", "casino", "roulette", "lottery", "ticketmaster", 
    "bowl", "tripadvisor", "spotify", "pandora", "hulu", "disney+", "apple tv", "hbo max", 
    "youtube premium", "twitch", "tiktok", "vlog", "podcast", "radio", "musical", "zoo", 
    "aquarium", "amusement park", "roller coaster", "skydiving", "paragliding", "fishing", 
    "travel blog", "adventure park", "outdoor adventure", "live music", "pop concert", 
    "classical concert", "live theater", "art gallery", "antique show", "cultural event", 
    "family fun", "kids' event", "childrens' activity", "boardwalk", "water park", "circus", 
    "golf", "mini golf", "bungee jumping", "yoga retreat", "meditation retreat", "Fremdwährung", 
    "Boulderhalle"
],
    "Miscellaneous": [
        "unexpected", "miscellaneous", "other", "donation", "charity", "gift",
        "cash withdrawal", "uncategorized", "random", "small purchase"
    ],
}