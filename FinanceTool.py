#!/usr/bin/env python
# coding: utf-8

# # A. Loading the required libraries 
# - importing standard python libraries 
# - importing self written utils functions 
# - declaring paths for finding data and saving text and images 


# Import the required libraries

# Data manipulation and math operations
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt  # For Matplotlib visualizations
plt.rcParams['text.usetex'] = False
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns            # For Seaborn visualizations
import plotly.express as px      # For interactive visualizations with Plotly
# %matplotlib inline  # Uncomment for Jupyter notebooks to display plots inline


#Libraries to make/construct the pdf for the final overview
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image

# System libraries
import warnings
import sys
import os
import inspect
from collections import Counter
# Ignore all warnings (useful for ignoring non-critical warnings)
warnings.filterwarnings("ignore")

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join('src')))

# Import utils library with custom functions
# Annotation: If there is a change in the utils function collect this window has to be executed twice 
# In order to get the utils functions loaded to the kernel to make the updated versions available for the notebook 
from utils_finance import *
import utils_finance
import importlib
importlib.reload(utils_finance)

# This library is for an offline translater which is required for the pretrained classifier model
# The model was trained for english language but the script is made for german financial data 
from translate import Translator

# This downloads a pretrained large language machine learning model from hugging face library 
# The download process will be visible below 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# This download will use the marianMTmodel for translation
from transformers import MarianMTModel, MarianTokenizer



# Path to the neighboring 'plots' folder in the local repository
plots_path = os.path.abspath(os.path.join(os.getcwd(), 'plots'))
print(plots_path)


# Path to the neighboring 'data' folder in the local repository
data_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
print(data_path)

# Path to the neighboring 'reports' folder in the local repository where the conclusive report will be saved as a pdf 
report_path = os.path.abspath(os.path.join(os.getcwd(), 'reports'))
print(report_path)


# # B. Data management
# + finding the data 
# + loading all available years  
# + concatenate all years of the data
# + saving the all years df  

# Retrieving data

# CSV files from DKB have a cryptic header; skip the first rows to make it readable
headerSize = 4

# List to store each DataFrame for later concatenation
dataframes = []

# Iterate through all files in the Data directory
for filename in os.listdir(data_path):
    # Skip "Data_All_Years.csv" if it already exists
    if filename == "Data_All_Years.csv":
        continue
    # Skip the training data for the ML model
    if filename == "Training_Data.csv":
        continue
    # Skip the finalized data from the runs before
    if filename == "Data_Treated.csv":
        continue
     
    # Check if the file is a .csv file
    if filename.endswith('.csv'):
        # Create the full path to the file
        file_path = os.path.join(data_path, filename)

        # Extract the last 4 digits of the filename (assuming they represent the year)
        last_four_digits = filename[-8:-4]
        
        # Make the year variable
        year = int(last_four_digits)

        # Read the .csv file into a DataFrame
        # print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path, sep=',', skiprows=headerSize, header=0, encoding='utf-8-sig')

        # Save the DataFrame under a dynamic name
        globals()[f'df_{last_four_digits}'] = df

        # Append the DataFrame to the list for concatenation
        dataframes.append(df)

        # Display the shape and column names of the DataFrame
        # print('Shape:', df.shape)
        # print('Column names:', df.columns)

# Concatenate all data to a single DataFrame
df_all_years = pd.concat(dataframes, ignore_index=True)

# Save the concatenated DataFrame to "Data_All_Years.csv"
output_path = os.path.join(data_path, "Data_All_Years.csv")
df_all_years.to_csv(output_path, index=False, sep=';')
print(f"Concatenated DataFrame saved to: {output_path}")


df = df_all_years


inspect_data(df)


# # C. First cleaning of data set
# + deleting empty rows and columns 
# + fixing Euro ',' and '.' problem in numbers 
# + changing type of columns to numerical (floats) and types (object)
# + filter all entrances which have very long empty entrys and make them to only one ' '


### Deleting empty columns and rows for each df before saving it in a global variable 
# Identify columns that contain only NaN values
nan_columns = df.columns[df.isna().all()].tolist()    
# Drop the columns with only NaN values
df.drop(columns=nan_columns, inplace=True)
# Delete all rows that are completely empty (i.e., contain only NaN values)
empty_rows = df[df.isna().all(axis=1)].index
df.drop(index=empty_rows, inplace=True)


# Set type of data of the columns 


# Fix US / EU decimal-point/comma
df['Betrag (€)'] = df['Betrag (€)'].str.replace('.','')
df['Betrag (€)'] = df['Betrag (€)'].str.replace(',','.')



# All columns will be set accordingly as dtype to what sort of stuff they contain 
df['Buchungsdatum'] = pd.to_datetime(df['Buchungsdatum'], format='%d.%m.%y', errors='coerce')
df['Wertstellung'] = pd.to_datetime(df['Wertstellung'], format='%d.%m.%y', errors='coerce')
df['Status'] = df['Status'].astype('category')

# Make sure euro is set to float
df['Betrag (€)'] = df['Betrag (€)'].astype(float)

# List of columns which should be set and transformed to strings 
string_columns = [
    'Zahlungspflichtige*r',
    'Zahlungsempfänger*in',
    'Verwendungszweck',
    'IBAN',
    'Gläubiger-ID',
    'Mandatsreferenz',
    'Kundenreferenz'
]

for col in string_columns:
    # Zwingend sicherstellen, dass alle Werte entweder str oder pd.NA sind
    df[col] = df[col].apply(lambda x: x if isinstance(x, str) else pd.NA)
    # Convert the column to typ 'string'
    df[col] = df[col].astype('string')


# Function to replace multiple spaces and new lines with a single space
df = reduce_multiple_spaces_and_newlines(df)


# # D. Categorizing 
# + translation of the "verwendungszweck" column into english due to pre trained model 
# + categorizing of the incomes and expanses 
# + making separate category column 
# + shorting strings in df which are too long (informations are not needed anymore)



# ## D. 1 Translate
# This code translates the "Verwendungszweck" (Purpose) column into English for various descriptions 
# of account transactions. 
# This is necessary because the pre-trained financial organizer operates in English.
# If the account transactions are already recorded in English, this function becomes redundant 
# and can be commented out.

# Load the model and tokenizer for German → English translation
model_name = "Helsinki-NLP/opus-mt-de-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


# Count the frequency of values in the "Verwendungszweck" column
value_counts = Counter(df["Verwendungszweck"])

# Print only the values that appear more than once
print("Values appearing more than once in 'Verwendungszweck':")
for value, count in value_counts.items():
    if count > 1:
        print(f"{value}: {count} times")

# Extract unique values from the "Verwendungszweck" column
unique_values = df["Verwendungszweck"].unique()

# Translate unique values and store them in a dictionary
translations = {value: translate_text(value) for value in unique_values}

# Add the translated values as a new column in the DataFrame
df["Description_English"] = df["Verwendungszweck"].map(translations)

# Print the number of unique values with duplicates
duplicate_count = sum(1 for count in value_counts.values() if count > 1)
print(f"Number of unique values with duplicates in 'Verwendungszweck': {duplicate_count}")


df.head()


# ## D. 2 Define categories & load model

# Load FinBERT-tone model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# The categories with their corresponding key words are writen down and saved in the utils_finance document of 
# the repository. See src folder -> utils_finance.py 


# ## D. 3 Apply categorizer and clean up df 

# Ensure that the 'Verwendungszweck' column only contains strings
df["Verwendungszweck"] = df["Verwendungszweck"].astype(str)
# Optionally, you can replace any "None" or other non-string values with an empty string or some other placeholder
df["Verwendungszweck"] = df["Verwendungszweck"].fillna('')

# Apply categorization to the DataFrame based on various columns
df["Category1"] = df["Description_English"].apply(categorize_transaction)
df["Category2"] = df["Zahlungspflichtige*r"].apply(categorize_transaction)
df["Category3"] = df["Zahlungsempfänger*in"].apply(categorize_transaction)
df["Category4"] = df["Verwendungszweck"].apply(categorize_transaction)

# Determine the final category based on the four intermediate categories
df['Category'] = df.apply(determine_final_category, axis=1)

# Remove the intermediate category columns to clean up the DataFrame
df = df.drop(columns=['Category1', 'Category2', 'Category3', 'Category4'])
# Display the first 20 rows of the updated DataFrame
df.head()

# Function to shorten all strings in the df to below 30 letters
# Default is set to 30 in the function, if none is given 
df = truncate_strings_in_dataframe(df, max_length=30)


# # E. Analysis of the financial data

# ## E. 1 Basic analysis of the data 
# + Calculate total income and expenses.
# + Determine monthly savings (income minus expenses).
# + Identify the largest expenses and sources of income.
# Performs basic financial analysis including total income, expenses, and largest transactions
text_1 = basic_analysis(df)
# print(text_1)


# ## E. 2 Income and Expense Categorization
# + Categorize transactions by Umsatztyp (e.g., groceries, rent, salary) to provide a breakdown of expenses and income per category. 
# + This helps understand which categories contribute the most to expenses and income.
# Group by 'Umsatztyp' and calculate total income and expenses per category
text_2 = analyze_by_category(df)
# print(text_2)


# ## E. 3 Income vs Expense Trend Over Time 
# + Create a time series analysis to visualize income and expenses trends over time (e.g., monthly, quarterly).
# Creates a time series analysis of income and expenses over time
income_expense_trend(df, plots_path)


# ## E. 4 Average Monthly Income/Expense
# Calculate the average monthly income and expense to get an overall view of typical monthly financial activity.
# Calculates average monthly income and expenses.
text3 = average_monthly_stats(df)
# print(text3)


# ## E. 5 Recurring Transactions Detection
# Identify recurring transactions such as regular payments (e.g., rent, subscriptions) or income (e.g., salary) based on similar amounts and descriptions.
# Identifies recurring transactions based on recipient and amount
recurring_summary = detect_recurring_transactions(df)
# Function which extracts the text info from the pandas df for the summary pdf later on
text4 = text_recurring_transactions(recurring_summary)
# Call the function and display the first few rows of the recurring transactions
recurring_summary.head()
# Visualizes the top recurring transactions in a bar plot
plot_recurring_summary(recurring_summary, plots_path)


# ## E. 6 Monthly Expense to Income Ratio
# Calculate the expense-to-income ratio to determine how much of your income is spent each month.
# Calculates the expense-to-income ratio per month
text5 = expense_income_ratio(df)


# ## E. 7 Expense Breakdown by Vendor/Recipient
# Provide a breakdown of total expenses by Zahlungsempfänger*in to identify who you are spending the most money on.
# Provides a breakdown of expenses by recipient
text6 = expenses_by_recipient(df)


# ## E. 8 Outlier Detection
# Identify unusually large transactions by detecting outliers (e.g., large expenses that are significantly different from the average).
# Identifies unusual transactions using standard deviation
text7 = detect_outliers(df)


# ## E. 9 Savings Percentage
# Calculate the percentage of income saved each month.
# Calculates and visualizes monthly savings percentage
savings_percentage(df, plots_path)


# ## E. 10 Cash Flow Stability
# Calculate cash flow volatility to measure the consistency of your monthly income and expenses.
# Calculates and visualizes cash flow stability over time
cash_flow_volatility(df, plots_path)


# ## E. 11 Income Sources Breakdown
# Break down the sources of income (e.g., salary, refunds) by Zahlungspflichtige*r and analyze which contribute the most.
# Analyzes and visualizes income sources with bar plots
income_by_source(df, plots_path)


# ## E. 12 Income VS. Expenses
# Plot total income vs expenses
plot_total_income_expenses(df, plots_path)
# Plot monthly income vs expenses
plot_monthly_income_vs_expenses(df, plots_path)


# ## E. 13 Savings and expanses distribution
# Savings and expanses plot execution
plot_monthly_savings(df, plots_path)
# Plot the larges income and expanses 
plot_largest_expenses_income(df, plots_path)
# Plot distribution of transactions amounts
plot_transaction_distribution(df, plots_path)    




# ## E. 14 Category comparisson
# Comparisson of expenses by category - absolut numbers 
saved_path = plot_expenses_barplot(df, plots_path)
# Comparisson of expenses by category - relative
saved_path = plot_expenses_barplot_percent(df, plots_path)
# Distribution of expenses by category in a violin plot
saved_path = plot_expenses_violin(df, plots_path)


# # F. Writing all the text output and images into a pdf for an overview sheet
# - iterating through all functions seen above 
# - iterating through all the plots which are saved in the plots folder 
# - write it all down in a pdf and save it in reports 



# Main programm for the pdf generation
# Separate functions will be called and the output (text and plots)
# will be placed in the summary pdf 
if __name__ == "__main__":
    plot_functions_1 = [
        ("Largest Expenses vs Income", lambda: plot_largest_expenses_income(df, plots_path)),
        ("Total Income vs Expenses", lambda: plot_total_income_expenses(df, plots_path)),
        ("Income by Source", lambda: income_by_source(df, plots_path)),
        ("Expenses Barplot", lambda: plot_expenses_barplot(df, plots_path)),
        ("Expenses Violin Plot", lambda: plot_expenses_violin(df, plots_path)),
        ("Expenses Barplot Percent", lambda: plot_expenses_barplot_percent(df, plots_path)),
        ("Monthly Income vs Expenses", lambda: plot_monthly_income_vs_expenses(df, plots_path)),

    ]
    text_functions_1 = [
        ("Basic Analysis", lambda: basic_analysis(df)),
        ("Average Monthly Stats", lambda: average_monthly_stats(df)),
        ("Analyze by Category", lambda: analyze_by_category(df)),
        ("Recurring Transactions Summary", lambda: text_recurring_transactions(recurring_summary)),
        #("Expense to Income Ratio", lambda: expense_income_ratio(df)),
    ]
    plot_functions_2 = [
        ("Cash Flow Volatility", lambda: cash_flow_volatility(df, plots_path)),
        ("Income Expense Trend", lambda: income_expense_trend(df, plots_path)),
        ("Recurring Summary", lambda: plot_recurring_summary(recurring_summary, plots_path)),
        ("Savings Percentage", lambda: savings_percentage(df, plots_path)),
        ("Monthly Savings", lambda: plot_monthly_savings(df, plots_path)),
        ("Transaction Distribution", lambda: plot_transaction_distribution(df, plots_path)),
    ]
    text_functions_2 = [
        #("Detect Outliers", lambda: detect_outliers(df)),
        #("Expenses by Recipient", lambda: expenses_by_recipient(df))
    ]    

    # Combine the functions from both sets
    plot_functions = plot_functions_1 + plot_functions_2
    text_functions = text_functions_1 + text_functions_2

    output_pdf_path = os.path.join(report_path, "financial_report.pdf")
    save_to_pdf(text_functions, plot_functions, output_pdf_path)


# # G. Writing treated data frame to file 
# Save the treated and evaluated DataFrame to "Data_Treated.csv"
output_path = os.path.join(data_path, "Data_Treated.csv")
df.to_csv(output_path, index=False, sep=';')
print(f"Finalized DataFrame saved to: {output_path}")

