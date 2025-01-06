# **FinanceGrouper**

### A tool for evaluating, grouping, and visualizing bank expenses and incomes.

---

Analyze CSV exports from your **DKB Bank Account** to generate a comprehensive **financial report** as a PDF.  
The tool identifies recurring expenses and uses a language model to assign them into specific categories (e.g., rent, groceries, entertainment, etc.).  

> **Note:** This is *NOT* an official tool by [Deutsche Kreditbank AG (DKB)](https://www.dkb.de/).  
> It is a local tool for training and personal use. All data remains private on your machine.
> Also the translation of the Verwendungszweck for the sorting tokenizer trained model is done locally with the pretrained translator module MarianMTModel 

![DKB Logo](https://upload.wikimedia.org/wikipedia/commons/d/d4/Deutsche_Kreditbank_AG_Logo_2016.svg)

---

## **1. Features**

- **Analyze your income and expenses** from CSV files.
- **View trends**: Generate visualizations (line charts, pie charts) for balance, income, and expenses.
- **Identify and optimize** your largest expenses.
- **Categorize transactions** into meaningful themes (e.g., rent, card payments, investments).
- **Privacy & Security**:
  - All computations run **locally** and **offline**.
  - No account access or third-party services are required.
* privacy: everything is computed locally and offline on our computer. no servers or companies you need to trust
* security: it simply uses the CSV you exported. it cannot access your account in any way.
* the translation of some words in the csv is also done locally an will not be sent to any API or online translator 

==============================

## **2.  Installation
Clone the repository:

1. Clone the repository:
    ```bash
    git clone https://github.com/TilloStralka/FinanceGrouper.git
    cd FinanceGrouper
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Annotation: 
For older mac versions it is recommended to use conda for the installation of the sentencepiece library, which is needed for the MarianMTModel translator.
bash 
conda install -c conda-forge sentencepiece


==============================


## **3. Project Structure**
------------

    ├── LICENSE
    ├── README.md                      <- This top-level README file.
    │
    ├── data                           <- Your raw and processed data 
    │                                     (In Git are dummy files included)
    │
    ├── notebooks             
    │   ├── FinanceTool.ipynb          <- Jupyter notebook for analysis and visualizations.
    │   │                                 (This contains the whole all in one process)
    │   └── Dummy_data_generator.ipynb <- This Jupyter notebooks is only for the dummy data generation / IBAN replacement    
    │
    ├── plots                          <- Here all the generated plots/visualizations are saved as pngs.
    │
    ├── reports                        <- Generated report which gives an financial overview.
    │                                     (It uses the plots in the folder above)
    │
    ├── requirements.txt               <- Python dependencies.
    │
    ├── src                            <- Source code for this project.
    │   ├── __init__.py                <- Makes src a Python module.
    │   └── utils_finance.py           <- This contains all the separate defined functions 
    │                                     which are deployed in the notebook.

==============================

## 4. Data 

### How to Export CSV from DKB
To export your financial data from DKB (Deutsche Bank), follow these steps:

1. **Log in to Your DKB Online Banking Account**: Start by accessing your DKB online banking account through the official website or mobile app.

2. **Select the Time Period**: Choose the largest available time period (e.g., 3 years) for your main Girokonto (current account). This ensures you capture a comprehensive dataset of your transactions.

3. **Export the Transactions as CSV Files**: After selecting the desired period, look for an option to export the transactions. Choose CSV as the export format. This will generate a file containing your transaction history.

4. **Save the Files**: Once exported, save the CSV files using a naming convention such as `<your-name_Year>.csv`, where you replace `<your-name>` with your name and `Year` with the year of the transactions. For example: `JohnDoe_2023.csv`.

5. **Place the Files in the Data Folder**: Move these CSV files into the `data/raw` folder of your project directory for easy access during data processing.

==============================

## 5. Execution of the Main Script 

### 5.1 Execute `finance_tool.ipynb`
To analyze your financial data, execute the `finance_tool.ipynb` script. This script will generate a summary and an overview report of your financial situation based on the data you’ve provided. It processes your transactions and provides insights into your financial trends over the years.

- The generated report will be saved in the `reports` folder of your project directory. You can view this report to get a clear understanding of your financial progress and analysis.

### Running the Functions Individually
You have the option to run individual functions separately, depending on the specific analysis you want to perform. Alternatively, you can execute all functions at once by clicking the **"Execute All"** button in VS Code.

This allows you to run the entire script in one go, generating all the required outputs without needing to manually trigger each step.

==============================

## 6. Categorization: 

The categories used are:  
1. **Housing**  
   Rent or mortgage payments, property taxes, HOA fees, home maintenance costs  
2. **Transportation**  
   Car payments, registration and DMV fees, gas, maintenance, parking and tolls, public transit or ridesharing costs  
3. **Food**  
   Groceries, restaurant meals, work lunches, food delivery  
4. **Utilities**  
   Gas, electricity, water, sewage bills, internet and cell phone services  
5. **Insurance**  
   Health insurance, homeowner's or renter's insurance, auto insurance, life insurance, disability insurance  
6. **Medical & Healthcare**  
   Out-of-pocket costs for primary care, specialty care, dental care, urgent care, prescriptions and OTC medications, supplements and medical devices  
7. **Saving, Investing, & Debt Payments**  
   Debt repayment (credit cards, loans), emergency fund, retirement savings (401(k), IRA)  
8. **Personal Spending**  
   Gym memberships, clothes and shoes, haircuts and personal care, home decor and furniture, gifts  
9. **Recreation & Entertainment**  
   Concert tickets, sporting events, family activities and vacations, streaming services and subscriptions, restaurants (if not included in Food), hobbies  
10. **Miscellaneous**  
    Overflow category for unexpected expenses, additional spending in other categories when needed  


==============================

## 7. Models:

### A) FinBERT (Pre-trained Financial BERT Model)
The model used for text classification is a pre-trained FinBERT model specifically designed for financial texts. It is based on the BERT architecture and was trained on a dataset of financial news to detect sentiments (positive, negative, or neutral) in texts. The model was loaded from the Hugging Face library and integrated with the PyTorch `transformers` library.

#### Details on Using the Model:
- **Model Name**: `yiyanghkust/finbert-tone`
- **Type**: BERT model for sentiment analysis of financial texts.
- **Implementation**: The model is loaded directly in the code using the Hugging Face library, as shown in the following example:

    ```python
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    ```

#### Manual Download and Local Usage:
If needed, you can manually download the model:
- **Model Download**: [yiyanghkust/finbert-tone on Hugging Face](https://huggingface.co/yiyanghkust/finbert-tone)
- **Required Files**:
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer.json`

Download these files and store them in a local folder, e.g., `finbert-tone/`. Then, update the code to reference the local path:

```python
# Local usage of the model
tokenizer = AutoTokenizer.from_pretrained("path/to/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("path/to/finbert-tone")
```

### B) MarianMT (Machine Translation Model)
Additionally, a MarianMT translation model is used to translate texts from German to English. This is particularly useful for translating financial transaction descriptions, which are in German, into a language (English) suitable for further processing. The model is based on the MarianMT architecture and is used through the Hugging Face library.

#### Details on Using the Model:
- **Model Name**: `Helsinki-NLP/opus-mt-de-en` (German → English)
- **Type**: Machine translation model

#### Manual Download and Local Usage:
If needed, you can manually download the model:
- **Model Download**: [Helsinki-NLP/opus-mt-de-en on Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-de-en)
- **Required Files**:
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer.json`

Download these files and store them in a local folder, e.g., `opus-mt-de-en/`. Then, update the code to reference the local path:
```python
# Local usage of the translation model
tokenizer = MarianTokenizer.from_pretrained("path/to/opus-mt-de-en")
model = MarianMTModel.from_pretrained("path/to/opus-mt-de-en")
```

==============================

## 8. Annotation:

The notebook may need updating, if the CSV format by DKB has changed.
Also if you are working with csv data coming from another bank the csv read function needs anpassungen accordingly. 