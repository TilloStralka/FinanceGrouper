# **FinanceGrouper**

### A tool for evaluating, grouping, and visualizing bank expenses and incomes.

---

Analyze CSV exports from your **DKB Bank Account** to generate a comprehensive **financial report** as a PDF.  
The tool identifies recurring expenses and uses a language model to assign them into specific categories (e.g., rent, groceries, entertainment, etc.).  

> **Note:** This is *NOT* an official tool by [Deutsche Kreditbank AG (DKB)](https://www.dkb.de/).  
> It is a local tool for training and personal use. All data remains private on your machine.
- The only exclusion to this is the translation via the open sourche tool of the python Translator which is translating ONLY! the Verwendungszweck in ENglisch because the pre trained categorizer is better with english 

![DKB Logo](https://upload.wikimedia.org/wikipedia/commons/d/d4/Deutsche_Kreditbank_AG_Logo_2016.svg)

---

## **Features**

- **Analyze your income and expenses** from CSV files.
- **View trends**: Generate visualizations (line charts, pie charts) for balance, income, and expenses.
- **Identify and optimize** your largest expenses.
- **Categorize transactions** into meaningful themes (e.g., rent, card payments, investments).
- **Privacy & Security**:
  - All computations run **locally** and **offline**.
  - No account access or third-party services are required.
* privacy: everything is computed locally and offline on our computer. no servers or companies you need to trust
* security: it simply uses the CSV you exported. it cannot access your account in any way.


2.  Installation
Clone the repository:

git clone https://github.com/TilloStralka/FinanceGrouper.git
cd FinanceGrouper
Create a virtual environment and activate it:

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install dependencies:

pip install -r requirements.txt
==============================



## **2. Project Structure**
------------
FinanceGrouper/
│
├── LICENSE
├── README.md             <- This top-level README file.
│
├── data                  <- Your raw and processed data (not included in Git).
│   ├── raw               <- Original, immutable data.
│   └── processed         <- Cleaned and processed data.
│
├── models                <- Trained models and summaries.
│
├── notebooks             <- Jupyter notebooks for analysis and visualizations.
│
├── reports               <- Generated reports.
│   └── figures           <- Plots and figures used in reporting.
│
├── requirements.txt      <- Python dependencies.
│
├── src                   <- Source code for this project.
│   ├── __init__.py       <- Makes src a Python module.
│   ├── features          <- Scripts for feature engineering.
│   ├── models            <- Scripts for model training and predictions.
│   └── visualization     <- Scripts to create visualizations.


==============================


### 3. Data 

How to Export CSV from DKB
Log in to your DKB online banking account.
Select the largest available time period (e.g., 3 years) for your main Girokonto.
Export the transactions as CSV files.
Save the files in the format <my-name_Year>.csv and place them into the data/raw folder.

4. Execution of the main script 

4.1 Execute finance tool.ipynb
it generates a summary and overview report of your financial situation over the course of the years you have bereit gestellt hast. 

Der bericht wird im folder reports gespeichert 


Die einzelenen funktionen können separat ausgeführt werden oder du cklickst einfach oben in vs code auf execute all 
---

5. Kategorisierung: 

die verwendeten categorien sind: 
1. Housing
Rent or mortgage payments, property taxes, HOA fees, home maintenance costs
2. Transportation
Car payments, registration and DMV fees, gas, maintenance, parking and tolls, public transit or ridesharing costs
3. Food
Groceries, restaurant meals, work lunches, food delivery
4. Utilities
Gas, electricity, water, sewage bills, internet and cell phone services
5. Insurance
Health insurance, homeowner's or renter's insurance, auto insurance, life insurance, disability insurance
6. Medical & Healthcare
Out-of-pocket costs for primary care, specialty care, dental care, urgent care, prescriptions and OTC medications, supplements and medical devices
7. Saving, Investing, & Debt Payments
Debt repayment (credit cards, loans), emergency fund, retirement savings (401(k), IRA)
8. Personal Spending
Gym memberships, clothes and shoes, haircuts and personal care, home decor and furniture, gifts
9. Recreation & Entertainment
Concert tickets, sporting events, family activities and vacations, streaming services and subscriptions, restaurants (if not included in Food), hobbies
10. Miscellaneous
Overflow category for unexpected expenses, additional spending in other categories when needed

Das verwendete Modell ist mit pytorch implementiert und aus der hugging face bibliothek. Es handelt sich um ein pre-trained sprach model von 
yiyanghkust/finbert-tone
Es ist ein Bert-Sprach Model basierend auf tokenisation. 
A) es wird direkt im code mit der bibliothek geladen (wie hier angewendet und verwendet)
B) Es kann bei bedarf manuel herunter geladen werden unter: 
https://huggingface.co/yiyanghkust/finbert-tone
C) Sie können doe Modell-Dateien herunterladen:
Lade die Dateien config.json, pytorch_model.bin, und tokenizer.json herunter.
D) Lokal verwenden:
Speichere die Dateien in einem Ordner, z. B. finbert-tone/.
Ändere den Pfad im Python-Code, um auf die Dateien zuzugreifen:
python
Copy code
tokenizer = AutoTokenizer.from_pretrained("path/to/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("path/to/finbert-tone")

## Annotation

The notebook may need updating, if the CSV format by DKB has changed.
Also if you are working with csv data coming from another bank the csv read function needs anpassungen accordingly. 