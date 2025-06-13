import sys
import os 
from pathlib import Path

# Add the src directory relative to the test file's location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# Import all functions from utils module to make them available for testing 
from utils_finance import *



def test_reduce_multiple_spaces_and_newlines():
    df = pd.DataFrame({
        'col1': ['  Hello   World  ', 'Test\nNewline', 'NoChange'],
        'col2': [123, 456, 789]  # numerical column should remain untouched
    })

    cleaned_df = reduce_multiple_spaces_and_newlines(df.copy())

    assert cleaned_df['col1'][0] == 'Hello World'
    assert cleaned_df['col1'][1] == 'Test Newline'
    assert cleaned_df['col1'][2] == 'NoChange'
    assert cleaned_df['col2'].equals(df['col2'])  # unchanged

def test_basic_analysis():
    df = pd.DataFrame({
        'Buchungsdatum': ['2023-01-01', '2023-01-15', '2023-02-01'],
        'Betrag (€)': [1000, -300, -100],
        'Zahlungspflichtige*r': ['Arbeitgeber', 'Ich', 'Ich'],
        'Zahlungsempfänger*in': ['Ich', 'Supermarkt', 'Miete']
    })

    result = basic_analysis(df)

    assert "Total Income: €1000.00" in result
    assert "Total Expenses: €-400.00" in result
    assert "Most frequent income source: Arbeitgeber" in result
    assert "Most frequent outgoing target: Supermarkt" in result or "Miete" in result


def test_truncate_strings_in_dataframe():
    df = pd.DataFrame({
        'text': ['a' * 50, 'some   text   with  spaces', 'short'],
        'number': [1, 2, 3]
    })

    truncated_df = truncate_strings_in_dataframe(df.copy(), max_length=10)

    assert truncated_df['text'][0] == 'aaaaaaaaaa'
    assert truncated_df['text'][1] == 'some text'
    assert truncated_df['text'][2] == 'short'
    assert truncated_df['number'].equals(df['number'])


def test_analyze_by_category():
    df = pd.DataFrame({
        'Umsatztyp': ['Lebensmittel', 'Miete', 'Lebensmittel', 'Transport'],
        'Betrag (€)': [-50, -700, -30, -20]
    })

    result = analyze_by_category(df)

    assert "Lebensmittel" in result
    assert "Miete" in result
    assert "Transport" in result
    assert "-700" in result or "-50" in result















if __name__ == "__main__":
    pytest.main()
