import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def app(data):
    st.title('Exploratory Data Analysis')
    # Your code from app.py here, adjust as needed
    def preprocess_data(data):
        current_year = datetime.now().year
        # convert data['date_of_birth'] to datetime
        data['date_of_birth'] = pd.to_datetime(data['date_of_birth'])
        data['age'] = current_year - data['date_of_birth'].dt.year
        data['txn_ts'] = pd.to_datetime(data['txn_ts'])
        return data

    
    # Streamlit interface
    st.title('Transactions Analysis Dashboard')

    
    if data is None:
        st.warning("No data loaded. Please upload data using the sidebar.")
    if data is not None:
        df = data
        data = preprocess_data(df)
        # Group data by account_id for demographic and transaction analysis
        grouped_data = data.groupby('account_id').agg(
            total_transactions=('txn_amount', 'count'),
            average_transaction_size=('txn_amount', 'mean'),
            end_balance=('txn_amount', 'sum')
        ).reset_index()

        # Separate debits and credits for calculating average sizes
        debits = data[data['txn_amount'] < 0]
        credits = data[data['txn_amount'] > 0]

        avg_debit = debits.groupby('account_id')['txn_amount'].mean().reset_index().rename(columns={'txn_amount': 'average_debit'})
        avg_credit = credits.groupby('account_id')['txn_amount'].mean().reset_index().rename(columns={'txn_amount': 'average_credit'})

        # Merging average debit and credit data back to the grouped data
        grouped_data = grouped_data.merge(avg_debit, on='account_id', how='left')
        grouped_data = grouped_data.merge(avg_credit, on='account_id', how='left')

        # Displaying the age distribution of account holders
        fig, ax = plt.subplots(figsize=(10, 5))
        age_distribution = data.groupby('account_id')['age'].first().value_counts().sort_index()
        ax.bar(age_distribution.index, age_distribution.values, color='skyblue')
        ax.set_xlabel('Age')
        ax.set_ylabel('Number of Account Holders')
        ax.set_title('Age Distribution of Account Holders')
        ax.grid(True)
        st.pyplot(fig)

        # Displaying the distribution of end balances and transaction counts
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # End Balances
        axs[0].hist(grouped_data['end_balance'], bins=50, color='lightgreen', log=True)
        axs[0].set_xlabel('End Balance')
        axs[0].set_ylabel('Frequency (log scale)')
        axs[0].set_title('Distribution of End Balances')
        # Total Transactions
        axs[1].hist(grouped_data['total_transactions'], bins=50, color='lightblue', log=True)
        axs[1].set_xlabel('Total Transactions per User')
        axs[1].set_ylabel('Frequency (log scale)')
        axs[1].set_title('Transactions per User Distribution')
        plt.tight_layout()
        st.pyplot(fig)

    st.header("Exploratory Data Analysis Findings on exercise dataset")
    st.markdown("""
        ### Demographic Information Analysis
        - The age distribution of account holders is visualized above. Key observations include:
            * The account holder ages range widely, which suggests a diverse customer base from very young users to elderly.
            * The median age of the account holders is around 23 years, indicating a younger customer demographic which is common in e-banking.
            * A significant number of account holders are in their mid-20s to mid-30s, a prime demographic for digital banking services.

        ### Average Balances: Approximate average balance per user based on transaction data:
        - The calculated end balances (assuming an initial balance of zero) show a wide range from negative values (indicating net debits) to very large positive values (indicating net credits).
        - The average balance is skewed towards higher values, suggesting a subset of users with significantly high transaction volumes or values.
        - Deeper analysis about the distribution of end balances could provide insights into the financial behavior of different customer segments and even forecast the liquidity needs of the bank.
        - One of the analysis above (customer segmentation) will be demo at next page.
                
        ### Transactions per User: Determine the number of transactions per user.
        - The number of transactions per user varies significantly, with a maximum of 723 transactions for a single user, and a minimum of just 1 transaction.
        - Most users engage in fewer than 38 transactions (75th percentile), indicating sporadic use by many customers.
                
        ### Average Transaction Sizes: Analyze the average size of debits and credits.
        - Debits and credits vary widely, with mean values indicating generally larger credit transactions compared to debits.
        - The averages are significantly affected by extreme values, suggesting that a few large transactions are skewing the averages.
        """)