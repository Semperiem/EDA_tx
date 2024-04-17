import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data['date_of_birth'] = pd.to_datetime(data['date_of_birth'])
    data['txn_ts'] = pd.to_datetime(data['txn_ts'])
    return data

def preprocess_data(data):
    current_year = datetime.now().year
    data['age'] = current_year - data['date_of_birth'].dt.year
    return data

def generate_statistics(data):
    age_stats = data['age'].describe()
    txn_amount_stats = data['txn_amount'].describe()
    transactions_per_user = data.groupby('account_id').agg(
        total_transactions=pd.NamedAgg(column='txn_amount', aggfunc='count'),
        average_transaction_size=pd.NamedAgg(column='txn_amount', aggfunc='mean'),
        total_amount=pd.NamedAgg(column='txn_amount', aggfunc='sum')
    ).reset_index()
    return age_stats, txn_amount_stats, transactions_per_user

def plot_figures(data, transactions_per_user):
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    sns.histplot(data['age'], bins=30, ax=axes[0], color='skyblue')
    axes[0].set_title('Age Distribution of Account Holders')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')

    filtered_txn = data[data['txn_amount'].between(data['txn_amount'].quantile(0.01), data['txn_amount'].quantile(0.99))]
    sns.histplot(filtered_txn['txn_amount'], bins=50, ax=axes[1], color='lightgreen')
    axes[1].set_title('Transaction Amount Distribution (1st to 99th Percentile)')
    axes[1].set_xlabel('Transaction Amount')
    axes[1].set_ylabel('Frequency')

    sns.histplot(transactions_per_user['total_transactions'], bins=30, ax=axes[2], color='salmon')
    axes[2].set_title('Number of Transactions Per User')
    axes[2].set_xlabel('Total Transactions')
    axes[2].set_ylabel('Frequency')

    return fig

# Streamlit interface
st.title('Financial Transactions Analysis Dashboard')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = preprocess_data(data)
    age_stats, txn_amount_stats, transactions_per_user = generate_statistics(data)
    fig = plot_figures(data, transactions_per_user)

    st.write("### Age Statistics")
    st.write(age_stats)

    st.write("### Transaction Amount Statistics")
    st.write(txn_amount_stats)

    st.write("### Transactions Per User")
    st.write(transactions_per_user.head())  # You might want to display this differently

    st.write("### Visualizations")
    st.pyplot(fig)
