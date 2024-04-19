import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from datetime import datetime

def app(data):
    st.title('Customer Segmentation Analysis')
    # Your code from app2.py here, adjust as needed
    @st.cache_data  # updated caching mechanism
    def load_data():
        if data is None:
            st.warning("No data loaded. Please upload data using the sidebar.")
        if data is not None:
            # data = pd.read_csv('txn_history_dummysample.csv')
            data['date_of_birth'] = pd.to_datetime(data['date_of_birth'])
            data['txn_ts'] = pd.to_datetime(data['txn_ts'])
            data['age'] = datetime.now().year - data['date_of_birth'].dt.year

            # Aggregate data by account_id
            grouped_data = data.groupby('account_id').agg(
                total_transactions=('txn_amount', 'count'),
                average_transaction_size=('txn_amount', 'mean'),
                end_balance=('txn_amount', 'sum')
            ).reset_index()

            # Include age
            grouped_data['age'] = data.groupby('account_id')['age'].first()
            return grouped_data

    @st.cache_data  # added preprocessing step
    def preprocess_features(data):
        features = ['total_transactions', 'average_transaction_size', 'end_balance', 'age']
        # Imputing and scaling
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        data_imputed = imputer.fit_transform(data[features])
        data_scaled = scaler.fit_transform(data_imputed)
        return data_scaled

    data = load_data()
    data_scaled = preprocess_features(data)

    # Set up session state for cluster count
    if 'n_clusters' not in st.session_state:
        st.session_state['n_clusters'] = 4  # default number of clusters

    # Define function to perform clustering
    def perform_clustering(n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        return labels

    data['cluster'] = perform_clustering(st.session_state['n_clusters'])

    # Streamlit interface for visualization
    st.title('Customer Segmentation Analysis')
    st.write("Here are the clusters based on transaction behavior and demographics:")
    st.dataframe(data[['account_id', 'total_transactions', 'average_transaction_size', 'end_balance', 'age', 'cluster']])

    # Visualization
    st.subheader('Cluster Distribution')
    fig, ax = plt.subplots()
    data['cluster'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Customers')
    st.pyplot(fig)

    st.subheader('Scatter Plot of Clusters')
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['average_transaction_size'], data['end_balance'], c=data['cluster'], cmap='viridis', alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    ax.set_xlabel('Average Transaction Size')
    ax.set_ylabel('End Balance')
    ax.add_artist(legend1)
    st.pyplot(fig)

    # Slider to adjust the number of clusters
    n_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=st.session_state['n_clusters'])

    # Button to re-run clustering
    if st.button('Re-run Clustering'):
        st.session_state['n_clusters'] = n_clusters  # update the number of clusters in session state
        st.experimental_rerun()  # rerun the app with the updated state

    # Updates the cluster data based on the slider input without needing to press the button
    data['cluster'] = perform_clustering(n_clusters)

    st.header("Cluster Interpretations on exercise dataset")
    st.markdown("""
        ### Interpret Each Cluster

        Based on the average values of the features for each cluster, here are the possible interpretations:

        - **Premium Customers**: High average transaction size and high end balance.
          - Customers who make large transactions and maintain high balances.

        - **Occasional Users**: Low total transactions, low average transaction size, low end balance.
          - Customers, potentially new or low-engagement customers.

        - **Frequent Users**: High total transactions with moderate average transaction size.
          - These could be frequent users who use the bank for regular, possibly smaller transactions.

        - **Young Customers**: Younger age group with variable transaction sizes.
          - These might be younger customers with inconsistent transaction patterns, possibly reflecting lifestyle or early career stages.

        ### Develop Strategic Actions

        **Premium Customers**
        - **Marketing Strategy**: Develop high-value loyalty programs and premium services.
        - **Product Development**: Offer high investment yield products or exclusive services.
        - **Customer Retention**: Provide personalized communication and dedicated support.

        **Occasional Users**
        - **Engagement Strategy**: Create initiatives to increase banking activity through promotional offers.
        - **Onboarding Experience**: Simplify the onboarding process to increase usage and familiarity with bank services.

        **Frequent Users**
        - **Cross-Selling**: Target with cross-selling opportunities for credit cards or overdrafts.
        - **Feedback and Improvement**: Regularly collect feedback to improve services and ensure satisfaction.

        **Young Customers**
        - **Educational Content**: Offer financial planning and savings advice.
        - **Digital Tools and Apps**: Enhance digital tools and app functionalities that appeal to their tech-savvy nature.

        ### Continuous Improvement Plan

        - **Re-evaluate Clusters Periodically**: Ensure the clusters still make sense as customer behaviors change over time.
        - **Adjust Cluster Number and Features as Necessary**: Refine the segmentation to better fit the data and capture emerging patterns.
        - **Incorporate Additional Data**: Use more comprehensive data over time to refine customer understanding and improve the accuracy of targeting.

        """)