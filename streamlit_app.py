import streamlit as st
import pandas as pd
import plotly.express as px

url = 'https://raw.githubusercontent.com/michalis0/MGT-502-Data-Science-and-Machine-Learning/main/data/yield_df.csv'
df_yield= pd.read_csv(url)

# Assuming df_yield is your DataFrame
data = df_yield['Item'].value_counts().reset_index()
data.columns = ['Item', 'count']  # Rename columns for clarity

# Streamlit app
st.title('Shares of crops')

# Create a pie chart using Plotly Express
fig = px.pie(data, names='Item', values='count', title='Shares of crops')

# Display the pie chart using Streamlit
st.plotly_chart(fig)