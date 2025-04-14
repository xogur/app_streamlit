# deproject/app/app.py
import streamlit as st
import psycopg2
import pandas as pd

st.title("🎨 Fashion Trend Color Dashboard")

conn = psycopg2.connect(
    host="postgres_custom",
    port=5432,
    user="xogur",
    password="xogur",
    dbname="deproject"
)

df = pd.read_sql("SELECT * FROM color_trend", conn)
st.dataframe(df)
conn.close()
