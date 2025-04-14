import streamlit as st
import psycopg2
import pandas as pd

st.set_page_config(layout="wide")
st.title("🎨 Fashion Trend Color Dashboard")

# PostgreSQL 연결
conn = psycopg2.connect(
    host="postgres_custom",
    port=5432,
    user="xogur",
    password="xogur",
    dbname="deproject"
)

# 데이터 불러오기
df = pd.read_sql("SELECT * FROM color_trend", conn)
conn.close()

# 시각화
for idx, row in df.iterrows():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(row['image'], width=100)
    with col2:
        st.markdown(f"**RGB:** ({row['r']}, {row['g']}, {row['b']})")
        st.markdown(f"`{row['image']}`")

    st.markdown("---")
