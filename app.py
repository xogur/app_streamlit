import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import webcolors

st.set_page_config(layout="wide")
st.title("👗 Fashion Trend Dashboard")

# 사이드바 메뉴
menu = st.sidebar.radio(
    "페이지 선택",
    ("🎨 색상 분석", "🧥 인기 상품 보기", "📸 전체 상품 이미지")
)

# PostgreSQL 연결
conn = psycopg2.connect(
    host="postgres_custom",
    port=5432,
    user="xogur",
    password="xogur",
    dbname="deproject"
)

df_color = pd.read_sql("SELECT * FROM color_trend", conn)
df_item = pd.read_sql("SELECT * FROM item_trend", conn)
conn.close()

# 👉 색상 이름 추정 함수
def closest_color_name(rgb):
    try:
        return webcolors.rgb_to_name(tuple(rgb))
    except ValueError:
        min_dist = float('inf')
        closest_name = None
        for name in webcolors.CSS3_NAMES_TO_HEX:
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            dist = sum((rgb[i] - [r_c, g_c, b_c][i])**2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

# ✅ 1. 색상 분석 페이지
if menu == "🎨 색상 분석":
    st.subheader("🎨 색상 군집 분석")
    rgb_array = df_color[['r', 'g', 'b']].values
    k = 20
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(rgb_array)
    centroids = kmeans.cluster_centers_.astype(int)
    color_names = [closest_color_name(centroid) for centroid in centroids]
    count = Counter(labels)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [centroids[i]/255 for i in range(k)]
    ax.bar(range(k), [count[i] for i in range(k)], color=colors)
    ax.set_xticks(range(k))
    ax.set_xticklabels(color_names, rotation=45, ha='right')
    ax.set_ylabel('Color Count')
    ax.set_title('Dominant Color Groups')
    st.pyplot(fig)

# ✅ 2. 인기 상품 보기
elif menu == "🧥 인기 상품 보기":
    st.subheader("🔥 상위 10개 인기 상품")

    top10 = df_item.sort_values(by='product_count', ascending=False).head(10).reset_index(drop=True)
    for idx, row in top10.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(row['image_url'], width=100)
        with col2:
            st.markdown(f"**상품명:** {row['product_name']}")
            st.progress(int(row['product_count']) / top10['product_count'].max())

# ✅ 3. 전체 이미지 보기
elif menu == "📸 전체 상품 이미지":
    st.subheader("📸 색상별 이미지 보기")
    for idx, row in df_color.iterrows():
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(row['image_url'], width=100)
        with col2:
            st.markdown(f"**RGB:** ({row['r']}, {row['g']}, {row['b']})")
            st.markdown(f"`{row['image_url']}`")
        st.markdown("---")
