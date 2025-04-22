import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
import requests
from requests.auth import HTTPBasicAuth

# 권태혁혁
st.set_page_config(layout="wide")
st.title("👗 Fashion Trend Dashboard")

st.sidebar.header("트렌드 분석석")

# 스타일 매핑
style_map = {
    "캐주얼": 8, "미니멀": 5, "스트릿": 10, "걸리시": 13, "스포티": 14,
    "워크웨어": 6, "로맨틱": 12, "시크": 16, "시티보이": 7, "고프코어": 20,
    "레트로": 18, "프레피": 17, "리조트": 15, "에스닉": 19
}
season_map = {"봄":4, "여름":3, "가을":2, "겨울":1}
gender_map = {"남": "MEN", "여": "WOMEN"}

selected_style  = st.sidebar.selectbox("스타일", list(style_map.keys()))
selected_season = st.sidebar.selectbox("계절", list(season_map.keys()))
selected_gender = st.sidebar.selectbox("성별", list(gender_map.keys()))

if st.sidebar.button("알림 설정 전송"):
    payload = {
        "conf": {
            "style":  style_map[selected_style],
            "season": season_map[selected_season],
            "gender": gender_map[selected_gender]
        }
    }
    try:
        resp = requests.post(
            "http://airflow-webserver:8080/api/v1/dags/dags_fashion_item_trend_load/dagRuns",
            auth=HTTPBasicAuth("airflow", "airflow"),
            json=payload,
            timeout=10
        )
        if resp.status_code in (200, 201):
            st.sidebar.success("✅ DAG 실행 요청 완료!")
        else:
            st.sidebar.error(f"❌ 실패: {resp.status_code} / {resp.text}")
    except Exception as e:
        st.sidebar.error(f"🚨 요청 실패: {e}")

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
        for name in webcolors.names("css3"):
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
    k = st.sidebar.slider("군집 수 선택 (k)", min_value=2, max_value=30, value=20)
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
    cols = st.columns(3)
    for idx, row in df_color.iterrows():
        with cols[idx % 3]:
            st.image(row['image_url'], width=150)
            st.markdown(f"**RGB:** ({row['r']}, {row['g']}, {row['b']})")
            st.markdown(f"`{row['image_url']}`")
