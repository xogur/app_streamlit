import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import webcolors

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




# 배열을 numpy로 변환
rgb_list = df[['r', 'g', 'b']].values.tolist()
rgb_array = np.array(rgb_list)

# KMeans 군집화 (예: 3개 계열)


# RGB를 가장 가까운 CSS 색상 이름으로 변환
def closest_color_name(rgb):
    try:
        return webcolors.rgb_to_name(tuple(rgb))
    except ValueError:
        # 정확한 이름이 없으면 가장 가까운 이름 추정
        min_dist = float('inf')
        closest_name = None
        for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
            dist = (rgb[0] - r_c)**2 + (rgb[1] - g_c)**2 + (rgb[2] - b_c)**2
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

# 색상 이름 리스트 생성
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(rgb_array)
centroids = kmeans.cluster_centers_.astype(int)
color_names = [closest_color_name(centroid) for centroid in centroids]

# 군집별 개수 계산
count = Counter(labels)

# 📊 결과 시각화
fig, ax = plt.subplots(figsize=(10, 4))
colors = [centroids[i]/255 for i in range(k)]
ax.bar(range(k), [count[i] for i in range(k)], color=colors)
ax.set_xticks(range(k))
ax.set_xticklabels(color_names, rotation=45, ha='right')
ax.set_ylabel('Number of Colors')
ax.set_title('Dominant Color Groups from RGB List')
plt.tight_layout()

# Streamlit에 출력
st.pyplot(fig)

# 시각화
for idx, row in df.iterrows():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(row['image_url'], width=100)
    with col2:
        st.markdown(f"**RGB:** ({row['r']}, {row['g']}, {row['b']})")
        st.markdown(f"`{row['image_url']}`")

    st.markdown("---")
