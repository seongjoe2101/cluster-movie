
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write("영화의 특성을 분석해 비슷한 영화끼리 세 가지 유형으로 묶습니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    return df


try:
    original_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

total_movies = len(original_df)

# --------------------------------------------------
# 숫자형 변환 및 속성 생성
# --------------------------------------------------
df = original_df.copy()

numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 첫 주 관객이 0이거나 필요한 값이 없는 영화 제외
df = df.dropna(
    subset=[
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]
).copy()

df = df[
    (df["first_week_audi"] > 0)
    & (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["days_in_top10"] >= 0)
].copy()

# 상용로그(log10) 적용
df["스크린 수(로그)"] = df["first_scrn"].apply(
    lambda x: __import__("math").log10(x)
)
df["누적 관객(로그)"] = df["total_audi"].apply(
    lambda x: __import__("math").log10(x)
)

# 10위권 일수는 원래 값 사용
df["10위권 일수"] = df["days_in_top10"]

# 롱런 지수 = 누적 관객 / 첫 주 관객, 최댓값 20
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

# 원래 단위로 표시할 통계 열
original_feature_names = {
    "스크린 수(로그)": "스크린 수",
    "누적 관객(로그)": "누적 관객",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수"
}

feature_columns = list(original_feature_names.keys())

# 계산 과정에서 생길 수 있는 비정상 값 제거
df = df.replace([float("inf"), float("-inf")], float("nan"))
df = df.dropna(subset=feature_columns).copy()

# --------------------------------------------------
# 데이터 개수 표시
# --------------------------------------------------
st.markdown(
    f"**전체 영화: {total_movies:,}편　|　"
    f"분석 대상 영화: {len(df):,}편**"
)

st.caption(
    "필요한 값이 없거나 첫 주 관객이 0인 영화, "
    "로그를 계산할 수 없는 영화는 분석 대상에서 제외합니다."
)

if len(df) < 3:
    st.error("세 개의 군집으로 나누려면 분석 대상 영화가 최소 3편 필요합니다.")
    st.stop()

# --------------------------------------------------
# 분류 속성 선택
# --------------------------------------------------
st.subheader("1. 영화 유형을 나눌 속성 선택")

selected_features = st.multiselect(
    "분류에 사용할 속성을 두 개 이상 선택하세요.",
    options=feature_columns,
    default=feature_columns
)

if len(selected_features) < 2:
    st.warning("분류 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# --------------------------------------------------
# 표준화 및 K-평균 군집화
# --------------------------------------------------
X = df[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["군집 번호"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("군집 번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_order[0]: "㉮",
    cluster_order[1]: "㉯",
    cluster_order[2]: "㉰"
}

df["영화 유형"] = df["군집 번호"].map(cluster_labels)

# 모든 그래프에서 같은 유형의 색을 유지
color_map = {
    "㉮": "#E45756",
    "㉯": "#4C78A8",
    "㉰": "#54A24B"
}

# --------------------------------------------------
# 2차원 산점도
# --------------------------------------------------
st.subheader("2. 2차원 산점도")
st.write("가로축과 세로축의 속성을 바꾸며 영화 유형을 비교해 보세요.")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        key="scatter_x"
    )

with col_y:
    default_y = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=default_y,
        key="scatter_y"
    )

plot_df = df.copy()

fig_2d = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    color_discrete_map=color_map,
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        "movieNm": False,
        "first_scrn": ":,.0f",
        "total_audi": ":,.0f",
        "days_in_top10": ":,.0f"
    },
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    labels={
        x_feature: x_feature,
        y_feature: y_feature,
        "영화 유형": "영화 유형",
        "first_scrn": "스크린 수",
        "total_audi": "누적 관객",
        "days_in_top10": "10위권 일수"
    },
    title=f"{x_feature}와 {y_feature}에 따른 영화 유형"
)

fig_2d.update_traces(marker={"size": 8, "opacity": 0.8})
fig_2d.update_layout(
    legend_title_text="영화 유형",
    height=550
)

st.plotly_chart(fig_2d, use_container_width=True)

# --------------------------------------------------
# 3차원 산점도
# --------------------------------------------------
st.subheader("3. 3차원 산점도")
st.write(
    "마우스로 그래프를 드래그하면 시점을 회전할 수 있습니다. "
    "점 위에 마우스를 올리면 영화 제목을 확인할 수 있습니다."
)

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 분류 속성을 세 개 이상 선택해 주세요."
    )
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축 속성",
            selected_features,
            index=0,
            key="x3"
        )

    with c2:
        y3 = st.selectbox(
            "Y축 속성",
            selected_features,
            index=1,
            key="y3"
        )

    with c3:
        z3 = st.selectbox(
            "Z축 속성",
            selected_features,
            index=2,
            key="z3"
        )

    fig_3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        color_discrete_map=color_map,
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            "first_scrn": ":,.0f",
            "total_audi": ":,.0f",
            "days_in_top10": ":,.0f"
        },
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        labels={
            x3: x3,
            y3: y3,
            z3: z3,
            "영화 유형": "영화 유형",
            "first_scrn": "스크린 수",
            "total_audi": "누적 관객",
            "days_in_top10": "10위권 일수"
        },
        title="영화 유형의 3차원 분포"
    )

    fig_3d.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )

    fig_3d.update_layout(
        height=650,
        legend_title_text="영화 유형",
        scene={
            "xaxis_title": x3,
            "yaxis_title": y3,
            "zaxis_title": z3
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 60}
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------------------------------
# 군집별 평균 통계표
# --------------------------------------------------
st.subheader("4. 영화 유형별 평균과 편수")

summary_rows = []

for label in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == label]

    summary_rows.append({
        "영화 유형": label,
        "영화 편수": len(group),
        "평균 스크린 수": group["first_scrn"].mean(),
        "평균 누적 관객": group["total_audi"].mean(),
        "평균 10위권 일수": group["days_in_top10"].mean(),
        "평균 롱런 지수": group["롱런 지수"].mean()
    })

summary = pd.DataFrame(summary_rows)

st.dataframe(
    summary.style.format({
        "영화 편수": "{:,.0f}",
        "평균 스크린 수": "{:,.1f}",
        "평균 누적 관객": "{:,.0f}",
        "평균 10위권 일수": "{:,.1f}",
        "평균 롱런 지수": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "분류에는 로그 변환된 스크린 수와 누적 관객을 사용하지만, "
    "위 표에는 원래 단위의 평균을 표시합니다."
)

# --------------------------------------------------
# 군집별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("5. 영화 유형별 누적 관객 상위 5편")

for label in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == label].copy()
    top5 = group.sort_values(
        "total_audi", ascending=False
    ).head(5)

    with st.expander(f"{label} 유형 — 누적 관객 상위 영화", expanded=True):
        if top5.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            display_df = top5[
                ["movieNm", "total_audi", "first_week_audi",
                 "first_scrn", "days_in_top10", "롱런 지수"]
            ].rename(columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객",
                "first_week_audi": "첫 주 관객",
                "first_scrn": "스크린 수",
                "days_in_top10": "10위권 일수"
            })

            st.dataframe(
                display_df.style.format({
                    "누적 관객": "{:,.0f}",
                    "첫 주 관객": "{:,.0f}",
                    "스크린 수": "{:,.0f}",
                    "10위권 일수": "{:,.0f}",
                    "롱런 지수": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

st.divider()
st.caption(
    "영화 유형은 선택한 속성에 대한 K-평균 군집화 결과입니다. "
    "유형 번호는 정답이나 영화의 품질 순위를 뜻하지 않습니다."
)
