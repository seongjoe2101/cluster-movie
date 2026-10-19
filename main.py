
import math

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==================================================
# 1. 페이지 설정
# ==================================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 특성을 바탕으로 비슷한 영화끼리 묶고, "
    "군집 수에 따라 결과가 어떻게 달라지는지 살펴봅니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)


# ==================================================
# 2. 데이터 불러오기
# ==================================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    return data


try:
    original_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# ==================================================
# 3. 데이터 정리 및 속성 만들기
# ==================================================
total_movies = len(original_df)
df = original_df.copy()

numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10"
    ]
).copy()

# 로그 및 롱런 지수 계산에 사용할 수 없는 행 제외
df = df[
    (df["first_scrn"] > 0)
    & (df["first_week_audi"] > 0)
    & (df["total_audi"] > 0)
    & (df["days_in_top10"] >= 0)
].copy()

# 상용로그
df["스크린 수(로그)"] = df["first_scrn"].apply(math.log10)
df["누적 관객(로그)"] = df["total_audi"].apply(math.log10)

# 원래 단위의 속성
df["10위권 일수"] = df["days_in_top10"]

# 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
df["롱런 지수"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

feature_columns = [
    "스크린 수(로그)",
    "누적 관객(로그)",
    "10위권 일수",
    "롱런 지수"
]

df = df.replace([float("inf"), float("-inf")], float("nan"))
df = df.dropna(subset=feature_columns).copy()
df = df.reset_index(drop=True)

st.markdown(
    f"**전체 영화: {total_movies:,}편　|　"
    f"분석 대상 영화: {len(df):,}편**"
)

st.caption(
    "필요한 값이 없거나 첫 주 관객이 0인 영화 등 "
    "계산할 수 없는 영화는 분석에서 제외합니다."
)

if len(df) < 7:
    st.error(
        "묶음 수 1~7의 평가 그래프를 만들려면 "
        "분석 대상 영화가 최소 7편 필요합니다."
    )
    st.stop()


# ==================================================
# 4. 분류 속성 및 묶음 수 선택
# ==================================================
st.subheader("1. 분류 설정")

selected_features = st.multiselect(
    "묶는 데 사용할 속성 (두 개 이상 선택)",
    options=feature_columns,
    default=feature_columns
)

if len(selected_features) < 2:
    st.warning("분류 속성을 두 개 이상 선택해 주세요.")
    st.stop()

selected_k = st.selectbox(
    "묶음 수",
    options=[2, 3, 4, 5, 6, 7],
    index=1,
    help="2개부터 7개까지 선택할 수 있습니다."
)

# 기호를 최대 7개까지 사용
cluster_symbols = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

# 선택한 속성으로 표준화
X = df[selected_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================================================
# 5. 현재 묶음 수로 K-평균 군집화
# ==================================================
@st.cache_data
def run_kmeans(values, k):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(values)
    return labels, model.inertia_


labels, current_wcss = run_kmeans(X_scaled, selected_k)

df["군집 번호"] = labels

# 누적 관객 평균이 높은 군집부터 기호 부여
cluster_order = (
    df.groupby("군집 번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: cluster_symbols[i]
    for i, cluster_id in enumerate(cluster_order)
}

df["영화 유형"] = df["군집 번호"].map(cluster_labels)

# 모든 그래프에서 동일한 색상 사용
color_map = {
    "㉮": "#E45756",
    "㉯": "#4C78A8",
    "㉰": "#54A24B",
    "㉱": "#F2CF5B",
    "㉲": "#B279A2",
    "㉳": "#FF9D98",
    "㉴": "#79706E"
}

symbol_order = cluster_symbols[:selected_k]


# ==================================================
# 6. 현재 군집 결과 요약
# ==================================================
st.subheader("2. 군집 결과 요약")

st.markdown(
    f"현재 **{selected_k}개 묶음**으로 분류했습니다. "
    f"사용 속성: **{', '.join(selected_features)}**"
)

st.caption(
    "기호는 각 군집의 평균 누적 관객이 많은 순서대로 "
    "㉮부터 부여합니다."
)


# ==================================================
# 7. 2차원 산점도
# ==================================================
st.subheader("3. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        key="scatter_x"
    )

with col_y:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        key="scatter_y"
    )

fig_2d = px.scatter(
    df,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    color_discrete_map=color_map,
    category_orders={"영화 유형": symbol_order},
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        "movieNm": False,
        "first_scrn": ":,.0f",
        "total_audi": ":,.0f",
        "days_in_top10": ":,.0f"
    },
    labels={
        x_feature: x_feature,
        y_feature: y_feature,
        "영화 유형": "영화 유형",
        "first_scrn": "스크린 수",
        "total_audi": "누적 관객",
        "days_in_top10": "10위권 일수"
    },
    title=f"{selected_k}개 묶음: {x_feature}와 {y_feature}"
)

fig_2d.update_traces(
    marker={"size": 8, "opacity": 0.8}
)

fig_2d.update_layout(
    height=550,
    legend_title_text="영화 유형"
)

st.plotly_chart(fig_2d, use_container_width=True)


# ==================================================
# 8. 3차원 산점도
# ==================================================
st.subheader("4. 3차원 산점도")

st.write(
    "그래프를 마우스로 드래그하면 시점을 회전할 수 있습니다. "
    "점에 마우스를 올리면 영화 제목이 표시됩니다."
)

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 분류 속성을 "
        "세 개 이상 선택해 주세요."
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
        category_orders={"영화 유형": symbol_order},
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            "first_scrn": ":,.0f",
            "total_audi": ":,.0f",
            "days_in_top10": ":,.0f"
        },
        labels={
            x3: x3,
            y3: y3,
            z3: z3,
            "영화 유형": "영화 유형",
            "first_scrn": "스크린 수",
            "total_audi": "누적 관객",
            "days_in_top10": "10위권 일수"
        },
        title=f"{selected_k}개 묶음의 3차원 분포"
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


# ==================================================
# 9. 군집별 평균 표
# ==================================================
st.subheader("5. 묶음별 편수와 속성 평균")

summary_rows = []

for symbol in symbol_order:
    group = df[df["영화 유형"] == symbol]

    summary_rows.append({
        "영화 유형": symbol,
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
    "평균 표에는 로그 변환 전의 원래 스크린 수와 누적 관객을 표시합니다."
)


# ==================================================
# 10. 묶음별 누적 관객 상위 5편
# ==================================================
st.subheader("6. 묶음별 누적 관객 상위 5편")

for symbol in symbol_order:
    group = df[df["영화 유형"] == symbol]

    top5 = group.sort_values(
        "total_audi",
        ascending=False
    ).head(5)

    with st.expander(
        f"{symbol} 유형 — 누적 관객 상위 영화",
        expanded=True
    ):
        if top5.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            display_df = top5[
                [
                    "movieNm",
                    "total_audi",
                    "first_week_audi",
                    "first_scrn",
                    "days_in_top10",
                    "롱런 지수"
                ]
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


# ==================================================
# 11. 묶음 수별 군집 내 제곱거리 합계(WCSS)
# ==================================================
st.subheader("7. 묶음 수 정하기: 꺾은선 그래프")

st.write(
    "묶음 수를 1~7로 바꾸면서 각 점과 자신이 속한 군집 중심 사이의 "
    "제곱거리를 모두 더합니다. 값이 작을수록 군집 안의 점들이 "
    "중심에 가깝습니다."
)

wcss_rows = []

for k in range(1, 8):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss_rows.append({
        "묶음 수": k,
        "군집 내 제곱거리 합계": model.inertia_
    })

wcss_df = pd.DataFrame(wcss_rows)

fig_elbow = px.line(
    wcss_df,
    x="묶음 수",
    y="군집 내 제곱거리 합계",
    markers=True,
    title="묶음 수에 따른 군집 내 제곱거리 합계",
    labels={
        "묶음 수": "묶음 수",
        "군집 내 제곱거리 합계": "군집 내 제곱거리 합계"
    }
)

# 현재 선택한 묶음 수에 세로선 표시
fig_elbow.add_vline(
    x=selected_k,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: {selected_k}",
    annotation_position="top"
)

fig_elbow.update_layout(
    height=450,
    xaxis={
        "tickmode": "linear",
        "dtick": 1,
        "range": [0.8, 7.2]
    },
    yaxis_title="군집 내 제곱거리 합계"
)

st.plotly_chart(fig_elbow, use_container_width=True)


# ==================================================
# 12. WCSS 값과 감소량 표
# ==================================================
st.subheader("8. 묶음 수별 제곱거리 합계와 감소량")

table_df = wcss_df.copy()

table_df["직전 값에서 감소한 양"] = (
    table_df["군집 내 제곱거리 합계"].shift(1)
    - table_df["군집 내 제곱거리 합계"]
)

# 첫 번째 행은 비교할 이전 값이 없으므로 빈칸
table_df["직전 값에서 감소한 양"] = (
    table_df["직전 값에서 감소한 양"].round(3).astype("object")
)
table_df.loc[0, "직전 값에서 감소한 양"] = ""

table_df["군집 내 제곱거리 합계"] = (
    table_df["군집 내 제곱거리 합계"].round(3)
)

table_df = table_df.rename(columns={
    "군집 내 제곱거리 합계": "제곱거리 합계",
    "직전 값에서 감소한 양": "직전 값 대비 감소량"
})

st.dataframe(
    table_df.style.format({
        "묶음 수": "{:.0f}",
        "제곱거리 합계": "{:,.3f}",
        "직전 값 대비 감소량": lambda x: (
            "" if x == "" else f"{float(x):,.3f}"
        )
    }),
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 13. 실루엣 점수
# ==================================================
st.subheader("9. 현재 묶음 수의 실루엣 점수")

current_silhouette = silhouette_score(
    X_scaled,
    labels
)

st.markdown(
    f"**현재 묶음 수: {selected_k}개　|　"
    f"실루엣 점수: {current_silhouette:.4f}**"
)

st.progress(
    max(0.0, min(1.0, (current_silhouette + 1) / 2))
)

st.caption(
    "실루엣 점수는 -1부터 1까지입니다. "
    "1에 가까울수록 같은 묶음 안의 영화는 서로 가깝고, "
    "다른 묶음의 영화와는 잘 구분되는 경향이 있습니다. "
    "0에 가까우면 묶음이 겹칠 수 있고, 음수이면 일부 영화가 "
    "다른 묶음에 더 가까울 수 있습니다."
)

st.divider()

st.caption(
    "K-평균은 선택한 속성을 표준화해 계산합니다. "
    "난수를 고정해 같은 데이터와 설정에서는 결과를 재현할 수 있습니다. "
    "묶음 수가 달라지면 군집 결과와 유형 기호가 다시 계산됩니다."
)
