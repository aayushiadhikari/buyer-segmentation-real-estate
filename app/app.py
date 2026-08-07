import streamlit as st
import pandas as pd
import plotly.express as px

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/final_clustered_dataset.csv")

df = load_data()

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🏠 Buyer Segmentation")

st.sidebar.markdown("### Filters")

country = st.sidebar.multiselect(
    "🌍 Country",
    sorted(df["country"].dropna().unique()),
    default=sorted(df["country"].dropna().unique())
)

region = st.sidebar.multiselect(
    "📍 Region",
    sorted(df["region"].dropna().unique()),
    default=sorted(df["region"].dropna().unique())
)

client = st.sidebar.multiselect(
    "👤 Client Type",
    sorted(df["client_type"].dropna().unique()),
    default=sorted(df["client_type"].dropna().unique())
)

purpose = st.sidebar.multiselect(
    "🎯 Acquisition Purpose",
    sorted(df["acquisition_purpose"].dropna().unique()),
    default=sorted(df["acquisition_purpose"].dropna().unique())
)

loan = st.sidebar.multiselect(
    "🏦 Loan Applied",
    sorted(df["loan_applied"].dropna().unique()),
    default=sorted(df["loan_applied"].dropna().unique())
)

filtered = df[
    (df["country"].isin(country)) &
    (df["region"].isin(region)) &
    (df["client_type"].isin(client)) &
    (df["acquisition_purpose"].isin(purpose)) &
    (df["loan_applied"].isin(loan))
]

# ======================================================
# TITLE
# ======================================================

st.title("🏠 Machine Learning Based Buyer Segmentation & Investment Profiling")

st.markdown(
"""
AI-powered dashboard for analyzing buyer segments, investment behavior,
customer demographics and real estate market intelligence.
"""
)

st.markdown("---")

# ======================================================
# KPI SECTION
# ======================================================

loan_rate = filtered["loan_applied"].eq("Yes").mean() * 100

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric(
        "👥 Buyers",
        f"{len(filtered):,}"
    )

with c2:
    st.metric(
        "🏡 Properties",
        f"{int(filtered.total_properties.sum()):,}"
    )

with c3:
    st.metric(
        "💰 Investment",
        f"${filtered.total_investment.sum()/1_000_000:.2f} M"
    )

with c4:
    st.metric(
        "📐 Avg Floor Area",
        f"{filtered.average_floor_area.mean():.0f} sqft"
    )

with c5:
    st.metric(
        "⭐ Satisfaction",
        f"{filtered.satisfaction_score.mean():.2f}"
    )

with c6:
    st.metric(
        "🏦 Loan Applied",
        f"{loan_rate:.1f}%"
    )

st.markdown("---")

# =====================================================
# ROW 1 : BUYER OVERVIEW
# =====================================================

col1, col2 = st.columns(2)

with col1:

    buyer_df = (
        filtered["client_type"]
        .value_counts()
        .reset_index()
    )

    buyer_df.columns = ["Client Type", "Count"]

    fig = px.pie(
        buyer_df,
        names="Client Type",
        values="Count",
        hole=0.55,
        title="Buyer Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(
        textinfo="percent+label",
        pull=[0.02]*len(buyer_df)
    )

    fig.update_layout(
        height=450,
        legend_title="Client Type"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:

    purpose_df = (
        filtered.groupby("acquisition_purpose")
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        purpose_df,
        x="acquisition_purpose",
        y="Count",
        color="acquisition_purpose",
        text="Count",
        title="Acquisition Purpose",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False,
        height=450,
        xaxis_title="Purpose",
        yaxis_title="Number of Buyers"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.markdown("---")

# =====================================================
# ROW 2 : CUSTOMER ANALYTICS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered,
        x="age",
        color="client_type",
        nbins=25,
        marginal="box",
        title="Age Distribution",
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig.update_layout(
        height=450,
        xaxis_title="Age",
        yaxis_title="Number of Buyers"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:

    fig = px.box(
        filtered,
        x="Buyer_Segment",
        y="total_investment",
        color="Buyer_Segment",
        title="Investment Distribution by Segment",
        points="outliers",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        height=450,
        xaxis_title="Buyer Segment",
        yaxis_title="Investment ($)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.markdown("---")

# =====================================================
# ROW 3 : INVESTMENT ANALYTICS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        filtered,
        x="average_floor_area",
        y="total_investment",
        color="Buyer_Segment",
        hover_name="client_id",
        hover_data=[
            "country",
            "region",
            "client_type",
            "acquisition_purpose"
        ],
        opacity=0.75,
        render_mode="svg",          # <<< WebGL FIX
        title="Investment vs Floor Area",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_traces(
        marker=dict(
            size=11,
            line=dict(width=1, color="white")
        )
    )

    fig.update_layout(
        height=500,
        xaxis_title="Average Floor Area (sqft)",
        yaxis_title="Total Investment ($)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:

    country_df = (
        filtered
        .groupby("country")
        .agg(
            Total_Investment=("total_investment","sum")
        )
        .sort_values("Total_Investment",ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        country_df,
        x="country",
        y="Total_Investment",
        color="country",
        text_auto=".2s",
        title="Top 10 Countries by Investment",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        showlegend=False,
        height=500
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.markdown("---")

# =====================================================
# ROW 4 : REGIONAL ANALYTICS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    region_df = (
        filtered
        .groupby("region")
        .agg(
            Total_Investment=("total_investment","sum")
        )
        .sort_values("Total_Investment",ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        region_df,
        x="region",
        y="Total_Investment",
        color="region",
        text_auto=".2s",
        title="Top 10 Regions by Investment",
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig.update_layout(
        showlegend=False,
        height=500
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:

    fig = px.box(
        filtered,
        x="Buyer_Segment",
        y="satisfaction_score",
        color="Buyer_Segment",
        points="all",
        title="Customer Satisfaction by Buyer Segment",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        height=500,
        xaxis_title="Buyer Segment",
        yaxis_title="Satisfaction Score"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.markdown("---")

# =====================================================
# ROW 5 : BUYER SEGMENT SUMMARY
# =====================================================

st.subheader("📊 Buyer Segment Summary")

summary = (
    filtered
    .groupby("Buyer_Segment")
    .agg(
        Total_Buyers=("client_id","count"),
        Average_Age=("age","mean"),
        Average_Investment=("total_investment","mean"),
        Total_Investment=("total_investment","sum"),
        Average_Satisfaction=("satisfaction_score","mean"),
        Average_Properties=("total_properties","mean")
    )
    .round(2)
    .reset_index()
)

st.dataframe(
    summary,
    width="stretch",
    hide_index=True
)

st.markdown("---")

# =====================================================
# EXECUTIVE INSIGHTS
# =====================================================

st.subheader("🧠 Executive Insights")

top_country = (
    filtered.groupby("country")["total_investment"]
    .sum()
    .idxmax()
)

top_region = (
    filtered.groupby("region")["total_investment"]
    .sum()
    .idxmax()
)

top_segment = (
    filtered.groupby("Buyer_Segment")["total_investment"]
    .mean()
    .idxmax()
)

loan_rate = filtered["loan_applied"].eq("Yes").mean()*100

avg_age = filtered["age"].mean()

avg_sat = filtered["satisfaction_score"].mean()

st.info(f"""

### Key Business Findings

🌍 **Highest Investment Country:** **{top_country}**

📍 **Highest Investment Region:** **{top_region}**

💰 **Highest Value Buyer Segment:** **{top_segment}**

👥 **Average Buyer Age:** **{avg_age:.1f} Years**

⭐ **Average Customer Satisfaction:** **{avg_sat:.2f}/5**

🏦 **Loan Application Rate:** **{loan_rate:.1f}%**

""")

st.markdown("---")

# =====================================================
# BUYER SEGMENT COMPARISON
# =====================================================

st.subheader("📈 Buyer Segment Comparison")

comparison = (
    filtered
    .groupby("Buyer_Segment")
    .agg(
        Avg_Investment=("total_investment","mean"),
        Avg_Age=("age","mean")
    )
    .reset_index()
)

fig = px.bar(
    comparison,
    x="Buyer_Segment",
    y="Avg_Investment",
    color="Buyer_Segment",
    text_auto=".2s",
    title="Average Investment by Buyer Segment",
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_layout(
    showlegend=False,
    height=450
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.markdown("---")

# =====================================================
# DOWNLOAD DATA
# =====================================================

st.subheader("📥 Export Dataset")

csv = filtered.to_csv(index=False)

st.download_button(
    label="⬇ Download Filtered Dataset",
    data=csv,
    file_name="Buyer_Segmentation_Filtered_Data.csv",
    mime="text/csv"
)

st.markdown("---")

# =====================================================
# PROJECT INFORMATION
# =====================================================

st.subheader("📌 Project Details")

col1,col2 = st.columns(2)

with col1:

    st.success("""

### Machine Learning

✔ K-Means Clustering

✔ Hierarchical Clustering

✔ StandardScaler

✔ Feature Encoding

✔ Customer Segmentation

""")

with col2:

    st.success("""

### Business Objectives

✔ Buyer Profiling

✔ Investment Intelligence

✔ Customer Analytics

✔ Market Segmentation

✔ Real Estate Insights

""")

st.markdown("---")

# =====================================================
# FOOTER
# =====================================================

st.markdown(
"""
<div style='text-align:center'>

## 🏠 Buyer Segmentation & Investment Profiling

Machine Learning Powered Real Estate Market Intelligence Dashboard

**Developed by Aayushi Adhikari**

Unified Mentor Internship Project

</div>
""",
unsafe_allow_html=True
)