import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

# ------------------------------
# 1. Load Data
# ------------------------------
facebook_df = pd.read_csv('data/Facebook.csv')
google_df = pd.read_csv('data/Google.csv')
tiktok_df = pd.read_csv('data/TikTok.csv')
business_df = pd.read_csv('data/Business.csv')

marketing_df = pd.concat([facebook_df, google_df, tiktok_df], ignore_index=True)
marketing_df['date'] = pd.to_datetime(marketing_df['date'])
business_df['date'] = pd.to_datetime(business_df['date'])

df = marketing_df.merge(business_df, on='date', how='left')

# ------------------------------
# 2. Calculate Metrics Function
# ------------------------------
def calculate_metrics(df):
    df['CTR'] = df['clicks'] / df['impression'].replace(0,1)
    df['CPC'] = df['spend'] / df['clicks'].replace(0,1)
    df['ROAS'] = df['attributed revenue'] / df['spend'].replace(0,1)
    df['Conversion Rate'] = df['# of orders'] / df['clicks'].replace(0,1)
    return df

df = calculate_metrics(df)

# ------------------------------
# 3. Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Marketing Dashboard", layout="wide")
st.title("📊 Marketing Intelligence Dashboard")
st.markdown("Analyze marketing campaigns and their impact on business outcomes.")

# ------------------------------
# CEO Action Summary
# ------------------------------
st.markdown("""
## 🧑‍💼 Action Summary

- **Invest more** in the campaigns and platforms listed below for best growth.  
- **Monitor risk alerts** in the detailed table to optimize or pause underperformers.  
- See glossary tooltips (ℹ️) for metric definitions.  
""")

# ------------------------------
# 4. Sidebar Filters
# ------------------------------
st.sidebar.header("Filters")
platform_options = df['tactic'].unique().tolist()
selected_platforms = st.sidebar.multiselect("Select Platform(s):", platform_options, default=platform_options)

campaign_options = df['campaign'].unique().tolist()
selected_campaigns = st.sidebar.multiselect("Select Campaign(s):", campaign_options, default=campaign_options)

min_date = df['date'].min()
max_date = df['date'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)

filtered_df = df[
    (df['tactic'].isin(selected_platforms)) &
    (df['campaign'].isin(selected_campaigns)) &
    (df['date'] >= pd.to_datetime(start_date)) &
    (df['date'] <= pd.to_datetime(end_date))
]

filtered_df = calculate_metrics(filtered_df)

# ------------------------------
# 5. KPI Cards
# ------------------------------
def kpi_trend(current, previous):
    if previous == 0:
        return ""
    change = (current - previous)/previous
    arrow = "⬆️" if change > 0 else "⬇️"
    return f"{arrow} {abs(change)*100:.1f}%"

def kpi_card(metric_name, current_value, previous_value, sparkline_data, unit=""):
    trend_text = kpi_trend(current_value, previous_value)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=sparkline_data,
        mode='lines',
        line=dict(color='blue'),
        showlegend=False
    ))
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0,r=0,t=0,b=0),
        height=50,
        width=150
    )
    st.metric(label=metric_name, value=f"{unit}{current_value:,.2f}", delta=trend_text)
    st.plotly_chart(fig, use_container_width=True)

# Previous period for comparison
previous_df = df[
    (df['tactic'].isin(selected_platforms)) &
    (df['campaign'].isin(selected_campaigns)) &
    (df['date'] >= pd.to_datetime(start_date) - (pd.to_datetime(end_date)-pd.to_datetime(start_date))) &
    (df['date'] < pd.to_datetime(start_date))
]
previous_df = calculate_metrics(previous_df)

# KPI values
total_spend = filtered_df['spend'].sum()
prev_spend = previous_df['spend'].sum()
total_revenue = filtered_df['attributed revenue'].sum()
prev_revenue = previous_df['attributed revenue'].sum()
total_orders = filtered_df['# of orders'].sum()
prev_orders = previous_df['# of orders'].sum()
avg_cpc = filtered_df['CPC'].mean()
prev_cpc = previous_df['CPC'].mean()
avg_conversion_rate = filtered_df['Conversion Rate'].mean() * 100
prev_conversion_rate = previous_df['Conversion Rate'].mean() * 100
overall_roas = total_revenue / total_spend if total_spend > 0 else 0

# Sparkline data
last_14_days = filtered_df.groupby('date').sum().tail(14)

col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card("💰 Total Spend", total_spend, prev_spend, last_14_days['spend'], unit="$")
with col2:
    kpi_card("📈 Total Revenue", total_revenue, prev_revenue, last_14_days['attributed revenue'], unit="$")
with col3:
    kpi_card("🛒 Total Orders", total_orders, prev_orders, last_14_days['# of orders'])
with col4:
    kpi_card("💲 Avg CPC", avg_cpc, prev_cpc, last_14_days['spend']/last_14_days['clicks'], unit="$")

col5, col6 = st.columns(2)
with col5:
    kpi_card("📊 Avg Conversion Rate", avg_conversion_rate, prev_conversion_rate, 
             last_14_days['# of orders']/last_14_days['clicks']*100, unit="%")
with col6:
    kpi_card("📈 Overall ROAS", overall_roas, 0, last_14_days['attributed revenue']/last_14_days['spend'])

# ------------------------------
# 6. Daily Spend vs Revenue
# ------------------------------
daily_df = filtered_df.groupby('date')[['spend','attributed revenue']].sum().reset_index()
daily_df['Profit'] = daily_df['attributed revenue'] - daily_df['spend']

fig1 = px.line(daily_df, x='date', y=['spend','attributed revenue'], title="Daily Spend vs Revenue")
loss_days = daily_df[daily_df['Profit'] < 0]
fig1.add_scatter(x=loss_days['date'], y=loss_days['attributed revenue'], 
                 mode='markers', marker=dict(color='red', size=10), name='Loss Day')
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------
# 7. Top Campaigns & Platforms
# ------------------------------
st.markdown("### 🏆 Top Campaigns & Platforms to Invest In")
platform_campaign_df = filtered_df.groupby(['tactic', 'campaign']).agg({
    'spend': 'sum',
    'attributed revenue': 'sum'
}).reset_index()
platform_campaign_df['ROAS'] = platform_campaign_df['attributed revenue'] / platform_campaign_df['spend'].replace(0,1)
platform_campaign_df['Predicted Revenue'] = platform_campaign_df['attributed revenue'] * 1.1
platform_campaign_df['Predicted Profit'] = platform_campaign_df['Predicted Revenue'] - platform_campaign_df['spend']

def ceo_recommend(row):
    if row['ROAS'] >= 2 and row['Predicted Profit'] > 0:
        return "Invest More"
    elif row['ROAS'] < 1 or row['Predicted Profit'] < 0:
        return "Reduce/Pause"
    else:
        return "Monitor"

platform_campaign_df['Recommendation'] = platform_campaign_df.apply(ceo_recommend, axis=1)
top_platform_campaigns = platform_campaign_df.sort_values('Predicted Profit', ascending=False).head(10)
st.dataframe(top_platform_campaigns[['tactic','campaign','ROAS','Predicted Revenue','Predicted Profit','Recommendation']])

# ------------------------------
# 8. Campaign Prioritization
# ------------------------------
st.markdown("### 🚀 Campaign Prioritization with Risk Alerts")
campaign_df = filtered_df.groupby('campaign')[['spend','attributed revenue','# of orders','clicks','impression']].sum().reset_index()
campaign_df['CTR'] = campaign_df['clicks'] / campaign_df['impression'].replace(0,1)
campaign_df['ROAS'] = campaign_df['attributed revenue'] / campaign_df['spend'].replace(0,1)
campaign_df['Conversion Rate'] = campaign_df['# of orders'] / campaign_df['clicks'].replace(0,1)
campaign_df['Predicted Revenue'] = campaign_df['attributed revenue'] * 1.1
campaign_df['Predicted Profit'] = campaign_df['Predicted Revenue'] - campaign_df['spend']

def campaign_recommendation(row):
    if row['Predicted Profit'] < 0:
        return "⚠️ Risk - Optimize/Pause"
    elif row['ROAS'] >= 2:
        return "Invest More"
    else:
        return "Monitor"

campaign_df['Recommendation'] = campaign_df.apply(campaign_recommendation, axis=1)

rec_filter = st.selectbox("Show campaigns with recommendation:", ["All", "Invest More", "Monitor", "⚠️ Risk - Optimize/Pause"])
if rec_filter == "All":
    display_df = campaign_df
else:
    display_df = campaign_df[campaign_df["Recommendation"] == rec_filter]
st.dataframe(display_df[['campaign','ROAS','Predicted Revenue','Predicted Profit','Recommendation']])

# ------------------------------
# 9. Top 3 Campaigns
# ------------------------------
st.markdown("### 🏆 Top 3 Campaigns to Prioritize")
top_campaigns = campaign_df[campaign_df['Recommendation']=="Invest More"].sort_values('Predicted Profit', ascending=False).head(3)
if not top_campaigns.empty:
    st.dataframe(top_campaigns[['campaign','ROAS','Predicted Revenue','Predicted Profit']])
else:
    st.info("No campaigns currently recommended for priority investment.")

# ------------------------------
# 10. Forecast
# ------------------------------
st.markdown("### 🔮 Platform-Level Profit Forecast (Next 14 Days)")
platform_forecast_list = []
for platform in selected_platforms:
    pf_df = filtered_df[filtered_df['tactic']==platform].groupby('date')['attributed revenue'].sum().reset_index()
    if len(pf_df) < 2:
        continue
    model = ExponentialSmoothing(pf_df['attributed revenue'], trend='add', seasonal=None)
    fit_model = model.fit()
    forecast = fit_model.forecast(14)
    forecast_df = pd.DataFrame({
        'date': pd.date_range(start=pf_df['date'].max()+pd.Timedelta(days=1), periods=14),
        'forecast_revenue': forecast
    })
    forecast_df['platform'] = platform
    forecast_df['forecast_profit'] = forecast_df['forecast_revenue'] - pf_df['attributed revenue'].mean()
    platform_forecast_list.append(forecast_df)

# ------------------------------
# 11. Spend Scenario
# ------------------------------
st.markdown("### ⚙️ Adjust Spend Scenario")
spend_increase_pct = st.slider("Increase Spend by %:", 0, 100, 20)
if platform_forecast_list:
    platform_forecast_df = pd.concat(platform_forecast_list)
    platform_forecast_df['adjusted_profit'] = platform_forecast_df['forecast_profit'] * (1 + spend_increase_pct/100)
    platforms = platform_forecast_df['platform'].unique().tolist()
    selected_pf = st.selectbox("Select Platform to View Scenario:", platforms)
    pf_df = platform_forecast_df[platform_forecast_df['platform']==selected_pf].copy()
    pf_df = pd.melt(
        pf_df,
        id_vars=['date','platform'],
        value_vars=['forecast_profit','adjusted_profit'],
        var_name='Scenario',
        value_name='Profit'
    )
    fig_scenario = px.line(pf_df, x='date', y='Profit', color='Scenario', markers=True,
                           title=f"{selected_pf} Profit Scenario with {spend_increase_pct}% Spend Increase")
    st.plotly_chart(fig_scenario, use_container_width=True)

# ------------------------------
# 12. Customer Lifetime Value (CLV)
# ------------------------------
st.markdown("### 💎 Segment-Level CLV Estimation")
SEGMENT_COL = 'tactic'
if SEGMENT_COL in filtered_df.columns:
    segment_df = filtered_df.groupby(SEGMENT_COL).agg({
        'attributed revenue': 'sum',
        '# of orders': 'sum'
    }).reset_index()
    segment_df['Avg Order Value'] = segment_df['attributed revenue'] / segment_df['# of orders'].replace(0,1)
    n_periods = filtered_df['date'].dt.to_period('M').nunique()
    segment_df['Purchase Frequency'] = segment_df['# of orders'] / n_periods
    assumed_lifespan = 12
    segment_df['CLV'] = segment_df['Avg Order Value'] * segment_df['Purchase Frequency'] * assumed_lifespan
    st.dataframe(segment_df[[SEGMENT_COL, 'CLV', 'Avg Order Value', 'Purchase Frequency']])
    st.markdown(f"_Assuming customer lifespan: {assumed_lifespan} months._")
else:
    st.info(f"No '{SEGMENT_COL}' column for CLV calculation.")

# ------------------------------
# 13. Marketing Efficiency Score
# ------------------------------
st.markdown("### 📊 Marketing Efficiency Score by Campaign")
campaign_df['Efficiency Score'] = campaign_df['ROAS'] * campaign_df['Conversion Rate'] * campaign_df['CTR']
top_efficiency = campaign_df.sort_values('Efficiency Score', ascending=False).head(10)
st.dataframe(top_efficiency[['campaign','Efficiency Score','ROAS','CTR','Conversion Rate']])

# ------------------------------
# 14. Platform Contribution
# ------------------------------
st.markdown("### 🌐 Platform Contribution")
platform_contrib = filtered_df.groupby('tactic')[['spend','attributed revenue']].sum().reset_index()
platform_contrib['ROAS'] = platform_contrib['attributed revenue'] / platform_contrib['spend'].replace(0,1)
fig_contrib = px.pie(platform_contrib, names='tactic', values='attributed revenue', title="Revenue Contribution by Platform")
st.plotly_chart(fig_contrib, use_container_width=True)

# ------------------------------
# 15. State Heatmap
# ------------------------------
if 'state' in filtered_df.columns:
    st.markdown("### 📍 State-wise Profit Heatmap")
    state_df = filtered_df.groupby('state')[['spend','attributed revenue']].sum().reset_index()
    state_df['Profit'] = state_df['attributed revenue'] - state_df['spend']
    fig_heatmap = px.choropleth(state_df, locations='state', locationmode='USA-states', color='Profit',
                                color_continuous_scale='Viridis', title="State-wise Profit")
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ------------------------------
# 17. Narrative Insights (Auto CEO Recommendations)
# ------------------------------
st.markdown("## 📝 Executive Narrative Insights")

insights = []

# Top campaign recommendation
if not campaign_df.empty:
    best_campaign = campaign_df.sort_values("ROAS", ascending=False).iloc[0]
    insights.append(
        f"📈 Invest more in **{best_campaign['campaign']}** "
        f"(ROAS {best_campaign['ROAS']:.2f}, Predicted Profit ${best_campaign['Predicted Profit']:,.0f})."
    )

# Risky campaign warning
risky = campaign_df[campaign_df['Recommendation'].str.contains("Risk")]
if not risky.empty:
    risky_top = risky.sort_values("Predicted Profit").iloc[0]
    insights.append(
        f"⚠️ Campaign **{risky_top['campaign']}** is at risk with Predicted Profit "
        f"${risky_top['Predicted Profit']:,.0f}. Consider pausing or optimizing."
    )

# Platform with highest CLV
if 'CLV' in segment_df.columns:
    best_segment = segment_df.sort_values("CLV", ascending=False).iloc[0]
    insights.append(
        f"💎 Customers from **{best_segment[SEGMENT_COL]}** show the highest CLV "
        f"(${best_segment['CLV']:,.0f}). Focus long-term investment here."
    )

# Most efficient campaign
if 'Efficiency Score' in campaign_df.columns:
    best_eff = campaign_df.sort_values("Efficiency Score", ascending=False).iloc[0]
    insights.append(
        f"🚀 **{best_eff['campaign']}** is the most efficient campaign "
        f"(Efficiency Score {best_eff['Efficiency Score']:.3f})."
    )

# Display insights
if insights:
    for rec in insights:
        st.write(rec)
else:
    st.info("No narrative insights available for the current selection.")


# ------------------------------
# 18. Downloadable Reports
# ------------------------------
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.markdown("## 📥 Download Reports")

# Excel Report
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    campaign_df.to_excel(writer, sheet_name="Campaigns", index=False)
    segment_df.to_excel(writer, sheet_name="Segments", index=False)
    platform_contrib.to_excel(writer, sheet_name="Platforms", index=False)

st.download_button(
    label="📊 Download Excel Report",
    data=excel_buffer.getvalue(),
    file_name="marketing_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# PDF Report
pdf_buffer = io.BytesIO()
doc = SimpleDocTemplate(pdf_buffer)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Executive Marketing Report", styles["Title"]))
story.append(Spacer(1, 12))

# Add KPIs
story.append(Paragraph(f"Total Spend: ${total_spend:,.0f}", styles["Normal"]))
story.append(Paragraph(f"Total Revenue: ${total_revenue:,.0f}", styles["Normal"]))
story.append(Paragraph(f"Overall ROAS: {overall_roas:.2f}", styles["Normal"]))
story.append(Spacer(1, 12))

# Add Insights
story.append(Paragraph("CEO Insights:", styles["Heading2"]))
for rec in insights:
    story.append(Paragraph(rec, styles["Normal"]))
    story.append(Spacer(1, 8))

doc.build(story)

st.download_button(
    label="📄 Download PDF Report",
    data=pdf_buffer.getvalue(),
    file_name="marketing_report.pdf",
    mime="application/pdf"
)


# ------------------------------
# 16. CEO Insights
# ------------------------------
st.markdown("## 💡 Insights")
st.markdown("""
- Focus on campaigns labeled 'Invest More' to maximize profit.  
- Watch campaigns flagged ⚠️ Risk and optimize/pause them.  
- Use CLV insights to guide long-term budget allocations.  
- Efficiency Score shows which campaigns are both cost-effective and high-performing.  
""")
