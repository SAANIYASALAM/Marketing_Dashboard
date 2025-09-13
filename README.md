# 📊 Marketing Intelligence Dashboard

A comprehensive Streamlit-based dashboard for analyzing marketing campaigns, measuring their impact on business outcomes, and providing actionable insights for executives.

---

## 🚀 Features

1. **KPI Tracking**
   - Total Spend, Total Revenue, Total Orders, Average CPC
   - Conversion Rate and ROAS
   - Sparkline trends for quick visualization of recent performance

2. **Daily Spend vs Revenue Analysis**
   - Interactive line chart with markers for loss days
   - Helps identify campaigns that may require attention

3. **Top Campaigns & Platforms**
   - ROI-based recommendations: Invest More, Monitor, Reduce/Pause
   - Predicted Revenue and Profit for future planning

4. **Campaign Prioritization**
   - Highlights campaigns with risk alerts
   - Allows filtering campaigns by recommendations

5. **Forecasting**
   - Platform-level profit forecast for the next 14 days using Holt-Winters Exponential Smoothing
   - Adjustable spend scenario for predictive analysis

6. **Customer Lifetime Value (CLV)**
   - Segment-level CLV estimation based on Avg Order Value, Purchase Frequency, and Customer Lifespan

7. **Marketing Efficiency Score**
   - Combines ROAS, Conversion Rate, and CTR to rank campaigns

8. **Platform Contribution**
   - Revenue contribution by platform visualized using a pie chart

9. **Geographical Insights**
   - State-wise profit heatmap (USA) to visualize regional performance

10. **Executive Narrative Insights**
    - Auto-generated CEO-level insights and recommendations

11. **Downloadable Reports**
    - Excel and PDF reports containing KPIs, insights, and campaign analysis

---

## 📂 Data Requirements

The app expects the following CSV files in a `data/` directory:

- `Facebook.csv`
- `Google.csv`
- `TikTok.csv`
- `business.csv`

### Required Columns in Marketing Files
- `date` – campaign date
- `tactic` – platform (e.g., Facebook, Google, TikTok)
- `campaign` – campaign name
- `clicks`
- `impression`
- `spend`
- `attributed revenue`
- `# of orders`

### Required Columns in Business File
- `date`
- Additional business metrics to merge (optional)

Optional:
- `state` column for state-wise profit visualization

---

## 🛠️ Installation

1. Clone this repository:

```bash
git clone https://github.com/SAANIYASALAM/Marketing_Dashboard.git
cd marketing_dashboard
