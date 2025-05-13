import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Page configuration
st.set_page_config(
    page_title="COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🦠 COVID-19 Global Data Tracker")
st.markdown("""
This interactive dashboard provides real-time analysis and visualization of COVID-19 data across different countries.
Data source: [Our World in Data](https://ourworldindata.org/coronavirus)
""")

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    file_name = 'owid-covid-data.csv'
    if not Path(file_name).exists():
        st.error("❌ Data file not found. Please run project.py first to download the data.")
        st.stop()
    return pd.read_csv(file_name)

# Load the data
df = load_data()

# Data preprocessing
df['date'] = pd.to_datetime(df['date'])

# Sidebar configuration
st.sidebar.header("📊 Dashboard Controls")

# Add about section in sidebar
with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown("""
    This dashboard is built using:
    - Streamlit
    - Plotly
    - Pandas
    
    Data is updated daily from Our World in Data.
    """)

# Country selection with search
available_countries = sorted(df['location'].unique())

# Print available countries to help debug
print("Available countries in dataset:", available_countries)

selected_countries = st.sidebar.multiselect(
    "🌍 Select Countries",
    available_countries,
    default=['United States', 'India', 'Kenya'],
    help="You can select multiple countries for comparison"
)

# Add a note about available countries
if not selected_countries:
    st.sidebar.info("ℹ️ Please select at least one country to view data")

# Metric selection
available_metrics = [
    'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
    'total_vaccinations', 'hosp_patients', 'icu_patients'
]
selected_metric = st.sidebar.selectbox(
    "📈 Select Primary Metric",
    available_metrics,
    format_func=lambda x: x.replace('_', ' ').title()
)

# Date range selection with presets
date_ranges = {
    'Last 30 days': 30,
    'Last 90 days': 90,
    'Last 6 months': 180,
    'Last year': 365,
    'All time': None
}
selected_range = st.sidebar.selectbox("📅 Select Time Range", list(date_ranges.keys()))

# Calculate date range
max_date = df['date'].max()
if date_ranges[selected_range]:
    min_date = max_date - timedelta(days=date_ranges[selected_range])
else:
    min_date = df['date'].min()

# Custom date range
use_custom_dates = st.sidebar.checkbox("📆 Use Custom Date Range")
if use_custom_dates:
    date_range = st.sidebar.date_input(
        "Select Custom Date Range",
        value=(max_date - timedelta(days=30), max_date),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    min_date = pd.to_datetime(date_range[0])
    max_date = pd.to_datetime(date_range[1])

# Filter data based on selection
filtered_df = df[
    (df['location'].isin(selected_countries)) &
    (df['date'] >= min_date) &
    (df['date'] <= max_date)
]

# Download filtered data
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

with st.sidebar.expander("💾 Download Data", expanded=False):
    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        "Download Filtered Data as CSV",
        csv,
        "covid_data.csv",
        "text/csv",
        key='download-csv'
    )

# Main dashboard area
# Metrics row with sparklines
st.header("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

latest_data = filtered_df.groupby('location').last()

# Helper function for sparklines
def create_sparkline(data, metric):
    fig = go.Figure(go.Scatter(
        x=data.index, y=data[metric],
        mode='lines',
        showlegend=False,
        line_color='#1f77b4'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=50,
        width=100,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={'visible': False},
        yaxis={'visible': False}
    )
    return fig

with col1:
    total_cases = latest_data['total_cases'].sum()
    st.metric(
        "Total Cases",
        f"{total_cases:,.0f}",
        delta=f"{filtered_df['new_cases'].sum():,.0f} new"
    )
    st.plotly_chart(create_sparkline(
        filtered_df.groupby('date')['total_cases'].sum(),
        'total_cases'
    ), use_container_width=True)

with col2:
    total_deaths = latest_data['total_deaths'].sum()
    st.metric(
        "Total Deaths",
        f"{total_deaths:,.0f}",
        delta=f"{filtered_df['new_deaths'].sum():,.0f} new"
    )
    st.plotly_chart(create_sparkline(
        filtered_df.groupby('date')['total_deaths'].sum(),
        'total_deaths'
    ), use_container_width=True)

with col3:
    avg_death_rate = (total_deaths / total_cases * 100)
    st.metric(
        "Average Death Rate",
        f"{avg_death_rate:.2f}%"
    )

with col4:
    total_vaccinations = latest_data['total_vaccinations'].sum()
    st.metric(
        "Total Vaccinations",
        f"{total_vaccinations:,.0f}"
    )

# Main visualization tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Trend Analysis",
    "💉 Vaccination Progress",
    "🏥 Healthcare Impact",
    "📋 Data Explorer"
])

with tab1:
    # Trend Analysis
    st.subheader("Trend Analysis")
    
    # Main metric trend
    fig_main = px.line(
        filtered_df,
        x='date',
        y=selected_metric,
        color='location',
        title=f'{selected_metric.replace("_", " ").title()} Over Time'
    )
    fig_main.update_layout(height=500)
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Daily changes
    if 'new' not in selected_metric:
        daily_metric = f'new_{selected_metric.replace("total_", "")}'
        if daily_metric in filtered_df.columns:
            fig_daily = px.bar(
                filtered_df,
                x='date',
                y=daily_metric,
                color='location',
                title=f'Daily New {selected_metric.replace("total_", "").title()}'
            )
            st.plotly_chart(fig_daily, use_container_width=True)

with tab2:
    # Vaccination Analysis
    st.subheader("Vaccination Progress")
    
    # Vaccination trends
    fig_vax = px.line(
        filtered_df,
        x='date',
        y='total_vaccinations',
        color='location',
        title='Vaccination Progress Over Time'
    )
    st.plotly_chart(fig_vax, use_container_width=True)
    
    # Vaccination rates
    latest_vax = filtered_df.groupby('location').last().reset_index()
    latest_vax['vax_rate'] = latest_vax['total_vaccinations'] / latest_vax['population'] * 100
    fig_vax_rate = px.bar(
        latest_vax,
        x='location',
        y='vax_rate',
        title='Vaccination Rate by Country (%)',
        text=latest_vax['vax_rate'].round(1)
    )
    fig_vax_rate.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig_vax_rate, use_container_width=True)

with tab3:
    # Healthcare Impact
    st.subheader("Healthcare System Impact")
    
    if 'hosp_patients' in df.columns and 'icu_patients' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hosp = px.line(
                filtered_df,
                x='date',
                y='hosp_patients',
                color='location',
                title='Hospital Patients Over Time'
            )
            st.plotly_chart(fig_hosp, use_container_width=True)
        
        with col2:
            fig_icu = px.line(
                filtered_df,
                x='date',
                y='icu_patients',
                color='location',
                title='ICU Patients Over Time'
            )
            st.plotly_chart(fig_icu, use_container_width=True)
        
        # Calculate hospital burden
        if 'hospital_beds_per_thousand' in df.columns:
            st.subheader("Hospital Capacity Analysis")
            latest_hosp = filtered_df.groupby('location').last()
            latest_hosp['beds_used_per_thousand'] = latest_hosp['hosp_patients'] / latest_hosp['population'] * 1000
            latest_hosp['capacity_used'] = (latest_hosp['beds_used_per_thousand'] / latest_hosp['hospital_beds_per_thousand'] * 100).round(1)
            
            fig_capacity = px.bar(
                latest_hosp.reset_index(),
                x='location',
                y='capacity_used',
                title='Estimated Hospital Capacity Used (%)',
                text='capacity_used'
            )
            fig_capacity.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_capacity, use_container_width=True)
    else:
        st.info("🏥 Hospital and ICU data not available for the selected countries/timeframe.")

with tab4:
    # Data Explorer
    st.subheader("Data Explorer")
    
    # Column selection
    selected_columns = st.multiselect(
        "Select columns to display",
        df.columns.tolist(),
        default=['date', 'location', 'total_cases', 'total_deaths', 'total_vaccinations']
    )
    
    # Show filtered data
    if selected_columns:
        st.dataframe(
            filtered_df[selected_columns].sort_values('date', ascending=False),
            use_container_width=True
        )
    
    # Summary statistics
    with st.expander("📊 View Summary Statistics"):
        st.dataframe(
            filtered_df[selected_columns].describe(),
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Data from Our World in Data</p>
</div>
""", unsafe_allow_html=True) 