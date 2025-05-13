import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
import nbformat
from nbconvert import HTMLExporter
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_covid_data(file_name):
    try:
        print("Downloading COVID-19 data...")
        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print("Data downloaded successfully!")
            return True
        else:
            print(f"Failed to download data. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

def load_dataset(file_name):
    try:
        df = pd.read_csv(file_name)
        print("Data loaded successfully!")
        print("\nDataset Overview:")
        print("Number of rows and columns:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        print("\nMissing values summary:")
        print(df.isnull().sum())
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")

def validate_data(df):
    """Validate the dataset and return a list of issues found."""
    issues = []
    
    # Check for required columns
    required_columns = ['date', 'location', 'total_cases', 'total_deaths', 'total_vaccinations', 'population']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for date range validity
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        if (max_date - min_date).days < 0:
            issues.append("Invalid date range detected")
    
    # Check for negative values in numeric columns
    numeric_columns = ['total_cases', 'total_deaths', 'total_vaccinations', 'population']
    for col in numeric_columns:
        if col in df.columns and (df[col] < 0).any():
            issues.append(f"Negative values found in {col}")
    
    # Check for unrealistic values
    if 'total_deaths' in df.columns and 'total_cases' in df.columns:
        death_rate = df['total_deaths'] / df['total_cases']
        if (death_rate > 1).any():
            issues.append("Death rate exceeds 100% in some records")
    
    return issues

def clean_dataset(df):
    try:
        logger.info("Starting dataset cleaning...")
        
        # Validate data first
        issues = validate_data(df)
        if issues:
            logger.warning("Data validation issues found:")
            for issue in issues:
                logger.warning(f"- {issue}")
        
        # Print available countries to help with selection
        logger.info("Available countries in dataset:")
        logger.info(df['location'].unique())
        
        # Filter countries of interest
        countries_of_interest = ['United States', 'India', 'Kenya']  # Updated country names
        df = df[df['location'].isin(countries_of_interest)]
        logger.info(f"Filtered data for countries: {countries_of_interest}")

        # Drop rows with missing dates or critical values
        critical_columns = ['date', 'total_cases', 'total_deaths']
        initial_rows = len(df)
        df = df.dropna(subset=critical_columns)
        rows_dropped = initial_rows - len(df)
        logger.info(f"Dropped {rows_dropped} rows with missing critical values")

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Handle missing numeric values with forward fill then backward fill
        numeric_columns = ['total_cases', 'total_deaths', 'total_vaccinations', 'new_cases', 'new_deaths']
        df[numeric_columns] = df.groupby('location')[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Add derived columns
        df['death_rate'] = (df['total_deaths'] / df['total_cases'] * 100).round(2)
        df['cases_per_million'] = (df['total_cases'] / df['population'] * 1_000_000).round(2)
        
        logger.info("Dataset cleaned successfully!")
        return df
    except Exception as e:
        logger.error(f"Failed to clean dataset: {str(e)}")
        raise

def safe_plot(func):
    """Decorator for safe plotting with error handling."""
    def wrapper(*args, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in plotting function {func.__name__}: {str(e)}")
            plt.close()  # Clean up any partial plots
    return wrapper

@safe_plot
def perform_eda(df):
    logger.info("Performing exploratory data analysis...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('COVID-19 Analysis Dashboard', fontsize=16)
    
    # Plot 1: Total cases over time
    sns.lineplot(x='date', y='total_cases', hue='location', data=df, ax=axes[0,0])
    axes[0,0].set_title('Total Cases Over Time')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Total deaths over time
    sns.lineplot(x='date', y='total_deaths', hue='location', data=df, ax=axes[0,1])
    axes[0,1].set_title('Total Deaths Over Time')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Daily new cases
    sns.lineplot(x='date', y='new_cases', hue='location', data=df, ax=axes[1,0])
    axes[1,0].set_title('Daily New Cases')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Death rate over time
    sns.lineplot(x='date', y='death_rate', hue='location', data=df, ax=axes[1,1])
    axes[1,1].set_title('Death Rate Over Time (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

@safe_plot
def visualize_vaccination_progress(df):
    logger.info("Visualizing vaccination progress...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative vaccinations
    sns.lineplot(x='date', y='total_vaccinations', hue='location', data=df, ax=ax1)
    ax1.set_title('Cumulative Vaccinations Over Time')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Vaccination percentage
    latest_data = df.groupby('location').last().reset_index()
    latest_data['percent_vaccinated'] = latest_data['total_vaccinations'] / latest_data['population'] * 100
    
    sns.barplot(x='location', y='percent_vaccinated', data=latest_data, ax=ax2)
    ax2.set_title('Percent Vaccinated Population')
    ax2.set_ylabel('Percent Vaccinated')
    
    plt.tight_layout()
    plt.show()

def create_interactive_dashboard(df):
    """Create an interactive dashboard using Plotly."""
    logger.info("Creating interactive dashboard...")
    
    # Create time series figure
    fig_cases = px.line(df, x='date', y='total_cases', color='location',
                        title='Interactive: Total COVID-19 Cases Over Time')
    fig_cases.show()
    
    # Create scatter plot of cases vs deaths
    fig_scatter = px.scatter(df, x='total_cases', y='total_deaths', color='location',
                            size='total_vaccinations', hover_name='location',
                            title='Cases vs Deaths (size represents vaccinations)')
    fig_scatter.show()
    
    # Create choropleth map
    latest_data = df.groupby('location').agg({
        'total_cases': 'max',
        'total_deaths': 'max',
        'total_vaccinations': 'max'
    }).reset_index()
    
    fig_map = px.choropleth(latest_data, 
                           locations='location',
                           locationmode='country names',
                           color='total_cases',
                           hover_name='location',
                           color_continuous_scale='Viridis',
                           title='Total COVID-19 Cases by Country')
    fig_map.show()

def generate_insights(df):
    print("\n📊 COVID-19 Data Analysis Insights:")
    print("=" * 50)
    
    # Get latest date in the dataset
    latest_date = df['date'].max().strftime('%Y-%m-%d')
    print(f"\n📅 Analysis as of: {latest_date}")
    
    # Cases Analysis
    max_cases = df.groupby('location')['total_cases'].max()
    max_daily_cases = df.groupby('location')['new_cases'].max()
    print("\n📈 Cases Analysis:")
    print(f"• Country with highest total cases: {max_cases.idxmax()} ({max_cases.max():,.0f} cases)")
    print(f"• Country with highest daily cases: {max_daily_cases.idxmax()} ({max_daily_cases.max():,.0f} cases)")
    
    # Deaths Analysis
    latest_data = df.groupby('location').last()
    death_rate = latest_data['total_deaths'] / latest_data['total_cases'] * 100
    print("\n💀 Mortality Analysis:")
    print(f"• Country with highest death rate: {death_rate.idxmax()} ({death_rate.max():.2f}%)")
    print(f"• Average death rate across countries: {death_rate.mean():.2f}%")
    
    # Vaccination Analysis
    vax_rate = latest_data['total_vaccinations'] / latest_data['population'] * 100
    print("\n💉 Vaccination Progress:")
    print(f"• Country with highest vaccination rate: {vax_rate.idxmax()} ({vax_rate.max():.2f}%)")
    print(f"• Average vaccination rate: {vax_rate.mean():.2f}%")
    
    # Time Series Analysis
    print("\n📈 Trend Analysis:")
    # Calculate daily growth rate
    df['growth_rate'] = df.groupby('location')['total_cases'].pct_change() * 100
    avg_growth = df.groupby('location')['growth_rate'].mean()
    print(f"• Country with highest avg daily growth rate: {avg_growth.idxmax()} ({avg_growth.max():.2f}%)")
    
    print("\n" + "=" * 50)

def export_to_notebook(df):
    """Export analysis to a Jupyter notebook."""
    try:
        logger.info("Exporting analysis to Jupyter notebook...")
        
        nb = nbformat.v4.new_notebook()
        
        # Add title markdown cell
        nb.cells.append(nbformat.v4.new_markdown_cell("# COVID-19 Data Analysis Report"))
        
        # Add introduction
        nb.cells.append(nbformat.v4.new_markdown_cell("""
        This notebook contains an analysis of COVID-19 data across selected countries.
        The analysis includes:
        - Data overview and cleaning
        - Exploratory Data Analysis
        - Vaccination Progress
        - Interactive Visualizations
        """))
        
        # Add code cells
        nb.cells.append(nbformat.v4.new_code_cell("# Import required libraries\n" + 
                                                 "import pandas as pd\n" +
                                                 "import matplotlib.pyplot as plt\n" +
                                                 "import seaborn as sns\n" +
                                                 "import plotly.express as px"))
        
        # Save notebook
        notebook_path = 'covid19_analysis.ipynb'
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        # Convert to HTML
        html_exporter = HTMLExporter()
        html_path = 'covid19_analysis.html'
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
            body, _ = html_exporter.from_notebook_node(notebook)
            with open(html_path, 'w', encoding='utf-8') as f_html:
                f_html.write(body)
        
        logger.info(f"Analysis exported to {notebook_path} and {html_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export analysis: {str(e)}")
        return False

def main():
    file_name = 'owid-covid-data.csv'
    
    # Check if file exists, if not download it
    if not Path(file_name).exists():
        if not download_covid_data(file_name):
            logger.error("Could not proceed without data file.")
            return
            
    df = load_dataset(file_name)
    if df is not None:
        try:
            df = clean_dataset(df)
            if df is not None:
                perform_eda(df)
                visualize_vaccination_progress(df)
                create_interactive_dashboard(df)
                generate_insights(df)
                export_to_notebook(df)
        except Exception as e:
            logger.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()