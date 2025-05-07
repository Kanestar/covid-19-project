import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_dataset(file_name):
    try:
        df = pd.read_csv(file_name)
        print("Data loaded successfully!")
        print("Number of rows and columns:", df.shape)
        print("Columns:", df.columns)
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")

def clean_dataset(df):
    try:
        print("Cleaning dataset...")
        # Filter countries of interest
        countries_of_interest = ['Kenya', 'USA', 'India']
        df = df[df['location'].isin(countries_of_interest)]

        # Drop rows with missing dates or critical values
        df = df.dropna(subset=['date', 'total_cases', 'total_deaths'])

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Handle missing numeric values
        df['total_cases'] = df['total_cases'].fillna(0)
        df['total_deaths'] = df['total_deaths'].fillna(0)

        print("Dataset cleaned successfully!")
        return df
    except Exception as e:
        print(f"Failed to clean dataset: {e}")

def perform_eda(df):
    try:
        print("Performing exploratory data analysis...")
        # Plot total cases over time for selected countries
        plt.figure(figsize=(10,6))
        sns.lineplot(x='date', y='total_cases', hue='location', data=df)
        plt.title('Total Cases Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Cases')
        plt.show()

        # Plot total deaths over time
        plt.figure(figsize=(10,6))
        sns.lineplot(x='date', y='total_deaths', hue='location', data=df)
        plt.title('Total Deaths Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Deaths')
        plt.show()

        # Compare daily new cases between countries
        plt.figure(figsize=(10,6))
        sns.lineplot(x='date', y='new_cases', hue='location', data=df)
        plt.title('Daily New Cases')
        plt.xlabel('Date')
        plt.ylabel('New Cases')
        plt.show()

        # Calculate the death rate: total_deaths / total_cases
        df['death_rate'] = df['total_deaths'] / df['total_cases']
        print("Death rate statistics:")
        print(df['death_rate'].describe())
    except Exception as e:
        print(f"Failed to perform EDA: {e}")

def visualize_vaccination_progress(df):
    try:
        print("Visualizing vaccination progress...")
        # Plot cumulative vaccinations over time for selected countries
        plt.figure(figsize=(10,6))
        sns.lineplot(x='date', y='total_vaccinations', hue='location', data=df)
        plt.title('Cumulative Vaccinations Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Vaccinations')
        plt.show()

        # Compare % vaccinated population
        df['percent_vaccinated'] = df['total_vaccinations'] / df['population']
        plt.figure(figsize=(10,6))
        sns.barplot(x='location', y='percent_vaccinated', data=df)
        plt.title('Percent Vaccinated Population')
        plt.xlabel('Country')
        plt.ylabel('Percent Vaccinated')
        plt.show()
    except Exception as e:
        print(f"Failed to visualize vaccination progress: {e}")

def main():
    file_name = 'owid-covid-data.csv'
    df = load_dataset(file_name)
    if df is not None:
        df = clean_dataset(df)
        if df is not None:
            perform_eda(df)
            visualize_vaccination_progress(df)

if __name__ == "__main__":
    main()