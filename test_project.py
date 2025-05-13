import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from project import validate_data, clean_dataset, download_covid_data

@pytest.fixture
def sample_data():
    """Create sample COVID data for testing."""
    return pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'location': ['USA', 'India', 'Kenya'],
        'total_cases': [1000, 2000, 3000],
        'total_deaths': [50, 100, 150],
        'total_vaccinations': [500, 1000, 1500],
        'population': [1000000, 2000000, 3000000],
        'new_cases': [100, 200, 300],
        'new_deaths': [5, 10, 15],
        'hosp_patients': [20, 40, 60],
        'icu_patients': [5, 10, 15]
    })

def test_validate_data_valid(sample_data):
    """Test data validation with valid data."""
    issues = validate_data(sample_data)
    assert len(issues) == 0, "Valid data should have no issues"

def test_validate_data_missing_columns():
    """Test data validation with missing columns."""
    df = pd.DataFrame({
        'date': ['2023-01-01'],
        'location': ['USA']
    })
    issues = validate_data(df)
    assert len(issues) > 0, "Should detect missing required columns"
    assert any('Missing required columns' in issue for issue in issues)

def test_validate_data_negative_values(sample_data):
    """Test data validation with negative values."""
    df = sample_data.copy()
    df.loc[0, 'total_cases'] = -100
    issues = validate_data(df)
    assert len(issues) > 0, "Should detect negative values"
    assert any('Negative values found' in issue for issue in issues)

def test_validate_data_unrealistic_death_rate(sample_data):
    """Test data validation with unrealistic death rate."""
    df = sample_data.copy()
    df.loc[0, 'total_deaths'] = 2000  # More deaths than cases
    df.loc[0, 'total_cases'] = 1000
    issues = validate_data(df)
    assert len(issues) > 0, "Should detect unrealistic death rate"
    assert any('Death rate exceeds 100%' in issue for issue in issues)

def test_clean_dataset(sample_data):
    """Test dataset cleaning functionality."""
    cleaned_df = clean_dataset(sample_data)
    assert 'death_rate' in cleaned_df.columns, "Should add death_rate column"
    assert 'cases_per_million' in cleaned_df.columns, "Should add cases_per_million column"
    assert cleaned_df['date'].dtype == 'datetime64[ns]', "Should convert date to datetime"

def test_download_covid_data(tmp_path):
    """Test data download functionality."""
    file_path = tmp_path / "test_covid_data.csv"
    result = download_covid_data(str(file_path))
    assert result == True, "Download should succeed"
    assert file_path.exists(), "File should exist after download"

@pytest.mark.parametrize("test_input,expected", [
    ({'total_cases': 1000, 'total_deaths': 50}, 5.0),  # 5% death rate
    ({'total_cases': 2000, 'total_deaths': 200}, 10.0),  # 10% death rate
])
def test_death_rate_calculation(test_input, expected):
    """Test death rate calculation with different inputs."""
    df = pd.DataFrame([test_input])
    df['death_rate'] = (df['total_deaths'] / df['total_cases'] * 100).round(2)
    assert df['death_rate'].iloc[0] == expected

def test_data_cleaning_with_missing_values(sample_data):
    """Test handling of missing values during cleaning."""
    df = sample_data.copy()
    df.loc[0, 'total_cases'] = np.nan
    cleaned_df = clean_dataset(df)
    assert cleaned_df['total_cases'].isna().sum() == 0, "Should handle missing values"

def test_data_cleaning_with_invalid_dates(sample_data):
    """Test handling of invalid dates during cleaning."""
    df = sample_data.copy()
    df.loc[0, 'date'] = 'invalid_date'
    with pytest.raises(Exception):
        clean_dataset(df) 