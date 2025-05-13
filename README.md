# COVID-19 Global Data Tracker

A comprehensive Python application for analyzing and visualizing global COVID-19 data trends, featuring both static analysis and an interactive dashboard.

## 🎯 Features

- Real-time COVID-19 data download from Our World in Data
- Automated data cleaning and validation
- Interactive visualizations using Plotly and Seaborn
- Comprehensive data analysis including:
  - Case trends
  - Death rates
  - Vaccination progress
  - Country comparisons
  - Hospital & ICU data (where available)
- Interactive Streamlit dashboard with:
  - Country selection
  - Date range filtering
  - Real-time metric updates
  - Dynamic visualizations
- Export capabilities to Jupyter Notebook and HTML
- Detailed logging and error handling

## 📁 Project Structure

The project consists of the following key files:

### Core Files
- `project.py`: Main analysis script that:
  - Downloads COVID-19 data from Our World in Data
  - Performs data cleaning and validation
  - Generates static visualizations
  - Creates analysis reports
  - Exports results to Jupyter notebook

- `dashboard.py`: Interactive Streamlit dashboard that:
  - Provides real-time data filtering
  - Shows interactive visualizations
  - Allows country and date selection
  - Displays key metrics and trends
  - Enables data exploration

### Setup and Configuration
- `setup.py`: Setup script that:
  - Creates virtual environment
  - Installs required dependencies
  - Generates run.bat for Windows users
  - Provides execution instructions

- `requirements.txt`: Lists all Python dependencies:
  - Data analysis libraries (pandas, numpy)
  - Visualization tools (matplotlib, seaborn, plotly)
  - Dashboard framework (streamlit)
  - Other required packages

### Testing
- `test_project.py`: Unit tests that verify:
  - Data validation functions
  - Cleaning procedures
  - Calculations accuracy
  - File operations
  - Error handling

### Output Files
- `covid19_analysis.ipynb`: Generated Jupyter notebook with analysis
- `covid19_analysis.html`: HTML export of the analysis
- `owid-covid-data.csv`: Downloaded COVID-19 data

## 📋 Prerequisites

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd covid-19-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Static Analysis
Run the main analysis script:
```bash
python project.py
```

The script will:
1. Download the latest COVID-19 data
2. Perform data cleaning and validation
3. Generate visualizations
4. Create an interactive dashboard
5. Export analysis to a Jupyter notebook

### Interactive Dashboard
Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The dashboard provides:
- Interactive country selection
- Date range filtering
- Real-time visualization updates
- Hospital & ICU data analysis
- Raw data exploration

## 📊 Output Files

- `covid19_analysis.ipynb`: Jupyter notebook with complete analysis
- `covid19_analysis.html`: HTML export of the analysis
- `owid-covid-data.csv`: Raw data file

## 🔍 Data Analysis

The project analyzes:
- Total cases and deaths over time
- Daily new cases
- Death rates
- Vaccination progress
- Hospital & ICU metrics
- Country-wise comparisons

## 📝 Logging

Logs are printed to console with detailed information about:
- Data validation issues
- Processing steps
- Error messages

## 🧪 Testing

Run the tests:
```bash
pytest
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data provided by [Our World in Data](https://ourworldindata.org/coronavirus)
- Built with Python, Pandas, Matplotlib, Seaborn, Plotly, and Streamlit 