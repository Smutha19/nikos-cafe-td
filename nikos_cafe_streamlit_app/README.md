# Nikos Cafe Topological Data Analysis (TDA)

## Project Overview

This application applies Topological Data Analysis to cafe operational data to reveal hidden patterns and generate actionable business insights. Created by Sakshi Mutha as part of a Master's project in Data Science.

## Features

- Upload and analyze service performance data from Oracle Symphony
- Examine labor cost and efficiency data from Homebase
- Analyze digital ordering patterns from the GET app
- Apply advanced topological data analysis to identify hidden patterns
- Generate business insights and recommendations
- Export processed data for Power BI

## Topological Data Analysis Workflow

1. **Initial Data Integration**: Load and merge data from multiple systems
2. **Feature Extraction**: Extract numerical features from raw data
3. **Data Cleaning**: Preprocess data for analysis
4. **Topological Analysis**: Compute persistent homology
5. **Feature Matrix Generation**: Organize topological features
6. **Pattern Comparison**: Compare patterns across data sources
7. **Insight Generation**: Translate patterns into business insights
8. **Visualization**: Present findings in an accessible format

## Technologies Used

- Streamlit for the interactive dashboard
- Ripser/GUDHI for topological data analysis
- Plotly for interactive visualizations
- Scikit-learn for data preprocessing
- Python numerical libraries (NumPy, Pandas)

## Getting Started

### Installation

```bash
pip install -r streamlit_requirements.txt
```

### Running Locally

```bash
streamlit run simple_app.py
```

### Sample Data

The `sample_data/` directory contains example CSV files you can use to test the application:
- Service Performance.csv
- sales_labor_analysis.csv
- Nikos GET APP.csv

## License

This project is for educational purposes only. Not for commercial use without permission.

## Acknowledgments

- Created for a Master's project in Data Science
- Nikos Cafe for providing operational data
- Faculty advisors for guidance on TDA applications