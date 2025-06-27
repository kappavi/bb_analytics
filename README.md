# BB Analytics Visualization Dashboard

This repository contains a Streamlit dashboard for analyzing item-centric promotion data and a Jupyter notebook for price change tracking analysis.

## Setup Instructions

### 1. Environment Variables

Create a `.env` file in the root directory with your database credentials:

```env
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=your_database_port
DB_DATABASE=your_database_name
```
### 1.5 (optional) Load data 

If you have the CSVs cached, then you can create a folder data/ from the root, and create two files:
1. analysis_cleaned.csv
2. analysis_staging.csv

The code will automatically detect that you have a CSV in this location and will that to use the DF instead, which will be signifncatly faster than loading from Postgres.

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Visualization Dashboard

```bash
streamlit run visualization.py
```

The dashboard will open in your default web browser, typically at `http://localhost:8501`.

## Additional Files

- `test_price_change_tracking.ipynb` - Avi's Jupyter notebook for detailed price change analysis
- `queries/` - SQL query files used by the application
- `checkpoints/` - Saved visualization configurations

## Requirements

- Python 3.8+
- PostgreSQL database with SSL support
- All dependencies listed in `requirements.txt` 