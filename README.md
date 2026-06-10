# AI-Based Meal Forecasting System

Built an AI-based Meal Forecasting system for different Indian States.

## Overview

The AI-Based Meal Forecasting System is a time-series forecasting project designed to predict future institutional meal demand using historical meal-consumption records.

The system automates data collection, preprocessing, forecasting, and visualization to help organizations optimize food inventory planning, reduce wastage, and improve resource allocation.

Using Facebook Prophet, the model captures daily, weekly, and yearly seasonal patterns to generate accurate meal demand forecasts for both short-term and long-term planning.

---

## Features

* Automated ingestion of multiple CSV data files
* Data cleaning and preprocessing pipeline
* Aggregation of meal statistics into a unified dataset
* Time-series forecasting using Prophet
* 30-day demand forecasting
* 365-day demand forecasting
* Daily, weekly, and yearly seasonality analysis
* Automated visualization and reporting
* Forecast plot and seasonal component generation

---

## Tech Stack

* Python
* Pandas
* Prophet
* Matplotlib
* Regular Expressions (Regex)

---

## Project Workflow

### 1. Data Collection

Historical meal-consumption data is collected from multiple CSV files.

### 2. Data Preprocessing

* Extract dates from source files
* Handle inconsistent formats
* Aggregate meal statistics
* Create a unified forecasting dataset

### 3. Feature Preparation

The dataset is transformed into Prophet's required format:

| Column | Description          |
| ------ | -------------------- |
| ds     | Date                 |
| y      | Total meals required |

### 4. Model Training

A Prophet forecasting model is trained with:

* Daily seasonality
* Weekly seasonality
* Yearly seasonality

### 5. Forecast Generation

The model generates:

* 30-Day Forecast
* 365-Day Forecast

### 6. Visualization

The system automatically generates:

* Forecast trend plots
* Seasonal component plots
* Demand projections

---

## Results

The forecasting pipeline successfully:

* Automated meal-demand prediction
* Reduced manual reporting effort
* Generated short-term and long-term demand forecasts
* Identified seasonal consumption patterns
* Improved planning support for inventory and resource allocation

---

## Repository Structure

```text
├── data/
│   ├── data1.csv
│   ├── data2.csv
│   └── ...
├── meal_prediction.py
├── forecast_plot.png
├── components_plot.png
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/meal-forecasting.git
```

Move into the project directory:

```bash
cd meal-forecasting
```

Install dependencies:

```bash
pip install pandas prophet matplotlib
```

---

## Usage

Run the forecasting script:

```bash
python meal_prediction.py
```

The script will:

1. Load historical meal data
2. Preprocess and aggregate records
3. Train the Prophet model
4. Generate future forecasts
5. Save forecast visualizations

---

## Future Improvements

* Incorporate weather data
* Include holiday effects
* Add web dashboard visualization
* Deploy as a cloud-based forecasting service
* Compare Prophet with LSTM and ARIMA models

---

## Author

Ankita Mishra

B.Tech CSE | Minor in AI & Data Science

Passionate about Machine Learning, Predictive Analytics, and Data Science.
