# Data Science Thesis Project May 2025

This project aims to forecast the loss of bee colonies over time.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for project evaluation.

### Prerequisites

What things you need to install the software and how to install them:

- Anaconda or Miniconda
- Python 3.11

### Environment Setup

A step by step explanation to get the provided notebook and environment running:

1. Clone the repo: `git clone https://github.com/TinneCJM/Bees.git`
2. Navigate to the project directory: `cd your_project`
3. Install the provided conda environment: `conda env create -f environment.yaml`
4. Activate the new environment: `conda activate your_env_name`
5. Open the Jupyter notebook: `jupyter notebook your_notebook.ipynb`

**Note:** Replace `your_env_name` with the name of your conda environment, and `your_notebook.ipynb` with the name of your Jupyter notebook.


## Introduction: Predicting Bee Colonie Loss for US states

Welcome to this comprehensive and hands-on journey where we explore the **loss of bee colonies** across varies US states using **forecasting** techniques. The bees dataset is collected from two surveys per year targeting beekeepers who manage five or more bee colonies. These surveys gathered information on number of colonies, colony losses, colony additions and bee health stressors. This data from 2015 up to 2022 is available on **Kaggle**. Weather and drought data is combined with the base bees data: historical weather data from **Open Meteo API** and historical drought data from **National Integrated Drought Information System (NIDIS)**. The overall goal is to **collect, combine, clean, validate and build forecasting models** using a variety of machine learning models.

The notebooks are structered chronologically starting with data collection and cleaning, followed by validation and imputation of missing data, exploratory data analysis and finally, forecasting models.

---

## Notebooks overview

###  🐝 1a Fetch weather data
- Extract daily weather data through **Open Meteo API**.
- Calls are made on US state latitude/longitude combinations provided by ChatGPT.
- API calls are performed in chunks over time due to API call restrictions.

### 🐝 1b Process weather data
- Daily weather data is **condensed to quarterly** data.
- Weather feature aggregations have been made.

### 🐝 1c process drought data
- Weekly drought data is **condensed to quarterly** data.
- Drought feature aggregations have been made.

### 🐝 2a combine datasets
- Weather, drought and bees data is combined.
- Checks are made for missing data.

### 🐝 2b impute missing data
- Bee related data for quarter 2 from 2019 has to be **imputed**.
- Full dataset is validated for missing data.

### 🐝 3 Exploratory data analysis
- The main feature of interest is explored into detail.
- Correlations to feature of interest are investigated.
- Various features are plotted per quarter.

### 🐝 4 Forecasting models
- Time Series Forecasting is done both in a **static and walk-forward** manner.
- RMSE and MAE are evaluated for various models
- Classical ML models with lag features and forecasting models are used.

---

