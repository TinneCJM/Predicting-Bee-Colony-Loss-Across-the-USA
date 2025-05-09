## Introduction: Predicting Bee Colonie Loss for US states

Welcome to this comprehensive and hands-on journey where we explore the **loss of bee colonies** across varies US states using **forecasting** techniques. Various datasets are combined with the base bees data being collected from **Kaggle**, historical weather data from **Open Meteo API** and historical drought data from **National Integrated Drought Information System (NIDIS)**. The overall goal is to **collect, combine, clean, validate and build forecasting models** using a variety of machine learning models.

The notebooks are structered chronologically starting with data collection and cleaning, followed by validation and imputation of missing data, exploratory data analysis and finally, forecasting models.

---

## Notebooks overview

### ![alt text](bee.png) 1a Fetch weather data
- Perform a thorough **EDA (Exploratory Data Analysis)**.
- Assess the integrity of SMILES: **standardization, sanitization, neutralization**.
- Ensure there are **no duplicates or obvious data quality issues**.

### ![alt text](bee.png) 1b Process weather data
- Use **Tanimoto similarity** to evaluate whether molecules in training and validation/test sets are structurally similar.
- Recreate the split using **Bemis–Murcko scaffolds** to assess scaffold-based evaluation strategies.
- Explore the implications of split quality on **data leakage** and **true generalization**.

### ![alt text](bee.png) 1c process aggricultural data
- Explore the power of different **fingerprints and descriptors**: MACCS, Morgan, Mol2Vec, Mordred, Padel, and RDKit.
- Evaluate them individually and in combination to understand their unique and shared contributions to prediction performance.

### ![alt text](bee.png) 1d process drought data
- Compare the performance of **Linear Regression, Ridge, Lasso, SVR, Random Forest, LightGBM, CatBoost, and XGBoost**.
- Assess models across **raw**, **MinMax normalized**, and **standardized** feature sets.
- Tune hyperparameters and identify the most performant model configuration.

### ![alt text](bee.png) 2a combine datasets
- Understand **which features and descriptors matter most**.
- Explore why **descriptor-based features often outperform fingerprints**, and what this tells us about molecular representation.

### ![alt text](bee.png) 2b impute missing data
- Benchmark against a recent XGBoost implementation trained on the same dataset.
- Discuss key methodological differences such as **data splitting strategies** and **feature sets**.

### ![alt text](bee.png) 3 Exploratory data analysis
- Use **PCA** to visually assess the spread and overlap of training and test molecules.
- Gain insights into **chemical diversity** and how well your model might generalize in practice.

### ![alt text](bee.png) 4 Forecasting models
- Use **PCA** to visually assess the spread and overlap of training and test molecules.
- Gain insights into **chemical diversity** and how well your model might generalize in practice.

---
