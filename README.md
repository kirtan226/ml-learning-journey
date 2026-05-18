# 🤖 ML Learning Journey

This repository contains my hands-on practice projects while learning **Machine Learning with Python**.

I am building this as part of my journey from a **Python Backend Developer (Django/DRF)** toward integrating **AI/ML into backend systems**. The repo includes small, focused examples for understanding model training, prediction, evaluation, and working with real-world datasets.

## 📁 Repository Structure

```text
ml-learning-journey/
|-- Supervised learning/
|   |-- classification/
|   |   |-- Decision Trees/
|   |   |-- K-Nearest Neighbors/
|   |   |-- Logistic Regression/
|   |   |-- Random Forest/
|   |   `-- Support Vector Machine(SVM)/
|   |-- regression/
|   |   `-- Polynomial Regression/
|   |-- Problems/
|   `-- files/
|-- Unsupervised learning/
|   |-- Clustering Models/
|   |   |-- DBSCAN model/
|   |   |-- Hierarchical Clustering/
|   |   `-- K-Means clustering/
|   `-- files/
|-- requirements.txt
`-- main.py
```

## 🚀 Topics Covered

### Supervised Learning

#### Regression
- Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- House price prediction
- Car price prediction
- Salary prediction
- Diabetes prediction
- Advertising sales prediction
- Car speed vs fuel efficiency prediction

#### Classification
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)
- Iris classification
- Student pass/fail prediction
- Customer purchase prediction
- Spam detection
- Customer churn prediction
- Employee attrition prediction
- Titanic survival prediction

### Unsupervised Learning

#### Clustering
- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- Mall customer segmentation
- Credit card customer segmentation
- Wholesale customer segmentation
- Customer location clustering
- DNA clustering problem

## 🗃️ Datasets

Datasets used for practice are stored inside:

- `Supervised learning/files/`
- `Unsupervised learning/files/`

These include CSV and Excel datasets for regression, classification, and clustering examples.

## 🧪 How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/ml-learning-journey.git
cd ml-learning-journey
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run any script from the project root. Examples:

```bash
python "Supervised learning/classification/Logistic Regression/student_pass_fail.py"
python "Supervised learning/regression/car_price_prediction.py"
python "Unsupervised learning/Clustering Models/K-Means clustering/mall_customer_segmentation.py"
```

## 📊 Requirements

Main libraries used in this repository:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenPyXL
- Joblib

See `requirements.txt` for the full dependency list.

## ✔ Purpose

This repository is for learning and practice. Each script is focused on understanding one concept or model workflow, such as:

- Loading datasets
- Cleaning and preparing data
- Splitting train/test data
- Training ML models
- Making predictions
- Evaluating model performance
- Visualizing results where applicable

## ⭐ Status

This is an active learning repository. I will continue adding more machine learning models, datasets, notebooks, experiments, and backend integration examples as I progress.
