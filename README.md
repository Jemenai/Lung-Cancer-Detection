Lung Cancer Risk Prediction

![Anatomy of the Lungs](licensed-image.jpg)
This project aims to develop a robust machine learning system to predict the risk of lung cancer in patients based on a range of demographic, lifestyle, and medical history attributes. The ultimate goal is to create an accessible tool for early-risk assessment, deployed as both a web and mobile application.

This project is undertaken as part of the DataLab Analytics Final Project.

📋 Table of Contents
Project Overview

Problem Statement

Dataset Description

Project Workflow

Models Implemented

Technologies & Libraries

Project Structure

How to Use

Results & Performance

Future Work

License

Contact

📝 Project Overview
As a Data Scientist in a Health Tech Startup, I was tasked with training machine learning algorithms on a dataset of 5,000 patient records to develop a model that can identify patterns and correlations between risk factors and the likelihood of developing lung cancer. This repository contains the complete workflow, from data exploration and preprocessing to model training, evaluation, and preparation for deployment. The project culminates in a Streamlit web application and a React Native mobile app that leverage the best-performing model to provide probability-based predictions for new patients, enabling early intervention and personalized risk assessment.

🎯 Problem Statement
The objective is to classify whether a patient is at risk of developing lung cancer based on 18 predictive features. This is a binary classification problem where the model will output a probability score indicating the risk level. The key challenge is to build a highly accurate and reliable model by exploring a diverse set of machine learning and deep learning algorithms.

🗂️ Dataset Description
The project utilizes a comprehensive dataset of 5,000 records, each containing 18 features that capture critical lung cancer risk factors.

Target Variable: PULMONARY_DISEASE - A binary indicator of the presence or risk of lung cancer.

Features:

Feature Name

Description

Relevance to Lung Cancer Risk

AGE

Age of the patient (in years).

Older age is a known risk factor.

GENDER

Gender of the patient.

Incidence rates can differ by gender.

SMOKING

Smoking status of the patient.

Smoking is the leading cause of lung cancer.

ALCOHOL

Alcohol consumption history.

Heavy alcohol use can increase risk.

FAMILY_HISTORY

Family history of lung cancer.

Genetic predisposition is a known risk factor.

...

(Other features as listed in the dataset)

(Relevance of other features)

(You can complete the table above with all 18 features from your dataset description file.)

⚙️ Project Workflow
The project is structured into the following key phases:

Data Preparation & Preprocessing:

Loading the dataset and performing initial inspections.

Checking for and handling inconsistencies like incorrect data types, duplicates, and outliers.

Encoding categorical features into a numerical format suitable for machine learning models.

Scaling numerical features to a standard range to ensure fair contribution from all variables.

Splitting the data into training (80%) and testing (20%) sets to prevent data leakage.

Exploratory Data Analysis (EDA):

Conducting univariate and bivariate analysis to understand feature distributions and relationships.

Using visualizations like histograms, box plots, and correlation heatmaps to uncover patterns and insights.

Model Building & Training:

Implementing and training nine different classification algorithms to comprehensively explore modeling techniques.

The models range from traditional machine learning classifiers to more complex deep learning architectures.

Model Evaluation:

Assessing the performance of each model on the unseen test set using standard classification metrics:

Accuracy: Overall correctness of predictions.

Precision: Ability of the model to identify only relevant instances.

Recall (Sensitivity): Ability of the model to find all relevant instances.

F1-Score: The harmonic mean of Precision and Recall.

ROC-AUC Score: Measure of the model's ability to distinguish between classes.

Model Selection & Deployment:

Comparing the evaluation metrics to select the single best-performing model.

Saving the trained model (.pkl or .joblib file) for future use.

Developing a user-friendly Streamlit web application to serve the model.

Building a simple React Native mobile application that consumes the model's prediction API.

🧠 Models Implemented
A diverse range of algorithms was used to ensure a thorough and robust analysis.

Machine Learning Models:
Logistic Regression: A baseline linear model for binary classification.

K-Nearest Neighbors (KNN): A non-parametric algorithm that classifies based on the majority class of its 'k' nearest neighbors.

Random Forest: An ensemble method using multiple decision trees to improve prediction accuracy and control over-fitting.

XGBoost (Extreme Gradient Boosting): A powerful and efficient gradient boosting framework known for its high performance.

Linear Regression: While typically used for regression, it was explored to provide a complete analytical perspective. Its coefficients can offer insights but it's not suitable for the final classification task.

Deep Learning Models:
LSTM (Long Short-Term Memory): A type of Recurrent Neural Network (RNN) capable of learning long-term dependencies, often used in sequence data.

BiLSTM (Bidirectional LSTM): An extension of LSTM that processes data in both forward and backward directions, capturing more context.

GRU (Gated Recurrent Unit): A simpler variant of LSTM that often performs comparably with less computational overhead.

Attention Model: Utilizes tf.keras.layers.MultiHeadAttention to allow the model to focus on the most relevant parts of the input data when making predictions.

💻 Technologies & Libraries
Language: Python 3.9

Data Analysis & ML: Pandas, NumPy, Scikit-learn, XGBoost

Deep Learning: TensorFlow, Keras

Data Visualization: Matplotlib, Seaborn

Web App: Streamlit

Mobile App: React Native

Environment: Jupyter Notebook, VS Code

📁 Project Structure
.
├── data/
│   └── lung_cancer_data.csv
├── notebooks/
│   └── Lung_Cancer_Risk_Prediction.ipynb
├── models/
│   └── best_model.joblib
├── app/
│   └── app.py
├── mobile_app/
│   ├── App.js
│   └── ... (React Native project files)
├── README.md
└── requirements.txt

🚀 How to Use
To replicate this project on your local machine, follow these steps:

Clone the repository:

git clone [https://github.com/your-username/lung-cancer-prediction.git](https://github.com/your-username/lung-cancer-prediction.git)
cd lung-cancer-prediction

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Launch the Jupyter Notebook:

jupyter notebook notebooks/Lung_Cancer_Risk_Prediction.ipynb

Run the Streamlit Web Application:

streamlit run app/app.py

📊 Results & Performance
(This section is a template. You should fill it with your final results after running the notebook.)

The performance of all nine models was evaluated on the test set. The results are summarized below:

Model

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Logistic Regression

-

-

-

-

-

K-Nearest Neighbors

-

-

-

-

-

Random Forest

-

-

-

-

-

XGBoost

-

-

-

-

-

LSTM

-

-

-

-

-

BiLSTM

-

-

-

-

-

GRU

-

-

-

-

-

Attention Model

-

-

-

-

-

Conclusion: The [Your Best Model, e.g., XGBoost] model was selected as the best-performing model due to its superior balance of accuracy and F1-score. This model will be used for the final deployment.

💡 Future Work
Hyperparameter Tuning: Conduct a more extensive hyperparameter search (e.g., using GridSearch or Optuna) for the top models to further boost performance.

Feature Engineering: Create new features from existing ones to potentially improve model accuracy.

Model Interpretability: Use tools like SHAP or LIME to explain the predictions of the best model, providing more transparency.

Expand Deployment: Deploy the model API on a cloud service (like AWS or GCP) to create a more scalable backend for the mobile application.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

📞 Contact
Ifeoma Okonma - [Your Email] - [Your LinkedIn Profile URL]

Project Link: https://github.com/your-username/lung-cancer-prediction