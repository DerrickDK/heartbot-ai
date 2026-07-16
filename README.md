# **Heart Disease Prediction and Support Chatbot**
This project is decicated to MSAI-699 Capstone and aims to build an AI-powered heart disease prediction and patient support chatbot using Gradio, Hugging Face, and Groq. The goal is to provide a user friendly interface for users to input their data and recieve predictions about the presense of heart disease and a chatbot to offer patient support and guidance for wellness.


# **Roadmap**

## [**Phase 1**](heart_disease_dataset_exploration.ipynb):
- Gathered a Cleveland UCI [heart disease dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/code) for pre-processing and EDA.
- Extracted relevant features from dataset and encoded categorical variables for future model training.

## [**Phase 2**](heart_disease_dataset_model_training.ipynb):
- Split the dataset into training and testing sets using the train_test_split function from scikit-learn.
- Implemented a baseline model using logistic regression.
- Trained the baseline model on the training set.
- Evaluated the performance of the baseline model on the testing set using various metrics such as accuracy, precision, recall, F1 score, and ROC AUC score.
- Generated a confusion matrix to visualize the model's performance.
- Plotted the ROC curve to assess the model's ability to distinguish between positive and negative classes.
- Observed the impact of applying SMOTE on the model's performance and made decision use in later phases to balance and improve evaluation metrics of model optimization.