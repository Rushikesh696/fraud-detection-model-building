# Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent transactions using machine learning techniques. It covers the full pipeline from exploratory data analysis (EDA), visualization, to building robust classification models capable of handling class imbalance effectively.

We used PySpark for big data preprocessing and Scikit-learn/XGBoost for building the fraud detection models. The final solution also includes a Streamlit dashboard for interactive data exploration.

Features
### Data Exploration and Cleaning

Schema inspection, class balance check, null/missing value handling

Duplicate detection and unique value counts

### Interactive Visualizations (Streamlit)

Univariate and Bivariate analysis

Histograms, boxplots, scatter plots, and heatmaps

Categorical feature comparisons with pie charts and stacked bar charts

### Machine Learning Models

Random Forest Classifier (with class_weight='balanced_subsample')

XGBoost Classifier with scale_pos_weight for class imbalance

SMOTE Oversampling for better minority class representation

### Model Evaluation

Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Comparison before and after applying SMOTE

### Model Persistence

Models and encoders saved using joblib for reuse

### Tech Stack
PySpark: Data preprocessing and EDA on large datasets

Pandas/Matplotlib/Seaborn: Visualization and data manipulation

Scikit-learn: Random Forest, preprocessing, evaluation

XGBoost: Gradient boosting classifier for imbalanced data

imblearn: SMOTE (Synthetic Minority Oversampling Technique)

Streamlit: Interactive data visualization app

### Conclusion
During this project, we identified that the fraud detection dataset was highly imbalanced, with very few fraudulent transactions compared to non-fraudulent ones.

Although Random Forest and XGBoost classifiers were trained (with techniques like class weights and SMOTE to handle imbalance), the models still showed bias toward the majority class. This made it challenging for the system to accurately classify both classes effectively.

Due to this limitation, the project could not be extended further into a production-ready deployment pipeline. Resolving such imbalance would require either a larger dataset with more fraud cases or advanced anomaly detection methods.

### Future Improvements
Experiment with anomaly detection techniques like Isolation Forest and Autoencoders.

Collect or synthesize more minority class data to improve model performance.

Integrate the model into a real-time fraud detection API once class imbalance is addressed.
