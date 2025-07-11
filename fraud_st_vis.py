import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder \
    .appName('Fraud Visualization') \
    .getOrCreate()

# Load data
df = spark.read.csv('fraud_detection_data.csv', header=True, inferSchema=True)
pandas_df = df.toPandas()  # Convert Spark DataFrame to Pandas

# Define feature types
categorical_features = ['transaction_type', 'country', 'fraud', 'device_trusted', 'is_international']
numerical_features = ['amount', 'account_age_days', 'num_prev_transactions']

# Main title
st.title('📊 Fraud Detection Data Visualization')

# Sidebar: Analysis type
st.sidebar.header('Analysis Options')
analysis_type = st.sidebar.radio('Choose Analysis Type:', ['Univariate Analysis', 'Bivariate Analysis'])

# ────────────────────────────────
# 📌 UNIVARIATE ANALYSIS
# ────────────────────────────────
if analysis_type == 'Univariate Analysis':
    st.sidebar.subheader('Univariate Feature Selection')
    feature_type = st.sidebar.radio('Select Feature Type', ['Categorical', 'Numerical'])

    if feature_type == 'Categorical':
        selected_feature = st.sidebar.selectbox('Choose a Categorical Feature:', categorical_features)

        st.subheader(f"📊 Univariate Analysis for '{selected_feature}' (Categorical)")

        # Countplot and Pie chart side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Countplot
        sns.countplot(data=pandas_df, x=selected_feature, 
                      order=pandas_df[selected_feature].value_counts().index, ax=axes[0])
        axes[0].set_title('Countplot')
        axes[0].tick_params(axis='x', rotation=45)

        # Pie Chart
        pandas_df[selected_feature].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1],
                                                            startangle=90, shadow=True)
        axes[1].set_title('Pie Chart')
        axes[1].set_ylabel('')  # Hide y-label for pie chart

        st.pyplot(fig)
        plt.close(fig)

    else:
        selected_feature = st.sidebar.selectbox('Choose a Numerical Feature:', numerical_features)

        st.subheader(f"📊 Univariate Analysis for '{selected_feature}' (Numerical)")

        # Histogram
        st.write("#### Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data=pandas_df, x=selected_feature, kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        # Boxplot
        st.write("#### Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(data=pandas_df, x=selected_feature, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

# ────────────────────────────────
# 📌 BIVARIATE ANALYSIS
# ────────────────────────────────
else:
    st.sidebar.subheader('Bivariate Feature Selection')
    bivariate_type = st.sidebar.radio('Select Bivariate Type:', [
        'Numerical vs Numerical',
        'Numerical vs Categorical',
        'Categorical vs Categorical'
    ])

    # Numerical vs Numerical
    if bivariate_type == 'Numerical vs Numerical':
        x_num = st.sidebar.selectbox('X-axis (Numerical):', numerical_features)
        y_num = st.sidebar.selectbox('Y-axis (Numerical):', numerical_features)

        st.subheader(f"📈 Scatter Plot: {x_num} vs {y_num}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=pandas_df, x=x_num, y=y_num, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.write("#### 📊 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pandas_df[numerical_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Numerical vs Categorical
    elif bivariate_type == 'Numerical vs Categorical':
        num_feature = st.sidebar.selectbox('Numerical Feature:', numerical_features)
        cat_feature = st.sidebar.selectbox('Categorical Feature:', categorical_features)

        st.subheader(f"📊 Boxplot of '{num_feature}' grouped by '{cat_feature}'")
        fig, ax = plt.subplots()
        sns.boxplot(data=pandas_df, x=cat_feature, y=num_feature, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    # Categorical vs Categorical
    elif bivariate_type == 'Categorical vs Categorical':
        cat_feature1 = st.sidebar.selectbox('First Categorical Feature:', categorical_features)
        cat_feature2 = st.sidebar.selectbox('Second Categorical Feature:', categorical_features)

        st.subheader(f"📊 Stacked Bar Plot: '{cat_feature1}' vs '{cat_feature2}'")
        cross_tab = pd.crosstab(pandas_df[cat_feature1], pandas_df[cat_feature2])
        cross_tab.plot(kind='bar', stacked=True)
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.close()
