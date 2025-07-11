import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

df = pd.read_csv('fraud_detection_data.csv')

# Label Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
print("Categorical columns encoded:", list(categorical_cols))

# seperating the data into features and target variable
X = df.drop('fraud', axis=1)
y = df['fraud']

# scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))
print('Confusion Matrix', confusion_matrix(y_test, y_pred))

# smote
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("Resampled class distribution:", y_resampled.value_counts())

model.fit(X_resampled, y_resampled)
y_pred_resampled = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_resampled))
print("Precision:", precision_score(y_test, y_pred_resampled))
print("Classification Report:", classification_report(y_test, y_pred_resampled))
print('Confusion Matrix', confusion_matrix(y_test, y_pred_resampled))


# xgboost 
from xgboost import XGBClassifier

model = XGBClassifier(
    scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_resampled, y_resampled)

print("Accuracy:", accuracy_score(y_test, y_pred_resampled))
print("Precision:", precision_score(y_test, y_pred_resampled))
print("Classification Report:", classification_report(y_test, y_pred_resampled))
print('Confusion Matrix', confusion_matrix(y_test, y_pred_resampled))
# # Save the model and encoders
# joblib.dump(model, 'fraud_detection_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(encoders, 'label_encoders.pkl')



