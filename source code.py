import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import warnings

warnings.filterwarnings('ignore')

# Load dataset
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Data Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Standardize text values
cols_to_fix = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols_to_fix:
    df[col] = df[col].replace('No internet service', 'No')

# Drop unique identifiers and encode categories
df = df.drop('customerID', axis=1)
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split into 80/20 train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fix class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Generate predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate performance
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Global explanations (SHAP)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Local explanations (LIME)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['No Churn', 'Churn'],
    mode='classification',
    random_state=42
)

# Explain a single high-risk customer
customer_idx = 5
explanation = lime_explainer.explain_instance(
    X_test.iloc[customer_idx].values, 
    xgb_model.predict_proba, 
    num_features=5
)
explanation.show_in_notebook(show_table=True)

# Segment customers by risk level for business action
results = X_test.copy()
results['Actual'] = y_test
results['Prob'] = y_pred_proba

def get_risk_level(p):
    if p >= 0.7: return 'High Risk'
    if p >= 0.4: return 'Medium Risk'
    return 'Low Risk'

results['Risk_Segment'] = results['Prob'].apply(get_risk_level)
results.to_csv('Churn_Priority_List.csv', index=False)
