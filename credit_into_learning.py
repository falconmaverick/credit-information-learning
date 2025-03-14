import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

"""
Home Credit Default Risk Competition Details:

1. What to learn and what to predict:
   - The goal is to predict the probability of a client defaulting on a loan.
   - The dataset contains information about clients' financial history and demographics.
   - We will train a machine learning model to determine how likely a client is to repay a loan.

2. What kind of file should be created and submitted to Kaggle?
   - The submission file should be a CSV with two columns:
     - SK_ID_CURR: The unique ID of each client.
     - TARGET: The predicted probability of loan default.

3. How will submissions be evaluated?
   - Submissions are evaluated based on the Area Under the ROC Curve (AUC-ROC), 
     which measures the ability of the model to distinguish between defaulters and non-defaulters.
"""

# Load dataset
df = pd.read_csv("train.csv")

# Select relevant features
df_selected = df[['SK_ID_CURR', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'TARGET']]

# Feature Engineering
df_selected['DAYS_EMPLOYED_PERC'] = df_selected['DAYS_EMPLOYED'] / df_selected['DAYS_BIRTH']
df_selected['INCOME_CREDIT_RATIO'] = df_selected['AMT_INCOME_TOTAL'] / df_selected['AMT_CREDIT']

# Define features and target
X = df_selected[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_RATIO']]
y = df_selected['TARGET']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict_proba(X_test)[:, 1]

# Compute AUC-ROC
auc_lr = roc_auc_score(y_test, y_pred_lr)
auc_rf = roc_auc_score(y_test, y_pred_rf)

# Store results in a table
auc_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "AUC-ROC": [auc_lr, auc_rf]
})

print("Model Performance:")
print(auc_results)

# Visualize feature importance from Random Forest
plt.figure(figsize=(8, 6))
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(7).plot(kind='barh')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest")
plt.show()

# Load test dataset
test_df = pd.read_csv("application_test.csv")
test_df['DAYS_EMPLOYED_PERC'] = test_df['DAYS_EMPLOYED'] / test_df['DAYS_BIRTH']
test_df['INCOME_CREDIT_RATIO'] = test_df['AMT_INCOME_TOTAL'] / test_df['AMT_CREDIT']
X_test_final = test_df[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_RATIO']]

# Predict probabilities using the best model (Random Forest)
test_predictions = rf_model.predict_proba(X_test_final)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "SK_ID_CURR": test_df["SK_ID_CURR"],
    "TARGET": test_predictions
})

# Save submission file
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
