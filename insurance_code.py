import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

df = pd.read_csv('insurance.csv')

# Drop some columns
columns_to_drop = [
    'client_id', 'quotation_id', 'pay_subscription', 'pay_cancellation',
    'premium_subscription', 'premium_cancellation', 'broker_id',
    'guarantees_purchased', 'uninsured_vehicles', 'natural_events',
    'kasko', 'license_revoked', 'vandalism', 'key_loss' #'car_brand', 'car_model',  # Decision tree try.
]
df = df.drop(columns=columns_to_drop)

# Most frequent value for these
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['gender', 'county', 'operating_system']] = cat_imputer.fit_transform(df[['gender', 'county', 'operating_system']])

# Mean value for these
float_cols = ['driver_injury', 'legal_protection', 'waive_right_compensation',
              'protected_bonus', 'windows', 'theft_fire', 'collision']
float_imputer = SimpleImputer(strategy='mean')
df[float_cols] = float_imputer.fit_transform(df[float_cols])

# Median value for these
date_cols = ['car_immatriculation_date', 'insurance_expires_at', 'birth_date', 'policy_quoted_at', 'base_subscription']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    median = df[col].median()
    df[col].fillna(median, inplace=True)

# Create onw values and drop original dates
df['car_age'] = df['policy_quoted_at'].dt.year - df['car_immatriculation_date'].dt.year
df['client_age'] = df['policy_quoted_at'].dt.year - df['birth_date'].dt.year
df['days_until_expiry'] = (df['insurance_expires_at'] - df['policy_quoted_at']).dt.days
df['subscription_duration_days'] = (df['policy_quoted_at'] - df['base_subscription']).dt.days
df.drop(columns=date_cols, inplace=True)

# Categorical encoding
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le # Save encoders if need to inverse

# Features and target, Split the data
X = df.drop('issued', axis=1)
y = df['issued'].astype(int)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train models an evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))


# Best i got

model = RandomForestClassifier(class_weight={0:1, 1:3}, random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

threshold = 0.35
preds = (probs >= threshold).astype(int)

best_rf = RandomForestClassifier(
    bootstrap=False,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]  

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))
