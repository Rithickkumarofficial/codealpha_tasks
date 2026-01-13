import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Load dataset
df = pd.read_csv("data/german_credit.csv")

# Clean target
df["credit_risk"] = df["credit_risk"].astype(str).str.strip().str.lower()
df["credit_risk"] = df["credit_risk"].map({"good": 0, "bad": 1})

# Split X and y
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

# Feature types
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Preprocessor
numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocess
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Train model
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = rf.predict(X_test_processed)
y_prob = rf.predict_proba(X_test_processed)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

# Save model
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("Saved model to models/random_forest.pkl")
print("Saved preprocessor to models/preprocessor.pkl")

