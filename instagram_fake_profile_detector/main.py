# main.py
import pandas as pd, numpy as np, pickle, os, warnings, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

if not os.path.exists('models'):
    os.makedirs('models')

# 🔹 Load and clean dataset
df = pd.read_csv("data/instagram_fake_profile.csv", encoding="utf-8", on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 🔹 Feature Engineering
df['followers_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
df['engagement_score'] = df['posts_count'] / (df['followers_count'] + 1)
df['bio_length'] = df.get('bio_description_length', 0)
df['posts_per_follower'] = df['posts_count'] / (df['followers_count'] + 1)
df['followers_per_post'] = df['followers_count'] / (df['posts_count'] + 1)

# 🔹 Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['fake_or_real'].astype(str))

FEATURE_COLS = [
    'profile_pic','username_length','bio_length','external_url','is_private',
    'posts_count','followers_count','following_count',
    'followers_following_ratio','engagement_score',
    'posts_per_follower','followers_per_post'
]

X = df[FEATURE_COLS].fillna(0)
y = df['label']

# 🔹 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Base Models (tuned)
models = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, eval_metric='logloss', random_state=42, verbosity=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.8, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# 🔹 Stacking Ensemble (upgrade from Voting)
ensemble = StackingClassifier(
    estimators=[('nb', models["Naive Bayes"]),
                ('rf', models["Random Forest"]),
                ('xgb', models["XGBoost"]),
                ('ada', models["AdaBoost"])],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method='predict_proba',
    passthrough=True,
    n_jobs=1
).fit(X_train_scaled, y_train)

# 🔹 Evaluate all models + ensemble
MODEL_METRICS = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:,1]
    MODEL_METRICS[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "fpr": roc_curve(y_test, y_prob)[0].tolist(),
        "tpr": roc_curve(y_test, y_prob)[1].tolist()
    }

# Ensemble metrics
y_pred = ensemble.predict(X_test_scaled)
y_prob = ensemble.predict_proba(X_test_scaled)[:,1]
MODEL_METRICS["Voting Ensemble"] = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_prob),
    "fpr": roc_curve(y_test, y_prob)[0].tolist(),
    "tpr": roc_curve(y_test, y_prob)[1].tolist()
}

# 🔹 Save metrics to JSON file
with open("models/metrics.json", "w") as f:
    json.dump(MODEL_METRICS, f, indent=4)

# 🔹 Save models and test data
pickle.dump(ensemble, open("models/voting.pkl","wb"))  # same filename so Flask app works
pickle.dump(scaler, open("models/scaler.pkl","wb"))
pickle.dump(FEATURE_COLS, open("models/feature_cols.pkl","wb"))
pickle.dump(X_test_scaled, open("models/X_test.pkl","wb"))
pickle.dump(y_test, open("models/y_test.pkl","wb"))
for name, model in models.items():
    pickle.dump(model, open(f"models/{name.lower().replace(' ','_')}.pkl","wb"))