from flask import Flask, request, jsonify
import joblib, numpy as np, os, json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

app = Flask(__name__)

# 🔹 Load trained models
models = {
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
    "Voting Ensemble": joblib.load("models/voting.pkl"),
    "AdaBoost": joblib.load("models/adaboost.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl")
}

# 🔹 Load scaler and feature columns
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

# 🔹 Load test data (optional)
X_test = joblib.load("models/X_test.pkl") if os.path.exists("models/X_test.pkl") else None
y_test = joblib.load("models/y_test.pkl") if os.path.exists("models/y_test.pkl") else None

# 🔹 Load precomputed metrics
MODEL_METRICS = {}
if os.path.exists("models/metrics.json"):
    with open("models/metrics.json", "r") as f:
        MODEL_METRICS = json.load(f)
else:
    if X_test is not None and y_test is not None:
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)

            MODEL_METRICS[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc_roc": roc_auc_score(y_test, y_prob),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }

# 🔹 Metrics endpoint
@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({"metrics": MODEL_METRICS})


# 🔹 Prediction endpoint
@app.route("/predict/all", methods=["POST"])
def predict_all():
    data = request.json

    # 🔹 Derived features
    data["followers_following_ratio"] = data["followers_count"] / (data["following_count"] + 1)
    data["engagement_score"] = data["posts_count"] / (data["followers_count"] + 1)
    data["posts_per_follower"] = data["posts_count"] / (data["followers_count"] + 1)
    data["followers_per_post"] = data["followers_count"] / (data["posts_count"] + 1)

    # 🔹 Build feature array
    features = np.array([[data[col] for col in feature_cols]]).astype(float)
    features_scaled = scaler.transform(features)

    results = {}
    votes_fake, votes_real = 0, 0

    # 🔹 Model predictions
    for name, model in models.items():
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        results[name] = {
            "prediction": int(pred),
            "fake_prob": float(prob[0]),
            "real_prob": float(prob[1])
        }

        if pred == 0:
            votes_fake += 1
        else:
            votes_real += 1

    # 🔹 Ensemble voting
    final_label = 0 if votes_fake > votes_real else 1
    final_decision = "Fake" if final_label == 0 else "Real"

    # 🔹 Rule-based override (with explanation)
    rule_triggered = None

    if data["profile_pic"] == 0 and data["bio_length"] == 0 and data["followers_count"] < 50 and data["following_count"] > 1000:
        final_label = 0
        final_decision = "Fake"
        rule_triggered = "Spam bot pattern (no pic, no bio, mass following)"

    elif data["posts_count"] == 0 and data["external_url"] == 1:
        final_label = 0
        final_decision = "Fake"
        rule_triggered = "Empty account with external link"

    elif data["followers_count"] > 10000 and data["posts_count"] < 5:
        final_label = 0
        final_decision = "Fake"
        rule_triggered = "High followers but no activity"

    # 🔹 Feature values
    explain_values = {
        col: float(val) for col, val in zip(feature_cols, features[0])
    }

    # 🔹 Feature importance (from models that support it)
    importance_list = []

    for model_name in ["Random Forest", "XGBoost", "AdaBoost"]:
        model = models.get(model_name)
        if hasattr(model, "feature_importances_"):
            importance_list.append(model.feature_importances_)

    # 🔹 Average importance
    if importance_list:
        avg_importance = np.mean(importance_list, axis=0)

        explain_importance = {
            col: float(score)
            for col, score in zip(feature_cols, avg_importance)
        }

        # ✅ 🔥 TOP FEATURES (YOUR FEATURE)
        top_features = sorted(
            explain_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

    else:
        explain_importance = {}
        top_features = []

    return jsonify({
        "all_predictions": results,
        "votes_fake": votes_fake,
        "votes_real": votes_real,
        "final_label": final_label,
        "final_decision": final_decision,
        "rule_triggered": rule_triggered,  # 👈 NEW (optional but powerful)
        "explain_prediction": {
            "values": explain_values,
            "importance": explain_importance,
            "top_features": top_features   # 👈 MAIN ADDITION
        }
    })


if __name__ == "__main__":
    app.run(debug=True)