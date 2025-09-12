import streamlit as st
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter

from datetime import datetime
# =========================
# CONFIGURATION
# =========================
DATA_DIR = "app_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
SCALER_FILE = "scaler.pkl"  # your saved scaler file

MODEL_FILES = {
    "XGBoost": "models/xgboost_model.joblib",
    "LSTM": "models/best_lstm_model.keras",
    "GRU": "models/best_gru_model.keras",
    "MHA": "models/best_mha_model.keras",
    "LinearRegression": "models/linear_regression_model.joblib",
    "LogisticRegression": "models/logistic_regression_model.joblib",
    "RandomForest": "models/random_forest_model.joblib",
    "KNN": "models/k_nearest_neighbors_model.joblib"
}

FULL_FEATURES = [
    "SMOKING",
    "BREATHING_ISSUE",
    "FAMILY_HISTORY_x_SMOKING_FAMILY_HISTORY",
    "SMOKING_x_SMOKING_FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY",
    "THROAT_DISCOMFORT",
    "ENERGY_LEVEL",
    "STRESS_IMMUNE_x_MENTAL_STRESS",
    "STRESS_IMMUNE",
    "IMMUNE_WEAKNESS_x_STRESS_IMMUNE",
    "FAMILY_HISTORY",
    "IMMUNE_WEAKNESS",
    "EXPOSURE_TO_POLLUTION",
    "OXYGEN_SATURATION",
    "AGE",
    "OXYGEN_LEVEL",
    "AGE_GROUP",
    "GENDER",
    "CHEST_TIGHTNESS",
    "ALCOHOL_CONSUMPTION",
    "GENDER_CATEGORY",
    "FINGER_COLORATION",
    "MENTAL_STRESS",
    "LONG_TERM_ILLNESS"
]


# Features: update with your real training columns
FEATURES = [
    "SMOKING",
    "BREATHING_ISSUE",
    "FAMILY_HISTORY_x_SMOKING_FAMILY_HISTORY",
    "SMOKING_x_SMOKING_FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY",
    "THROAT_DISCOMFORT",
    "ENERGY_LEVEL",
    "STRESS_IMMUNE_x_MENTAL_STRESS",
    "STRESS_IMMUNE",
    "IMMUNE_WEAKNESS_x_STRESS_IMMUNE",
    "FAMILY_HISTORY",
    "IMMUNE_WEAKNESS",
    "EXPOSURE_TO_POLLUTION",
    "OXYGEN_SATURATION",
    "AGE"
]

FEATURE_TYPES = {
    "SMOKING": "binary",
    "BREATHING_ISSUE": "binary",
    "FAMILY_HISTORY_x_SMOKING_FAMILY_HISTORY": "binary",
    "SMOKING_x_SMOKING_FAMILY_HISTORY": "binary",
    "SMOKING_FAMILY_HISTORY": "binary",
    "THROAT_DISCOMFORT": "binary",
    "ENERGY_LEVEL": "numeric",
    "STRESS_IMMUNE_x_MENTAL_STRESS": "binary",
    "STRESS_IMMUNE": "binary",
    "IMMUNE_WEAKNESS_x_STRESS_IMMUNE": "binary",
    "FAMILY_HISTORY": "binary",
    "IMMUNE_WEAKNESS": "binary",
    "EXPOSURE_TO_POLLUTION": "binary",
    "OXYGEN_SATURATION": "numeric",
    "AGE": "numeric"
}

LABEL_MAP = {
    "FAMILY_HISTORY_x_SMOKING_FAMILY_HISTORY": "FH x Smoking FH",
    "SMOKING_x_SMOKING_FAMILY_HISTORY": "Smoking x Smoking FH",
    "STRESS_IMMUNE_x_MENTAL_STRESS": "StressImm x MentalS",
    "IMMUNE_WEAKNESS_x_STRESS_IMMUNE": "ImmWeak x StressImm",
    "OXYGEN_SATURATION": "O2 Sat",
    "EXPOSURE_TO_POLLUTION": "Pollution Exp",
    "BREATHING_ISSUE": "Breathing",
    "THROAT_DISCOMFORT": "Throat Discomf",
    "ENERGY_LEVEL": "EnergyLvl",
    "IMMUNE_WEAKNESS": "Immune Weak",
    "STRESS_IMMUNE": "Stress Imm",
    "SMOKING_FAMILY_HISTORY": "Smoking FH",
    "FAMILY_HISTORY": "FH",
    "SMOKING": "Smoking",
    "AGE": "Age"
}



# Example metrics (replace with your actual)
MODEL_METRICS = {
    "XGBoost": {"accuracy": 0.91, "f1": 0.89},
    "LSTM": {"accuracy": 0.88, "f1": 0.87},
    "GRU": {"accuracy": 0.87, "f1": 0.86},
    "MHA": {"accuracy": 0.89, "f1": 0.88},
    "LinearRegression": {"accuracy": 0.72, "f1": 0.70},
    "LogisticRegression": {"accuracy": 0.84, "f1": 0.82},
    "RandomForest": {"accuracy": 0.90, "f1": 0.89},
    "KNN": {"accuracy": 0.80, "f1": 0.78}
}

ADMIN_USERS = ["admin@example.com"]

def get_label(name):
    """Return acronym if defined, else full name."""
    return LABEL_MAP.get(name, name)


def to_serializable(val):
    """Convert numpy types to plain Python types."""
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

from sklearn.linear_model import LinearRegression
import numpy as np

class LinearRegressionClassifier:
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        continuous_predictions = self.model.predict(X)
        return (continuous_predictions > 0.5).astype(int)

    def predict_proba(self, X):
        probs = self.model.predict(X)
        probs = np.clip(probs, 0, 1)
        return np.vstack([1 - probs, probs]).T

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self


# =========================
# UTILITIES
# =========================
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # reset to empty history if file is invalid
        return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)  # default=str handles stray non-serializable types

def load_scaler():
    with open(SCALER_FILE, "rb") as f:
        return joblib.load(f)

def load_model(model_file):
    if model_file.endswith(".joblib"):
        return joblib.load(model_file)
    elif model_file.endswith(".h5") or model_file.endswith(".keras"):
        return tf.keras.models.load_model(model_file)
    else:
        raise ValueError(f"Unsupported model format: {model_file}")

def preprocess_input(inputs):
    # Step 1: Create a placeholder for all 24 features
    arr_full = np.zeros(len(FULL_FEATURES))

    # Step 2: Map collected inputs into their correct positions
    for feat, val in zip(FEATURES, inputs):   # FEATURES = your 15 selected ones
        if feat in FULL_FEATURES:
            idx = FULL_FEATURES.index(feat)
            arr_full[idx] = val

    arr_full = arr_full.reshape(1, -1)

    # Step 3: Scale with saved scaler (trained on all 24 features)
    scaler = joblib.load("scaler.pkl")
    arr_scaled = scaler.transform(arr_full)

    # Step 4: Select only the features you want to pass to the model
    selected_indices = [FULL_FEATURES.index(f) for f in FEATURES]
    arr_selected = arr_scaled[:, selected_indices]

    return arr_selected

def predict_with_model(model_name, model, X):
    try:
        # Handle special case for RNNs (expecting 3D input)
        if model_name in ["LSTM", "GRU", "MHA"]:
            X_input = X.reshape((X.shape[0], 1, X.shape[1]))  # (batch, timesteps=1, features)
            y_pred = model.predict(X_input)
            y_pred = y_pred.reshape(-1)  # flatten back to 2D-like output
        else:
            y_pred = model.predict(X)

        # For classifiers that output probabilities
        if hasattr(y_pred, "__len__") and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_class = np.argmax(y_pred, axis=1)[0]
        else:
            y_pred_class = int(y_pred[0] > 0.5) if hasattr(y_pred, "__len__") else int(y_pred > 0.5)

        return y_pred_class, y_pred
    except Exception as e:
        st.error(f"Prediction failed for {model_name}: {e}")
        return None, None


def predict(model, inputs):
    try:
        # If using the custom LinearRegressionClassifier
        if hasattr(model, "predict_proba") or hasattr(model, "predict"):
            pred = model.predict(inputs)
        else:
            pred = model.predict(inputs)

        if isinstance(pred, np.ndarray) and pred.ndim > 1:
            return int(pred.argmax(axis=1)[0])
        return int(round(pred[0]))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


# =========================
# AUTH
# =========================
def register_user(email, password, name):
    users = load_users()
    if email in users:
        return False, "User already exists."
    users[email] = {"password": password, "name": name, "profile": {"age": None}}
    save_users(users)
    return True, "Registration successful."

def authenticate_user(email, password):
    users = load_users()
    if email in users and users[email]["password"] == password:
        return True, users[email]
    return False, None

def update_profile(email, profile_data):
    users = load_users()
    if email in users:
        users[email]["profile"].update(profile_data)
        save_users(users)

# =========================
# STREAMLIT APP
# =========================
def main():
#     st.markdown("""
#     <style>
#         .stApp {
#             background-color: #f9f9f9;
#         }
#         .css-1d391kg { 
#             max-width: 900px; 
#             margin: auto;
#         }
#         .stNumberInput, .stSelectbox {
#             margin-bottom: 12px;
#         }
#     </style>
# """, unsafe_allow_html=True)
    
    ensure_data_dir()
    st.title("🫁 Lung Cancer Prediction App")

    if "user" not in st.session_state:
        st.session_state.user = None

    menu = ["Login", "Register"] if not st.session_state.user else ["Dashboard", "Profile", "History", "Logout"]
    if st.session_state.user and st.session_state.user["email"] in ADMIN_USERS:
        menu.insert(-1, "Admin")

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        st.subheader("Register")
        email = st.text_input("Email")
        name = st.text_input("Name")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            success, msg = register_user(email, password, name)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    elif choice == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            ok, user = authenticate_user(email, password)
            if ok:
                st.session_state.user = {"email": email, "name": user["name"]}
                st.rerun()
            else:
                st.error("Invalid login credentials.")

    elif choice == "Dashboard":
        st.subheader(f"Welcome, {st.session_state.user['name']}")

        # Patient Questionnaire
        st.subheader("Patient Questionnaire")
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        inputs = []
        for idx, feat in enumerate(FEATURES):
            with cols[idx % 3]:
                label = get_label(feat)  # ✅ Use acronym/shortened label
                if FEATURE_TYPES[feat] == "numeric":
                    val = st.number_input(
                        label, 
                        min_value=0, 
                        value=0, 
                        help=feat  # full name as tooltip
                    )
                else:
                    val = st.selectbox(
                        label, 
                        [0, 1], 
                        index=0, 
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        help=feat  # full name as tooltip
                    )
                inputs.append(val)

        algo_choice = st.selectbox("Choose Algorithm", ["All"] + list(MODEL_FILES.keys()))
        if st.button("Predict"):
            scaled_inputs = preprocess_input(inputs)
            predictions = {}

            st.write("### Predictions")

            if algo_choice == "All":
                predictions = {}
                for model_name, model_file in MODEL_FILES.items():
                    model = load_model(model_file)
                    y_class, y_pred = predict_with_model(model_name, model, scaled_inputs)  # ✅ FIXED
                    if y_class is not None:
                        predictions[model_name] = {"class": y_class, "raw": y_pred}
                # handle consensus voting here if needed

                # Consensus (majority)
                if len(predictions) > 1:
                    class_votes = [v["class"] for v in predictions.values()]
                    consensus = Counter(class_votes).most_common(1)[0][0]
                    avg_acc = np.mean([
                        MODEL_METRICS.get(name, {}).get("accuracy", 0)
                        for name in predictions if name in MODEL_METRICS
                    ])
                    predictions["Consensus"] = {
                        "class": consensus,
                        "raw": class_votes,
                        "accuracy": round(avg_acc, 2)
                    }

            else:
                model_name = algo_choice
                model = load_model(MODEL_FILES[model_name])
                y_class, y_pred = predict_with_model(model_name, model, scaled_inputs)  # ✅ FIXED
                if y_class is not None:
                    # st.success(f"Prediction with {model_name}: {y_class} (raw={y_pred})")
                    predictions[model_name] = {"class": y_class, "raw": float(y_pred)}

            
            for algo, pred in predictions.items():
                pred_class = pred["class"] if isinstance(pred, dict) else pred
                if algo == "Consensus":
                    acc = pred.get("accuracy", "N/A")
                else:
                    acc = MODEL_METRICS.get(algo, {}).get("accuracy", "N/A")
                st.write(f"- {algo}: {'Positive' if pred_class == 1 else 'Negative'} (Acc: {acc})")


            # Save history
            history = load_history()
            # Convert predictions dict to JSON serializable
            serializable_predictions = {k: to_serializable(v) for k, v in predictions.items()}

            history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": st.session_state.user["email"],
                "inputs": dict(zip(FEATURES, [to_serializable(x) for x in inputs])),
                "predictions": serializable_predictions
            })
            save_history(history)

    elif choice == "Profile":
        st.subheader("Edit Profile")
        users = load_users()
        profile = users[st.session_state.user["email"]]["profile"]
        age = st.number_input("Age", min_value=0, max_value=120, value=profile.get("age") or 30)
        if st.button("Save Profile"):
            update_profile(st.session_state.user["email"], {"age": age})
            st.success("Profile updated.")

    elif choice == "History":
        st.subheader("Your Prediction History")
        history = load_history()
        user_history = [h for h in history if h["user"] == st.session_state.user["email"]]

        if user_history:
            with st.container(height=500):
                # st.markdown('<div class="scrollable-history">', unsafe_allow_html=True)

                for i, record in enumerate(reversed(user_history), 1):
                    ts = record.get("timestamp", "Unknown time")
                    # Inside History tab, when building each tile
                    preds = record.get("predictions", {})

                    if "Consensus" in preds:
                        # Show only consensus result
                        consensus_class = preds["Consensus"]["class"]
                        short_summary = f"Consensus: {'Positive' if consensus_class == 1 else 'Negative'}"
                    else:
                        # Fallback to showing all models (single model case)
                        short_summary = ", ".join([
                            f"{k}: {'Positive' if v['class'] == 1 else 'Negative'}"
                            for k, v in preds.items()
                        ])

                    with st.expander(f"#{i} • {ts} • {short_summary}"):
                        st.write("**Inputs:**", record["inputs"])
                        st.write("**Predictions:**", preds)

                        # Delete confirmation
                        if st.button(f"🗑️ Delete Record #{i}", key=f"del_btn_{i}"):
                            st.session_state[f"confirm_delete_{i}"] = True

                        if st.session_state.get(f"confirm_delete_{i}", False):
                            st.warning(f"Confirm delete record #{i}?")
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button(f"✅ Yes, delete #{i}", key=f"yes_{i}"):
                                    new_history = [h for h in history if h != record]
                                    save_history(new_history)
                                    st.success(f"Deleted record #{i}.")
                                    st.session_state[f"confirm_delete_{i}"] = False
                                    st.rerun()
                            with c2:
                                if st.button(f"❌ Cancel", key=f"no_{i}"):
                                    st.session_state[f"confirm_delete_{i}"] = False
                st.markdown('</div>', unsafe_allow_html=True)

            # Delete all history for this user
            if st.button("🗑️ Delete ALL My History"):
                st.session_state["confirm_delete_all_user"] = True

            if st.session_state.get("confirm_delete_all_user", False):
                st.warning("Are you sure you want to delete ALL your history?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Yes, delete all my history"):
                        new_history = [h for h in history if h["user"] != st.session_state.user["email"]]
                        save_history(new_history)
                        st.success("All your history has been deleted.")
                        st.session_state["confirm_delete_all_user"] = False
                        st.rerun()
                with c2:
                    if st.button("❌ Cancel delete all"):
                        st.session_state["confirm_delete_all_user"] = False
        else:
            st.info("No history yet for your account.")


    elif choice == "Admin":
        st.subheader("Admin - Global Predictions")
        history = load_history()

        if not history:
            st.info("No history records found.")
        else:
            df = pd.DataFrame(history)
            st.dataframe(df)

            st.markdown("### Manage History")

            # Delete ALL
            if st.button("🗑️ Delete ALL history", type="primary"):
                st.session_state["confirm_delete_all_admin"] = True

            if st.session_state.get("confirm_delete_all_admin", False):
                st.warning("⚠️ Are you sure you want to delete ALL history?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Yes, delete everything"):
                        save_history([])
                        st.success("All history deleted.")
                        st.session_state["confirm_delete_all_admin"] = False
                        st.rerun()
                with c2:
                    if st.button("❌ Cancel delete all"):
                        st.session_state["confirm_delete_all_admin"] = False

            # Delete by user
            users = sorted(set([h["user"] for h in history]))
            selected_user = st.selectbox("Delete by user", users)
            if st.button("Delete selected user's history"):
                st.session_state["confirm_delete_user"] = selected_user

            if st.session_state.get("confirm_delete_user"):
                st.warning(f"⚠️ Are you sure you want to delete ALL history for **{st.session_state['confirm_delete_user']}**?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Yes, delete user history"):
                        filtered = [h for h in history if h["user"] != st.session_state["confirm_delete_user"]]
                        save_history(filtered)
                        st.success(f"History for {st.session_state['confirm_delete_user']} deleted.")
                        st.session_state["confirm_delete_user"] = None
                        st.rerun()
                with c2:
                    if st.button("❌ Cancel user delete"):
                        st.session_state["confirm_delete_user"] = None

            # Delete by timestamp
            timestamps = [h["timestamp"] for h in history]
            selected_ts = st.selectbox("Delete by timestamp", timestamps)
            if st.button("Delete selected record"):
                st.session_state["confirm_delete_ts"] = selected_ts

            if st.session_state.get("confirm_delete_ts"):
                st.warning(f"⚠️ Are you sure you want to delete record at {st.session_state['confirm_delete_ts']}?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Yes, delete record"):
                        filtered = [h for h in history if h["timestamp"] != st.session_state["confirm_delete_ts"]]
                        save_history(filtered)
                        st.success(f"Record at {st.session_state['confirm_delete_ts']} deleted.")
                        st.session_state["confirm_delete_ts"] = None
                        st.rerun()
                with c2:
                    if st.button("❌ Cancel record delete"):
                        st.session_state["confirm_delete_ts"] = None


    elif choice == "Logout":
        st.session_state.user = None
        st.rerun()

if __name__ == "__main__":
    main()