import joblib
import pandas as pd
from data_loader import load_cricket_data
from feature_engineering import clean_and_engineer_features

MODEL_CLASSIFIER_PATH = "models/home_win_model.pkl"
MODEL_SCORE_PATH = "models/winning_score_model.pkl"

def load_models():
    clf = joblib.load(MODEL_CLASSIFIER_PATH)
    reg = joblib.load(MODEL_SCORE_PATH)
    return clf, reg

def make_prediction(input_dict):
    df_input = pd.DataFrame([input_dict])
    df_template = load_cricket_data().iloc[0:1].copy()
    for col in df_input.columns:
        df_template[col] = df_input[col].values[0]
    df_template['start_date'] = pd.to_datetime(df_template['start_date'])

    X_input, _ = clean_and_engineer_features(df_template)
    clf, reg = load_models()

    win_pred = clf.predict(X_input)[0]
    win_conf = clf.predict_proba(X_input)[0][win_pred]

    score_pred = reg.predict(X_input)[0]

    return {
        "home_team_will_win": bool(win_pred),
        "win_confidence": round(win_conf, 3),
        "predicted_score_for_winner": round(score_pred)
    }

# Example test
if __name__ == "__main__":
    test_input = {
        "home_team": "CSK",
        "away_team": "MI",
        "toss_won": "MI",
        "decision": "BOWL FIRST",
        "venue_name": "MA Chidambaram Stadium, Chepauk, Chennai",
        "description": "68th Match (N), Indian Premier League",
        "start_date": "2023-05-20T14:00Z"
    }
    result = make_prediction(test_input)
    print("✅ Prediction:", "Home team will win" if result["home_team_will_win"] else "Home team will lose")
    print("🔮 Confidence:", result["win_confidence"])
    print("📊 Predicted winning score:", result["predicted_score_for_winner"])
