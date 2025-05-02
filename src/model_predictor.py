import joblib
import pandas as pd
from feature_engineering import clean_and_engineer_features
from data_loader import load_cricket_data

MODEL_PATH = "models/home_win_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def make_prediction(input_dict):
    """
    input_dict example:
    {
        "home_team": "CSK",
        "away_team": "MI",
        "toss_won": "MI",
        "decision": "BOWL FIRST",
        "venue_name": "MA Chidambaram Stadium, Chepauk, Chennai",
        "description": "68th Match (N), Indian Premier League",
        "start_date": "2023-05-20T14:00Z"
    }
    """
    # Convert dict to DataFrame
    df_input = pd.DataFrame([input_dict])

    # Fill other required columns to match expected format
    df_template = load_cricket_data().iloc[0:1].copy()
    for col in df_input.columns:
        df_template[col] = df_input[col].values[0]
    df_template['start_date'] = pd.to_datetime(df_template['start_date'])

    # Process features
    X_input, _ = clean_and_engineer_features(df_template)

    # Predict
    model = load_model()
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][prediction]

    return {
        "prediction": int(prediction),
        "confidence": round(probability, 3)
    }

# Test
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
    print("✅ Prediction:", "Home team will win" if result["prediction"] else "Home team will lose")
    print("🔮 Confidence:", result["confidence"])
