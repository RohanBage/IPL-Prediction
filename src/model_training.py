from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

from data_loader import load_cricket_data
from feature_engineering import clean_and_engineer_features

def train_winning_score_model():
    # Load and preprocess
    df = load_cricket_data()
    X, _ = clean_and_engineer_features(df)

    # Create target: max of 1st and 2nd inning runs
    df['winning_score'] = df[['1st_inning_runs', '2nd_inning_runs']].max(axis=1)

    # Only keep rows where score is present
    df = df[df['winning_score'].notna()].copy()

    # Ensure X matches the filtered df
    X = X.loc[df.index]
    y = df['winning_score']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("✅ R² Score:", round(r2_score(y_test, y_pred), 3))

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/winning_score_model.pkl")
    print("✅ Score model saved to models/winning_score_model.pkl")

    return model


if __name__=="__main__":
    train_winning_score_model()

