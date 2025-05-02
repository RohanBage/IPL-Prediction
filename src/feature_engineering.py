import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_and_engineer_features(df):
    df = df.copy()

    # Label encode teams
    team_cols = ['home_team', 'away_team', 'toss_won', 'winner']
    le_team = LabelEncoder()
    all_teams = pd.concat([df[col] for col in team_cols]).dropna().unique()
    le_team.fit(all_teams)

    for col in team_cols:
        df[col + '_encoded'] = le_team.transform(df[col].fillna('Unknown'))

    # Encode venue
    le_venue = LabelEncoder()
    df['venue_encoded'] = le_venue.fit_transform(df['venue_name'].fillna('Unknown'))

    # Toss decision binary
    df['toss_decision_encoded'] = df['decision'].map({'BAT FIRST': 0, 'BOWL FIRST': 1}).fillna(-1)

    # Match type (Qualifier/Final/League)
    df['match_type'] = df['description'].apply(lambda x: 'Qualifier' if 'Qualifier' in str(x) else ('Final' if 'Final' in str(x) else 'League'))
    df['match_type_encoded'] = df['match_type'].map({'League': 0, 'Qualifier': 1, 'Final': 2})

    # Target column: did home_team win?
    df['home_win'] = (df['winner'] == df['home_team']).astype(int)

    # Time-based features
    df['month'] = df['start_date'].dt.month
    df['year'] = df['start_date'].dt.year
    df['day_night'] = df['description'].str.contains('N', na=False).astype(int)

    # Select features
    feature_cols = [
        'home_team_encoded', 'away_team_encoded', 'toss_won_encoded',
        'toss_decision_encoded', 'venue_encoded', 'match_type_encoded',
        'month', 'year', 'day_night'
    ]

    target_col = 'home_win'

    return df[feature_cols], df[target_col]

if __name__ == "__main__":
    from data_loader import load_cricket_data
    df_raw = load_cricket_data()
    X, y = clean_and_engineer_features(df_raw)
    print(f"✅ Feature matrix shape: {X.shape}")
    print(X.head())
    print(f"✅ Target distribution:\n{y.value_counts()}")
