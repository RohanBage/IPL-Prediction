import pandas as pd
import os

def load_cricket_data(filepath='data/Cricket_data.csv'):
    # Load the CSV file
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

    # Drop completely empty columns (seen in your sample)
    df.dropna(axis=1, how='all', inplace=True)

    # Convert dates
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

    # Clean column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Drop any rows where essential info is missing (e.g., no winner or scores)
    df = df[df['winner'].notna() & df['1st_inning_score'].notna() & df['2nd_inning_score'].notna()]

    # Split scores into runs/wickets if needed
    def parse_score(score):
        try:
            runs, wickets = score.split("/")
            return int(runs), int(wickets)
        except:
            return None, None

    df[['1st_inning_runs', '1st_inning_wkts']] = df['1st_inning_score'].apply(lambda x: pd.Series(parse_score(x)))
    df[['2nd_inning_runs', '2nd_inning_wkts']] = df['2nd_inning_score'].apply(lambda x: pd.Series(parse_score(x)))

    return df

if __name__ == "__main__":
    df = load_cricket_data()
    print(df.head(3))
