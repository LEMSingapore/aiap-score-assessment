from pathlib import Path
import sqlite3
import pandas as pd

from . import settings


def load_data() -> pd.DataFrame:
    if not settings.DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {settings.DB_PATH}")
    with sqlite3.connect(settings.DB_PATH) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
            conn,
        )["name"].tolist()
        table_name = tables[0]
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    print(f'Loaded table "{table_name}": {df.shape[0]:,} rows × {df.shape[1]} columns')
    return df


def main() -> None:
    print("[main] Loading data from score.db...")
    df = load_data()
    print("[main] Head of dataframe:")
    print(df.head())
    # TODO: call preprocessing, feature_engineering, models here


if __name__ == "__main__":
    main()
