import sqlite3
import pandas as pd
from typing import Optional, Union, List
import os
import matplotlib.pyplot as plt


def load_df(db_path: str, table_name: str) -> pd.DataFrame:
    """Load table from SQLite and clean data."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}";', conn)
    conn.close()
    # Rename generic headers if present
    if all(str(col).startswith('field') for col in df.columns) and not df.empty:
        header = df.iloc[0].tolist()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = header
    # Replace empty strings with NA
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    # Cast date
    if '일자' in df.columns:
        df['일자'] = pd.to_datetime(df['일자'], errors='coerce')
    # Remove commas and cast numerics
    for col in ['수량(박스)', '수량(낱개)', '판매금액', '순번', '단수']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def add_time_dims(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, week, period, weekday columns to DataFrame."""
    df['year'] = df['일자'].dt.year
    df['month'] = df['일자'].dt.month
    df['week'] = df['일자'].dt.isocalendar().week
    df['day'] = df['일자'].dt.day
    df['period'] = df['day'].apply(lambda x: 'start' if x <= 10 else ('mid' if x <= 20 else 'end'))
    df['weekday'] = df['일자'].dt.weekday + 1
    return df


def aggregate_dimension(df: pd.DataFrame) -> dict:
    """Aggregate sums by time dimension."""
    df2 = add_time_dims(df.copy())
    agg_cols = [c for c in ['수량(박스)', '수량(낱개)', '판매금액'] if c in df2.columns]
    results = {}
    for dim in ['year', 'month', 'week', 'period', 'weekday']:
        grp = df2.groupby(dim)[agg_cols].sum().reset_index()
        results[dim] = grp
    return results


def aggregate_trend(df: pd.DataFrame, item: str = None, category: str = None,
                    from_date: str = None, to_date: str = None) -> pd.DataFrame:
    """Aggregate daily trend filtered by item/category/date range."""
    df2 = df.copy()
    if item:
        if isinstance(item, list):
            df2 = df2[df2['품목'].isin(item)]
        else:
            df2 = df2[df2['품목'] == item]
    if category:
        if isinstance(category, list):
            df2 = df2[df2['분류'].isin(category)]
        else:
            df2 = df2[df2['분류'] == category]
    if from_date:
        df2 = df2[df2['일자'] >= pd.to_datetime(from_date)]
    if to_date:
        df2 = df2[df2['일자'] <= pd.to_datetime(to_date)]
    agg_cols = [c for c in ['수량(박스)', '수량(낱개)', '판매금액'] if c in df2.columns]
    grp = df2.groupby('일자')[agg_cols].sum().reset_index()
    return grp


def time_grouped_analysis(df: pd.DataFrame, date_col: str, period: str, agg_col: str, output_dir: str = "reports/analysis"):
    """
    Group df by given time period ("M" or "W"), aggregate agg_col, save CSV and trend plot, return JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    df[date_col] = pd.to_datetime(df[date_col])
    if period == "M":
        grouped = df.groupby(df[date_col].dt.to_period("M"))[agg_col].sum()
    elif period == "W":
        grouped = df.groupby(df[date_col].dt.to_period("W"))[agg_col].sum()
    else:
        raise ValueError("Unsupported period: choose 'M' or 'W'")
    # save results
    csv_path = os.path.join(output_dir, f"{agg_col}_{period}_summary.csv")
    grouped.to_csv(csv_path)
    # plot
    plt.figure()
    grouped.index = grouped.index.astype(str)
    grouped.plot(marker='o', title=f"{agg_col} per {period}")
    plt.xlabel("Period")
    plt.ylabel(agg_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{agg_col}_{period}_trend.png")
    plt.savefig(plot_path)
    plt.close()
    return grouped.to_json() 