import sqlite3
import pandas as pd

def load_df(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}";', conn)
    conn.close()
    # If table uses generic field names and first row is header
    if all(str(col).startswith('field') for col in df.columns) and not df.empty:
        header = df.iloc[0].tolist()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = header
        print(f"Renamed generic columns to: {header}")
    # Replace empty strings with NA
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    # Cast date
    if '일자' in df.columns:
        df['일자'] = pd.to_datetime(df['일자'], errors='coerce')
    # Remove commas and cast numeric
    for col in ['수량(박스)', '수량(낱개)', '판매금액', '순번', '단수']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def aggregate_dimension(df):
    df = df.copy()
    # Create time dimensions
    df['year'] = df['일자'].dt.year
    df['month'] = df['일자'].dt.month
    df['week'] = df['일자'].dt.isocalendar().week
    df['day'] = df['일자'].dt.day
    df['period'] = df['day'].apply(lambda x: 'start' if x <= 10 else ('mid' if x <= 20 else 'end'))
    df['weekday'] = df['일자'].dt.weekday + 1  # 1=Monday
    # Define aggregation columns
    agg_cols = [c for c in ['수량(박스)', '수량(낱개)', '판매금액'] if c in df.columns]
    results = {}
    for dim in ['year', 'month', 'week', 'period', 'weekday']:
        grp = df.groupby(dim)[agg_cols].sum().reset_index()
        results[dim] = grp
    return results

def aggregate_by_item_category(df):
    df = df.copy()
    df['year'] = df['일자'].dt.year
    df['month'] = df['일자'].dt.month
    df['week'] = df['일자'].dt.isocalendar().week
    df['day'] = df['일자'].dt.day
    df['period'] = df['day'].apply(lambda x: 'start' if x <= 10 else ('mid' if x <= 20 else 'end'))
    df['weekday'] = df['일자'].dt.weekday + 1
    agg_cols = [c for c in ['수량(박스)', '수량(낱개)', '판매금액'] if c in df.columns]
    grp = df.groupby(['품목', '분류', 'year', 'month', 'week', 'period', 'weekday'])[agg_cols].sum().reset_index()
    return grp

def main():
    db_path = 'vf.db'
    table = 'vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)'
    print(f"Loading table: {table}")
    df = load_df(db_path, table)
    print(f"Total records after load: {len(df)}")
    # Time-dimension aggregations
    results = aggregate_dimension(df)
    for dim, grp in results.items():
        print(f"\nAggregation by {dim} (top 5):")
        print(grp.head())
    # Item/Category aggregations
    item_grp = aggregate_by_item_category(df)
    print("\nAggregation by item and category (top 5):")
    print(item_grp.head())

if __name__ == '__main__':
    main() 