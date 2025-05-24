import sqlite3
import pandas as pd

def list_tables(conn):
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]

def get_schema(conn, table_name):
    cursor = conn.execute(f'PRAGMA table_info("{table_name}");')
    return [(col[1], col[2]) for col in cursor.fetchall()]

def sample_data(conn, table_name, limit=5):
    cursor = conn.execute(f'SELECT * FROM "{table_name}" LIMIT {limit};')
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return cols, rows

def infer_types(df):
    type_map = {}
    for col in df.columns:
        if '일자' in col or 'date' in col.lower():
            type_map[col] = 'datetime'
        elif any(x in col for x in ['수량','금액','단','순번']):
            type_map[col] = 'numeric'
        else:
            type_map[col] = 'text'
    return type_map

def analyze_table(conn, table_name):
    print(f"\nAnalyzing {table_name}:")
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}";', conn)
    # If table has generic field1, field2... and first row contains actual headers, rename columns
    if all(str(col).startswith('field') for col in df.columns) and not df.empty:
        header_row = df.iloc[0].tolist()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = header_row
        print(f"Renamed generic columns to: {header_row}")
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    print("Inferred types:")
    inferred = infer_types(df)
    for col, t in inferred.items():
        print(f"  - {col}: {t}")
    for col, t in inferred.items():
        if t == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif t == 'numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("\nData types after casting:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nOutlier counts (beyond 3 sigma) per numeric column:")
    for col, t in inferred.items():
        if t == 'numeric':
            series = df[col].dropna()
            mean = series.mean()
            std = series.std()
            outliers = series[(series < mean - 3*std) | (series > mean + 3*std)]
            print(f"  - {col}: {len(outliers)}")

def main():
    db_path = 'vf.db'
    conn = sqlite3.connect(db_path)
    tables = list_tables(conn)
    print("Tables:", tables)
    for table in tables:
        print(f"\nSchema for {table}:")
        schema = get_schema(conn, table)
        for col_name, col_type in schema:
            print(f"  - {col_name}: {col_type}")
        print(f"\nSample data for {table} (first 5 rows):")
        cols, rows = sample_data(conn, table)
        print("\t".join(cols))
        for row in rows:
            print("\t".join(str(v) for v in row))
        analyze_table(conn, table)
    conn.close()

if __name__ == '__main__':
    import traceback
    print("DEBUG: Starting inspect_db", flush=True)
    try:
        main()
    except Exception:
        traceback.print_exc() 