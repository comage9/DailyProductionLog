#!/usr/bin/env python3
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
import requests
import io

load_dotenv()
# Configuration from environment or defaults
DB_PATH = os.getenv("DB_PATH", "vf.db")
TABLE_NAME = os.getenv("TABLE_NAME", "vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)")
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQwqI0BG-d2aMrql7DK4fQQTjvu57VtToSLAkY_nq92a4Cg5GFVbIn6_IR7Fq6_O-2TloFSNlXT8ZWC/pub?gid=1152588885&single=true&output=csv"

# Fetch sheet data
resp = requests.get(CSV_URL)
resp.raise_for_status()
sheet = pd.read_csv(io.BytesIO(resp.content), dtype=str)
# Normalize column names and drop irrelevant
sheet.columns = [c.strip() for c in sheet.columns]
if '미입고 수량' in sheet.columns:
    sheet = sheet.drop(columns=['미입고 수량'])
# Parse and aggregate
sheet['일자'] = pd.to_datetime(sheet['일자'], errors='coerce').dt.strftime('%Y-%m-%d')
sheet['수량(박스)'] = pd.to_numeric(sheet['수량(박스)'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
sheet_grp = sheet.groupby('일자')['수량(박스)'].sum()

# Load DB data
conn = sqlite3.connect(DB_PATH)
db = pd.read_sql_query(
    f'SELECT "일자", "수량(박스)" FROM "{TABLE_NAME}"',
    conn,
    parse_dates=['일자']
)
conn.close()
db['일자'] = db['일자'].dt.strftime('%Y-%m-%d')
db_grp = db.groupby('일자')['수량(박스)'].sum()

# Compare
cmp_df = pd.concat([sheet_grp, db_grp], axis=1, keys=['sheet', 'db']).fillna(0)
cmp_df['diff'] = cmp_df['db'] - cmp_df['sheet']
# sheet과 db 값이 다른 행(불일치)만 필터링
mismatches = cmp_df[cmp_df['diff'] != 0]
if mismatches.empty:
    print("No mismatches found")
else:
    print(mismatches.to_string()) 