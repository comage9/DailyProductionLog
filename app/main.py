# -*- coding: utf-8 -*-
from fastapi import FastAPI, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import Optional, Union, List, Dict
from datetime import date, timedelta, datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import subprocess
import logging
import sqlite3
import requests
import io
import time
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import sys
from fastapi import HTTPException, status
import shutil
from .cache import initialize_cache, get_cached_report, cache_report
from .background import schedule_periodic_reports, queue_report_generation, process_report_queue
from app.scrapers.coupan_scraper import get_coupan_report_for_bono_house

# Configure logger for debugging
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

from app.analysis import load_df, aggregate_dimension, aggregate_trend
from forecast import forecast_series, create_events_df, train_residual_model, predict_with_residual_correction, safe_forecast_series
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.eda import generate_eda_report
from app.analysis import time_grouped_analysis
from app.decomposition import decompose_time_series, forecast_arima

# Load settings
load_dotenv()
db_path = os.getenv("DB_PATH", "vf.db")
table_name = os.getenv("TABLE_NAME", "vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)")

# CSV URL for real-time data updates
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQwqI0BG-d2aMrql7DK4fQQTjvu57VtToSLAkY_nq92a4Cg5GFVbIn6_IR7Fq6_O-2TloFSNlXT8ZWC/pub?gid=1152588885&single=true&output=csv"

REALTIME_CSV_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQYW_XI-stT0t4KqqpDW0DcBud_teV8223_vupnZsO3DrbqRqZkwXBplXSld8sB_qEXL92Ckn7J8B29/pub?gid=572466553&single=true&output=csv'
# CSV URL for inventory data (전산 재고 수량)
INVENTORY_CSV_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQwqI0BG-d2aMrql7DK4fQQTjvu57VtToSLAkY_nq92a4Cg5GFVbIn6_IR7Fq6_O-2TloFSNlXT8ZWC/pub?gid=2125795373&single=true&output=csv'
DB_PATH = db_path  # use main database path for realtime and inventory
last_realtime_update: str = ""

# In-memory caches for rapid access
real_time_cache: Dict[str, List[Optional[float]]] = {}
history_daily_sum: Dict[str, float] = {}

# Helper for running LLM prompts via Ollama CLI
def run_llm_prompt(model: str, prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logger.error(f"LLM error: {result.stderr.strip()}")
            return result.stderr.strip()
    except Exception as e:
        logger.error(f"run_llm_prompt exception: {e}")
        return str(e)

def fetch_csv_to_db():
    try:
        df_csv = pd.read_csv(CSV_URL, dtype=str, low_memory=False)
        # 고유키 후보: 일자, 품목, 분류 (컬럼명 확인 필요)
        key_cols = ['일자', '품목', '분류']
        conn = sqlite3.connect(db_path)
        # 테이블이 없으면 생성 (컬럼명 자동 추출)
        cols = ', '.join([f'"{c}" TEXT' for c in df_csv.columns])
        key_str = ', '.join([f'"{c}"' for c in key_cols])
        # 기본적으로 TEXT로 생성, 필요시 타입 조정
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols}, PRIMARY KEY ({key_str}))')
        # Ensure unique index on key columns to enable ON CONFLICT upsert; wrap in try to ignore existing violations
        try:
            conn.execute(
                f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{table_name}_key" ON "{table_name}" ({key_str})'
            )
        except Exception as e:
            logger.warning(f'Unique index creation skipped due to: {e}')
        # Clear existing data to avoid stale or duplicate rows
        conn.execute(f'DELETE FROM "{table_name}"')
        conn.commit()
        # upsert all CSV data
        if not df_csv.empty:
            placeholders = ','.join(['?'] * len(df_csv.columns))
            update_cols = ','.join([f'"{c}"=excluded."{c}"' for c in df_csv.columns if c not in key_cols])
            sql = f'INSERT INTO "{table_name}" ({",".join([f"\"{c}\"" for c in df_csv.columns])}) VALUES ({placeholders}) ON CONFLICT({key_str}) DO UPDATE SET {update_cols}'
            conn.executemany(sql, df_csv.values.tolist())
            conn.commit()
            # Delete rows not present in the CSV to remove stale items
            existing_keys = set(conn.execute(f'SELECT {key_str} FROM "{table_name}"').fetchall())
            csv_keys = set(tuple(r[c] for c in key_cols) for _, r in df_csv.iterrows())
            for key in existing_keys - csv_keys:
                where_clause = ' AND '.join([f'"{col}"=?' for col in key_cols])
                conn.execute(f'DELETE FROM "{table_name}" WHERE {where_clause}', key)
            conn.commit()
        conn.close()
        logger.debug(f"Fetched CSV data ({df_csv.shape}), upserted {len(df_csv)} rows to {table_name}")
    except Exception as e:
        logger.error(f"Error fetching CSV data: {e}")

def download_and_store_realtime():
    global last_realtime_update
    # retry download up to 3 times
    for attempt in range(3):
        try:
            resp = requests.get(REALTIME_CSV_URL, timeout=10)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            else:
                raise
    # Try utf-8-sig, then cp949, then fallback
    try:
        df = pd.read_csv(io.BytesIO(resp.content), encoding='utf-8-sig')
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(resp.content), encoding='cp949')
        except Exception:
            df = pd.read_csv(io.BytesIO(resp.content))
    df.columns = [str(c).strip() for c in df.columns]
    # If columns are field1, field2, ... use first row as header
    if all(str(col).startswith('field') for col in df.columns):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
    # Auto-detect columns
    date_col = next((c for c in df.columns if '날짜' in c or '일자' in c), None)
    day_col = next((c for c in df.columns if '요일' in c), None)
    total_col = next((c for c in df.columns if '합계' in c), None)
    if not (date_col and day_col and total_col):
        raise Exception(f'필수 컬럼이 없습니다: {df.columns}')
    melt = df.melt(id_vars=[date_col, day_col, total_col], var_name='hour', value_name='shipment')
    melt['hour'] = pd.to_numeric(melt['hour'], errors='coerce')
    melt = melt.dropna(subset=['hour'])
    melt['hour'] = melt['hour'].astype(int)
    melt[date_col] = pd.to_datetime(melt[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    melt = melt.rename(columns={date_col: '날짜'})
    # Generate EDA report for realtime data
    try:
        generate_eda_report(melt, output_dir="reports/eda/realtime")
    except Exception as e:
        logger.warning(f"EDA report generation failed: {e}")
    conn = sqlite3.connect(DB_PATH)
    # ensure table exists with primary key for upsert semantics
    conn.execute(
        """CREATE TABLE IF NOT EXISTS realtime_shipments (
            날짜 TEXT,
            hour INTEGER,
            shipment REAL,
            PRIMARY KEY (날짜, hour)
        )"""
    )
    # ensure unique index on (날짜, hour) to support ON CONFLICT upsert
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_realtime_shipments_date_hour ON realtime_shipments(날짜, hour)"
    )
    # upsert each record into realtime_shipments
    upsert_sql = (
        "INSERT INTO realtime_shipments (날짜, hour, shipment) VALUES (?, ?, ?) "
        "ON CONFLICT(날짜, hour) DO UPDATE SET shipment=excluded.shipment"
    )
    data = melt[['날짜', 'hour', 'shipment']].values.tolist()
    conn.executemany(upsert_sql, data)
    conn.commit()
    conn.close()
    # record last update timestamp
    last_realtime_update = datetime.now().isoformat()

# Add inventory sync and cache loader
def fetch_inventory_to_db():
    try:
        df_csv = pd.read_csv(INVENTORY_CSV_URL, dtype=str, low_memory=False)
        # strip whitespace from column names for consistent matching
        df_csv.columns = [str(c).strip() for c in df_csv.columns]
        display_cols = ['일자','시간','로케이션','분류','품명','바코드','전산 재고 수량','일 평균 출고 수량','단수']
        drop_col = '미입고 수량'
        if drop_col in df_csv.columns:
            df_csv = df_csv.drop(columns=[drop_col])
        df_csv = df_csv.loc[:, display_cols]
        df_csv = df_csv.where(pd.notnull(df_csv), None)
        conn = sqlite3.connect(DB_PATH)
        key_cols = ['일자','시간','로케이션','바코드']
        cols_def = ', '.join([f'"{c}" TEXT' for c in display_cols])
        key_str = ', '.join([f'"{c}"' for c in key_cols])
        conn.execute(f'CREATE TABLE IF NOT EXISTS inventory ({cols_def}, PRIMARY KEY ({key_str}))')
        conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS idx_inventory_key ON inventory({key_str})')
        placeholders = ','.join(['?'] * len(display_cols))
        col_list = ','.join([f'"{c}"' for c in display_cols])
        update_cols = ','.join([f'"{c}"=excluded."{c}"' for c in display_cols if c not in key_cols])
        upsert_sql = f'INSERT INTO inventory ({col_list}) VALUES ({placeholders}) ON CONFLICT({key_str}) DO UPDATE SET {update_cols}'
        conn.executemany(upsert_sql, df_csv.values.tolist())
        existing_keys = set(conn.execute(f'SELECT {key_str} FROM inventory').fetchall())
        csv_keys = set([tuple([r[c] for c in key_cols]) for _, r in df_csv.iterrows()])
        for key in existing_keys - csv_keys:
            where_clause = ' AND '.join([f'"{col}"=?' for col in key_cols])
            conn.execute(f'DELETE FROM inventory WHERE {where_clause}', key)
        conn.commit()
        conn.close()
        logger.debug(f"Fetched inventory data ({len(df_csv)}) rows into 'inventory' table")
    except Exception as e:
        logger.error(f"Error syncing inventory to DB: {e}")

inventory_cache: List[Dict] = []

def load_inventory_cache():
    global inventory_cache
    conn = sqlite3.connect(DB_PATH)
    df_inv = pd.read_sql_query("SELECT * FROM inventory", conn)
    conn.close()
    df_inv = df_inv.where(pd.notnull(df_inv), None)
    inventory_cache = df_inv.to_dict(orient='records')

# Initialize application and data
app = FastAPI(title="출고 수량 분석 API")

# Add CORS middleware to allow external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log all requests to access.log (development only)
@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    response = await call_next(request)
    # Skip logging for the access.log endpoint to prevent file modification during streaming
    if request.url.path == "/access.log":
        return response
    try:
        # Determine base directory for log file
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.getcwd()
        log_path = os.path.join(base_dir, 'access.log')
        ts = datetime.now().isoformat()
        client_ip = request.client.host
        method = request.method
        path = request.url.path
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{ts}\t{client_ip}\t{method}\t{path}\n")
    except Exception as e:
        logger.error(f"Failed to write access log: {e}")
    return response


# Endpoint for Coupang Bono House report
@app.get("/api/reports/coupan/bono_house", summary="쿠팡 '보노 하우스' 관련 상품 리포트 생성")
async def get_coupan_bono_house_report():
    """
    쿠팡에서 '보노 하우스' 관련 상품의 구매 후기, 댓글, Q&A를 스크랩하여
    보고서 형태로 반환합니다.
    """
    try:
        report_data = await get_coupan_report_for_bono_house()

        # --- 디버깅 코드 추가 시작 ---
        logger.debug(f"Type of report_data: {type(report_data)}")
        if isinstance(report_data, list):
            for i, item in enumerate(report_data):
                logger.debug(f"Type of report_data[{i}]: {type(item)}")
                if isinstance(item, dict):
                    for key, value in item.items():
                        logger.debug(f"  Type of report_data[{i}]['{key}']: {type(value)}")
        # --- 디버깅 코드 추가 끝 ---

        return JSONResponse(content=report_data)
    except Exception as e:
        logger.error(f"Error generating Coupang report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"보고서 생성 중 오류 발생: {str(e)}")


# Initial data load from CSV
fetch_csv_to_db()
df = load_df(db_path, table_name)
dimensions = aggregate_dimension(df)

# NEW: Download realtime data at startup
try:
    download_and_store_realtime()
except Exception as e:
    logger.error(f'Failed to download realtime data at startup: {e}')

# INITIAL inventory load
fetch_inventory_to_db()
load_inventory_cache()

# Generic function to generate and store AI insight for a given date
def generate_insight_for_date(date_str: str) -> str:
    # Build context from specified date's data
    try:
        df_prev = df[(df['일자'] >= pd.to_datetime(date_str)) & (df['일자'] <= pd.to_datetime(date_str))]
        item_qty = df_prev.groupby('품목')['수량(박스)'].sum().to_dict()
        item_sales = df_prev.groupby('품목')['판매금액'].sum().to_dict()
        context_str = f"조회 기간 {date_str}의 품목별 출고량 합계: {item_qty}. 판매금액 합계: {item_sales}."
        instruction = "생각 과정을 생략하고 간결하게 한국어로, 마크다운 형식으로 요약하세요."
        prompt = f"{instruction} {context_str}"
        summary = run_llm_prompt('qwen3:4b', prompt)
        # Store in DB
        conn2 = sqlite3.connect(DB_PATH)
        conn2.execute(
            "INSERT OR REPLACE INTO insights (date, summary) VALUES (?, ?)",
            (date_str, summary)
        )
        conn2.commit()
        conn2.close()
        logger.info(f"AI insight generated and stored for {date_str}")
        return summary
    except Exception as e:
        logger.error(f"Error generating insight for {date_str}: {e}")
        raise

# Job wrapper to generate yesterday's insight after data refresh
def daily_insight_job():
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    generate_insight_for_date(yesterday)

# Initialize insights storage table
conn_ins = sqlite3.connect(DB_PATH)
conn_ins.execute("CREATE TABLE IF NOT EXISTS insights (date TEXT PRIMARY KEY, summary TEXT)")
conn_ins.close()

# Scheduler to refresh data daily at midnight
def refresh_data():
    global df, dimensions
    # Refresh data from CSV and reload
    fetch_csv_to_db()
    df = load_df(db_path, table_name)
    dimensions = aggregate_dimension(df)
    # Refresh inventory data
    fetch_inventory_to_db()
    load_inventory_cache()

scheduler = AsyncIOScheduler()
scheduler.add_job(refresh_data, 'cron', hour=0)  # Main historical data (daily)
scheduler.add_job(download_and_store_realtime, 'interval', hours=1)  # Real-time hourly data (every hour)
scheduler.add_job(fetch_inventory_to_db, 'interval', hours=3)  # Inventory sync every 3 hours
scheduler.add_job(daily_insight_job, 'cron', hour=0, minute=5)  # Daily AI insight after data sync
scheduler.start()

# Initialize in-memory caches for rapid access
def load_history_cache():
    global history_daily_sum
    # aggregate daily sums from historical DataFrame
    df_hist = load_df(db_path, table_name)
    df_hist['일자'] = pd.to_datetime(df_hist['일자'], errors='coerce').dt.strftime('%Y-%m-%d')
    history_daily_sum = df_hist.groupby('일자')['수량(박스)'].sum().to_dict()

def load_realtime_cache():
    global real_time_cache
    conn2 = sqlite3.connect(DB_PATH)
    df_rt = pd.read_sql_query("SELECT 날짜, hour, shipment FROM realtime_shipments", conn2)
    conn2.close()
    cache = {}
    for date, group in df_rt.groupby('날짜'):
        arr = [None] * 24
        for _, row in group.iterrows():
            h = int(row['hour'])
            val = row['shipment']
            arr[h] = float(val) if pd.notna(val) else None
        cache[date] = arr
    real_time_cache = cache

# Load caches at startup
load_history_cache()
load_realtime_cache()
load_inventory_cache()

# Models
class TrendParams(BaseModel):
    item: Optional[Union[str, List[str]]] = None
    category: Optional[Union[str, List[str]]] = None
    from_date: Optional[date] = None
    to_date: Optional[date] = None

class ForecastParams(BaseModel):
    item: Optional[Union[str, List[str]]] = None
    category: Optional[Union[str, List[str]]] = None
    periods: int = Field(30, gt=0)
    from_date: Optional[date] = None
    last_date: Optional[date] = None
    use_custom: bool = False
    use_exog: bool = False
    freq: Optional[str] = None

class BacktestParams(BaseModel):
    item: Optional[Union[str, List[str]]] = None
    category: Optional[Union[str, List[str]]] = None
    from_date: Optional[date] = None
    to_date: Optional[date] = None

# Routes
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "출고 수량 분석 API",
        "overview_endpoint": "/api/overview",
        "trend_endpoint": "/api/trend",
        "forecast_endpoint": "/api/forecast",
        "docs": "/docs"
    }

@app.get("/access.log", include_in_schema=False)
async def get_access_log():
    """Serve the access.log file (development only)."""
    # Determine base directory
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.getcwd()
    log_path = os.path.join(base_dir, 'access.log')
    return FileResponse(log_path, media_type='text/plain')

@app.get("/api/overview")
def get_overview(dimension: str = Query("year", enum=["year", "month", "week", "period", "weekday"])):
    """Return aggregated sums by the specified time dimension."""
    return dimensions.get(dimension, [])

@app.post("/api/trend")
def get_trend(params: TrendParams):
    """Return daily trend filtered by item, category, and optional date range."""
    df2 = df.copy()
    # filter by item(s)
    if params.item:
        if isinstance(params.item, list):
            df2 = df2[df2['품목'].isin(params.item)]
        else:
            df2 = df2[df2['품목'] == params.item]
    # filter by category(ies)
    if params.category:
        if isinstance(params.category, list):
            df2 = df2[df2['분류'].isin(params.category)]
        else:
            df2 = df2[df2['분류'] == params.category]
    # filter by dates
    if params.from_date:
        df2 = df2[df2['일자'] >= pd.to_datetime(params.from_date)]
    if params.to_date:
        df2 = df2[df2['일자'] <= pd.to_datetime(params.to_date)]
    # aggregate daily sums
    grp = df2.groupby('일자')[['수량(박스)', '판매금액']].sum().reset_index()
    grp['일자'] = grp['일자'].dt.strftime('%Y-%m-%d')
    return grp.to_dict(orient='records')

@app.post("/api/trend-by-category")
def get_trend_by_category(params: TrendParams):
    """Return daily sums by category filtered by item/category/date range."""
    df2 = df.copy()
    # filter by item(s)
    if params.item:
        if isinstance(params.item, list):
            df2 = df2[df2['품목'].isin(params.item)]
        else:
            df2 = df2[df2['품목'] == params.item]
    # filter by category(ies)
    if params.category:
        if isinstance(params.category, list):
            df2 = df2[df2['분류'].isin(params.category)]
        else:
            df2 = df2[df2['분류'] == params.category]
    if params.from_date:
        df2 = df2[df2['일자'] >= pd.to_datetime(params.from_date)]
    if params.to_date:
        df2 = df2[df2['일자'] <= pd.to_datetime(params.to_date)]
    # 그룹핑: 일자, 분류별 합계
    grp = df2.groupby(['일자', '분류'])[['수량(박스)', '판매금액']].sum().reset_index()
    # 날짜 포맷을 문자열로 변환
    grp['일자'] = grp['일자'].dt.strftime('%Y-%m-%d')
    return grp.to_dict(orient='records')


async def _prepare_shipment_analysis_data(params: TrendParams, df_global: pd.DataFrame):
    """주어진 조건에 따라 출고 데이터를 분석하고 필요한 모든 데이터를 반환하는 내부 함수"""
    logger.info(f"Preparing shipment analysis data for: {params}")

    df_report = df_global.copy()
    if params.item:
        if isinstance(params.item, list):
            df_report = df_report[df_report['품목'].isin(params.item)]
        else:
            df_report = df_report[df_report['품목'] == params.item]
    if params.category:
        if isinstance(params.category, list):
            df_report = df_report[df_report['분류'].isin(params.category)]
        else:
            df_report = df_report[df_report['분류'] == params.category]

    query_from_date = pd.to_datetime(params.from_date) if params.from_date else None
    query_to_date = pd.to_datetime(params.to_date) if params.to_date else None

    if query_from_date:
        df_report = df_report[df_report['일자'] >= query_from_date]
    if query_to_date:
        df_report = df_report[df_report['일자'] <= query_to_date]

    if df_report.empty:
        return {
            "error": "해당 조건에 맞는 데이터가 없습니다.",
            "filtered_df": pd.DataFrame(),
            "daily_summary_list": [],
            "overall_metrics_dict": {},
            "llm_analysis_text": "",
            "llm_recommendation_text": "",
            "generation_date_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    daily_summary_df = df_report.groupby(df_report['일자'].dt.date)[['수량(박스)', '판매금액']].sum().reset_index()
    daily_summary_df.columns = ['일자', '수량(박스)', '판매금액']
    daily_summary_df['일자'] = pd.to_datetime(daily_summary_df['일자']).dt.strftime('%Y-%m-%d')
    daily_summary_list = daily_summary_df.sort_values(by='일자').to_dict(orient='records')

    if not daily_summary_list:
        return {
            "error": "데이터 집계 후 분석할 내용이 없습니다.",
            "filtered_df": df_report, # 원본 df_report는 있을 수 있음
            "daily_summary_list": [],
            "overall_metrics_dict": {},
            "llm_analysis_text": "",
            "llm_recommendation_text": "",
            "generation_date_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    total_quantity_boxes = sum(item['수량(박스)'] for item in daily_summary_list)
    total_sales_amount = sum(item['판매금액'] for item in daily_summary_list)
    num_days = len(daily_summary_list)
    average_daily_quantity_boxes = round(total_quantity_boxes / num_days, 2) if num_days > 0 else 0
    average_daily_sales_amount = round(total_sales_amount / num_days, 2) if num_days > 0 else 0
    peak_day_by_quantity = max(daily_summary_list, key=lambda x: x['수량(박스)']) if daily_summary_list else {}
    peak_day_by_sales = max(daily_summary_list, key=lambda x: x['판매금액']) if daily_summary_list else {}

    overall_metrics_dict = {
        "total_quantity_boxes": total_quantity_boxes,
        "total_sales_amount": total_sales_amount,
        "num_days": num_days,
        "average_daily_quantity_boxes": average_daily_quantity_boxes,
        "average_daily_sales_amount": average_daily_sales_amount,
        "peak_day_by_quantity_date": peak_day_by_quantity.get('일자'),
        "peak_day_by_quantity_value": peak_day_by_quantity.get('수량(박스)'),
        "peak_day_by_sales_date": peak_day_by_sales.get('일자'),
        "peak_day_by_sales_value": peak_day_by_sales.get('판매금액')
    }

    data_sample_for_llm = ""
    if len(daily_summary_list) > 10:
        data_sample_for_llm += "최근 주요 데이터 (상위 5개 일자):\n" + "\n".join([f"- {d['일자']}: {d['수량(박스)']}박스, {d['판매금액']:,}원" for d in daily_summary_list[-5:]])
        data_sample_for_llm += "\n\n과거 주요 데이터 (하위 5개 일자):\n" + "\n".join([f"- {d['일자']}: {d['수량(박스)']}박스, {d['판매금액']:,}원" for d in daily_summary_list[:5]])
    else:
        data_sample_for_llm = "\n".join([f"- {d['일자']}: {d['수량(박스)']}박스, {d['판매금액']:,}원" for d in daily_summary_list])

    prompt_template_analysis = f"""
다음 출고 데이터 요약을 기반으로, 지정된 기간 동안의 전반적인 출고량 및 판매금액 추세, 주요 변동 사항, 관찰된 패턴에 대해 한국어로 상세히 분석해 주십시오.
분석은 객관적인 데이터를 바탕으로 하며, 전문적인 보고서 스타일로 작성합니다. (200자 이내 권장)

데이터 요약:
{data_sample_for_llm}

기간: {params.from_date.strftime('%Y-%m-%d') if params.from_date else daily_summary_list[0]['일자']} ~ {params.to_date.strftime('%Y-%m-%d') if params.to_date else daily_summary_list[-1]['일자']}
분석 내용:
"""

    prompt_template_recommendation = f"""
다음 출고 데이터 요약과 앞선 기간별 추이 분석을 바탕으로, 주목할 만한 특이사항, 데이터 기반의 사업적 제언 또는 개선 아이디어를 한국어로 구체적으로 제시해 주십시오.
제언은 실질적이고 실행 가능한 내용 위주로 작성합니다. (200자 이내 권장)

데이터 요약:
{data_sample_for_llm}

기간: {params.from_date.strftime('%Y-%m-%d') if params.from_date else daily_summary_list[0]['일자']} ~ {params.to_date.strftime('%Y-%m-%d') if params.to_date else daily_summary_list[-1]['일자']}
제언 내용:
"""
    logger.debug(f"LLM Analysis Prompt (first 200 chars): {prompt_template_analysis[:200]}")
    logger.debug(f"LLM Recommendation Prompt (first 200 chars): {prompt_template_recommendation[:200]}")

    llm_analysis_text = run_llm_prompt(model="qwen2:7b-instruct-q4_K_M", prompt=prompt_template_analysis)
    llm_recommendation_text = run_llm_prompt(model="qwen2:7b-instruct-q4_K_M", prompt=prompt_template_recommendation)

    return {
        "filtered_df": df_report,
        "daily_summary_list": daily_summary_list,
        "overall_metrics_dict": overall_metrics_dict,
        "llm_analysis_text": llm_analysis_text,
        "llm_recommendation_text": llm_recommendation_text,
        "generation_date_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/api/report/shipment", summary="출고 데이터 기반 AI 분석 보고서 생성")
async def generate_shipment_report(params: TrendParams):
    """
    주어진 기간 및 필터 조건에 따라 출고 데이터를 분석하고,
    AI를 활용하여 서술적 분석이 포함된 Markdown 보고서를 생성합니다.
    """
    analysis_data = await _prepare_shipment_analysis_data(params, df_global=df) # df는 전역 DataFrame으로 가정

    report_parts = [
        f"# 출고 데이터 분석 보고서\n",
        f"**보고서 생성일:** {report_generation_date}",
        f"**조회 기간:** {start_date_str} ~ {end_date_str}"
    ]
    if item_filter_str:
        report_parts.append(f"**품목 필터:** {item_filter_str}")
    if category_filter_str:
        report_parts.append(f"**카테고리 필터:** {category_filter_str}")

    report_parts.extend([
        "\n---\n",
        "## 1. 종합 요약\n",
        f"*   **총 출고 수량 (박스):** {total_quantity_boxes:,}",
        f"*   **총 판매 금액:** {total_sales_amount:,.0f}원",
        f"*   **일 평균 출고 수량 (박스):** {average_daily_quantity_boxes:,.2f}",
        f"*   **일 평균 판매 금액:** {average_daily_sales_amount:,.0f}원",
        f"*   **최다 출고일 (수량 기준):** {peak_day_by_quantity.get('일자', 'N/A')} ({peak_day_by_quantity.get('수량(박스)', 0):,} 박스)",
        f"*   **최다 판매일 (금액 기준):** {peak_day_by_sales.get('일자', 'N/A')} ({peak_day_by_sales.get('판매금액', 0):,.0f}원)\n",
        "\n---\n",
        "## 2. 기간별 출고량 추이 분석\n",
        f"{llm_period_trend_analysis}\n",
        "\n---\n",
        "## 3. 주요 관찰 사항 및 AI 제언\n",
        f"{llm_observations_and_recommendations}\n",
        "\n---\n",
        "## 부록: 일별 상세 데이터\n",
        "| 일자       | 수량(박스) | 판매금액     |",
        "|------------|------------|--------------|"
    ])

    for row in daily_summary:
        report_parts.append(f"| {row['일자']} | {row['수량(박스)']:,} | {row['판매금액']:,.0f}원 |")

    final_report = "\n".join(report_parts)
    logger.info("Shipment report generated successfully.")
    return {"report_markdown": final_report}


@app.post("/api/forecast")
def get_forecast(params: ForecastParams):
    """Return forecasted values for 수량(박스) filtered by item, category, and date range."""
    # 시간별 예측 요청 시, 최근 7일간 평균 시간별 출고량 누적으로 예측
    if getattr(params, 'freq', None) == 'H':
        logger.debug("Handling hourly forecast request based on avg increments")
        today_dt = pd.to_datetime(params.from_date or datetime.now().strftime('%Y-%m-%d'))
        # 실제 오늘 데이터
        conn = sqlite3.connect(DB_PATH)
        try:
            df_today = pd.read_sql_query(
                "SELECT hour, shipment FROM realtime_shipments WHERE 날짜 = ? ORDER BY hour",
                conn, params=[today_dt.strftime('%Y-%m-%d')]
            )
            logger.debug(f"Today's data (df_today) shape: {df_today.shape}\n{df_today.head()}")
        except Exception as e:
            logger.error(f"Error fetching today's data: {e}")
            df_today = pd.DataFrame({'hour':[], 'shipment':[]})
        # 과거 7일 데이터
        seven = (today_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        try:
            df_hist = pd.read_sql_query(
                "SELECT 날짜, hour, shipment FROM realtime_shipments WHERE 날짜 >= ? AND 날짜 < ? ORDER BY 날짜, hour",
                conn, params=[seven, today_dt.strftime('%Y-%m-%d')]
            )
            logger.debug(f"History data (df_hist) shape: {df_hist.shape}\n{df_hist.head()}")
        except Exception as e:
            logger.error(f"Error fetching history data: {e}")
            df_hist = pd.DataFrame({'날짜':[], 'hour':[], 'shipment':[]})
        conn.close()

        # 오늘 데이터에서 실적이 있는 시간대 찾기
        present_hours = df_today.dropna(subset=['shipment'])['hour'].astype(int).tolist()
        first_missing_hour = next((h for h in range(24) if h not in present_hours), 24)
        forecastStart = first_missing_hour  # 실적이 없는 첫 시간부터 예측 시작

        # 누적 실적의 마지막 값을 예측의 시작점으로 사용
        if present_hours:
            last_actual_hour = max(present_hours)
            # shipment 값이 NaN일 경우 0으로 대체
            shipment_val = df_today[df_today['hour'] == last_actual_hour]['shipment'].values
            if len(shipment_val) > 0 and pd.notna(shipment_val[0]):
                last_actual_cum = int(shipment_val[0])
            else:
                # 마지막 유효한(숫자인) shipment 값 찾기
                valid_shipments = df_today.dropna(subset=['shipment'])
                if not valid_shipments.empty:
                    last_actual_cum = int(valid_shipments.iloc[-1]['shipment'])
                else:
                    last_actual_cum = 0
        else:
            last_actual_cum = 0

        logger.debug(f"실적 데이터가 있는 시간대: {present_hours}")
        logger.debug(f"실적이 없는 첫 시간: {forecastStart}")
        logger.debug(f"누적 실적 마지막 값: {last_actual_cum}")

        # 시간별 평균 *증가량* 계산
        df_hist['shipment'] = pd.to_numeric(df_hist['shipment'], errors='coerce')
        df_hist = df_hist.dropna(subset=['shipment']) # NaN shipment 값 제거
        df_hist = df_hist.sort_values(by=['날짜', 'hour'])
        df_hist['increment'] = df_hist.groupby('날짜')['shipment'].diff() # diff 후 NaN 가능성 있음 (각 날짜의 첫 시간)
        
        # 평균 증가량 계산 시 NaN을 0으로 채우고, 음수 증가량은 0으로 처리
        avg_inc_per_hour = df_hist.groupby('hour')['increment'].mean().fillna(0).clip(lower=0).to_dict()
        logger.debug(f"Average increment per hour: {avg_inc_per_hour}")

        # 예측 누적값 계산 (평균 증가량 기반)
        result = []
        pred_cum = last_actual_cum 
        for h in range(forecastStart, 24):
            inc = avg_inc_per_hour.get(h, 0) 
            if pd.isna(inc):
                inc = 0
            pred_cum += inc 
            ds = today_dt + pd.Timedelta(hours=h)
            result.append({'ds': ds.strftime('%Y-%m-%dT%H:%M:%S'), 'yhat': int(round(pred_cum))})
        # 0~23시 전체에 대해 yhat 값을 채움
        full_result = []
        # 오늘 시간별 실제 출고량을 딕셔너리로 준비 (NaN은 특정 값으로 대체하거나, 여기서는 int로 변환하므로 유효한 숫자만 포함됨)
        actuals_map = df_today.dropna(subset=['shipment']).set_index('hour')['shipment'].astype(int).to_dict()

        for h in range(24):
            ds_str = (today_dt + pd.Timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M:%S')
            
            if h in actuals_map: # 실제 데이터가 있는 시간
                full_result.append({'ds': ds_str, 'yhat': actuals_map[h]})
            elif h >= forecastStart: # 예측해야 하는 시간 (실제 데이터 없고, 예측 시작 시간 이후)
                # 'result'는 forecastStart부터 23시까지의 예측값을 담고 있음
                predicted_val_obj = next((r for r in result if pd.to_datetime(r['ds']).hour == h), None)
                if predicted_val_obj and predicted_val_obj['yhat'] is not None:
                    full_result.append({'ds': ds_str, 'yhat': predicted_val_obj['yhat']})
                else:
                    # 예측값이 없는 경우 (이론상 발생하면 안되지만, 방어 코드로 last_actual_cum 사용)
                    full_result.append({'ds': ds_str, 'yhat': last_actual_cum}) 
            else: # 실제 데이터도 없고, 예측 시작 시간(forecastStart) 이전 시간
                full_result.append({'ds': ds_str, 'yhat': last_actual_cum})
        
        logger.debug(f"Final forecast result (hourly, full 0~23, yhat populated): {full_result}")
        return {'forecast': full_result}
    # 그 외 일별 예측
    df2 = df
    # Debug: log input parameters
    logger.debug(f"get_forecast called with: {params}")
    # Exogenous regressors: price per box (판매금액 / 수량)
    exog_cols = None
    if params.use_exog:
        df2['price_per_box'] = df2['판매금액'] / df2['수량(박스)']
        exog_cols = ['price_per_box']
    # Debug: initial data size
    logger.debug(f"Initial df size: {df2.shape}")
    # Filter by item and/or category
    if params.item:
        if isinstance(params.item, list):
            df2 = df2[df2['품목'].isin(params.item)]
        else:
            df2 = df2[df2['품목'] == params.item]
        logger.debug(f"Filtered by item '{params.item}', size: {df2.shape}")
    if params.category:
        if isinstance(params.category, list):
            df2 = df2[df2['분류'].isin(params.category)]
        else:
            df2 = df2[df2['분류'] == params.category]
        logger.debug(f"Filtered by category '{params.category}', size: {df2.shape}")
    # Create event flags for improved accuracy and residual correction
    start_date = df2['일자'].min()
    events_df = create_events_df(start_date, start_date + timedelta(days=params.periods))
    # Force using custom Prophet model with all regressors (lag, events, holidays)
    forecast_full = safe_forecast_series(
        df2,
        '일자',
        '수량(박스)',
        periods=params.periods,
        freq=params.freq or 'D',
        use_custom=True,
        exog_cols=exog_cols,
        events_df=events_df
    )
    logger.debug(f"Full forecast results sample:\n{forecast_full.head()}")
    if params.last_date:
        cutoff = pd.to_datetime(params.last_date)
        # Prepare actual historical data
        ts_hist = df2[['일자', '수량(박스)']].dropna().rename(columns={'일자':'ds','수량(박스)':'y'})
        ts_hist = ts_hist.groupby('ds')['y'].sum().reset_index()
        try:
            # Align predictions with actuals by merging on ds
            hist_pred = forecast_full[['ds', 'yhat']].merge(ts_hist, on='ds', how='inner')
            actuals = hist_pred['y']
            preds = hist_pred['yhat']
            # Compute error metrics (skip if empty)
            if len(actuals) > 0 and len(preds) > 0:
                mse = mean_squared_error(actuals, preds)
                mae = mean_absolute_error(actuals, preds)
                mape = mean_absolute_percentage_error(actuals, preds)
            else:
                mse = mae = mape = None
            # Generate future forecasts using last_date cutoff
            forecast_future = forecast_full[forecast_full['ds'] > cutoff].copy()
            # 잔차 보정도 데이터가 있을 때만
            if not hist_pred.empty and not forecast_future.empty:
                residual_model = train_residual_model(ts_hist, hist_pred[['ds','yhat']], events_df=events_df)
                corrected_df = predict_with_residual_correction(residual_model, forecast_future, events_df=events_df)
                forecast_future = forecast_future.merge(corrected_df, on='ds', how='left')
                # Use the residual-corrected yhat as the primary forecast
                forecast_future['yhat'] = forecast_future['yhat_corrected'].fillna(forecast_future['yhat'])
                return {
                    'metrics': {'mse': mse, 'mae': mae, 'mape': mape},
                    'forecast': forecast_future.to_dict(orient='records')
                }
        except Exception as e:
            logger.error(f"Residual correction failed, returning raw forecast. Error: {e}")
            # Fallback: return only future forecast values
            forecast_future = forecast_full[forecast_full['ds'] > cutoff].copy() if 'cutoff' in locals() else forecast_full
            return forecast_future.to_dict(orient='records')
        # Fallback for cases where no residual correction path returned
        return forecast_future.to_dict(orient='records')
    else:
        return forecast_full.to_dict(orient='records')

# Endpoint to get unique items (품목)
@app.get("/api/items")
def get_items(category: str = Query(None, description="분류 이름")):
    """Return list of unique items (품목), optionally filtered by category"""
    if category:
        items = df[df['분류'] == category]['품목'].dropna().unique().tolist()
    else:
        items = df['품목'].dropna().unique().tolist()
    return items

# Endpoint to get categories (분류) for a given item
@app.get("/api/categories")
def get_categories(item: str = Query(None, description="품목 이름")):
    """Return list of unique categories (분류), optionally filtered by item"""
    if item:
        cats = df[df['품목'] == item]['분류'].dropna().unique().tolist()
    else:
        cats = df['분류'].dropna().unique().tolist()
    return cats

@app.get("/api/models")
def get_models():
    """Return list of local Ollama models."""
    # static list; replace with dynamic `ollama list` if needed
    # default AI model is qwen3:4b
    return ["qwen3:4b", "gemma3:latest", "exaone-deep:7.8b", "gemma3:4b"]

class InsightParams(BaseModel):
    item: Optional[Union[str, List[str]]] = None
    category: Optional[Union[str, List[str]]] = None
    from_date: date
    to_date: date
    model: str
    question: Optional[str] = None  # optional follow-up question for chat

@app.post("/api/insight")
def get_insight(params: InsightParams):
    """Handle initial summary of top-performing categories and follow-up Q&A."""
    logger.debug(f"get_insight called with: {params}")
    # If no follow-up question, generate default summary of categories with most increase
    if params.question is None:
        # Filter by item if provided
        df2 = df.copy()
        if params.item:
            if isinstance(params.item, list):
                df2 = df2[df2['품목'].isin(params.item)]
            else:
                df2 = df2[df2['품목'] == params.item]
        if params.category:
            if isinstance(params.category, list):
                df2 = df2[df2['분류'].isin(params.category)]
            else:
                df2 = df2[df2['분류'] == params.category]
        # Current period sums by category
        df_curr = df2[(df2['일자'] >= pd.to_datetime(params.from_date)) & (df2['일자'] <= pd.to_datetime(params.to_date))]
        curr_sum = df_curr.groupby('분류')['수량(박스)'].sum()
        # Previous year same period
        prev_from = params.from_date - timedelta(days=365)
        prev_to = params.to_date - timedelta(days=365)
        df_prev = df2[(df2['일자'] >= pd.to_datetime(prev_from)) & (df2['일자'] <= pd.to_datetime(prev_to))]
        prev_sum = df_prev.groupby('분류')['수량(박스)'].sum()
        # Combine and compute percent change
        df_cat = pd.DataFrame({'curr': curr_sum, 'prev': prev_sum}).fillna(0)
        df_cat = df_cat[df_cat['prev'] > 0]
        if not df_cat.empty:
            # Compute percent changes and pick top 5 categories
            df_cat['pct'] = (df_cat['curr'] - df_cat['prev']) / df_cat['prev'] * 100
            top5 = df_cat.sort_values('pct', ascending=False).head(5)
            # Format category summaries
            cats = [f"{cat}({row.pct:+.1f}%↑)" for cat, row in top5.iterrows()]
            # For each top category, find top 5 items by percent change
            details = []
            for cat in top5.index:
                # current and previous for items within category
                df_curr_cat = df2[(df2['분류'] == cat) & (df2['일자'] >= pd.to_datetime(params.from_date)) & (df2['일자'] <= pd.to_datetime(params.to_date))]
                df_prev_cat = df2[(df2['분류'] == cat) & (df2['일자'] >= pd.to_datetime(params.from_date - timedelta(days=365))) & (df2['일자'] <= pd.to_datetime(params.to_date - timedelta(days=365)))]
                curr_items = df_curr_cat.groupby('품목')['수량(박스)'].sum()
                prev_items = df_prev_cat.groupby('품목')['수량(박스)'].sum()
                df_item = pd.DataFrame({'curr': curr_items, 'prev': prev_items}).fillna(0)
                df_item = df_item[df_item['prev'] > 0]
                if not df_item.empty:
                    df_item['pct'] = (df_item['curr'] - df_item['prev']) / df_item['prev'] * 100
                    top_items = df_item.sort_values('pct', ascending=False).head(5).index.tolist()
                    details.append(f"{cat}: {', '.join(top_items)}")
            default_summary = (
                f"이번 기간 수량이 가장 크게 증가한 상위 5개 분류는 {', '.join(cats)}입니다. "
                f"세부 품목별 상위 5개는 { '; '.join(details) }입니다."
            )
        else:
            default_summary = "이번 기간 급격한 증가를 보인 분류가 없습니다."
        return {"insight": default_summary}
    # Otherwise, handle follow-up with AI
    # Provide full per-item context for the LLM to answer arbitrary questions
    # Filter data for the requested period
    df_period = df[(df['일자'] >= pd.to_datetime(params.from_date)) & (df['일자'] <= pd.to_datetime(params.to_date))]
    # Aggregate per-item totals
    item_qty = df_period.groupby('품목')['수량(박스)'].sum().to_dict()
    item_sales = df_period.groupby('품목')['판매금액'].sum().to_dict()
    # Build context string for AI
    context_str = (
        f"조회 기간 {params.from_date}~{params.to_date}의 품목별 출고량 합계: {item_qty}. "
        f"품목별 판매금액 합계: {item_sales}."
    )
    # Build prompt including user question, instruct model to omit chain-of-thought
    instruction = "생각 과정을 생략하고, 질문에 대한 정답만 간결하게 한국어로, 마크다운 형식으로 답변하세요."
    prompt = f"{instruction} {context_str} 추가 질문: {params.question}"
    logger.debug(f"Insight prompt (Q&A): {prompt}")
    # Call ollama CLI
    result = run_llm_prompt(params.model, prompt)
    return {"insight": result}

# Endpoint to manually refresh data from CSV to DB
@app.post("/api/refresh-data")
def refresh_data_endpoint():
    """Fetch the remote CSV, replace DB table, and reload in-memory data."""
    fetch_csv_to_db()
    global df, dimensions
    df = load_df(db_path, table_name)
    dimensions = aggregate_dimension(df)
    # refresh in-memory history cache
    load_history_cache()
    # refresh inventory data and cache
    fetch_inventory_to_db()
    load_inventory_cache()
    return {"status": "data refreshed"}

@app.post("/api/backtest")
def get_backtest(params: BacktestParams):
    """Return historical daily forecast and error rate between from_date and to_date."""
    df2 = df
    if params.item:
        if isinstance(params.item, list):
            df2 = df2[df2['품목'].isin(params.item)]
        else:
            df2 = df2[df2['품목'] == params.item]
    if params.category:
        if isinstance(params.category, list):
            df2 = df2[df2['분류'].isin(params.category)]
        else:
            df2 = df2[df2['분류'] == params.category]
    if params.from_date:
        df2 = df2[df2['일자'] >= pd.to_datetime(params.from_date)]
    if params.to_date:
        df2 = df2[df2['일자'] <= pd.to_datetime(params.to_date)]
    # Get forecast series including historical dates
    hist_fc = forecast_series(df2, '일자', '수량(박스)', periods=0, freq='D')
    # Prepare actuals
    actual = df2[['일자', '수량(박스)']].dropna().rename(columns={'일자':'ds', '수량(박스)':'y'})
    actual = actual.groupby('ds')['y'].sum().reset_index()
    # Merge and compute error rate
    merged = hist_fc.merge(actual, on='ds', how='inner')
    merged['error_rate'] = (merged['yhat'] - merged['y']).abs() / merged['y'] * 100
    # Return per-day records
    return merged[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', 'error_rate']].to_dict(orient='records')

@app.post('/api/realtime/refresh')
def refresh_realtime():
    # 즉시 realtime 데이터를 다운로드하고 캐시를 업데이트합니다
    download_and_store_realtime()
    load_realtime_cache()
    return {'status': 'realtime data refreshed'}

@app.get('/api/realtime/status')
def get_realtime_status():
    """Return last realtime data update timestamp."""
    return {'last_update': last_realtime_update}

@app.get('/api/realtime/today')
def get_today_realtime():
    today = datetime.now().strftime('%Y-%m-%d')
    shipments = real_time_cache.get(today, [None] * 24)
    return {'date': today, 'shipments': shipments}

@app.get('/api/realtime/history')
def get_realtime_history(date: str):
    shipments = real_time_cache.get(date, [None] * 24)
    return {'date': date, 'shipments': shipments}

@app.get('/api/realtime/weekday-trend')
def get_weekday_trend():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT 날짜, hour, shipment FROM realtime_shipments", conn)
    conn.close()
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['weekday'] = df['날짜'].dt.weekday  # Monday=0, Sunday=6
    # 최근 4주만 사용
    max_date = df['날짜'].max()
    min_date = max_date - pd.Timedelta(days=28)
    df_recent = df[df['날짜'] >= min_date]
    # 요일별, 시간별 평균
    weekday_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    result = {}
    for wd in range(7):
        arr = [None]*24
        for h in range(24):
            vals = df_recent[(df_recent['weekday']==wd) & (df_recent['hour']==h)]['shipment']
            arr[h] = float(vals.mean()) if not vals.empty else None
        result[weekday_map[wd]] = arr 
    return result

@app.get('/api/status/daily-sum')
def get_daily_sums():
    """Return daily total shipments for historical analysis."""
    # Fetch fresh sheet CSV into DB and reload cache to ensure up-to-date
    fetch_csv_to_db()
    load_history_cache()
    return [{'date': d, 'sum': s} for d, s in history_daily_sum.items()]

@app.get("/api/top-item")
async def get_top_item(date: date = Query(None, description="조회할 날짜 (YYYY-MM-DD), 기본: 어제")):
    """Return the item with the highest shipment quantity for a specified date (default: yesterday)."""
    # Determine target date
    target_date = date or (datetime.now().date() - timedelta(days=1))
    # Filter data for that date
    df_date = df[df['일자'] == pd.to_datetime(target_date)]
    # Group by item and sum quantities
    grp = df_date.groupby('품목')['수량(박스)'].sum()
    if grp.empty:
        return {"date": str(target_date), "item": None, "quantity": 0}
    # Identify top item
    top_item = grp.idxmax()
    top_qty  = int(grp.max())
    logger.debug(f"Top item for {target_date}: {top_item} ({top_qty})")
    return {"date": str(target_date), "item": top_item, "quantity": top_qty}

@app.get("/api/inventory")
def get_inventory():
    """Return cached inventory data to avoid CSV loading issues."""
    return inventory_cache 

# Analysis API endpoints
@app.get("/api/analysis/monthly")
async def monthly_analysis(
    agg_col: str = Query("수량(박스)", description="Aggregation column"),
    date_col: str = Query("일자")
):
    conn = sqlite3.connect(DB_PATH)
    df_hist = pd.read_sql_query(
        f"SELECT `{date_col}` as date, `{agg_col}` as value FROM `{table_name}`",
        conn
    )
    conn.close()
    result_json = time_grouped_analysis(df_hist, "date", "M", "value", output_dir="reports/analysis/monthly")
    return JSONResponse(content=json.loads(result_json))

@app.get("/api/analysis/weekly")
async def weekly_analysis(
    agg_col: str = Query("수량(박스)", description="Aggregation column"),
    date_col: str = Query("일자")
):
    conn = sqlite3.connect(DB_PATH)
    df_hist = pd.read_sql_query(
        f"SELECT `{date_col}` as date, `{agg_col}` as value FROM `{table_name}`",
        conn
    )
    conn.close()
    result_json = time_grouped_analysis(df_hist, "date", "W", "value", output_dir="reports/analysis/weekly")
    return JSONResponse(content=json.loads(result_json))

@app.get("/api/analysis/decompose")
async def decompose(
    agg_col: str = Query("수량(박스)"),
    date_col: str = Query("일자"),
    period: int = Query(7, description="Decomposition period")
):
    conn = sqlite3.connect(DB_PATH)
    df_hist = pd.read_sql_query(
        f"SELECT `{date_col}` as date, `{agg_col}` as value FROM `{table_name}`",
        conn
    )
    conn.close()
    # Aggregate values by date to ensure unique index
    series = df_hist.groupby("date")["value"].sum().sort_index()
    comps = decompose_time_series(series, period, output_dir="reports/decomposition")
    return JSONResponse(content={k: json.loads(v) for k, v in comps.items()})

@app.get("/api/analysis/arima")
async def arima_forecast(
    agg_col: str = Query("수량(박스)"),
    date_col: str = Query("일자"),
    order: str = Query("1,1,1"),
    steps: int = Query(10)
):
    conn = sqlite3.connect(DB_PATH)
    df_hist = pd.read_sql_query(
        f"SELECT `{date_col}` as date, `{agg_col}` as value FROM `{table_name}`",
        conn
    )
    conn.close()
    series = df_hist.set_index("date")["value"]
    order_tuple = tuple(map(int, order.split(",")))
    forecast_json = forecast_arima(series, order_tuple, steps)
    return JSONResponse(content=json.loads(forecast_json))

# === Insight storage retrieval and Q&A endpoints ===
class InsightQuestion(BaseModel):
    question: str

@app.get("/api/insights/{date_str}")
def get_daily_insight(date_str: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT summary FROM insights WHERE date=?", (date_str,)).fetchone()
        conn.close()
        if row:
            return {"date": date_str, "summary": row[0]}
        # Generate on demand if missing
        summary = generate_insight_for_date(date_str)
        return {"date": date_str, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"AI insight generation failed: {e}")

@app.post("/api/insights/{date_str}/question")
def question_insight(date_str: str, payload: InsightQuestion):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT summary FROM insights WHERE date=?", (date_str,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No insight for given date")
    try:
        instruction = "생각 과정을 생략하고, 질문에 대한 정답만 간결하게 한국어로 답변하세요."
        prompt = f"{instruction} 참고 요약문: {row[0]} 추가 질문: {payload.question}"
        answer = run_llm_prompt('qwen3:4b', prompt)
        return {"date": date_str, "question": payload.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"AI Q&A failed: {e}")

@app.get("/api/insights/health")
def insights_health():
    # Check if Ollama CLI is available
    if shutil.which('ollama') is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Ollama CLI not found")
    return {"status": "ok"} 

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup sequence initiated.")
    initialize_cache()
    logger.info("Cache initialized.")
    
    # 주기적 보고서 생성을 위한 파라미터를 큐에 추가하는 작업 예약
    # (schedule_periodic_reports는 이제 큐에 추가만 하고 즉시 생성 안 함)
    await schedule_periodic_reports()
    logger.info("Periodic report scheduling (queuing) completed.")

    # 보고서 큐를 주기적으로 처리하는 작업 스케줄링
    if scheduler:
        try:
            # next_run_time을 현재 시간으로부터 1분 뒤로 설정하여 즉시 실행 방지
            run_time = datetime.now() + timedelta(minutes=1)
            scheduler.add_job(process_report_queue, 'interval', minutes=1, misfire_grace_time=60, id='process_report_queue_job', next_run_time=run_time)
            logger.info(f"Job 'process_report_queue_job' added to scheduler (1-minute interval, next run at {run_time.strftime('%Y-%m-%d %H:%M:%S')}).")
        except Exception as e:
            logger.error(f"Error adding 'process_report_queue_job' to scheduler: {e}")
    else:
        logger.error("Scheduler not initialized. Cannot add 'process_report_queue_job'.")

    # 스케줄러 시작
    if scheduler and not scheduler.running:
        try:
            scheduler.start()
            logger.info("Scheduler started successfully.")
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    elif scheduler and scheduler.running:
        logger.info("Scheduler is already running.")
    else:
        logger.warning("Scheduler was not started because it's not initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown sequence initiated.")
    if scheduler and scheduler.running:
        try:
            # Give pending jobs a chance to complete
            scheduler.shutdown(wait=True) 
            logger.info("Scheduler shut down successfully.")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    else:
        logger.info("Scheduler was not running or not initialized, no shutdown needed.")

@app.get("/api/report/test_route_cascade")
async def test_route_cascade():
    return {"message": "Cascade test route is working!"}

@app.post("/api/report/detailed", summary="상세 출고 데이터 분석 보고서 생성")
async def generate_detailed_report(params: TrendParams):
    cache_key = f"detailed_{params.from_date}_{params.to_date}_{params.item or ''}_{params.category or ''}"
    cached = get_cached_report(cache_key)
    if cached:
        logger.info(f"Returning cached detailed report for key: {cache_key}")
        return cached

    logger.info(f"Generating new detailed report for: {params}")
    analysis_data = await _prepare_shipment_analysis_data(params, df_global=df) # df는 전역 DataFrame으로 가정

    if analysis_data.get("error"):
        # 오류 발생 시 간단한 메시지 반환 또는 적절한 HTTP 오류 반환
        raise HTTPException(status_code=404, detail=analysis_data['error'])

    df_filtered = analysis_data["filtered_df"]
    item_sales_summary = []
    category_sales_summary = []

    if df_filtered is not None and not df_filtered.empty:
        item_sales_summary = df_filtered.groupby('품목').agg(
            total_quantity=('수량(박스)', 'sum'),
            total_amount=('판매금액', 'sum')
        ).reset_index().sort_values(by='total_amount', ascending=False).to_dict(orient='records')

        category_sales_summary = df_filtered.groupby('분류').agg(
            total_quantity=('수량(박스)', 'sum'),
            total_amount=('판매금액', 'sum')
        ).reset_index().sort_values(by='total_amount', ascending=False).to_dict(orient='records')

    detailed_report_payload = {
        "report_info": {
            "title": "상세 판매 분석 리포트",
            "generated_at": analysis_data['generation_date_str'],
            "period": f"{params.from_date.strftime('%Y-%m-%d') if params.from_date else 'N/A'} ~ {params.to_date.strftime('%Y-%m-%d') if params.to_date else 'N/A'}"
        },
        "summary_markdown": f"## AI 분석 요약\n\n{analysis_data['llm_analysis_text']}\n\n## AI 사업적 제언\n\n{analysis_data['llm_recommendation_text']}",
        "overall_metrics": analysis_data['overall_metrics_dict'],
        "charts_data": {
            "sales_trend": {
                "labels": [d['일자'] for d in analysis_data.get("daily_summary_list", [])],
                "datasets": [
                    {"label": "수량(박스)", "data": [d['수량(박스)'] for d in analysis_data.get("daily_summary_list", [])], "borderColor": "#36A2EB", "backgroundColor": "#9BD0F5"},
                    {"label": "판매금액", "data": [d['판매금액'] for d in analysis_data.get("daily_summary_list", [])], "yAxisID": "y1", "borderColor": "#FF6384", "backgroundColor": "#FFB1C1"}
                ]
            },
            "item_distribution_quantity": {
                "labels": [item['품목'] for item in item_sales_summary[:10]], # 상위 10개
                "datasets": [{"label": "판매량(박스)", "data": [item['total_quantity'] for item in item_sales_summary[:10]]}]
            },
            "item_distribution_amount": {
                "labels": [item['품목'] for item in item_sales_summary[:10]],
                "datasets": [{"label": "판매금액", "data": [item['total_amount'] for item in item_sales_summary[:10]]}]
            },
            "category_distribution_quantity": {
                "labels": [cat['분류'] for cat in category_sales_summary],
                "datasets": [{"label": "판매량(박스)", "data": [cat['total_quantity'] for cat in category_sales_summary]}]
            },
            "category_distribution_amount": {
                "labels": [cat['분류'] for cat in category_sales_summary],
                "datasets": [{"label": "판매금액", "data": [cat['total_amount'] for cat in category_sales_summary]}]
            }
        },
        "tables_data": {
            "daily_sales": analysis_data.get("daily_summary_list", []),
            "item_summary": item_sales_summary,
            "category_summary": category_sales_summary
        }
    }
    cache_report(cache_key, detailed_report_payload)
    logger.info(f"Detailed report generated and cached for key: {cache_key}")
    return detailed_report_payload