import os
import json
import time
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DURATION = 3600 * 24  # 24시간 캐시

def initialize_cache():
    """캐시 디렉토리 초기화"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_report(cache_key):
    """캐시된 보고서 조회"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_DURATION:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def cache_report(cache_key, report_data):
    """보고서 캐싱"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False)
