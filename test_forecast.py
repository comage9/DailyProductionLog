import sqlite3
import requests
import json

# DB 설정
DB_PATH = 'vf.db'
TABLE_NAME = 'vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)'

# DB에서 최신 날짜 조회
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(f'SELECT MAX("일자") FROM "{TABLE_NAME}"')
last_date = cur.fetchone()[0]
conn.close()

print(f'DB Latest Date: {last_date}')

# 예측 API 호출 (1년치)
url = 'http://127.0.0.1:8001/api/forecast'
payload = {
    'periods': 365,
    'from_date': last_date,
    'last_date': last_date
}
response = requests.post(url, json=payload)
response.raise_for_status()
data = response.json()

# yhat 또는 yhat_corrected 값 추출
yhats = [item.get('yhat_corrected', item.get('yhat')) for item in data]

print(f'Forecast count: {len(data)}')
print(f'Yhat min: {min(yhats)}, max: {max(yhats)}')
print('First 5 forecasts:', json.dumps(data[:5], ensure_ascii=False))
print('Last 5 forecasts:', json.dumps(data[-5:], ensure_ascii=False)) 