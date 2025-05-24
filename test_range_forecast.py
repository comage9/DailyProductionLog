import pandas as pd
from forecast import load_df, forecast_series, create_events_df

# DB 설정
DB_PATH = 'vf.db'
TABLE_NAME = 'vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)'

# 데이터 로드 및 최신 날짜 조회
df = load_df(DB_PATH, TABLE_NAME)
last_date = df['일자'].max()

# 테스트 대상 기간
start = pd.to_datetime('2025-05-05')
end = pd.to_datetime('2025-12-31')
# 예측 기간 계산
periods = (end - last_date).days
if periods <= 0:
    raise ValueError('end date must be after last_date')

# 이벤트 플래그 생성
events_df = create_events_df(last_date, end)
# 예측 수행
forecast_df = forecast_series(
    df, '일자', '수량(박스)', periods=periods, freq='D', use_custom=False, events_df=events_df
)
# 범위 필터링
subset = forecast_df[(forecast_df['ds'] >= start) & (forecast_df['ds'] <= end)]

# 요약 출력
print(f"Forecast count: {len(subset)}")
print(f"Date range: {subset['ds'].min()} ~ {subset['ds'].max()}")
print(f"Yhat min: {subset['yhat'].min()}, max: {subset['yhat'].max()}")
print(f"Any negative values: { (subset['yhat'] < 0).any() }")
print("First 5 records:", subset.head().to_dict(orient='records'))
print("Last 5 records:", subset.tail().to_dict(orient='records')) 