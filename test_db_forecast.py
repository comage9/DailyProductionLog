import pandas as pd
from forecast import load_df, forecast_series, create_events_df

# DB 설정
DB_PATH = 'vf.db'
TABLE_NAME = 'vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)'

# 데이터 로드 및 최신 날짜 조회
df = load_df(DB_PATH, TABLE_NAME)
last_date = df['일자'].max()
print(f'DB Latest Date: {last_date}')

# 이벤트 플래그 생성 (latest부터 1년 뒤까지)
events_df = create_events_df(last_date, last_date + pd.Timedelta(days=365))

# 1년(365일) 예측 수행
forecast_df = forecast_series(
    df, '일자', '수량(박스)', periods=365, freq='D', use_custom=False, events_df=events_df
)

# 요약 출력
print('First 5 forecasts:')
print(forecast_df.head())
print('Last 5 forecasts:')
print(forecast_df.tail())
print(f"Yhat range: min={forecast_df['yhat'].min()}, max={forecast_df['yhat'].max()}") 