import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from app.analysis import load_df
from forecast import forecast_series

# 설정: 데이터베이스 경로 및 테이블 이름
DB_PATH = 'vf.db'
TABLE_NAME = 'vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)'

# 1. 전체 데이터를 불러와 정렬
df = load_df(DB_PATH, TABLE_NAME)
df = df.sort_values('일자')

# 2. 마지막 날짜 및 학습/테스트 분할
last_date = df['일자'].max()
train_end = last_date - pd.Timedelta(days=7)  # 7일 전까지 학습

df_train = df[df['일자'] <= train_end]
df_test = df[(df['일자'] > train_end) & (df['일자'] <= last_date)]

print(f"학습 데이터: {df_train['일자'].min().date()} ~ {df_train['일자'].max().date()} ({len(df_train)} 건)")
print(f"테스트 데이터: {df_test['일자'].min().date()} ~ {df_test['일자'].max().date()} ({len(df_test)} 건)")

# 3. 예측 (다음 7일)
forecast_df = forecast_series(df_train, '일자', '수량(박스)', periods=7, freq='D')
# 테스트 기간(학습 종료일 이후 ~ last_date)과 일치하는 부분만 추출
forecast_test = forecast_df[(forecast_df['ds'] > train_end) & (forecast_df['ds'] <= last_date)].reset_index(drop=True)

# 4. 실제값 정리
actual = df_test[['일자', '수량(박스)']].rename(columns={'일자': 'ds', '수량(박스)': 'y'})
actual = actual.reset_index(drop=True)

# 5. 병합 및 오차 계산
df_eval = forecast_test.merge(actual, on='ds', how='left')

mse = mean_squared_error(df_eval['y'], df_eval['yhat'])
mae = mean_absolute_error(df_eval['y'], df_eval['yhat'])
mape = mean_absolute_percentage_error(df_eval['y'], df_eval['yhat'])

print("\n=== 예측 vs 실제 비교 (" + str(train_end.date()+timedelta(days=1)) + " ~ " + str(last_date.date()) + ") ===")
print(df_eval[['ds', 'yhat', 'y']])
print(f"\n오차 지표: MSE={mse:.2f}, MAE={mae:.2f}, MAPE={mape*100:.2f}%") 