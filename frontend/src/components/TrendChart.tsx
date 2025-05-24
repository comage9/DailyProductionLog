import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, TimeScale } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels';
import 'chartjs-adapter-date-fns'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)
ChartJS.register(ChartDataLabels);
// Set global default font size to 15px
ChartJS.defaults.font.size = 15;

interface TrendDataItem {
  일자: string
  '수량(박스)': number
  '판매금액': number
  분류?: string;
}

interface BacktestDataItem {
  ds: string
  y: number
  yhat: number
  yhat_lower: number
  yhat_upper: number
  error_rate: number
}

interface ForecastResultItem {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
  yhat_corrected?: number;
}

interface TrendChartProps {
  item?: string | string[]
  category?: string | string[]
  fromDate?: string
  toDate?: string
  data?: TrendDataItem[];
  setData?: React.Dispatch<React.SetStateAction<TrendDataItem[]>>;
  refreshKey?: number;
}

const TrendChart: React.FC<TrendChartProps> = ({ item, category, fromDate, toDate, data: propData, setData: propSetData, refreshKey }) => {
  const [internalData, internalSetData] = useState<TrendDataItem[]>([])
  const data = propData ?? internalData;
  const setData = propSetData ?? internalSetData;
  const [prevData, setPrevData] = useState<TrendDataItem[]>([])
  const [showPrevYear, setShowPrevYear] = useState<boolean>(false)
  const [showActualValues, setShowActualValues] = useState<boolean>(true)
  const [showForecast, setShowForecast] = useState<boolean>(false)
  const [forecastDays, setForecastDays] = useState<number>(3)
  const [showErrorRate, setShowErrorRate] = useState<boolean>(false)
  const [showSales, setShowSales] = useState<boolean>(false)
  const [actualMeasure, setActualMeasure] = useState<'quantity' | 'sales'>('quantity')
  const [prevMeasure, setPrevMeasure] = useState<'quantity' | 'sales'>('quantity')
  const [backtestData, setBacktestData] = useState<BacktestDataItem[]>([])
  const [futureForecast, setFutureForecast] = useState<ForecastResultItem[]>([])

  useEffect(() => {
    axios.post('/api/trend', { item, category, from_date: fromDate, to_date: toDate })
      .then(res => {
        console.debug('Actual trend data response:', res.data)
        setData(res.data)
      })
      .catch(err => console.error(err))
    // fetch previous-year same period
    if (fromDate && toDate) {
      const f = new Date(fromDate), t = new Date(toDate);
      const pf = new Date(f.setFullYear(f.getFullYear() - 1));
      const pt = new Date(t.setFullYear(t.getFullYear() - 1));
      const prevFrom = pf.toISOString().split('T')[0];
      const prevTo = pt.toISOString().split('T')[0];
      axios.post('/api/trend', { item, category, from_date: prevFrom, to_date: prevTo })
        .then(res => {
          console.debug('Prev-year trend data response:', res.data)
          setPrevData(res.data)
        })
        .catch(err => console.error(err));
    }
  }, [item, category, fromDate, toDate, refreshKey])

  useEffect(() => {
    axios.post('/api/backtest', { item, category, from_date: fromDate, to_date: toDate })
      .then(res => setBacktestData(res.data))
      .catch(err => console.error(err))
  }, [item, category, fromDate, toDate, refreshKey])

  useEffect(() => {
    // Clear or fetch forecast based on last actual data
    if (!showForecast || !data.length) {
      setFutureForecast([])
      return
    }
    // Compute last actual data date
    const dates = data.map(d => d.일자.split('T')[0]).sort()
    const lastDate = dates[dates.length - 1]
    // Request future forecasts starting the day after last actual
    axios.post('/api/forecast', { item, category, periods: forecastDays, last_date: lastDate })
      .then(res => {
        const arr = Array.isArray(res.data) ? res.data : res.data.forecast
        setFutureForecast(arr as ForecastResultItem[])
      })
      .catch(err => console.error(err))
  }, [item, category, forecastDays, data, showForecast])

  // Determine the last actual data date for forecast basis
  const actualDates = data.map(d => d.일자.split('T')[0]).sort()
  const lastActual = actualDates.length ? actualDates[actualDates.length - 1] : undefined

  // Build continuous date labels from fromDate to toDate
  const dateLabels: string[] = []
  if (fromDate && toDate) {
    const start = new Date(fromDate)
    // 실제 데이터의 마지막 날짜와 toDate 중 더 이른 날짜까지만 x축 생성
    const lastDataDate = lastActual ? new Date(lastActual) : new Date(toDate)
    const end = lastDataDate < new Date(toDate) ? lastDataDate : new Date(toDate)
    for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
      dateLabels.push(d.toISOString().split('T')[0])
    }
  }
  // Build future forecast dates starting the day after last actual data
  const futureDates: string[] = []
  if (lastActual && showForecast) {
    const start = new Date(lastActual)
    for (let i = 1; i <= forecastDays; i++) {
      const d = new Date(start)
      d.setDate(d.getDate() + i)
      futureDates.push(d.toISOString().split('T')[0])
    }
  }
  const allDates = [...dateLabels, ...futureDates]
  const labels = allDates.map(d => d.slice(5))

  // Map historical and future data to arrays
  const dataMap = new Map(data.map(d => [d.일자.split('T')[0], d['수량(박스)']]))
  const salesMap = new Map(data.map(d => [d.일자.split('T')[0], d['판매금액']]))
  const errorMap = new Map(backtestData.map(b => [b.ds.split('T')[0], b.error_rate]))
  // Historical values arrays
  const histValues = dateLabels.map(d => dataMap.get(d) ?? null)
  const histSales = dateLabels.map(d => salesMap.get(d) ?? null)
  const histError = dateLabels.map(d => errorMap.get(d) ?? null)
  // Forecast mapping
  const fcMap = new Map(futureForecast.map(f => [f.ds.split('T')[0], (f.yhat_corrected ?? f.yhat)]))
  const fcValues = futureDates.map(d => fcMap.get(d) ?? null)

  // Compute linear regression for trend line on historical values (exclude nulls)
  const validPoints = histValues
    .map((v, i) => ({ v, i }))
    .filter(pt => pt.v != null) as { v: number; i: number }[]
  const xArr = validPoints.map(pt => pt.i)
  const yArr = validPoints.map(pt => pt.v)
  const n = xArr.length
  const xSum = xArr.reduce((acc, x) => acc + x, 0)
  const ySum = yArr.reduce((acc, y) => acc + y, 0)
  const xySum = xArr.reduce((acc, x, idx) => acc + x * yArr[idx], 0)
  const xxSum = xArr.reduce((acc, x) => acc + x * x, 0)
  const slope = n ? (n * xySum - xSum * ySum) / (n * xxSum - xSum * xSum) : 0
  const intercept = n ? (ySum - slope * xSum) / n : 0

  // Prev-year mapping and padded arrays
  const prevMapQuantity = new Map(prevData.map(d => [d.일자.split('T')[0], d['수량(박스)']]))
  const prevMapSales = new Map(prevData.map(d => [d.일자.split('T')[0], d['판매금액']]))
  // Helper: get previous year date string for a given date string
  const prevYearDate = (d: string) => {
    const dt = new Date(d);
    dt.setFullYear(dt.getFullYear() - 1);
    return dt.toISOString().split('T')[0];
  };
  const paddedPrev = [...dateLabels.map(d => prevMapQuantity.get(prevYearDate(d)) ?? null), ...futureDates.map(() => null)]
  const paddedPrevSales = [...dateLabels.map(d => prevMapSales.get(prevYearDate(d)) ?? null), ...futureDates.map(() => null)]
  // Combined padded arrays
  const paddedValues = [...histValues, ...futureDates.map(() => null)]
  const paddedSales = [...histSales, ...futureDates.map(() => null)]
  const paddedError = [...histError, ...futureDates.map(() => null)]
  const paddedForecast = [...dateLabels.map(() => null), ...fcValues]
  // Full trendline values (historical + nulls)
  const trendlineFull = [...histValues.map((_, idx) => intercept + slope * idx), ...futureDates.map(() => null)]

  // Build datasets in control order: Actual -> Prev-year -> Forecast -> Error -> Sales -> Trendline
  const datasets: any[] = []
  // Actual series
  if (showActualValues) {
    datasets.push({
      label: actualMeasure === 'quantity' ? '수량(박스)' : '판매금액',
      data: paddedValues,
      yAxisID: actualMeasure === 'quantity' ? 'quantity' : 'sales',
      borderColor: actualMeasure === 'quantity' ? 'rgba(75,192,192,1)' : 'rgba(255,99,132,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Prev-year series
  if (showPrevYear) {
    datasets.push({
      label: prevMeasure === 'quantity' ? '전년동기 수량' : '전년동기 판매금액',
      data: prevMeasure === 'quantity' ? paddedPrev : paddedPrevSales,
      yAxisID: prevMeasure === 'quantity' ? 'quantity' : 'sales',
      borderColor: 'rgba(153,102,255,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Forecast series
  if (showForecast) {
    datasets.push({
      label: `예측값(${forecastDays}일)`,
      data: paddedForecast,
      yAxisID: 'quantity',
      borderColor: 'rgba(255,159,64,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Error rate series
  if (showErrorRate) {
    datasets.push({
      label: '오차율 (%)',
      data: paddedError,
      yAxisID: 'error',
      borderColor: 'rgba(255,206,86,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Sales series
  if (showSales) {
    datasets.push({
      label: '판매금액',
      data: paddedSales,
      yAxisID: 'sales',
      borderColor: 'rgba(255,99,132,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Trendline
  if (showActualValues || showPrevYear || showForecast || showErrorRate || showSales) {
    datasets.push({
      label: '추세선',
      data: trendlineFull,
      yAxisID: 'quantity',
      borderColor: 'rgba(0,0,0,0.6)',
      borderDash: [5,5],
      fill: false,
      pointRadius: 0,
      tension: 0
    })
  }
  const chartData = { labels, datasets }

  return (
    <div className="chart-container">
      {/* Chart series controls */}
      <fieldset className="chart-controls">
        <legend><span className="comp-label">6</span> 데이터 시리즈</legend>
        <label>실제값:
          <select value={actualMeasure} onChange={e => setActualMeasure(e.target.value as 'quantity'|'sales')}>
            <option value="quantity">수량(박스)</option>
            <option value="sales">판매금액</option>
          </select>
        </label>
        <label>전년동기:
          <select value={prevMeasure} onChange={e => setPrevMeasure(e.target.value as 'quantity'|'sales')}>
            <option value="quantity">수량(박스)</option>
            <option value="sales">판매금액</option>
          </select>
        </label>
        <label>전년동기 표시:
          <input type="checkbox" checked={showPrevYear} onChange={e => setShowPrevYear(e.target.checked)} />
        </label>
        <label>예측값 표시:
          <input type="checkbox" checked={showForecast} onChange={e => setShowForecast(e.target.checked)} />
          일수:
          <select value={forecastDays} onChange={e => setForecastDays(Number(e.target.value))}>
            <option value={3}>3일</option>
            <option value={5}>5일</option>
            <option value={7}>7일</option>
          </select>
        </label>
        <label>오차율 표시:
          <input type="checkbox" checked={showErrorRate} onChange={e => setShowErrorRate(e.target.checked)} />
        </label>
        <label>판매금액 표시:
          <input type="checkbox" checked={showSales} onChange={e => setShowSales(e.target.checked)} />
        </label>
      </fieldset>
      <h2>일별 출고 수량 및 판매금액 추이</h2>
      <Line
        data={chartData}
        options={{
          responsive: true,
          aspectRatio: 3,
          scales: {
            x: { ticks: { font: { size: 15 } } },
            quantity: { type: 'linear', position: 'left', title: { display: true, text: '수량(박스)' } },
            sales: {
              type: 'linear',
              position: 'right',
              display: showSales,
              title: { display: true, text: '판매금액(백만)' },
              grid: { drawOnChartArea: false },
              ticks: { callback: (val: any) => `${(val / 1000000).toFixed(1)}M` }
            },
            error: {
              type: 'linear',
              position: 'right',
              offset: true,
              display: showErrorRate,
              title: { display: true, text: '오차율 (%)' },
              grid: { drawOnChartArea: false },
              ticks: { callback: (val: any) => `${val.toFixed(1)}%` }
            }
          },
          plugins: {
            legend: { labels: { filter: (item: any, chart: any) => {
                // maintain order: actual, prev-year, forecast, error, sales, trendline
                return true;
            } } },
            datalabels: {
              display: (context: any) => {
                if (context.dataset.label === '추세선') {
                  const idx = context.dataIndex;
                  return idx === 0 || idx === dateLabels.length - 1;
                }
                // 모든 시리즈(추세선 제외)는 항상 표시
                return context.dataset.label !== undefined;
              },
              align: 'end' as const,
              formatter: (value: number, context: any) => {
                // 판매금액 시리즈는 백만원 단위로 1자리 소수
                if (context.dataset.label && context.dataset.label.includes('판매금액')) {
                  return (value / 1_000_000).toFixed(1) + 'M';
                }
                // 오차율 시리즈에 % 단위 추가
                if (context.dataset.label && context.dataset.label.includes('오차율')) {
                  return Math.round(value) + '%';
                }
                return Math.round(value);
              }
            },
            tooltip: { callbacks: { label: ctx => {
              if (ctx.dataset.label === '판매금액') return `${(ctx.parsed.y / 1000000).toFixed(2)}M`;
              if (ctx.dataset.label?.startsWith('오차율')) return `${ctx.parsed.y.toFixed(1)}%`;
              return `${ctx.dataset.label}: ${ctx.parsed.y}`;
            } } }
          }
        }}
      />
    </div>
  )
}

// 출고량/판매금액 표 컴포넌트
export const ShippingTable: React.FC<{ data: TrendDataItem[]; fromDate?: string; toDate?: string }> = ({ data }) => {
  if (!data || data.length === 0) return <div>데이터가 없습니다.</div>;
  // Determine whether to group by '분류' or '품목'
  const keyField = (data[0] as any).분류 !== undefined ? '분류' : '품목';
  // Sorting: by box (수량) or sales (판매금액)
  const [sortKey, setSortKey] = useState<'box' | 'sales'>('box');
  // 날짜별로 그룹핑
  const dateSet = new Set<string>();
  const categorySet = new Set<string>();
  data.forEach(row => {
    const date = row.일자.split('T')[0];
    dateSet.add(date);
    const catKey = (row as any)[keyField] as string;
    categorySet.add(catKey || '기타');
  });
  const allDates = Array.from(dateSet).sort().reverse();
  const dates = allDates.slice(0, 5);
  // 카테고리별, 날짜별 데이터 매핑
  const table: Record<string, Record<string, { box: number; sales: number }>> = {};
  data.forEach(row => {
    const date = row.일자.split('T')[0];
    const cat = row['분류'] || '기타';
    if (!table[cat]) table[cat] = {};
    table[cat][date] = {
      box: row['수량(박스)'],
      sales: row['판매금액'],
    };
  });
  // 합계 계산
  const totalRow: Record<string, { box: number; sales: number }> = {};
  dates.forEach(date => {
    let box = 0, sales = 0;
    categorySet.forEach(cat => {
      box += table[cat]?.[date]?.box || 0;
      sales += table[cat]?.[date]?.sales || 0;
    });
    totalRow[date] = { box, sales };
  });
  // 카테고리 정렬: 최신 날짜 기준, 선택된 키(sortKey) 순으로 내림차순
  const sortedCategories = Array.from(categorySet).sort((a, b) => {
    const latest = dates[0];
    const aVal = table[a]?.[latest]?.[sortKey] || 0;
    const bVal = table[b]?.[latest]?.[sortKey] || 0;
    return bVal - aVal;
  });
  const categories = sortedCategories;
  // 색상 함수 (수량/금액 기준으로 파랑 계열 강조)
  function getCellColor(val: number, max: number) {
    if (!max) return '';
    const pct = Math.min(val / max, 1);
    const alpha = 0.15 + pct * 0.55;
    return `background: rgba(79,140,255,${alpha}); color: ${pct > 0.5 ? '#fff' : '#222'};`;
  }
  // 각 날짜별 최대값
  const maxBox: Record<string, number> = {};
  const maxSales: Record<string, number> = {};
  dates.forEach(date => {
    maxBox[date] = Math.max(...categories.map(cat => table[cat]?.[date]?.box || 0));
    maxSales[date] = Math.max(...categories.map(cat => table[cat]?.[date]?.sales || 0));
  });
  return (
    <div className="shipping-table-container">
      {/* Table number badge and title */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', gap: '16px' }}>
        <span className="comp-label">11</span>
        <span style={{ fontWeight: 600 }}>출고량/판매금액 표</span>
        <label style={{ marginLeft: 'auto' }}>정렬 기준:</label>
        <select value={sortKey} onChange={e => setSortKey(e.target.value as 'box' | 'sales')}>
          <option value="box">수량(박스)</option>
          <option value="sales">판매금액</option>
        </select>
      </div>
      <table className="shipping-table">
        <thead>
          <tr>
            <th rowSpan={2}>{keyField}</th>
            {dates.map(date => {
              const [y, m, d] = date.split('-');
              return (
                <th key={date} colSpan={2}>
                  {`${y}. ${Number(m)}. ${Number(d)}.`}
                </th>
              );
            })}
          </tr>
          <tr>
            {dates.map(date => (
              <React.Fragment key={date}>
                <th key={date+"box"}>수량(박스)</th>
                <th key={date+"sales"}>판매금액</th>
              </React.Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {categories.map(cat => (
            <tr key={cat}>
              <td>{cat}</td>
              {dates.map(date => [
                <td key={date+cat+"box"} style={{...getCellColor(table[cat]?.[date]?.box || 0, maxBox[date]) ? {background:getCellColor(table[cat]?.[date]?.box || 0, maxBox[date]).split(';')[0].split(':')[1], color:getCellColor(table[cat]?.[date]?.box || 0, maxBox[date]).split(';')[1].split(':')[1]} : {}}}>{table[cat]?.[date]?.box ?? '-'}</td>,
                <td key={date+cat+"sales"} style={{...getCellColor(table[cat]?.[date]?.sales || 0, maxSales[date]) ? {background:getCellColor(table[cat]?.[date]?.sales || 0, maxSales[date]).split(';')[0].split(':')[1], color:getCellColor(table[cat]?.[date]?.sales || 0, maxSales[date]).split(';')[1].split(':')[1]} : {}}}>{table[cat]?.[date]?.sales?.toLocaleString() ?? '-'}</td>
              ])}
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr style={{fontWeight:'bold', background:'#f0f4fa'}}>
            <td>총 합계</td>
            {dates.map(date => [
              <td key={date+"totalbox"}>{totalRow[date].box}</td>,
              <td key={date+"totalsales"}>{totalRow[date].sales.toLocaleString()}</td>
            ])}
          </tr>
        </tfoot>
      </table>
    </div>
  );
};

export default TrendChart