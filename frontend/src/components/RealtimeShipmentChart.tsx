import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Chart } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, BarController, LineController, Title, Tooltip, Legend } from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

// Props for triggering global refresh
interface RealtimeShipmentChartProps {
  refreshKey: number;
}

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, BarController, LineController, Title, Tooltip, Legend);
ChartJS.register(ChartDataLabels);

const HOURS = Array.from({ length: 24 }, (_, i) => i);

function getDateStr(offset: number) {
  const d = new Date();
  d.setDate(d.getDate() - offset);
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

const COLORS = {
  today: 'rgba(54,162,235,1)',
  todayBg: 'rgba(54,162,235,0.2)',
  pred: 'rgba(255,159,64,1)',
  predDash: [5,5],
  yest: 'rgba(153,102,255,0.4)',
  yest2: 'rgba(201,203,207,0.4)'
};

const RealtimeShipmentChart: React.FC<RealtimeShipmentChartProps> = ({ refreshKey }) => {
  const [shipments, setShipments] = useState<(number|null)[]>(Array(24).fill(null));
  const [forecast, setForecast] = useState<(number|null)[]>(Array(24).fill(null));
  const [date, setDate] = useState('');
  const [yesterday, setYesterday] = useState<(number|null)[]>(Array(24).fill(null));
  const [yesterday2, setYesterday2] = useState<(number|null)[]>(Array(24).fill(null));
  const [dateYest, setDateYest] = useState('');
  const [dateYest2, setDateYest2] = useState('');

  // Fetch today, yesterday, two days ago
  const fetchAll = async () => {
    const todayStr = getDateStr(0);
    const yestStr = getDateStr(1);
    const yest2Str = getDateStr(2);
    setDate(todayStr);
    setDateYest(yestStr);
    setDateYest2(yest2Str);
    // Today
    const res = await axios.get('/api/realtime/today');
    setShipments(res.data.shipments);
    // Yesterday
    const resY = await axios.get('/api/realtime/history', { params: { date: yestStr } });
    setYesterday(resY.data.shipments);
    // Two days ago
    const resY2 = await axios.get('/api/realtime/history', { params: { date: yest2Str } });
    setYesterday2(resY2.data.shipments);
    // Forecast for today (remaining hours after current time)
    const now = new Date();
    const currentHour = now.getHours();
    const forecastStart = currentHour + 1;
    if (forecastStart < 24) {
      const periods = 24 - forecastStart;
      const forecastRes = await axios.post('/api/forecast', {
        periods,
        freq: 'H',
        from_date: todayStr,
        last_date: todayStr
      });
      console.log('Forecast API Response:', JSON.stringify(forecastRes.data, null, 2));
      const fcArr = Array(24).fill(null);
      if (forecastRes.data && forecastRes.data.forecast) {
        forecastRes.data.forecast.forEach((f: any) => {
          try {
            const dateObj = new Date(f.ds);
            const hour = dateObj.getHours();
            console.log(`Processing forecast: ds=${f.ds}, hour=${hour}, yhat=${f.yhat}, forecastStart=${forecastStart}`);
            if (!isNaN(hour) && hour >= forecastStart && hour < 24) {
              fcArr[hour] = f.yhat;
            }
          } catch (e) {
            console.error("Error processing forecast data point:", f, e);
          }
        });
      }
      console.log('Processed Forecast Array (fcArr):', JSON.stringify(fcArr));
      setForecast(fcArr);
    } else {
      setForecast(Array(24).fill(null));
    }
  };

  // fetch data on mount and when global refreshKey changes
  useEffect(() => {
    fetchAll();
  }, [refreshKey]);

  // Build chart data
  const actualData = shipments.map((v, i) => (v !== null ? v : null));
  const forecastData = forecast;
  const yestData = yesterday.map((v, i) => (v !== null ? v : null));
  const yest2Data = yesterday2.map((v, i) => (v !== null ? v : null));

  // 증감 계산 (오늘/예측)
  const increments = actualData.map((v, i) => (i === 0 || v === null || actualData[i-1] === null) ? null : v - (actualData[i-1] as number));
  const predIncrements = forecastData.map((v, i) => (i === 0 || v === null || forecastData[i-1] === null) ? null : v - (forecastData[i-1] as number));

  const chartData = {
    labels: HOURS.map(h => `${h}시`),
    datasets: [
      // 실적 증감 막대
      {
        type: 'bar' as const,
        label: '실적 증감',
        data: increments,
        backgroundColor: 'rgba(54,162,235,0.3)',
        borderColor: COLORS.today,
        borderWidth: 1,
        yAxisID: 'y',
        order: 1,
        datalabels: {
          display: true,
          align: 'end' as const,
          anchor: 'end' as const,
          color: COLORS.today,
          font: { weight: 'bold' as const },
          formatter: (value: number|null) => typeof value === 'number' ? `+${value}` : ''
        }
      },
      // 예측 증감 막대
      {
        type: 'bar' as const,
        label: '예측 증감',
        data: predIncrements,
        backgroundColor: 'rgba(255,159,64,0.3)',
        borderColor: COLORS.pred,
        borderWidth: 1,
        yAxisID: 'y',
        order: 1,
        datalabels: {
          display: true,
          align: 'end' as const,
          anchor: 'end' as const,
          color: COLORS.pred,
          font: { weight: 'bold' as const },
          formatter: (value: number|null) => typeof value === 'number' ? `+${value}` : ''
        }
      },
      // 오늘 실측 곡선
      {
        type: 'line' as const,
        label: '실측 출고',
        data: actualData,
        borderColor: COLORS.today,
        backgroundColor: COLORS.todayBg,
        fill: false,
        tension: 0.4,
        order: 2,
        datalabels: {
          display: true,
          align: 'end' as const,
          color: COLORS.today,
          font: { weight: 'bold' as const },
          formatter: (value: number | null) => typeof value === 'number' ? Math.round(value) : ''
        }
      },
      // 오늘 예측 곡선
      {
        type: 'line' as const,
        label: '예측 출고',
        data: forecastData,
        borderColor: COLORS.pred,
        borderDash: COLORS.predDash,
        fill: false,
        tension: 0.4,
        order: 2,
        datalabels: {
          display: true,
          align: 'end' as const,
          color: COLORS.pred,
          font: { weight: 'bold' as const },
          formatter: (value: number | null) => typeof value === 'number' ? Math.round(value) : ''
        }
      },
      // 어제
      {
        type: 'line' as const,
        label: `어제 (${dateYest})`,
        data: yestData,
        borderColor: COLORS.yest,
        fill: false,
        tension: 0.4,
        order: 3,
        datalabels: { display: false }
      },
      // 그제
      {
        type: 'line' as const,
        label: `그제 (${dateYest2})`,
        data: yest2Data,
        borderColor: COLORS.yest2,
        fill: false,
        tension: 0.4,
        order: 3,
        datalabels: { display: false }
      }
    ]
  };

  return (
    <div className="chart-container">
      <h2>당일 시간대별 출고 수량 ({date})</h2>
      {/* Removed local refresh; use global 데이터 새로고침 버튼 */}
      <Chart
        key={refreshKey}
        redraw
        type="bar"
        data={chartData}
        options={{
          responsive: true,
          plugins: {
            datalabels: {
              display: (context: any) => context.dataset.label === '실측 출고',
              align: 'end' as const,
              color: COLORS.today,
              font: { weight: 'bold' as const },
              formatter: (value: number | null, context: any) => {
                const idx = context.dataIndex;
                if (typeof value !== 'number' || isNaN(value)) return '';
                if (idx === 0 || actualData[idx-1] === null || typeof actualData[idx-1] !== 'number') return Math.round(value);
                const inc = increments[idx];
                if (typeof inc !== 'number' || isNaN(inc)) return Math.round(value);
                return `${Math.round(value)}\n(+${Math.round(inc)})`;
              }
            },
            legend: { position: 'top' },
            tooltip: { enabled: true }
          },
          scales: {
            y: { title: { display: true, text: '출고 수량' } },
            x: { title: { display: true, text: '시간' } }
          }
        }}
      />
    </div>
  );
};

export default RealtimeShipmentChart; 