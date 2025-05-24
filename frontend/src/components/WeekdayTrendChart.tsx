import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const HOURS = Array.from({ length: 24 }, (_, i) => i);
const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const COLORS = [
  'rgba(54,162,235,0.5)', // Mon
  'rgba(75,192,192,0.5)', // Tue
  'rgba(255,206,86,0.5)', // Wed
  'rgba(255,99,132,0.5)', // Thu
  'rgba(153,102,255,0.5)', // Fri
  'rgba(201,203,207,0.5)', // Sat
  'rgba(255,0,0,1)' // Sun - bold red
];

const WeekdayTrendChart: React.FC = () => {
  const [trend, setTrend] = useState<{[key: string]: (number|null)[]}>({});
  const [loading, setLoading] = useState(false);

  const fetchTrend = async () => {
    setLoading(true);
    try {
      const res = await axios.get('/api/realtime/weekday-trend', { params: { t: Date.now() } });
      setTrend(res.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrend();
  }, []);

  // trend가 없으면 로딩 표시
  if (!trend || !trend['Mon']) {
    return <div>로딩 중...</div>;
  }

  const datasets = WEEKDAYS.map((wd, idx) => ({
    label: wd === 'Sun' ? '일요일(Sun)' : wd,
    data: trend[wd] || Array(24).fill(null),
    borderColor: COLORS[idx],
    borderWidth: wd === 'Sun' ? 3 : 2,
    fill: false,
    tension: 0.4,
    pointRadius: wd === 'Sun' ? 4 : 2,
    pointBackgroundColor: COLORS[idx],
    borderDash: wd === 'Sun' ? undefined : [2,2],
    hidden: false
  }));

  const chartData = {
    labels: HOURS.map(h => `${h}시`),
    datasets
  };

  return (
    <div className="chart-container">
      <h2>요일별 시간대별 출고 평균 (최근 4주)</h2>
      <button onClick={fetchTrend} disabled={loading}>{loading ? '갱신 중...' : '리프레시'}</button>
      <Line
        data={chartData}
        options={{
          responsive: true,
          plugins: {
            legend: { position: 'top' },
            tooltip: { enabled: true }
          },
          scales: {
            y: { title: { display: true, text: '평균 출고 수량' } },
            x: { title: { display: true, text: '시간' } }
          }
        }}
      />
    </div>
  );
};

export default WeekdayTrendChart; 