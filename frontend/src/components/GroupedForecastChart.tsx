import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels'

// Register ChartJS modules
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)
ChartJS.register(ChartDataLabels)

interface ForecastDataItem {
  ds: string
  yhat: number
  yhat_lower: number
  yhat_upper: number
  yhat_corrected?: number
}

interface GroupedForecastChartProps {
  groupBy: 'category' | 'item'
  selectedCategory?: string
  periods?: number
  useCustom?: boolean
}

const palette = [
  'rgba(75,192,192,1)',
  'rgba(255,99,132,1)',
  'rgba(153,102,255,1)',
  'rgba(255,159,64,1)',
  'rgba(54,162,235,1)',
  'rgba(255,206,86,1)',
  'rgba(201,203,207,1)'
]

const GroupedForecastChart: React.FC<GroupedForecastChartProps> = ({ groupBy, selectedCategory, periods = 30, useCustom = false }) => {
  const [groups, setGroups] = useState<string[]>([])
  const [chartData, setChartData] = useState<any>(null)
  const [loading, setLoading] = useState<boolean>(false)

  // Fetch the list of categories or items based on grouping
  useEffect(() => {
    if (groupBy === 'category') {
      axios.get('/api/categories')
        .then(res => setGroups(res.data))
        .catch(err => console.error(err))
    } else if (groupBy === 'item') {
      if (!selectedCategory) return
      axios.get(`/api/items?category=${encodeURIComponent(selectedCategory)}`)
        .then(res => setGroups(res.data))
        .catch(err => console.error(err))
    }
  }, [groupBy, selectedCategory])

  // Fetch forecast for each group
  useEffect(() => {
    if (!groups.length) return
    setLoading(true)
    const requests = groups.map(g => {
      const params: any = { periods, use_custom: useCustom }
      if (groupBy === 'category') params.category = g
      if (groupBy === 'item') {
        params.category = selectedCategory
        params.item = g
      }
      return axios.post('/api/forecast', params)
        .then(res => {
          const arr = Array.isArray(res.data) ? res.data : res.data.forecast
          return { group: g, data: arr as ForecastDataItem[] }
        })
    })
    Promise.all(requests)
      .then(results => {
        if (!results.length) return
        const labels = results[0].data.map(item => item.ds)
        const datasets = results.map((r, idx) => {
          const useCorrected = r.data[0]?.yhat_corrected !== undefined
          const values = r.data.map(item => Math.max(0, useCorrected ? (item.yhat_corrected as number) : item.yhat))
          return {
            label: r.group,
            data: values,
            borderColor: palette[idx % palette.length],
            fill: false,
            tension: 0.4
          }
        })
        setChartData({ labels, datasets })
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [groups, periods, useCustom, selectedCategory])

  if (loading) return <p>Loading grouped forecast...</p>
  if (!chartData) return <p>No data to display</p>

  return (
    <div className="chart-container">
      <h2>{groupBy === 'category' ? '분류별 예측 수량' : '품목별 예측 수량'}</h2>
      <Line
        data={chartData}
        options={{
          responsive: true,
          scales: {
            x: { title: { display: true, text: '날짜' } },
            y: { title: { display: true, text: '수량(박스)' } }
          },
          plugins: {
            legend: { position: 'bottom' },
            datalabels: { display: false },
            tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}` } }
          }
        }}
      />
    </div>
  )
}

export default GroupedForecastChart 