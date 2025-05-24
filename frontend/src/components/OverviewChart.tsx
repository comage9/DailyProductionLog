import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Bar } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

interface DataItem {
  [key: string]: any
  '수량(박스)': number
}

interface OverviewChartProps {
  dimension: string
}

const OverviewChart: React.FC<OverviewChartProps> = ({ dimension }) => {
  const [data, setData] = useState<DataItem[]>([])

  useEffect(() => {
    axios.get(`/api/overview?dimension=${dimension}`)
      .then(res => setData(res.data))
      .catch(err => console.error(err))
  }, [dimension])

  const labels = data.map(item => String(item[dimension]))
  const values = data.map(item => item['수량(박스)'] || 0)

  const chartData = {
    labels,
    datasets: [
      {
        label: '수량(박스)',
        data: values,
        backgroundColor: 'rgba(75,192,192,0.6)'
      }
    ]
  }

  return (
    <div className="chart-container">
      <Bar data={chartData} />
    </div>
  )
}

export default OverviewChart 