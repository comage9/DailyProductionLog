import React, { useEffect, useState, useRef, useMemo } from 'react'
import axios from 'axios'

interface InventoryRow { [key: string]: string }

// API endpoint for inventory data
const API_URL = '/api/inventory'
// Fallback CSV URL for direct fetch
const CSV_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQwqI0BG-d2aMrql7DK4fQQTjvu57VtToSLAkY_nq92a4Cg5GFVbIn6_IR7Fq6_O-2TloFSNlXT8ZWC/pub?gid=2125795373&single=true&output=csv'

// 표시할 컬럼 순서 정의 (미입고 수량 컬럼은 제외)
const displayColumns = [
  '일자',
  '로케이션',
  '분류',
  '품명',
  '바코드',
  '전산 재고 수량',
  '일 평균 출고 수량',
  '단수'
]

interface InventoryTableProps {
  refreshKey?: number
  onRefresh: () => void
  isRefreshing: boolean
  lastRefreshTime?: string
}
const InventoryTable: React.FC<InventoryTableProps> = ({ refreshKey, onRefresh, isRefreshing, lastRefreshTime }) => {
  const [rows, setRows] = useState<InventoryRow[]>([])
  // Filtering state
  const [filteredRows, setFilteredRows] = useState<InventoryRow[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [items, setItems] = useState<string[]>([])
  const [selectedCategories, setSelectedCategories] = useState<string[]>([])
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [openCategories, setOpenCategories] = useState(false)
  const [openItems, setOpenItems] = useState(false)
  const catWrapperRef = useRef<HTMLDivElement>(null)
  const itemWrapperRef = useRef<HTMLDivElement>(null)
  const [countdown, setCountdown] = useState<number>(15)
  // Header filter state
  const [filters, setFilters] = useState<Record<string, string[]>>({})
  const [openFilter, setOpenFilter] = useState<Record<string, boolean>>(
    () => displayColumns.reduce((acc, col) => ({ ...acc, [col]: false }), {})
  )
  const tableWrapperRef = useRef<HTMLDivElement>(null)

  // sorted rows by location for consistent ordering across filters
  const sortedRowsByLocation = useMemo(() => {
    if (rows.length === 0) return []
    // determine the most recent date in '일자'
    const dates = rows.map(r => r['일자'] || '')
    const maxDate = dates.reduce((a, b) => (a > b ? a : b), '')
    // filter to rows of the latest date
    const latestRows = rows.filter(r => r['일자'] === maxDate)
    // prioritize items with numeric location and place others at the bottom
    return [...latestRows].sort((a, b) => {
      const locA = a['로케이션'] || ''
      const locB = b['로케이션'] || ''
      const numA = Number(locA)
      const numB = Number(locB)
      const hasA = locA !== '' && !isNaN(numA)
      const hasB = locB !== '' && !isNaN(numB)
      if (hasA && hasB) {
        return numA - numB
      }
      if (hasA && !hasB) {
        return -1
      }
      if (!hasA && hasB) {
        return 1
      }
      return 0
    })
  }, [rows])

  useEffect(() => {
    setCountdown(15)
    axios.get(API_URL)
      .then(res => setRows(res.data as InventoryRow[]))
      .catch(err => {
        console.error('Inventory API fetch error:', err)
        // Fallback to direct CSV fetch
        axios.get(CSV_URL, { responseType: 'text' })
          .then(res2 => {
            const text = res2.data as string
            const lines = text.trim().split('\n')
            if (lines.length < 1) return
            const header = lines[0].split(',')
            const dataRows = lines.slice(1)
              .filter(line => line.trim() !== '')
              .map(line => {
                const values = line.split(',')
                const row: InventoryRow = {}
                header.forEach((col, idx) => { row[col] = values[idx] || '' })
                return row
              })
            setRows(dataRows)
          })
          .catch(err2 => {
            console.error('Inventory direct CSV fetch error:', err2)
          })
      })
  }, [refreshKey])

  // Derive categories, items, and initial filteredRows when rows change
  useEffect(() => {
    const cats = Array.from(new Set(rows.map(r => r['분류']))).sort()
    setCategories(cats)
    const its = Array.from(new Set(rows.map(r => r['품명']))).sort()
    setItems(its)
    // show only the most recent date sorted by location
    setFilteredRows(sortedRowsByLocation)
  }, [rows])

  // Apply filters when selectedCategories or selectedItems change
  useEffect(() => {
    let data = sortedRowsByLocation
    if (selectedCategories.length > 0) {
      data = data.filter(r => selectedCategories.includes(r['분류']))
    }
    if (selectedItems.length > 0) {
      data = data.filter(r => selectedItems.includes(r['품명']))
    }
    setFilteredRows(data)
  }, [sortedRowsByLocation, selectedCategories, selectedItems])

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (openCategories && catWrapperRef.current && !catWrapperRef.current.contains(event.target as Node)) {
        setOpenCategories(false)
      }
      if (openItems && itemWrapperRef.current && !itemWrapperRef.current.contains(event.target as Node)) {
        setOpenItems(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => { document.removeEventListener('mousedown', handleClickOutside) }
  }, [openCategories, openItems])

  // 카운트다운 타이머
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>
    if (countdown > 0) {
      timer = setInterval(() => setCountdown(prev => Math.max(prev - 1, 0)), 1000)
    }
    return () => clearInterval(timer)
  }, [countdown])

  // Compute unique values for each column
  const uniqueValues = useMemo(() => {
    const map: Record<string, string[]> = {}
    displayColumns.forEach(col => {
      const vals = Array.from(new Set(rows.map(r => r[col] ?? '')))
      map[col] = vals.sort((a, b) =>
        col === '로케이션'
          ? a.localeCompare(b, undefined, { numeric: true })
          : a.localeCompare(b)
      )
    })
    return map
  }, [rows])

  // Apply header filters to rows
  useEffect(() => {
    let data = sortedRowsByLocation
    for (const col of displayColumns) {
      const sel = filters[col]
      if (sel && sel.length > 0) {
        data = data.filter(r => sel.includes(r[col] ?? ''))
      }
    }
    setFilteredRows(data)
  }, [sortedRowsByLocation, filters])

  // Close header dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (tableWrapperRef.current && !tableWrapperRef.current.contains(event.target as Node)) {
        setOpenFilter(displayColumns.reduce((acc, col) => ({ ...acc, [col]: false }), {}))
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // CSV 다운로드 (Excel로 열 수 있음)
  const downloadCSV = () => {
    const header = displayColumns.join(',')
    const body = rows.map(row =>
      displayColumns.map(col => `"${row[col] || ''}"`).join(',')
    ).join('\n')
    const csv = header + '\n' + body
    // UTF-8 BOM을 추가해 Excel에서 한글 인코딩 문제 해결
    const blob = new Blob(["\uFEFF" + csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.setAttribute('download', `inventory_${new Date().toISOString().slice(0,10)}.csv`)
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const isLoading = rows.length === 0

  return (
    <div className="chart-container">
      <div className="comp-item">
        <span className="comp-label">6</span>
        <h2>전산 재고 수량</h2>
      </div>
      {isLoading && (
        <div style={{ textAlign: 'center', color: '#555', margin: '8px 0' }}>
          ⚠️ 데이터 로딩 중... {countdown}초 남음
        </div>
      )}
      <div className="control-panel comp-item" style={{ gap: '8px' }}>
        <button className="modern-btn" onClick={downloadCSV}>엑셀 다운로드</button>
        <button className="modern-btn" onClick={onRefresh} disabled={isRefreshing}>
          <span className="comp-label">2</span> {isRefreshing ? '새로고침 중...' : '데이터 새로고침'}
        </button>
        {lastRefreshTime && (
          <span className="refresh-time" style={{ marginLeft: '12px', color: '#555' }}>
            최근: {lastRefreshTime}
          </span>
        )}
      </div>
      {false && (
      <div className="control-panel multi-selects">
        <label>분류: </label>
        <div className="multi-select-wrapper" ref={catWrapperRef}>
          <button type="button" className="multi-select-btn" onClick={() => setOpenCategories(o => !o)}>
            {selectedCategories.length === 0 ? '전체 선택' : selectedCategories.length === 1 ? selectedCategories[0] : `${selectedCategories[0]} 외 ${selectedCategories.length - 1}개`}
          </button>
          {openCategories && (
            <div className="multi-dropdown">
              <label>
                <input
                  type="checkbox"
                  checked={selectedCategories.length === categories.length}
                  onChange={e => setSelectedCategories(e.target.checked ? [...categories] : [])}
                /> 전체 선택
              </label>
              {categories.map(cat => (
                <label key={cat}>
                  <input
                    type="checkbox"
                    value={cat}
                    checked={selectedCategories.includes(cat)}
                    onChange={e => {
                      const value = e.target.value
                      const checked = e.target.checked
                      setSelectedCategories(prev =>
                        checked ? Array.from(new Set([...prev, value])) : prev.filter(v => v !== value)
                      )
                    }}
                  /> {cat}
                </label>
              ))}
            </div>
          )}
        </div>

        <label>품명: </label>
        <div className="multi-select-wrapper" ref={itemWrapperRef}>
          <button type="button" className="multi-select-btn" onClick={() => setOpenItems(o => !o)}>
            {selectedItems.length === 0 ? '전체 선택' : selectedItems.length === 1 ? selectedItems[0] : `${selectedItems[0]} 외 ${selectedItems.length - 1}개`}
          </button>
          {openItems && (
            <div className="multi-dropdown">
              <label>
                <input
                  type="checkbox"
                  checked={selectedItems.length === items.length}
                  onChange={e => setSelectedItems(e.target.checked ? [...items] : [])}
                /> 전체 선택
              </label>
              {items.map(it => (
                <label key={it}>
                  <input
                    type="checkbox"
                    value={it}
                    checked={selectedItems.includes(it)}
                    onChange={e => {
                      const value = e.target.value
                      const checked = e.target.checked
                      setSelectedItems(prev =>
                        checked ? Array.from(new Set([...prev, value])) : prev.filter(v => v !== value)
                      )
                    }}
                  /> {it}
                </label>
              ))}
            </div>
          )}
        </div>
      </div>
      )}

      <div style={{ overflowX: 'auto' }} ref={tableWrapperRef}>
        <table className="inventory-table">
          <thead>
            <tr>
              {displayColumns.map(col => (
                <th
                  key={col}
                  style={{
                    textAlign: 'center',
                    fontSize: '18px',
                    padding: '8px 12px',
                    backgroundColor: filters[col]?.length > 0 ? '#ffe082' : undefined,
                    minWidth: filters[col]?.length > 0 ? '200px' : undefined
                  }}
                >
                  <div style={{ position: 'relative', display: 'inline-block' }}>
                    {col}
                    <span
                      style={{ marginLeft: '4px', cursor: 'pointer' }}
                      onClick={() => setOpenFilter(prev => ({ ...prev, [col]: !prev[col] }))}
                    >▾</span>
                    {openFilter[col] && (
                      <div
                        className="multi-dropdown"
                        style={{
                          position: 'absolute',
                          top: '100%',
                          left: 0,
                          zIndex: 1000,
                          background: '#fff',
                          border: '1px solid #ccc',
                          padding: '8px',
                          textAlign: 'left',
                          height: '800px',
                          maxHeight: '800px',
                          overflowY: 'auto'
                        }}
                        onMouseLeave={() => setOpenFilter(prev => ({ ...prev, [col]: false }))}
                      >
                        <label>
                          <input
                            type="checkbox"
                            checked={filters[col]?.length === uniqueValues[col].length}
                            onChange={e =>
                              setFilters(prev => ({
                                ...prev,
                                [col]: e.target.checked ? [...uniqueValues[col]] : []
                              }))
                            }
                          /> 전체 선택
                        </label>
                        {uniqueValues[col].map(val => (
                          <label key={val} style={{ display: 'block' }}>
                            <input
                              type="checkbox"
                              value={val}
                              checked={filters[col]?.includes(val) ?? false}
                              onChange={e => {
                                const { value, checked } = e.target
                                setFilters(prev => {
                                  const prevSel = prev[col] || []
                                  const newSel = checked
                                    ? Array.from(new Set([...prevSel, value]))
                                    : prevSel.filter(v => v !== value)
                                  return { ...prev, [col]: newSel }
                                })
                              }}
                            /> {val}
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row, idx) => {
              const inv = Number(row['전산 재고 수량'])
              const avg = Number(row['일 평균 출고 수량'])
              const isLow = inv < avg * 4
              return (
                <tr key={idx} style={{ color: isLow ? '#d32f2f' : undefined }}>
                  {displayColumns.map(col => (
                    <td
                      key={col}
                      style={{
                        textAlign: 'center',
                        whiteSpace: col === '품명' ? 'nowrap' : 'normal',
                        padding: '8px 12px',
                        backgroundColor: filters[col]?.length > 0 ? '#ffe082' : undefined,
                        minWidth: filters[col]?.length > 0 ? '200px' : undefined
                      }}
                    >
                      {
                        col === '바코드'
                          ? (row[col]?.replace(/\.0+$/, '') || '')
                          : row[col]
                      }
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default InventoryTable
