import React, { useState, useEffect, useMemo, ReactElement } from 'react';
import axios from 'axios'
import TrendChart, { ShippingTable } from './components/TrendChart'
import ForecastChart from './components/ForecastChart'
import ItemCategorySelector from './components/ItemCategorySelector'
import RealtimeShipmentChart from './components/RealtimeShipmentChart'
import WeekdayTrendChart from './components/WeekdayTrendChart'
import InventoryTable from './components/InventoryTable'
import './App.css'
import './ModernNav.css'
import { useState as useAppState } from 'react'

// Removed OverviewChart and dimension selector per user request

const App = (): ReactElement => {
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [selectedCategories, setSelectedCategories] = useState<string[]>([])
  const [rangeType, setRangeType] = useState<string>('week')
  const [fromDate, setFromDate] = useState<string>('')
  const [toDate, setToDate] = useState<string>('')
  // Forecast period selection
  const [forecastRangeType, setForecastRangeType] = useState<string>('week')
  const [forecastDays, setForecastDays] = useState<number>(7)
  // Toggle custom Prophet seasonalities in forecast
  const [useCustomForecast, setUseCustomForecast] = useState<boolean>(false)
  const [useExogenousForecast, setUseExogenousForecast] = useState<boolean>(false)
  // AI 모델 및 인사이트
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('qwen3:4b')
  const [insight, setInsight] = useState<string>('')
  const [page, setPage] = useState<string>('realtime')
  const [loading, setLoading] = useState<boolean>(false)
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([])
  const [question, setQuestion] = useState<string>('')
  // For triggering manual refresh: realtime/status 개별 처리
  const [realtimeRefreshKey, setRealtimeRefreshKey] = useState<number>(0)
  const [statusRefreshKey, setStatusRefreshKey] = useState<number>(0)
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false)
  const [lastRefreshTime, setLastRefreshTime] = useState<string>('')
  const [showWeekdayTrend, setShowWeekdayTrend] = useState(false)
  // 현황분석 데이터 상태를 App에서 관리
  const [trendData, setTrendData] = useAppState<any[]>([])
  // 카테고리별 일별 합계 데이터 상태
  const [categoryTrend, setCategoryTrend] = useState<any[]>([])
  // Inventory page refresh key
  const [inventoryRefreshKey, setInventoryRefreshKey] = useState<number>(0)
  // Access log viewer state, dynamic IP-to-name mapping, and hourly summary (dev only)
  const [accessLogs, setAccessLogs] = useState<{ ts: string; ip: string; method: string; path: string; name: string }[]>([])
  const [ipNameMap, setIpNameMap] = useState<Record<string, string>>({ '59.9.19.188': '회사 내컴' })
  const [mappingChanged, setMappingChanged] = useState<boolean>(false)
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false)
  const [logSummary, setLogSummary] = useState<{ hour: number; count: number }[]>([])
  const [uploadedFileName, setUploadedFileName] = useState<string>('')
  // New AI report states
  const [reportDate, setReportDate] = useState<string>(new Date().toISOString().split('T')[0])
  const [dailySummary, setDailySummary] = useState<string>('')
  const [aiMessages, setAiMessages] = useState<{ role: string; content: string }[]>([])
  const [aiQuestion, setAiQuestion] = useState<string>('')
  const [aiLoading, setAiLoading] = useState<boolean>(false)

  // global refresh handler
  const handleGlobalRefresh = async () => {
    setIsRefreshing(true);
    try {
      if (page === 'realtime') {
        await axios.post('/api/realtime/refresh');
        setRealtimeRefreshKey(prev => prev + 1);
      } else if (page === 'status') {
        await axios.post('/api/refresh-data');
        setStatusRefreshKey(prev => prev + 1);
      } else if (page === 'inventory') {
        // refresh inventory data
        await axios.post('/api/refresh-data');
        setInventoryRefreshKey(prev => prev + 1);
      }
      setLastRefreshTime(new Date().toLocaleString());
    } catch (err) {
      console.error('Global refresh error:', err);
      alert('데이터 새로고침 중 오류가 발생했습니다.');
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    if (rangeType === 'custom') return
    const now = new Date()
    const to = now.toISOString().split('T')[0]
    let from = to
    switch (rangeType) {
      case 'day':
        from = to
        break
      case 'week':
        from = new Date(now.getTime() - 6*24*60*60*1000).toISOString().split('T')[0]
        break
      case 'month': {
        const m = new Date(now.getFullYear(), now.getMonth()-1, now.getDate())
        from = m.toISOString().split('T')[0]
      } break
      case 'quarter': {
        const qm = Math.floor(now.getMonth() / 3) * 3
        const qStart = new Date(now.getFullYear(), qm, 1)
        from = qStart.toISOString().split('T')[0]
      } break
      case 'year': {
        const y = new Date(now.getFullYear()-1, now.getMonth(), now.getDate())
        from = y.toISOString().split('T')[0]
      } break
    }
    setFromDate(from)
    setToDate(to)
  }, [rangeType])

  useEffect(() => {
    switch (forecastRangeType) {
      case 'week': setForecastDays(7); break
      case 'month': setForecastDays(30); break
      case 'year': setForecastDays(365); break
      default: break
    }
  }, [forecastRangeType])

  // fetch available Ollama models
  useEffect(() => {
    axios.get('/api/models')
      .then(res => {
        setModels(res.data);
        if (res.data.length && !res.data.includes(selectedModel)) setSelectedModel('qwen3:4b');
      })
      .catch(err => console.error('Models fetch error:', err));
  }, [])

  // 카테고리별 일별 합계 가져오기
  useEffect(() => {
    if (page !== 'status' || !fromDate || !toDate) return;
    // status 페이지에서 statusRefreshKey 변경시 데이터 갱신
    axios.post('/api/trend-by-category', {
      item: selectedItems,
      category: selectedCategories,
      from_date: fromDate,
      to_date: toDate
    })
    .then(res => setCategoryTrend(res.data))
    .catch(err => console.error(err));
  }, [page, selectedItems, selectedCategories, fromDate, toDate, statusRefreshKey]);

  // Fetch access.log when on Forecast or Logs page (development only)
  useEffect(() => {
    if ((page === 'forecast' || page === 'logs') && import.meta.env.DEV) {
      fetch('/access.log')
        .then(res => res.text())
        .then(text => {
          const lines = text.trim().split('\n').filter(l => l)
          const parsed = lines.map(line => {
            const [ts, ip, method, path] = line.split('\t');
            return { ts, ip, method, path, name: ipNameMap[ip] || ip };
          });
          setAccessLogs(parsed);
          // compute hourly summary
          const summaryMap: Record<number, number> = {};
          parsed.forEach(log => {
            const hour = new Date(log.ts).getHours();
            summaryMap[hour] = (summaryMap[hour] || 0) + 1;
          });
          const summaryArray = Object.entries(summaryMap)
            .map(([h, cnt]) => ({ hour: +h, count: cnt }))
            .sort((a, b) => a.hour - b.hour);
          setLogSummary(summaryArray);
        })
        .catch(err => console.error('Access log fetch error:', err));
    }
  }, [page]);

  // Load saved IP-name mappings from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('ipNameMap')
    if (saved) {
      try { setIpNameMap(JSON.parse(saved)) } catch {}
    }
  }, [])

  // Persist IP-name mapping whenever it changes
  useEffect(() => {
    localStorage.setItem('ipNameMap', JSON.stringify(ipNameMap))
  }, [ipNameMap])

  // Handle user-uploaded log file
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadedFileName(file.name)
    const reader = new FileReader()
    reader.onload = () => {
      const text = reader.result as string
      const lines = text.trim().split('\n').filter(l => l)
      const parsed = lines.map(line => {
        const [ts, ip, method, path] = line.split('\t')
        return { ts, ip, method, path, name: ipNameMap[ip] || ip }
      })
      setAccessLogs(parsed)
      // compute summary
      const summaryMap: Record<number, number> = {}
      parsed.forEach(log => {
        const hour = new Date(log.ts).getHours()
        summaryMap[hour] = (summaryMap[hour] || 0) + 1
      })
      const summaryArray = Object.entries(summaryMap)
        .map(([h, cnt]) => ({ hour: +h, count: cnt }))
        .sort((a, b) => a.hour - b.hour)
      setLogSummary(summaryArray)
    }
    reader.readAsText(file)
  }

  // Save IP-name mappings manually
  const handleSaveMappings = () => {
    try {
      localStorage.setItem('ipNameMap', JSON.stringify(ipNameMap))
      setMappingChanged(false)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 2000)
    } catch (e) {
      console.error('Failed to save mappings', e)
      alert('이름 저장 중 오류가 발생했습니다.')
    }
  }

  // Compute unique IPs and detect missing names
  const uniqueIps = Array.from(new Set(accessLogs.map(log => log.ip)));
  const missingIps = uniqueIps.filter(ip => !ipNameMap[ip] || ipNameMap[ip] === ip);
  // Compute unique visitor names per hour
  const visitorByHour = useMemo(() => {
    const map: Record<number, Set<string>> = {};
    accessLogs.forEach(log => {
      const hour = new Date(log.ts).getHours();
      const name = ipNameMap[log.ip] || log.ip;
      if (!map[hour]) map[hour] = new Set();
      map[hour].add(name);
    });
    return Object.entries(map)
      .map(([h, set]) => ({ hour: +h, names: Array.from(set) }))
      .sort((a, b) => a.hour - b.hour);
  }, [accessLogs, ipNameMap]);

  // Fetch daily AI report summary when page is ai-report or date changes
  useEffect(() => {
    if (page === 'ai-report' && reportDate) {
      setAiLoading(true);
      setDailySummary('AI 요약 로딩 중...'); 
      axios.get(`/api/insights/${reportDate}`)
        .then(res => {
          setDailySummary(res.data.summary || '해당 날짜에 대한 요약이 없습니다.');
        })
        .catch(err => {
          console.error('Fetch daily AI summary error:', err);
          if (axios.isAxiosError(err) && err.response && err.response.status === 404) {
            setDailySummary(`해당 날짜(${reportDate})에 대한 AI 요약이 아직 생성되지 않았습니다. '조회' 버튼을 눌러 상세 보고서 생성을 시도하면 요약도 함께 생성될 수 있습니다.`);
          } else if (axios.isAxiosError(err) && err.response) {
            setDailySummary(`일일 AI 요약 로딩 중 오류가 발생했습니다: ${err.response.status} ${err.response.data?.detail || err.response.statusText}. 상세 내용은 콘솔을 확인하세요.`);
          } 
          else {
            setDailySummary('일일 AI 요약 로딩 중 네트워크 또는 알 수 없는 오류가 발생했습니다.');
          }
        })
        .finally(() => {
          setAiLoading(false);
        });
    }
  }, [page, reportDate]); // Dependencies: page and reportDate

  return (
    <div className="app-container">
      {/* 1. 페이지 제목 */}
      <div className="comp-item">
        <span className="comp-label">1</span>
        <h1>출고 수량 분석 대시보드</h1>
      </div>
      {/* 3. 페이지 네비게이션 */}
      <nav className="modern-nav">
        <span className="comp-label">3</span>
        <button className={`modern-nav-btn${page === 'realtime' ? ' active' : ''}`} onClick={() => { setPage('realtime'); setRealtimeRefreshKey(prev => prev + 1); }}>당일 출고 현황</button>
        <button className={`modern-nav-btn${page === 'status' ? ' active' : ''}`} onClick={() => setPage('status')} disabled={page === 'status'}>현황 분석</button>
        <button className={`modern-nav-btn${page === 'inventory' ? ' active' : ''}`} onClick={() => { setPage('inventory'); setInventoryRefreshKey(prev => prev + 1); }} disabled={page === 'inventory'}>전산 재고 수량</button>
        <button className={`modern-nav-btn${page === 'looker' ? ' active' : ''}`} onClick={() => setPage('looker')} disabled={page === 'looker'}>Looker Studio 리포트</button>
        <button className={`modern-nav-btn${page === 'forecast' ? ' active' : ''}`} onClick={() => setPage('forecast')} disabled={page === 'forecast'}>예측 분석</button>
        <button className={`modern-nav-btn${page === 'ai-report' ? ' active' : ''}`} onClick={() => setPage('ai-report')} disabled={page === 'ai-report'}>AI 보고서</button>
        {import.meta.env.DEV && (
          <button className={`modern-nav-btn${page === 'logs' ? ' active' : ''}`} onClick={() => setPage('logs')} disabled={page === 'logs'}>접속 로그</button>
        )}
      </nav>
      {/* 4. 기간 유형 선택 및 데이터 새로고침 */}
      {page !== 'inventory' && (
      <div className="control-panel comp-item">
        <span className="comp-label">4</span>
        <label>기간 유형: </label>
        <select value={rangeType} onChange={e => setRangeType(e.target.value)}>
          <option value="day">일간</option>
          <option value="week">주간</option>
          <option value="month">월간</option>
          <option value="quarter">분기</option>
          <option value="year">연간</option>
          <option value="custom">사용자 지정</option>
        </select>
        {rangeType === 'custom' && (
          <>
            <label htmlFor="from-date">시작 일자: </label>
            <input type="date" id="from-date" value={fromDate} onChange={e => setFromDate(e.target.value)} />
            <label htmlFor="to-date">종료 일자: </label>
            <input type="date" id="to-date" value={toDate} onChange={e => setToDate(e.target.value)} />
          </>
        )}
        <button className="modern-btn" onClick={handleGlobalRefresh} disabled={isRefreshing}>
          <span className="comp-label">2</span> {isRefreshing ? '새로고침 중...' : '데이터 새로고침'}
        </button>
        {lastRefreshTime && (
          <span className="refresh-time" style={{ marginLeft: '12px', color: '#555' }}>
            최근: {lastRefreshTime}
          </span>
        )}
      </div>
      )}

      {/* 당일 출고 현황 페이지 */}
      {page === 'realtime' && (
        <>
          <div className="comp-item">
            <button onClick={() => setShowWeekdayTrend(v => !v)}>
              {showWeekdayTrend ? '당일 출고 현황 보기' : '요일별 출고 트렌드 보기'}
            </button>
          </div>
          {showWeekdayTrend
            ? <WeekdayTrendChart />
            : <RealtimeShipmentChart refreshKey={realtimeRefreshKey} />}
        </>
      )}

      {/* 현황 분석 페이지 */}
      {page === 'status' && (
        <>
          {/* 5. 아이템/분류 선택 */}
          <div className="comp-item">
            <span className="comp-label">5</span>
            <ItemCategorySelector
              selectedItems={selectedItems}
              selectedCategories={selectedCategories}
              onItemChange={setSelectedItems}
              onCategoryChange={setSelectedCategories}
            />
          </div>
          {/* 6. 트렌드 차트 */}
          <div className="chart-container">
            <TrendChart
              item={selectedItems}
              category={selectedCategories}
              fromDate={fromDate}
              toDate={toDate}
              data={trendData}
              setData={setTrendData}
              refreshKey={statusRefreshKey}
            />
          </div>
          {/* 6-1. 출고량/판매금액 표 */}
          <div className="chart-container">
            <ShippingTable
              data={categoryTrend}
              fromDate={fromDate}
              toDate={toDate}
            />
          </div>
          {/* 7. AI 인사이트 챗봇 */}
          <div className="comp-item ai-container">
            <span className="comp-label">7</span>
            {loading ? (
              <p>심층 출고량 분석 보고서 생성 중...</p>
            ) : (
              <div className="insight-report-box">
                {messages.map((m, idx) => (
                  <div key={idx} className={`chat-message ${m.role}`}> 
                    <strong>{m.role === 'user' ? '사용자' : 'AI'}:</strong> {m.content}
                  </div>
                ))}
              </div>
            )}
            {/* 질문 입력 */}
            <div className="chat-input">
              <input type="text" value={question} placeholder="추가 질문을 입력하세요" onChange={e => setQuestion(e.target.value)} />
              <button disabled={loading || !question} onClick={async () => {
                const q = question;
                setLoading(true);
                setMessages(prev => [...prev, { role: 'user', content: q }]);
                setQuestion('');
                try {
                  const res = await axios.post('/api/insight', {
                    item: selectedItems,
                    category: selectedCategories,
                    from_date: fromDate,
                    to_date: toDate,
                    model: selectedModel,
                    question: q
                  });
                  setMessages(prev => [...prev, { role: 'assistant', content: res.data.insight }]);
                } catch (err) {
                  console.error(err);
                  setMessages(prev => [...prev, { role: 'assistant', content: '질문 처리 중 오류가 발생했습니다.' }]);
                } finally {
                  setLoading(false);
                }
              }}>질문</button>
              <button disabled={loading} onClick={() => { setMessages([]); setQuestion(''); }}>초기화</button>
            </div>
          </div>
        </>
      )}
      {/* 전산 재고 수량 페이지 */}
      {page === 'inventory' && (
        <InventoryTable
          refreshKey={inventoryRefreshKey}
          onRefresh={handleGlobalRefresh}
          isRefreshing={isRefreshing}
          lastRefreshTime={lastRefreshTime}
        />
      )}
      {/* 예측 분석 페이지 */}
      {page === 'forecast' && (
        <div style={{ display: 'flex' }}>
          {/* Forecast chart */}
          <div style={{ flex: 1 }}>
            <ForecastChart
              item={selectedItems}
              category={selectedCategories}
              periods={forecastDays}
              fromDate={fromDate}
              lastDate={toDate}
              useCustom={useCustomForecast}
              useExog={useExogenousForecast}
            />
          </div>
          {/* Access log sidebar (dev only) */}
          {import.meta.env.DEV && (
            <div className="access-log-sidebar" style={{ width: '240px', marginLeft: '20px', overflowY: 'auto', maxHeight: '500px', padding: '10px', border: '1px solid #ccc' }}>
              <h3>접속 로그</h3>
              <ul style={{ listStyle: 'none', padding: 0 }}>
                {accessLogs.map((log, idx) => (
                  <li key={idx} style={{ marginBottom: '8px', fontSize: '12px' }}>
                    <div>{log.ts}</div>
                    <div>{log.name} ({log.ip})</div>
                    <div>{log.method} {log.path}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      {/* Looker Studio 리포트 페이지 */}
      {page === 'looker' && (
        <div className="chart-container">
          <h2>Looker Studio 리포트</h2>
          <p>
            아래 버튼을 클릭하면 새 창에서 리포트가 열립니다.<br />
            <a
              href="https://lookerstudio.google.com/s/iPtHIDHt2Zk"
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: '1.2em', color: '#1976d2', textDecoration: 'underline' }}
            >
              Looker Studio 리포트 바로가기
            </a>
          </p>
        </div>
      )}
      {/* AI 보고서 페이지 */}
      {page === 'ai-report' && (
        <div className="chart-container">
          <h2>AI 보고서</h2>
          <div className="comp-item">
            <label>날짜: </label>
            <input type="date" value={reportDate} onChange={e => setReportDate(e.target.value)} />
            <button onClick={async () => {
                if (!reportDate) {
                  alert('날짜를 선택해주세요.');
                  return;
                }
                console.log('[AI Report] Fetching report for date:', reportDate, 'with rangeType:', rangeType);
                setAiLoading(true);
                setDailySummary('AI 보고서 생성 중...'); // Provide immediate feedback
                try {
                  // Calculate from_date based on rangeType and reportDate (acting as to_date for the report)
                  let calculatedFromDate = reportDate;
                  const tempToDate = new Date(reportDate);
                  switch (rangeType) {
                    case 'day':
                      calculatedFromDate = reportDate;
                      break;
                    case 'week':
                      calculatedFromDate = new Date(tempToDate.getTime() - 6 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                      break;
                    case 'month':
                      calculatedFromDate = new Date(tempToDate.getFullYear(), tempToDate.getMonth() - 1, tempToDate.getDate() + 1).toISOString().split('T')[0];
                        break;
                      default:
                        console.warn('[AI Report] Unknown rangeType:', rangeType, 'defaulting to week logic.');
                        calculatedFromDate = new Date(tempToDate.getTime() - 6 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                        break;
                    }
                    console.log('[AI Report] Calculated date range for API:', { from_date: calculatedFromDate, to_date: reportDate });

                  const res = await axios.post('/api/report/shipment', {
                    from_date: calculatedFromDate, // Use calculated from_date
                    to_date: reportDate, // reportDate acts as to_date for the AI report query
                    // item: selectedItems, // Add if AI report needs filtering by item/category
                    // category: selectedCategories
                  });
                  console.log('[AI Report] API Response:', res);

                  if (res.data && res.data.report_markdown) {
                    setDailySummary(res.data.report_markdown);
                  } else {
                    console.warn('[AI Report] Report content is missing in API response:', res.data);
                    setDailySummary('보고서 내용을 가져오지 못했습니다. 응답 데이터 확인 필요.');
                  }
                } catch (err) {
                  console.error('[AI Report] Fetch shipment report error:', err);
                  if (axios.isAxiosError(err) && err.response) {
                    console.error('[AI Report] Error response data:', err.response.data);
                    setDailySummary(`보고서 생성 중 오류가 발생했습니다: ${err.response.status} ${err.response.statusText}. 상세 내용은 콘솔 확인.`);
                  } else {
                    setDailySummary('보고서 생성 중 네트워크 또는 알 수 없는 오류가 발생했습니다.');
                  }
                } finally {
                  setAiLoading(false);
                }
              }} style={{ marginLeft: '8px' }}>조회</button>
          </div>
          <div className="comp-item">
            <h3>요약문</h3>
            <div style={{ whiteSpace: 'pre-wrap', background: '#f9f9f9', padding: '10px', borderRadius: '4px' }}>
              {dailySummary}
            </div>
          </div>
          <div className="comp-item">
            <h3>추가 질문</h3>
            <input type="text" value={aiQuestion} placeholder="질문을 입력하세요" onChange={e => setAiQuestion(e.target.value)} />
            <button disabled={!aiQuestion || aiLoading} onClick={async () => {
              setAiLoading(true)
              try {
                const res = await axios.post(`/api/insights/${reportDate}/question`, { question: aiQuestion })
                setAiMessages(prev => [...prev, { role: 'user', content: aiQuestion }, { role: 'assistant', content: res.data.answer }])
                setAiQuestion('')
              } catch (err) {
                console.error('AI question error:', err)
              }
              setAiLoading(false)
            }} style={{ marginLeft: '8px' }}>질문</button>
          </div>
          <div className="comp-item">
            {aiMessages.map((m, idx) => (
              <div key={idx} className={`chat-message ${m.role}`}>
                <strong>{m.role === 'user' ? '사용자' : 'AI'}:</strong> {m.content}
              </div>
            ))}
          </div>
        </div>
      )}
      {import.meta.env.DEV && page === 'logs' && (
        <div className="chart-container">
          <h2>접속 로그</h2>
          {/* Warn about IPs without names */}
          {missingIps.length > 0 && (
            <div style={{ color: 'red', marginBottom: '12px' }}>
              다음 IP에 이름을 입력하세요: {missingIps.join(', ')}
            </div>
          )}
          <div style={{ display: 'flex' }}>
            {/* Left: upload and mapping inputs */}
            <div style={{ flex: 1, paddingRight: '20px' }}>
              <h3>로그 파일 업로드</h3>
              <input type="file" accept=".log,.txt" onChange={handleFileUpload} />
              {uploadedFileName && <p>파일: {uploadedFileName}</p>}
              <div style={{ marginTop: '16px' }}>
                <h3>IP 별 이름 설정</h3>
                {uniqueIps.map(ip => {
                  const isMissing = missingIps.includes(ip);
                  return (
                  <div key={ip} style={{ marginBottom: '8px', display: 'flex', alignItems: 'center' }}>
                    <label style={{ width: '120px' }}>{ip}:</label>
                    <input
                      type="text"
                      placeholder="이름을 입력하세요"
                      value={ipNameMap[ip] ?? ''}
                      onChange={e => {
                        setIpNameMap(prev => ({ ...prev, [ip]: e.target.value })); setMappingChanged(true)
                      }}
                      style={{ flex: 1, padding: '4px 8px', border: isMissing ? '1px solid red' : '1px solid #ccc', borderRadius: '4px' }}
                    />
                  </div>
                )})}
                {/* Save mappings button */}
                <div style={{ marginTop: '12px' }}>
                  <button onClick={handleSaveMappings} disabled={!mappingChanged} style={{ padding: '6px 12px', borderRadius: '4px', cursor: mappingChanged ? 'pointer' : 'not-allowed' }}>
                    저장
                  </button>
                  {saveSuccess && <span style={{ color: 'green', marginLeft: '8px' }}>저장되었습니다.</span>}
                </div>
              </div>
            </div>
            {/* Right: hourly summary and detailed logs */}
            <div style={{ flex: 2 }}>
              <div style={{ marginBottom: '16px' }}>
                <h3>시간별 요청 요약</h3>
                <ul>
                  {logSummary.map(item => (
                    <li key={item.hour}>{`${item.hour.toString().padStart(2, '0')}:00 - ${item.count}건`}</li>
                  ))}
                </ul>
              </div>
              <div style={{ marginBottom: '16px' }}>
                <h3>시간별 방문자</h3>
                <ul>
                  {visitorByHour.map(item => (
                    <li key={item.hour}>{`${item.hour.toString().padStart(2, '0')}:00 - ${item.names.join(', ')}`}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h3>상세 로그</h3>
                <ul style={{ listStyle: 'none', padding: 0 }}>
                  {accessLogs.map((log, idx) => (
                    <li key={idx} style={{ marginBottom: '8px' }}>
                      <div>{log.ts}</div>
                      <div>{(ipNameMap[log.ip] ?? log.ip)} ({log.ip})</div>
                      <div>{log.method} {log.path}</div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App