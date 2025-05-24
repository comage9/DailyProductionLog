import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line, Bar, Pie } from 'react-chartjs-2';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import '../components/DetailedReport.css';

interface DetailedReportProps {
  fromDate: string;
  toDate: string;
  item?: string | string[];
  category?: string | string[];
}

const DetailedReport: React.FC<DetailedReportProps> = ({ fromDate, toDate, item, category }) => {
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const fetch = async () => {
      setLoading(true);
      try {
        const res = await axios.post('/api/report/detailed', {
          from_date: fromDate,
          to_date: toDate,
          item,
          category
        });
        setReport(res.data);
      } catch (e) {
        setError('보고서 로드 실패');
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, [fromDate, toDate, item, category]);

  const exportPdf = async () => {
    const el = document.getElementById('detailed-report');
    if (!el) return;
    const canvas = await html2canvas(el);
    const img = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const w = 210;
    const h = (canvas.height * w) / canvas.width;
    pdf.addImage(img, 'PNG', 0, 0, w, h);
    pdf.save(`report_${fromDate}_${toDate}.pdf`);
  };

  if (loading) return <div className="loading">생성 중...</div>;
  if (error || !report) return <div className="error">{error || '데이터 없음'}</div>;

  return (
    <div id="detailed-report" className="detailed-report">
      {/* 여기에 보고서 구조 렌더링 */}
      <button onClick={exportPdf} className="export-button">PDF 저장</button>
    </div>
  );
};

export default DetailedReport;
