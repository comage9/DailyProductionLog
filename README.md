# Analytics Dashboard 프로젝트

## 프로젝트 개요

**Analytics Dashboard**는 출고/판매 데이터를 기반으로 실시간 현황, 트렌드, 예측, AI 인사이트 분석을 제공하는 통합 대시보드입니다. 구글 시트 데이터를 자동으로 동기화하여, 웹 기반으로 시각화 및 심층 분석을 지원합니다.

---

## 주요 기능
- **구글 시트 연동**: 출고/판매 데이터를 주기적으로 자동 동기화(증분 upsert)
- **현황 분석**: 일별/주간/월간 출고량, 판매금액, 품목/분류별 트렌드 시각화
- **예측 분석**: Prophet 기반 미래 출고량 예측, 신뢰구간 제공
- **AI 인사이트**: (옵션) LLM 기반 심층 보고서 자동 생성(현재 중지됨)
- **실시간/요일별 출고 현황**: 당일 및 최근 4주 요일별 시간대별 출고량 분석
- **쿠팡 '보노 하우스' 상품 리포트**: 쿠팡에서 '보노 하우스' 관련 상품의 리뷰, Q&A 등 스크랩 및 제공
- **Looker Studio 연동**: 외부 BI 리포트 바로가기

### 쿠팡 스크래퍼 개발 현황 및 이슈 (2025-05-23 기준)

현재 쿠팡 상품 정보(상품명, 가격, 이미지, 리뷰, Q&A 등)를 스크랩하는 기능의 안정화 및 개선 작업이 진행 중입니다. 주요 진행 사항 및 당면 과제는 다음과 같습니다.

**1. 주요 진행 상황**
*   **알림창 자동 처리**: 스크래핑 중 발생하는 예기치 않은 알림창(예: "데이터 처리에 실패했습니다. 다시 시도해 주세요.")을 감지하고 자동으로 닫는 로직을 구현하여 안정성을 높였습니다.
*   **CSS 선택자 검토 및 수정 시도**: 상품명(`h2.prod-buy-header__title`) 및 가격(`.prod-buy-price .total-price strong`) 등 주요 정보 추출을 위한 CSS 선택자를 지속적으로 검토하고 수정 시도하고 있습니다.
*   **대기 시간 증가**: 동적 컨텐츠 로딩을 위해 `WebDriverWait`의 대기 시간을 30초로 늘려, 페이지 요소가 나타날 시간을 충분히 확보하고자 했습니다.
*   **디버깅을 위한 상세 로깅 추가**:
    *   `TimeoutException` 또는 `UnexpectedAlertPresentException`과 같은 주요 예외 발생 시, 해당 시점의 페이지 소스 일부를 로깅하여 원인 분석 자료로 활용하고 있습니다.
    *   상품명 요소 탐색 직전에 현재 페이지 소스의 일부를 로깅하는 기능을 추가하여, 요소 미발견 시점의 DOM(Document Object Model) 상태를 파악할 수 있도록 개선했습니다. (현재 이 로그가 터미널에 정상적으로 출력되는지 확인하는 과정에 있습니다.)

**2. 현재 주요 문제점 및 해결 과제**
*   **상품명/가격 추출의 지속적 실패**: 가장 큰 문제점으로, 상품명과 가격 정보 추출 시 `TimeoutException`이 계속 발생하고 있습니다. 이는 현재 사용 중인 CSS 선택자가 실제 쿠팡 페이지의 HTML 구조와 일치하지 않거나, 페이지가 완전히 로드되지 않은 상태에서 요소 탐색을 시도하기 때문일 가능성이 높습니다.
*   **핵심 정보 확보의 어려움**: 위의 문제로 인해, 스크래퍼의 핵심 목표인 상품명, 가격, 이미지 등의 주요 정보를 안정적으로 가져오지 못하고 있습니다.
*   **추가된 디버깅 로그 미출력**: 상품명 탐색 직전의 페이지 소스를 로깅하도록 코드를 수정했음에도 불구하고, 최근 테스트 실행 시 해당 로그가 터미널에 나타나지 않는 현상이 관찰되었습니다. 이 문제의 원인을 파악하여 디버깅 효율을 높여야 합니다.

**3. 향후 진행 방향**
*   **로그 분석 및 선택자 정밀 검증**: 먼저, 추가된 디버깅 로그(상품명 탐색 직전 페이지 소스)가 정상적으로 출력되도록 문제를 해결합니다. 이후, 확보된 로그를 바탕으로 실제 HTML 구조를 면밀히 분석하여 현재 CSS 선택자들의 유효성을 철저히 재검증하고, 필요시 더욱 정확하고 견고한 선택자로 수정합니다.
*   **동적 컨텐츠 로딩 전략 개선**: 쿠팡 페이지의 JavaScript 실행, AJAX 호출 등 동적 컨텐츠 로딩 메커니즘을 심층적으로 분석합니다. 특정 사용자 인터랙션(스크롤, 클릭 등)이나 추가적인 대기 조건이 필요한지 확인하고, 스크래핑 로직에 반영합니다.
*   **예외 처리 로직 강화**: 네트워크 불안정, 예기치 않은 페이지 구조 변경 등 다양한 예외 상황에 대해 스크래퍼가 더욱 유연하고 안정적으로 대응할 수 있도록 예외 처리 로직을 전반적으로 보강합니다.
*   **이미지, 리뷰, Q&A 스크래핑 로직 구현**: 상품명과 가격 추출 문제가 안정화되는 대로, 이미지 URL, 사용자 리뷰, Q&A 데이터 등 나머지 중요 정보들에 대한 스크래핑 로직 구현을 본격적으로 진행할 예정입니다.

---

## 기술 스택
- **백엔드**: Python, FastAPI, pandas, sqlite3, APScheduler, BeautifulSoup4
- **프론트엔드**: React, TypeScript, Chart.js, axios
- **AI/예측**: Prophet, scikit-learn, (옵션) Ollama LLM
- **데이터 소스**: Google Sheets (CSV API)

---

## 폴더 구조
```
├── app/                # FastAPI 백엔드 및 데이터 처리
│   ├── scrapers/       # 웹 스크래핑 모듈
│   │   └── coupan_scraper.py # 쿠팡 스크래퍼
│   └── main.py         # API, 데이터 동기화, 분석 로직
├── frontend/           # React 프론트엔드
│   └── src/            # 주요 컴포넌트, 스타일 등
├── forecast.py         # 예측 관련 함수
├── requirements.txt    # Python 의존성
├── README.md           # 프로젝트 설명 및 안내
└── ...
```

---

## 설치 및 실행 방법
1. **Python 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```
2. **프론트엔드 설치/실행 (개발 서버: 포트 3000)**
   ```bash
   cd frontend
   npm install
   npm run dev -- --port 3000
   ```
3. **백엔드 실행 (개발 서버: 포트 8001)**
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```
4. **웹 브라우저에서 접속**
   - 프론트엔드 개발 서버: http://localhost:3000
   - 백엔드 API 문서: http://localhost:8001/docs

### 주요 API 엔드포인트
- `/api/overview`: 데이터 개요 및 요약
- `/api/trend`: 기간별 판매 트렌드 분석
- `/api/forecast`: 판매량 예측
- `/api/report/detailed`: 상세 판매 분석 보고서 (JSON 형식, 차트 데이터 포함)
- `/api/report/shipment`: 출고 데이터 기반 AI 분석 보고서 (Markdown 형식)
- `/api/reports/coupan/bono_house`: 쿠팡 '보노 하우스' 상품 리포트 (스크랩된 데이터)
- `/api/eda/monthly_summary`: 월별 판매 요약
- `/api/eda/weekly_summary`: 주간 판매 요약
- `/api/insight`: (현재 목업) AI 기반 데이터 분석 및 질의응답
- `/api/refresh_data`: 원격 CSV 데이터베이스 강제 동기화
- *더 많은 엔드포인트는 `http://localhost:8001/docs`에서 확인 가능합니다.*

### 개발용 기능: 접속 로그 뷰어
- **접속 로그 엔드포인트**: `/access.log` (개발 환경 전용)
- **프론트엔드 탭**: "접속 로그" 탭을 클릭하여 로그 파일 업로드, IP별 이름 매핑 및 시간별 요약 확인
- **IP 매핑 저장**: 입력한 별칭은 `localStorage`에 저장되어 새로고침 시에도 유지됩니다

## 테스트 실행
- **백엔드 유닛 테스트**:
  ```bash
  pytest test_api.py test_db_forecast.py test_forecast.py test_range_forecast.py
  ```
- **프론트엔드 테스트**:
  ```bash
  cd frontend
  npm test
  ```

## 환경변수 설정
- 프로젝트 루트에 `.env` 파일 생성 후 다음 변수 설정:
  ```dotenv
  DB_PATH=your_database.db
  TABLE_NAME="vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)"
  ```

## 라이선스
- 본 프로젝트는 MIT 라이선스를 따릅니다.

---

## 실행 파일 패키징 (프리징)

서버 및 프론트엔드를 단일 실행 파일(.exe)로 묶어 배포하는 방법입니다:

1. **release_build 디렉터리 생성 및 파일 복사**
   ```bash
   mkdir release_build
   robocopy . release_build /E /XD .git release_build venv
   ```
2. **번들링 디렉터리로 이동**
   ```bash
   cd release_build
   ```
3. **PyInstaller로 실행 파일 생성**
   ```bash
   pyinstaller --clean --onefile \
     --collect-all prophet \
     --collect-all holidays \
     --hidden-import holidays.countries \
     --add-data "frontend/dist;frontend/dist" \
     run.py
   ```
4. **생성된 실행 파일 실행**
   ```bash
   dist/run.exe
   ```
5. **브라우저 접속 (배포 서버 기본 포트: 5173)**
   ```bash
   http://localhost:5173
   ```

### 배포 서버 포트 변경
패키징된 EXE는 기본적으로 5173번 포트에서 실행됩니다. 다른 포트를 사용하려면 `release_build/app/main.py` 파일의 `uvicorn.run(app, host="0.0.0.0", port=5173)` 부분을 원하는 포트로 수정한 뒤, 다시 패키징하세요.

---

## 데이터 흐름 및 동기화 구조
- **구글 시트 → CSV 다운로드 → pandas DataFrame → sqlite3 DB upsert**
- 리프레시 시 변경분만 upsert하여 속도 최적화
- DB → API → 프론트엔드로 데이터 전달 및 시각화

---

## 주요 특징 및 장점
- **증분 동기화**: 전체 데이터가 아닌 변경분만 반영하여 빠른 리프레시
- **모듈화**: 백엔드/프론트엔드 분리, 유지보수 용이
- **확장성**: AI 분석, 외부 BI 연동 등 기능 확장 가능
- **실시간성**: 구글 시트 데이터가 변경되면 빠르게 반영
- **사용자 친화적 UI**: 현대적 대시보드 스타일, 다양한 시각화 제공

---

## 활용 예시
- 물류/유통/제조사의 출고/판매 실적 모니터링
- 실시간 트렌드 및 예측 기반 의사결정 지원
- AI 기반 심층 분석 보고서 자동화(옵션)

---

## 향후 개선 사항
아래는 프로젝트의 성능, 예측 정밀도 및 안정성을 향상시키기 위한 주요 개선 항목입니다.

1. 백엔드 최적화
   - 데이터 파이프라인 최적화
     * 전체 CSV/DB 재처리 대신 증분(delta)만 처리하도록 로직 개선
     * APScheduler 작업을 하루치·시간치 업데이트로 분리 및 캐싱 적용
   - Pandas 벡터화 및 메모리 절감
     * 반복문 제거, `groupby`·`merge` 등 벡터 연산 활용
     * 카테고리(dtype='category') 및 다운캐스팅(downcast='integer') 적용
   - 데이터베이스 튜닝
     * 자주 조회/업서트하는 `날짜`, `hour` 컬럼에 복합 인덱스 생성
     * PRAGMA 설정, `VACUUM` 주기적 실행
     * SQLite → PostgreSQL/MySQL 교체 및 커넥션 풀링 고려
   - API 서버 최적화
     * FastAPI 비동기 핸들러(async) 적용, Uvicorn/Gunicorn 멀티 워커 기동
     * GZIP 압축, `Cache-Control` 헤더 적용
     * Redis 또는 in-memory LRU 캐시로 중복 쿼리 방지

2. 예측 모델 정밀도·속도 개선
   - 시계열 특화 라이브러리(Darts, Prophet) 도입 및 하이퍼파라미터 튜닝
   - Gradient Boosting(LightGBM)·GPU 활용, Numba/Cython 경량화
   - 예측 결과 모니터링 및 자동 재학습

3. 프론트엔드 성능·안정성
   - Vite 코드 분할 및 동적 `import()` 적용
   - CDN 또는 ESM 트리 쉐이킹으로 Chart.js 최적화
   - React `memo`, `useMemo`, `useCallback`으로 렌더링 제어
   - 가상 스크롤링(react-virtualized)으로 대용량 테이블 처리

4. 인프라·배포·테스트 자동화
   - Docker + GitHub Actions 기반 CI/CD 구축
   - Production: Uvicorn+Gunicorn, Nginx 리버스 프록시, HTTPS 설정
   - Sentry/Prometheus + Grafana 모니터링
   - pytest + FastAPI TestClient, Jest + Testing Library 테스트 커버리지 확보

### 1. 백엔드 최적화 심층 계획
아래는 1번 백엔드 최적화를 단계별로 수행하기 위한 상세 계획입니다.

1.1. 증분 데이터 처리 설계 및 구현
- 메타데이터 테이블(`last_update`) 생성: 마지막 처리한 날짜/시간 기록
- CSV/DB 처리 로직 수정: 메타데이터 기반으로 신규 레코드만 추출 및 upsert

1.2. 스케줄러 최적화
- APScheduler 작업 분리: daily_job, hourly_job으로 역할 분리
- 병렬 실행 지원: ThreadPoolExecutor/ProcessPoolExecutor 활용

1.3. Pandas 최적화 적용
- 기존 `melt`-기반 로직 리팩토링: `wide_to_long` 등 고성능 함수 도입
- 불필요 복사본 제거, `SettingWithCopyWarning` 해결

1.4. 데이터 타입 및 메모리 최적화
- 주요 컬럼에 `categorical` 타입 적용
- `pd.to_numeric(downcast='integer')` 활용 데이터 크기 축소

1.5. 데이터베이스 인덱스 및 PRAGMA 튜닝
- SQL 스크립트로 인덱스 생성(`CREATE INDEX IF NOT EXISTS idx_date_hour ON realtime(date, hour)`)
- `PRAGMA synchronous = NORMAL` 등 쓰기 성능 튜닝

1.6. API 서버 비동기 및 캐시
- FastAPI 핸들러 `async def`로 변경
- `fastapi-cache` 또는 `cachetools.LRUCache` 적용

1.7. 모니터링 및 검증
- `cProfile`·`pyinstrument`로 처리 시간 프로파일링
- 주요 API 응답 시간 · DB 쿼리 시간 로깅 및 분석

---

## 유지보수 및 확장 가이드
- **구글 시트 구조 변경 시**: app/main.py의 컬럼 매핑/키 조정 필요
- **AI 분석 재활성화**: /api/insight 엔드포인트 주석 해제 및 LLM 서버 준비
- **DB 교체**: sqlite3 → MySQL/PostgreSQL 등으로 확장 가능
- **프론트엔드 커스터마이즈**: src/components 내 컴포넌트 수정

---

## 기여 방법
1. 이슈 등록 또는 Pull Request 제출
2. 주요 변경 시 README, 주석, 예제 코드 보강 권장
3. 문의: comage9@gmail.com

---

# Analytics Dashboard 프로젝트 깃허브 업로드 안내

이 문서는 **다른 컴퓨터에서 이 프로젝트를 깃허브에 업로드(푸시)하는 방법**과 필요한 준비 사항을 안내합니다.

---

## 1. 사전 준비

1. **Git 설치**
   - [Git 공식 다운로드](https://git-scm.com/downloads)에서 운영체제에 맞게 설치

2. **깃허브 계정**
   - [GitHub](https://github.com/)에서 계정 생성 및 로그인

3. **깃허브 Personal Access Token(토큰) 준비**
   - [토큰 생성 가이드](https://docs.github.com/ko/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
   - 토큰은 비밀번호 대신 사용됨 (권장: repo 권한 포함)

---

## 2. 프로젝트 폴더에서 깃허브 업로드(푸시) 방법

### 1) 터미널(명령 프롬프트, PowerShell, Git Bash 등) 실행 후 프로젝트 폴더로 이동
```bash
cd 프로젝트_폴더_경로
```

### 2) git 초기화(이미 되어 있다면 생략)
```bash
git init
```

### 3) 원격 저장소(origin) 설정 (이미 있다면 변경)
```bash
git remote remove origin  # 기존 origin이 있다면 삭제
```
```bash
git remote add origin https://github.com/comage9/analytics-dashboard.git
```

### 4) 변경사항 스테이징 및 커밋
```bash
git add .
git commit -m "프로젝트 업로드"
```

### 5) 강제 푸시(기존 내용 덮어쓰기)
```bash
git push -f origin main
```
- **토큰 입력 요청 시**: 아이디 대신 토큰을 비밀번호 자리에 붙여넣기

---

## 3. 자주 발생하는 문제 및 해결법

- **인증 오류**: 토큰이 만료되었거나 권한이 부족할 수 있음 → 새 토큰 생성 후 사용
- **브랜치 이름(main/master) 불일치**: `git branch -M main` 명령으로 main 브랜치로 통일
- **권한 오류**: 저장소 Collaborator로 추가되어 있는지 확인

---

## 4. 참고
- [깃허브 공식 문서](https://docs.github.com/ko)
- [토큰 인증 안내](https://docs.github.com/ko/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

---

**문의: comage9@gmail.com 또는 깃허브 이슈 등록**

## 빌드 전 정리 (Pre-build Cleanup)

패키징에 앞서 반드시 다음 단계를 수행하세요:

1. 실행 중인 `run.exe` 프로세스를 종료합니다.
2. 이전 `dist/run.exe` 파일을 삭제합니다:

```powershell
Remove-Item -Force .\dist\run.exe
```

이후 아래 명령으로 패키징을 진행합니다:

```powershell
pyinstaller --clean --onefile --collect-all prophet \
  --collect-all holidays \
  --hidden-import holidays.countries \
  --add-data "frontend/dist;frontend/dist" run.py
``` 