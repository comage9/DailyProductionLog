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
*   **대기 시간 증가**: 동적 컨텐츠 로딩을 위해 `WebDriverWait`의 대기 시간을 30초로 늘려, 페이지 요소가 나타날 시간을 충분히 확보했습니다.
*   **예외 처리 강화**: `try-except` 블록을 사용하여 특정 요소(예: 리뷰, Q&A)가 없는 경우에도 스크래핑이 중단되지 않고 다음 단계로 진행하도록 개선했습니다.
*   **리뷰/Q&A 수집 로직 개선**: "더보기" 버튼을 클릭하여 모든 리뷰와 Q&A를 가져오는 로직을 구현 중입니다. 현재 일부 상품에서 모든 내용을 가져오지 못하는 경우가 있어 디버깅 중입니다.
*   **데이터 저장**: 수집된 데이터는 CSV 파일로 저장되며, 향후 데이터베이스 연동을 고려하고 있습니다.

**2. 당면 과제 및 해결 방안**
*   **선택자 불안정성**: 쿠팡 웹사이트 구조 변경 시 CSS 선택자가 무효화될 수 있습니다.
    *   **해결 방안**: 좀 더 견고한 선택자(예: ID, 복합 선택자) 사용, XPath 사용 고려, 정기적인 선택자 유효성 검사 및 업데이트 자동화 스크립트 개발.
*   **동적 컨텐츠 로딩 지연**: 일부 상품 페이지에서 특정 요소(특히 리뷰, Q&A) 로딩이 매우 느리거나 실패하는 경우가 있습니다.
    *   **해결 방안**: `WebDriverWait`와 `expected_conditions` 조합 최적화, 필요시 `time.sleep()` 추가, JavaScript 실행을 통한 직접 데이터 추출 고려.
*   **IP 차단 가능성**: 빈번한 요청 시 IP가 차단될 수 있습니다.
    *   **해결 방안**: 요청 간 랜덤 지연 시간 추가, 프록시 서버 사용, User-Agent 변경. (현재는 구현되지 않음)
*   **"더보기" 버튼 처리**: 모든 리뷰/Q&A를 가져오기 위한 "더보기" 버튼 클릭 로직이 일부 상품에서 완벽하게 동작하지 않습니다.
    *   **해결 방안**: 버튼 상태(활성화/비활성화) 정확히 감지, 클릭 후 컨텐츠 로딩 대기 시간 충분히 부여, JavaScript 직접 실행으로 버튼 클릭 시도.
*   **로그인 필요 컨텐츠**: 일부 정보는 로그인해야 접근 가능할 수 있습니다. (현재는 로그인 없이 접근 가능한 정보만 수집)
    *   **해결 방안**: 필요시 로그인 기능 구현 (쿠키 사용 등), 또는 수집 범위 제한.
*   **오류 로깅 및 모니터링**: 스크래핑 실패 시 원인 파악을 위한 상세 로깅 및 알림 시스템이 부족합니다.
    *   **해결 방안**: `logging` 모듈을 사용하여 단계별 로그 기록, 예외 발생 시 상세 정보(URL, 시간, 에러 메시지, 스크린샷 등) 기록, 주요 오류 발생 시 이메일/슬랙 알림 기능 추가.

**3. 향후 개선 계획**
*   스크래핑 주기 설정 및 자동 실행 (APScheduler 등 활용)
*   수집 데이터 DB 저장 및 API 연동
*   스크래핑 현황 대시보드 제공

---

## 기술 스택

**백엔드**:
- Python 3.10+
- FastAPI: 고성능 API 프레임워크
- Pandas: 데이터 분석 및 처리
- SQLite3: 경량 데이터베이스 (vf.db, your.db)
- APScheduler: 백그라운드 작업 스케줄링
- Prophet (Meta): 시계열 데이터 예측
- Uvicorn: ASGI 서버

**프론트엔드**:
- React 18+
- TypeScript
- Vite: 차세대 프론트엔드 빌드 도구
- Chart.js: 인터랙티브 차트 라이브러리
- Axios: HTTP 클라이언트

**데이터 소스**:
- Google Sheets API (CSV 다운로드 방식)
- 쿠팡 웹사이트 (스크래핑)

---

## 프로젝트 구조

```
analytics-dashboard-main/
├── app/                      # FastAPI 백엔드 애플리케이션
│   ├── __init__.py
│   ├── analysis.py           # 데이터 분석 로직
│   ├── background.py         # 백그라운드 작업 (APScheduler)
│   ├── cache.py              # 캐시 관련 유틸리티
│   ├── decomposition.py      # 시계열 분해 로직
│   ├── eda.py                # 탐색적 데이터 분석 API
│   ├── main.py               # FastAPI 메인 라우터, 핵심 API
│   ├── scrapers/             # 웹 스크래퍼
│   │   └── coupan_scraper.py # 쿠팡 상품 정보 스크래퍼
│   └── ... (기타 유틸리티)
├── frontend/                 # React 프론트엔드 애플리케이션
│   ├── public/
│   ├── src/
│   │   ├── components/       # React 컴포넌트
│   │   ├── App.tsx           # 메인 애플리케이션 컴포넌트
│   │   └── main.tsx          # React 애플리케이션 진입점
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── data/                     # (주의) 실제 데이터 파일은 .gitignore 처리됨
│   ├── sample_data.csv
│   └── ...
├── reports/                  # (주의) 생성된 리포트 파일은 .gitignore 처리됨
│   └── ...
├── tests/                    # 테스트 코드 (pytest)
│   └── ...
├── .env.example              # 환경변수 설정 예시
├── .gitignore                # Git 무시 파일 목록
├── Dockerfile                # Docker 빌드 설정 (선택 사항)
├── requirements.txt          # Python 의존성 목록
├── README.md                 # 프로젝트 설명 파일
└── run.py                    # FastAPI 애플리케이션 실행 스크립트
```

---

## 설치 및 실행

### 1. 사전 준비
- Python 3.10 이상 설치
- Node.js 및 npm (또는 yarn) 설치 (프론트엔드 빌드 시 필요)
- Google Cloud Platform 프로젝트 생성 및 Google Sheets API 활성화
- Google 서비스 계정 생성 및 JSON 키 파일 다운로드 (`google_credentials.json` - 프로젝트 루트에 배치)

### 2. 백엔드 설정 및 실행
1.  **가상환경 생성 및 활성화**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\activate    # Windows
    ```
2.  **Python 의존성 설치**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **환경변수 설정**: `.env.example` 파일을 복사하여 `.env` 파일 생성 후, 내부 값들을 실제 환경에 맞게 수정합니다. (특히 `GOOGLE_SHEET_ID`, `DB_PATH` 등)
4.  **데이터베이스 초기화** (최초 실행 시): `vf.db` 파일이 자동으로 생성됩니다. 필요시 `app.db_utils`의 스키마 생성 함수를 직접 실행할 수 있습니다.
5.  **FastAPI 서버 실행**:
    ```bash
    python run.py
    ```
    또는 Uvicorn 직접 실행:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    API 문서는 `http://localhost:8000/docs` 에서 확인 가능합니다.

### 3. 프론트엔드 설정 및 실행
1.  **프론트엔드 디렉토리로 이동**:
    ```bash
    cd frontend
    ```
2.  **Node.js 의존성 설치**:
    ```bash
    npm install
    # 또는 yarn install
    ```
3.  **프론트엔드 개발 서버 실행**:
    ```bash
    npm run dev
    # 또는 yarn dev
    ```
    애플리케이션은 `http://localhost:5173` (또는 다른 포트)에서 실행됩니다.

### 4. 프론트엔드 빌드 (배포용)
1.  프론트엔드 디렉토리(`frontend/`)에서 다음 명령 실행:
    ```bash
    npm run build
    # 또는 yarn build
    ```
2.  빌드 결과물은 `frontend/dist` 디렉토리에 생성됩니다. FastAPI는 이 디렉토리의 정적 파일을 자동으로 서빙하도록 설정되어 있습니다 (`app/main.py`의 `StaticFiles` 마운트 부분 참조).

---

## 테스트

프로젝트 루트 디렉토리에서 다음 명령으로 `pytest` 실행:
```bash
pytest
```
- 특정 테스트 파일 실행: `pytest tests/test_api.py`
- 특정 테스트 함수 실행: `pytest tests/test_api.py::test_get_some_data`

---

## 데이터베이스 관리

- 현재 SQLite 데이터베이스 파일(`vf.db`, `your.db`)은 `.gitignore`에 의해 버전 관리에서 제외됩니다.
- 데이터베이스 스키마 변경이나 마이그레이션은 수동으로 관리해야 합니다. (Alembic 등 마이그레이션 도구 도입 고려 가능)
- 데이터 백업은 별도로 관리해야 합니다.

---

## 깃허브(GitHub) 연동 가이드

### 1. 개인 액세스 토큰(PAT) 생성
- 깃허브 > Settings > Developer settings > Personal access tokens > Tokens (classic)
- "Generate new token (classic)" 선택
- Note: "DailyProductionLog Access" 등
- Expiration: (적절히 선택, 예: 90 days)
- Select scopes:
    - `repo` (Full control of private repositories) 필수
- "Generate token" 클릭 후 생성된 토큰 복사 (다시 볼 수 없으므로 안전한 곳에 보관)

### 중요: `analytics-dashboard` 리포지토리 (구 버전 관리용)
이 프로젝트의 이전 버전 또는 다른 목적으로 사용되던 `https://github.com/comage9/analytics-dashboard.git` 리포지토리가 존재할 수 있습니다.
**현재 주 개발 및 코드 관리는 `DailyProductionLog` 리포지토리에서 이루어집니다.** (아래 섹션 참조)

만약 `analytics-dashboard` 리포지토리에 특정 내용을 업로드해야 한다면, 해당 프로젝트 폴더로 이동하여 아래와 유사한 절차를 따르되, 원격 저장소 URL을 정확히 지정해야 합니다.

```bash
# 예시: analytics-dashboard 리포지토리로 푸시하는 경우
# cd /path/to/your/analytics-dashboard/project
# git remote set-url origin https://github.com/comage9/analytics-dashboard.git
# git add .
# git commit -m "커밋 메시지"
# git push origin main # 또는 해당 브랜치명
```
토큰 인증 방식은 동일하게 적용됩니다.

---

## 1. 깃허브(GitHub)에 프로젝트 업로드(푸시)하는 방법 (일반적인 경우)

### 1) 로컬 프로젝트 폴더를 Git 저장소로 만들기 (이미 되어 있다면 생략)
```bash
git init
```

### 2) 원격 저장소(origin) 연결 (아직 안되어 있다면)
- **HTTPS 방식 (권장: 토큰 사용)**
  ```bash
  git remote add origin https://github.com/사용자이름/저장소이름.git
  ```
  예: `git remote add origin https://github.com/comage9/DailyProductionLog.git`

- **SSH 방식 (SSH 키 설정 필요)**
  ```bash
  git remote add origin git@github.com:사용자이름/저장소이름.git
  ```
  예: `git remote add origin git@github.com:comage9/DailyProductionLog.git`

### 3) 변경사항 스테이징 및 커밋
```bash
git add .                 # 모든 변경사항 스테이징
# 또는 git add 특정파일   # 특정 파일만 스테이징
git commit -m "여기에 커밋 메시지 작성"
```

### 4) 원격 저장소로 푸시
```bash
git push -u origin main   # 'main'은 현재 브랜치 이름 (master일 수도 있음)
```
- `-u` 옵션은 한 번만 사용하면 다음부터 `git push`만으로 해당 브랜치에 푸시 가능
- **HTTPS 방식 사용 시**: 사용자 이름과 비밀번호(개인 액세스 토큰) 입력 요청
   - 사용자 이름: GitHub 사용자 이름
   - 비밀번호: 위에서 생성한 개인 액세스 토큰(PAT) 붙여넣기 (입력 시 화면에 표시되지 않음)

### 5) 기존 원격 저장소 URL 변경 (필요시)
```bash
git remote set-url origin 새로운_저장소_URL
```
예: `git remote set-url origin https://github.com/comage9/NewRepository.git`

### 6) 강제 푸시 (주의: 원격 저장소의 내용을 덮어쓰므로 신중히 사용)
로컬 저장소의 내용으로 원격 저장소를 강제로 덮어쓰려면:
```bash
git push -f origin main
```
- **경고**: 팀원과 협업 시에는 절대 사용하지 마세요. 개인 프로젝트의 히스토리 정리 등 특별한 경우에만 사용합니다.
- HTTPS 방식 사용 시 토큰 인증 필요 (위와 동일)

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

---

## 5. 현재 GitHub 리포지토리 (`DailyProductionLog`) 관리

이 프로젝트의 현재 주 개발 리포지토리는 `https://github.com/comage9/DailyProductionLog.git` 입니다.
이 리포지토리에는 데이터 파일, 데이터베이스 파일, 빌드 아티팩트 (`dist/`, `build/`), `node_modules`, `__pycache__`, 백업 파일 (`*.bak`), 캐시 파일 (`cache/`) 등 불필요한 대용량 파일 및 디렉토리를 제외한 순수 코드베이스만 관리됩니다.

**주요 `.gitignore` 설정:**
```
venv/
*.db
*.sqlite
*.sqlite3
reports/
release_build/
access.log
**/__pycache__/
*.pyc
*.log
*.DS_Store
**/node_modules/
build/
dist/
frontend/dist/
*.zip
*.bak
cache/
backup/
```

**업로드 과정 요약 (참고용):**
1.  `DailyProductionLog` 리포지토리를 임시 폴더에 복제.
2.  기존 리포지토리 내용 삭제 (새 커밋으로 덮어쓰기 준비).
3.  현재 `analytics-dashboard-main` 프로젝트의 전체 내용을 임시 폴더에 복사.
4.  `.gitignore` 파일을 신규 생성/수정하여 불필요한 파일 및 디렉토리 제외 패턴 명시.
5.  여러 차례의 파일 삭제, `.gitignore` 수정, `git commit --amend`를 통해 커밋 정제.
6.  최종 정리된 코드베이스를 `DailyProductionLog`의 `main` 브랜치로 강제 푸시 (`git push origin main --force`).

**향후 이 프로젝트의 변경사항은 `e:\python\analytics-dashboard-main` 디렉토리에서 작업 후, 다음 명령어를 사용하여 `DailyProductionLog` 리포지토리로 푸시합니다:**

1.  프로젝트 루트 디렉토리 (`e:\python\analytics-dashboard-main`)로 이동합니다.
2.  원격 저장소 설정 확인 (현재 `https://github.com/comage9/DailyProductionLog.git`로 설정되어 있어야 함):
    ```bash
    git remote -v 
    ```
    만약 다르게 설정되어 있다면, 다음 명령으로 URL을 업데이트합니다:
    ```bash
    git remote set-url origin https://github.com/comage9/DailyProductionLog.git 
    ```
3.  변경사항 스테이징 및 커밋:
    ```bash
    git add .
    git commit -m "커밋 메시지"
    ```
4.  `main` 브랜치로 푸시:
    ```bash
    git push origin main
    ```
    (일반적인 작업 흐름에서는 `--force`를 사용하지 않습니다. 히스토리 충돌 시에는 원인을 파악하고 rebase 또는 merge를 고려해야 합니다.)

---

---

## 6. Windsurf (Cascade) 설정 백업 및 이전 가이드

Windsurf (Cascade)의 전역 규칙, 작업 공간 규칙, 메모리 및 일부 MCP 서버 설정을 다른 컴퓨터로 이전하거나 백업하는 방법입니다.

1.  **Codeium/Windsurf 설치 (새 컴퓨터)**:
    *   새 컴퓨터에 현재 사용하고 계신 버전과 동일한 Codeium 및 Windsurf를 설치합니다.

2.  **설정 및 메모리 폴더 복사 (핵심 단계)**:
    *   **기존 컴퓨터**에서 다음 디렉토리의 **전체 내용**을 복사합니다:
        *   `C:\Users\<사용자 이름>\.codeium\windsurf\memories\`
        *   (예: `C:\Users\kis\.codeium\windsurf\memories\`)
    *   이 `memories` 폴더에는 전역 규칙 (`global_rules.md`), 작업 공간별 규칙 파일, 그리고 생성된 모든 메모리 파일들이 포함되어 있습니다.
    *   **새 컴퓨터**의 동일한 위치 (`C:\Users\<새 컴퓨터 사용자 이름>\.codeium\windsurf\memories\`)에 복사한 `memories` 폴더 전체를 붙여넣습니다.
    *   만약 `.codeium` 또는 하위 폴더가 없다면, Windsurf를 한 번 실행하여 폴더가 생성되도록 하거나 수동으로 생성할 수 있습니다.

3.  **프로젝트 작업 공간 이전**:
    *   현재 작업 중인 프로젝트 폴더(예: `e:\python\analytics-dashboard-main`) 전체를 새 컴퓨터의 원하는 위치로 복사합니다.
    *   새 컴퓨터에서 Windsurf를 통해 해당 작업 공간을 열면, 2단계에서 이전한 메모리 덕분에 작업 공간 규칙이 자동으로 적용됩니다.

4.  **`desktop_commander` (mcp2) 서버 설정 적용 (새 컴퓨터에서)**:
    *   `desktop_commander` (mcp2) 서버의 설정(예: `blockedCommands`, `defaultShell`, `allowedDirectories`)은 필요에 따라 새 컴퓨터에서 재현합니다.
    *   Cascade에게 요청하여 `mcp2_get_config`로 현재 설정을 확인하고, `mcp2_set_config_value`로 새 컴퓨터에 동일하게 설정할 수 있습니다.
    *   **`allowedDirectories` 설정 주의**: 이 설정을 빈 배열 `[]`로 하면 Windsurf가 컴퓨터의 모든 파일 시스템에 접근할 수 있게 됩니다. 보안상 위험할 수 있으므로, 필요한 최소한의 디렉토리만 허용하는 것이 안전합니다.

**참고: 개인 설정 백업 (예시)**
사용자께서는 `C:\Users\kis\.codeium\windsurf\memories\` 폴더의 내용을 개인 GitHub 리포지토리(예: `git@github.com:comage9/coin.git`의 `windsurf_settings` 폴더)에 주기적으로 백업하여 관리하고 계십니다. 이 방법은 개인 설정을 안전하게 보관하고 여러 환경에서 일관성을 유지하는 좋은 전략입니다.

---
