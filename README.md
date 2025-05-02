# CSV 데이터 분석기

CSV 파일을 업로드하고 Ollama를 사용하여 로컬 LLM으로 데이터를 분석하는 Streamlit 애플리케이션입니다.

## 기능

- CSV 파일 업로드 및 데이터 미리보기
- 분석할 특정 컬럼 선택
- 다양한 Ollama 모델 선택 (gemma3:27b, llama3:8b, mistral:latest 등)
- 사용자 정의 프롬프트로 데이터 분석
- 분석 결과를 CSV로 다운로드
- 진행 상황 실시간 표시
- 오류 처리 및 재시도 기능

## 설치 방법

1. 저장소 클론:

   ```
   git clone https://github.com/DaniyolKim/csv-llm-analyzer.git
   cd csv-llm-analyzer
   ```

2. 가상 환경 생성 및 활성화:

   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. 의존성 설치:

   ```
   pip install -r requirements.txt
   ```

4. Ollama 설치:

   - [Windows](https://ollama.com/download/windows)
   - [macOS](https://ollama.com/download/mac)
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`

5. 모델 다운로드:
   ```
   ollama run exaone3.5:7.8b
   ```
   또는 다른 모델을 선택할 수 있습니다.

## 사용 방법

1. Streamlit 앱 실행:

   ```
   streamlit run app.py
   ```

2. 웹 브라우저에서 앱 열기 (기본 URL: http://localhost:8501)

3. CSV 파일 업로드

4. 분석할 컬럼 선택

5. 사용할 모델 선택

6. 분석 프롬프트 입력 (예: "이 텍스트가 광고글인지 판단해줘")

7. "분석 요청" 버튼 클릭

8. 분석 결과 확인 및 CSV로 다운로드

## 고급 설정

- **분석 행 수 제한**: 대용량 데이터셋에서 일부 행만 분석
- **배치 크기**: 한 번에 처리할 행 수 설정
- **API 타임아웃**: 각 API 호출의 최대 대기 시간
- **최대 재시도 횟수**: API 호출 실패 시 재시도 횟수

## 시스템 요구사항

- Python 3.8 이상
- Ollama 설치 (로컬 LLM 실행용)
- 충분한 RAM (선택한 모델에 따라 다름, 최소 8GB 권장)

## 라이선스

MIT License
