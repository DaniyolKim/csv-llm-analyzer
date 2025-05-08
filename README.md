# CSV-LLM-Analyzer

CSV 파일의 데이터를 Ollama API를 사용하여 분석하는 Streamlit 애플리케이션입니다.

## 기능

- CSV 파일 업로드 및 분석
- Ollama API를 사용한 텍스트 분석
- 병렬 처리를 통한 성능 향상
- ChromaDB를 활용한 결과 캐싱
- 텍스트 압축을 통한 API 요청 최적화

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/csv-llm-analyzer.git
cd csv-llm-analyzer
```

2. 가상 환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. Ollama 설치
- [Ollama 공식 사이트](https://ollama.com/download)에서 운영체제에 맞는 버전 다운로드 및 설치

## 실행 방법

```bash
streamlit run app.py
```

## 문제 해결

### ChromaDB 캐싱 오류

ChromaDB 캐싱 관련 오류가 발생하는 경우 다음과 같이 해결할 수 있습니다:

1. `app.py` 파일 상단의 `disable_chromadb_cache()` 주석을 해제하여 캐싱을 비활성화합니다.
2. `.chromadb` 디렉토리를 삭제하고 다시 시작합니다.

### PyArrow 변환 오류

데이터프레임 표시 중 PyArrow 변환 오류가 발생하는 경우:

1. 데이터프레임의 모든 컬럼이 문자열로 변환되도록 `fix_dataframe.py` 모듈을 사용합니다.
2. 특히 'Unnamed:' 컬럼이 있는 경우 이 문제가 발생할 수 있습니다.

## 고급 설정

- **배치 크기**: 한 번에 처리할 행 수를 설정합니다.
- **API 타임아웃**: API 호출 시 최대 대기 시간을 설정합니다.
- **재시도 횟수**: API 호출 실패 시 재시도 횟수를 설정합니다.
- **병렬 처리 작업자 수**: 동시에 실행할 작업자 수를 설정합니다.
- **캐싱 사용**: 이전 결과를 캐싱하여 중복 API 호출을 방지합니다.
- **최대 텍스트 길이**: API에 전송할 텍스트의 최대 길이를 설정합니다.