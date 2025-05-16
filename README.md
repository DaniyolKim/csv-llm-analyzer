# CSV LLM Analyzer

텍스트 CSV 파일을 분석하고 RAG(Retrieval-Augmented Generation) 시스템을 구성하는 도구입니다.

(kss lib 사용을 위해 python 3.10에서 동작함)

## 주요 기능

- CSV 파일 미리보기 및 분석
- 텍스트 데이터 전처리 (결측치 제거, 특수문자 제거)
- ChromaDB를 사용한 벡터 데이터베이스 구축
- Ollama를 통한 로컬 LLM 연동
- RAG 시스템을 통한 질의응답
- 유사도 기반 문서 필터링

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/csv_llm_analyzer.git
cd csv_llm_analyzer
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. Ollama 설치 (선택사항)
- [Ollama 웹사이트](https://ollama.ai/download)에서 운영체제에 맞는 버전을 다운로드하여 설치합니다.
- 모델 다운로드: `ollama pull llama2`

## 사용 방법

1. Streamlit 앱 실행
```bash
streamlit run app.py
```

2. 웹 브라우저에서 앱 접속
- 기본 URL: http://localhost:8501

## 파일 구조

- `app.py`: 메인 Streamlit 애플리케이션
- `text_utils.py`: 텍스트 처리 관련 유틸리티
- `data_utils.py`: 데이터프레임 처리 관련 유틸리티
- `embedding_utils.py`: 임베딩 모델 관련 유틸리티
- `chroma_utils.py`: ChromaDB 관련 유틸리티
- `ollama_utils.py`: Ollama 관련 유틸리티
- `rag_utils.py`: RAG 시스템 관련 유틸리티
- `utils.py`: 하위 호환성을 위한 임포트 모음
- `requirements.txt`: 필요한 패키지 목록

## 임베딩 모델

다음과 같은 임베딩 모델을 지원합니다:

| 목적 | 추천 모델 |
| --- | --- |
| 전반적 의미 임베딩 (가장 추천) | `snunlp/KR-SBERT-V40K-klueNLI-augSTS` |
| 속도 중요 & 경량화 필요 | `BM-K/KoMiniLM-Sentence-Transformers` |
| 감정, 문장 유사도 | `jhgan/ko-sbert-sts` |
| 금융/도메인 특화 | `jinmang2/kpf-sbert` |

## 라이선스

MIT License
