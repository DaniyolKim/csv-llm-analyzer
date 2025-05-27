# CSV LLM 분석기

텍스트 기반 CSV 파일의 내용을 심층 분석하고, Ollama 로컬 LLM과 연동하여 RAG(Retrieval-Augmented Generation) 질의응답 시스템을 손쉽게 구축 및 활용할 수 있도록 돕는 Streamlit 웹 애플리케이션입니다.

## 사전 준비 사항

- **Python 3.10**: `kss` 라이브러리와의 호환성을 위해 필요합니다.
- **Java Development Kit (JDK) 1.8 이상**: `konlpy` 라이브러리 실행 환경을 위해 필요합니다. `java -version`으로 설치 여부 및 버전을 확인하세요. (Konlpy 설치 가이드 참고)
- **Ollama**: 로컬 LLM 실행 환경입니다.

## 주요 기능

- CSV 파일 미리보기, 기본 분석 및 다양한 인코딩(utf-8, cp949, euc-kr 등) 자동 감지 지원
- 텍스트 데이터 전처리 (결측치 제거, 특수문자 제거)
- ChromaDB를 사용한 벡터 데이터베이스 구축 및 그래픽 시각화
- Ollama를 통한 로컬 LLM 연동
- RAG 시스템을 통한 질의응답 및 참조 문서 확인
- 유사도 기반 문서 필터링 및 하이브리드 검색(키워드 + 임베딩)

## 설치 방법
1. **사전 준비 사항 확인**
    - Python 3.10 설치 여부를 확인합니다.
    - Java (JDK 1.8 이상) 설치 및 `JAVA_HOME` 환경 변수 설정 여부를 확인합니다.

2. **Ollama 설치 및 모델 다운로드**
    - Ollama 웹사이트에서 운영체제에 맞는 버전을 다운로드하여 설치합니다. (https://ollama.com/download)
    - Ollama 서버 실행: 터미널에서 `ollama serve` 명령을 실행합니다 (백그라운드 실행 권장).
    - 원하는 LLM 모델 다운로드: 예) `ollama pull exaone3.5:7.8b`, `ollama pull llama2`, `ollama pull gemma:2b` (사용 가능한 모델 목록: Ollama models(https://ollama.com/search))

3. **저장소 클론**

```bash
git clone https://github.com/DaniyolKim/csv-rag.git
cd csv-rag
```

3. python 가상 환경 생성

```bash
python -m venv venv
# windows
venv\Scripts\activate
# linux, mac
source venv/bin/activate
```

4. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

1. Ollama 서버 실행 확인

터미널에서 ollama serve 명령이 실행 중인지 확인합니다. (백그라운드 실행 권장)

2. Streamlit 앱 실행

```bash
streamlit run Home.py
```

3. 웹 브라우저에서 앱 접속

- 기본 URL: http://localhost:8501

## 프로젝트 구조

```
csv-llm-analyzer/
│
├── Home.py                   # 메인 Streamlit 애플리케이션 (RAG 챗봇 인터페이스)
├── pages/                    # Streamlit 멀티페이지 앱을 위한 추가 페이지들
│   ├── 01_csv_upload.py      # CSV 파일 업로드 및 벡터 DB 구축 페이지
│   └── 02_db_search.py       # 벡터 DB 검색 및 시각화 페이지
│
├── chroma_utils.py           # ChromaDB 관련 유틸리티
├── data_utils.py             # 데이터프레임 처리 관련 유틸리티
├── embedding_utils.py        # 임베딩 모델 관련 유틸리티
├── ollama_integration.py     # Ollama 통합 모듈
├── ollama_utils.py           # Ollama 관련 유틸리티
├── rag_utils.py              # RAG 시스템 관련 유틸리티
├── text_utils.py             # 텍스트 처리 관련 유틸리티
├── utils.py                  # 하위 호환성을 위한 임포트 모음
└── requirements.txt          # 필요한 패키지 목록
```

## CSV 파일 처리 과정

1. CSV 파일 업로드 또는 경로 지정
2. 인코딩 자동 감지 및 데이터 로드
3. 데이터 분석 및 전처리
4. 임베딩 모델을 통한 텍스트 벡터화
5. ChromaDB에 벡터 데이터 저장
6. RAG 시스템 구성 또는 시각화

## 시각화 기능

- t-SNE 알고리즘을 활용한 임베딩 2차원 시각화
- K-means 클러스터링을 통한 문서 그룹화
- 클러스터별 문서 통계 및 분석

## RAG 시스템

- 사용자 질문에 대해 관련 문서 검색
- Ollama 로컬 LLM을 활용한 응답 생성
- 참조 문서 제공으로 응답 신뢰성 향상
- 시스템 프롬프트 설정 기능

## 사용 TIP

- 한국어 데이터는 한국어 특화 모델(`snunlp/KR-SBERT-V40K-klueNLI-augSTS`)을 사용하는 것이 좋습니다.
- Ollama 서버는 별도로 실행해야 합니다: `ollama serve`
- 시각화는 데이터 크기에 따라 처리 시간이 길어질 수 있습니다.
- ChromaDB 컬렉션은 재사용이 가능하며, 여러 CSV 파일의 데이터를 하나의 컬렉션에 통합할 수 있습니다.

## 주의사항

- 대용량 CSV 파일(수십만 행 이상)의 경우 배치 처리 크기를 조정하여 메모리 사용량을 관리하세요.

## 라이선스

MIT License
