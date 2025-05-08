# CSV 분석기 & RAG 시스템

CSV 파일을 업로드하여 데이터를 분석하고 시각화하는 Streamlit 애플리케이션입니다. 또한 ChromaDB와 Ollama를 활용한 RAG(Retrieval-Augmented Generation) 시스템을 구성할 수 있습니다.

## 기능

- CSV 파일 업로드 및 미리보기
- 데이터 요약 통계 제공
- 데이터 시각화 (분포, 히스토그램)
- 상관 관계 분석 및 히트맵
- 데이터 탐색 및 분석
- ChromaDB에 데이터 저장
- Ollama를 통한 RAG 시스템 구성

## 설치 방법

1. 저장소를 클론합니다:
```
git clone <repository-url>
cd csv_analyzer
```

2. 가상 환경을 생성하고 활성화합니다:

Windows:
```
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```
python -m venv venv
source venv/bin/activate
```

3. 필요한 패키지를 설치합니다:
```
pip install -r requirements.txt
```

4. Ollama 설치 (선택 사항, RAG 기능 사용 시 필요):
   - [Ollama 웹사이트](https://ollama.ai/)에서 Ollama를 다운로드하고 설치합니다.
   - 터미널에서 다음 명령어로 모델을 다운로드합니다:
     ```
     ollama pull llama2
     ```
   - Ollama 서버를 실행합니다:
     ```
     ollama serve
     ```

## 실행 방법

### 스크립트 사용 (권장)

Windows:
```
setup_venv.bat    # 처음 한 번만 실행
run_app.bat       # 애플리케이션 실행
```

macOS/Linux:
```
chmod +x setup_venv.sh run_app.sh  # 스크립트 실행 권한 부여 (처음 한 번만)
./setup_venv.sh   # 처음 한 번만 실행
./run_app.sh      # 애플리케이션 실행
```

### 수동 실행

가상 환경을 활성화한 후 애플리케이션을 실행합니다:

Windows:
```
venv\Scripts\activate
streamlit run app.py
```

macOS/Linux:
```
source venv/bin/activate
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용할 수 있습니다.

## 사용 방법

### 데이터 분석

1. 웹 인터페이스에서 CSV 파일을 업로드합니다.
2. 업로드된 데이터의 미리보기와 기본 정보를 확인합니다.
3. 탭을 선택하여 다양한 분석 결과를 확인합니다:
   - 요약 통계: 데이터의 기술 통계를 제공합니다.
   - 데이터 시각화: 선택한 열의 분포를 시각화합니다.
   - 상관 관계: 숫자형 데이터 간의 상관 관계를 히트맵으로 표시합니다.
   - 데이터 탐색: 열별 정보와 값 분포를 확인합니다.

### RAG 시스템 구성

1. "RAG 시스템" 탭으로 이동합니다.
2. ChromaDB에 저장할 열을 선택합니다.
3. "ChromaDB에 데이터 저장" 또는 "LangChain Chroma에 데이터 저장" 버튼을 클릭합니다.
4. 저장이 완료되면 질의를 입력하고 "ChromaDB 쿼리" 또는 "LangChain Chroma 쿼리" 버튼을 클릭하여 검색 결과를 확인합니다.
5. Ollama와 연동하려면 제공된 코드를 참조하거나 `ollama_integration.py` 스크립트를 사용합니다.

### Ollama 연동 스크립트 사용

```
python ollama_integration.py --db_path ./langchain_chroma_db --model llama2 --query "여기에 질문을 입력하세요"
```

대화형 모드로 실행하려면 `--query` 인자를 생략합니다:

```
python ollama_integration.py
```

## 요구사항

- Python 3.8 이상
- Streamlit 1.32.0
- Pandas 2.1.4
- NumPy 1.26.3
- Matplotlib 3.8.2
- Seaborn 0.13.1
- Plotly 5.18.0
- ChromaDB 0.4.22
- LangChain 0.1.4
- Sentence Transformers 2.2.2
- Ollama (선택 사항, RAG 기능 사용 시 필요)