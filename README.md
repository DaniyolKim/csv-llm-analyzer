# CSV 분석기 & RAG 시스템

CSV 파일을 업로드하여 데이터를 분석하고, ChromaDB와 Ollama를 활용한 RAG(Retrieval-Augmented Generation) 시스템을 구성할 수 있는 Streamlit 애플리케이션입니다.

## 기능

- CSV 파일 업로드 및 미리보기
- 다양한 인코딩 지원 (utf-8, cp949, euc-kr, latin1)
- 데이터 요약 통계 제공
- 텍스트 데이터 정제 및 분석
- ChromaDB에 데이터 저장 및 관리
- 기존 ChromaDB 컬렉션 로드 기능
- Ollama를 통한 RAG 시스템 구성
- 대화형 질의응답 시스템

## 설치 방법

1. 저장소를 클론합니다:
```
git clone <repository-url>
cd csv-llm-analyzer
```

2. 가상 환경을 생성하고 활성화합니다:

Windows (CMD):
```
python -m venv venv
call venv\Scripts\activate.bat
```

Windows (PowerShell):
```
python -m venv venv
PowerShell -ExecutionPolicy Bypass -File venv\Scripts\Activate.ps1
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

## Ollama 설치 (RAG 기능 사용 시 필요)

### Windows
1. [Ollama 웹사이트](https://ollama.ai/download/windows)에서 Windows 설치 파일을 다운로드합니다.
2. 다운로드한 설치 파일을 실행하고 설치 지침을 따릅니다.
3. 설치가 완료되면 시스템 트레이에 Ollama 아이콘이 나타납니다.
4. 모델을 다운로드하려면 명령 프롬프트를 열고 다음 명령어를 실행합니다:
   ```
   ollama pull llama2
   ```

### macOS
1. [Ollama 웹사이트](https://ollama.ai/download/mac)에서 macOS 설치 파일을 다운로드합니다.
2. 다운로드한 .dmg 파일을 열고 Ollama를 Applications 폴더로 드래그합니다.
3. Applications 폴더에서 Ollama를 실행합니다.
4. 모델을 다운로드하려면 터미널을 열고 다음 명령어를 실행합니다:
   ```
   ollama pull llama2
   ```

### Linux
1. 터미널을 열고 다음 명령어를 실행합니다:
   ```
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
2. 설치가 완료되면 Ollama 서버를 시작합니다:
   ```
   ollama serve
   ```
3. 새 터미널 창을 열고 모델을 다운로드합니다:
   ```
   ollama pull llama2
   ```

## 실행 방법

가상 환경을 활성화한 후 애플리케이션을 실행합니다:

Windows (CMD):
```
call venv\Scripts\activate.bat
streamlit run app.py
```

macOS/Linux:
```
source venv/bin/activate
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용할 수 있습니다.

## 사용 방법

### 1. CSV 파일 업로드 및 분석

1. 웹 인터페이스에서 CSV 파일을 업로드합니다.
2. 적절한 인코딩을 선택합니다 (utf-8, cp949, euc-kr, latin1).
3. 업로드된 데이터의 미리보기와 기본 정보를 확인합니다.
4. 텍스트 열을 선택하여 텍스트 데이터 미리보기 및 정제된 텍스트를 확인합니다.

### 2. RAG 시스템 구성

1. ChromaDB에 저장할 열을 선택합니다.
2. 처리할 최대 행 수와 배치 처리 크기를 설정합니다.
3. 컬렉션 이름과 저장 경로를 지정합니다.
4. "ChromaDB에 데이터 저장" 버튼을 클릭합니다.
5. 저장이 완료되면 Ollama 모델을 선택하고 질의를 입력하여 RAG 시스템을 사용합니다.

### 3. 기존 ChromaDB 로드

1. 사이드바에서 ChromaDB 경로를 입력합니다.
2. 사용 가능한 컬렉션 목록에서 원하는 컬렉션을 선택합니다.
3. "컬렉션 로드" 버튼을 클릭합니다.
4. 로드가 완료되면 RAG 시스템을 사용할 수 있습니다.

## 주요 파일 구조

- `app.py`: Streamlit 애플리케이션의 메인 파일
- `utils.py`: 유틸리티 함수 모음 (데이터 전처리, ChromaDB 관리, Ollama 연동 등)
- `ollama_integration.py`: 명령줄에서 RAG 시스템을 사용하기 위한 스크립트
- `requirements.txt`: 필요한 패키지 목록

## 요구사항

- Python 3.8 이상
- Streamlit
- Pandas
- NumPy
- ChromaDB
- langchain-text-splitters
- requests
- ollama (Python 라이브러리)
- Ollama (로컬 LLM 서버)