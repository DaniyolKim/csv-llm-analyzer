import pandas as pd
import numpy as np
import os
import re
import chromadb
import subprocess
import json
import platform
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def clean_text(text):
    """
    텍스트에서 산술 기호를 제외한 특수문자를 제거합니다.
    
    Args:
        text (str): 정제할 텍스트
        
    Returns:
        str: 정제된 텍스트
    """
    if not isinstance(text, str):
        return str(text)
    
    # 산술 기호(+, -, *, /, %, =)를 제외한 특수문자 제거
    # 영문, 숫자, 한글, 공백, 산술 기호만 유지
    cleaned_text = re.sub(r'[^\w\s+\-*/=%.,()[\]]+', ' ', text)
    
    # 연속된 공백을 하나의 공백으로 치환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def preprocess_dataframe(df, selected_columns, max_rows=None):
    """
    데이터프레임을 전처리합니다.
    
    Args:
        df (pandas.DataFrame): 원본 데이터프레임
        selected_columns (list): 선택한 열 목록
        max_rows (int, optional): 처리할 최대 행 수
        
    Returns:
        pandas.DataFrame: 전처리된 데이터프레임
    """
    # 선택한 열만 추출
    selected_df = df[selected_columns].copy()
    
    # 결측치가 있는 행 제거
    selected_df = selected_df.dropna()
    
    # 최대 행 수 제한
    if max_rows is not None and max_rows > 0 and max_rows < len(selected_df):
        selected_df = selected_df.head(max_rows)
    
    # 텍스트 정제
    for col in selected_df.columns:
        if selected_df[col].dtype == 'object':
            selected_df[col] = selected_df[col].apply(clean_text)
    
    return selected_df

def create_chroma_db(collection_name="csv_test", persist_directory="./chroma_db", overwrite=False):
    """
    ChromaDB 클라이언트와 컬렉션을 생성합니다.
    
    Args:
        collection_name (str): 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        overwrite (bool): 기존 컬렉션을 덮어쓸지 여부
        
    Returns:
        tuple: (chromadb.Client, chromadb.Collection) 클라이언트와 컬렉션
    """
    # 디렉토리가 없으면 생성
    os.makedirs(persist_directory, exist_ok=True)
    
    # ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 컬렉션 존재 여부 확인
    collections = client.list_collections()
    collection_exists = collection_name in [c.name for c in collections]
    
    if collection_exists:
        if overwrite:
            # 기존 컬렉션 삭제 후 새로 생성
            client.delete_collection(collection_name)
            collection = client.create_collection(name=collection_name)
        else:
            # 기존 컬렉션 사용
            collection = client.get_collection(name=collection_name)
    else:
        # 새 컬렉션 생성
        collection = client.create_collection(name=collection_name)
    
    return client, collection

def load_chroma_collection(collection_name="csv_test", persist_directory="./chroma_db"):
    """
    기존 ChromaDB 컬렉션을 로드합니다.
    
    Args:
        collection_name (str): 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        
    Returns:
        tuple: (chromadb.Client, chromadb.Collection) 클라이언트와 컬렉션
    """
    # 디렉토리가 없으면 오류
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {persist_directory}")
    
    # ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 컬렉션 목록 확인
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    
    # 컬렉션이 없으면 오류
    if collection_name not in collection_names:
        raise ValueError(f"컬렉션 '{collection_name}'을 찾을 수 없습니다. 사용 가능한 컬렉션: {collection_names}")
    
    # 컬렉션 로드
    collection = client.get_collection(collection_name)
    
    return client, collection

def get_available_collections(persist_directory="./chroma_db"):
    """
    사용 가능한 ChromaDB 컬렉션 목록을 가져옵니다.
    
    Args:
        persist_directory (str): 데이터베이스 저장 경로
        
    Returns:
        list: 컬렉션 이름 목록
    """
    # 디렉토리가 없으면 빈 목록 반환
    if not os.path.exists(persist_directory):
        return []
    
    try:
        # ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(path=persist_directory)
        
        # 컬렉션 목록 가져오기
        collections = client.list_collections()
        return [c.name for c in collections]
    except:
        return []

def store_data_in_chroma(df, selected_columns, collection_name="csv_test", persist_directory="./chroma_db", chunk_size=500, chunk_overlap=50, max_rows=None, batch_size=100, append=True):
    """
    선택한 열의 데이터를 ChromaDB에 저장합니다.
    
    Args:
        df (pandas.DataFrame): 데이터프레임
        selected_columns (list): 저장할 열 이름 목록
        collection_name (str): 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        chunk_size (int): 청크 크기
        chunk_overlap (int): 청크 오버랩 크기
        max_rows (int, optional): 처리할 최대 행 수
        batch_size (int): 배치 처리 크기
        append (bool): 기존 컬렉션에 데이터를 추가할지 여부
        
    Returns:
        tuple: (chromadb.Client, chromadb.Collection) 클라이언트와 컬렉션
    """
    # 데이터 전처리
    selected_df = preprocess_dataframe(df, selected_columns, max_rows)
    
    # 전처리 후 데이터가 없는 경우
    if selected_df.empty:
        raise ValueError("선택한 열에 유효한 데이터가 없습니다. 결측치가 있는 행은 모두 제거됩니다.")
    
    # ChromaDB 생성 또는 로드
    client, collection = create_chroma_db(collection_name, persist_directory, overwrite=not append)
    
    # 텍스트 분할기 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # 배치 처리를 위한 변수 초기화
    batch_documents = []
    batch_metadatas = []
    batch_ids = []
    
    # 기존 문서 ID 가져오기 (중복 방지)
    existing_ids = set()
    if append:
        try:
            # 기존 컬렉션에서 모든 ID 가져오기
            # 참고: 대용량 컬렉션의 경우 이 부분이 메모리 문제를 일으킬 수 있음
            existing_ids = set(collection.get()["ids"])
        except:
            # ID를 가져올 수 없는 경우 빈 세트 사용
            existing_ids = set()
    
    # 각 행을 처리
    for idx, row in selected_df.iterrows():
        # 행 데이터를 문자열로 변환
        row_text = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
        
        # 텍스트가 충분히 길면 분할
        if len(row_text) > chunk_size:
            chunks = text_splitter.split_text(row_text)
            for i, chunk in enumerate(chunks):
                # 고유 ID 생성 (UUID 사용)
                doc_id = f"row_{idx}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                
                # ID 중복 확인
                while doc_id in existing_ids:
                    doc_id = f"row_{idx}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                
                existing_ids.add(doc_id)
                batch_documents.append(chunk)
                batch_metadatas.append({"source": f"row_{idx}", "chunk": i})
                batch_ids.append(doc_id)
                
                # 배치 크기에 도달하면 저장
                if len(batch_documents) >= batch_size:
                    collection.add(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    batch_documents = []
                    batch_metadatas = []
                    batch_ids = []
        else:
            # 짧은 텍스트는 그대로 저장
            # 고유 ID 생성 (UUID 사용)
            doc_id = f"row_{idx}_{uuid.uuid4().hex[:8]}"
            
            # ID 중복 확인
            while doc_id in existing_ids:
                doc_id = f"row_{idx}_{uuid.uuid4().hex[:8]}"
            
            existing_ids.add(doc_id)
            batch_documents.append(row_text)
            batch_metadatas.append({"source": f"row_{idx}"})
            batch_ids.append(doc_id)
            
            # 배치 크기에 도달하면 저장
            if len(batch_documents) >= batch_size:
                collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                batch_documents = []
                batch_metadatas = []
                batch_ids = []
    
    # 남은 배치 처리
    if batch_documents:
        collection.add(
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    
    return client, collection

def query_chroma(collection, query_text, n_results=5):
    """
    ChromaDB에서 쿼리를 실행합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query_text (str): 쿼리 텍스트
        n_results (int): 반환할 결과 수 (기본값: 5)
        
    Returns:
        dict: 쿼리 결과
    """
    # 쿼리 텍스트 정제
    cleaned_query = clean_text(query_text)
    
    results = collection.query(
        query_texts=[cleaned_query],
        n_results=n_results
    )
    
    return results

def is_ollama_installed():
    """
    Ollama가 설치되어 있는지 확인합니다.
    
    Returns:
        bool: Ollama가 설치되어 있으면 True, 아니면 False
    """
    try:
        if platform.system() == "Windows":
            # Windows에서는 where 명령어 사용
            subprocess.run(["where", "ollama"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            # Linux/macOS에서는 which 명령어 사용
            subprocess.run(["which", "ollama"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def is_ollama_running():
    """
    Ollama 서버가 실행 중인지 확인합니다.
    
    Returns:
        bool: Ollama 서버가 실행 중이면 True, 아니면 False
    """
    try:
        import ollama
        try:
            ollama.list()
            return True
        except Exception:
            return False
    except ImportError:
        return False

def is_ollama_lib_available():
    """
    Ollama 라이브러리가 설치되어 있는지 확인합니다.
    
    Returns:
        bool: Ollama 라이브러리가 설치되어 있으면 True, 아니면 False
    """
    try:
        import ollama
        return True
    except ImportError:
        return False

def get_ollama_models():
    """
    설치된 Ollama 모델 목록을 가져옵니다.
    
    Returns:
        list: 설치된 모델 목록
    """
    try:
        import ollama
        result = ollama.list()
        
        # 디버깅을 위해 전체 응답 출력
        print("Ollama API 응답:", result)
        
        # 응답 구조 확인
        if hasattr(result, 'models') and isinstance(result.models, list):
            # 새로운 API 구조: result.models는 Model 객체 리스트
            return [model.model for model in result.models]
        elif isinstance(result, dict) and 'models' in result:
            # 이전 API 구조: result['models']는 딕셔너리 리스트
            return [model['name'] for model in result['models']]
        else:
            # 다른 구조인 경우
            print("예상치 못한 응답 구조:", result)
            if isinstance(result, list):
                # 리스트인 경우 각 항목의 model 속성 또는 문자열 변환
                return [getattr(item, 'model', str(item)) for item in result]
            else:
                # 그 외의 경우 빈 리스트 반환
                return []
    except Exception as e:
        print(f"모델 목록을 가져오는 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []

def query_ollama(prompt, model_name="llama2"):
    """
    Ollama 모델에 질의합니다.
    
    Args:
        prompt (str): 프롬프트
        model_name (str): Ollama 모델 이름
        
    Returns:
        str: 모델 응답
    """
    try:
        import ollama
        
        # Ollama 라이브러리 사용
        response = ollama.generate(model=model_name, prompt=prompt)
        
        # 응답 구조 확인
        if isinstance(response, dict) and 'response' in response:
            # 이전 API 구조: 딕셔너리 형태
            return response['response']
        elif hasattr(response, 'response'):
            # 새로운 API 구조: 객체 형태
            return response.response
        else:
            # 다른 구조인 경우
            print("예상치 못한 응답 구조:", response)
            return str(response)
    except ImportError:
        return "ollama 패키지가 설치되어 있지 않습니다. 'pip install ollama'를 실행하세요."
    except Exception as e:
        print(f"Ollama 질의 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return f"Ollama 오류: {e}"

def get_ollama_install_guide():
    """
    Ollama 설치 가이드를 반환합니다.
    
    Returns:
        str: 설치 가이드 마크다운
    """
    system = platform.system()
    
    if system == "Windows":
        return """
        ### Windows에 Ollama 설치하기

        1. [Ollama 웹사이트](https://ollama.ai/download/windows)에서 Windows 설치 파일을 다운로드합니다.
        2. 다운로드한 설치 파일을 실행하고 설치 지침을 따릅니다.
        3. 설치가 완료되면 시스템 트레이에 Ollama 아이콘이 나타납니다.
        4. 모델을 다운로드하려면 명령 프롬프트를 열고 다음 명령어를 실행합니다:
           ```
           ollama pull llama2
           ```
        5. 이 애플리케이션으로 돌아와 새로고침 버튼을 클릭하세요.
        
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 직접 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        """
    elif system == "Darwin":  # macOS
        return """
        ### macOS에 Ollama 설치하기

        1. [Ollama 웹사이트](https://ollama.ai/download/mac)에서 macOS 설치 파일을 다운로드합니다.
        2. 다운로드한 .dmg 파일을 열고 Ollama를 Applications 폴더로 드래그합니다.
        3. Applications 폴더에서 Ollama를 실행합니다.
        4. 모델을 다운로드하려면 터미널을 열고 다음 명령어를 실행합니다:
           ```
           ollama pull llama2
           ```
        5. 이 애플리케이션으로 돌아와 새로고침 버튼을 클릭하세요.
        
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 직접 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        """
    else:  # Linux
        return """
        ### Linux에 Ollama 설치하기

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
        4. 이 애플리케이션으로 돌아와 새로고침 버튼을 클릭하세요.
        
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 직접 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        """

def rag_query_with_ollama(collection, query, model_name="llama2", n_results=5):
    """
    RAG 시스템에 질의합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query (str): 질의 텍스트 (프롬프트와 질문이 결합된 형태일 수 있음)
        model_name (str): Ollama 모델 이름
        n_results (int): 검색할 결과 수, 0이면 최대 20개 문서 사용, 최소 3개 문서 사용
        
    Returns:
        dict: 질의 결과
    """
    # 쿼리 텍스트 정제
    cleaned_query = clean_text(query)
    
    # n_results가 0이면 최대 20개 문서로 제한
    if n_results == 0:
        n_results = 20  # 일반적인 최대값
    
    # n_results가 None이면 기본값 5 사용
    elif n_results is None:
        n_results = 5
        
    # n_results가 3보다 작으면 최소 3으로 설정
    elif n_results < 3:
        n_results = 3
        
    # n_results가 20보다 크면 최대 20으로 제한
    elif n_results > 20:
        n_results = 20
    
    # ChromaDB에서 관련 문서 검색
    results = collection.query(
        query_texts=[cleaned_query],
        n_results=n_results
    )
    
    # 검색 결과를 컨텍스트로 사용
    context = "\n".join(results["documents"][0])
    
    # 사용자 입력이 프롬프트와 질문으로 구성되어 있는지 확인
    if "질문:" in query or "Question:" in query:
        # 사용자가 직접 프롬프트와 질문을 입력한 경우, 그대로 사용
        prompt = f"""
    다음 정보를 바탕으로 응답해주세요:
    
    {context}
    
    {query}
    """
    else:
        # 기존 방식 - 질문 형식으로 구성
        prompt = f"""
    다음 정보를 바탕으로 질문에 답변해주세요:
    
    {context}
    
    질문: {query}
    답변:
    """
    
    response = query_ollama(prompt, model_name)
    
    return {
        "query": query,
        "context": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
        "response": response
    }