"""
ChromaDB 관련 유틸리티 함수 모음
"""
import os
import uuid
import kss
import chromadb
from text_utils import clean_text
from data_utils import preprocess_dataframe
from embedding_utils import get_embedding_function, get_embedding_status, get_normalized_embedding_function

def create_chroma_db(collection_name="csv_test", persist_directory="./chroma_db", overwrite=False, embedding_model="all-MiniLM-L6-v2"):
    """
    ChromaDB 클라이언트와 컬렉션을 생성합니다.
    
    Args:
        collection_name (str): 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        overwrite (bool): 기존 컬렉션을 덮어쓸지 여부
        embedding_model (str): 임베딩 모델 이름
        
    Returns:
        tuple: (chromadb.Client, chromadb.Collection) 클라이언트와 컬렉션
    """
    # 디렉토리가 없으면 생성
    os.makedirs(persist_directory, exist_ok=True)
    
    # ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 임베딩 함수 설정 (L2 정규화 적용)
    embedding_function = get_normalized_embedding_function(embedding_model)
    
    # 컬렉션 존재 여부 확인
    collections = client.list_collections()
    collection_exists = collection_name in [c.name for c in collections]
    
    if collection_exists:
        if overwrite:
            # 기존 컬렉션 삭제 후 새로 생성
            client.delete_collection(collection_name)
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"embedding_model": embedding_model}  # 임베딩 모델 정보 저장
            )
        else:
            # 기존 컬렉션 사용
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            # 기존 컬렉션에 임베딩 모델 메타데이터 업데이트
            try:
                collection.metadata = {"embedding_model": embedding_model}
            except:
                # 일부 버전에서는 메타데이터 직접 업데이트가 지원되지 않을 수 있음
                pass
    else:
        # 새 컬렉션 생성
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"embedding_model": embedding_model}  # 임베딩 모델 정보 저장
        )
    
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
    
    # 컬렉션 존재 여부 확인
    collections = client.list_collections()
    collection_exists = collection_name in [c.name for c in collections]
    embedding_model = "all-MiniLM-L6-v2"  # 기본값
    
    if collection_exists:
        # 임시로 임베딩 함수 없이 컬렉션 가져오기 (메타데이터 확인용)
        try:
            temp_collection = client.get_collection(name=collection_name)
            # 컬렉션 메타데이터에서 임베딩 모델 정보 확인
            if temp_collection.metadata and "embedding_model" in temp_collection.metadata:
                stored_model = temp_collection.metadata["embedding_model"]
                print(f"컬렉션에 저장된 임베딩 모델 정보를 찾았습니다: {stored_model}")
                # 저장된 임베딩 모델 사용
                embedding_model = stored_model
        except Exception as e:
            # 메타데이터 접근 실패 시 기본 모델 사용
            print(f"컬렉션 메타데이터 접근 실패: {str(e)}, 기본 임베딩 모델을 사용합니다: {embedding_model}")
    
    # 임베딩 모델이 지정되지 않았으면 컬렉션에서 가져오기
    if embedding_model is None:
        embedding_model = "all-MiniLM-L6-v2"  # 기본값
    
    # 임베딩 함수 설정 (L2 정규화 적용)
    embedding_function = get_normalized_embedding_function(embedding_model)
    
    if collection_exists:
        # 기존 컬렉션 사용
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            # 임베딩 모델 정보가 없으면 추가
            if not collection.metadata or "embedding_model" not in collection.metadata:
                try:
                    collection.metadata = {"embedding_model": embedding_model}
                except:
                    # 일부 버전에서는 메타데이터 직접 업데이트가 지원되지 않을 수 있음
                    pass
        except Exception as e:
            print(f"임베딩 함수로 컬렉션 로드 실패: {str(e)}, 임베딩 함수 없이 로드를 시도합니다.")
            # 임베딩 함수 없이 컬렉션 로드 시도
            collection = client.get_collection(name=collection_name)
    else:
        # 새 컬렉션 생성
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"embedding_model": embedding_model}
        )
    
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

def store_data_in_chroma(df, selected_columns, collection_name="csv_test", persist_directory="./chroma_db", 
                          chunk_size=500, chunk_overlap=50, max_rows=None, batch_size=100, append=True, 
                          embedding_model="all-MiniLM-L6-v2"):
    """
    선택한 열의 데이터를 ChromaDB에 저장합니다.
    
    Args:
        df (pandas.DataFrame): 데이터프레임
        selected_columns (list): 저장할 열 이름 목록
        collection_name (str): 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        chunk_size (int): 청크 크기 (최대 문장 길이)
        chunk_overlap (int): 청크 오버랩 크기
        max_rows (int, optional): 처리할 최대 행 수
        batch_size (int): 배치 처리 크기
        append (bool): 기존 컬렉션에 데이터를 추가할지 여부
        embedding_model (str): 임베딩 모델 이름
        
    Returns:
        tuple: (chromadb.Client, chromadb.Collection) 클라이언트와 컬렉션
    """
    # 데이터 전처리
    selected_df = preprocess_dataframe(df, selected_columns, max_rows)
    
    # 전처리 후 데이터가 없는 경우
    if selected_df.empty:
        raise ValueError("선택한 열에 유효한 데이터가 없습니다. 결측치가 있는 행은 모두 제거됩니다.")
    
    # ChromaDB 생성 또는 로드 (임베딩 모델 지정)
    client, collection = create_chroma_db(collection_name, persist_directory, overwrite=not append, embedding_model=embedding_model)
    
    # 임베딩 함수가 제대로 설정되었는지 확인
    if not hasattr(collection, "_embedding_function") or collection._embedding_function is None:
        print("경고: 컬렉션에 임베딩 함수가 설정되지 않았습니다. 시각화 기능이 작동하지 않을 수 있습니다.")
    
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
        
        try:
            # kss를 사용하여 한글 문장 단위로 분할
            sentences = kss.split_sentences(row_text)
            
            # 문장들을 최대 길이(chunk_size)를 고려하여 청크로 결합
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # 하나의 문장이 chunk_size보다 길면 그대로 청크로 추가
                if len(sentence) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    chunks.append(sentence)
                    continue
                
                # 현재 청크에 문장을 추가했을 때 chunk_size를 초과하는지 확인
                if len(current_chunk) + len(sentence) + 1 > chunk_size:  # +1: 공백 고려
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # 공백 추가 (첫 문장이 아닌 경우)
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # 남은 청크가 있으면 추가
            if current_chunk:
                chunks.append(current_chunk)
                
            # 청크가 없으면 원본 텍스트를 그대로 사용
            if not chunks:
                chunks = [row_text]
                
        except Exception as e:
            print(f"문장 분할 중 오류 발생: {e}, 기본 분할 방식 사용")
            # kss 분할에 실패한 경우, 간단히 텍스트 길이로 분할
            chunks = []
            for i in range(0, len(row_text), chunk_size - chunk_overlap):
                end = min(i + chunk_size, len(row_text))
                chunks.append(row_text[i:end])
                if end >= len(row_text):
                    break
            
            # 청크가 너무 작으면 하나로 합치기
            if len(chunks) == 1 or all(len(chunk) < chunk_size // 4 for chunk in chunks):
                chunks = [row_text]
        
        # 청크 처리
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

def hybrid_query_chroma(collection, query_text, n_results=5):
    """
    ChromaDB에서 하이브리드 쿼리(임베딩 + 키워드)를 실행합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query_text (str): 쿼리 텍스트
        n_results (int): 반환할 결과 수 (기본값: 5)
        
    Returns:
        dict: 쿼리 결과
    """
    # 쿼리 텍스트 정제
    cleaned_query = clean_text(query_text)
    
    # 1. 임베딩 기반 검색 실행
    embedding_results = collection.query(
        query_texts=[cleaned_query],
        n_results=n_results * 2  # 더 많은 결과를 가져와서 나중에 필터링
    )
    
    # 2. 키워드 기반 검색 실행 (ChromaDB의 where 필터 사용)
    # 검색어를 소문자로 변환하여 대소문자 구분 없이 검색
    query_keywords = cleaned_query.lower().split()
    
    # 결과 저장을 위한 변수
    combined_results = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "distances": [],
        "search_type": []  # 검색 유형 추가 (임베딩 또는 키워드)
    }
    
    # 임베딩 결과 처리
    if embedding_results["documents"] and embedding_results["documents"][0]:
        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            embedding_results["ids"][0],
            embedding_results["documents"][0],
            embedding_results["metadatas"][0],
            embedding_results["distances"][0]
        )):
            # 이미 추가된 문서인지 확인
            if doc_id not in combined_results["ids"]:
                combined_results["ids"].append(doc_id)
                combined_results["documents"].append(doc)
                combined_results["metadatas"].append(metadata)
                combined_results["distances"].append(distance)
                combined_results["search_type"].append("embedding")
    
    # 키워드 검색 결과 처리
    # 모든 문서를 가져와서 키워드 매칭 확인
    all_docs = collection.get()
    
    if all_docs and all_docs["documents"]:
        for i, (doc_id, doc, metadata) in enumerate(zip(
            all_docs["ids"],
            all_docs["documents"],
            all_docs["metadatas"]
        )):
            # 이미 임베딩 결과에 포함된 문서는 건너뜀
            if doc_id in combined_results["ids"]:
                continue
            
            # 문서를 소문자로 변환하여 키워드 검색
            doc_lower = doc.lower()
            
            # 모든 키워드가 문서에 포함되어 있는지 확인
            keyword_match = any(keyword in doc_lower for keyword in query_keywords)
            
            if keyword_match:
                # 키워드 매칭 문서 추가
                combined_results["ids"].append(doc_id)
                combined_results["documents"].append(doc)
                combined_results["metadatas"].append(metadata)
                # 키워드 매칭은 거리를 0.5로 설정 (임베딩 결과와 구분)
                combined_results["distances"].append(0.5)
                combined_results["search_type"].append("keyword")
    
    # 결과 정렬 (거리 기준)
    sorted_indices = sorted(range(len(combined_results["distances"])), key=lambda i: combined_results["distances"][i])
    
    # 정렬된 결과 생성
    sorted_results = {
        "ids": [combined_results["ids"][i] for i in sorted_indices[:n_results]],
        "documents": [combined_results["documents"][i] for i in sorted_indices[:n_results]],
        "metadatas": [combined_results["metadatas"][i] for i in sorted_indices[:n_results]],
        "distances": [combined_results["distances"][i] for i in sorted_indices[:n_results]],
        "search_type": [combined_results["search_type"][i] for i in sorted_indices[:n_results]]
    }
    
    # ChromaDB 결과 형식에 맞게 변환
    return {
        "ids": [sorted_results["ids"]],
        "documents": [sorted_results["documents"]],
        "metadatas": [sorted_results["metadatas"]],
        "distances": [sorted_results["distances"]],
        "search_type": [sorted_results["search_type"]]
    }

def delete_collection(collection_name, persist_directory="./chroma_db"):
    """
    ChromaDB 컬렉션을 삭제합니다.
    
    Args:
        collection_name (str): 삭제할 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        
    Returns:
        bool: 삭제 성공 여부
    """
    # 디렉토리가 없으면 False 반환
    if not os.path.exists(persist_directory):
        return False
    
    try:
        # ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(path=persist_directory)
        
        # 컬렉션 목록 확인
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        # 컬렉션이 없으면 False 반환
        if collection_name not in collection_names:
            return False
        
        # 컬렉션 삭제
        client.delete_collection(collection_name)
        return True
    except Exception as e:
        print(f"컬렉션 삭제 중 오류 발생: {e}")
        return False