"""
ChromaDB 관련 유틸리티 함수 모음
"""
import os
import uuid
import kss
import chromadb
import gc
import time
import logging
import pandas as pd
import torch # For GPU monitoring
import threading # For GPU monitoring thread
from text_utils import clean_text
from data_utils import preprocess_dataframe
from embedding_utils import get_embedding_status, get_normalized_embedding_function, is_gpu_available


# 로깅 설정
logger = logging.getLogger("chroma_utils")

def create_chroma_db(collection_name="csv_test", persist_directory="./chroma_db", overwrite=False, embedding_model="all-MiniLM-L6-v2", embedding_device_preference="auto"):
    """
    ChromaDB 클라이언트와 컬렉션을 생성합니다.
    
    Args:
        collection_name (str): 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        overwrite (bool): 기존 컬렉션을 덮어쓸지 여부
        embedding_model (str): 요청할 임베딩 모델 이름
        embedding_device_preference (str): 임베딩 연산에 사용할 장치 ("auto", "cuda", "cpu")
        
    Returns:
        tuple: (chromadb.Client, chromadb.Collection) 클라이언트와 컬렉션
    """
    # 디렉토리가 없으면 생성
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        # ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(path=persist_directory)
        
        # 임베딩 함수 설정 (L2 정규화 적용)
        embedding_function = get_normalized_embedding_function(embedding_model, device_preference=embedding_device_preference)
        
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
                    metadata={"embedding_model": get_embedding_status()["actual_model"]}  # 실제 사용된 임베딩 모델 정보 저장
                )
            else:
                # 기존 컬렉션 사용
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
                # 기존 컬렉션에 임베딩 모델 메타데이터 업데이트
                try:
                    collection.metadata = {"embedding_model": get_embedding_status()["actual_model"]}
                except:
                    # 일부 버전에서는 메타데이터 직접 업데이트가 지원되지 않을 수 있음
                    pass
        else:
            # 새 컬렉션 생성
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"embedding_model": get_embedding_status()["actual_model"]}  # 실제 사용된 임베딩 모델 정보 저장
            )
        
        return client, collection
    except Exception as e:
        logger.error(f"ChromaDB 생성 중 오류 발생: {e}")
        raise

def load_chroma_collection(collection_name="csv_test", persist_directory="./chroma_db", embedding_device_preference="auto"):
    """
    기존 ChromaDB 컬렉션을 로드합니다.
    
    Args:
        collection_name (str): 컬렉션 이름
        persist_directory (str): 데이터베이스 저장 경로
        embedding_device_preference (str): 임베딩 연산에 사용할 장치 ("auto", "cuda", "cpu")
        
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
    embedding_model_to_load = "all-MiniLM-L6-v2"  # 기본값
    
    if collection_exists:
        # 임시로 임베딩 함수 없이 컬렉션 가져오기 (메타데이터 확인용)
        try:
            temp_collection = client.get_collection(name=collection_name)
            # 컬렉션 메타데이터에서 임베딩 모델 정보 확인
            if temp_collection.metadata and "embedding_model" in temp_collection.metadata:
                stored_model = temp_collection.metadata["embedding_model"]
                print(f"컬렉션에 저장된 임베딩 모델 정보를 찾았습니다: {stored_model}")
                embedding_model_to_load = stored_model # 컬렉션 생성 시 사용된 모델을 우선 로드 시도
        except Exception as e:
            # 메타데이터 접근 실패 시 기본 모델 사용
            print(f"컬렉션 메타데이터 접근 실패: {str(e)}, 기본 임베딩 모델을 사용합니다: {embedding_model_to_load}")
    
    # 임베딩 모델이 지정되지 않았으면 컬렉션에서 가져오기
    if embedding_model_to_load is None: # 이 경우는 거의 없지만 안전장치
        embedding_model_to_load = "all-MiniLM-L6-v2"
    
    # 임베딩 함수 설정 (L2 정규화 적용)
    embedding_function = get_normalized_embedding_function(embedding_model_to_load, device_preference=embedding_device_preference)
    actual_embedding_model_used = get_embedding_status()["actual_model"] # 실제 로드된 모델
    
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
                    collection.metadata = {"embedding_model": actual_embedding_model_used}
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
            metadata={"embedding_model": actual_embedding_model_used}
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
                          embedding_model="all-MiniLM-L6-v2", embedding_device_preference="auto", progress_bar=None, status_text=None, gpu_status_placeholder=None):
    """ # noqa: E501
    선택한 열의 데이터를 ChromaDB에 저장합니다.
    의미 기반 청킹 기능을 사용하여 검색 정확도와 유사도 계산의 품질을 향상시킵니다.
    
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
        embedding_device_preference (str): 임베딩 연산에 사용할 장치 ("auto", "cuda", "cpu")
        progress_bar (streamlit.Progress, optional): Streamlit progress bar 객체
        status_text (streamlit.empty, optional): Streamlit status text 객체
        gpu_status_placeholder (streamlit.empty, optional): Streamlit empty object for GPU status
        
    Returns:
        tuple: (chromadb.Client, chromadb.Collection) 클라이언트와 컬렉션
    """
    start_time = time.time()
    logger.info(f"ChromaDB 데이터 저장 시작: {collection_name}")
    
    monitor_thread = None
    stop_event = None

    # GPU 모니터링 스레드 설정
    if embedding_device_preference != "cpu" and is_gpu_available() and gpu_status_placeholder:
        stop_event = threading.Event()

        def monitor_gpu_memory(stop_event_ref): # placeholder 인자 제거
            initial_total_gpu_mem_mb = 0
            if torch.cuda.is_available():
                try:
                    initial_total_gpu_mem_mb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**2)
                    logger.info(f"GPU 총 메모리: {initial_total_gpu_mem_mb:.0f} MB. 사용량 모니터링 중...") # logger 사용
                except Exception as e:
                    logger.error(f"GPU 총 메모리 조회 실패: {e}")
                    logger.warning("GPU 총 메모리 조회 실패.") # logger 사용

            while not stop_event_ref.is_set():
                try:
                    if torch.cuda.is_available(): # Check again in loop in case GPU becomes unavailable
                        allocated = torch.cuda.memory_allocated() / (1024**2) # MB
                        reserved = torch.cuda.memory_reserved() / (1024**2) # MB
                        logger.info(f"GPU Mem: 사용 {allocated:.0f}MB | 예약 {reserved:.0f}MB (총 {initial_total_gpu_mem_mb:.0f}MB)") # logger 사용
                    else:
                        logger.warning("GPU를 사용할 수 없습니다.") # logger 사용
                        break
                    time.sleep(1) # 1초마다 업데이트
                except Exception: # GPU 사용 불가 또는 기타 오류 처리
                    logger.warning("GPU 모니터링 중 오류 발생. 모니터링 중단.") # logger 사용
                    break
            logger.info("GPU 모니터링 스레드 종료.") # logger 사용

    try:
        # 데이터 전처리
        from text_utils import chunk_text_semantic, extract_keywords # 원본 호출 유지
        selected_df = preprocess_dataframe(df, selected_columns, max_rows)
        
        # 전처리 후 데이터가 없는 경우
        if selected_df.empty:
            raise ValueError("선택한 열에 유효한 데이터가 없습니다. 결측치가 있는 행은 모두 제거됩니다.")
        
        # 처리할 전체 행 수
        total_rows = len(selected_df)
        logger.info(f"처리할 전체 행 수: {total_rows}")

        # GPU 모니터링 스레드 시작
        if monitor_thread is None and stop_event is not None and gpu_status_placeholder is not None: # Ensure thread is not already started
            monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(stop_event,)) # placeholder 인자 전달 제거
            monitor_thread.start()
        
        # ChromaDB 생성 또는 로드 (임베딩 모델 지정)
        client, collection = create_chroma_db(collection_name, persist_directory, overwrite=not append, embedding_model=embedding_model, embedding_device_preference=embedding_device_preference)
        
        # 임베딩 함수가 제대로 설정되었는지 확인
        if not hasattr(collection, "_embedding_function") or collection._embedding_function is None:
            logger.warning("경고: 컬렉션에 임베딩 함수가 설정되지 않았습니다. 시각화 기능이 작동하지 않을 수 있습니다.")
        
        # 기존 문서 ID 가져오기 (중복 방지)
        existing_ids = set()
        if append:
            try:
                # 기존 컬렉션에서 모든 ID 가져오기 시도
                try:
                    existing_ids = set(collection.get()["ids"])
                    logger.info(f"기존 문서 ID {len(existing_ids)}개 로드됨")
                except Exception as e:
                    # get() 메서드가 제대로 작동하지 않는 경우 query() 메서드로 시도
                    logger.warning(f"기존 ID를 get() 메서드로 가져오는데 실패: {e}, query() 메서드 시도 중...")
                    result = collection.query(query_texts=[""], n_results=1)
                    if result and "ids" in result and result["ids"]:
                        # 컬렉션이 존재하지만 문서가 없는 경우
                        existing_ids = set()
                    else:
                        logger.error(f"query() 메서드도 실패: {e}")
                        existing_ids = set()
            except Exception as e:
                # ID를 가져올 수 없는 경우 빈 세트 사용
                logger.error(f"기존 ID를 가져오는 중 오류 발생: {e}, 빈 세트 사용")
                existing_ids = set()
        
        # 총 처리 건수 추적
        total_documents_processed = 0
        
        # 병렬 처리 대신 부분적으로 행을 처리하여 메모리 사용량 최적화
        # 최대 처리 행 수가 지정된 경우 슬라이싱
        if max_rows:
            selected_df = selected_df.head(max_rows)
        
        # 더 작은 청크 단위로 나누어 처리
        chunk_size_rows = min(batch_size * 5, 500)  # 한 번에 최대 500행씩 처리
        total_chunks = (len(selected_df) + chunk_size_rows - 1) // chunk_size_rows
        logger.info(f"전체 처리할 행 청크 수: {total_chunks}")

        if status_text:
            status_text.text(f"데이터 처리 준비 중... 총 {total_chunks}개 대형 배치 예정")
        if progress_bar:
            progress_bar.progress(0)
        
        for chunk_idx in range(total_chunks):
            # 행 청크 계산
            start_idx = chunk_idx * chunk_size_rows
            end_idx = min((chunk_idx + 1) * chunk_size_rows, len(selected_df))
            df_chunk = selected_df.iloc[start_idx:end_idx]
            
            if status_text:
                status_text.text(f"데이터 저장 진행: {chunk_idx + 1}/{total_chunks} 번째 대형 배치 처리 중...")
            if progress_bar:
                progress_bar.progress((chunk_idx + 1) / total_chunks)
            
            # 배치 처리를 위한 변수 초기화
            batch_documents = []
            batch_metadatas = []
            batch_ids = []
            
            # 각 행을 처리
            for idx, row in df_chunk.iterrows():
                # 메모리 상태 확인 및 최적화
                if idx % 100 == 0 and idx > 0:
                    # 가비지 컬렉션 호출 (메모리 최적화)
                    gc.collect()
                
                # 행 데이터를 문자열로 변환
                try:
                    row_text = " ".join([str(val) for val in row.values if not pd.isna(val)])
                except Exception as e:
                    logger.error(f"행 {idx} 데이터 변환 중 오류: {e}, 건너뜁니다")
                    continue
                
                # 행이 비어 있으면 건너뛰기
                if not row_text.strip():
                    logger.warning(f"행 {idx}이(가) 비어 있어 건너뜁니다")
                    continue
                
                try:
                    # 의미 기반 청킹 함수 사용
                    try:
                        chunks = chunk_text_semantic(row_text, chunk_size, chunk_overlap)
                    except Exception as e:
                        logger.warning(f"의미 기반 청킹 중 오류 발생: {e}, 기본 분할 방식 사용")
                        # 텍스트 길이가 너무 길면 분할
                        chunks = []
                        if len(row_text) > chunk_size:
                            for i in range(0, len(row_text), chunk_size - chunk_overlap):
                                end = min(i + chunk_size, len(row_text))
                                chunks.append(row_text[i:end])
                                if end >= len(row_text):
                                    break
                        else:
                            chunks = [row_text]
                    
                    # 청크가 없으면 원본 텍스트를 그대로 사용
                    if not chunks:
                        chunks = [row_text]
                    
                    # 청크가 너무 작으면 하나로 합치기
                    if len(chunks) == 1 or all(len(chunk) < chunk_size // 4 for chunk in chunks):
                        chunks = [row_text]
                        
                except Exception as e:
                    logger.error(f"행 {idx} 청킹 중 오류 발생: {e}, 원본 텍스트 사용")
                    chunks = [row_text]
                
                # 청크 처리
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                        
                    # 키워드 추출 (메타데이터로 저장)
                    try:
                        # 청크의 길이와 문장 수에 따라 키워드 추출 개수 동적 조절
                        text_length = len(chunk)
                        
                        # 문장 수 계산 (마침표, 물음표, 느낌표 기준)
                        import re
                        sentences = re.split(r'[.!?]+', chunk)
                        sentences = [s.strip() for s in sentences if s.strip()]
                        sentence_count = len(sentences)
                        
                        # 텍스트 길이와 문장 수에 따른 키워드 수 계산
                        # 기본값 5개, 텍스트가 길거나 문장이 많으면 증가
                        if text_length < 100:  # 매우 짧은 텍스트
                            keywords_count = max(2, min(3, sentence_count))
                        elif text_length < 300:  # 짧은 텍스트
                            keywords_count = max(3, min(5, sentence_count))
                        elif text_length < 800:  # 중간 길이 텍스트
                            keywords_count = max(5, min(8, sentence_count))
                        else:  # 긴 텍스트
                            keywords_count = max(8, min(12, sentence_count))
                        
                        # 키워드 추출 (동적으로 결정된 개수)
                        keywords = extract_keywords(chunk, top_n=keywords_count)
                        keywords_str = ", ".join(keywords)
                    except Exception as e:
                        logger.warning(f"키워드 추출 중 오류 발생: {e}")
                        keywords_str = ""
                    
                    # 고유 ID 생성 (UUID 사용)
                    doc_id = f"row_{idx}_chunk_{i}_{uuid.uuid4().hex[:8]}"
    
                    # ID 중복 확인 (무한 루프 방지를 위한 최대 시도 횟수 제한)
                    max_attempts = 5
                    attempts = 0
                    while doc_id in existing_ids and attempts < max_attempts:
                        doc_id = f"row_{idx}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                        attempts += 1
    
                    existing_ids.add(doc_id)
                    batch_documents.append(chunk)
                    
                    # 메타데이터에 키워드 정보 추가 (None 값 방지)
                    metadata = {
                        "source": f"row_{idx}",
                        "chunk": i,
                        "keywords": keywords_str if keywords_str else ""  # None 대신 빈 문자열 사용
                    }
                    # None 값이 있는지 확인하고 제거
                    for key in list(metadata.keys()):
                        if metadata[key] is None:
                            metadata[key] = ""  # None 값을 빈 문자열로 대체
                    batch_metadatas.append(metadata)
                    batch_ids.append(doc_id)
                    
                    # 배치 크기에 도달하면 저장
                    if len(batch_documents) >= batch_size:
                        try:
                            collection.add(
                                documents=batch_documents,
                                metadatas=batch_metadatas,
                                ids=batch_ids
                            )
                            total_documents_processed += len(batch_documents)
                            logger.info(f"배치 저장 성공: {len(batch_documents)}개 문서, 전체 진행률: {total_documents_processed}/{total_rows*2} 문서")
                        except Exception as e:
                            logger.error(f"배치 저장 중 오류 발생: {e}, 계속 진행합니다")
                            
                        # 배치 초기화
                        batch_documents = []
                        batch_metadatas = []
                        batch_ids = []
                        
                        # 메모리 최적화를 위한 가비지 컬렉션 호출
                        gc.collect()
            
            # 남은 배치 처리
            if batch_documents:
                try:
                    collection.add(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    total_documents_processed += len(batch_documents)
                    logger.info(f"남은 배치 저장 성공: {len(batch_documents)}개 문서, 전체 진행률: {total_documents_processed}/{total_rows*2} 문서")
                except Exception as e:
                    logger.error(f"남은 배치 저장 중 오류 발생: {e}")
            
            # 청크마다 가비지 컬렉션 호출 및 약간의 지연 추가
            gc.collect()
            time.sleep(0.5)  # 메모리 정리를 위한 짧은 지연
            
            logger.info(f"행 청크 {chunk_idx+1}/{total_chunks} 처리 완료")
        
        end_time = time.time()
        logger.info(f"ChromaDB 데이터 저장 완료: {total_documents_processed}개 문서, 소요 시간: {end_time - start_time:.2f}초")
        
        if status_text:
            status_text.text("ChromaDB 데이터 저장 완료!")
        if progress_bar:
            progress_bar.progress(1.0) # Ensure it reaches 100%
        
        return client, collection
    except Exception as e:
        logger.error(f"ChromaDB 데이터 저장 중 오류 발생: {e}", exc_info=True)
        raise
    finally:
        # GPU 모니터링 스레드 종료
        if stop_event and monitor_thread and monitor_thread.is_alive():
            stop_event.set()
            monitor_thread.join(timeout=2) # 스레드가 종료될 때까지 최대 2초 대기
        if gpu_status_placeholder: # Ensure placeholder is cleared if it was used
            gpu_status_placeholder.empty()
            
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
    
    # 2. 키워드 기반 검색 준비
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
    def process_embedding_results():
        if not embedding_results["documents"] or not embedding_results["documents"][0]:
            return
            
        for doc_id, doc, metadata, distance in zip(
            embedding_results["ids"][0],
            embedding_results["documents"][0],
            embedding_results["metadatas"][0],
            embedding_results["distances"][0]
        ):
            # 이미 추가된 문서인지 확인
            if doc_id not in combined_results["ids"]:
                combined_results["ids"].append(doc_id)
                combined_results["documents"].append(doc)
                combined_results["metadatas"].append(metadata)
                combined_results["distances"].append(distance)
                combined_results["search_type"].append("embedding")
    
    # 키워드 검색 결과 처리
    def process_keyword_results():
        # 모든 문서를 가져와서 키워드 매칭 확인
        all_docs = collection.get()
        
        if not all_docs or not all_docs["documents"]:
            return
            
        for doc_id, doc, metadata in zip(
            all_docs["ids"],
            all_docs["documents"],
            all_docs["metadatas"]
        ):
            # 이미 임베딩 결과에 포함된 문서는 건너뜀
            if doc_id in combined_results["ids"]:
                continue
            
            # 문서를 소문자로 변환하여 키워드 검색
            doc_lower = doc.lower()
            
            # 키워드가 문서에 포함되어 있는지 확인
            keyword_match = any(keyword in doc_lower for keyword in query_keywords)
            
            if keyword_match:
                # 키워드 매칭 문서 추가
                combined_results["ids"].append(doc_id)
                combined_results["documents"].append(doc)
                combined_results["metadatas"].append(metadata)
                # 키워드 매칭은 거리를 0.5로 설정 (임베딩 결과와 구분)
                combined_results["distances"].append(0.5)
                combined_results["search_type"].append("keyword")
    
    # 결과 처리 실행
    process_embedding_results()
    process_keyword_results()
    
    # 결과 정렬 및 제한
    def sort_and_limit_results():
        # 결과가 없으면 빈 결과 반환
        if not combined_results["distances"]:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "search_type": [[]]
            }
            
        # 결과 정렬 (거리 기준)
        sorted_indices = sorted(range(len(combined_results["distances"])), 
                               key=lambda i: combined_results["distances"][i])
        
        # 결과 수 제한
        sorted_indices = sorted_indices[:n_results]
        
        # 정렬된 결과 생성
        sorted_results = {
            "ids": [combined_results["ids"][i] for i in sorted_indices],
            "documents": [combined_results["documents"][i] for i in sorted_indices],
            "metadatas": [combined_results["metadatas"][i] for i in sorted_indices],
            "distances": [combined_results["distances"][i] for i in sorted_indices],
            "search_type": [combined_results["search_type"][i] for i in sorted_indices]
        }
        
        # ChromaDB 결과 형식에 맞게 변환
        return {
            "ids": [sorted_results["ids"]],
            "documents": [sorted_results["documents"]],
            "metadatas": [sorted_results["metadatas"]],
            "distances": [sorted_results["distances"]],
            "search_type": [sorted_results["search_type"]]
        }
    
    return sort_and_limit_results()

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