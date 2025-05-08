import pandas as pd
import ollama
import time
import subprocess
import threading
import re
import concurrent.futures
import chromadb
import hashlib
import os
from typing import List, Dict, Any, Optional, Callable

# ChromaDB 설정
COLLECTION_NAME = "ollama_cache"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chromadb')

# ChromaDB 클라이언트 초기화
def get_chroma_client():
    """ChromaDB 클라이언트를 초기화하고 반환합니다."""
    try:
        # 디렉토리가 존재하는지 확인하고 생성
        if not os.path.exists(CHROMA_PERSIST_DIR):
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            print(f"ChromaDB 디렉토리 생성: {CHROMA_PERSIST_DIR}")
        
        # 클라이언트 초기화
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        print("ChromaDB 클라이언트 초기화 성공")
        return client
    except Exception as e:
        print(f"ChromaDB 초기화 오류: {str(e)}")
        # 캐싱 없이도 작동할 수 있도록 None 반환
        return None

def get_available_models():
    """
    Get a list of available models from Ollama
    
    Returns:
        tuple: (list of model names, installation guide if needed)
    """
    installation_guide = """
    # Ollama 설치 및 모델 다운로드 가이드
    
    ## Ollama 설치 방법
    1. Windows: https://ollama.com/download/windows 에서 설치 프로그램을 다운로드하여 실행하세요.
    2. macOS: https://ollama.com/download/mac 에서 설치 프로그램을 다운로드하여 실행하세요.
    3. Linux: 터미널에서 다음 명령어를 실행하세요:
       ```
       curl -fsSL https://ollama.com/install.sh | sh
       ```
    
    ## exaone 모델 다운로드 방법
    Ollama가 설치된 후, 터미널 또는 명령 프롬프트에서 다음 명령어를 실행하세요:
    ```
    ollama run exaone3.5:7.8b
    ```
    
    다운로드가 완료되면 이 애플리케이션을 다시 시작하세요.
    """
    
    try:
        # First try using the Python API
        try:
            models_response = ollama.list()
            print(f"Models response: {models_response}")
            
            # 새로운 ollama 패키지 버전에 맞게 응답 처리
            if hasattr(models_response, 'models') and models_response.models:
                return [model.model for model in models_response.models], None
            else:
                print("No models found or unexpected API response format")
                return [], installation_guide
        except Exception as api_error:
            print(f"API error: {str(api_error)}, falling back to command line")
            
        # Fall back to command line if API fails
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the command line output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    models = []
                    for line in lines[1:]:  # Skip header
                        parts = line.split()
                        if parts:
                            models.append(parts[0])  # First column is the name
                    if models:
                        return models, None
            
            # If we get here, no models were found
            return [], installation_guide
        except Exception as cmd_error:
            print(f"Command line error: {str(cmd_error)}")
            return [], installation_guide
            
    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return [], installation_guide

def compress_text(text: str, max_length: int = 4000) -> str:
    """
    텍스트를 압축하여 최대 길이를 제한합니다.
    
    Args:
        text (str): 압축할 텍스트
        max_length (int): 최대 길이
        
    Returns:
        str: 압축된 텍스트
    """
    # None이나 숫자 등 문자열이 아닌 값이 들어오면 문자열로 변환
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text or len(text) <= max_length:
        return text
    
    # 1. 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= max_length:
        return text
    
    # 2. 중복 패턴 제거 (예: 반복되는 문장이나 단어)
    # 간단한 중복 패턴 감지 및 제거
    words = text.split()
    unique_words = []
    for word in words:
        if len(unique_words) < 2 or word != unique_words[-1] or word != unique_words[-2]:
            unique_words.append(word)
    
    compressed = ' '.join(unique_words)
    if len(compressed) <= max_length:
        return compressed
    
    # 3. 텍스트가 여전히 너무 길면 앞부분과 뒷부분만 유지
    front_part = text[:max_length//2 - 10]  # 앞부분
    back_part = text[-(max_length//2 - 10):]  # 뒷부분
    
    return front_part + " [...내용 생략...] " + back_part

def generate_cache_key(model_name: str, prompt: str, text: str) -> str:
    """
    모델 이름, 프롬프트, 텍스트를 기반으로 캐시 키를 생성합니다.
    
    Args:
        model_name (str): 모델 이름
        prompt (str): 프롬프트
        text (str): 분석할 텍스트
        
    Returns:
        str: 캐시 키
    """
    # 입력 데이터를 결합하여 해시 생성
    combined = f"{model_name}:{prompt}:{text}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def get_from_cache(cache_key: str) -> Optional[str]:
    """
    ChromaDB 캐시에서 결과를 가져옵니다.
    
    Args:
        cache_key (str): 캐시 키
        
    Returns:
        str or None: 캐시된 결과 또는 None
    """
    try:
        client = get_chroma_client()
        if not client:
            print("ChromaDB 클라이언트가 없어 캐시 조회를 건너뜁니다.")
            return None
        
        try:
            # 컬렉션이 존재하는지 확인
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                print(f"캐시 컬렉션 '{COLLECTION_NAME}' 접근 성공")
            except Exception as coll_error:
                print(f"캐시 컬렉션이 존재하지 않음: {str(coll_error)}")
                return None
            
            # 캐시 항목 조회
            try:
                results = collection.get(
                    ids=[cache_key], 
                    include=["metadatas", "documents"]
                )
                print(f"캐시 조회 성공: {cache_key[:10]}...")
                
                if results and len(results["ids"]) > 0 and len(results["documents"]) > 0:
                    # 캐시 만료 시간 확인 (24시간)
                    metadata = results["metadatas"][0]
                    timestamp = metadata.get("timestamp", 0)
                    
                    if time.time() - timestamp < 86400:  # 24시간 이내
                        print("유효한 캐시 항목 발견")
                        return results["documents"][0]
                    else:
                        print("캐시 항목이 만료됨")
                else:
                    print("캐시 항목이 없음")
            except Exception as get_error:
                print(f"캐시 항목 조회 오류: {str(get_error)}")
                return None
            
        except Exception as e:
            print(f"컬렉션 조회 오류: {str(e)}")
            return None
        
        return None
    except Exception as e:
        print(f"캐시 읽기 오류: {str(e)}")
        return None

def save_to_cache(cache_key: str, content: str) -> None:
    """
    결과를 ChromaDB 캐시에 저장합니다.
    
    Args:
        cache_key (str): 캐시 키
        content (str): 저장할 내용
    """
    # 내용이 없으면 저장하지 않음
    if not content:
        print("저장할 내용이 없어 캐싱을 건너뜁니다.")
        return
    
    try:
        client = get_chroma_client()
        if not client:
            print("ChromaDB 클라이언트가 없어 캐싱을 건너뜁니다.")
            return
        
        try:
            # 컬렉션 가져오기 시도
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                print(f"기존 컬렉션 '{COLLECTION_NAME}' 가져오기 성공")
            except Exception as get_error:
                print(f"컬렉션 가져오기 실패: {str(get_error)}, 새 컬렉션 생성 시도")
                collection = client.create_collection(name=COLLECTION_NAME)
                print(f"새 컬렉션 '{COLLECTION_NAME}' 생성 성공")
            
            # 기존 항목이 있으면 삭제
            try:
                collection.delete(ids=[cache_key])
                print(f"기존 캐시 항목 삭제: {cache_key}")
            except Exception as del_error:
                print(f"캐시 항목 삭제 실패 (무시됨): {str(del_error)}")
            
            # 새 항목 추가
            # 임베딩 없이 저장하기 위해 embeddings 매개변수를 명시적으로 None으로 설정
            collection.add(
                ids=[cache_key],
                documents=[content],
                metadatas=[{"timestamp": time.time()}],
                embeddings=None
            )
            print(f"캐시 항목 저장 성공: {cache_key[:10]}...")
            
        except Exception as e:
            print(f"컬렉션 저장 오류: {str(e)}")
            
    except Exception as e:
        print(f"캐시 저장 오류: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    텍스트를 적절한 크기의 청크로 분할합니다.
    
    Args:
        text (str): 분할할 텍스트
        chunk_size (int): 각 청크의 최대 크기
        
    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    # 텍스트가 충분히 짧으면 분할하지 않음
    if len(text) <= chunk_size:
        return [text]
    
    # 문장 단위로 분할 (마침표, 물음표, 느낌표 기준)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 현재 청크에 문장을 추가했을 때 최대 크기를 초과하는지 확인
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            # 현재 청크가 있으면 저장하고 새 청크 시작
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 새 문장이 청크 크기보다 크면 강제로 분할
            if len(sentence) > chunk_size:
                # 단어 단위로 분할
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk += word + " "
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = word + " "
            else:
                current_chunk = sentence + " "
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def call_ollama_with_timeout(model_name: str, prompt: str, row_text: str, timeout: int = 30, use_cache: bool = True, max_text_length: int = 4000) -> str:
    """
    Call Ollama API with a timeout and caching
    
    Args:
        model_name (str): Name of the model to use
        prompt (str): The prompt to send
        row_text (str): The text data to analyze
        timeout (int): Timeout in seconds
        use_cache (bool): Whether to use caching
        max_text_length (int): Maximum text length
        
    Returns:
        str: Response content or error message
    """
    try:
        # 텍스트 길이 확인 및 제한
        if len(row_text) > max_text_length:
            print(f"텍스트가 너무 깁니다 ({len(row_text)} 문자). 청킹을 수행합니다.")
            
            # 텍스트를 청크로 분할
            chunks = chunk_text(row_text, chunk_size=max_text_length//2)
            
            # 첫 번째 청크만 사용 (간단한 접근법)
            compressed_text = compress_text(chunks[0], max_length=max_text_length)
            print(f"원본 텍스트 ({len(row_text)} 문자)를 {len(compressed_text)} 문자로 압축했습니다.")
        else:
            # 텍스트가 충분히 짧으면 그냥 압축
            compressed_text = compress_text(row_text, max_length=max_text_length)
        
        # 캐싱 사용 시 캐시 확인
        cache_key = None
        if use_cache:
            try:
                cache_key = generate_cache_key(model_name, prompt, compressed_text)
                cached_result = get_from_cache(cache_key)
                if cached_result:
                    print("캐시에서 결과를 찾았습니다.")
                    return cached_result
            except Exception as cache_error:
                print(f"캐시 조회 오류: {str(cache_error)}")
                # 캐시 오류가 발생해도 API 호출은 계속 진행
        
        result = {"content": None, "error": None}
        
        def api_call():
            try:
                # 스트리밍 API 호출로 변경
                response_content = []
                
                try:
                    # 스트리밍 응답 처리
                    for chunk in ollama.chat(
                        model=model_name,
                        messages=[
                            {
                                'role': 'user',
                                'content': f"{prompt}: {compressed_text}"
                            }
                        ],
                        stream=True
                    ):
                        if 'message' in chunk and 'content' in chunk['message']:
                            response_content.append(chunk['message']['content'])
                    
                    # 모든 응답 청크 결합
                    result["content"] = "".join(response_content)
                except Exception as stream_error:
                    print(f"스트리밍 오류, 일반 API로 대체: {str(stream_error)}")
                    
                    # 스트리밍 실패 시 일반 API 호출로 폴백
                    response = ollama.chat(
                        model=model_name,
                        messages=[
                            {
                                'role': 'user',
                                'content': f"{prompt}: {compressed_text}"
                            }
                        ]
                    )
                    if 'message' in response and 'content' in response['message']:
                        result["content"] = response['message']['content']
                    else:
                        result["error"] = "API 응답 형식 오류"
            except Exception as e:
                result["error"] = str(e)
        
        # Create and start the thread
        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete or timeout
        thread.join(timeout)
        
        # Check if the thread is still alive (timeout occurred)
        if thread.is_alive():
            return f"Error: API call timed out after {timeout} seconds"
        
        # Return the result or error
        if result["error"]:
            return f"Error: {result['error']}"
        
        # 결과 캐싱
        if use_cache and result["content"] and cache_key:
            try:
                save_to_cache(cache_key, result["content"])
            except Exception as cache_error:
                print(f"캐시 저장 오류: {str(cache_error)}")
                # 캐시 저장 오류는 무시하고 결과 반환
        
        return result["content"]
    
    except Exception as e:
        print(f"API 호출 처리 중 예외 발생: {str(e)}")
        return f"Error: Exception during API call processing: {str(e)}"

def process_batch(batch_df: pd.DataFrame, prompt: str, model_name: str, timeout: int, 
                 max_retries: int, progress_callback: Optional[Callable] = None, 
                 start_idx: int = 0, total_rows: int = 0, use_cache: bool = True,
                 max_text_length: int = 4000) -> pd.DataFrame:
    """
    배치 단위로 데이터를 처리합니다.
    
    Args:
        batch_df (pandas.DataFrame): 처리할 배치 데이터프레임
        prompt (str): 프롬프트
        model_name (str): 모델 이름
        timeout (int): 타임아웃 시간
        max_retries (int): 최대 재시도 횟수
        progress_callback (function): 진행 상황 콜백 함수
        start_idx (int): 시작 인덱스
        total_rows (int): 전체 행 수
        use_cache (bool): 캐시 사용 여부
        max_text_length (int): 최대 텍스트 길이
        
    Returns:
        pandas.DataFrame: 처리된 데이터프레임
    """
    try:
        result_df = batch_df.copy()
        result_df['response'] = None
        
        for idx, row in result_df.iterrows():
            # 텍스트 데이터 준비 - 컬럼별 최대 길이 제한
            try:
                row_text_parts = []
                for col in result_df.columns:
                    if col != 'response':
                        # None 값 처리
                        if pd.isna(row[col]):
                            value = ""
                        else:
                            try:
                                value = str(row[col])[:300]
                            except Exception:
                                value = "변환 불가 데이터"
                        row_text_parts.append(f"{col}: {value}")
                row_text = ' '.join(row_text_parts)
            except Exception as e:
                print(f"행 텍스트 생성 오류 (인덱스 {idx}): {str(e)}")
                row_text = "데이터 변환 오류"
            
            # 재시도 로직
            retry_count = 0
            response_content = None
            
            while retry_count < max_retries:
                try:
                    # API 호출 간 간격 두기
                    if idx > start_idx or retry_count > 0:
                        time.sleep(0.5)
                    
                    # 진행 상황 업데이트
                    if progress_callback:
                        current_idx = start_idx + (idx - batch_df.index[0])
                        progress_callback(min(1.0, current_idx / total_rows) if total_rows > 0 else 0, 
                                        f"처리 중: {current_idx}/{total_rows} 행")
                    
                    # Ollama API 호출
                    response_content = call_ollama_with_timeout(
                        model_name, prompt, row_text, timeout, use_cache=use_cache,
                        max_text_length=max_text_length
                    )
                    
                    # 오류가 아니면 종료
                    if response_content and not response_content.startswith("Error:"):
                        break
                    
                    # 오류 발생 시 재시도
                    retry_count += 1
                    print(f"API 호출 오류 (인덱스 {idx}, 재시도 {retry_count}/{max_retries}): {response_content}")
                    
                    if retry_count < max_retries:
                        if progress_callback:
                            current_idx = start_idx + (idx - batch_df.index[0])
                            progress_callback(min(1.0, current_idx / total_rows) if total_rows > 0 else 0, 
                                            f"처리 중: {current_idx}/{total_rows} 행 (재시도 {retry_count}/{max_retries})")
                        time.sleep(1)  # 재시도 전 대기
                except Exception as e:
                    retry_count += 1
                    print(f"API 호출 예외 (인덱스 {idx}, 재시도 {retry_count}/{max_retries}): {str(e)}")
                    response_content = f"Error: {str(e)}"
                    
                    if retry_count < max_retries:
                        if progress_callback:
                            current_idx = start_idx + (idx - batch_df.index[0])
                            progress_callback(min(1.0, current_idx / total_rows) if total_rows > 0 else 0, 
                                            f"처리 중: {current_idx}/{total_rows} 행 (재시도 {retry_count}/{max_retries})")
                        time.sleep(1)
            
            # 최종 응답 저장
            result_df.at[idx, 'response'] = response_content if response_content else f"Error: Failed after {max_retries} retries"
            
            # 진행 상황 업데이트
            if progress_callback:
                current_idx = start_idx + (idx - batch_df.index[0] + 1)
                progress_callback(min(1.0, current_idx / total_rows) if total_rows > 0 else 0, 
                                f"처리 중: {current_idx}/{total_rows} 행")
        
        return result_df
    
    except Exception as e:
        print(f"배치 처리 중 예외 발생: {str(e)}")
        # 배치 처리 실패 시 오류 메시지 추가
        error_df = batch_df.copy()
        error_df['response'] = f"Error: Batch processing exception: {str(e)}"
        return error_df

def analyze_text_with_prompt(prompt: str, df: pd.DataFrame, model_name: str = "exaone3.5:7.8b", 
                            progress_callback: Optional[Callable] = None, batch_size: int = 10, 
                            max_retries: int = 3, timeout: int = 30, max_workers: int = 2, 
                            use_cache: bool = True, max_text_length: int = 4000) -> pd.DataFrame:
    """
    Analyze text data in a dataframe using Ollama and add responses as a new column.
    
    Args:
        prompt (str): The prompt to send to Ollama
        df (pandas.DataFrame): The dataframe containing the text data to analyze
        model_name (str): Name of the Ollama model to use
        progress_callback (function): Optional callback function to report progress
        batch_size (int): Number of rows to process in each batch
        max_retries (int): Maximum number of retries for failed API calls
        timeout (int): Timeout in seconds for each API call
        max_workers (int): Maximum number of parallel workers
        use_cache (bool): Whether to use caching
        max_text_length (int): Maximum text length
        
    Returns:
        pandas.DataFrame: The original dataframe with an additional 'response' column
    """
    # 결과를 저장할 빈 데이터프레임 생성
    result_df = pd.DataFrame()
    total_rows = len(df)
    
    # 병렬 처리를 위한 배치 생성
    batches = []
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batches.append((start, df.iloc[start:end]))
    
    # 병렬 처리 실행 - 오류 처리 강화
    try:
        # 병렬 처리 작업자 수가 1보다 크면 ThreadPoolExecutor 사용
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 각 배치에 대한 작업 제출
                future_to_batch = {}
                batch_results = []
                
                for start_idx, batch_df in batches:
                    try:
                        future = executor.submit(
                            process_batch, 
                            batch_df, 
                            prompt, 
                            model_name, 
                            timeout, 
                            max_retries, 
                            progress_callback, 
                            start_idx, 
                            total_rows,
                            use_cache,
                            max_text_length
                        )
                        future_to_batch[future] = (start_idx, batch_df)
                    except Exception as submit_error:
                        print(f"작업 제출 오류 (인덱스 {start_idx}): {str(submit_error)}")
                        # 오류 발생 시 빈 결과 생성
                        empty_df = batch_df.copy()
                        empty_df['response'] = f"Error: Job submission failed: {str(submit_error)}"
                        batch_results.append((start_idx, empty_df))
                
                # 결과 수집
                for future in concurrent.futures.as_completed(future_to_batch):
                    start_idx, batch_df = future_to_batch[future]
                    try:
                        batch_result = future.result(timeout=timeout+10)  # 약간의 여유 시간 추가
                        batch_results.append((start_idx, batch_result))
                    except concurrent.futures.TimeoutError:
                        print(f"배치 처리 타임아웃: 시작 인덱스 {start_idx}")
                        empty_df = batch_df.copy()
                        empty_df['response'] = "Error: Batch processing timed out"
                        batch_results.append((start_idx, empty_df))
                    except Exception as e:
                        print(f"배치 처리 오류 (인덱스 {start_idx}): {str(e)}")
                        # 오류 발생 시 빈 결과 생성
                        empty_df = batch_df.copy()
                        empty_df['response'] = f"Error: Batch processing failed: {str(e)}"
                        batch_results.append((start_idx, empty_df))
        else:
            # 병렬 처리 없이 순차적으로 처리
            batch_results = []
            for start_idx, batch_df in batches:
                try:
                    if progress_callback:
                        progress_callback(start_idx / total_rows if total_rows > 0 else 0, 
                                         f"배치 처리 중: {start_idx}/{total_rows}")
                    
                    batch_result = process_batch(
                        batch_df, 
                        prompt, 
                        model_name, 
                        timeout, 
                        max_retries, 
                        progress_callback, 
                        start_idx, 
                        total_rows,
                        use_cache,
                        max_text_length
                    )
                    batch_results.append((start_idx, batch_result))
                except Exception as e:
                    print(f"배치 처리 오류 (순차 처리, 인덱스 {start_idx}): {str(e)}")
                    # 오류 발생 시 빈 결과 생성
                    empty_df = batch_df.copy()
                    empty_df['response'] = f"Error: Sequential batch processing failed: {str(e)}"
                    batch_results.append((start_idx, empty_df))
        
        # 결과 정렬 및 병합
        batch_results.sort(key=lambda x: x[0])
        for _, batch_df in batch_results:
            result_df = pd.concat([result_df, batch_df], ignore_index=True)
        
    except Exception as e:
        print(f"전체 분석 처리 오류: {str(e)}")
        # 전체 처리 실패 시 원본 데이터프레임에 오류 메시지 추가
        result_df = df.copy()
        result_df['response'] = f"Error: Analysis failed: {str(e)}"
    
    # 최종 진행 상황 업데이트
    if progress_callback:
        progress_callback(1.0, "분석 완료!")
    
    return result_df