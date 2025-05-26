"""
임베딩 모델 관련 유틸리티 함수 모음
"""
import ssl
import warnings
import requests
import numpy as np
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Union, Optional
from chromadb.utils import embedding_functions

# SSL 인증서 검증 우회 옵션 (기본값은 검증함)
ssl._create_default_https_context_backup = ssl._create_default_https_context
VERIFY_SSL = True

# 임베딩 모델 로드 상태를 추적하기 위한 변수
EMBEDDING_MODEL_STATUS = {
    "requested_model": None,  # 요청한 임베딩 모델
    "actual_model": None,     # 실제 사용중인 임베딩 모델
    "fallback_used": False,   # 폴백(기본 모델)을 사용했는지 여부
    "error_message": None     # 오류 메시지 (있는 경우)
}

# 임베딩 캐시 사전 설정 (메모리 효율성 향상)
EMBEDDING_CACHE = {}
CACHE_HITS = 0
CACHE_MISSES = 0
MAX_CACHE_SIZE = 10000  # 최대 캐시 크기

def reset_embedding_status():
    """임베딩 모델 상태를 초기화합니다."""
    global EMBEDDING_MODEL_STATUS
    EMBEDDING_MODEL_STATUS = {
        "requested_model": None,
        "actual_model": None,
        "fallback_used": False,
        "error_message": None
    }
    
def get_embedding_status():
    """현재 임베딩 모델 상태를 반환합니다."""
    return EMBEDDING_MODEL_STATUS.copy()

def set_ssl_verification(verify=True):
    """
    SSL 인증서 검증 여부를 설정합니다.
    
    Args:
        verify (bool): SSL 인증서 검증 여부 (True: 검증함, False: 검증하지 않음)
    """
    global VERIFY_SSL
    VERIFY_SSL = verify
    
    if verify:
        # 기본 SSL 컨텍스트 복원 (SSL 인증서 검증 활성화)
        ssl._create_default_https_context = ssl._create_default_https_context_backup
        # requests 라이브러리 설정
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    else:
        # SSL 인증서 검증을 우회하는 컨텍스트 설정
        ssl._create_default_https_context = ssl._create_unverified_context
        # requests 라이브러리의 SSL 검증 경고 무시
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

def clear_embedding_cache():
    """임베딩 캐시를 초기화합니다."""
    global EMBEDDING_CACHE, CACHE_HITS, CACHE_MISSES
    EMBEDDING_CACHE = {}
    CACHE_HITS = 0
    CACHE_MISSES = 0
    print(f"임베딩 캐시가 초기화되었습니다.")

def get_cache_stats():
    """임베딩 캐시 통계를 반환합니다."""
    global CACHE_HITS, CACHE_MISSES
    total = CACHE_HITS + CACHE_MISSES
    hit_ratio = 0 if total == 0 else (CACHE_HITS / total) * 100
    return {
        "hits": CACHE_HITS,
        "misses": CACHE_MISSES, 
        "total_lookups": total,
        "hit_ratio": hit_ratio,
        "cache_size": len(EMBEDDING_CACHE)
    }

# 텍스트 해싱 함수 (캐시 키로 사용)
def _hash_text(text: str) -> str:
    """
    텍스트를 해시값으로 변환합니다. 캐시 키로 사용됩니다.
    
    Args:
        text: 해시할 텍스트
        
    Returns:
        str: 해시된 텍스트 값
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_embedding_function(embedding_model="all-MiniLM-L6-v2", use_cache=True):
    """
    임베딩 함수를 생성합니다. 캐싱 기능이 포함되어 있습니다.
    
    Args:
        embedding_model (str): 임베딩 모델 이름
        use_cache (bool): 캐시 사용 여부
        
    Returns:
        embedding_function: 임베딩 함수
    """
    # 임베딩 상태 초기화
    global EMBEDDING_MODEL_STATUS, EMBEDDING_CACHE, CACHE_HITS, CACHE_MISSES
    reset_embedding_status()
    EMBEDDING_MODEL_STATUS["requested_model"] = embedding_model
    
    # 임베딩 함수 설정
    if embedding_model == "default" or embedding_model is None:
        # 기본 임베딩 모델(all-MiniLM-L6-v2) 명시적으로 사용
        try:
            print("기본 임베딩 모델(all-MiniLM-L6-v2)을 로드합니다.")
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            EMBEDDING_MODEL_STATUS["actual_model"] = "all-MiniLM-L6-v2"
            EMBEDDING_MODEL_STATUS["fallback_used"] = False
        except Exception as e:
            print(f"기본 임베딩 모델 로드 중 오류 발생: {e}")
            print("내장 기본 임베딩 함수를 사용합니다.")
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
            EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
            EMBEDDING_MODEL_STATUS["fallback_used"] = True
            EMBEDDING_MODEL_STATUS["error_message"] = str(e)
    else:
        # 먼저 SSL 검증 활성화 상태로 시도
        try:
            print(f"임베딩 모델 '{embedding_model}' 로드 시도 중...")
            # sentence-transformers 기반 임베딩 함수 생성
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            print(f"임베딩 모델 '{embedding_model}' 로드 성공!")
            EMBEDDING_MODEL_STATUS["actual_model"] = embedding_model
            EMBEDDING_MODEL_STATUS["fallback_used"] = False
        except Exception as e:
            print(f"임베딩 모델 '{embedding_model}' 로드 중 오류 발생: {e}")
            error_msg = str(e)
            
            # SSL 오류인 경우 검증 우회 시도
            if "CERTIFICATE_VERIFY_FAILED" in error_msg or "SSL" in error_msg:
                print("SSL 인증서 검증 실패, 검증을 우회하여 다시 시도합니다...")
                
                # SSL 인증서 검증 비활성화
                old_verify = VERIFY_SSL
                set_ssl_verification(False)
                
                try:
                    # 검증 우회 모드로 다시 시도
                    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=embedding_model
                    )
                    print(f"SSL 검증 우회 모드로 '{embedding_model}' 임베딩 모델을 성공적으로 로드했습니다.")
                    EMBEDDING_MODEL_STATUS["actual_model"] = embedding_model
                    EMBEDDING_MODEL_STATUS["fallback_used"] = False
                except Exception as inner_e:
                    inner_error_msg = str(inner_e)
                    print(f"SSL 검증을 우회해도 임베딩 모델 로드 실패: {inner_error_msg}")
                    
                    # 'unrecognized model' 오류 또는 'No model with name' 오류인 경우 상세 정보 출력
                    if "unrecognized model" in inner_error_msg.lower() or "no model with name" in inner_error_msg.lower():
                        print(f"모델 '{embedding_model}'이(가) 인식되지 않습니다. 해당 모델이 Hugging Face에 존재하는지 확인하세요.")
                        print("대신 기본 임베딩 모델(all-MiniLM-L6-v2)을 사용합니다.")
                        
                        try:
                            # 기본 모델로 대체
                            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                                model_name="all-MiniLM-L6-v2"
                            )
                            EMBEDDING_MODEL_STATUS["actual_model"] = "all-MiniLM-L6-v2"
                            EMBEDDING_MODEL_STATUS["fallback_used"] = True
                            EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않아 기본 모델로 대체됨"
                        except Exception:
                            print("기본 임베딩 모델도 로드 실패, 내장 기본 임베딩 함수를 사용합니다.")
                            embedding_function = embedding_functions.DefaultEmbeddingFunction()
                            EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                            EMBEDDING_MODEL_STATUS["fallback_used"] = True
                            EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않고, 기본 모델 로드도 실패함"
                    else:
                        print("내장 기본 임베딩 함수를 사용합니다.")
                        embedding_function = embedding_functions.DefaultEmbeddingFunction()
                        EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                        EMBEDDING_MODEL_STATUS["fallback_used"] = True
                        EMBEDDING_MODEL_STATUS["error_message"] = inner_error_msg
                
                # SSL 인증서 검증 상태 복원
                set_ssl_verification(old_verify)
            else:
                # 'unrecognized model' 오류 또는 'No model with name' 오류인 경우 상세 정보 출력
                if "unrecognized model" in error_msg.lower() or "no model with name" in error_msg.lower():
                    print(f"모델 '{embedding_model}'이(가) 인식되지 않습니다. 해당 모델이 Hugging Face에 존재하는지 확인하세요.")
                    print("대신 기본 임베딩 모델(all-MiniLM-L6-v2)을 사용합니다.")
                    
                    try:
                        # 기본 모델로 대체
                        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name="all-MiniLM-L6-v2"
                        )
                        EMBEDDING_MODEL_STATUS["actual_model"] = "all-MiniLM-L6-v2"
                        EMBEDDING_MODEL_STATUS["fallback_used"] = True
                        EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않아 기본 모델로 대체됨"
                    except Exception:
                        print("기본 임베딩 모델도 로드 실패, 내장 기본 임베딩 함수를 사용합니다.")
                        embedding_function = embedding_functions.DefaultEmbeddingFunction()
                        EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                        EMBEDDING_MODEL_STATUS["fallback_used"] = True
                        EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않고, 기본 모델 로드도 실패함"
                else:
                    print("내장 기본 임베딩 함수를 사용합니다.")
                    embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                    EMBEDDING_MODEL_STATUS["fallback_used"] = True
                    EMBEDDING_MODEL_STATUS["error_message"] = error_msg
    
    # 기본 임베딩 함수
    base_embedding_function = _get_base_embedding_function(embedding_model)
    
    # 캐싱 기능이 활성화된 경우 래퍼 함수 생성
    if use_cache:
        def cached_embedding_function(texts: List[str]) -> List[List[float]]:
            """
            캐싱 기능이 적용된 임베딩 함수 래퍼
            
            Args:
                texts: 임베딩할 텍스트 목록
                
            Returns:
                List[List[float]]: 임베딩 벡터 목록
            """
            global CACHE_HITS, CACHE_MISSES, EMBEDDING_CACHE
            
            # 캐시에서 히트한 텍스트와 미스한 텍스트 분리
            cached_vectors = {}
            texts_to_embed = []
            text_positions = {}
            
            # 캐시 확인
            for i, text in enumerate(texts):
                text_hash = _hash_text(text)
                
                if text_hash in EMBEDDING_CACHE:
                    # 캐시 히트
                    cached_vectors[i] = EMBEDDING_CACHE[text_hash]
                    CACHE_HITS += 1
                else:
                    # 캐시 미스
                    texts_to_embed.append(text)
                    text_positions[len(texts_to_embed) - 1] = i
                    CACHE_MISSES += 1
            
            # 캐시되지 않은 텍스트 임베딩
            if texts_to_embed:
                new_embeddings = base_embedding_function(texts_to_embed)
                
                # 새로운 임베딩 캐싱 및 결과에 추가
                for i, embedding in enumerate(new_embeddings):
                    orig_index = text_positions[i]
                    text = texts[orig_index]
                    text_hash = _hash_text(text)
                    
                    # 캐시 크기 제한 확인
                    if len(EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
                        # 캐시 1/4 비우기
                        keys_to_remove = list(EMBEDDING_CACHE.keys())[:MAX_CACHE_SIZE // 4]
                        for key in keys_to_remove:
                            EMBEDDING_CACHE.pop(key, None)
                    
                    # 새로운 임베딩 캐싱
                    EMBEDDING_CACHE[text_hash] = embedding
                    cached_vectors[orig_index] = embedding
            
            # 원래 순서대로 결과 반환
            result = [cached_vectors[i] for i in range(len(texts))]
            return result
        
        return cached_embedding_function
    else:
        return base_embedding_function

def _get_base_embedding_function(embedding_model="all-MiniLM-L6-v2"):
    """
    기본 임베딩 함수를 생성합니다. (내부 함수)
    
    Args:
        embedding_model (str): 임베딩 모델 이름
        
    Returns:
        function: 기본 임베딩 함수
    """
    global EMBEDDING_MODEL_STATUS
    
    # 임베딩 함수 설정
    if embedding_model == "default" or embedding_model is None:
        # 기본 임베딩 모델(all-MiniLM-L6-v2) 명시적으로 사용
        try:
            print("기본 임베딩 모델(all-MiniLM-L6-v2)을 로드합니다.")
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            EMBEDDING_MODEL_STATUS["actual_model"] = "all-MiniLM-L6-v2"
            EMBEDDING_MODEL_STATUS["fallback_used"] = False
        except Exception as e:
            print(f"기본 임베딩 모델 로드 중 오류 발생: {e}")
            print("내장 기본 임베딩 함수를 사용합니다.")
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
            EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
            EMBEDDING_MODEL_STATUS["fallback_used"] = True
            EMBEDDING_MODEL_STATUS["error_message"] = str(e)
    else:
        # 먼저 SSL 검증 활성화 상태로 시도
        try:
            print(f"임베딩 모델 '{embedding_model}' 로드 시도 중...")
            # sentence-transformers 기반 임베딩 함수 생성
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            print(f"임베딩 모델 '{embedding_model}' 로드 성공!")
            EMBEDDING_MODEL_STATUS["actual_model"] = embedding_model
            EMBEDDING_MODEL_STATUS["fallback_used"] = False
        except Exception as e:
            print(f"임베딩 모델 '{embedding_model}' 로드 중 오류 발생: {e}")
            error_msg = str(e)
            
            # SSL 오류인 경우 검증 우회 시도
            if "CERTIFICATE_VERIFY_FAILED" in error_msg or "SSL" in error_msg:
                print("SSL 인증서 검증 실패, 검증을 우회하여 다시 시도합니다...")
                
                # SSL 인증서 검증 비활성화
                old_verify = VERIFY_SSL
                set_ssl_verification(False)
                
                try:
                    # 검증 우회 모드로 다시 시도
                    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=embedding_model
                    )
                    print(f"SSL 검증 우회 모드로 '{embedding_model}' 임베딩 모델을 성공적으로 로드했습니다.")
                    EMBEDDING_MODEL_STATUS["actual_model"] = embedding_model
                    EMBEDDING_MODEL_STATUS["fallback_used"] = False
                except Exception as inner_e:
                    inner_error_msg = str(inner_e)
                    print(f"SSL 검증을 우회해도 임베딩 모델 로드 실패: {inner_error_msg}")
                    
                    # 'unrecognized model' 오류 또는 'No model with name' 오류인 경우 상세 정보 출력
                    if "unrecognized model" in inner_error_msg.lower() or "no model with name" in inner_error_msg.lower():
                        print(f"모델 '{embedding_model}'이(가) 인식되지 않습니다. 해당 모델이 Hugging Face에 존재하는지 확인하세요.")
                        print("대신 기본 임베딩 모델(all-MiniLM-L6-v2)을 사용합니다.")
                        
                        try:
                            # 기본 모델로 대체
                            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                                model_name="all-MiniLM-L6-v2"
                            )
                            EMBEDDING_MODEL_STATUS["actual_model"] = "all-MiniLM-L6-v2"
                            EMBEDDING_MODEL_STATUS["fallback_used"] = True
                            EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않아 기본 모델로 대체됨"
                        except Exception:
                            print("기본 임베딩 모델도 로드 실패, 내장 기본 임베딩 함수를 사용합니다.")
                            embedding_function = embedding_functions.DefaultEmbeddingFunction()
                            EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                            EMBEDDING_MODEL_STATUS["fallback_used"] = True
                            EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않고, 기본 모델 로드도 실패함"
                    else:
                        print("내장 기본 임베딩 함수를 사용합니다.")
                        embedding_function = embedding_functions.DefaultEmbeddingFunction()
                        EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                        EMBEDDING_MODEL_STATUS["fallback_used"] = True
                        EMBEDDING_MODEL_STATUS["error_message"] = inner_error_msg
                
                # SSL 인증서 검증 상태 복원
                set_ssl_verification(old_verify)
            else:
                # 'unrecognized model' 오류 또는 'No model with name' 오류인 경우 상세 정보 출력
                if "unrecognized model" in error_msg.lower() or "no model with name" in error_msg.lower():
                    print(f"모델 '{embedding_model}'이(가) 인식되지 않습니다. 해당 모델이 Hugging Face에 존재하는지 확인하세요.")
                    print("대신 기본 임베딩 모델(all-MiniLM-L6-v2)을 사용합니다.")
                    
                    try:
                        # 기본 모델로 대체
                        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name="all-MiniLM-L6-v2"
                        )
                        EMBEDDING_MODEL_STATUS["actual_model"] = "all-MiniLM-L6-v2"
                        EMBEDDING_MODEL_STATUS["fallback_used"] = True
                        EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않아 기본 모델로 대체됨"
                    except Exception:
                        print("기본 임베딩 모델도 로드 실패, 내장 기본 임베딩 함수를 사용합니다.")
                        embedding_function = embedding_functions.DefaultEmbeddingFunction()
                        EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                        EMBEDDING_MODEL_STATUS["fallback_used"] = True
                        EMBEDDING_MODEL_STATUS["error_message"] = f"모델 '{embedding_model}'이(가) 인식되지 않고, 기본 모델 로드도 실패함"
                else:
                    print("내장 기본 임베딩 함수를 사용합니다.")
                    embedding_function = embedding_functions.DefaultEmbeddingFunction()
                    EMBEDDING_MODEL_STATUS["actual_model"] = "default_embedding_function"
                    EMBEDDING_MODEL_STATUS["fallback_used"] = True
                    EMBEDDING_MODEL_STATUS["error_message"] = error_msg
    
    return embedding_function

class L2NormalizedEmbeddingFunction:
    """
    L2 정규화를 적용한 임베딩 함수 래퍼 클래스
    
    기본 임베딩 함수에서 생성된 벡터에 L2 정규화를 적용하여 유사도 계산의 일관성을 높입니다.
    """
    
    def __init__(self, base_embedding_function, use_cache=True):
        """
        L2 정규화 임베딩 함수 초기화
        
        Args:
            base_embedding_function: 기본 임베딩 함수
            use_cache: 캐싱 사용 여부
        """
        self.base_embedding_function = base_embedding_function
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        
    def __call__(self, input):
        """
        텍스트를 임베딩 벡터로 변환하고 L2 정규화를 적용합니다.
        
        Args:
            input: 임베딩할 텍스트 목록
            
        Returns:
            list: L2 정규화된 임베딩 벡터 목록
        """
        # 캐싱 기능 사용 시 캐시 확인
        if self.use_cache:
            global CACHE_HITS, CACHE_MISSES
            cached_vectors = {}
            texts_to_embed = []
            text_positions = {}
            
            # 캐시 확인
            for i, text in enumerate(input):
                text_hash = _hash_text(text)
                
                if text_hash in EMBEDDING_CACHE:
                    # 캐시 히트
                    cached_vectors[i] = EMBEDDING_CACHE[text_hash]
                    CACHE_HITS += 1
                else:
                    # 캐시 미스
                    texts_to_embed.append(text)
                    text_positions[len(texts_to_embed) - 1] = i
                    CACHE_MISSES += 1
            
            # 캐시되지 않은 텍스트 임베딩
            if texts_to_embed:
                # 기본 임베딩 함수로 벡터 생성
                embeddings = self.base_embedding_function(texts_to_embed)
                
                # L2 정규화 및 캐싱
                for i, emb in enumerate(embeddings):
                    # L2 정규화 적용
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        normalized_emb = emb / norm
                    else:
                        normalized_emb = emb
                    
                    # 원래 인덱스와 텍스트 가져오기
                    orig_index = text_positions[i]
                    text = input[orig_index]
                    text_hash = _hash_text(text)
                    
                    # 캐시 크기 제한 확인
                    if len(EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
                        # 캐시 1/4 비우기
                        keys_to_remove = list(EMBEDDING_CACHE.keys())[:MAX_CACHE_SIZE // 4]
                        for key in keys_to_remove:
                            EMBEDDING_CACHE.pop(key, None)
                    
                    # 캐싱 및 결과 저장
                    EMBEDDING_CACHE[text_hash] = normalized_emb
                    cached_vectors[orig_index] = normalized_emb
            
            # 원래 순서대로 결과 반환
            return [cached_vectors[i] for i in range(len(input))]
        else:
            # 캐싱 없이 임베딩 생성 및 정규화
            embeddings = self.base_embedding_function(input)
            
            # L2 정규화 적용
            normalized_embeddings = []
            for emb in embeddings:
                # L2 노름(벡터 크기) 계산
                norm = np.linalg.norm(emb)
                # 0으로 나누기 방지
                if norm > 0:
                    normalized_emb = emb / norm
                else:
                    normalized_emb = emb
                normalized_embeddings.append(normalized_emb)
                
            return normalized_embeddings

def get_normalized_embedding_function(embedding_model="all-MiniLM-L6-v2", use_cache=True):
    """
    L2 정규화가 적용된 임베딩 함수를 생성합니다.
    
    Args:
        embedding_model (str): 임베딩 모델 이름
        use_cache (bool): 캐시 사용 여부
        
    Returns:
        L2NormalizedEmbeddingFunction: L2 정규화된 임베딩 함수
    """
    # 기본 임베딩 함수 생성
    base_function = get_embedding_function(embedding_model, use_cache=False)
    
    # L2 정규화 래퍼 적용
    return L2NormalizedEmbeddingFunction(base_function, use_cache=use_cache)

def get_available_embedding_models():
    """
    사용 가능한 임베딩 모델 목록을 반환합니다.
    
    Returns:
        dict: 카테고리별 추천 임베딩 모델
    """
    return {
        "다국어 모델": [
            "all-MiniLM-L6-v2",  # 기본 모델
            "paraphrase-multilingual-MiniLM-L12-v2",  # 다국어 지원 모델
            "distiluse-base-multilingual-cased-v2"  # 다국어 지원 모델 (크기 큼)
        ],
        "한국어 특화 모델": [
            "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
            "jhgan/ko-sbert-sts"
        ],
        "영어 특화 모델": [
            "all-mpnet-base-v2",  # 영어 특화 고성능 모델
            "all-distilroberta-v1",  # 영어 특화 경량 모델
            "all-MiniLM-L12-v2"  # 영어 특화 모델
        ]
    }