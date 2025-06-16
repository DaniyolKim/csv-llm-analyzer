"""
임베딩 모델 관련 유틸리티 함수 모음
"""
import ssl
import warnings
import requests
import torch # GPU 확인을 위해 추가
import numpy as np
import hashlib
from functools import lru_cache
import streamlit as st # Streamlit 캐싱을 위해 추가
from typing import List, Dict, Any, Union, Optional, Callable
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer # For OOM fallback and direct use
import logging

# SSL 인증서 검증 우회 옵션 (기본값은 검증함)
ssl._create_default_https_context_backup = ssl._create_default_https_context
VERIFY_SSL = True

# 임베딩 모델 로드 상태를 추적하기 위한 변수
EMBEDDING_MODEL_STATUS = {
    "requested_model": None,  # 요청한 임베딩 모델
    "actual_model": None,     # 실제 사용중인 임베딩 모델
    "fallback_used": False,   # 폴백(기본 모델)을 사용했는지 여부
    "error_message": None,    # 오류 메시지 (있는 경우)
    "device_preference": None, # 사용자가 요청한 장치
    "device_used": None       # 실제 사용된 장치
}

# 임베딩 캐시 사전 설정 (메모리 효율성 향상)
EMBEDDING_CACHE = {}
CACHE_HITS = 0
CACHE_MISSES = 0
MAX_CACHE_SIZE = 10000  # 최대 캐시 크기

# 로거 설정
logger = logging.getLogger("embedding_utils")

def reset_embedding_status():
    """임베딩 모델 상태를 초기화합니다."""
    global EMBEDDING_MODEL_STATUS
    EMBEDDING_MODEL_STATUS = {
        "requested_model": None,
        "actual_model": None,
        "fallback_used": False,
        "error_message": None,
        "device_preference": None,
        "device_used": None
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

# GPU 관련 함수 추가
def is_gpu_available():
    """CUDA GPU 사용 가능 여부를 확인합니다."""
    return torch.cuda.is_available()

def get_gpu_info():
    """사용 가능한 GPU 정보를 반환합니다."""
    if is_gpu_available():
        gpu_count = torch.cuda.device_count()
        gpus = []
        for i in range(gpu_count):
            gpus.append({
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3), # GB
            })
        return {"available": True, "count": gpu_count, "devices": gpus}
    return {"available": False, "count": 0, "devices": []}

def get_default_embedding_model_name():
    """기본 임베딩 모델 이름을 반환합니다."""
    return "all-MiniLM-L6-v2"

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

def _create_oom_safe_embedding_function(
    underlying_ef: Callable[[List[str]], List[List[float]]],
    model_status_ref: Dict[str, Any],
    model_name_for_fallback: str
) -> Callable[[List[str]], List[List[float]]]:
    """
    기존 임베딩 함수를 감싸 GPU OOM 발생 시 CPU로 폴백하는 기능을 추가합니다.

    Args:
        underlying_ef: 원본 임베딩 함수.
        model_status_ref: EMBEDDING_MODEL_STATUS 딕셔너리에 대한 참조.
        model_name_for_fallback: OOM 발생 시 CPU에서 로드할 모델의 이름.

    Returns:
        OOM 안전 기능이 추가된 임베딩 함수.
    """
    def oom_safe_function(texts_to_embed: List[str]) -> List[List[float]]:
        try:
            return underlying_ef(texts_to_embed)
        except torch.cuda.OutOfMemoryError as e:
            if model_status_ref.get("device_used") == "cuda":
                logger.warning(f"GPU OutOfMemoryError for model '{model_name_for_fallback}': {e}. Attempting batch on CPU.")
                model_status_ref["error_message"] = f"GPU OOM, batch on CPU: {e}" # 에러 메시지 업데이트
                # model_status_ref["last_error_type"] = "GPU_OOM_CPU_FALLBACK" # 상태 업데이트

                try:
                    if model_name_for_fallback == "chromadb_default_ef": # ChromaDB 기본 임베딩은 SentenceTransformer로 재로드 불가
                        logger.error("Cannot create CPU fallback for 'chromadb_default_ef' during OOM.")
                        raise # 원래 OOM 오류 다시 발생
                    
                    logger.info(f"OOM Fallback: Creating temporary CPU model for '{model_name_for_fallback}'.")
                    temp_model_cpu = SentenceTransformer(model_name_for_fallback, device='cpu')
                    embeddings = temp_model_cpu.encode(texts_to_embed).tolist()
                    del temp_model_cpu
                    torch.cuda.empty_cache() # GPU 캐시 정리 시도
                    logger.info(f"OOM Fallback: Successfully processed batch on CPU for '{model_name_for_fallback}'.")
                    return embeddings
                except Exception as cpu_fallback_e:
                    logger.error(f"OOM Fallback: CPU fallback for model '{model_name_for_fallback}' failed: {cpu_fallback_e}", exc_info=True)
                    raise e # CPU 폴백 실패 시 원래 OOM 오류 다시 발생
            else:
                logger.error(f"Caught torch.cuda.OutOfMemoryError, but EMBEDDING_MODEL_STATUS['device_used'] was '{model_status_ref.get('device_used')}'. Re-raising original OOM for model '{model_name_for_fallback}'.")
                raise
        except Exception as other_e:
            logger.error(f"General error during embedding with model '{model_name_for_fallback}': {other_e}", exc_info=True)
            raise
    return oom_safe_function


@st.cache_resource # 임베딩 모델 인스턴스를 캐싱하여 재사용
def get_embedding_function(embedding_model_request="all-MiniLM-L6-v2", use_cache=True, device_preference="auto"):
    """ # noqa: E501
    임베딩 함수를 생성합니다. 캐싱 기능이 포함되어 있습니다. CPU 사용을 기본으로 합니다.
    
    Args:
        embedding_model_request (str): 사용자가 요청한 임베딩 모델 이름
        use_cache (bool): 캐시 사용 여부
        device_preference (str): 사용할 장치 ("auto", "cuda", "cpu")
        
    Returns:
        embedding_function: 임베딩 함수
    """
    global EMBEDDING_MODEL_STATUS, EMBEDDING_CACHE, CACHE_HITS, CACHE_MISSES
    reset_embedding_status()

    # 사용자의 요청에 따라 device_preference가 "auto" 또는 명시적 "cpu"일 경우 CPU를 사용하도록 설정
    # 명시적으로 "cuda"가 요청된 경우에만 GPU 사용 시도
    actual_device_preference = "cpu" # 기본적으로 CPU
    if device_preference == "cuda":
        actual_device_preference = "cuda"

    EMBEDDING_MODEL_STATUS["requested_model"] = embedding_model_request
    EMBEDDING_MODEL_STATUS["device_preference"] = device_preference
    model_to_try = embedding_model_request
    if model_to_try == "default" or model_to_try is None:
        model_to_try = get_default_embedding_model_name()

    # 장치 결정 로직
    model_kwargs = {}
    final_device_to_use = "cpu"  # 최종적으로 사용할 장치, 기본값 CPU
    if actual_device_preference == "cuda":
        if is_gpu_available():
            final_device_to_use = "cuda"
        else:
            print("경고: GPU 사용이 요청되었으나 사용 가능한 CUDA 장치가 없습니다. CPU를 사용합니다.")
    # actual_device_preference가 "cpu"인 경우 final_device_to_use는 이미 "cpu"
    
    model_kwargs['device'] = final_device_to_use
    EMBEDDING_MODEL_STATUS["device_used"] = final_device_to_use
    print(f"임베딩 모델 로드 시도: '{model_to_try}', 장치: '{final_device_to_use}'")

    base_ef = None
    
    # 1차 시도: 요청된 모델 로드
    try:
        # SentenceTransformerEmbeddingFunction에 device 인자를 직접 전달하도록 수정
        base_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_to_try, device=final_device_to_use)
        EMBEDDING_MODEL_STATUS["actual_model"] = model_to_try
        print(f"임베딩 모델 '{model_to_try}' 로드 성공 (장치: {final_device_to_use}).")
    except Exception as e:
        print(f"임베딩 모델 '{model_to_try}' 로드 중 오류 발생 (1차 시도): {e}")
        EMBEDDING_MODEL_STATUS["error_message"] = str(e)
        
        # SSL 오류인 경우, SSL 검증 우회 후 재시도
        if "CERTIFICATE_VERIFY_FAILED" in str(e) or "SSL" in str(e):
            print("SSL 인증서 검증 실패, 검증을 우회하여 다시 시도합니다...")
            old_verify_ssl = VERIFY_SSL
            set_ssl_verification(False)
            try:
                # SentenceTransformerEmbeddingFunction에 device 인자를 직접 전달하도록 수정
                base_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_to_try, device=final_device_to_use)
                EMBEDDING_MODEL_STATUS["actual_model"] = model_to_try
                print(f"SSL 검증 우회 모드로 '{model_to_try}' 임베딩 모델 로드 성공 (장치: {final_device_to_use}).")
                EMBEDDING_MODEL_STATUS["error_message"] = None # 성공했으므로 이전 오류 메시지 제거
            except Exception as inner_e:
                print(f"SSL 검증 우회 후에도 임베딩 모델 '{model_to_try}' 로드 실패: {inner_e}")
                EMBEDDING_MODEL_STATUS["error_message"] = str(inner_e) # 새 오류 메시지로 업데이트
            finally:
                set_ssl_verification(old_verify_ssl) # SSL 설정 복원

        # 1차 시도 (또는 SSL 우회 시도) 실패 시 기본 모델로 폴백
        if base_ef is None:
            print(f"'{model_to_try}' 모델 로드에 실패하여 기본 모델 '{get_default_embedding_model_name()}'로 폴백합니다.")
            EMBEDDING_MODEL_STATUS["fallback_used"] = True
            try:
                # 기본 모델은 항상 SSL 검증 활성화 상태로 시도
                current_ssl_setting = VERIFY_SSL
                if not current_ssl_setting: set_ssl_verification(True)
                
                # SentenceTransformerEmbeddingFunction에 device 인자를 직접 전달하도록 수정
                base_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=get_default_embedding_model_name(), device=final_device_to_use)
                EMBEDDING_MODEL_STATUS["actual_model"] = get_default_embedding_model_name()
                print(f"기본 임베딩 모델 '{get_default_embedding_model_name()}' 로드 성공 (장치: {final_device_to_use}).")
                # 폴백 성공 시, 이전 오류 메시지는 유지 (왜 폴백했는지 알려주기 위함)
                if not EMBEDDING_MODEL_STATUS["error_message"]: # 이전 오류가 없었다면 (이 경우는 거의 없음)
                    EMBEDDING_MODEL_STATUS["error_message"] = f"'{model_to_try}' 로드 실패로 기본 모델 사용."
                
                if not current_ssl_setting: set_ssl_verification(False) # 원래 SSL 설정으로 복원

            except Exception as fallback_e:
                print(f"기본 임베딩 모델 '{get_default_embedding_model_name()}' 로드 중 오류 발생: {fallback_e}")
                EMBEDDING_MODEL_STATUS["error_message"] = f"요청 모델 실패({str(e)}), 기본 모델도 실패({str(fallback_e)})"

    # 모든 SentenceTransformer 모델 로드 실패 시 ChromaDB 기본 임베딩 함수 사용
    if base_ef is None:
        print("모든 SentenceTransformer 모델 로드에 실패하여 ChromaDB 내장 기본 임베딩 함수를 사용합니다.")
        base_ef = embedding_functions.DefaultEmbeddingFunction()
        EMBEDDING_MODEL_STATUS["actual_model"] = "chromadb_default_ef"
        EMBEDDING_MODEL_STATUS["fallback_used"] = True
        EMBEDDING_MODEL_STATUS["device_used"] = "cpu" # DefaultEmbeddingFunction은 CPU 사용
        if not EMBEDDING_MODEL_STATUS["error_message"]: # 이전 오류 메시지가 없다면
             EMBEDDING_MODEL_STATUS["error_message"] = "SentenceTransformer 모델 로드 불가."

    # OOM 안전 래퍼 적용
    actual_model_name_for_oom = EMBEDDING_MODEL_STATUS.get("actual_model", get_default_embedding_model_name())
    # base_ef가 None일 수 있는 경우 (모든 모델 로드 실패) 처리 - DefaultEmbeddingFunction은 OOM 발생 안 함
    oom_safe_base_ef = _create_oom_safe_embedding_function(base_ef, EMBEDDING_MODEL_STATUS, actual_model_name_for_oom) if base_ef else base_ef

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
            nonlocal oom_safe_base_ef # OOM 안전 래퍼를 사용
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
                if oom_safe_base_ef is None: # 모든 모델 로드 실패 시
                    logger.error("No embedding function available for encoding.")
                    raise ValueError("Embedding function could not be initialized.")
                new_embeddings = oom_safe_base_ef(texts_to_embed)
                
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
        return oom_safe_base_ef

def get_available_embedding_models():
    """
    사용 가능한 임베딩 모델 목록을 반환합니다.
    
    Returns:
        dict: 카테고리별 추천 임베딩 모델
    """
    return {
        "다국어 모델": [
            get_default_embedding_model_name(),  # 기본 모델
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

class NormalizedEmbeddingFunction:
    """
    ChromaDB 호환 L2 정규화 임베딩 함수 클래스
    ChromaDB의 EmbeddingFunction 인터페이스를 준수합니다.
    """
    
    def __init__(self, base_embedding_function):
        """
        L2 정규화 임베딩 함수를 초기화합니다.
        
        Args:
            base_embedding_function: 기본 임베딩 함수
        """
        self.base_embedding_function = base_embedding_function
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB 호환 임베딩 함수 인터페이스
        
        Args:
            input: 임베딩할 텍스트 목록
            
        Returns:
            List[List[float]]: L2 정규화된 임베딩 벡터 목록
        """
        # 기본 임베딩 함수로 임베딩 생성
        embeddings = self.base_embedding_function(input)
        
        # L2 정규화 적용
        import numpy as np
        normalized_embeddings = []
        for embedding in embeddings:
            # numpy 배열로 변환
            embedding_array = np.array(embedding, dtype=float)
            
            # L2 norm 계산
            l2_norm = np.linalg.norm(embedding_array)
            
            # 0으로 나누기 방지
            if l2_norm > 0:
                normalized_embedding = embedding_array / l2_norm
            else:
                normalized_embedding = embedding_array
            
            normalized_embeddings.append(normalized_embedding.tolist())
        
        return normalized_embeddings

def get_normalized_embedding_function(embedding_model_request="all-MiniLM-L6-v2", device_preference="auto"):
    """
    L2 정규화가 적용된 ChromaDB 호환 임베딩 함수를 생성합니다.
    
    Args:
        embedding_model_request (str): 사용자가 요청한 임베딩 모델 이름
        device_preference (str): 사용할 장치 ("auto", "cuda", "cpu")
        
    Returns:
        NormalizedEmbeddingFunction: ChromaDB 호환 L2 정규화 임베딩 함수
    """
    # 기본 임베딩 함수 가져오기
    base_embedding_function = get_embedding_function(
        embedding_model_request=embedding_model_request,
        use_cache=True,
        device_preference=device_preference
    )
    
    # ChromaDB 호환 정규화 임베딩 함수 클래스로 래핑
    return NormalizedEmbeddingFunction(base_embedding_function)