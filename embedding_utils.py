"""
임베딩 모델 관련 유틸리티 함수 모음
"""
import ssl
import warnings
import requests
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

def get_embedding_function(embedding_model="all-MiniLM-L6-v2"):
    """
    임베딩 함수를 생성합니다.
    
    Args:
        embedding_model (str): 임베딩 모델 이름
        
    Returns:
        embedding_function: 임베딩 함수
    """
    # 임베딩 상태 초기화
    global EMBEDDING_MODEL_STATUS
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
    
    return embedding_function

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