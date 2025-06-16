# utils 패키지 초기화 파일
"""
CSV LLM Analyzer 유틸리티 모듈 패키지

이 패키지는 다음과 같은 유틸리티 모듈들을 포함합니다:
- chroma_utils: ChromaDB 관련 유틸리티
- data_utils: 데이터 처리 관련 유틸리티
- db_search_utils: DB 검색 관련 유틸리티
- embedding_utils: 임베딩 모델 관련 유틸리티
- ollama_utils: Ollama 연동 관련 유틸리티
- rag_utils: RAG 관련 유틸리티
- text_utils: 텍스트 처리 관련 유틸리티
- visualization_utils: 시각화 관련 유틸리티
"""

__version__ = "1.0.0"
__author__ = "CSV LLM Analyzer Team"

# ChromaDB 관련 함수들
from .chroma_utils import (
    create_chroma_db,
    load_chroma_collection,
    get_available_collections,
    store_data_in_chroma,
    query_chroma,
    delete_collection
)

# 데이터 처리 관련 함수들
from .data_utils import preprocess_dataframe

# 임베딩 모델 관련 함수들
from .embedding_utils import (
    reset_embedding_status,
    get_embedding_status,
    set_ssl_verification,
    get_embedding_function,
    get_available_embedding_models,
    is_gpu_available,
    get_gpu_info
)

# Ollama 관련 함수들
from .ollama_utils import (
    is_ollama_installed,
    is_ollama_running,
    is_ollama_lib_available,
    get_ollama_models,
    query_ollama,
    get_ollama_install_guide
)

# RAG 시스템 관련 함수들
from .rag_utils import (
    rag_query_with_ollama,
    rag_query_with_metadata_filter,
    rag_chat_with_ollama
)

# 텍스트 처리 관련 함수들
from .text_utils import (
    clean_text,
    KOREAN_STOPWORDS,
    IMPORTANT_SINGLE_CHAR_NOUNS
)

# 가장 자주 사용되는 함수들을 __all__에 포함
__all__ = [
    # ChromaDB 관련
    'create_chroma_db',
    'load_chroma_collection',
    'get_available_collections',
    'store_data_in_chroma',
    'query_chroma',
    'delete_collection',
    
    # 데이터 처리
    'preprocess_dataframe',
    
    # 임베딩 관련
    'reset_embedding_status',
    'get_embedding_status',
    'set_ssl_verification',
    'get_embedding_function',
    'get_available_embedding_models',
    'is_gpu_available',
    'get_gpu_info',
    
    # Ollama 관련
    'is_ollama_installed',
    'is_ollama_running',
    'is_ollama_lib_available',
    'get_ollama_models',
    'query_ollama',
    'get_ollama_install_guide',
    
    # RAG 관련
    'rag_query_with_ollama',
    'rag_query_with_metadata_filter',
    'rag_chat_with_ollama',
    
    # 텍스트 처리
    'clean_text',
    'KOREAN_STOPWORDS',
    'IMPORTANT_SINGLE_CHAR_NOUNS'
]