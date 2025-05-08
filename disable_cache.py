import streamlit as st
import os
import shutil

# ChromaDB 디렉토리 경로
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chromadb')

def disable_chromadb_cache():
    """
    ChromaDB 캐싱을 비활성화하는 함수입니다.
    이 함수를 app.py 시작 부분에서 호출하면 캐싱 관련 오류를 방지할 수 있습니다.
    """
    # 세션 상태에 캐싱 비활성화 플래그 설정
    if 'use_cache' not in st.session_state:
        st.session_state.use_cache = False
    
    print("ChromaDB 캐싱이 비활성화되었습니다.")
    return st.session_state.use_cache

def is_cache_enabled():
    """
    캐싱이 활성화되어 있는지 확인합니다.
    """
    return st.session_state.get('use_cache', False)

def toggle_cache(enable=None):
    """
    캐싱을 활성화하거나 비활성화합니다.
    
    Args:
        enable (bool, optional): True면 활성화, False면 비활성화, None이면 현재 상태 반전
    
    Returns:
        bool: 변경 후 캐싱 상태
    """
    if enable is None:
        st.session_state.use_cache = not st.session_state.get('use_cache', False)
    else:
        st.session_state.use_cache = enable
    
    status = "활성화" if st.session_state.use_cache else "비활성화"
    print(f"ChromaDB 캐싱이 {status}되었습니다.")
    return st.session_state.use_cache

def safe_remove_cache_dir():
    """
    안전하게 캐시 디렉토리를 제거합니다.
    앱 시작 시 호출하면 이전 캐시로 인한 문제를 방지할 수 있습니다.
    """
    try:
        if os.path.exists(CHROMA_PERSIST_DIR):
            print(f"캐시 디렉토리 제거 시도: {CHROMA_PERSIST_DIR}")
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
            print("캐시 디렉토리 제거 완료")
        return True
    except Exception as e:
        print(f"캐시 디렉토리 제거 실패: {str(e)}")
        return False