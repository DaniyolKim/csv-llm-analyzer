import streamlit as st
from typing import Dict, List, Optional, Any
from utils import (
    get_available_collections,
    load_chroma_collection,
    get_embedding_function,
    get_embedding_status,
    get_gpu_info,
    is_ollama_installed,
    is_ollama_running,
    is_ollama_lib_available,
    get_ollama_models,
    rag_chat_with_ollama
)

class HomeModel:
    """Home 페이지의 데이터 모델"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        session_defaults = {
            'chroma_client': None,
            'chroma_collection': None,
            'rag_enabled': False,
            'ollama_models': [],
            'ollama_status_checked': False,
            'ollama_installed': False,
            'ollama_running': False,
            'chroma_path': "./chroma_db",
            'collection_name': "csv_test",
            'chat_history': [],
            'current_question': "",
            'ollama_num_gpu': 0,
            'prompt': "",
            'current_collection_name': None,
            'current_db_path': None,
            'collection_loaded': False,
            'embedding_model': None
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Ollama 상태 확인"""
        if not st.session_state.ollama_status_checked:
            # Ollama 설치 확인
            st.session_state.ollama_installed = is_ollama_installed()
            
            if st.session_state.ollama_installed:
                # Ollama 서버 실행 확인
                st.session_state.ollama_running = is_ollama_running()
                
                if st.session_state.ollama_running:
                    # 모델 목록 가져오기
                    st.session_state.ollama_models = get_ollama_models()
            
            st.session_state.ollama_status_checked = True
        
        return {
            'installed': st.session_state.ollama_installed,
            'running': st.session_state.ollama_running,
            'models': st.session_state.ollama_models,
            'lib_available': is_ollama_lib_available()
        }
    
    def get_available_collections(self, db_path: str) -> List[str]:
        """사용 가능한 컬렉션 목록 반환"""
        return get_available_collections(db_path)
    
    def load_collection(self, collection_name: str, db_path: str) -> Dict[str, Any]:
        """컬렉션 로드"""
        try:
            client, collection = load_chroma_collection(
                collection_name=collection_name,
                persist_directory=db_path
            )
            
            st.session_state.chroma_client = client
            st.session_state.chroma_collection = collection
            st.session_state.rag_enabled = True
            st.session_state.collection_loaded = True
            st.session_state.current_collection_name = collection_name
            st.session_state.current_db_path = db_path
            
            # 임베딩 모델 상태 확인
            embedding_status = get_embedding_status()
            
            # 컬렉션에 저장된 임베딩 모델 정보 확인
            stored_model = None
            try:
                if collection.metadata and "embedding_model" in collection.metadata:
                    stored_model = collection.metadata["embedding_model"]
                    st.session_state.embedding_model = stored_model
            except:
                pass
            
            return {
                'success': True,
                'embedding_status': embedding_status,
                'stored_model': stored_model,
                'collection': collection
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_collection_info(self, collection) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        try:
            collection_info = collection.count()
            
            # 컬렉션에 저장된 임베딩 모델 정보 확인
            embedding_model = "알 수 없음"
            try:
                if collection.metadata and "embedding_model" in collection.metadata:
                    embedding_model = collection.metadata["embedding_model"]
            except:
                pass
            
            return {
                'success': True,
                'count': collection_info,
                'embedding_model': embedding_model
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def refresh_ollama_models(self):
        """Ollama 모델 목록 새로고침"""
        st.session_state.ollama_models = get_ollama_models()
    
    def reset_ollama_status(self):
        """Ollama 상태 재확인을 위한 리셋"""
        st.session_state.ollama_status_checked = False
    
    def add_chat_message(self, role: str, content: str, **kwargs):
        """채팅 메시지 추가"""
        import time
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 추가 정보가 있으면 포함
        message.update(kwargs)
        
        st.session_state.chat_history.append(message)
    
    def clear_chat_history(self):
        """채팅 기록 지우기"""
        st.session_state.chat_history = []
    
    def new_chat(self):
        """새 대화 시작"""
        st.session_state.chat_history = []
        st.session_state.current_question = ""
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 반환"""
        return get_gpu_info()
    
    def update_session_state(self, key: str, value: Any):
        """세션 상태 업데이트"""
        st.session_state[key] = value
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """세션 상태 값 반환"""
        return st.session_state.get(key, default)