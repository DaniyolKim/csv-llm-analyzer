import streamlit as st
import time
import os
from typing import Dict, List, Optional, Any
from models.home_model import HomeModel
from utils import rag_chat_with_ollama, clean_text, get_ollama_install_guide

class HomeController:
    """Home 페이지의 컨트롤러"""
    
    def __init__(self):
        self.model = HomeModel()
    
    def handle_db_path_validation(self, db_path: str) -> str:
        """DB 경로 검증 및 처리"""
        default_db_path = "./chroma_db"
        
        if not os.path.exists(db_path):
            return default_db_path
        return db_path
    
    def handle_collection_load(self, collection_name: str, db_path: str) -> Dict[str, Any]:
        """컬렉션 로드 처리"""
        result = self.model.load_collection(collection_name, db_path)
        
        if result['success']:
            # 성공 메시지 및 경고 처리
            messages = []
            
            if result['stored_model']:
                messages.append({
                    'type': 'info',
                    'content': f"컬렉션에 저장된 임베딩 모델 '{result['stored_model']}'을 사용합니다."
                })
            
            if result['embedding_status']['fallback_used']:
                messages.append({
                    'type': 'warning',
                    'content': f"""⚠️ **임베딩 모델 변경됨**: 요청하신 모델 대신 기본 임베딩 모델이 사용되었습니다.
                    - 요청 모델: {result['embedding_status']['requested_model']}
                    - 사용된 모델: {result['embedding_status']['actual_model']}
                    - 원인: {result['embedding_status']['error_message']}"""
                })
            
            messages.append({
                'type': 'success',
                'content': f"컬렉션 '{collection_name}'을 성공적으로 로드했습니다."
            })
            
            return {
                'success': True,
                'messages': messages
            }
        else:
            return {
                'success': False,
                'error': f"컬렉션 로드 중 오류 발생: {result['error']}"
            }
    
    def handle_ollama_status_check(self) -> Dict[str, Any]:
        """Ollama 상태 확인 처리"""
        status = self.model.check_ollama_status()
        
        if not status['lib_available']:
            return {
                'status': 'lib_not_available',
                'message': "❌ Ollama 라이브러리가 설치되어 있지 않습니다.",
                'guide': """### Ollama 라이브러리 설치하기
                
                Python에서 Ollama를 사용하려면 다음 명령어로 라이브러리를 설치하세요:
                ```
                pip install ollama
                ```
                
                설치 후 애플리케이션을 다시 시작하세요."""
            }
        
        if not status['installed']:
            return {
                'status': 'not_installed',
                'message': "❌ Ollama가 설치되어 있지 않습니다.",
                'guide': get_ollama_install_guide()
            }
        
        if not status['running']:
            return {
                'status': 'not_running',
                'message': "❌ Ollama 서버가 실행되고 있지 않습니다.",
                'guide': """### Ollama 서버 실행하기
                
                터미널에서 다음 명령어를 실행하여 Ollama 서버를 시작하세요:
                ```
                ollama serve
                ```
                
                서버가 실행되면 '상태 다시 확인' 버튼을 클릭하세요."""
            }
        
        if not status['models']:
            return {
                'status': 'no_models',
                'message': "⚠️ 설치된 모델이 없습니다.",
                'guide': """### 모델 설치하기
                
                터미널에서 다음 명령어를 실행하여 모델을 설치하세요:
                ```
                ollama pull llama2
                ```
                
                또는 다른 모델을 설치할 수 있습니다:
                ```
                ollama pull mistral
                ollama pull gemma:2b
                ```
                
                모델 설치 후 '상태 다시 확인' 버튼을 클릭하세요."""
            }
        
        return {
            'status': 'ready',
            'message': "✅ Ollama가 준비되었습니다.",
            'models': status['models']
        }
    
    def handle_chat_submission(self, question: str, prompt: str, selected_model: str, n_results: int) -> Dict[str, Any]:
        """채팅 질문 처리"""
        if not question.strip():
            return {
                'success': False,
                'error': "질문을 입력하세요."
            }
        
        if st.session_state.chroma_collection is None:
            return {
                'success': False,
                'error': "컬렉션이 로드되지 않았습니다."
            }
        
        try:
            # 사용자 메시지 추가
            self.model.add_chat_message("user", question)
            
            # 프롬프트와 질문을 합쳐서 query 생성
            combined_query = ""
            if prompt:
                combined_query += prompt.strip() + "\n\n"
            combined_query += question.strip()
            
            # 쿼리 텍스트 정제
            cleaned_query = clean_text(combined_query)
            
            # 채팅 히스토리 준비
            chat_history = [msg for msg in st.session_state.chat_history if msg["role"] in ["user", "assistant"]]
            
            # RAG 쿼리 실행
            result = rag_chat_with_ollama(
                collection=st.session_state.chroma_collection,
                query=question,
                model_name=selected_model,
                n_results=n_results,
                system_prompt=prompt if prompt else None,
                chat_history=chat_history
            )
            
            # AI 응답 메시지 추가
            self.model.add_chat_message(
                "assistant", 
                result["response"],
                references={
                    "docs": result["context"],
                    "metadatas": result["metadatas"],
                    "distances": result["distances"]
                }
            )
            
            return {
                'success': True,
                'response': result["response"]
            }
            
        except Exception as e:
            # 오류 메시지 추가
            self.model.add_chat_message("error", f"오류 발생: {e}")
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def handle_collection_change_check(self, selected_collection: str, db_path: str) -> bool:
        """컬렉션이나 경로 변경 확인"""
        return (selected_collection != st.session_state.get('current_collection_name') or 
                db_path != st.session_state.get('current_db_path'))
    
    def reset_collection_status(self):
        """컬렉션 상태 리셋"""
        self.model.update_session_state('collection_loaded', False)
    
    def refresh_models(self):
        """모델 목록 새로고침"""
        self.model.refresh_ollama_models()
    
    def reset_status(self):
        """상태 리셋"""
        self.model.reset_ollama_status()
    
    def clear_chat(self):
        """채팅 기록 지우기"""
        self.model.clear_chat_history()
    
    def start_new_chat(self):
        """새 대화 시작"""
        self.model.new_chat()
    
    def get_available_collections(self, db_path: str) -> List[str]:
        """사용 가능한 컬렉션 목록 반환"""
        return self.model.get_available_collections(db_path)
    
    def get_collection_info(self, collection) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        return self.model.get_collection_info(collection)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 반환"""
        return self.model.get_gpu_info()