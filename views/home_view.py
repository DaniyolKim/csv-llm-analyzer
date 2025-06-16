import streamlit as st
import time
from typing import Dict, List, Optional, Any

class HomeView:
    """Home 페이지의 뷰 컴포넌트"""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_css()
    
    def setup_page_config(self):
        """페이지 설정"""
        st.set_page_config(
            page_title="텍스트 CSV 분석기 & RAG",
            page_icon="📊",
            layout="wide"
        )
    
    def setup_css(self):
        """CSS 스타일 설정"""
        st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 60vh;
            overflow-y: auto;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: #f0f2f6;
        }
        .user-message {
            background-color: #e1f5fe;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            align-self: flex-end;
        }
        .assistant-message {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            align-self: flex-start;
        }
        .message-input {
            position: sticky;
            bottom: 0;
            background-color: white;
            padding: 10px;
            border-top: 1px solid #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_title(self):
        """제목 렌더링"""
        st.title("Custom RAG")
    
    def render_sidebar_db_settings(self, db_path: str, available_collections: List[str], 
                                 selected_collection: str) -> Dict[str, Any]:
        """사이드바 DB 설정 렌더링"""
        with st.sidebar:
            st.header("DB 설정")
            
            # ChromaDB 경로 설정
            new_db_path = st.text_input(
                "ChromaDB 경로",
                value=db_path,
                help="ChromaDB가 저장된 경로를 입력하세요. 기본값은 './chroma_db'입니다."
            )
            
            # 경로 검증 메시지
            if db_path != "./chroma_db":
                st.warning(f"입력한 경로({db_path})가 존재하지 않습니다. 기본 경로를 사용합니다.")
            
            return {
                'db_path': new_db_path,
                'path_changed': new_db_path != db_path
            }
    
    def render_sidebar_collection_selection(self, available_collections: List[str], 
                                          selected_collection: str) -> Dict[str, Any]:
        """사이드바에서 컬렉션 선택 UI 렌더링"""
        with st.sidebar:
            st.header("컬렉션 설정")
            
            if available_collections:
                st.success(f"✅ {len(available_collections)}개의 컬렉션을 찾았습니다.")
                
                # 컬렉션 선택
                new_selected_collection = st.selectbox(
                    "컬렉션 선택",
                    options=available_collections,
                    index=0 if available_collections and available_collections[0] == selected_collection else 0,
                    help="검색할 ChromaDB 컬렉션을 선택하세요."
                )
                
                # 컬렉션 로드 버튼
                load_button = st.button("컬렉션 로드", key="load_collection_btn", type="primary")
                
                return {
                    'selected_collection': new_selected_collection,
                    'load_button_clicked': load_button,
                    'has_collections': True
                }
            else:
                st.error(f"사용 가능한 컬렉션이 없습니다.")
                return {
                    'has_collections': False
                }
    
    def render_sidebar_collection_info(self, collection_name: str, collection_info: Dict[str, Any], 
                                     db_path: str):
        """사이드바에서 컬렉션 정보 표시"""
        with st.sidebar:
            with st.expander("컬렉션 정보", expanded=True):
                if collection_info['success']:
                    st.write(f"**컬렉션:** {collection_name}")
                    st.write(f"**문서 수:** {collection_info['count']}")
                    st.write(f"**임베딩 모델:** {collection_info['embedding_model']}")
                    st.write(f"**DB 경로:** {db_path}")
                else:
                    st.error(f"컬렉션 정보 오류: {collection_info['error']}")
    
    def render_collection_selection(self, available_collections: List[str], 
                                  selected_collection: str) -> Dict[str, Any]:
        """컬렉션 선택 UI 렌더링 (메인 컨텐츠용 - 사용 안함)"""
        if available_collections:
            st.success(f"✅ {len(available_collections)}개의 컬렉션을 찾았습니다.")
            
            # 컬렉션 선택
            new_selected_collection = st.selectbox(
                "컬렉션 선택",
                options=available_collections,
                index=0 if available_collections and available_collections[0] == selected_collection else 0,
                help="검색할 ChromaDB 컬렉션을 선택하세요."
            )
            
            # 컬렉션 로드 버튼
            load_button = st.button("컬렉션 로드", key="load_collection_btn")
            
            return {
                'selected_collection': new_selected_collection,
                'load_button_clicked': load_button,
                'has_collections': True
            }
        else:
            st.error(f"사용 가능한 컬렉션이 없습니다.")
            return {
                'has_collections': False
            }
    
    def render_collection_info(self, collection_name: str, collection_info: Dict[str, Any], 
                             db_path: str):
        """컬렉션 정보 표시"""
        with st.expander("컬렉션 정보"):
            if collection_info['success']:
                st.write(f"컬렉션 이름: {collection_name}")
                st.write(f"문서 수: {collection_info['count']}")
                st.write(f"임베딩 모델: {collection_info['embedding_model']}")
                st.write(f"DB 경로: {db_path}")
            else:
                st.error(f"컬렉션 정보를 가져오는 중 오류가 발생했습니다: {collection_info['error']}")
    
    def render_loading_messages(self, messages: List[Dict[str, str]]):
        """로딩 후 메시지들 표시"""
        for message in messages:
            if message['type'] == 'success':
                st.success(message['content'])
            elif message['type'] == 'warning':
                st.warning(message['content'])
            elif message['type'] == 'info':
                st.info(message['content'])
            elif message['type'] == 'error':
                st.error(message['content'])
    
    def render_ollama_status(self, status_result: Dict[str, Any]) -> bool:
        """Ollama 상태 렌더링 및 재확인 버튼 처리"""
        st.subheader("Ollama 연동")
        
        if status_result['status'] == 'lib_not_available':
            st.error(status_result['message'])
            st.markdown(status_result['guide'])
            return False
        
        elif status_result['status'] in ['not_installed', 'not_running', 'no_models']:
            if status_result['status'] == 'not_installed':
                st.error(status_result['message'])
            elif status_result['status'] == 'not_running':
                st.error(status_result['message'])
            else:  # no_models
                st.warning(status_result['message'])
            
            st.markdown(status_result['guide'])
            
            return st.button("Ollama 상태 다시 확인")
        
        elif status_result['status'] == 'ready':
            st.success(status_result['message'])
            return True
        
        return False
    
    def render_model_selection(self, models: List[str]) -> Dict[str, Any]:
        """모델 선택 UI 렌더링"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_model = st.selectbox(
                "사용할 모델 선택", 
                models,
                index=0 if "llama2" not in models else models.index("llama2")
            )
        
        with col2:
            refresh_button = st.button("모델 목록 새로고침", use_container_width=True)
        
        return {
            'selected_model': selected_model,
            'refresh_clicked': refresh_button
        }
    
    def render_system_prompt_settings(self, current_prompt: str, gpu_info: Dict[str, Any],
                                    current_gpu_count: int) -> Dict[str, Any]:
        """시스템 프롬프트 설정 렌더링"""
        with st.expander("시스템 프롬프트 설정", expanded=False):
            # 프롬프트 입력
            prompt = st.text_area(
                "시스템 프롬프트 (지시사항)",
                height=150,
                placeholder="모델에게 전달할 지시사항을 입력하세요.",
                value=current_prompt
            )
            
            # 참조할 문서 수 설정
            n_results = st.slider(
                "참조할 문서 수", 
                min_value=3, 
                max_value=20, 
                value=5, 
                step=1,
                help="참조할 문서 수를 선택하세요. 일반적으로 3-5개가 적당합니다."
            )
            
            # GPU 설정
            ollama_num_gpu = 0
            if gpu_info["available"]:
                ollama_num_gpu = st.number_input(
                    "Ollama에 할당할 GPU 수", 
                    min_value=0, 
                    max_value=gpu_info["count"], 
                    value=current_gpu_count if current_gpu_count <= gpu_info["count"] else (1 if gpu_info["count"] > 0 else 0),
                    step=1,
                    help=f"Ollama 모델 추론에 사용할 GPU 개수입니다. (시스템에 사용 가능: {gpu_info['count']}개). 0으로 설정하면 CPU를 사용합니다."
                )
            else:
                st.info("Ollama 추론에 사용 가능한 GPU가 없습니다. CPU를 사용합니다.")
            
            return {
                'prompt': prompt,
                'n_results': n_results,
                'ollama_num_gpu': ollama_num_gpu
            }
    
    def render_chat_interface(self, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """채팅 인터페이스 렌더링"""
        chat_container = st.container()
        
        with chat_container:
            st.subheader("대화")
            
            # 채팅 기록 컨트롤 버튼
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                clear_button = st.button("대화 기록 지우기", key="clear_history")
            with col2:
                new_chat_button = st.button("새 대화 시작", key="new_chat")
            
            # 채팅 기록을 표시할 컨테이너
            chat_history_container = st.container(height=600)
            
            # 기존 채팅 기록 표시
            with chat_history_container:
                self._render_chat_messages(chat_history)
            
            # 구분선
            st.markdown("---")
            
            # 질문 입력 영역
            col1, col2 = st.columns([5, 1])
            with col1:
                question = st.text_area(
                    "질문을 입력하세요",
                    key="question_input",
                    height=80,
                    placeholder="질문을 입력한 후 전송 버튼을 클릭하세요.",
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                submit_button = st.button("전송", key="submit_question", use_container_width=True)
            
            return {
                'question': question,
                'submit_clicked': submit_button,
                'clear_clicked': clear_button,
                'new_chat_clicked': new_chat_button
            }
    
    def _render_chat_messages(self, chat_history: List[Dict[str, Any]]):
        """채팅 메시지들 렌더링"""
        for chat in chat_history:
            if chat["role"] == "user":
                self._render_user_message(chat)
            elif chat["role"] == "assistant":
                self._render_assistant_message(chat)
            elif chat["role"] == "error":
                self._render_error_message(chat)
    
    def _render_user_message(self, chat: Dict[str, Any]):
        """사용자 메시지 렌더링"""
        message_container = st.container()
        with message_container:
            col1, col2 = st.columns([1, 9])
            with col1:
                st.markdown("### 🧑")
            with col2:
                st.markdown(f"**사용자** <span style='color:gray;font-size:0.8em;'>{chat['timestamp']}</span>", unsafe_allow_html=True)
                st.markdown(chat["content"])
    
    def _render_assistant_message(self, chat: Dict[str, Any]):
        """AI 어시스턴트 메시지 렌더링"""
        message_container = st.container()
        with message_container:
            col1, col2 = st.columns([1, 9])
            with col1:
                st.markdown("### 🤖")
            with col2:
                st.markdown(f"**AI 어시스턴트** <span style='color:gray;font-size:0.8em;'>{chat['timestamp']}</span>", unsafe_allow_html=True)
                st.markdown(chat["content"])
                
                # 참조 문서가 있으면 확장 가능한 섹션으로 표시
                if "references" in chat:
                    with st.expander("참조 문서", expanded=False):
                        for i, (doc, metadata, distance) in enumerate(zip(
                            chat["references"]["docs"],
                            chat["references"]["metadatas"],
                            chat["references"]["distances"]
                        )):
                            st.markdown(f"**문서 {i+1}** (유사도: {1-distance:.4f})")
                            st.info(doc)
                            st.write(f"메타데이터: {metadata}")
                            if i < len(chat["references"]["docs"]) - 1:
                                st.markdown("---")
    
    def _render_error_message(self, chat: Dict[str, Any]):
        """오류 메시지 렌더링"""
        message_container = st.container()
        with message_container:
            col1, col2 = st.columns([1, 9])
            with col1:
                st.markdown("### ⚠️")
            with col2:
                st.markdown(f"**시스템 오류** <span style='color:gray;font-size:0.8em;'>{chat['timestamp']}</span>", unsafe_allow_html=True)
                st.error(chat["content"])
    
    def show_spinner(self, message: str = "처리 중..."):
        """스피너 표시"""
        return st.spinner(message)
    
    def show_no_rag_message(self):
        """RAG 비활성화 메시지 표시"""
        st.info("Ollama 연동을 사용하려면 먼저 데이터를 ChromaDB에 저장하거나 기존 컬렉션을 로드하세요.")
    
    def add_sidebar_separator(self):
        """사이드바 구분선 추가"""
        with st.sidebar:
            st.markdown("---")