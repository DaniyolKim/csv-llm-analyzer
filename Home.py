import streamlit as st
import time
from utils import (
    is_ollama_installed,
    is_ollama_running,
    is_ollama_lib_available,
    get_ollama_models,
    get_ollama_install_guide,
    rag_chat_with_ollama,
    get_available_collections,
    load_chroma_collection,
    delete_collection,
    get_embedding_status
)
from embedding_utils import get_available_embedding_models

# 페이지 설정
st.set_page_config(
    page_title="텍스트 CSV 분석기 & RAG",
    page_icon="📊",
    layout="wide"
)

# 세션 상태 초기화
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'chroma_collection' not in st.session_state:
    st.session_state.chroma_collection = None
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = False
if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []
if 'ollama_status_checked' not in st.session_state:
    st.session_state.ollama_status_checked = False
if 'ollama_installed' not in st.session_state:
    st.session_state.ollama_installed = False
if 'ollama_running' not in st.session_state:
    st.session_state.ollama_running = False
if 'chroma_path' not in st.session_state:
    st.session_state.chroma_path = "./chroma_db"
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "csv_test"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# 제목
st.title("Custom RAG")

# 사이드바에 기존 ChromaDB 로드 옵션 추가
with st.sidebar:
    st.header("기존 ChromaDB 로드")
    
    # ChromaDB 경로 입력
    chroma_path = st.text_input("ChromaDB 경로", value=st.session_state.chroma_path)
    st.session_state.chroma_path = chroma_path
    
    # 사용 가능한 컬렉션 목록 가져오기
    available_collections = get_available_collections(chroma_path)
    
    if available_collections:
        st.success(f"✅ {len(available_collections)}개의 컬렉션을 찾았습니다.")
        
        # 삭제할 컬렉션 상태 관리
        if 'collection_to_delete' not in st.session_state:
            st.session_state.collection_to_delete = None
            
        if 'show_delete_confirm' not in st.session_state:
            st.session_state.show_delete_confirm = False
            
        # 컬렉션 선택을 위한 컨테이너
        collection_container = st.container()
        
        with collection_container:
            # 컬렉션 목록 표시
            st.write("### 컬렉션 목록")
            for collection in available_collections:
                col1, col2 = st.columns([7, 2])
                with col1:
                    # 컬렉션 이름 표시
                    is_selected = st.radio(
                        label="",
                        options=[collection],
                        key=f"radio_{collection}",
                        label_visibility="collapsed",
                        index=0 if collection == st.session_state.collection_name else None
                    )
                    if is_selected:
                        st.session_state.collection_name = collection
                
                with col2:
                    # 삭제 버튼
                    if st.button("삭제", key=f"delete_{collection}", type="secondary"):
                        st.session_state.collection_to_delete = collection
                        st.session_state.show_delete_confirm = True
        

                        
        # 삭제 확인 다이얼로그
        if st.session_state.show_delete_confirm and st.session_state.collection_to_delete:
            with st.expander(f"'{st.session_state.collection_to_delete}' 컬렉션을 삭제하시겠습니까?", expanded=True):
                st.warning(f"'{st.session_state.collection_to_delete}' 컬렉션의 모든 데이터가 삭제됩니다.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("확인", key="confirm_delete", type="primary"):
                        # 컬렉션 삭제 수행
                        success = delete_collection(st.session_state.collection_to_delete, chroma_path)
                        if success:
                            # 현재 로드된 컬렉션이 삭제되었다면 상태 초기화
                            if st.session_state.collection_name == st.session_state.collection_to_delete:
                                st.session_state.chroma_collection = None
                                st.session_state.chroma_client = None
                                st.session_state.rag_enabled = False
                            
                            st.success(f"'{st.session_state.collection_to_delete}' 컬렉션이 삭제되었습니다.")
                            # 상태 초기화
                            st.session_state.collection_to_delete = None
                            st.session_state.show_delete_confirm = False
                            
                            # 페이지 새로고침
                            st.rerun()
                        else:
                            st.error(f"'{st.session_state.collection_to_delete}' 컬렉션 삭제 중 오류가 발생했습니다.")
                
                with col2:
                    if st.button("취소", key="cancel_delete"):
                        st.session_state.collection_to_delete = None
                        st.session_state.show_delete_confirm = False
                        st.rerun()
                    
        # 컬렉션 로드 버튼
        if st.button("컬렉션 로드"):
            try:
                client, collection = load_chroma_collection(
                    st.session_state.collection_name, 
                    chroma_path,
                    embedding_model=st.session_state.embedding_model
                )
                st.session_state.chroma_client = client
                st.session_state.chroma_collection = collection
                st.session_state.rag_enabled = True
                
                # 임베딩 모델 상태 확인
                embedding_status = get_embedding_status()
                
                # 컬렉션에 저장된 임베딩 모델 정보 확인
                stored_model = None
                try:
                    if collection.metadata and "embedding_model" in collection.metadata:
                        stored_model = collection.metadata["embedding_model"]
                        # 저장된 모델 정보가 있으면 세션 상태 업데이트
                        st.session_state.embedding_model = stored_model
                        st.info(f"컬렉션에 저장된 임베딩 모델 '{stored_model}'을 사용합니다.")
                except:
                    pass
                
                if embedding_status["fallback_used"]:
                    st.warning(f"""
                    ⚠️ **임베딩 모델 변경됨**: 요청하신 모델 대신 기본 임베딩 모델이 사용되었습니다.
                    - 요청 모델: {embedding_status["requested_model"]}
                    - 사용된 모델: {embedding_status["actual_model"]}
                    - 원인: {embedding_status["error_message"]}
                    """)
                
                st.success(f"컬렉션 '{st.session_state.collection_name}'을 성공적으로 로드했습니다.")
            except Exception as e:
                st.error(f"컬렉션 로드 중 오류 발생: {e}")
    else:
        st.info(f"'{chroma_path}' 경로에 사용 가능한 컬렉션이 없습니다.")
    
    st.markdown("---")

# Ollama 연동 섹션
st.subheader("Ollama 연동")

if st.session_state.rag_enabled:
    
    # Ollama 라이브러리 확인
    if not is_ollama_lib_available():
        st.error("❌ Ollama 라이브러리가 설치되어 있지 않습니다.")
        st.markdown("""
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        
        설치 후 애플리케이션을 다시 시작하세요.
        """)
        st.stop()
    
    # Ollama 상태 확인 (처음 한 번만)
    if not st.session_state.ollama_status_checked:
        with st.spinner("Ollama 상태 확인 중..."):
            # Ollama 설치 확인
            st.session_state.ollama_installed = is_ollama_installed()
            
            if st.session_state.ollama_installed:
                # Ollama 서버 실행 확인
                st.session_state.ollama_running = is_ollama_running()
                
                if st.session_state.ollama_running:
                    # 모델 목록 가져오기
                    st.session_state.ollama_models = get_ollama_models()
            
            st.session_state.ollama_status_checked = True
    
    # Ollama 상태에 따른 UI 표시
    if not st.session_state.ollama_installed:
        st.error("❌ Ollama가 설치되어 있지 않습니다.")
        st.markdown(get_ollama_install_guide())
        
        if st.button("Ollama 상태 다시 확인"):
            st.session_state.ollama_status_checked = False
            st.rerun()
    
    elif not st.session_state.ollama_running:
        st.error("❌ Ollama 서버가 실행되고 있지 않습니다.")
        st.markdown("""
        ### Ollama 서버 실행하기
        
        터미널에서 다음 명령어를 실행하여 Ollama 서버를 시작하세요:
        ```
        ollama serve
        ```
        
        서버가 실행되면 '상태 다시 확인' 버튼을 클릭하세요.
        """)
        
        if st.button("Ollama 상태 다시 확인"):
            st.session_state.ollama_status_checked = False
            st.rerun()
    
    elif not st.session_state.ollama_models:
        st.warning("⚠️ 설치된 모델이 없습니다.")
        st.markdown("""
        ### 모델 설치하기
        
        터미널에서 다음 명령어를 실행하여 모델을 설치하세요:
        ```
        ollama pull llama2
        ```
        
        또는 다른 모델을 설치할 수 있습니다:
        ```
        ollama pull mistral
        ollama pull gemma:2b
        ```
        
        모델 설치 후 '상태 다시 확인' 버튼을 클릭하세요.
        """)
        
        if st.button("Ollama 상태 다시 확인"):
            st.session_state.ollama_status_checked = False
            st.rerun()
    
    else:
        # 모든 조건이 충족되면 Ollama 사용 가능
        st.success("✅ Ollama가 준비되었습니다.")
        
        # 모델 선택
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_model = st.selectbox(
                "사용할 모델 선택", 
                st.session_state.ollama_models,
                index=0 if "llama2" not in st.session_state.ollama_models else st.session_state.ollama_models.index("llama2")
            )
        
        with col2:
            # 모델 새로고침 버튼
            if st.button("모델 목록 새로고침", use_container_width=True):
                with st.spinner("모델 목록을 가져오는 중..."):
                    st.session_state.ollama_models = get_ollama_models()
                    st.rerun()
        
        # 프롬프트 입력 (여러 줄 입력 가능)
        with st.expander("시스템 프롬프트 설정", expanded=False):
            prompt = st.text_area(
                "시스템 프롬프트 (지시사항)",
                height=150,
                placeholder="모델에게 전달할 지시사항을 입력하세요.",
                value=st.session_state.get('prompt', '')
            )
            st.session_state['prompt'] = prompt
            
            # 참조할 문서 수 설정 (최소 3, 최대 20)
            n_results = st.slider(
                "참조할 문서 수", 
                min_value=3, 
                max_value=20, 
                value=5, 
                step=1,
                help="참조할 문서 수를 선택하세요. 일반적으로 3-5개가 적당합니다."
            )
        
        # 채팅 인터페이스 컨테이너
        chat_container = st.container()
        
        # 채팅 인터페이스 스타일 적용
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
        
        # 채팅 기록 표시
        with chat_container:
            st.subheader("대화")
            
            # 채팅 기록 컨트롤 버튼
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("대화 기록 지우기", key="clear_history"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                if st.button("새 대화 시작", key="new_chat"):
                    st.session_state.chat_history = []
                    st.session_state.current_question = ""
                    st.rerun()
            
            # 채팅 기록을 표시할 컨테이너
            chat_history_container = st.container(height=600)
            
            # 기존 채팅 기록 표시
            with chat_history_container:
                for chat in st.session_state.chat_history:
                    if chat["role"] == "user":
                        message_container = st.container()
                        with message_container:
                            col1, col2 = st.columns([1, 9])
                            with col1:
                                st.markdown("### 🧑")
                            with col2:
                                st.markdown(f"**사용자** <span style='color:gray;font-size:0.8em;'>{chat['timestamp']}</span>", unsafe_allow_html=True)
                                st.markdown(chat["content"])
                    elif chat["role"] == "assistant":
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
                    elif chat["role"] == "error":
                        message_container = st.container()
                        with message_container:
                            col1, col2 = st.columns([1, 9])
                            with col1:
                                st.markdown("### ⚠️")
                            with col2:
                                st.markdown(f"**시스템 오류** <span style='color:gray;font-size:0.8em;'>{chat['timestamp']}</span>", unsafe_allow_html=True)
                                st.error(chat["content"])
            
            # 구분선
            st.markdown("---")
            
            # 질문 입력 영역과 전송 버튼 (하단에 고정)
            col1, col2 = st.columns([5, 1])
            with col1:
                question = st.text_area(
                    "질문을 입력하세요",
                    key="question_input",
                    height=80,
                    placeholder="질문을 입력한 후 전송 버튼을 클릭하세요.",
                    value=st.session_state.current_question,
                    on_change=lambda: setattr(st.session_state, 'current_question', '')
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # 간격 조정
                submit_question = st.button("전송", key="submit_question", use_container_width=True)
        
        # 질문이 입력되었고 전송 버튼이 클릭되었을 때
        if (question or st.session_state.current_question) and submit_question and st.session_state.chroma_collection is not None:
            # 현재 질문 가져오기
            current_question = question if question else st.session_state.current_question
            st.session_state.current_question = ""  # 질문 초기화
            
            # 프롬프트와 질문을 합쳐서 query 생성
            combined_query = ""
            
            # 프롬프트가 있으면 먼저 추가
            if prompt:
                combined_query += prompt.strip() + "\n\n"
                
            # 질문 추가
            if current_question:
                combined_query += current_question.strip()
            
            if not combined_query.strip():
                st.warning("프롬프트 또는 질문을 입력하세요.")
            else:
                # 채팅 기록에 질문 추가
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": current_question,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # 응답 생성 중 표시
                message_container = st.container()
                with message_container:
                    col1, col2 = st.columns([1, 9])
                    with col1:
                        st.markdown("### 🤖")
                    with col2:
                        st.markdown(f"**AI 어시스턴트**")
                        status_text = st.empty()
                        status_text.markdown("*응답 생성 중...*")
                
                try:
                    import time
                    
                    # 쿼리 텍스트 정제 및 ChromaDB 검색 준비
                    from utils import clean_text
                    cleaned_query = clean_text(combined_query)
                    
                    # n_results 처리
                    actual_n_results = n_results
                    
                    # RAG 쿼리 실행
                    result = rag_chat_with_ollama(
                        st.session_state.chroma_collection,
                        combined_query,
                        selected_model,
                        actual_n_results
                    )
                    # RAG 쿼리 실행
                    chat_history = [msg for msg in st.session_state.chat_history if msg["role"] in ["user", "assistant"]]
                    result = rag_chat_with_ollama(
                        collection=st.session_state.chroma_collection,
                        query=current_question,  # combined_query 대신 current_question 사용
                        model_name=selected_model,
                        n_results=actual_n_results,
                        system_prompt=prompt if prompt else None,  # 시스템 프롬프트 전달
                        chat_history=chat_history  # 이전 대화 기록 전달
                    )
                    
                    # 채팅 기록에 응답 추가
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["response"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "references": {
                            "docs": result["context"],
                            "metadatas": result["metadatas"],
                            "distances": result["distances"]
                        }
                    })
                    
                    # 페이지 새로고침하여 채팅 기록 업데이트
                    st.rerun()
                except Exception as e:
                    # 오류 메시지를 채팅 기록에 추가
                    st.session_state.chat_history.append({
                        "role": "error",
                        "content": f"오류 발생: {e}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # 페이지 새로고침하여 채팅 기록 업데이트
                    st.rerun()
else:
    st.info("Ollama 연동을 사용하려면 먼저 데이터를 ChromaDB에 저장하거나 기존 컬렉션을 로드하세요.")