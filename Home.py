import streamlit as st
from controllers.home_controller import HomeController
from views.home_view import HomeView

def main():
    """메인 애플리케이션 함수"""
    # MVC 컴포넌트 초기화
    controller = HomeController()
    view = HomeView()
    
    # 제목 렌더링
    view.render_title()
    
    # 사이드바 DB 설정
    db_path_result = view.render_sidebar_db_settings(
        db_path=st.session_state.chroma_path,
        available_collections=[],
        selected_collection=st.session_state.collection_name
    )
    
    # DB 경로 검증 및 업데이트
    validated_db_path = controller.handle_db_path_validation(db_path_result['db_path'])
    st.session_state.chroma_path = validated_db_path
    
    # 사용 가능한 컬렉션 목록 가져오기
    available_collections = controller.get_available_collections(validated_db_path)
    
    # 사이드바에서 컬렉션 선택 및 로드
    collection_result = view.render_sidebar_collection_selection(
        available_collections=available_collections,
        selected_collection=st.session_state.collection_name
    )
    
    if collection_result.get('has_collections', False):
        # 컬렉션 변경 확인
        if controller.handle_collection_change_check(
            collection_result['selected_collection'], 
            validated_db_path
        ):
            controller.reset_collection_status()
        
        # 세션 상태 업데이트
        st.session_state.collection_name = collection_result['selected_collection']
        
        # 컬렉션 로드 처리
        if collection_result['load_button_clicked']:
            with view.show_spinner("컬렉션을 로드하는 중..."):
                load_result = controller.handle_collection_load(
                    collection_result['selected_collection'],
                    validated_db_path
                )
                
                if load_result['success']:
                    view.render_loading_messages(load_result['messages'])
                else:
                    st.error(load_result['error'])
        
        # 사이드바에서 컬렉션이 로드된 경우 정보 표시
        if st.session_state.collection_loaded:
            collection_info = controller.get_collection_info(st.session_state.chroma_collection)
            view.render_sidebar_collection_info(
                collection_result['selected_collection'],
                collection_info,
                validated_db_path
            )
    
    # 사이드바 구분선
    view.add_sidebar_separator()
    
    # RAG가 활성화된 경우 Ollama 연동 처리
    if st.session_state.rag_enabled:
        # Ollama 상태 확인
        with view.show_spinner("Ollama 상태 확인 중..."):
            ollama_status = controller.handle_ollama_status_check()
        
        # Ollama 상태에 따른 UI 처리
        status_check_result = view.render_ollama_status(ollama_status)
        
        if status_check_result is True:  # Ollama가 준비된 경우
            # 모델 선택
            model_result = view.render_model_selection(st.session_state.ollama_models)
            
            # 모델 새로고침 처리
            if model_result['refresh_clicked']:
                with view.show_spinner("모델 목록을 가져오는 중..."):
                    controller.refresh_models()
                    st.rerun()
            
            # 시스템 프롬프트 설정
            gpu_info = controller.get_gpu_info()
            prompt_settings = view.render_system_prompt_settings(
                current_prompt=st.session_state.get('prompt', ''),
                gpu_info=gpu_info,
                current_gpu_count=st.session_state.ollama_num_gpu
            )
            
            # 세션 상태 업데이트
            st.session_state.prompt = prompt_settings['prompt']
            st.session_state.ollama_num_gpu = prompt_settings['ollama_num_gpu']
            
            # 채팅 인터페이스
            chat_result = view.render_chat_interface(st.session_state.chat_history)
            
            # 채팅 버튼 처리
            if chat_result['clear_clicked']:
                controller.clear_chat()
                st.rerun()
            
            if chat_result['new_chat_clicked']:
                controller.start_new_chat()
                st.rerun()
            
            # 질문 제출 처리
            if chat_result['submit_clicked'] and chat_result['question']:
                # 응답 생성 중 표시
                message_container = st.container()
                with message_container:
                    col1, col2 = st.columns([1, 9])
                    with col1:
                        st.markdown("### 🤖")
                    with col2:
                        st.markdown("**AI 어시스턴트**")
                        status_text = st.empty()
                        status_text.markdown("*응답 생성 중...*")
                
                # 채팅 처리
                chat_submission_result = controller.handle_chat_submission(
                    question=chat_result['question'],
                    prompt=prompt_settings['prompt'],
                    selected_model=model_result['selected_model'],
                    n_results=prompt_settings['n_results']
                )
                
                if not chat_submission_result['success']:
                    st.error(chat_submission_result['error'])
                
                # 페이지 새로고침하여 채팅 기록 업데이트
                st.rerun()
        
        elif status_check_result is False:  # 재확인 버튼 클릭
            controller.reset_status()
            st.rerun()
    
    else:
        # RAG가 비활성화된 경우
        view.show_no_rag_message()

if __name__ == "__main__":
    main()