import streamlit as st
from controllers.csv_upload_controller import CsvUploadController
from views.csv_upload_view import CsvUploadView

def main():
    """메인 애플리케이션 함수"""
    # MVC 컴포넌트 초기화
    controller = CsvUploadController()
    view = CsvUploadView()
    
    # 제목 렌더링
    view.render_title()
    
    # 사이드바 DB 설정
    db_path_result = view.render_sidebar_db_settings(st.session_state.chroma_path)
    
    # DB 경로 검증 및 업데이트
    validated_db_path = controller.handle_db_path_validation(db_path_result['db_path'])
    controller.update_session_state('chroma_path', validated_db_path)
    
    # 사용 가능한 컬렉션 목록 가져오기
    available_collections = controller.get_available_collections(validated_db_path)
    
    # 사이드바에서 컬렉션 관리 UI
    collection_result = view.render_sidebar_collection_management(
        available_collections=available_collections,
        selected_collection=st.session_state.collection_name
    )
    
    if collection_result.get('has_collections', False):
        # 컬렉션 상태 업데이트
        controller.update_collection_state(
            collection_result['selected_collection'], 
            validated_db_path
        )
        
        # 삭제 버튼 처리
        if collection_result['delete_button_clicked']:
            controller.prepare_delete_confirmation(collection_result['selected_collection'])
        
        # 삭제 확인 다이얼로그 (메인 컨텐츠에서 표시)
        if st.session_state.show_delete_confirm and st.session_state.collection_to_delete:
            delete_result = view.render_delete_confirmation(st.session_state.collection_to_delete)
            
            if delete_result['confirm_clicked']:
                # 컬렉션 삭제 수행
                deletion_result = controller.handle_collection_deletion(
                    st.session_state.collection_to_delete, 
                    validated_db_path
                )
                
                if deletion_result['success']:
                    view.show_success(deletion_result['message'])
                    st.rerun()
                else:
                    view.show_error(deletion_result['error'])
            
            elif delete_result['cancel_clicked']:
                controller.cancel_delete_confirmation()
                st.rerun()
        
        # 사이드바에서 컬렉션이 로드된 경우 정보 표시
        if st.session_state.collection_loaded:
            collection_info = controller.get_collection_info()
            view.render_sidebar_collection_info(
                collection_result['selected_collection'],
                collection_info,
                validated_db_path
            )
    
    # 사이드바 구분선
    view.add_sidebar_separator()
    
    # 파일 입력 섹션
    file_input_result = view.render_file_input_section()
    
    # 파일이 있는지 확인
    has_file = False
    file_source = None
    
    if file_input_result['method'] == 'upload' and file_input_result['file_source'] is not None:
        has_file = True
        file_source = file_input_result['file_source']
    elif file_input_result['method'] == 'path' and controller.validate_file_path(file_input_result['file_path']):
        has_file = True
        file_source = file_input_result['file_path']
    elif file_input_result['method'] == 'path' and file_input_result['file_path']:
        view.show_error(f"파일을 찾을 수 없습니다: {file_input_result['file_path']}")
    
    # 파일이 있는 경우 처리
    if has_file:
        # 인코딩 선택
        selected_encoding = view.render_encoding_selection()
        
        # 파일 로드
        file_load_result = controller.handle_file_upload(file_source, selected_encoding)
        
        if file_load_result['success']:
            view.show_success(file_load_result['message'])
            df = file_load_result['dataframe']
            
            # 데이터프레임 분석
            df_analysis = controller.handle_dataframe_analysis(df)
            view.render_dataframe_preview(df_analysis)
            
            if df_analysis['success']:
                # 열 선택
                all_columns = df.columns.tolist()
                selected_columns = view.render_column_selection(
                    all_columns, 
                    st.session_state.selected_columns
                )
                controller.update_session_state('selected_columns', selected_columns)
                
                # 데이터 전처리 옵션
                preprocessing_options = view.render_preprocessing_options()
                
                # 전처리 미리보기 (선택된 열이 있는 경우)
                if selected_columns:
                    preview_result = controller.handle_preprocessing_preview(
                        df, selected_columns, preprocessing_options['max_rows']
                    )
                    view.render_preprocessing_preview(preview_result)
                
                # ChromaDB 저장 옵션
                storage_options = view.render_chroma_storage_options(
                    st.session_state.collection_name,
                    st.session_state.chroma_path
                )
                
                # 임베딩 모델 선택
                embedding_models = controller.get_available_embedding_models()
                selected_embedding_model = view.render_embedding_model_selection(
                    embedding_models,
                    st.session_state.embedding_model
                )
                controller.update_session_state('embedding_model', selected_embedding_model)
                
                # 하드웨어 가속 설정
                gpu_info = controller.get_gpu_info()
                device_options = controller.get_embedding_device_options(gpu_info)
                selected_device = view.render_hardware_acceleration_settings(
                    gpu_info,
                    device_options,
                    st.session_state.embedding_device_preference
                )
                controller.update_session_state('embedding_device_preference', selected_device)
                
                # 저장 버튼
                if view.render_storage_button():
                    # 진행 상황 UI 생성
                    progress_ui = view.render_storage_progress()
                    
                    with view.show_spinner("ChromaDB에 데이터 저장 중..."):
                        # ChromaDB 저장 처리
                        storage_result = controller.handle_chroma_storage(
                            df=df,
                            selected_columns=selected_columns,
                            collection_name=storage_options['collection_name'],
                            persist_directory=storage_options['persist_directory'],
                            max_rows=preprocessing_options['max_rows'],
                            batch_size=preprocessing_options['batch_size'],
                            progress_bar=progress_ui['progress_bar'],
                            status_text=progress_ui['status_text'],
                            gpu_status_placeholder=progress_ui['gpu_status_placeholder']
                        )
                        
                        if storage_result['success']:
                            view.render_storage_messages(storage_result['messages'])
                        else:
                            view.show_error(storage_result['error'])
        else:
            view.show_error(file_load_result['error'])

if __name__ == "__main__":
    main()