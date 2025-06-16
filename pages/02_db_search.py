import streamlit as st
from controllers.db_search_controller import DbSearchController
from views.db_search_view import DbSearchView

def main():
    """메인 애플리케이션 함수"""
    # MVC 컴포넌트 초기화
    controller = DbSearchController()
    view = DbSearchView()
    
    # 제목 렌더링
    view.render_title()
    
    # 사이드바 DB 설정
    db_path_result = view.render_sidebar_db_settings("./chroma_db")
    
    # DB 경로 검증 및 업데이트
    validated_db_path = controller.handle_db_path_validation(db_path_result['db_path'])
    
    # GPU 정보 및 하드웨어 설정
    gpu_info = controller.get_gpu_info()
    device_options = controller.get_embedding_device_options(gpu_info)
    current_preference = controller.get_session_state('embedding_device_preference', 'auto')
    
    selected_device = view.render_hardware_settings(gpu_info, device_options, current_preference)
    controller.update_session_state('embedding_device_preference', selected_device)
    
    # 사용 가능한 컬렉션 목록 가져오기
    available_collections = controller.get_available_collections(validated_db_path)
    
    # 사이드바에서 컬렉션 선택 및 로드
    collection_result = view.render_sidebar_collection_selection(
        available_collections, 
        controller.get_session_state('current_collection_name')
    )
    
    if collection_result['has_collections']:
        selected_collection = collection_result['selected_collection']
        
        # 컬렉션이나 경로 변경 확인
        if controller.check_collection_change(selected_collection, validated_db_path):
            controller.update_session_state('collection_loaded', False)
            controller.update_session_state('current_collection_name', selected_collection)
            controller.update_session_state('current_db_path', validated_db_path)
            controller.update_session_state('embedding_model', None)
        
        # 컬렉션 로드 버튼 처리
        if not st.session_state.collection_loaded and collection_result['load_button_clicked']:
            with view.show_spinner("컬렉션을 로드하는 중..."):
                load_result = controller.handle_collection_load(selected_collection, validated_db_path)
                
                if load_result['success']:
                    view.render_loading_messages(load_result['messages'])
                else:
                    view.show_error(load_result['error'])
        
        # 사이드바에서 컬렉션이 로드된 경우 정보 표시
        if st.session_state.collection_loaded:
            collection_info = controller.get_collection_info()
            view.render_sidebar_collection_info(selected_collection, collection_info, validated_db_path)
    
    # 메인 탭 생성
    tabs = view.render_tabs()
    
    # 컬렉션이 없는 경우 모든 탭에 에러 메시지
    if not available_collections:
        for tab in tabs:
            with tab:
                view.show_no_collections_message(validated_db_path)
    else:
        # 컬렉션이 로드되지 않은 경우 안내 메시지
        if not st.session_state.collection_loaded:
            for tab in tabs:
                with tab:
                    view.show_load_collection_message()
        else:
            # 탭 1: 컬렉션 데이터
            with tabs[0]:
                if view.render_collection_data_tab(selected_collection):
                    with view.show_spinner("데이터를 가져오는 중..."):
                        data_result = controller.handle_collection_data_load()
                        
                        if data_result['success']:
                            view.render_collection_data(data_result['data'], data_result['count'])
                        else:
                            view.show_error(data_result['error'])
            
            # 탭 2: 텍스트 검색
            with tabs[1]:
                # 삭제 성공 메시지 표시
                view.show_delete_success_message()
                
                # 검색 설정
                search_settings = view.render_search_settings()
                
                # 검색 입력
                search_input = view.render_search_input()
                
                # 검색 실행
                if search_input['search_clicked'] and search_input['query']:
                    with view.show_spinner("검색 중..."):
                        search_result = controller.handle_search(
                            search_input['query'],
                            search_settings['similarity_threshold'],
                            search_settings['use_search_for_viz']
                        )
                        
                        if search_result['success']:
                            result_df = search_result['data']
                            
                            # 시각화용 검색 결과 메시지
                            if search_result['use_for_viz'] and not result_df.empty:
                                view.show_success(f"{len(result_df)}개의 문서가 시각화를 위해 준비되었습니다. '시각화' 탭으로 이동하세요.")
                            
                            # 검색 결과 표시
                            selected_docs = view.render_search_results(result_df)
                            
                            # 선택한 문서 삭제 처리
                            if selected_docs is not None and not selected_docs.empty:
                                with view.show_spinner(f"{len(selected_docs)}개 문서 삭제 중..."):
                                    delete_result = controller.handle_document_deletion(selected_docs)
                                    
                                    if delete_result['success']:
                                        st.rerun()
                                    else:
                                        view.show_error(delete_result['error'])
                        else:
                            view.show_error(search_result['error'])
                
                # 이전 검색 결과 재표시
                elif st.session_state.last_search_results is not None:
                    selected_docs = view.render_search_results(st.session_state.last_search_results)
                    
                    # 선택한 문서 삭제 처리
                    if selected_docs is not None and not selected_docs.empty:
                        with view.show_spinner(f"{len(selected_docs)}개 문서 삭제 중..."):
                            delete_result = controller.handle_document_deletion(selected_docs)
                            
                            if delete_result['success']:
                                st.rerun()
                            else:
                                view.show_error(delete_result['error'])
            
            # 탭 3: 시각화
            with tabs[2]:
                st.subheader(f"컬렉션 시각화: {selected_collection}")
                
                # 컬렉션의 전체 문서 수 가져오기
                try:
                    total_docs = st.session_state.chroma_collection.count()
                except:
                    total_docs = 0
                
                # 검색 결과 시각화 사용 가능 여부 확인
                search_results_available = ('search_results_for_viz' in st.session_state and 
                                          st.session_state.search_results_for_viz is not None)
                
                # 시각화 설정
                viz_settings = view.render_visualization_settings(
                    use_search_results=False,
                    search_results_available=search_results_available,
                    total_docs=total_docs
                )
                
                # 세션 상태 업데이트
                controller.update_session_state('perplexity', viz_settings['perplexity'])
                controller.update_session_state('max_words_wc_slider', viz_settings['max_words_wc'])
                controller.update_session_state('lda_topics', viz_settings['lda_topics'])
                
                # 시각화 생성 버튼
                if view.render_visualization_button():
                    with view.show_spinner("시각화를 생성하는 중... 이 작업은 데이터 크기에 따라 시간이 걸릴 수 있습니다."):
                        viz_result = controller.handle_visualization_creation(
                            viz_settings['use_search_results'],
                            viz_settings['docs_percentage'],
                            viz_settings['find_optimal'],
                            viz_settings['max_clusters'],
                            viz_settings['n_clusters'],
                            viz_settings['perplexity']
                        )
                        
                        if viz_result['success']:
                            # 성공 메시지들 표시
                            view.render_loading_messages(viz_result['messages'])
                            
                            # 최적 클러스터 분석 결과 표시
                            if viz_result['optimal_result']:
                                optimal = viz_result['optimal_result']
                                view.render_optimal_cluster_analysis(
                                    optimal['silhouette_df'],
                                    optimal['optimal_clusters']
                                )
                            
                            st.rerun()
                        else:
                            view.show_error(viz_result['error'])
                
                # 이미 완료된 시각화 표시
                elif st.session_state.get('viz_completed', False):
                    # 최적 클러스터 수 분석 결과 표시 (있는 경우)
                    if st.session_state.get('show_optimal_cluster_analysis', False):
                        silhouette_df = st.session_state.get('silhouette_df_for_plot')
                        optimal_clusters = st.session_state.get('optimal_clusters_info')
                        if silhouette_df is not None and optimal_clusters is not None:
                            view.render_optimal_cluster_analysis(silhouette_df, optimal_clusters)
                    
                    # 시각화 데이터 표시
                    viz_data = st.session_state.get('viz_data')
                    n_clusters = st.session_state.get('n_clusters')
                    
                    if viz_data is not None and n_clusters is not None:
                        max_words_wc = st.session_state.get('max_words_wc_slider', 100)
                        lda_result = view.render_visualizations(viz_data, n_clusters, max_words_wc)
                        
                        # LDA 토픽 모델링 처리
                        if lda_result and lda_result['run_lda_clicked']:
                            with view.show_spinner("LDA 토픽 모델링 중..."):
                                lda_processing_result = controller.handle_lda_topic_modeling(
                                    lda_result['selected_cluster'],
                                    lda_result['lda_topics']
                                )
                                
                                if lda_processing_result['success']:
                                    view.render_lda_results(lda_processing_result['results'])
                                    view.show_success("LDA 토픽 모델링이 완료되었습니다.")
                                else:
                                    view.show_error(lda_processing_result['error'])
                    else:
                        view.show_warning("시각화 데이터가 세션에 없습니다. 먼저 '시각화 생성'을 실행하세요.")

if __name__ == "__main__":
    main()