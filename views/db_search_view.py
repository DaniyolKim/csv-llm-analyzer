import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import utils.db_search_utils as db_search_utils
import utils.visualization_utils as visualization_utils
from utils import KOREAN_STOPWORDS

class DbSearchView:
    """DB Search 페이지의 뷰 컴포넌트"""
    
    def __init__(self):
        self.setup_page_config()
    
    def setup_page_config(self):
        """페이지 설정"""
        st.set_page_config(
            page_title="DB 검색",
            page_icon="🔍",
            layout="wide"
        )
    
    def render_title(self):
        """제목 렌더링"""
        st.title("DB 검색")
    
    def render_sidebar_db_settings(self, db_path: str) -> Dict[str, Any]:
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
            
            return {'db_path': new_db_path}
    
    def render_hardware_settings(self, gpu_info: Dict[str, Any], 
                                device_options: Dict[str, str],
                                current_preference: str) -> str:
        """하드웨어 설정 렌더링"""
        with st.sidebar:
            st.header("하드웨어 설정")
            
            if gpu_info["available"]:
                st.success(f"✅ GPU 사용 가능: {gpu_info['count']}개의 GPU 감지됨.")
                for i, gpu_device in enumerate(gpu_info["devices"]):
                    st.markdown(f"  - GPU {i}: {gpu_device['name']} (메모리: {gpu_device['memory_total']:.2f} GB)")
                
                current_index = list(device_options.values()).index(current_preference) if current_preference in device_options.values() else 0
                
                selected_device_label = st.radio(
                    "임베딩 연산 장치 선택",
                    options=list(device_options.keys()),
                    index=current_index,
                    help="임베딩 계산에 사용할 장치를 선택합니다. '자동'은 GPU가 있으면 GPU를, 없으면 CPU를 사용합니다."
                )
                
                return device_options[selected_device_label]
            else:
                st.info("ℹ️ 사용 가능한 GPU가 감지되지 않았습니다. 임베딩 연산은 CPU를 사용합니다.")
                return "cpu"
    
    def render_collection_selection(self, available_collections: List[str], 
                                  selected_collection: Optional[str]) -> Dict[str, Any]:
        """컬렉션 선택 UI 렌더링 (메인 컨텐츠용 - 사용 안함)"""
        if not available_collections:
            st.error("사용 가능한 컬렉션이 없습니다.")
            return {
                'has_collections': False,
                'selected_collection': None,
                'load_button_clicked': False
            }
        
        new_selected_collection = st.selectbox(
            "컬렉션 선택",
            options=available_collections,
            index=0 if available_collections else None,
            help="검색할 ChromaDB 컬렉션을 선택하세요."
        )
        
        load_button = st.button("컬렉션 로드", key="load_collection_btn")
        
        return {
            'has_collections': True,
            'selected_collection': new_selected_collection,
            'load_button_clicked': load_button
        }
    
    def render_sidebar_collection_selection(self, available_collections: List[str], 
                                          selected_collection: Optional[str]) -> Dict[str, Any]:
        """사이드바에서 컬렉션 선택 UI 렌더링"""
        with st.sidebar:
            st.header("컬렉션 설정")
            
            if not available_collections:
                st.error("사용 가능한 컬렉션이 없습니다.")
                return {
                    'has_collections': False,
                    'selected_collection': None,
                    'load_button_clicked': False
                }
            
            new_selected_collection = st.selectbox(
                "컬렉션 선택",
                options=available_collections,
                index=0 if available_collections else None,
                help="검색할 ChromaDB 컬렉션을 선택하세요."
            )
            
            load_button = st.button("컬렉션 로드", key="load_collection_btn", type="primary")
            
            return {
                'has_collections': True,
                'selected_collection': new_selected_collection,
                'load_button_clicked': load_button
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
                
                # 현재 로드된 임베딩 모델 상태 표시
                st.write(f"**현재 사용 중인 임베딩 모델:** {collection_info['current_model']}")
                st.write(f"**사용 장치:** {collection_info['device_used']} (요청: {collection_info['device_preference']})")
                
                if collection_info['fallback_used']:
                    st.warning("모델 로드 시 폴백(fallback)이 사용되었습니다.")
                    if collection_info['error_message']:
                        st.warning(f"  - 원인: {collection_info['error_message']}")
            else:
                st.error(collection_info['error'])
    
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
                    
                    # 현재 로드된 임베딩 모델 상태 표시
                    st.write(f"**사용 모델:** {collection_info['current_model']}")
                    st.write(f"**사용 장치:** {collection_info['device_used']}")
                    
                    if collection_info['fallback_used']:
                        st.warning("⚠️ 모델 로드 시 폴백 사용됨")
                        if collection_info['error_message']:
                            st.caption(f"원인: {collection_info['error_message']}")
                else:
                    st.error(collection_info['error'])
    
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
    
    def render_tabs(self) -> List:
        """메인 탭 렌더링"""
        return st.tabs(["컬렉션 데이터", "텍스트 검색", "시각화"])
    
    def render_collection_data_tab(self, selected_collection: str) -> bool:
        """컬렉션 데이터 탭 UI 렌더링"""
        st.subheader(f"컬렉션: {selected_collection}")
        return st.button("데이터 표시", key="show_data_btn")
    
    def render_search_settings(self) -> Dict[str, Any]:
        """검색 설정 UI 렌더링"""
        with st.expander("검색 설정", expanded=True):
            similarity_threshold = st.slider(
                "유사도 임계값",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                format="%.2f",
                help="이 값 이상의 유사도를 가진 문서만 표시합니다. 값이 클수록 검색어와 더 유사한 문서만 표시됩니다."
            )
            
            use_search_for_viz = st.checkbox(
                "검색 결과를 시각화에 사용",
                value=False,
                help="체크하면 검색 결과를 시각화 탭에서 사용할 수 있습니다."
            )
        
        return {
            'similarity_threshold': similarity_threshold,
            'use_search_for_viz': use_search_for_viz
        }
    
    def render_search_input(self) -> Dict[str, Any]:
        """검색 입력 UI 렌더링"""
        query = st.text_input("검색어를 입력하세요", key="search_query")
        search_button = st.button("검색", type="primary")
        
        return {
            'query': query,
            'search_clicked': search_button
        }
    
    def render_search_results(self, result_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """검색 결과 표시 및 선택된 문서 반환"""
        return db_search_utils.display_search_results(result_df)
    
    def render_collection_data(self, result_df: pd.DataFrame, count: int):
        """컬렉션 데이터 표시"""
        st.success(f"총 {count}개의 문서를 로드했습니다.")
        db_search_utils.display_collection_data(result_df)
    
    def render_visualization_settings(self, use_search_results: bool, 
                                    search_results_available: bool,
                                    total_docs: int) -> Dict[str, Any]:
        """시각화 설정 UI 렌더링"""
        with st.expander("시각화 설정", expanded=True):
            # 검색 결과 사용 여부
            if search_results_available:
                search_results_info = st.session_state.search_results_for_viz
                search_df = search_results_info['df']
                use_search_results = st.checkbox(
                    f"검색 결과 시각화 ('{search_results_info['query']}' 검색 결과 {len(search_df)}개 문서)",
                    value=True,
                    help="체크하면 검색 결과를 시각화합니다. 체크 해제하면 전체 컬렉션에서 시각화 데이터를 생성합니다."
                )
            
            # 문서 비율 설정
            if not (use_search_results and search_results_available):
                docs_percentage = st.slider(
                    "사용할 문서 비율(%)",
                    min_value=1,
                    max_value=100,
                    value=20,
                    step=1,
                    key="docs_percentage_slider",
                    help="전체 문서 중 시각화에 사용할 문서의 비율을 설정합니다."
                )
                
                st.markdown(f"<p style='margin-top:-15px; font-size:0.85em;'>선택된 문서 수: {max(1, int(total_docs * docs_percentage / 100))}개 (전체 {total_docs}개 중)</p>", 
                          unsafe_allow_html=True)
            else:
                docs_percentage = 100
                search_df = st.session_state.search_results_for_viz['df']
                st.markdown(f"<p style='margin-top:-15px; font-size:0.85em;'>검색 결과 문서 수: {len(search_df)}개</p>", 
                          unsafe_allow_html=True)
            
            # 최적 클러스터 수 찾기
            find_optimal = st.checkbox(
                "최적 클러스터 수 자동 찾기",
                value=False,
                help="실루엣 스코어를 사용하여 최적의 클러스터 수를 자동으로 찾습니다."
            )
            
            # 클러스터 수 설정
            if find_optimal:
                max_clusters = st.slider(
                    "최대 클러스터 수",
                    min_value=3,
                    max_value=20,
                    value=10,
                    step=1,
                    help="검색할 최대 클러스터 수를 설정합니다."
                )
                n_clusters = st.slider(
                    "클러스터 수",
                    min_value=2,
                    max_value=20,
                    value=5,
                    step=1,
                    help="자동 찾기 옵션이 켜져 있어 이 설정은 무시됩니다.",
                    disabled=True
                )
            else:
                n_clusters = st.slider(
                    "클러스터 수",
                    min_value=2,
                    max_value=20,
                    value=st.session_state.get('n_clusters', 5),
                    step=1,
                    help="문서를 그룹화할 클러스터의 수를 설정합니다."
                )
                max_clusters = None
            
            # t-SNE 복잡도 설정
            perplexity = st.slider(
                "t-SNE 복잡도(Perplexity)",
                min_value=5,
                max_value=50,
                value=st.session_state.get('perplexity', 30),
                step=5,
                help="t-SNE 알고리즘의 핵심 매개변수로, 각 데이터 포인트 주변의 '유효 이웃 수'를 결정합니다."
            )
            
            # WordCloud 최대 단어 수
            max_words_wc = st.slider(
                "WordCloud 최대 단어 수",
                min_value=20,
                max_value=200,
                value=st.session_state.get('max_words_wc_slider', 100),
                step=10,
                help="WordCloud에 표시할 최대 단어 수를 설정합니다."
            )
            
            # LDA 토픽 수
            lda_topics = st.slider(
                "LDA 토픽 수",
                min_value=2,
                max_value=10,
                value=st.session_state.get('lda_topics', 6),
                step=1,
                help="LDA 토픽 모델링에서 사용할 토픽의 수를 설정합니다."
            )
        
        return {
            'use_search_results': use_search_results,
            'docs_percentage': docs_percentage,
            'find_optimal': find_optimal,
            'max_clusters': max_clusters,
            'n_clusters': n_clusters,
            'perplexity': perplexity,
            'max_words_wc': max_words_wc,
            'lda_topics': lda_topics
        }
    
    def render_visualization_button(self) -> bool:
        """시각화 생성 버튼 렌더링"""
        return st.button("시각화 생성", key="create_viz_btn", type="primary")
    
    def render_optimal_cluster_analysis(self, silhouette_df, optimal_clusters):
        """최적 클러스터 수 분석 결과 표시"""
        st.subheader("최적 클러스터 수 분석")
        visualization_utils.plot_elbow_method(silhouette_df)
        visualization_utils.display_optimal_cluster_info(optimal_clusters)
    
    def render_visualizations(self, viz_data, n_clusters: int, max_words_wc: int):
        """시각화 렌더링"""
        # 시각화 기능을 탭으로 분리
        viz_tabs = st.tabs(["클러스터 시각화", "주요 문서", "워드클라우드", "LDA 토픽 모델링"])
        
        # 탭 1: 클러스터 시각화
        with viz_tabs[0]:
            st.subheader("문서 클러스터 시각화")
            visualization_utils.create_cluster_visualization(viz_data, n_clusters)
        
        # 탭 2: 클러스터별 주요 문서
        with viz_tabs[1]:
            st.subheader("클러스터별 주요 문서")
            visualization_utils.display_cluster_documents(viz_data, n_clusters)
        
        # 탭 3: 클러스터별 워드클라우드
        with viz_tabs[2]:
            st.subheader("클러스터별 주요 단어 (WordCloud)")
            visualization_utils.display_cluster_wordclouds(viz_data, n_clusters, KOREAN_STOPWORDS, max_words_wc=max_words_wc)
        
        # 탭 4: LDA 토픽 모델링
        with viz_tabs[3]:
            return self.render_lda_tab(n_clusters)
    
    def render_lda_tab(self, n_clusters: int) -> Dict[str, Any]:
        """LDA 토픽 모델링 탭 렌더링"""
        st.subheader("클러스터별 LDA 토픽 모델링")
        
        # 클러스터 선택 드롭다운
        cluster_options = ["모든 클러스터"] + [f"클러스터 {i}" for i in range(n_clusters)]
        selected_cluster = st.selectbox(
            "분석할 클러스터 선택",
            options=cluster_options,
            index=0,
            key="lda_cluster_select"
        )
        
        lda_topics = st.session_state.get('lda_topics', 6)
        
        st.write("lambda=1일 때는 빈도 기반, lambda=0일 때는 토픽 내 특이성 기반으로 단어를 정렬합니다. 0.6 ~ 0.8 사이의 값을 추천합니다.")
        
        run_lda = st.button("LDA 토픽 모델링 실행", key="run_lda_btn", type="primary")
        
        return {
            'selected_cluster': selected_cluster,
            'lda_topics': lda_topics,
            'run_lda_clicked': run_lda
        }
    
    def render_lda_results(self, results: List[Dict[str, Any]]):
        """LDA 결과 표시"""
        for result in results:
            if result['success']:
                st.write(f"클러스터 {result['cluster_id']} LDA 토픽 분석 ({result['doc_count']}개 문서)")
            else:
                st.write(f"클러스터 {result['cluster_id']}: {result['reason']}")
    
    def show_spinner(self, message: str = "처리 중..."):
        """스피너 표시"""
        return st.spinner(message)
    
    def show_error(self, message: str):
        """에러 메시지 표시"""
        st.error(message)
    
    def show_success(self, message: str):
        """성공 메시지 표시"""
        st.success(message)
    
    def show_warning(self, message: str):
        """경고 메시지 표시"""
        st.warning(message)
    
    def show_info(self, message: str):
        """정보 메시지 표시"""
        st.info(message)
    
    def show_no_collections_message(self, db_path: str):
        """컬렉션 없음 메시지 표시"""
        st.error(f"선택한 경로({db_path})에 사용 가능한 컬렉션이 없습니다. 먼저 CSV 파일을 업로드하고 DB에 저장해주세요.")
    
    def show_load_collection_message(self):
        """컬렉션 로드 안내 메시지 표시"""
        st.info("사이드바에서 컬렉션을 로드하세요.")
    
    def show_delete_success_message(self):
        """삭제 성공 메시지 표시 (세션 상태에서)"""
        if st.session_state.get('delete_success_message'):
            st.success(st.session_state.delete_success_message)
            st.session_state.delete_success_message = None