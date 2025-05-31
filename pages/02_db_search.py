import streamlit as st
import pandas as pd
import os
import sys

# 상위 디렉토리를 경로에 추가합니다
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from chroma_utils import load_chroma_collection, get_available_collections
from text_utils import KOREAN_STOPWORDS

# 모듈 임포트 방식 변경
import db_search_utils
# 시각화 모듈 직접 임포트
import visualization_utils

st.set_page_config(
    page_title="DB 검색",
    page_icon="🔍",
    layout="wide"
)

# 세션 상태 초기화
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'chroma_collection' not in st.session_state:
    st.session_state.chroma_collection = None
if 'collection_loaded' not in st.session_state:
    st.session_state.collection_loaded = False
if 'current_collection_name' not in st.session_state:
    st.session_state.current_collection_name = None
if 'current_db_path' not in st.session_state:
    st.session_state.current_db_path = None

st.title("DB 검색")

# 사이드바 설정
def setup_sidebar():
    with st.sidebar:
        st.header("DB 설정")
        
        # ChromaDB 경로 설정
        default_db_path = "./chroma_db"
        db_path = st.text_input(
            "ChromaDB 경로",
            value=default_db_path,
            help="ChromaDB가 저장된 경로를 입력하세요. 기본값은 './chroma_db'입니다."
        )
        
        # 경로가 존재하는지 확인
        if not os.path.exists(db_path):
            st.warning(f"입력한 경로({db_path})가 존재하지 않습니다. 기본 경로를 사용합니다.")
            db_path = default_db_path
        
        # 사용 가능한 컬렉션 목록 가져오기
        collections = get_available_collections(persist_directory=db_path)
        
        if not collections:
            st.error(f"'{db_path}' 경로에 사용 가능한 컬렉션이 없습니다.")
            selected_collection = None
            # 컬렉션이 없으면 세션 상태 초기화
            st.session_state.collection_loaded = False
            st.session_state.chroma_client = None
            st.session_state.chroma_collection = None
        else:
            selected_collection = st.selectbox(
                "컬렉션 선택",
                options=collections,
                index=0 if collections else None,
                help="검색할 ChromaDB 컬렉션을 선택하세요."
            )
            
            # 컬렉션이나 경로가 변경되면 세션 상태 업데이트
            if (selected_collection != st.session_state.current_collection_name or 
                db_path != st.session_state.current_db_path):
                st.session_state.collection_loaded = False
                st.session_state.current_collection_name = selected_collection
                st.session_state.current_db_path = db_path
                
        # 컬렉션 로드 버튼
        if selected_collection and not st.session_state.collection_loaded:        
            if st.button("컬렉션 로드", key="load_collection_btn"):
                with st.spinner("컬렉션을 로드하는 중..."):
                    try:
                        client, collection = load_chroma_collection(
                            collection_name=selected_collection,
                            persist_directory=db_path
                        )
                        st.session_state.chroma_client = client
                        st.session_state.chroma_collection = collection
                        st.session_state.collection_loaded = True
                        st.success(f"컬렉션 '{selected_collection}'을 성공적으로 로드했습니다.")
                    except Exception as e:
                        st.error(f"컬렉션 로드 중 오류 발생: {e}")
        
        # 컬렉션이 로드된 경우 상태 표시
        if st.session_state.collection_loaded:
            # 컬렉션 정보 표시
            with st.expander("컬렉션 정보"):
                try:
                    # 이미 로드된 컬렉션 사용
                    collection = st.session_state.chroma_collection
                    collection_info = collection.count()
                    
                    # 컬렉션에 저장된 임베딩 모델 정보 확인
                    embedding_model = "알 수 없음"
                    try:
                        if collection.metadata and "embedding_model" in collection.metadata:
                            embedding_model = collection.metadata["embedding_model"]
                    except:
                        pass
                    
                    st.write(f"컬렉션 이름: {selected_collection}")
                    st.write(f"문서 수: {collection_info}")
                    st.write(f"임베딩 모델: {embedding_model}")
                    st.write(f"DB 경로: {db_path}")
                except Exception as e:
                    st.error(f"컬렉션 정보를 가져오는 중 오류가 발생했습니다: {str(e)}")
    
    return db_path, selected_collection, collections

# 컬렉션 데이터 탭 UI
def render_collection_data_tab(selected_collection):
    st.subheader(f"컬렉션: {selected_collection}")
    
    # 데이터 로드 버튼
    if st.button("데이터 표시", key="show_data_btn"):
        with st.spinner("데이터를 가져오는 중..."):
            try:
                # 이미 로드된 컬렉션 사용
                collection = st.session_state.chroma_collection
                
                # 컬렉션 데이터 로드
                result_df, _ = db_search_utils.load_collection_data(collection)
                
                if result_df is not None:
                    st.success(f"총 {len(result_df)}개의 문서를 로드했습니다.")
                    db_search_utils.display_collection_data(result_df)
                else:
                    st.info("컬렉션에 데이터가 없습니다.")
            except Exception as e:
                st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")

# 텍스트 검색 탭 UI
def render_search_tab():
    # 검색 설정
    with st.expander("검색 설정", expanded=True):
        # 검색 결과 수 설정
        n_results = st.slider(
            "검색 결과 수",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="반환할 검색 결과의 최대 개수를 설정합니다."
        )
    
    # 검색 입력 필드
    query = st.text_input("검색어를 입력하세요", key="search_query")
    
    # 검색 버튼
    search_button = st.button("검색", type="primary")
    
    # 검색 결과 저장 세션 상태
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = None
    
    # 삭제 성공 메시지 세션 상태
    if 'delete_success_message' not in st.session_state:
        st.session_state.delete_success_message = None
    
    # 삭제 성공 메시지 표시 (있을 경우)
    if st.session_state.delete_success_message:
        st.success(st.session_state.delete_success_message)
        st.session_state.delete_success_message = None
    
    # 검색 실행
    if search_button and query:
        with st.spinner("검색 중..."):
            try:
                # 이미 로드된 컬렉션 사용
                collection = st.session_state.chroma_collection
                
                # 검색 실행
                result_df = db_search_utils.search_collection(collection, query, n_results=n_results)
                
                # 검색 결과 저장 (삭제 기능을 위해)
                st.session_state.last_search_results = result_df
                
                # 검색 결과 표시
                selected_docs = db_search_utils.display_search_results(result_df)
                
                # 선택한 문서 삭제 처리
                if selected_docs is not None and not selected_docs.empty:
                    delete_selected_documents(selected_docs)
                
            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
    elif st.session_state.last_search_results is not None:
        # 이전 검색 결과 재표시
        selected_docs = db_search_utils.display_search_results(st.session_state.last_search_results)
        
        # 선택한 문서 삭제 처리
        if selected_docs is not None and not selected_docs.empty:
            delete_selected_documents(selected_docs)

# 선택한 문서를 삭제하는 함수
def delete_selected_documents(selected_docs):
    """선택한 문서를 컬렉션에서 삭제하는 함수"""
    try:
        # 이미 로드된 컬렉션 사용
        collection = st.session_state.chroma_collection
        
        # 선택된 문서의 ID 추출 (결과에 ID가 없는 경우 처리)
        if "ID" in selected_docs.columns:
            doc_ids = selected_docs["ID"].tolist()
        else:
            # 선택된 문서의 내용으로 ID 찾기
            docs_to_delete = selected_docs["내용"].tolist()
            # 컬렉션에서 모든 문서 가져와서 내용이 일치하는 것의 ID 찾기
            all_docs, all_data = db_search_utils.load_collection_data(collection)
            if all_docs is not None:
                doc_ids = []
                for doc in docs_to_delete:
                    # 내용이 일치하는 문서의 ID 찾기
                    matching_rows = all_docs[all_docs["내용"] == doc]
                    if not matching_rows.empty:
                        for _, row in matching_rows.iterrows():
                            doc_ids.append(row["ID"])
            else:
                st.error("문서 ID를 찾을 수 없어 삭제할 수 없습니다.")
                return
        
        if not doc_ids:
            st.error("삭제할 문서의 ID를 찾을 수 없습니다.")
            return
        
        # 문서 삭제 확인
        with st.spinner(f"{len(doc_ids)}개 문서 삭제 중..."):
            collection.delete(ids=doc_ids)
            
            # 성공 메시지 설정
            st.session_state.delete_success_message = f"{len(doc_ids)}개 문서가 성공적으로 삭제되었습니다."
            
            # 마지막 검색 결과에서 삭제된 문서 제거
            if st.session_state.last_search_results is not None:
                # 내용 기반으로 삭제된 문서 필터링
                contents_to_delete = selected_docs["내용"].tolist()
                st.session_state.last_search_results = st.session_state.last_search_results[
                    ~st.session_state.last_search_results["내용"].isin(contents_to_delete)
                ]
                
            # 페이지 리프레시 (최신 Streamlit 버전 호환성)
            st.rerun()
            
    except Exception as e:
        st.error(f"문서 삭제 중 오류가 발생했습니다: {str(e)}")

# 시각화 탭 UI
def render_visualization_tab(selected_collection):
    st.subheader(f"컬렉션 시각화: {selected_collection}")
    
    # 시각화 설정
    with st.expander("시각화 설정", expanded=True):
        # 자동 최적 클러스터 수 찾기 옵션
        find_optimal = st.checkbox(
            "최적 클러스터 수 자동 찾기", 
            value=False,
            help="실루엣 스코어를 사용하여 최적의 클러스터 수를 자동으로 찾습니다."
        )
        
        if find_optimal:
            # 최대 클러스터 수 설정
            max_clusters = st.slider(
                "최대 클러스터 수",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                help="검색할 최대 클러스터 수를 설정합니다."
            )
            # 클러스터 수를 비활성화 표시용으로만 설정
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
            # 클러스터 수 설정
            n_clusters = st.slider(
                "클러스터 수",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                help="문서를 그룹화할 클러스터의 수를 설정합니다."
            )
        
        # 최대 문서 수 설정
        max_docs = st.slider(
            "최대 문서 수",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="시각화할 최대 문서 수를 설정합니다. 문서가 많을수록 처리 시간이 길어집니다."
        )
        
        # 차원 축소 방법 선택
        perplexity = st.slider(
            "t-SNE 복잡도(Perplexity)",
            min_value=5,
            max_value=50,
            value=30,
            step=5,
            help="""
            t-SNE 알고리즘의 핵심 매개변수로, 각 데이터 포인트 주변의 '유효 이웃 수'를 결정합니다.
            - 낮은 값(5~10): 지역적 구조 보존, 작은 클러스터 식별에 효과적
            - 높은 값(30~50): 전역적 구조 보존, 데이터 전체 패턴 파악에 유리
            - 일반적으로 10~50 사이 값 권장, 데이터셋 크기에 따라 조정
            
            너무 작은 값: 파편화된 클러스터 발생
            너무 큰 값: 클러스터 간 경계가 모호해짐
            """
        )
        
        # LDA 토픽 수 설정 추가
        lda_topics = st.slider(
            "LDA 토픽 수",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="각 클러스터에서 LDA로 추출할 토픽의 수를 설정합니다. 작은 클러스터의 경우 자동으로 조정됩니다."
        )
    
    # 시각화 버튼
    if st.button("시각화 생성", key="create_viz_btn", type="primary"):
        with st.spinner("시각화를 생성하는 중... 이 작업은 데이터 크기에 따라 시간이 걸릴 수 있습니다."):
            try:
                # 이미 로드된 컬렉션 사용
                collection = st.session_state.chroma_collection
                
                # 컬렉션의 모든 데이터 가져오기
                _, all_data = db_search_utils.load_collection_data(collection)
                
                if all_data and all_data["documents"]:
                    # 결과 표시
                    total_docs = len(all_data["documents"])
                    st.success(f"총 {total_docs}개의 문서를 로드했습니다.")
                    
                    # 임베딩 데이터 가져오기
                    documents, metadatas, ids, embeddings = db_search_utils.get_embeddings_data(collection, all_data, max_docs)
                    
                    # 임베딩이 없는 경우 처리
                    if len(embeddings) == 0:
                        st.error("임베딩 데이터를 가져올 수 없습니다.")
                        embeddings = db_search_utils.handle_missing_embeddings(collection, documents)
                    
                    # 임베딩 배열로 변환
                    import numpy as np
                    embeddings_array = np.array(embeddings)
                    
                    # 최적 클러스터 수 찾기
                    if find_optimal:
                        st.subheader("최적 클러스터 수 분석")
                        with st.spinner("최적 클러스터 수 계산 중..."):
                            silhouette_df, optimal_clusters = db_search_utils.find_optimal_clusters(embeddings_array, max_clusters)
                            
                            # 엘보우 방법 시각화
                            db_search_utils.plot_elbow_method(silhouette_df)
                            
                            # 최적 클러스터 수 정보 표시
                            db_search_utils.display_optimal_cluster_info(optimal_clusters)
                            
                            # 최적 클러스터 수를 사용하도록 설정
                            n_clusters = int(optimal_clusters["클러스터 수"])
                            st.success(f"최적의 클러스터 수로 {n_clusters}을(를) 사용합니다.")
                    
                    # 시각화 데이터 준비 부분 수정
                    viz_data = visualization_utils.prepare_visualization_data(
                        embeddings_array, documents, ids, metadatas, perplexity, n_clusters
                    )
                    
                    # 클러스터 시각화
                    visualization_utils.create_cluster_visualization(viz_data, n_clusters)
                    
                    # 클러스터별 주요 문서 표시
                    visualization_utils.display_cluster_documents(viz_data, n_clusters)
                    
                    # 클러스터별 WordCloud 표시
                    visualization_utils.display_cluster_wordclouds(viz_data, n_clusters, KOREAN_STOPWORDS)
                    
                    # 클러스터별 LDA 토픽 모델링 표시
                    visualization_utils.display_cluster_lda(viz_data, n_clusters, KOREAN_STOPWORDS, lda_topics)
                else:
                    st.info("컬렉션에 데이터가 없습니다.")
            
            except Exception as e:
                st.error(f"시각화 생성 중 오류가 발생했습니다: {str(e)}")
                st.exception(e)

# 메인 앱 실행
def main():
    # 사이드바 설정
    db_path, selected_collection, collections = setup_sidebar()
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["컬렉션 데이터", "텍스트 검색", "시각화"])
    
    # 메인 영역
    if not collections:
        for tab in [tab1, tab2, tab3]:
            with tab:
                st.error(f"선택한 경로({db_path})에 사용 가능한 컬렉션이 없습니다. 먼저 CSV 파일을 업로드하고 DB에 저장해주세요.")
    else:
        # 컬렉션이 로드되지 않은 경우 안내 메시지
        if not st.session_state.collection_loaded:
            for tab in [tab1, tab2, tab3]:
                with tab:
                    st.info("사이드바에서 컬렉션을 로드하세요.")
        else:
            # 탭 1: 컬렉션 데이터 표시
            with tab1:
                render_collection_data_tab(selected_collection)
            
            # 탭 2: DB 검색
            with tab2:
                render_search_tab()
            
            # 탭 3: 시각화
            with tab3:
                render_visualization_tab(selected_collection)

if __name__ == "__main__":
    main()