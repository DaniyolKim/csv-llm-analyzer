import streamlit as st
import pandas as pd
import os
import sys

# 상위 디렉토리를 경로에 추가합니다
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 필요한 유틸리티 함수 임포트 (get_embedding_status 추가)
from utils import load_chroma_collection, get_available_collections, get_embedding_function, get_embedding_status
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
if 'embedding_model' not in st.session_state: # 임베딩 모델 세션 상태 추가
    st.session_state.embedding_model = None
# 시각화 관련 세션 상태 추가
if 'viz_completed' not in st.session_state:
    st.session_state.viz_completed = False
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = None
if 'max_words_wc_slider' not in st.session_state: # WordCloud 최대 단어 수 세션 상태
    st.session_state.max_words_wc_slider = 100
# 최적 클러스터 분석 결과 표시를 위한 세션 상태
if 'show_optimal_cluster_analysis' not in st.session_state:
    st.session_state.show_optimal_cluster_analysis = False
if 'silhouette_df_for_plot' not in st.session_state:
    st.session_state.silhouette_df_for_plot = None
if 'optimal_clusters_info' not in st.session_state:
    st.session_state.optimal_clusters_info = None

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
            st.session_state.embedding_model = None # 임베딩 모델도 초기화
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
                st.session_state.embedding_model = None # 컬렉션 변경 시 임베딩 모델 초기화
                
        # 컬렉션 로드 버튼
        if selected_collection and not st.session_state.collection_loaded:        
            if st.button("컬렉션 로드", key="load_collection_btn"):
                with st.spinner("컬렉션을 로드하는 중..."):
                    try:
                        # load_chroma_collection 함수는 이제 utils.py에 있습니다.
                        # 이 함수 내부에서 임베딩 모델을 로드하고 컬렉션 객체에 연결합니다.
                        # 하지만 st.cache_resource를 사용하기 위해 여기서 직접 임베딩 함수를 가져와 세션에 저장합니다.
                        client, collection = load_chroma_collection(
                            collection_name=selected_collection,
                            persist_directory=db_path
                            # load_chroma_collection 내부에서 임베딩 모델을 로드하므로 여기서는 device_preference를 전달하지 않습니다.
                            # load_chroma_collection 함수가 컬렉션 메타데이터에서 모델 정보를 가져와 로드합니다.
                        )
                        
                        # 컬렉션 메타데이터에서 임베딩 모델 정보 가져오기 (저장 시 사용된 모델)
                        # load_chroma_collection 함수가 이미 임베딩 함수를 컬렉션에 연결했으므로,
                        # 여기서는 해당 임베딩 함수를 가져와 세션 상태에 저장합니다.
                        # collection._embedding_function 속성을 직접 사용하는 것은 내부 구현에 의존하므로 권장되지 않습니다.
                        # 대신, 컬렉션 메타데이터에서 모델 이름을 가져와 get_embedding_function을 다시 호출하는 것이 안전합니다.
                        # get_embedding_function은 st.cache_resource가 적용되어 있으므로 중복 로드되지 않습니다.
                        embedding_model_name = collection.metadata.get("embedding_model", "all-MiniLM-L6-v2") # 기본값
                        
                        # 임베딩 모델 로드 (st.cache_resource 덕분에 한 번만 로드됨)
                        # load_chroma_collection에서 이미 로드되었을 가능성이 높지만, 안전하게 다시 호출합니다.
                        # load_chroma_collection에서 사용된 device_preference를 알 수 없으므로 'auto'를 사용합니다.
                        # 더 정확하게 하려면 load_chroma_collection에서 사용된 device_preference도 반환하도록 수정해야 합니다.
                        # 현재 구조에서는 컬렉션 로드 시 임베딩 모델 로드와 세션 저장을 분리하는 것이 더 명확합니다.
                        
                        # 임베딩 모델 로드 (st.cache_resource 덕분에 한 번만 로드됨)
                        # 컬렉션 메타데이터에 저장된 모델 이름으로 임베딩 함수를 가져옵니다.
                        # load_chroma_collection 함수는 임베딩 함수를 컬렉션 객체에 연결하지만,
                        # Streamlit 세션 상태에서 직접 관리하기 위해 여기서 다시 가져옵니다.
                        # get_embedding_function은 st.cache_resource가 적용되어 있어 효율적입니다.
                        
                        # 컬렉션 로드 시 임베딩 모델 로드 및 세션 저장
                        # load_chroma_collection 함수는 이제 임베딩 함수를 컬렉션 객체에 연결합니다.
                        # 우리는 이 연결된 임베딩 함수를 세션 상태에 저장하여 재사용할 것입니다.
                        # collection._embedding_function 속성을 직접 사용하는 것은 내부 구현에 의존하므로
                        # 컬렉션 메타데이터에서 모델 이름을 가져와 get_embedding_function을 호출하는 것이 더 안전합니다.
                        
                        # 컬렉션 메타데이터에서 임베딩 모델 이름 가져오기
                        embedding_model_name = collection.metadata.get("embedding_model", "all-MiniLM-L6-v2") # 기본값
                        
                        # 임베딩 모델 로드 (st.cache_resource 덕분에 한 번만 로드됨)
                        # load_chroma_collection에서 사용된 device_preference를 알 수 없으므로 'auto'를 사용합니다.
                        # 만약 load_chroma_collection에서 특정 device_preference를 사용했다면,
                        # 해당 정보를 메타데이터에 저장하거나 load_chroma_collection에서 반환하도록 수정해야 합니다.
                        # 현재는 'auto'로 가정하고 로드합니다.
                        embedding_model_func = get_embedding_function(embedding_model_name, device_preference="auto")
                        
                        if embedding_model_func is None:
                             st.error("임베딩 모델 로드에 실패했습니다. 애플리케이션을 다시 시작하거나 다른 모델을 시도하세요.")
                             # 로드 실패 시 세션 상태 초기화
                             st.session_state.chroma_client = None
                             st.session_state.chroma_collection = None
                             st.session_state.collection_loaded = False
                             st.session_state.embedding_model = None
                             return # 함수 종료
                             
                        st.session_state.chroma_client = client
                        st.session_state.chroma_collection = collection
                        st.session_state.embedding_model = embedding_model_func # 로드된 임베딩 함수 저장
                        st.session_state.collection_loaded = True
                        st.success(f"컬렉션 '{selected_collection}'을 성공적으로 로드했습니다.")
                        
                        # 임베딩 모델 로드 상태 표시
                        embedding_status = get_embedding_function(embedding_model_name, device_preference="auto", use_cache=True) # 상태 확인용으로 다시 호출 (캐시 사용)
                        status_info = get_embedding_status()
                        if status_info["fallback_used"]:
                             st.warning(f"""
                             ⚠️ **임베딩 모델 변경됨**: 요청하신 모델 대신 기본 임베딩 모델이 사용되었습니다.
                             - 요청 모델: {status_info["requested_model"]}
                             - 사용된 모델: {status_info["actual_model"]}
                             - 사용된 장치: {status_info["device_used"]} (요청: {status_info["device_preference"]})
                             """)
                             if status_info["error_message"]:
                                 st.warning(f"- 원인: {status_info['error_message']}")
                        else:
                             st.info(f"✅ 임베딩 모델 로드됨: {status_info['actual_model']} (장치: {status_info['device_used']})")


                    except Exception as e:
                        st.error(f"컬렉션 로드 중 오류 발생: {e}")
                        # 오류 발생 시 세션 상태 초기화
                        st.session_state.chroma_client = None
                        st.session_state.chroma_collection = None
                        st.session_state.collection_loaded = False
                        st.session_state.embedding_model = None
        
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
                    
                    # 현재 로드된 임베딩 모델 상태 표시 (세션 상태 기준)
                    if st.session_state.embedding_model:
                         status_info = get_embedding_status() # utils에서 상태 가져옴
                         st.write(f"**현재 사용 중인 임베딩 모델:** {status_info.get('actual_model', 'N/A')}")
                         st.write(f"**사용 장치:** {status_info.get('device_used', 'N/A')} (요청: {status_info.get('device_preference', 'N/A')})")
                         if status_info.get('fallback_used'):
                             st.warning("모델 로드 시 폴백(fallback)이 사용되었습니다.")
                             if status_info.get('error_message'):
                                 st.warning(f"  - 원인: {status_info['error_message']}")
                    else:
                         st.warning("임베딩 모델이 로드되지 않았습니다.")
                         
                except Exception as e:
                    st.error(f"컬렉션 정보를 가져오는 중 오류가 발생했습니다: {str(e)}")
    
    return db_path, selected_collection, collections

# 컬렉션 데이터 탭 UI
def render_collection_data_tab(selected_collection):
    st.subheader(f"컬렉션: {selected_collection}")
    
    # 데이터 로드 버튼
    if st.button("데이터 표시", key="show_data_btn"):
        if not st.session_state.collection_loaded or st.session_state.chroma_collection is None:
            st.error("먼저 사이드바에서 컬렉션을 로드하세요.")
            return
            
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
        # 검색 결과 수 대신 유사도 임계값 설정
        similarity_threshold = st.slider(
            "유사도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.2f",
            help="이 값 이상의 유사도를 가진 문서만 표시합니다. 값이 클수록 검색어와 더 유사한 문서만 표시됩니다."
        )
        
        # 검색 결과를 시각화에 사용할지 여부
        use_search_for_viz = st.checkbox(
            "검색 결과를 시각화에 사용",
            value=False,
            help="체크하면 검색 결과를 시각화 탭에서 사용할 수 있습니다."
        )
    
    # 검색 입력 필드
    query = st.text_input("검색어를 입력하세요", key="search_query")
    
    # 검색 버튼
    search_button = st.button("검색", type="primary")
    
    # 검색 결과 저장 세션 상태
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = None
    
    # 검색 결과 임베딩 저장 세션 상태 (시각화에 사용)
    if 'search_results_for_viz' not in st.session_state:
        st.session_state.search_results_for_viz = None
    
    # 삭제 성공 메시지 세션 상태
    if 'delete_success_message' not in st.session_state:
        st.session_state.delete_success_message = None
    
    # 삭제 성공 메시지 표시 (있을 경우)
    if st.session_state.delete_success_message:
        st.success(st.session_state.delete_success_message)
        st.session_state.delete_success_message = None
    
    # 검색 실행
    if search_button and query:
        if not st.session_state.collection_loaded or st.session_state.chroma_collection is None:
            st.error("먼저 사이드바에서 컬렉션을 로드하세요.")
            return
        
        # 세션 상태에서 임베딩 함수 가져오기
        embed_fn = st.session_state.embedding_model
        if embed_fn is None:
             st.error("임베딩 모델이 로드되지 않았습니다. 컬렉션을 다시 로드해 보세요.")
             return
             
        with st.spinner("검색 중..."):
            try:
                # 이미 로드된 컬렉션 사용
                collection = st.session_state.chroma_collection
                
                # 유사도 기반 검색 실행
                # 시각화에 사용할 경우 임베딩도 함께 가져옴
                if use_search_for_viz:
                    # 새로운 함수 사용 - 모든 문서 대상 검색 (1000개 제한 해결)
                    # 임베딩 모델을 인자로 전달
                    result_df, embeddings = db_search_utils.search_collection_by_similarity_full(
                        collection, query, similarity_threshold, include_embeddings=True
                        , embed_fn=embed_fn # 임베딩 함수 전달
                    )
                    # 검색 결과 및 임베딩 저장 (시각화에 사용)
                    st.session_state.search_results_for_viz = {
                        'query': query,
                        'df': result_df,
                        'embeddings': embeddings,
                        'threshold': similarity_threshold
                    }
                    # 검색 결과 시각화 준비 완료 메시지
                    if not result_df.empty:
                        st.success(f"{len(result_df)}개의 문서가 시각화를 위해 준비되었습니다. '시각화' 탭으로 이동하세요.")
                else:
                    # 새로운 함수 사용 - 모든 문서 대상 검색 (1000개 제한 해결)
                    # 임베딩 모델을 인자로 전달
                    result_df = db_search_utils.search_collection_by_similarity_full(
                        collection, query, similarity_threshold
                        , embed_fn=embed_fn # 임베딩 함수 전달
                    )
                
                # 검색 결과 저장 (삭제 기능을 위해)
                st.session_state.last_search_results = result_df
                
                # 검색 결과 표시
                selected_docs = db_search_utils.display_search_results(result_df)
                
                # 선택한 문서 삭제 처리
                if selected_docs is not None and not selected_docs.empty:
                    delete_selected_documents(selected_docs)
                
            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
                st.exception(e) # 상세 오류 정보 표시
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
        if not st.session_state.collection_loaded or st.session_state.chroma_collection is None:
            st.error("컬렉션이 로드되지 않아 문서를 삭제할 수 없습니다.")
            return
            
        # 이미 로드된 컬렉션 사용
        collection = st.session_state.chroma_collection
        
        # 선택된 문서의 ID 추출 (결과에 ID가 없는 경우 처리)
        if "ID" in selected_docs.columns:
            doc_ids = selected_docs["ID"].tolist()
        else:
            # 선택된 문서의 내용으로 ID 찾기 (비효율적일 수 있음)
            st.warning("검색 결과에 문서 ID가 포함되어 있지 않아 내용 기반으로 ID를 찾습니다. 시간이 오래 걸릴 수 있습니다.")
            docs_to_delete = selected_docs["내용"].tolist()
            # 컬렉션에서 모든 문서 가져와서 내용이 일치하는 것의 ID 찾기
            all_docs, all_data = db_search_utils.load_collection_data(collection)
            if all_docs is not None:
                doc_ids = []
                for doc in docs_to_delete:
                    # 내용이 일치하는 문서의 ID 찾기
                    matching_rows = all_docs[all_docs["내용"] == doc]
                    if not matching_rows.empty:
                        # 중복 ID 방지를 위해 set 사용 후 list로 변환
                        found_ids = matching_rows["ID"].tolist()
                        doc_ids.extend(found_ids)
                doc_ids = list(set(doc_ids)) # 중복 제거
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
                # ID 기반으로 삭제된 문서 필터링
                st.session_state.last_search_results = st.session_state.last_search_results[
                    ~st.session_state.last_search_results["ID"].isin(doc_ids)
                ]
                
            # 페이지 리프레시 (최신 Streamlit 버전 호환성)
            st.rerun()
            
    except Exception as e:
        st.error(f"문서 삭제 중 오류가 발생했습니다: {str(e)}")
        st.exception(e) # 상세 오류 정보 표시

# 시각화 탭 UI
def render_visualization_tab(selected_collection):
    st.subheader(f"컬렉션 시각화: {selected_collection}")
    
    if not st.session_state.collection_loaded or st.session_state.chroma_collection is None:
        st.info("먼저 사이드바에서 컬렉션을 로드하세요.")
        return
        
    # 시각화 설정
    with st.expander("시각화 설정", expanded=True):
        # 검색 결과를 시각화에 사용할지 여부 확인
        use_search_results = False
        search_results_available = 'search_results_for_viz' in st.session_state and st.session_state.search_results_for_viz is not None
        
        if search_results_available:
            search_results_info = st.session_state.search_results_for_viz
            search_df = search_results_info['df']
            use_search_results = st.checkbox(
                f"검색 결과 시각화 ('{search_results_info['query']}' 검색 결과 {len(search_df)}개 문서)",
                value=True,
                help="체크하면 검색 결과를 시각화합니다. 체크 해제하면 전체 컬렉션에서 시각화 데이터를 생성합니다."
            )
        
        # 컬렉션의 전체 문서 수 가져오기
        try:
            collection = st.session_state.chroma_collection
            total_docs = collection.count()
        except:
            total_docs = 0
        
        # 검색 결과를 사용하지 않을 경우에만 문서 비율 슬라이더 표시
        if not (use_search_results and search_results_available):
            # 문서 비율 설정 슬라이더 (백분율) - 라벨에 실제 문서 수 표시
            docs_percentage = st.slider(
                "사용할 문서 비율(%)",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                key="docs_percentage_slider",
                help="전체 문서 중 시각화에 사용할 문서의 비율을 설정합니다. 100%는 모든 문서를 사용합니다. 문서가 많을수록 처리 시간이 길어집니다."
            )
            
            # 슬라이더 값이 변경될 때마다 라벨 업데이트
            st.markdown(f"<p style='margin-top:-15px; font-size:0.85em;'>선택된 문서 수: {max(1, int(total_docs * docs_percentage / 100))}개 (전체 {total_docs}개 중)</p>", unsafe_allow_html=True)
        else:
             # 검색 결과를 사용하는 경우 문서 비율 슬라이더를 숨기거나 비활성화
             docs_percentage = 100 # 검색 결과는 100% 사용으로 간주
             st.markdown(f"<p style='margin-top:-15px; font-size:0.85em;'>검색 결과 문서 수: {len(search_df)}개</p>", unsafe_allow_html=True)


        # 자동 최적 클러스터 수 찾기 옵션
        find_optimal = st.checkbox(
            "최적 클러스터 수 자동 찾기", 
            value=False,
            help="실루엣 스코어를 사용하여 최적의 클러스터 수를 자동으로 찾습니다."
        )
        
        # 클러스터 수 설정 (자동 찾기 옵션에 따라 활성화/비활성화)
        if find_optimal:
            max_clusters = st.slider(
                "최대 클러스터 수",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                help="검색할 최대 클러스터 수를 설정합니다."
            )
            # 클러스터 수를 비활성화 표시용으로만 설정
            n_clusters_input = st.slider(
                "클러스터 수",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                help="자동 찾기 옵션이 켜져 있어 이 설정은 무시됩니다.",
                disabled=True
            )
            # 실제 사용될 n_clusters는 자동 찾기 결과로 결정됨
            n_clusters = None # 초기값 None
        else:
            # 클러스터 수 설정
            n_clusters = st.slider(
                "클러스터 수",
                min_value=2,
                max_value=20,
                value=st.session_state.get('n_clusters', 5), # 세션 상태에서 기본값 가져오기
                step=1,
                help="문서를 그룹화할 클러스터의 수를 설정합니다."
            )
            max_clusters = None # 자동 찾기 사용 안 함
        
        # t-SNE 복잡도 설정
        perplexity = st.slider(
            "t-SNE 복잡도(Perplexity)",
            min_value=5,
            max_value=50,
            value=st.session_state.get('perplexity', 30), # 세션 상태에서 기본값 가져오기
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

        # WordCloud 최대 단어 수 설정
        current_max_words_val = st.session_state.get('max_words_wc_slider', 100)
        max_words_wc_slider = st.slider(
            "WordCloud 최대 단어 수",
            min_value=20,
            max_value=200,
            value=current_max_words_val,
            step=10,
            key="max_words_wc_slider_key",
            help="WordCloud에 표시할 최대 단어 수를 설정합니다. 일반적으로 50-100개가 적절합니다."
        )
        # 세션 상태 업데이트 (슬라이더 값이 변경된 경우)
        if current_max_words_val != max_words_wc_slider:
            st.session_state.max_words_wc_slider = max_words_wc_slider
        
        st.session_state.perplexity = perplexity # 세션 상태에 저장

        # LDA 토픽 수 설정은 여기서 제거하고 아래로 이동
    
    # 시각화 버튼
    if st.button("시각화 생성", key="create_viz_btn", type="primary"):
        # "시각화 생성" 버튼 클릭 시, 이전 분석 결과 및 시각화 관련 세션 상태 초기화
        st.session_state.show_optimal_cluster_analysis = False
        st.session_state.silhouette_df_for_plot = None
        st.session_state.optimal_clusters_info = None
        st.session_state.viz_completed = False # 새로운 시각화가 완료될 때까지 이전 결과 숨김
        st.session_state.viz_data = None
        # n_clusters는 아래 로직에서 find_optimal 여부에 따라 다시 설정되므로 여기서 None으로 초기화
        st.session_state.n_clusters = None 

        with st.spinner("시각화를 생성하는 중... 이 작업은 데이터 크기에 따라 시간이 걸릴 수 있습니다."):
            try:
                # 이미 로드된 컬렉션 사용
                collection = st.session_state.chroma_collection
                
                # 검색 결과를 사용하는 경우와 전체 컬렉션을 사용하는 경우 분리
                if use_search_results and search_results_available:
                    # 검색 결과 정보 가져오기
                    search_results_info = st.session_state.search_results_for_viz
                    search_df = search_results_info['df']
                    search_embeddings = search_results_info['embeddings']
                    
                    if not search_df.empty and len(search_embeddings) > 0:
                        # 검색 결과의 문서 수 표시
                        total_docs_for_viz = len(search_df)
                        st.success(f"검색 결과 '{search_results_info['query']}'에서 {total_docs_for_viz}개의 문서를 시각화합니다.")
                        
                        # 검색 결과에서 필요한 데이터 추출
                        documents = search_df['내용'].tolist()
                        ids = search_df['ID'].tolist()
                        
                        # 메타데이터 구성
                        metadatas = []
                        for _, row in search_df.iterrows():
                            metadata = {'source': row.get('source', '알 수 없음')}
                            if 'chunk' in row:
                                metadata['chunk'] = row['chunk']
                            if 'keywords' in row:
                                metadata['keywords'] = row['keywords']
                            metadatas.append(metadata)
                        
                        # 임베딩 배열로 변환
                        import numpy as np
                        embeddings_array = np.array(search_embeddings)
                        
                        # 최소 필요 문서 수 확인 (클러스터링을 위해)
                        min_docs_for_clustering = 3 # 최소 3개 문서 필요 (t-SNE, KMeans)
                        if total_docs_for_viz < min_docs_for_clustering:
                             st.warning(f"시각화를 위해서는 최소 {min_docs_for_clustering}개 이상의 문서가 필요합니다. 현재 문서 수: {total_docs_for_viz}")
                             st.stop()
                             
                        # 클러스터 수 조정 (문서 수보다 클 수 없음)
                        if find_optimal:
                             # 자동 찾기 시에도 최대 클러스터 수는 문서 수보다 작아야 함
                             max_clusters_actual = min(max_clusters, total_docs_for_viz - 1) if total_docs_for_viz > 1 else 2
                             if max_clusters_actual < 2:
                                 st.warning(f"클러스터링을 위해서는 최소 2개 이상의 클러스터가 필요하지만, 문서 수가 부족하여 자동 찾기를 수행할 수 없습니다. (문서 수: {total_docs_for_viz})")
                                 find_optimal = False # 자동 찾기 비활성화
                                 n_clusters = max(2, total_docs_for_viz // 2) # 기본 클러스터 수 설정
                                 st.info(f"클러스터 수를 {n_clusters}로 자동 조정합니다.")
                             else:
                                 max_clusters = max_clusters_actual # 조정된 최대 클러스터 수 사용
                                 st.info(f"자동 찾기 최대 클러스터 수를 {max_clusters}로 조정합니다.")
                                 
                        # 클러스터 수 조정 (문서 수보다 클 수 없음) - 자동 찾기 사용 안 할 때
                        if not find_optimal and n_clusters >= total_docs_for_viz:
                             st.warning(f"클러스터 수({n_clusters})가 문서 수({total_docs_for_viz})보다 많습니다. 클러스터 수를 문서 수보다 작게 조정합니다.")
                             n_clusters = max(2, total_docs_for_viz // 2) # 문서 수의 절반 또는 최소 2개
                             st.info(f"클러스터 수를 {n_clusters}로 자동 조정합니다.")
                             
                        # 클러스터링을 위한 최소 문서 수 확인
                        if (find_optimal and total_docs_for_viz < 3) or (not find_optimal and total_docs_for_viz < n_clusters):
                             st.warning(f"클러스터링을 위해서는 최소 {n_clusters}개 이상의 문서가 필요합니다. 현재 문서 수: {total_docs_for_viz}")
                             if total_docs_for_viz >= 2:
                                 st.info(f"문서 수에 맞춰 클러스터 수를 {max(2, total_docs_for_viz // 2)}로 조정합니다.")
                                 n_clusters = max(2, total_docs_for_viz // 2)
                             else:
                                 st.error("클러스터링을 수행할 수 없습니다. 문서 수를 늘려주세요.")
                                 st.stop()


                    else:
                        st.error("검색 결과가 없거나 임베딩 데이터를 가져올 수 없습니다.")
                        st.stop()
                else:
                    # 컬렉션의 모든 데이터 가져오기
                    _, all_data = db_search_utils.load_collection_data(collection)
                    
                    if all_data and all_data["documents"]:
                        # 결과 표시
                        total_docs_for_viz = len(all_data["documents"])
                        st.success(f"총 {total_docs_for_viz}개의 문서를 로드했습니다.")
                        
                        # 문서 비율(%) 기반으로 시각화 데이터 가져오기
                        documents, metadatas, ids, embeddings = visualization_utils.get_embeddings_data(collection, all_data, docs_percentage)
                        
                        # 임베딩이 없는 경우 처리
                        if len(embeddings) == 0:
                            st.error("임베딩 데이터를 가져올 수 없습니다.")
                            # 임베딩이 없을 때 대체 시각화 로직 (visualization_utils 내부에서 처리)
                            embeddings_array = visualization_utils.handle_missing_embeddings(collection, documents)
                            if embeddings_array is None or len(embeddings_array) == 0:
                                st.error("임베딩 데이터가 없거나 대체 임베딩 생성에 실패하여 시각화를 진행할 수 없습니다.")
                                st.stop()
                        else:
                            # 임베딩 배열로 변환
                            import numpy as np
                            embeddings_array = np.array(embeddings)
                            
                        # 최소 필요 문서 수 확인 (클러스터링을 위해)
                        min_docs_for_clustering = 3 # 최소 3개 문서 필요 (t-SNE, KMeans)
                        if len(documents) < min_docs_for_clustering:
                             st.warning(f"시각화를 위해서는 최소 {min_docs_for_clustering}개 이상의 문서가 필요합니다. 현재 문서 수: {len(documents)}")
                             st.stop()
                             
                        # 클러스터 수 조정 (문서 수보다 클 수 없음)
                        if find_optimal:
                             # 자동 찾기 시에도 최대 클러스터 수는 문서 수보다 작아야 함
                             max_clusters_actual = min(max_clusters, len(documents) - 1) if len(documents) > 1 else 2
                             if max_clusters_actual < 2:
                                 st.warning(f"클러스터링을 위해서는 최소 2개 이상의 클러스터가 필요하지만, 문서 수가 부족하여 자동 찾기를 수행할 수 없습니다. (문서 수: {len(documents)})")
                                 find_optimal = False # 자동 찾기 비활성화
                                 n_clusters = max(2, len(documents) // 2) # 기본 클러스터 수 설정
                                 st.info(f"클러스터 수를 {n_clusters}로 자동 조정합니다.")
                             else:
                                 max_clusters = max_clusters_actual # 조정된 최대 클러스터 수 사용
                                 st.info(f"자동 찾기 최대 클러스터 수를 {max_clusters}로 조정합니다.")
                                 
                        # 클러스터 수 조정 (문서 수보다 클 수 없음) - 자동 찾기 사용 안 할 때
                        if not find_optimal and n_clusters >= len(documents):
                             st.warning(f"클러스터 수({n_clusters})가 문서 수({len(documents)})보다 많습니다. 클러스터 수를 문서 수보다 작게 조정합니다.")
                             n_clusters = max(2, len(documents) // 2) # 문서 수의 절반 또는 최소 2개
                             st.info(f"클러스터 수를 {n_clusters}로 자동 조정합니다.")
                             
                        # 클러스터링을 위한 최소 문서 수 확인
                        if (find_optimal and len(documents) < 3) or (not find_optimal and len(documents) < n_clusters):
                             st.warning(f"클러스터링을 위해서는 최소 {n_clusters}개 이상의 문서가 필요합니다. 현재 문서 수: {len(documents)}")
                             if len(documents) >= 2:
                                 st.info(f"문서 수에 맞춰 클러스터 수를 {max(2, len(documents) // 2)}로 조정합니다.")
                                 n_clusters = max(2, len(documents) // 2)
                             else:
                                 st.error("클러스터링을 수행할 수 없습니다. 문서 수를 늘려주세요.")
                                 st.stop()

                    else:
                        st.info("컬렉션에 데이터가 없습니다.")
                        st.stop()
                
                # 최적 클러스터 수 찾기
                if find_optimal:
                    st.subheader("최적 클러스터 수 분석")
                    with st.spinner("최적 클러스터 수 계산 중..."):
                        # 임베딩 배열과 최대 클러스터 수를 전달
                        silhouette_df, optimal_clusters = visualization_utils.find_optimal_clusters(embeddings_array, max_clusters)
                        
                        # 엘보우 방법 시각화
                        visualization_utils.plot_elbow_method(silhouette_df)
                        # 최적 클러스터 수 정보 표시
                        visualization_utils.display_optimal_cluster_info(optimal_clusters)
                        
                        # 최적 클러스터 수를 사용하도록 설정
                        n_clusters = int(optimal_clusters["클러스터 수"])
                        st.success(f"최적의 클러스터 수로 {n_clusters}을(를) 사용합니다.")

                        # 세션 상태에 분석 결과 저장
                        st.session_state.show_optimal_cluster_analysis = True
                        st.session_state.silhouette_df_for_plot = silhouette_df
                        st.session_state.optimal_clusters_info = optimal_clusters
                else:
                    # 자동 찾기를 사용하지 않으면 관련 세션 상태 초기화
                    st.session_state.show_optimal_cluster_analysis = False
                    st.session_state.silhouette_df_for_plot = None
                    st.session_state.optimal_clusters_info = None
                
                # 시각화 데이터 준비
                viz_data = visualization_utils.prepare_visualization_data(
                    embeddings_array, documents, ids, metadatas, perplexity, n_clusters
                )
                
                # 세션 상태에 시각화 데이터와 클러스터 수 저장
                st.session_state.viz_data = viz_data
                st.session_state.n_clusters = n_clusters # 최종 결정된 클러스터 수 저장
                st.session_state.viz_completed = True
                st.rerun()
            except Exception as e:
                st.error(f"시각화 생성 중 오류가 발생했습니다: {str(e)}")
                st.exception(e)
    
    # 시각화가 이미 완료된 경우, 저장된 시각화 데이터를 다시 표시
    elif 'viz_completed' in st.session_state and st.session_state.viz_completed:
        # 시각화 데이터를 사용하여 기존 시각화 표시
        
        # 최적 클러스터 수 분석 결과가 있으면 먼저 표시
        if st.session_state.get('show_optimal_cluster_analysis', False):
            st.subheader("최적 클러스터 수 분석")
            silhouette_df_to_plot = st.session_state.get('silhouette_df_for_plot')
            optimal_clusters_to_display = st.session_state.get('optimal_clusters_info')
            if silhouette_df_to_plot is not None and optimal_clusters_to_display is not None:
                visualization_utils.plot_elbow_method(silhouette_df_to_plot)
                visualization_utils.display_optimal_cluster_info(optimal_clusters_to_display)
            else:
                # 이 경우는 거의 발생하지 않아야 함
                st.info("최적 클러스터 수 분석 정보가 세션에 없습니다.")

        viz_data = st.session_state.get('viz_data')
        n_clusters = st.session_state.get('n_clusters')
        if viz_data is not None and n_clusters is not None:
            max_words_to_use = st.session_state.get('max_words_wc_slider', 100)
            # 기본 시각화 표시
            render_visualizations(viz_data, n_clusters, max_words_to_use)
            # LDA 토픽 모델링 섹션 - 시각화 이후에 배치하여 UI 흐름 일관성 유지
            st.subheader("클러스터별 LDA 토픽 모델링 설정")
            with st.container():
                lda_cols = st.columns([3,1])
                with lda_cols[0]:
                    lda_topics = st.slider(
                        "LDA 토픽 수",
                        min_value=2,
                        max_value=10,
                        value=st.session_state.get('lda_topics', 3),
                        step=1,
                        help="각 클러스터에서 LDA로 추출할 토픽의 수를 설정합니다. 작은 클러스터의 경우 자동으로 조정됩니다.",
                        key="lda_topics_slider"
                    )
                    if 'lda_topics' not in st.session_state or st.session_state.lda_topics != lda_topics:
                        st.session_state.lda_topics = lda_topics
                with lda_cols[1]:
                    st.write("")
                    run_lda_again = st.button(
                        "LDA 토픽 모델링", 
                        key="run_lda_again_btn", 
                        type="primary"
                    )
            st.write("lambda=1일 때는 빈도 기반, lambda=0일 때는 토픽 내 특이성 기반으로 단어를 정렬합니다. 0.6 ~ 0.8 사이의 값을 추천합니다.")
            if run_lda_again:
                with st.spinner("LDA 토픽 모델링 중..."):
                    try:
                        viz_data = st.session_state.viz_data
                        n_clusters = st.session_state.n_clusters
                        lda_topics = st.session_state.lda_topics
                        
                        # 각 클러스터별로 LDA 토픽 모델링 실행
                        for cluster_id in range(n_clusters):
                            cluster_texts = viz_data[viz_data['cluster'] == cluster_id]['full_text'].tolist()
                            if len(cluster_texts) >= 5:  # 최소 5개 문서 필요
                                st.write(f"클러스터 {cluster_id} LDA 토픽 분석 ({len(cluster_texts)}개 문서)")
                                visualization_utils.process_cluster_lda(cluster_texts, cluster_id, KOREAN_STOPWORDS, lda_topics)
                            else:
                                st.write(f"클러스터 {cluster_id}: 문서가 부족하여 LDA 분석을 수행할 수 없습니다. (최소 5개 필요)")
                        
                        st.success("LDA 토픽 모델링이 완료되었습니다.")
                    except Exception as e:
                        st.error(f"LDA 토픽 모델링 중 오류가 발생했습니다: {str(e)}")
                        st.exception(e)
        else:
            st.warning("시각화 데이터가 세션에 없습니다. 먼저 '시각화 생성'을 실행하세요.")

# 시각화 렌더링 함수를 분리하여 재사용 가능하게 함
def render_visualizations(viz_data, n_clusters, max_words_for_wordcloud):
    """시각화 데이터를 사용하여 클러스터 시각화, 문서, WordCloud를 표시"""
    # 클러스터 시각화
    visualization_utils.create_cluster_visualization(viz_data, n_clusters)
    
    # 클러스터별 주요 문서 표시
    visualization_utils.display_cluster_documents(viz_data, n_clusters)
    # 클러스터별 WordCloud 표시
    visualization_utils.display_cluster_wordclouds(viz_data, n_clusters, KOREAN_STOPWORDS, max_words_wc=max_words_for_wordcloud)

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