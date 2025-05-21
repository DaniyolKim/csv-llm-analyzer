import streamlit as st
import pandas as pd
import os
from chroma_utils import load_chroma_collection, get_available_collections, hybrid_query_chroma


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
        st.error(f"선택한 경로({db_path})에 사용 가능한 컬렉션이 없습니다.")
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
        # st.success(f"✅ 컬렉션 '{selected_collection}'이 로드되었습니다.")

# 탭 생성
tab1, tab2 = st.tabs(["컬렉션 데이터", "텍스트 검색"])

# 메인 영역
if not collections:
    for tab in [tab1, tab2]:
        with tab:
            st.warning(f"선택한 경로({db_path})에 사용 가능한 컬렉션이 없습니다. 먼저 CSV 파일을 업로드하고 DB에 저장해주세요.")
else:
    # 컬렉션이 로드되지 않은 경우 안내 메시지
    if not st.session_state.collection_loaded:
        for tab in [tab1, tab2]:
            with tab:
                st.info("사이드바에서 컬렉션을 로드하세요.")
    else:
        # 탭 1: 컬렉션 데이터 표시
        with tab1:
            st.subheader(f"컬렉션: {selected_collection}")
            
            # 데이터 로드 버튼
            if st.button("데이터 표시", key="show_data_btn"):
                with st.spinner("데이터를 가져오는 중..."):
                    try:
                        # 이미 로드된 컬렉션 사용
                        collection = st.session_state.chroma_collection
                        
                        # 컬렉션의 모든 데이터 가져오기
                        all_data = collection.get()
                        
                        if all_data and all_data["documents"]:
                            # 결과 표시
                            st.success(f"총 {len(all_data['documents'])}개의 문서를 로드했습니다.")
                            
                            # 결과를 데이터프레임으로 변환
                            result_data = []
                            for i, (doc, metadata, id) in enumerate(zip(
                                all_data["documents"], 
                                all_data["metadatas"],
                                all_data["ids"]
                            )):
                                result_data.append({
                                    "ID": id,
                                    "출처": metadata.get("source", "알 수 없음"),
                                    "청크": metadata.get("chunk", "알 수 없음"),
                                    "내용": doc
                                })
                            
                            # 데이터프레임 생성 및 표시
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(
                                result_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "ID": st.column_config.TextColumn(width="medium"),
                                    "출처": st.column_config.TextColumn(width="small"),
                                    "청크": st.column_config.NumberColumn(width="small"),
                                    "내용": st.column_config.TextColumn(width="large")
                                }
                            )
                            
                            # 데이터 통계
                            st.subheader("데이터 통계")
                            st.write(f"총 문서 수: {len(all_data['documents'])}")
                            
                            # 출처별 문서 수 계산
                            source_counts = {}
                            for metadata in all_data["metadatas"]:
                                source = metadata.get("source", "알 수 없음")
                                source_counts[source] = source_counts.get(source, 0) + 1
                            
                            # 출처별 문서 수 차트
                            source_df = pd.DataFrame({
                                "출처": list(source_counts.keys()),
                                "문서 수": list(source_counts.values())
                            })
                            st.bar_chart(source_df.set_index("출처"))
                        else:
                            st.info("컬렉션에 데이터가 없습니다.")
                    
                    except Exception as e:
                        st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
        
        # 탭 2: DB 검색
        with tab2:
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
            
            # 검색 실행
            if search_button and query:
                with st.spinner("검색 중..."):
                    try:
                        # 이미 로드된 컬렉션 사용
                        collection = st.session_state.chroma_collection
                        
                        # 하이브리드 검색 실행
                        results = hybrid_query_chroma(collection, query, n_results=n_results)
                        
                        if results and results["documents"] and results["documents"][0]:
                            # 결과 표시
                            st.subheader(f"검색 결과: {len(results['documents'][0])}개")
                            
                            # 결과를 데이터프레임으로 변환
                            result_data = []
                            for i, (doc, metadata, distance) in enumerate(zip(
                                results["documents"][0], 
                                results["metadatas"][0], 
                                results["distances"][0]
                            )):
                                # 유사도 점수 계산 (거리를 유사도로 변환)
                                similarity = 1 - distance
                                
                                # 검색 유형 확인
                                search_type = "임베딩"
                                if "search_type" in results:
                                    search_type = "키워드" if results["search_type"][0][i] == "keyword" else "임베딩"
                                
                                # 모든 결과 표시
                                result_data.append({
                                    "순위": i + 1,
                                    "유사도": f"{similarity:.4f}",
                                    "검색 유형": search_type,
                                    "출처": metadata.get("source", "알 수 없음"),
                                    "청크": metadata.get("chunk", "알 수 없음"),
                                    "내용": doc
                                })
                            
                            # 데이터프레임 생성 및 표시
                            result_df = pd.DataFrame(result_data)
                            st.dataframe(
                                result_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "순위": st.column_config.NumberColumn(width="small"),
                                    "유사도": st.column_config.TextColumn(width="small"),
                                    "검색 유형": st.column_config.TextColumn(width="small"),
                                    "출처": st.column_config.TextColumn(width="small"),
                                    "청크": st.column_config.NumberColumn(width="small"),
                                    "내용": st.column_config.TextColumn(width="large")
                                }
                            )
                            
                            # 시각화: 유사도 차트
                            if len(result_data) > 1:
                                st.subheader("유사도 분포")
                                chart_data = pd.DataFrame({
                                    "순위": [item["순위"] for item in result_data],
                                    "유사도": [float(item["유사도"]) for item in result_data]
                                })
                                st.bar_chart(chart_data.set_index("순위"))
                        else:
                            st.info("검색 결과가 없습니다.")
                    
                    except Exception as e:
                        st.error(f"검색 중 오류가 발생했습니다: {str(e)}")

# 도움말 섹션
with st.expander("사용 방법"):
    st.markdown("""
    ### DB 검색 사용 방법
    
    #### 공통 설정
    1. 사이드바에서 ChromaDB 경로를 입력합니다. (기본값: './chroma_db')
    2. 검색할 컬렉션을 선택합니다.
    3. '컬렉션 로드' 버튼을 클릭하여 컬렉션을 메모리에 로드합니다.
    
    #### 컬렉션 데이터 탭
    - '데이터 표시' 버튼을 클릭하여 선택한 컬렉션의 모든 데이터를 확인할 수 있습니다.
    - 데이터 통계를 통해 출처별 문서 수를 확인할 수 있습니다.
    
    #### DB 검색 탭
    1. 검색 결과 수를 조정합니다.
    2. 검색어를 입력하고 '검색' 버튼을 클릭합니다.
    3. 검색 결과는 유사도가 높은 순으로 정렬됩니다.
    
    ### 하이브리드 검색
    
    하이브리드 검색은 임베딩 기반 의미 검색과 키워드 기반 검색을 결합합니다.
    - 단어 검색에 더 효과적이며, 정확한 단어 매칭을 포함합니다.
    - 검색 결과에 '검색 유형'이 표시됩니다. (임베딩 또는 키워드)
    
    ### ChromaDB 경로
    
    다른 폴더에 저장된 ChromaDB를 검색하려면 해당 경로를 입력하세요.
    상대 경로(예: './chroma_db') 또는 절대 경로(예: 'C:/Users/username/chroma_db')를 사용할 수 있습니다.
    
    ### 임베딩 모델
    
    검색 시 컬렉션에 저장된 임베딩 모델이 자동으로 사용됩니다.
    컬렉션 정보에서 사용 중인 임베딩 모델을 확인할 수 있습니다.
    
    ### 유사도 점수
    
    유사도 점수는 0에서 1 사이의 값으로, 1에 가까울수록 검색어와 유사한 내용입니다.
    키워드 검색 결과의 경우 유사도 점수는 임의로 설정됩니다.
    """)