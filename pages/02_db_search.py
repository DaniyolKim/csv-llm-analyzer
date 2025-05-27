import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from chroma_utils import load_chroma_collection, get_available_collections, hybrid_query_chroma
# WordCloud 추가
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from text_utils import KOREAN_STOPWORDS # text_utils에서 불용어 목록 가져오기
import matplotlib.font_manager as fm
import random # 무작위 선택을 위해 추가

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
                                    "키워드": metadata.get("keywords", "알 수 없음"),
                                    "내용": doc,
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
                                    "키워드": st.column_config.TextColumn(width="medium", help="문서에서 추출된 주요 키워드입니다."),
                                    "내용": st.column_config.TextColumn(width="large"),
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
                                    "키워드": metadata.get("keywords", "알 수 없음"),
                                    "내용": doc,
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
                                    "키워드": st.column_config.TextColumn(width="medium", help="문서에서 추출된 주요 키워드입니다."),
                                    "내용": st.column_config.TextColumn(width="large"),
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
        
        # 탭 3: 시각화
        with tab3:
            st.subheader(f"컬렉션 시각화: {selected_collection}")
            
            # 시각화 설정
            with st.expander("시각화 설정", expanded=True):
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
                    help="t-SNE가 각 데이터 포인트 주변의 '유효 이웃 수'를 결정하는 값입니다. 데이터의 지역적 구조와 전역적 구조 사이의 균형에 영향을 줍니다.\n\n"
                         "- **낮은 값 (5-15):** 매우 가까운 이웃에 집중하여 세밀한 지역적 구조를 파악하는 데 유용합니다. 너무 낮으면 노이즈에 민감하거나 불필요하게 많은 작은 군집이 생길 수 있습니다.\n"
                         "- **중간 값 (20-30):** 대부분의 데이터셋에서 좋은 시작점입니다. 지역적 구조와 전역적 구조 간의 적절한 균형을 제공합니다. (기본값: 30)\n"
                         "- **높은 값 (35-50):** 더 넓은 범위의 이웃을 고려하여 데이터의 전역적인 구조나 큰 군집을 파악하는 데 도움이 됩니다. 너무 높으면 세밀한 구조가 뭉개질 수 있습니다.\n\n"
                         "데이터셋의 크기와 특성에 따라 여러 값을 시도하며 가장 의미 있는 패턴을 보여주는 값을 선택하세요."
                )
            
            # 임베딩 데이터 가져오기 함수
            def get_embeddings_data(collection, all_data, max_docs):
                """컬렉션에서 임베딩 데이터를 가져오는 함수"""
                total_docs = len(all_data["documents"])
                
                # 최대 문서 수 제한 적용
                if total_docs > max_docs:
                    st.info(f"문서가 너무 많아 무작위로 {max_docs}개를 선택하여 시각화합니다.")
                    # 무작위 인덱스 선택
                    random_indices = random.sample(range(total_docs), max_docs)
                    
                    # 선택된 인덱스를 사용하여 데이터 추출
                    documents = [all_data["documents"][i] for i in random_indices]
                    metadatas = [all_data["metadatas"][i] for i in random_indices]
                    ids = [all_data["ids"][i] for i in random_indices]

                    # documents = all_data["documents"][:max_docs] # 이전 방식
                    # metadatas = all_data["metadatas"][:max_docs] # 이전 방식
                    # ids = all_data["ids"][:max_docs] # 이전 방식


                    # 제한된 ID로 임베딩 가져오기
                    try:
                        embeddings_result = collection.get(
                            ids=ids,
                            include=["embeddings"]
                        )
                        embeddings = embeddings_result.get("embeddings", [])
                    except Exception as e:
                        st.warning(f"임베딩 데이터 가져오기 실패: {str(e)}")
                        embeddings = []
                else:
                    documents = all_data["documents"]
                    metadatas = all_data["metadatas"]
                    ids = all_data["ids"]
                    
                    # 모든 임베딩 가져오기
                    try:
                        embeddings_result = collection.get(
                            include=["embeddings"]
                        )
                        embeddings = embeddings_result.get("embeddings", [])
                    except Exception as e:
                        st.warning(f"임베딩 데이터 가져오기 실패: {str(e)}")
                        embeddings = []
                
                return documents, metadatas, ids, embeddings
            
            # 임베딩 없을 때 처리 함수
            def handle_missing_embeddings(collection, documents):
                """임베딩 데이터가 없을 때 처리하는 함수"""
                # 컬렉션 메타데이터에서 임베딩 모델 정보 확인
                embedding_model = "알 수 없음"
                try:
                    if collection.metadata and "embedding_model" in collection.metadata:
                        embedding_model = collection.metadata["embedding_model"]
                except:
                    pass
                
                # 임베딩 함수 확인
                has_embedding_function = hasattr(collection, "_embedding_function") and collection._embedding_function is not None
                
                if has_embedding_function:
                    st.warning(f"이 컬렉션은 '{embedding_model}' 임베딩 모델로 생성되었지만, 임베딩 데이터를 가져올 수 없습니다.")
                    st.info("컬렉션을 다시 로드하거나, 데이터를 다시 저장해보세요.")
                else:
                    st.warning("컬렉션에 임베딩 함수가 설정되지 않았습니다.")
                    st.info(f"이 컬렉션은 '{embedding_model}' 임베딩 모델로 생성되었습니다. 동일한 모델로 데이터를 다시 저장해보세요.")
                
                # 대체 시각화 방법 제안
                st.info("임베딩 데이터 없이 시각화를 진행하시겠습니까? 임의의 임베딩을 생성하여 시각화할 수 있습니다.")
                if st.button("임의 임베딩으로 시각화 진행", key="random_viz_btn"):
                    # 임의의 임베딩 생성 (문서 수 x 384 차원)
                    st.text("임의 임베딩 생성 중...")
                    import numpy as np
                    random_dim = 384  # 일반적인 임베딩 차원
                    num_docs = len(documents)
                    embeddings = np.random.rand(num_docs, random_dim)
                    st.success(f"임의의 {num_docs}x{random_dim} 임베딩을 생성했습니다.")
                    return embeddings
                else:
                    st.stop()
                    return []
            
            # 시각화 데이터 준비 함수
            def prepare_visualization_data(embeddings_array, documents, ids, metadatas, perplexity, n_clusters):
                """시각화를 위한 데이터를 준비하는 함수"""
                # t-SNE로 차원 축소
                st.text("t-SNE로 차원 축소 중...")
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings_array)
                
                # K-means 클러스터링
                st.text("클러스터링 중...")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(embeddings_array)
                
                # 데이터프레임 생성
                viz_data = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'cluster': clusters,
                    'id': ids,
                    'text': documents,  # 전체 텍스트 표시 (hover용)
                    'full_text': documents  # 원본 텍스트 (WordCloud 및 상세 내용 표시용)
                })
                
                # 출처 정보 추가
                viz_data['source'] = [metadata.get("source", "알 수 없음") for metadata in metadatas]
                
                return viz_data
            
            # WordCloud 생성 함수
            # @st.cache_data(show_spinner=False) # 결과 캐싱하여 반복 생성 방지
            def generate_wordcloud_for_cluster(_texts, _stopwords, _collection_name_for_cache, _cluster_id_for_cache): # 캐시 키에 클러스터 ID 추가
                from text_utils import clean_text # 여기서 import 해야 캐싱에 문제 없음
                okt = Okt()
                nouns = []
                for text_content in _texts:
                    cleaned_text_for_nouns = clean_text(str(text_content)) # str()로 명시적 변환
                    for noun in okt.nouns(cleaned_text_for_nouns):
                        if noun not in _stopwords and len(noun) > 1: # 한 글자 명사 제외
                            nouns.append(noun)
            
                if not nouns:
                    return None
            
                # 폰트 경로 설정
                font_path = None
                preferred_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR']
                font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                
                for font_file in font_list:
                    try:
                        font_name = fm.FontProperties(fname=font_file).get_name()
                        if any(preferred in font_name for preferred in preferred_fonts):
                            font_path = font_file
                            break
                    except RuntimeError:
                        continue # 일부 폰트 파일 파싱 오류 발생 가능성
            
                if not font_path:
                    # 캐시 함수 내에서 st UI 요소 직접 호출 지양
                    print("선호하는 한글 폰트(NanumGothic, Malgun Gothic 등)를 시스템에서 찾지 못했습니다.")
            
                # 새로운 Figure 객체 생성
                fig, ax = plt.subplots(figsize=(12, 6))
                try:
                    wordcloud = WordCloud(
                        font_path=font_path,
                        width=800,
                        height=400,
                        background_color='white',
                        collocations=False # 연어(collocations) 방지
                    ).generate(' '.join(nouns))
                    
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    return fig # WordCloud 이미지 자체가 아닌 Figure 객체를 반환
                except Exception as e:
                    print(f"WordCloud 생성 중 오류: {e}. 폰트 문제일 수 있습니다.")
                    plt.close(fig) # 오류 발생 시 생성된 figure 닫기
                    return None

            # 클러스터별 WordCloud 표시 함수
            def display_cluster_wordclouds(viz_data, n_clusters, stopwords, current_collection_name):
                st.subheader("클러스터별 주요 단어 (WordCloud)")

                # 폰트 경로 미리 확인 (경고 메시지용)
                font_path_exists = False
                preferred_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans KR']
                font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                for font_file in font_list:
                    try:
                        font_name = fm.FontProperties(fname=font_file).get_name()
                        if any(preferred in font_name for preferred in preferred_fonts):
                            font_path_exists = True
                            break
                    except RuntimeError:
                        continue
                if not font_path_exists:
                    st.sidebar.warning("선호하는 한글 폰트(NanumGothic, Malgun Gothic 등)를 시스템에서 찾지 못했습니다. WordCloud가 깨질 수 있습니다. 폰트 설치를 권장합니다.")

                for cluster_id in range(n_clusters):
                    cluster_texts = viz_data[viz_data['cluster'] == cluster_id]['full_text'].tolist()
                    
                    with st.expander(f"클러스터 {cluster_id} WordCloud ({len(cluster_texts)}개 문서)"):
                        if not cluster_texts:
                            st.write(f"클러스터 {cluster_id}: 분석할 텍스트가 없습니다.")
                            continue
                        # generate_wordcloud_for_cluster 함수 호출 시 클러스터 ID도 전달
                        wordcloud_fig = generate_wordcloud_for_cluster(cluster_texts, stopwords, current_collection_name, cluster_id)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                            plt.close(wordcloud_fig) # 사용 후 figure 닫기
                        else:
                            st.write("WordCloud를 생성할 충분한 단어가 없거나, 생성 중 오류가 발생했습니다.")

            # 시각화 버튼
            if st.button("시각화 생성", key="create_viz_btn", type="primary"):
                with st.spinner("시각화를 생성하는 중... 이 작업은 데이터 크기에 따라 시간이 걸릴 수 있습니다."):
                    try:
                        # 이미 로드된 컬렉션 사용
                        collection = st.session_state.chroma_collection
                        
                        # 컬렉션의 모든 데이터 가져오기
                        all_data = collection.get()
                        
                        if all_data and all_data["documents"]:
                            # 결과 표시
                            total_docs = len(all_data["documents"])
                            st.success(f"총 {total_docs}개의 문서를 로드했습니다.")
                            
                            # 임베딩 데이터 가져오기
                            documents, metadatas, ids, embeddings = get_embeddings_data(collection, all_data, max_docs)
                            
                            # 임베딩이 없는 경우 처리
                            if len(embeddings) == 0:
                                st.error("임베딩 데이터를 가져올 수 없습니다.")
                                embeddings = handle_missing_embeddings(collection, documents)
                            
                            # 임베딩 배열로 변환
                            embeddings_array = np.array(embeddings)
                            
                            # 시각화 데이터 준비
                            viz_data = prepare_visualization_data(
                                embeddings_array, documents, ids, metadatas, perplexity, n_clusters
                            )
                            
                            # Plotly로 시각화
                            st.subheader("문서 클러스터 시각화")
                            
                            # 클러스터별 색상 설정
                            colors = px.colors.qualitative.Plotly
                            
                            # 클러스터 수에 맞게 색상 확장
                            while len(colors) < n_clusters:
                                colors.extend(colors)
                            colors = colors[:n_clusters]
                            
                            # 그래프 생성
                            fig = go.Figure()
                            
                            # 클러스터별로 점 추가
                            def add_cluster_traces(fig, viz_data, n_clusters, colors):
                                """클러스터별로 그래프에 점 추가"""
                                for cluster_id in range(n_clusters):
                                    cluster_data = viz_data[viz_data['cluster'] == cluster_id]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=cluster_data['x'],
                                        y=cluster_data['y'],
                                        mode='markers',
                                        marker=dict(
                                            size=10,
                                            color=colors[cluster_id],
                                            line=dict(width=1, color='DarkSlateGrey')
                                        ),
                                        name=f'클러스터 {cluster_id}',
                                        text=cluster_data['text'],
                                        hoverinfo='text',
                                        hovertemplate='<b>출처:</b> %{customdata}<br><b>내용:</b> %{text}<extra></extra>',
                                        customdata=cluster_data['source']
                                    ))
                                return fig
                            
                            # 클러스터별로 점 추가
                            fig = add_cluster_traces(fig, viz_data, n_clusters, colors)
                            
                            # 레이아웃 설정
                            fig.update_layout(
                                title='문서 클러스터 시각화 (t-SNE + K-means)',
                                xaxis=dict(title='t-SNE 차원 1', showgrid=True),
                                yaxis=dict(title='t-SNE 차원 2', showgrid=True),
                                hovermode='closest',
                                legend_title='클러스터',
                                width=800,
                                height=600
                            )
                            
                            # 그래프 표시
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 클러스터 통계 시각화
                            def visualize_cluster_statistics(viz_data, colors):
                                """클러스터 통계를 시각화하는 함수"""
                                st.subheader("클러스터 통계")
                                cluster_counts = viz_data['cluster'].value_counts().reset_index()
                                cluster_counts.columns = ['클러스터', '문서 수']
                                
                                # 클러스터별 문서 수 차트
                                fig_bar = go.Figure(go.Bar(
                                    x=cluster_counts['클러스터'],
                                    y=cluster_counts['문서 수'],
                                    marker_color=colors[:len(cluster_counts)]
                                ))
                                fig_bar.update_layout(
                                    title='클러스터별 문서 수',
                                    xaxis=dict(title='클러스터'),
                                    yaxis=dict(title='문서 수')
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                            
                            # 클러스터 통계 시각화
                            visualize_cluster_statistics(viz_data, colors)
                            
                            # 클러스터별 WordCloud 표시
                            display_cluster_wordclouds(viz_data, n_clusters, KOREAN_STOPWORDS, selected_collection)
                            
                            # 클러스터별 주요 문서 표시
                            def display_cluster_documents(viz_data, n_clusters):
                                """클러스터별 주요 문서를 표시하는 함수"""
                                st.subheader("클러스터별 주요 문서")
                                for cluster_id in range(n_clusters):
                                    cluster_docs = viz_data[viz_data['cluster'] == cluster_id]
                                    with st.expander(f"클러스터 {cluster_id} 주요 문서 ({len(cluster_docs)}개 문서)"):
                                        for _, row in cluster_docs.head(5).iterrows():
                                            # 원본 텍스트 전체를 표시
                                            st.markdown(f"**출처:** {row['source']}")
                                            st.markdown(f"**내용:** {row['full_text']}")
                                            st.markdown("---")
                            
                            # 클러스터별 주요 문서 표시
                            display_cluster_documents(viz_data, n_clusters)
                        else:
                            st.info("컬렉션에 데이터가 없습니다.")
                    
                    except Exception as e:
                        st.error(f"시각화 생성 중 오류가 발생했습니다: {str(e)}")
                        st.exception(e)

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
    
    #### 시각화 탭
    1. 클러스터 수, 최대 문서 수, t-SNE 복잡도 등의 매개변수를 설정합니다.
    2. '시각화 생성' 버튼을 클릭하여 문서 클러스터 시각화를 생성합니다.
    3. 클러스터별 문서 분포와 주요 문서를 확인할 수 있습니다.
    
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
    
    ### 시각화 정보
    
    시각화 탭에서는 t-SNE 알고리즘을 사용하여 고차원 임베딩을 2차원으로 축소하고, K-means 클러스터링을 통해 유사한 문서들을 그룹화합니다.
    - 각 점은 하나의 문서를 나타내며, 색상은 클러스터를 구분합니다.
    - 점 위에 마우스를 올리면 문서 내용과 출처를 확인할 수 있습니다.
    - 클러스터별 문서 수와 주요 문서를 확인할 수 있습니다.
    """)