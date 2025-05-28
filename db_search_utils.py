import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from text_utils import KOREAN_STOPWORDS, clean_text

# 데이터 로딩 및 표시 함수
def load_collection_data(collection):
    """컬렉션에서 모든 데이터를 로드하여 DataFrame으로 반환"""
    try:
        all_data = collection.get()
        
        if all_data and all_data["documents"]:
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
            
            # 데이터프레임 생성 및 반환
            return pd.DataFrame(result_data), all_data
        else:
            return None, None
    except Exception as e:
        raise Exception(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")

def display_collection_data(result_df):
    """컬렉션 데이터를 표시하는 함수"""
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
    st.write(f"총 문서 수: {len(result_df)}")
    
    # 출처별 문서 수 계산
    source_counts = result_df["출처"].value_counts().reset_index()
    source_counts.columns = ["출처", "문서 수"]
    
    # 출처별 문서 수 차트
    st.bar_chart(source_counts.set_index("출처"))

# 검색 함수
def search_collection(collection, query, n_results=10):
    """컬렉션에서 쿼리에 가장 관련 있는 문서를 검색"""
    from chroma_utils import hybrid_query_chroma  # 함수 내에서 import하여 circular import 방지
    
    results = hybrid_query_chroma(collection, query, n_results=n_results)
    
    if results and results["documents"] and results["documents"][0]:
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
        
        # 데이터프레임 생성 및 반환
        return pd.DataFrame(result_data)
    else:
        return None

def display_search_results(result_df):
    """검색 결과를 표시하는 함수"""
    if result_df is None or len(result_df) == 0:
        st.info("검색 결과가 없습니다.")
        return
        
    st.subheader(f"검색 결과: {len(result_df)}개")
    
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
    if len(result_df) > 1:
        st.subheader("유사도 분포")
        chart_data = pd.DataFrame({
            "순위": result_df["순위"],
            "유사도": result_df["유사도"].astype(float)
        })
        st.bar_chart(chart_data.set_index("순위"))

# 시각화 유틸리티 함수
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
        random_dim = 384  # 일반적인 임베딩 차원
        num_docs = len(documents)
        embeddings = np.random.rand(num_docs, random_dim)
        st.success(f"임의의 {num_docs}x{random_dim} 임베딩을 생성했습니다.")
        return embeddings
    else:
        st.stop()
        return []

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

def generate_wordcloud_for_cluster(texts, stopwords):
    """클러스터의 텍스트에서 워드클라우드 생성"""
    okt = Okt()
    nouns = []
    for text_content in texts:
        cleaned_text_for_nouns = clean_text(str(text_content)) # str()로 명시적 변환
        for noun in okt.nouns(cleaned_text_for_nouns):
            if noun not in stopwords and len(noun) > 1: # 한 글자 명사 제외
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

def display_cluster_wordclouds(viz_data, n_clusters, stopwords):
    """클러스터별 워드클라우드 표시"""
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
            wordcloud_fig = generate_wordcloud_for_cluster(cluster_texts, stopwords)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
                plt.close(wordcloud_fig) # 사용 후 figure 닫기
            else:
                st.write("WordCloud를 생성할 충분한 단어가 없거나, 생성 중 오류가 발생했습니다.")

def create_cluster_visualization(viz_data, n_clusters):
    """클러스터 시각화 생성"""
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

def visualize_cluster_statistics(viz_data, n_clusters):
    """클러스터 통계를 시각화"""
    st.subheader("클러스터 통계")
    cluster_counts = viz_data['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['클러스터', '문서 수']
    
    # 클러스터별 색상 설정
    colors = px.colors.qualitative.Plotly
    while len(colors) < n_clusters:
        colors.extend(colors)
    colors = colors[:n_clusters]
    
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

def display_cluster_documents(viz_data, n_clusters):
    """클러스터별 주요 문서를 표시"""
    st.subheader("클러스터별 주요 문서")
    for cluster_id in range(n_clusters):
        cluster_docs = viz_data[viz_data['cluster'] == cluster_id]
        with st.expander(f"클러스터 {cluster_id} 주요 문서 ({len(cluster_docs)}개 문서)"):
            for _, row in cluster_docs.head(5).iterrows():
                # 원본 텍스트 전체를 표시
                st.markdown(f"**출처:** {row['source']}")
                st.markdown(f"**내용:** {row['full_text']}")
                st.markdown("---")