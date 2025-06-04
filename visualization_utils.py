import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from text_utils import KOREAN_STOPWORDS, clean_text

# LDA 모델링을 위한 추가 라이브러리 - 선택적 임포트
LDA_AVAILABLE = False
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    import pyLDAvis
    import pyLDAvis.gensim_models
    LDA_AVAILABLE = True
except ImportError:
    pass

def get_embeddings_data(collection, all_data, docs_percentage):
    """컬렉션에서 임베딩 데이터를 가져오는 함수"""
    total_docs = len(all_data["documents"])
    
    # 백분율에 따른 문서 수 계산
    if docs_percentage < 100:
        # 백분율 기준으로 문서 수 계산
        num_docs = max(1, int(total_docs * docs_percentage / 100))
        st.info(f"전체 {total_docs}개 문서 중 {docs_percentage}%인 {num_docs}개의 문서를 무작위로 선택하여 시각화합니다.")
        
        # 무작위 인덱스 선택
        random_indices = random.sample(range(total_docs), num_docs)
        
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
        # 100%인 경우 모든 문서 사용
        st.info(f"모든 {total_docs}개 문서를 시각화합니다. 처리 시간이 길어질 수 있습니다.")
        
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

def prepare_visualization_data(embeddings_input, documents, ids, metadatas, perplexity, n_clusters):
    """시각화를 위한 데이터를 준비하는 함수"""
    
    # 1. Convert to NumPy array and ensure it's numeric
    try:
        # Ensure embeddings_input is treated as a NumPy array of floats
        current_embeddings_array = np.array(embeddings_input, dtype=float)
    except ValueError as e:
        st.error(f"임베딩 데이터를 숫자형 배열로 변환하는 중 오류 발생: {e}")
        st.info("일부 문서의 임베딩이 유효하지 않거나 형식이 일관되지 않을 수 있습니다.")
        st.error("심각한 임베딩 데이터 형식 오류로 시각화를 진행할 수 없습니다. DB의 임베딩 데이터를 확인해주세요.")
        return pd.DataFrame() # Return empty DataFrame

    # 2. Check array dimension
    if current_embeddings_array.ndim != 2:
        st.error(f"임베딩 배열이 2차원이 아닙니다 (현재 차원: {current_embeddings_array.ndim}). 시각화를 진행할 수 없습니다.")
        st.info("컬렉션의 임베딩 데이터 구조를 확인해주세요. 각 임베딩은 동일한 길이의 숫자 리스트여야 합니다.")
        return pd.DataFrame()
    
    # 3. Check for NaN values and filter
    nan_rows_mask = np.isnan(current_embeddings_array).any(axis=1)
    
    if np.any(nan_rows_mask): # If there are any NaN rows
        num_removed = np.sum(nan_rows_mask)
        st.warning(f"임베딩 데이터에 NaN 값이 포함된 {num_removed}개의 문서가 시각화에서 제외됩니다.")
        
        embeddings_array_filtered = current_embeddings_array[~nan_rows_mask]
        documents_filtered = [doc for i, doc in enumerate(documents) if not nan_rows_mask[i]]
        ids_filtered = [id_val for i, id_val in enumerate(ids) if not nan_rows_mask[i]]
        metadatas_filtered = [meta for i, meta in enumerate(metadatas) if not nan_rows_mask[i]]
        
        if embeddings_array_filtered.shape[0] == 0:
            st.error("NaN 값을 포함한 문서를 제외한 후 시각화할 데이터가 남아있지 않습니다.")
            return pd.DataFrame()
    else:
        embeddings_array_filtered = current_embeddings_array
        documents_filtered = documents
        ids_filtered = ids
        metadatas_filtered = metadatas

    # 4. Check if enough data points remain for t-SNE and K-Means
    num_samples = embeddings_array_filtered.shape[0]

    if num_samples == 0:
        st.error("시각화할 데이터 포인트가 없습니다.")
        return pd.DataFrame()

    # Adjust perplexity for t-SNE: must be less than n_samples
    adjusted_perplexity = perplexity
    if num_samples <= perplexity:
        adjusted_perplexity = max(1, num_samples - 1) 
        if num_samples > 1:
             st.warning(f"데이터 포인트 수({num_samples})가 Perplexity 설정값({perplexity})보다 작거나 같아 Perplexity를 {adjusted_perplexity}로 조정합니다.")
    
    if num_samples <= 1: # TSNE requires at least 2 samples
        st.error(f"t-SNE를 실행하기에 데이터 포인트가 너무 적습니다 ({num_samples}개). 최소 2개 이상의 데이터 포인트가 필요합니다.")
        return pd.DataFrame()

    # Adjust n_clusters for K-Means: must be <= n_samples and >= 1
    actual_n_clusters = n_clusters
    if num_samples < n_clusters:
        actual_n_clusters = num_samples
        st.warning(f"요청된 클러스터 수({n_clusters})가 사용 가능한 데이터 포인트 수({num_samples})보다 많아 클러스터 수를 {actual_n_clusters}로 조정합니다.")
    
    if actual_n_clusters < 1:
        st.error(f"클러스터링을 위한 데이터 포인트가 없습니다 ({num_samples}개). 최소 1개의 클러스터가 필요합니다.")
        return pd.DataFrame()

    # t-SNE로 차원 축소
    st.text("t-SNE로 차원 축소 중...")
    tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings_array_filtered)
    
    # K-means 클러스터링
    st.text("클러스터링 중...")
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(embeddings_array_filtered)
    
    # 데이터프레임 생성
    viz_data = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': clusters,
        'id': ids_filtered,
        'text': documents_filtered,
        'full_text': documents_filtered
    })
    
    # 출처 정보 추가
    viz_data['source'] = [metadata.get("source", "알 수 없음") for metadata in metadatas_filtered]
    
    return viz_data

def create_cluster_visualization(viz_data, n_clusters):
    """클러스터 시각화 생성"""
    st.subheader("문서 클러스터 시각화")
    
    # 클러스터별 색상 설정
    colors = px.colors.qualitative.Plotly
    
    # 클러스터 수에 맞게 색상 확장
    while len(colors) < n_clusters:
        colors.extend(colors)
    colors = colors[:n_clusters]
    
    # 텍스트 길이 제한 함수
    def truncate_text(text, max_length=200):
        """텍스트를 지정된 길이로 제한하고 줄바꿈 추가"""
        if len(text) <= max_length:
            # 80자마다 줄바꿈 추가
            return '<br>'.join([text[i:i+80] for i in range(0, len(text), 80)])
        return '<br>'.join([text[:max_length][i:i+80] for i in range(0, len(text[:max_length]), 80)]) + "..."
    
    # 그래프 생성
    fig = go.Figure()
    
    # 클러스터별로 점 추가
    for cluster_id in range(n_clusters):
        cluster_data = viz_data[viz_data['cluster'] == cluster_id]
        
        # 텍스트 길이 제한 적용
        hover_texts = [truncate_text(text) for text in cluster_data['text']]
        
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
            text=hover_texts,
            hoverinfo='text',
            hovertemplate='<b>출처:</b> %{customdata}<br><b>내용:</b> %{text}<extra></extra>',
            customdata=cluster_data['source']
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        title='문서 클러스터 시각화 (t-SNE + K-means)',
        xaxis=dict(title='t-SNE 차원 1', showgrid=True),
        yaxis=dict(title='t-SNE 차원 2', showgrid=True),
        hovermode='closest',  # 호버 모드 설정 (closest: 가장 가까운 포인트만)
        legend_title='클러스터',
        width=800,
        height=600,
        # 호버 모드와 호버 정보 스타일 지정
        hoverlabel=dict(
            font_size=12,
            font_family="Arial",
            # 호버 정보의 최대 너비 지정
            namelength=-1  # 호버 라벨 이름 길이 제한 없음
        )
    )
    
    # 그래프 표시
    st.plotly_chart(fig, use_container_width=True)

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

def generate_wordcloud_for_cluster(texts, stopwords, max_words_wc=100):
    """클러스터의 텍스트에서 워드클라우드 생성"""
    from text_utils import IMPORTANT_SINGLE_CHAR_NOUNS  # 중요 한 글자 명사 목록 가져오기
    
    okt = Okt()
    nouns = []
    
    for text_content in texts:
        try:
            # 텍스트를 문자열로 변환하고 정제
            cleaned_text_for_nouns = clean_text(str(text_content))
            
            # 형태소 분석으로 명사 추출 (인코딩 오류 예외 처리 추가)
            try:
                extracted_nouns = okt.nouns(cleaned_text_for_nouns)
                for noun in extracted_nouns:
                    # 불용어가 아니고, 2글자 이상이거나 중요 한 글자 명사 목록에 있는 단어만 포함
                    if noun not in stopwords and (len(noun) > 1 or noun in IMPORTANT_SINGLE_CHAR_NOUNS):
                        nouns.append(noun)
            except UnicodeDecodeError as ude:
                # 인코딩 오류 발생 시 해당 텍스트 건너뛰기
                print(f"인코딩 오류 발생, 텍스트 건너뛰기: {str(ude)}")
                continue
            except Exception as e:
                # 기타 오류 발생 시 건너뛰기
                print(f"명사 추출 중 오류 발생: {str(e)}")
                continue
                
        except Exception as e:
            print(f"텍스트 처리 중 오류 발생: {str(e)}")
            continue

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
            collocations=False, # 연어(collocations) 방지
            max_words=max_words_wc # 표시할 최대 단어 수 설정
        ).generate(' '.join(nouns))
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig # WordCloud 이미지 자체가 아닌 Figure 객체를 반환
    except Exception as e:
        print(f"WordCloud 생성 중 오류: {e}. 폰트 문제일 수 있습니다.")
        plt.close(fig) # 오류 발생 시 생성된 figure 닫기
        return None

def display_cluster_wordclouds(viz_data, n_clusters, stopwords, max_words_wc=100):
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
            wordcloud_fig = generate_wordcloud_for_cluster(cluster_texts, stopwords, max_words_wc)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
                plt.close(wordcloud_fig) # 사용 후 figure 닫기
            else:
                st.write("WordCloud를 생성할 충분한 단어가 없거나, 생성 중 오류가 발생했습니다.")

# LDA 관련 함수
def preprocess_text_for_lda(texts, stopwords):
    """LDA 모델링을 위한 텍스트 전처리"""
    from text_utils import IMPORTANT_SINGLE_CHAR_NOUNS  # 중요 한 글자 명사 목록 가져오기
    okt = Okt()
    processed_texts = []
    
    for text in texts:
        # 텍스트 정제 및 명사 추출
        cleaned_text = clean_text(str(text))
        # 불용어가 아니고, 2글자 이상이거나 중요 한 글자 명사 목록에 있는 단어만 포함
        nouns = [noun for noun in okt.nouns(cleaned_text) 
                if noun not in stopwords and (len(noun) > 1 or noun in IMPORTANT_SINGLE_CHAR_NOUNS)]
        
        if nouns:  # 빈 리스트가 아닌 경우만 추가
            processed_texts.append(nouns)
            
    return processed_texts

def train_lda_model(texts, num_topics=5, passes=15):
    """LDA 모델 학습"""
    # 텍스트가 충분한지 확인
    if len(texts) < 5:
        return None, None, None
    
    # 사전 및 코퍼스 생성
    dictionary = corpora.Dictionary(texts)
    
    # 너무 빈도가 낮거나 높은 단어 필터링 (옵션)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    
    # 코퍼스 생성
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # LDA 모델 학습
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, corpus, dictionary

def visualize_lda_topics(lda_model, corpus, dictionary):
    """LDA 토픽을 시각화"""
    # pyLDAvis로 시각화
    try:
        vis_data = pyLDAvis.gensim_models.prepare(
            lda_model, corpus, dictionary, mds='mmds'
        )
        
        # HTML 시각화 생성
        html_vis = pyLDAvis.prepared_data_to_html(vis_data)
        return html_vis
    except Exception as e:
        st.error(f"LDA 시각화 중 오류 발생: {str(e)}")
        return None

def display_cluster_lda(viz_data, n_clusters, stopwords, num_topics=3):
    """클러스터별 LDA 토픽 모델링 및 시각화"""
    if not LDA_AVAILABLE:
        st.warning("LDA 토픽 모델링을 위한 패키지(gensim, pyLDAvis)가 설치되어 있지 않습니다.")
        st.info("pip install gensim pyLDAvis 명령으로 필요한 패키지를 설치하세요.")
        return
        
    # st.subheader("클러스터별 LDA 토픽 모델링")
    # st.write("lambda=1일 때는 빈도 기반, lambda=0일 때는 토픽 내 특이성 기반으로 단어를 정렬합니다. 0.6 ~ 0.8 사이의 값을 추천합니다.")
    
    for cluster_id in range(n_clusters):
        cluster_texts = viz_data[viz_data['cluster'] == cluster_id]['full_text'].tolist()
        
        with st.expander(f"클러스터 {cluster_id} LDA 토픽 분석 ({len(cluster_texts)}개 문서)"):
            if not cluster_texts or len(cluster_texts) < 5:
                st.write(f"클러스터 {cluster_id}: 문서가 부족하여 LDA 분석을 수행할 수 없습니다. (최소 5개 필요)")
                continue
                
            st.info(f"{len(cluster_texts)}개 문서에 대한 LDA 토픽 모델링을 수행합니다.")
            
            # 텍스트 전처리
            processed_texts = preprocess_text_for_lda(cluster_texts, stopwords)
            
            if not processed_texts:
                st.write("처리된 텍스트가 없습니다.")
                continue
                
            # LDA 모델 학습
            with st.spinner("LDA 모델 학습 중..."):
                # 클러스터 크기에 따라 토픽 수 조정
                adjusted_topics = min(num_topics, max(2, len(processed_texts) // 3))
                
                lda_model, corpus, dictionary = train_lda_model(
                    processed_texts, 
                    num_topics=adjusted_topics
                )
            
            if lda_model is None:
                st.write("LDA 모델 학습에 실패했습니다.")
                continue
            
            # 인터랙티브 LDA 시각화 (PyLDAvis)
            st.subheader(f"클러스터 {cluster_id}의 인터랙티브 토픽 시각화")
            with st.spinner("인터랙티브 시각화 생성 중..."):
                html_vis = visualize_lda_topics(lda_model, corpus, dictionary)
                if html_vis:
                    # 높이는 고정, 너비는 반응형
                    html_height = 900
                    
                    # 좌측 영역이 잘리지 않도록 패딩과 여백 추가
                    html_vis = html_vis.replace(
                        '<div id="ldavis_el"', 
                        '<div id="ldavis_el" style="padding-left: 40px; box-sizing: border-box;"'
                    )
                    
                    # HTML 내부의 width 속성을 100%로 수정하여 반응형으로 만듦
                    html_vis = html_vis.replace('width="100%"', 'width="100%"').replace('height="530px"', f'height="{html_height}px"')
                    
                    # iframe 태그를 이용해 HTML 시각화를 반응형으로 표시
                    st.components.v1.html(
                        f"""
                        <div style="width:100%; padding-left: 20px; box-sizing: border-box; overflow-x: visible;">
                            {html_vis}
                        </div>
                        """, 
                        height=html_height + 50, # 높이를 약간 증가
                        scrolling=True
                    )
                else:
                    st.write("인터랙티브 시각화를 생성할 수 없습니다.")

def process_cluster_lda(cluster_texts, cluster_id, stopwords, num_topics=3):
    """
    단일 클러스터에 대한 LDA 토픽 모델링을 수행합니다. (expander 없이)
    """
    if not LDA_AVAILABLE:
        st.warning("LDA 토픽 모델링을 위한 패키지(gensim, pyLDAvis)가 설치되어 있지 않습니다.")
        st.info("pip install gensim pyLDAvis 명령으로 필요한 패키지를 설치하세요.")
        return
    
    st.info(f"{len(cluster_texts)}개 문서에 대한 LDA 토픽 모델링을 수행합니다.")
    
    # 텍스트 전처리
    processed_texts = preprocess_text_for_lda(cluster_texts, stopwords)
    
    if not processed_texts:
        st.write("처리된 텍스트가 없습니다.")
        return
        
    # LDA 모델 학습
    with st.spinner("LDA 모델 학습 중..."):
        # 클러스터 크기에 따라 토픽 수 조정
        adjusted_topics = min(num_topics, max(2, len(processed_texts) // 3))
        
        lda_model, corpus, dictionary = train_lda_model(
            processed_texts, 
            num_topics=adjusted_topics
        )
    
    if lda_model is None:
        st.write("LDA 모델 학습에 실패했습니다.")
        return
    
    # 인터랙티브 LDA 시각화 (PyLDAvis)
    st.subheader("인터랙티브 토픽 시각화")
    with st.spinner("인터랙티브 시각화 생성 중..."):
        html_vis = visualize_lda_topics(lda_model, corpus, dictionary)
        if html_vis:
            # 높이는 고정, 너비는 반응형
            html_height = 900
            
            # 좌측 영역이 잘리지 않도록 패딩과 여백 추가
            html_vis = html_vis.replace(
                '<div id="ldavis_el"', 
                '<div id="ldavis_el" style="padding-left: 40px; box-sizing: border-box;"'
            )
            
            # HTML 내부의 width 속성을 100%로 수정하여 반응형으로 만듦
            html_vis = html_vis.replace('width="100%"', 'width="100%"').replace('height="530px"', f'height="{html_height}px"')
            
            # iframe 태그를 이용해 HTML 시각화를 반응형으로 표시
            st.components.v1.html(
                f"""
                <div style="width:100%; padding-left: 20px; box-sizing: border-box; overflow-x: visible;">
                    {html_vis}
                </div>
                """, 
                height=html_height + 50, # 높이를 약간 증가
                scrolling=True
            )
        else:
            st.write("인터랙티브 시각화를 생성할 수 없습니다.")

def find_optimal_clusters(embeddings, max_clusters=10):
    """
    실루엣 스코어를 사용하여 최적의 클러스터 수를 찾기 위한 함수
    
    Args:
        embeddings: 임베딩 배열
        max_clusters: 분석할 최대 클러스터 수
        
    Returns:
        (pd.DataFrame, dict): 클러스터 수별 실루엣 스코어 데이터프레임, 최적 클러스터 정보
    """
    iters = []
    silhouette_avg = []
    
    # 클러스터 수에 따른 실루엣 스코어 계산
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 실루엣 스코어 계산
        silhouette_avg.append(silhouette_score(embeddings, cluster_labels))
        iters.append(n_clusters)
    
    # 결과 데이터프레임 생성
    cluster_silhouette_df = pd.DataFrame({
        '클러스터 수': iters,
        '실루엣 스코어': silhouette_avg
    })
    
    # 최적의 클러스터 수 찾기 (실루엣 스코어가 가장 높은 지점)
    optimal_clusters = cluster_silhouette_df.loc[cluster_silhouette_df['실루엣 스코어'].idxmax()]
    
    return cluster_silhouette_df, optimal_clusters

def plot_elbow_method(silhouette_df):
    """
    엘보우 방법 결과를 시각화하는 함수
    """
    st.subheader("엘보우 방법에 의한 최적 클러스터 수")
    
    # 도움말 표시
    with st.expander("엘보우 방법과 실루엣 스코어 이해하기", expanded=True):
        st.markdown("""
        ### 엘보우 방법(Elbow Method)
        클러스터 개수를 결정하기 위한 시각적 방법으로, 클러스터 수에 따른 성능 지표(여기서는 실루엣 스코어)를 
        그래프로 그려 급격한 변화가 일어나는 '팔꿈치' 지점을 찾는 방법입니다.
        
        ### 실루엣 스코어(Silhouette Score)
        각 데이터 포인트가 자신의 클러스터와 얼마나 잘 맞는지, 그리고 다른 클러스터와 얼마나 잘 분리되는지를 측정합니다.
        
        - **값의 범위**: -1(최악) ~ 1(최상)
        - **해석**:
          - 1에 가까울수록: 데이터가 자신의 클러스터와 잘 맞고, 다른 클러스터와 잘 분리됨
          - 0에 가까울수록: 데이터가 클러스터 경계에 위치함
          - -1에 가까울수록: 데이터가 잘못된 클러스터에 배정됨
          
        ### 최적 클러스터 수 결정
        실루엣 스코어가 가장 높은 클러스터 수가 데이터의 자연스러운 군집 구조를 가장 잘 표현하는 것으로 판단합니다.
        """)
    
    fig = go.Figure()
    
    # 실루엣 스코어 선 그래프
    fig.add_trace(go.Scatter(
        x=silhouette_df['클러스터 수'],
        y=silhouette_df['실루엣 스코어'],
        mode='lines+markers',
        marker=dict(size=10, color='blue'),
        line=dict(width=2),
        name='실루엣 스코어'
    ))
    
    # 최적 클러스터 수 점 표시
    optimal_point = silhouette_df.loc[silhouette_df['실루엣 스코어'].idxmax()]
    fig.add_trace(go.Scatter(
        x=[optimal_point['클러스터 수']],
        y=[optimal_point['실루엣 스코어']],
        mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        name='최적 클러스터 수'
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title='엘보우 방법: 클러스터 수에 따른 실루엣 스코어',
        xaxis=dict(title='클러스터 수'),
        yaxis=dict(title='실루엣 스코어'),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_optimal_cluster_info(optimal_clusters):
    """최적 클러스터 수 정보 표시"""
    st.subheader("최적 클러스터 수")
    
    # 최적 클러스터 정보 표시
    st.write(f"최적의 클러스터 수는 **{optimal_clusters['클러스터 수']}** 개입니다.")
    st.write(f"해당 클러스터 수에서의 실루엣 스코어: **{optimal_clusters['실루엣 스코어']:.4f}**")
    
    # 실루엣 스코어에 대한 해석 추가
    score = optimal_clusters['실루엣 스코어']
    
    st.subheader("결과 해석")
    if score > 0.7:
        st.success("실루엣 스코어가 0.7 이상으로 매우 강한 클러스터 구조를 가지고 있습니다.")
        st.write("데이터가 자연스럽게 잘 분리된 군집 구조를 가지고 있으며, 각 클러스터가 명확하게 구분됩니다.")
    elif score > 0.5:
        st.success("실루엣 스코어가 0.5 이상으로 합리적인 클러스터 구조를 가지고 있습니다.")
        st.write("데이터가 비교적 잘 분리된 군집 구조를 가지고 있으며, 클러스터 간 구분이 양호합니다.")
    elif score > 0.3:
        st.warning("실루엣 스코어가 0.3~0.5 사이로 약한 클러스터 구조를 가지고 있습니다.")
        st.write("클러스터 간 일부 중첩이 있을 수 있으나, 전반적으로 의미 있는 패턴이 발견됩니다.")
    else:
        st.error("실루엣 스코어가 0.3 미만으로 클러스터 구조가 불분명합니다.")
        st.write("데이터에 명확한 자연스러운 군집이 없거나, 선택한 임베딩 방법이 데이터의 특성을 잘 반영하지 못할 수 있습니다.")
    
    st.write("### 최적 클러스터 수가 의미하는 것")
    st.write("""
    이 최적의 클러스터 수는 데이터의 자연스러운 군집 구조를 가장 잘 표현하는 값입니다. 
    이상적인 클러스터링에서는:
    
    1. **동일 클러스터 내 문서**는 서로 유사한 주제나 내용을 가집니다.
    2. **다른 클러스터의 문서**는 서로 다른 주제나 내용을 가집니다.
    3. **각 클러스터**는 고유한 특성이나 주제를 대표합니다.
    
    클러스터 수가 너무 적으면 다른 주제의 문서들이 하나의 클러스터에 포함되고, 
    클러스터 수가 너무 많으면 유사한 주제의 문서들이 여러 클러스터로 불필요하게 분리됩니다.
    """)
    
    st.write("### 클러스터 분석 활용 방법")
    st.write("""
    - 클러스터별 WordCloud와 주요 문서를 검토하여 각 클러스터의 핵심 주제를 파악하세요.
    - LDA 토픽 모델링 결과를 통해 각 클러스터 내의 세부 주제 구조를 확인하세요.
    - 클러스터 간 비교를 통해 데이터의 전체적인 구조와 패턴을 이해하세요.
    """)
