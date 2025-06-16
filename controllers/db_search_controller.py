import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Optional, Any
from models.db_search_model import DbSearchModel
import utils.db_search_utils as db_search_utils
import utils.visualization_utils as visualization_utils
from utils import KOREAN_STOPWORDS

class DbSearchController:
    """DB Search 페이지의 컨트롤러"""
    
    def __init__(self):
        self.model = DbSearchModel()
    
    def handle_db_path_validation(self, db_path: str) -> str:
        """DB 경로 검증 및 처리"""
        default_db_path = "./chroma_db"
        
        if not os.path.exists(db_path):
            return default_db_path
        return db_path
    
    def handle_collection_load(self, collection_name: str, db_path: str) -> Dict[str, Any]:
        """컬렉션 로드 처리"""
        result = self.model.load_collection(collection_name, db_path)
        
        if result['success']:
            messages = []
            embedding_status = result['embedding_status']
            
            messages.append({
                'type': 'success',
                'content': f"컬렉션 '{result['collection_name']}'을 성공적으로 로드했습니다."
            })
            
            if embedding_status.get("fallback_used"):
                messages.append({
                    'type': 'warning',
                    'content': f"""⚠️ **임베딩 모델 변경됨**: 요청하신 모델 대신 기본 임베딩 모델이 사용되었습니다.
                    - 요청 모델: {embedding_status["requested_model"]}
                    - 사용된 모델: {embedding_status["actual_model"]}
                    - 사용된 장치: {embedding_status["device_used"]} (요청: {embedding_status["device_preference"]})"""
                })
                if embedding_status.get("error_message"):
                    messages[-1]['content'] += f"\n- 원인: {embedding_status['error_message']}"
            else:
                messages.append({
                    'type': 'info',
                    'content': f"✅ 임베딩 모델 로드됨: {embedding_status['actual_model']} (장치: {embedding_status['device_used']})"
                })
            
            return {
                'success': True,
                'messages': messages
            }
        else:
            return result
    
    def handle_collection_data_load(self) -> Dict[str, Any]:
        """컬렉션 데이터 로드 처리"""
        if not st.session_state.collection_loaded:
            return {
                'success': False,
                'error': "먼저 사이드바에서 컬렉션을 로드하세요."
            }
        
        return self.model.load_collection_data()
    
    def handle_search(self, query: str, similarity_threshold: float, 
                     use_search_for_viz: bool = False) -> Dict[str, Any]:
        """검색 처리"""
        if not query.strip():
            return {
                'success': False,
                'error': "검색어를 입력하세요."
            }
        
        if not st.session_state.collection_loaded:
            return {
                'success': False,
                'error': "먼저 사이드바에서 컬렉션을 로드하세요."
            }
        
        try:
            # 검색 실행
            search_result = self.model.search_collection(
                query, similarity_threshold, include_embeddings=use_search_for_viz
            )
            
            if not search_result['success']:
                return search_result
            
            result_df = search_result['data']
            
            # 검색 결과 저장
            st.session_state.last_search_results = result_df
            
            # 시각화용 검색 결과 저장
            if use_search_for_viz and 'embeddings' in search_result:
                st.session_state.search_results_for_viz = {
                    'query': query,
                    'df': result_df,
                    'embeddings': search_result['embeddings'],
                    'threshold': similarity_threshold
                }
            
            return {
                'success': True,
                'data': result_df,
                'use_for_viz': use_search_for_viz,
                'query': query
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"검색 중 오류가 발생했습니다: {str(e)}"
            }
    
    def handle_document_deletion(self, selected_docs: pd.DataFrame) -> Dict[str, Any]:
        """문서 삭제 처리"""
        if selected_docs is None or selected_docs.empty:
            return {
                'success': False,
                'error': "삭제할 문서를 선택하세요."
            }
        
        result = self.model.delete_documents(selected_docs)
        
        if result['success']:
            # 삭제 성공 메시지 설정
            st.session_state.delete_success_message = f"{result['deleted_count']}개 문서가 성공적으로 삭제되었습니다."
        
        return result
    
    def handle_visualization_creation(self, use_search_results: bool, docs_percentage: int,
                                    find_optimal: bool, max_clusters: int, n_clusters: int,
                                    perplexity: int) -> Dict[str, Any]:
        """시각화 생성 처리"""
        if not st.session_state.collection_loaded:
            return {
                'success': False,
                'error': "먼저 사이드바에서 컬렉션을 로드하세요."
            }
        
        # 시각화 상태 초기화
        self.model.reset_visualization_state()
        
        try:
            # 시각화 데이터 준비
            prep_result = self.model.prepare_visualization_data(
                use_search_results, docs_percentage, find_optimal, 
                max_clusters, n_clusters, perplexity
            )
            
            if not prep_result['success']:
                return prep_result
            
            documents = prep_result['documents']
            metadatas = prep_result['metadatas']
            ids = prep_result['ids']
            embeddings_array = prep_result['embeddings_array']
            n_clusters = prep_result['adjusted_n_clusters']
            max_clusters = prep_result.get('adjusted_max_clusters')
            
            messages = []
            
            # 검색 결과 사용 메시지
            if use_search_results and st.session_state.search_results_for_viz:
                search_info = st.session_state.search_results_for_viz
                messages.append({
                    'type': 'success',
                    'content': f"검색 결과 '{search_info['query']}'에서 {len(documents)}개의 문서를 시각화합니다."
                })
            else:
                messages.append({
                    'type': 'success', 
                    'content': f"총 {len(documents)}개의 문서를 로드했습니다."
                })
            
            # 최적 클러스터 수 찾기
            optimal_result = None
            if find_optimal:
                silhouette_df, optimal_clusters = visualization_utils.find_optimal_clusters(
                    embeddings_array, max_clusters
                )
                n_clusters = int(optimal_clusters["클러스터 수"])
                
                # 세션 상태에 분석 결과 저장
                st.session_state.show_optimal_cluster_analysis = True
                st.session_state.silhouette_df_for_plot = silhouette_df
                st.session_state.optimal_clusters_info = optimal_clusters
                
                optimal_result = {
                    'silhouette_df': silhouette_df,
                    'optimal_clusters': optimal_clusters,
                    'n_clusters': n_clusters
                }
                
                messages.append({
                    'type': 'success',
                    'content': f"최적의 클러스터 수로 {n_clusters}을(를) 사용합니다."
                })
            
            # 시각화 데이터 준비
            viz_data = visualization_utils.prepare_visualization_data(
                embeddings_array, documents, ids, metadatas, perplexity, n_clusters
            )
            
            # 세션 상태에 시각화 데이터 저장
            st.session_state.viz_data = viz_data
            st.session_state.n_clusters = n_clusters
            st.session_state.viz_completed = True
            
            return {
                'success': True,
                'messages': messages,
                'optimal_result': optimal_result,
                'viz_data': viz_data,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"시각화 생성 중 오류가 발생했습니다: {str(e)}"
            }
    
    def handle_lda_topic_modeling(self, selected_cluster: str, lda_topics: int) -> Dict[str, Any]:
        """LDA 토픽 모델링 처리"""
        try:
            viz_data = st.session_state.get('viz_data')
            n_clusters = st.session_state.get('n_clusters')
            
            if viz_data is None or n_clusters is None:
                return {
                    'success': False,
                    'error': "시각화 데이터가 없습니다. 먼저 시각화를 생성하세요."
                }
            
            results = []
            
            if selected_cluster == "모든 클러스터":
                # 모든 클러스터 처리
                for cluster_id in range(n_clusters):
                    cluster_texts = viz_data[viz_data['cluster'] == cluster_id]['full_text'].tolist()
                    if len(cluster_texts) >= 5:
                        result = visualization_utils.process_cluster_lda(
                            cluster_texts, cluster_id, KOREAN_STOPWORDS, lda_topics
                        )
                        results.append({
                            'cluster_id': cluster_id,
                            'doc_count': len(cluster_texts),
                            'success': True
                        })
                    else:
                        results.append({
                            'cluster_id': cluster_id,
                            'doc_count': len(cluster_texts),
                            'success': False,
                            'reason': '문서 부족 (최소 5개 필요)'
                        })
            else:
                # 특정 클러스터만 처리
                cluster_id = int(selected_cluster.split(" ")[1])
                cluster_texts = viz_data[viz_data['cluster'] == cluster_id]['full_text'].tolist()
                if len(cluster_texts) >= 5:
                    result = visualization_utils.process_cluster_lda(
                        cluster_texts, cluster_id, KOREAN_STOPWORDS, lda_topics
                    )
                    results.append({
                        'cluster_id': cluster_id,
                        'doc_count': len(cluster_texts),
                        'success': True
                    })
                else:
                    results.append({
                        'cluster_id': cluster_id,
                        'doc_count': len(cluster_texts),
                        'success': False,
                        'reason': '문서 부족 (최소 5개 필요)'
                    })
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"LDA 토픽 모델링 중 오류가 발생했습니다: {str(e)}"
            }
    
    def get_available_collections(self, db_path: str) -> List[str]:
        """사용 가능한 컬렉션 목록 반환"""
        return self.model.get_available_collections(db_path)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 반환"""
        return self.model.get_gpu_info()
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        return self.model.get_collection_info()
    
    def check_collection_change(self, selected_collection: str, db_path: str) -> bool:
        """컬렉션이나 경로 변경 확인"""
        return self.model.check_collection_change(selected_collection, db_path)
    
    def get_embedding_device_options(self, gpu_info: Dict[str, Any]) -> Dict[str, str]:
        """임베딩 장치 옵션 반환"""
        if gpu_info["available"]:
            return {
                "자동 (GPU 우선 사용)": "auto",
                "GPU 강제 사용": "cuda",
                "CPU 전용 사용": "cpu"
            }
        else:
            return {"CPU 전용 사용": "cpu"}
    
    def update_session_state(self, key: str, value: Any):
        """세션 상태 업데이트"""
        self.model.update_session_state(key, value)
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """세션 상태 반환"""
        return self.model.get_session_state(key, default)