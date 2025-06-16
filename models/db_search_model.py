import streamlit as st
import pandas as pd
import os
import sys
from typing import Dict, List, Optional, Any
from utils import (
    load_chroma_collection, 
    get_available_collections, 
    get_embedding_function, 
    get_embedding_status,
    get_gpu_info
)
import utils.db_search_utils as db_search_utils
import utils.visualization_utils as visualization_utils

class DbSearchModel:
    """DB Search 페이지의 데이터 모델"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_path()
    
    def setup_path(self):
        """상위 디렉토리를 경로에 추가"""
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        session_defaults = {
            'chroma_client': None,
            'chroma_collection': None,
            'collection_loaded': False,
            'current_collection_name': None,
            'current_db_path': None,
            'embedding_model': None,
            'embedding_device_preference': "auto",
            'viz_completed': False,
            'viz_data': None,
            'n_clusters': None,
            'max_words_wc_slider': 100,
            'show_optimal_cluster_analysis': False,
            'silhouette_df_for_plot': None,
            'optimal_clusters_info': None,
            'last_search_results': None,
            'search_results_for_viz': None,
            'delete_success_message': None,
            'perplexity': 30,
            'lda_topics': 6
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def get_available_collections(self, db_path: str) -> List[str]:
        """사용 가능한 컬렉션 목록 반환"""
        return get_available_collections(persist_directory=db_path)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 반환"""
        return get_gpu_info()
    
    def load_collection(self, collection_name: str, db_path: str) -> Dict[str, Any]:
        """컬렉션 로드"""
        try:
            client, collection = load_chroma_collection(
                collection_name=collection_name,
                persist_directory=db_path
            )
            
            # 컬렉션 메타데이터에서 임베딩 모델 정보 가져오기
            embedding_model_name = collection.metadata.get("embedding_model", "all-MiniLM-L6-v2")
            
            # 임베딩 모델 로드
            device_preference = st.session_state.get('embedding_device_preference', 'auto')
            embedding_model_func = get_embedding_function(embedding_model_name, device_preference=device_preference)
            
            if embedding_model_func is None:
                return {
                    'success': False,
                    'error': "임베딩 모델 로드에 실패했습니다."
                }
            
            # 세션 상태 업데이트
            st.session_state.chroma_client = client
            st.session_state.chroma_collection = collection
            st.session_state.embedding_model = embedding_model_func
            st.session_state.collection_loaded = True
            st.session_state.current_collection_name = collection_name
            st.session_state.current_db_path = db_path
            
            # 임베딩 상태 확인
            embedding_status = get_embedding_status()
            
            return {
                'success': True,
                'embedding_status': embedding_status,
                'collection_name': collection_name
            }
            
        except Exception as e:
            # 오류 발생 시 세션 상태 초기화
            self.reset_collection_state()
            return {
                'success': False,
                'error': f"컬렉션 로드 중 오류 발생: {e}"
            }
    
    def reset_collection_state(self):
        """컬렉션 관련 세션 상태 초기화"""
        st.session_state.chroma_client = None
        st.session_state.chroma_collection = None
        st.session_state.collection_loaded = False
        st.session_state.embedding_model = None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        try:
            collection = st.session_state.chroma_collection
            if collection is None:
                return {
                    'success': False,
                    'error': "로드된 컬렉션이 없습니다."
                }
            
            collection_count = collection.count()
            
            # 컬렉션에 저장된 임베딩 모델 정보 확인
            embedding_model = "알 수 없음"
            try:
                if collection.metadata and "embedding_model" in collection.metadata:
                    embedding_model = collection.metadata["embedding_model"]
            except:
                pass
            
            # 현재 로드된 임베딩 모델 상태
            embedding_status = get_embedding_status()
            
            return {
                'success': True,
                'count': collection_count,
                'embedding_model': embedding_model,
                'current_model': embedding_status.get('actual_model', 'N/A'),
                'device_used': embedding_status.get('device_used', 'N/A'),
                'device_preference': embedding_status.get('device_preference', 'N/A'),
                'fallback_used': embedding_status.get('fallback_used', False),
                'error_message': embedding_status.get('error_message')
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"컬렉션 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"
            }
    
    def load_collection_data(self) -> Dict[str, Any]:
        """컬렉션 데이터 로드"""
        try:
            collection = st.session_state.chroma_collection
            if collection is None:
                return {
                    'success': False,
                    'error': "로드된 컬렉션이 없습니다."
                }
            
            result_df, _ = db_search_utils.load_collection_data(collection)
            
            if result_df is not None:
                return {
                    'success': True,
                    'data': result_df,
                    'count': len(result_df)
                }
            else:
                return {
                    'success': False,
                    'error': "컬렉션에 데이터가 없습니다."
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"데이터 로드 중 오류가 발생했습니다: {str(e)}"
            }
    
    def search_collection(self, query: str, similarity_threshold: float, 
                         include_embeddings: bool = False) -> Dict[str, Any]:
        """컬렉션 검색"""
        try:
            collection = st.session_state.chroma_collection
            embed_fn = st.session_state.embedding_model
            
            if collection is None or embed_fn is None:
                return {
                    'success': False,
                    'error': "컬렉션 또는 임베딩 모델이 로드되지 않았습니다."
                }
            
            if include_embeddings:
                result_df, embeddings = db_search_utils.search_collection_by_similarity_full(
                    collection, query, similarity_threshold, 
                    include_embeddings=True, embed_fn=embed_fn
                )
                return {
                    'success': True,
                    'data': result_df,
                    'embeddings': embeddings
                }
            else:
                result_df = db_search_utils.search_collection_by_similarity_full(
                    collection, query, similarity_threshold, embed_fn=embed_fn
                )
                return {
                    'success': True,
                    'data': result_df
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"검색 중 오류가 발생했습니다: {str(e)}"
            }
    
    def delete_documents(self, selected_docs: pd.DataFrame) -> Dict[str, Any]:
        """선택한 문서 삭제"""
        try:
            collection = st.session_state.chroma_collection
            if collection is None:
                return {
                    'success': False,
                    'error': "컬렉션이 로드되지 않아 문서를 삭제할 수 없습니다."
                }
            
            # 선택된 문서의 ID 추출
            if "ID" in selected_docs.columns:
                doc_ids = selected_docs["ID"].tolist()
            else:
                # 내용 기반으로 ID 찾기
                docs_to_delete = selected_docs["내용"].tolist()
                all_docs, _ = db_search_utils.load_collection_data(collection)
                
                if all_docs is not None:
                    doc_ids = []
                    for doc in docs_to_delete:
                        matching_rows = all_docs[all_docs["내용"] == doc]
                        if not matching_rows.empty:
                            found_ids = matching_rows["ID"].tolist()
                            doc_ids.extend(found_ids)
                    doc_ids = list(set(doc_ids))  # 중복 제거
                else:
                    return {
                        'success': False,
                        'error': "문서 ID를 찾을 수 없어 삭제할 수 없습니다."
                    }
            
            if not doc_ids:
                return {
                    'success': False,
                    'error': "삭제할 문서의 ID를 찾을 수 없습니다."
                }
            
            # 문서 삭제
            collection.delete(ids=doc_ids)
            
            # 마지막 검색 결과에서 삭제된 문서 제거
            if st.session_state.last_search_results is not None:
                st.session_state.last_search_results = st.session_state.last_search_results[
                    ~st.session_state.last_search_results["ID"].isin(doc_ids)
                ]
            
            return {
                'success': True,
                'deleted_count': len(doc_ids)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"문서 삭제 중 오류가 발생했습니다: {str(e)}"
            }
    
    def prepare_visualization_data(self, use_search_results: bool, docs_percentage: int,
                                 find_optimal: bool, max_clusters: int, n_clusters: int,
                                 perplexity: int) -> Dict[str, Any]:
        """시각화 데이터 준비"""
        try:
            collection = st.session_state.chroma_collection
            
            if use_search_results and st.session_state.search_results_for_viz:
                # 검색 결과 사용
                search_results_info = st.session_state.search_results_for_viz
                search_df = search_results_info['df']
                search_embeddings = search_results_info['embeddings']
                
                if search_df.empty or len(search_embeddings) == 0:
                    return {
                        'success': False,
                        'error': "검색 결과가 없거나 임베딩 데이터를 가져올 수 없습니다."
                    }
                
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
                
                import numpy as np
                embeddings_array = np.array(search_embeddings)
                
            else:
                # 전체 컬렉션 사용
                _, all_data = db_search_utils.load_collection_data(collection)
                
                if not all_data or not all_data["documents"]:
                    return {
                        'success': False,
                        'error': "컬렉션에 데이터가 없습니다."
                    }
                
                documents, metadatas, ids, embeddings = visualization_utils.get_embeddings_data(
                    collection, all_data, docs_percentage
                )
                
                if len(embeddings) == 0:
                    embeddings_array = visualization_utils.handle_missing_embeddings(collection, documents)
                    if embeddings_array is None or len(embeddings_array) == 0:
                        return {
                            'success': False,
                            'error': "임베딩 데이터가 없거나 대체 임베딩 생성에 실패했습니다."
                        }
                else:
                    import numpy as np
                    embeddings_array = np.array(embeddings)
            
            # 최소 문서 수 확인
            min_docs = 3
            if len(documents) < min_docs:
                return {
                    'success': False,
                    'error': f"시각화를 위해서는 최소 {min_docs}개 이상의 문서가 필요합니다. 현재: {len(documents)}개"
                }
            
            # 클러스터 수 조정
            if find_optimal:
                max_clusters = min(max_clusters, len(documents) - 1) if len(documents) > 1 else 2
                if max_clusters < 2:
                    return {
                        'success': False,
                        'error': "클러스터링을 위한 문서 수가 부족합니다."
                    }
            else:
                if n_clusters >= len(documents):
                    n_clusters = max(2, len(documents) // 2)
            
            return {
                'success': True,
                'documents': documents,
                'metadatas': metadatas,
                'ids': ids,
                'embeddings_array': embeddings_array,
                'adjusted_n_clusters': n_clusters,
                'adjusted_max_clusters': max_clusters if find_optimal else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"시각화 데이터 준비 중 오류: {str(e)}"
            }
    
    def update_session_state(self, key: str, value: Any):
        """세션 상태 업데이트"""
        st.session_state[key] = value
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """세션 상태 값 반환"""
        return st.session_state.get(key, default)
    
    def check_collection_change(self, selected_collection: str, db_path: str) -> bool:
        """컬렉션이나 경로 변경 확인"""
        return (selected_collection != st.session_state.current_collection_name or 
                db_path != st.session_state.current_db_path)
    
    def reset_visualization_state(self):
        """시각화 관련 세션 상태 초기화"""
        st.session_state.show_optimal_cluster_analysis = False
        st.session_state.silhouette_df_for_plot = None
        st.session_state.optimal_clusters_info = None
        st.session_state.viz_completed = False
        st.session_state.viz_data = None
        st.session_state.n_clusters = None