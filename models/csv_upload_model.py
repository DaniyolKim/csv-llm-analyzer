import streamlit as st
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Any
from utils import (
    get_available_collections,
    get_available_embedding_models,
    get_embedding_status,
    get_gpu_info,
    store_data_in_chroma,
    preprocess_dataframe,
    delete_collection
)

class CsvUploadModel:
    """CSV Upload 페이지의 데이터 모델"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("csv_upload.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("csv_uploader")
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        session_defaults = {
            'df': None,
            'chroma_client': None,
            'chroma_collection': None,
            'selected_columns': [],
            'rag_enabled': False,
            'chroma_path': "./chroma_db",
            'collection_name': "csv_test",
            'embedding_model': "all-MiniLM-L6-v2",
            'embedding_device_preference': "auto",
            'collection_to_delete': None,
            'show_delete_confirm': False,
            'current_collection_name': None,
            'current_db_path': None,
            'collection_loaded': False
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # 현재 상태 초기화
        if st.session_state.current_collection_name is None:
            st.session_state.current_collection_name = st.session_state.collection_name
        if st.session_state.current_db_path is None:
            st.session_state.current_db_path = st.session_state.chroma_path
    
    def load_csv_file(self, file_source, encoding: str = "utf-8") -> Dict[str, Any]:
        """CSV 파일 로드"""
        try:
            if hasattr(file_source, 'read'):  # 업로드된 파일
                df = pd.read_csv(file_source, encoding=encoding)
                st.session_state.df = df
                return {
                    'success': True,
                    'message': "파일 업로드 성공!",
                    'dataframe': df
                }
            else:  # 파일 경로
                if not os.path.isfile(file_source):
                    return {
                        'success': False,
                        'error': f"파일을 찾을 수 없습니다: {file_source}"
                    }
                
                df = pd.read_csv(file_source, encoding=encoding)
                st.session_state.df = df
                return {
                    'success': True,
                    'message': f"파일 로드 성공: {file_source}",
                    'dataframe': df
                }
        except UnicodeDecodeError:
            return {
                'success': False,
                'error': f"{encoding} 인코딩으로 파일을 읽을 수 없습니다. 다른 인코딩을 선택해 주세요."
            }
        except Exception as e:
            self.logger.error(f"파일 로드 중 오류 발생: {e}")
            return {
                'success': False,
                'error': f"파일을 처리하는 중 오류가 발생했습니다: {e}"
            }
    
    def get_dataframe_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터프레임 기본 정보 반환"""
        try:
            col_info = pd.DataFrame({
                '데이터 타입': [str(dtype) for dtype in df.dtypes],
                '고유값 수': df.nunique(),
                '결측치 수': df.isna().sum(),
                '결측치 비율(%)': (df.isna().sum() / len(df) * 100).round(2)
            })
            
            return {
                'success': True,
                'basic_info': {
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'null_values': df.isna().sum().sum(),
                    'duplicates': df.duplicated().sum()
                },
                'column_info': col_info,
                'preview': df.head(10)
            }
        except Exception as e:
            self.logger.error(f"데이터프레임 정보 생성 중 오류: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_preprocessing_preview(self, df: pd.DataFrame, selected_columns: List[str], 
                                max_rows: Optional[int] = None) -> Dict[str, Any]:
        """전처리 미리보기"""
        try:
            max_preview_rows = max_rows if max_rows and max_rows > 0 else None
            processed_df = preprocess_dataframe(df, selected_columns, max_preview_rows)
            
            dropped_rows = df.shape[0] - len(df.dropna(subset=selected_columns))
            
            return {
                'success': True,
                'processed_df': processed_df,
                'original_rows': df.shape[0],
                'processed_rows': processed_df.shape[0],
                'dropped_rows': dropped_rows,
                'max_rows_applied': max_preview_rows,
                'preview': processed_df.head(5) if not processed_df.empty else None
            }
        except Exception as e:
            self.logger.error(f"전처리 미리보기 중 오류: {e}")
            return {
                'success': False,
                'error': f"데이터 전처리 중 오류 발생: {e}"
            }
    
    def get_available_collections(self, db_path: str) -> List[str]:
        """사용 가능한 컬렉션 목록 반환"""
        return get_available_collections(db_path)
    
    def get_available_embedding_models(self) -> Dict[str, List[str]]:
        """사용 가능한 임베딩 모델 반환"""
        return get_available_embedding_models()
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 반환"""
        return get_gpu_info()
    
    def delete_collection(self, collection_name: str, db_path: str) -> bool:
        """컬렉션 삭제"""
        success = delete_collection(collection_name, db_path)
        
        if success and st.session_state.collection_name == collection_name:
            # 현재 로드된 컬렉션이 삭제되었다면 상태 초기화
            st.session_state.chroma_collection = None
            st.session_state.chroma_client = None
            st.session_state.rag_enabled = False
            st.session_state.collection_loaded = False
        
        return success
    
    def store_data_to_chroma(self, df: pd.DataFrame, selected_columns: List[str],
                           collection_name: str, persist_directory: str,
                           max_rows: Optional[int], batch_size: int,
                           progress_bar, status_text, gpu_status_placeholder) -> Dict[str, Any]:
        """데이터를 ChromaDB에 저장"""
        try:
            max_process_rows = max_rows if max_rows and max_rows > 0 else None
            
            client, collection = store_data_in_chroma(
                df,
                selected_columns,
                collection_name,
                persist_directory,
                max_rows=max_process_rows,
                batch_size=batch_size,
                embedding_model=st.session_state.embedding_model,
                embedding_device_preference=st.session_state.embedding_device_preference,
                progress_bar=progress_bar,
                status_text=status_text,
                gpu_status_placeholder=gpu_status_placeholder
            )
            
            # 세션 상태 업데이트
            st.session_state.chroma_client = client
            st.session_state.chroma_collection = collection
            st.session_state.collection_name = collection_name
            st.session_state.chroma_path = persist_directory
            st.session_state.rag_enabled = True
            
            # 임베딩 상태 확인
            embedding_status = get_embedding_status()
            
            return {
                'success': True,
                'embedding_status': embedding_status,
                'max_process_rows': max_process_rows,
                'collection_name': collection_name
            }
            
        except Exception as e:
            self.logger.error(f"ChromaDB 저장 중 오류 발생: {e}")
            return {
                'success': False,
                'error': f"ChromaDB 저장 중 오류 발생: {e}"
            }
    
    def update_session_state(self, key: str, value: Any):
        """세션 상태 업데이트"""
        st.session_state[key] = value
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """세션 상태 값 반환"""
        return st.session_state.get(key, default)
    
    def reset_delete_confirmation(self):
        """삭제 확인 상태 리셋"""
        st.session_state.collection_to_delete = None
        st.session_state.show_delete_confirm = False
    
    def check_collection_change(self, selected_collection: str, db_path: str) -> bool:
        """컬렉션이나 경로 변경 확인"""
        return (selected_collection != st.session_state.current_collection_name or 
                db_path != st.session_state.current_db_path)
    
    def update_collection_state(self, selected_collection: str, db_path: str):
        """컬렉션 상태 업데이트"""
        if self.check_collection_change(selected_collection, db_path):
            st.session_state.collection_loaded = False
        
        st.session_state.current_collection_name = selected_collection
        st.session_state.current_db_path = db_path