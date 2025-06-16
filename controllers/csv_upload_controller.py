import streamlit as st
import pandas as pd
import os
import traceback
from typing import Dict, List, Optional, Any
from models.csv_upload_model import CsvUploadModel

class CsvUploadController:
    """CSV Upload 페이지의 컨트롤러"""
    
    def __init__(self):
        self.model = CsvUploadModel()
    
    def handle_db_path_validation(self, db_path: str) -> str:
        """DB 경로 검증 및 처리"""
        default_db_path = "./chroma_db"
        
        if not os.path.exists(db_path):
            return default_db_path
        return db_path
    
    def handle_file_upload(self, file_source, encoding: str) -> Dict[str, Any]:
        """파일 업로드 처리"""
        return self.model.load_csv_file(file_source, encoding)
    
    def handle_dataframe_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터프레임 분석 처리"""
        return self.model.get_dataframe_info(df)
    
    def handle_preprocessing_preview(self, df: pd.DataFrame, selected_columns: List[str], 
                                   max_rows: Optional[int] = None) -> Dict[str, Any]:
        """전처리 미리보기 처리"""
        if not selected_columns:
            return {
                'success': False,
                'error': "열을 선택해주세요."
            }
        
        return self.model.get_preprocessing_preview(df, selected_columns, max_rows)
    
    def handle_collection_deletion(self, collection_name: str, db_path: str) -> Dict[str, Any]:
        """컬렉션 삭제 처리"""
        try:
            success = self.model.delete_collection(collection_name, db_path)
            
            if success:
                self.model.reset_delete_confirmation()
                return {
                    'success': True,
                    'message': f"'{collection_name}' 컬렉션이 삭제되었습니다."
                }
            else:
                return {
                    'success': False,
                    'error': f"'{collection_name}' 컬렉션 삭제 중 오류가 발생했습니다."
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"컬렉션 삭제 중 오류: {e}"
            }
    
    def handle_chroma_storage(self, df: pd.DataFrame, selected_columns: List[str],
                            collection_name: str, persist_directory: str,
                            max_rows: Optional[int], batch_size: int,
                            progress_bar, status_text, gpu_status_placeholder) -> Dict[str, Any]:
        """ChromaDB 저장 처리"""
        if not selected_columns:
            return {
                'success': False,
                'error': "저장할 열을 하나 이상 선택하세요."
            }
        
        try:
            result = self.model.store_data_to_chroma(
                df, selected_columns, collection_name, persist_directory,
                max_rows, batch_size, progress_bar, status_text, gpu_status_placeholder
            )
            
            if result['success']:
                # 성공 메시지 생성
                embedding_status = result['embedding_status']
                messages = []
                
                # 임베딩 상태 메시지 처리
                if embedding_status["fallback_used"]:
                    warning_message = f"""⚠️ **임베딩 모델 변경됨**:
                    - 요청 모델: {embedding_status["requested_model"]}
                    - 실제 사용 모델: {embedding_status["actual_model"]}
                    - 사용된 장치: {embedding_status["device_used"]} (요청: {embedding_status["device_preference"]})"""
                    
                    if embedding_status["error_message"]:
                        warning_message += f"\n- 원인: {embedding_status['error_message']}"
                    
                    messages.append({
                        'type': 'warning',
                        'content': warning_message
                    })
                else:
                    messages.append({
                        'type': 'info',
                        'content': f"임베딩 모델: {embedding_status['actual_model']}, 사용 장치: {embedding_status['device_used']} (요청: {embedding_status['device_preference']})"
                    })
                
                # GPU OOM 메시지
                if embedding_status.get("error_message") and "GPU OOM" in embedding_status["error_message"]:
                    messages.append({
                        'type': 'info',
                        'content': "GPU 메모리 부족으로 일부 또는 전체 배치가 CPU에서 처리되었을 수 있습니다. 자세한 내용은 로그를 확인하세요."
                    })
                
                # 성공 메시지
                if result['max_process_rows'] is not None:
                    success_message = f"ChromaDB에 데이터가 성공적으로 저장되었습니다. 컬렉션: {result['collection_name']} (처리된 행 수: {result['max_process_rows']})"
                else:
                    success_message = f"ChromaDB에 데이터가 성공적으로 저장되었습니다. 컬렉션: {result['collection_name']}"
                
                messages.append({
                    'type': 'success',
                    'content': success_message
                })
                
                return {
                    'success': True,
                    'messages': messages
                }
            else:
                return result
                
        except Exception as e:
            self.model.logger.error(f"ChromaDB 저장 중 오류 발생: {e}")
            self.model.logger.debug(traceback.format_exc())
            return {
                'success': False,
                'error': f"ChromaDB 저장 중 오류 발생: {e}"
            }
    
    def get_available_collections(self, db_path: str) -> List[str]:
        """사용 가능한 컬렉션 목록 반환"""
        return self.model.get_available_collections(db_path)
    
    def get_available_embedding_models(self) -> Dict[str, List[str]]:
        """사용 가능한 임베딩 모델 반환"""
        return self.model.get_available_embedding_models()
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 반환"""
        return self.model.get_gpu_info()
    
    def prepare_delete_confirmation(self, collection_name: str):
        """삭제 확인 준비"""
        self.model.update_session_state('collection_to_delete', collection_name)
        self.model.update_session_state('show_delete_confirm', True)
    
    def cancel_delete_confirmation(self):
        """삭제 확인 취소"""
        self.model.reset_delete_confirmation()
    
    def update_collection_state(self, selected_collection: str, db_path: str):
        """컬렉션 상태 업데이트"""
        self.model.update_collection_state(selected_collection, db_path)
        self.model.update_session_state('collection_name', selected_collection)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        try:
            collection = st.session_state.chroma_collection
            if collection is None:
                return {
                    'success': False,
                    'error': "로드된 컬렉션이 없습니다."
                }
            
            collection_info = collection.count()
            
            # 컬렉션에 저장된 임베딩 모델 정보 확인
            embedding_model = "알 수 없음"
            try:
                if collection.metadata and "embedding_model" in collection.metadata:
                    embedding_model = collection.metadata["embedding_model"]
            except:
                pass
            
            return {
                'success': True,
                'count': collection_info,
                'embedding_model': embedding_model
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"컬렉션 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"
            }
    
    def validate_file_path(self, file_path: str) -> bool:
        """파일 경로 검증"""
        return file_path and os.path.isfile(file_path)
    
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