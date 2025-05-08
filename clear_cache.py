import os
import shutil
import chromadb
import sys
import time

# ChromaDB 디렉토리 경로
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chromadb')

def clear_chromadb_cache():
    """
    ChromaDB 캐시를 초기화합니다.
    1. 기존 컬렉션 삭제
    2. 디렉토리 삭제 및 재생성
    """
    print(f"ChromaDB 캐시 초기화를 시작합니다...")
    print(f"캐시 디렉토리: {CHROMA_PERSIST_DIR}")
    
    try:
        # 1. 클라이언트 생성 시도
        if os.path.exists(CHROMA_PERSIST_DIR):
            try:
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                print("ChromaDB 클라이언트에 연결했습니다.")
                
                # 2. 모든 컬렉션 삭제
                try:
                    collections = client.list_collections()
                    print(f"발견된 컬렉션: {len(collections)}개")
                    
                    for collection in collections:
                        collection_name = collection.name
                        try:
                            client.delete_collection(name=collection_name)
                            print(f"컬렉션 '{collection_name}' 삭제 완료")
                        except Exception as del_error:
                            print(f"컬렉션 '{collection_name}' 삭제 실패: {str(del_error)}")
                except Exception as list_error:
                    print(f"컬렉션 목록 조회 실패: {str(list_error)}")
            except Exception as client_error:
                print(f"ChromaDB 클라이언트 연결 실패: {str(client_error)}")
        
        # 클라이언트 연결 해제를 위한 대기
        print("리소스 해제를 위해 3초 대기 중...")
        time.sleep(3)
        
        # 3. 디렉토리 삭제 시도 (최대 3번)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if os.path.exists(CHROMA_PERSIST_DIR):
                    shutil.rmtree(CHROMA_PERSIST_DIR)
                    print(f"디렉토리 '{CHROMA_PERSIST_DIR}' 삭제 완료")
                    break
            except Exception as rm_error:
                print(f"디렉토리 삭제 시도 {attempt+1}/{max_attempts} 실패: {str(rm_error)}")
                if attempt < max_attempts - 1:
                    print(f"5초 후 다시 시도합니다...")
                    time.sleep(5)
                else:
                    print("수동으로 디렉토리를 삭제해주세요:")
                    print(f"1. 모든 Python 프로세스를 종료하세요")
                    print(f"2. 다음 디렉토리를 삭제하세요: {CHROMA_PERSIST_DIR}")
                    return False
        
        # 4. 디렉토리 재생성
        try:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            print(f"디렉토리 '{CHROMA_PERSIST_DIR}' 재생성 완료")
        except Exception as mkdir_error:
            print(f"디렉토리 재생성 실패: {str(mkdir_error)}")
            return False
        
        print("ChromaDB 캐시 초기화가 완료되었습니다.")
        return True
    
    except Exception as e:
        print(f"캐시 초기화 중 오류 발생: {str(e)}")
        return False

def manual_cleanup_instructions():
    """수동 정리 지침을 출력합니다."""
    print("\n===== 수동 정리 지침 =====")
    print("1. 모든 Python 프로세스 종료:")
    print("   - Windows: 작업 관리자에서 모든 python.exe 프로세스 종료")
    print("   - macOS/Linux: 터미널에서 'pkill -f python' 실행")
    print(f"2. ChromaDB 디렉토리 삭제:")
    print(f"   - 다음 디렉토리를 삭제하세요: {CHROMA_PERSIST_DIR}")
    print("3. 애플리케이션 재시작")
    print("===========================\n")

if __name__ == "__main__":
    success = clear_chromadb_cache()
    if not success:
        manual_cleanup_instructions()
    sys.exit(0 if success else 1)