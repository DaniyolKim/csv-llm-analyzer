"""
텍스트 처리 관련 유틸리티 함수 모음
"""
import re
import kss
from typing import List, Dict, Any, Tuple, Optional

def clean_text(text):
    """
    텍스트에서 불필요한 특수문자를 제거하되, 문장 구분 기호와 필수 문법 부호는 보존합니다.
    URL도 제거합니다. ', 조합과 '] 조합은 .로 변환합니다.
    
    Args:
        text (str): 정제할 텍스트
        
    Returns:
        str: 정제된 텍스트
    """
    if not isinstance(text, str):
        return str(text)
    
    # URL 패턴 제거 (http://, https://, www. 로 시작하는 전체 URL)
    cleaned_text = re.sub(r'(https?:\/\/|www\.)[^\s\"\'\(\)\[\]<>]+', ' ', text)
    
    # ', 조합과 '] 조합을 .로 변환
    cleaned_text = re.sub(r"',", ".", cleaned_text)
    cleaned_text = re.sub(r"']", ".", cleaned_text)
    
    # 연속된 하이픈(---)을 공백으로 변환
    cleaned_text = re.sub(r'-{2,}', ' ', cleaned_text)
    
    # 문장 구분 기호와 필수 문법 부호를 보존하면서 불필요한 특수문자 제거
    # 보존할 문자: 
    # - 영문, 숫자, 한글, 공백
    # - 문장 구분 기호: . ? ! ; : , 등
    # - 산술 기호: + - * / % = 등
    # - 기타 필수 문법 부호: & @ # 등
    cleaned_text = re.sub(r'[^\w\s\.\?!;:,\+\-\*/%=&@#]', ' ', cleaned_text)
    
    # 연속된 공백을 하나의 공백으로 치환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def chunk_text_semantic(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    텍스트를 의미 단위(문장)로 청크로 나눕니다.
    한국어 문장 분할 기능을 사용하여 문장 단위를 유지하면서 청킹합니다.
    
    Args:
        text (str): 청킹할 텍스트
        chunk_size (int): 최대 청크 크기
        chunk_overlap (int): 청크 간 중복 크기
        
    Returns:
        List[str]: 청킹된 텍스트 목록
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        # kss를 사용하여 한글 문장 단위로 분할
        sentences = kss.split_sentences(text)
        
        # 문장들을 최대 길이(chunk_size)를 고려하여 청크로 결합
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 하나의 문장이 chunk_size보다 길면 그대로 청크로 추가
            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                chunks.append(sentence[:chunk_size])  # 길면 잘라서 추가
                current_chunk = ""
                continue
            
            # 현재 청크에 문장을 추가했을 때 chunk_size를 초과하는지 확인
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                chunks.append(current_chunk)
                
                # 중복(오버랩) 설정: 이전 청크의 마지막 부분을 이어서 시작
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                # 공백 추가 (첫 문장이 아닌 경우)
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # 남은 청크가 있으면 추가
        if current_chunk:
            chunks.append(current_chunk)
        
        # 청크가 없으면 원본 텍스트를 그대로 사용
        if not chunks:
            chunks = [text]
        
        return chunks
        
    except Exception as e:
        print(f"텍스트 청킹 중 오류 발생: {e}")
        # 오류 발생 시 단순 분할 방식으로 폴백
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            end = min(i + chunk_size, len(text))
            chunks.append(text[i:end])
            if end >= len(text):
                break
        return chunks

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    텍스트에서 주요 키워드를 추출합니다.
    
    Args:
        text (str): 키워드를 추출할 텍스트
        top_n (int): 추출할 키워드 수
        
    Returns:
        List[str]: 추출된 키워드 목록
    """
    try:
        # 단어 분리 (한글, 영문, 숫자만 포함)
        words = re.findall(r'[\w가-힣]+', text.lower())
        
        # 불용어 정의 (한국어)
        stopwords = {
            '있다', '하다', '되다', '이다', '돼다', '않다', '그리고', '그러나', '또한', '그런데',
            '것', '등', '및', '이', '그', '저', '에서', '에게', '으로', '으로써', '로써', '를', '이런',
            '저런', '그런', '때문에', '이러한', '저러한', '그러한', '매우', '아주', '너무', '이렇게'
        }
        
        # 단어 필터링 (3글자 이상, 불용어 제외)
        filtered_words = [word for word in words if len(word) >= 2 and word not in stopwords]
        
        # 빈도 계산
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도 기준 정렬 및 상위 키워드 추출
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_keywords[:top_n]]
    
    except Exception as e:
        print(f"키워드 추출 중 오류 발생: {e}")
        return []