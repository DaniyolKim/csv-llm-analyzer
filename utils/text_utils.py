"""
텍스트 처리 관련 유틸리티 함수 모음
"""
import re
from typing import List
from konlpy.tag import Okt

# 한국어 불용어 리스트 (중앙 관리)
KOREAN_STOPWORDS = [
    '이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나', '사람', '주', '아니', '등', '같', '우리',
    '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하', '때문', '그것', '두', '말하', '알', '그러나',
    '받', '못하', '일', '그런', '또', '문제', '더', '많', '그리고', '좋', '크', '따르', '중', '나오', '가지', '씨', '시키',
    '만들', '지금', '생각하', '그러', '속', '하나', '집', '살', '모르', '적', '월', '데', '자신', '안', '어떤', '내', '경우',
    '명', '생각', '시간', '그녀', '다시', '이런', '앞', '보이', '번', '다른', '어떻', '여자', '남자', '개', '정도', '좀',
    '원', '잘', '통하', '소리', '놓', '부분', '그냥', '정말', '지금', '오늘', '어제', '내일', '여기', '저기', '거기',
    '매우', '아주', '너무', '정말', '진짜', '완전', '같은', '다른', '모든', '여러', '몇', '사실', '경우', '내용', '부분',
    '결과', '자료', '정보', '데이터', '분석', '처리', '기능', '구현', '요청', '확인', '문서', '텍스트', '단어', '추가',
    '사용', '선택', '입력', '출력', '표시', '생성', '제거', '포함', '위치', '사이', '기반', '형태', '위주', '다음', '파일',
    '페이지', '항목', '항상', '보통', '자주', '가끔', '거의', '매일', '매주', '매년', '통해', '위해', '대한', '관련', '따라',
    '이다', '돼다', '및', '에서', '에게', '으로', '으로써', '로써', '를', '저런', '그러한', '이러한', '저러한', '이렇게'
]

# 보존할 중요 1글자 명사 목록
IMPORTANT_SINGLE_CHAR_NOUNS = {
    '물', '불', '밥', '집', '꿈', '말', '옷', '땀', '차', '술', '꽃', '돈', '눈', '귀', '손', '발', '코',
    '입', '빛', '꿀', '약', '솜', '털', '피', '잠', '힘', '김', '국', '밤', '낮', '죽', '쑥', '점', '숲',
    '책', '병', '강', '산', '바다', '풀', '길', '몸', '삶', '팔', '짐', '봄', '여름', '가을', '겨울'
}

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

def chunk_text_semantic(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    텍스트를 의미론적으로 청킹합니다.
    문장 단위로 먼저 분할한 후, 청크 크기에 맞춰 조합합니다.
    
    Args:
        text (str): 청킹할 텍스트
        chunk_size (int): 각 청크의 최대 크기 (문자 수)
        chunk_overlap (int): 청크 간 겹치는 문자 수
        
    Returns:
        List[str]: 청킹된 텍스트 리스트
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    # 문장 분할 (한국어 문장 구분자 기준)
    sentences = re.split(r'[.!?]\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 현재 청크에 문장을 추가했을 때의 길이 계산
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            # 현재 청크가 비어있지 않으면 저장
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # 오버랩 처리: 현재 청크의 마지막 부분을 다음 청크 시작에 포함
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                # 문장이 청크 크기보다 큰 경우, 강제로 분할
                if len(sentence) > chunk_size:
                    # 문장을 청크 크기로 분할
                    for i in range(0, len(sentence), chunk_size - chunk_overlap):
                        chunk_part = sentence[i:i + chunk_size]
                        if chunk_part.strip():
                            chunks.append(chunk_part.strip())
                else:
                    current_chunk = sentence
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    텍스트에서 키워드를 추출합니다.
    한국어 형태소 분석을 사용하여 명사를 추출하고, 빈도를 기준으로 정렬합니다.
    
    Args:
        text (str): 키워드를 추출할 텍스트
        max_keywords (int): 추출할 최대 키워드 수
        
    Returns:
        List[str]: 추출된 키워드 리스트
    """
    if not text:
        return []
    
    try:
        # 텍스트 정제
        cleaned_text = clean_text(text)
        
        # Okt 형태소 분석기 초기화
        okt = Okt()
        
        # 명사 추출
        nouns = okt.nouns(cleaned_text)
        
        # 불용어 제거 및 필터링
        filtered_nouns = []
        for noun in nouns:
            # 불용어가 아니고, 2글자 이상이거나 중요 한 글자 명사 목록에 있는 단어만 포함
            if (noun not in KOREAN_STOPWORDS and 
                (len(noun) > 1 or noun in IMPORTANT_SINGLE_CHAR_NOUNS) and
                len(noun) <= 10):  # 너무 긴 단어 제외
                filtered_nouns.append(noun)
        
        # 빈도 계산
        word_freq = {}
        for noun in filtered_nouns:
            word_freq[noun] = word_freq.get(noun, 0) + 1
        
        # 빈도순으로 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_keywords[:max_keywords]]
        
        return keywords
        
    except Exception as e:
        print(f"키워드 추출 중 오류 발생: {str(e)}")
        # 오류 발생 시 간단한 공백 기준 분할로 fallback
        words = text.split()
        return [word for word in words[:max_keywords] if len(word) > 2]