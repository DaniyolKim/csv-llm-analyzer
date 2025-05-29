"""
텍스트 처리 관련 유틸리티 함수 모음
"""
import re
import kss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

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
    # extract_keywords 함수에 있던 불용어 추가 (중복 제거됨)
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
        last_sentence = ""  # 마지막으로 처리한 문장 저장
        
        for sentence in sentences:
            # 하나의 문장이 chunk_size보다 길면 그대로 청크로 추가
            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                chunks.append(sentence[:chunk_size])  # 길면 잘라서 추가
                current_chunk = ""
                last_sentence = ""
                continue
            
            # 현재 청크에 문장을 추가했을 때 chunk_size를 초과하는지 확인
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                chunks.append(current_chunk)
                
                # 중복(오버랩) 설정: 전체 문장 단위로 오버랩 처리
                if chunk_overlap > 0:
                    # 현재 청크의 마지막 문장들을 파악
                    chunk_sentences = kss.split_sentences(current_chunk)
                    
                    # 오버랩에 포함될 문장들 결정
                    overlap_size = 0
                    overlap_sentences = []
                    
                    # 뒤에서부터 문장을 추가하여 오버랩 크기 결정
                    for s in reversed(chunk_sentences):
                        if overlap_size + len(s) + 1 <= chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s) + 1
                        else:
                            # 오버랩 크기를 초과하면 중단
                            break
                    
                    # 완전한 문장들로 구성된 오버랩 청크 생성
                    if overlap_sentences:
                        current_chunk = " ".join(overlap_sentences) + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                # 공백 추가 (첫 문장이 아닌 경우)
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            
            last_sentence = sentence
        
        # 남은 청크가 있으면 추가
        if current_chunk:
            chunks.append(current_chunk)
        
        # 청크 후처리: 중복 내용 검사 및 누락된 텍스트 확인
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # 연속된 청크 간에 단어가 잘리지 않았는지 검사
            if i > 0:
                prev_chunk = chunks[i-1]
                # 현재 청크의 시작 부분과 이전 청크의 마지막 부분이 올바르게 연결되었는지 확인
                current_words = chunk.split()
                prev_words = prev_chunk.split()
                
                # 중복 시작 단어 확인 (이전 청크의 마지막 단어로 시작하는지)
                if len(current_words) > 0 and len(prev_words) > 0:
                    # 단어 중복/누락 방지
                    processed_chunk = chunk
                    processed_chunks.append(processed_chunk)
                else:
                    processed_chunks.append(chunk)
            else:
                processed_chunks.append(chunk)
        
        # 청크가 없으면 원본 텍스트를 그대로 사용
        if not processed_chunks:
            processed_chunks = [text]
        
        return processed_chunks
        
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

def extract_keywords_advanced(text: str, top_n: int = 10, method: str = "tfidf") -> List[str]:
    """
    향상된 키워드 추출 함수.
    다양한 방법(TF-IDF, TextRank, 하이브리드)을 지원합니다.
    중요 1글자 명사도 포함합니다.
    
    Args:
        text (str): 키워드를 추출할 텍스트
        top_n (int): 추출할 키워드 수, 기본값은 10
        method (str): 키워드 추출 방법 ("tfidf", "textrank", "hybrid")
        
    Returns:
        List[str]: 추출된 키워드 목록
    """
    try:
        from konlpy.tag import Okt
        okt = Okt()
        
        # 텍스트 전처리
        cleaned_text = clean_text(text)
        
        # 명사 추출 및 필터링 (중요 1글자 명사 포함)
        all_nouns = okt.nouns(cleaned_text)
        
        # 명사 필터링:
        # 1. 불용어 제거
        # 2. 2글자 이상 단어는 모두 포함
        # 3. 1글자 단어는 중요 1글자 명사 목록에 있는 것만 포함
        nouns = [noun for noun in all_nouns if 
                (noun not in KOREAN_STOPWORDS) and 
                (len(noun) >= 2 or noun in IMPORTANT_SINGLE_CHAR_NOUNS)]
        
        if not nouns:
            return []
            
        # 방법에 따라 키워드 추출
        if method == "tfidf":
            return _extract_keywords_tfidf(nouns, top_n)
        elif method == "textrank":
            return _extract_keywords_textrank(nouns, cleaned_text, top_n)
        elif method == "hybrid":
            return _extract_keywords_hybrid(nouns, cleaned_text, top_n)
        else:
            # 기본 방법 (빈도 기반)
            word_counts = Counter(nouns)
            return [word for word, _ in word_counts.most_common(top_n)]
            
    except Exception as e:
        print(f"고급 키워드 추출 중 오류 발생: {e}")
        # 오류 발생 시 기존 방식으로 폴백
        return extract_keywords(text, top_n)

def _extract_keywords_tfidf(nouns: List[str], top_n: int) -> List[str]:
    """TF-IDF를 이용한 키워드 추출"""
    # 단어 빈도수 계산
    word_counts = Counter(nouns)
    
    # 전체 단어 수
    total_words = len(nouns)
    
    # 고유 단어 목록
    unique_words = list(word_counts.keys())
    
    # 문서를 단일 문서로 간주하고 TF-IDF 계산
    tfidf_dict = {}
    
    # 단어별 TF 계산 (단어 빈도 / 전체 단어 수)
    for word in unique_words:
        tf = word_counts[word] / total_words
        # IDF는 단일 문서에서는 의미가 없으므로 TF만 사용
        tfidf_dict[word] = tf
    
    # TF-IDF 값으로 정렬하여 상위 키워드 추출
    sorted_keywords = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords[:top_n]]

def _extract_keywords_textrank(nouns: List[str], text: str, top_n: int) -> List[str]:
    """TextRank 알고리즘을 이용한 키워드 추출"""
    # 단어 중복 제거
    unique_nouns = list(set(nouns))
    
    # 단어 수가 너무 적으면 빈도 기반 추출로 대체
    if len(unique_nouns) < 4:
        word_counts = Counter(nouns)
        return [word for word, _ in word_counts.most_common(top_n)]
    
    # 동시 출현 행렬 생성 (window size = 2)
    window_size = 2
    co_occurrence_matrix = np.zeros((len(unique_nouns), len(unique_nouns)))
    
    # 동시 출현 계산
    word_to_idx = {word: i for i, word in enumerate(unique_nouns)}
    for i in range(len(nouns) - window_size + 1):
        window = nouns[i:i + window_size]
        for word1 in window:
            if word1 not in unique_nouns:
                continue
            for word2 in window:
                if word2 not in unique_nouns or word1 == word2:
                    continue
                idx1, idx2 = word_to_idx[word1], word_to_idx[word2]
                co_occurrence_matrix[idx1, idx2] += 1
    
    # 그래프 생성
    graph = nx.from_numpy_array(co_occurrence_matrix)
    
    # TextRank 알고리즘 적용
    scores = nx.pagerank(graph)
    
    # 점수순으로 정렬
    ranked_words = sorted(((scores[i], unique_nouns[i]) for i in range(len(unique_nouns))), 
                          reverse=True)
    
    # 상위 키워드 반환
    return [word for _, word in ranked_words[:top_n]]

def _extract_keywords_hybrid(nouns: List[str], text: str, top_n: int) -> List[str]:
    """TF-IDF와 TextRank를 결합한 하이브리드 방식"""
    # TF-IDF 키워드 추출
    tfidf_keywords = _extract_keywords_tfidf(nouns, top_n * 2)
    
    # 단어 수가 충분하면 TextRank도 적용
    try:
        textrank_keywords = _extract_keywords_textrank(nouns, text, top_n * 2)
        
        # 결과 결합 및 순위 계산
        combined_keywords = {}
        
        # TF-IDF 결과에 가중치 부여
        for i, keyword in enumerate(tfidf_keywords):
            score = (len(tfidf_keywords) - i) / len(tfidf_keywords)
            combined_keywords[keyword] = combined_keywords.get(keyword, 0) + score * 0.6
        
        # TextRank 결과에 가중치 부여
        for i, keyword in enumerate(textrank_keywords):
            score = (len(textrank_keywords) - i) / len(textrank_keywords)
            combined_keywords[keyword] = combined_keywords.get(keyword, 0) + score * 0.4
        
        # 결합 점수로 정렬
        sorted_combined = sorted(combined_keywords.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_combined[:top_n]]
    except:
        # TextRank 실패 시 TF-IDF 결과만 사용
        return tfidf_keywords[:top_n]

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    텍스트에서 주요 키워드를 추출합니다.
    OKT를 사용하여 명사만 추출하고, 가변적인 키워드 수를 지원합니다.
    
    Args:
        text (str): 키워드를 추출할 텍스트
        top_n (int): 추출할 키워드 수, 기본값은 10
        
    Returns:
        List[str]: 추출된 키워드 목록
    """
    # 상위 호환성을 위해 advanced 방식으로 대체
    return extract_keywords_advanced(text, top_n, method="hybrid")