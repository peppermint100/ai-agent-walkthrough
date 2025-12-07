# 커밋 메시지 작성 규칙

## 패턴 1: 언어
- 한국어 사용

## 패턴 2: 제목 형식
- 명사형으로 끝남 (예: "~추가", "~구현", "~챗봇")
- 구현한 기능을 간결하게 요약
- 예시:
  - "비동기 + 재시도 로직이 포함된 Gemini 챗봇 추가"
  - "Gemini를 활용한 기본, 스트리밍 방식 챗봇"

# 파일명 작성 규칙

## 패턴
- 소문자 사용
- 언더스코어(_)로 단어 구분
- 주요 기능/기술/방식을 이름에 포함
- 명확하고 설명적인 이름 사용

## 예시
- `basic_gemini_chat.py` - 기본_기술_기능
- `streaming_gemini_chat.py` - 방식_기술_기능
- `async_and_retry_gemini_chat.py` - 기능1_and_기능2_기술_기능
- `prompt_template_str_output_parser.py` - 사용컴포넌트들
