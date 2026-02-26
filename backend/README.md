# multimodal-face-auth-security
CV-based face authentication with deepfake defense using image frequency-domain cues and biosignal-based liveness verification.

## OpenAI AI 코멘트 설정
`OPENAI_API_KEY`가 설정되면 AI 코멘트를 OpenAI로 생성합니다.

권장 환경 변수:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# 선택
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT_SEC=20
```

키가 없거나 호출 실패 시에는 기존 규칙 기반 코멘트로 자동 폴백됩니다.

## URL 추론(Instagram/YouTube) 환경변수
공개 URL은 무쿠키로도 시도하며, 로그인/레이트리밋이 걸리는 경우에만 쿠키/세션이 필요할 수 있습니다.

환경 변수:

```bash
# 선택: YouTube 접근 제한 시 사용
YTDLP_COOKIEFILE=/run/secrets/ig_cookies.txt

# 선택: Instaloader 접근 안정화용
INSTAGRAM_SESSION_ID=...

# 선택: YouTube extractor client 우선순위
YTDLP_YOUTUBE_CLIENTS=android,web,tv_embedded

# 선택: YouTube 보조 client 우선순위(기본값: ios,mweb,web,web_safari)
YTDLP_YOUTUBE_ALT_CLIENTS=ios,mweb,web,web_safari

# 선택: YouTube Shorts(pytubefix) client 우선순위
PYTUBEFIX_YOUTUBE_CLIENTS=ANDROID,WEB,IOS,MWEB
```

Docker Compose 예시:

```yaml
services:
  backend:
    environment:
      - YTDLP_COOKIEFILE=/run/secrets/ig_cookies.txt
      - INSTAGRAM_SESSION_ID=${INSTAGRAM_SESSION_ID}
      - YTDLP_YOUTUBE_CLIENTS=android,web,tv_embedded
      - YTDLP_YOUTUBE_ALT_CLIENTS=ios,mweb,web,web_safari
      - PYTUBEFIX_YOUTUBE_CLIENTS=${PYTUBEFIX_YOUTUBE_CLIENTS:-ANDROID,WEB,IOS,MWEB}
    volumes:
      - /opt/dbdbdeep-dev/secrets/ig_cookies.txt:/run/secrets/ig_cookies.txt:ro
```

주의:
- 쿠키 파일은 저장소에 커밋하지 마세요.
- YouTube Shorts는 pytubefix로 직접 다운로드하며, 해당 경로에서는 yt-dlp를 사용하지 않습니다.
- Shorts에서 스트림 추출 실패 시 pytubefix client 순서를 바꿔 재시도할 수 있습니다(`PYTUBEFIX_YOUTUBE_CLIENTS`).
- Instagram private/제한 게시물은 세션/쿠키가 없으면 실패할 수 있습니다.
- Instagram 공개 게시물은 Instaloader 실패 시 OpenGraph/yt-dlp 순서로 자동 폴백합니다.
