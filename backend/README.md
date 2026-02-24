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

## URL 추론(Instagram/YouTube) 쿠키 설정
Instagram Reels 등 일부 URL은 로그인/레이트리밋 제한으로 쿠키가 필요할 수 있습니다.

환경 변수:

```bash
YTDLP_COOKIEFILE=/run/secrets/ig_cookies.txt
```

Docker Compose 예시:

```yaml
services:
  backend:
    environment:
      - YTDLP_COOKIEFILE=/run/secrets/ig_cookies.txt
    volumes:
      - /opt/dbdbdeep-dev/secrets/ig_cookies.txt:/run/secrets/ig_cookies.txt:ro
```

주의:
- 쿠키 파일은 저장소에 커밋하지 마세요.
- 파일 경로가 잘못되면 `/api/analyze-url`에서 400 에러가 반환됩니다.
