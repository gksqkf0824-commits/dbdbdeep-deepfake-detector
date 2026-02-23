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
