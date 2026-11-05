"""VLM 호출 예산 및 재시도 관리.

무료 등급(Gemini 하루 20회 등)의 호출 한도를 초과하지 않도록 관리하고,
호출 실패 시 지수 백오프로 재시도한다.
"""
import os
import threading
import time


class VLMBudget:
    """세션/일 단위 VLM 호출 예산 관리자."""

    def __init__(self, daily_limit: int | None = None,
                 per_video_limit: int | None = None):
        self.daily_limit = daily_limit
        self.per_video_limit = per_video_limit
        self.used_today = 0
        self.used_this_video = 0
        self._lock = threading.Lock()

    def can_call(self) -> bool:
        with self._lock:
            if self.daily_limit is not None \
                    and self.used_today >= self.daily_limit:
                return False
            if self.per_video_limit is not None \
                    and self.used_this_video >= self.per_video_limit:
                return False
            return True

    def record(self, n: int = 1):
        with self._lock:
            self.used_today += n
            self.used_this_video += n

    def reset_video(self):
        with self._lock:
            self.used_this_video = 0

    def remaining(self) -> int:
        """남은 호출 가능 횟수를 반환한다."""
        with self._lock:
            daily_rem = (self.daily_limit - self.used_today
                         if self.daily_limit is not None else 10**9)
            video_rem = (self.per_video_limit - self.used_this_video
                         if self.per_video_limit is not None else 10**9)
            return max(0, min(daily_rem, video_rem))


class VLMRetryPolicy:
    """지수 백오프 재시도 정책."""

    def __init__(self, max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 30.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """재시도할지 결정한다. 429(한도 초과)는 즉시 재시도하지 않는다."""
        msg = str(error).lower()
        if attempt >= self.max_retries:
            return False
        # 한도 초과(quota exceeded)는 재시도해도 소용없음.
        if any(k in msg for k in
               ("429", "quota", "rate limit", "resource_exhausted",
                "daily", "per day", "freetier")):
            return False
        # 인증 오류도 재시도 불가.
        if any(k in msg for k in ("401", "403", "invalid api key",
                                   "unauthorized", "permission")):
            return False
        # 5xx 서버 오류는 재시도 대상.
        return any(k in msg for k in ("500", "502", "503", "504",
                                       "unavailable", "internal"))

    def delay(self, attempt: int) -> float:
        """attempt(0-based) 번째 재시도 전 대기 시간(초)."""
        return min(self.base_delay * (2 ** attempt), self.max_delay)


def classify_vlm_error(error: Exception) -> str:
    """VLM 오류를 사용자 표시용 메시지 카테고리로 분류한다."""
    msg = str(error).lower()
    if any(k in msg for k in ("429", "quota", "rate limit",
                               "resource_exhausted", "daily", "per day",
                               "freetier")):
        return "quota_exceeded"
    if any(k in msg for k in ("401", "403", "invalid api key",
                               "unauthorized", "permission")):
        return "auth_error"
    if any(k in msg for k in ("404", "model not found", "not found")):
        return "model_not_found"
    if any(k in msg for k in ("500", "502", "503", "504",
                               "unavailable", "internal")):
        return "server_error"
    if "json" in msg and ("해석" in msg or "parse" in msg):
        return "parse_error"
    return "unknown_error"


def friendly_vlm_message(category: str, provider: str = "",
                          model: str = "") -> str:
    """사용자에게 보여줄 친절한 VLM 오류 메시지."""
    if category == "quota_exceeded":
        return (f"VLM({provider}) 무료 등급의 일일 호출 한도를 초과했습니다. "
                "내일 다시 시도하거나 유료 키를 사용하세요."
                + (f" (모델: {model})" if model else ""))
    if category == "auth_error":
        return ("API 키가 올바르지 않거나 권한이 없습니다. "
                "제공자와 키를 확인하세요.")
    if category == "model_not_found":
        return ("모델명을 찾을 수 없습니다. 사이드바의 모델명을 확인하세요.")
    if category == "server_error":
        return ("VLM 서버가 일시적으로 응답하지 않습니다. 잠시 후 다시 시도하세요.")
    if category == "parse_error":
        return ("VLM 응답을 해석하지 못했습니다. 다시 시도해 주세요.")
    return "VLM 호출 중 알 수 없는 오류가 발생했습니다."
