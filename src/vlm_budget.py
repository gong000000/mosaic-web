"""VLM 호출 예산 및 재시도 관리.

Gemini 호출 한도를 초과하지 않도록 영상 단위 예산을 관리하고,
일시적 호출 실패 시 지수 백오프로 재시도한다. 무료 티어는 '분당 요청
수' 제한도 걸려 있으므로, 전역 스로틀(분당 N회)로 호출 자체를 천천히
분산한다.
"""
import threading
import time
from collections import deque

# 무료 티어 분당 한도는 모델에 따라 15회 내외다. 여유를 두어 12회로
# 제한하고, 429가 나면 아래 재시도 경로가 대신 기다린다.
DEFAULT_CALLS_PER_MINUTE = 12


class VLMThrottle:
    """전역 분당 호출 스로틀.

    여러 세션/스레드가 같은 워크스페이스 키를 공유하므로, 프로세스
    단위로 호출 속도를 제한해 429(분당 한도)를 예방한다.
    """

    def __init__(self, per_minute=DEFAULT_CALLS_PER_MINUTE):
        self.per_minute = max(1, int(per_minute))
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        """호출 가능해질 때까지 기다린다."""
        while True:
            with self._lock:
                now = time.time()
                while self._calls and (now - self._calls[0]) > 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.per_minute:
                    self._calls.append(now)
                    return
                wait = 60.0 - (now - self._calls[0])
            time.sleep(min(max(0.5, wait), 30.0))

    def used(self):
        with self._lock:
            return len(self._calls)


_THROTTLE = VLMThrottle()
_THROTTLE_LOCK = threading.Lock()


def throttle():
    """프로세스 전역 스로틀."""
    global _THROTTLE
    with _THROTTLE_LOCK:
        return _THROTTLE


def set_calls_per_minute(per_minute):
    """분당 호출 상한을 바꾼다. 0 이하면 제한하지 않는다."""
    global _THROTTLE
    with _THROTTLE_LOCK:
        _THROTTLE = VLMThrottle(per_minute)



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

    def reserve(self) -> bool:
        """호출 1회를 원자적으로 예약한다. 병렬 워커가 한도를 넘지 않게 한다."""
        with self._lock:
            if self.daily_limit is not None \
                    and self.used_today >= self.daily_limit:
                return False
            if self.per_video_limit is not None \
                    and self.used_this_video >= self.per_video_limit:
                return False
            self.used_today += 1
            self.used_this_video += 1
            return True

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
                 max_delay: float = 30.0,
                 rate_limit_delay: float = 25.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rate_limit_delay = rate_limit_delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """재시도할지 결정한다. 일일 한도 초과는 즉시 재시도하지 않는다."""
        msg = str(error).lower()
        if attempt >= self.max_retries:
            return False
        # 일일 한도(하루 총량)는 기다려도 소용없다.
        if _is_daily_quota(msg):
            return False
        # 인증 오류도 재시도 불가.
        if _is_auth_error(msg):
            return False
        # 분당 한도(429)는 잠시 기다리면 풀린다. 재시도한다.
        if _is_rate_limit(msg):
            return True
        # 5xx 서버 오류는 재시도 대상.
        return any(k in msg for k in ("500", "502", "503", "504",
                                       "unavailable", "internal"))

    def delay(self, attempt: int) -> float:
        """attempt(0-based) 번째 재시도 전 대기 시간(초)."""
        return min(self.base_delay * (2 ** attempt), self.max_delay)


def _is_daily_quota(msg):
    """하루 총량 한도 초과인지."""
    return any(k in msg for k in ("per day", "daily", "per 24 hours",
                                  "requests per day", "perday"))


def _is_auth_error(msg):
    """인증/권한 오류인지."""
    return any(k in msg for k in ("401", "403", "invalid api key",
                                  "unauthorized", "permission"))


def _is_rate_limit(msg):
    """분당 요청 수 한도(429)인지."""
    return any(k in msg for k in ("429", "rate limit", "resource_exhausted",
                                  "per minute", "requests per minute",
                                  "too many requests", "rpm"))


def classify_vlm_error(error: Exception) -> str:
    """VLM 오류를 사용자 표시용 메시지 카테고리로 분류한다.

    '하루 총량 한도(quota_exceeded)'와 '분당 한도(rate_limited)'를 구분한다.
    분당 한도는 기다리면 풀리므로 재시도하고, 일일 한도는 바로 알린다.
    """
    msg = str(error).lower()
    if _is_daily_quota(msg):
        return "quota_exceeded"
    if _is_auth_error(msg):
        return "auth_error"
    if any(k in msg for k in ("404", "model not found", "not found")):
        return "model_not_found"
    if _is_rate_limit(msg):
        return "rate_limited"
    if any(k in msg for k in ("500", "502", "503", "504",
                               "unavailable", "internal")):
        return "server_error"
    if "json" in msg and ("해석" in msg or "parse" in msg):
        return "parse_error"
    return "unknown_error"
