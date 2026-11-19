"""파이프라인/큐 공용 예외."""


class RenderCancelled(Exception):
    """사용자가 렌더 작업을 취소했다."""
