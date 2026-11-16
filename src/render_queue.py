"""한 프로세스 안에서 GPU 렌더를 한 번에 하나만 실행하는 게이트.

Streamlit은 세션마다 스레드가 따로 돌기 때문에, 두 사용자가 동시에
'모자이크 시작'을 누르면 같은 GPU에서 두 렌더가 겹쳐 메모리가 터질 수
있다. 기존 Lock은 순서를 알 수 없어서, 대기 인원을 세어 화면에
'대기 순번'을 보여준다.
"""
import threading


class RenderGate:
    def __init__(self):
        self._lock = threading.Lock()
        self._guard = threading.Lock()
        self._waiting = 0

    def enter(self):
        """대기열에 들어간다. 렌더를 시작하기 전에 호출한다."""
        with self._guard:
            self._waiting += 1

    def waiting(self):
        with self._guard:
            return max(0, self._waiting - 1)

    def leave_waiting(self):
        with self._guard:
            self._waiting = max(0, self._waiting - 1)

    def acquire(self, on_wait=None, timeout=None):
        """Lock을 얻을 때까지 기다린다. on_wait(앞에_있는_사람_수)."""
        while True:
            if self._lock.acquire(timeout=0.5):
                self.leave_waiting()
                return True
            if on_wait is not None:
                try:
                    on_wait(self.waiting())
                except Exception:
                    pass
            if timeout is not None and timeout <= 0:
                self.leave_waiting()
                return False
            if timeout is not None:
                timeout -= 0.5

    def release(self):
        try:
            self._lock.release()
        except RuntimeError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
