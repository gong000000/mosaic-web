"""렌더 큐 워커 프로세스.

Modal 웹 컨테이너에서 Streamlit과 별도 프로세스로 실행한다. 브라우저
세션이 없거나 Streamlit 스크립트가 다시 실행되지 않아도 큐를 계속
처리하고, 워커가 죽으면 이 프로세스가 다시 띄운다.

로컬에서는 app.py가 같은 프로세스 안에서 워커를 시작하므로 이 모듈을
직접 실행할 필요는 없다. 두 곳에서 동시에 실행해도 SQLite 리스로 한쪽만
실제 워커가 된다.
"""
import logging
import time

from . import jobqueue

log = logging.getLogger("mosaic.worker")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("렌더 큐 워커 프로세스 시작")
    while True:
        try:
            jobqueue.start_worker()
        except Exception:
            log.exception("렌더 큐 워커 시작 실패")
        time.sleep(5)


if __name__ == "__main__":
    main()
