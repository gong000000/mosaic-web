import logging
import os

# 앱 전체 로깅 설정(한 번만). LOG_LEVEL 환경변수로 조절한다.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(),
                      logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
