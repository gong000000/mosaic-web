import os
import sys
import types

# CI/개발 PC에 ultralytics/torch가 없어도 핵심 테스트를 실행할 수 있게
# import 전에 최소 스텁을 둔다. 실제 모델 추론은 이 테스트들에서 쓰지 않는다.
try:
    import ultralytics  # noqa: F401
except ImportError:
    stub = types.ModuleType("ultralytics")
    stub.YOLO = object
    sys.modules["ultralytics"] = stub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
