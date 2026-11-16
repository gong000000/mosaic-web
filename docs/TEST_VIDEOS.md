# 테스트 영상 (로컬 전용)

아래 파일은 실제 얼굴이 포함된 검증용 영상이라 **git에 커밋하지 않는다.**
필요할 때 각자 `docs/`에 같은 이름으로 넣고 로컬에서만 사용한다.
외부에 다시 배포하지 않는다.

| 파일명 | 용도 |
|---|---|
| `face.mp4` | 근접 얼굴 5명, 얼굴형/깜빡임/sticky box 검증 |
| `street.mp4` | 야간 거리, 얼굴·번호판·주소·간판, 원거리 군중 검증 |
| `Walking in New York City.mp4` | 보행자 추적, 택시 번호판, 60fps 빠른 이동 검증 |
| `YTDown.com_YouTube_Media__VuJA-VQRcY_Central-Cee-Doja-Official-Music-Video_002_720p.mp4` | 긴 영상/장면 전환 검증 |

## 실행 예

```powershell
.\.venv\Scripts\python.exe scripts\apptest.py
.\.venv\Scripts\python.exe scripts\qualitytest.py
.\.venv\Scripts\python.exe scripts\robustness.py
.\.venv\Scripts\python.exe scripts\leak_metrics.py "docs\street.mp4"
.\.venv\Scripts\python.exe scripts\live_vlm.py "docs\street.mp4"
```

live(Modal)에서 전체 흐름을 볼 때는 영상을 Volume에 올린 뒤
`scripts/live_apptest.py`에 `/data/...` 경로를 넘긴다.

```powershell
modal volume put --force mosaic-data "docs\face.mp4" "/_smoke/face.mp4"
modal container exec <container-id> -- python /root/app/scripts/live_apptest.py "/data/_smoke/face.mp4"
```

검증이 끝나면 `modal volume rm -r mosaic-data "/_smoke"`로 올린 파일을 지운다.
