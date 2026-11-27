"""One Euro Filter for face-box jitter reduction.

The tracker already has velocity and size damping, but a face box can still
wobble by one or two pixels because the detector alternates between slightly
different landmarks.  A One Euro filter keeps the box steady while the face is
almost still and opens up the cutoff when the face moves quickly, so it does not
add the lag a plain low-pass filter would add.

The filter is deliberately optional and frame-index based.  It is used only
after the offline box interpolation step, and the filtered box is expanded to
cover the raw box so a filtered edge can never uncover a face.
"""
from __future__ import annotations

import math


class OneEuroFilter:
    """Scalar One Euro filter (Casiez et al., 2012)."""

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        self.freq = max(1e-6, float(freq))
        self.min_cutoff = max(1e-6, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.d_cutoff = max(1e-6, float(d_cutoff))
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / max(1e-6, dt))

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, x, t=None):
        x = float(x)
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        if t is not None and self._t_prev is not None:
            dt = max(1e-3, float(t) - float(self._t_prev))
        else:
            dt = 1.0 / self.freq
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t if t is not None else self._t_prev
        return x_hat


def _cut_between(cuts, start, end):
    return any(start < c <= end for c in (cuts or ()))


def smooth_boxes_one_euro(frames, cuts=(), fps=30.0, min_cutoff=1.1,
                          beta=0.012, d_cutoff=1.0, coverage_pad=0.015):
    """Filter boxes per track and never shrink below the raw detection.

    ``frames`` is a list of ``{track_id: [x1, y1, x2, y2]}`` dictionaries.
    A scene cut resets the filter state for that track.  The output box is the
    union of the filtered box and the raw box, expanded by ``coverage_pad`` of
    the box diagonal, so any filter lag is on the safe (covered) side.
    """
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 30.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    cuts = set(cuts or ())
    out = [dict(f) for f in frames]
    filters = {}
    for i, boxes in enumerate(frames):
        for tid, box in boxes.items():
            try:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
            except (TypeError, ValueError, IndexError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            st = filters.get(tid)
            gap = st is not None and i - int(st.get("last", i - 1)) > 1
            if st is None or gap or _cut_between(cuts, i - 1, i):
                st = {
                    "cx": OneEuroFilter(fps, min_cutoff, beta, d_cutoff),
                    "cy": OneEuroFilter(fps, min_cutoff, beta, d_cutoff),
                    "w": OneEuroFilter(fps, min_cutoff, beta, d_cutoff),
                    "h": OneEuroFilter(fps, min_cutoff, beta, d_cutoff),
                    "box": None,
                    "last": i,
                }
                filters[tid] = st
            t = i / fps
            # Re-seed after a gap.  Carrying a stale filter across a gap would
            # move a new face towards the old position.
            if st.get("box") is None:
                for k in ("cx", "cy", "w", "h"):
                    st[k].reset()
            fcx = st["cx"].filter((x1 + x2) / 2.0, t)
            fcy = st["cy"].filter((y1 + y2) / 2.0, t)
            fw = st["w"].filter(max(1.0, x2 - x1), t)
            fh = st["h"].filter(max(1.0, y2 - y1), t)
            st["last"] = i
            pad = max(1.0, coverage_pad * math.hypot(x2 - x1, y2 - y1))
            fx1 = fcx - fw / 2.0 - pad
            fy1 = fcy - fh / 2.0 - pad
            fx2 = fcx + fw / 2.0 + pad
            fy2 = fcy + fh / 2.0 + pad
            # Union with the raw box: a filter that lags behind a fast edge
            # must never expose the face that was actually detected.
            out[i][tid] = [min(fx1, x1), min(fy1, y1),
                           max(fx2, x2), max(fy2, y2)]
            st["box"] = out[i][tid]
    return out
