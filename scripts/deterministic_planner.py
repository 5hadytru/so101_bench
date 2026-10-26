#!/usr/bin/env python3
"""Deterministic planner + G0.5: the Sol + PyRUA-Lean + G0.5 harness (v9.6) with the LLM removed.

Run by scripts/pyrua_eval.py in place of the model (``--scripted_check``), in the PyRUA venv and
against the frozen v9.6 runtime, so it uses exactly Sol's ``robo`` API: the same SAM3 ``locate``
(two-camera 3D), IK motions, gripper, ``vla_run`` hand-offs, takeover clock, speed and descent
limits and ``end_episode``.  It sees what Sol sees: the instruction, the cameras and the robot
state (the task card lists no object names).  Nothing reads simulator state.

What replaces Sol's judgement (all fixed rules, from the same perception tools):

- instruction: the benchmark's templates, parsed by regex;
- bin verdicts at each G0.5 pause: ``jaw_reopened`` with the grasp point over the located bin =
  placed, elsewhere = failed attempt; ``arm_still`` / ``arm_lagging`` = stuck; jaw stopped on
  something (measured opening 4-75 mm) with the grasp point 4 cm up and not over the bin =
  lifted; otherwise progress;
- spatial verdicts: re-``locate`` the target at each pause; lifted = its top rose >= 2 cm (or its
  centre >= 1.5 cm) with the grasp point within 6 cm of it; ``arm_still`` = stuck; else progress;
- takeover grasp: top-down on ``locate``'s centre and ``grasp_yaw``, close, lift;
- placement geometry from the task rules (v9.6 targets): next to = the referent's side nearest
  the target's start, surface gap 1.5 cm; between = midpoint of the referents' centres, the
  target's long axis across the referent line; move = 6 cm along the direction (the blocker is
  not identifiable without names: see the paper note);
- placement: carry at 0.15 m, descend (the harness's 1 cm steps below 0.10 m) to the height the
  grasp point had when the target rested, release half open, lift, ``end_episode``.

The ledger is written to <out>/calls.jsonl (as pyrualean.play does), so the annotated videos work.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np

from pyrualean._backend import EpisodeFinished, Ledger
from pyrualean.hosts import so101_bench_host as host
from pyrualean.so101_bench import So101BenchRobot

CARRY_Z = 0.15
HOVER_ABOVE_TOP = 0.07  # >= the harness's 0.0508 m
NEXT_TO_GAP = 0.015  # v9.6: next-to surface gap 1-2 cm
MOVE_GAP = 0.015  # v9.6: move gap to the blocker 1-2 cm
MOVE_NO_BLOCKER = 0.08  # v9.2 guidance: about 8 cm when nothing is ahead (>= 5.08 cm needed)
MOVE_SCAN = 0.20  # how far ahead the corridor is scanned for a blocker
GRID = 0.005  # free-space grid on the table plane
#: PLANNER_GEOMETRY=depth: the VFM variant (metric depth height map from scripts/vfm_depth_server.py)
GEOMETRY = os.environ.get("PLANNER_GEOMETRY", "colour")
DEPTH_PORT = int(os.environ.get("PLANNER_DEPTH_PORT", "8010"))
HEIGHT_OCC = 0.008  # a cell this high above the table is occupied
HEIGHT_OBJ = 0.006  # flood-fill threshold for object outlines
OCC_STRICT_DIST = 90.0  # colour test used alongside depth (flat objects only)
#: PLANNER_SEMANTIC=owl (depth mode): cross-check SAM3 with OWLv2 open-vocabulary detection
SEMANTIC = os.environ.get("PLANNER_SEMANTIC", "sam3")
OWL_MIN_SCORE = 0.10
OWL_MAX_Z = 0.12  # a box whose lifted centre is higher is the arm, not a tabletop object
OWL_MAX_Y = 0.15  # the robot's base stands at the +y edge
OWL_BASE_CLEAR = 0.10  # a box this close to the robot's base (task card) is the robot
OWL_OVERRIDE = 0.30
SAM_TRUST = 0.40
SAM3_MIN_SCORE = 0.05  # all-instances mode: candidates below this are not considered
OCC_MIN_DIST = 40.0  # RGB distance from the table colour that counts as an object
DIRECTIONS = {"left": (1.0, 0.0), "right": (-1.0, 0.0), "forward": (0.0, -1.0), "forwards": (0.0, -1.0),
              "backward": (0.0, 1.0), "backwards": (0.0, 1.0)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse(task: str) -> dict:
    t = task.strip().rstrip(".")
    if re.fullmatch(r"Place each object in the plastic bin", t, re.I):
        return {"kind": "bin"}
    m = re.fullmatch(r"Place the (.+?) between the (.+?) and the (.+)", t, re.I)
    if m:
        return {"kind": "between", "target": m[1], "refs": [m[2], m[3]]}
    m = re.fullmatch(r"Place the (.+?) next to the (.+)", t, re.I)
    if m:
        return {"kind": "next_to", "target": m[1], "refs": [m[2]]}
    m = re.fullmatch(r"Move the (.+?) (left|right|forwards?|backwards?)", t, re.I)
    if m:
        return {"kind": "move", "target": m[1], "dir": DIRECTIONS[m[2].lower()]}
    raise ValueError(f"unparsed instruction: {task!r}")


class Planner:
    def __init__(self, robo: So101BenchRobot, debug_dir: Path | None = None, base_xy=None, sam3=None):
        self.r = robo
        self.sam3 = sam3  # the SAM 3 behind locate (same process): all instance masks
        self.debug_dir = debug_dir
        self.base_xy = base_xy  # the robot's base, from the task card (Sol is given it too)

    # -- plumbing ------------------------------------------------------------
    def cell(self):
        """What the end of one of Sol's cells does: the show requests are taken (attached)."""
        self.r._take_show_requests()

    def loc(self, name: str):
        """The best SAM3 measurement of ``name``: both cameras, two phrasings; prefer two-camera
        measurements, then the higher score."""
        best = None
        for prompt in (f"the {name}", f"the {name} on the table"):
            for cam in ("overhead", "wrist"):
                try:
                    s = self.r.locate(prompt, camera=cam)
                except Exception as exc:  # noqa: BLE001
                    log("locate error", name, cam, exc)
                    continue
                if s.found and s.center_xyz is not None:
                    key = (min(s.views, 2), s.score or 0)
                    if best is None or key > (min(best.views, 2), best.score or 0):
                        best = s
            if best is not None and best.views >= 2 and (best.score or 0) >= 0.5:
                break
        if best is None:  # SAM3 found nothing: the bare name with a lower score threshold
            for prompt in (name, f"a {name}"):
                for cam in ("overhead", "wrist"):
                    try:
                        s = self.r.locate(prompt, camera=cam, min_score=0.08)
                    except Exception:  # noqa: BLE001
                        continue
                    if s.found and s.center_xyz is not None and (best is None or (s.score or 0) > (best.score or 0)):
                        best = s
                if best is not None:
                    break
        return best

    def tcp(self):
        return self.r.state().tcp_pos

    def move(self, xyz, **kw):
        """move_to; when a pitch was asked for and IK finds no way, step the pitch down (far from the
        base the gripper cannot point straight down), as Sol does."""
        res = self.r.move_to(list(xyz), **kw)
        self.cell()
        if not res.ik_error:
            return res
        asked = kw.get("pitch")
        for pitch in (90, 75, 60, 45, 30, 15):  # any pitch the arm can hold there
            if asked is not None and pitch == asked:
                continue
            for path in ("auto", "joint"):
                res = self.r.move_to(list(xyz), **{**kw, "pitch": pitch, "path": path})
                self.cell()
                if not res.ik_error:
                    return res
        log("unreachable:", [round(v, 3) for v in xyz], res.ik_error)
        return res

    def find_bin(self):
        """The bin: the largest bin-sized mask over a few prompts (a single prompt can latch onto a
        small object)."""
        best = None
        for prompt in ("plastic bin", "large plastic bin", "storage bin", "plastic container"):
            try:
                s = self.r.locate(f"the {prompt}", camera="overhead")
            except Exception:  # noqa: BLE001
                continue
            if s.found and s.center_xyz is not None and (s.length_m or 0) >= 0.12:
                if best is None or (s.area_px or 0) > (best.area_px or 0):
                    best = s
        log("bin", None if best is None else (best.center_xyz, best.length_m, best.width_m))
        return best

    # -- shared geometry -----------------------------------------------------
    @staticmethod
    def rect(s):
        """Oriented footprint of an ObjectSight: centre, unit long axis, half length, half width."""
        a = math.radians(s.long_axis_deg or 0.0)
        return (s.center_xyz[0], s.center_xyz[1]), (math.cos(a), math.sin(a)), (s.length_m or 0.04) / 2, (s.width_m or 0.04) / 2

    @staticmethod
    def half_extent_along(s, n):
        c, u, hl, hw = Planner.rect(s)
        v = (-u[1], u[0])
        return abs(n[0] * u[0] + n[1] * u[1]) * hl + abs(n[0] * v[0] + n[1] * v[1]) * hw

    @staticmethod
    def inside(s, xy, margin=0.0):
        c, u, hl, hw = Planner.rect(s)
        d = (xy[0] - c[0], xy[1] - c[1])
        return abs(d[0] * u[0] + d[1] * u[1]) <= hl + margin and abs(-d[0] * u[1] + d[1] * u[0]) <= hw + margin

    # -- own grasp (takeovers) ----------------------------------------------
    def grasp(self, s) -> bool:
        x, y, _ = s.center_xyz
        top = s.top_z or 0.03
        yaw = s.grasp_yaw
        z_now = self.tcp()[2]
        self.r.set_gripper(self.r.OPEN)
        self.cell()
        self.move([x, y, max(z_now, top + 0.06)], pitch=90, yaw=yaw)
        grasp_z = max(0.012, top - min(0.02, 0.5 * (s.height_m or top)))
        for _ in range(12):
            if self.tcp()[2] <= grasp_z + 0.006:
                break
            before = self.tcp()[2]
            res = self.move([x, y, max(grasp_z, before - 0.03)], pitch=90, yaw=yaw)
            if res.ik_error or abs(self.tcp()[2] - before) < 0.002:
                log("grasp descent stalled:", res.ik_error or "no motion")
                break
        self.r.set_gripper(self.r.CLOSE)
        self.cell()
        self.move([x, y, top + 0.08], gripper=self.r.CLOSE, path="line")
        return True

    # -- bin -----------------------------------------------------------------
    def bin_task(self):
        bin_s = self.find_bin()
        h = self.r.vla_run()
        self.cell()
        for _ in range(300):
            if h.ended or self.r.state().truncated:
                return
            if h.control == "you":  # takeover: grasp what is under the wrist camera, drop it in the bin
                self.bin_takeover(bin_s)
                h = self.r.vla_run("placed")
                self.cell()
                continue
            st = self.r.state()
            over_bin = bin_s is not None and self.inside(bin_s, st.tcp_pos[:2], margin=0.01)
            holding = 0.004 < st.opening_m < 0.075 and st.tcp_pos[2] > 0.06
            why = h.paused_because
            if why == "jaw_reopened":
                verdict = "placed" if over_bin else "failed_attempt"
            elif why in ("arm_still", "arm_lagging"):
                verdict = "lifted" if (holding and not over_bin) else "stuck"
            else:
                verdict = "lifted" if (holding and not over_bin) else "progress"
            obj = "object" if verdict in ("failed_attempt", "stuck") else None
            try:
                h = self.r.vla_run(verdict, object=obj)
            except ValueError as exc:  # e.g. failed_attempt without a close in that stretch
                log("verdict refused:", exc)
                h = self.r.vla_run("stuck", object="object")
            self.cell()

    def bin_takeover(self, bin_s):
        s = None
        for prompt, cam in (("object", "wrist"), ("object on the table", "overhead")):
            try:
                c = self.r.locate(f"the {prompt}", camera=cam)
            except Exception:  # noqa: BLE001
                continue
            if c.found and c.center_xyz is not None and not (bin_s and self.inside(bin_s, c.center_xyz[:2])):
                s = c
                break
        if s is None:
            log("takeover: nothing to grasp found")
            return
        self.grasp(s)
        bx, by = bin_s.center_xyz[:2] if bin_s else (s.center_xyz[0], s.center_xyz[1])
        self.move([self.tcp()[0], self.tcp()[1], CARRY_Z], gripper=self.r.CLOSE)
        self.move([bx, by, CARRY_Z], gripper=self.r.CLOSE)
        self.r.set_gripper(self.r.OPEN)
        self.cell()

    # -- free space from the overhead image (RGB only, no names) -----------------
    def occupancy(self, free_rects, strict=False):
        """Occupancy of the table plane on a 5 mm grid: a grid point is occupied when the overhead
        pixel it projects to is not table-coloured.  ``free_rects`` (ObjectSight, margin) are
        cleared (the target's own start, the bin, referents where the caller handles them)."""
        img = np.asarray(self.r.image("overhead")).astype(float)
        meta = self.r.camera_meta("overhead")
        K = np.asarray(meta["intrinsic"], float)
        T = np.linalg.inv(np.asarray(meta["camera_to_environment"], float))
        xs, ys = np.arange(-0.32, 0.46, GRID), np.arange(-0.30, 0.22, GRID)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.ones(X.size)])
        C = T @ P
        z = C[2]
        u = (K[0, 0] * C[0] / np.maximum(z, 1e-6) + K[0, 2]).round().astype(int)
        v = (K[1, 1] * C[1] / np.maximum(z, 1e-6) + K[1, 2]).round().astype(int)
        h, w = img.shape[:2]
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        col = np.zeros((X.size, 3))
        col[ok] = img[v[ok], u[ok]]
        core = ok & (X.ravel() > -0.18) & (X.ravel() < 0.36) & (Y.ravel() > -0.16) & (Y.ravel() < 0.14)
        med = np.median(col[core], axis=0)
        dist = np.linalg.norm(col - med, axis=1)
        mad = np.median(np.abs(dist[core] - np.median(dist[core])))
        thr = OCC_STRICT_DIST if strict else max(OCC_MIN_DIST, np.median(dist[core]) + 6 * mad)
        occ = (~ok) | (dist > thr)
        occ = occ.reshape(X.shape)
        for s, margin in free_rects:
            if s is None:
                continue
            occ &= ~self._rect_mask(s, X, Y, margin)
        self._occ = (xs, ys, occ, X, Y)
        log("occupancy: %.1f%% of the grid occupied" % (100 * occ.mean()))
        if self.debug_dir is not None:  # for checking the map against the image
            np.savez_compressed(self.debug_dir / "occupancy.npz", xs=xs, ys=ys, occ=occ, dist=dist.reshape(X.shape),
                                ok=ok.reshape(X.shape), med=med, K=K, T=np.asarray(meta["camera_to_environment"], float))
            import cv2
            cv2.imwrite(str(self.debug_dir / "overhead_start.png"), img[..., ::-1].astype(np.uint8))
        return self._occ

    def _rect_mask(self, s, X, Y, margin, center=None, axis_deg=None):
        c, uax, hl, hw = self.rect(s)
        if center is not None:
            c = center
        if axis_deg is not None:
            a = math.radians(axis_deg)
            uax = (math.cos(a), math.sin(a))
        dx, dy = X - c[0], Y - c[1]
        along = dx * uax[0] + dy * uax[1]
        across = -dx * uax[1] + dy * uax[0]
        return (np.abs(along) <= hl + margin) & (np.abs(across) <= hw + margin)

    def blocked_fraction(self, s, center, axis_deg, margin):
        xs, ys, occ, X, Y = self._occ
        m = self._rect_mask(s, X, Y, margin, center=center, axis_deg=axis_deg)
        return float(occ[m].mean()) if m.any() else 1.0

    def occupied_at(self, x, y):
        xs, ys, occ, X, Y = self._occ
        i, j = int(round((x - xs[0]) / GRID)), int(round((y - ys[0]) / GRID))
        if i < 0 or j < 0 or i >= len(xs) or j >= len(ys):
            return True
        return bool(occ[i, j])

    # -- VFM geometry (PLANNER_GEOMETRY=depth): metric depth -> height map ------------
    def depth_points(self):
        """World points for every overhead pixel: Depth Anything V2 (metric) calibrated to the
        known table plane (scale and offset fitted where each ray meets z = 0, outliers dropped)."""
        import base64, io, urllib.request
        img = np.asarray(self.r.image("overhead"))
        meta = self.r.camera_meta("overhead")
        K = np.asarray(meta["intrinsic"], float)
        T = np.asarray(meta["camera_to_environment"], float)
        ok, buf = __import__("cv2").imencode(".png", img[..., ::-1])
        fov_x = math.degrees(2 * math.atan(img.shape[1] / (2 * K[0, 0])))
        body = json.dumps({"image_png_b64": base64.b64encode(buf.tobytes()).decode(), "fov_x_deg": fov_x}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{DEPTH_PORT}/depth", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            reply = json.loads(resp.read())
            pred = np.load(io.BytesIO(base64.b64decode(reply["depth_npy_b64"])))
            log("depth served:", reply.get("mode"))
        h, w = pred.shape
        vv, uu = np.mgrid[0:h, 0:w]
        rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)  # z = 1
        R, o = T[:3, :3], T[:3, 3]
        wr = rays @ R.T  # world direction per unit camera-z
        t_plane = -o[2] / np.where(np.abs(wr[..., 2]) > 1e-6, wr[..., 2], np.nan)  # camera-z where the ray meets z=0
        hit = o + wr * t_plane[..., None]
        core = (t_plane > 0) & (hit[..., 0] > -0.18) & (hit[..., 0] < 0.36) & (hit[..., 1] > -0.16) & (hit[..., 1] < 0.14)
        core &= np.isfinite(pred)
        x, y = pred[core].ravel(), t_plane[core].ravel()
        keep = np.ones_like(x, bool)
        for _ in range(4):  # robust affine fit: table pixels dominate the core region
            A = np.stack([x[keep], np.ones(keep.sum())], 1)
            (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            res = np.abs(a * x + b - y)
            keep = res < max(0.01, np.percentile(res, 60))
        d = a * pred + b
        pts = o + wr * d[..., None]
        log("depth calibration: a %.3f b %.3f, table residual %.1f mm" % (a, b, 1000 * np.median(res[keep])))
        self._pts, self._img = pts, img
        return pts, img

    def height_map(self, free_rects):
        pts, img = (self._pts, self._img) if getattr(self, "_pts", None) is not None else self.depth_points()
        xs, ys = np.arange(-0.32, 0.46, GRID), np.arange(-0.30, 0.22, GRID)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        H = np.full(X.shape, np.nan)
        P = pts.reshape(-1, 3)
        i = np.round((P[:, 0] - xs[0]) / GRID).astype(int)
        j = np.round((P[:, 1] - ys[0]) / GRID).astype(int)
        inb = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys)) & np.isfinite(P[:, 2])
        Hf = np.full(X.size, -np.inf)
        np.maximum.at(Hf, i[inb] * len(ys) + j[inb], P[inb, 2])
        H = Hf.reshape(X.shape)
        seen = np.isfinite(H)
        # colour test, strict, for flat things depth cannot lift off the table (tape, cards)
        colour = self.occupancy(free_rects, strict=True)[2]
        occ = (~seen) | (np.where(seen, H, 0) > HEIGHT_OCC) | colour
        for s, margin in free_rects:
            if s is not None:
                occ &= ~self._rect_mask(s, X, Y, margin)
        self._occ = (xs, ys, occ, X, Y)
        self._H = np.where(seen, H, np.nan)
        log("height map: %.1f%% occupied" % (100 * occ.mean()))
        if self.debug_dir is not None:
            np.savez_compressed(self.debug_dir / "height_map.npz", xs=xs, ys=ys, H=self._H, occ=occ)
        return self._occ

    def owl_locate(self, name):
        """OWLv2 open-vocabulary detection of ``name`` in the overhead image, lifted to 3D through
        the calibrated depth (median of the raised points in the box's inner part)."""
        import base64, urllib.request
        from types import SimpleNamespace
        img = self._img
        ok, buf = __import__("cv2").imencode(".png", img[..., ::-1])
        body = json.dumps({"image_png_b64": base64.b64encode(buf.tobytes()).decode(), "queries": [name]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{DEPTH_PORT}/detect", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                dets = json.loads(resp.read())["detections"][0]
        except Exception as exc:  # noqa: BLE001
            log("owl error", exc)
            return None
        h, w = img.shape[:2]
        for d in dets:
            if d["score"] < OWL_MIN_SCORE:
                break
            x0, y0, x1, y1 = d["box"]
            if x1 - x0 > 0.6 * w or y1 - y0 > 0.6 * h:  # the table / the bin, not an object
                continue
            cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            r0, r1 = int(max(0, cy - 0.3 * bh)), int(min(h, cy + 0.3 * bh + 1))
            c0, c1 = int(max(0, cx - 0.3 * bw)), int(min(w, cx + 0.3 * bw + 1))
            P = self._pts[r0:r1, c0:c1].reshape(-1, 3)
            P = P[np.isfinite(P).all(1)]
            if len(P) == 0:
                continue
            raised = P[P[:, 2] > HEIGHT_OBJ]
            Q = raised if len(raised) >= 5 else P
            c = np.median(Q, axis=0)
            top = float(np.percentile(Q[:, 2], 95)) if len(raised) >= 5 else 0.02
            # not a tabletop object: the arm (far above the table), the robot's base, off the table
            near_base = self.base_xy is not None and math.dist(c[:2], self.base_xy) < OWL_BASE_CLEAR
            if near_base or c[2] > OWL_MAX_Z or c[1] > OWL_MAX_Y or not (-0.32 < c[0] < 0.46 and -0.30 < c[1] < 0.22) \
                    or (self._bin_s is not None and self.inside(self._bin_s, c[:2], margin=0.0)):
                log("owl skip", name, "score %.2f" % d["score"], [round(float(v), 3) for v in c])
                continue
            sight = SimpleNamespace(found=True, score=d["score"], views=1, centroid=(int(cy), int(cx)), area_px=int(bw * bh),
                                    camera="overhead", note="owlv2", center_xyz=(float(c[0]), float(c[1]), top / 2),
                                    length_m=0.06, width_m=0.04, long_axis_deg=0.0, grasp_yaw=90.0, top_z=top, height_m=top)
            log("owl", name, "score %.2f" % d["score"], [round(float(v), 3) for v in c[:2]])
            return sight
        log("owl", name, "nothing above", OWL_MIN_SCORE)
        return None

    def sam3_instances(self, prompt: str):
        """Every instance SAM 3 finds for ``prompt`` in the overhead image: (mask, score), best first.
        Uses the SAM 3 already loaded in this process (the one behind locate)."""
        cl = self.sam3
        if cl is None:
            return []
        import warnings
        pixels = np.ascontiguousarray(np.asarray(self._img)[..., :3].astype(np.uint8))
        with cl._lock, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = cl._state_for(pixels)
            with cl._autocast():
                out = cl._processor.set_text_prompt(prompt=prompt.strip(), state=state)
        masks, scores = out.get("masks"), out.get("scores")
        if masks is None or scores is None:
            return []
        to_np = lambda v: v if isinstance(v, np.ndarray) else v.detach().float().cpu().numpy()  # noqa: E731
        masks, scores = to_np(masks), to_np(scores).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        order = np.argsort(-scores)
        return [(masks[i] > 0, float(scores[i])) for i in order if (masks[i] > 0).any()]

    def sam3_locate(self, name: str):
        """The object called ``name``: SAM 3's instance masks lifted to 3D with the calibrated depth;
        instances that are the robot, off the table or inside the bin are dropped; the most confident
        valid one wins.  Its outline and top come from the mask's 3D points (no box guesswork)."""
        from types import SimpleNamespace
        best = None
        for prompt in (name, f"{name} object"):
            for mask, score in self.sam3_instances(prompt)[:8]:
                if score < SAM3_MIN_SCORE:
                    break
                P = self._pts[mask]
                P = P[np.isfinite(P).all(1)]
                raised = P[P[:, 2] > HEIGHT_OBJ]
                Q = raised if len(raised) >= 8 else P
                if len(Q) < 8:
                    continue
                c = np.median(Q, axis=0)
                near_base = self.base_xy is not None and math.dist(c[:2], self.base_xy) < OWL_BASE_CLEAR
                off = not (-0.32 < c[0] < 0.46 and -0.30 < c[1] < 0.22)
                in_bin = self._bin_s is not None and self.inside(self._bin_s, c[:2], margin=0.0)
                if near_base or off or in_bin or c[2] > OWL_MAX_Z:
                    log("sam3 skip", name, "score %.2f" % score, [round(float(v), 3) for v in c])
                    continue
                xy = Q[:, :2]
                m = xy.mean(0)
                ev, evec = np.linalg.eigh(np.cov((xy - m).T)) if len(xy) > 2 else (None, np.eye(2))
                ax = evec[:, -1]
                perp = np.array([-ax[1], ax[0]])
                al, ac = (xy - m) @ ax, (xy - m) @ perp
                centre = m + ax * (al.max() + al.min()) / 2 + perp * (ac.max() + ac.min()) / 2
                top = float(np.percentile(Q[:, 2], 95))
                heading = math.degrees(math.atan2(ax[1], ax[0]))
                sight = SimpleNamespace(found=True, score=score, views=2, centroid=None, area_px=int(mask.sum()),
                                        camera="overhead", note="sam3 instance + depth", from_mask=True,
                                        center_xyz=(float(centre[0]), float(centre[1]), top / 2),
                                        length_m=float(al.max() - al.min()), width_m=float(ac.max() - ac.min()),
                                        long_axis_deg=heading, grasp_yaw=((heading + 180.0) % 180.0) - 90.0,
                                        top_z=top, height_m=top)
                if best is None or score > best.score:
                    best = sight
            if best is not None:
                break
        if best is not None:
            log("sam3all", name, "score %.2f" % best.score, [round(v, 3) for v in best.center_xyz[:2]],
                "L %.3f W %.3f top %.3f" % (best.length_m, best.width_m, best.top_z))
        return best

    def semantic_locate(self, name):
        """SAM3 (locate) cross-checked with OWLv2: OWLv2 is used when SAM3 misses, is unsure, or
        disagrees while OWLv2 is the more confident."""
        if SEMANTIC == "sam3all" and getattr(self, "_pts", None) is not None:
            got = self.sam3_locate(name)
            return got if got is not None else self.loc(name)
        sam = self.loc(name)
        if SEMANTIC != "owl" or getattr(self, "_pts", None) is None:
            return sam
        owl = self.owl_locate(name)
        if owl is None:
            return sam
        if sam is None or (sam.score or 0) < SAM_TRUST:
            log("semantic: OWLv2 for", name, "(SAM3 %s)" % (None if sam is None else round(sam.score or 0, 2)))
            return owl
        far = math.dist(sam.center_xyz[:2], owl.center_xyz[:2]) > 0.04
        if far and owl.score >= OWL_OVERRIDE and (sam.score or 0) < 0.6:
            log("semantic: SAM3 and OWLv2 disagree for", name, "- OWLv2 more confident")
            return owl
        return sam

    def refine(self, s):
        """Outline of a located object from the height map: flood fill of the raised cells around
        its centre (principal axes = heading, extents = length/width, highest cell = top)."""
        if s is None or getattr(self, "_H", None) is None or getattr(s, "from_mask", False):
            return s
        xs, ys, _, X, Y = self._occ
        H = np.nan_to_num(self._H, nan=0.0)
        ci, cj = int(round((s.center_xyz[0] - xs[0]) / GRID)), int(round((s.center_xyz[1] - ys[0]) / GRID))
        radius = int(((max(s.length_m or 0.05, 0.05)) * 0.75 + 0.03) / GRID)
        if not (0 <= ci < len(xs) and 0 <= cj < len(ys)):
            return s
        # seed: the highest cell near the centre
        win = H[max(0, ci - 3):ci + 4, max(0, cj - 3):cj + 4]
        if win.size == 0 or win.max() < HEIGHT_OBJ:
            return s
        di, dj = np.unravel_index(np.argmax(win), win.shape)
        seed = (max(0, ci - 3) + di, max(0, cj - 3) + dj)
        stack, seen, cells = [seed], {seed}, []
        while stack and len(cells) < 4000:
            a, b = stack.pop()
            if H[a, b] < HEIGHT_OBJ or abs(a - ci) > radius or abs(b - cj) > radius:
                continue
            cells.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (a + da, b + db)
                if 0 <= n[0] < len(xs) and 0 <= n[1] < len(ys) and n not in seen:
                    seen.add(n)
                    stack.append(n)
        if len(cells) < 6:
            return s
        pts = np.array([(xs[a], ys[b]) for a, b in cells])
        c = pts.mean(0)
        ev, evec = np.linalg.eigh(np.cov((pts - c).T))
        ax = evec[:, 1]
        along, across = (pts - c) @ ax, (pts - c) @ np.array([-ax[1], ax[0]])
        mid = np.array([(along.max() + along.min()) / 2, (across.max() + across.min()) / 2])
        centre = c + mid[0] * ax + mid[1] * np.array([-ax[1], ax[0]])
        top = float(max(H[a, b] for a, b in cells))
        heading = math.degrees(math.atan2(ax[1], ax[0]))
        from types import SimpleNamespace
        r = SimpleNamespace(**{k: getattr(s, k) for k in ("found", "score", "views", "centroid", "area_px", "camera", "note")})
        r.center_xyz = (float(centre[0]), float(centre[1]), top / 2)
        r.length_m = float(along.max() - along.min() + GRID)
        r.width_m = float(across.max() - across.min() + GRID)
        r.long_axis_deg = heading
        r.grasp_yaw = ((heading + 90.0 + 90.0) % 180.0) - 90.0
        r.top_z = top
        r.height_m = top
        log("refined", [round(v, 3) for v in r.center_xyz[:2]], "L %.3f W %.3f top %.3f (locate L %s W %s top %s)" % (
            r.length_m, r.width_m, top, s.length_m, s.width_m, s.top_z))
        return r

    # -- spatial ---------------------------------------------------------------
    def spatial_task(self, plan):
        name = plan["target"]
        self._bin_s = None
        if GEOMETRY == "depth":
            self.depth_points()  # the calibrated point cloud first: OWLv2 detections are lifted with it
            self._bin_s = self.find_bin()
            tgt0 = self.semantic_locate(name)
        else:
            tgt0 = self.loc(name)
        if tgt0 is None:
            log("target not found; ending")
            self.r.end_episode()
            return
        refs0 = [self.semantic_locate(n) if GEOMETRY == "depth" else self.loc(n) for n in plan.get("refs", [])]
        bin_s = self.find_bin()
        # the scene as it stands, before the arm goes over it: free space for the placement
        free = [(tgt0, 0.015), (bin_s, 0.01)] + [(r, 0.008) for r in refs0]
        if GEOMETRY == "depth":
            self.height_map(free)
            tgt0 = self.refine(tgt0)
            refs0 = [self.refine(r) for r in refs0]
            self.height_map([(tgt0, 0.015), (bin_s, 0.01)] + [(r, 0.008) for r in refs0])
        else:
            self.occupancy(free)
        goal_c, axis_deg = self.goal(plan, tgt0, refs0, bin_s)
        log("goal", [round(v, 3) for v in goal_c], axis_deg)
        x, y, _ = tgt0.center_xyz
        top = tgt0.top_z or 0.03
        self.move([self.tcp()[0], self.tcp()[1], max(self.tcp()[2], CARRY_Z)])
        self.move([x, y, max(CARRY_Z, top + HOVER_ABOVE_TOP)], pitch=90)
        self.move([x, y, top + HOVER_ABOVE_TOP], pitch=90)
        h = self.r.vla_run(target_top_z=top)
        self.cell()
        rest_tcp_z, grip_offset, grip_yaw = None, (0.0, 0.0), None
        while h.control == "policy" and not h.ended and not self.r.state().truncated:
            s = self.loc(name)
            tcp = self.tcp()
            verdict = "progress"
            if s is not None:
                rise_top = (s.top_z or 0) - top
                rise_c = s.center_xyz[2] - tgt0.center_xyz[2]
                near = math.dist(tcp[:2], s.center_xyz[:2]) < 0.06
                if near and (rise_top >= 0.02 or rise_c >= 0.015):
                    verdict = "lifted"
                    rest_tcp_z = tcp[2] - max(rise_c, 0.0)
                    grip_offset = (s.center_xyz[0] - tcp[0], s.center_xyz[1] - tcp[1])
                    grip_yaw = self.r.state().yaw
            if verdict == "progress" and h.paused_because in ("rose_2cm", "moved_3cm"):
                # the held object is often hidden by the jaw (SAM3 loses it): the gripper rose or moved,
                # the jaw stopped on something, and the target is no longer where it stood = lifted
                st = self.r.state()
                gone = s is None or (math.dist(s.center_xyz[:2], tgt0.center_xyz[:2]) > 0.03
                                     and math.dist(s.center_xyz[:2], tcp[:2]) < 0.06)
                if 0.004 < st.opening_m < 0.075 and tcp[2] > top + 0.01 and gone:
                    verdict = "lifted"
                    rest_tcp_z = max(0.012, tcp[2] - 0.02)
                    # the grasp point was over the target's start when G0.5 closed: offset from there
                    grip_offset = (tgt0.center_xyz[0] - tcp[0], tgt0.center_xyz[1] - tcp[1]) \
                        if math.dist(tcp[:2], tgt0.center_xyz[:2]) < 0.05 else (0.0, 0.0)
                    grip_yaw = st.yaw
            if verdict == "progress" and h.paused_because in ("arm_still", "arm_lagging"):
                verdict = "stuck"
            h = self.r.vla_run(verdict, object=name if verdict == "stuck" else None)
            self.cell()
        if h.ended or self.r.state().truncated:
            return
        if h.paused_because == "takeover":  # G0.5 never got it up: grasp it ourselves
            s = self.loc(name) or tgt0
            self.grasp(s)
            grip_offset, grip_yaw = (0.0, 0.0), self.r.state().yaw
            rest_tcp_z = max(0.012, (s.top_z or top) - min(0.02, 0.5 * (s.height_m or top)))
        if rest_tcp_z is None:
            rest_tcp_z = max(0.012, top - 0.015)
        self.place(goal_c, axis_deg, rest_tcp_z, grip_offset, grip_yaw, refs0)
        self.verify_and_correct(plan, name, tgt0, refs0, goal_c, axis_deg)
        self.r.end_episode()

    def place(self, goal_c, axis_deg, rest_tcp_z, grip_offset, grip_yaw, refs0):
        """Carry the held target so its centre ends at ``goal_c`` (long axis along ``axis_deg`` if
        given), lower it in the harness's steps, release half open and lift clear."""
        yaw = None
        ox, oy = grip_offset
        if axis_deg is not None and grip_yaw is not None:
            # the jaw closes across the target's narrow side: the long axis lies across the jaw line
            yaw = (((axis_deg + 90.0) + 90.0) % 180.0) - 90.0
            turn = math.radians(yaw - grip_yaw)
            ox, oy = ox * math.cos(turn) - oy * math.sin(turn), ox * math.sin(turn) + oy * math.cos(turn)
        goal = (goal_c[0] - ox, goal_c[1] - oy)
        z = self.tcp()[2]
        self.move([self.tcp()[0], self.tcp()[1], max(z, CARRY_Z)], gripper=self.r.CLOSE)
        self.move([goal[0], goal[1], CARRY_Z], gripper=self.r.CLOSE, yaw=yaw)
        release_z = rest_tcp_z + 0.008
        retries = 0
        for _ in range(60):  # the harness allows one <= 1 cm descent per cell below 0.10 m
            now = self.tcp()[2]
            if now <= release_z + 0.003:
                break
            nxt = max(release_z, 0.10, now - 0.03) if now > 0.101 else max(release_z, now - 0.01)
            try:
                res = self.move([goal[0], goal[1], nxt], gripper=self.r.CLOSE)
            except ValueError as exc:  # a descent the harness refuses: take a smaller one next time
                log("descent refused:", exc)
                self.cell()
                continue
            if res.blocked and now > release_z + 0.02 and retries < 2:
                retries += 1
                refs = [r for r in refs0 if r is not None]
                if refs:
                    rc = min(refs, key=lambda r: math.dist(r.center_xyz[:2], goal))
                    dx, dy = goal[0] - rc.center_xyz[0], goal[1] - rc.center_xyz[1]
                    n = math.hypot(dx, dy) or 1.0
                    goal = (goal[0] + 0.015 * dx / n, goal[1] + 0.015 * dy / n)
                self.move([goal[0], goal[1], min(CARRY_Z, self.tcp()[2] + 0.03)], gripper=self.r.CLOSE)
                continue
            if res.ik_error or abs(self.tcp()[2] - now) < 0.002:
                break
        self.r.set_gripper(self.r.OPEN, opening=0.5)
        self.cell()
        p = self.tcp()
        self.move([p[0], p[1], min(CARRY_Z, p[2] + 0.06)])

    def goal(self, plan, tgt0, refs0, bin_s):
        """Where the target's centre should end (and its long-axis direction, if it must turn), from
        the task rule, the measured outlines and the free space; v9.6 gap targets."""
        tc = tgt0.center_xyz[:2]
        t_axis = tgt0.long_axis_deg or 0.0
        if plan["kind"] == "move":
            d = plan["dir"]
            lat = (-d[1], d[0])
            lead = self.half_extent_along(tgt0, d)
            side = self.half_extent_along(tgt0, lat)
            s0 = tc[0] * d[0] + tc[1] * d[1] + lead
            # scan the target's corridor ahead for the first object: the blocker (bin excluded)
            hit = None
            for k in range(1, int(MOVE_SCAN / GRID) + 1):
                s = k * GRID
                n_occ = 0
                for f in np.linspace(-0.9, 0.9, 7):
                    px = tc[0] + d[0] * (lead + s) + lat[0] * f * side
                    py = tc[1] + d[1] * (lead + s) + lat[1] * f * side
                    if bin_s is not None and self.inside(bin_s, (px, py), margin=0.01):
                        continue
                    n_occ += self.occupied_at(px, py)
                if n_occ >= 2:
                    hit = s
                    break
            travel = MOVE_NO_BLOCKER if hit is None else max(0.026, hit - MOVE_GAP)
            if bin_s is not None:  # never into the bin
                for k in range(1, int(travel / GRID) + 1):
                    p = (tc[0] + d[0] * (lead + k * GRID), tc[1] + d[1] * (lead + k * GRID))
                    if self.inside(bin_s, p, margin=0.015):
                        travel = max(0.026, (k - 1) * GRID)
                        break
            log("move: blocker at", hit, "travel", round(travel, 3), "s0", round(s0, 3))
            return (tc[0] + travel * d[0], tc[1] + travel * d[1]), None
        refs = [r for r in refs0 if r is not None]
        if plan["kind"] == "between" and len(refs) == 2:
            a, b = refs[0].center_xyz[:2], refs[1].center_xyz[:2]
            L = math.dist(a, b) or 1e-6
            u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
            w = (-u[1], u[0])
            ha, hb = self.half_extent_along(refs[0], u), self.half_extent_along(refs[1], u)
            gap = max(0.0, L - ha - hb)
            centre_s = ha + gap / 2
            line_deg = math.degrees(math.atan2(u[1], u[0]))
            # a long target goes across the line when it would not fit along it
            fits_along = (tgt0.length_m or 0.04) + 0.02 <= gap
            axis = line_deg if fits_along else line_deg + 90.0
            best = None
            for dw in (0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03):
                for ds in (0.0, 0.01, -0.01, 0.02, -0.02):
                    s = centre_s + ds
                    p = (a[0] + u[0] * s + w[0] * dw, a[1] + u[1] * s + w[1] * dw)
                    frac = self.blocked_fraction(tgt0, p, axis, 0.008)
                    key = (frac > 0.02, abs(dw) + 0.5 * abs(ds), frac)
                    if best is None or key < best[0]:
                        best = (key, p)
            return best[1], axis
        if refs:  # next to: a free side of the referent, nearest the target's start, 1.5 cm edge gap
            ref = refs[0]
            c, uax, hl, hw = self.rect(ref)
            best = None
            base = math.atan2(uax[1], uax[0])
            for k in range(8):
                a = base + k * math.pi / 4
                n = (math.cos(a), math.sin(a))
                off = self.half_extent_along(ref, n) + self.half_extent_along(tgt0, n) + NEXT_TO_GAP
                p = (c[0] + n[0] * off, c[1] + n[1] * off)
                frac = self.blocked_fraction(tgt0, p, t_axis, 0.008)
                key = (frac > 0.02, frac > 0.0, math.dist(p, tc))
                if best is None or key < best[0]:
                    best = (key, p)
            return best[1], None
        return tc, None

    def verify_and_correct(self, plan, name, tgt0, refs0, goal_c, axis_deg):
        """After the release: measure the result against the task rule; one correction if it is off
        and there is time (own grasp, lift, shift, lower, release)."""
        st = self.r.state()
        left_s = ((st.native_step_limit or 0) - (st.native_step or 0)) / 30.0
        s = self.loc(name)
        if s is None or left_s < 10.0:
            return
        c = s.center_xyz[:2]
        refs = [r for r in refs0 if r is not None]
        err = math.dist(c, goal_c)
        bad = False
        if plan["kind"] == "between" and len(refs) == 2:
            a, b = refs[0].center_xyz[:2], refs[1].center_xyz[:2]
            L = math.dist(a, b) or 1e-6
            off = abs((b[0] - a[0]) * (a[1] - c[1]) - (a[0] - c[0]) * (b[1] - a[1])) / L
            bad = off > 0.02 or not (math.dist(c, a) < L and math.dist(c, b) < L)
        elif plan["kind"] == "next_to" and refs:
            ref = refs[0]
            dvec = (c[0] - ref.center_xyz[0], c[1] - ref.center_xyz[1])
            dn = math.hypot(*dvec) or 1e-6
            n = (dvec[0] / dn, dvec[1] / dn)
            gap = dn - self.half_extent_along(ref, n) - self.half_extent_along(s, n)
            bad = gap > 0.035 or gap < 0.002
        elif plan["kind"] == "move":
            d = plan["dir"]
            progress = (c[0] - tgt0.center_xyz[0]) * d[0] + (c[1] - tgt0.center_xyz[1]) * d[1]
            lateral = abs((c[0] - tgt0.center_xyz[0]) * -d[1] + (c[1] - tgt0.center_xyz[1]) * d[0])
            want = (goal_c[0] - tgt0.center_xyz[0]) * d[0] + (goal_c[1] - tgt0.center_xyz[1]) * d[1]
            bad = progress < want - 0.02 or lateral > 0.03
        log("verify: err %.3f bad %s left %.1fs" % (err, bad, left_s))
        if not bad or err < 0.008:
            return
        self.grasp(s)
        rest = max(0.012, (s.top_z or 0.03) - min(0.02, 0.5 * (s.height_m or 0.03)))
        held_yaw = self.r.state().yaw
        self.place(goal_c, axis_deg, rest, (0.0, 0.0), held_yaw, refs0)

    def run(self):
        plan = parse(self.r.task)
        log("plan", plan)
        if plan["kind"] == "bin":
            self.bin_task()
        else:
            self.spatial_task(plan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-port", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args, _ = ap.parse_known_args()
    cell = host.Cell(suite="planner", task="planner", seed=0, cuda_device=0)
    toolkit, daemons = host.boot(cell, args.out, bridge_port=args.bridge_port)
    robo = So101BenchRobot(host.make_backend(toolkit), ledger=Ledger(), primitives="full")
    card = host.dump_card(toolkit, cell, args.out)
    base_xy = None
    try:  # "robot base [x, y, z]": "[0.052, 0.181, -0.031] m"
        txt = (getattr(card, "facts", {}) or {}).get("robot base [x, y, z]", "")
        vals = [float(v) for v in re.findall(r"-?\d+\.\d+", txt)]
        base_xy = (vals[0], vals[1]) if len(vals) >= 2 else None
    except Exception:  # noqa: BLE001
        base_xy = None
    started = time.time()
    status = "finished"
    try:
        sam3 = getattr(getattr(toolkit, "_primitives", None), "sam3_client", None)
        Planner(robo, debug_dir=args.out, base_xy=base_xy, sam3=sam3).run()
    except EpisodeFinished:
        status = "episode_finished"
    except Exception:  # noqa: BLE001
        status = "planner_exception"
        log("ERROR", traceback.format_exc())
    finally:
        robo._ledger.write_jsonl(args.out / "calls.jsonl")
        (args.out / "result.json").write_text(json.dumps({
            "status": status, "planner": "deterministic", "policy_wall_s": round(time.time() - started, 1),
            "stateful_calls": sum(1 for r in robo._ledger.records if r.stateful)}, indent=1))
        toolkit.conclude()
        time.sleep(1)
        for d in daemons:
            d.stop()


if __name__ == "__main__":
    main()
