from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import cv2

from portrait_evaluator.config import EvaluatorConfig
from portrait_evaluator.models import ImageFeatures, ROISet


@dataclass(slots=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    source: str


class FaceDetector:
    def __init__(self) -> None:
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, rgb: np.ndarray) -> FaceDetection | None:
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)); boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
        if len(boxes) == 0: return None
        x, y, w, h = max(boxes, key=lambda b: int(b[2]) * int(b[3])); return FaceDetection((int(x), int(y), int(w), int(h)), "opencv_haar")


def _normalize_bbox(box: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape; x, y, bw, bh = box
    if max(abs(x), abs(y), abs(bw), abs(bh)) <= 1.5:
        x, bw = x * w, bw * w; y, bh = y * h, bh * h
    xi = max(0, min(w - 1, int(round(x)))); yi = max(0, min(h - 1, int(round(y)))); wi = max(1, min(w - xi, int(round(bw)))); hi = max(1, min(h - yi, int(round(bh))))
    return xi, yi, wi, hi


def _ellipse_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], scale: float = 1.0) -> np.ndarray:
    h, w = shape; x, y, bw, bh = bbox; cx = x + bw / 2; cy = y + bh / 2
    axes = (max(1, int(round(bw * 0.46 * scale))), max(1, int(round(bh * 0.50 * scale)))); mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), axes, 0, 0, 360, 1, -1); return mask.astype(bool)


def _geometry_skin_zone(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape; x, y, bw, bh = bbox; mask = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in [(0.25, 0.12, 0.75, 0.34), (0.12, 0.43, 0.43, 0.72), (0.57, 0.43, 0.88, 0.72)]:
        xa=max(0,int(x+x0*bw)); xb=min(w,int(x+x1*bw)); ya=max(0,int(y+y0*bh)); yb=min(h,int(y+y1*bh)); mask[ya:yb,xa:xb]=True
    return mask


def build_rois(features: ImageFeatures, config: EvaluatorConfig, bbox_override: tuple[float, float, float, float] | None = None, detector: FaceDetector | None = None) -> tuple[ROISet | None, list[str]]:
    h, w = features.rgb.shape[:2]; reasons: list[str] = []
    if bbox_override is not None: bbox = _normalize_bbox(bbox_override, (h, w)); source = "manifest_bbox"
    else:
        detection = (detector or FaceDetector()).detect(features.rgb)
        if detection is None: return None, ["main face not detected"]
        bbox, source = detection.bbox, detection.source
    x, y, bw, bh = bbox
    if bw * bh / float(h * w) < float(config.get("roi.min_face_area_ratio")): return None, ["face area below minimum"]
    face = _ellipse_mask((h, w), bbox, 1.0); outer = _ellipse_mask((h, w), bbox, float(config.get("roi.ring_scale"))); ring = outer & ~face; background = ~outer; geometry = _geometry_skin_zone((h, w), bbox) & face
    ycrcb = cv2.cvtColor(features.rgb, cv2.COLOR_RGB2YCrCb); cr = ycrcb[...,1]; cb = ycrcb[...,2]
    skin_color = (cr >= int(config.get("roi.skin_cr_min"))) & (cr <= int(config.get("roi.skin_cr_max"))) & (cb >= int(config.get("roi.skin_cb_min"))) & (cb <= int(config.get("roi.skin_cb_max")))
    skin = geometry & skin_color & (features.luminance > 0.03) & (features.luminance < 0.97); min_pixels = int(config.get("roi.min_skin_pixels")); skin_source = "geometry_ycrcb"
    if int(skin.sum()) < min_pixels:
        skin = geometry & (features.luminance > 0.03) & (features.luminance < 0.97); skin_source = "geometry_fallback"; reasons.append("skin segmentation sparse; geometry fallback used")
    if int(skin.sum()) < min_pixels: return None, reasons + ["skin ROI below minimum"]
    if int(ring.sum()) < 20 or int(background.sum()) < 50: return None, reasons + ["insufficient face ring/background area"]
    reasons.append(f"face source: {source}"); return ROISet(face, skin, ring, background, skin_source, bbox), reasons


def face_composition_delta(a: ROISet, b: ROISet, shape_a: tuple[int, int], shape_b: tuple[int, int]) -> tuple[float, float]:
    if a.face_bbox is None or b.face_bbox is None: return 0.0, 0.0
    ah_img, aw_img = shape_a; bh_img, bw_img = shape_b; ax, ay, aw, ah = a.face_bbox; bx, by, bw, bh = b.face_bbox
    ac=((ax+aw/2)/aw_img,(ay+ah/2)/ah_img); bc=((bx+bw/2)/bw_img,(by+bh/2)/bh_img); center=math.hypot(ac[0]-bc[0],ac[1]-bc[1])/math.sqrt(2.0)
    a_area=(aw*ah)/float(aw_img*ah_img); b_area=(bw*bh)/float(bw_img*bh_img); scale_ev=abs(math.log2((a_area+1e-9)/(b_area+1e-9)))
    return center, scale_ev
