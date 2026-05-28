import time
from pathlib import Path

from ai_inference_engine import (
    DEFAULT_MODEL_PATH,
    LocalYOLOObjectDetector,
    ObjectDecisionEngine,
    load_actions,
)


DEFAULT_ACTIONS_CONFIG = Path(__file__).resolve().with_name(
    "actions_config_rada_tpii_complete.json"
)

STOP_SIGNAL_LABELS = {"stop", "stop_sign"}
STOP_REARM_AFTER_MISSING_SECONDS = 1.0


def normalize_label(label: str):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def is_stop_signal(detection):
    return normalize_label(detection.label) in STOP_SIGNAL_LABELS


def box_area(detection):
    if detection.width is None or detection.height is None:
        return 0.0
    return float(detection.width) * float(detection.height)


def detection_center(detection):
    if detection.center_x is None or detection.center_y is None:
        return None
    return float(detection.center_x), float(detection.center_y)


def update_detection_metrics(detections, state, now):
    previous_tracks = state["tracks"]
    used_track_ids = set()
    updated_tracks = {}
    metrics = []

    for detection in detections:
        center = detection_center(detection)
        area = box_area(detection)
        metric = {"area": area, "speed_px_s": None}

        if center is None:
            metrics.append(metric)
            continue

        best_track_id = None
        best_distance_sq = None

        for track_id, track in previous_tracks.items():
            if track_id in used_track_ids or track["label"] != detection.label:
                continue

            dx = center[0] - track["center_x"]
            dy = center[1] - track["center_y"]
            distance_sq = dx * dx + dy * dy
            max_match_distance = max(
                float(detection.width or 0),
                float(detection.height or 0),
                80.0,
            )

            if distance_sq > max_match_distance * max_match_distance:
                continue

            if best_distance_sq is None or distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_track_id = track_id

        if best_track_id is None:
            track_id = state["next_track_id"]
            state["next_track_id"] += 1
            speed_px_s = None
        else:
            track_id = best_track_id
            used_track_ids.add(track_id)
            previous = previous_tracks[track_id]
            dt = max(now - previous["timestamp"], 1e-6)
            speed_px_s = (
                (best_distance_sq ** 0.5) / dt
                if best_distance_sq is not None
                else None
            )

        metric["speed_px_s"] = speed_px_s
        metrics.append(metric)
        updated_tracks[track_id] = {
            "label": detection.label,
            "center_x": center[0],
            "center_y": center[1],
            "timestamp": now,
        }

    state["tracks"] = updated_tracks
    return metrics


def attach_detection_metrics(detections, metrics):
    for detection, metric in zip(detections, metrics):
        detection.area = float(metric.get("area", 0.0) or 0.0)
        speed = metric.get("speed_px_s", None)
        detection.speed_px_s = float(speed) if speed is not None else None

        if None not in (
            detection.center_x,
            detection.center_y,
            detection.width,
            detection.height,
        ):
            detection.x1 = float(detection.center_x - detection.width / 2)
            detection.y1 = float(detection.center_y - detection.height / 2)
            detection.x2 = float(detection.center_x + detection.width / 2)
            detection.y2 = float(detection.center_y + detection.height / 2)
        else:
            detection.x1 = 0.0
            detection.y1 = 0.0
            detection.x2 = 0.0
            detection.y2 = 0.0


def action_thresholds(action):
    if not isinstance(action, dict):
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(action.get("min_area", 0.0)),
        float(action.get("min_width", 0.0)),
        float(action.get("min_height", 0.0)),
        float(action.get("min_center_y", 0.0)),
    )


def distance_filter_enabled(actions):
    return any(
        value > 0
        for action in actions.values()
        for value in action_thresholds(action)
    )


def action_for_detection(detection, actions):
    normalized = detection.label.strip().lower()
    if normalized in actions:
        return actions[normalized]

    normalized = normalize_label(detection.label)
    return actions.get(normalized)


def detection_distance_thresholds(detection, actions):
    return action_thresholds(action_for_detection(detection, actions))


def detection_passes_distance_filter(detection, actions):
    width = float(detection.width) if detection.width is not None else None
    height = float(detection.height) if detection.height is not None else None
    center_y = float(detection.center_y) if detection.center_y is not None else None
    area = box_area(detection)
    min_area, min_width, min_height, min_center_y = detection_distance_thresholds(
        detection,
        actions,
    )

    if min_width > 0 and (width is None or width < min_width):
        return False

    if min_height > 0 and (height is None or height < min_height):
        return False

    if min_area > 0 and area < min_area:
        return False

    if min_center_y > 0 and (center_y is None or center_y < min_center_y):
        return False

    return True


def filter_distant_detections(detections, actions):
    if not distance_filter_enabled(actions):
        return list(detections), []

    kept = []
    rejected = []

    for detection in detections:
        if detection_passes_distance_filter(detection, actions):
            kept.append(detection)
        else:
            rejected.append(detection)

    return kept, rejected


def suppress_already_handled_stop(detections, stop_state):
    now = time.time()
    close_stop_seen = any(is_stop_signal(detection) for detection in detections)

    if close_stop_seen:
        stop_state["last_seen_ts"] = now
    elif (
        stop_state["handled"]
        and now - stop_state["last_seen_ts"] >= STOP_REARM_AFTER_MISSING_SECONDS
    ):
        stop_state["handled"] = False

    if not close_stop_seen:
        return list(detections), []

    if not stop_state["handled"]:
        stop_state["handled"] = True
        return list(detections), []

    kept = []
    suppressed = []
    for detection in detections:
        if is_stop_signal(detection):
            suppressed.append(detection)
        else:
            kept.append(detection)

    return kept, suppressed


def configure_slow_action_speeds(actions, slow_throttle_cap, slow_throttle_scale):
    updated_actions = {}
    changed = []

    for label, action in actions.items():
        if not isinstance(action, dict):
            updated_actions[label] = action
            continue

        updated_action = dict(action)
        if updated_action.get("type") == "slow":
            original_cap = float(updated_action.get("throttle_cap", 0.20))
            new_cap = original_cap

            if slow_throttle_scale is not None:
                new_cap *= slow_throttle_scale

            if slow_throttle_cap is not None:
                new_cap = slow_throttle_cap

            new_cap = max(-1.0, min(new_cap, 1.0))
            updated_action["throttle_cap"] = new_cap

            if new_cap != original_cap:
                changed.append((label, original_cap, new_cap))

        updated_actions[label] = updated_action

    return updated_actions, changed


def format_threshold_settings(min_area, min_width, min_height, min_center_y):
    parts = []
    if min_area > 0:
        parts.append(f"area>={min_area:.0f}")
    if min_width > 0:
        parts.append(f"width>={min_width:.0f}")
    if min_height > 0:
        parts.append(f"height>={min_height:.0f}")
    if min_center_y > 0:
        parts.append(f"center_y>={min_center_y:.0f}")
    return ", ".join(parts) if parts else "desactivado"


def format_filter_settings(actions):
    configured = []
    for label, action in sorted(actions.items()):
        settings = format_threshold_settings(*action_thresholds(action))
        if settings != "desactivado":
            configured.append(f"{label}: {settings}")

    if not configured:
        return "json=desactivado"

    return "json=" + "; ".join(configured)


def resolve_device(requested_device):
    device_text = str(requested_device or "cpu").strip()

    if device_text.lower() == "cpu":
        return "cpu"

    if device_text.lower() == "auto":
        device_text = "0"

    try:
        import torch

        if torch.cuda.is_available():
            return device_text

        print("[CORE LOCAL IA] CUDA no disponible para PyTorch. Uso CPU.")
        return "cpu"
    except Exception as error:
        print(f"[CORE LOCAL IA] No se pudo comprobar CUDA ({error}). Uso CPU.")
        return "cpu"


def summarize_detections(detections, top_k):
    if not detections:
        return "sin detecciones"

    return ", ".join(
        f"{detection.label}({detection.confidence:.2f})"
        for detection in detections[: max(top_k, 1)]
    )


def summarize_decision(decision):
    if decision.stop:
        return f"STOP por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.control_mode == 1:
        return f"IZQUIERDA por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.control_mode == 2:
        return f"RECTO por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.control_mode == 3:
        return f"DERECHA por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.throttle_cap is not None:
        return (
            f"LENTO {decision.throttle_cap:.2f} "
            f"por {decision.source_label} ({decision.source_confidence:.2f})"
        )
    return "sin accion"


class LocalAIProcessor:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        actions_config=DEFAULT_ACTIONS_CONFIG,
        device="cpu",
        image_size=640,
        iou_threshold=0.7,
        max_detections=50,
        min_confidence=0.5,
        decision_hold_seconds=1.5,
        current_speed=0.0,
        inference_every=1,
        slow_throttle_cap=None,
        slow_throttle_scale=None,
    ):
        self.device = resolve_device(device)
        self.actions = load_actions(actions_config)
        self.actions, self.slow_speed_changes = configure_slow_action_speeds(
            self.actions,
            slow_throttle_cap=slow_throttle_cap,
            slow_throttle_scale=slow_throttle_scale,
        )

        self.detector = LocalYOLOObjectDetector(
            model_path=model_path,
            min_confidence=min_confidence,
            device=self.device,
            image_size=image_size,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        self.decision_engine = ObjectDecisionEngine(
            actions=self.actions,
            decision_hold_seconds=decision_hold_seconds,
        )
        self.current_speed = current_speed
        self.inference_every = max(int(inference_every), 1)

        self.frame_counter = 0
        self.last_raw_detections = []
        self.last_raw_metrics = []
        self.last_used_detections = []
        self.last_rejected_detections = []
        self.last_suppressed_stop_detections = []
        self.last_decision = None
        self.last_inference_ms = 0.0
        self.detection_motion_state = {"tracks": {}, "next_track_id": 1}
        self.stop_state = {"handled": False, "last_seen_ts": 0.0}

    def process_frame(self, img):
        self.frame_counter += 1

        if self.frame_counter % self.inference_every == 0:
            start = time.perf_counter()
            _, self.last_raw_detections = self.detector.infer(img)
            self.last_raw_metrics = update_detection_metrics(
                self.last_raw_detections,
                self.detection_motion_state,
                time.perf_counter(),
            )
            attach_detection_metrics(self.last_raw_detections, self.last_raw_metrics)
            (
                self.last_used_detections,
                self.last_rejected_detections,
            ) = filter_distant_detections(
                self.last_raw_detections,
                self.actions,
            )
            (
                self.last_used_detections,
                self.last_suppressed_stop_detections,
            ) = suppress_already_handled_stop(
                self.last_used_detections,
                self.stop_state,
            )
            self.last_inference_ms = (time.perf_counter() - start) * 1000.0
            self.last_decision = self.decision_engine.decide(
                self.last_used_detections,
                self.current_speed,
            )

        if self.last_decision is None:
            self.last_decision = self.decision_engine.decide([], self.current_speed)

        return {
            "detections": self.last_used_detections,
            "raw_detections": self.last_raw_detections,
            "decision": self.last_decision,
            "inference_ms": float(self.last_inference_ms),
            "raw_detection_count": len(self.last_raw_detections),
            "used_detection_count": len(self.last_used_detections),
            "rejected_detection_count": len(self.last_rejected_detections),
            "suppressed_stop_count": len(self.last_suppressed_stop_detections),
            "rejected_stop_count": len(self.last_rejected_detections),
            "ok": True,
        }
