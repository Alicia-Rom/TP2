"""
JETSON - Servidor de IA para DeepRacer/ARTEMIS.

Este script es la parte separada de IA del codigo local de giro_sin_trayectoria.2.py.
La Jetson SOLO hace:
  1. Recibir imagenes UDP del CORE con paquete tipo b'I'.
  2. Ejecutar YOLO.
  3. Filtrar senales lejanas usando los umbrales del JSON de acciones.
  4. Aplicar la logica de STOP de una sola vez mientras la misma senal siga visible.
  5. Devolver al CORE un diccionario con detecciones + decision.

El CORE sigue encargandose de:
  - recibir coche/LIDAR/bateria,
  - calcular trayectoria con artemis_autonomous_car,
  - enviar control b'C' al coche.
"""

import argparse
import pickle
import socket
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ai_inference_enginev2 import (
    DEFAULT_MODEL_PATH,
    LocalYOLOObjectDetector,
    ObjectDecisionEngine,
    load_actions,
)


MAX_PACKET_SIZE = 99999
DEFAULT_WARMUP_RUNS = 5
DEFAULT_ACTIONS_CONFIG = Path(__file__).resolve().with_name(
    "actions_config_rada_tpii_complete.json"
)

# STOP ya gestionado: se ignora mientras siga visible y se rearma cuando desaparece.
STOP_SIGNAL_LABELS = {"stop", "stop_sign"}
STOP_REARM_AFTER_MISSING_SECONDS = 1.0


def normalize_label(label: str):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def is_stop_signal(detection):
    return normalize_label(detection.label) in STOP_SIGNAL_LABELS


def decode_image_packet(data):
    """Decodifica un paquete UDP tipo b'I' recibido desde el CORE."""
    payload = bytes(data[1:])
    encoded_image = pickle.loads(payload, encoding="latin1")
    return cv2.imdecode(encoded_image, 1)


def box_area(detection):
    if detection.width is None or detection.height is None:
        return 0.0
    return float(detection.width) * float(detection.height)


def detection_center(detection):
    if detection.center_x is None or detection.center_y is None:
        return None
    return float(detection.center_x), float(detection.center_y)


def update_detection_metrics(detections, state, now):
    """
    Calcula metricas extra parecidas al codigo local:
    - area de caja,
    - velocidad aparente px/s comparando con el frame anterior.
    Estas metricas se devuelven al CORE para poder dibujarlas o depurar.
    """
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
            speed_px_s = (best_distance_sq ** 0.5) / dt if best_distance_sq is not None else None

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


def action_thresholds(action):
    """Lee del JSON los umbrales de cercania para cada senal."""
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
    """
    Una senal solo se usa para decidir cuando cumple los umbrales configurados
    en actions_config_rada_tpii_complete.json.
    """
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
    """
    Misma idea del codigo local: el STOP cercano se ejecuta una vez. Luego se
    oculta al motor de decision hasta que desaparezca de la imagen.
    """
    now = time.time()
    close_stop_seen = any(is_stop_signal(detection) for detection in detections)

    if close_stop_seen:
        stop_state["last_seen_ts"] = now
    elif stop_state["handled"] and now - stop_state["last_seen_ts"] >= STOP_REARM_AFTER_MISSING_SECONDS:
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


def detection_to_dict(detection, metric=None):
    center_x = detection.center_x
    center_y = detection.center_y
    width = detection.width
    height = detection.height

    if None not in (center_x, center_y, width, height):
        x1 = float(center_x - width / 2)
        y1 = float(center_y - height / 2)
        x2 = float(center_x + width / 2)
        y2 = float(center_y + height / 2)
        box = [x1, y1, x2, y2]
    else:
        box = [0.0, 0.0, 0.0, 0.0]

    item = {
        "label": detection.label,
        "confidence": float(detection.confidence),
        "center_x": float(center_x) if center_x is not None else None,
        "center_y": float(center_y) if center_y is not None else None,
        "width": float(width) if width is not None else None,
        "height": float(height) if height is not None else None,
        "box": box,
    }

    if metric is not None:
        item["area"] = float(metric.get("area", 0.0) or 0.0)
        speed = metric.get("speed_px_s", None)
        item["speed_px_s"] = float(speed) if speed is not None else None

    return item


def decision_to_dict(decision):
    return {
        "stop": bool(decision.stop),
        "control_mode": int(decision.control_mode) if decision.control_mode else 0,
        "throttle_cap": (
            float(decision.throttle_cap)
            if decision.throttle_cap is not None
            else None
        ),
        "source_label": decision.source_label if decision.source_label else "",
        "source_confidence": (
            float(decision.source_confidence)
            if decision.source_confidence is not None
            else 0.0
        ),
        # Compatibilidad con tu ai_inference_engine.py actual:
        # algunas maniobras especiales, como u_turn o parking_cones,
        # fuerzan control manual desde la IA.
        "force_manual_control": bool(getattr(decision, "force_manual_control", False)),
    }


def metric_for_detection(target, raw_detections, raw_metrics):
    """Busca la metrica que corresponde al mismo objeto de deteccion."""
    for detection, metric in zip(raw_detections, raw_metrics):
        if detection is target:
            return metric
    return None


def build_response(
    raw_detections,
    used_detections,
    rejected_detections,
    suppressed_stop_detections,
    raw_metrics,
    decision,
    inference_ms,
    stop_state,
):
    return {
        "type": "AI_DECISION",
        "detections": [
            detection_to_dict(item, metric_for_detection(item, raw_detections, raw_metrics))
            for item in used_detections
        ],
        "raw_detections": [
            detection_to_dict(item, metric)
            for item, metric in zip(raw_detections, raw_metrics)
        ],
        "decision": decision_to_dict(decision),
        "inference_ms": float(inference_ms),
        "raw_detection_count": len(raw_detections),
        "used_detection_count": len(used_detections),
        "rejected_detection_count": len(rejected_detections),
        "suppressed_stop_count": len(suppressed_stop_detections),
        # Compatibilidad con tu CORE anterior.
        "rejected_stop_count": len(rejected_detections),
        "stop_handled": bool(stop_state["handled"]),
        "timestamp": time.time(),
    }


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
        return f"LENTO {decision.throttle_cap:.2f} por {decision.source_label} ({decision.source_confidence:.2f})"
    return "sin accion"


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

            new_cap = max(0.0, min(new_cap, 1.0))
            updated_action["throttle_cap"] = new_cap

            if new_cap != original_cap:
                changed.append((label, original_cap, new_cap))

        updated_actions[label] = updated_action

    return updated_actions, changed


def model_uses_tensorrt(model_path):
    return Path(model_path).suffix.lower() == ".engine"


def inspect_cuda():
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else ""
        cuda_version = getattr(torch.version, "cuda", None)
        return {
            "available": cuda_available,
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "error": None,
        }
    except Exception as error:
        return {
            "available": False,
            "gpu_name": "",
            "cuda_version": None,
            "error": str(error),
        }


def resolve_device(requested_device, model_path, cuda_info):
    device_text = str(requested_device).strip()
    uses_tensorrt = model_uses_tensorrt(model_path)

    if not device_text or device_text.lower() in ("none", "auto"):
        device_text = "0" if uses_tensorrt else "cpu"

    if uses_tensorrt and device_text.lower() == "cpu":
        raise ValueError(
            "El modelo .engine usa TensorRT y necesita GPU/CUDA. "
            "Ejecuta con --device 0, no con --device cpu."
        )

    if cuda_info["available"]:
        return device_text

    if uses_tensorrt:
        detail = f" Detalle: {cuda_info['error']}" if cuda_info["error"] else ""
        raise RuntimeError(
            "CUDA no esta disponible y el modelo .engine no puede ejecutarse en CPU."
            + detail
        )

    print("[JETSON IA] CUDA no disponible para PyTorch. Uso CPU.")
    return "cpu"


def warm_up_detector(detector, image_size, runs=DEFAULT_WARMUP_RUNS):
    runs = max(int(runs), 0)
    if runs == 0:
        return 0, 0.0

    warmup_frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    start = time.perf_counter()
    for _ in range(runs):
        detector.infer(warmup_frame)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return runs, elapsed_ms / runs


def recv_latest_image_packet(sock, drop_stale_frames):
    data, address = sock.recvfrom(MAX_PACKET_SIZE)

    if not drop_stale_frames:
        return data, address, 0

    latest_data = data
    latest_address = address
    dropped = 0

    sock.setblocking(False)
    try:
        while True:
            new_data, new_address = sock.recvfrom(MAX_PACKET_SIZE)
            if not new_data:
                continue

            packet_type = struct.unpack("c", bytes([new_data[0]]))[0]
            if packet_type == b"I":
                latest_data = new_data
                latest_address = new_address
                dropped += 1
    except BlockingIOError:
        pass
    finally:
        sock.setblocking(True)

    return latest_data, latest_address, dropped


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Servidor IA Jetson separado del CORE para DeepRacer/ARTEMIS."
    )
    parser.add_argument("--server-ip", default="192.168.50.2")
    parser.add_argument("--server-port", type=int, default=21000)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--actions-config", default=str(DEFAULT_ACTIONS_CONFIG))
    parser.add_argument("--device", default="0", help="Ejemplos: cpu, 0, 0,1")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--decision-hold-seconds", type=float, default=1.5)
    parser.add_argument(
        "--current-speed",
        type=float,
        default=0.0,
        help=(
            "Velocidad aproximada del coche en m/s para ObjectDecisionEngine.decide(). "
            "Si no se mide desde el CORE, se deja en 0.0; el motor de decision aplica "
            "su minimo interno cuando lo necesita."
        ),
    )
    parser.add_argument("--inference-every", type=int, default=1)
    parser.add_argument("--drop-stale-frames", dest="drop_stale_frames", action="store_true")
    parser.add_argument("--keep-queued-frames", dest="drop_stale_frames", action="store_false")
    parser.set_defaults(drop_stale_frames=True)
    parser.add_argument("--slow-throttle-cap", type=float, default=None)
    parser.add_argument("--slow-throttle-scale", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--log-top-k", type=int, default=3)
    return parser


def main():
    args = build_arg_parser().parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    uses_tensorrt = model_uses_tensorrt(model_path)
    cuda_info = inspect_cuda()
    args.device = resolve_device(args.device, model_path, cuda_info)

    actions = load_actions(args.actions_config)
    actions, slow_speed_changes = configure_slow_action_speeds(
        actions,
        slow_throttle_cap=args.slow_throttle_cap,
        slow_throttle_scale=args.slow_throttle_scale,
    )

    detector = LocalYOLOObjectDetector(
        model_path=args.model_path,
        min_confidence=args.min_confidence,
        device=args.device,
        image_size=args.image_size,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )

    warmup_runs, warmup_avg_ms = warm_up_detector(
        detector,
        image_size=args.image_size,
        runs=DEFAULT_WARMUP_RUNS,
    )

    decision_engine = ObjectDecisionEngine(
        actions=actions,
        decision_hold_seconds=args.decision_hold_seconds,
    )

    server_address = (args.server_ip, args.server_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)

    print(f"[JETSON IA SPLIT] Escuchando UDP {args.server_ip}:{args.server_port}")
    print(f"[JETSON IA SPLIT] Modelo YOLO: {model_path}")
    print(f"[JETSON IA SPLIT] TensorRT: {'activo' if uses_tensorrt else 'no activo'}")
    print(f"[JETSON IA SPLIT] Device: {args.device}")
    print(f"[JETSON IA SPLIT] CUDA disponible: {cuda_info['available']}")
    print(f"[JETSON IA SPLIT] GPU: {cuda_info['gpu_name'] or 'no detectada'}")
    print(f"[JETSON IA SPLIT] CUDA version PyTorch: {cuda_info['cuda_version'] or 'desconocida'}")
    print(
        "[JETSON IA SPLIT] Warm-up TensorRT/GPU: "
        f"{warmup_runs} inferencias, media={warmup_avg_ms:.1f} ms"
    )
    print(f"[JETSON IA SPLIT] Config acciones: {args.actions_config}")
    print(f"[JETSON IA SPLIT] Acciones configuradas: {sorted(actions.keys())}")
    print(f"[JETSON IA SPLIT] Current speed para decide(): {args.current_speed:.2f} m/s")
    print(f"[JETSON IA SPLIT] Filtro cercania: {format_filter_settings(actions)}")
    print(f"[JETSON IA SPLIT] Drop stale frames: {args.drop_stale_frames}")
    if slow_speed_changes:
        print("[JETSON IA SPLIT] Velocidades type=slow ajustadas:")
        for label, old_cap, new_cap in slow_speed_changes:
            print(f"- {label}: {old_cap:.2f} -> {new_cap:.2f}")

    frame_counter = 0
    last_raw_detections = []
    last_raw_metrics = []
    last_used_detections = []
    last_rejected_detections = []
    last_suppressed_stop_detections = []
    last_decision = None
    last_inference_ms = 0.0
    dropped_input_packets = 0
    detection_motion_state = {"tracks": {}, "next_track_id": 1}
    stop_state = {"handled": False, "last_seen_ts": 0.0}

    while True:
        data, core_address, dropped_now = recv_latest_image_packet(
            sock,
            args.drop_stale_frames,
        )
        dropped_input_packets += dropped_now

        if not data:
            continue

        packet_type = struct.unpack("c", bytes([data[0]]))[0]
        if packet_type != b"I":
            continue

        try:
            img = decode_image_packet(data)
            if img is None:
                print("[JETSON IA SPLIT] Imagen no valida")
                continue

            frame_counter += 1

            if frame_counter % max(args.inference_every, 1) == 0:
                start = time.perf_counter()
                _, last_raw_detections = detector.infer(img)
                last_raw_metrics = update_detection_metrics(
                    last_raw_detections,
                    detection_motion_state,
                    time.perf_counter(),
                )
                last_used_detections, last_rejected_detections = filter_distant_detections(
                    last_raw_detections,
                    actions,
                )
                last_used_detections, last_suppressed_stop_detections = suppress_already_handled_stop(
                    last_used_detections,
                    stop_state,
                )
                last_inference_ms = (time.perf_counter() - start) * 1000.0
                last_decision = decision_engine.decide(last_used_detections, args.current_speed)

            if last_decision is None:
                last_decision = decision_engine.decide([], args.current_speed)

            response = build_response(
                raw_detections=last_raw_detections,
                used_detections=last_used_detections,
                rejected_detections=last_rejected_detections,
                suppressed_stop_detections=last_suppressed_stop_detections,
                raw_metrics=last_raw_metrics,
                decision=last_decision,
                inference_ms=last_inference_ms,
                stop_state=stop_state,
            )
            sock.sendto(pickle.dumps(response), core_address)

            if args.log_every and frame_counter % args.log_every == 0:
                print(
                    f"[JETSON IA SPLIT] frame={frame_counter} "
                    f"infer_ms={last_inference_ms:.1f} "
                    f"raw={len(last_raw_detections)} "
                    f"usadas={summarize_detections(last_used_detections, args.log_top_k)} "
                    f"filtradas={len(last_rejected_detections)} "
                    f"stops_ignorados={len(last_suppressed_stop_detections)} "
                    f"cola_descartados={dropped_input_packets} "
                    f"decision={summarize_decision(last_decision)}"
                )
                dropped_input_packets = 0

        except Exception as error:
            print(f"[JETSON IA SPLIT] Error procesando frame: {error}")
            detection_motion_state = {"tracks": {}, "next_track_id": 1}


if __name__ == "__main__":
    main()
