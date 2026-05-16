import argparse
import pickle
import socket
import struct
import time
from pathlib import Path

import cv2

import artemis_autonomous_car
from ai_inference_engine import (
    DEFAULT_MODEL_PATH,
    LocalYOLOObjectDetector,
    ObjectDecisionEngine,
    load_actions,
)


MAX_PACKET_SIZE = 99999

# IP fija del servidor UDP. Se deja en codigo para evitar tener que pasarla
# como argumento cada vez que se arranca el script.
SERVER_IP = "172.16.0.1"
DEFAULT_ACTIONS_CONFIG = Path(__file__).resolve().with_name("actions_config_rada_tpii_complete.json")

# Cuando un STOP ya ha provocado una parada, se ignora mientras siga visible.
# Se rearma cuando deja de verse durante este tiempo.
STOP_SIGNAL_LABELS = {"stop", "stop_sign"}
STOP_REARM_AFTER_MISSING_SECONDS = 1.0


# Envia al vehiculo un paquete de control UDP con el formato que espera ARTEMIS:
# un byte de tipo ('C') seguido de dos dobles con giro y aceleracion.
def send_control(sock, control_giro, control_acelerador, address):
    payload = (
        struct.pack("c", bytes("C", "ascii"))
        + struct.pack("d", round(control_giro, 3))
        + struct.pack("d", round(control_acelerador, 3))
    )
    sock.sendto(payload, address)


def clamp_control(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def controls_without_road(control_mode, base_throttle, args):
    if control_mode == 1:
        return args.no_road_left_steering + args.steering_calibration, base_throttle
    if control_mode == 3:
        return args.no_road_right_steering + args.steering_calibration, base_throttle
    return args.no_road_straight_steering + args.steering_calibration, base_throttle


# Convierte la ruta definida por linea de comandos en una lista de enteros.
# Cada numero indica que hacer en el siguiente cruce:
# 1 = izquierda, 2 = recto, 3 = derecha.
def parse_route(route_text: str):
    route = []
    for raw_value in route_text.split(","):
        value = raw_value.strip()
        if not value:
            continue
        route.append(int(value))

    if not route:
        raise ValueError("La ruta no puede estar vacia")

    return route


# Guarda el paquete mas reciente de cada tipo y cuenta cuantos se han descartado.
def _remember_packet(packet_store, packet_counts, data, address):
    data_type = struct.unpack("c", bytes([data[0]]))[0]
    packet_store[data_type] = (data, address)
    packet_counts[data_type] = packet_counts.get(data_type, 0) + 1


# Lee al menos un paquete del socket y, opcionalmente, vacia la cola para
# quedarse con el mas reciente de cada tipo. Asi se evita procesar frames viejos.
def recv_latest_packets(sock, drop_stale_frames: bool):
    packet_store = {}
    packet_counts = {}

    data, address = sock.recvfrom(MAX_PACKET_SIZE)
    _remember_packet(packet_store, packet_counts, data, address)

    if not drop_stale_frames:
        return packet_store, packet_counts

    sock.setblocking(False)
    try:
        while True:
            data, address = sock.recvfrom(MAX_PACKET_SIZE)
            _remember_packet(packet_store, packet_counts, data, address)
    except BlockingIOError:
        pass
    finally:
        sock.setblocking(True)

    return packet_store, packet_counts


def decode_payload(data):
    payload = bytes(data[1:])
    return pickle.loads(payload, encoding="latin1")


def box_area(detection):
    # Si YOLO no devuelve dimensiones de caja, no podemos estimar cercania por area.
    if detection.width is None or detection.height is None:
        return 0.0
    return float(detection.width) * float(detection.height)


def detection_center(detection):
    if detection.center_x is None or detection.center_y is None:
        return None
    return float(detection.center_x), float(detection.center_y)


def update_detection_metrics(detections, state, now):
    # Estima velocidad aparente en pixeles/segundo asociando cada deteccion con
    # la deteccion previa de la misma clase cuyo centro quede mas cerca.
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
            max_match_distance = max(float(detection.width or 0), float(detection.height or 0), 80.0)
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


def draw_detection_metrics(frame, detections, metrics):
    for detection, metric in zip(detections[:5], metrics[:5]):
        if None in (detection.center_x, detection.center_y, detection.width, detection.height):
            continue

        x1 = int(detection.center_x - detection.width / 2)
        y1 = int(detection.center_y - detection.height / 2)
        y_text = max(y1 + 18, 20)
        speed = metric["speed_px_s"]
        speed_text = "--" if speed is None else f"{speed:.1f}"
        text = f"area={metric['area']:.0f}px2 v={speed_text}px/s"

        cv2.putText(
            frame,
            text,
            (max(x1, 5), y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def draw_detection_overlay(frame, detections, metrics):
    for detection, metric in zip(detections[:5], metrics[:5]):
        if None in (detection.center_x, detection.center_y, detection.width, detection.height):
            continue

        x1 = int(detection.center_x - detection.width / 2)
        y1 = int(detection.center_y - detection.height / 2)
        x2 = int(detection.center_x + detection.width / 2)
        y2 = int(detection.center_y + detection.height / 2)
        label_text = f"{detection.label}: {detection.confidence:.2f}"
        speed = metric["speed_px_s"]
        speed_text = "--" if speed is None else f"{speed:.1f}"
        metrics_text = f"area={metric['area']:.0f}px2 v={speed_text}px/s"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            label_text,
            (max(x1, 5), max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            metrics_text,
            (max(x1, 5), min(y1 + 20, frame.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


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
    # El filtro queda activo si alguna accion del JSON configura un umbral mayor que 0.
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
    # Extraemos las medidas de la caja de YOLO. Cuando falta una medida obligatoria,
    # la deteccion no pasa el filtro para evitar actuar por una senal lejana o incompleta.
    width = float(detection.width) if detection.width is not None else None
    height = float(detection.height) if detection.height is not None else None
    center_y = float(detection.center_y) if detection.center_y is not None else None
    area = box_area(detection)
    min_area, min_width, min_height, min_center_y = detection_distance_thresholds(detection, actions)

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
    # Si una senal esta lejos, se retira temporalmente de las detecciones usadas
    # para decidir. Cuando su caja cumple la cercania, pasa y se ejecuta su accion.
    if not distance_filter_enabled(actions):
        return list(detections), []

    kept = []
    rejected = []
    for detection in detections:
        if not detection_passes_distance_filter(detection, actions):
            rejected.append(detection)
        else:
            kept.append(detection)
    return kept, rejected


def normalize_label(label: str):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def is_stop_signal(detection):
    return normalize_label(detection.label) in STOP_SIGNAL_LABELS


def suppress_already_handled_stop(detections, stop_state):
    # Permite que el STOP cercano llegue una sola vez al motor de decision.
    # Despues lo oculta hasta que la senal desaparezca de la imagen, para que
    # el coche pueda continuar y no reinicie la parada en cada fotograma.
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
    # Texto usado en consola para confirmar que los umbrales salen del JSON.
    configured = []
    for label, action in sorted(actions.items()):
        settings = format_threshold_settings(*action_thresholds(action))
        if settings != "desactivado":
            configured.append(f"{label}: {settings}")

    if not configured:
        return "json=desactivado"

    return "json=" + "; ".join(configured)


def summarize_detections(detections, top_k: int):
    if not detections:
        return "sin detecciones"

    items = []
    for detection in detections[: max(top_k, 1)]:
        items.append(f"{detection.label} ({detection.confidence:.2f})")
    return ", ".join(items)


def summarize_decision(decision):
    if decision.stop:
        return f"STOP por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.control_mode == 1:
        return f"GIRAR IZQUIERDA por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.control_mode == 2:
        return f"SEGUIR RECTO por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.control_mode == 3:
        return f"GIRAR DERECHA por {decision.source_label} ({decision.source_confidence:.2f})"
    if decision.throttle_cap is not None:
        return (
            f"REDUCIR VELOCIDAD a {decision.throttle_cap:.2f} "
            f"por {decision.source_label} ({decision.source_confidence:.2f})"
        )
    return "sin accion"


def maybe_log_detection(detections, decision, top_k, min_interval, state):
    now = time.time()
    if not detections:
        return

    summary = summarize_detections(detections, top_k)
    decision_text = summarize_decision(decision)
    log_key = (summary, decision_text)

    if log_key != state["last_key"] or now - state["last_ts"] >= min_interval:
        print(f"[DETECCION] {summary} | decision: {decision_text}")
        state["last_key"] = log_key
        state["last_ts"] = now


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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Servidor UDP para Deep Racer con baja latencia, acciones completas y logs de deteccion."
    )
    parser.add_argument("--server-port", type=int, default=20001)
    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--show-inference", action="store_true")
    parser.add_argument("--route", default="2,2,2,2,2,2,2,2,2,2,0")
    parser.add_argument("--steering-calibration", type=float, default=0.19)
    parser.add_argument(
        "--steering-gain",
        type=float,
        default=1.25,
        help="Multiplica el giro calculado. Valores >1 cierran mas el giro; ejemplo: 1.25.",
    )
    parser.add_argument(
        "--no-road-straight-steering",
        type=float,
        default=0.0,
        help="Giro usado sin carretera cuando no hay senal de direccion.",
    )
    parser.add_argument(
        "--no-road-left-steering",
        type=float,
        default=0.45,
        help="Giro usado sin carretera si la senal ordena izquierda.",
    )
    parser.add_argument(
        "--no-road-right-steering",
        type=float,
        default=-0.45,
        help="Giro usado sin carretera si la senal ordena derecha.",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default=None, help="Ejemplos: cpu, 0, 0,1")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--decision-hold-seconds", type=float, default=1.5)
    parser.add_argument("--inference-every", type=int, default=1)
    parser.add_argument("--actions-config", default=str(DEFAULT_ACTIONS_CONFIG))
    parser.add_argument(
        "--slow-throttle-cap",
        type=float,
        default=None,
        help="Sobrescribe el throttle_cap de todas las acciones type=slow. Ejemplo: 0.22.",
    )
    parser.add_argument(
        "--slow-throttle-scale",
        type=float,
        default=None,
        help="Multiplica todos los throttle_cap type=slow del JSON. Ejemplo: 1.2 para ir un 20%% mas rapido.",
    )
    parser.add_argument("--drop-stale-frames", dest="drop_stale_frames", action="store_true")
    parser.add_argument("--keep-queued-frames", dest="drop_stale_frames", action="store_false")
    parser.set_defaults(drop_stale_frames=True)
    parser.add_argument(
        "--stats-every",
        type=int,
        default=60,
        help="Muestra tiempos y frames descartados cada N imagenes. Usa 0 para desactivarlo.",
    )
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=1.0,
        help="Segundos minimos entre logs identicos para no saturar la terminal.",
    )
    parser.add_argument(
        "--log-top-k",
        type=int,
        default=3,
        help="Numero maximo de detecciones a mostrar en cada log.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    route = parse_route(args.route)
    actions = load_actions(args.actions_config)
    actions, slow_speed_changes = configure_slow_action_speeds(
        actions,
        slow_throttle_cap=args.slow_throttle_cap,
        slow_throttle_scale=args.slow_throttle_scale,
    )

    auto_utils = artemis_autonomous_car.artemis_autonomous_car(route, args.steering_calibration)
    detector = LocalYOLOObjectDetector(
        model_path=args.model_path,
        min_confidence=args.min_confidence,
        device=args.device,
        image_size=args.image_size,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    decision_engine = ObjectDecisionEngine(
        actions=actions,
        decision_hold_seconds=args.decision_hold_seconds,
    )

    # El servidor escucha siempre en la IP fija definida arriba; el puerto sigue
    # siendo configurable porque no afecta a la logica de deteccion.
    server_address = (SERVER_IP, args.server_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_address)

    print(f"Escuchando en UDP {server_address[0]}:{server_address[1]}")
    print(f"Ruta por defecto: {route}")
    print(f"Modelo YOLO: {Path(detector.model_path)}")
    print(f"Config de acciones: {args.actions_config}")
    print(f"Frames obsoletos descartados: {args.drop_stale_frames}")
    print(f"Acciones configuradas: {sorted(actions.keys())}")
    print(f"Filtro de cercania para senales: {format_filter_settings(actions)}")
    print(f"Ganancia de giro: {args.steering_gain:.2f}")
    print(f"Compensacion de direccion: {args.steering_calibration:.2f}")
    print(
        "Avance sin carretera: "
        f"throttle=artemis/lidar "
        f"recto={args.no_road_straight_steering:.2f} "
        f"izquierda={args.no_road_left_steering:.2f} "
        f"derecha={args.no_road_right_steering:.2f}"
    )
    if slow_speed_changes:
        print("Velocidades type=slow ajustadas:")
        for label, old_cap, new_cap in slow_speed_changes:
            print(f"- {label}: {old_cap:.2f} -> {new_cap:.2f}")

    frame_counter = 0
    # raw_detections conserva todo lo que devuelve YOLO. last_detections contiene
    # lo que realmente se entrega al motor de decision tras filtrar senales lejanas.
    raw_detections = []
    last_detections = []
    detection_metrics = []
    detection_motion_state = {"tracks": {}, "next_track_id": 1}
    filtered_detection_counter = 0
    suppressed_stop_counter = 0
    dropped_image_packets = 0
    inference_ms_acc = 0.0
    loop_ms_acc = 0.0
    log_state = {"last_key": None, "last_ts": 0.0}
    stop_state = {"handled": False, "last_seen_ts": 0.0}

    while True:
        loop_start = time.perf_counter()
        latest_packets, packet_counts = recv_latest_packets(sock, args.drop_stale_frames)
        dropped_image_packets += max(packet_counts.get(b"I", 0) - 1, 0)

        if b"L" in latest_packets:
            decoded_lidar = decode_payload(latest_packets[b"L"][0])
            auto_utils.proceso_lidar(decoded_lidar, False)

        if b"B" in latest_packets:
            decoded_battery = decode_payload(latest_packets[b"B"][0])
            auto_utils.set_battery_level(decoded_battery)

        image_packet = latest_packets.get(b"I")
        if image_packet is None:
            continue

        data, address = image_packet
        decoded_image = decode_payload(data)
        img = cv2.imdecode(decoded_image, 1)
        if img is None:
            continue

        frame_counter += 1

        inference_elapsed_ms = 0.0
        if frame_counter % max(args.inference_every, 1) == 0:
            try:
                inference_start = time.perf_counter()
                _, raw_detections = detector.infer(img)
                detection_metrics = update_detection_metrics(
                    raw_detections,
                    detection_motion_state,
                    time.perf_counter(),
                )
                # Antes de decidir la accion, retiramos las senales que todavia
                # estan lejos segun el tamano y posicion de su caja YOLO.
                last_detections, rejected_detections = filter_distant_detections(raw_detections, actions)
                filtered_detection_counter += len(rejected_detections)
                last_detections, suppressed_stop_detections = suppress_already_handled_stop(
                    last_detections,
                    stop_state,
                )
                suppressed_stop_counter += len(suppressed_stop_detections)
                inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000.0
                inference_ms_acc += inference_elapsed_ms
            except Exception as error:
                print(f"[WARN] Error ejecutando inferencia: {error}")
                raw_detections = []
                last_detections = []
                detection_metrics = []
                detection_motion_state = {"tracks": {}, "next_track_id": 1}

        # El motor de decision trabaja con las detecciones ya filtradas: cualquier
        # senal debe cumplir la cercania antes de activar su accion.
        decision = decision_engine.decide(last_detections)
        #decision = decision_engine.decide(last_detections, auto_utils.lidar_throttle_control)
        maybe_log_detection(
            last_detections,
            decision,
            top_k=args.log_top_k,
            min_interval=max(args.log_every_seconds, 0.0),
            state=log_state,
        )

        auto_utils.set_stop(1 if decision.stop else 0)
        control_mode = decision.control_mode if decision.control_mode else 0

        control_giro, control_acelerador, trayectory_not_found = auto_utils.proceso_fotograma(
            img,
            args.show_info,
            control_mode,
        )

        if trayectory_not_found:
            control_giro, control_acelerador = controls_without_road(
                control_mode,
                auto_utils.lidar_throttle_control,
                args,
            )

        control_giro = clamp_control(control_giro * args.steering_gain)

        if decision.throttle_cap is not None:
            control_acelerador = min(control_acelerador, decision.throttle_cap)

        if decision.stop:
            # Refuerzo final: si la decision activa STOP, anulamos acelerador aunque
            # el controlador de trayectoria haya calculado una velocidad positiva.
            control_acelerador = 0

        send_control(sock, control_giro, control_acelerador, address)

        if args.show_inference:
            img_overlay = img.copy()
            draw_detection_overlay(img_overlay, raw_detections, detection_metrics)
            cv2.imshow("Automatic object control low latency", img_overlay)
            cv2.waitKey(1)

        loop_elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
        loop_ms_acc += loop_elapsed_ms

        if args.stats_every and frame_counter % args.stats_every == 0:
            avg_loop_ms = loop_ms_acc / args.stats_every
            avg_inference_ms = inference_ms_acc / max(args.stats_every // max(args.inference_every, 1), 1)
            approx_fps = 1000.0 / avg_loop_ms if avg_loop_ms > 0 else 0.0
            print(
                "[STATS] "
                f"frame={frame_counter} "
                f"loop_avg_ms={avg_loop_ms:.1f} "
                f"infer_last_ms={inference_elapsed_ms:.1f} "
                f"infer_avg_ms={avg_inference_ms:.1f} "
                f"fps_aprox={approx_fps:.1f} "
                f"frames_descartados={dropped_image_packets} "
                f"detecciones_filtradas={filtered_detection_counter} "
                f"stops_ignorados_tras_parada={suppressed_stop_counter}"
            )
            inference_ms_acc = 0.0
            loop_ms_acc = 0.0
            dropped_image_packets = 0
            filtered_detection_counter = 0
            suppressed_stop_counter = 0


if __name__ == "__main__":
    main()
