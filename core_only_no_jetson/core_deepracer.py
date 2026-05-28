"""
CORE - DeepRacer/ARTEMIS con IA local, sin Jetson.
Version RectoSinTrayectoria: no usa nunca la trayectoria del suelo.
El coche avanza recto y aplica --steering-calibration como pulso periodico,
pero sigue ejecutando las decisiones de senales de la IA.

Este script aplica la idea del codigo dividido anterior, pero ejecuta YOLO y
la logica de decision dentro del propio CORE:

  COCHE  -> CORE:    paquetes UDP I/L/B
  CORE   -> CORE:    inferencia local YOLO + decision IA
  CORE   -> COCHE:   paquete C con giro/acelerador

El CORE conserva:
  - artemis_autonomous_car solo como fuente de throttle_o/max_throttle,
  - throttle_cap/STOP,
  - control manual por teclado opcional.
"""

import argparse
import pickle
import socket
import struct
import time

import cv2

try:
    import msvcrt
except ImportError:
    msvcrt = None

import artemis_autonomous_car
from ai_inference_engine import DEFAULT_MODEL_PATH, Decision
from local_ai_pipeline import (
    DEFAULT_ACTIONS_CONFIG,
    LocalAIProcessor,
    format_filter_settings,
)


MAX_PACKET_SIZE = 99999

def send_control(sock, control_giro, control_acelerador, address):
    """Envia al coche byte b'C' + double giro + double acelerador."""
    payload = (
        struct.pack("c", bytes("C", "ascii"))
        + struct.pack("d", round(control_giro, 3))
        + struct.pack("d", round(control_acelerador, 3))
    )
    sock.sendto(payload, address)


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def apply_steering_gain(control_giro, steering_gain, steering_calibration):
    base_steering = control_giro - steering_calibration
    return clamp((base_steering * steering_gain) + steering_calibration)


def straight_without_road_steering(now, state, steering_calibration, every_seconds, pulse_seconds):
    if steering_calibration == 0 or every_seconds <= 0 or pulse_seconds <= 0:
        return 0.0

    if state["next_ts"] is None:
        state["next_ts"] = now + every_seconds
        state["until_ts"] = 0.0

    if now < state["until_ts"]:
        return steering_calibration

    if now >= state["next_ts"]:
        state["until_ts"] = now + pulse_seconds
        state["next_ts"] = now + every_seconds
        return steering_calibration

    return 0.0


def artemis_base_throttle(auto_utils):
    throttle = getattr(auto_utils, "throttle_o", 0.0)
    max_throttle = getattr(auto_utils, "max_throttle", 1.0)
    return clamp(min(throttle, max_throttle))


def controls_without_road(control_mode, base_throttle, steering_calibration):
    """
    Control de emergencia si artemis no encuentra trayectoria.
    control_mode: 1=izquierda, 2=recto, 3=derecha.
    La intensidad del giro se aplica despues con --steering-gain.
    """
    if control_mode == 1:
        return 1.0 + steering_calibration, base_throttle
    if control_mode == 3:
        return -1.0 + steering_calibration, base_throttle
    return 0.0, base_throttle


def parse_route(route_text: str):
    route = []
    for raw_value in route_text.split(","):
        value = raw_value.strip()
        if value:
            route.append(int(value))

    if not route:
        raise ValueError("La ruta no puede estar vacia")

    return route


def remember_packet(packet_store, packet_counts, data, address):
    if not data:
        return

    data_type = struct.unpack("c", bytes([data[0]]))[0]
    packet_store[data_type] = (data, address)
    packet_counts[data_type] = packet_counts.get(data_type, 0) + 1


def recv_latest_packets(sock, drop_stale_frames: bool):
    """Recibe del coche y se queda con el ultimo paquete de cada tipo."""
    packet_store = {}
    packet_counts = {}

    data, address = sock.recvfrom(MAX_PACKET_SIZE)
    remember_packet(packet_store, packet_counts, data, address)

    if not drop_stale_frames:
        return packet_store, packet_counts

    sock.setblocking(False)
    try:
        while True:
            data, address = sock.recvfrom(MAX_PACKET_SIZE)
            remember_packet(packet_store, packet_counts, data, address)
    except BlockingIOError:
        pass
    finally:
        sock.setblocking(True)

    return packet_store, packet_counts


def decode_payload(data):
    payload = bytes(data[1:])
    return pickle.loads(payload, encoding="latin1")


# -------------------------------------------------------------------------
# Control manual opcional
# -------------------------------------------------------------------------
def build_manual_state():
    return {
        "steering_override": None,
        "throttle_override": None,
        "last_steering_ts": 0.0,
        "last_throttle_ts": 0.0,
    }


def log_manual_state(state):
    steering_text = (
        f"{state['steering_override']:+.2f}"
        if state["steering_override"] is not None
        else "AUTO"
    )
    throttle_text = (
        f"{state['throttle_override']:+.2f}"
        if state["throttle_override"] is not None
        else "AUTO"
    )
    print(f"[MANUAL] giro={steering_text} acelerador={throttle_text}")


def apply_manual_command(command, state, args):
    now = time.time()
    changed = False

    if command in ("a", "A"):
        state["steering_override"] = -abs(args.manual_turn_speed)
        state["last_steering_ts"] = now
        changed = True
    elif command in ("d", "D"):
        state["steering_override"] = abs(args.manual_turn_speed)
        state["last_steering_ts"] = now
        changed = True
    elif command in ("w", "W"):
        state["throttle_override"] = abs(args.manual_forward_speed)
        state["last_throttle_ts"] = now
        changed = True
    elif command in ("2",):
        state["throttle_override"] = abs(args.manual_fast_forward_speed)
        state["last_throttle_ts"] = now
        changed = True
    elif command in ("s", "S"):
        state["throttle_override"] = -abs(args.manual_reverse_speed)
        state["last_throttle_ts"] = now
        changed = True
    elif command in ("x", "X"):
        state["throttle_override"] = -abs(args.manual_fast_reverse_speed)
        state["last_throttle_ts"] = now
        changed = True

    if changed:
        log_manual_state(state)


def poll_console_input(state, args):
    if msvcrt is None:
        return

    while msvcrt.kbhit():
        key = msvcrt.getwch()

        if key in ("\x00", "\xe0"):
            special = msvcrt.getwch()
            arrow_map = {
                "K": "a",
                "M": "d",
                "H": "w",
                "P": "s",
            }
            key = arrow_map.get(special)
            if key is None:
                continue

        apply_manual_command(key, state, args)


def poll_window_input(state, args):
    key_code = cv2.waitKeyEx(1)

    if key_code < 0:
        return

    if key_code in (81, 2424832):
        key = "a"
    elif key_code in (83, 2555904):
        key = "d"
    elif key_code in (82, 2490368):
        key = "w"
    elif key_code in (84, 2621440):
        key = "s"
    else:
        key = chr(key_code & 0xFF) if 0 <= (key_code & 0xFF) <= 255 else None

    if key:
        apply_manual_command(key, state, args)


def decay_manual_overrides(state, hold_seconds):
    now = time.time()

    if (
        state["steering_override"] is not None
        and now - state["last_steering_ts"] > hold_seconds
    ):
        state["steering_override"] = None

    if (
        state["throttle_override"] is not None
        and now - state["last_throttle_ts"] > hold_seconds
    ):
        state["throttle_override"] = None


def apply_manual_overrides(control_giro, control_acelerador, state):
    if state["steering_override"] is not None:
        control_giro = state["steering_override"]

    if state["throttle_override"] is not None:
        control_acelerador = state["throttle_override"]

    return clamp(control_giro), clamp(control_acelerador)


def empty_decision():
    return Decision()


def empty_ai_response():
    return {
        "detections": [],
        "raw_detections": [],
        "decision": empty_decision(),
        "inference_ms": 0.0,
        "raw_detection_count": 0,
        "used_detection_count": 0,
        "rejected_detection_count": 0,
        "suppressed_stop_count": 0,
        "rejected_stop_count": 0,
        "ok": False,
    }


# -------------------------------------------------------------------------
# Logs y visualizacion
# -------------------------------------------------------------------------
def summarize_detections(detections, top_k):
    if not detections:
        return "sin detecciones"

    summary_parts = []
    for detection in detections[: max(top_k, 1)]:
        width = detection.width
        height = detection.height
        if width is None or height is None:
            width = max(0.0, detection.x2 - detection.x1)
            height = max(0.0, detection.y2 - detection.y1)

        area = detection.area if detection.area else width * height
        speed = "--" if detection.speed_px_s is None else f"{detection.speed_px_s:.1f}px/s"
        summary_parts.append(
            f"{detection.label} ({detection.confidence:.2f}) "
            f"tam={width:.0f}x{height:.0f} area={area:.0f} vel={speed}"
        )

    return ", ".join(summary_parts)


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
        print(f"[DETECCION IA LOCAL] {summary} | decision: {decision_text}")
        state["last_key"] = log_key
        state["last_ts"] = now


def draw_detections(frame, detections, decision):
    for index, detection in enumerate(detections[:5]):
        if None not in (
            detection.center_x,
            detection.center_y,
            detection.width,
            detection.height,
        ):
            x1 = int(detection.center_x - detection.width / 2)
            y1 = int(detection.center_y - detection.height / 2)
            x2 = int(detection.center_x + detection.width / 2)
            y2 = int(detection.center_y + detection.height / 2)
        else:
            x1, y1, x2, y2 = (
                int(detection.x1),
                int(detection.y1),
                int(detection.x2),
                int(detection.y2),
            )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        speed = "--" if detection.speed_px_s is None else f"{detection.speed_px_s:.1f}"
        label_text = (
            f"{detection.label}: {detection.confidence:.2f} "
            f"area={detection.area:.0f} v={speed}px/s"
        )

        cv2.putText(
            frame,
            label_text,
            (max(x1, 5), max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y_pos = 25 + index * 22
        cv2.putText(
            frame,
            f"{detection.label}: {detection.confidence:.2f}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if decision.source_label:
        cv2.putText(
            frame,
            f"Decision: {summarize_decision(decision)}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CORE DeepRacer/ARTEMIS con IA local, sin Jetson."
    )

    parser.add_argument("--server-ip", default="172.16.0.1")
    parser.add_argument("--server-port", type=int, default=20001)

    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--actions-config", default=str(DEFAULT_ACTIONS_CONFIG))
    parser.add_argument("--device", default="cpu", help="Ejemplos: cpu, auto, 0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--decision-hold-seconds", type=float, default=1.5)
    parser.add_argument("--current-speed", type=float, default=0.0)
    parser.add_argument("--inference-every", type=int, default=1)
    parser.add_argument("--slow-throttle-cap", type=float, default=None)
    parser.add_argument("--slow-throttle-scale", type=float, default=None)

    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--show-inference", action="store_true")

    parser.add_argument("--route", default="2,2,2,2,2,2,2,2,2,2,0")
    parser.add_argument("--steering-calibration", type=float, default=-0.25)
    parser.add_argument(
        "--steering-gain",
        type=float,
        default=1.35,
        help="Multiplica solo los giros por senal. Valores >1 cierran mas el giro.",
    )
    parser.add_argument(
        "--straight-calibration-every-seconds",
        type=float,
        default=1.0,
        help="Intervalo entre pulsos de calibracion al avanzar recto sin trayectoria.",
    )
    parser.add_argument(
        "--straight-calibration-pulse-seconds",
        type=float,
        default=0.15,
        help="Duracion del pulso de calibracion al avanzar recto sin trayectoria.",
    )

    parser.add_argument("--drop-stale-frames", dest="drop_stale_frames", action="store_true")
    parser.add_argument("--keep-queued-frames", dest="drop_stale_frames", action="store_false")
    parser.set_defaults(drop_stale_frames=True)

    parser.add_argument("--stats-every", type=int, default=0)
    parser.add_argument("--log-every-seconds", type=float, default=1.0)
    parser.add_argument("--log-top-k", type=int, default=3)

    parser.add_argument("--manual-hold-seconds", type=float, default=0.20)
    parser.add_argument("--manual-forward-speed", type=float, default=0.20)
    parser.add_argument("--manual-reverse-speed", type=float, default=0.18)
    parser.add_argument("--manual-fast-forward-speed", type=float, default=0.35)
    parser.add_argument("--manual-fast-reverse-speed", type=float, default=0.30)
    parser.add_argument("--manual-turn-speed", type=float, default=1.0)

    return parser


def main():
    args = build_arg_parser().parse_args()
    route = parse_route(args.route)

    auto_utils = artemis_autonomous_car.artemis_autonomous_car(
        route,
        args.steering_calibration,
    )

    ai_processor = LocalAIProcessor(
        model_path=args.model_path,
        actions_config=args.actions_config,
        device=args.device,
        image_size=args.image_size,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
        min_confidence=args.min_confidence,
        decision_hold_seconds=args.decision_hold_seconds,
        current_speed=args.current_speed,
        inference_every=args.inference_every,
        slow_throttle_cap=args.slow_throttle_cap,
        slow_throttle_scale=args.slow_throttle_scale,
    )

    server_address = (args.server_ip, args.server_port)

    car_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    car_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    car_sock.bind(server_address)

    print(f"[CORE LOCAL] Escuchando coche UDP {args.server_ip}:{args.server_port}")
    print(f"[CORE LOCAL] IA local activa en device={ai_processor.device}")
    print(f"[CORE LOCAL] Modelo YOLO: {args.model_path}")
    print(f"[CORE LOCAL] Config acciones: {args.actions_config}")
    print(f"[CORE LOCAL] Acciones configuradas: {sorted(ai_processor.actions.keys())}")
    print(f"[CORE LOCAL] Filtro cercania: {format_filter_settings(ai_processor.actions)}")
    print(f"[CORE LOCAL] Inferencia cada {ai_processor.inference_every} frame(s)")
    print(f"[CORE LOCAL] Ruta: {route}")
    print(f"[CORE LOCAL] Modo: recto sin trayectoria")
    print(f"[CORE LOCAL] Ganancia de giro por senal: {args.steering_gain:.2f}")
    print(f"[CORE LOCAL] Compensacion direccion por pulso: {args.steering_calibration:.2f}")
    print(
        "[CORE LOCAL] Throttle recto: "
        f"throttle_o={auto_utils.throttle_o:.2f}, "
        f"max_throttle={auto_utils.max_throttle:.2f}"
    )
    print(
        "[CORE LOCAL] Pulso recto sin trayectoria: "
        f"cada {args.straight_calibration_every_seconds:.2f}s "
        f"durante {args.straight_calibration_pulse_seconds:.2f}s"
    )
    print(f"[CORE LOCAL] Drop stale frames: {args.drop_stale_frames}")
    print(
        "[CORE LOCAL] Direccion: "
        "sin trayectoria; se obedecen senales IA con control_mode"
    )
    if ai_processor.slow_speed_changes:
        print("[CORE LOCAL] Velocidades type=slow ajustadas:")
        for label, old_cap, new_cap in ai_processor.slow_speed_changes:
            print(f"- {label}: {old_cap:.2f} -> {new_cap:.2f}")
    print("[CORE LOCAL] Manual: w delante, s atras, a izquierda, d derecha, 2 rapido, x atras rapido.")
    print("[CORE LOCAL] En Linux usa --show-inference para capturar teclado desde la ventana OpenCV.")

    frame_counter = 0
    last_detections = []
    last_raw_detections = []
    last_decision = empty_decision()
    last_inference_ms = 0.0
    last_raw_detection_count = 0
    last_used_detection_count = 0
    last_rejected_detection_count = 0
    last_suppressed_stop_count = 0

    dropped_image_packets = 0
    ai_ok_count = 0
    ai_error_count = 0
    loop_ms_acc = 0.0
    ai_ms_acc = 0.0
    rejected_detection_acc = 0
    suppressed_stop_acc = 0

    log_state = {"last_key": None, "last_ts": 0.0}
    manual_state = build_manual_state()
    straight_calibration_state = {"next_ts": None, "until_ts": 0.0}

    while True:
        loop_start = time.perf_counter()

        latest_packets, packet_counts = recv_latest_packets(
            car_sock,
            args.drop_stale_frames,
        )
        dropped_image_packets += max(packet_counts.get(b"I", 0) - 1, 0)

        poll_console_input(manual_state, args)

        # Esta version no usa LIDAR para velocidad ni trayectoria.
        if b"L" in latest_packets:
            pass

        if b"B" in latest_packets:
            decoded_battery = decode_payload(latest_packets[b"B"][0])
            auto_utils.set_battery_level(decoded_battery)

        image_packet = latest_packets.get(b"I")
        if image_packet is None:
            continue

        data, car_address = image_packet
        decoded_image = decode_payload(data)
        img = cv2.imdecode(decoded_image, 1)
        if img is None:
            continue

        frame_counter += 1

        ai_start = time.perf_counter()
        try:
            ai_response = ai_processor.process_frame(img)
        except Exception as error:
            print(f"[CORE LOCAL] Error en inferencia local: {error}")
            ai_response = empty_ai_response()
        ai_elapsed_ms = (time.perf_counter() - ai_start) * 1000.0
        ai_ms_acc += ai_elapsed_ms

        ok = ai_response["ok"]
        if ok:
            last_detections = ai_response["detections"]
            last_raw_detections = ai_response["raw_detections"]
            last_decision = ai_response["decision"]
            last_inference_ms = ai_response["inference_ms"]
            last_raw_detection_count = ai_response["raw_detection_count"]
            last_used_detection_count = ai_response["used_detection_count"]
            last_rejected_detection_count = ai_response["rejected_detection_count"]
            last_suppressed_stop_count = ai_response["suppressed_stop_count"]
            rejected_detection_acc += last_rejected_detection_count
            suppressed_stop_acc += last_suppressed_stop_count
            ai_ok_count += 1
        else:
            ai_error_count += 1
            # Seguridad: si falla la IA local, no mantenemos un STOP antiguo infinito.
            last_decision = empty_decision()

        if ok:
            maybe_log_detection(
                last_detections,
                last_decision,
                top_k=args.log_top_k,
                min_interval=max(args.log_every_seconds, 0.0),
                state=log_state,
            )

        control_mode = last_decision.control_mode if last_decision.control_mode else 0
        control_giro = straight_without_road_steering(
            time.time(),
            straight_calibration_state,
            args.steering_calibration,
            args.straight_calibration_every_seconds,
            args.straight_calibration_pulse_seconds,
        )
        control_acelerador = artemis_base_throttle(auto_utils)
        trayectory_not_found = 1
        force_manual_control = bool(getattr(last_decision, "force_manual_control", False))

        if force_manual_control:
            straight_calibration_state["next_ts"] = None
            straight_calibration_state["until_ts"] = 0.0

            if control_mode in (1, 3):
                control_giro, _ = controls_without_road(
                    control_mode,
                    control_acelerador,
                    args.steering_calibration,
                )
                control_giro = apply_steering_gain(
                    control_giro,
                    args.steering_gain,
                    args.steering_calibration,
                )
            elif control_mode == 2:
                control_giro = 0.0

            if last_decision.throttle_cap is not None:
                control_acelerador = clamp(last_decision.throttle_cap)
        else:
            if control_mode in (1, 3):
                control_giro, control_acelerador = controls_without_road(
                    control_mode,
                    control_acelerador,
                    args.steering_calibration,
                )
                control_giro = apply_steering_gain(
                    control_giro,
                    args.steering_gain,
                    args.steering_calibration,
                )
                straight_calibration_state["next_ts"] = None
                straight_calibration_state["until_ts"] = 0.0

            if last_decision.throttle_cap is not None:
                control_acelerador = min(control_acelerador, last_decision.throttle_cap)

        if last_decision.stop:
            control_acelerador = 0

        decay_manual_overrides(manual_state, args.manual_hold_seconds)
        control_giro, control_acelerador = apply_manual_overrides(
            control_giro,
            control_acelerador,
            manual_state,
        )

        send_control(car_sock, control_giro, control_acelerador, car_address)

        if args.show_inference:
            img_overlay = img.copy()

            try:
                draw_detections(img_overlay, last_detections, last_decision)
            except Exception as error:
                print(f"[CORE LOCAL] No se pudo dibujar detecciones: {error}")

            manual_text = (
                "Manual: "
                f"giro {manual_state['steering_override']:+.2f} "
                if manual_state["steering_override"] is not None
                else "Manual: giro AUTO "
            )
            manual_text += (
                f"acel {manual_state['throttle_override']:+.2f}"
                if manual_state["throttle_override"] is not None
                else "acel AUTO"
            )

            cv2.putText(
                img_overlay,
                manual_text,
                (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                img_overlay,
                f"IA local: {'OK' if ok else 'ERROR'} | IA {last_inference_ms:.1f} ms",
                (10, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                img_overlay,
                (
                    f"Det usadas: {last_used_detection_count} / crudas: {last_raw_detection_count} | "
                    f"filtradas: {last_rejected_detection_count} | "
                    f"STOP ignorados: {last_suppressed_stop_count}"
                ),
                (10, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if trayectory_not_found:
                cv2.putText(
                    img_overlay,
                    "Recto sin trayectoria",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("CORE YOLO local", img_overlay)
            poll_window_input(manual_state, args)

        loop_elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
        loop_ms_acc += loop_elapsed_ms

        if args.stats_every and frame_counter % args.stats_every == 0:
            avg_loop_ms = loop_ms_acc / args.stats_every
            avg_ai_ms = ai_ms_acc / args.stats_every
            approx_fps = 1000.0 / avg_loop_ms if avg_loop_ms > 0 else 0.0

            print(
                "[STATS CORE LOCAL] "
                f"frame={frame_counter} "
                f"loop_avg_ms={avg_loop_ms:.1f} "
                f"ai_avg_ms={avg_ai_ms:.1f} "
                f"fps_aprox={approx_fps:.1f} "
                f"ai_ok={ai_ok_count} "
                f"ai_error={ai_error_count} "
                f"frames_descartados={dropped_image_packets} "
                f"detecciones_filtradas={rejected_detection_acc} "
                f"stops_ignorados={suppressed_stop_acc}"
            )

            loop_ms_acc = 0.0
            ai_ms_acc = 0.0
            dropped_image_packets = 0
            ai_ok_count = 0
            ai_error_count = 0
            rejected_detection_acc = 0
            suppressed_stop_acc = 0


if __name__ == "__main__":
    main()
