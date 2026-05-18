"""
CORE - DeepRacer/ARTEMIS + Jetson IA separada.

Este script aplica la idea del codigo local giro_sin_trayectoria.2.py, pero
separando la IA en la Jetson:

  COCHE  -> CORE:    paquetes UDP I/L/B
  CORE   -> JETSON:  reenvia paquete I original
  JETSON -> CORE:    decision IA + detecciones
  CORE   -> COCHE:   paquete C con giro/acelerador

El CORE conserva:
  - artemis_autonomous_car para trayectoria y LIDAR,
  - giro sin carretera segun control_mode IA,
  - ganancia de giro,
  - throttle_cap/STOP,
  - control manual por teclado opcional.
"""

import argparse
import pickle
import socket
import struct
import time
from types import SimpleNamespace

import cv2

try:
    import msvcrt
except ImportError:
    msvcrt = None

import artemis_autonomous_car


MAX_PACKET_SIZE = 99999

# Valores de giro usados cuando la trayectoria no se encuentra o cuando la IA
# mantiene una orden de direccion. Son los mismos conceptos del codigo local.
MANUAL_LEFT_STEERING = 0.45
MANUAL_RIGHT_STEERING = -0.45
MANUAL_STRAIGHT_STEERING = 0.0


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


def controls_without_road(control_mode, base_throttle, steering_calibration):
    """
    Control de emergencia si artemis no encuentra trayectoria.
    control_mode: 1=izquierda, 2=recto, 3=derecha.
    """
    if control_mode == 1:
        return MANUAL_LEFT_STEERING + steering_calibration, base_throttle
    if control_mode == 3:
        return MANUAL_RIGHT_STEERING + steering_calibration, base_throttle
    return MANUAL_STRAIGHT_STEERING + steering_calibration, base_throttle


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


# -------------------------------------------------------------------------
# Respuesta Jetson -> objetos simples en el CORE
# -------------------------------------------------------------------------
def detection_from_dict(item):
    center_x = item.get("center_x", None)
    center_y = item.get("center_y", None)
    width = item.get("width", None)
    height = item.get("height", None)
    box = item.get("box", [0, 0, 0, 0])

    if len(box) < 4:
        box = [0, 0, 0, 0]

    return SimpleNamespace(
        label=item.get("label", ""),
        confidence=float(item.get("confidence", 0.0) or 0.0),
        center_x=float(center_x) if center_x is not None else None,
        center_y=float(center_y) if center_y is not None else None,
        width=float(width) if width is not None else None,
        height=float(height) if height is not None else None,
        x1=float(box[0]),
        y1=float(box[1]),
        x2=float(box[2]),
        y2=float(box[3]),
        area=float(item.get("area", 0.0) or 0.0),
        speed_px_s=(
            float(item.get("speed_px_s"))
            if item.get("speed_px_s", None) is not None
            else None
        ),
    )


def decision_from_dict(item):
    return SimpleNamespace(
        stop=bool(item.get("stop", False)),
        control_mode=int(item.get("control_mode", 0) or 0),
        throttle_cap=item.get("throttle_cap", None),
        source_label=item.get("source_label", ""),
        source_confidence=float(item.get("source_confidence", 0.0) or 0.0),
    )


def empty_decision():
    return SimpleNamespace(
        stop=False,
        control_mode=0,
        throttle_cap=None,
        source_label="",
        source_confidence=0.0,
    )


def ask_jetson_for_decision(jetson_sock, jetson_address, image_packet_data, timeout):
    """Reenvia a Jetson el paquete I original recibido del coche."""
    jetson_sock.settimeout(timeout)

    try:
        jetson_sock.sendto(image_packet_data, jetson_address)
        response_data, _ = jetson_sock.recvfrom(MAX_PACKET_SIZE)
        response = pickle.loads(response_data, encoding="latin1")

        detections = [
            detection_from_dict(item)
            for item in response.get("detections", [])
        ]
        raw_detections = [
            detection_from_dict(item)
            for item in response.get("raw_detections", [])
        ]
        decision = decision_from_dict(response.get("decision", {}))

        return {
            "detections": detections,
            "raw_detections": raw_detections,
            "decision": decision,
            "inference_ms": float(response.get("inference_ms", 0.0) or 0.0),
            "raw_detection_count": int(response.get("raw_detection_count", len(raw_detections)) or 0),
            "used_detection_count": int(response.get("used_detection_count", len(detections)) or 0),
            "rejected_detection_count": int(response.get("rejected_detection_count", 0) or 0),
            "suppressed_stop_count": int(response.get("suppressed_stop_count", 0) or 0),
            # Compatibilidad con nombres anteriores.
            "rejected_stop_count": int(response.get("rejected_stop_count", 0) or 0),
            "ok": True,
        }

    except socket.timeout:
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

    except Exception as error:
        print(f"[CORE] Error hablando con Jetson: {error}")
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

    return ", ".join(
        f"{detection.label} ({detection.confidence:.2f})"
        for detection in detections[: max(top_k, 1)]
    )


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
        print(f"[DETECCION JETSON] {summary} | decision: {decision_text}")
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
        description="CORE DeepRacer/ARTEMIS con IA remota en Jetson."
    )

    parser.add_argument("--server-ip", default="10.0.128.177")
    parser.add_argument("--server-port", type=int, default=20001)

    parser.add_argument("--jetson-ip", default="192.168.50.2")
    parser.add_argument("--jetson-port", type=int, default=21000)
    parser.add_argument("--jetson-timeout", type=float, default=0.25)

    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--show-inference", action="store_true")

    parser.add_argument("--route", default="2,2,2,2,2,2,2,2,2,2,0")
    parser.add_argument("--steering-calibration", type=float, default=0.19)
    parser.add_argument(
        "--steering-gain",
        type=float,
        default=1.25,
        help="Multiplica el giro calculado. Valores >1 cierran mas el giro.",
    )

    parser.add_argument("--drop-stale-frames", dest="drop_stale_frames", action="store_true")
    parser.add_argument("--keep-queued-frames", dest="drop_stale_frames", action="store_false")
    parser.set_defaults(drop_stale_frames=True)

    parser.add_argument("--stats-every", type=int, default=60)
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

    server_address = (args.server_ip, args.server_port)

    car_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    car_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    car_sock.bind(server_address)

    jetson_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    jetson_address = (args.jetson_ip, args.jetson_port)

    print(f"[CORE SPLIT] Escuchando coche UDP {args.server_ip}:{args.server_port}")
    print(f"[CORE SPLIT] Jetson IA UDP {args.jetson_ip}:{args.jetson_port}")
    print(f"[CORE SPLIT] Ruta: {route}")
    print(f"[CORE SPLIT] Ganancia de giro: {args.steering_gain:.2f}")
    print(f"[CORE SPLIT] Compensacion direccion: {args.steering_calibration:.2f}")
    print(f"[CORE SPLIT] Drop stale frames: {args.drop_stale_frames}")
    print(
        "[CORE SPLIT] Avance sin carretera: "
        f"recto={MANUAL_STRAIGHT_STEERING:.2f} "
        f"izquierda={MANUAL_LEFT_STEERING:.2f} "
        f"derecha={MANUAL_RIGHT_STEERING:.2f}"
    )
    print("[CORE SPLIT] Manual: w delante, s atras, a izquierda, d derecha, 2 rapido, x atras rapido.")
    print("[CORE SPLIT] En Linux usa --show-inference para capturar teclado desde la ventana OpenCV.")

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
    jetson_ok_count = 0
    jetson_timeout_count = 0
    loop_ms_acc = 0.0
    jetson_ms_acc = 0.0
    rejected_detection_acc = 0
    suppressed_stop_acc = 0

    log_state = {"last_key": None, "last_ts": 0.0}
    manual_state = build_manual_state()

    while True:
        loop_start = time.perf_counter()

        latest_packets, packet_counts = recv_latest_packets(
            car_sock,
            args.drop_stale_frames,
        )
        dropped_image_packets += max(packet_counts.get(b"I", 0) - 1, 0)

        poll_console_input(manual_state, args)

        if b"L" in latest_packets:
            decoded_lidar = decode_payload(latest_packets[b"L"][0])
            auto_utils.proceso_lidar(decoded_lidar, False)

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

        jetson_start = time.perf_counter()
        jetson_response = ask_jetson_for_decision(
            jetson_sock=jetson_sock,
            jetson_address=jetson_address,
            image_packet_data=data,
            timeout=args.jetson_timeout,
        )
        jetson_elapsed_ms = (time.perf_counter() - jetson_start) * 1000.0
        jetson_ms_acc += jetson_elapsed_ms

        ok = jetson_response["ok"]
        if ok:
            last_detections = jetson_response["detections"]
            last_raw_detections = jetson_response["raw_detections"]
            last_decision = jetson_response["decision"]
            last_inference_ms = jetson_response["inference_ms"]
            last_raw_detection_count = jetson_response["raw_detection_count"]
            last_used_detection_count = jetson_response["used_detection_count"]
            last_rejected_detection_count = jetson_response["rejected_detection_count"]
            last_suppressed_stop_count = jetson_response["suppressed_stop_count"]
            rejected_detection_acc += last_rejected_detection_count
            suppressed_stop_acc += last_suppressed_stop_count
            jetson_ok_count += 1
        else:
            jetson_timeout_count += 1
            # Seguridad: si Jetson no responde, no mantenemos un STOP antiguo infinito,
            # pero si quieres mantener ultima decision, comenta la linea siguiente.
            last_decision = empty_decision()

        maybe_log_detection(
            last_detections,
            last_decision,
            top_k=args.log_top_k,
            min_interval=max(args.log_every_seconds, 0.0),
            state=log_state,
        )

        auto_utils.set_stop(1 if last_decision.stop else 0)
        control_mode = last_decision.control_mode if last_decision.control_mode else 0

        manual_turn_active = (
            not last_decision.stop
            and control_mode in (1, 3)
        )

        if manual_turn_active:
            control_giro, control_acelerador = controls_without_road(
                control_mode,
                auto_utils.lidar_throttle_control,
                args.steering_calibration,
            )
            trayectory_not_found = 1
        else:
            control_giro, control_acelerador, trayectory_not_found = auto_utils.proceso_fotograma(
                img,
                args.show_info,
                control_mode,
            )

            if trayectory_not_found:
                control_giro, control_acelerador = controls_without_road(
                    control_mode,
                    auto_utils.lidar_throttle_control,
                    args.steering_calibration,
                )

        control_giro = clamp(control_giro * args.steering_gain)

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
                print(f"[CORE SPLIT] No se pudo dibujar detecciones: {error}")

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
                f"Jetson: {'OK' if ok else 'TIMEOUT'} | IA {last_inference_ms:.1f} ms",
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
                    "Trayectoria no encontrada / giro por IA",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("CORE + Jetson YOLO Split", img_overlay)
            poll_window_input(manual_state, args)

        loop_elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
        loop_ms_acc += loop_elapsed_ms

        if args.stats_every and frame_counter % args.stats_every == 0:
            avg_loop_ms = loop_ms_acc / args.stats_every
            avg_jetson_ms = jetson_ms_acc / args.stats_every
            approx_fps = 1000.0 / avg_loop_ms if avg_loop_ms > 0 else 0.0

            print(
                "[STATS CORE SPLIT] "
                f"frame={frame_counter} "
                f"loop_avg_ms={avg_loop_ms:.1f} "
                f"jetson_avg_ms={avg_jetson_ms:.1f} "
                f"fps_aprox={approx_fps:.1f} "
                f"jetson_ok={jetson_ok_count} "
                f"jetson_timeout={jetson_timeout_count} "
                f"frames_descartados={dropped_image_packets} "
                f"detecciones_filtradas={rejected_detection_acc} "
                f"stops_ignorados={suppressed_stop_acc}"
            )

            loop_ms_acc = 0.0
            jetson_ms_acc = 0.0
            dropped_image_packets = 0
            jetson_ok_count = 0
            jetson_timeout_count = 0
            rejected_detection_acc = 0
            suppressed_stop_acc = 0


if __name__ == "__main__":
    main()
