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
    draw_detections,
    load_actions,
)


MAX_PACKET_SIZE = 99999
DEFAULT_ACTIONS_CONFIG = Path(__file__).resolve().with_name("actions_config_rada_tpii_complete.json")


# Envia al vehiculo un paquete de control UDP con el formato que espera ARTEMIS:
# un byte de tipo ('C') seguido de dos dobles con giro y aceleracion.
def send_control(sock, control_giro, control_acelerador, address):
    payload = (
        struct.pack("c", bytes("C", "ascii"))
        + struct.pack("d", round(control_giro, 3))
        + struct.pack("d", round(control_acelerador, 3))
    )
    sock.sendto(payload, address)


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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Servidor UDP para Deep Racer con baja latencia, acciones completas y logs de deteccion."
    )
    parser.add_argument("--server-ip", default="10.0.128.177")
    parser.add_argument("--server-port", type=int, default=20001)
    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--show-inference", action="store_true")
    parser.add_argument("--route", default="2,3,2,3,2,2,2,2,0")
    parser.add_argument("--steering-calibration", type=float, default=0.0)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default=None, help="Ejemplos: cpu, 0, 0,1")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--decision-hold-seconds", type=float, default=1.5)
    parser.add_argument("--inference-every", type=int, default=1)
    parser.add_argument("--actions-config", default=str(DEFAULT_ACTIONS_CONFIG))
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

    server_address = (args.server_ip, args.server_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(server_address)

    print(f"Escuchando en UDP {server_address[0]}:{server_address[1]}")
    print(f"Ruta por defecto: {route}")
    print(f"Modelo YOLO: {Path(detector.model_path)}")
    print(f"Config de acciones: {args.actions_config}")
    print(f"Frames obsoletos descartados: {args.drop_stale_frames}")
    print(f"Acciones configuradas: {sorted(actions.keys())}")

    frame_counter = 0
    last_detections = []
    dropped_image_packets = 0
    inference_ms_acc = 0.0
    loop_ms_acc = 0.0
    log_state = {"last_key": None, "last_ts": 0.0}

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
                _, last_detections = detector.infer(img)
                inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000.0
                inference_ms_acc += inference_elapsed_ms
            except Exception as error:
                print(f"[WARN] Error ejecutando inferencia: {error}")
                last_detections = []

        decision = decision_engine.decide(last_detections)
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

        if decision.throttle_cap is not None:
            control_acelerador = min(control_acelerador, decision.throttle_cap)

        if decision.stop:
            control_acelerador = 0

        send_control(sock, control_giro, control_acelerador, address)

        if args.show_inference:
            img_overlay = img.copy()
            draw_detections(img_overlay, last_detections, decision)
            if trayectory_not_found:
                cv2.putText(
                    img_overlay,
                    "Trayectoria no encontrada",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
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
                f"frames_descartados={dropped_image_packets}"
            )
            inference_ms_acc = 0.0
            loop_ms_acc = 0.0
            dropped_image_packets = 0


if __name__ == "__main__":
    main()
