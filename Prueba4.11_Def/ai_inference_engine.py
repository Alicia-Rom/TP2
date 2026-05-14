import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depende del entorno local
    YOLO = None


DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("best.pt")

# Acciones por defecto que relacionan una clase detectada con una respuesta
# de alto nivel del vehiculo. Incluyen tanto nombres genericos como las clases
# en espanol del dataset actual para que el sistema funcione sin config extra.
DEFAULT_ACTIONS = {
    "stop": {"type": "stop", "cooldown": 2.5},
    "stop_sign": {"type": "stop", "cooldown": 2.5},
    "slow": {"type": "slow", "throttle_cap": 0.20},
    "slow_down": {"type": "slow", "throttle_cap": 0.20},
    "speed_limit": {"type": "slow", "throttle_cap": 0.25},
    "left": {"type": "direction", "control": 1},
    "turn_left": {"type": "direction", "control": 1},
    "straight": {"type": "direction", "control": 2},
    "forward": {"type": "direction", "control": 2},
    "right": {"type": "direction", "control": 3},
    "turn_right": {"type": "direction", "control": 3},
    "obligatorio_girar_izquierda": {"type": "direction", "control": 1, "cooldown": 1.5},
    "obligatorio_continuar_recto": {"type": "direction", "control": 2, "cooldown": 1.5},
    "obligatorio_girar_derecha": {"type": "direction", "control": 3, "cooldown": 1.5},
    "obras": {"type": "slow", "throttle_cap": 0.18},
    "peligro_obras": {"type": "slow", "throttle_cap": 0.18},
    "cono": {"type": "slow", "throttle_cap": 0.16},
    "valla": {"type": "slow", "throttle_cap": 0.16},
    "ninos": {"type": "slow", "throttle_cap": 0.16},
    "semaforos": {"type": "slow", "throttle_cap": 0.16},
    "bajada_peligrosa": {"type": "slow", "throttle_cap": 0.18},
    "estrechamiento_calzada_derecha": {"type": "slow", "throttle_cap": 0.18},
    "otros_peligros": {"type": "slow", "throttle_cap": 0.18},
    "peligro_trex": {"type": "slow", "throttle_cap": 0.18},
    "velocidad_max_30": {"type": "slow", "throttle_cap": 0.15},
    "velocidad_max_90": {"type": "slow", "throttle_cap": 0.30},
}


# Representa una deteccion individual devuelta por el modelo local.
@dataclass
class Detection:
    label: str
    confidence: float
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


# Representa la decision final que tomara el sistema tras analizar todas las
# detecciones de una imagen.
@dataclass
class Decision:
    control_mode: int = 0
    stop: bool = False
    throttle_cap: Optional[float] = None
    source_label: Optional[str] = None
    source_confidence: float = 0.0


# Encapsula la inferencia local con Ultralytics sobre best.pt.
# Su trabajo consiste en recibir un frame, ejecutarlo sobre YOLO y convertir
# el resultado a una lista uniforme de Detection para el resto del programa.
class LocalYOLOObjectDetector:
    def __init__(
        self,
        model_path: str,
        min_confidence: float = 0.5,
        device: Optional[str] = None,
        image_size: int = 640,
        iou_threshold: float = 0.7,
        max_detections: int = 50,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "Ultralytics no esta instalado. Instala la dependencia con 'pip install ultralytics'."
            )

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontro el modelo YOLO: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.min_confidence = min_confidence
        self.device = device
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    # Ejecuta la inferencia sobre un frame con el modelo local.
    def infer(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.min_confidence,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )
        raw_result = results[0]
        return raw_result, self._collect_detections(raw_result)

    # Convierte la salida propia de Ultralytics a Detection.
    def _collect_detections(self, result):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names or getattr(self.model, "names", {})
        xywh = boxes.xywh.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()

        detections = []
        for box, confidence, class_id in zip(xywh, confidences, class_ids):
            detections.append(
                Detection(
                    label=self._resolve_label(names, int(class_id)),
                    confidence=float(confidence),
                    center_x=float(box[0]),
                    center_y=float(box[1]),
                    width=float(box[2]),
                    height=float(box[3]),
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    @staticmethod
    def _resolve_label(names, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)


# Alias de compatibilidad para el servidor existente.
WorkflowObjectDetector = LocalYOLOObjectDetector


# Traduce una lista de detecciones en una accion concreta del vehiculo.
# La idea es separar claramente la "vision" de la "toma de decisiones".
class ObjectDecisionEngine:
    def __init__(self, actions: Dict[str, dict], decision_hold_seconds: float = 1.5) -> None:
        self.actions = {key.lower(): value for key, value in actions.items()}
        self.decision_hold_seconds = decision_hold_seconds
        self.stop_until = 0.0
        self.direction_until = 0.0
        self.direction_control = 0

    # Recorre las detecciones en orden y escoge la primera que tenga una accion
    # configurada. A partir de ella construye la Decision final.
    def decide(self, detections):
        now = time.time()
        decision = Decision()

        for detection in detections:
            action = self._resolve_action(detection.label)
            if action is None:
                continue

            action_type = action.get("type")
            if action_type == "stop":
                cooldown = float(action.get("cooldown", self.decision_hold_seconds))
                self.stop_until = max(self.stop_until, now + cooldown)
                decision.stop = True
                decision.source_label = detection.label
                decision.source_confidence = detection.confidence
                break

            if action_type == "direction":
                self.direction_control = int(action.get("control", 0))
                self.direction_until = now + float(action.get("cooldown", self.decision_hold_seconds))
                decision.control_mode = self.direction_control
                decision.source_label = detection.label
                decision.source_confidence = detection.confidence
                break

            if action_type == "slow":
                throttle_cap = float(action.get("throttle_cap", 0.20))
                decision.throttle_cap = throttle_cap
                decision.source_label = detection.label
                decision.source_confidence = detection.confidence
                break

        if now < self.stop_until:
            decision.stop = True

        if now < self.direction_until and self.direction_control:
            decision.control_mode = self.direction_control

        return decision

    # Busca la accion asociada a una clase detectada.
    # Tambien contempla pequenas variaciones de nombre sustituyendo espacios
    # y guiones por guiones bajos.
    def _resolve_action(self, label: str):
        normalized = label.strip().lower()
        if normalized in self.actions:
            return self.actions[normalized]

        normalized = normalized.replace(" ", "_").replace("-", "_")
        return self.actions.get(normalized)


# Carga un mapa de acciones desde JSON. Si no se pasa fichero, usa la
# configuracion por defecto definida en este mismo modulo.
def load_actions(actions_config_path: Optional[str]):
    if not actions_config_path:
        return DEFAULT_ACTIONS

    with open(actions_config_path, "r", encoding="utf-8") as file:
        loaded_actions = json.load(file)

    if not isinstance(loaded_actions, dict):
        raise ValueError("El fichero de acciones debe contener un objeto JSON")

    return loaded_actions


# Dibuja en una imagen las detecciones encontradas y la decision tomada.
# Esta funcion es solo para depuracion visual; no modifica el resultado
# de la inferencia ni la logica de decision.
def draw_detections(frame, detections, decision):
    for index, detection in enumerate(detections[:5]):
        if None not in (detection.center_x, detection.center_y, detection.width, detection.height):
            x1 = int(detection.center_x - detection.width / 2)
            y1 = int(detection.center_y - detection.height / 2)
            x2 = int(detection.center_x + detection.width / 2)
            y2 = int(detection.center_y + detection.height / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{detection.label}: {detection.confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        y_pos = 25 + index * 22
        text = f"{detection.label}: {detection.confidence:.2f}"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if decision.source_label:
        decision_text = f"Decision: {decision.source_label} ({decision.source_confidence:.2f})"
        cv2.putText(frame, decision_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


# Define los argumentos del modo de prueba por imagen.
# Este script puede ejecutarse de forma aislada para comprobar que la IA
# detecta bien y devuelve la decision esperada.
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Prueba la inferencia local con YOLO11 y la toma de decisiones sobre una imagen."
    )
    parser.add_argument("image_path")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default=None, help="Ejemplos: cpu, 0, 0,1")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--decision-hold-seconds", type=float, default=1.5)
    parser.add_argument("--actions-config")
    parser.add_argument("--show", action="store_true")
    return parser


# Punto de entrada del modo de prueba.
# Carga una imagen desde disco, la procesa con el modelo local y muestra tanto
# las detecciones como la decision resultante.
def main():
    args = build_arg_parser().parse_args()
    frame = cv2.imread(args.image_path)
    if frame is None:
        raise FileNotFoundError(f"No se ha podido leer la imagen: {args.image_path}")

    detector = LocalYOLOObjectDetector(
        model_path=args.model_path,
        min_confidence=args.min_confidence,
        device=args.device,
        image_size=args.image_size,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    decision_engine = ObjectDecisionEngine(
        actions=load_actions(args.actions_config),
        decision_hold_seconds=args.decision_hold_seconds,
    )

    _, detections = detector.infer(frame)
    decision = decision_engine.decide(detections)

    print("Modelo:", detector.model_path)
    print("Detecciones:")
    for detection in detections:
        print(f"- {detection.label}: {detection.confidence:.3f}")

    print("Decision:")
    print(
        json.dumps(
            {
                "control_mode": decision.control_mode,
                "stop": decision.stop,
                "throttle_cap": decision.throttle_cap,
                "source_label": decision.source_label,
                "source_confidence": decision.source_confidence,
            },
            indent=2,
        )
    )

    if args.show:
        frame_overlay = frame.copy()
        draw_detections(frame_overlay, detections, decision)
        cv2.imshow("AI inference", frame_overlay)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
