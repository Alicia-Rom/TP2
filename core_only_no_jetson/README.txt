Version CORE-only, sin Jetson
=============================

Esta carpeta contiene una copia independiente de los ficheros necesarios para
ejecutar el coche desde el CORE haciendo la inferencia YOLO en local.

Ficheros principales:
- core_deepracer_core.py
- local_ai_pipeline.py
- ai_inference_engine.py
- artemis_autonomous_car.py
- actions_config_rada_tpii_complete.json
- best.pt

Ejecucion recomendada:

python core_deepracer_core_only_RectoSinTrayectoria.py ^
  --server-ip 172.16.0.1 ^
  --server-port 20001 ^
  --show-inference ^
  --steering-calibration 0.12 ^
  --straight-calibration-every-seconds 1 ^
  --straight-calibration-pulse-seconds 0.43

python3 core_deepracer.py   --server-ip 172.16.0.1   --server-port 20001  --show-inference --steering-calibration 0.12 --straight-calibration-every-seconds 1 --straight-calibration-pulse-seconds 0.43


Notas:
- Ya no se usa jetson_ai_server_v5.1.py.
- Ya no existen parametros --jetson-ip, --jetson-port ni --jetson-timeout.
- Por defecto la IA local usa CPU: --device cpu.
- Si el CORE tiene GPU/CUDA disponible, puedes probar --device auto o --device 0.
- El modelo por defecto es best.pt y la configuracion por defecto es
  actions_config_rada_tpii_complete.json, ambos copiados en esta carpeta.
- Dependencias esperadas: opencv-python, numpy, imutils, ultralytics y torch.
