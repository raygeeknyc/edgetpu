#!/bin/bash
time python3 /usr/local/lib/python3.5/dist-packages/edgetpu/demo/object_detection.py --model ./resources/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite --input ./resources/face.jpg --output ~/detection_results.jpg
