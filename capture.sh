#!/bin/bash
# output only goes to HDMI not VNC
time python3 /usr/local/lib/python3.5/dist-packages/edgetpu/demo/classify_capture.py --model ./resources/mobilenet_v2_1.0_224_quant_edgetpu.tflite --label ./resources/imagenet_labels.txt
