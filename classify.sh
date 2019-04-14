#!/bin/bash
time python3 /usr/local/lib/python3.5/dist-packages/edgetpu/demo/classify_image.py --model ./resources/mobilenet_v2_1.0_224_quant_edgetpu.tflite --label ./resources/imagenet_labels.txt --image ./resources/parrot.jpg
