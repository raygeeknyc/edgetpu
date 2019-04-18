# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import time

import numpy as np
import picamera

import edgetpu.classification.engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    with open(args.label, 'r', encoding="utf-8") as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = edgetpu.classification.engine.ClassificationEngine(args.model)

    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 30
        _, width, height, channels = engine.get_input_tensor_shape()
        try:
            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream,
                                                 format='rgb',
                                                 use_video_port=True,
                                                 resize=(width, height)):
                stream.truncate()
                stream.seek(0)
                frame = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                start_ms = time.time()
                results = engine.ClassifyWithInputTensor(frame, top_k=1)
                elapsed_ms = time.time() - start_ms
                if results:
                    logging.info("frame has {} objects".format(len(results)))
                    for detected in results:
                        logging.info("label: {}".format(labels[detected[0], detected[1]))
        finally:
            logging.info("done capturing")


if __name__ == '__main__':
    main()
