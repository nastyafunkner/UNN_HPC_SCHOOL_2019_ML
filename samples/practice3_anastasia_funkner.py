import sys
import cv2
import argparse
import logging as log
import time
import numpy as np

sys.path.append('../src')
from ie_detector import InferenceEngineDetector


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', help='input image path', type=str)
    parser.add_argument('-v', '--video_path', help='input video path', type=str)

    return parser


MODELS = {
    'FP16': [r'C:\public\mobilenet-ssd\FP16\mobilenet-ssd.bin',
             r'C:\public\mobilenet-ssd\FP16\mobilenet-ssd.xml',
             'CPU',
             r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll',
             20,
             r'C:\UNN_HPC_SCHOOL_2019_ML\src\names_classes'],
    'FP32': [r'C:\public\mobilenet-ssd\FP32\mobilenet-ssd.bin',
             r'C:\public\mobilenet-ssd\FP32\mobilenet-ssd.xml',
             'CPU',
             r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll',
             20,
             r'C:\UNN_HPC_SCHOOL_2019_ML\src\names_classes'],
    'I8': [r'C:\public\mobilenet-ssd\FP32\mobilenet-ssd_i8.bin',
           r'C:\public\mobilenet-ssd\FP32\mobilenet-ssd_i8.xml',
           'CPU',
           r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll',
           20,
           r'C:\UNN_HPC_SCHOOL_2019_ML\src\names_classes']
}


def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO,
                    stream=sys.stdout)

    args = build_argparse().parse_args()

    # if args.image_path is not None:
    #     image_path = args.image_path
    #
    #     image = cv2.imread(image_path)
    #
    #     for detector_param in MODELS:
    #         my_detector = InferenceEngineDetector(*detector_param)
    #         cv2.imshow('Frame', my_detector.detect(image))
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    if args.video_path is not None:

        for name_model, detector_param in MODELS.items():
            my_detector = InferenceEngineDetector(*detector_param)
            times = []
            t0_total = time.time()
            cap = cv2.VideoCapture(args.video_path)

            if not cap.isOpened():
                print("Error opening video stream or file")

            # Read until video is completed
            while cap.isOpened():
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret:
                    # print(len(times), end='...')
                    t0 = time.time()
                    my_detector.detect(frame)
                    t1 = time.time()
                    # times.append(t1 - t0)
                    times.append(1)

                else:
                    log.info('ret False')
                    break

            cap.release()

            t1_total = time.time()
            # latency = np.median(times)
            fps = len(times) / (t1_total - t0_total)
            print(name_model)
            # print(times)
            # print('latency', latency, sep=' : ')
            print('fps', fps, sep=' : ')

    return


if __name__ == '__main__':
    sys.exit(main())
