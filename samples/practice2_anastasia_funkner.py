import sys
import cv2
import argparse
import logging as log

sys.path.append('../src')
from ie_detector import InferenceEngineDetector


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help='input image path', type=str)

    return parser


def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO,
                    stream=sys.stdout)
    log.info("Hello object detection!")
    args = build_argparse().parse_args()

    my_detector = InferenceEngineDetector(
        r'C:\public\mobilenet-ssd\FP32\mobilenet-ssd.bin',
        r'C:\public\mobilenet-ssd\FP32\mobilenet-ssd.xml',
        'CPU',
        r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll'
    )

    image_path = args.input_path
    log.info(args.input_path)

    image = cv2.imread(image_path)

    print(my_detector.detect(image))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return


if __name__ == '__main__':
    sys.exit(main())
