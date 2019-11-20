import sys
import cv2
import argparse
import logging as log

sys.path.append('../src')
from ie_detector import InferenceEngineDetector


def build_argparse():
    parser = argparse.ArgumentParser()

    #
    # Add your code here
    #

    return parser


def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO,
                    stream=sys.stdout)
    log.info("Hello object detection!")
    args = build_argparse().parse_args()

    #
    # Add your code here
    #

    return


if __name__ == '__main__':
    sys.exit(main())
