import sys
import cv2
import logging as log
import argparse
import os

sys.path.append('..\\src')

from imagefilter import ImageFilter


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help='input image path', type=str)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Hello image filtering")
    args = build_argparse().parse_args()

    image_path = args.input_path
    log.info(args.input_path)

    image = cv2.imread(image_path)
    my_filter = ImageFilter(grey=True, shape=None, crop=True)

    cv2.imshow("Image1", image)
    cv2.imshow("Image2", my_filter.process_image(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    sys.exit(main())
