import sys
import cv2
import argparse
import logging as log

sys.path.append('../src')
from ie_detector import InferenceEngineDetector


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', help='input image path', type=str)
    parser.add_argument('-v', '--video_path', help='input video path', type=str)

    return parser


def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO,
                    stream=sys.stdout)

    args = build_argparse().parse_args()

    my_detector = InferenceEngineDetector(
        r'C:\public\mobilenet-ssd\FP16\mobilenet-ssd.bin',
        r'C:\public\mobilenet-ssd\FP16\mobilenet-ssd.xml',
        'CPU',
        r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll',
        20,
        r'C:\UNN_HPC_SCHOOL_2019_ML\src\names_classes',

    )

    if args.image_path is not None:
        image_path = args.image_path

        image = cv2.imread(image_path)
        cv2.imshow('Frame', my_detector.detect(image))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.video_path is not None:
        log.info("Press 'Q' to exit!")
        cap = cv2.VideoCapture(args.video_path)
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")

        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:

                # Display the resulting frame
                cv2.imshow('Frame', my_detector.detect(frame))

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    sys.exit(main())
