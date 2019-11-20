import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

CLASS_NAMES = dict(enumerate(open(r'C:\UNN_HPC_SCHOOL_2019_ML\src\names_classes').readlines()))


def renormalize(n, range2):
    delta2 = range2[1] - range2[0]
    return int(delta2 * n + range2[0])


def renormalize_coordinates(point, scale_len):
    return (renormalize(point[0], (0, scale_len)),
            renormalize(point[1], (0, scale_len)))


def write_conf(conf, point, img, shift=20):
    new_point = point[0] + shift, point[1] + shift
    cv2.putText(img, 'conf=' + str(round(conf * 100, 2)), new_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                2, cv2.LINE_AA)
    return img


def write_class(class_num, point, img, shift=20):
    new_point = point[0] + shift, point[1] + shift
    cv2.putText(img, CLASS_NAMES[class_num].strip(), new_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                1, cv2.LINE_AA)
    return img


class InferenceEngineDetector:
    def __init__(self, weightsPath=None, configPath=None,
                 device='CPU', extension=None):
        """
        :param weightsPath: Путь до bin-файла модели.
        :param configPath:Путь до xml-файла модели.
        :param device: Тип устройства, на котором запускаемся (CPU или GPU).
        :param extension: Для CPU необходим путь до библиотеки со слоями,
        реализации которых нет в MKL-DNN
        (C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll).
        """

        self.ie = IECore()
        if device == 'CPU':
            self.ie.add_extension(extension, device)

        self.net = IENetwork(model=configPath, weights=weightsPath)

        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        return

    @staticmethod
    def draw_detection(detections, img):
        print(CLASS_NAMES)
        for det in detections:
            if any(det[1:]):
                image_id, label, conf, *init_coors = det
                point_1 = renormalize_coordinates(init_coors[:2], img.shape[0])
                point_2 = renormalize_coordinates(init_coors[2:], img.shape[1])
                cv2.rectangle(img, point_1, point_2, (0, 255, 0), 1)
                write_class(label, point_1, img)

        return img

    @staticmethod
    def _prepare_image(image, h, w):
        image = cv2.resize(image, (h, w))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape

        blob = self._prepare_image(image, h, w)

        output = self.exec_net.infer(inputs={input_blob: blob})

        output = output[out_blob]

        cv2.imshow("Detections", self.draw_detection(output[0][0], image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # !!!!
        detection = output

        return detection
