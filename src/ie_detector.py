import cv2
import numpy as np

from matplotlib import cm, colors
from openvino.inference_engine import IENetwork, IECore


def renormalize(n, range2):
    delta2 = range2[1] - range2[0]
    return int(delta2 * n + range2[0])


def renormalize_coordinates(point, scale_len):
    return (renormalize(point[0], (0, scale_len[1])),
            renormalize(point[1], (0, scale_len[0])))


def write_conf(conf, point, img, shift=20):
    new_point = point[0] + shift, point[1] + shift
    cv2.putText(img, 'conf=' + str(round(conf * 100, 2)), new_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                2, cv2.LINE_AA)
    return img


class InferenceEngineDetector:
    def __init__(self, weightsPath=None, configPath=None,
                 device='CPU', extension=None, class_num=10,
                 class_names_path=None,
                 color_map=cm.Set3):
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

        self.class_names_dict = dict(enumerate([" "] * class_num))

        if class_names_path is not None:
            self.class_names_dict = dict(enumerate(open(class_names_path).readlines()))
            class_num = len(self.class_names_dict)

        self.color_dict = dict(enumerate(map(lambda nums: tuple([int(c * 255) for c in nums]),
                                             color_map(np.linspace(0, 1, class_num)))))

        return

    def write_class(self, class_num, point, img, shift=20):
        new_point = point[0] + shift, point[1] + shift
        cv2.putText(img, self.class_names_dict[class_num].strip(), new_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_dict[class_num],
                    1, cv2.LINE_AA)
        return img

    def draw_detection(self, detections, img):
        for det in detections:
            if any(det[1:]):
                image_id, label, conf, *init_coors = det
                # print(init_coors)
                point_1 = renormalize_coordinates(init_coors[:2], img.shape)
                point_2 = renormalize_coordinates(init_coors[2:], img.shape)
                # print(point_1, point_2)
                cv2.rectangle(img, point_1, point_2, self.color_dict[label], 1)
                self.write_class(label, point_1, img)

        return img

    @staticmethod
    def _prepare_image(image, h, w):
        image = cv2.resize(image, (h, w))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    def detect(self, image, return_detection=True):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape

        blob = self._prepare_image(image, h, w)

        output = self.exec_net.infer(inputs={input_blob: blob})

        output = output[out_blob]

        # cv2.imshow("Detections", self.draw_detection(output[0][0], image))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # !!!!
        if return_detection:
            return output
        else:
            return self.draw_detection(output[0][0], image)
