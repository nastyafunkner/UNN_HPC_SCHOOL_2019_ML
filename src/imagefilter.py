import cv2


class ImageFilter():
    def __init__(self, grey=False, shape=None, crop=None):
        self.grey = grey
        self.crop = crop

        if shape:
            self.shape = shape
        else:
            self.shape = None

    def process_image(self, image):

        if self.grey:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.shape is not None:
            image = cv2.resize(image, dsize=self.shape)

        if self.crop:
            two_size = image.shape[:2]
            min_size = min(two_size)
            new_slice = []

            for size in two_size:
                if size != min_size:
                    diff_half = (size - min_size) // 2
                    new_slice.append(slice(diff_half, diff_half + min_size, 1))
                else:
                    new_slice.append(slice(0, min_size, 1))

            image = image[tuple(new_slice)]

        return image
