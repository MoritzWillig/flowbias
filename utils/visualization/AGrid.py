import numpy as np
from PIL import Image, ImageFont, ImageDraw


class AGrid:

    def __init__(self, grid, image_shape, padding=5, text_height=50, title_height=70, text_params=None, title_params=None):
        self._padding = padding
        self._text_height = text_height
        self._title_height = title_height
        self.text_params = {} if text_params is None else text_params
        self.title_params = {} if title_params is None else title_params

        self._cell_shape = list(image_shape).copy()
        self._cell_shape[0] += self._title_height + self._text_height + (2 * padding)
        self._cell_shape[1] += (2 * padding)

        self._a = np.full((grid[1]*self._cell_shape[0], grid[0]*self._cell_shape[1], 3), 1.0)

    def _put(self, xx, yy, image):
        if len(image.shape) != 3:
            if len(image.shape) == 2:
                image = np.repeat(image[:,:,np.newaxis], 3, axis=2)
        else:
            if len(image.shape) == 3:
                if image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)

        # print(">>", image.shape, self._a.shape, xx, yy)
        self._a[
            yy:yy + image.shape[0],
            xx:xx + image.shape[1],
            :
        ] = image

    def _cell_pos(self, x, y):
        """
        returns the upper left position of a cell (excluding padding or text)
        :param x:
        :param y:
        :return:
        """
        yy = self._title_height + y * self._cell_shape[0]
        xx = x * self._cell_shape[1]
        return xx, yy

    def place(self, x, y, image, label=None):
        xx, yy = self._cell_pos(x, y)
        yy += self._padding + self._text_height
        xx += self._padding
        self._put(xx, yy, image)

        if label is not None:
            self.label(x,y,label)

    def label(self, x, y, text):
        if self._text_height == 0:
            return

        xx, yy = self._cell_pos(x, y)
        yy += self._padding
        xx += self._padding

        text_im = Image.new("F", (self._cell_shape[1] - (2 * self._padding), self._text_height))
        ImageDraw.Draw(text_im).text((0, 0), text, **self.text_params)
        self._put(xx, yy, np.asarray(1-np.asarray(text_im)))

    def title(self, title):
        if self._title_height == 0:
            return

        text_im = Image.new("F", (self._a.shape[1], self._title_height))
        ImageDraw.Draw(text_im).text((0, 0), title, **self.title_params)
        self._put(0, 0, np.asarray(1 - np.asarray(text_im)))

    def get_image(self):
        return self._a
