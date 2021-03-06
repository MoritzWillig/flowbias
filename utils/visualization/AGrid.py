import numpy as np
from PIL import Image, ImageFont, ImageDraw
import imageio
import matplotlib.cm

class AGrid:

    def __init__(self, grid, image_shape, padding=5, text_height=50, title_height=70, text_params=None, title_params=None):
        if len(image_shape) != 2:
            raise ValueError(f"expected a two dimensional image shape, but got dim {len(image_shape)}")
        self._padding = padding
        self._text_height = text_height
        self._title_height = title_height
        self.text_params = {} if text_params is None else text_params
        self.title_params = {} if title_params is None else title_params

        self._cell_shape = list(image_shape).copy()
        self._cell_shape[0] += self._text_height + (2 * padding)
        self._cell_shape[1] += (2 * padding)

        self._grid_cols = grid[0]
        self._grid_rows = grid[1]

        self._a = np.full((self._title_height + grid[1]*self._cell_shape[0], grid[0]*self._cell_shape[1], 3), 1.0)

    def _put(self, xx, yy, image, colormap=None):
        if colormap is not None:
            if (len(image.shape) == 2) or ((len(image.shape) == 3) and (image.shape[2] == 1)):
                mappable = matplotlib.cm.ScalarMappable(cmap=colormap)
                image = mappable.to_rgba(image)[:, :, :3]
            else:
                raise ValueError("color map is only allowed for 1 dim images")

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
        if x < 0:
            x = self._grid_cols + x + 1
        if y < 0:
            y = self._grid_rows + y + 1

        yy = self._title_height + y * self._cell_shape[0]
        xx = x * self._cell_shape[1]
        return xx, yy

    def place(self, x, y, image, label=None, colormap=None):
        xx, yy = self._cell_pos(x, y)
        yy += self._padding + self._text_height
        xx += self._padding
        self._put(xx, yy, image, colormap=colormap)

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

    def save_image(self, file_path):
        imageio.imwrite(file_path, self.get_image())
