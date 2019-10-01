import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImageSplit:

    def read_mark(self, path, size):
        img = Image.open(path)
        width, height = img.size
        coordinate = []
        for i in range(width):
            for j in range(height):
                r, g, b = img.getpixel((i, j))
                if 220 <= r <= 240 and 20 <= g <= 30 and 25 <= b <= 30:
                    pixel = [int(i/size), int(j/size)]
                    coordinate.append(pixel)
        return np.unique(np.array(coordinate), axis=0)

    def write_mark(self, path, coordinate,  size):
        img = Image.open(path)
        for i in range(coordinate.shape[0]):
            x, y = coordinate[i][0] * size, coordinate[i][1] * size
            for a in range(32):
                for b in range(32):
                    img.putpixel((x+a, y+b), (255, 25, 25))
        img.save(path)
        img.show()


if __name__ == '__main__':
    path1 = 'aa.jpg'
    path2 = 'a1.jpg'
    image = ImageSplit()
    coordinate = image.read_mark(path1, 64)
    image.write_mark(path2, coordinate, 64)
