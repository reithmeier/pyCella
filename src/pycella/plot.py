"""
Helpers to plot results using openCV
"""

from typing import Optional, List

import cv2
import numpy as np


class CvGridWindow:
    """
    Window to display a matrix as a grid using OpenCV
    """

    def __init__(
        self,
        name: str,
        height: int = 480,
        width: int = 640,
        color_map: Optional[dict[int, tuple[int, int, int]]] = None,
    ) -> None:
        self.name = name
        self.color_map = color_map
        # create window
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, width, height)

    def show(self, matrix: np.ndarray) -> None:
        """
        update the window with the values of a 2d matrix

        :param matrix: 2d matrix
        """
        # Create a colored image
        colored_image = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)

        for val, color in self.color_map.items():
            colored_image[matrix == val] = color

        cv2.imshow(self.name, colored_image)


class CvHistogramWindow:
    """
    Window to display a histogram of an 1d array using openCV
    """

    def __init__(
        self, name: str, height: int = 480, width: int = 640, bins: int = 64
    ) -> None:
        self.name = name
        self.height = height
        self.width = width
        self.bins = bins

        # create window
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, width, height)

    def show(self, array: np.ndarray):
        """
        update the window with an 1d array

        :param array: 1d array
        """
        if len(array) == 0:
            return

        data = np.array(array)
        hist, bin_edges = np.histogram(
            data, bins=self.bins, range=(0, np.max(data))
        )  # range covers min and max
        hist = hist.astype(np.float32)

        hist_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Normalize histogram to fit the image height
        hist = cv2.normalize(hist, hist, 0, self.height, cv2.NORM_MINMAX)

        # Width of each bin in pixels
        bin_width = int(self.width / self.bins)

        for i in range(1, self.bins):
            cv2.line(
                hist_img,
                (bin_width * (i - 1), self.height - int(hist[i - 1])),
                (bin_width * i, self.height - int(hist[i])),
                (255, 255, 255),
                2,  # White color, thickness 2
            )

        # Show the histogram
        cv2.imshow(self.name, hist_img)


class CvLinePlotWindow:
    """
    Window to display a histogram of an 1d array using openCV
    """

    def __init__(self, name: str, height: int = 480, width: int = 640) -> None:
        self.name = name
        self.height = height
        self.width = width
        self.margin = 5

        # create window
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, width, height)

    def show(self, series: List):
        # Create white canvas
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for data, color in series:
            self.draw_line(img, data, color)

        cv2.imshow(self.name, img)

    def draw_line(self, img, data, color):
        if len(data) < 2:
            return

        # Normalize data to fit graph height
        y_max = max(data)
        y_min = min(data)
        y_range = y_max - y_min if y_max != y_min else 1

        y_values = [
            self.height - int((val - y_min) / y_range * self.height) for val in data
        ]

        # step width in x-axis
        x_interval = self.width // (len(data) - 1)
        points = [(i * x_interval, y) for i, y in enumerate(y_values)]

        # Draw lines between points
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, 2)
