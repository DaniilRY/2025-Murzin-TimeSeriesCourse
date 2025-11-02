import numpy as np
import pandas as pd
import math
import cv2
import imutils
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

from typing import Union

class Image2TimeSeries:
    """
    Converter from image to time series by angle-based method
        
    Parameters
    ----------
    angle_step: angle step for finding the contour points
    """
    
    def __init__(self, angle_step: int = 10) -> None:
        self.angle_step: int = angle_step

    # (остальные методы без изменений...)

    def _img_show(self, img, contour, edge_coordinates, center):
        """
        Корректная визуализация через matplotlib (совместима с Colab/Jupyter).
        Показывает контур, центр и лучи от центра к контуру.
        """

        # Нарисуем контур
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        # Центр
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

        # Линии от центра до краёв
        for pt in edge_coordinates:
            x, y = int(pt[0]), int(pt[1])
            cv2.line(img, (int(center[0]), int(center[1])), (x, y), (255, 0, 255), 2)

        # Показ изображения в правильных цветах
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(imutils.resize(img, width=400), cv2.COLOR_BGR2RGB))
        plt.title("Image with Contours and Rays")
        plt.axis("off")
        plt.show()

    def _img_preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the raw image: convert to grayscale, inverse, blur slightly, and threshold it
        
        Parameters
        ----------
        img: raw image
        
        Returns
        -------
        prep_img: image after preprocessing
        """

        # Инверсия изображения
        inverted_img = cv2.bitwise_not(img)
        
        # Размытие изображения
        blurred_img = cv2.GaussianBlur(inverted_img, (5, 5), 0)
        
        # Бинаризация изображения
        _, binary_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        eroded_img = cv2.erode(binary_img, kernel, iterations=1)
        dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)
        
        # Медианная фильтрация
        prep_img = cv2.medianBlur(dilated_img, 5)
        
        return prep_img


    def _get_contour(self, img: np.ndarray) -> np.ndarray:
        """
        Find the largest contour in the preprocessed image

        Parameters
        ----------
        img: preprocessed image
        
        Returns
        -------
        contour: object contour
        """

        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = [cnt for cnt in contours if cv2.contourArea(cnt) > 500][0]

        return contour


    def _get_center(self, contour: np.ndarray) -> tuple[float, float]:
        """
        Compute the object center

        Parameters
        ----------
        contour: object contour
        
        Returns
        -------
            coordinates of the object center
        """

        M = cv2.moments(contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        return (center_x, center_y)


    def _find_nearest_idx(self, array: np.ndarray, value: int) -> int:
        """
        Find index of element that is the nearest to the defined value

        Parameters
        ----------
        array: array of values
        value: defined value
     
        Returns
        -------
        idx: index of element that is the nearest to the defined value
        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        
        return idx


    def _get_coordinates_at_angle(self, contour: np.ndarray, center: tuple[float, float], angle: int) -> np.ndarray:
        """
        Find one point on contour that are located at the angle

        Parameters
        ----------
        contour: object contour
        center: object center
        angle: angle
     
        Returns
        -------
            coordinates of one point on the contour
        """

        angles = np.rad2deg(np.arctan2(*(center - contour).T))
        angles = np.where(angles < -90, angles + 450, angles + 90)
        found = np.rint(angles) == angle
        
        if np.any(found):
            return contour[found][0]
        else:
            idx = self._find_nearest_idx(angles, angle)
            return contour[idx]


    def _get_edge_coordinates(self, contour: np.ndarray, center: tuple[float, float]) -> list[np.ndarray]:
        """
        Find points on contour that are located from each other at the angle step

        Parameters
        ----------
        contour: object contour
        center: object center
     
        Returns
        -------
        edge_coordinates: coordinates of the object center
        """

        edge_coordinates = []
        for angle in range(0, 360, self.angle_step):
            pt = self._get_coordinates_at_angle(contour, center, angle)
            if np.any(pt):
                edge_coordinates.append(pt)

        return edge_coordinates

    def convert(self, img: np.ndarray, is_visualize: bool = False) -> np.ndarray:
        """
        Convert image to time series by angle-based method

        Parameters
        ----------
        img: input image
        is_visualize: visualize or not image with contours, center and rais from starting center
        
        Returns
        -------
        ts: time series representation
        """

        ts = []

        prep_img = self._img_preprocess(img)
        contour = self._get_contour(prep_img)
        center = self._get_center(contour)
        edge_coordinates = self._get_edge_coordinates(contour.reshape(-1, 2), center)

        if (is_visualize):
            self._img_show(img.copy(), contour, edge_coordinates, center)

        for coord in edge_coordinates:
            #dist = math.sqrt((coord[0] - center[0])**2 + (coord[1] - center[1])**2)
            dist = math.fabs(coord[0] - center[0]) + math.fabs(coord[1] - center[1])
            ts.append(dist)

        return np.array(ts)