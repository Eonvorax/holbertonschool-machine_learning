#!/usr/bin/env python3
"""
Initialize Yolo
"""

from tensorflow import keras as K


class Yolo:
    """
    This class implements the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class.

        Args:
            model_path (str): path to the YOLO model.
            classes_path (str): path to the file containing the class names.
            class_t (float): box score threshold for the initial filtering
                step.
            nms_t (float): The IOU threshold for non-max suppression.
            anchors (numpy.ndarray): The anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
