#!/usr/bin/env python3
"""
Filter Boxes
"""

from tensorflow import keras as K
import numpy as np


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

    def sigmoid(self, x):
        """A simple sigmoid method"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes outputs from the YOLO model and returns the bounding boxes,
        box confidences, and class probabilities for each detected object.

        Parameters:
        - outputs: a list of numpy.ndarrays containing predictions from YOLO
        - image_size: a numpy.ndarray containing the original size of the
            image [image_height, image_width]

        Returns:
        - boxes: a list of numpy.ndarrays containing the processed boundary
            boxes for each output
        - box_confidences: a list of numpy.ndarrays containing the box
            confidences for each output
        - box_class_probs: a list of numpy.ndarrays containing the class
            probabilities for each output
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]

            # Extract box parameters (output coordinates)
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # grid cells coordinates for width and height
            c_x, c_y = np.meshgrid(np.arange(grid_width),
                                   np.arange(grid_height))

            # Add axis to match dimensions of t_x & t_y
            c_x = np.expand_dims(c_x, axis=-1)
            c_y = np.expand_dims(c_y, axis=-1)

            # Calculate bounding box coordinates
            # NOTE apply sigmoid activation and offset by grid cell location
            # then normalize by grid dimensions
            bx = (self.sigmoid(t_x) + c_x) / grid_width
            by = (self.sigmoid(t_y) + c_y) / grid_height
            # NOTE apply exponential and scale by anchor dimensions
            bw = (np.exp(t_w) * self.anchors[i,
                  :, 0]) / self.model.input.shape[1]
            bh = (np.exp(t_h) * self.anchors[i,
                  :, 1]) / self.model.input.shape[2]

            # Convert to original image scale
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # Stack coordinates to form final box coordinates
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            # Extract sigmoid-normalized box confidence and class prob.
            # NOTE 4:5 instead of 4 to preserve last dimension
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the boxes based on box confidences and class probabilities.

        Parameters:
        - boxes: list of numpy.ndarrays containing the processed boundary
            boxes for each output
        - box_confidences: list of numpy.ndarrays containing the box
            confidences for each output
        - box_class_probs: list of numpy.ndarrays containing the class
            probabilities for each output
        - box_class_threshold: float representing the threshold for class
            scores

        Returns:
        - filtered_boxes: a numpy.ndarray containing filtered bounding boxes
        - box_classes: a numpy.ndarray containing the class number that each
            box in filtered_boxes predicts
        - box_scores: a numpy.ndarray containing the box scores for each box
            in filtered_boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_confidence, box_class_prob in zip(boxes,
                                                       box_confidences,
                                                       box_class_probs):
            # Calc. box scores from confidences and class probabilities
            box_score = box_confidence * box_class_prob

            # Find the class (index) with the maximum score for each box
            box_class = np.argmax(box_score, axis=-1)

            # Keep only the highest score for each box
            box_score = np.max(box_score, axis=-1)

            # Create a mask for boxes with a score over the threshold
            # NOTE equivalent to np.where()
            filter_mask = box_score >= self.class_t

            # Filter each list using the mask
            filtered_boxes.append(box[filter_mask])
            box_classes.append(box_class[filter_mask])
            box_scores.append(box_score[filter_mask])

        # Turn the resulting lists into numpy arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
