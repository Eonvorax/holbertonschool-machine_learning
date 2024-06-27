#!/usr/bin/env python3
"""
This module contains the Yolo class implementing the YOLO algorithm
"""

import os
from glob import iglob
from tensorflow import keras as K
import numpy as np
import cv2


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

    def iou(self, box1, boxes):
        """
        Calculates the Intersection Over Union (IoU) between a box and an
        array of boxes.

        Parameters:
        - box1: a numpy.ndarray of shape (4,) representing the first box
        - boxes: a numpy.ndarray of shape (?, 4) representing the other boxes

        Returns:
        - iou_scores: a numpy.ndarray of shape (?) containing the IoU scores
        """
        x1, y1, x2, y2 = box1
        box1_area = (x2 - x1) * (y2 - y1)

        # Extract dimensions for all other boxes to compare
        x1s = boxes[:, 0]
        y1s = boxes[:, 1]
        x2s = boxes[:, 2]
        y2s = boxes[:, 3]

        boxes_area = (x2s - x1s) * (y2s - y1s)

        inter_x1 = np.maximum(x1, x1s)
        inter_y1 = np.maximum(y1, y1s)
        inter_x2 = np.minimum(x2, x2s)
        inter_y2 = np.minimum(y2, y2s)

        inter_area = np.maximum(inter_x2 - inter_x1, 0) * \
            np.maximum(inter_y2 - inter_y1, 0)
        union_area = box1_area + boxes_area - inter_area

        iou_scores = inter_area / union_area
        return iou_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) to filter the bounding boxes.

        Parameters:
        - filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes
        - box_classes: a numpy.ndarray of shape (?,) containing the class
            number for each box in filtered_boxes
        - box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes
        - iou_threshold: a float representing the Intersection Over Union
            (IoU) threshold for NMS

        Returns:
        - box_predictions: a numpy.ndarray of shape (?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
        - predicted_box_classes: a numpy.ndarray of shape (?,) containing the
            class number for box_predictions ordered by class and box score
        - predicted_box_scores: a numpy.ndarray of shape (?) containing the
            box scores for box_predictions ordered by class and box score
        """
        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            # Sort the boxes by their unique class
            cls_indices = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            # Sort the boxes by their scores (in descending order)
            sorted_indices = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            while len(cls_boxes) > 0:
                # Select the box with the highest score
                box = cls_boxes[0]
                score = cls_scores[0]

                box_predictions.append(box)
                predicted_box_classes.append(cls)
                predicted_box_scores.append(score)

                # If this was the last box, no need to keep going
                if len(cls_boxes) == 1:
                    break

                # Calculate IoU between the selected box and the rest
                ious = self.iou(box, cls_boxes[1:])
                # Select boxes with IoU lower than the threshold
                remaining_indices = np.where(ious < self.nms_t)[0]

                # Exclude the box we just added to the output
                cls_boxes = cls_boxes[1:][remaining_indices]
                cls_scores = cls_scores[1:][remaining_indices]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a specified folder.

        Parameters:
        - folder_path: a string representing the path to the folder holding
            all the images to load

        Returns:
        - images: a list of images as numpy.ndarrays
        - image_paths: a list of paths to the individual images in images
        """
        image_paths = []
        images = []
        # Iterator over .jpg image files
        for path in iglob(os.path.join(folder_path, '*.jpg')):
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales images for the Darknet model.

        Parameters:
        - images: a list of images as numpy.ndarrays

        Returns:
        - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
            containing all of the preprocessed images
        - image_shapes: a numpy.ndarray of shape (ni, 2) containing the
            original height and width of the images
        """
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for img in images:
            # Resize image with inter-cubic interpolation
            resized_img = cv2.resize(
                img, (input_h, input_w), interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values from [0, 255] to [0, 1]
            pimages.append(resized_img / 255.0)

            # Add image shape to shapes array
            orig_h, orig_w = img.shape[:2]
            image_shapes.append([orig_h, orig_w])

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes
