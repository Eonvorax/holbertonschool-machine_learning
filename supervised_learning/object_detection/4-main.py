#!/usr/bin/env python3

if __name__ == '__main__':
    import cv2
    import numpy as np
    Yolo = __import__('4-yolo').Yolo

    np.random.seed(2)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images('yolo_images/yolo/')
    image_paths, images = zip(*sorted(zip(image_paths, images)))
    i = np.random.randint(0, len(images))
    print(i)
    print(image_paths[i])
    cv2.imshow(image_paths[i], images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
