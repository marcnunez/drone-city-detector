import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOWorld
from collections import Counter


def is_white_car(car_crop):
    if car_crop.size == 0: return False
    hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 195])
    upper_white = np.array([180, 45, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return (np.sum(mask > 0) / mask.size) > 0.35


def run_detection(image_path, output_path, conf_threshold):
    # Use absolute path for Docker
    model = YOLOWorld('/app/yolov8l-worldv2.pt')
    prompt_list = ["car", "person", "motorcyclist", "tree canopy", "garbage bag"]
    display_classes = ["car", "person", "motorbike", "tree", "trash", "white car"]
    model.set_classes(prompt_list)

    def callback(image_slice):
        results = model.predict(image_slice, conf=conf_threshold, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=(0.3, 0.3)
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: {image_path} not found")
        return

    detections = slicer(image).with_nms(threshold=0.3, class_agnostic=True)

    # Color Verification Logic
    final_class_ids = []
    for i, class_id in enumerate(detections.class_id):
        if prompt_list[class_id] == "car":
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if is_white_car(image[y1:y2, x1:x2]):
                final_class_ids.append(5)
            else:
                final_class_ids.append(0)
        else:
            final_class_ids.append(class_id)

    detections.class_id = np.array(final_class_ids)

    # Annotation and Saving
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4)
    labels = [f"{display_classes[cid]} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]

    annotated_scene = label_annotator.annotate(
        scene=box_annotator.annotate(scene=image.copy(), detections=detections),
        detections=detections, labels=labels
    )

    cv2.imwrite(output_path, annotated_scene)
    print(f"YOLO Result saved to: {output_path}")