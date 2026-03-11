import cv2
import argparse
import supervision as sv
from ultralytics import YOLOWorld
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="SkyScout-Zero: High-Recall Drone Object Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="high_res_summary.jpg", help="Output path")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    args = parser.parse_args()

    # Initialize YOLO-World
    model = YOLOWorld('yolov8l-worldv2.pt')
    classes = ["car", "white car", "person", "motorbike", "tree", "trash"]
    model.set_classes(classes)

    # Define Slicing Logic for 1904x1071 Resolution
    # This ensures we don't downsample and lose small objects
    def callback(image_slice):
        results = model.predict(image_slice, conf=args.conf, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),  # Standard YOLO window
        overlap_ratio_wh=(0.2, 0.2),  # 20% overlap to handle objects cut by tiles
        thread_workers=2  # Parallel processing for efficiency
    )

    # Execute Detection
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    print(f"Processing image at native resolution: {image.shape[1]}x{image.shape[0]}...")
    detections = slicer(image)

    # Global NMS (Non-Maximum Suppression)
    # Slicing can create duplicate detections at the edges of tiles
    detections = detections.with_nms(threshold=0.5)

    # Annotation
    # Using a modern thickness that scales with high-res imagery
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels = [
        f"{classes[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Save and Summary
    cv2.imwrite(args.output, annotated_image)

    # Generate terminal summary
    class_counts = Counter([classes[class_id] for class_id in detections.class_id])

    print("\n" + "=" * 30)
    print(" DETECTION SUMMARY")
    print("=" * 30)
    if not class_counts:
        print("No objects detected.")
    for obj, count in class_counts.items():
        print(f"{obj:15} : {count}")
    print("=" * 30)
    print(f"Output saved to: {args.output}\n")


if __name__ == "__main__":
    main()