import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from collections import Counter
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def run_detection(image_path, output_path, conf_threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    queries = "car . white car . person . motorcycle . tree canopy . garbage bag ."

    def callback(image_slice: np.ndarray) -> sv.Detections:
        pil_img = Image.fromarray(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, text=queries, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=conf_threshold,
            text_threshold=0.25,
            target_sizes=[pil_img.size[::-1]]
        )[0]

        display_classes = ["car", "white car", "person", "motorbike", "tree", "trash"]

        # Mapping dictionary
        label_map = {name: i for i, name in enumerate(display_classes)}

        # Convert string labels to integer IDs
        # If DINO returns a string not in our list, we default to 0
        int_labels = np.array([label_map.get(l, 0) for l in results["labels"]])

        # Create the Detections object manually
        return sv.Detections(
            xyxy=results["boxes"].cpu().numpy(),
            confidence=results["scores"].cpu().numpy(),
            class_id=int_labels
        )

    slicer = sv.InferenceSlicer(callback=callback, slice_wh=(512, 512), overlap_ratio_wh=(0.2, 0.2))

    image = cv2.imread(image_path)
    if image is None: return

    detections = slicer(image).with_nms(threshold=0.4, class_agnostic=True)

    # Annotation
    annotated_image = sv.BoundingBoxAnnotator().annotate(scene=image.copy(), detections=detections)
    annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

    cv2.imwrite(output_path, annotated_image)
    print(f"DINO Result saved to: {output_path}")