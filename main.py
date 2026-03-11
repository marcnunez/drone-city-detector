import argparse
import sys

def run_orchestrator():
    parser = argparse.ArgumentParser(description="Drone Detection Orchestrator")

    # 1. Model Choice
    parser.add_argument(
        "--model",
        choices=["yolo", "dino"],
        default="yolo",
        help="Choose which detection script to run"
    )

    # 2. Shared Parameters
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Custom output path (optional)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")

    args, unknown = parser.parse_known_args()

    # Route to the correct script
    if args.model == "yolo":
        print("Starting YOLO-World Detector...")
        import yolo
        # Use provided output or default to yolo specific result
        output_path = args.output if args.output else "results/output_yolo.jpg"
        yolo.run_detection(args.image, output_path, args.conf)

    elif args.model == "dino":
        print("Starting Grounding DINO Detector...")
        import dino
        output_path = args.output if args.output else "results/output_dino.jpg"
        dino.run_detection(args.image, output_path, args.conf)

if __name__ == "__main__":
    run_orchestrator()