import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Inference with Thermal Sentinel YOLO11")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to best.pt weights")
    parser.add_argument("--source", type=str, required=True, help="Path to thermal image or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=True)
    print("Inference completed! Results saved to runs/detect/predict/")

if __name__ == "__main__":
    main()
