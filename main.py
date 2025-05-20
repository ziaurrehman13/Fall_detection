from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import math
from ultralytics import YOLO

app = FastAPI()

# Load YOLOv11 pose estimation pretrained model once
model = YOLO('yolo11n-pose.pt')  # Replace with your model path

# COCO keypoint connections for skeleton (optional, for client visualization)
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def is_fall(keypoints, bbox):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    width = x2 - x1
    aspect_ratio = height / (width + 1e-5)

    if aspect_ratio < 1.0:
        return True

    y_coords = keypoints[:, 1]
    y_var = np.var(y_coords / (height + 1e-5))

    if y_var < 0.0015:
        return True

    return False

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Resize to 640x480 for consistent processing
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, verbose=False)

    detections = []

    for info in results:
        boxes = info.boxes
        keypoints_all = info.keypoints.xy.cpu().numpy() if info.keypoints is not None else []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            # Person class id is 0 in COCO
            if conf > 0.65 and cls_id == 0:
                kpts = keypoints_all[i] if i < len(keypoints_all) else None

                fall = False
                keypoints_list = []
                if kpts is not None:
                    fall = is_fall(kpts, (x1, y1, x2, y2))
                    keypoints_list = [{"x": float(kp[0]), "y": float(kp[1])} for kp in kpts]

                detections.append({
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": conf,
                    "label": f"Person {int(conf*100)}%",
                    "fall_detected": fall,
                    "keypoints": keypoints_list,
                    "connections": connections  # Optional: send for client skeleton drawing
                })

    return {"detections": detections}

# Add at the end of your main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
