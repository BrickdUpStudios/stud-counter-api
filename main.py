from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image
import base64

app = FastAPI()

# Allow frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_studs(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    np_image = np.array(pil_image)

    # Convert to grayscale
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles (studs)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=30
    )

    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0])
        for (x, y, r) in circles[0, :]:
            cv2.circle(np_image, (x, y), r, (0, 255, 0), 3)

    # Encode annotated image to base64
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "count": count,
        "annotated_image": base64_image
    })
