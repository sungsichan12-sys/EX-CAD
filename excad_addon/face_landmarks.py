# SPDX-License-Identifier: GPL-2.0-or-later
"""Isolated MediaPipe entry point. Only landmark coordinates leave this local process."""

import json
import sys


def detect(image_path, model):
    import mediapipe as mp
    vision = mp.tasks.vision
    image = mp.Image.create_from_file(image_path)
    options = vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model), num_faces=2)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        faces = detector.detect(image).face_landmarks
    if len(faces) != 1:
        raise ValueError("Expected one clearly visible frontal face")
    return [[p.x*image.width, p.y*image.height] for p in faces[0]]


if __name__ == "__main__":
    print(json.dumps(detect(sys.argv[1], sys.argv[2]), separators=(",", ":")))
