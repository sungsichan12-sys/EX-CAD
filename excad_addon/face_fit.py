# SPDX-License-Identifier: GPL-2.0-or-later
"""Measure frontal landmarks locally and apply non-accumulating, bilateral MPFB targets."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen

import bpy
from mathutils import Vector

from .human_engine import _import_mpfb

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def cache_dir():
    cache = os.getenv("EXCAD_CACHE") or bpy.utils.user_resource("DATAFILES", path="excad", create=True)
    path = Path(cache)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_download(url, dest, digest):
    def valid(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == digest

    if dest.is_file() and valid(dest):
        return dest
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
        temp = Path(out.name)
    try:
        with urlopen(url, timeout=45) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out)
        if not valid(temp):
            raise ValueError(f"Checksum mismatch for {dest.name}")
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)
    return dest


def _python(cache):
    configured = os.getenv("EXCAD_FACE_PYTHON")
    if configured:
        return configured
    runtime = cache / "face-runtime"
    exe = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = runtime / "mediapipe-1.2.3.ready"
    if not marker.is_file() or not exe.is_file():
        host = ["py", "-3.12"] if os.name == "nt" and shutil.which("py") else ["python3.12"]
        _run(host + ["-m", "venv", str(runtime)], 90)
        _run([str(exe), "-m", "pip", "install", "--disable-pip-version-check", "mediapipe==1.2.3"], 300)
        marker.touch()
    return str(exe)


def _run(args, timeout):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Facial landmark runtime timed out after {timeout} seconds") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"Facial landmark runtime exited with status {result.returncode}")
    return result.stdout


def measure_image(image_path, cache):
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(image)
    model = os.getenv("EXCAD_FACE_MODEL") or cached_download(MODEL_URL, cache / "face_landmarker.task", MODEL_SHA)
    output = _run([_python(cache), str(Path(__file__).with_name("face_landmarks.py")), str(image), str(model)], 90)
    return measure([Vector(p) for p in json.loads(output)])


def measure(p):
    def distance(a, b):
        return (p[a] - p[b]).length

    width, height = distance(234, 454), distance(10, 152)
    if min(width, height) < 20:
        raise ValueError("Face is too small for reliable fitting")
    left, right = (p[33] + p[133]) * 0.5, (p[263] + p[362]) * 0.5
    if abs(distance(234, 1) - distance(454, 1)) / width > 0.23:
        raise ValueError("Use a more frontal view for symmetric fitting")
    return {
        "aspect": height / width,
        "jaw_angle": math.degrees((p[172] - p[152]).angle(p[397] - p[152])),
        "cheek_width": width / height,
        "eye_spacing": (left - right).length / width,
        "eye_size": (distance(33, 133) + distance(263, 362)) / (2 * width),
        "nose_width": distance(98, 327) / width,
        "nose_length": distance(6, 2) / height,
        "mouth_width": distance(61, 291) / width,
    }


def basemesh_services(obj):
    mpfb = _import_mpfb()
    if not mpfb:
        raise RuntimeError("Enable the MPFB extension before fitting a human")
    from bl_ext.blender_org.mpfb.services.objectservice import ObjectService

    if obj is None or not ObjectService.object_is_basemesh(obj):
        raise ValueError("Facial targets require an MPFB basemesh")
    return mpfb


def set_targets(obj, values, channel):
    human, targets, _, location = basemesh_services(obj)
    root = Path(location.get_mpfb_data("targets"))
    paths = {name: root / (name + ".target.gz") for name in values}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    previous = json.loads(obj.get(channel, "{}"))
    keys = obj.data.shape_keys
    blocks = keys.key_blocks if keys else {}
    saved = {name: blocks[name].value for name in set(values) | set(previous) if name in blocks}
    created = []
    try:
        for name in previous.keys() - values.keys():
            if name in blocks:
                blocks[name].value = 0
        for name, weight in values.items():
            if name in blocks:
                blocks[name].value = weight
            else:
                created.append(targets.load_target(obj, str(paths[name]), weight=weight, name=name))
        human.refit(obj)
        from .wardrobe import refit_eyes, refit_fringe

        refit_eyes(obj)
        refit_fringe(obj)
        obj["excad_character"] = True
        obj[channel] = json.dumps(values)
    except Exception:
        for key in created:
            obj.shape_key_remove(key)
        for name, value in saved.items():
            obj.data.shape_keys.key_blocks[name].value = value
        raise


def fit_face_from_image(obj, image_path):
    return apply_measurements(obj, measure_image(bpy.path.abspath(image_path), cache_dir()))


def apply_measurements(obj, ratios):
    mapping = (
        ("aspect", 1.31, 0.20, "head/head-scale-vert"),
        ("jaw_angle", 102, 21, "chin/chin-width"),
        ("cheek_width", 0.77, 0.12, "cheek/{side}-cheek-bones"),
        ("eye_spacing", 0.44, 0.08, "eyes/{side}-eye-trans"),
        ("eye_size", 0.20, 0.06, "eyes/{side}-eye-scale"),
        ("nose_width", 0.255, 0.07, "nose/nose-scale-horiz"),
        ("nose_length", 0.22, 0.08, "nose/nose-scale-vert"),
        ("mouth_width", 0.36, 0.09, "mouth/mouth-scale-horiz"),
    )
    values = {}
    for metric, neutral, span, target in mapping:
        delta = (ratios[metric] - neutral) / span
        direction = ("out" if delta >= 0 else "in") if metric == "eye_spacing" else ("incr" if delta >= 0 else "decr")
        for side in ("l", "r") if "{side}" in target else ("",):
            values[target.format(side=side) + "-" + direction] = round(0.15 + 0.35 * min(1.0, abs(delta)), 4)
    set_targets(obj, values, "excad_face_fit")
    return {"object": obj.name, "measurements": {k: round(v, 4) for k, v in ratios.items()}, "weights": values}
