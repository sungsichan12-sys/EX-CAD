# SPDX-License-Identifier: GPL-2.0-or-later
"""Exercise both TRELLIS queue layouts in the server's Gradio environment, outside Blender."""

import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "excad_addon"))
from ai_3d_provider import download_model


def verify(directory, combined):
    portrait = directory / "input.png"
    Image.new("RGB", (32, 24), (120, 140, 155)).save(portrait)
    payload = json.dumps({"asset": {"version": "2.0"}}).encode()
    payload += b" " * (-len(payload) % 4)
    model = directory / "result.glb"
    model.write_bytes(b"glTF" + struct.pack("<III", 2, len(payload) + 20, len(payload)) + b"JSON" + payload)
    events = []

    def start(req: gr.Request):
        events.append(("start", req.session_hash))

    def preprocess(image, req: gr.Request):
        assert image.size == (32, 24)
        events.append(("preprocess", req.session_hash))
        return image

    def generate(
        image, gallery, multi, seed, shape_guidance, shape_steps, latent_guidance, latent_steps, mode, req: gr.Request
    ):
        assert image.size == (32, 24) and not gallery and not multi
        assert [seed, shape_guidance, shape_steps, latent_guidance, latent_steps, mode] == [
            0,
            7.5,
            12,
            3.0,
            12,
            "stochastic",
        ]
        events.append(("generate", req.session_hash))
        return {"session": req.session_hash}

    def extract(state, simplify, resolution, req: gr.Request):
        assert state["session"] == req.session_hash and simplify == 0.95 and resolution == 1024
        events.append(("extract", req.session_hash))
        return str(model)

    def generate_glb(image, gallery, multi, seed, sg, ss, lg, ls, mode, simplify, resolution, req: gr.Request):
        state = generate(image, gallery, multi, seed, sg, ss, lg, ls, mode, req)
        output = extract(state, simplify, resolution, req)
        return state, output, output

    with gr.Blocks() as server:
        action = gr.Button()
        photo, gallery = gr.Image(type="pil"), gr.Gallery()
        multi, state = gr.State(False), gr.State()
        numbers = [gr.Number() for _ in range(5)]
        mode, file = gr.Textbox(), gr.File()
        action.click(start, [], [], api_name="start_session")
        action.click(preprocess, [photo], [photo], api_name="preprocess_image")
        generation = [photo, gallery, multi, *numbers, mode]
        if combined:
            action.click(generate_glb, [*generation, *numbers[:2]], [state, file, gr.DownloadButton()],
                         api_name="generate_and_extract_glb")
        else:
            action.click(generate, generation, state, api_name="image_to_3d")
            action.click(extract, [state, *numbers[:2]], file, api_name="extract_glb")
        server.queue()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    try:
        server.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            quiet=True,
            show_error=True,
            allowed_paths=[str(directory)],
        )
        os.environ["TRELLIS_URL"] = f"http://127.0.0.1:{port}"
        output = download_model(provider="TRELLIS", image_path=portrait, timeout=40)
        try:
            assert output.read_bytes() == model.read_bytes()
        finally:
            output.unlink()
        assert [name for name, _ in events] == ["start", "preprocess", "generate", "extract"]
        assert len({session for _, session in events}) == 1
    finally:
        server.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for combined in (False, True):
            verify(Path(tmp), combined)
    print("TRELLIS TRANSPORT PASS", gr.__version__)
