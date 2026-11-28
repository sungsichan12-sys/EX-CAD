# SPDX-License-Identifier: GPL-2.0-or-later
"""Run with Blender --background --factory-startup --python-exit-code 1 --python FILE."""

import json
import threading
import time
from pathlib import Path

import bpy

from excad_addon.ai_bridge import AiBridgeServer
from excad_addon.client_compact import rpc

bridge = AiBridgeServer.instance()
run = bridge._dispatch_dict
schema = bridge._dispatch('{"cmd":"schema"}').encode("utf8")
assert len(schema) <= 1200, len(schema)
for topic in ("sketch", "cad", "character", "organic", "scene"):
    assert len(bridge._dispatch(json.dumps({"cmd": "schema", "topic": topic})).encode("utf8")) <= 1200
steps = [{"cmd": "ping"}, {"cmd": "absent"}, {"cmd": "list"}]
response = run({"cmd": "batch", "steps": steps, "return_all": True, "stop_on_error": False})
assert not response["ok"] and response["steps_run"] == 3 and response["last"]["ok"], response
assert run({"cmd": "batch", "steps": steps})["steps_run"] == 2
code = 'S XY; R 0 0 2 1; sketch_extrude {"depth":0.5,"name":"A; B"}; DIGEST'
result = run({"cmd": "dsl", "code": code})
assert result["ok"] and result["n"] == 4 and bpy.data.objects.get("A; B"), result
digest = run({"cmd": "scene_digest", "since_hash": result["last"]["h"]})
assert digest["unchanged"]

recipe_path = Path(__file__).resolve().parents[2] / "doc" / "excad" / "feature_tree.json"
recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
created = run({"cmd": "feature_tree", "name": "Housing", "recipe": recipe})
assert created["ok"], created
assert run({"cmd": "set_material", "object": "Housing", "preset": "BRUSHED_METAL"})["ok"]
changed = run(
    {
        "cmd": "rebuild_feature_tree",
        "object": "Housing",
        "parameters": {"pad.depth": 0.8, "sketches.plate.shapes.0.width": 3.4},
    }
)
assert changed["ok"], changed

# Repeatedly compare edits with a checkpoint. Keep that active baseline while
# evicting older one-off edits from the bounded digest cache.
housing = bpy.data.objects["Housing"]
base_x = housing.location.x
baseline = run({"cmd": "scene_digest"})["h"]
for i in range(22):
    housing.location.x = base_x + (i + 1) / 10
    delta = run({"cmd": "scene_digest", "since_hash": baseline})
    assert delta.get("delta") and any(r[0] == housing.name for r in delta["changed"]), (i, delta)
    if i == 0:
        first_edit = delta["h"]
    housing.location.x = base_x
    assert run({"cmd": "scene_digest", "since_hash": baseline}).get("unchanged")
assert len(bridge._digest_cache) == 16 and set(bridge._sketch_cache) == set(bridge._digest_cache)
assert "objs" in run({"cmd": "scene_digest", "since_hash": first_edit})

# A reconnecting client may already know the open scene while the bridge is new.
reconnected = AiBridgeServer()._dispatch_dict
assert reconnected({"cmd": "scene_digest", "since_hash": baseline}).get("unchanged")
housing.location.x = base_x + 1
assert reconnected({"cmd": "scene_digest", "since_hash": baseline}).get("delta")
housing.location.x = base_x

# Stop between receipt and the main-thread timer. A fresh listener must never run
# the previous request, even when the client is still waiting for its reply.
bridge.start(0)
pending = {}
thread = threading.Thread(
    target=lambda: pending.update(rpc({"cmd": "sketch_new", "sketch_id": "Stale"}, port=bridge.port, timeout=5))
)
thread.start()
deadline = time.monotonic() + 4
while bridge._queue.empty() and time.monotonic() < deadline:
    time.sleep(0.005)
assert not bridge._queue.empty()
bridge.stop()
bridge.start(0)
try:
    bridge._timer_poll()
    thread.join(2)
    assert not thread.is_alive() and pending == {"ok": False, "error": "Bridge stopped"}, pending
    assert bpy.context.scene.get("excad_active_sketch") != "Stale"
finally:
    bridge.stop()
print("PROTOCOL PASS", bpy.app.version_string, len(schema), "schema bytes")
