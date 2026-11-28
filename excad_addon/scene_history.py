# SPDX-License-Identifier: GPL-2.0-or-later
"""Named scene checkpoints and bounded request-level undo, without switching .blend files."""

from collections import deque
from pathlib import Path
import tempfile
import uuid

import bpy

DATA = (
    "scenes",
    "objects",
    "collections",
    "meshes",
    "curves",
    "cameras",
    "lights",
    "materials",
    "armatures",
    "actions",
    "images",
    "node_groups",
    "textures",
    "worlds",
    "metaballs",
    "fonts",
    "volumes",
    "pointclouds",
    "hair_curves",
    "lattices",
)


def _dependencies():
    graph = {}
    for block, users in bpy.data.user_map().items():
        for user in users:
            graph.setdefault(user, set()).add(block)
    return graph


def _reachable(roots, graph, excluded=()):
    result, excluded = set(roots), set(excluded)
    queue = list(result)
    while queue:
        additional = graph.get(queue.pop(), set()) - result - excluded
        result.update(additional)
        queue.extend(additional)
    return result


class SceneHistory:
    def __init__(self):
        self.storage = tempfile.TemporaryDirectory(prefix="excad-history-")
        self.named = {}
        self.history = deque()
        self.owner = None
        self.identities = {}
        self.relative_paths = {}

    def _drop(self, path):
        getattr(self, "identities", {}).pop(path, None)
        getattr(self, "relative_paths", {}).pop(path, None)
        path.unlink(missing_ok=True)

    def scope(self):
        uid = bpy.context.scene.session_uid
        if self.owner != uid:
            for path in set(self.named.values()) | set(self.history):
                self._drop(path)
            self.named.clear()
            self.history.clear()
            self.owner = uid

    def capture(self):
        self.scope()
        path = Path(self.storage.name) / (uuid.uuid4().hex + ".blend")
        scene = bpy.context.scene
        bpy.data.libraries.write(str(path), {scene}, path_remap="ABSOLUTE", compress=True)
        if not hasattr(self, "identities"):
            self.identities = {}
        self.identities[path] = (
            scene.name,
            {
                (key, block.name): block.session_uid
                for key in DATA
                for block in getattr(bpy.data, key, ())
                if not block.library
            },
        )
        if not hasattr(self, "relative_paths"):
            self.relative_paths = {}
        self.relative_paths[path] = (
            bpy.data.filepath,
            {
                (key, block.name): block.filepath
                for key in ("images", "volumes", "fonts")
                for block in getattr(bpy.data, key, ())
                if not block.library and block.filepath.startswith("//")
            },
        )
        return path

    def checkpoint(self, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Checkpoint name must be nonempty")
        snapshot = self.capture()
        old = self.named.get(name)
        self.named[name] = snapshot
        if old:
            self._drop(old)
        return {"name": name}

    def remember(self):
        self.history.append(self.capture())
        if len(self.history) > 12:
            self._drop(self.history.popleft())

    def undo(self):
        self.scope()
        if not self.history:
            raise ValueError("No bridge edit to undo")
        snapshot = self.history[-1]
        self.restore(snapshot)
        self._drop(self.history.pop())
        return {}

    def rollback(self, name):
        self.scope()
        if name not in self.named:
            raise ValueError(f"Unknown checkpoint {name}")
        self.remember()
        self.restore(self.named[name])
        return {"name": name}

    def restore(self, path):
        if bpy.context.mode != "OBJECT":
            raise ValueError("Switch to Object Mode before restoring a scene")
        current = bpy.context.scene
        other_scenes = [s for s in bpy.data.scenes if s != current]
        shared = {o for scene in other_scenes for o in scene.objects}
        shared.update(c for scene in other_scenes for c in scene.collection.children_recursive)
        old_objects = set(current.objects) - shared
        old_collections = set(current.collection.children_recursive) - shared
        dependencies = _dependencies()
        external = set(other_scenes)
        owners = _reachable({current, *old_objects, *old_collections}, dependencies, external)
        # References from other scenes (including modifier and collection instances)
        # must survive a restore even when their objects are not linked there directly.
        protected = _reachable(other_scenes, dependencies)
        old_objects -= protected
        old_collections -= protected
        disposable = {
            block.session_uid for block in owners - {current} - old_objects - old_collections - protected - shared
        }
        requested = {}
        root_name, identities = getattr(self, "identities", {}).get(path, (None, {}))
        initial_file, relative = getattr(self, "relative_paths", {}).get(path, (None, {}))
        existing = {block.session_uid: block for block in protected if not block.library}
        with bpy.data.libraries.load(str(path), link=False) as (source, dest):
            if root_name is None and len(source.scenes) == 1:
                root_name = source.scenes[0]
            if root_name not in source.scenes:
                raise ValueError("Invalid scene checkpoint")
            for key in DATA:
                names = list(getattr(source, key, ()))
                if names:
                    requested[key] = names
                    setattr(dest, key, names.copy())
        restored = dest.scenes[requested["scenes"].index(root_name)]
        captured = _reachable(
            {restored, *restored.objects, *restored.collection.children_recursive},
            _dependencies(),
            set(bpy.data.scenes) - {restored},
        )
        # An object linked into two scenes represents one edit. Restore the shared
        # datablock for all its users, retaining links and names across subsequent undo.
        replacements, discard = {}, set()
        external_copies, external_data = [], set()
        for key, names in requested.items():
            for name, block in zip(names, getattr(dest, key)):
                previous_uid = identities.get((key, name))
                old = existing.get(previous_uid)
                # Removing a part can orphan its mesh, finishes or images before
                # rollback. Release exactly the captured originals when unused,
                # and track their replacements for repeated restores.
                if previous_uid and old is None and block in captured and not block.library:
                    disposable.add(previous_uid)
                    replacements[previous_uid] = block.session_uid
                if old is None or not block or block == old or block.library:
                    continue
                if old in external:
                    block.user_remap(old)
                    external_copies.append(block)
                elif old in owners:
                    replacements[old.session_uid] = block.session_uid
                    old.user_remap(block)
                    discard.add(old)
                else:
                    block.user_remap(old)
                    external_data.add(block)
        for _, mapping in getattr(self, "identities", {}).values():
            for key, uid in mapping.items():
                if uid in replacements:
                    mapping[key] = replacements[uid]
        for win in bpy.context.window_manager.windows:
            if win.scene == current:
                win.scene = restored
        bpy.data.scenes.remove(current)
        bpy.data.batch_remove(old_objects | old_collections | discard)
        # Remove zero-user dependencies in order without purging unrelated orphan assets.
        # Removing a mesh can immediately free its embedded shape keys or node tree.
        # Reacquire live IDs after each pass instead of dereferencing invalid RNA handles.
        while disposable:
            unused = {
                i
                for i in bpy.data.user_map()
                if i.session_uid in disposable and not i.library and i.users == 0 and not i.use_fake_user
            }
            if not unused:
                break
            disposable.difference_update(i.session_uid for i in unused)
            bpy.data.batch_remove(unused)
        for key, names in requested.items():
            for name, block in zip(names, getattr(dest, key)):
                if block and not block.library:
                    block.name = name
                    if (key, name) in relative:
                        value = (
                            relative[key, name]
                            if initial_file == bpy.data.filepath
                            else bpy.path.relpath(block.filepath)
                        )
                        if key == "images":
                            block.filepath_raw = value
                        else:
                            block.filepath = value
        for scene in external_copies:
            bpy.data.scenes.remove(scene)
        if external_data:
            bpy.data.batch_remove(external_data)
        self.owner = restored.session_uid
        bpy.context.view_layer.update()
