# SPDX-License-Identifier: GPL-2.0-or-later
"""EX-CAD's four-tool MCP stdio server. Run with ordinary Python, alongside Blender."""

import base64
import json
import os
from pathlib import Path
import socket
import sys

MODERN = "2026-07-28"
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
INFO = {"name": "excad", "version": "1.1"}
CAPABILITIES = {"tools": {}}


def tool(name, description, properties, **schema):
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "additionalProperties": False, **schema},
    }


TOOLS = [
    tool(
        "excad_schema",
        "Get the compact Blender command reference.",
        {"topic": {"enum": ["sketch", "cad", "character", "organic", "scene"]}},
    ),
    tool(
        "excad_run",
        "Run a DSL program or an ordered batch in Blender.",
        {
            "code": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object"}},
            "return_all": {"type": "boolean"},
            "stop_on_error": {"type": "boolean"},
        },
        oneOf=[{"required": ["code"]}, {"required": ["steps"]}],
    ),
    tool(
        "excad_digest",
        "Return scene changes since the given digest.",
        {"since_hash": {"type": "string"}, "include_all": {"type": "boolean"}},
    ),
    tool(
        "excad_inspect",
        "Inspect evaluated geometry and return a 2×2 PNG with bounds and warnings.",
        {
            "target": {"type": "string"},
            "views": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"enum": ["FRONT", "SIDE", "ISO", "TOP"]},
            },
            "size": {"type": "integer", "minimum": 64, "maximum": 1024},
        },
    ),
]

SCHEMAS = {entry["name"]: entry["inputSchema"] for entry in TOOLS}


def checked_value(field, value, schema):
    """Validate the JSON types, enums and bounds used in the four advertised inputs."""
    kind = schema.get("type")
    if kind == "integer":
        if type(value) not in (int, float) or value % 1:
            raise ValueError(f"{field} must be an integer")
        value = int(value)
        if not schema["minimum"] <= value <= schema["maximum"]:
            raise ValueError(f"{field} must be between {schema['minimum']} and {schema['maximum']}")
    elif kind and type(value) is not {"object": dict, "array": list, "string": str, "boolean": bool}[kind]:
        raise ValueError(f"{field} must have type {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{field} must be one of {', '.join(schema['enum'])}")
    if kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{field} must contain {schema['minItems']} items")
        return [checked_value(f"{field}[{i}]", item, schema["items"]) for i, item in enumerate(value)]
    return value


def rpc(command):
    with socket.create_connection(("127.0.0.1", int(os.getenv("EXCAD_PORT", "9877"))), timeout=5) as sock:
        sock.settimeout(1200)
        sock.sendall((json.dumps(command, separators=(",", ":")) + "\n").encode("utf8"))
        with sock.makefile("rb") as stream:
            raw = stream.readline()
        if not raw:
            raise ConnectionError("Blender closed the bridge connection")
        return json.loads(raw)


def call(name, args):
    if name not in SCHEMAS:
        raise ValueError(f"Unknown tool: {name}")
    commands = {"excad_schema": "schema", "excad_digest": "scene_digest", "excad_inspect": "render_inspect"}
    try:
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        properties = SCHEMAS[name]["properties"]
        unknown = args.keys() - properties.keys()
        if unknown:
            raise ValueError("Unknown argument " + ", ".join(sorted(unknown)))
        args = {field: checked_value(field, value, properties[field]) for field, value in args.items()}
        if name == "excad_run":
            if ("steps" in args) == ("code" in args):
                raise ValueError("Specify either code or steps")
            cmd = "dsl" if "code" in args else "batch"
        else:
            cmd = commands[name]
        result = rpc(dict(args, cmd=cmd))
        items = [{"type": "text", "text": json.dumps(result, ensure_ascii=False, separators=(",", ":"))}]
        if name == "excad_inspect" and result.get("ok"):
            raw = Path(result["path"]).read_bytes()
            items.append({"type": "image", "mimeType": "image/png", "data": base64.b64encode(raw).decode("ascii")})
        return {"content": items, "isError": not result.get("ok", False)}
    except (OSError, ValueError, KeyError) as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}


def handle(message):
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return {"id": None, "jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}}
    if "id" not in message:
        return None
    reply = {"id": message["id"], "jsonrpc": "2.0"}
    method = message["method"]
    params = message.get("params", {})
    try:
        if not isinstance(params, dict):
            raise ValueError("Expected an object for params")
        version = params.get("_meta", {}).get("io.modelcontextprotocol/protocolVersion")
        if version and version not in (MODERN, *LEGACY):
            reply["error"] = {
                "code": -32022,
                "message": "Unsupported protocol version",
                "data": {"supported": [MODERN, *LEGACY], "requested": version},
            }
        elif method == "server/discover":
            reply["result"] = {"supportedVersions": [MODERN, *LEGACY], "capabilities": CAPABILITIES}
        elif method == "initialize":
            requested = params.get("protocolVersion")
            reply["result"] = {
                "protocolVersion": requested if requested in LEGACY else LEGACY[0],
                "capabilities": CAPABILITIES,
                "serverInfo": INFO,
            }
        elif method == "ping":
            reply["result"] = {}
        elif method == "tools/list":
            reply["result"] = {"tools": TOOLS}
        elif method == "tools/call":
            reply["result"] = call(params["name"], params.get("arguments", {}))
        else:
            reply["error"] = {"code": -32601, "message": "Method not found"}
        if "result" in reply and (version == MODERN or method == "server/discover"):
            result = reply["result"]
            result["resultType"] = "complete"
            result["_meta"] = {"io.modelcontextprotocol/serverInfo": INFO}
            if method in {"server/discover", "tools/list"}:
                result.update(ttlMs=0, cacheScope="private")
    except (TypeError, KeyError, ValueError, AttributeError) as exc:
        reply["error"] = {"code": -32602, "message": str(exc)}
    return reply


def main():
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        try:
            reply = handle(json.loads(line))
        except (ValueError, UnicodeError):
            reply = {"id": None, "jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}
        if reply is not None:
            sys.stdout.buffer.write(
                (json.dumps(reply, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf8")
            )
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
