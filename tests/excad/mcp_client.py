# SPDX-License-Identifier: GPL-2.0-or-later
"""Exercise EX-CAD's stdio server through the official Python MCP client."""

import asyncio
import base64
from importlib.metadata import version
import json
import os
from pathlib import Path
import struct
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client


async def check(server):
    parameters = StdioServerParameters(
        command=sys.executable, args=[str(server)], env={"EXCAD_PORT": os.environ["EXCAD_PORT"], "PYTHONUTF8": "1"}
    )
    async with stdio_client(parameters) as streams, ClientSession(*streams) as client:
        sdk = version("mcp")
        if int(sdk.split(".")[0]) >= 2:
            info = (await client.discover()).model_dump(by_alias=True)
            protocol = info["supportedVersions"][0]
            assert info["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "excad"
        else:
            info = (await client.initialize()).model_dump(by_alias=True)
            protocol = info["protocolVersion"]
            assert info["serverInfo"]["name"] == "excad"
        listing = await client.list_tools()
        if int(sdk.split(".")[0]) >= 2:
            wire = listing.model_dump(by_alias=True)
            assert wire["resultType"] == "complete" and wire["cacheScope"] == "private" and wire["ttlMs"] == 0
        tools = listing.tools
        expected = {"excad_schema", "excad_run", "excad_digest", "excad_inspect"}
        assert {t.name for t in tools} == expected and len(tools) == 4

        async def call(name, **arguments):
            result = (await client.call_tool(name, arguments)).model_dump(by_alias=True)
            assert not result["isError"], result["content"][0]
            return json.loads(result["content"][0]["text"]), result["content"][1:]

        schema, _ = await call("excad_schema")
        assert len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf8")) <= 1200
        initial, _ = await call("excad_digest")
        name = "SDK test " + sdk
        mark = "Before " + name
        bad_steps = [
            {"cmd": "sdf_generate", "name": "Invalid batch prefix", "tree": {"type": "sphere", "radius": 0.3}},
            "not a command object",
        ]
        for tool, args, field in (
            ("excad_run", {"steps": bad_steps}, "steps[1]"),
            ("excad_inspect", {"size": True}, "size"),
            ("excad_inspect", {"views": ["TOP"]}, "views"),
            ("excad_schema", {"topic": "unknown"}, "topic"),
        ):
            rejected = (await client.call_tool(tool, args)).model_dump(by_alias=True)
            assert rejected["isError"] and field in rejected["content"][0]["text"]
        same, _ = await call("excad_digest", since_hash=initial["h"])
        assert same["unchanged"]
        await call("excad_run", steps=[{"cmd": "checkpoint", "name": mark}])
        try:
            await call(
                "excad_run",
                steps=[
                    {
                        "cmd": "sdf_generate",
                        "name": name,
                        "tree": {"type": "ellipsoid", "radii": [0.6, 0.4, 0.3]},
                        "resolution": 38,
                    },
                    {"cmd": "place_relative", "obj": name, "relation": "ALIGN_FLOOR"},
                ],
            )
            delta, preview = await asyncio.gather(
                call("excad_digest", since_hash=initial["h"]), call("excad_inspect", target=name, size=96.0)
            )
            assert delta[0]["changed"] and not preview[0]["warnings"]
            assert preview[0]["views"] == ["FRONT", "SIDE", "ISO", "TOP"]
            (image,) = preview[1]
            png = base64.b64decode(image["data"], validate=True)
            assert image["mimeType"] == "image/png" and struct.unpack(">II", png[16:24]) == (194, 194)
            path = Path(preview[0]["path"])
            assert png == path.read_bytes() and png.startswith(b"\x89PNG")
            assert path.name.startswith("excad-inspect-") and path.suffix == ".png"
            path.unlink()
            error = (await client.call_tool("excad_inspect", {"target": name + " missing"})).model_dump(by_alias=True)
            assert error["isError"] and error["content"][0]["type"] == "text"
            reloaded, _ = await call("excad_run", steps=[{"cmd": "reload_addon"}])
            assert reloaded["last"]["port"] == int(os.environ["EXCAD_PORT"])
        finally:
            await call("excad_run", steps=[{"cmd": "rollback", "name": mark}])
        restored, _ = await call("excad_digest", since_hash=initial["h"])
        assert restored["unchanged"]
        print("OFFICIAL MCP SDK PASS", sdk, protocol, len(tools), len(png), flush=True)


if __name__ == "__main__":
    asyncio.run(check(Path(__file__).resolve().parents[2] / "excad_mcp_server.py"))
