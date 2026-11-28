"""Token-efficient EX-CAD AI Direct Bridge client (TCP 127.0.0.1:9877).

Usage:
    python client_compact.py
"""
import json
import socket


def rpc(payload: dict, host="127.0.0.1", port=9877, timeout=1200.0) -> dict:
    with socket.create_connection((host, port), timeout=min(timeout, 10.0)) as s:
        s.settimeout(timeout)
        s.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8"))


if __name__ == "__main__":
    print("schema:", rpc({"cmd": "schema"}))
    # Single-turn DSL: sketch + extrude + MPFB2 human + scene digest
    print(
        "dsl:",
        rpc({
            "cmd": "dsl",
            "code": "S XY; R -0.5 -0.5 1 1; SOLVE; EX 0.6 0 Box1; H ATHLETIC_FEMALE ASIAN NONE 0 0; DIGEST",
        }),
    )
