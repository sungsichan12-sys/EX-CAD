# SPDX-License-Identifier: GPL-2.0-or-later
"""Small, shared wire grammar. JSON command arguments keep paths and nested recipes intact."""

import json


def dsl_lines(code):
    quote = None
    escaped = False
    start = 0
    for i, char in enumerate(code):
        if escaped:
            escaped = False
        elif quote:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char in ";\n":
            if line := code[start:i].strip():
                yield line
            start = i + 1
    if quote:
        raise ValueError("Unterminated DSL string")
    if line := code[start:].strip():
        yield line


def json_command(line):
    if line.startswith("{"):
        data = json.loads(line)
    else:
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[1].startswith("{"):
            return None
        data = json.loads(parts[1])
        if not isinstance(data, dict) or "cmd" in data:
            raise ValueError("Command arguments must be an object without cmd")
        data["cmd"] = parts[0].lower()
    if not isinstance(data, dict) or not data.get("cmd"):
        raise ValueError("Expected a JSON command object")
    return data
