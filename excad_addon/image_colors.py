# SPDX-License-Identifier: GPL-2.0-or-later
"""Conversions between Blender's byte image pixels and scene-linear material colors."""

import numpy as np


def has_srgb_bytes(image):
    return not image.is_float and image.colorspace_settings.name == "sRGB"


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(0, rgb) ** (1 / 2.4) - 0.055)
