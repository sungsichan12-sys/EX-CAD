# SPDX-License-Identifier: GPL-2.0-or-later
"""Install official, version-pinned CC0 packs in a reusable cache and fit by asset name."""

from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

from .face_fit import cache_dir, cached_download
from .human_engine import _import_mpfb

PACKS = {
    "system": ("makehuman_system_assets", "b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107"),
    "shirts": ("shirts01", "a5a723b0e84a109bb190fcfeac7f1de4138d875da3e30fe5b3340eac9f38bcd3"),
    "pants": ("pants01", "e4e0ec60db34f279be291a83cfd7b342a7c5cf09bb7676682a5f39f4f6ac4ad9"),
    "skirts": ("skirts01", "293fa0c15e28e8dcea8dfff20cae31aa4b0d1268a6e6ba0a53a92ebb0f36c882"),
}


def install_asset_pack(pack="system", cache=None):
    cache = cache if cache is not None else cache_dir()
    root = cache / "makehuman"
    if pack.lower() == "all":
        prepare_packs(PACKS, cache)
        return root
    slug, digest = PACKS[pack.lower()]
    root.mkdir(exist_ok=True)
    dest = root / (slug + "-" + digest[:12])
    if dest.is_dir():
        return dest
    url = f"https://files.makehumancommunity.org/asset_packs/{slug}/{slug}_cc0.zip"
    archive = cached_download(url, root / (slug + ".zip"), digest)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        staging = Path(temp) / "assets"
        staging.mkdir()
        with ZipFile(archive) as z:
            for entry in z.infolist():
                path = (staging / entry.filename).resolve()
                if not path.is_relative_to(staging):
                    raise ValueError("Invalid asset pack member")
                if not entry.is_dir():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(entry) as src, path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        try:
            staging.rename(dest)
        except OSError:
            if not dest.is_dir():
                raise
    return dest


def prepare_packs(packs, cache):
    for pack in packs:
        install_asset_pack(pack, cache)


def styling_packs(options):
    from .wardrobe import CLOTHES

    clothing = options.get("clothes", [])
    if isinstance(clothing, str):
        clothing = [clothing]
    needed = {CLOTHES[item][0] for item in clothing if item in CLOTHES}
    portrait_assets = ("skin", "lip_tint", "lashes", "teeth", "eyes", "iris_color", "iris_size")
    if isinstance(options.get("hair"), str) or any(options.get(k) for k in portrait_assets):
        needed.add("system")
    return sorted(needed)


def equip_asset(human, name, pack="system", subdiv=0):
    mpfb = _import_mpfb()
    if not mpfb:
        raise RuntimeError("Enable the MPFB extension before equipping assets")
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Use an asset name in an installed pack")
    root = install_asset_pack(pack)
    stem = Path(name).stem.casefold()
    matches = [p for p in root.rglob("*.mhclo") if p.stem.casefold() == stem]
    if len(matches) != 1:
        raise ValueError(f"Expected one asset named {name} in pack {pack}")
    path = matches[0]
    category = path.relative_to(root).parts[0]
    asset_type = category.capitalize()
    # MPFB refits by searching its own asset roots whenever a macro or face target changes.
    library = Path(mpfb[3].get_user_data(category)) / "EX-CAD" / root.name
    library.mkdir(parents=True, exist_ok=True)
    published = library / path.parent.name
    if not published.is_dir():
        with tempfile.TemporaryDirectory(dir=library) as tmp:
            folder = Path(tmp) / "asset"
            shutil.copytree(path.parent, folder)
            try:
                folder.rename(published)
            except OSError:
                if not published.is_dir():
                    raise
    path = published / path.name
    from bl_ext.blender_org.mpfb.services.clothesservice import ClothesService

    obj = mpfb[0].add_mhclo_asset(
        str(path), human, asset_type=asset_type, subdiv_levels=subdiv, interpolate_weights=True, set_up_rigging=True
    )
    ClothesService.fit_clothes_to_human(obj, human, set_parent=False)
    obj["excad_wearer"] = human.name
    obj["excad_human"] = human
    obj["excad_asset_type"] = category.lower()
    obj["excad_mask_group"] = "Delete." + path.stem.replace(" ", "_")
    from .inspect_shading import warm_materials

    warm_materials(obj)
    return obj
