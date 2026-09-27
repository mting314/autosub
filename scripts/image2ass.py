#!/usr/bin/env python3
"""Embed PNG/JPEG images into an .ass file as ASS vector drawings.

Inspired by lyger's Image2ASS.lua, but takes standard images (PNG/JPEG/...)
instead of bitmaps and fills `TODO` placeholder lines instead of the Aegisub
GUI selection.

Placeholder lines look like::

    Comment: ...,,TODO: Add "That damn smile" meme
    Comment: ...,,TODO: Insert "..." meme

Images are matched to placeholders by **filename**: the image's stem and the
placeholder's quoted name are both slugified (lowercased, non-alphanumerics
collapsed to '-') and must match exactly. So `that_damn_smile.jpeg` fills the
`TODO: ... "That damn smile" ...` line.

Two modes:
  * Batch (no IMAGE):   scan the .ass file's directory for images and fill every
                        placeholder with a matching image. Reports unmatched
                        placeholders and unused images.
  * Single (IMAGE):     fill the one placeholder whose name matches that image.

The matched placeholder's timing/style is reused for the inserted Dialogue line.

Rendering model (verified against libass/ffmpeg):
  * 1 drawing unit == 1 source pixel; the drawing is scaled on screen with
    ``\\fscx/\\fscy`` so the source image can stay small (compact .ass).
  * Each image row is one ``\\p1`` segment; rows are separated by ``{\\p0}\\N{\\p1}``
    and stack at exactly one drawing-unit pitch (libass line height == drawing
    height).
  * Within a row, consecutive same-colour pixels are run-length encoded. libass
    lays out consecutive drawing segments like glyphs (auto-advancing the pen),
    so every run uses *local* coordinates starting at ``m 0 0``.

Idempotent: a generated line carries ``img2ass:<slug>`` in its Effect field;
re-running replaces the prior line for that slug instead of duplicating it.

Usage:
    uv run python scripts/image2ass.py ASS_FILE                 # batch
    uv run python scripts/image2ass.py IMAGE ASS_FILE           # single
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

# Placeholder pattern: TODO: <Add|Insert|...> "<name>" ...
TODO_RE = re.compile(
    r"""TODO\s*:?\s*(?:add|insert|place|put)?\s*["“](?P<name>[^"”]+)["”]""",
    re.IGNORECASE,
)
EVENT_RE = re.compile(r"^(?P<kind>Dialogue|Comment)\s*:\s*(?P<body>.*)$")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def ass_alpha(opacity: int) -> str:
    """ASS alpha hex: 00 == opaque, FF == transparent (inverse of opacity)."""
    return f"{255 - opacity:02X}"


def parse_playres(lines: list[str]) -> tuple[int, int]:
    x = y = None
    for ln in lines:
        m = re.match(r"\s*PlayResX\s*:\s*(\d+)", ln, re.IGNORECASE)
        if m:
            x = int(m.group(1))
        m = re.match(r"\s*PlayResY\s*:\s*(\d+)", ln, re.IGNORECASE)
        if m:
            y = int(m.group(1))
    return x or 1920, y or 1080


def split_event(body: str) -> list[str]:
    """Split an event body into 10 fields (Text keeps embedded commas)."""
    return body.split(",", 9)


def load_image(path: Path, max_dim: int, colors: int, alpha_levels: int):
    img = Image.open(path).convert("RGBA")
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new, Image.LANCZOS)

    rgb = img.convert("RGB")
    if colors:
        rgb = rgb.quantize(colors=colors, method=Image.MEDIANCUT).convert("RGB")

    rgb_px = rgb.load()
    a_px = img.getchannel("A").load()

    def quant_a(a: int) -> int:
        if alpha_levels <= 1:
            return a
        step = 255 / (alpha_levels - 1)
        return min(255, round(round(a / step) * step))

    w, h = img.size
    grid = []
    for y in range(h):
        row = []
        for x in range(w):
            r, g, b = rgb_px[x, y]
            row.append((r, g, b, quant_a(a_px[x, y])))
        grid.append(row)
    return grid, w, h


def build_drawing(grid, align: int, pos: tuple[int, int], scale: float) -> str:
    sx, sy = pos
    header = (
        f"{{\\an{align}\\pos({sx},{sy})\\bord0\\shad0"
        f"\\fscx{scale:g}\\fscy{scale:g}\\p1}}"
    )
    prev_c = prev_a = None
    rows = []
    for row in grid:
        segs = []
        # run-length encode the row
        x = 0
        n = len(row)
        while x < n:
            r, g, b, a = row[x]
            run = 1
            while x + run < n and row[x + run] == (r, g, b, a):
                run += 1
            tag = ""
            bgr = f"{b:02X}{g:02X}{r:02X}"
            if bgr != prev_c:
                tag += f"\\c&H{bgr}&"
                prev_c = bgr
            ah = ass_alpha(a)
            if ah != prev_a:
                tag += f"\\alpha&H{ah}&"
                prev_a = ah
            prefix = f"{{{tag}}}" if tag else ""
            segs.append(f"{prefix}m 0 0 l 0 1 {run} 1 {run} 0")
            x += run
        rows.append("".join(segs))
    body = "{\\p0}\\N{\\p1}".join(rows)
    return header + body + "{\\p0}"


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def find_placeholders(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    """Return (name, kind, fields) for every TODO placeholder event."""
    out = []
    for ln in lines:
        ev = EVENT_RE.match(ln)
        if not ev:
            continue
        fields = split_event(ev.group("body"))
        if len(fields) < 10:
            continue
        m = TODO_RE.search(fields[9])
        if m:
            out.append((m.group("name").strip(), ev.group("kind"), fields))
    return out


def render_and_insert(lines, fields, kind, image_path, play_x, play_y, args):
    """Render image_path and insert it after the placeholder. Returns (lines, info)."""
    name = TODO_RE.search(fields[9]).group("name").strip()
    start, end, style = fields[1], fields[2], fields[3]
    effect = f"img2ass:{slugify(name)}"

    grid, w, h = load_image(image_path, args.max_dim, args.colors, args.alpha_levels)
    target_w = args.width if args.width else args.width_frac * play_x
    scale = round(target_w / w * 100, 3)
    if args.pos:
        px, py = (int(float(v)) for v in args.pos.split(","))
    else:
        px, py = play_x // 2, play_y // 2
    drawing = build_drawing(grid, args.align, (px, py), scale)
    new_line = f"Dialogue: {args.layer},{start},{end},{style},,0,0,0,{effect},{drawing}"

    # Drop any prior generated line for this slug (idempotency).
    out = []
    for ln in lines:
        ev = EVENT_RE.match(ln)
        if ev:
            f = split_event(ev.group("body"))
            if len(f) >= 10 and f[8] == effect:
                continue
        out.append(ln)

    # Insert right after the placeholder (matched by kind + exact text).
    insert_at = len(out)
    for i, ln in enumerate(out):
        ev = EVENT_RE.match(ln)
        if ev and ev.group("kind") == kind:
            f = split_event(ev.group("body"))
            if len(f) >= 10 and f[9].strip() == fields[9].strip():
                insert_at = i + 1
                break
    out.insert(insert_at, new_line)

    info = {
        "name": name, "image": image_path.name, "w": w, "h": h,
        "on": (round(w * scale / 100), round(h * scale / 100)),
        "pos": (px, py), "scale": scale, "kb": len(new_line) // 1000,
    }
    return out, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", type=Path, help="image to embed; omit for batch mode over the .ass file's directory")
    ap.add_argument("ass_file", type=Path, help="target .ass subtitle file")
    ap.add_argument("--images-dir", type=Path, help="directory to scan for images in batch mode (default: the .ass file's directory)")
    ap.add_argument("--max-dim", type=int, default=240, help="downsample longest side to N px (drawing resolution; default 240)")
    ap.add_argument("--colors", type=int, default=64, help="quantize to N colors to shrink the drawing (0=off; default 64)")
    ap.add_argument("--alpha-levels", type=int, default=8, help="quantize alpha to N levels (default 8)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--width", type=int, help="on-screen width in video px")
    grp.add_argument("--width-frac", type=float, default=0.28, help="on-screen width as fraction of PlayResX (default 0.28)")
    ap.add_argument("--pos", help="anchor position X,Y in video px (default: screen center)")
    ap.add_argument("--align", type=int, default=5, help="ASS alignment 1-9 for the anchor (default 5=center)")
    ap.add_argument("--layer", type=int, default=0, help="event layer for the drawing (default 0)")
    ap.add_argument("-o", "--output", type=Path, help="write to this path (default: in place; ignored for batch unless single)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, do not write")
    args = ap.parse_args(argv)

    if not args.ass_file.exists():
        ap.error(f"ass file not found: {args.ass_file}")
    if args.image is not None and not args.image.exists():
        ap.error(f"image not found: {args.image}")

    text = args.ass_file.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    play_x, play_y = parse_playres(lines)

    placeholders = find_placeholders(lines)
    if not placeholders:
        print("No TODO placeholder lines found.", file=sys.stderr)
        return 1

    # Build the (placeholder, image) work list by slug correspondence.
    by_slug: dict[str, list[tuple[str, str, list[str]]]] = {}
    for ph in placeholders:
        by_slug.setdefault(slugify(ph[0]), []).append(ph)

    jobs: list[tuple[tuple[str, str, list[str]], Path]] = []  # (placeholder, image_path)
    unused_images: list[str] = []

    if args.image is not None:
        # Single mode: match this image's filename to one placeholder.
        slug = slugify(args.image.stem)
        cands = by_slug.get(slug, [])
        if not cands:
            print(f"No TODO placeholder matches image {args.image.name!r} (slug {slug!r}). Placeholders:", file=sys.stderr)
            for name, *_ in placeholders:
                print(f"  - {name}  [{slugify(name)}]", file=sys.stderr)
            return 1
        if len(cands) > 1:
            print(f"Image {args.image.name!r} (slug {slug!r}) matches {len(cands)} placeholders; names must be unique.", file=sys.stderr)
            return 1
        jobs.append((cands[0], args.image))
    else:
        # Batch mode: scan a directory for images, match each by slug.
        img_dir = args.images_dir or args.ass_file.parent
        if not img_dir.is_dir():
            ap.error(f"images dir not found: {img_dir}")
        img_by_slug: dict[str, Path] = {}
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                s = slugify(p.stem)
                if s in img_by_slug:
                    print(f"warning: multiple images slug to {s!r}: keeping {img_by_slug[s].name}, skipping {p.name}", file=sys.stderr)
                    continue
                img_by_slug[s] = p
        used = set()
        for ph in placeholders:
            s = slugify(ph[0])
            img = img_by_slug.get(s)
            if img is not None:
                jobs.append((ph, img))
                used.add(s)
        unused_images = [p.name for s, p in img_by_slug.items() if s not in used]

    matched_slugs = {slugify(ph[0]) for ph, _ in jobs}
    unmatched = [name for name, *_ in placeholders if slugify(name) not in matched_slugs]

    if not jobs:
        print("Nothing to do: no image filenames matched any TODO placeholder.", file=sys.stderr)
        for name, *_ in placeholders:
            print(f"  unmatched placeholder: {name}  [{slugify(name)}]", file=sys.stderr)
        return 1

    # Apply all jobs. Each is keyed by exact placeholder text, so order is safe.
    for (name, kind, fields), image_path in jobs:
        lines, info = render_and_insert(lines, fields, kind, image_path, play_x, play_y, args)
        prefix = "[dry-run] would fill" if args.dry_run else "filled"
        print(
            f"{prefix} {info['name']!r}  <- {info['image']}  "
            f"draw={info['w']}x{info['h']}  on-screen={info['on'][0]}x{info['on'][1]}px "
            f"@{info['pos']} an{args.align}  scale={info['scale']:g}%  line={info['kb']}KB",
            file=sys.stderr,
        )

    for name in unmatched:
        print(f"  no image for placeholder: {name}  [{slugify(name)}]", file=sys.stderr)
    for img in unused_images:
        print(f"  unused image (no matching placeholder): {img}", file=sys.stderr)

    if args.dry_run:
        return 0

    result = "\n".join(lines) + "\n"
    dest = args.output if (args.output and args.image is not None) else args.ass_file
    # Aegisub writes a UTF-8 BOM; preserve it if the source had one.
    had_bom = text[:1] == "﻿"
    dest.write_text(result, encoding="utf-8-sig" if had_bom else "utf-8")
    print(f"Wrote {dest}  ({len(jobs)} image{'s' if len(jobs) != 1 else ''})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
