import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from autosub.core.speaker_map import calculate_speaker_slot_layout, load_speaker_map

logger = logging.getLogger(__name__)


# Where the subject's face sits in a standard portrait headshot, and how much of
# the frame to keep around it. Measured against the official Love Live cast
# photos: a square of 66% of the source height, centred 35% down, holds the whole
# head without clipping hair, and stops well above any caption band.
_FACE_CENTRE_Y = 0.35
_FACE_CROP_SCALE = 0.66


def _content_bounds_x(img: Image.Image) -> Tuple[int, int]:
    """Horizontal extent of the artwork, excluding a plain white page margin.

    Official cast photos are supplied as a card floating on white. Cropping that
    margin away first keeps the face centred on the subject rather than on the
    page.
    """
    grey = img.convert("L")
    width, height = grey.size
    pixels = grey.load()
    step = max(1, height // 64)

    def column_is_margin(x: int) -> bool:
        return all(pixels[x, y] > 235 for y in range(0, height, step))

    left = 0
    while left < width - 1 and column_is_margin(left):
        left += 1
    right = width
    while right > left + 1 and column_is_margin(right - 1):
        right -= 1

    # A thin coloured frame usually hugs the margin; step inside it.
    inset = max(2, int((right - left) * 0.02))
    left, right = left + inset, right - inset
    if right - left < width * 0.2:  # detection went wrong, keep the full width
        return 0, width
    return left, right


def _crop_to_face(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Crop a headshot to the subject's face at the requested aspect ratio.

    Source photos often carry a white page margin, a coloured frame and a caption
    band with the subject's name. Framing on the face drops all three, because
    they sit outside the head.
    """
    width, height = img.size
    left, right = _content_bounds_x(img)
    centre_x = (left + right) / 2

    crop_h = height * _FACE_CROP_SCALE
    crop_w = crop_h * (target_w / target_h)
    if crop_w > right - left:  # never widen past the artwork into the margin
        crop_w = right - left
        crop_h = crop_w * (target_h / target_w)

    top = max(0.0, height * _FACE_CENTRE_Y - crop_h / 2)
    top = min(top, height - crop_h)
    box = (
        int(max(0, centre_x - crop_w / 2)),
        int(top),
        int(min(width, centre_x + crop_w / 2)),
        int(top + crop_h),
    )
    return img.crop(box).resize((target_w, target_h), Image.Resampling.LANCZOS)


def _build_va_card(
    va_img_path: Union[Path, str, None],
    title_text: str,
    subtitle_text: str,
    color_hex: str,
    card_size: Tuple[int, int] = (260, 310),
    banner_height: int = 70,
    radius: int = 14,
) -> Image.Image:
    """Renders a vertical portrait VA card with a large top photo and VA + character name banner at the bottom."""
    total_w, card_h = card_size
    img_h = card_h - banner_height

    # Parse hex color
    try:
        hex_clean = color_hex.lstrip("#")
        r, g, b = (
            int(hex_clean[0:2], 16),
            int(hex_clean[2:4], 16),
            int(hex_clean[4:6], 16),
        )
    except Exception:
        r, g, b = 255, 127, 39

    # 1. Top Portion: Large Portrait Photo
    top_container = Image.new("RGBA", (total_w, img_h), (r, g, b, 120))
    if va_img_path and Path(va_img_path).exists():
        try:
            va_raw = Image.open(va_img_path).convert("RGBA")
            cropped = _crop_to_face(va_raw, total_w, img_h)
            top_container.paste(cropped, (0, 0))
        except Exception as e:
            logger.warning(f"Failed to process VA photo {va_img_path}: {e}")

    # 2. Bottom Portion: Accent Color Banner for VA + Char Name
    banner = Image.new("RGBA", (total_w, banner_height), (r, g, b, 255))
    draw_b = ImageDraw.Draw(banner)

    # Determine contrast text color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > 140:
        dark_text_color = (20, 20, 30, 255)
        sub_text_color = (40, 40, 60, 240)
    else:
        dark_text_color = (255, 255, 255, 255)
        sub_text_color = (235, 235, 245, 230)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 20)
        font_subtitle = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        try:
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_subtitle = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()

    t_bbox = font_title.getbbox(title_text)
    s_bbox = font_subtitle.getbbox(subtitle_text) if subtitle_text else (0, 0, 0, 0)

    t_w = t_bbox[2] - t_bbox[0]
    t_h = t_bbox[3] - t_bbox[1]
    s_w = s_bbox[2] - s_bbox[0]
    s_h = s_bbox[3] - s_bbox[1]

    line_spacing = 4
    text_block_h = t_h + (line_spacing + s_h if subtitle_text else 0)
    start_y = max(4, int((banner_height - text_block_h) // 2 - 2))

    t_x = (total_w - t_w) // 2
    draw_b.text((t_x, start_y), title_text, font=font_title, fill=dark_text_color)

    if subtitle_text:
        s_y = start_y + t_h + line_spacing
        s_x = (total_w - s_w) // 2
        draw_b.text((s_x, s_y), subtitle_text, font=font_subtitle, fill=sub_text_color)

    # 3. Combine Top & Bottom flush
    card_raw = Image.new("RGBA", (total_w, card_h), (0, 0, 0, 0))
    card_raw.paste(top_container, (0, 0))
    card_raw.paste(banner, (0, img_h))

    # Outer subtle border line
    draw_card = ImageDraw.Draw(card_raw)
    draw_card.rectangle(
        [0, 0, total_w - 1, card_h - 1], outline=(255, 255, 255, 60), width=1
    )

    # 4. Single outer rounded mask around entire card
    mask = Image.new("L", (total_w, card_h), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle([(0, 0), (total_w, card_h)], radius=radius, fill=255)

    final_card = Image.new("RGBA", (total_w, card_h), (0, 0, 0, 0))
    final_card.paste(card_raw, (0, 0), mask=mask)
    return final_card


def generate_radio_overlay_image(
    speaker_map_or_path: Union[Dict[str, Dict[str, Any]], Path],
    output_path: Path,
    canvas_size: Tuple[int, int] = (1920, 1080),
) -> Path:
    """Renders a 1920x1080 transparent PNG with vertical VA cards and translucent text background bars.

    :param speaker_map_or_path: Speaker map dictionary or path to speaker_map.toml
    :param output_path: Destination PNG filepath
    :param canvas_size: Output image resolution (default 1920x1080)
    :return: Output Path
    """
    if isinstance(speaker_map_or_path, (str, Path)):
        speaker_map = load_speaker_map(Path(speaker_map_or_path))
    else:
        speaker_map = speaker_map_or_path

    canvas_w, canvas_h = canvas_size
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    valid_slots = [
        entry.get("slot")
        for entry in speaker_map.values()
        if entry.get("slot") is not None
    ]
    total_slots = max(valid_slots) if valid_slots else 1

    # 1. Draw translucent dark background bars behind subtitle text slots
    bar_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    bar_draw = ImageDraw.Draw(bar_layer)

    drawn_bar_slots = set()
    for label, entry in speaker_map.items():
        slot = entry.get("slot", 1)
        if slot in drawn_bar_slots:
            continue
        drawn_bar_slots.add(slot)

        layout = calculate_speaker_slot_layout(
            slot=slot,
            total_slots=total_slots,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            card_width=260,
            card_margin_left=30,
            text_gap=40,
        )

        cx, cy, cw, ch = (
            layout["card_x"],
            layout["card_y"],
            layout["card_width"],
            layout["card_height"],
        )

        center_y = int(cy + ch / 2)
        bar_h = 180
        bar_y1 = center_y - (bar_h // 2)
        bar_y2 = bar_y1 + bar_h
        bar_x1 = cx + cw + 20
        bar_x2 = canvas_w - 30

        # Translucent dark charcoal background bar (approx 70% opacity)
        bar_draw.rounded_rectangle(
            [(bar_x1, bar_y1), (bar_x2, bar_y2)],
            radius=14,
            fill=(14, 14, 20, 175),
            outline=(255, 255, 255, 30),
            width=1,
        )

    img.paste(bar_layer, (0, 0), bar_layer)

    # 2. Render and paste VA cards
    drawn_card_slots = set()
    for label, entry in speaker_map.items():
        slot = entry.get("slot", 1)
        if slot in drawn_card_slots:
            continue
        drawn_card_slots.add(slot)

        name = entry.get("name", label)
        character = entry.get("character", "")
        color_hex = entry.get("color") or "#FFFFFF"
        avatar_path = entry.get("avatar")

        layout = calculate_speaker_slot_layout(
            slot=slot,
            total_slots=total_slots,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            card_width=260,
            card_margin_left=30,
            text_gap=40,
        )

        cx, cy, cw, ch = (
            layout["card_x"],
            layout["card_y"],
            layout["card_width"],
            layout["card_height"],
        )

        card_img = _build_va_card(
            va_img_path=avatar_path,
            title_text=name,
            subtitle_text=character if character else "",
            color_hex=color_hex,
            card_size=(cw, 330),
            banner_height=70,
            radius=14,
        )

        # Paste card onto main canvas centered in slot Y
        card_y = int(cy + (ch - card_img.height) / 2)
        img.paste(card_img, (cx, card_y), card_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    logger.info(f"Generated radio overlay PNG image at {output_path}")
    return output_path
