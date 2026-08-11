import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from autosub.core.speaker_map import calculate_speaker_slot_layout, load_speaker_map

logger = logging.getLogger(__name__)


def _crop_and_resize_avatar(img: Image.Image, target_size: Tuple[int, int], radius: int = 10) -> Image.Image:
    """Crops an image to target size without stretching and applies rounded corners."""
    orig_w, orig_h = img.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h
    orig_ratio = orig_w / orig_h

    if orig_ratio > target_ratio:
        new_w = int(orig_h * target_ratio)
        left = (orig_w - new_w) // 2
        crop_box = (left, 0, left + new_w, orig_h)
    else:
        new_h = int(orig_w / target_ratio)
        top = int((orig_h - new_h) * 0.15)
        crop_box = (0, max(0, top), orig_w, min(orig_h, top + new_h))

    cropped = img.crop(crop_box).resize((target_w, target_h), Image.Resampling.LANCZOS)

    mask = Image.new("L", (target_w, target_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, target_w, target_h], radius=radius, fill=255)

    output = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    output.paste(cropped, (0, 0), mask)
    return output


def _build_va_card(
    va_img_path: Union[Path, str, None],
    title_text: str,
    subtitle_text: str,
    color_hex: str,
    card_size: Tuple[int, int] = (350, 110),
    radius: int = 14,
) -> Image.Image:
    """Renders a sleek horizontal glassmorphic card with VA photo, accent border, and VA/character text."""
    cw, ch = card_size

    # Parse color
    try:
        hex_clean = color_hex.lstrip("#")
        r, g, b = (
            int(hex_clean[0:2], 16),
            int(hex_clean[2:4], 16),
            int(hex_clean[4:6], 16),
        )
    except Exception:
        r, g, b = 255, 127, 39

    # Glassmorphic card container
    card_bg = Image.new("RGBA", (cw, ch), (24, 24, 34, 225))
    card_draw = ImageDraw.Draw(card_bg)

    # Accent left bar
    card_draw.rectangle([0, 0, 8, ch], fill=(r, g, b, 255))
    card_draw.rectangle([8, 0, cw, ch], outline=(255, 255, 255, 35), width=1)

    # Photo thumbnail inside card (placed left)
    thumb_w = 90
    thumb_h = ch - 20
    thumb_x = 20
    thumb_y = 10

    if va_img_path and Path(va_img_path).exists():
        try:
            va_raw = Image.open(va_img_path).convert("RGBA")
            va_processed = _crop_and_resize_avatar(va_raw, (thumb_w, thumb_h), radius=10)
            card_bg.paste(va_processed, (thumb_x, thumb_y), va_processed)
        except Exception as e:
            logger.warning(f"Failed to load avatar photo {va_img_path}: {e}")
            card_draw.rounded_rectangle(
                [thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h],
                radius=10,
                fill=(r, g, b, 100),
            )
    else:
        card_draw.rounded_rectangle(
            [thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h],
            radius=10,
            fill=(r, g, b, 100),
        )

    # Text block on right side of avatar photo
    text_x = thumb_x + thumb_w + 16

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 22)
        font_subtitle = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            font_title = ImageFont.truetype("arial.ttf", 22)
            font_subtitle = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()

    t_bbox = font_title.getbbox(title_text)
    s_bbox = font_subtitle.getbbox(subtitle_text) if subtitle_text else (0, 0, 0, 0)

    t_h = t_bbox[3] - t_bbox[1]
    s_h = s_bbox[3] - s_bbox[1]
    line_spacing = 6

    text_block_h = t_h + (line_spacing + s_h if subtitle_text else 0)
    start_y = max(10, (ch - text_block_h) // 2 - 2)

    card_draw.text((text_x, start_y), title_text, font=font_title, fill=(255, 255, 255, 255))
    if subtitle_text:
        card_draw.text((text_x, start_y + t_h + line_spacing), subtitle_text, font=font_subtitle, fill=(r, g, b, 230))

    # Outer rounded corner mask around entire card
    mask = Image.new("L", (cw, ch), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rounded_rectangle([(0, 0), (cw, ch)], radius=radius, fill=255)

    final_card = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    final_card.paste(card_bg, (0, 0), mask=mask)
    return final_card


def generate_radio_overlay_image(
    speaker_map_or_path: Union[Dict[str, Dict[str, Any]], Path],
    output_path: Path,
    canvas_size: Tuple[int, int] = (1920, 1080),
) -> Path:
    """Renders a 1920x1080 transparent PNG with avatar cards for each speaker slot.

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

    total_slots = max(len(speaker_map), 1)

    for label, entry in speaker_map.items():
        slot = entry.get("slot", 1)
        name = entry.get("name", label)
        character = entry.get("character", "")
        color_hex = entry.get("color") or "#FFFFFF"
        avatar_path = entry.get("avatar")

        layout = calculate_speaker_slot_layout(
            slot=slot,
            total_slots=total_slots,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            card_width=350,
            card_margin_left=50,
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
            card_size=(cw, min(ch, 110)),
            radius=14,
        )

        # Paste card onto main canvas centered in slot Y
        card_y = int(cy + (ch - card_img.height) / 2)
        img.paste(card_img, (cx, card_y), card_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    logger.info(f"Generated radio overlay PNG image at {output_path}")
    return output_path
