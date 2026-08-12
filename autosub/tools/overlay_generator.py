import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from autosub.core.speaker_map import calculate_speaker_slot_layout, load_speaker_map

logger = logging.getLogger(__name__)


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
            orig_w, orig_h = va_raw.size
            target_ratio = total_w / img_h
            orig_ratio = orig_w / orig_h

            if orig_ratio > target_ratio:
                # Image is wider: crop sides (center crop)
                new_w = int(orig_h * target_ratio)
                left = (orig_w - new_w) // 2
                crop_box = (left, 0, left + new_w, orig_h)
            else:
                # Image is taller: crop top/bottom (12% offset from top for headshots)
                new_h = int(orig_w / target_ratio)
                top = int((orig_h - new_h) * 0.12)
                crop_box = (0, max(0, top), orig_w, min(orig_h, top + new_h))

            cropped = va_raw.crop(crop_box).resize(
                (total_w, img_h), Image.Resampling.LANCZOS
            )
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

    total_slots = max(len(speaker_map), 1)

    # 1. Draw translucent dark background bars behind subtitle text slots
    bar_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    bar_draw = ImageDraw.Draw(bar_layer)

    for label, entry in speaker_map.items():
        slot = entry.get("slot", 1)
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
        bar_h = 100
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
            card_size=(cw, 310),
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
