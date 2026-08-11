import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from autosub.core.speaker_map import calculate_speaker_slot_layout, load_speaker_map

logger = logging.getLogger(__name__)


def _crop_and_resize_avatar(img: Image.Image, target_size: Tuple[int, int], radius: int = 12) -> Image.Image:
    """Crops an image to aspect ratio without stretching and applies rounded corners."""
    orig_w, orig_h = img.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h
    orig_ratio = orig_w / orig_h

    if orig_ratio > target_ratio:
        # Image is wider: crop sides (center crop)
        new_w = int(orig_h * target_ratio)
        left = (orig_w - new_w) // 2
        crop_box = (left, 0, left + new_w, orig_h)
    else:
        # Image is taller: crop top/bottom (top-weighted 20% offset for portrait faces)
        new_h = int(orig_w / target_ratio)
        top = int((orig_h - new_h) * 0.20)
        crop_box = (0, max(0, top), orig_w, min(orig_h, top + new_h))

    cropped = img.crop(crop_box).resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Apply rounded corner alpha mask
    mask = Image.new("L", (target_w, target_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, target_w, target_h], radius=radius, fill=255)

    output = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    output.paste(cropped, (0, 0), mask)
    return output


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
        character = entry.get("character")
        color_hex = entry.get("color") or "#FFFFFF"
        avatar_path = entry.get("avatar")

        layout = calculate_speaker_slot_layout(
            slot=slot,
            total_slots=total_slots,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            card_width=340,
            card_margin_left=50,
        )

        cx, cy, cw, ch = (
            layout["card_x"],
            layout["card_y"],
            layout["card_width"],
            layout["card_height"],
        )

        # Parse hex color for accent border / tag
        try:
            hex_clean = color_hex.lstrip("#")
            r, g, b = (
                int(hex_clean[0:2], 16),
                int(hex_clean[2:4], 16),
                int(hex_clean[4:6], 16),
            )
        except Exception:
            r, g, b = 255, 255, 255

        # Draw glassmorphic card background
        card_bg = Image.new("RGBA", (cw, ch), (24, 24, 34, 225))
        card_draw = ImageDraw.Draw(card_bg)

        # Draw accent left border
        card_draw.rectangle([0, 0, 8, ch], fill=(r, g, b, 255))
        card_draw.rectangle([8, 0, cw, ch], outline=(255, 255, 255, 35), width=1)

        # Draw Avatar thumbnail if available
        thumb_size = ch - 24
        thumb_x = 20
        thumb_y = 12

        if avatar_path and Path(avatar_path).exists():
            try:
                avatar_raw = Image.open(avatar_path).convert("RGBA")
                avatar_processed = _crop_and_resize_avatar(avatar_raw, (thumb_size, thumb_size), radius=10)
                card_bg.paste(avatar_processed, (thumb_x, thumb_y), avatar_processed)
            except Exception as e:
                logger.warning(f"Failed to load avatar image {avatar_path}: {e}")
                card_draw.rounded_rectangle(
                    [thumb_x, thumb_y, thumb_x + thumb_size, thumb_y + thumb_size],
                    radius=10,
                    fill=(r, g, b, 100),
                )
        else:
            card_draw.rounded_rectangle(
                [thumb_x, thumb_y, thumb_x + thumb_size, thumb_y + thumb_size],
                radius=10,
                fill=(r, g, b, 100),
            )

        # Render speaker name and character text on the card
        text_start_x = thumb_x + thumb_size + 14
        text_y_offset = 24

        try:
            # Attempt default system sans-serif font
            font_title = ImageFont.truetype("arial.ttf", 22)
            font_subtitle = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()

        card_draw.text((text_start_x, text_y_offset), name, fill=(255, 255, 255, 255), font=font_title)
        if character:
            card_draw.text((text_start_x, text_y_offset + 28), character, fill=(r, g, b, 230), font=font_subtitle)

        # Paste card onto main canvas
        img.paste(card_bg, (cx, cy), card_bg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    logger.info(f"Generated radio overlay PNG image at {output_path}")
    return output_path
