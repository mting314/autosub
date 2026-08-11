import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from autosub.core.speaker_map import calculate_speaker_slot_layout, load_speaker_map

logger = logging.getLogger(__name__)


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
            card_width=320,
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
        card_bg = Image.new("RGBA", (cw, ch), (30, 30, 42, 220))
        card_draw = ImageDraw.Draw(card_bg)

        # Draw accent left border
        card_draw.rectangle([0, 0, 8, ch], fill=(r, g, b, 255))
        card_draw.rectangle([8, 0, cw, ch], outline=(255, 255, 255, 40), width=1)

        # Draw Avatar thumbnail if available
        thumb_size = ch - 24
        thumb_x = 20
        thumb_y = 12

        if avatar_path and Path(avatar_path).exists():
            try:
                avatar_img = Image.open(avatar_path).convert("RGBA")
                avatar_img = avatar_img.resize(
                    (thumb_size, thumb_size), Image.Resampling.LANCZOS
                )
                card_bg.paste(avatar_img, (thumb_x, thumb_y), avatar_img)
            except Exception as e:
                logger.warning(f"Failed to load avatar image {avatar_path}: {e}")
                card_draw.rounded_rectangle(
                    [thumb_x, thumb_y, thumb_x + thumb_size, thumb_y + thumb_size],
                    radius=8,
                    fill=(r, g, b, 100),
                )
        else:
            card_draw.rounded_rectangle(
                [thumb_x, thumb_y, thumb_x + thumb_size, thumb_y + thumb_size],
                radius=8,
                fill=(r, g, b, 100),
            )

        # Paste card onto main canvas
        img.paste(card_bg, (cx, cy), card_bg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    logger.info(f"Generated radio overlay PNG image at {output_path}")
    return output_path
