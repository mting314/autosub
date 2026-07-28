FROM python:3.12-slim

# ffmpeg + fonts. The fonts are REQUIRED, not optional: the `autosub hardsub`
# stage burns subtitles with the libass `ass=` filter, and libass renders every
# style's Fontname via fontconfig. Without these, libass silently falls back to a
# default face and the .ass styling (e.g. `Lato ExtraBold`, Japanese TL notes,
# emoji) does not apply. Keep this list whenever editing the image:
#   fonts-lato            -> Lato / Lato ExtraBold (proseka subtitle styles)
#   fonts-liberation      -> Arial-metric-compatible fallback ("Arial" styles)
#   fonts-noto-cjk        -> Japanese/CJK glyphs (TL notes, romaji, names)
#   fonts-noto-color-emoji-> emoji used in subs (e.g. the pinched-fingers glyph)
#   fontconfig            -> font discovery; fc-cache builds the index at build time
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        ffmpeg \
        fontconfig \
        fonts-lato \
        fonts-liberation \
        fonts-noto-cjk \
        fonts-noto-color-emoji && \
    fc-cache -f && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "autosub"]
