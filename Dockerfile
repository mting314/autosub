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

# SCXvid — scene-change keyframe generator for the keyframes step. No apt package
# exists, so build the standalone cross-platform port (pinned source) against
# libxvidcore. Build-only deps are purged; only the runtime lib (libxvidcore4) stays.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        ca-certificates curl gcc libc6-dev libxvidcore-dev libxvidcore4 && \
    curl -fsSL "https://raw.githubusercontent.com/soyokaze/SCXvid-standalone/faf6f3b3d1a2e0b400fad2d6b7534f073044cc65/scxvid.c" \
        -o /tmp/scxvid.c && \
    cc -O2 -o /usr/local/bin/SCXvid /tmp/scxvid.c -lxvidcore && \
    apt-mark manual libxvidcore4 && \
    apt-get purge -y --auto-remove curl gcc libc6-dev libxvidcore-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/scxvid.c

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "autosub"]
