# Speaker colours

Each speaker in a `speaker_map.toml` carries a `color`. It drives the subtitle
outline and, for radio overlays, the VA card's name banner.

```toml
[speakers."0"]
name = "Sakakura Sakura"       # the voice actress
character = "Tomari Onitsuka"  # who she plays
color = "#5DD1CA"              # the CHARACTER's colour, not the VA's
slot = 1
```

## Where to get them

**Love Live (Liella!, Lieraji, concerts): <https://ratius.github.io/LLS/color.html>**

That page lists the official Love Live! Superstar!! character colours. Look up the
**character**, not the voice actress, and copy the hex verbatim. Do not sample
colours from artwork or guess them — the official values are what the fandom
recognises, and they are the same values used across every project here.

**Project Sekai** colours come from a different place: `meta.json` in
`sekai-story-indexer`, which `scripts/generate_overlays.py` in the
subtitling-projects repo reads by character name. There is no equivalent registry
for Love Live, which is why those are transcribed by hand.

Record the source in a comment at the top of each speaker map, so the next person
knows where the values came from:

```toml
# Speaker map for Lieraji Episode 277 (Team Midori)
# Selection 5 official character colors from https://ratius.github.io/LLS/color.html
```

## How the colour is used

`#RRGGBB` in the TOML; `hex_to_pyass_color()` handles the BGR byte order ASS
wants, so never pre-swap it yourself.

- **Subtitles** — the colour goes in the **outline**, with a white fill.
- **Overlay cards** — the colour fills the name banner, and its luminance decides
  whether the banner text is drawn dark or light.

### Why the colour is the outline and not the fill

Slot subtitles sit on a dark translucent backdrop. Filling the text with a dark
character colour makes it unreadable there:

| character | colour | as fill | as outline, white fill |
| --- | --- | --- | --- |
| Sakakura Sakura | `#5DD1CA` | 5.75:1 | 10.56:1 |
| Ookuma Wakana | `#ABDFD0` | 7.14:1 | 10.56:1 |
| Aoyama Nagisa | `#172B80` | **1.17:1** | 10.56:1 |

WCAG asks for at least 3.0:1 on large text. Navy `#172B80` scores 1.17:1 against
the backdrop, and a black outline only adds 1.69:1, so the outline cannot rescue
it. White fill with the colour in the outline reads identically well for every
speaker and still carries the colour cue. This matches the convention the ProSeka
files already use (`Style: Shiho` → white primary, `&H001EC3A5` outline).

## A missing colour does not fail

If a speaker has no `color`, the format stage assigns one from a rotating list of
five pastels, by sorted speaker index, without warning. A typo'd or omitted colour
therefore produces a plausible-looking but wrong colour, and the assignment
shifts if the cast changes. Check the generated `[V4+ Styles]` block after adding
a speaker.
