# SpinoML — Brand & Design System

The visual language shared by the app, the README, and the marketing site.
Derived from the colors already used in the product UI, so everything stays
consistent.

## Logo

The mark is a **line-art spinosaurus** facing left, its outline filled with a
warm yellow-orange → pink gradient (head warm, tail pink).

- `assets/logo.svg` — square mark (use on dark surfaces)
- `assets/banner.svg` — horizontal lockup (mark + wordmark + tagline)
- `public/favicon.svg` — app/browser icon
- `src-tauri/icons/` — desktop app icons (regenerate with
  `tauri icon <1024px-master.png>`)

The mark gradient runs left→right across the figure:

`linear-gradient(100deg, #e8a70a 0%, #e95f10 46%, #d82474 100%)`

Clear space ≥ the height of the saurus’ head. Keep the gradient on marketing
surfaces; a flat mark is allowed only where gradients can’t render (e.g.
monochrome print).

## Color tokens

### Brand gradient
The signature is a 135° gradient, light-violet → violet → blue:

| Token | Hex | Use |
|---|---|---|
| `--brand-1` | `#a866ff` | gradient start (light violet) |
| `--brand-2` | `#7e14ff` | gradient core (violet) |
| `--brand-3` | `#47bfff` | gradient end (blue) |
| `--accent`  | `#6ab7ff` | solid interactive accent (links, focus) |

`linear-gradient(135deg, #a866ff 0%, #7e14ff 55%, #47bfff 100%)`

### Surfaces & text (dark theme — the only theme)
| Token | Hex | Use |
|---|---|---|
| `--bg`        | `#0b0d10` | page background |
| `--bg-elev`   | `#0e1115` | raised section background |
| `--panel`     | `#14181c` | cards, panels |
| `--border`    | `#1f2429` | hairline borders |
| `--border-2`  | `#2a3038` | stronger borders / dividers |
| `--text`      | `#e6e8eb` | primary text |
| `--text-dim`  | `#9aa1a8` | secondary text |
| `--text-mute` | `#7a8088` | captions, labels |

### Category accents (the “node graph” palette)
Reused from the canvas layer categories for decorative dots, chips, and
illustrations:

`#6ab7ff` Conv · `#ffb84d` Linear · `#b39dff` Norm · `#4dd0a8` Activation ·
`#ff8a65` Pool · `#ff6b9d` Attention · `#ffd166` Recurrent · `#34d399` Graph ·
`#ec4899` Merge · `#a78bfa` Reshape

## Typography

System UI stack (matches the app, zero web-font fetch):

```
ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
"Helvetica Neue", Arial, sans-serif
```

- **Display / headings** — weight 800, tight tracking (`-0.02em` to `-0.03em`).
- **Body** — weight 400–500, `--text-dim` for long copy.
- **Code / technical labels** — `ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace`.

## Tone

Confident, technical, calm. Present the tool; don’t oversell it. Short verbs,
concrete capabilities, no growth-hacky superlatives.

## Assets in this repo

| Path | What |
|---|---|
| `assets/logo.svg` / `assets/banner.svg` | logo mark + README banner |
| `~/spinoml-website/` *(outside the repo)* | static marketing landing page — self-contained `index.html` using these tokens |
