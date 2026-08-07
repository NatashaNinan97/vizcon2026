# Vizcon — How the World Lives, Thrives & Connects

An interactive visual essay on 15 years of the **Social Progress Index**
(2011–2025, 171 countries, 57 scored indicators) joined with the
**World Happiness Report**. Built with Dash + Plotly.

## Run it

```bash
cd Vizcon
python3 -m venv .venv                       # once
./.venv/bin/pip install pandas numpy plotly dash \
    dash-bootstrap-components openpyxl      # once
./.venv/bin/python dashboard.py             # → http://127.0.0.1:8050
```

## What's inside

| File | Role |
|---|---|
| `dashboard.py` | App, layout, callbacks (Overview / Explore / Social Progress / AI Lab) |
| `viz_data.py` | Data loading & prep — incl. WHR↔SPI country-name alignment and emoji flags |
| `viz_theme.py` | Design system: validated palettes, tokens, shared Plotly chrome |
| `viz_charts.py` | Every figure builder — each returns `(figure, data_table)` |
| `assets/styles.css` | HTML-layer design system (auto-loaded by Dash) |
| `baseline/` | The original dashboard, untouched: code, screenshots, and `PARITY.md` |

## Features

- **Year timeline with playback** on Overview, Explore, and Social Progress (loops).
- **Region lens** (Social Progress) — spotlight one region across the bubble
  strip and both scatters; everything else recedes.
- **Component drill-down** (Explore) — click a component card to unpack its
  sub-indicators by region.
- **Country deep dive** — flags, stat tiles, SPI trend, locator map, dumbbell
  comparison vs region & world, and a four-form indicator breakdown by world
  percentile.
- **Chart forms chosen per job** — lollipops for top-25 rankings, a Cleveland
  dot plot for regional comparison, dumbbells for country-vs-world, SPI-tier
  shading on scatters, residual coloring on happiness-vs-SPI (blue = happier
  than progress predicts, red = less happy).
- **"View data" on every chart** — each figure ships its data-table twin, so
  no value is gated behind a hover.
- **Scoped updates** — interacting with any control re-renders only that
  chart's body; the page never jumps back to the top.
- **AI Lab tab** — eight concrete GenAI integration concepts for this dashboard.

## Design notes

Colors are not hand-picked: the 8-slot categorical palette (regions), the blue
sequential ramp (maps, SPI tiers), the blue↔red diverging pair (gains/losses,
over/under-performers, happiness residuals) and the fixed status scale
(percentile tiers) are validated for CVD separation, lightness band, chroma
and surface contrast. Region ↔ color mapping is fixed everywhere; filtering
never repaints surviving series.

Data sources: [Social Progress Imperative](https://www.socialprogress.org/) ·
[World Happiness Report](https://worldhappiness.report/) (Figure 2.1 data).
