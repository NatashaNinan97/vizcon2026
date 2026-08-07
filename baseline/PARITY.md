# Vizcon parity inventory — everything the redesign must preserve

Snapshot of the original app: `baseline/dashboard_original.py`, screenshots in
`baseline/screenshots/` (captured 2026-08-01 against the original running app).
Every item below must exist in the redesigned dashboard. "Restyled" is allowed;
"removed" is not.

## Data

- `SPI.csv` — 2,595 rows; 170 countries × 15 years (2011–2025) + World rows;
  8 regions; index/dimension/component scores + ~57 scored indicators
  (bare-named columns, 0–100, higher = better) + raw-unit twins + population,
  GDP per capita.
- `Happiness_report.xlsx` — World Happiness Report figure 2.1 data, 2011–2025:
  rank, life evaluation, whiskers, 6 "Explained by" factors, dystopia+residual.
  App uses years ≥ 2019.

## Tab 1 — Overview

1. Narrative intro: "Does money buy social progress?" + 170-countries/15-years
   paragraph + Live / Thrive / Connect explainer (verbatim copy).
2. Key insights strip — 5 facts (leader + gap-to-last computed; Qatar-vs-Finland
   GDP/SPI computed; Costa Rica-vs-Qatar computed; Europe-vs-SSA averages
   computed; Mexico #12-happiest/#75-SPI).
3. SPI framework diagram: 3 pillars × 4 components × every indicator name.
4. View selector: Dimensions Distributions / Population / GDP per Capita.
5. Dims view: 3-panel jittered strip plot (Basic Needs, Foundations of
   Wellbeing, Opportunity) by region, y 0–105, region colors, country-highlight
   support.
6. Population view: log-scale choropleth (0.1M–1B tick labels) + top-25
   horizontal bar with ranks, region colors.
7. GDP view: log-scale choropleth ($1K–$100K ticks) + top-25 bar with ranks.

## Tab 2 — Explore

8. Controls: Dimension (Basic Needs=Live / Foundations=Thrive / Societal
   Opportunity=Connect), View (Region Analysis / Country Deep Dive), Year +
   Play/step (region view), Country picker (deep dive), 8-region color legend.
9. Region Analysis: grouped bars — mean component score per region, regions
   sorted by overall average, selected-country diamond overlay.
10. Component drill-down: click a component → sub-indicator regional bars
    (drops no-variation indicators, sorted by overall mean, per-indicator
    descending region sort, value labels). Donut may be replaced by an
    equivalent click-to-drill affordance; indicator counts stay visible.
11. Country deep dive (World default when no country): header "{X} — Deep Dive
    ({year})"; stat tiles Population / GDP per capita / Social Progress; SPI
    over-time line (all 15 years); locator map (continent scope); grouped bars
    country vs region-avg vs world per component.
12. Indicator deep dive (4 quadrant forms preserved): lollipop, gauges,
    diverging-from-median, horizontal percentile bars; three-tier coloring
    (≥70 top / 40–70 mid / <40 bottom); per-quadrant explainer captions.
    NOTE: original "↑ higher raw value is better" copy was misleading (columns
    are SPI-scored, all higher-better) — copy corrected in redesign.

## Tab 3 — Social Progress

13. Year + Play/step controls.
14. SPI bubble strip: x = SPI, jittered y, bubble area = population, 7 tiers
    (bins 35/45/55/65/75/85), top-30 most-populous labeled, "bubble size =
    population" note.
15. GDP vs SPI scatter: log trend line, r annotation, "plateaus" caption.
16. Over/Under-performers toggle: residual ±8 vs log fit; over/under/expected
    classes, labels on outliers, dashed trend.
17. Biggest movers 2011→2025: top-8 gainers + bottom-5 decliners, signed value
    labels, zero baseline.
18. What Drives Happiness: top-20 by rank for chosen year (2019–2025), 6
    stacked factor contributions rescaled to happiness score, #rank labels,
    score at bar end.
19. Happiness vs SPI: nearest-SPI-year merge, region identity, trend + r.

## Cross-cutting

20. Region ↔ color identity stable across every chart (color follows entity).
21. Year animation (~800 ms/frame) with play/pause + step controls.
22. Hover tooltips on every mark (country, values, year context).
23. All 3 tabs; app runs with `python dashboard.py` on :8050.
