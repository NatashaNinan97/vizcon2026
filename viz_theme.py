"""Vizcon design system — tokens, palettes, and shared Plotly chrome.

Palette: validated 8-slot categorical set (light + dark steps), blue
sequential ramp, blue<->red diverging pair, fixed status scale. Categorical
sets pass the dataviz gates (lightness band, chroma floor, CVD >= 8 adjacent,
normal-vision >= 15, contrast) — light-mode aqua/yellow/magenta sit below 3:1
on the surface, so every chart that leans on them ships a data-table view.
"""

import textwrap

FONT_STACK = ('Inter, system-ui, -apple-system, "Segoe UI", Roboto, '
              '"Helvetica Neue", sans-serif')

# ---------------------------------------------------------------- tokens ---

TOKENS = {
    'light': {
        'page': '#f9f9f7', 'surface': '#fcfcfb',
        'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781',
        'grid': '#e1e0d9', 'axis': '#c3c2b7', 'ring': 'rgba(11,11,11,0.10)',
        'slots': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                  '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
        # sequential blue, steps 100 -> 700 (light end recedes on light surface)
        'seq': ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
                '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
                '#184f95', '#104281', '#0d366b'],
        'div_pos': '#2a78d6', 'div_neg': '#e34948', 'div_mid': '#f0efec',
        'good': '#0ca30c', 'warn': '#fab219', 'crit': '#d03b3b',
        'delta_up': '#006300', 'delta_down': '#c22a2a',
        'dim': '#d6d4cc',          # de-emphasis gray for context marks
        'land': '#edece6',         # map land with no data
    },
    'dark': {
        'page': '#0d0d0d', 'surface': '#1a1a19',
        'ink': '#ffffff', 'ink2': '#c3c2b7', 'muted': '#898781',
        'grid': '#2c2c2a', 'axis': '#383835', 'ring': 'rgba(255,255,255,0.10)',
        'slots': ['#3987e5', '#d95926', '#199e70', '#c98500',
                  '#d55181', '#008300', '#9085e9', '#e66767'],
        'seq': ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
                '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
                '#184f95', '#104281', '#0d366b'],
        'div_pos': '#3987e5', 'div_neg': '#e66767', 'div_mid': '#383835',
        'good': '#0ca30c', 'warn': '#fab219', 'crit': '#d03b3b',
        'delta_up': '#0ca30c', 'delta_down': '#e66767',
        'dim': '#3a3a37',
        'land': '#242422',
    },
}


def t(theme):
    return TOKENS['dark' if theme == 'dark' else 'light']


# -------------------------------------------------------------- palettes ---

# Region names — use the original SPI World-Bank-style groupings as-is.
# They are geographically and politically accurate.
REGION_ORDER = ['North America', 'Latin America & Caribbean', 'Europe',
                'Sub-Saharan Africa', 'Middle East & North Africa',
                'Central Asia', 'South Asia', 'East Asia & Pacific']

REGION_SHORT = {'East Asia & Pacific': 'EAP', 'South Asia': 'SA',
                'Central Asia': 'CA', 'Middle East & North Africa': 'MENA',
                'Sub-Saharan Africa': 'SSA', 'Europe': 'EUR',
                'North America': 'NAM', 'Latin America & Caribbean': 'LAC'}

# No remapping needed — original names are correct
REGION_REMAP = {}
NORTH_AFRICA_COUNTRIES = set()  # unused, kept for import compatibility


# Explicit, hand-picked colors per region. These must be maximally
# distinguishable from each other and from the page background, validated for
# normal vision and CVD (deuteranopia/protanopia). The CSS vars --r1..--r8
# in styles.css mirror these in REGION_ORDER sequence.
_REGION_COLORS_LIGHT = {
    'North America':              '#8B4513',  # brown (warm, dark)
    'Latin America & Caribbean':  '#e65100',  # orange/red
    'Europe':                     '#7c3aed',  # purple
    'Sub-Saharan Africa':         '#f9a825',  # yellow-gold (bright, distinct from orange)
    'Middle East & North Africa': '#ad1457',  # magenta/wine (distinct from pink & red)
    'Central Asia':               '#2e7d32',  # forest green
    'South Asia':                 '#f06292',  # light pink
    'East Asia & Pacific':        '#1565c0',  # strong blue
}
_REGION_COLORS_DARK = {
    'North America':              '#CD853F',
    'Latin America & Caribbean':  '#ff7043',
    'Europe':                     '#9085e9',
    'Sub-Saharan Africa':         '#fdd835',
    'Middle East & North Africa': '#ec407a',
    'Central Asia':               '#66bb6a',
    'South Asia':                 '#f48fb1',
    'East Asia & Pacific':        '#42a5f5',
}


def region_colors(theme):
    return _REGION_COLORS_DARK.copy() if theme == 'dark' else _REGION_COLORS_LIGHT.copy()


# Display name for a region — identity for most, since we now use the
# original SPI region names which are already clear and accurate.
REGION_DISPLAY = {}


def region_display(name):
    return REGION_DISPLAY.get(name, name)


# Pillar accents — first three slots validate all-pairs in both modes.
def dim_accent(theme):
    s = t(theme)['slots']
    return {'Basic Needs': s[0],
            'Foundations of Wellbeing': s[1],
            'Societal Opportunity': s[2]}


# Happiness factor slots (6 series, stacked/adjacent form).
FACTOR_ORDER = ['GDP per capita', 'Social support', 'Healthy life expectancy',
                'Freedom to make life choices', 'Generosity',
                'Perceptions of corruption']


def factor_colors(theme):
    s = t(theme)['slots']
    return {f: s[i] for i, f in enumerate(FACTOR_ORDER)}


def seq_scale(theme):
    """Sequential blue colorscale; anchor flips in dark so low recedes."""
    ramp = t(theme)['seq']
    steps = ramp if theme != 'dark' else ramp[::-1]
    n = len(steps) - 1
    return [[i / n, c] for i, c in enumerate(steps)]


def tier_ramp(theme):
    """7 SPI tiers, worst -> best: an ordered heat scale — red (struggling)
    through amber and green to deep blue (leading). Tier identity also rides
    x-position, boundary rules, the legend, and tooltips — never color alone."""
    return ['#c0392b', '#e8833a', '#eda100', '#f4d03f',
            '#7fb069', '#2a9d8f', '#1c5cab']


def status_color(pct, theme):
    """Percentile tier -> status color (>=70 good, 40-70 warning, <40 critical)."""
    k = t(theme)
    if pct is None:
        return k['muted']
    if pct >= 70:
        return k['good']
    if pct >= 40:
        return k['warn']
    return k['crit']


# --------------------------------------------------------- plotly chrome ---

GRAPH_CONFIG = {'displayModeBar': False, 'scrollZoom': False,
                'staticPlot': False, 'responsive': True}


def base_layout(theme, height=450, **overrides):
    """Shared figure chrome: transparent surfaces, Inter, hairline grid."""
    k = t(theme)
    layout = dict(
        height=height,
        autosize=True,   # reflow to container width on laptop/tablet/phone
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_STACK, size=12, color=k['ink2']),
        margin=dict(l=56, r=24, t=28, b=48),
        hoverlabel=dict(bgcolor=k['surface'], bordercolor=k['grid'],
                        font=dict(family=FONT_STACK, size=12, color=k['ink'])),
        xaxis=dict(gridcolor=k['grid'], gridwidth=1, zeroline=False,
                   linecolor=k['axis'], tickfont=dict(size=11, color=k['muted']),
                   title=dict(font=dict(size=12, color=k['muted']))),
        yaxis=dict(gridcolor=k['grid'], gridwidth=1, zeroline=False,
                   linecolor=k['axis'], tickfont=dict(size=11, color=k['muted']),
                   title=dict(font=dict(size=12, color=k['muted']))),
        legend=dict(font=dict(size=11, color=k['ink2']),
                    bgcolor='rgba(0,0,0,0)'),
    )
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(layout.get(key), dict):
            layout[key].update(val)
        else:
            layout[key] = val
    return layout


def wrap(label, width=16):
    """Wrap a long label onto <br> lines for plotly titles."""
    return '<br>'.join(textwrap.wrap(label, width)) or label
