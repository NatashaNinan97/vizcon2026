"""Vizcon — How the World Lives, Thrives & Connects.

An interactive visual essay on the Social Progress Index (2011–2025) and the
World Happiness Report. Run with:  python dashboard.py  →  http://127.0.0.1:8050

The original dashboard is preserved verbatim in baseline/ (code + screenshots);
baseline/PARITY.md inventories everything this redesign keeps.1. Year not working for  country analysis
2. For basic needs specify country way of handling
3. Mention slider for regional analysis
4. Over under perform tooltip not rounding
5. Move legend of happiness dirvers down
6. Key insight - nearly double Sub-Saharan Africa’s nan makes no sense
7. Make Key insights less text heavy

"""

import json
from functools import lru_cache
from threading import Thread

import pandas as pd
from dash import (Dash, dcc, html, Input, Output, State, callback, ctx, ALL,
                  MATCH, no_update, clientside_callback, ClientsideFunction)
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from viz_data import (df, world_df, happy_df, years, countries, regions,
                      LATEST, EARLIEST, SPI_COL, GDP_COL, POP_COL,
                      DIMENSIONS, N_INDICATORS, happy_years, flag,
                      INDICATOR_DEFINITIONS, DIMENSION_COMPONENTS)
from viz_theme import (dim_accent, GRAPH_CONFIG, REGION_ORDER, region_display)
import viz_charts as C
import ai_query as AI

THEME = 'light'
PLAY_SPEED = 800
FONTS = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
FAVICON = ("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' "
           "viewBox='0 0 100 100'><circle cx='50' cy='50' r='42' fill='%232a78d6'/>"
           "<path d='M8 50h84M50 8c14 12 14 72 0 84M50 8c-14 12-14 72 0 84' "
           "stroke='%23fcfcfb' stroke-width='5' fill='none'/></svg>")

app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP, FONTS],
           suppress_callback_exceptions=True,
           title='Amazon VizCon 2026 — How the World Lives, Thrives & Connects',
           update_title=None)
# Flask instance underneath Dash — gunicorn/WSGI hosts (Render, etc.) import
# this module-level `server` name to run the app in production.
server = app.server

app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
<link rel="icon" href="''' + FAVICON + '''">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
{%favicon%}{%css%}
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''


# ================================================================ helpers ==

def head_block(kicker, title, subtitle=None, accent=False):
    kids = [html.Div(kicker, className='kicker' + (' k-accent' if accent else ''))]
    if title:
        kids.append(html.H3(title, className='card-title'))
    if subtitle:
        kids.append(html.Div(subtitle, className='card-sub'))
    return html.Div(kids, className='card-head')


def table_el(tdf, max_rows=400, show_median=True):
    """Scrollable HTML table for a chart's data twin.

    When show_median is True, a footer row reports the median of every numeric
    column (the median-country score for that metric)."""
    full = tdf
    tdf = tdf.head(max_rows)

    def fmt(v):
        if pd.isna(v):
            return '—'
        if isinstance(v, float):
            return f'{v:,.2f}'.rstrip('0').rstrip('.')
        return str(v)

    footer = None
    if show_median:
        numeric_cols = full.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            cells = []
            first = True
            for c in full.columns:
                if c in numeric_cols:
                    cells.append(html.Td(f'{full[c].median():,.2f}'.rstrip('0').rstrip('.'),
                                         className='median-cell'))
                elif first:
                    cells.append(html.Td('Median', className='median-cell median-label'))
                    first = False
                else:
                    cells.append(html.Td('', className='median-cell'))
            footer = html.Tfoot(html.Tr(cells))

    table_children = [
        html.Thead(html.Tr([html.Th(str(c)) for c in tdf.columns])),
        html.Tbody([html.Tr([html.Td(fmt(v)) for v in row])
                    for row in tdf.itertuples(index=False)]),
    ]
    if footer is not None:
        table_children.append(footer)
    table = html.Table(className='viz-table', children=table_children)
    return html.Div(table, className='table-scroll')


def data_view(tdf, label='View data', max_rows=400):
    """Native-disclosure data-table twin for a chart (the no-hover channel)."""
    if tdf is None or len(tdf) == 0:
        return None
    return html.Details(className='data-view', children=[
        html.Summary(label),
        table_el(tdf, max_rows),
    ])


def graph(fig, **kw):
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG, **kw)


def how_to_read(text):
    """Collapsible 'How to read this chart' helper shown under a chart.
    Accepts a string or a list of bullet strings."""
    if isinstance(text, (list, tuple)):
        body = html.Ul([html.Li(t) for t in text], className='htr-list')
    else:
        body = html.Div(text, className='htr-body')
    return html.Details(className='how-to-read', children=[
        html.Summary(['\U0001f4d6  How to read this chart']),
        body,
    ])


def next_tab_nudge(key, label, target, wrapper_id=None, style=None):
    """Small pointer placed right before a Takeaway card, nudging the reader
    onward instead of scrolling past the summary unnoticed. Also doubles as a
    quick "back to top" so long tabs don't trap you.

    `key` must be unique per nudge on the page (e.g. 'ov', 'sp-spi').
    `target` is either 'tab:<tab_id>' (switch the top nav) or
    'subview:<value>' (switch the Progress tab's own View toggle)."""
    extra = {}
    if wrapper_id:
        extra['id'] = wrapper_id
    if style:
        extra['style'] = style
    return html.Div(className='next-tab-nudge', **extra, children=[
        html.Span('Done exploring this section?', className='next-tab-hint'),
        html.Button([f'Switch to {label} \u2192'],
                    id={'type': 'next-tab-btn', 'index': key},
                    className='next-tab-btn', n_clicks=0),
        html.Button('\u2191 Back to Top',
                    id={'type': 'next-tab-top-btn', 'index': key},
                    className='next-tab-btn next-tab-btn--ghost', n_clicks=0),
        dcc.Store(id={'type': 'next-tab-target', 'index': key}, data=target),
    ])


def stat_tile(label, value, delta=None, up=True, note=None):
    kids = [html.Div(label, className='stat-label'),
            html.Div(value, className='stat-value')]
    if delta:
        kids.append(html.Div(delta, className='stat-delta ' + ('up' if up else 'down')))
    if note:
        kids.append(html.Div(note, className='stat-note'))
    return html.Div(kids, className='stat-tile')


def region_chips():
    def color_for(r):
        if r == 'Europe':
            return 'var(--europe)'
        return f'var(--r{REGION_ORDER.index(r) + 1})'
    return html.Div(className='chip-row', children=[
        html.Span([html.Span(className='c-dot',
                             style={'backgroundColor': color_for(r)}),
                   region_display(r)], className='chip') for r in regions])


def segmented(cid, options, value):
    return dcc.RadioItems(id=cid, className='segmented', options=options,
                          value=value, inline=True)


def timeline(prefix, note=None, wrapper_id=None):
    """Year slider + play/step controls + interval, one per tab."""
    kids = [
        html.Div([
            html.Button('▶ Play', id=f'{prefix}-play-btn', className='play-btn',
                        n_clicks=0),
            html.Button('‹', id=f'{prefix}-prev-btn', className='step-btn',
                        n_clicks=0, title='Previous year'),
            html.Button('›', id=f'{prefix}-next-btn', className='step-btn',
                        n_clicks=0, title='Next year'),
        ], className='play-group'),
        html.Div(dcc.Slider(
            id=f'{prefix}-year', min=min(years), max=max(years), step=1,
            value=LATEST, marks={y: str(y) for y in years if y % 2 == 1},
            included=True, persistence=True, persistence_type='session',
            updatemode='drag'),
            className='slider-wrap'),
        html.Div(str(LATEST), id=f'{prefix}-year-badge', className='year-badge'),
        dcc.Interval(id=f'{prefix}-play-interval', interval=PLAY_SPEED,
                     disabled=True),
    ]
    wrap = [html.Div('Year', className='control-label'),
            html.Div(kids, className='timeline')]
    if note:
        wrap.append(html.Div(note, className='stat-note', style={'marginTop': '2px'}))
    extra = {'id': wrapper_id} if wrapper_id else {}
    return html.Div(wrap, className='control-group grow', **extra)


def make_play_callbacks(prefix):
    @callback(Output(f'{prefix}-play-interval', 'disabled'),
              Output(f'{prefix}-play-btn', 'children'),
              Output(f'{prefix}-play-btn', 'className'),
              Output(f'{prefix}-year', 'value', allow_duplicate=True),
              Input(f'{prefix}-play-btn', 'n_clicks'),
              Input(f'{prefix}-year', 'value'),
              State(f'{prefix}-play-interval', 'disabled'),
              prevent_initial_call=True)
    def toggle(n, year, disabled):
        trig = ctx.triggered_id
        # Auto-stop on reaching the last year in the range
        if trig == f'{prefix}-year' and year == years[-1] and not disabled:
            return True, '▶ Play', 'play-btn', no_update
        if trig == f'{prefix}-play-btn':
            playing = bool(disabled)
            # The slider defaults to the latest year, so hitting play there
            # would advance nowhere. Rewind to the first year instead and
            # play the series from the start.
            rewind = years[0] if playing and year == years[-1] else no_update
            return (not playing,
                    '❚❚ Pause' if playing else '▶ Play',
                    'play-btn playing' if playing else 'play-btn',
                    rewind)
        return no_update, no_update, no_update, no_update

    @callback(Output(f'{prefix}-year', 'value'),
              Input(f'{prefix}-play-interval', 'n_intervals'),
              Input(f'{prefix}-prev-btn', 'n_clicks'),
              Input(f'{prefix}-next-btn', 'n_clicks'),
              State(f'{prefix}-year', 'value'),
              prevent_initial_call=True)
    def nav(n_int, prev, nxt, cur):
        idx = years.index(cur) if cur in years else len(years) - 1
        if ctx.triggered_id == f'{prefix}-prev-btn':
            return years[max(0, idx - 1)]
        if ctx.triggered_id == f'{prefix}-next-btn':
            return years[min(len(years) - 1, idx + 1)]
        # Stop at 2025 instead of looping
        nxt_idx = idx + 1
        if nxt_idx >= len(years):
            return years[-1]
        return years[nxt_idx]

    # Keep the big year badge in sync with the slider — clientside, no
    # server roundtrip needed.
    app.clientside_callback(
        "function(year) { return String(year); }",
        Output(f'{prefix}-year-badge', 'children'),
        Input(f'{prefix}-year', 'value'),
    )


# ================================================================= layout ==

app.layout = html.Div(id='app-root', children=[
    html.Div(className='topbar', children=html.Div(className='topbar-inner', children=[
        dbc.Tabs([
            dbc.Tab(label='Overview', tab_id='overview'),
            dbc.Tab(label='Explore', tab_id='explore'),
            dbc.Tab(label='Progress', tab_id='social_progress'),
            dbc.Tab(label='Key Insights', tab_id='insights'),
            dbc.Tab(label='Appendix', tab_id='appendix'),
            dbc.Tab(label='Ask the Data', tab_id='ask'),
        ], id='tabs', active_tab='overview'),
        html.Button('🌙', id='theme-toggle', className='theme-toggle',
                    n_clicks=0, title='Toggle dark mode',
                    style={'display': 'none'}),
    ])),
    # Lazy loading: all tab containers exist but only the active one is visible.
    # Content is built on first visit, then cached in the DOM (no rebuild on tab switch).
    dcc.Store(id='tabs-loaded', data={'overview': False, 'explore': False,
                                      'social_progress': False, 'insights': False,
                                      'appendix': False, 'ask': False}),
    dcc.Store(id='fw-scroll-dummy'),
    html.Div(id='tab-overview', className='page-wrap', style={'display': 'block'},
             children=[html.Div(className='skeleton')]),
    html.Div(id='tab-explore', className='page-wrap', style={'display': 'none'},
             children=[html.Div(className='skeleton')]),
    html.Div(id='tab-social-progress', className='page-wrap', style={'display': 'none'},
             children=[html.Div(className='skeleton')]),
    html.Div(id='tab-insights', className='page-wrap', style={'display': 'none'},
             children=[html.Div(className='skeleton')]),
    html.Div(id='tab-appendix', className='page-wrap', style={'display': 'none'},
             children=[html.Div(className='skeleton')]),
    html.Div(id='tab-ask', className='page-wrap', style={'display': 'none'},
             children=[html.Div(className='skeleton')]),
])

# Dark mode toggle — switches data-theme attribute on the root element
app.clientside_callback(
    """
    function(n) {
        var root = document.getElementById('app-root');
        if (!root) return '☀️';
        var dark = root.getAttribute('data-theme') === 'dark';
        root.setAttribute('data-theme', dark ? '' : 'dark');
        return dark ? '🌙' : '☀️';
    }
    """,
    Output('theme-toggle', 'children'),
    Input('theme-toggle', 'n_clicks'),
    prevent_initial_call=True,
)

# Clientside callback for instant tab visibility toggle (no server roundtrip)
app.clientside_callback(
    """
    function(tab) {
        var tabs = ['overview', 'explore', 'social_progress', 'insights', 'appendix', 'ask'];
        var styles = [];
        for (var i = 0; i < tabs.length; i++) {
            styles.push(tab === tabs[i] ? {display: 'block'} : {display: 'none'});
        }
        return styles;
    }
    """,
    [Output('tab-overview', 'style'),
     Output('tab-explore', 'style'),
     Output('tab-social-progress', 'style'),
     Output('tab-insights', 'style'),
     Output('tab-appendix', 'style'),
     Output('tab-ask', 'style')],
    Input('tabs', 'active_tab')
)


@callback(Output('tab-overview', 'children'),
          Output('tab-explore', 'children'),
          Output('tab-social-progress', 'children'),
          Output('tab-insights', 'children'),
          Output('tab-appendix', 'children'),
          Output('tab-ask', 'children'),
          Output('tabs-loaded', 'data'),
          Input('tabs', 'active_tab'),
          State('tabs-loaded', 'data'))
def render_tab(tab, loaded):
    """Build tab content only on first visit — subsequent switches are instant
    because the DOM is already populated (just hidden/shown via clientside)."""
    ov = ex = sp = ins = ap = ask = no_update
    new_loaded = loaded.copy() if loaded else {'overview': False, 'explore': False,
                                               'social_progress': False,
                                               'insights': False, 'appendix': False,
                                               'ask': False}

    if tab == 'overview' and not new_loaded.get('overview'):
        ov = build_overview_tab()
        new_loaded['overview'] = True
    elif tab == 'explore' and not new_loaded.get('explore'):
        ex = build_explore_tab()
        new_loaded['explore'] = True
    elif tab == 'social_progress' and not new_loaded.get('social_progress'):
        sp = build_social_progress_tab()
        new_loaded['social_progress'] = True
    elif tab == 'insights' and not new_loaded.get('insights'):
        ins = build_insights_tab()
        new_loaded['insights'] = True
    elif tab == 'appendix' and not new_loaded.get('appendix'):
        ap = build_appendix_tab()
        new_loaded['appendix'] = True
    elif tab == 'ask' and not new_loaded.get('ask'):
        ask = build_ask_tab()
        new_loaded['ask'] = True

    return ov, ex, sp, ins, ap, ask, new_loaded


# =============================================================== overview ==

def build_hero():
    def chip(word, var, desc):
        # --pillar drives the chip's top edge, dot halo and hover glow in CSS,
        # so each of Live / Thrive / Connect carries its own colour identity.
        return html.Div([
            html.Div([
                html.Span(className='p-dot', style={'backgroundColor': f'var({var})'}),
                html.B(word)]),
            html.Div(desc, className='pillar-chip-desc'),
        ], className='pillar-chip', style={'--pillar': f'var({var})'})

    return html.Div(className='hero', children=[
        # Exhibition theme ribbon — the three pillar words in their own
        # colours, so the theme reads at a glance before any chart loads.
        html.Div([
            html.Span(className='tr-pulse'),
            html.Span('How the World'),
            html.Span('Lives', className='tr-word w-live'),
            html.Span('·', className='tr-sep'),
            html.Span('Thrives', className='tr-word w-thrive'),
            html.Span('·', className='tr-sep'),
            html.Span('Connects', className='tr-word w-connect'),
        ], className='theme-ribbon'),
        html.H1('DOES MONEY BUY SOCIAL PROGRESS?', className='hero-title',
                style={'textAlign': 'center'}),
        html.P('What does it actually mean for a society to progress? GDP tells us '
               'how much money flows through an economy — but it says nothing about '
               'whether people can breathe clean air, access education, or live '
               'without fear. This visualization explores that gap.',
               className='hero-lede'),
        html.P([html.B('Social Progress Index (SPI)'),
                ' — is a framework that measures the real quality of life of a '
                'country, entirely independent of economic output. The SPI asks '
                'three fundamental questions:'],
               className='hero-lede'),
        html.Div([
            chip('Live', '--live', 'Are basic human needs met? Can people survive?'),
            chip('Thrive', '--thrive', 'Can people build better lives?'),
            chip('Connect', '--connect', 'Can people participate freely in society?'),
        ], className='pillar-chips'),
        html.Div(className='kpi-row', children=[
            stat_tile('Countries', f'{len(countries)}',
                      note=f'{len(regions)} world regions'),
            stat_tile('Years', f'{EARLIEST}–{LATEST}',
                      note=f'{len(years)} annual editions'),
            stat_tile('Indicators', f'{N_INDICATORS}', note='scored 0–100'),
            stat_tile('Datasets', '2',
                      note='Social Progress Index and World Happiness Index'),
        ]),
        html.P(['This dashboard reveals a provocative truth: wealth and wellbeing '
                'are correlated, but far from synonymous. The question isn\u2019t '
                'just \u201chow rich is a country?\u201d \u2014 it\u2019s \u201chow well '
                'does a society convert its resources into real outcomes for its '
                'people?\u201d That\u2019s the story these visualizations tell.'],
               className='hero-lede',
               style={'borderTop': '1px solid var(--grid)', 'paddingTop': '16px',
                      'marginTop': '8px'}),
    ])


def build_key_insights():
    """Surprising comparative facts, computed from the data."""
    latest = df[df['SPI year'] == LATEST]
    top = latest.nlargest(1, SPI_COL).iloc[0]
    bot = latest.nsmallest(1, SPI_COL).iloc[0]
    facts = [('The Leader',
              f"{flag(top['Country'])} {top['Country']} leads the world with an "
              f"SPI of {top[SPI_COL]:.1f}",
              f"Gap from last ({flag(bot['Country'])} {bot['Country']}): "
              f"{top[SPI_COL] - bot[SPI_COL]:.0f} points")]

    qatar = latest[latest['Country'] == 'Qatar']
    finland = latest[latest['Country'] == 'Finland']
    if not qatar.empty and not finland.empty:
        q, f = qatar.iloc[0], finland.iloc[0]
        facts.append(('Money ≠ progress',
                      f"{flag('Qatar')} Qatar’s GDP is "
                      f"{q[GDP_COL] / f[GDP_COL]:.1f}× {flag('Finland')} Finland’s",
                      f"Yet Finland’s SPI is {f[SPI_COL] - q[SPI_COL]:.0f} points higher"))
    cr = latest[latest['Country'] == 'Costa Rica']
    if not cr.empty and not qatar.empty:
        c, q = cr.iloc[0], qatar.iloc[0]
        facts.append(('Efficiency',
                      f"{flag('Costa Rica')} Costa Rica outscores "
                      f"{flag('Qatar')} Qatar on SPI",
                      f"With {q[GDP_COL] / c[GDP_COL]:.0f}× less GDP "
                      f"({c[SPI_COL]:.1f} vs {q[SPI_COL]:.1f})"))
    eu = latest[latest['Region'] == 'Europe'][SPI_COL].mean()
    afr = latest[latest['Region'] == 'Sub-Saharan Africa'][SPI_COL].mean()
    facts.append(('Regions',
                  'Europe leads all regions in SPI and happiness',
                  f"Average SPI of {eu:.0f} — nearly double Sub-Saharan Africa’s {afr:.0f}"))
    facts.append(('The paradox',
                  f"{flag('Mexico')} Mexico is #12 happiest but #75 in Social Progress",
                  'People report high happiness despite lower systemic progress'))

    return html.Div(className='viz-card', children=[
        head_block('Key Insights', 'What the Data Says'),
        html.Div(className='insight-grid', children=[
            html.Div([
                html.Div(k, className='insight-kicker'),
                html.Div(m, className='insight-main'),
                html.Div(s, className='insight-sub'),
            ], className='insight-tile') for k, m, s in facts]),
    ])


def _anchor(text):
    """Slugify a component name into a safe HTML id fragment."""
    return ''.join(c.lower() if c.isalnum() else '-' for c in text).strip('-')


def build_framework():
    dim_vars = {'Basic Needs': '--live',
                'Foundations of Wellbeing': '--thrive',
                'Societal Opportunity': '--connect'}
    tags = {'Basic Needs': 'LIVE', 'Foundations of Wellbeing': 'THRIVE',
            'Societal Opportunity': 'CONNECT'}
    cols = []
    for dim_name, data in DIMENSIONS.items():
        var = f'var({dim_vars[dim_name]})'
        comps = []
        for comp, indicators in data['raw_indicators'].items():
            comps.append(html.Button(
                id={'type': 'fw-comp-btn', 'index': _anchor(comp)}, n_clicks=0,
                className='fw-comp', title=f'View {comp} indicators in the Appendix',
                children=[
                    html.Div([html.Span(className='fw-dot',
                                        style={'backgroundColor': var}), comp],
                             className='fw-comp-name'),
                    html.Div(className='fw-inds', children=[
                        html.Span(ind.split(' (')[0].strip(), className='fw-ind')
                        for ind, _ in indicators]),
                ]))
        cols.append(html.Div(className='pillar-col', children=[
            html.Div([html.Span(dim_name.upper()),
                      html.Span(tags[dim_name], className='pillar-tag',
                                style={'color': var})],
                     className='pillar-name',
                     style={'borderBottomColor': var}),
            html.Div(comps),
        ]))
    return html.Div(className='viz-card', children=[
        head_block('The Framework', 'The Social Progress Index',
                   f'3 dimensions · 12 components · {N_INDICATORS} scored indicators.'),
        html.Div(cols, className='framework-grid'),
        html.Div('Click any component to jump to its full indicator glossary '
                 'in the Appendix tab.', className='stat-note',
                 style={'marginTop': '8px'}),
        dcc.Store(id='fw-jump-target'),
    ])


@callback(Output('tabs', 'active_tab', allow_duplicate=True),
          Output('sp-sub-view', 'value', allow_duplicate=True),
          Input({'type': 'next-tab-btn', 'index': ALL}, 'n_clicks'),
          State({'type': 'next-tab-target', 'index': ALL}, 'data'),
          prevent_initial_call=True)
def jump_to_next_tab(_clicks, targets):
    """Handles tab switches and Progress sub-view switches. Overview sub-view
    switches are handled by the dedicated per-button callbacks below."""
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not ctx.triggered[0].get('value'):
        return no_update, no_update
    triggered_idx = trig.get('index', '')
    if triggered_idx.startswith('ov-'):
        return no_update, no_update
    for entry in ctx.states_list[0]:
        if entry['id'].get('index') == triggered_idx:
            target = entry.get('value')
            if not target:
                break
            kind, _, value = target.partition(':')
            if kind == 'tab':
                return value, no_update
            if kind == 'subview':
                return no_update, value
            break
    return no_update, no_update


# "Back to Top" nudge buttons — pure client-side smooth scroll, no server
# roundtrip needed.
app.clientside_callback(
    """
    function(clicks) {
        window.scrollTo({top: 0, behavior: 'smooth'});
        return '';
    }
    """,
    Output({'type': 'next-tab-top-btn', 'index': ALL}, 'title'),
    Input({'type': 'next-tab-top-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)


@callback(Output('tabs', 'active_tab', allow_duplicate=True),
          Output('fw-jump-target', 'data'),
          Input({'type': 'fw-comp-btn', 'index': ALL}, 'n_clicks'),
          prevent_initial_call=True)
def jump_to_appendix(_clicks):
    """Clicking a framework component switches to the Appendix tab and
    records which component to scroll to once it's rendered."""
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not ctx.triggered[0].get('value'):
        return no_update, no_update
    return 'appendix', trig['index']


# Once the Appendix tab has rendered (its anchors exist in the DOM), scroll
# the targeted component into view. A short delay covers the render tick.
app.clientside_callback(
    """
    function(target, tabsLoaded) {
        if (!target || !tabsLoaded || !tabsLoaded.appendix) {
            return window.dash_clientside.no_update;
        }
        setTimeout(function () {
            var el = document.getElementById('appx-anchor-' + target);
            if (el) {
                // Auto-open the details disclosure inside this card
                var details = el.querySelector('details');
                if (details) { details.open = true; }
                el.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        }, 150);
        return '';
    }
    """,
    Output('fw-scroll-dummy', 'data'),
    Input('fw-jump-target', 'data'),
    Input('tabs-loaded', 'data'),
    prevent_initial_call=True,
)


def build_overview_conclusion():
    return html.Div(className='viz-card', children=[
        head_block('Conclusions', None),
        html.Div(className='prose prose-wide', children=[
            html.Ul([
                html.Li('The world has made measurable progress over 15 years, but '
                        'vast inequalities persist.'),
                html.Li('Basic Needs are the most evenly distributed dimension; '
                        'Opportunity remains the widest gap between regions.'),
                html.Li("India overtook China as the world's most populous country "
                        'in 2022 — China’s population has since plateaued, while '
                        'India continues to grow (1.45B in 2025).'),
                html.Li("Guyana's GDP per capita grew 6.8× between 2011 and 2025, "
                        'the fastest of any country — largely driven by its oil boom.'),
                html.Li("Oman's population grew 91.5% between 2011 and 2025, the "
                        'fastest of any country with over a million people — Gulf '
                        'states (Oman, Qatar, Kuwait) dominate the fastest-growing '
                        'list, driven by labor migration.'),
                html.Li("Ukraine's population fell 18.5% since 2011, the steepest "
                        'decline of any country — a combination of war, '
                        'displacement, and outmigration.'),
                html.Li('Singapore, Luxembourg, and Ireland lead the world in GDP '
                        'per capita, all exceeding $115,000 — small, highly '
                        'globalized economies dominate the wealth rankings.'),
                html.Li('Burundi and Central African Republic remain the world’s '
                        'poorest by GDP per capita, both under $1,200 — virtually '
                        'unchanged in relative position over the 15-year record.'),
            ]),
        ]),
    ])


@lru_cache(maxsize=32)
def _dims_strip(year):
    """Cached figure for Overview's "Dimension Distributions" year slider.
    Every other year-driven chart in the app (bubble, GDP scatter, movers,
    happiness) is cached the same way — this one wasn't, so every tick paid
    a fresh groupby + 3-panel figure rebuild (~43ms) plus an unconditional
    data-table render, instead of being instant on repeat/scrub."""
    dff = df[df['SPI year'] == year]
    return C.strip_chart(
        dff, ['Basic Needs', 'Foundations of Wellbeing', 'Opportunity'],
        None, year, THEME,
        titles=['Basic Needs', 'Foundations of Wellbeing', 'Societal Opportunity'])


def build_overview_tab():
    region_trend_fig, region_trend_table = C.region_spi_trend(THEME)
    return html.Div([
        build_hero(),
        build_framework(),
        html.Div(className='viz-card', children=[
            head_block('Regional Trends', 'Social Progress Index (SPI) Over Time by Region',
                       'Mean Social Progress Index per region, every year on record.'),
            graph(region_trend_fig),
            how_to_read([
                'Each line is one world region; its height is the average SPI '
                '(0\u2013100, higher is better) of all countries in that region.',
                'Follow a line left to right to see whether that region\u2019s '
                'social progress has risen or stalled over 2011\u20132025.',
                'Lines that stay far apart show a persistent gap between '
                'regions; lines converging would mean the gap is closing.']),
            data_view(region_trend_table)]),
        html.Div(className='filter-bar', children=[
            html.Div([html.Div('View', className='control-label'),
                      segmented('ov-view', [
                          {'label': 'Dimension Distributions', 'value': 'dims'},
                          {'label': 'Population', 'value': 'pop'},
                          {'label': 'GDP per Capita', 'value': 'gdp'},
                      ], 'dims')], className='control-group'),
        ]),
        html.Div(id='ov-body-1', style={'minHeight': '500px'}, children=[
            html.Div(className='viz-card', children=[
                html.Div(id='ov-body-1-head'),
                html.Div(id='ov-dims-timeline-wrap', children=timeline('ov'),
                         style={'marginBottom': '12px'}),
                dcc.Graph(id='ov-graph-1', figure={}, config=GRAPH_CONFIG),
                html.Div([html.Div('Regions', className='control-label'),
                          region_chips()], id='ov-region-chips-wrap',
                         className='control-group',
                         style={'marginTop': '12px'}),
                html.Div(id='ov-htr'),
                html.Div(id='ov-body-1-table'),
            ]),
        ]),
        html.Div(id='ov-body-2', style={'minHeight': '500px'}, children=[
            html.Div(className='viz-card', children=[
                html.Div(id='ov-body-2-head'),
                html.Div(id='ov-top25-timeline-wrap', children=timeline('ov25'),
                         style={'marginBottom': '12px'}),
                dcc.Graph(id='ov-graph-2', figure={}, config=GRAPH_CONFIG),
                how_to_read([
                    'A ranked list of the 25 leading countries \u2014 longest bar '
                    'at the top (#1), shortest at the bottom (#25).',
                    'The colored dot is the value for the selected year; the grey '
                    'dot marks the 2011 baseline, so the gap between them shows how '
                    'far the country has moved.',
                    'Press play or drag the year slider to watch the ranking '
                    'reshuffle over time.']),
                html.Div(id='ov-body-2-table'),
            ]),
        ]),
        next_tab_nudge('ov-dims', 'Population view', 'subview-ov:pop',
                      wrapper_id='ov-nudge-dims'),
        next_tab_nudge('ov-pop', 'GDP per Capita view', 'subview-ov:gdp',
                      wrapper_id='ov-nudge-pop',
                      style={'display': 'none'}),
        next_tab_nudge('ov-gdp', 'Dimension Distributions view', 'subview-ov:dims',
                      wrapper_id='ov-nudge-gdp',
                      style={'display': 'none'}),
        # View-specific takeaway — one per Overview sub-view, toggled by the
        # same clientside callback that shows/hides the nudges.
        html.Div(id='ov-takeaway-dims', className='viz-card',
                 style={'padding': '20px 28px'}, children=[
            html.Div('Takeaway', className='kicker'),
            html.P('Look at how tightly the dots cluster on each strip above. '
                   'Basic Needs bunches near the top (mean 73) \u2014 most countries '
                   'have nutrition, water, and shelter largely solved. Opportunity '
                   'is the opposite: dots scatter across the entire strip, from 14 '
                   'to 91. Rights, freedom, and inclusion are where countries at the '
                   'same income level still land in very different places.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
        ]),
        html.Div(id='ov-takeaway-pop', className='viz-card',
                 style={'padding': '20px 28px', 'display': 'none'}, children=[
            html.Div('Takeaway', className='kicker'),
            html.P('Population is extraordinarily concentrated. India (1.45B) and '
                   'China (1.41B) alone hold about 35% of humanity \u2014 more than '
                   'the next 20 countries combined. China peaked in 2022 and is now '
                   'shrinking, while India keeps climbing, so the world\u2019s two '
                   'demographic giants are now moving in opposite directions. Where '
                   'people live, though, tells you little about how well they live: '
                   'the ten most populous countries range from an SPI of 44 '
                   '(Ethiopia) to 82 (United States) \u2014 a 38-point spread, with '
                   'no relationship between size and social progress.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
        ]),
        html.Div(id='ov-takeaway-gdp', className='viz-card',
                 style={'padding': '20px 28px', 'display': 'none'}, children=[
            html.Div('Takeaway', className='kicker'),
            html.P('The income gap between nations is staggering: Singapore '
                   '($133K per capita) is roughly 159\u00d7 richer than Burundi '
                   '($0.8K). Wealth is dominated by small, finance- and '
                   'trade-driven economies \u2014 Singapore, Luxembourg, Ireland, '
                   'Qatar \u2014 rather than the largest ones. The log scale is '
                   'telling: most of the world sits at the low end, and only a '
                   'handful of countries reach the deep-green top. Whether that '
                   'wealth translates into social progress is the question the '
                   'rest of this story explores.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
        ]),
    ])


make_play_callbacks('ov')
make_play_callbacks('ov25')

# Toggle which year slider is visible: "dims" shows the top-filter-bar slider,
# "pop"/"gdp" show the one next to the Top 25 chart instead.
app.clientside_callback(
    """
    function(view) {
        var showDims = view === 'dims';
        var showPop = view === 'pop';
        var showGdp = view === 'gdp';
        // Scroll the chart area into view when switching sub-views so the
        // user sees the new chart immediately, not the bottom of the old one.
        var body1 = document.getElementById('ov-body-1');
        if (body1) { body1.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        var cardPad = {padding: '20px 28px'};
        var hidden = {display: 'none'};
        return [showDims ? {} : hidden,
                showDims ? hidden : {},
                showDims ? {marginTop: '12px'} : hidden,
                showDims ? {} : hidden,
                showPop ? {} : hidden,
                showGdp ? {} : hidden,
                showDims ? cardPad : hidden,
                showPop ? cardPad : hidden,
                showGdp ? cardPad : hidden];
    }
    """,
    Output('ov-dims-timeline-wrap', 'style'),
    Output('ov-top25-timeline-wrap', 'style'),
    Output('ov-region-chips-wrap', 'style'),
    Output('ov-nudge-dims', 'style'),
    Output('ov-nudge-pop', 'style'),
    Output('ov-nudge-gdp', 'style'),
    Output('ov-takeaway-dims', 'style'),
    Output('ov-takeaway-pop', 'style'),
    Output('ov-takeaway-gdp', 'style'),
    Input('ov-view', 'value'),
)

# Direct callbacks for the Overview sub-view switch nudge buttons.
# These bypass the pattern-matching jump_to_next_tab callback entirely —
# simpler and more reliable.
@callback(Output('ov-view', 'value', allow_duplicate=True),
          Input({'type': 'next-tab-btn', 'index': 'ov-dims'}, 'n_clicks'),
          prevent_initial_call=True)
def nudge_to_pop(n):
    return 'pop' if n else no_update


@callback(Output('ov-view', 'value', allow_duplicate=True),
          Input({'type': 'next-tab-btn', 'index': 'ov-pop'}, 'n_clicks'),
          prevent_initial_call=True)
def nudge_to_gdp(n):
    return 'gdp' if n else no_update


@callback(Output('ov-view', 'value', allow_duplicate=True),
          Input({'type': 'next-tab-btn', 'index': 'ov-gdp'}, 'n_clicks'),
          prevent_initial_call=True)
def nudge_to_dims(n):
    return 'dims' if n else no_update


@callback(Output('exp-view', 'value', allow_duplicate=True),
          Input({'type': 'next-tab-btn', 'index': 'exp-to-region'}, 'n_clicks'),
          prevent_initial_call=True)
def nudge_to_region_view(n):
    return 'region' if n else no_update


@callback(Output('ov-graph-1', 'figure'),
          Output('ov-body-1-head', 'children'),
          Output('ov-body-1-table', 'children'),
          Input('ov-view', 'value'), Input('ov-year', 'value'),
          Input('ov-play-interval', 'disabled'))
def update_overview_map(view, year, paused):
    """Prop-only updates — no DOM rebuild, no glitch."""
    if view == 'dims':
        year = year if year in years else LATEST
        fig, table = _dims_strip(year)   # cached — instant on repeat/scrub
        return (fig,
                head_block('Distributions', f'The Three Pillars, Country by Country ({year})',
                           'Each dot is a country\u2019s score (0\u2013100) on that dimension.'),
                data_view(table) if paused else no_update)
    dff = df[df['SPI year'] == LATEST]
    if view == 'pop':
        fig, table = C.population_map(dff, LATEST, THEME)
        return (fig,
                head_block('Population', 'Where People Live',
                           'Each shade step represents roughly a 10\u00d7 difference in population size.'),
                data_view(table))
    if view == 'gdp':
        fig, table = C.gdp_map(dff, LATEST, THEME)
        return (fig,
                head_block('Wealth', 'GDP per Capita Around the World',
                           'GDP per capita on a log scale, showing the vast income divide between nations.'),
                data_view(table))
    return {}, [], []


# "How to read" text for the Overview map/strip views, swapped by view.
OV_HTR = {
    'dims': ['Three vertical strips \u2014 Basic Needs, Foundations of Wellbeing, '
             'and Opportunity. Each dot is one country.',
             'Higher up = a better score (0\u2013100) on that pillar. Dot color '
             'is the country\u2019s region.',
             'Tightly bunched dots mean countries score similarly; a wide '
             'vertical spread means big gaps between countries on that pillar.'],
    'pop': ['A world map shaded by population \u2014 darker means more people.',
            'The shading is on a log scale: each step up represents roughly a '
            '10\u00d7 jump in population, so India and China stand out sharply.',
            'Hover any country for its exact population.'],
    'gdp': ['A world map shaded by GDP per capita \u2014 deeper green means '
            'higher average income.',
            'The scale is logarithmic, so equal color steps represent equal '
            '*multiples* of income, not equal dollar amounts.',
            'Grey countries have no data for this year.'],
}


@callback(Output('ov-htr', 'children'),
          Input('ov-view', 'value'))
def update_ov_htr(view):
    return how_to_read(OV_HTR.get(view, OV_HTR['dims']))


@callback(Output('ov-graph-2', 'figure'),
          Output('ov-body-2-head', 'children'),
          Output('ov-body-2-table', 'children'),
          Output('ov-body-2', 'style'),
          Input('ov-view', 'value'), Input('ov25-year', 'value'))
def update_overview_top25(view, year):
    """Prop-only updates for the Top 25 chart."""
    year = year if year in years else LATEST
    dff = df[df['SPI year'] == year]
    if view == 'pop':
        fig, table = C.population_bar(dff, None, year, THEME)
        return (fig,
                head_block('Population', f'Top 25 Most Populous Countries ({year})',
                           'Watch how the order shifts as you slide through the years.'),
                data_view(table), {'minHeight': '500px'})
    if view == 'gdp':
        fig, table = C.gdp_bar(dff, None, year, THEME)
        return (fig,
                head_block('Wealth', f'Top 25 Countries by GDP per Capita ({year})',
                           'Watch how the order shifts as you slide through the years.'),
                data_view(table), {'minHeight': '500px'})
    return {}, [], [], {'display': 'none'}


# ================================================================ explore ==

DIM_OPTIONS = [{'label': 'Basic Needs · Live', 'value': 'Basic Needs'},
               {'label': 'Wellbeing · Thrive', 'value': 'Foundations of Wellbeing'},
               {'label': 'Opportunity · Connect', 'value': 'Societal Opportunity'}]

# Dimension-specific narrative for region analysis and country deep dive
DIM_NARRATIVE = {
    'Basic Needs': {
        'region': 'How well each region meets its population\u2019s survival needs \u2014 '
                  'nutrition, water, shelter, and safety scored 0\u2013100.',
        'country': 'A single country\u2019s Basic Needs profile \u2014 select a country '
                   'to see where it excels and where it falls short.',
    },
    'Foundations of Wellbeing': {
        'region': 'Which regions give their people the tools to build better lives — '
                  'education, health, information access, and environmental quality '
                  'compared side by side.',
        'country': 'One country\u2019s capacity to help people develop — how it stacks '
                   'up on schooling, connectivity, health outcomes, and air quality '
                   'against regional and global benchmarks.',
    },
    'Societal Opportunity': {
        'region': 'Where in the world people can participate freely — rights, personal '
                  'freedom, inclusivity, and access to advanced education scored by region.',
        'country': 'A country\u2019s openness and inclusion — how much freedom, voice, '
                   'and opportunity its people actually have relative to the rest of '
                   'the world.',
    },
}


def build_explore_tab():
    return html.Div([
        html.Div(id='exp-filter-bar', className='filter-bar collapsible',
                 style={'position': 'sticky', 'top': '56px', 'zIndex': 1020},
                 children=[
            html.Button('⌃ Hide filters', id='exp-collapse-btn', n_clicks=0,
                        className='filter-collapse-btn', title='Collapse filters'),
            html.Div([html.Div('Dimension', className='control-label'),
                      segmented('exp-dim', DIM_OPTIONS, 'Basic Needs')],
                     className='control-group'),
            html.Div([html.Div('View', className='control-label'),
                      segmented('exp-view', [
                          {'label': 'Country Deep Dive', 'value': 'country'},
                          {'label': 'Region Analysis', 'value': 'region'},
                      ], 'country')], className='control-group'),
            html.Div([html.Div('Country', className='control-label'),
                      dcc.Dropdown(id='exp-country',
                                   options=[{'label': f'{flag(c)} {c}'.strip(),
                                             'value': c} for c in countries],
                                   value=None, placeholder='Select a country…',
                                   clearable=True, style={'minWidth': '240px'})],
                     id='exp-country-wrap', className='control-group'),
            html.Div(id='exp-dim-narrative', className='control-group full-row',
                     style={'marginTop': '4px'}),
            timeline('exp', note='Controls both regional drill-down and country deep dive.',
                    wrapper_id='exp-time-wrap'),
            # "You are here" pill inside the sticky bar so it stays visible
            # even when filters are collapsed or you scroll down.
            html.Div(id='exp-here', className='here-pill',
                     style={'flex': '1 1 100%', 'margin': '4px 0 0'}),
            html.Div([html.Div('Regions', className='control-label'),
                      region_chips()], className='control-group grow full-row'),
        ]),
        dcc.Store(id='exp-comp'),
        html.Div(id='exp-content'),
    ])


@callback(Output('exp-here', 'children'),
          Input('exp-dim', 'value'), Input('exp-view', 'value'),
          Input('exp-country', 'value'), Input('exp-year', 'value'))
def update_here_pill(dim, view, country, year):
    year = year if year in years else LATEST
    dim_label = next((o['label'] for o in DIM_OPTIONS if o['value'] == dim), dim)
    parts = [html.Span('You are here:', className='here-label'),
             html.Span(dim_label, className='here-chip here-chip--dim')]
    if view == 'region':
        parts.append(html.Span('Region Analysis', className='here-chip'))
        if country:
            parts.append(html.Span([f'{flag(country)} {country} highlighted'],
                                   className='here-chip here-chip--country'))
    else:
        parts.append(html.Span('Country Deep Dive', className='here-chip'))
        parts.append(html.Span(
            [f'{flag(country)} {country}'] if country else 'World (no country selected)',
            className='here-chip here-chip--country'))
    parts.append(html.Span(str(year), className='here-chip here-chip--year'))
    return parts


make_play_callbacks('exp')

# Manual collapse toggle for the Explore filter bar — folds everything except
# the year slider so it stops eating half the viewport. Clientside so it's
# instant and never rebuilds the DOM.
app.clientside_callback(
    """
    function(n) {
        var bar = document.getElementById('exp-filter-bar');
        var collapsed = (n || 0) % 2 === 1;
        if (bar) { bar.classList.toggle('manual-collapsed', collapsed); }
        return collapsed ? '⌄ Show filters' : '⌃ Hide filters';
    }
    """,
    Output('exp-collapse-btn', 'children'),
    Input('exp-collapse-btn', 'n_clicks'),
    prevent_initial_call=True,
)


@callback(Output('exp-dim-narrative', 'children'),
          Input('exp-dim', 'value'), Input('exp-view', 'value'))
def update_dim_narrative(dim, view):
    """Show dimension-specific narrative text before the timeline."""
    narr = DIM_NARRATIVE.get(dim, {})
    dim_label = {'Basic Needs': 'Basic Needs · Live',
                 'Foundations of Wellbeing': 'Foundations of Wellbeing · Thrive',
                 'Societal Opportunity': 'Societal Opportunity · Connect'}.get(dim, dim)
    region_text = narr.get('region', '')
    country_text = narr.get('country', '')
    if view == 'region':
        desc = region_text
    else:
        desc = country_text
    return [html.Div(dim_label, style={'fontWeight': '700', 'fontSize': '13px',
                                        'color': 'var(--ink)', 'marginBottom': '2px'}),
            html.Div(desc, className='card-sub')]


@callback(Output('exp-comp', 'data'),
          Input({'type': 'comp-btn', 'index': ALL}, 'n_clicks'),
          Input('exp-dim', 'value'),
          State('exp-comp', 'data'),
          prevent_initial_call=False)
def select_component(clicks, dim, current):
    comps = DIMENSIONS[dim]['components']
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get('type') == 'comp-btn':
        # ignore the phantom fire when buttons are (re)created (n_clicks 0/None)
        if ctx.triggered and ctx.triggered[0].get('value'):
            return trig['index']
        return no_update if current in comps else comps[0]
    if trig == 'exp-dim' or current not in comps:
        return comps[0]
    return no_update


@callback(Output('exp-content', 'children'), Input('exp-view', 'value'))
def render_explore_scaffold(view):
    """Static scaffolding per view — chart bodies update in place, so
    interacting with a chart never rebuilds (or re-scrolls) the page."""
    if view == 'region':
        return html.Div([
            html.Div(className='viz-card', children=html.Div(id='exp-region-body')),
            html.Div(className='viz-card', children=[
                html.Div(id='exp-drill-head'),
                html.Div(id='exp-comp-cards', className='comp-cards'),
                html.Div(id='exp-drill-body'),
            ]),
        ])
    # Country deep dive — fully static scaffold with fixed graph IDs.
    # Callbacks update figure props + text in place (never rebuild the DOM),
    # so ticking the year slider or hitting play stays smooth, exactly like
    # the Overview maps.
    return html.Div([
        html.Div(className='viz-card', children=[
            html.Div(id='dd-head'),
            html.Div(id='dd-tiles'),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div('Population vs GDP Over Time', className='card-title',
                             style={'fontSize': '14px', 'marginBottom': '2px'}),
                    html.Div('How population and income have moved together \u2014 '
                             'the selected year is marked.',
                             className='card-sub'),
                    dcc.Graph(id='dd-trend-graph', figure={}, config=GRAPH_CONFIG),
                    how_to_read([
                        'Two lines on a shared timeline: green = population (left '
                        'axis, millions), the other = GDP per capita (right axis, '
                        '$K). Each axis has its own scale.',
                        'The dotted vertical line and the enlarged dots mark the '
                        'year you\u2019ve selected on the slider.',
                        'Rising lines mean the country grew on that measure; if the '
                        'two move together, population and income rose in step.']),
                    html.Div(id='dd-trend-table'),
                ]), md=7),
                dbc.Col(html.Div(id='dd-locator-col', children=[
                    html.Div(id='dd-locator-title', className='card-title',
                             style={'fontSize': '14px', 'marginBottom': '2px'}),
                    html.Div(id='dd-locator-sub', className='card-sub'),
                    dcc.Graph(id='dd-locator-graph', figure={}, config=GRAPH_CONFIG),
                    how_to_read('A locator map highlighting where this country '
                                'sits in the world. Tiny countries get a labelled '
                                'pin and a zoomed-in view so they\u2019re visible.'),
                ]), md=5),
            ], align='center', className='g-4'),
            html.Div(style={'marginTop': '8px'}, children=[
                html.Div(id='dd-comp-title', className='card-title',
                         style={'fontSize': '14px'}),
                html.Div('Component scores side by side — the line spans the gap '
                         'from world average to this country.', className='card-sub'),
                dcc.Graph(id='dd-comparison-graph', figure={}, config=GRAPH_CONFIG),
                how_to_read([
                    'One row per component of this dimension. On each row: the grey '
                    'dot is the world average, the light dot is the region average, '
                    'and the colored dot is this country.',
                    'The connecting line spans the gap from the world average to '
                    'the country \u2014 a long line means this country is far from '
                    'the global norm on that component.',
                    'Further right = a higher score (0\u2013100).']),
                html.Div(id='dd-comp-table'),
            ]),
        ]),
        html.Div(id='dd-ind-wrap', children=[
            html.Div(className='viz-card', children=[
                html.Div(id='dd-gdpspi-head'),
                dcc.Graph(id='dd-gdpspi-graph', figure={}, config=GRAPH_CONFIG),
                how_to_read([
                    'Two lines over time: one is SPI (left axis), the other is GDP '
                    'per capita (right axis, $K).',
                    'Watching them together shows whether rising income has been '
                    'matched by rising social progress \u2014 or whether one has '
                    'outpaced the other.']),
                html.Div(id='dd-gdpspi-table'),
            ]),
            html.Div(className='viz-card', children=[
                html.Div(id='dd-ind-head'),
                dcc.Graph(id='dd-indicator-graph', figure={}, config=GRAPH_CONFIG),
                how_to_read([
                    'Each panel is one indicator. The bar/marker shows this '
                    'country\u2019s world percentile \u2014 how it ranks against '
                    'every other country, not a raw value.',
                    'Green = top tier (\u2265 70th percentile), amber = middle '
                    '(40\u201370), red = bottom (< 40).',
                    'A high percentile means the country handles that specific '
                    'challenge better than most of the world.']),
                html.Div(className='tier-legend', children=[
                    html.Span([html.Span(className='c-dot',
                                         style={'backgroundColor': '#0ca30c'}),
                               'Top tier (≥ 70th percentile)'], className='chip'),
                    html.Span([html.Span(className='c-dot',
                                         style={'backgroundColor': '#fab219'}),
                               'Middle (40–70)'], className='chip'),
                    html.Span([html.Span(className='c-dot',
                                         style={'backgroundColor': '#d03b3b'}),
                               'Bottom (< 40)'], className='chip'),
                ]),
                html.Div(id='dd-ind-table'),
            ]),
        ]),
        next_tab_nudge('exp-to-region', 'Regional Analysis', 'exp-view:region',
                      wrapper_id='exp-nudge-region'),
    ])


@callback(Output('exp-region-body', 'children'),
          Input('exp-dim', 'value'), Input('exp-year', 'value'),
          Input('exp-country', 'value'),
          Input('exp-play-interval', 'disabled'))
def update_region_analysis(dim, year, country, paused):
    year = year if year in years else LATEST
    dff = df[df['SPI year'] == year]
    fig, table = C.region_analysis(dff, DIMENSIONS[dim]['components'],
                                   country, year, dim, THEME)
    kids = [head_block('Region Analysis', f'{dim} — Regional Comparison',
                       DIM_NARRATIVE.get(dim, {}).get('region',
                       'Each region\u2019s mean score across years.')),
            graph(fig),
            how_to_read([
                'Each line is one region\u2019s average score on this dimension, '
                'plotted over time.',
                'Higher = better. Compare where the lines sit and whether the '
                'gaps between regions are widening or narrowing.',
                'If you\u2019ve selected a country, its line is highlighted against '
                'the regional averages for context.'])]
    if paused:   # keep playback ticks light — tables return on pause
        kids.append(data_view(table))
    return kids


@callback(Output('exp-comp-cards', 'children'),
          Input('exp-dim', 'value'), Input('exp-comp', 'data'))
def update_comp_cards(dim, comp):
    dim_data = DIMENSIONS[dim]
    comps = list(dim_data['raw_indicators'].keys())
    if comp not in comps:
        comp = comps[0]
    accent = dim_accent(THEME)[dim]
    return [html.Button([
        html.Div(c, className='cc-name'),
        html.Div(f"{len(dim_data['raw_indicators'][c])} indicators",
                 className='cc-n')],
        id={'type': 'comp-btn', 'index': c}, n_clicks=0,
        className='comp-card' + (' selected' if c == comp else ''),
        style={'borderLeftColor': accent}) for c in comps]


@callback(Output('exp-drill-head', 'children'),
          Input('exp-dim', 'value'), Input('exp-comp', 'data'),
          Input('exp-year', 'value'))
def update_drill_head(dim, comp, year):
    year = year if year in years else LATEST
    comps = list(DIMENSIONS[dim]['raw_indicators'].keys())
    if comp not in comps:
        comp = comps[0]
    return head_block('Drill Down', f'{comp} \u2014 Sub-Indicators by Region ({year})',
                      'Click a component to unpack its indicators. '
                      'Use the year slider above to see how scores change over time.')


@callback(Output('exp-drill-body', 'children'),
          Input('exp-dim', 'value'), Input('exp-comp', 'data'),
          Input('exp-year', 'value'),
          Input('exp-play-interval', 'disabled'))
def update_drill(dim, comp, year, paused):
    year = year if year in years else LATEST
    dff = df[df['SPI year'] == year]
    dim_data = DIMENSIONS[dim]
    comps = list(dim_data['raw_indicators'].keys())
    if comp not in comps:
        comp = comps[0]
    drill_fig, drill_table = C.subindicator_fig(dff, dim_data, comp, year, THEME)
    if drill_fig is None:
        return html.Div('No regional variation to display for this component.',
                        className='card-sub')
    kids = [graph(drill_fig),
            how_to_read([
                'Each small panel is one sub-indicator of the selected component. '
                'Within a panel, each bar is a region.',
                'Bar height is the region\u2019s average score (higher is better); '
                'bars are sorted tallest-to-shortest so the leader is on the left.',
                'The color legend below maps each bar color to a region and stays '
                'fixed across years so playback doesn\u2019t reshuffle it.'])]
    if paused:
        kids.append(data_view(drill_table))
    return kids


def _dd_region_of(country):
    """Region for a country (constant across years; uses latest-year rows)."""
    latest = df[df['SPI year'] == LATEST]
    if country and country in latest['Country'].values:
        return latest[latest['Country'] == country]['Region'].iloc[0]
    return None


@callback(
    Output('dd-locator-graph', 'figure'), Output('dd-locator-title', 'children'),
    Output('dd-locator-sub', 'children'), Output('dd-locator-col', 'style'),
    Output('dd-gdpspi-head', 'children'), Output('dd-gdpspi-graph', 'figure'),
    Output('dd-gdpspi-table', 'children'), Output('dd-ind-wrap', 'style'),
    Input('exp-country', 'value'),
    Input('exp-view', 'value'))
def update_deep_dive_structure(country, view):
    """Country/view-driven scaffold. These pieces don't depend on the year, so
    they're built only when the country changes — the year callback below only
    touches figure props, keeping playback perfectly smooth."""
    if view != 'country':
        raise PreventUpdate
    is_world = not country
    latest = df[df['SPI year'] == LATEST]

    if is_world:
        return ({}, '', '', {'display': 'none'},
                [], {}, None, {'display': 'none'})

    region_name = _dd_region_of(country)
    locator = C.locator_map(country, region_name, latest, THEME)
    gdp_spi_fig, gdp_spi_table = C.gdp_spi_country_line(country, THEME)

    gdp_head = head_block('GDP vs SPI Over Time',
                          f'{flag(country)} {country} — GDP per Capita vs SPI Score',
                          'How economic output and social progress have co-evolved.')

    return (locator, f'{flag(country)} {country}',
            region_display(region_name) if region_name else '', {},
            gdp_head, gdp_spi_fig, data_view(gdp_spi_table), {})


@callback(
    Output('dd-head', 'children'), Output('dd-tiles', 'children'),
    Output('dd-comp-title', 'children'), Output('dd-comparison-graph', 'figure'),
    Output('dd-comp-table', 'children'), Output('dd-ind-head', 'children'),
    Output('dd-indicator-graph', 'figure'), Output('dd-ind-table', 'children'),
    Output('dd-trend-graph', 'figure'), Output('dd-trend-table', 'children'),
    Input('exp-dim', 'value'), Input('exp-country', 'value'),
    Input('exp-year', 'value'), Input('exp-view', 'value'),
    Input('exp-play-interval', 'disabled'))
def update_deep_dive_year(dim, country, year, view, paused):
    """Year-driven, prop-only updates — never rebuilds the DOM, so the slider
    and play button transition smoothly like every other chart."""
    if view != 'country':
        raise PreventUpdate
    year = year if year in years else LATEST
    # Population vs GDP trend with the selected year highlighted
    trend_fig, trend_table = C.pop_gdp_country_line(
        None if not country else country, THEME, highlight_year=year)
    dff = df[df['SPI year'] == year]
    dim_data = DIMENSIONS[dim]
    comps = dim_data['components']
    is_world = not country
    label = 'World' if is_world else country

    if is_world:
        w = world_df[world_df['SPI year'] == year]
        src = w.iloc[0] if not w.empty else None
    else:
        src = (dff[dff['Country'] == country].iloc[0]
               if country in dff['Country'].values else None)

    head = head_block(
        'Country Deep Dive',
        f'{flag(country) if country else "🌍"} {label} — Deep Dive ({year})',
        DIM_NARRATIVE.get(dim, {}).get('country') if not is_world else None)

    if src is None:
        empty_tiles = html.Div(f'No data for {label} in {year}.', className='card-sub')
        return (head, empty_tiles, f'{dim}: {label} vs Region vs World', {}, None,
                [], {}, None, trend_fig, data_view(trend_table))

    pop_v = pd.to_numeric(src.get(POP_COL), errors='coerce')
    gdp_v = pd.to_numeric(src.get(GDP_COL), errors='coerce')
    spi_v = pd.to_numeric(src.get(SPI_COL), errors='coerce')
    tiles = html.Div(className='dd-stats', children=[
        html.Div([html.Div('Population', className='stat-label'),
                  html.Div(f'{pop_v / 1e6:,.1f}M' if pd.notna(pop_v) else '—',
                           className='stat-value')], className='dd-tile'),
        html.Div([html.Div('GDP per capita', className='stat-label'),
                  html.Div(f'${gdp_v / 1e3:,.1f}K' if pd.notna(gdp_v) else '—',
                           className='stat-value')], className='dd-tile'),
        html.Div([html.Div('Social progress', className='stat-label'),
                  html.Div(f'{spi_v:.1f}' if pd.notna(spi_v) else '—',
                           className='stat-value')], className='dd-tile'),
    ])

    wrow_df = world_df[world_df['SPI year'] == year]
    wrow = wrow_df.iloc[0] if not wrow_df.empty else None
    region_name = None if is_world else _dd_region_of(country)

    country_vals = None if is_world else [pd.to_numeric(src.get(c), errors='coerce')
                                          for c in comps]
    world_vals = ([pd.to_numeric(wrow.get(c), errors='coerce') for c in comps]
                  if wrow is not None else [None] * len(comps))
    region_vals = None
    if region_name:
        rd = dff[dff['Region'] == region_name]
        region_vals = [rd[c].mean() for c in comps]

    comparison, comp_table = C.comparison_fig(
        comps, country_vals, region_vals, world_vals, label, region_name, THEME)
    comp_title = f'{dim}: {label} vs Region vs World'

    if is_world:
        return (head, tiles, comp_title, comparison,
                data_view(comp_table) if paused else no_update, [], {}, None,
                trend_fig, data_view(trend_table) if paused else no_update)

    ind_head = head_block(
        'Indicator Deep Dive',
        f'{flag(country)} {country}, Indicator by Indicator ({year})',
        'Scores reflect how well a country manages each indicator \u2014 '
        'a higher score means the country handles that challenge better than most.')
    ind_fig, ind_table = C.indicator_charts(dff, country, dim_data, THEME)
    return (head, tiles, comp_title, comparison,
            data_view(comp_table) if paused else no_update,
            ind_head, ind_fig, data_view(ind_table) if paused else no_update,
            trend_fig, data_view(trend_table) if paused else no_update)


# ======================================================== social progress ==

@lru_cache(maxsize=64)
def _bubble(year, focus):
    return C.spi_bubble(df[df['SPI year'] == year], year, THEME, focus)


@lru_cache(maxsize=64)
def _gdp(year, focus):
    return C.gdp_scatter(df[df['SPI year'] == year], year, THEME, focus)


@lru_cache(maxsize=4)
def _movers():
    return C.movers_fig(THEME)


@lru_cache(maxsize=4)
def _outliers():
    return C.outlier_fig(THEME)


@lru_cache(maxsize=32)
def _happy(hy):
    return C.happiness_factors(happy_df[happy_df['Year'] == hy], hy, THEME)


@lru_cache(maxsize=64)
def _happy_spi(hy, focus):
    return C.happiness_vs_spi(hy, THEME, focus)


def _focus(value):
    return None if value in (None, 'ALL') else value


def _happy_year(year):
    if year in happy_years:
        return year
    return min(happy_years, key=lambda y: abs(y - (year or LATEST)))


def _sp_card(kicker, title_id, title, sub, graph_id, fig, table=None,
             table_key=None, extra=None, htr=None):
    """Card with a stable graph id — callbacks update figure props in place
    instead of re-mounting the DOM (keeps playback smooth). Pass `table` for
    a static data twin, or `table_key` for one built lazily on first open.
    `htr` adds a 'How to read this chart' block under the graph."""
    kids = [
        html.Div([html.Div(kicker, className='kicker'),
                  html.H3(title, className='card-title', id=title_id),
                  html.Div(sub, className='card-sub')], className='card-head'),
        graph(fig, id=graph_id),
    ]
    if htr:
        kids.append(how_to_read(htr))
    if table is not None:
        kids.append(data_view(table))
    elif table_key:
        kids.append(html.Details(className='data-view', children=[
            html.Summary('View data',
                         id={'type': 'sp-dv-sum', 'index': table_key},
                         n_clicks=0),
            html.Div(id={'type': 'sp-dv-body', 'index': table_key}),
        ]))
    if extra:
        kids.extend(extra)
    return html.Div(className='viz-card', children=kids)


# One-line description per Progress sub-view, shown inside the View tile.
SP_VIEW_BLURB = {
    'spi': 'Who improved, who declined, and how the world spreads across SPI tiers.',
    'gdp': 'Where money buys progress, where it stops mattering, and who beats the curve.',
    'happy': 'What drives happiness, and how it tracks social progress.',
}


def build_social_progress_tab():
    """Fully rendered in one shot (cached builders) — no callback waterfall.
    Data tables are lazy (built on first open) so year changes only ship
    figure updates."""
    year, focus, hy = LATEST, None, _happy_year(LATEST)
    bub_fig, _ = _bubble(year, focus)
    gdp_fig, _ = _gdp(year, focus)
    mov_fig, mov_table = _movers()
    hap_fig, _ = _happy(hy)
    hs_fig, _ = _happy_spi(hy, focus)

    # --- Sub-tab: Social Progress ---
    sp_section = html.Div([
        html.Div(className='filter-bar', style={'marginBottom': '16px'}, children=[
            html.Div([html.Div('Region Lens', className='control-label'),
                      dcc.Dropdown(id='sp-focus',
                                   options=([{'label': 'All Regions', 'value': 'ALL'}] +
                                            [{'label': region_display(r), 'value': r}
                                             for r in regions]),
                                   value='ALL', clearable=False,
                                   style={'minWidth': '230px'})],
                     className='control-group'),
            timeline('sp'),
        ]),
        _sp_card('Global View', 'sp-bubble-title',
                 f'Social Progress Index — Every Country ({year})',
                 'Every bubble represents a country, bubble size = population.',
                 'sp-bubble-graph', bub_fig, table_key='bubble',
                 htr=['Each bubble is a country. Left\u2013right position is its '
                      'SPI score (higher is better); bubble size is population.',
                      'The chart is split into 7 vertical tiers from struggling '
                      '(left) to leading (right) \u2014 the color legend maps each '
                      'tier.',
                      'Select a region above to spotlight its countries, or press '
                      'play to watch bubbles drift right as the world improves.']),
        html.Div(className='viz-card', children=[
            head_block('SPI Progress', f'SPI Change vs {LATEST}',
                       'Countries with improving SPI (\u2191 0.5+ points), no significant '
                       'change (0 to 0.49 points), or declining (\u2193 less than 0) \u2014 '
                       'comparison against 2025. Select a section to view countries.'),
            html.Div(className='filter-bar', style={'marginBottom': '12px'}, children=[
                html.Div([
                    html.Button('\u25b6 Play', id='sp-pie-play-btn',
                                className='play-btn', n_clicks=0),
                    html.Button('\u2039', id='sp-pie-prev-btn',
                                className='step-btn', n_clicks=0),
                    html.Button('\u203a', id='sp-pie-next-btn',
                                className='step-btn', n_clicks=0),
                ], className='play-group'),
                html.Div(dcc.Slider(
                    id='sp-pie-year', min=min(years), max=max(years) - 1, step=1,
                    value=EARLIEST,
                    marks={y: str(y) for y in years if y % 2 == 1 and y < max(years)},
                    included=True, persistence=True, persistence_type='session'),
                    className='slider-wrap'),
                html.Div(str(EARLIEST), id='sp-pie-year-badge', className='year-badge'),
                dcc.Interval(id='sp-pie-play-interval', interval=PLAY_SPEED,
                             disabled=True),
            ]),
            html.Div(id='sp-pie-title', className='card-title',
                     style={'fontSize': '14px', 'marginBottom': '4px'}),
            dcc.Graph(id='sp-pie-graph', figure={}, config=GRAPH_CONFIG),
            how_to_read([
                'The pie splits all countries into three groups by how their SPI '
                'changed from the chosen baseline year to 2025: Improving, No '
                'significant change, or Declining.',
                'Each slice\u2019s size is the number of countries in that group.',
                'Click a slice to list the countries in it; move the baseline-year '
                'slider to change the comparison window.']),
            html.Div(id='sp-pie-table-body'),
            html.Div(id='sp-pie-country-body')]),
        _sp_card('Momentum', 'sp-movers-title',
                 f'Biggest Movers, {EARLIEST} \u2192 {LATEST}',
                 'Countries which have had the biggest change in SPI from 2011 to 2025.',
                 'sp-movers-graph', mov_fig, table=mov_table,
                 htr=['A ranked bar chart of change in SPI from 2011 to 2025 '
                      '(not the score itself).',
                      'Bars to the right are gains; bars to the left (if any) are '
                      'declines. Longer bar = bigger move.']),
        next_tab_nudge('sp-spi', 'GDP', 'subview:gdp'),
        html.Div(className='viz-card', style={'padding': '20px 28px'}, children=[
            html.Div('Takeaway', className='kicker'),
            html.P('The biggest movers are not the richest countries.',
                   className='takeaway-lead'),
            html.P('Every one of the eight largest gains since 2011 came from a '
                   'low- or middle-income country: Fiji (+11.8), Saudi Arabia '
                   '(+10.8), Moldova, The Gambia and Armenia (+10.0 each), then '
                   'Uzbekistan, Vietnam and Eswatini. The Gambia manages its '
                   '10-point gain on $3.0K per person.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
            html.P('That pattern holds across the whole distribution. The poorer '
                   'half of the world in 2011 gained an average of +5.8 points by '
                   '2025, against +4.1 for the richer half \u2014 the bottom is '
                   'rising about 1.4\u00d7 faster than the top.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
            html.P('But catching up is slow work. The spread between the best and '
                   'worst country has barely budged \u2014 63.2 points in 2011, '
                   '64.0 today. Countries are climbing; the ladder itself is the '
                   'same length.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
        ]),
    ])

    # --- Sub-tab: GDP ---
    gdp_section = html.Div([
        html.Div(className='filter-bar', style={'marginBottom': '16px'}, children=[
            html.Div([html.Div('Region Lens', className='control-label'),
                      dcc.Dropdown(id='sp-gdp-focus',
                                   options=([{'label': 'All Regions', 'value': 'ALL'}] +
                                            [{'label': region_display(r), 'value': r}
                                             for r in regions]),
                                   value='ALL', clearable=False,
                                   style={'minWidth': '230px'})],
                     className='control-group'),
            timeline('spgdp'),
        ]),
        _sp_card('Wealth vs Progress', 'sp-gdp-title',
                 f'GDP vs Social Progress ({year})',
                 'Does more money mean more progress? The relationship is logarithmic \u2014 '
                 'GDP helps early on, but beyond ~$20K per capita the gains flatten sharply.',
                 'sp-gdp-graph', gdp_fig, table_key='gdp',
                 htr=['Each dot is a country: horizontal position is GDP per '
                      'capita, vertical position is SPI. Color is the SPI tier.',
                      'The x-axis is a fixed range so dots move as you scrub years, '
                      'and the curve is the best-fit log trend line.',
                      'Dots above the curve get more social progress than their '
                      'income predicts; dots below get less. The r value shows how '
                      'tightly income and progress track.'],
                 extra=[
                     html.Button('Show Over Performers and Under Performers (2025)',
                                 id='sp-outlier-toggle', className='ghost-btn',
                                 n_clicks=0,
                                 style={'display': 'block', 'margin': '10px auto 2px'}),
                     dbc.Collapse(html.Div(id='sp-outlier-body'),
                                  id='sp-outlier-collapse', is_open=False),
                 ]),
        next_tab_nudge('sp-gdp', 'Happiness Index', 'subview:happy'),
        html.Div(className='viz-card', style={'padding': '20px 28px'}, children=[
            html.Div('Takeaway', className='kicker'),
            html.P('Money matters most when you have little of it.',
                   className='takeaway-lead'),
            html.P('Give every country an extra $5,000 per person and the poorest '
                   'ones gain about 15 SPI points. The richest gain less than 1. '
                   'The same money, twenty times the impact.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
            html.P('That is why the curve flattens: wealth buys the basics \u2014 '
                   'food, clean water, hospitals, schools \u2014 and once those are '
                   'in place, more money adds little. What separates rich countries '
                   'from each other is not income but governance, rights, and '
                   'inclusion. Money buys the floor, not the ceiling.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
        ]),
    ])

    # --- Sub-tab: Happiness Index ---
    happiness_section = html.Div([
        html.Div(className='viz-card', children=[
            html.Div([html.Div('Happiness', className='kicker'),
                      html.H3(f'Happiness Drivers ({hy})',
                              className='card-title', id='sp-happy-title'),
                      html.Div('Top 20 happiest countries \u2014 how much each factor '
                               '(GDP, social support, health, freedom, generosity, '
                               'corruption) contributes to their happiness score.',
                               className='card-sub')], className='card-head'),
            html.Div(className='filter-bar', style={'marginBottom': '10px',
                     'padding': '12px 18px', 'border': 'none', 'boxShadow': 'none'},
                     children=[
                html.Div([
                    html.Button('\u25b6 Play', id='sp-happy-play-btn',
                                className='play-btn', n_clicks=0),
                    html.Button('\u2039', id='sp-happy-prev-btn',
                                className='step-btn', n_clicks=0, title='Previous year'),
                    html.Button('\u203a', id='sp-happy-next-btn',
                                className='step-btn', n_clicks=0, title='Next year'),
                ], className='play-group'),
                html.Div(dcc.Slider(
                    id='sp-happy-year', min=min(happy_years), max=max(happy_years),
                    step=1, value=hy,
                    marks={y: str(y) for y in happy_years},
                    included=True, persistence=True, persistence_type='session'),
                    className='slider-wrap'),
                html.Div(str(hy), id='sp-happy-year-badge', className='year-badge'),
                dcc.Interval(id='sp-happy-play-interval', interval=PLAY_SPEED,
                             disabled=True),
            ]),
            graph(hap_fig, id='sp-happy-graph'),
            how_to_read([
                'One horizontal bar per country (the 20 happiest). Each bar is '
                'split into the six factors that make up its happiness score \u2014 '
                'GDP, social support, health, freedom, generosity, and '
                'perceptions of corruption.',
                'Longer coloured segments contributed more to that country\u2019s '
                'happiness; the number at the end is the total score.',
                'Use the year slider to see how the drivers shift over time.']),
            html.Details(className='data-view', children=[
                html.Summary('View data',
                             id={'type': 'sp-dv-sum', 'index': 'happy'},
                             n_clicks=0),
                html.Div(id={'type': 'sp-dv-body', 'index': 'happy'}),
            ]),
        ]),
        _sp_card('Happiness \u00d7 Progress', 'sp-happyspi-title',
                 f'Happiness vs Social Progress ({hy})',
                 'Blue = happier than their progress predicts; red = less happy than predicted.',
                 'sp-happyspi-graph', hs_fig, table_key='happyspi',
                 htr=['Each dot is a country: horizontal position is SPI, vertical '
                      'position is its self-reported happiness score (0\u201310).',
                      'The diagonal trend line is what happiness the SPI predicts. '
                      'Blue dots sit above it (happier than predicted); red dots '
                      'below (less happy than predicted).',
                      'The biggest vertical gaps are the most interesting \u2014 '
                      'places where wellbeing and measured progress diverge.']),
        next_tab_nudge('sp-happy', 'Key Insights', 'tab:insights'),
        html.Div(className='viz-card', style={'padding': '20px 28px'}, children=[
            html.Div('Takeaway', className='kicker'),
            html.P('Latin America is happier than its scorecard; South Asia is unhappier.',
                   className='takeaway-lead'),
            html.P('Guatemala, El Salvador, Nicaragua, and Mexico all report '
                   'happiness roughly 1.2\u20131.4 points above what their SPI '
                   'predicts \u2014 the region\u2019s biggest systematic gap. Costa '
                   'Rica does too, on top of already being a Basic Needs '
                   'over-performer. Family ties and social trust appear to buy '
                   'wellbeing that nutrition and safety scores don\u2019t capture.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
            html.P('The steepest shortfalls are Afghanistan (\u22122.1) and Botswana '
                   '(\u22121.9), followed by Lebanon, Sri Lanka, and Jordan \u2014 '
                   'countries where war, economic crisis, or deep inequality '
                   'weigh on how people feel far more than their SPI score alone '
                   'would suggest. Across all 140 countries, SPI still tracks '
                   'happiness closely (r = 0.83) \u2014 about as well as income '
                   'does (r = 0.80) \u2014 but these are the places where the '
                   'index and lived experience diverge most.',
                   className='card-sub', style={'fontSize': '13.5px', 'lineHeight': '1.7',
                                                'color': 'var(--ink-2)'}),
        ]),
    ])

    return html.Div([
        html.Div(className='filter-bar', style={'marginBottom': '16px'}, children=[
            html.Div([html.Div('View', className='control-label'),
                      segmented('sp-sub-view', [
                          {'label': 'Social Progress', 'value': 'spi'},
                          {'label': 'GDP', 'value': 'gdp'},
                          {'label': 'Happiness Index', 'value': 'happy'},
                      ], 'spi'),
                      # Description sits inside the View tile, right under the
                      # buttons, and swaps with the selected sub-view.
                      html.Div(SP_VIEW_BLURB['spi'], id='sp-sub-blurb',
                               className='card-sub',
                               style={'marginTop': '8px', 'whiteSpace': 'nowrap'}),
                      ], className='control-group'),
        ]),
        html.Div(id='sp-sec-spi', children=sp_section),
        html.Div(id='sp-sec-gdp', children=gdp_section, style={'display': 'none'}),
        html.Div(id='sp-sec-happy', children=happiness_section, style={'display': 'none'}),
    ])


make_play_callbacks('sp')
make_play_callbacks('spgdp')

# Pie chart play callbacks (years up to max-1)
_pie_years = [y for y in years if y < max(years)]

@callback(Output('sp-pie-play-interval', 'disabled'),
          Output('sp-pie-play-btn', 'children'),
          Output('sp-pie-play-btn', 'className'),
          Output('sp-pie-year', 'value', allow_duplicate=True),
          Input('sp-pie-play-btn', 'n_clicks'),
          Input('sp-pie-year', 'value'),
          State('sp-pie-play-interval', 'disabled'),
          prevent_initial_call=True)
def toggle_pie_play(n, year, disabled):
    trig = ctx.triggered_id
    if trig == 'sp-pie-year' and year == _pie_years[-1] and not disabled:
        return True, '\u25b6 Play', 'play-btn', no_update
    if trig == 'sp-pie-play-btn':
        playing = bool(disabled)
        # Parked on the last year: rewind so play has a series to run.
        rewind = _pie_years[0] if playing and year == _pie_years[-1] else no_update
        return (not playing,
                '\u275a\u275a Pause' if playing else '\u25b6 Play',
                'play-btn playing' if playing else 'play-btn',
                rewind)
    return no_update, no_update, no_update, no_update

@callback(Output('sp-pie-year', 'value'),
          Input('sp-pie-play-interval', 'n_intervals'),
          Input('sp-pie-prev-btn', 'n_clicks'),
          Input('sp-pie-next-btn', 'n_clicks'),
          State('sp-pie-year', 'value'),
          prevent_initial_call=True)
def nav_pie_year(n_int, prev, nxt, cur):
    idx = _pie_years.index(cur) if cur in _pie_years else 0
    if ctx.triggered_id == 'sp-pie-prev-btn':
        return _pie_years[max(0, idx - 1)]
    if ctx.triggered_id == 'sp-pie-next-btn':
        return _pie_years[min(len(_pie_years) - 1, idx + 1)]
    nxt_idx = idx + 1
    if nxt_idx >= len(_pie_years):
        return _pie_years[-1]
    return _pie_years[nxt_idx]

app.clientside_callback(
    "function(year) { return String(year); }",
    Output('sp-pie-year-badge', 'children'),
    Input('sp-pie-year', 'value'),
)

# Toggle SP sub-sections visibility + swap the View-tile blurb (clientside, instant)
app.clientside_callback(
    """
    function(view) {
        var blurb = %s;
        return [
            view === 'spi' ? {display: 'block'} : {display: 'none'},
            view === 'gdp' ? {display: 'block'} : {display: 'none'},
            view === 'happy' ? {display: 'block'} : {display: 'none'},
            blurb[view] || ''
        ];
    }
    """ % json.dumps(SP_VIEW_BLURB),
    [Output('sp-sec-spi', 'style'),
     Output('sp-sec-gdp', 'style'),
     Output('sp-sec-happy', 'style'),
     Output('sp-sub-blurb', 'children')],
    Input('sp-sub-view', 'value'),
)

# Happiness play callbacks — uses happy_years instead of years
@callback(Output('sp-happy-play-interval', 'disabled'),
          Output('sp-happy-play-btn', 'children'),
          Output('sp-happy-play-btn', 'className'),
          Output('sp-happy-year', 'value', allow_duplicate=True),
          Input('sp-happy-play-btn', 'n_clicks'),
          Input('sp-happy-year', 'value'),
          State('sp-happy-play-interval', 'disabled'),
          prevent_initial_call=True)
def toggle_happy_play(n, year, disabled):
    trig = ctx.triggered_id
    if trig == 'sp-happy-year' and year == max(happy_years) and not disabled:
        return True, '\u25b6 Play', 'play-btn', no_update
    if trig == 'sp-happy-play-btn':
        playing = bool(disabled)
        # Happiness data starts at 2019, so rewind there rather than 2011.
        rewind = (happy_years[0] if playing and year == happy_years[-1]
                  else no_update)
        return (not playing,
                '\u275a\u275a Pause' if playing else '\u25b6 Play',
                'play-btn playing' if playing else 'play-btn',
                rewind)
    return no_update, no_update, no_update, no_update


@callback(Output('sp-happy-year', 'value'),
          Input('sp-happy-play-interval', 'n_intervals'),
          Input('sp-happy-prev-btn', 'n_clicks'),
          Input('sp-happy-next-btn', 'n_clicks'),
          State('sp-happy-year', 'value'),
          prevent_initial_call=True)
def nav_happy_year(n_int, prev, nxt, cur):
    idx = happy_years.index(cur) if cur in happy_years else len(happy_years) - 1
    if ctx.triggered_id == 'sp-happy-prev-btn':
        return happy_years[max(0, idx - 1)]
    if ctx.triggered_id == 'sp-happy-next-btn':
        return happy_years[min(len(happy_years) - 1, idx + 1)]
    nxt_idx = idx + 1
    if nxt_idx >= len(happy_years):
        return happy_years[-1]
    return happy_years[nxt_idx]


app.clientside_callback(
    "function(year) { return String(year); }",
    Output('sp-happy-year-badge', 'children'),
    Input('sp-happy-year', 'value'),
)


@callback(Output('sp-bubble-graph', 'figure'), Output('sp-bubble-title', 'children'),
          Input('sp-year', 'value'), Input('sp-focus', 'value'),
          prevent_initial_call=True)
def update_sp_bubble(year, focus):
    """Update only the SPI bubble chart from the SP sub-tab's region/year controls."""
    year = year if year in years else LATEST
    f = _focus(focus)
    bub_fig, _ = _bubble(year, f)
    return bub_fig, f'Social Progress Index — Every Country ({year})'


@callback(Output('sp-gdp-graph', 'figure'), Output('sp-gdp-title', 'children'),
          Input('spgdp-year', 'value'), Input('sp-gdp-focus', 'value'),
          prevent_initial_call=True)
def update_sp_gdp(year, focus):
    """Update the GDP scatter from the GDP sub-tab's region/year controls."""
    year = year if year in years else LATEST
    f = _focus(focus)
    gdp_fig, _ = _gdp(year, f)
    return gdp_fig, f'GDP vs Social Progress ({year})'


@callback(Output('sp-happy-graph', 'figure'), Output('sp-happy-title', 'children'),
          Input('sp-happy-year', 'value'),
          prevent_initial_call=True)
def update_happy_factors(hy):
    """Update the happiness factors chart from its own year slider."""
    hy = hy if hy in happy_years else max(happy_years)
    hap_fig, _ = _happy(hy)
    return hap_fig, f'Happiness Drivers ({hy})'


@callback(Output('sp-pie-title', 'children'),
          Output('sp-pie-graph', 'figure'),
          Output('sp-pie-table-body', 'children'),
          Input('sp-pie-year', 'value'))
def update_pie_chart(base_year):
    """Prop-only update — no DOM rebuild, no alignment shift."""
    base_year = base_year if base_year in years else EARLIEST
    pie_fig, pie_table = C.spi_change_pie(THEME, baseline_year=base_year)
    return f'{base_year} \u2192 {LATEST}', pie_fig, data_view(pie_table)


SP_TABLE_BUILDERS = {
    'bubble': lambda year: _bubble(year, None)[1],
    'gdp': lambda year: _gdp(year, None)[1],
    'happy': lambda year: _happy(_happy_year(year))[1],
    'happyspi': lambda year: _happy_spi(_happy_year(year), None)[1],
}


@callback(Output({'type': 'sp-dv-body', 'index': MATCH}, 'children'),
          Input({'type': 'sp-dv-sum', 'index': MATCH}, 'n_clicks'),
          Input('sp-year', 'value'),
          prevent_initial_call=True)
def fill_sp_table(n, year):
    """Build a data table only once its disclosure has been opened; after
    that, keep it in sync with the year."""
    if not n:
        return no_update
    out = ctx.outputs_list
    key = (out['id'] if isinstance(out, dict) else out[0]['id'])['index']
    year = year if year in years else LATEST
    return table_el(SP_TABLE_BUILDERS[key](year))


@callback(Output('sp-outlier-collapse', 'is_open'),
          Output('sp-outlier-toggle', 'children'),
          Input('sp-outlier-toggle', 'n_clicks'),
          State('sp-outlier-collapse', 'is_open'),
          prevent_initial_call=True)
def toggle_outliers(n, is_open):
    now = not is_open
    return now, ('Hide Over Performers and Under Performers (2025)' if now
                 else 'Show Over Performers and Under Performers (2025)')


@callback(Output('sp-outlier-body', 'children'),
          Input('sp-outlier-collapse', 'is_open'))
def update_outliers(is_open):
    if not is_open:
        return []
    fig, table = _outliers()
    return [html.Div(style={'marginTop': '14px'}, children=[
        head_block('Over and Under Performers', 'Over Performers and Under Performers (2025)',
                   'Countries more than 8 SPI points above or below what GDP alone predicts.'),
        graph(fig), data_view(table)])]


@callback(Output('sp-pie-country-body', 'children'),
          Input('sp-pie-graph', 'clickData'),
          State('sp-pie-year', 'value'),
          prevent_initial_call=True)
def show_pie_category_countries(click_data, base_year):
    """On slice click, list every country in that SPI-change category."""
    if not click_data:
        return no_update
    category = click_data['points'][0]['label']
    table = C.spi_change_category_table(category, baseline_year=base_year)
    return html.Div(style={'marginTop': '10px'}, children=[
        html.Div(f'{category} — {len(table)} countries', className='card-title',
                 style={'fontSize': '14px', 'marginBottom': '2px'}),
        table_el(table, max_rows=len(table))])


# ==================================================================== run ==

# ============================================================= insights ==

def build_insights_tab():
    """Key Insights tab: the most surprising data-driven findings."""
    # Compute the overview facts dynamically
    latest = df[df['SPI year'] == LATEST]
    top = latest.nlargest(1, SPI_COL).iloc[0]
    bot = latest.nsmallest(1, SPI_COL).iloc[0]
    qatar = latest[latest['Country'] == 'Qatar']
    finland = latest[latest['Country'] == 'Finland']
    cr = latest[latest['Country'] == 'Costa Rica']
    eu = latest[latest['Region'] == 'Europe'][SPI_COL].mean()
    afr = latest[latest['Region'] == 'Sub-Saharan Africa'][SPI_COL].mean()

    overview_insights = [
        ('\U0001f4b0', 'The 159\u00d7 Income Gap',
         f'{flag("Singapore")} Singapore ($133K per capita) is 159\u00d7 richer '
         f'than {flag("Burundi")} Burundi ($0.8K). Singapore is a global trade '
         f'and financial hub; Burundi is landlocked, agrarian, and historically '
         f'unstable.'),
        ('\U0001f465', '35% of Humanity in Two Countries',
         f'{flag("India")} India (1.45B) + {flag("China")} China (1.41B) = 2.86B '
         f'of 8.1B people. Monsoon-fed river plains ideal for rice farming let '
         f'these regions scale agriculture \u2014 and population \u2014 earlier '
         f'and faster than anywhere else.'),
        ('\U0001f3c6', 'The Leader',
         f"{flag(top['Country'])} {top['Country']} leads the world with an SPI of "
         f"{top[SPI_COL]:.1f} \u2014 a {top[SPI_COL] - bot[SPI_COL]:.0f}-point gap over "
         f"{flag(bot['Country'])} {bot['Country']}, the lowest-ranked country. "
         f"Nordic countries dominate due to decades of investment in universal "
         f"welfare, education, and trust."),
        ('\U0001f30d', 'Europe Dominates',
         f"Europe leads all regions with an average SPI of {eu:.0f}, a "
         f"{eu - afr:.0f}-point gap over Sub-Saharan Africa\u2019s {afr:.0f}. "
         f"Post-war institution-building, EU integration, and strong social "
         f"safety nets created a compounding advantage."),
        ('\U0001f914', 'The Happiness Paradox',
         f"{flag('Mexico')} Mexico is #12 happiest but #75 in Social Progress. "
         f"Strong social bonds, family culture, and community resilience drive "
         f"happiness even when systemic infrastructure lags behind."),
        ('\U0001f4b8', 'Money \u2260 Progress',
         f"{flag('Singapore')} Singapore has the world\u2019s highest GDP/capita ($132K) "
         f"but ranks only #15 in SPI. Wealth concentrates in a small population; "
         f"personal freedoms and societal opportunity pull the score down."),
        ('\u2b50', 'Efficiency Champion',
         f"{flag('Costa Rica')} Costa Rica scores 78.8 on $27K per capita \u2014 "
         f"beating {flag('Qatar')} Qatar (74.1) on four times the income. "
         f"It abolished its military in 1948 and redirected spending to health, "
         f"education, and environment \u2014 proving policy beats income."),
    ]

    insights = overview_insights + [
        ('\U0001f4c9', 'The US Paradox',
         f'{flag("United States")} US: only country that got richer (+$15.7K GDP) yet '
         'declined in SPI (\u22122.4). Driven by rising inequality, political '
         'polarization, and eroding trust in institutions since 2016.'),
        ('\u26fd', 'Oil \u2260 Progress',
         f'{flag("Guyana")} Guyana: #10 in GDP, #96 in SPI \u2014 an 85-rank gap. '
         'Massive offshore oil discoveries (2015+) inflated GDP overnight, but '
         'infrastructure and healthcare haven\u2019t caught up.'),
        ('\U0001f4aa', 'Poorer but Better',
         f'{flag("Oman")} Oman lost $11.5K GDP/capita (oil price crash) but gained '
         '+8 SPI. Diversification reforms in education, health, and governance '
         'paid off even as oil revenues shrank.'),
        ('\u2764\ufe0f', 'Happy Despite the Odds',
         f'{flag("El Salvador")} El Salvador and {flag("Guatemala")} Guatemala: '
         'happiness above 6.5, SPI below 65. Strong family bonds and community '
         'ties compensate for weak institutions.'),
        ('\U0001f4ca', '95% Improved Since 2011',
         f'163 of 171 countries gained SPI; only 8 declined. '
         f'Steepest falls: {flag("Venezuela")} Venezuela (\u22124.7, economic '
         f'collapse), {flag("Syria")} Syria (\u22124.2, civil war), '
         f'{flag("Afghanistan")} Afghanistan (\u22123.2, Taliban takeover) \u2014 '
         f'but {flag("United States")} the US (\u22122.4) and '
         f'{flag("Canada")} Canada (\u22120.8) are on that list too, proving '
         f'wealth alone doesn\u2019t protect against backsliding.'),
        ('\U0001f3ed', 'Saudi Vision 2030',
         f'{flag("Saudi Arabia")} Saudi Arabia gained +10.8 SPI points since 2011 \u2014 '
         'the 2nd largest gain globally. Vision 2030 reforms expanded women\u2019s rights, '
         'entertainment access, and social freedoms at unprecedented speed.'),
        ('\U0001f30f', 'India Overtakes China',
         f'{flag("India")} India passed {flag("China")} China in 2022 (1,414M vs '
         f'1,412M) and now leads by 42M. China peaked that same year and has '
         f'shrunk every year since \u2014 decades of one-child policy and rising '
         f'costs of living catching up.'),
        ('\U0001f3c3', 'Oman\u2019s Population Doubled',
         f'{flag("Oman")} Oman\u2019s population grew 92% between 2011 and 2025 '
         f'\u2014 the fastest growth of any country over 1 million people. Gulf '
         f'states (Oman, {flag("Qatar")} Qatar +77%, {flag("Kuwait")} Kuwait +69%) '
         f'dominate the list, all driven by labor migration rather than birth rates.'),
    ]

    # Split each insight into front (fact) and back (reasoning)
    def split_insight(desc):
        """Split at the first sentence-ending period that's followed by a reason."""
        parts = desc.split('. ', 1)
        if len(parts) == 2:
            return parts[0] + '.', parts[1]
        return desc, ''

    cards = [html.Div(className='flip-card', children=[
        html.Div(className='flip-card-inner', children=[
            # Front
            html.Div(className='flip-card-front', children=[
                html.Div(emoji, className='insight-emoji'),
                html.Div([
                    html.Div(title, className='insight-full-title'),
                    html.Div(split_insight(desc)[0], className='insight-full-desc'),
                ]),
            ]),
            # Back
            html.Div(className='flip-card-back', children=[
                html.Div('\U0001f4a1', className='insight-emoji'),
                html.Div([
                    html.Div('Why?', className='insight-full-title'),
                    html.Div(split_insight(desc)[1] or desc,
                             className='insight-full-desc'),
                ]),
            ]),
        ]),
    ]) for emoji, title, desc in insights]

    return html.Div([
        html.Div(className='hero hero--white', style={'padding': '24px 24px 16px'},
                 children=[
            html.Div('Discovery', className='kicker', style={'textAlign': 'left'}),
            html.H1('KEY INSIGHTS', className='hero-title',
                    style={'textAlign': 'left', 'fontSize': '32px'}),
            html.P('The most surprising findings from 15 years of Social Progress '
                   'Index data \u2014 patterns that challenge conventional assumptions '
                   'about wealth, happiness, and human development.',
                   className='card-sub', style={'textAlign': 'left', 'marginTop': '6px'}),
            html.P('Hover over a card to flip it and see the reasoning. '
                   'Click "Flip all" to reveal all reasons at once.',
                   className='card-sub', style={'textAlign': 'left', 'marginTop': '4px',
                                                'fontStyle': 'italic'}),
        ]),
        # The dashboard's closing argument, stated before the individual
        # findings so the cards below read as evidence for it.
        html.Div(className='viz-card conclusion-card',
                 style={'padding': '26px 30px'}, children=[
            html.Div('The Bottom Line', className='kicker'),
            html.P('GDP is not destiny.',
                   className='takeaway-lead', style={'fontSize': '24px'}),
            html.P('Economic wealth does not automatically guarantee wellbeing \u2014 '
                   'and global quality of life is now stagnating, driven by rising '
                   'restrictions on human rights.',
                   className='card-sub',
                   style={'fontSize': '15px', 'lineHeight': '1.7',
                          'color': 'var(--ink)', 'marginBottom': '18px'}),
            html.Div('Key Findings', className='kicker',
                     style={'marginBottom': '10px'}),
            html.Div(className='findings-list', children=[
                html.Div([
                    html.Div('Economic Disconnect', className='finding-name'),
                    html.P('Money helps poorer nations build basic infrastructure, '
                           'but among wealthier nations high GDP does not predict '
                           'high social progress. The United States ranks 19th of '
                           '23 countries at similar income, lagging its peers most '
                           'on safety (\u221211 points) and health (\u221210).',
                           className='finding-body'),
                ], className='finding'),
                html.Div([
                    html.Div('The Rights Recession', className='finding-name'),
                    html.P('Global social progress has stalled since 2021 \u2014 '
                           'from +0.43 points a year to +0.14. Rights and Voice is '
                           'the only one of the twelve components in decline, and '
                           'it spills over into health, safety, and environmental '
                           'conditions.',
                           className='finding-body'),
                ], className='finding'),
                html.Div([
                    html.Div('Policy Prioritization', className='finding-name'),
                    html.P('Governments must measure success by direct social and '
                           'environmental outcomes rather than economic output '
                           'alone, to target what communities actually need.',
                           className='finding-body'),
                ], className='finding'),
            ]),
        ]),
        html.Button('Flip all', id='flip-all-btn', className='ghost-btn',
                    n_clicks=0, style={'margin': '0 24px 16px'}),
        html.Div(cards, className='insights-grid', id='insights-grid'),
    ])


# Flip all cards clientside
app.clientside_callback(
    """
    function(n) {
        var grid = document.getElementById('insights-grid');
        if (!grid) return '';
        var cards = grid.querySelectorAll('.flip-card');
        var anyFlipped = Array.from(cards).some(c => c.classList.contains('flipped'));
        cards.forEach(c => {
            if (anyFlipped) c.classList.remove('flipped');
            else c.classList.add('flipped');
        });
        return '';
    }
    """,
    Output('flip-all-btn', 'title'),
    Input('flip-all-btn', 'n_clicks'),
    prevent_initial_call=True,
)

# ================================================================ appendix ==

def build_appendix_tab():
    """Card grid: 12 component cards split into 3 dimension sections (Live,
    Thrive, Connect), 2 cards per row within each section."""
    dim_vars = {'Basic Needs (Live)': '--live',
                'Foundations of Wellbeing (Thrive)': '--thrive',
                'Societal Opportunity (Connect)': '--connect'}
    dim_tags = {'Basic Needs (Live)': 'LIVE',
                'Foundations of Wellbeing (Thrive)': 'THRIVE',
                'Societal Opportunity (Connect)': 'CONNECT'}

    sections = []
    for dim_name, comps in DIMENSION_COMPONENTS.items():
        var = dim_vars[dim_name]
        cards = []
        for comp in comps:
            defs = INDICATOR_DEFINITIONS.get(comp, {})
            ind_list = [html.Li([html.Span(ind, className='appx-card-ind-name'),
                                 html.Span(f' — {meaning}', className='appx-card-ind-def')])
                        for ind, meaning in defs.items()]
            cards.append(html.Div(
                className='appx-card', id=f'appx-anchor-{_anchor(comp)}',
                style={'borderTopColor': f'var({var})'},
                children=[
                    html.Div([
                        html.Span(className='fw-dot',
                                  style={'backgroundColor': f'var({var})'}),
                        html.Span(comp, className='appx-card-title'),
                    ], className='appx-card-header'),
                    html.Div(f'{len(defs)} indicators',
                             className='appx-card-meta'),
                    html.Details(className='appx-card-details', children=[
                        html.Summary('Show Indicators'),
                        html.Ul(ind_list, className='appx-card-list'),
                    ]),
                ]))
        sections.append(html.Div(className='appx-section', children=[
            html.Div([html.Span(dim_name.split('(')[0].strip().upper()),
                      html.Span(dim_tags[dim_name], className='pillar-tag',
                                style={'color': f'var({var})'})],
                     className='pillar-name',
                     style={'borderBottomColor': f'var({var})',
                            'marginLeft': '24px', 'marginRight': '24px'}),
            html.Div(cards, className='appx-grid'),
        ]))

    return html.Div([
        html.Div(className='hero hero--white', style={'padding': '24px 24px 16px'},
                 children=[
            html.Div('Reference', className='kicker', style={'textAlign': 'left'}),
            html.H1('APPENDIX — INDICATOR GLOSSARY', className='hero-title',
                    style={'textAlign': 'left', 'fontSize': '32px'}),
            html.P(['A plain-language meaning for every one of the ',
                    html.B(f'{N_INDICATORS} scored indicators'),
                    ' in the Social Progress Index. Click any card to expand '
                    'its indicator list.'],
                   className='card-sub', style={'textAlign': 'left', 'marginTop': '6px'}),
        ]),
        *sections,
    ])


# =============================================================== ask tab ===

def build_ask_tab():
    """Ask the Data — natural-language Q&A over the dataset. Deterministic
    parsing (see ai_query.py), not a free-generated LLM answer, so every
    response is traceable to an actual pandas computation."""
    return html.Div([
        html.Div(className='hero hero--white', style={'padding': '24px 24px 16px'},
                 children=[
            html.Div('Ask', className='kicker', style={'textAlign': 'left'}),
            html.H1('ASK THE DATA', className='hero-title',
                    style={'textAlign': 'left', 'fontSize': '32px'}),
            html.P('Ask a question about social progress, GDP, '
                   'population, or happiness \u2014 across 171 countries and 15 '
                   'years. Answers are computed directly from the dataset, not '
                   'generated freely, so every number here is traceable back to '
                   'a chart elsewhere in this dashboard.',
                   className='card-sub', style={'textAlign': 'left', 'marginTop': '6px'}),
        ]),
        html.Div(className='viz-card ask-input-row-card', children=[
            html.Div(className='ask-input-row', children=[
                dcc.Input(id='ask-input', type='text', debounce=False,
                         placeholder='e.g. "Which countries improved the most since 2011?"',
                         className='ask-input',
                         n_submit=0),
                html.Button('Ask', id='ask-submit-btn', className='ask-submit-btn',
                           n_clicks=0),
            ]),
            html.Div(className='ask-suggestions', children=[
                html.Span('Try:', className='ask-suggestions-label'),
                *[html.Button(q, id={'type': 'ask-suggestion', 'index': i},
                              className='ask-suggestion-chip', n_clicks=0)
                  for i, q in enumerate(AI.SUGGESTED_QUESTIONS)],
            ]),
        ]),
        dcc.Loading(type='dot', children=html.Div(id='ask-answer-area')),
        dcc.Store(id='ask-active-question'),
    ])


@callback(Output('ask-active-question', 'data'),
          Input('ask-submit-btn', 'n_clicks'),
          Input('ask-input', 'n_submit'),
          Input({'type': 'ask-suggestion', 'index': ALL}, 'n_clicks'),
          State('ask-input', 'value'),
          State({'type': 'ask-suggestion', 'index': ALL}, 'children'),
          prevent_initial_call=True)
def set_active_question(_n1, _n2, _n3, typed, suggestion_labels):
    """Either the free-text box or a suggestion chip sets the active question."""
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get('type') == 'ask-suggestion':
        all_ids = ctx.states_list[1]
        for id_dict, label in zip((c['id'] for c in all_ids), suggestion_labels):
            if id_dict['index'] == trig['index']:
                return label
        return no_update
    return (typed or '').strip() or no_update


@callback(Output('ask-answer-area', 'children'),
          Output('ask-input', 'value'),
          Input('ask-active-question', 'data'),
          prevent_initial_call=True)
def answer_active_question(question):
    if not question:
        return no_update, no_update
    result = AI.answer_question(question)
    kids = [
        html.Div(className='ask-answer-card', children=[
            html.Div(f'\u201c{question}\u201d', className='ask-echo'),
            html.Div(result['answer'], className='ask-answer-text',
                     style={'whiteSpace': 'pre-line'}),
        ]),
    ]
    # A question asking about several metrics at once returns one chart/table
    # per metric, so render them all rather than only the first.
    figs = result.get('figs')
    tables = result.get('tables')
    if figs:
        for i, fig in enumerate(figs):
            tbl = tables[i] if tables and i < len(tables) else None
            kids.append(html.Div(className='viz-card', children=[
                graph(fig),
                data_view(tbl) if tbl is not None else None,
            ]))
    elif result.get('fig') is not None:
        kids.append(html.Div(className='viz-card', children=[
            graph(result['fig']),
            data_view(result['table']) if result.get('table') is not None else None,
        ]))
    elif tables:
        for tbl in tables:
            kids.append(html.Div(className='viz-card',
                                 children=data_view(tbl, label='View data')))
    elif result.get('table') is not None:
        kids.append(html.Div(className='viz-card',
                             children=data_view(result['table'], label='View data')))
    return kids, ''


def _preload_caches():
    """Background thread: warm LRU caches for the Social Progress tab while
    the user is still looking at the Overview. This means switching to the SP
    tab is near-instant instead of waiting for figure generation."""
    dff = df[df['SPI year'] == LATEST]
    _bubble(LATEST, None)
    _gdp(LATEST, None)
    _movers()
    _outliers()
    hy = _happy_year(LATEST)
    _happy(hy)
    _happy_spi(hy, None)
    # Also preload a few year variations for smooth playback
    for y in years[-3:]:
        _bubble(y, None)
        _gdp(y, None)
    # Overview's "Dimension Distributions" slider — warm every year up front
    # (cheap, ~45ms each) so scrubbing/playing is instant from the first tick,
    # not just on revisits.
    for y in years:
        _dims_strip(y)


# Preload expensive figures in background so tab switches are instant. This
# runs at import time (not just __main__) so it also fires under gunicorn.
# On hosting with limited CPU (Render free tier), this runs slower but still
# ensures second visits to each tab are instant.
Thread(target=_preload_caches, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=False, port=8050)
