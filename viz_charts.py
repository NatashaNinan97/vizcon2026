"""Vizcon chart builders.

Every builder returns (figure, table_df) — the table is the chart's
no-hover-required data twin, rendered behind a "View data" toggle.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz_data import (df, world_df, happy_df, years, regions, LATEST, EARLIEST,
                      SPI_COL, GDP_COL, POP_COL, flag)
from viz_theme import (t, base_layout, region_colors, seq_scale, tier_ramp,
                       status_color, dim_accent, factor_colors, FACTOR_ORDER,
                       REGION_SHORT, FONT_STACK, REGION_ORDER, region_display)


# Words kept lowercase inside a title (unless they're the first word), and
# tokens whose existing capitalisation must be preserved verbatim.
_TC_SMALL = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'from', 'in',
             'into', 'nor', 'of', 'on', 'or', 'per', 'the', 'to', 'vs', 'with'}
_TC_KEEP = {'GDP', 'SPI', 'US', 'UK', 'EU', 'CO2', 'GNI', 'HIV', 'AIDS', 'DTP',
            'PM2.5', 'ICT', 'GPI', 'LGBT', 'NOx', 'TB'}


def _title_case(text):
    """Title-case a chart heading while protecting acronyms, small words, and
    anything already mixed-case (e.g. 'PM2.5', 'CO2')."""
    if not text:
        return text
    words = text.split(' ')
    out = []
    for i, w in enumerate(words):
        if not w:
            out.append(w)
            continue
        core = w.strip('()[],:;—-')
        if core.upper() in _TC_KEEP or (core.isupper() and len(core) > 1):
            out.append(w)                      # acronym — leave as-is
        elif core.lower() in _TC_SMALL and i != 0:
            out.append(w.lower())              # small word, not first
        elif any(ch.isdigit() for ch in core):
            out.append(w)                      # has digits — leave as-is
        else:
            out.append(w[0].upper() + w[1:])   # capitalise, keep rest intact
    return ' '.join(out)


def _style_all_axes(fig, theme):
    """Apply hairline grid / muted ticks to every subplot axis."""
    k = t(theme)
    fig.update_xaxes(gridcolor=k['grid'], zeroline=False, linecolor=k['axis'],
                     tickfont=dict(size=11, color=k['muted']),
                     title_font=dict(size=12, color=k['muted']))
    fig.update_yaxes(gridcolor=k['grid'], zeroline=False, linecolor=k['axis'],
                     tickfont=dict(size=11, color=k['muted']),
                     title_font=dict(size=12, color=k['muted']))


def _subplot_titles(fig, theme, size=13, only_subplot=None):
    """Style subplot titles. Subplot titles are added first by make_subplots,
    so when other annotations are appended later, restrict styling to the
    original subplot titles by passing only_subplot with their count."""
    k = t(theme)
    anns = fig.layout.annotations
    targets = anns[:only_subplot] if only_subplot else anns
    for ann in targets:
        if ann.text:
            ann.font = dict(size=size, color=k['ink'], family=FONT_STACK)


# ============================================================ overview ====

def strip_chart(dff, metrics, highlight_country, year, theme, titles=None):
    """Three-panel jittered strip plot of dimension scores by region."""
    np.random.seed(42)
    k = t(theme)
    rc = region_colors(theme)
    n = len(metrics)
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles or metrics,
                        horizontal_spacing=0.05, shared_yaxes=True)
    # Pre-compute region index mapping
    region_idx = {r: i for i, r in enumerate(regions)}

    for ci, metric in enumerate(metrics, 1):
        valid = dff[['Country', 'Region', metric]].dropna(subset=[metric])
        if valid.empty:
            continue
        scores = valid[metric].values
        cnames = valid['Country'].values
        regs = valid['Region'].values
        base_op = 0.25 if highlight_country else 0.85
        colors = [rc.get(r, k['muted']) for r in regs]
        ops = [1.0 if c == highlight_country else base_op for c in cnames]
        xp = np.array([region_idx.get(r, 0) for r in regs]) + np.random.uniform(-0.22, 0.22, len(scores))

        fig.add_trace(go.Scatter(
            x=xp, y=scores, mode='markers',
            marker=dict(size=10, color=colors, opacity=ops,
                        line=dict(width=1.5, color=k['surface'])),
            customdata=np.stack([cnames, regs], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b><br>' + metric +
                          ': %{y:.2f}<br>%{customdata[1]} · ' + str(year) +
                          '<extra></extra>',
            showlegend=False), row=1, col=ci)

        if highlight_country and highlight_country in cnames:
            mask = cnames == highlight_country
            fig.add_trace(go.Scatter(
                x=xp[mask], y=scores[mask], mode='markers',
                marker=dict(size=14, color=[rc.get(r, k['muted']) for r in regs[mask]],
                            opacity=1, line=dict(width=2.5, color=k['ink'])),
                hoverinfo='skip', showlegend=False), row=1, col=ci)

        fig.update_xaxes(tickvals=list(range(len(regions))),
                         ticktext=[REGION_SHORT[r] for r in regions],
                         showgrid=False, row=1, col=ci)
        fig.update_yaxes(range=[0, 105], row=1, col=ci)
    fig.update_yaxes(title_text='Score (0–100)', row=1, col=1)
    fig.update_layout(**base_layout(theme, height=560,
                                    margin=dict(l=56, r=24, t=48, b=48)))
    _style_all_axes(fig, theme)
    _subplot_titles(fig, theme)

    cols = ['Country', 'Region'] + list(metrics)
    table = dff[cols].sort_values(metrics[0], ascending=False).round(1)
    table.columns = ['Country', 'Region'] + list(titles or metrics)
    table['Region'] = table['Region'].map(region_display)
    return fig, table


def population_map(dff, year, theme):
    k = t(theme)
    m = dff[['Country', POP_COL]].dropna(subset=[POP_COL]).copy()
    m['log_pop'] = np.log10((m[POP_COL] / 1e6).clip(lower=0.01))
    fig = go.Figure(go.Choropleth(
        locations=m['Country'], locationmode='country names', z=m['log_pop'],
        colorscale=seq_scale(theme), customdata=m[POP_COL] / 1e6,
        marker_line_color=k['surface'], marker_line_width=0.4,
        colorbar=dict(title=dict(text='Population', font=dict(size=11, color=k['muted'])),
                      tickvals=[-1, 0, 1, 2, 3],
                      ticktext=['0.1M', '1M', '10M', '100M', '1B'],
                      tickfont=dict(size=10, color=k['muted']),
                      thickness=10, outlinewidth=0, len=0.7),
        hovertemplate='<b>%{location}</b><br>Population: %{customdata:.2f}M<extra></extra>'))
    fig.update_layout(**base_layout(theme, height=470,
                                    margin=dict(l=8, r=8, t=8, b=8)),
                      dragmode=False)
    fig.update_geos(showframe=False, showcoastlines=False, showland=True,
                    landcolor=k['land'], bgcolor='rgba(0,0,0,0)',
                    projection_type='natural earth', resolution=50,
                    lonaxis_range=[-180, 180], lataxis_range=[-60, 85])
    table = m[['Country', POP_COL]].sort_values(POP_COL, ascending=False).copy()
    table[POP_COL] = (table[POP_COL] / 1e6).round(2)
    table.columns = ['Country', 'Population (M)']
    return fig, table


def population_bar(dff, hc, year, theme):
    return _top25_bar(dff, POP_COL, hc, theme, scale=1e6,
                      value_fmt=lambda v: f'{v:,.0f}M',
                      axis_title='Population (Millions)')


def gdp_map(dff, year, theme):
    k = t(theme)
    m = dff[['Country', GDP_COL]].dropna().copy()
    m = m[m[GDP_COL] > 0]
    m['GK'] = m[GDP_COL] / 1e3
    m['log_gdp'] = np.log10(m['GK'].clip(lower=0.1))
    # Green sequential ramp (low=pale green, high=deep green)
    green_scale = [[0, '#e8f5e9'], [0.15, '#c8e6c9'], [0.3, '#81c784'],
                   [0.5, '#4caf50'], [0.7, '#2e7d32'], [0.85, '#1b5e20'],
                   [1, '#0d3d14']]
    fig = go.Figure(go.Choropleth(
        locations=m['Country'], locationmode='country names', z=m['log_gdp'],
        colorscale=green_scale, customdata=m['GK'],
        marker_line_color=k['surface'], marker_line_width=0.4,
        colorbar=dict(title=dict(text='GDP / Capita', font=dict(size=11, color=k['muted'])),
                      tickvals=[0, 0.5, 1, 1.5, 2],
                      ticktext=['$1K', '$3K', '$10K', '$30K', '$100K'],
                      tickfont=dict(size=10, color=k['muted']),
                      thickness=10, outlinewidth=0, len=0.7),
        hovertemplate='<b>%{location}</b><br>GDP per capita: $%{customdata:.2f}K<extra></extra>'))
    fig.update_layout(**base_layout(theme, height=470,
                                    margin=dict(l=8, r=8, t=8, b=8)),
                      dragmode=False)
    fig.update_geos(showframe=False, showcoastlines=False, showland=True,
                    landcolor=k['land'], bgcolor='rgba(0,0,0,0)',
                    projection_type='natural earth', resolution=50,
                    lonaxis_range=[-180, 180], lataxis_range=[-60, 85])
    table = m[['Country', 'GK']].sort_values('GK', ascending=False).round(2)
    table.columns = ['Country', 'GDP per capita ($K)']
    return fig, table


def gdp_bar(dff, hc, year, theme):
    return _top25_bar(dff, GDP_COL, hc, theme, scale=1e3,
                      value_fmt=lambda v: f'${v:,.1f}K',
                      axis_title='GDP per Capita (Thousands $)')


def _top25_axis_max(col, scale):
    """Fixed x-axis ceiling for the Top-25 lollipops, computed once across ALL
    years. Keeping the axis constant means the dots visibly move as you scrub
    the year slider, instead of the axis rescaling under them."""
    global_max = pd.to_numeric(df[col], errors='coerce').max() / scale
    if not np.isfinite(global_max) or global_max <= 0:
        return None
    return global_max * 1.04   # a little headroom; value labels spill into the margin


def _top25_bar(dff, col, hc, theme, scale, value_fmt, axis_title):
    """Top-25 lollipop: hairline stem + region-colored dot + value label."""
    k = t(theme)
    rc = region_colors(theme)
    p = dff[['Country', 'Region', col]].dropna().sort_values(col, ascending=False).head(25)
    p['Rank'] = range(1, len(p) + 1)
    p['V'] = p[col] / scale
    p = p.sort_values(col, ascending=True)
    # Rank prefix on the axis label, same convention as the Happiness Drivers chart
    p['Label'] = [f"#{int(rk)} {flag(c)} {c}"
                  for rk, c in zip(p['Rank'], p['Country'])]
    colors = [rc.get(r, k['muted']) for r in p['Region']]
    ops = [1.0 if hc and c == hc else (0.25 if hc else 0.95) for c in p['Country']]

    fig = go.Figure()
    # Grey baseline dot: the 2011 value for each of the Top 25 countries (where
    # available) so as the year slider moves you can see how far each moved
    # from its starting position.
    baseline_yr = min(years)
    base_df = df[df['SPI year'] == baseline_yr][['Country', col]].dropna()
    base_map = base_df.set_index('Country')[col].to_dict()
    base_vals = [(base_map.get(c, None)) for c in p['Country']]
    base_x = [v / scale if v is not None else None for v in base_vals]
    fig.add_trace(go.Scatter(
        x=base_x, y=p['Label'].tolist(), mode='markers',
        marker=dict(size=7, color=k['dim'], opacity=0.6,
                    line=dict(width=1, color=k['surface'])),
        name=f'{baseline_yr} baseline', showlegend=True,
        hovertemplate='%{y}<br>' + str(baseline_yr) + ': %{x:.2f}<extra></extra>'))
    for lab, v in zip(p['Label'], p['V']):
        fig.add_trace(go.Scatter(x=[0, v], y=[lab, lab], mode='lines',
                                 line=dict(color=k['grid'], width=2),
                                 showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=p['V'], y=p['Label'], mode='markers+text',
        marker=dict(size=11, color=colors, opacity=ops,
                    line=dict(width=2, color=k['surface'])),
        text=[value_fmt(v) for v in p['V']], textposition='middle right',
        textfont=dict(size=9.5, color=k['muted']), cliponaxis=False,
        customdata=[[value_fmt(v), region_display(r), rk] for v, r, rk in
                    zip(p['V'], p['Region'], p['Rank'])],
        hovertemplate='<b>%{y}</b><br>#%{customdata[2]} · %{customdata[0]} · '
                      '%{customdata[1]}<extra></extra>',
        showlegend=False))
    for region in regions:
        if region in p['Region'].values:
            fig.add_trace(go.Scatter(y=[None], x=[None], mode='markers',
                                     name=region_display(region),
                                     marker=dict(size=9, color=rc[region]),
                                     showlegend=True))
    # Pin the x-axis so the dots move across a constant scale year to year.
    x_max = _top25_axis_max(col, scale)
    xaxis = dict(title=dict(text=axis_title))
    if x_max:
        xaxis['range'] = [0, x_max]
        xaxis['autorange'] = False
    else:
        xaxis['rangemode'] = 'tozero'
    fig.update_layout(**base_layout(
        theme, height=620, margin=dict(l=182, r=72, t=16, b=100),
        xaxis=xaxis,
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center',
                    yanchor='top', font=dict(size=9), tracegroupgap=4)))
    fig.update_yaxes(showgrid=False,
                     categoryorder='array', categoryarray=list(p['Label']))
    _style_all_axes(fig, theme)

    table = p.sort_values('Rank')[['Rank', 'Country', 'Region', 'V']].copy()
    table['V'] = table['V'].map(value_fmt)
    table.columns = ['Rank', 'Country', 'Region', axis_title]
    return fig, table


# ============================================================= explore ====

def region_analysis(dff, components, highlight_country, year, dim_name, theme):
    """Line chart: one line per region showing component values across years."""
    k = t(theme)
    rc = region_colors(theme)

    # Build a multi-year line chart where each line is a region
    fig = go.Figure()
    for region in regions:
        region_data = df[df['Region'] == region]
        yearly_means = region_data.groupby('SPI year')[components].mean()
        # Use the first component's mean as representative dimension score
        dim_score_col = components[0] if len(components) == 1 else components
        if len(components) > 1:
            yearly_means['_dim_avg'] = yearly_means[components].mean(axis=1)
            col_to_plot = '_dim_avg'
        else:
            col_to_plot = components[0]
        yearly_means = yearly_means.sort_index()
        fig.add_trace(go.Scatter(
            x=yearly_means.index, y=yearly_means[col_to_plot],
            mode='lines+markers', name=region_display(region),
            line=dict(color=rc[region], width=2.5),
            marker=dict(size=5, color=rc[region]),
            hovertemplate=region_display(region) +
                          '<br>Year: %{x}<br>Score: %{y:.2f}<extra></extra>'))

    if highlight_country and highlight_country in dff['Country'].values:
        c_data = df[df['Country'] == highlight_country]
        c_yearly = c_data.groupby('SPI year')[components].mean()
        if len(components) > 1:
            c_yearly['_dim_avg'] = c_yearly[components].mean(axis=1)
            col_to_plot = '_dim_avg'
        else:
            col_to_plot = components[0]
        c_yearly = c_yearly.sort_index()
        fig.add_trace(go.Scatter(
            x=c_yearly.index, y=c_yearly[col_to_plot],
            mode='lines+markers', name=highlight_country,
            line=dict(color=k['ink'], width=3, dash='dash'),
            marker=dict(size=7, color=k['ink'], symbol='diamond'),
            hovertemplate=f'{highlight_country}<br>Year: %{{x}}<br>Score: %{{y:.2f}}<extra></extra>'))

    fig.update_layout(**base_layout(
        theme, height=400, margin=dict(l=56, r=24, t=16, b=72),
        xaxis=dict(title=dict(text='Year'), dtick=2),
        yaxis=dict(title=dict(text='Mean Score (0–100)')),
        legend=dict(orientation='h', y=-0.22, font=dict(size=10))))
    _style_all_axes(fig, theme)

    # Table: current year means per region
    means = dff.groupby('Region')[components].mean().reindex(regions)
    table = means.round(2).reset_index().rename(columns={'index': 'Region'})
    table['Region'] = table['Region'].map(region_display)
    return fig, table


def subindicator_fig(dff, dim_data, selected_comp, year, theme):
    """Small multiples: each sub-indicator's regional means as bars."""
    k = t(theme)
    rc = region_colors(theme)
    indicators = dim_data['raw_indicators'][selected_comp]
    ind_cols = [c for c, _ in indicators if c in dff.columns]
    means = dff.groupby('Region')[ind_cols].mean().reindex(regions)
    # Keep panels in the indicator declaration order. Sorting by each year's
    # mean made the subplots swap positions on every year tick / play frame.
    keep = [c for c in ind_cols
            if means[c].dropna().nunique() > 1 and means[c].std() > 0.01]
    if not keep:
        return None, None
    labels = [_title_case(c.split(' (')[0].strip()) for c in keep]

    n = len(keep)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=labels,   # keep each heading on 1 line
                        vertical_spacing=0.24 if nrows > 1 else 0.12,
                        horizontal_spacing=0.07)
    for i, col in enumerate(keep):
        r, c = i // ncols + 1, i % ncols + 1
        order = means[col].dropna().sort_values(ascending=False).index
        for region in order:
            if region not in regions:
                continue
            val = means.loc[region, col]
            fig.add_trace(go.Bar(
                x=[REGION_SHORT[region]], y=[val], name=region_display(region),
                marker=dict(color=rc[region], cornerradius=3,
                            line=dict(width=1, color=k['surface'])),
                showlegend=False,
                text=[f"{val:.0f}"], textposition='outside',
                textfont=dict(size=9, color=k['muted']),
                hovertemplate=region_display(region) +
                              ': %{y:.2f}<extra></extra>'), row=r, col=c)
        fig.update_xaxes(tickfont=dict(size=9), showgrid=False, row=r, col=c)

    # Legend as dedicated invisible stubs in a FIXED region order. The bars
    # themselves are sorted by value (which changes year to year), so driving
    # the legend off them made entries reshuffle on every play tick. These
    # stubs keep the legend identical across all years — it only encodes color.
    for region in regions:
        fig.add_trace(go.Bar(
            x=[None], y=[None], name=region_display(region),
            marker=dict(color=rc[region]),
            showlegend=True, hoverinfo='skip'), row=1, col=1)

    fig.update_layout(**base_layout(
        theme, height=300 * nrows, margin=dict(l=44, r=20, t=48, b=56),
        legend=dict(orientation='h', y=-0.12, font=dict(size=10),
                    traceorder='normal')))
    _style_all_axes(fig, theme)
    _subplot_titles(fig, theme, size=12)

    table = means[keep].round(1)
    table.columns = labels
    table = table.reset_index()
    table['Region'] = table['Region'].map(region_display)
    return fig, table


# =============================================== country deep dive ========

def spi_trend_fig(country, theme):
    k = t(theme)
    accent = t(theme)['slots'][0]
    if country:
        trend = df[df['Country'] == country][['SPI year', SPI_COL]].dropna()
    else:
        trend = world_df[['SPI year', SPI_COL]].dropna()
    trend = trend.sort_values('SPI year')
    fig = go.Figure(go.Scatter(
        x=trend['SPI year'], y=trend[SPI_COL], mode='lines+markers',
        line=dict(color=accent, width=2, shape='spline', smoothing=0.6),
        marker=dict(size=5, color=accent),
        hovertemplate='%{x}: %{y:.2f}<extra></extra>'))
    if not trend.empty:
        last = trend.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last['SPI year']], y=[last[SPI_COL]], mode='markers+text',
            marker=dict(size=9, color=accent, line=dict(width=2, color=k['surface'])),
            text=[f"{last[SPI_COL]:.1f}"], textposition='middle right',
            textfont=dict(size=11, color=k['ink']),
            hoverinfo='skip', showlegend=False, cliponaxis=False))
    fig.update_layout(**base_layout(
        theme, height=280, margin=dict(l=48, r=52, t=16, b=36),
        showlegend=False,
        xaxis=dict(dtick=2), yaxis=dict(title=dict(text='SPI'))))
    _style_all_axes(fig, theme)
    table = trend.rename(columns={'SPI year': 'Year', SPI_COL: 'SPI'}).round(1)
    return fig, table


# Countries whose land area is too small to be visible as a filled polygon on
# a continental choropleth. For these we drop a labelled pin and zoom the map
# in around the country instead of relying on the fill alone.
# (lon, lat, half-width of the lon/lat window in degrees)
SMALL_COUNTRY_PINS = {
    'Singapore': (103.82, 1.35, 14),
    'Hong Kong': (114.17, 22.32, 14),
    'Bahrain': (50.55, 26.07, 12),
    'Malta': (14.38, 35.90, 11),
    'Luxembourg': (6.13, 49.61, 10),
    'Cyprus': (33.38, 35.13, 12),
    'Mauritius': (57.55, -20.35, 14),
    'Comoros': (43.33, -11.65, 13),
    'Cabo Verde': (-23.62, 15.12, 13),
    'Maldives': (73.51, 3.20, 14),
    'Sao Tome and Principe': (6.61, 0.19, 12),
    'Barbados': (-59.54, 13.19, 11),
    'Trinidad and Tobago': (-61.22, 10.69, 11),
    'Jamaica': (-77.30, 18.11, 12),
    'Djibouti': (42.59, 11.83, 12),
    'Eswatini': (31.47, -26.52, 11),
    'Lesotho': (28.23, -29.61, 11),
    'Rwanda': (29.87, -1.94, 11),
    'Burundi': (29.92, -3.37, 11),
    'Guinea-Bissau': (-15.18, 11.80, 11),
    'Timor-Leste': (125.73, -8.87, 13),
    'Qatar': (51.18, 25.35, 12),
    'Kuwait': (47.48, 29.31, 12),
    'Lebanon': (35.86, 33.85, 11),
    'Israel': (34.85, 31.05, 11),
    'Slovenia': (14.82, 46.15, 10),
    'Montenegro': (19.37, 42.71, 10),
    'Moldova': (28.37, 47.41, 10),
    'Armenia': (45.04, 40.07, 10),
    'Estonia': (25.01, 58.60, 10),
    'Latvia': (24.60, 56.88, 10),
    'Equatorial Guinea': (10.27, 1.65, 12),
    'Fiji': (177.97, -17.71, 14),
    'Solomon Islands': (160.16, -9.65, 15),
}


def locator_map(country, region_name, dff, theme):
    k = t(theme)
    # Instead of Plotly's built-in scopes (which are fixed continent boxes
    # that don't match our region definitions — Mexico gets cut off by
    # "south america", Oman gets cut off by "africa"), use explicit
    # lon/lat viewports per region that actually contain every country
    # in that region.
    _REGION_VIEWPORT = {
        'Europe':                      (-12, 42, 28, 72),
        'Sub-Saharan Africa':          (-20, 60, -36, 18),
        'Middle East & North Africa':  (-18, 65, 10, 42),
        'North America':               (-170, -50, 10, 75),
        'Latin America & Caribbean':   (-120, -30, -58, 34),
        'East Asia & Pacific':         (70, 180, -48, 55),
        'South Asia':                  (60, 100, 2, 40),
        'Central Asia':                (45, 90, 30, 56),
    }
    # Countries that span far beyond their region's viewport and need a
    # wider or different view. Show these at world scale instead.
    _WIDE_COUNTRIES = {'Russia', 'United States', 'France'}
    pin = SMALL_COUNTRY_PINS.get(country)

    # ONE choropleth trace, not two. Two overlapping traces (all countries in
    # grey, then the selected country on top) left the highlight hidden until
    # some interaction forced a redraw and reordered the paint — the
    # "highlighted but not visible until I move" bug. Encoding the highlight
    # as a z-value inside a single trace removes the overlap entirely, so the
    # colour is correct on first paint.
    locs = dff['Country'].tolist()
    if country not in locs:
        locs = locs + [country]
    zvals = [1 if c == country else 0 for c in locs]
    accent = k['slots'][0]
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=locs, locationmode='country names',
        z=zvals, showscale=False,
        # Two-stop scale with a hard break: 0 -> land grey, 1 -> accent.
        colorscale=[[0.0, k['land']], [0.5, k['land']],
                    [0.5, accent], [1.0, accent]],
        zmin=0, zmax=1,
        marker_line_color=k['surface'], marker_line_width=0.4,
        text=locs, hoverinfo='skip'))

    geo = dict(showframe=False, showcoastlines=False, showland=False,
               bgcolor='rgba(0,0,0,0)', resolution=50,
               projection_type='natural earth')

    if pin:
        # Too small to see as a polygon: add a ring + label pin, and zoom the
        # viewport to a tight window around the country so it's identifiable.
        lon, lat, span = pin
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat], mode='markers',
            marker=dict(size=26, color='rgba(0,0,0,0)',
                        line=dict(width=2.5, color=accent)),
            hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat], mode='markers+text',
            marker=dict(size=8, color=accent,
                        line=dict(width=1.5, color=k['surface'])),
            text=[f'  {country}'], textposition='middle right',
            textfont=dict(size=12, color=k['ink']),
            hovertemplate=f'<b>{country}</b><extra></extra>', showlegend=False))
        geo.update(lonaxis=dict(range=[lon - span, lon + span]),
                   lataxis=dict(range=[lat - span * 0.72, lat + span * 0.72]),
                   showcountries=True, countrycolor=k['surface'],
                   countrywidth=0.5, showland=True, landcolor=k['land'])
    else:
        vp = _REGION_VIEWPORT.get(region_name)
        if country in _WIDE_COUNTRIES or not vp:
            # Country spans multiple continents or has no region viewport —
            # show the whole world so nothing gets clipped.
            geo.update(showcountries=True, countrycolor=k['surface'],
                       countrywidth=0.5, showland=True, landcolor=k['land'])
        else:
            lon0, lon1, lat0, lat1 = vp
            geo.update(lonaxis=dict(range=[lon0, lon1]),
                       lataxis=dict(range=[lat0, lat1]),
                       showcountries=True, countrycolor=k['surface'],
                       countrywidth=0.5, showland=True, landcolor=k['land'])

    fig.update_layout(**base_layout(theme, height=300,
                                    margin=dict(l=0, r=0, t=8, b=0)))
    fig.update_geos(**geo)
    return fig


def comparison_fig(comps, country_vals, region_vals, world_vals,
                   country, region_name, theme):
    """Dumbbell rows: world → country connector per component, region as
    context dot. The country is the point; the rest is context."""
    k = t(theme)
    accent = k['slots'][0]
    context = '#86b6ef' if theme != 'dark' else '#256abf'
    rows = comps[::-1]
    idx = {c: i for i, c in enumerate(comps)}

    def vals(source):
        return [None if source is None else source[idx[c]] for c in rows]

    cv, rv, wv = vals(country_vals), vals(region_vals), vals(world_vals)

    fig = go.Figure()
    if country_vals is not None:
        for c, cval, wval in zip(rows, cv, wv):
            if cval is None or wval is None or pd.isna(cval) or pd.isna(wval):
                continue
            fig.add_trace(go.Scatter(x=[wval, cval], y=[c, c], mode='lines',
                                     line=dict(color=k['grid'], width=2),
                                     showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=wv, y=rows, mode='markers', name='World',
        marker=dict(size=18, color=k['muted'], opacity=0.75,
                    line=dict(width=2, color=k['surface'])),
        hovertemplate='%{y}<br>World: %{x:.2f}<extra></extra>'))
    if region_vals is not None:
        region_label = f'{region_display(region_name)} avg'
        fig.add_trace(go.Scatter(
            x=rv, y=rows, mode='markers', name=region_label,
            marker=dict(size=13, color=context,
                        line=dict(width=2, color=k['surface'])),
            hovertemplate='%{y}<br>' + region_label +
                          ': %{x:.2f}<extra></extra>'))
    if country_vals is not None:
        fig.add_trace(go.Scatter(
            x=cv, y=rows, mode='markers+text', name=country,
            marker=dict(size=9, color=accent,
                        line=dict(width=2, color=k['surface'])),
            text=[f'{v:.0f}' if v is not None and pd.notna(v) else '' for v in cv],
            textposition='top center', textfont=dict(size=10, color=k['ink2']),
            cliponaxis=False,
            hovertemplate='%{y}<br>' + (country or '') + ': %{x:.2f}<extra></extra>'))
    fig.update_layout(**base_layout(
        theme, height=320, margin=dict(l=24, r=32, t=16, b=64),
        xaxis=dict(title=dict(text='Score (0–100)'), range=[0, 100]),
        yaxis=dict(tickfont=dict(size=12, color=k['ink2'])),
        legend=dict(orientation='h', y=-0.26, font=dict(size=11))))
    _style_all_axes(fig, theme)
    fig.update_yaxes(tickfont=dict(size=12, color=k['ink2']))
    table = pd.DataFrame({'Component': comps})
    if country_vals is not None:
        table[country or 'Country'] = country_vals
    if region_vals is not None:
        table[f'{region_display(region_name)} avg'] = region_vals
    table['World'] = world_vals
    return fig, table.round(1)


def indicator_charts(dff, country, dim_data, theme):
    """Four component quadrants, four deliberate chart forms (as designed
    originally): lollipop / gauges / diverging-from-median / percentile bars.
    Values are SPI-scored indicators expressed as world percentiles."""
    k = t(theme)
    row = dff[dff['Country'] == country].iloc[0]

    def percentile(col):
        """Tie-adjusted (mid-rank) percentile: countries strictly below count
        fully, countries tied with this one count as half — otherwise a
        country tied with many others at a floor/ceiling value (common on
        indicators like Undernourishment) gets an artificially low percentile."""
        series = dff[col].dropna()
        val = row.get(col)
        if pd.isna(val) or series.empty:
            return None, val
        below = (series < val).sum()
        tied = (series == val).sum()
        return (below + 0.5 * tied) / len(series) * 100, val

    comps = list(dim_data['raw_indicators'].keys())
    table_rows = []
    # Blank the top-right (gauge) subplot title — a matching bold heading is
    # added manually below so all 4 quadrant headers share the same font,
    # size, and weight.
    sub_titles = [f'<b>{comps[0]}</b>', '', f'<b>{comps[2]}</b>', f'<b>{comps[3]}</b>']
    # HSPACE needs to be wide enough that row 2's right-column y-axis tick
    # labels (indicator names, rendered just left of that subplot) don't
    # bleed into the left column's plot area — a narrow gap here is what
    # caused the "Inclusive Society" / "Advanced Education" overlap.
    VSPACE, HSPACE = 0.14, 0.26
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'domain'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        subplot_titles=sub_titles,
        vertical_spacing=VSPACE, horizontal_spacing=HSPACE)

    # -- 1: lollipop ------------------------------------------------------
    labels, pcts, raws, colors1 = [], [], [], []
    for col, _hb in dim_data['raw_indicators'][comps[0]]:
        if col not in dff.columns:
            continue
        p, v = percentile(col)
        if p is None or pd.isna(v):   # drop indicators with no value
            continue
        labels.append(_title_case(col.split(' (')[0].strip()))   # keep label on 1 line
        pcts.append(p)
        raws.append(v)
        colors1.append(status_color(p, theme))
        table_rows.append((comps[0], col.split(' (')[0].strip(), v, p))
    for lab, p in zip(labels, pcts):
        fig.add_trace(go.Scatter(x=[0, p], y=[lab, lab], mode='lines',
                                 line=dict(color=k['grid'], width=2),
                                 showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pcts, y=labels, mode='markers+text',
        marker=dict(size=13, color=colors1, line=dict(width=2, color=k['surface'])),
        text=[f'{p:.0f}' for p in pcts], textposition='middle right',
        textfont=dict(size=9, color=k['ink2']), cliponaxis=False,
        customdata=raws, showlegend=False,
        hovertemplate='%{y}<br>Percentile: %{x:.2f}/100<br>Score: %{customdata:.2f}<extra></extra>'),
        row=1, col=1)
    fig.update_xaxes(range=[-2, 104], title_text='World Percentile', row=1, col=1)

    # -- 2: gauges (2 per row) ------------------------------------------------
    # The gauge quadrant uses free-floating paper coordinates (go.Indicator
    # domains aren't tied to the subplot grid), so its vertical extent is
    # explicitly clamped to stay inside row 1's domain — that's what caused
    # gauges to bleed into the bar chart quadrant below. With VSPACE between
    # the two subplot rows, row 1 spans y in [0.5 + VSPACE/2, 1.0].
    row1_y_bottom = 0.5 + VSPACE / 2
    ind2 = [iv for iv in dim_data['raw_indicators'][comps[1]] if iv[0] in dff.columns]
    # drop indicators with no value
    ind2 = [(col, hb) for col, hb in ind2 if percentile(col)[0] is not None]
    tint = dict(good='rgba(12,163,12,0.16)', warn='rgba(250,178,25,0.16)',
                crit='rgba(208,59,59,0.16)')
    # Component heading for the whole gauge quadrant — same size/weight/family
    # as the other three quadrant headers (which are bolded via sub_titles).
    # Column 2's actual x-domain, derived from HSPACE (2-col grid: col2 spans
    # [(1+HSPACE)/2, 1]) — keeps the gauge quadrant aligned with the grid
    # even if HSPACE changes later.
    col2_x_lo = (1 + HSPACE) / 2
    gauge_x_center = (col2_x_lo + 1.0) / 2
    fig.add_annotation(
        x=gauge_x_center, y=0.99, xref='paper', yref='paper',
        text=f'<b>{comps[1]}</b>', showarrow=False,
        font=dict(size=16, color=k['ink'], family=FONT_STACK),
        xanchor='center', yanchor='bottom')
    # 2 gauges per row within the top-right quadrant. Always use the full
    # available height inside row 1 (down to its own bottom edge, with a
    # small buffer) — this maximizes gauge size while guaranteeing the grid
    # never overlaps the bar-chart quadrant below, regardless of how many
    # gauge rows a given component needs.
    gcols = 2
    grows = max((len(ind2) + gcols - 1) // gcols, 1)
    x_lo, x_hi = col2_x_lo + 0.01, 0.99
    y_hi = 0.94    # small, consistent gap below the heading — the other 3
                   # quadrants have their subplot title sit almost flush
                   # with their axis top, so match that here
    y_lo = row1_y_bottom + 0.02
    col_span = (x_hi - x_lo) / gcols
    row_span = (y_hi - y_lo) / grows
    for i, (col, _hb) in enumerate(ind2):
        p, v = percentile(col)
        table_rows.append((comps[1], col.split(' (')[0].strip(), v, p))
        gr, gc = i // gcols, i % gcols
        x0 = x_lo + gc * col_span + 0.015
        x1 = x_lo + (gc + 1) * col_span - 0.015
        cell_top = y_hi - gr * row_span
        cell_bot = cell_top - row_span
        # Within each cell: gauge occupies the top ~50%, its label sits
        # immediately beneath it, and the remaining bottom ~35% is empty
        # padding — an extra line-gap after each gauge so rows breathe.
        g_top = cell_top
        g_bot = cell_top - row_span * 0.50
        label_y = g_bot - row_span * 0.02
        fig.add_trace(go.Indicator(
            mode='gauge+number', value=p if p is not None else 0,
            number=dict(font=dict(size=13, color=k['ink'], family=FONT_STACK),
                        suffix='<span style="font-size:0.6em">/100</span>'),
            gauge=dict(
                axis=dict(range=[0, 100], showticklabels=False, ticks=''),
                bar=dict(color=status_color(p, theme), thickness=0.55),
                bgcolor='rgba(0,0,0,0)', borderwidth=0,
                steps=[dict(range=[0, 40], color=tint['crit']),
                       dict(range=[40, 70], color=tint['warn']),
                       dict(range=[70, 100], color=tint['good'])]),
            domain=dict(x=[x0, x1], y=[g_bot, g_top])))
        # label below the gauge, on a single line
        fig.add_annotation(
            x=(x0 + x1) / 2, y=label_y, xref='paper', yref='paper',
            text=_title_case(col.split(' (')[0].strip()), showarrow=False,
            font=dict(size=10, color=k['ink2'], family=FONT_STACK),
            xanchor='center', yanchor='top')

    # -- 3: diverging from median ------------------------------------------
    h_labels, h_diffs, h_colors, h_raws = [], [], [], []
    for col, _hb in dim_data['raw_indicators'][comps[2]]:
        if col not in dff.columns:
            continue
        p, v = percentile(col)
        if p is None or pd.isna(v):   # drop indicators with no value
            continue
        table_rows.append((comps[2], col.split(' (')[0].strip(), v, p))
        h_labels.append(_title_case(col.split(' (')[0].strip()))   # keep label on 1 line
        h_diffs.append(round(p - 50, 2))
        h_colors.append(status_color(p, theme))
        h_raws.append(round(v, 2) if pd.notna(v) else v)
    fig.add_trace(go.Bar(
        y=h_labels, x=h_diffs, orientation='h', width=0.55,
        marker=dict(color=h_colors, cornerradius=4,
                    line=dict(width=1, color=k['surface'])),
        text=[f'{d:+.0f}' for d in h_diffs], textposition='outside',
        textfont=dict(size=9, color=k['ink2']), cliponaxis=False,
        customdata=h_raws, showlegend=False,
        hovertemplate='%{y}<br>%{x:+.2f} vs world median<br>Score: %{customdata:.2f}<extra></extra>'),
        row=2, col=1)
    fig.update_xaxes(range=[-55, 55], title_text='Distance from world median',
                     zeroline=True, zerolinecolor=k['axis'], zerolinewidth=1.5,
                     row=2, col=1)

    # -- 4: percentile bars --------------------------------------------------
    s_labels, s_pcts, s_raws, s_colors = [], [], [], []
    for col, _hb in dim_data['raw_indicators'][comps[3]]:
        if col not in dff.columns:
            continue
        p, v = percentile(col)
        if p is None or pd.isna(v):   # drop indicators with no value
            continue
        s_labels.append(_title_case(col.split(' (')[0].strip()))   # keep label on 1 line
        s_pcts.append(p)
        s_raws.append(v)
        s_colors.append(status_color(p, theme))
        table_rows.append((comps[3], col.split(' (')[0].strip(), v, p))
    fig.add_trace(go.Bar(
        y=s_labels, x=s_pcts, orientation='h', width=0.55,
        marker=dict(color=s_colors, cornerradius=4,
                    line=dict(width=1, color=k['surface'])),
        text=[f'{p:.0f}' for p in s_pcts], textposition='outside',
        textfont=dict(size=9, color=k['ink2']), cliponaxis=False,
        customdata=s_raws, showlegend=False,
        hovertemplate='%{y}<br>Percentile: %{x:.2f}/100<br>Score: %{customdata:.2f}<extra></extra>'),
        row=2, col=2)
    fig.update_xaxes(range=[0, 104], title_text='World Percentile', row=2, col=2)

    fig.update_layout(**base_layout(theme, height=1080,
                                    margin=dict(l=210, r=30, t=110, b=56)))
    _style_all_axes(fig, theme)
    # The 4 subplot titles already carry <b> tags (added above) so every
    # quadrant heading — including the gauge one, added as its own matching
    # annotation — shares the same font, size, weight, and color.
    _subplot_titles(fig, theme, size=16, only_subplot=4)

    table = pd.DataFrame(table_rows,
                         columns=['Component', 'Indicator', 'Score (0–100)',
                                  'World Percentile'])
    table = table.round(1)
    # Median country score per indicator, for comparison in the data view.
    medians = {}
    for _dim, indicators in dim_data['raw_indicators'].items():
        for col, _hb in indicators:
            if col in dff.columns:
                medians[col.split(' (')[0].strip()] = dff[col].median()
    table['Median Country Score'] = table['Indicator'].map(medians).round(1)
    return fig, table


# ====================================================== social progress ====

TIER_BINS = [0, 35, 45, 55, 65, 75, 85, 100]
TIER_LABELS = ['Tier 7', 'Tier 6', 'Tier 5', 'Tier 4', 'Tier 3', 'Tier 2', 'Tier 1']


def spi_bubble(dff, year, theme, focus=None):
    """Beeswarm strip: x = SPI, bubble area = population, 7 tiers.

    Tier color is a redundant channel — position, boundary rules, legend and
    tooltips all carry tier identity."""
    k = t(theme)
    ramp = tier_ramp(theme)  # worst -> best
    tier_color = {lab: ramp[i] for i, lab in enumerate(TIER_LABELS)}

    d = dff.dropna(subset=[SPI_COL, POP_COL]).copy()
    d = d[d[POP_COL] > 0]
    d['Tier'] = pd.cut(d[SPI_COL], bins=TIER_BINS, labels=TIER_LABELS,
                       include_lowest=True)
    # Deterministic y-jitter per country (hash-based, not position-based)
    # so bubbles don't jump around when the country set changes across years.
    d['yj'] = d['Country'].apply(lambda c: (hash(c) % 10000) / 10000 * 5 - 2.5)
    d['sz'] = np.sqrt(d[POP_COL] / 1e6) * 3
    # World view: label top 25 most populous countries
    # Region focus: label ALL countries in that region
    top_pop_25 = set(d.nlargest(25, POP_COL)['Country'])

    # --- label collision avoidance -----------------------------------------
    # Only two label positions (top/bottom center) meant countries close in
    # both SPI and jitter printed their names on top of each other (e.g. New
    # Zealand vs Singapore). Walk the labelled points left-to-right and, for
    # each one, pick the placement that sits furthest from already-placed
    # labels — cycling through 6 offsets instead of 2.
    _POSITIONS = ['top center', 'bottom center', 'middle right', 'middle left',
                  'top right', 'bottom left']
    # approximate occupied box per placement, in data units
    _OFFSETS = {'top center': (0.0, 0.42), 'bottom center': (0.0, -0.42),
                'middle right': (2.2, 0.0), 'middle left': (-2.2, 0.0),
                'top right': (1.8, 0.34), 'bottom left': (-1.8, -0.34)}

    # Shared across every tier trace — bubbles are drawn one trace per tier, so
    # a per-trace list would let labels from different tiers still collide.
    placed = []              # committed (x, y) label anchors

    def _assign_positions(sub, labelled_mask):
        """Greedy placement: for each labelled point choose the offset whose
        resulting anchor is furthest from anchors already committed."""
        order = np.argsort(sub[SPI_COL].values)
        result = [None] * len(sub)
        xs = sub[SPI_COL].values
        ys = sub['yj'].values
        for i in order:
            if not labelled_mask[i]:
                result[i] = 'top center'
                continue
            if not placed:
                best = 'top center'
            else:
                best, best_score = 'top center', -1.0
                for pos in _POSITIONS:
                    dx, dy = _OFFSETS[pos]
                    ax, ay = xs[i] + dx, ys[i] + dy
                    # distance to nearest committed label (x scaled: 1 SPI pt is
                    # visually ~0.28 jitter units, so weight x less than y)
                    near = min(((ax - px) * 0.28) ** 2 + (ay - py) ** 2
                               for px, py in placed)
                    if near > best_score:
                        best, best_score = pos, near
            dx, dy = _OFFSETS[best]
            placed.append((xs[i] + dx, ys[i] + dy))
            result[i] = best
        return result

    fig = go.Figure()
    # tier boundary hairlines + tier labels along the top
    for b in TIER_BINS[1:-1]:
        fig.add_shape(type='line', x0=b, x1=b, y0=-3.2, y1=3.6,
                      line=dict(color=k['grid'], width=1))
    edges = [10] + TIER_BINS[1:-1] + [100]
    for i, lab in enumerate(TIER_LABELS):         # left->right = Tier 7..Tier 1
        lo, hi = edges[i], edges[i + 1]
        fig.add_annotation(x=(lo + hi) / 2, y=4.35, text=lab, showarrow=False,
                           font=dict(size=10, color=k['muted']))
    for tier in TIER_LABELS[::-1]:
        td = d[d['Tier'] == tier].sort_values('sz', ascending=False)
        # Draw bigger bubbles first so smaller ones render on top (stay visible)
        if td.empty:
            continue
        if focus:
            colors = [tier_color[tier] if r == focus else k['dim']
                      for r in td['Region']]
            ops = [0.9 if r == focus else 0.35 for r in td['Region']]
            # Show ALL country names for the focused region
            texts = [c if r == focus else ''
                     for c, r in zip(td['Country'], td['Region'])]
        else:
            colors, ops = tier_color[tier], 0.75
            texts = [c if c in top_pop_25 else '' for c in td['Country']]
        label_mask = [bool(s) for s in texts]
        fig.add_trace(go.Scatter(
            x=td[SPI_COL], y=td['yj'], mode='markers+text',
            marker=dict(size=td['sz'], color=colors, opacity=ops,
                        line=dict(width=1, color=k['surface'])),
            text=texts,
            textposition=_assign_positions(td, label_mask),
            textfont=dict(size=9, color=k['ink2']),
            name=tier, legendrank=TIER_LABELS.index(tier),
            showlegend=not focus,   # hide from legend when focused (dummy below)
            customdata=np.stack([td['Country'], td[POP_COL] / 1e6,
                                 td['Region'].map(region_display),
                                 td['Tier'].astype(str)], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b><br>SPI: %{x:.2f} (%{customdata[3]})'
                          '<br>Population: %{customdata[1]:.2f}M<br>%{customdata[2]}'
                          '<extra></extra>'))
    # When a region is focused, add invisible dummy traces so the legend always
    # shows the correct tier colors (not the dimmed gray).
    if focus:
        for tier in TIER_LABELS[::-1]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=tier_color[tier]),
                name=tier, legendrank=TIER_LABELS.index(tier),
                showlegend=True))
    fig.update_layout(**base_layout(
        theme, height=640, margin=dict(l=40, r=40, t=40, b=64),
        xaxis=dict(title=dict(text='Social Progress Index'), range=[10, 100],
                   showgrid=False),
        yaxis=dict(visible=False, range=[-3.7, 4.8]),
        legend=dict(orientation='h', y=-0.1, font=dict(size=10))))
    _style_all_axes(fig, theme)

    table = d[['Country', 'Region', SPI_COL, 'Tier', POP_COL]].copy()
    table[POP_COL] = (table[POP_COL] / 1e6).round(2)
    table.columns = ['Country', 'Region', 'SPI', 'Tier', 'Population (M)']
    table['Region'] = table['Region'].map(region_display)
    return fig, table.sort_values('SPI', ascending=False).round(1)


def gdp_scatter(dff, year, theme, focus=None):
    k = t(theme)
    rc = region_colors(theme)
    d = dff[['Country', 'Region', SPI_COL, GDP_COL]].dropna()
    d = d[d[GDP_COL] > 0]
    xv = d[GDP_COL] / 1e3

    fig = go.Figure()
    if focus:
        rest, sel = d[d['Region'] != focus], d[d['Region'] == focus]
        for part, color, op, name, show in (
                (rest, k['dim'], 0.55, 'Other regions', True),
                (sel, rc[focus], 0.9, region_display(focus), True)):
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part[GDP_COL] / 1e3, y=part[SPI_COL], mode='markers',
                marker=dict(size=9, color=color, opacity=op,
                            line=dict(width=1, color=k['surface'])),
                name=name, showlegend=show,
                customdata=np.stack([part['Country'], part['Region'].map(region_display)],
                                    axis=-1),
                hovertemplate='<b>%{customdata[0]}</b><br>GDP: $%{x:.2f}K'
                              '<br>SPI: %{y:.2f}<br>%{customdata[1]}<extra></extra>'))
    else:
        # dots shaded by SPI tier (same ramp as the global view)
        ramp = tier_ramp(theme)
        tier_color = {lab: ramp[i] for i, lab in enumerate(TIER_LABELS)}
        dt = d.copy()
        dt['Tier'] = pd.cut(dt[SPI_COL], bins=TIER_BINS, labels=TIER_LABELS,
                            include_lowest=True)
        for tier in TIER_LABELS:
            td = dt[dt['Tier'] == tier]
            if td.empty:
                continue
            fig.add_trace(go.Scatter(
                x=td[GDP_COL] / 1e3, y=td[SPI_COL], mode='markers',
                marker=dict(size=9, color=tier_color[tier], opacity=0.85,
                            line=dict(width=1, color=k['surface'])),
                name=tier, legendrank=TIER_LABELS.index(tier),
                customdata=np.stack([td['Country'], td['Region'].map(region_display)],
                                    axis=-1),
                hovertemplate='<b>%{customdata[0]}</b><br>GDP: $%{x:.2f}K'
                              '<br>SPI: %{y:.2f} (' + tier + ')'
                              '<br>%{customdata[1]}<extra></extra>'))
    if len(d) > 3:
        logx = np.log(xv.values)
        a, b = np.polyfit(logx, d[SPI_COL].values, 1)
        xl = np.linspace(xv.min(), xv.max(), 200)
        fig.add_trace(go.Scatter(x=xl, y=a * np.log(xl) + b, mode='lines',
                                 line=dict(color=k['ink2'], width=2),
                                 showlegend=False, hoverinfo='skip'))
        # Correlate against LOG GDP — that's the curve actually fitted above.
        # Using raw GDP here reported r = 0.79 while the trend line and the
        # Takeaway text describe the log relationship (r = 0.91).
        corr = np.corrcoef(logx, d[SPI_COL].values)[0, 1]
        fig.add_annotation(text=f"log GDP vs SPI:  r = {corr:.2f} · "
                                f"R\u00b2 = {corr**2:.2f}",
                           xref='paper', yref='paper',
                           x=0.97, y=0.04, showarrow=False, xanchor='right',
                           font=dict(size=12, color=k['muted']))
        # Annotate diminishing returns inflection
        fig.add_annotation(x=20, y=a * np.log(20) + b,
                           text='Diminishing Returns →',
                           showarrow=True, arrowhead=2, arrowsize=0.8,
                           arrowcolor=k['muted'], ax=-60, ay=-30,
                           font=dict(size=10, color=k['muted']))
    fig.update_layout(**base_layout(
        theme, height=440,
        xaxis=dict(title=dict(text='GDP per Capita (Thousands $)'),
                   range=[0, 140]),   # fixed range so dots move, axis stays still
        yaxis=dict(title=dict(text='Social Progress Index'), range=[10, 100]),
        legend=dict(orientation='h', y=-0.16, font=dict(size=11))))
    _style_all_axes(fig, theme)

    table = d.copy()
    table[GDP_COL] = (table[GDP_COL] / 1e3).round(1)
    table.columns = ['Country', 'Region', 'SPI', 'GDP per capita ($K)']
    table['Region'] = table['Region'].map(region_display)
    return fig, table.sort_values('SPI', ascending=False).round(1)


def outlier_fig(theme):
    """Over/under-performers vs the GDP–SPI trend (residual beyond ±8)."""
    k = t(theme)
    latest = df[df['SPI year'] == LATEST]
    d = latest[['Country', 'Region', SPI_COL, GDP_COL]].dropna()
    d = d[d[GDP_COL] > 0].copy()
    xv = d[GDP_COL] / 1e3
    logx = np.log(xv.values)
    a, b = np.polyfit(logx, d[SPI_COL].values, 1)
    d['residual'] = d[SPI_COL] - (a * logx + b)
    d['category'] = 'As expected'
    d.loc[d['residual'] > 8, 'category'] = 'Overperformer'
    d.loc[d['residual'] < -8, 'category'] = 'Underperformer'
    cat_colors = {'Overperformer': k['div_pos'],
                  'Underperformer': k['div_neg'],
                  'As expected': k['dim']}

    fig = go.Figure()
    xl = np.linspace(xv.min(), xv.max(), 200)
    fig.add_trace(go.Scatter(x=xl, y=a * np.log(xl) + b, mode='lines',
                             line=dict(color=k['muted'], width=1.5, dash='dot'),
                             name='Expected from GDP', showlegend=True,
                             hoverinfo='skip'))
    for cat in ['As expected', 'Overperformer', 'Underperformer']:
        cd = d[d['category'] == cat]
        if cd.empty:
            continue
        notable = cat != 'As expected'
        # Show ALL outliers (8+ point difference) with smart text placement
        if notable:
            texts = list(cd['Country'])
            # Cycle through 4 positions to minimize overlap
            pos_cycle = ['top center', 'bottom center', 'top right', 'top left']
            positions = [pos_cycle[i % len(pos_cycle)] for i in range(len(cd))]
        else:
            texts = None
            positions = 'top center'
        fig.add_trace(go.Scatter(
            x=cd[GDP_COL] / 1e3, y=cd[SPI_COL],
            mode='markers+text' if notable else 'markers',
            marker=dict(size=10 if notable else 7, color=cat_colors[cat],
                        opacity=0.95 if notable else 0.5,
                        line=dict(width=1, color=k['surface'])),
            text=texts,
            textposition=positions,
            textfont=dict(size=8, color=k['ink2']),
            name=cat,
            customdata=np.stack([cd['Country'], cd['residual'].round(2).astype(str)], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b><br>GDP: $%{x:.2f}K · SPI: %{y:.2f}'
                          '<br>%{customdata[1]} vs expected<extra></extra>'))
    fig.update_layout(**base_layout(
        theme, height=500, margin=dict(l=56, r=30, t=24, b=64),
        xaxis=dict(title=dict(text='GDP per Capita (Thousands $)')),
        yaxis=dict(title=dict(text='Social Progress Index'), range=[15, 100]),
        legend=dict(orientation='h', y=-0.14, font=dict(size=11))))
    _style_all_axes(fig, theme)

    out = d[d['category'] != 'As expected'].sort_values('residual', ascending=False)
    table = out[['Country', 'Region', SPI_COL, GDP_COL, 'residual', 'category']].copy()
    table[GDP_COL] = (table[GDP_COL] / 1e3).round(1)
    table['residual'] = table['residual'].round(1)
    table.columns = ['Country', 'Region', 'SPI', 'GDP per capita ($K)',
                     'SPI vs expected', 'Category']
    table['Region'] = table['Region'].map(region_display)
    return fig, table.round(1)


def movers_fig(theme):
    k = t(theme)
    earliest = df[df['SPI year'] == EARLIEST][['Country', 'Region', SPI_COL]].dropna()
    latest = df[df['SPI year'] == LATEST][['Country', 'Region', SPI_COL]].dropna()
    both = earliest.merge(latest, on=['Country', 'Region'], suffixes=('_start', '_end'))
    both['change'] = (both[f'{SPI_COL}_end'] - both[f'{SPI_COL}_start']).round(2)
    show = pd.concat([both.nsmallest(5, 'change').sort_values('change'),
                      both.nlargest(8, 'change').sort_values('change')])
    colors = [k['div_pos'] if c > 0 else k['div_neg'] for c in show['change']]

    fig = go.Figure(go.Bar(
        y=show['Country'], x=show['change'], orientation='h',
        marker=dict(color=colors, cornerradius=4,
                    line=dict(width=1, color=k['surface'])),
        text=[f"{c:+.1f}" for c in show['change']], textposition='outside',
        textfont=dict(size=10, color=k['ink2']), cliponaxis=False,
        customdata=np.stack([show[f'{SPI_COL}_start'], show[f'{SPI_COL}_end']], axis=-1),
        hovertemplate='<b>%{y}</b><br>%{customdata[0]:.2f} → %{customdata[1]:.2f}'
                      '  (%{x:+.2f})<extra></extra>'))
    fig.update_layout(**base_layout(
        theme, height=480, margin=dict(l=130, r=60, t=16, b=48),
        xaxis=dict(title=dict(text=f'SPI Change, {EARLIEST} → {LATEST}'),
                   zeroline=True, zerolinecolor=k['axis'], zerolinewidth=1.5),
        showlegend=False))
    fig.update_yaxes(showgrid=False)
    _style_all_axes(fig, theme)

    table = show.sort_values('change', ascending=False)[
        ['Country', 'Region', f'{SPI_COL}_start', f'{SPI_COL}_end', 'change']].copy()
    table.columns = ['Country', 'Region', f'SPI {EARLIEST}', f'SPI {LATEST}', 'Change']
    table['Region'] = table['Region'].map(region_display)
    return fig, table.round(1)


def happiness_factors(hdf, year, theme):
    """Stacked horizontal bar: top 20 happiest countries, 6 factors stacked,
    with total happiness score annotated at the end of each bar."""
    k = t(theme)
    fc = factor_colors(theme)
    factor_cols = {f'Explained by: {"Log GDP per capita" if f == "GDP per capita" else f}': f
                   for f in FACTOR_ORDER}
    top = hdf.nsmallest(20, 'Rank').sort_values('Happiness Score', ascending=True).copy()
    top['Label'] = ['#' + str(int(r)) + ' ' + flag(c) + ' ' + c
                    for r, c in zip(top['Rank'], top['Country'])]
    pf = [c for c in factor_cols if c in top.columns]
    factor_names = [factor_cols[c] for c in pf]

    # Rescale factors so they sum to the total happiness score
    fs = top[pf].sum(axis=1)
    scale = top['Happiness Score'] / fs.replace(0, np.nan)
    for c in pf:
        top[c] = top[c] * scale

    fig = go.Figure()
    for i, col in enumerate(pf):
        lbl = factor_cols[col]
        is_last = i == len(pf) - 1
        fig.add_trace(go.Bar(
            y=top['Label'], x=top[col], orientation='h', name=lbl,
            marker=dict(color=fc[lbl], line=dict(width=1, color=k['surface'])),
            text=[f"{s:.1f}" for s in top['Happiness Score']] if is_last else None,
            textposition='outside' if is_last else 'none',
            textfont=dict(size=10, color=k['ink2']), cliponaxis=False,
            hovertemplate='%{y}<br>' + lbl + ': %{x:.2f}<extra></extra>'))
    fig.update_layout(**base_layout(
        theme, height=600, margin=dict(l=130, r=54, t=16, b=68),
        barmode='stack', bargap=0.35,
        xaxis=dict(title=dict(text='Happiness Score (Factors Rescaled to Total)')),
        legend=dict(orientation='h', y=-0.18, font=dict(size=10),
                    traceorder='normal')))
    fig.update_yaxes(showgrid=False)
    _style_all_axes(fig, theme)

    table = top.sort_values('Rank')[['Rank', 'Country', 'Happiness Score'] + pf].copy()
    table.columns = (['Rank', 'Country', 'Happiness Score'] + factor_names)
    return fig, table.round(2)


def happiness_vs_spi(year, theme, focus=None):
    k = t(theme)
    rc = region_colors(theme)
    spi_year = year if year in years else min(years, key=lambda y: abs(y - year))
    sdf = df[df['SPI year'] == spi_year][['Country', 'Region', SPI_COL]].dropna()
    hdf = happy_df[happy_df['Year'] == year][['Country', 'Happiness Score', 'Rank']].dropna()
    merged = hdf.merge(sdf, on='Country')
    if merged.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout(theme, height=300))
        return fig, merged

    # residual vs the SPI→happiness trend: who is happier than progress predicts?
    a, b, corr = None, None, None
    if len(merged) > 3:
        corr = np.corrcoef(merged[SPI_COL], merged['Happiness Score'])[0, 1]
        a, b = np.polyfit(merged[SPI_COL], merged['Happiness Score'], 1)
        merged['resid'] = merged['Happiness Score'] - (a * merged[SPI_COL] + b)
    else:
        merged['resid'] = 0.0
    THRESH = 0.6
    merged['Mood'] = 'On trend'
    merged.loc[merged['resid'] > THRESH, 'Mood'] = 'Happier than SPI predicts'
    merged.loc[merged['resid'] < -THRESH, 'Mood'] = 'Less happy than SPI predicts'

    fig = go.Figure()
    if focus:
        rest, sel = merged[merged['Region'] != focus], merged[merged['Region'] == focus]
        groups = [(rest, k['dim'], 0.55, 'Other regions', False),
                  (sel, rc[focus], 0.9, region_display(focus), False)]
    else:
        mood_colors = {'On trend': k['dim'],
                       'Happier than SPI predicts': k['div_pos'],
                       'Less happy than SPI predicts': k['div_neg']}
        groups = [(merged[merged['Mood'] == m], mood_colors[m],
                   0.55 if m == 'On trend' else 0.95, m, m != 'On trend')
                  for m in ['On trend', 'Happier than SPI predicts',
                            'Less happy than SPI predicts']]
    for part, color, op, name, label_extremes in groups:
        if part.empty:
            continue
        texts = [''] * len(part)
        if label_extremes:
            tops = set(part.reindex(part['resid'].abs()
                                    .sort_values(ascending=False).index)
                       .head(3)['Country'])
            texts = [c if c in tops else '' for c in part['Country']]
        fig.add_trace(go.Scatter(
            x=part[SPI_COL], y=part['Happiness Score'], mode='markers+text',
            marker=dict(size=9, color=color, opacity=op,
                        line=dict(width=1, color=k['surface'])),
            text=texts, textposition='top center',
            textfont=dict(size=9, color=k['ink2']),
            name=name or '', showlegend=bool(name),
            customdata=np.stack([part['Country'],
                                 part['Region'].fillna('—').map(region_display),
                                 part['resid']], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b><br>SPI: %{x:.2f}'
                          '<br>Happiness: %{y:.2f} (%{customdata[2]:+.2f} vs trend)'
                          '<br>%{customdata[1]}<extra></extra>'))
    if a is not None:
        xl = np.linspace(merged[SPI_COL].min(), merged[SPI_COL].max(), 100)
        fig.add_trace(go.Scatter(x=xl, y=a * xl + b, mode='lines',
                                 line=dict(color=k['ink2'], width=2),
                                 name='Trend', showlegend=False, hoverinfo='skip'))
        fig.add_annotation(text=f"SPI vs happiness:  r = {corr:.2f} · "
                                f"R\u00b2 = {corr**2:.2f}",
                           xref='paper', yref='paper',
                           x=0.97, y=0.04, showarrow=False, xanchor='right',
                           font=dict(size=12, color=k['muted']))
    fig.update_layout(**base_layout(
        theme, height=440,
        xaxis=dict(title=dict(text='Social Progress Index')),
        yaxis=dict(title=dict(text='Happiness Score')),
        legend=dict(orientation='h', y=-0.16, font=dict(size=11))))
    _style_all_axes(fig, theme)

    table = merged[['Rank', 'Country', 'Region', SPI_COL,
                    'Happiness Score', 'resid']].copy()
    table.columns = ['Happiness rank', 'Country', 'Region',
                     f'SPI ({spi_year})', 'Happiness score', 'Vs trend']
    table['Region'] = table['Region'].fillna('—').map(region_display)
    return fig, table.sort_values('Happiness rank').round(2)


def spi_change_pie(theme, baseline_year=None):
    """Pie chart: countries whose SPI improved, declined, or stayed constant
    from baseline_year to LATEST (2025)."""
    k = t(theme)
    base = baseline_year if baseline_year and baseline_year in years else EARLIEST
    earliest_df = df[df['SPI year'] == base][['Country', SPI_COL]].dropna()
    latest_df = df[df['SPI year'] == LATEST][['Country', SPI_COL]].dropna()
    merged = earliest_df.merge(latest_df, on='Country', suffixes=('_start', '_end'))
    merged['change'] = merged[f'{SPI_COL}_end'] - merged[f'{SPI_COL}_start']

    # Threshold: ≥0.5 = improving, <0 = declining, 0–0.49 = no significant change
    increased = int((merged['change'] >= 0.5).sum())
    decreased = int((merged['change'] < 0).sum())
    constant = int(len(merged) - increased - decreased)

    labels = ['Improving', 'No significant change', 'Declining']
    values = [increased, constant, decreased]
    colors = [k['div_pos'], k['muted'], k['div_neg']]

    # Three attempts at labeling this pie via Plotly's automatic per-slice
    # text all failed for the same underlying reason: "No significant
    # change" sits directly between two other slices with very little
    # angular room, so whatever Plotly (or a workaround) does with its
    # label ends up either colliding with "Declining" (outside text),
    # invisible on thin years (inside text hidden by uniformtext), or
    # visually disconnected from its own wedge (a floating annotation).
    # It also looked inconsistent: only one of the three slices was
    # special-cased differently from the other two.
    #
    # Fix: stop relying on Plotly's per-slice text placement entirely. No
    # slice draws its own label. Instead every slice gets an identical,
    # explicit callout to the right of the pie — a colored square plus its
    # label and count, stacked at three FIXED vertical positions. Because
    # the three destination slots never move, they cannot collide with each
    # other regardless of how big or small any slice is in a given year.
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(width=2, color=k['surface'])),
        textinfo='none',
        sort=False,   # keep slice order fixed so pie doesn't rotate between years
        showlegend=False,   # replaced by the callout stack below
        # Pin the pie to an explicit domain and stop Plotly auto-scaling it.
        domain=dict(x=[0.05, 0.55], y=[0.08, 1.0]),
        hovertemplate='<b>%{label}</b><br>Countries: %{value}<br>%{percent}<extra></extra>'))

    callout_y = [0.82, 0.5, 0.18]   # fixed slots, same every year
    annotations = [
        dict(x=0.62, y=cy, xref='paper', yref='paper',
             xanchor='left', yanchor='middle', showarrow=False, align='left',
             font=dict(size=13, color=k['ink']),
             text=(f"<span style='color:{color}'>\u25a0</span> "
                   f"<b>{lbl}</b><br>&nbsp;&nbsp;{v} countries"))
        for cy, lbl, v, color in zip(callout_y, labels, values, colors)
    ]

    fig.update_layout(**base_layout(theme, height=420,
                                    margin=dict(l=20, r=20, t=30, b=30)),
                      showlegend=False, annotations=annotations)

    table = pd.DataFrame({'Category': labels, 'Countries': values,
                          'Percentage': [f'{v/sum(values)*100:.1f}%' for v in values]})
    return fig, table


def spi_change_category_table(category, baseline_year=None):
    """Countries in one SPI-change category, comparing baseline_year to LATEST."""
    base = baseline_year if baseline_year and baseline_year in years else EARLIEST
    earliest_df = df[df['SPI year'] == base][['Country', 'Region', SPI_COL]].dropna()
    latest_df = df[df['SPI year'] == LATEST][['Country', SPI_COL]].dropna()
    merged = earliest_df.merge(latest_df, on='Country', suffixes=('_start', '_end'))
    merged['change'] = merged[f'{SPI_COL}_end'] - merged[f'{SPI_COL}_start']

    if category == 'Improving':
        sub = merged[merged['change'] >= 0.5]
    elif category == 'Declining':
        sub = merged[merged['change'] < 0]
    else:
        sub = merged[(merged['change'] >= 0) & (merged['change'] < 0.5)]

    sub = sub.sort_values('change', ascending=False)
    table = sub[['Country', 'Region', f'{SPI_COL}_start', f'{SPI_COL}_end', 'change']].copy()
    table.columns = ['Country', 'Region', f'SPI ({base})', f'SPI ({LATEST})', 'Change']
    table['Region'] = table['Region'].map(region_display)
    return table.round(2)


def pop_gdp_country_line(country, theme, highlight_year=None):
    """Population vs GDP per capita over time for a country, dual-axis. The
    selected year is highlighted with a vertical marker + enlarged dots so the
    deep-dive's year slider is reflected in this chart."""
    k = t(theme)
    if not country:
        fig = go.Figure()
        fig.update_layout(**base_layout(theme, height=300))
        fig.add_annotation(text='Select a country to view Population vs GDP',
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=k['muted']))
        return fig, pd.DataFrame()

    cdf = df[df['Country'] == country][['SPI year', POP_COL, GDP_COL]].dropna()
    cdf = cdf.sort_values('SPI year')
    if cdf.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout(theme, height=300))
        return fig, pd.DataFrame()

    cdf = cdf.copy()
    cdf['PopM'] = cdf[POP_COL] / 1e6
    cdf['GK'] = cdf[GDP_COL] / 1e3
    hy = highlight_year if highlight_year in set(cdf['SPI year']) else None

    def _sizes(colname):
        return [11 if (hy is not None and y == hy) else 5
                for y in cdf['SPI year']]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Vertical highlight band for the selected year
    if hy is not None:
        fig.add_vline(x=hy, line=dict(color=k['muted'], width=1.5, dash='dot'))

    fig.add_trace(go.Scatter(
        x=cdf['SPI year'], y=cdf['PopM'], mode='lines+markers',
        name='Population (M)', line=dict(color=k['slots'][2], width=2.5),
        marker=dict(size=_sizes('PopM'), color=k['slots'][2],
                    line=dict(width=1, color=k['surface'])),
        hovertemplate='Year: %{x}<br>Population: %{y:.1f}M<extra></extra>'),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cdf['SPI year'], y=cdf['GK'], mode='lines+markers',
        name='GDP per capita ($K)', line=dict(color=k['slots'][3], width=2.5),
        marker=dict(size=_sizes('GK'), color=k['slots'][3],
                    line=dict(width=1, color=k['surface'])),
        hovertemplate='Year: %{x}<br>GDP: $%{y:.2f}K<extra></extra>'),
        secondary_y=True)
    fig.update_yaxes(title_text='Population (M)', secondary_y=False)
    fig.update_yaxes(title_text='GDP per capita ($K)', secondary_y=True)
    fig.update_layout(**base_layout(
        theme, height=350, margin=dict(l=56, r=56, t=16, b=48),
        xaxis=dict(title=dict(text='Year'), dtick=2),
        legend=dict(orientation='h', y=-0.2, font=dict(size=11))))
    _style_all_axes(fig, theme)

    table = cdf[['SPI year', 'PopM', 'GK']].copy()
    table.columns = ['Year', 'Population (M)', 'GDP per capita ($K)']
    return fig, table.round(2)


def gdp_spi_country_line(country, theme):
    """Line chart of GDP per capita vs SPI score over time for a selected country."""
    k = t(theme)
    if not country:
        fig = go.Figure()
        fig.update_layout(**base_layout(theme, height=300))
        fig.add_annotation(text='Select a country to view GDP vs SPI trend',
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=k['muted']))
        return fig, pd.DataFrame()

    cdf = df[df['Country'] == country][['SPI year', SPI_COL, GDP_COL]].dropna()
    cdf = cdf.sort_values('SPI year')
    if cdf.empty:
        fig = go.Figure()
        fig.update_layout(**base_layout(theme, height=300))
        return fig, pd.DataFrame()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=cdf['SPI year'], y=cdf[SPI_COL], mode='lines+markers',
        name='SPI Score', line=dict(color=k['slots'][0], width=2.5),
        marker=dict(size=6, color=k['slots'][0]),
        hovertemplate='Year: %{x}<br>SPI: %{y:.2f}<extra></extra>'),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cdf['SPI year'], y=cdf[GDP_COL] / 1e3, mode='lines+markers',
        name='GDP per capita ($K)', line=dict(color=k['slots'][1], width=2.5),
        marker=dict(size=6, color=k['slots'][1]),
        hovertemplate='Year: %{x}<br>GDP: $%{y:.2f}K<extra></extra>'),
        secondary_y=True)
    fig.update_yaxes(title_text='SPI Score', secondary_y=False)
    fig.update_yaxes(title_text='GDP per capita ($K)', secondary_y=True)
    fig.update_layout(**base_layout(
        theme, height=350, margin=dict(l=56, r=56, t=16, b=48),
        xaxis=dict(title=dict(text='Year'), dtick=2),
        legend=dict(orientation='h', y=-0.2, font=dict(size=11))))
    _style_all_axes(fig, theme)

    table = cdf.copy()
    table[GDP_COL] = (table[GDP_COL] / 1e3).round(2)
    table.columns = ['Year', 'SPI Score', 'GDP per capita ($K)']
    return fig, table.round(2)


def region_spi_trend(theme):
    """Line chart: SPI over time, one line per region (colored by region)."""
    k = t(theme)
    rc = region_colors(theme)
    fig = go.Figure()
    for region in regions:
        rd = df[df['Region'] == region]
        yearly = rd.groupby('SPI year')[SPI_COL].mean().sort_index()
        if yearly.empty:
            continue
        fig.add_trace(go.Scatter(
            x=yearly.index, y=yearly.values, mode='lines+markers',
            name=region_display(region), line=dict(color=rc[region], width=2.5),
            marker=dict(size=5, color=rc[region]),
            hovertemplate=region_display(region) +
                          '<br>Year: %{x}<br>SPI: %{y:.2f}<extra></extra>'))
    fig.update_layout(**base_layout(
        theme, height=420, margin=dict(l=56, r=24, t=16, b=64),
        xaxis=dict(title=dict(text='Year'), dtick=2),
        yaxis=dict(title=dict(text='Mean SPI (0–100)')),
        legend=dict(orientation='h', y=-0.18, font=dict(size=10))))
    _style_all_axes(fig, theme)

    # Table with region means per year
    piv = df.pivot_table(index='SPI year', columns='Region', values=SPI_COL,
                         aggfunc='mean').round(2)
    table = piv.reset_index().rename(columns={'SPI year': 'Year'})
    table = table.rename(columns={r: region_display(r) for r in piv.columns})
    return fig, table


def median_deviation_fig(dff, theme):
    """Diverging bars: each sub-metric's world mean position relative to the
    world median — which metrics sit above vs below the median country."""
    k = t(theme)
    # Collect all scored sub-indicators across all dimensions
    from viz_data import DIMENSIONS
    rows = []
    for dim_name, data in DIMENSIONS.items():
        for comp, indicators in data['raw_indicators'].items():
            for ind, _ in indicators:
                col = ind
                if col not in dff.columns:
                    continue
                series = dff[col].dropna()
                if series.empty:
                    continue
                median = series.median()
                mean = series.mean()
                rows.append({'Metric': ind.split(' (')[0].strip(),
                             'Component': comp, 'Dimension': dim_name,
                             'Median': median, 'Mean': mean,
                             'Deviation': mean - median})
    if not rows:
        return go.Figure(), pd.DataFrame()

    mdf = pd.DataFrame(rows).sort_values('Deviation')
    mdf['Mean'] = mdf['Mean'].round(2)
    mdf['Median'] = mdf['Median'].round(2)
    mdf['Deviation'] = mdf['Deviation'].round(2)
    colors = [k['div_pos'] if d >= 0 else k['div_neg'] for d in mdf['Deviation']]

    fig = go.Figure(go.Bar(
        y=mdf['Metric'], x=mdf['Deviation'], orientation='h',
        marker=dict(color=colors, cornerradius=3,
                    line=dict(width=1, color=k['surface'])),
        text=[f'{d:+.2f}' for d in mdf['Deviation']], textposition='outside',
        textfont=dict(size=9, color=k['ink2']), cliponaxis=False,
        customdata=np.stack([mdf['Mean'], mdf['Median'], mdf['Component']], axis=-1),
        hovertemplate='<b>%{y}</b><br>Mean: %{customdata[0]:.2f} · Median: %{customdata[1]:.2f}'
                      '<br>%{x:+.2f} vs median<br>%{customdata[2]}<extra></extra>'))
    fig.update_layout(**base_layout(
        theme, height=max(500, len(mdf) * 16), margin=dict(l=200, r=60, t=16, b=48),
        xaxis=dict(title=dict(text='World Mean Minus World Median'),
                   zeroline=True, zerolinecolor=k['axis'], zerolinewidth=1.5),
        showlegend=False))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=9))
    _style_all_axes(fig, theme)

    table = mdf[['Metric', 'Component', 'Dimension', 'Mean', 'Median', 'Deviation']].round(2)
    return fig, table
