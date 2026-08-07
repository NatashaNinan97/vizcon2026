"""Ask the Data — a small natural-language query engine over the SPI /
happiness / GDP / population dataset.

No external LLM call: this is a deterministic parser that maps a question to
a safe, constrained set of pandas operations (metric, entity, region, year,
comparison, top/bottom-N, correlation) and returns a plain-language answer
plus, where useful, a figure and a data table. This keeps answers verifiably
grounded in the dataset rather than free-generated text — the recurring
problem this whole project ran into with hand-written Takeaway copy.
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz_data import (df, happy_df, world_df, years, happy_years, regions,
                      LATEST, EARLIEST, SPI_COL, GDP_COL, POP_COL,
                      DIMENSIONS, countries, flag)
from viz_theme import t, base_layout, region_colors, region_display, REGION_ORDER

THEME = 'light'

# --------------------------------------------------------------- metrics ---
# Every metric the engine understands, with aliases a user might type and the
# dataframe + column needed to resolve it. `frame` is 'spi' (df, per SPI
# year) or 'happy' (happy_df, per WHR year).
def _signed(fmt):
    """Wrap a value formatter so it also handles signed deltas correctly
    (e.g. GDP's $/K scaling applied to a *change* in GDP, not just a level)."""
    def _f(v):
        s = fmt(abs(v))
        return f'-{s}' if v < 0 else f'+{s}'
    return _f


METRICS = {
    'spi': {'aliases': ['social progress', 'spi', 'progress index', 'progress score'],
            'label': 'Social Progress Index',
            'frame': 'spi', 'col': SPI_COL, 'fmt': lambda v: f'{v:.1f}', 'good_high': True},
    'gdp': {'aliases': ['gdp per capita', 'gdp/capita', 'gdp', 'income', 'wealth',
                        'wealthy', 'richest', 'poorest'],
            'label': 'GDP per Capita',
            'frame': 'spi', 'col': GDP_COL, 'fmt': lambda v: f'${v/1e3:,.1f}K', 'good_high': True},
    'population': {'aliases': ['population', 'people', 'populous', 'inhabitants'],
                   'label': 'Population',
                   'frame': 'spi', 'col': POP_COL,
                   'fmt': lambda v: f'{v/1e6:,.1f}M', 'good_high': True},
    'happiness': {'aliases': ['happiness', 'happy', 'happiest', 'wellbeing score',
                              'life satisfaction'],
                  'label': 'Happiness Score',
                  'frame': 'happy', 'col': 'Happiness Score',
                  'fmt': lambda v: f'{v:.2f}', 'good_high': True},
}
# Dimension / component scores are also queryable metrics, keyed by their
# exact column name in df (Basic Needs, Foundations of Wellbeing, Opportunity,
# plus every component like 'Safety', 'Health', etc).
DIM_SCORE_COLS = {'Basic Needs': 'Basic Needs',
                  'Foundations of Wellbeing': 'Foundations of Wellbeing',
                  'Societal Opportunity': 'Opportunity'}
COMPONENT_COLS = {c for d in DIMENSIONS.values() for c in d['components']}

for dim_name, col in DIM_SCORE_COLS.items():
    key = dim_name.lower()
    METRICS[key] = {'aliases': [dim_name.lower(), key.replace('societal ', '')],
                     'label': dim_name,
                     'frame': 'spi', 'col': col,
                     'fmt': lambda v: f'{v:.1f}', 'good_high': True}
for comp in COMPONENT_COLS:
    METRICS[comp.lower()] = {'aliases': [comp.lower()], 'label': comp,
                             'frame': 'spi', 'col': comp,
                             'fmt': lambda v: f'{v:.1f}', 'good_high': True}

# Signed-delta formatter for every metric (applies the same $/M/K scaling to
# a *change* in value, not just a level — a plain f"{change:+.1f}" on a raw
# GDP delta printed nonsense like "+4757.8" instead of "+$4.8K").
for _spec in METRICS.values():
    _spec['delta_fmt'] = _signed(_spec['fmt'])

_ALIAS_TO_METRIC = []
for key, spec in METRICS.items():
    for alias in spec['aliases']:
        _ALIAS_TO_METRIC.append((alias, key))
_ALIAS_TO_METRIC.sort(key=lambda p: -len(p[0]))   # longest alias first

_REGION_ALIASES = {r.lower(): r for r in regions}
_REGION_ALIASES.update({
    'mena': 'Middle East & North Africa', 'middle east': 'Middle East & North Africa',
    'north africa': 'Middle East & North Africa',
    'ssa': 'Sub-Saharan Africa', 'africa': 'Sub-Saharan Africa',
    'latin america': 'Latin America & Caribbean', 'caribbean': 'Latin America & Caribbean',
    'asia pacific': 'East Asia & Pacific', 'east asia': 'East Asia & Pacific',
})
_COUNTRY_LOOKUP = {c.lower(): c for c in countries}
_COUNTRY_LOOKUP.update({
    'usa': 'United States', 'us': 'United States', 'america': 'United States',
    'uk': 'United Kingdom', 'britain': 'United Kingdom',
    'uae': 'United Arab Emirates', 'south korea': 'Republic of Korea',
    'north korea': 'Democratic Republic of Korea', 'ivory coast': "Côte d'Ivoire",
    'drc': 'Democratic Republic of Congo', 'congo-brazzaville': 'Republic of Congo',
    'russia': 'Russia', 'gambia': 'The Gambia', 'macedonia': 'Republic of North Macedonia',
})


def _find_metric(q):
    for alias, key in _ALIAS_TO_METRIC:
        if re.search(r'\b' + re.escape(alias) + r'\b', q):
            return key
    return None


def _find_metrics(q):
    """All distinct metrics mentioned, in the order they appear in the
    question (supports "show GDP, happiness, and SPI for India")."""
    hits = []
    for alias, key in _ALIAS_TO_METRIC:
        m = re.search(r'\b' + re.escape(alias) + r'\b', q)
        if m and key not in hits:
            hits.append((m.start(), key))
    hits.sort()
    return [k for _, k in hits]


def _find_region(q):
    for alias, region in sorted(_REGION_ALIASES.items(), key=lambda p: -len(p[0])):
        if re.search(r'\b' + re.escape(alias) + r'\b', q):
            return region
    return None


def _find_countries(q):
    found = []
    for alias, name in sorted(_COUNTRY_LOOKUP.items(), key=lambda p: -len(p[0])):
        if re.search(r'\b' + re.escape(alias) + r'\b', q) and name not in found:
            found.append(name)
    return found


def _find_year(q):
    m = re.search(r'\b(19|20)\d{2}\b', q)
    if m:
        y = int(m.group(0))
        return y
    return None


def _top_n(q, default=5):
    m = re.search(r'\btop\s+(\d+)\b', q)
    if m:
        return min(int(m.group(1)), 25)
    return default


def _metric_frame(key, year=None):
    """Return (dataframe, column, year_used) for a metric at the given year
    (or latest available)."""
    spec = METRICS[key]
    if spec['frame'] == 'happy':
        y = year if year in happy_years else max(happy_years)
        return happy_df[happy_df['Year'] == y], spec['col'], y
    y = year if year in years else LATEST
    return df[df['SPI year'] == y], spec['col'], y


# ---------------------------------------------------------------- intents ---

def _intent_correlation(q, metric_a, metric_b, region):
    fa, cola, ya = _metric_frame(metric_a)
    fb, colb, yb = _metric_frame(metric_b)
    if METRICS[metric_a]['frame'] != METRICS[metric_b]['frame']:
        merged = fa[['Country', cola]].dropna().merge(
            fb[['Country', colb]].dropna(), on='Country')
    else:
        merged = fa[['Country', cola, colb]].dropna() if cola != colb else None
        if merged is None:
            return None
    if region:
        reg_map = df.drop_duplicates('Country').set_index('Country')['Region']
        merged = merged[merged['Country'].map(reg_map) == region]
    if len(merged) < 5:
        return {'answer': "Not enough overlapping data to compute that correlation.",
               'fig': None, 'table': None}
    r = np.corrcoef(merged[cola], merged[colb])[0, 1]
    a_label = METRICS[metric_a]['label']
    b_label = METRICS[metric_b]['label']
    strength = ('very strong' if abs(r) > 0.8 else 'strong' if abs(r) > 0.6 else
                'moderate' if abs(r) > 0.4 else 'weak')
    direction = 'positive' if r > 0 else 'negative'
    scope = f' in {region_display(region)}' if region else ''
    answer = (f"{a_label} and {b_label} have a {strength} {direction} correlation "
             f"(r = {r:.2f}, R\u00b2 = {r**2:.2f}) across {len(merged)} countries{scope}.")
    k = t(THEME)
    fig = go.Figure(go.Scatter(
        x=merged[cola], y=merged[colb], mode='markers',
        marker=dict(size=8, color=k['slots'][0], opacity=0.7,
                    line=dict(width=1, color=k['surface'])),
        customdata=merged['Country'],
        hovertemplate='<b>%{customdata}</b><br>%{x:.2f} / %{y:.2f}<extra></extra>'))
    fig.update_layout(**base_layout(
        THEME, height=380, margin=dict(l=56, r=24, t=16, b=48),
        xaxis=dict(title=dict(text=a_label)), yaxis=dict(title=dict(text=b_label))))
    return {'answer': answer, 'fig': fig,
           'table': merged.sort_values(cola, ascending=False).round(2)}


def _intent_topn(q, metric, region, year, n, ascending):
    frame, col, y = _metric_frame(metric, year)
    d = frame[['Country', col]].dropna()
    if region:
        reg_map = df.drop_duplicates('Country').set_index('Country')['Region']
        d = d[d['Country'].map(reg_map) == region]
        if d.empty:
            return {'answer': f'No data for {region_display(region)} on that metric.',
                   'fig': None, 'table': None}
    d = d.sort_values(col, ascending=ascending).head(n)
    label = METRICS[metric]['label']
    scope = f' in {region_display(region)}' if region else ''
    which = 'lowest' if ascending else 'highest'
    lines = [f"{i+1}. {flag(row['Country'])} {row['Country']} \u2014 "
            f"{METRICS[metric]['fmt'](row[col])}"
            for i, (_, row) in enumerate(d.iterrows())]
    answer = f"{which.title()} {label}{scope} ({y}):\n" + '\n'.join(lines)
    k = t(THEME)
    rc = region_colors(THEME)
    reg_map = df.drop_duplicates('Country').set_index('Country')['Region']
    colors = [rc.get(reg_map.get(c, ''), k['muted']) for c in d['Country']]
    fig = go.Figure(go.Bar(
        x=d[col][::-1], y=[f"{flag(c)} {c}" for c in d['Country']][::-1],
        orientation='h', marker=dict(color=colors[::-1]),
        text=[METRICS[metric]['fmt'](v) for v in d[col][::-1]], textposition='outside'))
    fig.update_layout(**base_layout(
        THEME, height=max(260, 34 * len(d)), margin=dict(l=160, r=60, t=16, b=40),
        xaxis=dict(title=dict(text=label))))
    return {'answer': answer, 'fig': fig, 'table': d.rename(columns={col: label}).round(2)}


def _one_metric_country_lines(metric, countries_found, year):
    """(lines, table_rows, year_used, note) for a single metric across the
    requested countries — always includes rank + region average so a
    single-country answer isn't just a bare number."""
    spec = METRICS[metric]
    frame, col, y = _metric_frame(metric, year)
    reg_map = df.drop_duplicates('Country').set_index('Country')['Region']
    rank_frame = frame[['Country', col]].dropna().copy()
    rank_frame['rank'] = rank_frame[col].rank(
        ascending=not spec['good_high'], method='min').astype(int)
    n_total = len(rank_frame)

    lines, rows = [], []
    note = None
    if spec['frame'] == 'happy' and year and year != y:
        note = (f"Happiness data only goes back to {min(happy_years)}, so {y} "
                "is shown instead of the year you asked for.")
    for c in countries_found:
        rk = rank_frame[rank_frame['Country'] == c]
        if rk.empty:
            lines.append(f"{flag(c)} {c}: no {spec['label']} data for {y}.")
            continue
        val = rk[col].iloc[0]
        rank = int(rk['rank'].iloc[0])
        region_name = reg_map.get(c)
        reg_avg = None
        if region_name is not None:
            reg_avg = rank_frame[rank_frame['Country'].map(
                lambda cc: reg_map.get(cc)) == region_name][col].mean()
        piece = (f"{flag(c)} {c}: {spec['fmt'](val)} \u2014 ranks #{rank} of "
                f"{n_total} worldwide")
        if region_name and reg_avg is not None and not pd.isna(reg_avg):
            vs = 'above' if (val > reg_avg) == spec['good_high'] else 'below'
            piece += (f", {vs} the {region_display(region_name)} average "
                     f"({spec['fmt'](reg_avg)})")
        lines.append(piece + '.')
        rows.append({'Country': c, spec['label']: val, 'Rank': rank,
                     'Region': region_display(region_name) if region_name else '—'})
    return lines, rows, y, note


def _intent_country_lookup(q, countries_found, metrics, year):
    """One or more metrics for one or more countries. Always reports rank and
    a region-average comparison so a single-country answer is a complete
    picture, not a bare number."""
    metrics = metrics or ['spi']
    all_lines, all_notes, all_tables = [], [], []
    for metric in metrics:
        lines, rows, y, note = _one_metric_country_lines(metric, countries_found, year)
        label = METRICS[metric]['label']
        all_lines.append(f"{label} ({y}):")
        all_lines.extend('  ' + ln for ln in lines)
        if note:
            all_notes.append(note)
        if rows:
            all_tables.append(pd.DataFrame(rows))

    answer = '\n'.join(all_lines)
    if all_notes:
        answer += '\n\n' + ' '.join(sorted(set(all_notes)))

    table = None
    if len(all_tables) == 1:
        table = all_tables[0].round(2)
    elif len(all_tables) > 1:
        # Merge one row per country, one column per metric, for a tidy table.
        table = all_tables[0][['Country', 'Region']].drop_duplicates()
        for tbl, metric in zip(all_tables, metrics):
            label = METRICS[metric]['label']
            table = table.merge(tbl[['Country', label]], on='Country', how='outer')
        table = table.round(2)
    return {'answer': answer, 'fig': None, 'table': table}


def _intent_change(q, metric, region, n, ascending, since_year):
    """"Which countries improved/declined the most (since <year>)" — a change
    intent, distinct from a plain top-N-by-level intent."""
    spec = METRICS[metric]
    note = ''
    if spec['frame'] == 'happy':
        y0 = max(min(happy_years), since_year or min(happy_years))
        y1 = max(happy_years)
        if (since_year or EARLIEST) < min(happy_years):
            note = (f" Happiness data only starts in {min(happy_years)}, so "
                    f"showing {y0}\u2192{y1} instead.")
        d0 = happy_df[happy_df['Year'] == y0][['Country', spec['col']]].dropna()
        d1 = happy_df[happy_df['Year'] == y1][['Country', spec['col']]].dropna()
    else:
        y0 = since_year if since_year in years else EARLIEST
        y1 = LATEST
        d0 = df[df['SPI year'] == y0][['Country', spec['col']]].dropna()
        d1 = df[df['SPI year'] == y1][['Country', spec['col']]].dropna()
    m = d0.merge(d1, on='Country', suffixes=('_start', '_end'))
    m['change'] = m[f"{spec['col']}_end"] - m[f"{spec['col']}_start"]
    if region:
        reg_map = df.drop_duplicates('Country').set_index('Country')['Region']
        m = m[m['Country'].map(reg_map) == region]
        if m.empty:
            return {'answer': f'No data for {region_display(region)} on that metric.',
                   'fig': None, 'table': None}
    m = m.sort_values('change', ascending=ascending).head(n)
    label = spec['label']
    which = 'declined' if ascending else 'improved'
    scope = f' in {region_display(region)}' if region else ''
    lines = [f"{i+1}. {flag(row['Country'])} {row['Country']} \u2014 "
            f"{spec['delta_fmt'](row['change'])}"
            for i, (_, row) in enumerate(m.iterrows())]
    answer = (f"Countries that {which} the most on {label}{scope} "
             f"({y0}\u2192{y1}):\n" + '\n'.join(lines))
    if note:
        answer += '\n' + note.strip()
    k = t(THEME)
    fig = go.Figure(go.Bar(
        x=m['change'][::-1], y=[f"{flag(c)} {c}" for c in m['Country']][::-1],
        orientation='h',
        marker=dict(color=[k['div_neg'] if v < 0 else k['div_pos']
                           for v in m['change'][::-1]]),
        text=[spec['delta_fmt'](v) for v in m['change'][::-1]], textposition='outside'))
    fig.update_layout(**base_layout(
        THEME, height=max(260, 34 * len(m)), margin=dict(l=160, r=60, t=16, b=40),
        xaxis=dict(title=dict(text=f'Change in {label} ({y0}\u2192{y1})'))))
    table = m[['Country', f"{spec['col']}_start", f"{spec['col']}_end", 'change']].copy()
    table.columns = ['Country', f'{label} ({y0})', f'{label} ({y1})', 'Change']
    return {'answer': answer, 'fig': fig, 'table': table.round(2)}


def _intent_country_trend(q, countries_found, metrics, since_year):
    """"Show India's GDP over the years" — a real year-by-year series with a
    line chart, not just a start-vs-end delta. Handles several countries and
    several metrics (one chart per metric)."""
    metrics = metrics or ['spi']
    k = t(THEME)
    lines, figs, tables = [], [], []

    for metric in metrics:
        spec = METRICS[metric]
        if spec['frame'] == 'happy':
            src, ycol, yr_list = happy_df, 'Year', happy_years
        else:
            src, ycol, yr_list = df, 'SPI year', years
        if since_year:
            yr_list = [y for y in yr_list if y >= since_year] or yr_list

        sub = src[src['Country'].isin(countries_found) & src[ycol].isin(yr_list)]
        sub = sub[['Country', ycol, spec['col']]].dropna().sort_values(ycol)
        if sub.empty:
            lines.append(f"{spec['label']}: no data for "
                        f"{', '.join(countries_found)}.")
            continue

        note = ''
        if spec['frame'] == 'happy' and since_year and since_year < min(happy_years):
            note = (f" (happiness data only starts in {min(happy_years)})")

        fig = go.Figure()
        for i, c in enumerate(countries_found):
            cd = sub[sub['Country'] == c]
            if cd.empty:
                continue
            first, last = cd.iloc[0], cd.iloc[-1]
            change = last[spec['col']] - first[spec['col']]
            lines.append(
                f"{spec['label']}{note} \u2014 {flag(c)} {c}: "
                f"{spec['fmt'](first[spec['col']])} in {int(first[ycol])} \u2192 "
                f"{spec['fmt'](last[spec['col']])} in {int(last[ycol])} "
                f"({spec['delta_fmt'](change)} over "
                f"{int(last[ycol]) - int(first[ycol])} years).")
            fig.add_trace(go.Scatter(
                x=cd[ycol], y=cd[spec['col']], mode='lines+markers',
                name=f'{flag(c)} {c}',
                line=dict(width=2.5, color=k['slots'][i % len(k['slots'])]),
                marker=dict(size=6),
                hovertemplate=('<b>' + c + '</b><br>%{x}: %{y:.2f}<extra></extra>')))
        fig.update_layout(**base_layout(
            THEME, height=380, margin=dict(l=60, r=24, t=16, b=48),
            xaxis=dict(title=dict(text='Year'), dtick=2),
            yaxis=dict(title=dict(text=spec['label'])),
            legend=dict(orientation='h', y=-0.2, font=dict(size=11))))
        figs.append(fig)
        tbl = sub.rename(columns={ycol: 'Year', spec['col']: spec['label']})
        tables.append(tbl.round(2))

    return {'answer': '\n'.join(lines),
            'fig': figs[0] if len(figs) == 1 else None,
            'figs': figs if len(figs) > 1 else None,
            'table': tables[0] if len(tables) == 1 else None,
            'tables': tables if len(tables) > 1 else None}


def _intent_country_change(q, countries_found, metrics, since_year):
    """"Show GDP/happiness/SPI change from 2011 to 2025 for India" — change
    over time for one or more NAMED countries (as opposed to _intent_change,
    which ranks ALL countries by change)."""
    metrics = metrics or ['spi']
    y0_default = since_year or EARLIEST
    lines = []
    for metric in metrics:
        spec = METRICS[metric]
        if spec['frame'] != 'spi':
            y0 = max(min(happy_years), y0_default)
            y1 = max(happy_years)
            d0 = happy_df[happy_df['Year'] == y0][['Country', spec['col']]].dropna()
            d1 = happy_df[happy_df['Year'] == y1][['Country', spec['col']]].dropna()
            note = (f" (happiness data only starts in {min(happy_years)}, so "
                    f"showing {y0}\u2192{y1})" if y0_default < min(happy_years) else '')
        else:
            y0 = y0_default if y0_default in years else EARLIEST
            y1 = LATEST
            d0 = df[df['SPI year'] == y0][['Country', spec['col']]].dropna()
            d1 = df[df['SPI year'] == y1][['Country', spec['col']]].dropna()
            note = ''
        m = d0.merge(d1, on='Country', suffixes=('_s', '_e'))
        m = m[m['Country'].isin(countries_found)]
        label = spec['label']
        if m.empty:
            lines.append(f"{label}: no data for {', '.join(countries_found)}.")
            continue
        col_s, col_e = f"{spec['col']}_s", f"{spec['col']}_e"
        for _, row in m.iterrows():
            change = row[col_e] - row[col_s]
            val_s, val_e = spec['fmt'](row[col_s]), spec['fmt'](row[col_e])
            lines.append(
                f"{label}{note}: {flag(row['Country'])} {row['Country']} went from "
                f"{val_s} ({y0}) to {val_e} ({y1}) \u2014 {spec['delta_fmt'](change)}.")
    return {'answer': '\n'.join(lines), 'fig': None, 'table': None}


THIN_SAMPLE = 5   # regions with fewer members than this get a caveat


def _intent_region_compare(q, metric, year):
    frame, col, y = _metric_frame(metric, year)
    reg_map = df.drop_duplicates('Country').set_index('Country')['Region']
    d = frame[['Country', col]].dropna().copy()
    d['Region'] = d['Country'].map(reg_map)
    d = d.dropna(subset=['Region'])
    order = [r for r in REGION_ORDER if r in d['Region'].unique()]
    grp = d.groupby('Region')[col]
    means = grp.mean().reindex(order)
    counts = grp.count().reindex(order)
    spec = METRICS[metric]
    label = spec['label']
    top_r, bot_r = means.idxmax(), means.idxmin()

    answer = (f"By average {label} in {y}: {region_display(top_r)} leads "
             f"({spec['fmt'](means[top_r])}, n={int(counts[top_r])} countries), "
             f"{region_display(bot_r)} is lowest "
             f"({spec['fmt'](means[bot_r])}, n={int(counts[bot_r])}) \u2014 a gap of "
             f"{spec['fmt'](means[top_r] - means[bot_r])}.")

    # A mean over 2 countries isn't comparable to a mean over 43. Say so, and
    # name the largest-sample region plus the single best country so the
    # headline can't be read as "this region is happiest/richest overall".
    if counts[top_r] < THIN_SAMPLE:
        big = counts[counts >= THIN_SAMPLE]
        best_country = d.loc[d[col].idxmax()] if spec['good_high'] else d.loc[d[col].idxmin()]
        caveat = (f"\n\nCaution: {region_display(top_r)} only has "
                 f"{int(counts[top_r])} countries in this dataset "
                 f"({', '.join(d[d['Region'] == top_r]['Country'])}), so its "
                 f"average isn't comparable to larger regions.")
        if not big.empty:
            big_leader = means[big.index].idxmax()
            caveat += (f" Among regions with at least {THIN_SAMPLE} countries, "
                      f"{region_display(big_leader)} leads at "
                      f"{spec['fmt'](means[big_leader])} "
                      f"(n={int(counts[big_leader])}).")
        caveat += (f" The single highest-scoring country is "
                  f"{flag(best_country['Country'])} {best_country['Country']} "
                  f"({spec['fmt'](best_country[col])}, "
                  f"{region_display(best_country['Region'])}).")
        answer += caveat

    k = t(THEME)
    rc = region_colors(THEME)
    fig = go.Figure(go.Bar(
        x=[region_display(r) for r in means.index], y=means.values,
        marker=dict(color=[rc.get(r, k['muted']) for r in means.index]),
        customdata=counts.values,
        text=[f"{spec['fmt'](v)}<br><span style='font-size:9px'>n={int(c)}</span>"
              for v, c in zip(means.values, counts.values)],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' + label +
                      ': %{y:.2f}<br>%{customdata} countries<extra></extra>'))
    fig.update_layout(**base_layout(
        THEME, height=380, margin=dict(l=48, r=24, t=16, b=80),
        yaxis=dict(title=dict(text=label))))
    table = pd.DataFrame({'Region': [region_display(r) for r in means.index],
                          label: means.values,
                          'Countries': counts.values})
    return {'answer': answer, 'fig': fig, 'table': table.round(2)}


# ------------------------------------------------------------------ router --

def answer_question(question):
    """Parse a natural-language question and return
    {'answer': str, 'fig': go.Figure|None, 'table': DataFrame|None}."""
    q = question.strip().lower()
    if not q:
        return {'answer': 'Ask a question about the data \u2014 try one of the '
                          'suggestions below.', 'fig': None, 'table': None}

    year = _find_year(q)
    region = _find_region(q)
    countries_found = _find_countries(q)
    metrics_found = _find_metrics(q)
    metric = metrics_found[0] if metrics_found else None
    # Direction of a ranking (ascending = worst/smallest first).
    ascending = bool(re.search(r'\b(lowest|worst|least|poorest|bottom|smallest|'
                               r'min\w*|declin\w*|decreas\w*|drop\w*|fell|fallen|'
                               r'shrank|shrunk|worsen\w*|lost)\b', q))
    is_superlative = bool(re.search(
        r'\b(top|highest|most|best|largest|biggest|richest|happiest|max\w*|min\w*|'
        r'improv\w*|gain\w*|ros\w*|rise|risen|increas\w*|grew|grown|grow\w*)\b', q))
    # Change over time. Uses stems (increas\w*, gain\w*, grow\w*) so present
    # AND past tense both match — matching only "increased" meant "most
    # significant increase in GDP" fell through to a top-N-by-level ranking
    # and answered "highest GDP" instead of "biggest GDP growth".
    is_change = bool(re.search(
        r'\b(improv\w*|declin\w*|decreas\w*|increas\w*|gain\w*|grow\w*|growth|'
        r'grew|grown|ros\w*|rise|risen|fell|fallen|drop\w*|shrank|shrunk|'
        r'worsen\w*|lost|since|chang\w*|trend\w*|'
        r'across the years|over time|over the years|from \d{4})\b', q))

    # correlation: "does X relate to Y", "correlation between X and Y",
    # "X vs Y", "does gdp predict happiness"
    corr_trigger = re.search(
        r'\b(correlat\w*|relate\w*|relationship\w*|predict\w*|driv\w*|link\w*|'
        r'connect\w*|vs\.?|versus)\b', q)
    if corr_trigger:
        metrics_hit = []
        for alias, key in _ALIAS_TO_METRIC:
            if re.search(r'\b' + re.escape(alias) + r'\b', q) and key not in metrics_hit:
                metrics_hit.append(key)
        if len(metrics_hit) >= 2:
            result = _intent_correlation(q, metrics_hit[0], metrics_hit[1], region)
            if result:
                return result

    # "over the years" / "trend" / "year by year" asks for the whole SERIES,
    # so return a line chart across every year rather than a start-vs-end
    # delta. Plain "changed since 2011" phrasing keeps the two-point answer.
    wants_series = bool(re.search(
        r'\b(over the years|across the years|over time|year[- ]by[- ]year|'
        r'trend\w*|history|historical|each year|every year|by year)\b', q))
    if countries_found and wants_series:
        return _intent_country_trend(q, countries_found, metrics_found, year)

    # A named country + change language ("GDP change from 2011 to 2025 for
    # India", "how has Nepal's SPI changed") is a per-country trend, distinct
    # from both a plain snapshot lookup and a "which countries improved most"
    # ranking across ALL countries.
    if countries_found and is_change:
        return _intent_country_change(q, countries_found, metrics_found, year)

    # A plain country lookup ("How does Nepal compare on GDP?", "show GDP,
    # happiness, SPI for India") — no ranking/change language.
    if countries_found and not is_superlative:
        return _intent_country_lookup(q, countries_found, metrics_found, year)

    # Every remaining intent runs once PER requested metric, so asking for
    # three things at once returns three answers instead of silently dropping
    # all but the first.
    metric_list = metrics_found or (['spi'] if is_change else [])
    is_region_q = bool(re.search(r'\bregion', q)) or bool(region)

    def _run(mkey):
        if is_change:
            return _intent_change(q, mkey, region, _top_n(q, default=5),
                                  ascending, year or EARLIEST)
        if is_region_q:
            return _intent_region_compare(q, mkey, year)
        return _intent_topn(q, mkey, region, year, _top_n(q, default=5), ascending)

    if metric_list:
        results = [(_r, mk) for mk in metric_list for _r in [_run(mk)] if _r]
        if len(results) == 1:
            return results[0][0]
        # Multiple metrics: stack the text answers, and keep each figure so the
        # UI can render one chart per metric.
        answer = '\n\n'.join(r['answer'] for r, _ in results)
        figs = [r['fig'] for r, _ in results if r.get('fig') is not None]
        tables = [r['table'] for r, _ in results if r.get('table') is not None]
        return {'answer': answer,
                'fig': figs[0] if len(figs) == 1 else None,
                'figs': figs if len(figs) > 1 else None,
                'table': tables[0] if len(tables) == 1 else None,
                'tables': tables if len(tables) > 1 else None}

    # Nothing recognizable — say so rather than silently guessing SPI top-5.
    return {'answer': "I couldn't quite parse that. Try mentioning a metric "
                      "(GDP, happiness, social progress, population, or a "
                      "component like safety/health), and optionally a "
                      "country, region, or year \u2014 or pick one of the "
                      "suggestions below.",
           'fig': None, 'table': None}


# ---------------------------------------------------------- example prompts ---
SUGGESTED_QUESTIONS = [
    "Which countries improved the most since 2011?",
    "Top 10 happiest countries",
    "Does GDP correlate with happiness?",
    "Which region has the highest GDP?",
    "Bottom 5 countries by social progress",
    "Correlation between social progress and happiness",
    "Which region scores lowest on safety?",
    "Top 5 countries by population",
    "Compare India and China on population",
]
