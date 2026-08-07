import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc

# ============ DATA LOADING ============

df_raw = pd.read_csv('SPI.csv')
world_df = df_raw[df_raw['Country'] == 'World'].copy()
df = df_raw[df_raw['Country'] != 'World']
df = df[df['Region'].notna() & (df['Region'] != '')]

# Numeric conversions
ALL_NUMERIC = [c for c in df.columns if c not in ['Country', 'SPI country code', 'Status', 'Region']]
for col in ALL_NUMERIC:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if col in world_df.columns:
        world_df[col] = pd.to_numeric(world_df[col], errors='coerce')

df['SPI year'] = df['SPI year'].astype(int)
world_df['SPI year'] = pd.to_numeric(world_df['SPI year'], errors='coerce').astype(int)

years = sorted(df['SPI year'].unique())
countries = sorted(df['Country'].unique())
region_order = ['East Asia & Pacific', 'South Asia', 'Central Asia',
                'Middle East & North Africa', 'Sub-Saharan Africa',
                'Europe', 'North America', 'Latin America & Caribbean']
present = set(df['Region'].unique())
regions = [r for r in region_order if r in present]

# Happiness data
happy_df = pd.read_excel('Happiness_report.xlsx', sheet_name=0)
happy_df = happy_df.rename(columns={'Country name': 'Country', 'Life evaluation (3-year average)': 'Happiness Score'})
happy_df['Happiness Score'] = pd.to_numeric(happy_df['Happiness Score'], errors='coerce')
happy_df['Year'] = pd.to_numeric(happy_df['Year'], errors='coerce').astype(int)
country_region = df.drop_duplicates('Country').set_index('Country')['Region'].to_dict()
happy_df['Region'] = happy_df['Country'].map(country_region)
happy_years = sorted([y for y in happy_df['Year'].unique() if y >= 2019])

PLAY_SPEED = 800

# CONSISTENT COLOR PALETTE - one shade per color used everywhere
RED = '#d32f2f'
ORANGE = '#e8833a'
YELLOW = '#f9a825'
GREEN = '#2e7d32'
BLUE = '#1565c0'
LIGHT_BLUE = '#64b5f6'
PURPLE = '#7b1fa2'
GREY = '#9e9e9e'
DARK = '#2c3e50'

region_colors = {
    'Central Asia': RED, 'East Asia & Pacific': LIGHT_BLUE,
    'Europe': GREEN, 'Latin America & Caribbean': '#7b2d8b',
    'Middle East & North Africa': YELLOW, 'North America': GREY,
    'South Asia': ORANGE, 'Sub-Saharan Africa': '#e91e8c',
}

# ============ DIMENSION DEFINITIONS ============
# Each dimension has: component-level scores + raw indicators grouped by component

DIMENSIONS = {
    'Basic Needs': {
        'components': ['Nutrition and Medical Care', 'Water and Sanitation', 'Housing', 'Safety'],
        'raw_indicators': {
            'Nutrition and Medical Care': [
                ('Maternal mortality', True), ('Child stunting', True), ('Child mortality', True),
                ('Diet low in fruits and vegetables', True), ('Undernourishment', True), ('Infectious diseases', True),
            ],
            'Water and Sanitation': [
                ('Basic water service', True), ('Basic sanitation service', True),
                ('Unsafe water sanitation and hygiene', True), ('Satisfaction with water quality', True),
            ],
            'Housing': [
                ('Usage of clean fuels and technology', True), ('Access to electricity', True),
                ('Household air pollution', True), ('Dissatisfaction with housing affordability', True),
            ],
            'Safety': [
                ('Interpersonal violence', True), ('Transportation related injuries', True),
                ('Money stolen', True), ('Feeling safe walking alone', True), ('Intimate partner violence', True),
            ],
        },
    },
    'Foundations of Wellbeing': {
        'components': ['Basic Education', 'Information and Communications', 'Health', 'Environmental Quality'],
        'raw_indicators': {
            'Basic Education': [
                ('Children grow and learn', True), ('Equal access to quality education', True),
                ('Secondary school attainment', True), ('Gender parity in secondary attainment', True),
                ('Primary school enrollment', True),
            ],
            'Information and Communications': [
                ('Internet users', True), ('Mobile telephone users', True),
                ('Online Service Index', True), ('World Press Freedom Index', True),
            ],
            'Health': [
                ('Life expectancy at 65', True), ('Non-communicable diseases', True),
                ('Health Problems', True), ('Equal access to quality healthcare', True),
                ('Access to essential health services', True),
            ],
            'Environmental Quality': [
                ('Outdoor air pollution', True), ('Lead exposure', True),
                ('Waste recovery', True), ('Particulate matter pollution', True),
            ],
        },
    },
    'Societal Opportunity': {
        'components': ['Rights and Voice', 'Freedom and Choice', 'Inclusive Society', 'Advanced Education'],
        'raw_indicators': {
            'Rights and Voice': [
                ('Freedom of peaceful assembly', True),
                ('Equality before the law and individual liberty index', True),
                ('Rights equality', True), ('Perception of corruption', True), ('Political rights', True),
            ],
            'Freedom and Choice': [
                ('CSOs repression', True), ('Freedom over life choices', True),
                ('Vulnerable employment', True), ('Satisfied demand for contraception', True),
                ('Early marriage', True),
            ],
            'Inclusive Society': [
                ('Acceptance of gays and lesbians', True), ('Count on help', True),
                ('Equal access index', True), ('Young people not in education employment or training', True),
                ('Discrimination and violence against minorities', True),
            ],
            'Advanced Education': [
                ('Expected years of tertiary schooling', True), ('Citable documents', True),
                ('Women with advanced education', True), ('Quality weighted universities', True),
                (' Academic freedom', True),
            ],
        },
    },
}


# SPI Framework diagram data (for overview)
SPI_FRAMEWORK = {
    'BASIC NEEDS': {'color': '#17a2b8', 'components': DIMENSIONS['Basic Needs']['raw_indicators']},
    'FOUNDATIONS OF WELLBEING': {'color': '#e8833a', 'components': DIMENSIONS['Foundations of Wellbeing']['raw_indicators']},
    'SOCIETAL OPPORTUNITY': {'color': '#a3c161', 'components': DIMENSIONS['Societal Opportunity']['raw_indicators']},
}

# ============ APP LAYOUT ============

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

# Custom CSS for tab text color and chart headings
app.index_string = '''<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
.nav-tabs .nav-link { color: #000 !important; font-weight: 600; }
.nav-tabs .nav-link.active { color: #000 !important; border-bottom: 3px solid #2c3e50; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''

TITLE_STYLE = {'fontSize': '1.5rem', 'fontWeight': '700', 'color': DARK, 'textAlign': 'center', 'marginBottom': '10px'}

app.layout = dbc.Container([
    html.H2(["HOW THE WORLD", html.Br(), "LIVES, THRIVES, AND CONNECTS"],
            className="text-center mt-3 mb-2", style={'color': '#2c3e50', 'fontWeight': '600'}),
    html.Hr(),
    dbc.Tabs([
        dbc.Tab(label="Overview", tab_id="overview"),
        dbc.Tab(label="Explore", tab_id="explore"),
        dbc.Tab(label="Social Progress", tab_id="social_progress"),
    ], id="tabs", active_tab="overview"),
    html.Div(id="tab-content", className="mt-3"),
], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px 40px 40px 40px'})


@callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(tab):
    if tab == "overview": return build_overview_tab()
    elif tab == "explore": return build_explore_tab()
    elif tab == "social_progress": return build_social_progress_tab()


# ============ SHARED CONTROLS ============

def build_controls(prefix, extra_controls=None):
    """Standard year/country/play controls."""
    controls = [
        dbc.Col([
            html.Label("Year:", className="fw-bold mb-1", style={'fontSize': '0.85rem'}),
            dcc.Dropdown(id=f'{prefix}-year-dropdown',
                         options=[{'label': str(y), 'value': y} for y in years],
                         value=max(years), clearable=False, style={'width': '90px', 'fontSize': '0.85rem'}),
        ], width=1),
        dbc.Col([
            html.Label("Country:", className="fw-bold mb-1", style={'fontSize': '0.85rem'}),
            dcc.Dropdown(id=f'{prefix}-country',
                         options=[{'label': c, 'value': c} for c in countries],
                         value=None, placeholder="Select country", clearable=True,
                         style={'fontSize': '0.85rem'}),
        ], width=3),
        dbc.Col([
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button("▶ Play", id=f'{prefix}-play-btn', color="secondary", size="sm", outline=True),
                dbc.Button("◀", id=f'{prefix}-prev-btn', color="secondary", size="sm", outline=True),
                dbc.Button("▶", id=f'{prefix}-next-btn', color="secondary", size="sm", outline=True),
            ]),
        ], width=2),
    ]
    if extra_controls:
        controls = extra_controls + controls
    return dbc.Card([dbc.CardBody([
        dbc.Row(controls, align="end"),
        html.Div([html.Span([
            html.Span("●", style={'color': region_colors[r], 'fontSize': '1rem', 'marginRight': '3px'}),
            html.Span(r, style={'marginRight': '12px', 'fontSize': '0.75rem', 'color': '#555'}),
        ]) for r in regions], className="mt-2 text-center"),
    ])], className="mb-3", style={'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'})


def make_play_callbacks(prefix):
    """Register play/prev/next callbacks for a prefix."""
    @callback(Output(f'{prefix}-play-interval', 'disabled'), Output(f'{prefix}-play-state', 'data'),
              Output(f'{prefix}-play-btn', 'children'),
              Input(f'{prefix}-play-btn', 'n_clicks'), State(f'{prefix}-play-state', 'data'),
              prevent_initial_call=True)
    def toggle(n, playing):
        s = not playing
        return not s, s, "⏸" if s else "▶ Play"

    @callback(Output(f'{prefix}-year-dropdown', 'value'),
              Input(f'{prefix}-play-interval', 'n_intervals'),
              Input(f'{prefix}-prev-btn', 'n_clicks'), Input(f'{prefix}-next-btn', 'n_clicks'),
              State(f'{prefix}-year-dropdown', 'value'), prevent_initial_call=True)
    def nav(n_int, prev, nxt, cur):
        idx = years.index(cur) if cur in years else 0
        if ctx.triggered_id == f'{prefix}-prev-btn':
            return years[max(0, idx - 1)]
        return years[min(len(years) - 1, idx + 1)]

# ============ OVERVIEW TAB ============

def build_narrative_intro():
    """Story framing — the question we're exploring."""
    return dbc.Card([dbc.CardBody([
        html.H3("Does money buy social progress?",
                style={'color': '#2c3e50', 'fontWeight': '700', 'marginBottom': '6px', 'textAlign': 'center'}),
        html.P("Across 170 countries and 15 years of data, we explore what really determines whether a nation's people live well (basic needs), thrive (education, health, environment), and connect (rights, freedom, inclusion). GDP explains part of the story — but not all of it. Some countries punch far above their weight, while others waste their wealth. The answers may surprise you.",
               style={'color': '#555', 'fontSize': '0.88rem', 'lineHeight': '1.5', 'textAlign': 'center', 'marginBottom': '6px'}),
        html.Div([
            html.Span("Why Live, Thrive, Connect? ", style={'fontWeight': '600', 'color': '#2c3e50'}),
            html.Span("Basic Needs = ", style={'color': '#555'}),
            html.Span("Live", style={'color': '#17a2b8', 'fontWeight': '600'}),
            html.Span(" (can people survive?) • Foundations of Wellbeing = ", style={'color': '#555'}),
            html.Span("Thrive", style={'color': '#e8833a', 'fontWeight': '600'}),
            html.Span(" (can people develop?) • Societal Opportunity = ", style={'color': '#555'}),
            html.Span("Connect", style={'color': '#a3c161', 'fontWeight': '600'}),
            html.Span(" (can people participate freely?)", style={'color': '#555'}),
        ], style={'fontSize': '0.82rem', 'textAlign': 'center'}),
    ])], className="mb-3", style={'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)',
                                  'background': 'linear-gradient(135deg, #f8f9fa 0%, #eef2f7 100%)'})


def build_key_insights():
    """Dynamically computed surprising comparative facts from the data."""
    latest = df[df['SPI year'] == max(years)]
    spi_col = 'Social Progress Index'
    gdp_col = 'GDP per capita (constant 2021 international dollars)'

    # Fact 1: Norway leads
    top_spi = latest.nlargest(1, spi_col).iloc[0]
    bot_spi = latest.nsmallest(1, spi_col).iloc[0]

    # Fact 2: Qatar has 2x Finland's GDP but 17 points less SPI
    qatar = latest[latest['Country'] == 'Qatar']
    finland = latest[latest['Country'] == 'Finland']
    fact2 = None
    if not qatar.empty and not finland.empty:
        q, f = qatar.iloc[0], finland.iloc[0]
        ratio = q[gdp_col] / f[gdp_col]
        gap = f[spi_col] - q[spi_col]
        fact2 = (f"Qatar's GDP is {ratio:.1f}x Finland's", f"Yet Finland's SPI is {gap:.0f} points higher")

    # Fact 3: Costa Rica beats Qatar with 4x less GDP
    cr = latest[latest['Country'] == 'Costa Rica']
    fact3 = None
    if not cr.empty and not qatar.empty:
        c, q = cr.iloc[0], qatar.iloc[0]
        ratio = q[gdp_col] / c[gdp_col]
        fact3 = (f"Costa Rica outscores Qatar on SPI", f"With {ratio:.0f}x less GDP ({c[spi_col]:.1f} vs {q[spi_col]:.1f})")

    # Fact 4: Europe leads in both SPI and happiness
    eu_avg = latest[latest['Region'] == 'Europe'][spi_col].mean()
    ssa_avg = latest[latest['Region'] == 'Sub-Saharan Africa'][spi_col].mean()
    fact4 = ("Europe leads all regions in SPI and happiness", f"Average SPI of {eu_avg:.0f} — nearly double Sub-Saharan Africa's {ssa_avg:.0f}")

    # Fact 5: Mexico ranked #12 happiest but only #75 in Social Progress
    fact5 = ("Mexico is #12 happiest but #75 in Social Progress", "People report high happiness despite lower systemic progress")

    # Build card layout
    insight_style = {'textAlign': 'center', 'padding': '10px 14px', 'flex': '1', 'minWidth': '180px'}

    facts_html = []
    facts_html.append(html.Div([
        html.Span("🏆", style={'fontSize': '1.4rem'}),
        html.Div(html.Strong(f"{top_spi['Country']} leads the world with SPI of {top_spi[spi_col]:.1f}"),
                 style={'fontSize': '0.82rem', 'color': '#333'}),
        html.Div(f"Gap from last ({bot_spi['Country']}): {top_spi[spi_col] - bot_spi[spi_col]:.0f} points",
                 style={'fontSize': '0.72rem', 'color': '#888'}),
    ], style=insight_style))

    if fact2:
        facts_html.append(html.Div([
            html.Span("💸", style={'fontSize': '1.4rem'}),
            html.Div(html.Strong(fact2[0]), style={'fontSize': '0.82rem', 'color': '#333'}),
            html.Div(fact2[1], style={'fontSize': '0.72rem', 'color': '#888'}),
        ], style=insight_style))

    if fact3:
        facts_html.append(html.Div([
            html.Span("⭐", style={'fontSize': '1.4rem'}),
            html.Div(html.Strong(fact3[0]), style={'fontSize': '0.82rem', 'color': '#333'}),
            html.Div(fact3[1], style={'fontSize': '0.72rem', 'color': '#888'}),
        ], style=insight_style))

    facts_html.append(html.Div([
        html.Span("🌍", style={'fontSize': '1.4rem'}),
        html.Div(html.Strong(fact4[0]), style={'fontSize': '0.82rem', 'color': '#333'}),
        html.Div(fact4[1], style={'fontSize': '0.72rem', 'color': '#888'}),
    ], style=insight_style))

    facts_html.append(html.Div([
        html.Span("🤔", style={'fontSize': '1.4rem'}),
        html.Div(html.Strong(fact5[0]), style={'fontSize': '0.82rem', 'color': '#333'}),
        html.Div(fact5[1], style={'fontSize': '0.72rem', 'color': '#888'}),
    ], style=insight_style))

    return dbc.Card([dbc.CardBody([
        html.Div("💡 Key Insights from the Data", style={'fontWeight': '700', 'color': '#2c3e50',
                                                          'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '1rem'}),
        html.Div(facts_html, style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
    ])], className="mb-3", style={'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'})


def build_outlier_chart():
    """GDP vs SPI with overperformers (green) and underperformers (red) highlighted cleanly."""
    latest = df[df['SPI year'] == max(years)].copy()
    spi_col = 'Social Progress Index'
    gdp_col = 'GDP per capita (constant 2021 international dollars)'
    d = latest[['Country', 'Region', spi_col, gdp_col]].dropna()
    d = d[d[gdp_col] > 0].copy()
    xv = d[gdp_col] / 1e3
    logx = np.log(xv.values)
    a, b = np.polyfit(logx, d[spi_col].values, 1)
    d['predicted'] = a * np.log(xv.values) + b
    d['residual'] = d[spi_col] - d['predicted']

    # Categorize
    d['category'] = 'As expected'
    d.loc[d['residual'] > 8, 'category'] = 'Overperformer'
    d.loc[d['residual'] < -8, 'category'] = 'Underperformer'
    cat_colors = {'Overperformer': GREEN, 'Underperformer': RED, 'As expected': '#cccccc'}

    fig = go.Figure()
    # Trend line
    xl = np.linspace(xv.min(), xv.max(), 200)
    fig.add_trace(go.Scatter(x=xl, y=a * np.log(xl) + b, mode='lines',
                             line=dict(color=DARK, width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
    # Points by category
    for cat in ['As expected', 'Overperformer', 'Underperformer']:
        cd = d[d['category'] == cat]
        if cd.empty: continue
        show_text = cat != 'As expected'
        fig.add_trace(go.Scatter(
            x=cd[gdp_col] / 1e3, y=cd[spi_col], mode='markers' + ('+text' if show_text else ''),
            marker=dict(size=9 if show_text else 6, color=cat_colors[cat], opacity=0.9 if show_text else 0.4,
                        line=dict(width=0.3, color='white')),
            text=cd['Country'] if show_text else None,
            textposition='top center', textfont=dict(size=9, color=cat_colors[cat]),
            name=cat, showlegend=True,
            hovertemplate='<b>%{text}</b><br>GDP: $%{x:.1f}K<br>SPI: %{y:.1f}<extra></extra>' if show_text else
                          '<b>' + cd['Country'] + '</b><br>GDP: $' + (cd[gdp_col]/1e3).round(1).astype(str) + 'K<extra></extra>'))

    fig.update_layout(height=500, margin=dict(l=60, r=30, t=30, b=50), plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(title='GDP per Capita (Thousands $)', gridcolor='#eee', tickfont=dict(size=10)),
                      yaxis=dict(title='Social Progress Index', range=[15, 100], gridcolor='#eee', tickfont=dict(size=10)),
                      legend=dict(orientation='h', y=-0.12, font=dict(size=10)),
                      font=dict(family="Segoe UI, sans-serif"))
    return fig


def build_movers_chart():
    """Horizontal bar chart of biggest SPI improvers and decliners - no overlapping labels."""
    spi_col = 'Social Progress Index'
    earliest = df[df['SPI year'] == min(years)][['Country', 'Region', spi_col]].dropna()
    latest = df[df['SPI year'] == max(years)][['Country', 'Region', spi_col]].dropna()
    both = earliest.merge(latest, on=['Country', 'Region'], suffixes=('_start', '_end'))
    both['change'] = both[f'{spi_col}_end'] - both[f'{spi_col}_start']

    top8 = both.nlargest(8, 'change')
    bot5 = both.nsmallest(5, 'change')
    show = pd.concat([bot5.sort_values('change'), top8.sort_values('change')])

    colors = [GREEN if c > 0 else RED for c in show['change']]

    fig = go.Figure(go.Bar(
        y=show['Country'], x=show['change'], orientation='h',
        marker=dict(color=colors),
        text=[f"{c:+.1f}" for c in show['change']], textposition='outside',
        textfont=dict(size=10, color=DARK),
        hovertemplate='<b>%{y}</b><br>Change: %{x:+.1f} points<extra></extra>'))

    fig.update_layout(height=500, margin=dict(l=130, r=60, t=30, b=50), plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(title=f'SPI Change ({min(years)} to {max(years)})', gridcolor='#eee',
                                 zeroline=True, zerolinecolor=DARK, zerolinewidth=1, tickfont=dict(size=10)),
                      yaxis=dict(tickfont=dict(size=11)),
                      font=dict(family="Segoe UI, sans-serif"))
    return fig



def build_framework_diagram():
    """HTML 3-column layout matching the reference SPI framework image."""
    dim_data = {
        'BASIC NEEDS': {'color': '#17a2b8', 'components': DIMENSIONS['Basic Needs']['raw_indicators']},
        'FOUNDATIONS OF WELLBEING': {'color': '#e8833a', 'components': DIMENSIONS['Foundations of Wellbeing']['raw_indicators']},
        'SOCIETAL OPPORTUNITY': {'color': '#a3c161', 'components': DIMENSIONS['Societal Opportunity']['raw_indicators']},
    }
    columns = []
    for dim_name, data in dim_data.items():
        color = data['color']
        comp_blocks = []
        for comp, indicators in data['components'].items():
            comp_blocks.append(html.Div([
                html.Div([
                    html.Span("\u25cf", style={'color': color, 'marginRight': '6px', 'fontSize': '0.9rem'}),
                    html.Span(comp, style={'fontWeight': '700', 'fontSize': '0.9rem', 'color': '#333'}),
                ], style={'marginBottom': '3px'}),
                html.Ul([
                    html.Li(ind.split(' (')[0], style={'fontSize': '0.73rem', 'color': '#666', 'lineHeight': '1.3'})
                    for ind, _ in indicators
                ], style={'marginLeft': '16px', 'marginBottom': '8px', 'listStyleType': 'circle', 'paddingLeft': '8px'}),
            ]))
        columns.append(dbc.Col([
            html.Div(dim_name, style={'fontWeight': '700', 'fontSize': '1rem', 'color': DARK,
                                      'letterSpacing': '1.5px', 'marginBottom': '4px'}),
            html.Hr(style={'borderTop': f'3px solid {color}', 'marginTop': '2px', 'marginBottom': '10px', 'opacity': 1}),
            html.Div(comp_blocks),
        ], width=4))
    return dbc.Card([dbc.CardBody([
        html.Div("The Social Progress Index Framework", style={'fontSize': '1.2rem', 'fontWeight': '700',
                 'color': DARK, 'textAlign': 'center', 'marginBottom': '12px'}),
        dbc.Row(columns),
    ])], className="mb-3", style={'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'})



def build_overview_tab():
    views = [{'label': 'Dimensions Distributions', 'value': 'dims'},
             {'label': 'Population', 'value': 'pop'},
             {'label': 'GDP per Capita', 'value': 'gdp'}]
    return html.Div([
        build_narrative_intro(),
        build_key_insights(),
        build_framework_diagram(),
        # Simple view selector only (no year/country needed here)
        dbc.Card([dbc.CardBody(dbc.Row([
            dbc.Col([
                html.Label("View:", className="fw-bold mb-1", style={'fontSize': '0.85rem'}),
                dcc.Dropdown(id='ov-view', options=views, value='dims', clearable=False, style={'fontSize': '0.85rem'}),
            ], width=3),
        ], align="end"))], className="mb-3", style={'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'}),
        # Hidden stores and buttons for callback compatibility (no visible play controls)
        dcc.Dropdown(id='ov-year-dropdown', value=max(years), style={'display': 'none'}),
        dcc.Store(id='ov-country', data=None),
        dcc.Store(id='ov-play-state', data=False),
        dcc.Interval(id='ov-play-interval', interval=PLAY_SPEED, disabled=True),
        html.Div([
            html.Button(id='ov-play-btn', style={'display': 'none'}),
            html.Button(id='ov-prev-btn', style={'display': 'none'}),
            html.Button(id='ov-next-btn', style={'display': 'none'}),
        ]),
        html.Div(id='ov-content'),
    ])

make_play_callbacks('ov')

@callback(Output('ov-content', 'children'),
          Input('ov-view', 'value'), Input('ov-year-dropdown', 'value'), Input('ov-country', 'data'))
def update_overview(view, year, hc):
    if year is None: year = max(years)
    dff = df[df['SPI year'] == year]
    cs = {'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'}
    if view == 'dims':
        fig = build_strip_chart(dff, ['Basic Needs', 'Foundations of Wellbeing', 'Opportunity'],
                                hc, year, titles=['Basic Needs', 'Foundations of Wellbeing', 'Societal Opportunity'])
        return html.Div([
            html.P("How do countries score across the three pillars of social progress? Each dot represents a country, grouped by region.",
                   style={'color': '#666', 'fontSize': '0.85rem', 'textAlign': 'center', 'marginBottom': '8px'}),
            dbc.Card([dbc.CardBody(dcc.Graph(figure=fig, config={'displayModeBar': False}))], style=cs)])
    elif view == 'pop':
        figs = [build_population_map(dff, year), build_population_chart(dff, hc, year)]
        return html.Div([dbc.Card([dbc.CardBody(dcc.Graph(figure=f, config={'displayModeBar': False}))],
                                  className="mb-2", style=cs) for f in figs])
    elif view == 'gdp':
        map_fig = build_gdp_map(dff, year)
        bar_fig = build_gdp_bar_chart(dff, hc, year)
        return html.Div([
            dbc.Card([dbc.CardBody(dcc.Graph(figure=map_fig, config={'displayModeBar': False}))], className="mb-2", style=cs),
            dbc.Card([dbc.CardBody(dcc.Graph(figure=bar_fig, config={'displayModeBar': False}))], style=cs)])
    return html.Div()

# ============ EXPLORE TAB (merged Live/Thrive/Connect) ============

def build_explore_tab():
    dim_opts = [{'label': 'Basic Needs (Live)', 'value': 'Basic Needs'},
                {'label': 'Foundations of Wellbeing (Thrive)', 'value': 'Foundations of Wellbeing'},
                {'label': 'Societal Opportunity (Connect)', 'value': 'Societal Opportunity'}]
    view_opts = [{'label': 'Region Analysis', 'value': 'region'},
                 {'label': 'Country Deep Dive', 'value': 'country'}]
    return html.Div([
        dbc.Card([dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Dimension:", className="fw-bold mb-1", style={'fontSize': '0.85rem'}),
                    dcc.Dropdown(id='exp-dim', options=dim_opts, value='Basic Needs', clearable=False, style={'fontSize': '0.85rem'}),
                ], width=3),
                dbc.Col([
                    html.Label("View:", className="fw-bold mb-1", style={'fontSize': '0.85rem'}),
                    dcc.Dropdown(id='exp-view', options=view_opts, value='region', clearable=False, style={'fontSize': '0.85rem'}),
                ], width=2),
                # Year + Play (visible for Region view)
                dbc.Col([
                    html.Label("Year:", className="fw-bold mb-1", style={'fontSize': '0.85rem'}),
                    dcc.Dropdown(id='exp-year-dropdown', options=[{'label': str(y), 'value': y} for y in years],
                                 value=max(years), clearable=False, style={'width': '90px', 'fontSize': '0.85rem'}),
                ], width=1, id='exp-year-col'),
                dbc.Col([
                    html.Br(),
                    dbc.ButtonGroup([
                        dbc.Button("\u25b6 Play", id='exp-play-btn', color="secondary", size="sm", outline=True),
                        dbc.Button("\u25c0", id='exp-prev-btn', color="secondary", size="sm", outline=True),
                        dbc.Button("\u25b6", id='exp-next-btn', color="secondary", size="sm", outline=True),
                    ]),
                ], width=2, id='exp-play-col'),
                # Country (visible for Country Deep Dive)
                dbc.Col([
                    html.Label("Country:", className="fw-bold mb-1", style={'fontSize': '0.85rem'}),
                    dcc.Dropdown(id='exp-country', options=[{'label': c, 'value': c} for c in countries],
                                 value=None, placeholder="Select country", clearable=True, style={'fontSize': '0.85rem'}),
                ], width=3, id='exp-country-col'),
            ], align="end"),
            html.Div([html.Span([
                html.Span("\u25cf", style={'color': region_colors[r], 'fontSize': '1rem', 'marginRight': '3px'}),
                html.Span(r, style={'marginRight': '12px', 'fontSize': '0.75rem', 'color': '#555'}),
            ]) for r in regions], className="mt-2 text-center"),
        ])], className="mb-3", style={'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'}),
        dcc.Interval(id='exp-play-interval', interval=PLAY_SPEED, disabled=True),
        dcc.Store(id='exp-play-state', data=False),
        html.Div(id='exp-content'),
    ])


@callback(Output('exp-year-col', 'style'), Output('exp-play-col', 'style'),
          Output('exp-country-col', 'style'), Input('exp-view', 'value'))
def toggle_explore_controls(view):
    if view == 'country':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}

make_play_callbacks('exp')


@callback(Output('exp-content', 'children'),
          Input('exp-dim', 'value'), Input('exp-view', 'value'),
          Input('exp-year-dropdown', 'value'), Input('exp-country', 'value'))
def update_explore(dim, view, year, country):
    dff = df[df['SPI year'] == year]
    cs = {'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'}
    dim_data = DIMENSIONS[dim]

    if view == 'region':
        if dim == 'Basic Needs':
            fig = build_region_grouped_bar(dff, dim_data['components'], country, year)
        elif dim == 'Foundations of Wellbeing':
            fig = build_region_radar(dff, dim_data['components'], year)
        else:  # Societal Opportunity
            fig = build_region_dumbbell(dff, dim_data['components'], country, year)
        sunburst_fig = build_dimension_sunburst(dim, dim_data)
        return html.Div([
            # Bar chart first
            dbc.Card([dbc.CardBody(dcc.Graph(figure=fig, config={'displayModeBar': False}))],
                     className="mb-2", style=cs),
            # Donut + sub-indicator drill-down below
            dbc.Card([dbc.CardBody([
                html.Div("Drill Down: Click a component in the donut chart to see how regions compare on each sub-indicator",
                         style={'fontSize': '0.85rem', 'color': '#888', 'textAlign': 'center', 'marginBottom': '8px'}),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=sunburst_fig, id='exp-sunburst', config={'displayModeBar': False}), width=4),
                    dbc.Col(html.Div(id='exp-subindicator-chart'), width=8),
                ]),
            ])], style=cs),
        ])

    # Country deep dive
    children = [dbc.Card([dbc.CardBody([
        build_country_profile(dff, country, year, dim, dim_data),
    ])], className="mb-2", style=cs)]

    if country:
        children.append(dbc.Card([dbc.CardBody([
            html.Div("Indicator Deep Dive", style={'fontSize': '1.2rem', 'fontWeight': '600', 'color': DARK, 'textAlign': 'center', 'marginBottom': '4px'}),
            html.P("Each chart shows a different component. Values are world percentiles (0 = worst outcome globally, 100 = best). "
                   "Green = top tier, Yellow = mid, Red = bottom tier. Arrows indicate if a higher raw value is better (\u2191) or worse (\u2193).",
                   style={'fontSize': '0.82rem', 'color': '#666', 'textAlign': 'center', 'marginBottom': '8px'}),
            dcc.Graph(figure=build_indicator_charts(dff, country, dim_data), config={'displayModeBar': False}),
        ])], style=cs))

    return html.Div(children)


@callback(Output('exp-subindicator-chart', 'children'),
          Input('exp-sunburst', 'clickData'),
          Input('exp-dim', 'value'), Input('exp-year-dropdown', 'value'))
def update_subindicator(click_data, dim, year):
    """When a component is clicked in the sunburst, show its sub-indicators as a bar chart."""
    dim_data = DIMENSIONS.get(dim)
    if not dim_data:
        return html.Div()

    selected_comp = None
    if click_data:
        try:
            label = click_data['points'][0]['label']
            if label in dim_data['raw_indicators']:
                selected_comp = label
        except (KeyError, IndexError, TypeError):
            pass

    if not selected_comp:
        return html.Div([
            html.P("\u2190 Click a component to see its sub-indicator breakdown by region",
                   style={'color': '#888', 'textAlign': 'center', 'padding': '60px 20px', 'fontSize': '0.9rem'})
        ])

    dff = df[df['SPI year'] == year]
    indicators = dim_data['raw_indicators'][selected_comp]
    ind_cols = [col for col, _ in indicators if col in dff.columns]

    means = dff.groupby('Region')[ind_cols].mean().reindex(regions)

    # Drop indicators where all regions have (essentially) the same value - not informative
    keep_cols = [c for c in ind_cols if means[c].dropna().nunique() > 1 and means[c].std() > 0.01]

    if not keep_cols:
        return html.P("No regional variation to display for this component.",
                      style={'color': '#888', 'textAlign': 'center', 'padding': '40px'})

    # Sort by overall mean descending
    overall_means = means[keep_cols].mean().sort_values(ascending=False)
    keep_cols = list(overall_means.index)
    ind_labels = [c.split(' (')[0] for c in keep_cols]

    # Each indicator gets its own subplot with its own y-axis (scales differ: DALYs vs % vs 0-4 scores)
    n = len(keep_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=ind_labels,
                        vertical_spacing=0.2, horizontal_spacing=0.08)

    for i, col in enumerate(keep_cols):
        r = i // ncols + 1
        c = i % ncols + 1
        show_legend = (i == 0)
        # Sort regions descending by value for this specific indicator
        col_sorted_regions = means[col].dropna().sort_values(ascending=False).index
        for region in col_sorted_regions:
            if region not in regions: continue
            val = means.loc[region, col]
            fig.add_trace(go.Bar(
                x=[region], y=[val], name=region, marker=dict(color=region_colors[region], opacity=1.0),
                showlegend=show_legend, text=[f"{val:.1f}"], textposition='outside', textfont=dict(size=8),
                hovertemplate=region + ': %{y:.1f}<extra></extra>'), row=r, col=c)
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(tickfont=dict(size=8), gridcolor='#eee', row=r, col=c)

    fig.update_layout(height=280 * nrows, margin=dict(l=40, r=20, t=50, b=40),
                      plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(orientation='h', y=-0.06, font=dict(size=8)),
                      title=dict(text=f'{selected_comp} - Sub-Indicators by Region ({year})',
                                 font=dict(size=14, color=DARK), x=0.5),
                      font=dict(family="Segoe UI, sans-serif"))
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=11, color=DARK)
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


# ============ SOCIAL PROGRESS TAB ============

def build_social_progress_tab():
    cs = {'borderRadius': '10px', 'boxShadow': '0 1px 6px rgba(0,0,0,0.05)'}
    return html.Div([
        dcc.Interval(id='sp-play-interval', interval=PLAY_SPEED, disabled=True),
        dcc.Store(id='sp-play-state', data=False),
        # SPI Bubble - year/play controls live inside this card now
        dbc.Card([dbc.CardBody([
            html.P("A snapshot of every country's social progress score. Larger bubbles represent more populous nations.", style={'color': '#666', 'fontSize': '0.82rem', 'textAlign': 'center', 'marginBottom': '4px'}),
            html.Div("Social Progress Index — Global View", style=TITLE_STYLE),
            dbc.Row([
                dbc.Col([html.Label("Year:", style={'fontSize': '0.85rem', 'fontWeight': '600'}),
                         dcc.Dropdown(id='sp-year-dropdown', options=[{'label': str(y), 'value': y} for y in years],
                                      value=max(years), clearable=False, style={'width': '90px', 'fontSize': '0.85rem'})], width=2),
                dbc.Col([html.Br(), dbc.ButtonGroup([
                    dbc.Button("\u25b6 Play", id='sp-play-btn', color="secondary", size="sm", outline=True),
                    dbc.Button("\u25c0", id='sp-prev-btn', color="secondary", size="sm", outline=True),
                    dbc.Button("\u25b6", id='sp-next-btn', color="secondary", size="sm", outline=True)])], width=2),
            ], align="end", justify="center", className="mb-2"),
            dcc.Graph(id='sp-chart', config={'displayModeBar': False}),
        ])], className="mb-3", style=cs),
        # GDP vs SPI with outlier toggle
        dbc.Card([dbc.CardBody([
            html.Div(f"GDP vs Social Progress ({max(years)})", style=TITLE_STYLE),
            html.P("Does higher GDP guarantee higher social progress? The curve shows the relationship plateaus at high income levels.", style={'color': '#666', 'fontSize': '0.82rem', 'textAlign': 'center', 'marginBottom': '4px'}),
            dcc.Graph(id='sp-gdp-chart', config={'displayModeBar': False}),
            dbc.Button("\U0001f50d  Show Overperformers vs Underperformers", id='sp-outlier-toggle',
                       color="primary", size="md", className="mt-3 mb-2 d-block mx-auto",
                       style={'fontWeight': '600', 'padding': '8px 20px'}),
            html.Div(id='sp-outlier-container', style={'marginTop': '12px'}),
        ])], className="mb-3", style=cs),
        # Biggest Movers - full width
        dbc.Card([dbc.CardBody([
            html.P("Which countries have gained or lost the most social progress over the full time period?", style={'color': '#666', 'fontSize': '0.82rem', 'textAlign': 'center', 'marginBottom': '4px'}),
            html.Div(f"Biggest Movers: {min(years)} → {max(years)}", style=TITLE_STYLE),
            dcc.Graph(id='sp-movers-chart', config={'displayModeBar': False}),
        ])], className="mb-3", style=cs),
        # Happiness factor bar - full width
        dbc.Card([dbc.CardBody([
            html.Div(f"What Drives Happiness? ({max(years)})", style=TITLE_STYLE),
            html.P("The stacked bars show how much each factor (GDP, social support, health, freedom, generosity, corruption) contributes to a country's happiness score.", style={'color': '#666', 'fontSize': '0.82rem', 'textAlign': 'center', 'marginBottom': '4px'}),
            dbc.Row([dbc.Col([
                dcc.Dropdown(id='happy-year', options=[{'label': str(y), 'value': y} for y in happy_years],
                             value=max(happy_years), clearable=False, style={'width': '90px', 'fontSize': '0.85rem'}),
            ], width=2)], className="mb-2"),
            dcc.Graph(id='happy-bar', config={'displayModeBar': False}),
        ])], className="mb-3", style=cs),
        # Happiness vs SPI - below happiness
        dbc.Card([dbc.CardBody([
            html.Div(f"Happiness vs Social Progress ({max(years)})", style=TITLE_STYLE),
            html.P("Does being a socially progressive country make its people happier? Points colored by region; the dashed line shows the trend.", style={'fontSize': '0.82rem', 'color': '#888', 'textAlign': 'center'}),
            dcc.Graph(id='sp-happy-spi', config={'displayModeBar': False}),
        ])], style=cs),
    ])

make_play_callbacks('sp')


@callback(Output('sp-chart', 'figure'), Input('sp-year-dropdown', 'value'))
def update_sp(year):
    return build_spi_bubble(df[df['SPI year'] == year], year)


@callback(Output('sp-gdp-chart', 'figure'), Input('sp-year-dropdown', 'value'))
def update_sp_gdp(year):
    return build_gdp_scatter(df[df['SPI year'] == year], year)


@callback(Output('sp-outlier-container', 'children'), Output('sp-outlier-toggle', 'children'),
          Input('sp-outlier-toggle', 'n_clicks'), prevent_initial_call=True)
def toggle_outlier(n):
    if n and n % 2 == 1:
        return dcc.Graph(figure=build_outlier_chart(), config={'displayModeBar': False}), "\u2716  Hide Overperformers vs Underperformers"
    return html.Div(), "\U0001f50d  Show Overperformers vs Underperformers"


@callback(Output('sp-movers-chart', 'figure'), Input('sp-year-dropdown', 'value'))
def update_sp_movers(year):
    return build_movers_chart()


@callback(Output('sp-happy-spi', 'figure'), Input('sp-year-dropdown', 'value'))
def update_sp_happy_spi(year):
    return build_happiness_vs_spi(year)


@callback(Output('happy-bar', 'figure'), Input('happy-year', 'value'))
def update_happy_bar(year):
    return build_happiness_factors(happy_df[happy_df['Year'] == year], year)


# ============ CHART BUILDERS ============

def build_strip_chart(dff, metrics, highlight_country, year, titles=None):
    np.random.seed(42)
    n = len(metrics)
    display_titles = titles or metrics
    fig = make_subplots(rows=1, cols=n, subplot_titles=display_titles, horizontal_spacing=0.04)
    for ci, metric in enumerate(metrics, 1):
        for region in regions:
            rd = dff[dff['Region'] == region]
            if rd.empty: continue
            scores = rd[metric].values
            cnames = rd['Country'].values
            opacities = [1.0 if c == highlight_country else (0.2 if highlight_country else 0.7) for c in cnames]
            ri = regions.index(region)
            jit = np.random.uniform(-0.2, 0.2, len(scores))
            xp = [ri + j for j in jit]
            fig.add_trace(go.Scatter(x=xp, y=scores, mode='markers',
                marker=dict(size=12, color=region_colors[region], opacity=opacities, line=dict(width=0.5, color='white')),
                text=[f"{c}<br>{metric}: {s:.1f}<br>Year: {year}" for c, s in zip(cnames, scores)],
                hoverinfo='text', showlegend=False), row=1, col=ci)
            if highlight_country and highlight_country in cnames:
                hm = cnames == highlight_country
                fig.add_trace(go.Scatter(x=[x for x, m in zip(xp, hm) if m], y=scores[hm], mode='markers',
                    marker=dict(size=16, color=region_colors[region], opacity=1, line=dict(width=2.5, color='#2c3e50')),
                    hoverinfo='skip', showlegend=False), row=1, col=ci)
        fig.update_xaxes(tickvals=list(range(len(regions))), ticktext=[r.replace(' & ', '\n& ') for r in regions],
                         tickangle=0, tickfont=dict(size=8), row=1, col=ci, showgrid=False)
        fig.update_yaxes(range=[0, 105], row=1, col=ci, gridcolor='#eee')
        if ci == 1: fig.update_yaxes(title_text="Score", row=1, col=ci)
    fig.update_layout(height=700, margin=dict(l=50, r=20, t=50, b=120), plot_bgcolor='white', paper_bgcolor='white',
                      font=dict(family="Segoe UI, sans-serif"))
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=12, color=DARK)
    return fig


def build_region_grouped_bar(dff, components, highlight_country, year):
    """Treemap of dimension sub-sections + grouped bar by region."""
    return build_region_analysis(dff, components, highlight_country, year, 'Basic Needs')


def build_region_radar(dff, components, year):
    """For Foundations of Wellbeing - same pattern."""
    return build_region_analysis(dff, components, None, year, 'Foundations of Wellbeing')


def build_region_dumbbell(dff, components, highlight_country, year):
    """For Societal Opportunity - same pattern."""
    return build_region_analysis(dff, components, highlight_country, year, 'Societal Opportunity')


def build_region_analysis(dff, components, highlight_country, year, dim_name):
    """Grouped bar chart: one color per region, sorted descending by score within each component."""
    means = dff.groupby('Region')[components].mean().reindex(regions)

    fig = go.Figure()
    # Sort regions descending by their average across all components (overall ranking)
    overall_avg = means.mean(axis=1).sort_values(ascending=False)
    sorted_regions = [r for r in overall_avg.index if r in means.index]

    for region in sorted_regions:
        vals = means.loc[region].tolist()
        fig.add_trace(go.Bar(
            x=components, y=vals, name=region,
            marker=dict(color=region_colors[region], opacity=1.0),
            hovertemplate='%{x}<br>' + region + ': %{y:.1f}<extra></extra>'))

    if highlight_country and highlight_country in dff['Country'].values:
        crow = dff[dff['Country'] == highlight_country].iloc[0]
        for comp in components:
            fig.add_trace(go.Scatter(x=[comp], y=[crow[comp]], mode='markers',
                marker=dict(size=14, color=DARK, symbol='diamond', line=dict(width=1.5, color='white')),
                showlegend=False, hovertemplate=f'{highlight_country}<br>{comp}: {crow[comp]:.1f}<extra></extra>'))

    fig.update_layout(barmode='group', height=500, margin=dict(l=50, r=30, t=60, b=60),
                      plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(tickangle=0, tickfont=dict(size=10)),
                      yaxis=dict(title='Score (0-100)', range=[0, 100], gridcolor='#eee', tickfont=dict(size=10)),
                      legend=dict(orientation='h', y=-0.15, font=dict(size=9)),
                      title=dict(text=f'{dim_name} - Regional Comparison ({year})', font=dict(size=16, color=DARK), x=0.5),
                      font=dict(family="Segoe UI, sans-serif"))
    return fig


def build_dimension_treemap(dim, dim_data):
    """For backward compat - calls sunburst."""
    return build_dimension_sunburst(dim, dim_data)


def build_dimension_sunburst(dim, dim_data):
    """Donut chart showing the dimension's components. Click to drill into sub-indicators."""
    comp_colors_map = {'Basic Needs': ['#17a2b8', '#20c9de', '#5dd9e8', '#8ae4ef'],
                       'Foundations of Wellbeing': ['#e8833a', '#f0a060', '#f5bc88', '#f9d4b0'],
                       'Societal Opportunity': ['#a3c161', '#b8d080', '#cde0a0', '#e0ecc0']}
    colors = comp_colors_map.get(dim, ['#888'] * 4)

    comps = list(dim_data['raw_indicators'].keys())
    counts = [len(dim_data['raw_indicators'][c]) for c in comps]
    # Only include components that have indicators
    valid = [(c, n) for c, n in zip(comps, counts) if n > 0]
    if not valid:
        fig = go.Figure()
        fig.update_layout(height=350)
        return fig
    comps, counts = zip(*valid)
    comps, counts = list(comps), list(counts)

    fig = go.Figure(go.Pie(
        labels=comps, values=counts, hole=0.45,
        marker=dict(colors=colors[:len(comps)], line=dict(width=2, color='white')),
        textinfo='label', textfont=dict(size=10),
        hovertemplate='<b>%{label}</b><br>%{value} indicators<extra></extra>'))

    fig.update_layout(height=350, margin=dict(l=10, r=10, t=45, b=10),
                      title=dict(text=f'{dim}', font=dict(size=14, color=DARK), x=0.5),
                      showlegend=False,
                      annotations=[dict(text=dim.split()[0], x=0.5, y=0.5, font_size=12, showarrow=False, font_color=DARK)])
    return fig


def build_population_map(dff, year):
    pop_col = 'Population size (no. of people)'
    m = dff[['Country', pop_col]].dropna(subset=[pop_col]).copy()
    m['log_pop'] = np.log10((m[pop_col] / 1e6).clip(lower=0.01))
    fig = px.choropleth(m, locations='Country', locationmode='country names', color='log_pop',
                        color_continuous_scale=[[0, '#1a9850'], [0.5, '#fee08b'], [1, '#d73027']],
                        hover_name='Country', custom_data=[m[pop_col] / 1e6])
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Pop: %{customdata[0]:.1f}M<extra></extra>')
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10),
                      coloraxis_colorbar=dict(title='Pop', tickvals=[-1, 0, 1, 2, 3], ticktext=['0.1M', '1M', '10M', '100M', '1B']),
                      geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
                      title=dict(text=f'Population ({year})', font=dict(size=20, color=DARK), x=0.5))
    return fig


def build_population_chart(dff, hc, year):
    pop_col = 'Population size (no. of people)'
    p = dff[['Country', 'Region', pop_col]].dropna().sort_values(pop_col, ascending=False).head(25)
    p['Rank'] = range(1, len(p) + 1)
    p['PM'] = p[pop_col] / 1e6
    p = p.sort_values(pop_col, ascending=True)  # so biggest at top in horizontal bar

    colors = [region_colors.get(r, '#888') for r in p['Region']]
    ops = [1.0 if hc and c == hc else (0.2 if hc else 0.8) for c in p['Country']]

    fig = go.Figure(go.Bar(
        y=p['Country'], x=p['PM'], orientation='h',
        marker=dict(color=colors, opacity=ops),
        text=[f"#{r}" for r in p['Rank']], textposition='outside',
        textfont=dict(size=8, color='#555'),
        hovertemplate='%{y}<br>Pop: %{x:.1f}M<extra></extra>'))

    # Legend traces for regions
    for region in regions:
        if region in p['Region'].values:
            fig.add_trace(go.Bar(y=[None], x=[None], name=region,
                                 marker=dict(color=region_colors[region]), showlegend=True))

    fig.update_layout(height=550, margin=dict(l=110, r=50, t=50, b=30),
                      plot_bgcolor='white', paper_bgcolor='white',
                      xaxis_title='Population (Millions)',
                      yaxis=dict(tickfont=dict(size=9), showgrid=False, categoryorder='array', categoryarray=list(p['Country'])),
                      xaxis=dict(gridcolor='#eee'),
                      legend=dict(orientation='h', y=-0.1, font=dict(size=9)),
                      title=dict(text=f'Population ({year})', font=dict(size=20, color=DARK), x=0.5))
    return fig



def build_gdp_map(dff, year):
    """Choropleth world map colored by GDP per capita."""
    gdp_col = 'GDP per capita (constant 2021 international dollars)'
    m = dff[['Country', gdp_col]].dropna().copy()
    m = m[m[gdp_col] > 0]
    m['GK'] = m[gdp_col] / 1e3
    m['log_gdp'] = np.log10(m['GK'].clip(lower=0.1))

    fig = px.choropleth(m, locations='Country', locationmode='country names', color='log_gdp',
                        color_continuous_scale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
                        hover_name='Country', custom_data=['GK'])
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>GDP: $%{customdata[0]:.1f}K<extra></extra>')
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10),
                      coloraxis_colorbar=dict(title='GDP/cap', tickvals=[0, 0.5, 1, 1.5, 2],
                                              ticktext=['$1K', '$3K', '$10K', '$30K', '$100K']),
                      geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
                      title=dict(text=f'GDP per Capita - World Map ({year})', font=dict(size=16, color=DARK), x=0.5))
    return fig


def build_gdp_bar_chart(dff, hc, year):
    """Top 25 countries by GDP per capita - sorted by rank."""
    gdp_col = 'GDP per capita (constant 2021 international dollars)'
    g = dff[['Country', 'Region', gdp_col]].dropna().sort_values(gdp_col, ascending=False).head(25)
    g['Rank'] = range(1, len(g) + 1)
    g['GK'] = g[gdp_col] / 1e3
    g = g.sort_values(gdp_col, ascending=True)  # so biggest at top

    colors = [region_colors.get(r, '#888') for r in g['Region']]
    ops = [1.0 if hc and c == hc else (0.2 if hc else 0.8) for c in g['Country']]

    fig = go.Figure(go.Bar(
        y=g['Country'], x=g['GK'], orientation='h',
        marker=dict(color=colors, opacity=ops),
        text=[f"#{r}" for r in g['Rank']], textposition='outside',
        textfont=dict(size=8, color='#555'),
        hovertemplate='%{y}<br>GDP: $%{x:.1f}K<extra></extra>'))

    for region in regions:
        if region in g['Region'].values:
            fig.add_trace(go.Bar(y=[None], x=[None], name=region,
                                 marker=dict(color=region_colors[region]), showlegend=True))

    fig.update_layout(height=550, margin=dict(l=110, r=50, t=50, b=30),
                      plot_bgcolor='white', paper_bgcolor='white',
                      xaxis_title='GDP per Capita (Thousands $)',
                      yaxis=dict(tickfont=dict(size=9), showgrid=False, categoryorder='array', categoryarray=list(g['Country'])),
                      xaxis=dict(gridcolor='#eee'),
                      legend=dict(orientation='h', y=-0.1, font=dict(size=9)),
                      title=dict(text=f'Top 25 Countries by GDP per Capita ({year})', font=dict(size=16, color=DARK), x=0.5))
    return fig



def build_gdp_scatter(dff, year):
    gdp_col = 'GDP per capita (constant 2021 international dollars)'
    spi_col = 'Social Progress Index'
    d = dff[['Country', 'Region', spi_col, gdp_col]].dropna()
    d = d[d[gdp_col] > 0]
    xv = d[gdp_col] / 1e3
    fig = go.Figure(go.Scatter(x=xv, y=d[spi_col], mode='markers', text=d['Country'], showlegend=False,
                               marker=dict(size=9, color=d[spi_col], colorscale=[[0, '#c0392b'], [0.5, '#f4d03f'], [1, '#0b6623']],
                                           cmin=20, cmax=95, opacity=0.8, line=dict(width=0.3, color='white'),
                                           colorbar=dict(title='SPI', tickfont=dict(size=9))),
                               hovertemplate='<b>%{text}</b><br>GDP: $%{x:.1f}K<br>SPI: %{y:.1f}<extra></extra>'))
    if len(d) > 3:
        logx = np.log(xv.values)
        a, b = np.polyfit(logx, d[spi_col].values, 1)
        xl = np.linspace(xv.min(), xv.max(), 200)
        fig.add_trace(go.Scatter(x=xl, y=a * np.log(xl) + b, mode='lines', line=dict(color='#2c3e50', width=2.5),
                                 showlegend=False, hoverinfo='skip'))
        corr = np.corrcoef(xv, d[spi_col])[0, 1]
        fig.add_annotation(text=f"r = {corr:.2f}", xref='paper', yref='paper', x=0.95, y=0.05,
                           showarrow=False, font=dict(size=12, color='#c0392b'), xanchor='right')
    fig.update_layout(height=450, margin=dict(l=50, r=30, t=30, b=50), plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(title='GDP per Capita (Thousands $)', gridcolor='#eee'),
                      yaxis=dict(title='Social Progress Index', range=[10, 100], gridcolor='#eee'),
                      font=dict(family="Segoe UI, sans-serif"))
    return fig


def build_gdp_treemap(dff, year):
    gdp_col = 'GDP per capita (constant 2021 international dollars)'
    t = dff[['Country', 'Region', gdp_col]].dropna().copy()
    t = t[t[gdp_col] > 0]
    t['GK'] = t[gdp_col] / 1e3
    fig = px.treemap(t, path=['Region', 'Country'], values=gdp_col, color='Region', color_discrete_map=region_colors,
                     hover_data={'GK': ':.1f'}, title=f'GDP per Capita Treemap ({year})')
    fig.update_traces(textinfo='label+text', texttemplate='%{label}<br>$%{customdata[0]:.1f}K',
                      hovertemplate='<b>%{label}</b><br>$%{customdata[0]:.1f}K<extra></extra>')
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=45, b=10), title=dict(font=dict(size=13, color='#2c3e50')))
    return fig


def build_spi_bubble(dff, year):
    spi_col, pop_col = 'Social Progress Index', 'Population size (no. of people)'
    d = dff.dropna(subset=[spi_col, pop_col]).copy()
    d = d[d[pop_col] > 0]
    bins = [0, 35, 45, 55, 65, 75, 85, 100]
    labels = ['Tier 7', 'Tier 6', 'Tier 5', 'Tier 4', 'Tier 3', 'Tier 2', 'Tier 1']
    d['Tier'] = pd.cut(d[spi_col], bins=bins, labels=labels, include_lowest=True)
    tier_colors = {'Tier 1': '#0b6623', 'Tier 2': '#8fd694', 'Tier 3': '#c9a0dc',
                   'Tier 4': '#5b7fa6', 'Tier 5': '#f4d03f', 'Tier 6': '#e87d2e', 'Tier 7': '#c0392b'}
    np.random.seed(42)
    d['yj'] = np.random.uniform(-2.5, 2.5, len(d))
    d['sz'] = np.sqrt(d[pop_col] / 1e6) * 3
    fig = go.Figure()
    # Label the top 30 most populous countries (the big bubbles)
    top_pop_countries = set(d.nlargest(30, pop_col)['Country'])

    for tier in labels[::-1]:
        td = d[d['Tier'] == tier]
        if td.empty: continue
        fig.add_trace(go.Scatter(x=td[spi_col], y=td['yj'], mode='markers+text',
            marker=dict(size=td['sz'], color=tier_colors[tier], opacity=0.6, line=dict(width=0.5, color='white')),
            text=[c if c in top_pop_countries else '' for c in td['Country']],
            textposition='top center', textfont=dict(size=9, color=DARK), name=tier,
            hovertemplate='<b>%{customdata[0]}</b><br>SPI: %{x:.1f}<br>Pop: %{customdata[1]:.0f}M<extra></extra>',
            customdata=list(zip(td['Country'], td[pop_col] / 1e6))))
    fig.update_layout(height=680, margin=dict(l=40, r=40, t=50, b=70), plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(title='Social Progress Index', range=[10, 100], showgrid=False),
                      yaxis=dict(visible=False, range=[-3.5, 3.5]),
                      legend=dict(title=dict(text='SPI Tiers'), orientation='h', y=-0.1, font=dict(size=9)),
                      font=dict(family="Segoe UI, sans-serif"))
    # Prominent population note at the top of the chart
    fig.add_annotation(text="\u25cf Bubble size = population size   (larger bubble = more people)",
                       xref='paper', yref='paper', x=0.5, y=1.08,
                       showarrow=False, font=dict(size=12, color=DARK), xanchor='center')
    return fig


def build_happiness_factors(hdf, year):
    factor_cols = {'Explained by: Log GDP per capita': 'GDP per capita',
                   'Explained by: Social support': 'Social support',
                   'Explained by: Healthy life expectancy': 'Healthy life expectancy',
                   'Explained by: Freedom to make life choices': 'Freedom to make life choices',
                   'Explained by: Generosity': 'Generosity',
                   'Explained by: Perceptions of corruption': 'Perceptions of corruption'}
    palette = {'GDP per capita': '#c0392b', 'Social support': '#e87d2e', 'Healthy life expectancy': '#f4d03f',
               'Freedom to make life choices': '#5b7fa6', 'Generosity': '#c9a0dc', 'Perceptions of corruption': '#8fd694'}
    top = hdf.nsmallest(20, 'Rank').sort_values('Happiness Score', ascending=True).copy()
    top['Label'] = '#' + top['Rank'].astype(int).astype(str) + '  ' + top['Country']
    pf = [c for c in factor_cols if c in top.columns]
    fs = top[pf].sum(axis=1)
    scale = top['Happiness Score'] / fs.replace(0, np.nan)
    for c in pf: top[c] = top[c] * scale
    fig = go.Figure()
    for i, col in enumerate(pf):
        lbl = factor_cols[col]
        is_last = i == len(pf) - 1
        fig.add_trace(go.Bar(y=top['Label'], x=top[col], orientation='h', name=lbl,
                             marker=dict(color=palette[lbl]), customdata=top['Country'],
                             text=[f"{s:.2f}" for s in top['Happiness Score']] if is_last else None,
                             textposition='outside' if is_last else 'none', textfont=dict(size=8, color='#555'),
                             hovertemplate='%{y}<br>' + lbl + ': %{x:.2f}<extra></extra>'))
    fig.update_layout(barmode='stack', height=600, margin=dict(l=110, r=50, t=50, b=30),
                      plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(title='Happiness Score', gridcolor='#eee'),
                      yaxis=dict(tickfont=dict(size=9)), legend=dict(orientation='h', y=-0.08, font=dict(size=8)),
                      title=dict(text=f'Top 20 Happiest Countries ({year})', font=dict(size=13, color='#2c3e50')))
    return fig


def build_happiness_vs_spi(year):
    """Scatter: Happiness Score vs SPI for the given year."""
    spi_col = 'Social Progress Index'
    spi_year = year if year in years else min(years, key=lambda y: abs(y - year))
    sdf = df[df['SPI year'] == spi_year][['Country', 'Region', spi_col]].dropna()
    hdf = happy_df[happy_df['Year'] == year][['Country', 'Happiness Score', 'Rank']].dropna()
    merged = hdf.merge(sdf, on='Country')
    if merged.empty:
        fig = go.Figure()
        fig.update_layout(height=300)
        return fig

    fig = go.Figure()
    for region in regions:
        rd = merged[merged['Region'] == region]
        if rd.empty: continue
        fig.add_trace(go.Scatter(
            x=rd[spi_col], y=rd['Happiness Score'], mode='markers', text=rd['Country'],
            marker=dict(size=10, color=region_colors.get(region, '#888'), opacity=0.8,
                        line=dict(width=0.3, color='white')),
            name=region,
            hovertemplate='<b>%{text}</b><br>SPI: %{x:.1f}<br>Happiness: %{y:.2f}<extra></extra>'))
    # Trend line
    if len(merged) > 3:
        corr = np.corrcoef(merged[spi_col], merged['Happiness Score'])[0, 1]
        a, b = np.polyfit(merged[spi_col], merged['Happiness Score'], 1)
        xl = np.linspace(merged[spi_col].min(), merged[spi_col].max(), 100)
        fig.add_trace(go.Scatter(x=xl, y=a * xl + b, mode='lines', line=dict(color='#2c3e50', width=2, dash='dash'),
                                 showlegend=False, hoverinfo='skip'))
        fig.add_annotation(text=f"r = {corr:.2f}", xref='paper', yref='paper', x=0.95, y=0.05,
                           showarrow=False, font=dict(size=12, color='#c0392b'), xanchor='right')
    fig.update_layout(height=400, margin=dict(l=50, r=30, t=20, b=50), plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(title='Social Progress Index', gridcolor='#eee'),
                      yaxis=dict(title='Happiness Score', gridcolor='#eee'),
                      font=dict(family="Segoe UI, sans-serif"))
    return fig


# ============ COUNTRY DEEP DIVE COMPONENTS ============

def build_flow_diagram(dim, dim_data):
    """HTML flow: dimension -> components -> sub-indicators."""
    comp_colors_list = ['#1f77b4', '#e8833a', '#0b6623', '#7b2d8b']
    blocks = []
    for i, (comp, indicators) in enumerate(dim_data['raw_indicators'].items()):
        col = comp_colors_list[i % len(comp_colors_list)]
        blocks.append(html.Div([
            html.Div(comp, style={'backgroundColor': col, 'color': 'white', 'padding': '5px 8px',
                                  'borderRadius': '5px', 'fontWeight': '600', 'fontSize': '0.75rem',
                                  'textAlign': 'center', 'marginBottom': '4px'}),
            html.Div([html.Div(ind.split(' (')[0], style={
                'border': f'1px solid {col}', 'borderRadius': '4px', 'padding': '2px 5px',
                'fontSize': '0.65rem', 'color': '#444', 'marginBottom': '3px', 'textAlign': 'center',
                'backgroundColor': '#fff'}) for ind, _ in indicators]),
        ], style={'flex': '1', 'padding': '0 4px', 'minWidth': '0'}))
    return html.Div([
        html.Div(dim, style={'textAlign': 'center', 'fontWeight': '700', 'fontSize': '0.9rem',
                             'color': '#2c3e50', 'backgroundColor': '#e9ecef', 'padding': '6px',
                             'borderRadius': '6px', 'marginBottom': '4px'}),
        html.Div("↓", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Div(blocks, style={'display': 'flex', 'justifyContent': 'space-between'}),
    ], style={'padding': '6px 0', 'marginBottom': '8px'})


def build_country_profile(dff, country, year, dim, dim_data):
    """Country profile card: map + stats + component comparison + SPI line."""
    spi_col = 'Social Progress Index'
    pop_col = 'Population size (no. of people)'
    gdp_col = 'GDP per capita (constant 2021 international dollars)'
    comps = dim_data['components']

    is_world = not country
    label = 'World' if is_world else country

    src = world_df[world_df['SPI year'] == year].iloc[0] if is_world else (
        dff[dff['Country'] == country].iloc[0] if country in dff['Country'].values else None)
    if src is None:
        return html.Div(f"No data for {label} in {year}.", style={'color': '#888', 'textAlign': 'center', 'padding': '40px'})

    pop_v = pd.to_numeric(src.get(pop_col), errors='coerce')
    gdp_v = pd.to_numeric(src.get(gdp_col), errors='coerce')
    spi_v = pd.to_numeric(src.get(spi_col), errors='coerce')
    wrow = world_df[world_df['SPI year'] == year].iloc[0] if not world_df[world_df['SPI year'] == year].empty else None

    # Region averages
    region_name = None
    if not is_world and country in dff['Country'].values:
        region_name = dff[dff['Country'] == country]['Region'].iloc[0]

    # Stat cards (HTML based, simpler than indicator traces)
    stat_style = {'display': 'inline-block', 'textAlign': 'center', 'padding': '8px 18px',
                  'backgroundColor': '#f0f7ff', 'borderRadius': '8px', 'margin': '4px 8px'}
    stat_cards = html.Div([
        html.Div([html.Div("👥 Population", style={'fontSize': '0.75rem', 'color': '#1f77b4'}),
                  html.Div(f"{pop_v / 1e6:.1f}M" if pd.notna(pop_v) else "N/A",
                           style={'fontSize': '1.3rem', 'fontWeight': '700', 'color': '#1f77b4'})], style=stat_style),
        html.Div([html.Div("💰 GDP per Capita", style={'fontSize': '0.75rem', 'color': '#1f77b4'}),
                  html.Div(f"${gdp_v / 1e3:.1f}K" if pd.notna(gdp_v) else "N/A",
                           style={'fontSize': '1.3rem', 'fontWeight': '700', 'color': '#1f77b4'})], style=stat_style),
        html.Div([html.Div("📈 Social Progress", style={'fontSize': '0.75rem', 'color': '#1f77b4'}),
                  html.Div(f"{spi_v:.1f}" if pd.notna(spi_v) else "N/A",
                           style={'fontSize': '1.3rem', 'fontWeight': '700', 'color': '#1f77b4'})], style=stat_style),
    ], style={'textAlign': 'center', 'marginBottom': '10px'})

    # Comparison bar chart: country vs region vs world
    country_vals = [pd.to_numeric(src.get(c), errors='coerce') for c in comps]
    world_vals = [pd.to_numeric(wrow.get(c), errors='coerce') for c in comps] if wrow is not None else [None] * len(comps)
    region_vals = None
    if region_name:
        rd = dff[dff['Region'] == region_name]
        region_vals = [rd[c].mean() for c in comps]

    bar_fig = go.Figure()
    if not is_world:
        bar_fig.add_trace(go.Bar(x=comps, y=country_vals, name=country, marker=dict(color='#1f77b4')))
    if region_vals:
        bar_fig.add_trace(go.Bar(x=comps, y=region_vals, name=f'{region_name} avg', marker=dict(color='#7fb0d4')))
    bar_fig.add_trace(go.Bar(x=comps, y=world_vals, name='World', marker=dict(color='#b0b0b0')))
    bar_fig.update_layout(barmode='group', height=380, margin=dict(l=50, r=20, t=55, b=60),
                          plot_bgcolor='white', paper_bgcolor='white', yaxis=dict(range=[0, 100], gridcolor='#eee', tickfont=dict(size=11)),
                          xaxis=dict(tickfont=dict(size=12), tickangle=0),
                          legend=dict(orientation='h', y=-0.22, font=dict(size=10)),
                          title=dict(text=f'{dim}: Country vs Region vs World', font=dict(size=16, color=DARK), x=0.5))
    bar_fig.add_annotation(text='How does this country compare to its regional peers and the global average on each component?', xref='paper', yref='paper', x=0.5, y=-0.32, showarrow=False, font=dict(size=10, color='#888'), xanchor='center')

    # SPI simple line
    if is_world:
        trend = world_df[['SPI year', spi_col]].dropna().sort_values('SPI year')
    else:
        trend = df[df['Country'] == country][['SPI year', spi_col]].dropna().sort_values('SPI year')
    spi_fig = go.Figure(go.Scatter(x=trend['SPI year'], y=trend[spi_col], mode='lines+markers',
                                   line=dict(color='#1f77b4', width=2), marker=dict(size=5)))
    spi_fig.update_layout(height=280, margin=dict(l=50, r=20, t=45, b=35), plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(dtick=2, gridcolor='#f0f0f0', tickfont=dict(size=10)),
                          yaxis=dict(title='SPI', gridcolor='#eee', tickfont=dict(size=10)),
                          title=dict(text='Social Progress Over Time', font=dict(size=16, color=DARK), x=0.5))
    spi_fig.add_annotation(text='How has this country\'s overall social progress changed across all years?', xref='paper', yref='paper', x=0.5, y=-0.25, showarrow=False, font=dict(size=10, color='#888'), xanchor='center')

    # Country locator map
    if not is_world:
        scope_map = {'Europe': 'europe', 'Sub-Saharan Africa': 'africa', 'Middle East & North Africa': 'africa',
                     'North America': 'north america', 'Latin America & Caribbean': 'south america',
                     'East Asia & Pacific': 'asia', 'South Asia': 'asia', 'Central Asia': 'asia'}
        scope = scope_map.get(region_name, None)
        map_fig = go.Figure()
        # Grey basemap for context
        map_fig.add_trace(go.Choropleth(
            locations=dff['Country'].tolist(), locationmode='country names',
            z=[0] * len(dff), showscale=False,
            colorscale=[[0, '#e8e8e8'], [1, '#e8e8e8']],
            marker_line_color='white', marker_line_width=0.3, hoverinfo='skip'))
        # Highlighted country in blue
        map_fig.add_trace(go.Choropleth(locations=[country], locationmode='country names', z=[1],
                                          showscale=False, colorscale=[[0, BLUE], [1, BLUE]],
                                          marker_line_color=DARK, marker_line_width=1, hoverinfo='location'))
        geo_opts = dict(showframe=False, showcoastlines=True, bgcolor='rgba(0,0,0,0)')
        if scope:
            geo_opts['scope'] = scope
        else:
            geo_opts['projection_type'] = 'natural earth'
        map_fig.update_layout(height=300, margin=dict(l=0, r=0, t=35, b=0), geo=geo_opts,
                              title=dict(text=country, font=dict(size=16, color=DARK), x=0.5))
        map_card = dcc.Graph(figure=map_fig, config={'displayModeBar': False})
    else:
        map_card = None

    has_map = map_card is not None
    top_row_cols = [dbc.Col(dcc.Graph(figure=spi_fig, config={'displayModeBar': False}), width=7 if has_map else 12)]
    if has_map:
        top_row_cols.append(dbc.Col(map_card, width=5))

    return html.Div([
        html.H4(f"{label} - Deep Dive ({year})", style={'textAlign': 'center', 'color': DARK, 'fontWeight': '700', 'marginBottom': '10px', 'fontSize': '1.5rem'}),
        # Stats row
        stat_cards,
        # Top row: SPI line (left) + map (right)
        dbc.Row(top_row_cols, className="mb-2", align="center"),
        # Below: bar chart full width
        dcc.Graph(figure=bar_fig, config={'displayModeBar': False}),
    ])


def build_indicator_charts(dff, country, dim_data):
    """Build indicator deep dive with VARIED chart types per component:
    Component 1 = lollipop (blue), Component 2 = gauge (green),
    Component 3 = diverging bar (yellow/amber), Component 4 = horizontal bar (red-to-green).
    Percentiles: 0 = worst outcome globally, 100 = best outcome globally."""
    row = dff[dff['Country'] == country].iloc[0]

    def percentile(col, higher_better):
        series = dff[col].dropna()
        val = row.get(col)
        if pd.isna(val) or series.empty: return None, val
        pct = (series < val).mean() * 100
        if not higher_better: pct = 100 - pct
        return pct, val

    comps = list(dim_data['raw_indicators'].keys())

    # All 4 charts share ONE consistent color scheme: green = good percentile, yellow = mid, red = poor
    def tier_color(p):
        if p is None: return GREY
        if p >= 70: return GREEN
        if p >= 40: return YELLOW
        return RED

    LABEL_FONT = 12
    AXIS_TITLE_FONT = 12
    TITLE_FONT = 18

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'domain'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        subplot_titles=comps, vertical_spacing=0.2, horizontal_spacing=0.12, row_heights=[0.5, 0.5])

    # --- Component 1: Lollipop ---
    ind1 = dim_data['raw_indicators'][comps[0]]
    labels, pcts, raws, colors1 = [], [], [], []
    for col, hb in ind1:
        if col not in dff.columns: continue
        p, v = percentile(col, hb)
        arrow = " \u2191" if hb else " \u2193"
        labels.append(col.split(' (')[0] + arrow)
        pcts.append(p if p is not None else 0)
        raws.append(v)
        colors1.append(tier_color(p))
    for lab, p in zip(labels, pcts):
        fig.add_trace(go.Scatter(x=[0, p], y=[lab, lab], mode='lines',
                                 line=dict(color='#cdd8e0', width=2), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pcts, y=labels, mode='markers',
        marker=dict(size=14, color=colors1, line=dict(width=1, color='white')),
        customdata=raws, showlegend=False,
        hovertemplate='%{y}<br>Percentile: %{x:.0f}/100<br>Raw: %{customdata:.2f}<extra></extra>'), row=1, col=1)
    fig.update_xaxes(range=[0, 100], title_text='Percentile (0=worst, 100=best)', title_font=dict(size=AXIS_TITLE_FONT),
                     row=1, col=1, gridcolor='#eee', tickfont=dict(size=LABEL_FONT))
    fig.update_yaxes(tickfont=dict(size=LABEL_FONT), row=1, col=1)

    # --- Component 2: Gauges (same tier colors) ---
    ind2 = dim_data['raw_indicators'][comps[1]]
    n2 = len(ind2)
    for i, (col, hb) in enumerate(ind2):
        if col not in dff.columns: continue
        p, v = percentile(col, hb)
        x0 = 0.55 + (i / n2) * 0.42
        x1 = 0.55 + ((i + 1) / n2 - 0.02) * 0.42
        fig.add_trace(go.Indicator(
            mode='gauge+number', value=p if p is not None else 0,
            number=dict(font=dict(size=16, color=tier_color(p)), suffix='/100'),
            title=dict(text=col.split(' (')[0] + (" \u2191" if hb else " \u2193"), font=dict(size=LABEL_FONT - 1, color='#333')),
            gauge=dict(axis=dict(range=[0, 100], tickfont=dict(size=9)),
                       bar=dict(color=tier_color(p), thickness=0.7),
                       steps=[dict(range=[0, 40], color='#f2c4c4'),
                              dict(range=[40, 70], color='#f5e6a8'),
                              dict(range=[70, 100], color='#c8e6c9')],
                       threshold=dict(line=dict(color=DARK, width=2), thickness=0.8, value=50)),
            domain=dict(x=[x0, x1], y=[0.58, 0.95])))

    # --- Component 3: Diverging bar (same tier colors) ---
    ind3 = dim_data['raw_indicators'][comps[2]]
    h_labels, h_diffs, h_colors, h_raws = [], [], [], []
    for col, hb in ind3:
        if col not in dff.columns: continue
        p, v = percentile(col, hb)
        if p is None: continue
        diff = p - 50
        arrow = " \u2191" if hb else " \u2193"
        h_labels.append(col.split(' (')[0] + arrow)
        h_diffs.append(diff)
        h_colors.append(tier_color(p))
        h_raws.append(v)
    fig.add_trace(go.Bar(y=h_labels, x=h_diffs, orientation='h', marker=dict(color=h_colors),
        customdata=h_raws, showlegend=False,
        hovertemplate='%{y}<br>%{x:+.0f} vs median<br>Raw: %{customdata:.2f}<extra></extra>'), row=2, col=1)
    fig.update_xaxes(range=[-55, 55], title_text='Distance from median (positive = better)',
                     title_font=dict(size=AXIS_TITLE_FONT), row=2, col=1, gridcolor='#eee', zeroline=True,
                     zerolinecolor=DARK, zerolinewidth=1, tickfont=dict(size=LABEL_FONT))
    fig.update_yaxes(tickfont=dict(size=LABEL_FONT), row=2, col=1)

    # --- Component 4: Horizontal bar (same tier colors) ---
    ind4 = dim_data['raw_indicators'][comps[3]]
    s_labels, s_pcts, s_raws, s_colors = [], [], [], []
    for col, hb in ind4:
        if col not in dff.columns: continue
        p, v = percentile(col, hb)
        arrow = " \u2191" if hb else " \u2193"
        s_labels.append(col.split(' (')[0] + arrow)
        pv = p if p is not None else 0
        s_pcts.append(pv)
        s_raws.append(v)
        s_colors.append(tier_color(p))
    fig.add_trace(go.Bar(y=s_labels, x=s_pcts, orientation='h', marker=dict(color=s_colors),
        customdata=s_raws, showlegend=False,
        hovertemplate='%{y}<br>Percentile: %{x:.0f}/100<br>Raw: %{customdata:.2f}<extra></extra>'), row=2, col=2)
    fig.update_xaxes(range=[0, 100], title_text='Percentile (0=worst, 100=best)', title_font=dict(size=AXIS_TITLE_FONT),
                     row=2, col=2, gridcolor='#eee', tickfont=dict(size=LABEL_FONT))
    fig.update_yaxes(tickfont=dict(size=LABEL_FONT), row=2, col=2)

    fig.update_layout(height=800, margin=dict(l=150, r=30, t=60, b=50), plot_bgcolor='white', paper_bgcolor='white',
                      font=dict(family="Segoe UI, sans-serif"))
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=TITLE_FONT, color=DARK)
    # Sub-chart descriptions
    fig.add_annotation(text="Lollipop: Where does this country rank globally? Dot position = percentile among all countries.",
                       xref='paper', yref='paper', x=0.22, y=0.48, showarrow=False,
                       font=dict(size=9, color='#888'), xanchor='center')
    fig.add_annotation(text="Gauges: How does this country compare? 50 = median country. Needle above 50 = better than most.",
                       xref='paper', yref='paper', x=0.78, y=0.48, showarrow=False,
                       font=dict(size=9, color='#888'), xanchor='center')
    fig.add_annotation(text="Diverging bar: Distance from the world median. Positive (right) = outperforming most countries.",
                       xref='paper', yref='paper', x=0.22, y=-0.02, showarrow=False,
                       font=dict(size=9, color='#888'), xanchor='center')
    fig.add_annotation(text="Horizontal bar: Full percentile position. Longer bar = better global ranking on that indicator.",
                       xref='paper', yref='paper', x=0.78, y=-0.02, showarrow=False,
                       font=dict(size=9, color='#888'), xanchor='center')
    fig.add_annotation(text="\u2191 = higher raw value is better    \u2193 = lower raw value is better    |    Green = top tier, Yellow = mid, Red = bottom tier",
                       xref='paper', yref='paper', x=0.5, y=-0.07, showarrow=False,
                       font=dict(size=10, color='#888'), xanchor='center')
    return fig


# ============ RUN ============

if __name__ == '__main__':
    app.run(debug=True, port=8050)
