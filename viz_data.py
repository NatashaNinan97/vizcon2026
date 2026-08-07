"""Vizcon data loading & preparation (unchanged semantics from the original).

Indicator columns referenced by DIMENSIONS are the SPI-scored versions
(bare names, 0-100, higher = better); raw-unit twins also exist in SPI.csv.
"""

import pandas as pd

from viz_theme import REGION_ORDER, REGION_REMAP, NORTH_AFRICA_COUNTRIES

# ------------------------------------------------------------------- SPI ---

df_raw = pd.read_csv('SPI.csv')
world_df = df_raw[df_raw['Country'] == 'World'].copy()
df = df_raw[df_raw['Country'] != 'World']
df = df[df['Region'].notna() & (df['Region'] != '')]

ALL_NUMERIC = [c for c in df.columns
               if c not in ['Country', 'SPI country code', 'Status', 'Region']]
for col in ALL_NUMERIC:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if col in world_df.columns:
        world_df[col] = pd.to_numeric(world_df[col], errors='coerce')

df['SPI year'] = df['SPI year'].astype(int)
world_df['SPI year'] = pd.to_numeric(world_df['SPI year'], errors='coerce').astype(int)

# Remap regions (currently identity — original SPI names are geographically correct)
if REGION_REMAP:
    df['Region'] = df['Region'].map(REGION_REMAP).fillna(df['Region'])

years = sorted(int(y) for y in df['SPI year'].unique())
LATEST = max(years)
EARLIEST = min(years)
countries = sorted(df['Country'].unique())
present = set(df['Region'].unique())
regions = [r for r in REGION_ORDER if r in present]

SPI_COL = 'Social Progress Index'
GDP_COL = 'GDP per capita (constant 2021 international dollars)'
POP_COL = 'Population size (no. of people)'

# -------------------------------------------------------------- happiness ---

happy_df = pd.read_csv('Happiness_report.csv')
happy_df = happy_df.rename(columns={
    'Country name': 'Country',
    'Life evaluation (3-year average)': 'Happiness Score'})
# Align WHR country names with SPI.csv spellings so the merge and the
# region lookup keep these countries (unambiguous renames only).
happy_df['Country'] = happy_df['Country'].replace({
    'Viet Nam': 'Vietnam',
    'Russian Federation': 'Russia',
    'Türkiye': 'Turkey',
    'Republic of Moldova': 'Moldova',
    'Côte d’Ivoire': "Côte d'Ivoire",
    'Lao PDR': 'Laos',
    'North Macedonia': 'Republic of North Macedonia',
    'DR Congo': 'Democratic Republic of Congo',
    'Congo': 'Republic of Congo',
    'Gambia': 'The Gambia',
    'Swaziland': 'Eswatini',
})
happy_df['Happiness Score'] = pd.to_numeric(happy_df['Happiness Score'], errors='coerce')
happy_df['Year'] = pd.to_numeric(happy_df['Year'], errors='coerce').astype(int)
country_region = df.drop_duplicates('Country').set_index('Country')['Region'].to_dict()
happy_df['Region'] = happy_df['Country'].map(country_region)
happy_years = sorted([int(y) for y in happy_df['Year'].unique() if y >= 2019])

# ------------------------------------------------------------- dimensions ---
# Component-level scores + scored indicators grouped by component.
# (Booleans kept from the original schema; every scored indicator is
# higher-is-better, so they are uniformly True.)

DIMENSIONS = {
    'Basic Needs': {
        'components': ['Nutrition and Medical Care', 'Water and Sanitation',
                       'Housing', 'Safety'],
        'raw_indicators': {
            'Nutrition and Medical Care': [
                ('Maternal mortality', True), ('Child stunting', True),
                ('Child mortality', True),
                ('Diet low in fruits and vegetables', True),
                ('Undernourishment', True), ('Infectious diseases', True),
            ],
            'Water and Sanitation': [
                ('Basic water service', True), ('Basic sanitation service', True),
                ('Unsafe water sanitation and hygiene', True),
                ('Satisfaction with water quality', True),
            ],
            'Housing': [
                ('Usage of clean fuels and technology', True),
                ('Access to electricity', True),
                ('Household air pollution', True),
                ('Dissatisfaction with housing affordability', True),
            ],
            'Safety': [
                ('Interpersonal violence', True),
                ('Transportation related injuries', True),
                ('Money stolen', True), ('Feeling safe walking alone', True),
                ('Intimate partner violence', True),
            ],
        },
    },
    'Foundations of Wellbeing': {
        'components': ['Basic Education', 'Information and Communications',
                       'Health', 'Environmental Quality'],
        'raw_indicators': {
            'Basic Education': [
                ('Children grow and learn', True),
                ('Equal access to quality education', True),
                ('Secondary school attainment', True),
                ('Gender parity in secondary attainment', True),
                ('Primary school enrollment', True),
            ],
            'Information and Communications': [
                ('Internet users', True), ('Mobile telephone users', True),
                ('Online Service Index', True),
                ('World Press Freedom Index', True),
            ],
            'Health': [
                ('Life expectancy at 65', True),
                ('Non-communicable diseases', True), ('Health Problems', True),
                ('Equal access to quality healthcare', True),
                ('Access to essential health services', True),
            ],
            'Environmental Quality': [
                ('Outdoor air pollution', True), ('Lead exposure', True),
                ('Waste recovery', True),
                ('Particulate matter pollution', True),
            ],
        },
    },
    'Societal Opportunity': {
        'components': ['Rights and Voice', 'Freedom and Choice',
                       'Inclusive Society', 'Advanced Education'],
        'raw_indicators': {
            'Rights and Voice': [
                ('Freedom of peaceful assembly', True),
                ('Equality before the law and individual liberty index', True),
                ('Rights equality', True), ('Perception of corruption', True),
                ('Political rights', True),
            ],
            'Freedom and Choice': [
                ('CSOs repression', True), ('Freedom over life choices', True),
                ('Vulnerable employment', True),
                ('Satisfied demand for contraception', True),
                ('Early marriage', True),
            ],
            'Inclusive Society': [
                ('Acceptance of gays and lesbians', True), ('Count on help', True),
                ('Equal access index', True),
                ('Young people not in education employment or training', True),
                ('Discrimination and violence against minorities', True),
            ],
            'Advanced Education': [
                ('Expected years of tertiary schooling', True),
                ('Citable documents', True),
                ('Women with advanced education', True),
                ('Quality weighted universities', True),
                (' Academic freedom', True),   # leading space matches the CSV
            ],
        },
    },
}

# The dimension score column for the third pillar is named 'Opportunity'
# in SPI.csv even though the pillar is titled 'Societal Opportunity'.
DIM_SCORE_COL = {'Basic Needs': 'Basic Needs',
                 'Foundations of Wellbeing': 'Foundations of Wellbeing',
                 'Societal Opportunity': 'Opportunity'}

N_INDICATORS = sum(len(v) for d in DIMENSIONS.values()
                   for v in d['raw_indicators'].values())

# ------------------------------------------------------------------ flags ---
# ISO3 (SPI country code) -> ISO2, for emoji flags. EU27 -> EU flag.

_ISO3_TO_2 = {
    'AFG': 'AF', 'ALB': 'AL', 'DZA': 'DZ', 'AGO': 'AO', 'ARG': 'AR',
    'ARM': 'AM', 'AUS': 'AU', 'AUT': 'AT', 'AZE': 'AZ', 'BHR': 'BH',
    'BGD': 'BD', 'BRB': 'BB', 'BLR': 'BY', 'BEL': 'BE', 'BEN': 'BJ',
    'BTN': 'BT', 'BOL': 'BO', 'BIH': 'BA', 'BWA': 'BW', 'BRA': 'BR',
    'BGR': 'BG', 'BFA': 'BF', 'BDI': 'BI', 'CPV': 'CV', 'KHM': 'KH',
    'CMR': 'CM', 'CAN': 'CA', 'CAF': 'CF', 'TCD': 'TD', 'CHL': 'CL',
    'CHN': 'CN', 'COL': 'CO', 'COM': 'KM', 'CRI': 'CR', 'HRV': 'HR',
    'CUB': 'CU', 'CYP': 'CY', 'CZE': 'CZ', 'CIV': 'CI', 'COD': 'CD',
    'PRK': 'KP', 'DNK': 'DK', 'DJI': 'DJ', 'DOM': 'DO', 'ECU': 'EC',
    'EGY': 'EG', 'SLV': 'SV', 'GNQ': 'GQ', 'ERI': 'ER', 'EST': 'EE',
    'SWZ': 'SZ', 'ETH': 'ET', 'EU27': 'EU', 'FJI': 'FJ', 'FIN': 'FI',
    'FRA': 'FR', 'GAB': 'GA', 'GEO': 'GE', 'DEU': 'DE', 'GHA': 'GH',
    'GRC': 'GR', 'GTM': 'GT', 'GIN': 'GN', 'GNB': 'GW', 'GUY': 'GY',
    'HTI': 'HT', 'HND': 'HN', 'HUN': 'HU', 'ISL': 'IS', 'IND': 'IN',
    'IDN': 'ID', 'IRN': 'IR', 'IRQ': 'IQ', 'IRL': 'IE', 'ISR': 'IL',
    'ITA': 'IT', 'JAM': 'JM', 'JPN': 'JP', 'JOR': 'JO', 'KAZ': 'KZ',
    'KEN': 'KE', 'KWT': 'KW', 'KGZ': 'KG', 'LAO': 'LA', 'LVA': 'LV',
    'LBN': 'LB', 'LSO': 'LS', 'LBR': 'LR', 'LBY': 'LY', 'LTU': 'LT',
    'LUX': 'LU', 'MDG': 'MG', 'MWI': 'MW', 'MYS': 'MY', 'MDV': 'MV',
    'MLI': 'ML', 'MLT': 'MT', 'MRT': 'MR', 'MUS': 'MU', 'MEX': 'MX',
    'MDA': 'MD', 'MNG': 'MN', 'MNE': 'ME', 'MAR': 'MA', 'MOZ': 'MZ',
    'MMR': 'MM', 'NAM': 'NA', 'NPL': 'NP', 'NLD': 'NL', 'NZL': 'NZ',
    'NIC': 'NI', 'NER': 'NE', 'NGA': 'NG', 'NOR': 'NO', 'OMN': 'OM',
    'PAK': 'PK', 'PAN': 'PA', 'PNG': 'PG', 'PRY': 'PY', 'PER': 'PE',
    'PHL': 'PH', 'POL': 'PL', 'PRT': 'PT', 'QAT': 'QA', 'COG': 'CG',
    'KOR': 'KR', 'MKD': 'MK', 'ROU': 'RO', 'RUS': 'RU', 'RWA': 'RW',
    'STP': 'ST', 'SAU': 'SA', 'SEN': 'SN', 'SRB': 'RS', 'SLE': 'SL',
    'SGP': 'SG', 'SVK': 'SK', 'SVN': 'SI', 'SLB': 'SB', 'SOM': 'SO',
    'ZAF': 'ZA', 'SSD': 'SS', 'ESP': 'ES', 'LKA': 'LK', 'SDN': 'SD',
    'SUR': 'SR', 'SWE': 'SE', 'CHE': 'CH', 'SYR': 'SY', 'TJK': 'TJ',
    'TZA': 'TZ', 'THA': 'TH', 'GMB': 'GM', 'TLS': 'TL', 'TGO': 'TG',
    'TTO': 'TT', 'TUN': 'TN', 'TUR': 'TR', 'TKM': 'TM', 'UGA': 'UG',
    'UKR': 'UA', 'ARE': 'AE', 'GBR': 'GB', 'USA': 'US', 'URY': 'UY',
    'UZB': 'UZ', 'VEN': 'VE', 'VNM': 'VN', 'WBG': 'PS', 'YEM': 'YE',
    'ZMB': 'ZM', 'ZWE': 'ZW',
}

_country_iso3 = (df.drop_duplicates('Country')
                 .set_index('Country')['SPI country code'].to_dict())


def flag(country):
    """Emoji flag for a country name, or '' when unknown."""
    iso2 = _ISO3_TO_2.get(str(_country_iso3.get(country, '')).strip())
    if not iso2:
        return ''
    return ''.join(chr(0x1F1E6 + ord(c) - ord('A')) for c in iso2)


# --------------------------------------------------------- cached helpers ---
from functools import lru_cache

@lru_cache(maxsize=32)
def get_year_df(year):
    """Cached year-filtered dataframe (avoid repeated filtering on the hot path)."""
    return df[df['SPI year'] == year]


# ------------------------------------------------------ indicator glossary ---
# Plain-language meaning for each scored indicator, grouped by dimension and
# component. Used by the Appendix tab.

INDICATOR_DEFINITIONS = {
    'Nutrition and Medical Care': {
        'Maternal mortality': 'Deaths of mothers per 100,000 live births.',
        'Child stunting': 'Share of children with low height-for-age from chronic undernutrition.',
        'Child mortality': 'Deaths of children under age five per 1,000 live births.',
        'Diet low in fruits and vegetables': 'Disease burden from insufficient fruit/vegetable intake.',
        'Undernourishment': 'Share of the population with insufficient caloric intake.',
        'Infectious diseases': 'Disease burden from communicable illnesses.',
    },
    'Water and Sanitation': {
        'Basic water service': 'Share of people with access to a basic drinking-water source.',
        'Basic sanitation service': 'Share of people with access to basic sanitation facilities.',
        'Unsafe water sanitation and hygiene': 'Disease burden attributable to unsafe water and hygiene.',
        'Satisfaction with water quality': 'Share of people satisfied with local water quality.',
    },
    'Housing': {
        'Usage of clean fuels and technology': 'Share of people using clean cooking fuels/technology.',
        'Access to electricity': 'Share of the population with electricity access.',
        'Household air pollution': 'Disease burden from indoor air pollution.',
        'Dissatisfaction with housing affordability': 'Share dissatisfied with the cost of housing.',
    },
    'Safety': {
        'Interpersonal violence': 'Disease/death burden from interpersonal violence.',
        'Transportation related injuries': 'Injury/death burden from transport accidents.',
        'Money stolen': 'Share of people reporting money/property stolen.',
        'Feeling safe walking alone': 'Share who feel safe walking alone at night.',
        'Intimate partner violence': 'Prevalence of violence by an intimate partner.',
    },
    'Basic Education': {
        'Children grow and learn': 'Share of children meeting growth and learning milestones.',
        'Equal access to quality education': 'Degree of equal educational access across groups.',
        'Secondary school attainment': 'Share of adults completing secondary school.',
        'Gender parity in secondary attainment': 'Balance between girls and boys completing secondary school.',
        'Primary school enrollment': 'Share of primary-age children enrolled in school.',
    },
    'Information and Communications': {
        'Internet users': 'Share of the population using the internet.',
        'Mobile telephone users': 'Share of the population with mobile phone subscriptions.',
        'Online Service Index': 'Quality/availability of government online services.',
        'World Press Freedom Index': 'Degree of press and media freedom.',
    },
    'Health': {
        'Life expectancy at 65': 'Expected remaining years of life at age 65.',
        'Non-communicable diseases': 'Death/disease burden from chronic (non-infectious) illness.',
        'Health Problems': 'Self-reported prevalence of health problems.',
        'Equal access to quality healthcare': 'Degree of equal healthcare access across groups.',
        'Access to essential health services': 'Coverage of essential health services.',
    },
    'Environmental Quality': {
        'Outdoor air pollution': 'Disease burden from ambient (outdoor) air pollution.',
        'Lead exposure': 'Health burden from lead exposure.',
        'Waste recovery': 'Share of waste recycled or recovered.',
        'Particulate matter pollution': 'Population exposure to fine particulate matter (PM2.5).',
    },
    'Rights and Voice': {
        'Freedom of peaceful assembly': 'Degree to which people can assemble peacefully.',
        'Equality before the law and individual liberty index': 'Equal legal treatment and personal liberty.',
        'Rights equality': 'Equality of civil and political rights across groups.',
        'Perception of corruption': 'Perceived level of public-sector corruption.',
        'Political rights': 'Degree of political rights and electoral freedom.',
    },
    'Freedom and Choice': {
        'CSOs repression': 'Degree of repression of civil society organizations.',
        'Freedom over life choices': 'Share satisfied with their freedom to choose in life.',
        'Vulnerable employment': 'Share of workers in insecure/informal employment.',
        'Satisfied demand for contraception': 'Share of contraception demand that is met.',
        'Early marriage': 'Prevalence of marriage before age 18.',
    },
    'Inclusive Society': {
        'Acceptance of gays and lesbians': 'Social acceptance of gay and lesbian people.',
        'Count on help': 'Share who have someone to count on in times of need.',
        'Equal access index': 'Degree of equal access to services across groups.',
        'Young people not in education employment or training': 'Share of youth not in education, work, or training (NEET).',
        'Discrimination and violence against minorities': 'Prevalence of discrimination/violence against minorities.',
    },
    'Advanced Education': {
        'Expected years of tertiary schooling': 'Expected years of higher education.',
        'Citable documents': 'Volume of internationally citable research output.',
        'Women with advanced education': 'Share of women with advanced (tertiary) education.',
        'Quality weighted universities': 'Count of universities weighted by quality/ranking.',
        'Academic freedom': 'Degree of academic freedom.',
    },
}

# Which components belong to which dimension (for grouping in the appendix)
DIMENSION_COMPONENTS = {
    'Basic Needs (Live)': ['Nutrition and Medical Care', 'Water and Sanitation',
                           'Housing', 'Safety'],
    'Foundations of Wellbeing (Thrive)': ['Basic Education',
                                          'Information and Communications',
                                          'Health', 'Environmental Quality'],
    'Societal Opportunity (Connect)': ['Rights and Voice', 'Freedom and Choice',
                                       'Inclusive Society', 'Advanced Education'],
}
