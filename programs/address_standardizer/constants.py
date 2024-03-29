"""
Source: https://pe.usps.com/text/pub28/welcome.htm

"""
STREET_NAME_POST_ABBREVIATIONS = {
    "ALLEE": "ALY",
    "ALLEY": "ALY",
    "ALLY": "ALY",
    "ALY": "ALY",
    "ANEX": "ANX",
    "ANNEX": "ANX",
    "ANNX": "ANX",
    "ANX": "ANX",
    "ARC": "ARC",
    "ARC ": "ARC",
    "ARCADE": "ARC",
    "ARCADE ": "ARC",
    "AV": "AVE",
    "AVE": "AVE",
    "AVEN": "AVE",
    "AVENU": "AVE",
    "AVENUE": "AVE",
    "AVN": "AVE",
    "AVNUE": "AVE",
    "BAYOO": "BYU",
    "BAYOU": "BYU",
    "BCH": "BCH",
    "BEACH": "BCH",
    "BEND": "BND",
    "BLF": "BLF",
    "BLUF": "BLF",
    "BLUFF": "BLF",
    "BLUFFS": "BLFS",
    "BLUFFS ": "BLFS",
    "BLVD": "BLVD",
    "BND": "BND",
    "BOT": "BTM",
    "BOTTM": "BTM",
    "BOTTOM": "BTM",
    "BOUL": "BLVD",
    "BOULEVARD": "BLVD",
    "BOULEVARD ": "BLVD",
    "BOULV": "BLVD",
    "BR": "BR",
    "BRANCH": "BR",
    "BRDGE": "BRG",
    "BRG": "BRG",
    "BRIDGE": "BRG",
    "BRK": "BRK",
    "BRNCH": "BR",
    "BROOK": "BRK",
    "BROOKS": "BRKS",
    "BROOKS ": "BRKS",
    "BTM": "BTM",
    "BURG": "BG",
    "BURGS": "BGS",
    "BYP": "BYP",
    "BYPA": "BYP",
    "BYPAS": "BYP",
    "BYPASS": "BYP",
    "BYPS": "BYP",
    "CAMP": "CP",
    "CANYN": "CYN",
    "CANYON": "CYN",
    "CAPE": "CPE",
    "CAUSEWAY": "CSWY",
    "CAUSWA": "CSWY",
    "CAUSWAY": "CSWY",
    "CEN": "CTR",
    "CENT": "CTR",
    "CENTER": "CTR",
    "CENTERS": "CTRS",
    "CENTERS ": "CTRS",
    "CENTR": "CTR",
    "CENTRE": "CTR",
    "CIR": "CIR",
    "CIRC": "CIR",
    "CIRCL": "CIR",
    "CIRCLE": "CIR",
    "CIRCLES": "CIRS",
    "CK": "CRK",
    "CLB": "CLB",
    "CLF": "CLF",
    "CLFS": "CLFS",
    "CLIFF": "CLF",
    "CLIFFS": "CLFS",
    "CLUB": "CLB",
    "CMP": "CP",
    "CNTER": "CTR",
    "CNTR": "CTR",
    "CNYN": "CYN",
    "COMMON": "CMN",
    "COMMONS": "CMNS",
    "COR": "COR",
    "CORNER": "COR",
    "CORNERS": "CORS",
    "CORS": "CORS",
    "COURSE": "CRSE",
    "COURT": "CT",
    "COURTS": "CTS",
    "COVE": "CV",
    "COVES": "CVS",
    "CP": "CP",
    "CPE": "CPE",
    "CR": "CRK",
    "CRCL": "CIR",
    "CRCLE": "CIR",
    "CRECENT": "CRES",
    "CREEK": "CRK",
    "CRES": "CRES",
    "CRESCENT": "CRES",
    "CRESENT": "CRES",
    "CREST": "CRST",
    "CRK": "CRK",
    "CROSSING": "XING",
    "CROSSING ": "XING",
    "CROSSROAD": "XRD",
    "CROSSROADS": "XRDS",
    "CRSCNT": "CRES",
    "CRSE": "CRSE",
    "CRSENT": "CRES",
    "CRSNT": "CRES",
    "CRSSING": "XING",
    "CRSSNG": "XING",
    "CRSSNG ": "XING",
    "CRT": "CT",
    "CSWY": "CSWY",
    "CT": "CT",
    "CTR": "CTR",
    "CTS": "CTS",
    "CURVE": "CURV",
    "CURVE ": "CURV",
    "CV": "CV",
    "CYN": "CYN",
    "DALE": "DL",
    "DALE ": "DL",
    "DAM": "DM",
    "DAM ": "DM",
    "DIV": "DV",
    "DIVIDE": "DV",
    "DL": "DL",
    "DL ": "DL",
    "DM": "DM",
    "DM ": "DM",
    "DR": "DR",
    "DRIV": "DR",
    "DRIVE": "DR",
    "DRIVES": "DRS",
    "DRV": "DR",
    "DV": "DV",
    "DVD": "DV",
    "EST": "EST",
    "ESTATE": "EST",
    "ESTATES": "ESTS",
    "ESTS": "ESTS",
    "EXP": "EXPY",
    "EXPR": "EXPY",
    "EXPRESS": "EXPY",
    "EXPRESSWAY": "EXPY",
    "EXPW": "EXPY",
    "EXPY": "EXPY",
    "EXT": "EXT",
    "EXTENSION": "EXT",
    "EXTENSIONS": "EXTS",
    "EXTN": "EXT",
    "EXTNSN": "EXT",
    "EXTS": "EXTS",
    "FALL": "FALL",
    "FALLS": "FLS",
    "FERRY": "FRY",
    "FIELD": "FLD",
    "FIELDS": "FLDS",
    "FLAT": "FLT",
    "FLATS": "FLTS",
    "FLD": "FLD",
    "FLDS": "FLDS",
    "FLS": "FLS",
    "FLT": "FLT",
    "FLTS": "FLTS",
    "FORD": "FRD",
    "FORDS": "FRDS",
    "FOREST": "FRST",
    "FORESTS": "FRST",
    "FORG": "FRG",
    "FORGE": "FRG",
    "FORGES": "FRGS",
    "FORK": "FRK",
    "FORKS": "FRKS",
    "FORT": "FT",
    "FRD": "FRD",
    "FREEWAY": "FWY",
    "FREEWY": "FWY",
    "FRG": "FRG",
    "FRK": "FRK",
    "FRKS": "FRKS",
    "FRRY": "FRY",
    "FRST": "FRST",
    "FRT": "FT",
    "FRWAY": "FWY",
    "FRWY": "FWY",
    "FRY": "FRY",
    "FT": "FT",
    "FWY": "FWY",
    "GARDEN": "GDN",
    "GARDENS": "GDNS",
    "GARDN": "GDN",
    "GATEWAY": "GTWY",
    "GATEWY": "GTWY",
    "GATWAY": "GTWY",
    "GDN": "GDN",
    "GDNS": "GDNS",
    "GLEN": "GLN",
    "GLENS": "GLNS",
    "GLN": "GLN",
    "GRDEN": "GDN",
    "GRDN": "GDN",
    "GRDNS": "GDNS",
    "GREEN": "GRN",
    "GREENS": "GRNS",
    "GRN": "GRN",
    "GROV": "GRV",
    "GROVE": "GRV",
    "GROVES": "GRVS",
    "GRV": "GRV",
    "GTWAY": "GTWY",
    "GTWY": "GTWY",
    "HARB": "HBR",
    "HARBOR": "HBR",
    "HARBORS": "HBRS",
    "HARBR": "HBR",
    "HAVEN": "HVN",
    "HAVN": "HVN",
    "HBR": "HBR",
    "HEIGHT": "HTS",
    "HEIGHTS": "HTS",
    "HGTS": "HTS",
    "HIGHWAY": "HWY",
    "HIGHWY": "HWY",
    "HILL": "HL",
    "HILLS": "HLS",
    "HIWAY": "HWY",
    "HIWY": "HWY",
    "HL": "HL",
    "HLLW": "HOLW",
    "HLS": "HLS",
    "HOLLOW": "HOLW",
    "HOLLOWS": "HOLW",
    "HOLW": "HOLW",
    "HOLWS": "HOLW",
    "HRBOR": "HBR",
    "HT": "HTS",
    "HTS": "HTS",
    "HVN": "HVN",
    "HWAY": "HWY",
    "HWY": "HWY",
    "INLET": "INLT",
    "INLT": "INLT",
    "IS": "IS",
    "ISLAND": "IS",
    "ISLANDS": "ISS",
    "ISLE": "ISLE",
    "ISLES": "ISLE",
    "ISLND": "IS",
    "ISLNDS": "ISS",
    "ISS": "ISS",
    "JCT": "JCT",
    "JCTION": "JCT",
    "JCTN": "JCT",
    "JCTNS": "JCTS",
    "JCTS": "JCTS",
    "JUNCTION": "JCT",
    "JUNCTIONS": "JCTS",
    "JUNCTN": "JCT",
    "JUNCTON": "JCT",
    "KEY": "KY",
    "KEYS": "KYS",
    "KNL": "KNL",
    "KNLS": "KNLS",
    "KNOL": "KNL",
    "KNOLL": "KNL",
    "KNOLLS": "KNLS",
    "KY": "KY",
    "KYS": "KYS",
    "LA": "LN",
    "LAKE": "LK",
    "LAKES": "LKS",
    "LAND": "LAND",
    "LANDING": "LNDG",
    "LANE": "LN",
    "LANES": "LN",
    "LCK": "LCK",
    "LCKS": "LCKS",
    "LDG": "LDG",
    "LDGE": "LDG",
    "LF": "LF",
    "LGT": "LGT",
    "LIGHT": "LGT",
    "LIGHTS": "LGTS",
    "LK": "LK",
    "LKS": "LKS",
    "LN": "LN",
    "LNDG": "LNDG",
    "LNDNG": "LNDG",
    "LOAF": "LF",
    "LOCK": "LCK",
    "LOCKS": "LCKS",
    "LODG": "LDG",
    "LODGE": "LDG",
    "LOOP": "LOOP",
    "LOOPS": "LOOP",
    "MALL": "MALL",
    "MANOR": "MNR",
    "MANORS": "MNRS",
    "MDW": "MDW",
    "MDWS": "MDWS",
    "MEADOW": "MDW",
    "MEADOWS": "MDWS",
    "MEDOWS": "MDWS",
    "MEWS": "MEWS",
    "MILL": "ML",
    "MILLS": "MLS",
    "MISSION": "MSN",
    "MISSN": "MSN",
    "ML": "ML",
    "MLS": "MLS",
    "MNR": "MNR",
    "MNRS": "MNRS",
    "MNT": "MT",
    "MNTAIN": "MTN",
    "MNTN": "MTN",
    "MNTNS": "MTNS",
    "MOTORWAY": "MTWY",
    "MOUNT": "MT",
    "MOUNTAIN": "MTN",
    "MOUNTAINS": "MTNS",
    "MOUNTIN": "MTN",
    "MSN": "MSN",
    "MSSN": "MSN",
    "MT": "MT",
    "MTIN": "MTN",
    "MTN": "MTN",
    "NCK": "NCK",
    "NECK": "NCK",
    "ORCH": "ORCH",
    "ORCHARD": "ORCH",
    "ORCHRD": "ORCH",
    "OVAL": "OVAL",
    "OVERPASS": "OPAS",
    "OVL": "OVAL",
    "PARK": "PARK",
    "PARKS": "PARK",
    "PARKWAY": "PKWY",
    "PARKWAYS": "PKWY",
    "PARKWY": "PKWY",
    "PASS": "PASS",
    "PASSAGE": "PSGE",
    "PATH": "PATH",
    "PATHS": "PATH",
    "PIKE": "PIKE",
    "PIKES": "PIKE",
    "PINE": "PNE",
    "PINES": "PNES",
    "PK": "PARK",
    "PKWAY": "PKWY",
    "PKWY": "PKWY",
    "PKWYS": "PKWY",
    "PKY": "PKWY",
    "PL": "PL",
    "PLACE": "PL",
    "PLAIN": "PLN",
    "PLAINES": "PLNS",
    "PLAINS": "PLNS",
    "PLAZA": "PLZ",
    "PLN": "PLN",
    "PLNS": "PLNS",
    "PLZ": "PLZ",
    "PLZA": "PLZ",
    "PNES": "PNES",
    "POINT": "PT",
    "POINTS": "PTS",
    "PORT": "PRT",
    "PORTS": "PRTS",
    "PR": "PR",
    "PRAIRIE": "PR",
    "PRARIE": "PR",
    "PRK": "PARK",
    "PRR": "PR",
    "PRT": "PRT",
    "PRTS": "PRTS",
    "PT": "PT",
    "PTS": "PTS",
    "RAD": "RADL",
    "RADIAL": "RADL",
    "RADIEL": "RADL",
    "RADL": "RADL",
    "RAMP": "RAMP",
    "RANCH": "RNCH",
    "RANCHES": "RNCH",
    "RAPID": "RPD",
    "RAPIDS": "RPDS",
    "RD": "RD",
    "RDG": "RDG",
    "RDGE": "RDG",
    "RDGS": "RDGS",
    "RDS": "RDS",
    "REST": "RST",
    "RIDGE": "RDG",
    "RIDGES": "RDGS",
    "RIV": "RIV",
    "RIVER": "RIV",
    "RIVR": "RIV",
    "RNCH": "RNCH",
    "RNCHS": "RNCH",
    "ROAD": "RD",
    "ROADS": "RDS",
    "ROUTE": "RTE",
    "ROW": "ROW",
    "RPD": "RPD",
    "RPDS": "RPDS",
    "RST": "RST",
    "RUE": "RUE",
    "RUN": "RUN",
    "RVR": "RIV",
    "SHL": "SHL",
    "SHLS": "SHLS",
    "SHOAL": "SHL",
    "SHOALS": "SHLS",
    "SHOAR": "SHR",
    "SHOARS": "SHRS",
    "SHORE": "SHR",
    "SHORES": "SHRS",
    "SHR": "SHR",
    "SHRS": "SHRS",
    "SKYWAY": "SKWY",
    "SMT": "SMT",
    "SPG": "SPG",
    "SPGS": "SPGS",
    "SPNG": "SPG",
    "SPNGS": "SPGS",
    "SPRING": "SPG",
    "SPRINGS": "SPGS",
    "SPRNG": "SPG",
    "SPRNGS": "SPGS",
    "SPUR": "SPUR",
    "SPURS": "SPUR",
    "SQ": "SQ",
    "SQR": "SQ",
    "SQRE": "SQ",
    "SQRS": "SQS",
    "SQU": "SQ",
    "SQUARE": "SQ",
    "SQUARES": "SQS",
    "ST": "ST",
    "STA": "STA",
    "STATION": "STA",
    "STATN": "STA",
    "STN": "STA",
    "STR": "ST",
    "STRA": "STRA",
    "STRAV": "STRA",
    "STRAVE": "STRA",
    "STRAVEN": "STRA",
    "STRAVENUE": "STRA",
    "STRAVN": "STRA",
    "STREAM": "STRM",
    "STREET": "ST",
    "STREETS": "STS",
    "STREME": "STRM",
    "STRM": "STRM",
    "STRT": "ST",
    "STRVN": "STRA",
    "STRVNUE": "STRA",
    "SUMIT": "SMT",
    "SUMITT": "SMT",
    "SUMMIT": "SMT",
    "TER": "TER",
    "TERR": "TER",
    "TERRACE": "TER",
    "THROUGHWAY": "TRWY",
    "TPK": "TPKE",
    "TPKE": "TPKE",
    "TR": "TRL",
    "TRACE": "TRCE",
    "TRACES": "TRCE",
    "TRACK": "TRAK",
    "TRACKS": "TRAK",
    "TRAFFICWAY": "TRFY",
    "TRAIL": "TRL",
    "TRAILER": "TRLR",
    "TRAILS": "TRL",
    "TRAK": "TRAK",
    "TRCE": "TRCE",
    "TRFY": "TRFY",
    "TRK": "TRAK",
    "TRKS": "TRAK",
    "TRL": "TRL",
    "TRLR": "TRLR",
    "TRLRS": "TRLR",
    "TRLS": "TRL",
    "TRNPK": "TPKE",
    "TRPK": "TPKE",
    "TUNEL": "TUNL",
    "TUNL": "TUNL",
    "TUNLS": "TUNL",
    "TUNNEL": "TUNL",
    "TUNNELS": "TUNL",
    "TUNNL": "TUNL",
    "TURNPIKE": "TPKE",
    "TURNPK": "TPKE",
    "UN": "UN",
    "UNDERPASS": "UPAS",
    "UNION": "UN",
    "UNIONS": "UNS",
    "VALLEY": "VLY",
    "VALLEYS": "VLYS",
    "VALLY": "VLY",
    "VDCT": "VIA",
    "VIA": "VIA",
    "VIADCT": "VIA",
    "VIADUCT": "VIA",
    "VIEW": "VW",
    "VIEWS": "VWS",
    "VILL": "VLG",
    "VILLAG": "VLG",
    "VILLAGE": "VLG",
    "VILLAGES": "VLGS",
    "VILLE": "VL",
    "VILLG": "VLG",
    "VILLIAGE": "VLG",
    "VIS": "VIS",
    "VIST": "VIS",
    "VISTA": "VIS",
    "VL": "VL",
    "VLG": "VLG",
    "VLGS": "VLGS",
    "VLLY": "VLY",
    "VLY": "VLY",
    "VLYS": "VLYS",
    "VST": "VIS",
    "VSTA": "VIS",
    "VW": "VW",
    "VWS": "VWS",
    "WALK": "WALK",
    "WALKS": "WALK",
    "WALL": "WALL",
    "WAY": "WAY",
    "WAYS": "WAYS",
    "WELL": "WL",
    "WELLS": "WLS",
    "WLS": "WLS",
    "WY": "WAY",
    "XING": "XING",
    "XING ": "XING"
}

# Even though we don't care about normalizing the state names themselves,
# state names may appear inside of street names (i.e. Kentucky Highway).
STATE_ABBREVIATIONS = {
    'ALABAMA': 'AL',
    'ALA': 'AL',
    'ALASKA': 'AK',
    'ALAS': 'AK',
    'ARIZONA': 'AZ',
    'ARIZ': 'AZ',
    'ARKANSAS': 'AR',
    'ARK': 'AR',
    'CALIFORNIA': 'CA',
    'CALIF': 'CA',
    'CAL': 'CA',
    'COLORADO': 'CO',
    'COLO': 'CO',
    'COL': 'CO',
    'CONNECTICUT': 'CT',
    'CONN': 'CT',
    'DELAWARE': 'DE',
    'DEL': 'DE',
    'DISTRICT OF COLUMBIA': 'DC',
    'FLORIDA': 'FL',
    'FLA': 'FL',
    'FLOR': 'FL',
    'GEORGIA': 'GA',
    'GA': 'GA',
    'HAWAII': 'HI',
    'IDAHO': 'ID',
    'IDA': 'ID',
    'ILLINOIS': 'IL',
    'ILL': 'IL',
    'INDIANA': 'IN',
    'IND': 'IN',
    'IOWA': 'IA',
    'KANSAS': 'KS',
    'KANS': 'KS',
    'KAN': 'KS',
    'KENTUCKY': 'KY',
    'KEN': 'KY',
    'KENT': 'KY',
    'LOUISIANA': 'LA',
    'MAINE': 'ME',
    'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA',
    'MASS': 'MA',
    'MICHIGAN': 'MI',
    'MICH': 'MI',
    'MINNESOTA': 'MN',
    'MINN': 'MN',
    'MISSISSIPPI': 'MS',
    'MISS': 'MS',
    'MISSOURI': 'MO',
    'MONTANA': 'MT',
    'MONT': 'MT',
    'NEBRASKA': 'NE',
    'NEBR': 'NE',
    'NEB': 'NE',
    'NEVADA': 'NV',
    'NEV': 'NV',
    'NEW HAMPSHIRE': 'NH',
    'NEW JERSEY': 'NJ',
    'NEW MEXICO': 'NM',
    'N MEX': 'NM',
    'NEW M': 'NM',
    'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC',
    'NORTH DAKOTA': 'ND',
    'N DAK': 'ND',
    'OHIO': 'OH',
    'OKLAHOMA': 'OK',
    'OKLA': 'OK',
    'OREGON': 'OR',
    'OREG': 'OR',
    'ORE': 'OR',
    'PENNSYLVANIA': 'PA',
    'PENN': 'PA',
    'RHODE ISLAND': 'RI',
    'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD',
    'S DAK': 'SD',
    'TENNESSEE': 'TN',
    'TENN': 'TN',
    'TEXAS': 'TX',
    'TEX': 'TX',
    'UTAH': 'UT',
    'VERMONT': 'VT',
    'VIRGINIA': 'VA',
    'WASHINGTON': 'WA',
    'WASH': 'WA',
    'WEST VIRGINIA': 'WV',
    'W VA': 'WV',
    'WISCONSIN': 'WI',
    'WIS': 'WI',
    'WISC': 'WI',
    'WYOMING': 'WY',
    'WYO': 'WY'
}

STREET_NAME_ABBREVIATIONS = {
    'COUNTY HWY': 'COUNTY HIGHWAY',
    'CNTY HWY': 'COUNTY HIGHWAY',
    'COUNTY RD': 'COUNTY ROAD',
    'CR': 'COUNTY ROAD',
    'CNTY RD': 'COUNTY ROAD',
    'CORD': 'COUNTY ROAD',
    'CO. RD': 'COUNTY ROAD',
    'CO RD': 'COUNTY ROAD',
    'CR-': 'COUNTY ROAD',
    'CR #': 'COUNTY ROAD',
    'CNTY. RD': 'COUNTY ROAD',
    'CR.': 'COUNTY ROAD',
    'FARM TO MARKET': 'FM',
    'HWY FM': 'FM',
    'HIWAY': 'HIGHWAY',
    'HWY': 'HIGHWAY',
    'FRONTAGE ROAD': 'FRONTAGE RD',
    'BYPASS': 'BYP',
    'BYP RD': 'BYPASS RD',
    'INTERSTATE HWY': 'INTERSTATE',
    'IH': 'INTERSTATE',
    'I': 'INTERSTATE', #to account for cases like I10 OR I 55
    'RD': 'ROAD',
    'RT': 'ROUTE',
    'RTE': 'ROUTE',
    'RANCH ROAD': 'RANCH ROAD',
    'ST HWY': 'STATE HIGHWAY',
    'STHWY': 'STATE HIGHWAY',
    'ST-HWY': 'STATE HIGHWAY',
    'ST.HWY.': 'STATE HIGHWAY',
    'STATE HIGH WAY': 'STATE HIGHWAY',
    'S HWY': 'STATE HIGHWAY',
    'ST HIGHWAY': 'STATE HIGHWAY',
    'STATE HWY': 'STATE HIGHWAY',
    'SR': 'STATE ROAD',
    'ST RT': 'STATE ROUTE',
    'STATE RTE': 'STATE ROUTE',
    'TSR': 'TOWNSHIP ROAD',
    'TWP HWY': 'TOWNSHIP HIGHWAY',
    'TWN HWY': 'TOWNSHIP HIGHWAY',
    'TNHW': 'TOWNSHIP HIGHWAY',
    'US': 'US HIGHWAY',
    'US HWY' : 'US HIGHWAY',
    'USHWY' : 'US HIGHWAY',
    'US HWY': 'US HIGHWAY',
    'US-HWY': 'US HIGHWAY',
    'US.HWY.': 'US HIGHWAY',
    'PR': 'PRIAVATE ROAD',
}

# Can be used for pre and post directional info
DIRECTIONAL_ABBREVIATIONS = {
    'EAST': 'E',
    'WEST': 'W',
    'NORTH': 'N',
    'SOUTH': 'S',
    'NORTHEAST': 'NE',
    'NORTHWEST': 'NW',
    'SOUTHEAST': 'SE',
    'SOUTHWEST': 'SW',
    "NORTE": "N",
    "NO": "N",
    "NORESTE": "NE",
    "NOROESTE": "NW",
    "SUR": "S",
    "SO": "S",
    "SURESTE": "SE",
    "SUROESTE": "SW",
    "ESTE": "E",
    "EA": "E",
    "OESTE": "W",
    "WE": "W"
}

#From USPS "C2 Secondary Unit Designators"
#Subaddress Type/WSDESC1 (?)
OCCUPANCY_TYPE_ABBREVIATIONS = {
    'APARTMENT': 'APT',
    'BUILDING': 'BLDG',
    'BASEMENT': 'BSMT',
    'DEPARTMENT': 'DEPT',
    'FLOOR': 'FL',
    'FRONT': 'FRNT',
    'HANGER': 'HNGR',
    'KEY': 'KEY',
    'LOBBY': 'LBBY',
    'LOT': 'LOT',
    'LOWER': 'LOWR',
    'OFFICE': 'OFC',
    'PENTHOUSE': 'PH',
    'PIER': 'PIER',
    'REAR': 'REAR',
    'ROOM': 'RM',
    'SIDE': 'SIDE',
    'SLIP': 'SLIP',
    'SPACE': 'SPC',
    'STOP': 'STOP',
    'SUITE': 'STE',
    'TRAILER': 'TRLR',
    'UNIT': 'UNIT',
    'UPPER': 'UPPER'
}

DIRECTION_CODES = {
    "N": 1,
    "S": 2,
    "E": 3,
    "W": 4,
    "NE": 5,
    "NW": 6,
    "SE": 7,
    "SW": 8
}

EXTENSION_CODES = {
    "EXTD": 1,
    "EXTN": 2,
    "LP": 3,
    "BYP": 4,
    "ALT": 5,
    "BUS": 6,
    "OLD": 7,
    "SPUR": 8
}

STREET_TYPE_CODES = {
    "ALY": 11,
    "ALT": 12,
    "ARC": 15,
    "ARRY": 16,
    "APTA": 17,
    "AVA": 18,
    "AVE": 19,
    "BLVD": 26,
    "BLV": 32,
    "BSRT": 33,
    "BYP": 34,
    "CLLE": 36,
    "CJA": 37,
    "CJON": 38,
    "CAM": 39,
    "CARR": 47,
    "CSWY": 48,
    "CTR": 51,
    "CIR": 57,
    "CORD": 70,
    "CT": 71,
    "CV": 73,
    "CRES": 76,
    "XING": 77,
    "CRU": 78,
    "DR": 87,
    "EXP": 93,
    "EXPY": 94,
    "FM": 99,
    "4WD": 110,
    "FWY": 112,
    "HWY": 122,
    "I-": 133,
    "JPTR": 138,
    "LN": 146,
    "LOOP": 151,
    "MARG": 154,
    "MTWY": 164,
    "MRO": 167,
    "OVPS": 178,
    "PARK": 179,
    "PKY": 180,
    "PAS": 182,
    "PSO": 183,
    "PASS": 185,
    "PATH": 187,
    "PIKE": 189,
    "PSTA": 191,
    "PL": 192,
    "PLZ": 193,
    "PTE": 202,
    "RML": 208,
    "RMP": 210,
    "ROAD": 223,
    "RT": 227,
    "ROW": 228,
    "RUE": 229,
    "RUN": 230,
    "RUTA": 232,
    "SNDR": 239,
    "SVRD": 240,
    "SKWY": 248,
    "SPWY": 253,
    "SQ": 256,
    "STHY": 259,
    "ST": 263,
    "TER": 268,
    "THFR": 269,
    "THWY": 270,
    "TWHY": 273,
    "TFWY": 274,
    "TRL": 275,
    "TUN": 278,
    "TUNL": 279,
    "TPKE": 280,
    "UNPS": 281,
    "USHY": 283,
    "UNRD": 286,
    "VRDA": 288,
    "VIA": 289,
    "WALK": 291,
    "WKWY": 292,
    "WALL": 293,
    "WAY": 296,
    "NFD": 302,
    "OVAL": 303,
    "EST": 304,
    "VLLA": 305,
    "DRWY": 306,
    "RDWY": 307,
    "STRA": 308,
    "CLUB": 309,
    "CTS": 310,
    "JCT": 311,
    "LNDG": 312,
    "LDGE": 313,
    "MALL": 314,
    "MNR": 315,
    "STA": 316,
    "VLG": 317,
    "CORS": 318,
    "COMN": 319,
    "PRRD": 320,
    "EMS": 321
}

#Frankly, I don't know when we actually use this
Building_Codes = {
    "AFB": 1,
    "ARPT": 2,
    "APTS": 3,
    "ARC": 4,
    "BAZR": 5,
    "BLDG": 6,
    "BSPK": 7,
    "CTR": 8,
    "CLUB": 9,
    "CLTN": 10,
    "CMMN": 11,
    "CPLX": 12,
    "COND": 13,
    "CNCN": 14,
    "CORS": 15,
    "CTHS": 16,
    "CTS": 17,
    "CTYD": 18,
    "XING": 19,
    "XRDS": 20,
    "EDIF": 21,
    "ESP": 22,
    "EXCH": 23,
    "FEST": 24,
    "GALR": 25,
    "HALL": 26,
    "HOME": 27,
    "HOSP": 28,
    "HOTEL": 29,
    "HOUSE": 30,
    "INPK": 31,
    "INN": 32,
    "JCT": 33,
    "LNDG": 34,
    "LDGE": 35,
    "MALL": 36,
    "MNR": 37,
    "MKT": 38,
    "MERC": 39,
    "MTL": 40,
    "NAS": 41,
    "OFPK": 42,
    "OTLT": 43,
    "PARK": 44,
    "PAVL": 45,
    "PLNT": 46,
    "PLZ": 47,
    "PROM": 49,
    "QTRS": 50,
    "RES": 51,
    "7 CO": 52,
    "SC": 53,
    "SQ": 54,
    "STA": 55,
    "STES": 56,
    "TOWER": 57,
    "TWNH": 58,
    "TRPK": 59,
    "VLLA": 60,
    "VLG": 61,
    "VIVI": 62,
    "ESTS": 63,
    "COLL": 64,
    "COTT": 65,
    "PROJ": 66,
    "TORRE": 67
}