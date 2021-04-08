"""
For each company in the DOW, a list of keywords to filter news and make sure they are relevant.
"""
COMPANIES_KEYWORDS = {'AAPL': ['aap', 'apple', 'phone', 'mac', 'microsoft'],
                      'XOM': ['xom', 'exxon', 'mobil', 'petrol', 'gas', 'energy'],
                      'KO': ['ko', 'coca', 'cola', 'pepsi', 'soda'],
                      'INTC': ['intc', 'intel', 'chip', 'cpu', 'computer'],
                      'WMT': ['wmt', 'walmart', 'food'],
                      'MSFT': ['msft', 'microsoft', 'gates', 'apple', 'computer'],
                      'IBM': ['ibm', 'business', 'machine'],
                      'CVX': ['cvx', 'chevron', 'petrol', 'gas', 'energy'],
                      'JNJ': ['jnj', 'johnson', 'health', 'medi', 'pharma'],
                      'PG': ['pg', 'procter', 'gamble', 'health', 'care'],
                      'PFE': ['pfe', 'pfizer', 'health', 'medi', 'pharma'],
                      'VZ': ['vz', 'verizon', 'comm'],
                      'BA': ['ba', 'boeing', 'plane', 'air'],
                      'MRK': ['mrk', 'merck', 'health', 'medi', 'pharma'],
                      'CSCO': ['csco', 'cisco', 'system', 'techn'],
                      'HD': ['hd', 'home', 'depot', 'construction'],
                      'MCD': ['mcd', 'donald', 'food', 'burger'],
                      'MMM': ['mmm', '3m'],
                      'GE': ['ge', 'general', 'electric', 'tech', 'energy'],
                      'NKE': ['nke', 'nike', 'sport', 'wear'],
                      'CAT': ['cat', 'caterpillar', 'construction'],
                      'V': ['visa', 'bank', 'card', 'pay'],
                      'JPM': ['jpm', 'morgan', 'chase', 'bank'],
                      'AXP': ['axp', 'american', 'express', 'bank', 'card', 'pay'],
                      'GS': ['gs', 'goldman', 'sachs', 'bank'],
                      'UNH': ['unh', 'united', 'health', 'insurance'],
                      'TRV': ['trv', 'travel', 'insurance'],
                      # 'UTX': ['utx', 'united', 'tech'],
                      }

"""
List of companies used in this project (subset of Dow Jones Industrial Average).
"""
DJIA = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'MSFT', 'IBM', 'CVX', 'JNJ', 'PG', 'PFE', 'VZ', 'BA', 'MRK',
        'CSCO', 'HD', 'MCD', 'MMM', 'GE', 'NKE', 'CAT', 'JPM', 'AXP', 'GS', 'UNH', 'V', 'TRV']

"""
Leverages on Trading 212 CFD.
"""
LEVERAGES = {'AAPL': 1, 'XOM': 1, 'KO': 1, 'INTC': 1, 'WMT': 1, 'MSFT': 1, 'IBM': 1, 'CVX': 1, 'JNJ': 1,
             'PG': 1, 'PFE': 1, 'VZ': 1, 'BA': 1, 'MRK': 1, 'CSCO': 1, 'HD': 1, 'MCD': 1, 'MMM': 1,
             'GE': 1, 'NKE': 1, 'CAT': 1, 'V': 1, 'JPM': 1, 'AXP': 1, 'GS': 1, 'UNH': 1, 'TRV': 1}

"""
Subset of DJIA companies identifies as performers.
"""
DJIA_PERFORMERS = ['AAPL', 'XOM', 'KO', 'INTC', 'WMT', 'MSFT', 'CVX', 'MMM', 'V', 'GS']

"""
Full list of Dow Jones companies.
"""
DOW = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD',
       'MMM', 'MRK', 'MSFT', 'NLE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW']

"""
List of main Poloniex currency pairs.
"""

PAIRS = ["USDC_ATOM", "USDC_BTC", "USDC_DASH", "USDC_DOGE", "USDC_EOS", "USDC_ETC", "USDC_ETH",
         "USDC_LTC", "USDC_STR", "USDC_TRX", "USDC_USDT", "USDC_XMR", "USDC_XRP", "USDC_ZEC"]

"""
Subset of CAC 40 components
"""

CAC40 = ["BNP.PA", "FP.PA", "EN.PA", "CA.PA", "SAN.PA", "ACA.PA", "VIE.PA", "DG.PA", "ORA.PA", "WLN.PA", "VIV.PA",
         "GLE.PA", "HO.PA", "ML.PA", "RI.PA", "CAP.PA", "AI.PA", "SW.PA", "SGO.PA", "OR.PA", "ATO.PA", "KER.PA",
         "SU.PA", "AC.PA", "MC.PA", "UG.PA", "ENGI.PA", "AIR.PA", "BN.PA", "LR.PA"]
