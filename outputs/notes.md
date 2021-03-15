# Experiments notes

## 1. T0 = '2010-01-01' / T1 = '2019-01-01' / T2 = '2020-01-01' / AVEC FILTRE / H=10
PORTFO - Positive days: 56.55%. Average daily return: 0.0617%.
MARKET - Positive days: 56.55%. Average daily return: 0.0378%.

## 2. T0 = '2010-01-01' / T1 = '2019-01-01' / T2 = '2020-01-01' / SANS FILTRE / H=10
PORTFO - Positive days: 56.55%. Average daily return: 0.0396%.
MARKET - Positive days: 56.55%. Average daily return: 0.0378%.

## 3. T0 = '2010-01-01' / T1 = '2019-01-01' / T2 = '2020-01-01' / AVEC FILTRE / H=20
PORTFO - Positive days: 54.86%. Average daily return: 0.0687%.
MARKET - Positive days: 56.03%. Average daily return: 0.0443%.

## 4. T0 = '2010-01-01' / T1 = '2019-01-01' / T2 = '2020-01-01' / AVEC FILTRE / H=7
PORTFO - Positive days: 56.67%. Average daily return: 0.0541%.
MARKET - Positive days: 56.67%. Average daily return: 0.0397%.

## 5. Un lstm par asset, same as 1.
PORTFO - Positive days: 55.06%. Average daily return: 0.0989%.
MARKET - Positive days: 56.55%. Average daily return: 0.0378%.

## 6. Same as 5, without entropy in loss
PORTFO - Positive days: 52.81%. Average daily return: 0.0033%.
MARKET - Positive days: 56.55%. Average daily return: 0.0378%.

## 7. Same as 5, entropy lambda set to 2e-4
PORTFO - Positive days: 55.43%. Average daily return: 0.1037%.
MARKET - Positive days: 56.55%. Average daily return: 0.0378%.

## 8. Same as 5, entropy lambda set to 3e-4
PORTFO - Positive days: 56.55%. Average daily return: 0.1124%.
MARKET - Positive days: 56.55%. Average daily return: 0.0378%.

## 9. Same as 5, entropy lambda set to 4e-4
PORTFO - Positive days: 56.18%. Average daily return: 0.0680%.
MARKET - Positive days: 56.55%. Average daily return: 0.0378%.

## 10. Same as 8 but open-to-close
PORTFO - Positive days: 56.84%. Average daily return: 0.1056%.
MARKET - Positive days: 58.12%. Average daily return: 0.1204%.

## 11. Same as 10,  entropy lambda set to 2e-4
