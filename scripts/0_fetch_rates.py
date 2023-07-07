from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


# Death rate data
page = requests.get('https://www.niti.gov.in/content/death-rate')
soup = BeautifulSoup(page.text)

table = soup.find("table")

drs = []
for row in table.findAll("tr"):
    cols = row.findAll("td")
    if len(cols) > 3:
        cols = [ele.text.strip() for ele in cols]
        cols = [col for col in cols if len(col) > 0]

        dr = np.array(cols[1:]).astype(float) / 1000

        drs.append({
            'State': cols[0],
            'DR_2014': dr[2],
            'DR_2015': dr[1],
            'DR_2016': dr[0],
            'DR': np.exp(np.log(dr).mean()),
            'dDR': - np.diff(np.log(dr)).mean()
        })


drs = pd.DataFrame(drs)
print(drs)


# Birth rate data
page = requests.get('https://www.niti.gov.in/content/birth-rate')
soup = BeautifulSoup(page.text)

table = soup.find("table")

brs = []
for row in table.findAll("tr"):
    cols = row.findAll("td")
    if len(cols) > 3:
        cols = [ele.text.strip() for ele in cols]
        cols = [col for col in cols if len(col) > 0]

        br = np.array(cols[1:]).astype(float) / 1000

        brs.append({
            'State': cols[0],
            'BR_2014': br[2],
            'BR_2015': br[1],
            'BR_2016': br[0],
            'BR': np.exp(np.log(br).mean()),
            'dBR': - np.diff(np.log(br)).mean()
        })
        ed = cols

brs = pd.DataFrame(brs)
print(brs)


# Output
dbr = pd.merge(drs, brs)
dbr.to_csv('../data/DeathBirthRates.csv', index=False)
