import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle as pkl
from dpdt.with_mig import find_pars, reform_pars_to_interp, regenerate_interp
import os

__author__ = 'Chu-Chang Ku'

pop = pd.read_csv('../data/Projection.csv')
ps = dict()
for k, v in pop.loc[pop.Sex == 'Total'].groupby('Location'):
    ps[k] = v.set_index("Year").Pop

dbr = pd.read_csv('../data/DeathBirthRates.csv')
dbs = dict()
for k, col in dbr.groupby('State'):
    dbs[k] = {i: float(v) for i, v in dict(col).items() if v.dtype == float}


locations = set.intersection(set(ps.keys()), set(dbs.keys()))


os.makedirs('../pars/with_mig/', exist_ok=True)
os.makedirs('../figs/with_mig/', exist_ok=True)


for loc in locations:
    lab_loc = loc.replace(' ', '_').replace('&', 'and')

    print(loc, '->', lab_loc)

    dbr = dbs[loc]
    pop = ps[loc]

    pars = find_pars(dbr, pop)
    years = pars['years']
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))

    axes[0].plot(years, pars['pop'])
    axes[0].scatter(pop.index, pop)
    axes[0].set_title('With function form')

    pars = reform_pars_to_interp(pars)

    x, y = regenerate_interp(pars)

    axes[1].plot(x, y)
    axes[1].scatter(pop.index, pop)
    axes[1].set_title('With interp form')

    plt.savefig(f'../figs/with_mig/g_fitness_{lab_loc}.png')
    plt.show()


    with open(f'../pars/with_mig/pars_{lab_loc}.json', 'w') as f:
        pars_json = {k: list(v) for k, v in pars.items()}
        json.dump(pars_json, f)

    with open(f'../pars/with_mig/pars_{lab_loc}.pkl', 'wb') as f:
        pars['dimnames'] = dict(Year=[str(yr) for yr in pars['Year']])
        pkl.dump(pars, f)
