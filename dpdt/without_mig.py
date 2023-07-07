import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d


__author__ = 'Chu-Chang Ku'
__all__ = ['find_pars', 'reform_pars_to_interp', 'regenerate_interp']


def ode_func(t, y, pars):
    dr = pars['r_die'] * np.exp((t - 2015) * pars['rt_die'])
    br = pars['r_bir'] * np.exp((t - 2015) * pars['rt_bir'])
    return (br - dr) * y


def ode_interp(t, y, pars):
    p = pars(t)
    dr = p['r_death']
    br = p['r_birth']
    return (br - dr) * y


def _fn(x, dbr, pop, ode):
    pars = dict(r_die=dbr['DR'], rt_die=dbr['dDR'], r_bir=x[0], rt_bir=x[1])

    ys = solve_ivp(ode, [1970, 2036], y0=np.ones(1), args=(pars,), dense_output=True)

    ratio = float(pop[pop.index == 2020].iloc[0]) / ys.sol(2020)
    ts = pop.index
    sim = ys.sol(ts).reshape(-1) * ratio
    return ((sim / pop - 1) ** 2).sum()


def extract_pars(x, dbr, pop, ode):
    pars = dict(r_die=dbr['DR'], rt_die=dbr['dDR'], r_bir=x[0], rt_bir=x[1], year0=2015)

    ys = solve_ivp(ode, [1970, 2036], y0=np.ones(1), args=(pars,), dense_output=True)

    ratio = float(pop[pop.index == 2020].iloc[0]) / ys.sol(2020)
    ts = np.linspace(1970, 2036, (2036 - 1970) * 2 + 1)

    pars['years'] = ts.tolist()
    pars['pop'] = (ys.sol(ts).reshape(-1) * ratio).tolist()

    return pars


def find_pars(dbr, pop):
    opt = minimize(_fn, np.array([dbr['BR'], dbr['dBR']]), args=(dbr, pop, ode_func, ),
                   method='L-BFGS-B', bounds=[(0, 0.1), (-0.1, 0.1)])

    return extract_pars(opt.x, dbr, pop, ode_func)


def reform_pars_to_interp(p0):
    p1 = {
        'Year': p0['years'],
        'N': p0['pop']
    }

    years = p1['Year']
    p1['RateAgeing'] = np.zeros_like(years)
    p1['RateMig'] = np.zeros_like(years)

    r_die, rt_die, t0 = p0['r_die'], p0['rt_die'], p0['year0']

    p1['RateDeath'] = np.array([r_die * np.exp((t - t0) * rt_die) for t in years])

    r_bir, rt_bir = p0['r_bir'], p0['rt_bir']
    p1['RateBirth'] = np.array([r_bir * np.exp((t - t0) * rt_bir) for t in years])

    return p1


def regenerate_interp(pars):
    years = pars['Year']

    pars_interp = {
        'N': interp1d(x=pars['Year'], y=pars['N']),
        'RateBirth': interp1d(x=pars['Year'], y=pars['RateBirth']),
        'RateDeath': interp1d(x=pars['Year'], y=pars['RateDeath'])
    }

    def fn_pars(t):
        return {
            'N': pars_interp['N'](t),
            'r_death': pars_interp['RateDeath'](t),
            'r_birth': pars_interp['RateBirth'](t)
        }

    ys = solve_ivp(ode_interp, [1970, 2036], y0=np.array([pars['N'][0]]), args=(fn_pars,), dense_output=True)
    return years, ys.sol(years).reshape(-1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    loc = 'Delhi'

    pop = pd.read_csv('../data/Projection.csv')
    pop = pop[pop.Sex == 'Total']
    pop = pop[pop.Location == loc]
    pop = pop.set_index('Year').Pop

    dbr = pd.read_csv('../data/DeathBirthRates.csv')
    dbr = dbr.loc[dbr.State == loc]
    dbr = {k: v[0] for k, v in dbr.reset_index().to_dict().items()}

    pars = find_pars(dbr, pop)

    years = pars['years']
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    axes[0].plot(years, pars['pop'])
    axes[0].scatter(pop.index, pop)

    pars = reform_pars_to_interp(pars)

    x, y = regenerate_interp(pars)

    axes[1].plot(x, y)
    axes[1].scatter(pop.index, pop)

    plt.show()

