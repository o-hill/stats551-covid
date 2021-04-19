"""File that will perform the analysis for all population interpolations."""
import argparse
import theano
import theano.tensor as tt

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az




def run_model(month=7, n_samples=1000, interp_type='ncs', binary=True, spike=0.9, hdi_prob=0.95, zero_inf=0.7):

    # preprocessing
    binary_str = 'binary' if binary else 'nonbinary'
    df = pd.read_csv('../data/' + interp_type +
                       '-pop-deaths-and-' + binary_str + '-mandates.csv', index_col=0)
    df = df.rename(columns={"Age Group": "Age_Group", "COVID-19 Deaths": "covid_19_deaths"})
    test_df = df[df["Month"]==month]
    sex = np.array(test_df["Sex"])
    mandates = test_df.iloc[:,-4:] # takes all of the 4 mandate columns that currently exist
    age = test_df["Age_Group"]
    covid_deaths = test_df["covid_19_deaths"]
    population = test_df["Population"]/1000000 # makes the population in units of millions
    n = len(test_df["Age_Group"].unique()) # should decrease by 1 after proper age filtering

    age_data = pd.get_dummies(test_df["Age_Group"]).drop("Under 1 year", axis=1)
    sex_data = pd.get_dummies(test_df["Sex"], drop_first=True)


    # run the model

    with pm.Model() as model:

        # spike and slab prior
        tau = pm.InverseGamma('tau', alpha=20, beta=20)
        xi = pm.Bernoulli('xi', p=spike, shape=len(mandates.columns))
        beta_mandates = pm.MvNormal('beta_mandate', mu=0, cov=tau*np.eye(len(mandates.columns)), shape=len(mandates.columns))
        
        # age prior
        mu_age_mean = np.linspace(-5,5,len(age_data.columns))
        V_age_mean = np.identity(len(age_data.columns))
        cov = pm.HalfNormal('cov', sigma=2)
        mu_age = pm.MvNormal('mu_age', mu=mu_age_mean, cov=np.identity(len(age_data.columns)), shape=(1, 10))
        beta_age = pm.MvNormal('beta_age', mu=mu_age, cov=cov*np.identity(10), shape=(1, 10))

        # sex prior
        mu_sex = pm.Normal('mu_sex', mu=0, sigma=1)
        sigma_sex = pm.HalfNormal('simga_sex', sigma=2)
        beta_sex = pm.Normal('beta_sex', mu=mu_sex, sigma=sigma_sex)

        # intercept prior
        mu_intercept = pm.Normal('mu_intercept', mu=0, sigma=1)
        sigma_intercept = pm.HalfNormal('simga_intercept', sigma=2)
        beta_intercept = pm.Normal('beta_intercept', mu=mu_intercept, sigma=sigma_intercept)

        # mean setup for likelihood
        mandates = np.array(mandates).astype(theano.config.floatX)
        population = np.array(population).astype(theano.config.floatX)
        sex = np.array(sex_data).astype(theano.config.floatX) 
        age = np.array(age_data).astype(theano.config.floatX)
        w_mandates = theano.shared(mandates, 'w_mandate')
        w_sex = theano.shared(sex, 'w_sex')
        w_age = theano.shared(age, 'w_age')
        mean = beta_intercept + pm.math.matrix_dot(w_mandates, xi*beta_mandates) \
                            + pm.math.matrix_dot(w_sex, beta_sex).T \
                            + pm.math.matrix_dot(w_age, beta_age.T).T

        # likelihood
        obs = pm.ZeroInflatedPoisson('y_obs', psi=zero_inf, theta=population*tt.exp(mean), observed=covid_deaths)
        # obs = pm.Normal('crap', mu=mean, sigma=3, observed=covid_deaths)

        # sample from posterior
        trace = pm.sample(n_samples, tune=n_samples, nuts={'target_accept': 0.98})

    # posterior hdis
    mandates = test_df.iloc[:,-4:]
    x = az.summary(trace, var_names=["beta_mandate"], hdi_prob=hdi_prob).iloc[:,2:4]
    x.index = mandates.columns
    x.to_csv('../images/posteriors/mandate_' + interp_type + '_' + binary_str + '_' + 'conf.csv')
    x = az.summary(trace, var_names=["beta_sex"], hdi_prob=hdi_prob).iloc[:,2:4]
    x.index = sex_data.columns
    x.to_csv('../images/posteriors/sex_' + interp_type + '_' + binary_str + '_' + 'conf.csv')
    x = az.summary(trace, var_names=["beta_age"], hdi_prob=hdi_prob).iloc[:,2:4]
    x.index = age_data.columns
    x.to_csv('../images/posteriors/age_' + interp_type + '_' + binary_str + '_' + 'conf.csv')
    x = az.summary(trace, var_names=["beta_intercept"], hdi_prob=hdi_prob).iloc[:,2:4]
    x.to_csv('../images/posteriors/intercept_' + interp_type + '_' + binary_str + '_' + 'conf.csv')

    # posterior distributions
    ax = az.plot_forest(trace, 'ridgeplot', var_names=["beta_intercept"], combined=True, hdi_prob=0.99999)
    ax[0].set_title(r'Posterior Distribution of $\beta_0$')
    plt.savefig('../images/posteriors/intercept_posteriors' + interp_type + '_' + binary_str + '.png')

    ax = az.plot_forest(trace, 'ridgeplot', var_names=["beta_age"], combined=True, hdi_prob=0.99999)
    ax[0].set_yticklabels(reversed(age_data.columns))
    ax[0].set_title(r'Posterior Distribution of $\beta_{age}$')
    plt.savefig('../images/posteriors/age_posteriors' + interp_type + '_' + binary_str + '.png')

    ax = az.plot_forest(trace, 'ridgeplot', var_names=["beta_sex"], combined=True, hdi_prob=0.99999)
    ax[0].set_yticklabels(reversed(sex_data.columns))
    ax[0].set_title(r'Posterior Distribution of $\beta_{sex}$')
    plt.savefig('../images/posteriors/sex_posteriors' + interp_type + '_' + binary_str + '.png')

    ax = az.plot_forest(trace, 'ridgeplot', var_names=["beta_mandate"], combined=True, hdi_prob=0.99999)
    ax[0].set_yticklabels(reversed(mandates.columns))
    ax[0].set_title(r'Posterior Distribution of $\beta_{mandate}$')
    plt.savefig('../images/posteriors/mandate_posteriors' + interp_type + '_' + binary_str + '.png')

    # ESS Plots
    ax = az.plot_ess(trace, var_names=["beta_intercept"])
    ax.set_title(r'$\beta_0$  ESS')
    plt.savefig('../images/ess/' + interp_type + '_' + binary_str + '_interceptESS.png')


    ax = az.plot_ess(trace, var_names=["beta_age"])
    ax[0,0].set_title(r'$\beta_{age[1-4]}$  ESS', fontsize=18)
    ax[0,1].set_title(r'$\beta_{age[5-14]}$  ESS', fontsize=18)
    ax[0,2].set_title(r'$\beta_{age[15-24]}$  ESS', fontsize=18)
    ax[1,0].set_title(r'$\beta_{age[25-34]}$  ESS', fontsize=18)
    ax[1,1].set_title(r'$\beta_{age[35-44]}$  ESS', fontsize=18)
    ax[1,2].set_title(r'$\beta_{age[45-54]}$  ESS', fontsize=18)
    ax[2,0].set_title(r'$\beta_{age[55-64]}$  ESS', fontsize=18)
    ax[2,1].set_title(r'$\beta_{age[65-74]}$  ESS', fontsize=18)
    ax[2,2].set_title(r'$\beta_{age[75-84]}$  ESS', fontsize=18)
    ax[3,0].set_title(r'$\beta_{age[85+]}$  ESS', fontsize=18)
    plt.savefig('../images/ess/' + interp_type + '_' + binary_str + '_ageESS.png')


    ax = az.plot_ess(trace, var_names=["beta_sex"])
    ax.set_title(r'$\beta_{sex}$  ESS')
    plt.savefig('../images/ess/' + interp_type + '_' + binary_str + '_sexESS.png')


    ax = az.plot_ess(trace, var_names=["beta_mandate"])
    ax[0].set_title(r'$\beta_{mandate[April]}$  ESS', fontsize=18)
    ax[1].set_title(r'$\beta_{mandate[May]}$  ESS', fontsize=18)
    ax[2].set_title(r'$\beta_{mandate[June]}$  ESS', fontsize=18)
    ax[3].set_title(r'$\beta_{mandate[July]}$  ESS', fontsize=18)
    plt.savefig('../images/ess/' + interp_type + '_' + binary_str + '_mandateESS.png')

    # posterior predictive checking
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
    az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model))
    plt.savefig('../images/posterior_predictive/' + interp_type + '_' + binary_str + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', required=False)
    parser.add_argument('--n', required=False)
    parser.add_argument('--interp', required=False)
    parser.add_argument('--binary', required=False)
    parser.add_argument('--spike', required=False)
    parser.add_argument('--hdi_prob', required=False)
    parser.add_argument('--zero_inf', required=False)
    args = parser.parse_args()
    month = args.month if args.month is not None else 7
    n_samples = args.n if args.n is not None else 1000
    binary = args.binary if args.binary is not None else True
    spike = args.spike if args.spike is not None and abs(args.spike) <= 1 else 0.9
    hdi_prob = args.hdi_prob if args.hdi_prob is not None else 0.95
    zero_inf = args.zero_inf if args.zero_inf is not None else 0.7
    if args.interp=="all":
        for interp_type in ['linear', 'cubic', 'ncs', 'polynomial', 'fakenodes']:
            run_model(month, n_samples, interp_type, binary, spike, hdi_prob, zero_inf)
    else:
        interp_type = args.interp if args.interp is not None else 'ncs'
        run_model(month, n_samples, interp_type, binary, spike, hdi_prob, zero_inf)
