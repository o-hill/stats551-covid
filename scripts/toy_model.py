"""First Pass at PyMC3 Model"""
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as tt

df = pd.read_csv('../data/ncs-pop-deaths-and-binary-mandates.csv', index_col=0)
df = df.rename(columns={"Age Group": "Age_Group", "COVID-19 Deaths": "covid_19_deaths"})
test_df = df[df["Month"]==7]
sex = np.array(test_df["Sex"])
mandate = test_df["Mandate"]
age = test_df["Age_Group"]
covid_deaths = test_df["covid_19_deaths"]
population = test_df["Population"]/1000000 # makes the population in units of millions
n = len(test_df["Age_Group"].unique()) # should decrease by 1 after proper age filtering

with pm.Model() as model:

    # Parameters I have excluded for now.
#     alpha = pm.DiscreteUniform('alpha', lower=0, upper=n, shape=len(age_data.columns))
#     delta = pm.Normal('delta', mu=1, sigma=.1)

    # Prior
    beta = pm.Normal('beta', mu=1, sigma=.1)

    # Creating Mean Structure for Likelihood
    mandate = np.array(mandate).astype(theano.config.floatX)
    population = np.array(population).astype(theano.config.floatX)
    w = theano.shared(mandate, 'w')
    mean = pm.math.matrix_dot(w, beta)

    # Likelihood w/ Offset
    obs = pm.Poisson('y_obs', mu=population*tt.exp(mean), observed=covid_deaths)

    # Posterior Sampling
    trace = pm.sample(1000)
