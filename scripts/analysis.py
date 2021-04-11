"""File that will perform the analysis for all population interpolations."""
import pandas as pd
# import pymc3 as pm


def read_data(interp_type='linear', binary=True):
    """Reads the appropriate data set for analysis.

        Parameters:
            interp_type (str): interpolation type wanted
            binary (bool): binary status of mandate data

        Returns:
            A dataset with population interpolated and
            mandate data included as specific by the parameters.

    """

    binary_str = 'binary' if binary else 'nonbinary'
    return pd.read_csv('../../data/' + interp_type +
                       '-pop-deaths-and-' + binary_str + '-mandates.csv')
