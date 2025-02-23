import numpy as np
import pandas as pd 
from scipy.stats import norm
import scipy.interpolate as interp


def generate_new_samples(observations, num_samples=1000, seed = None, round = True):

    if seed is not None:
        np.random.seed(seed)

    # Step 1: Sort the observations
    sorted_observations = np.sort(observations)
    
    # Step 2: Compute the empirical CDF
    cdf_values = np.linspace(0, 1, len(sorted_observations))
    
    # Step 3: Create an interpolation function for the inverse CDF
    # This is the key to inverse transform sampling
    cdf_interpolation = interp.PchipInterpolator(cdf_values, sorted_observations)
    
    # Step 4: Generate uniform random samples in the range [0, 1]
    uniform_samples = np.random.uniform(0, 1, num_samples)
    
    # Step 5: Use the inverse CDF to generate new samples
    new_samples = cdf_interpolation(uniform_samples)
    if round:
        return np.round(new_samples, 3)
    else:
        return new_samples
    

def make_new_dataset(df, seed = None, round = True):
    df_new = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for _test_time in df.index:
        _obs = df.loc[_test_time]
        df_new.loc[_test_time] = generate_new_samples(_obs, num_samples=len(_obs), seed = seed, round = round)

    return df_new