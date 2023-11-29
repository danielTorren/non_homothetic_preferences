"""Runs a single simulation to produce data which is saved

A module that use dictionary of data for the simulation run. The single shot simulztion is run
for a given initial set seed.



Created: 10/10/2022
"""
# imports
from package.resources.run import generate_data
from package.resources.utility import (
    createFolder, 
    save_object, 
    produce_name_datetime
)
from package.plotting_data import single_experiment_plot
import pyperclip
import time

def main(
    base_params
) -> str: 

    root = "single_experiment"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)

    Data = generate_data(base_params)  # run the simulation
    #print(Data.average_identity)

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    base_params = {
    "save_timeseries_data": 1, 
    "utility_function_state": "addilog_CES",#"addilog_CES",#"min_nested_CES", "nested_CES"
    "budget_inequality_state":0,
    "redistribution_state": 0,
    "heterogenous_preferences": 1,
    "carbon_tax_implementation": "flat",
    "dividend_progressiveness": 1,
    "compression_factor":10,
    "network_structure_seed": 8,
    "init_vals_seed": 14,
    "set_seed": 4,
    "seed_reps": 5,
    "carbon_price_duration": 3000,
    "burn_in_duration": 0,
    "N": 200,
    "M": 5,
    "network_density": 0.1,
    "prob_rewire": 0.1,
    "learning_error_scale": 0.01,
    "homophily": 0.95,
    "phi_lower": 0.01,
    "phi_upper": 0.01,
    "sector_substitutability": 5,
    "sector_substitutability_lower": 5,
    "sector_substitutability_upper": 5,
    "low_carbon_substitutability_lower":2,
    "low_carbon_substitutability_upper":10,
    "min_H_m_lower": 0,
    "min_H_m_upper": 0,
    "a_identity": 2,
    "b_identity": 2,
    "clipping_epsilon": 1e-5,
    "std_low_carbon_preference":0.01,
    "confirmation_bias":5,
    "init_carbon_price": 0,
    "carbon_price_increased": 0.0,
    "budget":10
    }
    
    print_simu = 1
    if print_simu:
        start_time = time.time()
    
    fileName = main(base_params=base_params)

    if print_simu:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    RUN_PLOT = 1

    if RUN_PLOT:
        single_experiment_plot.main(fileName = fileName)
