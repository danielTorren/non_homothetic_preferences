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
import pandas as pd
import numpy as np

def gen_data():
    # Use ExcelFile to open the Excel spreadsheet
    xls = pd.ExcelFile("package/constants/figures_data.xlsx")

    df_expenditure = xls.parse("fig_1_a")
    df_carbon_footprint = xls.parse("fig_1_c")
    df_sector_expenditure_share = xls.parse("fig_2_a")
    df_sector_carbon_intensity = xls.parse("fig_2_c")

    #Get data population from other csv
    df_pop = pd.read_csv("package/constants/data_population_deciles.csv")

    #first generate the expenditures for each of the deciles, I normalise them such the total expenditure for the system is 1 each time step
    expenditure_deciles_vals = np.asarray(df_expenditure["value"])
    norm_expenditure_deciles_vals = expenditure_deciles_vals/np.sum(expenditure_deciles_vals)

    #next question is the fact that i need differnt carbon intensities for the different sectors? i could somehow nomarlise them and spread them between and 1 so that in teh aggregate they are the same?

    #get the prefereces for each sector
    deciles = np.arange(1,11)
    #print(deciles)

    #print(df_sector_expenditure_share)
    data_sector_preferences = []
    for decile in deciles:
        data_dec_subset = df_sector_expenditure_share[df_sector_expenditure_share["eu_expenture_decile"] == decile]
        values = np.asarray(data_dec_subset["value"])/100
        data_sector_preferences.append(values)

    #how to represent the carbon intensities of the low and high: use the lowest and highest options for eahc sectors
    data_sector_low_carbon_intensity = []#THE UNITS ARE in kgC02equiv/euro
    data_sector_high_carbon_intensity = []#THE UNITS ARE in kgC02equiv/euro

    sectors = ["services", "mobility", "housing", "food", "goods"]#NEED TO LOAD THIS IN

    for sector in sectors:
        data_dec_subset = df_sector_carbon_intensity[df_sector_carbon_intensity["sector_aggregate"] == sector]
        values = np.asarray(data_dec_subset["value"])
        data_sector_low_carbon_intensity.append(min(values))
        data_sector_high_carbon_intensity.append(max(values))

    data_start = {
        "deciles": deciles,
        "norm_expenditure_deciles_vals": np.asarray(norm_expenditure_deciles_vals),
        "data_sector_preferences": np.asarray(data_sector_preferences),
        "data_sector_low_carbon_intensity": np.asarray(data_sector_low_carbon_intensity),
        "data_sector_high_carbon_intensity": np.asarray(data_sector_high_carbon_intensity)
    }

    return data_start



def main(
    base_params
) -> str: 

    root = "single_experiment"
    fileName = produce_name_datetime(root)
    pyperclip.copy(fileName)
    print("fileName:", fileName)

    data_start =  gen_data()

    base_params.update(data_start)

    Data = generate_data(base_params)  # run the simulation
    #print(Data.average_identity)

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName

if __name__ == '__main__':
    base_params = {
    "save_timeseries_data": 1, 
    "imperfect_learning_state": 1,
    "vary_seed_imperfect_learning_state_or_initial_preferences_state": 1,
    "heterogenous_intrasector_preferences": 1,
    "heterogenous_intrasector_substitutabilities": 1,
    "heterogenous_carbon_price": 0,
    "heterogenous_phi":0,
    "budget_homophily":1,
    "utility_function_state": "nested_CES",#"addilog_CES",#"min_nested_CES", "nested_CES"
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
    "network_density": 0.1,
    "prob_rewire": 0.1,
    "homophily": 0.95,
    "phi_lower": 0.01,
    "phi_upper": 0.01,
    "sector_substitutability_lower": 1.5,
    "sector_substitutability_upper": 2,
    "low_carbon_substitutability_lower":1.5,
    "low_carbon_substitutability_upper":2,
    "min_H_m_lower": 0,
    "min_H_m_upper": 0,
    "a_identity": 3,
    "b_identity": 2,
    "clipping_epsilon": 1e-5,
    "clipping_epsilon_init_preference": 1e-5,
    "std_low_carbon_preference":0.01,
    "std_learning_error": 0.01,
    "confirmation_bias":5,
    "init_carbon_price": 0,
    "carbon_price_increased": 0.0
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
