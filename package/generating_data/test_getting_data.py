import pandas as pd
import numpy as np

N = 200

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
#print(norm_expenditure_deciles_vals)
#so essentially the expenditure that i assign to each member in a group will be 
indivdiual_expenditure =  10*norm_expenditure_deciles_vals/N#10 comes from the fact that there are 10 deciles

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
data_sector_carbon_intensity = []
for decile in deciles:
    data_dec_subset = df_sector_carbon_intensity[df_sector_carbon_intensity["eu_expenture_decile"] == decile]
    values = np.asarray(data_dec_subset["value"])
    data_sector_carbon_intensity.append([min(values),max(values)])


