import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from package.resources.utility import get_cmap_colours
from package.resources.utility import check_other_folder
# modules
class Individual_test:

    """
    Class to represent individuals with identities, preferences and consumption

    """

    def __init__(
        self,
        individual_params
    ):
 
        self.low_carbon_preferences = individual_params["low_carbon_preferences"] 
        self.carbon_price = individual_params["carbon_price"]
        self.M = individual_params["M"]

        self.low_carbon_substitutability_array = individual_params["low_carbon_substitutability"]
        self.prices_low_carbon = individual_params["prices_low_carbon"]
        self.prices_high_carbon = individual_params["prices_high_carbon"]
        self.utility_function_state = individual_params["utility_function_state"]
        self.sector_substitutability_m = individual_params["sector_substitutability_m"]
        self.carbon_intensity_high_carbon = individual_params["carbon_intensity_high_carbon"]
        self.carbon_intensity_low_carbon = individual_params["carbon_intensity_low_carbon"]

        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price


        if self.utility_function_state == "nested_CES":
            self.min_a = np.asarray(individual_params["min_a"])
            self.max_a = np.asarray(individual_params["max_a"])
            if sum(self.min_a) !=1:
                Exception("Invalid min_a")
            if sum(self.max_a) !=1:
                Exception("Invalid max_a")            
            self.min_B = individual_params["min_B"]
            self.max_B = individual_params["max_B"]
        elif self.utility_function_state == "addilog_CES":
            self.sector_preferences =  np.asarray([1/self.M]*self.M)
            self.sector_substitutability_base = self.sector_substitutability_m[0]

    #####################################################################################
    #NESTED CES

    def calc_sector_preferences(self):
        a_m = self.min_a + ((self.max_a-self.min_a)/(self.max_B - self.min_B))*(self.instant_budget-self.min_B)
        return a_m


    def calc_consumption_quantities_nested_CES(self):
        H_m = self.instant_budget*(self.chi_m**self.sector_substitutability_m)/self.Z
        L_m = H_m*self.Omega_m
        
        return H_m,L_m
    
    def calc_utility_nested_CES(self):
        psuedo_utility = (self.low_carbon_preferences*(self.L_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1 - self.low_carbon_preferences)*(self.H_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))

        if self.M == 1:
            U = psuedo_utility
        else:
            interal_components_utility = self.sector_preferences*(psuedo_utility**((self.sector_substitutability_m -1)/self.sector_substitutability_m))
            sum_utility = sum(interal_components_utility)
            U = sum(sum_utility**(self.sector_substitutability_m/(self.sector_substitutability_m-1)))
        return U,psuedo_utility
    
    #############################################
    #ADDILOG
    def func_jacobian(self, x, chi_base):
        """Derivative of function with respect to H_q or base good"""
        term_1 = self.sector_substitutability_m*(self.chi_m**(self.sector_substitutability_m-1))*(chi_base**(-self.sector_substitutability_m))
        term_2 = self.prices_high_carbon_instant + self.prices_low_carbon*self.Omega_m
        term_3 = x**(self.sector_substitutability_m/self.sector_substitutability_base)
        jacobian = term_1*term_2*term_3

        return jacobian
    
    def func_jacobian_alt(self, x, chi_base):
        """Derivative of function with respect to H_q or base good"""
        term_1 = self.sector_substitutability_m*(self.chi_m**(self.sector_substitutability_m-1))*(chi_base**(-self.sector_substitutability_m))
        term_2 = self.prices_high_carbon_instant + self.prices_low_carbon*self.Omega_m
        term_3 = x**(self.sector_substitutability_m/self.sector_substitutability_base)
        jacobian = sum(term_1*term_2*term_3)

        return jacobian

    def func_to_solve(self, x, chi_base):
        """In this function subscript 0 refers to q in equuation 121, the final one in derivation"""
        term_1 = (self.chi_m/chi_base)**self.sector_substitutability_m
        term_2 = self.prices_high_carbon_instant + self.prices_low_carbon*self.Omega_m
        term_3 = x**(self.sector_substitutability_m/self.sector_substitutability_base)
        f = np.sum(term_1*term_2*term_3) - self.instant_budget
        return f
    
    def calc_H_addilog_CES(self, H_0, chi_base):
        #DONE
        H = ((chi_base/self.chi_m)**(self.sector_substitutability_m))*(H_0**(self.sector_substitutability_m/self.sector_substitutability_base))
        return H
    
    def calc_consumption_quantities_addilog_CES(self):
        chi_base = self.chi_m[0]
        root = least_squares(self.func_to_solve, x0=self.init_vals_H, jac=self.func_jacobian, bounds = (0, np.inf), args = ([chi_base]))
        H_0 = root["x"][0]
        H_m = self.calc_H_addilog_CES(H_0, chi_base)
        L_m = self.Omega_m*H_m

        return H_m,L_m
    
    def calc_utility_addilog_CES(self):
        psuedo_utility = (self.low_carbon_preferences*(self.L_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)) + (1 - self.low_carbon_preferences)*(self.H_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array)))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))

        if self.M == 1:
            U = psuedo_utility
        else:
            ratio_term = (self.sector_substitutability_m*(self.sector_substitutability_base-1))/(self.sector_substitutability_base*(self.sector_substitutability_m-1))
            interal_components_utility = self.sector_preferences*ratio_term*(psuedo_utility**((self.sector_substitutability_m -1)/self.sector_substitutability_m))
            sum_utility = sum(interal_components_utility)
            U = sum_utility**(self.sector_substitutability_base/(self.sector_substitutability_base-1))
        return U, psuedo_utility
    

    ###########################################################################
    #SHARED   
    
    def calc_Omega_m(self):       
        term_1 = (self.prices_high_carbon_instant*self.low_carbon_preferences)
        term_2 = (self.prices_low_carbon*(1- self.low_carbon_preferences))
        omega_vector = (term_1/term_2)**(self.low_carbon_substitutability_array)
        return omega_vector
    
    def calc_n_tilde_m(self):
        n_tilde_m = (self.low_carbon_preferences*(self.Omega_m**((self.low_carbon_substitutability_array-1)/self.low_carbon_substitutability_array))+(1-self.low_carbon_preferences))**(self.low_carbon_substitutability_array/(self.low_carbon_substitutability_array-1))
        return n_tilde_m
    
    def calc_chi_m(self):
        chi_m = (self.sector_preferences*(self.n_tilde_m**((self.sector_substitutability_m-1)/self.sector_substitutability_m)))/self.prices_high_carbon_instant
        return chi_m
    
    def calc_Z(self):
        common_vector_denominator = self.Omega_m*self.prices_low_carbon + self.prices_high_carbon_instant
        chi_pow = self.chi_m**self.sector_substitutability_m
        
        Z = np.matmul(chi_pow, common_vector_denominator)     
        return Z   

    ########################################################################################################### 
        

    def calc_total_emissions(self):      
        emissions_by_sector = self.H_m*self.carbon_intensity_high_carbon + self.L_m*self.carbon_intensity_low_carbon
        return  emissions_by_sector,sum(emissions_by_sector) 

    def update_consumption(self):
        self.Omega_m = self.calc_Omega_m()
        self.n_tilde_m = self.calc_n_tilde_m()
        self.chi_m = self.calc_chi_m()
        #calculate consumption
        if self.utility_function_state == "nested_CES":
            self.Z = self.calc_Z()
            self.H_m, self.L_m = self.calc_consumption_quantities_nested_CES()
            self.utility,self.pseudo_utility = self.calc_utility_nested_CES()
        elif self.utility_function_state == "addilog_CES":
            self.H_m, self.L_m = self.calc_consumption_quantities_addilog_CES()
            self.init_vals_H = self.H_m[0]
            self.utility,self.pseudo_utility = self.calc_utility_addilog_CES()

    def calc_stuff(self, instant_budget):

        self.instant_budget = instant_budget

        if self.utility_function_state == "nested_CES":
            self.sector_preferences = self.calc_sector_preferences()
            #print("sum self.sector_preferences", self.sector_preferences, sum(self.sector_preferences))
        elif self.utility_function_state == "addilog_CES":
            self.init_vals_H = (self.instant_budget/self.M)*(self.prices_low_carbon/self.prices_high_carbon_instant) #assume initially its uniformaly spread    

        #update_consumption
        self.update_consumption()

        #calc_emissions
        self.flow_carbon_emissions_sectors, self.flow_carbon_emissions = self.calc_total_emissions()

        return self.H_m, self.L_m, self.flow_carbon_emissions, self.utility, self.flow_carbon_emissions_sectors

if __name__ == '__main__' :
    min_B = 1 
    max_B = 10
    budget_list = np.linspace(min_B,max_B,100)
    #"""
    params = {
        "utility_function_state": "nested_CES",#"addilog_CES",
        "M": 3,
        "low_carbon_preferences": np.asarray([0.9,0.9,0.1]),
        "carbon_price":  np.asarray([0.0,0.0,0.0]),
        "low_carbon_substitutability": np.asarray([10,2,1.1]),
        "sector_substitutability_m": np.asarray([5,5,5]),#must all be the same if nested_CES
        "prices_high_carbon": np.asarray([1,1,1]),
        "prices_low_carbon": np.asarray([1,1,1]),
        "min_a": np.asarray([0.7,0.29,0.01]),#have to sum to 1, represent energy, housing, aviation
        "max_a": np.asarray([0.1,0.1,0.8]),#have to sum to 1,represent energy, housing, aviation
        "min_B": min_B,
        "max_B": max_B,
        "carbon_intensity_high_carbon": np.asarray([1,1,5]),#has no affect on preferences, just emissions
        "carbon_intensity_low_carbon": np.asarray([0,0,5]),#has no affect on preferences, just emissions
    }   
    #"""
    """
    params = {
        "utility_function_state": "nested_CES",#"addilog_CES",
        "M": 2,
        "low_carbon_preferences": np.asarray([0.2,0.7]),
        "carbon_price":  np.asarray([0.0,0.0]),
        "low_carbon_substitutability": np.asarray([5,5]),
        "sector_substitutability_m": np.asarray([2,2]),#must all be the same if nested_CES
        "prices_high_carbon": np.asarray([1,1]),
        "prices_low_carbon": np.asarray([1,1]),
        "min_a": np.asarray([0.2,0.8]),#have to sum to 1
        "max_a": np.asarray([0.6,0.4]),#have to sum to 1
        "min_B": min_B,
        "max_B": max_B
    }   
    """


    test_subject = Individual_test(params)

    data_H = []
    data_L = []
    data_E = []
    data_U = []
    data_E_m = []
    for i in budget_list:
        
        data_point_H, data_point_L, data_point_E, data_point_U,  data_point_E_m= test_subject.calc_stuff(i)
        data_H.append(data_point_H)
        data_L.append(data_point_L)
        data_E.append(data_point_E)
        data_U.append(data_point_U)
        data_E_m.append(data_point_E_m)

    data_array_H = np.asarray(data_H)
    data_array_L = np.asarray(data_L)
    data_array_E = np.asarray(data_E)
    data_array_U = np.asarray(data_U)
    data_array_E_m = np.asarray(data_E_m)

    #print("data h",data_array_H,data_array_H.shape)

    cmap = get_cmap_colours(params["M"])

    data_array_H_t = data_array_H.T
    data_array_L_t = data_array_L.T
    fig, ax = plt.subplots(figsize=(10,6))
    if params["utility_function_state"] ==  "nested_CES":
        for i , a_min in enumerate(params["min_a"]):
            ax.plot(budget_list, data_array_H_t[i], label = "$H,a_{min,%s} = %s,a_{max,%s} = %s$" % (i+1,a_min,i+1,params["max_a"][i]),color = cmap(i),linestyle='--')
            ax.plot(budget_list, data_array_L_t[i], label = "$L,a_{min,%s} = %s,a_{max,%s} = %s$" % (i+1,a_min,i+1,params["max_a"][i]),color = cmap(i),linestyle='-')
    elif params["utility_function_state"] ==  "addilog_CES": 
        for i , nu_m in enumerate(params["sector_substitutability_m"]):
            ax.plot(budget_list, data_array_H_t[i], label = "$H,\\nu_{%s}$ = %s" % (i+1,nu_m),color = cmap(i),linestyle='--')
            ax.plot(budget_list, data_array_L_t[i], label = "$L,\\nu_{%s}$ = %s" % (i+1,nu_m),color = cmap(i),linestyle='-')
    ax.legend()
    ax.set_xlabel("Budget, B")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Quantity")#row
    check_other_folder()
    fig.tight_layout()
    plotName = "results/Other"
    f = plotName + "/quantity_evo_lux_basicy_a_func_B" 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(budget_list, data_array_U)
    ax.set_xlabel("Budget, B")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Utility")#row
    fig.tight_layout()
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/utility_evo_lux_basic_a_func_B" 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(budget_list, data_array_E)
    ax.set_xlabel("Budget, B")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Emissions flow")#row
    fig.tight_layout()
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/emissions_evo_lux_basic_a_func_B" 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

    fig, ax = plt.subplots(figsize=(10,6))
    data_array_E_m_T = data_array_E_m.T
    for i in range(params["M"]):
        ax.plot(budget_list, data_array_E_m_T[i], label = "m = %s" % (i+1),color = cmap(i))
    ax.legend()
    ax.set_xlabel("Budget, B")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Sectoral emissions flow")#row
    fig.tight_layout()
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "sector_emissions_evo_lux_basic_a_func_B" 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

    plt.show()

