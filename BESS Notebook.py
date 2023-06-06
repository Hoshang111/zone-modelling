# Databricks notebook source
from IPython.display import display
from datetime import datetime
import ipywidgets as widgets
from scenario_class import Scenario
from databricks.sdk.runtime import spark
import pandas as pd
import numpy as np
import pvlib as pl

scenario = Scenario()

# COMMAND ----------

optimisation = scenario.optimisation.df
assumption = scenario.assumption.df.set_index('period_end').sort_index()

PC_VSC_flow = optimisation['PC_VSC_flow']
PC_solar_production = optimisation['PC_solar_production']
operational_capacity = assumption.loc[assumption['identifier'] == 'PC_battery_operational_capacity']

BESS_df = pd.DataFrame()
num_inverters = 1554

BESS_df['Demand/Inverter'] = PC_VSC_flow.div(num_inverters)
BESS_df['PV Production/Inverter'] = PC_solar_production.div(num_inverters)
BESS_df['BESS Size'] = 20

# bess_index = []
# oc_index = operational_capacity.index
# for index in BESS_df.index:
#     i = index.strftime("%Y-%m-%d")
#     if i in oc_index:
#         BESS_df['BESS Size'].loc[index] = operational_capacity['numeric_value'].loc[i]

BESS_df['BESS Size'] = BESS_df['BESS Size']
BESS_df['Net Flow of Energy'] = BESS_df['PV Production/Inverter'] - BESS_df['Demand/Inverter']

charge_conditions = [BESS_df['Net Flow of Energy'] >= 0, BESS_df['Net Flow of Energy'] < 0]
charge_choices = ['Charge', 'Discharge']
BESS_df['Charge/Discharge'] = np.select(charge_conditions, charge_choices)

bess_states = []
spill = []
bess_state = 20
bess_effiency = 0.95
inverter_effiency = 0.95
bess_power = 2
current_spill = 0
for row in BESS_df.iterrows():
    net_flow = row[1][3]
    if bess_state + net_flow >= 0 and bess_state + net_flow <= bess_size:
        if net_flow > bess_power:
            current_spill += net_flow - bess_power
            bess_state += bess_power*bess_effiency
        else:
            bess_state += net_flow*bess_effiency
        bess_states.append(bess_state)
    spill.append(current_spill)

BESS_df['BESS State of Charge'] = b_size
BESS_df['Spill'] = spill
BESS_df

# COMMAND ----------

BESS_df.tail(30)

# COMMAND ----------

BESS_df.plot(y='Spill', use_index=True)

# COMMAND ----------

solcast_query = "select * from sandbox.pv_data.solcast__18_201125_133_408077"
df_solcast = spark.sql(solcast_query).toPandas()
optimisation = scenario.optimisation.df

num_rows = len(optimisation.index)
df_solcast = df_solcast.iloc[:num_rows]
df_solcast = df_solcast.drop(columns=['PeriodEnd', 'PeriodStart'])
optimisation = optimisation.reset_index(drop=True)
df = optimisation.join(df_solcast, how='outer')
df

# COMMAND ----------

# Matts attempt (work in progress)

"""
SCENARIO: AA_POS5_R01_SBP01_S1
Assume 1554 

PCU:
SMA SC-NXT 3520
AC Power @35degC (MVA)          = 3.520
# AC Power @50degC (MVA)        = ?
# MPPT Range Low @35degC (V)    = 962
# MPPT Range Low @35degC (V)    = 1325
DC-DC Power @35degC (MW)        = 9.6 (6x DC-DC converters per PCU, each with 1.6MW power)
Max Bess per DC-DC (4hr rate)   = 1
Max Bess per DC-DC (5hr rate)   = 1.5

BESS:
CATL EnerC+
Usable Energy (MWh)             = 3.82862

Zone Configuration:
1x PCU

6x BESS (total energy = 6x3.82862 = 22.97172) (4hr Rate)
OR
9x BESS (total energy = 9x3.82862 = 34.45758) (5hr Rate)

PV = JA Solar 144HC N-Type
28modules x 692strings (monofacial) (approximately 11.82687MWp)
OR
28modules x 653 (bifacial) (approximately 11.15743MWp)

"""

# Rules
"""
1. meet output power demand
2. 
"""

df = pd.read_csv('/dbfs/FileStore/pvlib_ui/JAM72D40_df610GB.csv')



# COMMAND ----------

df_mpp = pl.pvsystem.max_power_point(photocurrent=df['IL'], saturation_current=df['I0'], resistance_series=df['Rs'], resistance_shunt=df['Rsh'], nNsVth=df['nNsVth'], d2mutau=0, NsVbi=np.inf, method='brentq')

# COMMAND ----------

df_working = df.merge(df_mpp, left_index=True, right_index=True).copy()

# COMMAND ----------

t_delta = 1/12                          # Time delta between rows is 5min or 1/12hrs
demand_ac = 3150/1554                   # Assume static demand of VSC power divided by number of inverters

# Subarray Details
modules = 28
strings = 692

# BESS details
bess_cap_mwh = 34.45                 # BESS capacity in MWh

# DC-DC details
dcdc_p_mw = 9.6                         # DC-DC converter power

# Inverter details
inv_p_mw = 3.520

# Initialise dataframes
df_working['pv_vmp_v'] = modules * df_working['v_mp']
df_working['pv_imp_a'] = strings * df_working['i_mp']
df_working['pv_pmp_mw'] = df_working['p_mp'] * modules * strings/1000000

df_working['dcdc_p_mw'] = 0
df_working['dcdc_p_lmt_chrg_mw'] = -dcdc_p_mw
df_working['dcdc_p_lmt_dschrg_mw'] = dcdc_p_mw

df_working['bess_p_mw'] = 0
df_working['bess_soc_start_mwh'] = bess_cap_mwh
df_working['bess_p_lmt_mw'] = 0

df_working['bess_p_lmt_chrg_mw'] = 0
df_working['bess_p_lmt_dschrg_mw'] = 0
df_working['bess_cap_lmt_chrg_mwh'] = 0
df_working['bess_cap_lmt_dschrg_mwh'] = 0

df_working['inv_p_dc_mw'] = np.nan
df_working['inv_p_dc_lmt_mw'] = 3.520

df_working['p_net_dc'] = 0
df_working['bess_soc_end_mwh'] = bess_cap_mwh
df_working['spill_mw'] = 0

# Define function that controls the BESS charging
def bess_chrg(df, idx, p, demand_dc):
    """
    Function for controlling BESS charging
    """
    # Determine bess charge power limit
    # Charging power is negative, therefore this will be the max of:
    #   Available charge power, p
    #   Bess p limit charge
    #   DC-DC Converter p limit charge
    p_limit = max(-p, df.at[idx, 'dcdc_p_lmt_chrg_mw'], df.at[idx, 'bess_p_lmt_chrg_mw'])

    df.at[idx, 'bess_p_lmt_mw'] = p_limit

    # Charging
    # Increase BESS SoC
    df.at[idx, 'bess_soc_end_mwh'] = df.at[idx, 'bess_soc_start_mwh'] + p_limit*t_delta

    # Output at inverter
    df.at[idx, 'inv_p_dc_mw'] = demand_dc

    # Record any spill
    df.at[idx, 'spill'] = df.at[idx, 'pv_pmp_mw'] - demand_dc - p_limit

    return df

# Define function that controls the BESS discharging
def bess_dschrg(df, idx, p, demand_dc):
    """
    Function for controlling BESS discharging
    """
    # Determine bess discharge power limit
    # Discharging power is positive, p_limit will be the minimum of available p, dcdc p limit and bess p limit
    p_limit = min(-p, df.at[idx, 'dcdc_p_lmt_dschrg_mw'], df.at[idx, 'bess_p_lmt_dschrg_mw'])

    df.at[idx, 'bess_p_lmt_mw'] = p_limit

    # Discharging
    # Reduce BESS SoC
    df.at[idx, 'bess_soc_end_mwh'] = df.at[idx, 'bess_soc_start_mwh'] - p_limit*t_delta
    
    # Output at inverter
    df.at[idx, 'inv_p_dc_mw'] = p_limit + df.at[idx, 'pv_pmp_mw']

    # Record any spill
    df.at[idx, 'spill'] = 0

    return df

def get_bess_params(df, idx):
    """
    Function for assigning BESS parameters for a given time step
    """
    df.at[idx, 'bess_cap_lmt_chrg_mwh'] = bess_cap_mwh - df.at[idx, 'bess_soc_start_mwh']
    df.at[idx, 'bess_cap_lmt_dschrg_mwh'] =  df.at[idx, 'bess_soc_start_mwh']
    df.at[idx, 'bess_p_lmt_chrg_mw'] = -1*df.at[idx, 'bess_cap_lmt_chrg_mwh']/t_delta
    df.at[idx, 'bess_p_lmt_dschrg_mw'] = df.at[idx, 'bess_cap_lmt_dschrg_mwh']/t_delta
    return df

def get_dcdc_params(df, idx):
    """
    Function for assigning DC-DC converter parameters for a given time step
    """
    df.at[idx, 'dcdc_p_lmt_chrg_mw'] = -dcdc_p_mw
    df.at[idx, 'dcdc_p_lmt_dschrg_mw'] = dcdc_p_mw
    return df

def get_inv_params(df, idx):
    """
    Function for assigning Inverter parameters for a given time step
    """
    df.at[idx, 'inv_p_dc_lmt_mw'] = inv_p_mw
    return df


def simulate(df):
    """
    Function for itterating through simulation time series
    """
    for idx, row in df.iterrows():

        # Calculate demand_dc. This will be a function of 'demand_ac' and inverter efficiency
        # This is a placeholder for a future function.
        demand_dc = demand_ac

        # Get BESS SoC from previous time step
        if idx > 0:
            df.at[idx, 'bess_soc_start_mwh'] = df.at[idx-1, 'bess_soc_end_mwh']

        # Calculate difference between supply and demand.
        df.at[idx, 'p_net_dc'] = df.at[idx, 'pv_pmp_mw'] - demand_dc

        # Calculate device parameters for given time step
        df = get_bess_params(df, idx)
        df = get_dcdc_params(df, idx)
        df = get_bess_params(df, idx)

        
        #######################################################
        # If p_net_dc is 0, supply meets demand, bess does nothing
        #######################################################
        if df.at[idx, 'p_net_dc'] == 0:
            df.at[idx, 'inv_p_dc_mw'] = demand_dc
        
        #######################################################
        # If p_net_dc is positive, excess power can charge bess
        #######################################################
        if df.at[idx, 'p_net_dc'] > 0:
            df = bess_chrg(df, idx, df.at[idx, 'p_net_dc'], demand_dc)

        #######################################################
        # If p_net_dc is negative, bess power must be used 
        #######################################################
        if df.at[idx, 'p_net_dc'] < 0:
            df = bess_dschrg(df, idx, df.at[idx, 'p_net_dc'], demand_dc)

    return df.round(decimals = 2)


# COMMAND ----------

df_results = simulate(df_working.iloc[:105120])
# df_results = simulate(df_working.iloc[:288])
# df_working.iloc[:105120]

# COMMAND ----------

spark.createDataFrame(df_results).display()

# COMMAND ----------


