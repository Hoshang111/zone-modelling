# Databricks notebook source
# DBTITLE 1,Install PVLib
# MAGIC %pip install pvlib

# COMMAND ----------

# DBTITLE 1,Custom System
from IPython.display import display, clear_output
from datetime import datetime
from databricks.sdk.runtime import spark
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' #suppress anoying warnings
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pvlib as pl
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from IPython.display import display
from datetime import datetime

class Custom_System:
    def __init__(self):
        '''
        This is a wrapper around the PVlib library classes ModelChain, PVSystem and Location. It is designed to provide a UI for inputing required parameters and then running a simulation.
        '''

        # Common Params - Generic
        self.met_data = None
        self.sim_data = None
        self.array_type = None
        self.modules_per_string = None
        self.strings_per_inverter = None
        self.racking_model = None
        # self.surface_type = None # future, related to
        self.model = None
        
        # Common Params - Module
        self.bifacial = None
        self.bifacial_factor = None
        self.module_type = None
        self.module_name = None

        # SAT Params
        self.sat_axis_tilt = None
        self.sat_axis_azimuth = None
        self.sat_max_angle = None
        self.sat_backtrack = None
        self.sat_pitch = None
        self.sat_height = None
        self.sat_mod_length = None

        # MAV Params
        self.mav_tilt = None
        self.mav_azimuth = None

        ####################
        # These are parameters that do not need gui inputs
        self.surface_tilt = None
        self.surface_azimuth = None
        self.module_parameters = None
        self.temperature_model_parameters = None
        self.SystemPV = None
        self.curves = None
        self.yearly_df = None
        self.modules = self.get_modules()

        self.ui()

    def print_status(self, status):
        print("{}:  {}".format(datetime.now(),status))

    def get_modules(self):
        """
        Return a list of all modules in the online database
        """
        # retrieve CEC module parameters from the SAM libraries
        # with the "name" argument set to "CECMod"
        # List also seen here:
        # https://solarequipment.energy.ca.gov/Home/PVModuleList

        # the CEC modules are a pandas DataFrame oriented as columns, transpose to arrange
        # as indices
        # CECMODS.T.head()
        # https://pvsc-python-tutorials.github.io/PVSC48-Python-Tutorial/Tutorial%204%20-%20Model%20a%20Module%27s%20Performance.html
        # CECMODS = pl.pvsystem.retrieve_sam(path="https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Modules.csv")

        # Code to retrieve custom modules from Databricks file store, format same as pl.pvsystem.retrieve_sam
        df = spark.sql("select * from sandbox.pv_data.ref_mods").toPandas().drop([0, 1]).set_index("Name",drop=True)

        df.columns = df.columns.str.replace(' ', '_')
        df.index = pl.pvsystem._normalize_sam_product_names(df.index)
        df = df.transpose()

        if 'ADRCoefficients' in df.index:
            ad_ce = 'ADRCoefficients'
            # for each inverter, parses a string of coefficients like
            # ' 1.33, 2.11, 3.12' into a list containing floats:
            # [1.33, 2.11, 3.12]
            df.loc[ad_ce] = df.loc[ad_ce].map(lambda x: list(
                map(float, x.strip(' []').split())))

        # Retrieve CEC Modules from web
        CECMODS = pl.pvsystem.retrieve_sam(path="https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Modules.csv")

        modules = pd.concat([df,CECMODS], axis=1)
        
        return modules


    def get_met_data_sources(self):
        """
        Returns a list of table names in the data repository schema "sandbox.met_data"
        """
        return spark.sql("SHOW TABLES in sandbox.met_data").rdd.map(lambda x: x.tableName).collect()
    
    def get_met_data(self, metdata):
        """
        function to extract relevant data out of Solcast dataframe.
        """
        solcast_query = "select * from sandbox.met_data." + metdata
        df_solcast = spark.sql(solcast_query).toPandas()
        simdata_df = pd.DataFrame()
        simdata_df['PeriodStart'] = df_solcast['PeriodStart']
        simdata_df['PeriodStart'] = pd.to_datetime(simdata_df["PeriodStart"], dayfirst=True, utc=True).dt.tz_convert("Australia/Darwin").dt.tz_localize(None)
        simdata_df['temp_air'] = df_solcast['AirTemp']
        simdata_df['dhi'] = df_solcast['Dhi']
        simdata_df['dni'] = df_solcast['Dni']
        simdata_df['ghi'] = df_solcast['Ghi']
        simdata_df['wind_speed'] = df_solcast['WindSpeed10m']
        simdata_df['solar_azimuth'] = df_solcast['Azimuth']
        simdata_df['solar_zenith'] = df_solcast['Zenith']
        simdata_df['albedo'] = df_solcast['AlbedoDaily']
        simdata_df['pressure'] = df_solcast['SurfacePressure']
        simdata_df = simdata_df.set_index('PeriodStart')

        self.met_data = metdata
        self.sim_data = simdata_df
        
    def set_generic_params(self, array_type, 
                          modules_per_string, 
                          strings_per_inverter, 
                          racking_model,
                          model):
        self.array_type = array_type
        self.modules_per_string = modules_per_string
        self.strings_per_inverter = strings_per_inverter
        self.racking_model = racking_model
        # self.surface_type = None # future, related to
        self.model = model

    def set_module_params(self, bifacial, 
                          bifacial_factor, 
                          module_name, 
                          module_type):
        self.bifacial = bifacial
        self.bifacial_factor = bifacial_factor
        self.module_name = module_name
        self.module_type = module_type 

    def set_sat_params(self, sat_axis_tilt, 
                       sat_axis_azimuth,
                       sat_max_angle,
                       sat_backtrack,
                       sat_pitch,
                       sat_height,
                       sat_mod_length):
        self.sat_axis_tilt = sat_axis_tilt
        self.sat_axis_azimuth = sat_axis_azimuth
        self.sat_max_angle = sat_max_angle
        self.sat_backtrack = sat_backtrack
        self.sat_pitch = sat_pitch
        self.sat_height = sat_height
        self.sat_mod_length = sat_mod_length
    
    def set_mav_params(self, mav_tilt, mav_azimuth):
        self.mav_tilt = mav_tilt
        self.mav_azimuth = mav_azimuth

    def get_tilt_azimuth(self):
        if self.array_type == "SAT":
            # load solar position and tracker orientation for use in pvsystem object
            sat_mount = pl.pvsystem.SingleAxisTrackerMount(axis_tilt=self.sat_axis_tilt,  # flat array
                                                        axis_azimuth=self.sat_axis_azimuth,  # north-facing azimuth
                                                        max_angle=self.sat_max_angle,  # a common maximum rotation
                                                        backtrack=self.sat_backtrack,
                                                        gcr=self.sat_mod_length / self.sat_pitch)
            # created for use in pvfactors timeseries, replace NaN with 0
            orientation = sat_mount.get_orientation(self.sim_data['solar_zenith'],self.sim_data['solar_azimuth']).fillna(0)
            self.surface_tilt = orientation['surface_tilt']
            self.surface_azimuth = orientation['surface_azimuth']
        else:
            self.surface_tilt = self.mav_tilt
            self.surface_azimuth = self.mav_azimuth
        
        self.sim_data['surface_tilt'] = self.surface_tilt
        self.sim_data['surface_azimuth'] = self.surface_azimuth

    def get_temperature_model_parameters(self):
        """
        Get temperature_model_parameters depending on module properties
        """
         # if bifacial (open_rack_glass_glass)
         # else (open_rack_glass_polymer)
        string = self.racking_model + "_" + self.module_type
        self.temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS[self.model][string]

    def get_module_parameters(self, module_name):
        self.module_parameters = self.modules[module_name]

    def get_location(self):
        string = self.met_data.strip('solcast')
        string = string.replace('__', '-')
        string = string.replace('_', '.')
        dot_count = 0
        i = 0
        for character in string:
            if character == "." and dot_count == 1:
                break
            elif character == "." and dot_count != 1:
                dot_count += 1
            i += 1
        latitude = string[:i]    
        longitude= string[i+1:]
        return float(latitude), float(longitude)

    def create_pvsystem(self):
            self.SystemPV = PVSystem(surface_tilt=self.surface_tilt,
                            surface_azimuth=self.surface_azimuth,
                            albedo=self.sim_data['albedo'].mean(),
                            module=self.module_name,
                            module_type=self.module_type,
                            module_parameters=self.module_parameters,
                            temperature_model_parameters=self.temperature_model_parameters,
                            modules_per_string=self.modules_per_string,
                            strings_per_inverter=self.strings_per_inverter,
                            racking_model=self.racking_model)
            
    def get_irradiance(self):
        if self.bifacial == True:
            # npoints set to 10, default 100. Reduction in accuracy deemed insignificant for power/voltage modelling.
            poa_irradiance = pl.bifacial.infinite_sheds.get_irradiance(surface_tilt=self.surface_tilt, 
                                                                       surface_azimuth=self.surface_azimuth, 
                                                                       solar_zenith=self.sim_data['solar_zenith'], 
                                                                       solar_azimuth=self.sim_data['solar_azimuth'], 
                                                                       gcr=self.sat_mod_length/self.sat_pitch, 
                                                                       height=self.sat_height, 
                                                                       pitch=self.sat_pitch, 
                                                                       ghi=self.sim_data['ghi'], 
                                                                       dhi=self.sim_data['dhi'], 
                                                                       dni=self.sim_data['dni'], 
                                                                       albedo=self.sim_data['albedo'], 
                                                                       bifaciality=self.bifacial_factor,
                                                                       npoints=10,
                                                                       vectorize=True)
        else:
            poa_irradiance = self.SystemPV.get_irradiance(solar_zenith=self.sim_data['solar_zenith'],
                                                          solar_azimuth=self.sim_data['solar_azimuth'],
                                                          dni=self.sim_data['dni'], 
                                                          ghi=self.sim_data['ghi'], 
                                                          dhi=self.sim_data['dhi'], 
                                                          dni_extra=None, 
                                                          airmass=None, 
                                                          albedo=self.sim_data['albedo'], 
                                                          model='haydavies')
        
        self.sim_data = self.sim_data.join(poa_irradiance, how="outer")

    def get_cell_temperature(self):
        cell_temp = self.SystemPV.get_cell_temperature(poa_global=self.sim_data['poa_global'], temp_air=self.sim_data['temp_air'],wind_speed=self.sim_data['wind_speed'],model="sapm")
        self.sim_data['cell_temperature'] = cell_temp

    def get_iv_curve(self):
        alpha_sc = self.module_parameters.loc["alpha_sc"]
        a_ref = self.module_parameters.loc["a_ref"]
        I_L_ref = self.module_parameters.loc["I_L_ref"]
        I_o_ref = self.module_parameters.loc["I_o_ref"]
        Adjust = self.module_parameters.loc["Adjust"]
        R_s = self.module_parameters.loc["R_s"]
        R_sh_ref = self.module_parameters.loc["R_sh_ref"]
        
        # yearly_df = system.sim_data.iloc[:105120]
        # yearly_df = system.sim_data
        # self.yearly_df = yearly_df


        # Calcparams_cec
        IL, I0, Rs, Rsh, nNsVth = pl.pvsystem.calcparams_cec(effective_irradiance=system.sim_data["poa_global"],
                        temp_cell=system.sim_data['cell_temperature'], 
                        alpha_sc=alpha_sc,
                        I_L_ref=I_L_ref,
                        I_o_ref=I_o_ref,
                        R_s=R_s,
                        R_sh_ref=R_sh_ref,
                        a_ref=a_ref,
                        Adjust=Adjust)
        
        self.sim_data['IL'] = IL
        self.sim_data['I0'] = I0
        self.sim_data['Rs'] = Rs
        self.sim_data['Rsh'] = Rsh
        self.sim_data['nNsVth'] = nNsVth
        # Solve singlediode
        # curve_info = pl.pvsystem.singlediode(photocurrent=IL,
        #                         saturation_current=I0,
        #                         resistance_series=Rs,
        #                         resistance_shunt=Rsh,
        #                         nNsVth=nNsVth,
        #                         ivcurve_pnts=15,
        #                         method='lambertw')

        # self.sim_data['p_mp'] = curve_info['p_mp']
        # self.sim_data['i_sc'] = curve_info['i_sc']
        # self.sim_data['v_oc'] = curve_info['v_oc']
        # self.sim_data['i_mp'] = curve_info['i_mp']
        # self.sim_data['v_mp'] = curve_info['v_mp']
        # self.curves = curve_info

    def ui(self):
        output = widgets.Output()

        def on_value_change(change):
            with output:
                output.clear_output()
                if change.new == "SAT":
                    sat_axis_tilt_widget.disabled = False
                    sat_axis_azimuth_widget.disabled = False
                    sat_max_angle_widget.disabled = False
                    sat_backtrack_widget.disabled = False
                    sat_pitch_widget.disabled = False
                    sat_height_widget.disabled = False
                    sat_mod_length_widget.disabled = False
                    mav_tilt_widget.disabled = True
                    mav_azimuth_widget.disabled = True
                if change.new == "MAV":
                    mav_tilt_widget.disabled = False
                    mav_azimuth_widget.disabled = False
                    sat_axis_tilt_widget.disabled = True
                    sat_axis_azimuth_widget.disabled = True
                    sat_max_angle_widget.disabled = True
                    sat_backtrack_widget.disabled = True
                    sat_pitch_widget.disabled = True
                    sat_height_widget.disabled = True
                    sat_mod_length_widget.disabled = True
                    
        def on_load_clicked(_):
            with output:
                # Clear any warnings.
                clear_output()
                self.print_status("Running, this may take a few seconds...")
                self.print_status("Loading met data...")
                self.get_met_data(met_data_widget.value)
                

                self.set_generic_params(array_type_widget.value, 
                            modules_per_string_widget.value, 
                            strings_per_inverter_widget.value, 
                            racking_model_widget.value,
                            model_widget.value)
                
                self.set_module_params(bifacial_widget.value,
                                    bifacial_factor_widget.value,
                                    module_widget.value,
                                    module_type_widget.value)
                
                if array_type_widget.value == "SAT":
                    self.set_sat_params(sat_axis_tilt_widget.value,
                                        sat_axis_azimuth_widget.value,
                                        sat_max_angle_widget.value,
                                        sat_backtrack_widget.value,
                                        sat_pitch_widget.value,
                                        sat_height_widget.value,
                                        sat_mod_length_widget.value)
                elif array_type_widget.value == "MAV":
                    self.set_mav_params(mav_tilt_widget.value, mav_azimuth_widget.value)

                self.print_status("Loading Module Parameters...")
                self.get_module_parameters(module_widget.value)

                self.print_status("Loading Temperature Parameters...")
                self.get_temperature_model_parameters()

                self.print_status("Calculating Tilt, Azimuth...")
                self.get_tilt_azimuth()

                latitude, longitude = self.get_location()

                self.print_status("Creating PVSystem...")
                self.create_pvsystem()

                self.print_status("Calculating Irradiance...")
                self.get_irradiance()

                self.print_status("Calculating Cell Temperature...")
                self.get_cell_temperature()

                self.print_status("Calculating IV curves...")
                self.get_iv_curve()
                self.print_status("Run Finished!")


        # Clear any warnings.
        clear_output()

        layout = widgets.Layout(width='auto')
        style = {'description_width': 'initial'}
        sources = self.get_met_data_sources()
        met_data_widget = widgets.Dropdown(options=sources, description='Met Data', value=None, style=style)
        
        # Create Common Param Widgets
        array_type_widget = widgets.RadioButtons(options=['SAT', 'MAV'], description='Array Type', value=None, layout=layout, style=style)
        array_type_widget.observe(on_value_change, 'value')
        modules_per_string_widget = widgets.IntText(value=1, description='Modules Per String', disabled=False, layout=layout, style=style)
        strings_per_inverter_widget = widgets.IntText(value=1, description='Strings Per Inverter', disabled=False, layout=layout, style=style)
        racking_model_widget = widgets.Dropdown(options=['open_rack', 'close_mount', 'insulated_back'], description='Racking Model', value='open_rack', layout=layout, style=style)
        model_widget = widgets.Dropdown(options=['sapm', 'pvsyst', 'faiman', 'fuentes', 'noct_sam'], description='Model', value='sapm', layout=layout, style=style)

        # Create Module Param Widgets
        bifacial_widget = widgets.Checkbox(value=False, description='Bifacial', disabled=False,indent=False, layout=layout, style=style)
        bifacial_factor_widget = widgets.FloatText(value=0.85, description='Bifacial Factor', disabled=False, layout=layout, style=style)
        module_widget = widgets.Dropdown(options=[col for col in self.modules], description='Module Name', value=None)
        module_type_widget = widgets.RadioButtons(options=['glass_polymer', 'glass_glass'], description='Module Type', value=None, layout=layout, style=style)

        # Create SAT Param Widgets
        sat_axis_tilt_widget = widgets.FloatText(value=0, description='Axis Tilt', disabled=True, layout=layout, style=style)
        sat_axis_azimuth_widget = widgets.FloatText(value=0, description='Axis Azimuth', disabled=True, layout=layout, style=style)
        sat_max_angle_widget = widgets.FloatText(value=60, description='Max Angle', disabled=True, layout=layout, style=style)
        sat_backtrack_widget = widgets.Checkbox(value=True, description='Backtrack', disabled=True,indent=False, layout=layout, style=style)
        sat_pitch_widget = widgets.FloatText(value=5, description='Pitch', disabled=True, layout=layout, style=style)
        sat_height_widget = widgets.FloatText(value=1.5, description='Height', disabled=True, layout=layout, style=style)
        sat_mod_length_widget = widgets.FloatText(value=2.1, description='Module Length', disabled=True, layout=layout, style=style)

        # Create MAV Param Widgets
        mav_tilt_widget = widgets.FloatText(value=10, description='Tilt', disabled=True, layout=layout, style=style)
        mav_azimuth_widget = widgets.FloatText(value=90, description='Azimuth', disabled=True, layout=layout, style=style)

        # Create Common Tab
        array_box =  widgets.HBox([array_type_widget])
        strings_box = widgets.HBox([modules_per_string_widget, strings_per_inverter_widget])
        model_box = widgets.HBox([model_widget, racking_model_widget])
        generic_box = widgets.VBox([met_data_widget, array_box, strings_box, model_box])

        # Create Module Tab
        bifacial_box = widgets.HBox([bifacial_factor_widget, bifacial_widget])
        mod_box = widgets.HBox([module_widget, module_type_widget])
        module_box = widgets.VBox([bifacial_box, mod_box])
        
        # Create SAT Tab
        sat1_box = widgets.HBox([sat_axis_tilt_widget, sat_axis_azimuth_widget, sat_max_angle_widget, sat_backtrack_widget])
        sat2_box = widgets.HBox([sat_pitch_widget, sat_height_widget, sat_mod_length_widget])
        sat_box = widgets.VBox([sat1_box, sat2_box])

        # Create MAV Tab
        mav_box = widgets.HBox([mav_tilt_widget, mav_azimuth_widget])

        load_button_widget = widgets.Button(description="Load", disabled=False)
        load_button_widget.on_click(on_load_clicked)

        children = [generic_box, module_box, sat_box, mav_box]
        tab = widgets.Tab(children=children)
        tab.set_title(0, 'Generic Parameters')
        tab.set_title(1, 'Module Parameters')
        tab.set_title(2, 'SAT Parameters')
        tab.set_title(3, 'MAV Parameters')

        display(tab, load_button_widget, output)




# COMMAND ----------

# DBTITLE 1,PVLib UI
system = Custom_System()

# COMMAND ----------

# DBTITLE 1,Sort Index
system.sim_data = system.sim_data.sort_index()
system.sim_data

# COMMAND ----------

# DBTITLE 1,Save  simulation data
LR5_72HGD_600M_df = system.sim_data
outdir = '/dbfs/FileStore/pvlib_ui/'

directory = outdir+ 'LR5_72hGD_600M' + '.csv'
LR5_72HGD_600M_df.to_csv(directory, index=True, encoding="utf-8")

# COMMAND ----------

# Code to output yearly_df using spark interface
# adds pandas datetime index back in as column
spark.createDataFrame(system.sim_data.reset_index()).display()

# COMMAND ----------

import pandas as pd
from IPython.display import display, clear_output
from datetime import datetime
import ipywidgets as widgets
from databricks.sdk.runtime import spark
import pytz
from pyspark.sql.functions import *
from pyspark.sql.types import *

class Scenario:
    """
    A class to represent a given Scenario.
    ...
    Attributes
    ----------
    name : str
        name of the scenario
    start : str
        start time of the scenario
    end : str
        end time of the scenario
    optmisation : object
        optimisation table given from scenario, start and end time
    assumption : object
        assumption table given from scenario, start and end time
    Methods
    -------
    None
    """
    def __init__(self, name=None, start=None, end=None):
        '''
        Initialises Scenario
        Args:
            name (str) : name of the scenario
            start (str) : start time of the scenario
            end (str) : end time of the scenario
        Returns:
            None
        '''
        self.name = name
        self.start = start
        self.end = end
        self.optimisation = None
        self.assumption = None
        self.scopti_databricks_ui(True)
        self.timezone = "Australia/Darwin"
    
    def print_scenario_details(self):
        print(f"Scenario Name   : {self.name}")
        print(f"Start Date/Time : {self.start}")
        print(f"End Date/Time   : {self.end}")
        print(f"Time Zone       : {self.timezone}")
    
    def load(self):
        '''
        Load the select scenario optimisation and assumptions.
        Args:
            None
        Returns:
            None
        '''
        print("Scenario Loading, this may take several seconds...")
        self.optimisation = Optimisation(self.name, self.start, self.end, self.timezone)
        self.assumption = Assumption(self.name, self.start, self.end)
        clear_output()
        print("Scenario Loaded")
        self.print_scenario_details()

    def filterdf(self):
        '''
        Filter the scenario DF based on selected parameters.
        Args:
            None
        Returns:
            None
        '''

    def get_all_scenarios(self):
        '''
        Gets all scenarios from the details table
        Args:
            None
        Returns:
                df_scenarios (pandas dataframe object): Dataframe that gives result of all scenarios
        '''
        scenario_details_query = "select * from hive_metastore.federated_postgres.federated_optimisations_scenario_details"
        df_scenarios = spark.sql(scenario_details_query).toPandas().sort_values('scenario')
        return df_scenarios

    def get_active_scenarios(self):
        '''
        Gets the active scenarios from the details table
        Args:
            None
            
        Returns:
                df_scenarios (pandas dataframe object): Dataframe that gives result of active scenarios
        '''
        scenario_details_query = "select * from hive_metastore.federated_postgres.federated_optimisations_scenario_details where is_active == true"
        df_scenarios = spark.sql(scenario_details_query).toPandas().sort_values('scenario')
        return df_scenarios
    
    def scopti_databricks_ui(self, is_active):
        '''
        Creates Databricks UI widgets for a Databricks notebook.
        Args:
            is_active (boolean) : Tells UI whether or not to select active scenarios
        Returns:
                None: This function displays the created widgets.
        '''

        # List of hours in a day in form '00:00:00'
        hours = ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00', '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00', 
        '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00', '24:00:00']

        # List of selectable timezones
        timezone_list = ["Australia/Darwin", "UTC", "Singapore"]

        # Create widgets for the timezone selection
        timezones = widgets.RadioButtons(options=timezone_list, value=timezone_list[0], description='Timezone:',disabled=False)

        # Creates widgets for start and end time parameters
        start_hour = widgets.SelectionSlider(options=hours,value=hours[0], description='Start Hour',
        disabled=False, continuous_update=False, orientation='horizontal', readout=True)
        start_date = widgets.DatePicker(description='Start Date', disabled=False)

        end_hour = widgets.SelectionSlider(options=hours,value=hours[0], description='End Hour',
        disabled=False, continuous_update=False, orientation='horizontal', readout=True)
        end_date = widgets.DatePicker(description='End Date', disabled=False)

        # Creates Scenario dropdown depending on active scenarios are required
        if is_active == True:
            scenarios = self.get_active_scenarios()
        elif is_active == False:
            scenarios = self.get_all_scenarios()
        scenario_widget = widgets.Dropdown(options=scenarios['scenario'], description='Scenario')  

        # Create button widget. Clicking this buttons loads scenario from queried table.
        load_button = widgets.Button(description="Load")

        # Output widget to display the loaded dataframes
        output = widgets.Output()

        def on_load_button_clicked(_):
            '''
            Handles load button, updates scenario object properties
            Args:
                None
            Returns:
                None
            '''
            with output:
                output.clear_output()
                self.name = scenario_widget.value
                if start_date.value is not None:
                    self.start = start_date.value.strftime("%Y-%m-%d") + ' ' + start_hour.value
                if end_date.value is not None:
                    self.end = end_date.value.strftime("%Y-%m-%d") + ' ' + end_hour.value
                self.timezone = timezones.value
                self.load()


        # Register the button's callback function to query df and display results to the output widget
        load_button.on_click(on_load_button_clicked)

        # Define Layout
        section_layout = widgets.Layout(
        border='solid 1px',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px')

        # Collect widgets in boxes
        scenario_box = widgets.VBox([scenario_widget,load_button])

        start_box = widgets.HBox([start_date, start_hour])
        end_box = widgets.HBox([end_date, end_hour])
        filter_box = widgets.VBox([start_box,end_box])

        full_box = widgets.HBox([scenario_box, filter_box, timezones])
        full_box.layout = section_layout

        # Display the widgets and output
        display(full_box, output)

class Optimisation:
    """
    A class to represent a given scenarios' optimisation table.
    ...
    Attributes
    ----------
    name : str
        name of the optimisation scenario
    start : str
        start time of the queried optimisation table
    end : str
        end time of the queried optimisation table
    Methods
    -------
    create_optimisation_df():
        Creates an optimisation table for the scenario
    """
    def __init__(self, scenario, start, end, timezone):
        '''
        Initialises Optimisation Class
        Args:
            name (str) : name of the optimised scenario
            start (str) : start time of the optimised scenario
            end (str) : end time of the optimised scenario
        Returns:
            None
        '''
        self.scenario = scenario
        self.start = start
        self.end = end
        self.timezone = timezone
        self.df = self.create_optimisation_df()

    def create_optimisation_df(self):
        '''
        Creates Pandas Dataframe for optimsation table
        Args:
            None
        Returns:
            df_optimsations (pandas dataframe object) : Dataframe that gives optimsation of scenario
        '''
        if self.timezone == "UTC":
            tz = "period_end_utc"
        if self.timezone == "Singapore":
            tz = "period_end_sg"
        if self.timezone == "Australia/Darwin":
            tz = "period_end_nt"
        optimisations_query = "select "+tz+", identifier, value from hive_metastore.federated_postgres.federated_timezoned_optimisations where scenario = '" + self.scenario + "'"
        df_optimisations = spark.sql(optimisations_query)
        # pandas doesnt have a "decimal" data type, this needs to be cast as "double" prior to conversion
        df_optimisations = df_optimisations.withColumn('value', col('value').cast(DoubleType()))
        df_optimisations = df_optimisations.toPandas()
        df_optimisations = df_optimisations.set_index(pd.DatetimeIndex(df_optimisations[tz]))
        df_optimisations = df_optimisations.pivot(index=tz, columns='identifier', values='value')
        if self.start != None or self.end != None:
            df_optimisations = df_optimisations[self.start:self.end]    
        return df_optimisations

class Assumption:
    """
    A class to represent a given scenarios' assumption table.
    ...
    Attributes
    ----------
    name : str
        name of the scenario
    start : str
        start time of the queried assumption table
    end : str
        end time of the queried assumption table
    Methods
    -------
    create_assumption_df():
        Creates an assumption table for the scenario
    """
    def __init__(self, scenario, start, end):
        '''
        Initialises Assumption Class
        Args:
            name (str) : name of the assumption scenario
            start (str) : start time of the assumption scenario
            end (str) : end time of the assumption scenario
        Returns:
            None
        '''
        self.scenario = scenario
        self.start = start
        self.end = end
        self.df = self.create_assumption_df()

    def create_assumption_df(self):
        '''
        Creates Pandas Dataframe for assumption table
        Args:
            None
        Returns:
            df_assumptions (pandas dataframe object) : Dataframe that gives assumptions of scenario
        '''
        scenario_assumptions_query = "select * from hive_metastore.federated_postgres.federated_optimisations_assumptions where scenario = '" + self.scenario + "' order by period_start asc"
        df_assumptions = spark.sql(scenario_assumptions_query).toPandas()
        df_assumptions['period_start']= pd.to_datetime(df_assumptions['period_start'])
        df_assumptions['period_end']= pd.to_datetime(df_assumptions['period_end'])
        if self.start != None or self.end != None:
            df_assumptions = df_assumptions.loc[(df_assumptions['period_start'] >= self.start) & (df_assumptions['period_end'] <= self.end)]  
        return df_assumptions



# COMMAND ----------

# DBTITLE 1,Scopti Tools UI
scenario = Scenario()

# COMMAND ----------

scenario.optimisation.df

# COMMAND ----------

newdf = pd.DataFrame(np.repeat(df.values, 12, axis=0))
newdf.columns = df.columns
<<<<<<< Updated upstream
newdf
=======
newdf = newdf.set_index(system.yearly_df.index)
optimisation = newdf.join(system.yearly_df, how="outer")
optimisation

assumption = scenario.assumption.df.set_index('period_end').sort_index()

PC_VSC_flow = optimisation['PC_VSC_flow']
p_mp = optimisation['p_mp']
operational_capacity = assumption.loc[assumption['identifier'] == 'PC_battery_operational_capacity']

BESS_df = pd.DataFrame()
num_inverters = 1554

BESS_df['Demand (MW)'] = PC_VSC_flow.div(num_inverters)
BESS_df['PV Production (MW)'] = (p_mp)*12930/1000000
BESS_df['PV Production (MW)'] = BESS_df['PV Production (MW)'].fillna(0)
BESS_df['BESS Size (MWh)'] = 34

BESS_df['Net Flow of Energy (MWh)'] = (BESS_df['PV Production (MW)'] - BESS_df['Demand (MW)'])/12

charge_conditions = [BESS_df['Net Flow of Energy (MWh)'] >= 0, BESS_df['Net Flow of Energy (MWh)'] < 0]
charge_choices = ['Charge', 'Discharge']
BESS_df['Charge/Discharge'] = np.select(charge_conditions, charge_choices)

bess_states = []
spill = []
bess_state = 34
bess_effiency = 0.95
inverter_effiency = 0.95
bess_power_limit = 2/12
current_spill = 0

# Energy Balance
for row in BESS_df.iterrows():
    net_flow = row[1][3]
    if bess_state + net_flow >= 0 and bess_state + net_flow <= 34:
        if net_flow > bess_power_limit:
            current_spill += net_flow - bess_power_limit
            bess_state += bess_power_limit*bess_effiency
        else:
            bess_state += net_flow*bess_effiency
    bess_states.append(bess_state)
    spill.append(current_spill)

BESS_df['BESS State of Charge (MWh)'] = bess_states
BESS_df['Spill (MWh)'] = spill
BESS_df
>>>>>>> Stashed changes

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/pvlib_ui/pmp_df.csv")
df

# COMMAND ----------

        plt.figure()
        v_mp = system.yearly_df['v_mp']
        i_mp = system.yearly_df['i_mp']
        plt.plot(system.curves['v'], system.curves['i'], label="label")
        plt.plot([v_mp], [i_mp], ls='', marker='o', c='k')
    
        plt.legend(loc=(1.0, 0))
        plt.xlabel('Module voltage [V]')
        plt.ylabel('Module current [A]')
        plt.title("SAT")
        plt.show()
        plt.gcf().set_tight_layout(True)

# COMMAND ----------


<<<<<<< Updated upstream
=======

# COMMAND ----------

# DBTITLE 1,Modelchain
df = system.yearly_df
latitude, longitude = system.get_location()
Location = pl.location.Location(latitude, longitude)

solcast_query = "select * from sandbox.met_data." + system.met_data
df_solcast = spark.sql(solcast_query).toPandas()
weather = pd.DataFrame()
weather['precipitable_water'] = df_solcast['PrecipitableWater']
weather = weather.iloc[:105120]
weather =weather.set_index(df.index)

df['precipitable_water'] = weather['precipitable_water']
df['poa_diffuse'] = df['poa_front_diffuse'] + df['poa_back_diffuse']
df['poa_direct'] = df['poa_front_direct'] + df['poa_back_direct']
 
mc = pl.modelchain.ModelChain(system.SystemPV, Location, ac_model=None, aoi_model="no_loss")
mc.run_model_from_poa(df)


# COMMAND ----------

# DBTITLE 1,IV Curves
plt.figure()
v_mp = system.yearly_df['v_mp']
i_mp = system.yearly_df['i_mp']
plt.plot(system.curves['v'], system.curves['i'], label="label")
plt.plot([v_mp], [i_mp], ls='', marker='o', c='k')

plt.legend(loc=(1.0, 0))
plt.xlabel('Module voltage [V]')
plt.ylabel('Module current [A]')
plt.title("SAT")
plt.show()
plt.gcf().set_tight_layout(True)

# COMMAND ----------

solcast_query = "select * from sandbox.met_data.solcast__18_201125_133_408077"
df_solcast = spark.sql(solcast_query)
display(df_solcast)

# COMMAND ----------


>>>>>>> Stashed changes
