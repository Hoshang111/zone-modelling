# Databricks notebook source
# MAGIC %pip install pvlib

# COMMAND ----------

from IPython.display import display
from datetime import datetime
import ipywidgets as widgets
import pvlib as pl
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

class pvlib_wrapper():
    def __init__(self):
        """
        This is a wrapper around the PVlib library classes ModelChain, PVSystem and Location. It is designed to provide a UI for inputing required parameters and then running a simulation.

        Simulation Steps (i.e. what should happen when you press the "load_all" button):

        1. Extract the relevant parameters from the Solcast data (funtion "get_met_data"), arange them into a dataframe called "sim_data" with the column names defined here:

        https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.prepare_inputs.html#pvlib.modelchain.ModelChain.prepare_inputs

        Solcast Name        PVLib Name
        -----------         -----------
        PeriodStart         PeriodStart
        AirTemp             temp_air
        Dhi                 dhi
        Dni                 dni
        Ghi                 ghi
        WindSpeed10m        wind_speed
        Azimuth             solar_azimuth
        Zenith              solar_zenith
        --                  albedo          Note: take the "albedo" value  from the Common Parameters input by the user in the UI 
        


        2. Next, we use the solcast "solar_zenith" and "solar_azimuth" values to calculate "surface_tilt" and "surface_azimuth".
        Use the function "get_tilt_azimuth" defined below.
        If SAT, determine tracking angle. If MAV, used fixed angles.

        Return df's "surface_tilt" and "surface_azimuth" and append to sim_data df


        3. Get temperature module parameters (function )


        4. Lets initialise the "PVSystem" class here.


        5. Next we want to calculate the POA Irradiance, this is the actual irradiance hitting the PV panel, taking into account the orientation of the panel.
        This is done using the PVSystem.get_irradiance() method, but this method has to be overwritten because the calculation is different depending on whether the array is SAT vs MAV.


        6. Calculate the Cell Temperature. 


        7. 
        Last Step. Return a dataframe with the hourly Pmp and curve_info. This is done using the "pvsystem.singlediode" equation

        """
        # this is where I plan to initialise the UI.
        # self.ui = init_ui()

        # HOSHANG ---   Build the gui (functions below) that is able to return the following params. Common params are common. SAT/MAV inputs depend on whether array_type is SAT or MAV
        #               There will need to be a load button that when pressed, returns the variables and runs the "load_all()" function.

        # Common Params - Generic
        self.met_data = None #this will be the solcast dataframe
        self.met_data_name = None #this should be the name of the met_data (solcast) file
        self.sim_data = None
        self.array_type = None # string: 'SAT' or 'MAV'
        self.modules_per_string = 1
        self.strings_per_inverter = 1
        self.racking_model = None # string: ‘open_rack’, ‘close_mount’ or ‘insulated_back’
        self.albedo = None # int
        # self.surface_type = None # future, related to
        self.model = None # string: 'sapm', 'pvsyst', 'faiman', 'fuentes', and 'noct_sam'
        
        # Common Params - Module
        self.bifacial = True
        self.bifacial_factor = 0.85
        self.module_type = None # string "glass_polymer" or "glass_glass"
        self.module_name = None # string from module_widget dropdown

        # SAT Params
        self.sat_axis_tilt = 0
        self.sat_axis_azimuth = 0
        self.sat_max_angle = 60
        self.sat_backtrack = True
        self.sat_pitch = 5
        self.sat_height = 1.5
        self.sat_mod_length = 2.1

        # MAV Params
        self.mav_tilt = 10
        self.mav_azimuth = 90


        ####################
        # These are parameters that do not need gui inputs
        self.surface_tilt = None
        self.surface_azimuth = None
        self.module_parameters = None
        self.temperature_model_parameters = None

    def get_tilt_azimuth(self):
        """
        This function needs to return values for surface_tilt and surface_azimuth for every time step in the simulation data.
        """
        if self.array_type == "SAT":
            # load solar position and tracker orientation for use in pvsystem object
            sat_mount = pl.pvsystem.SingleAxisTrackerMount(axis_tilt=self.sat_axis_tilt,  # flat array
                                                        axis_azimuth=self.sat_axis_azimuth,  # north-facing azimuth
                                                        max_angle=self.sat_max_angle,  # a common maximum rotation
                                                        backtrack=self.sat_backtrack,
                                                        gcr=self.sat_mod_length / self.sat_pitch)
            # created for use in pvfactors timeseries
            orientation = sat_mount.get_orientation(self.sim_data['solar_zenith'],self.sim_data['solar_azimuth'])
            self.surface_tilt = orientation['surface_tilt']
            self.surface_azimuth = orientation['surface_azimuth']

        else:
            self.surface_tilt = self.mav_tilt
            self.surface_azimuth = self.mav_azimuth

    def get_met_data(self, metdata, albedo):
        """
        function to extract relevant data out of Solcast dataframe.
        """
        solcast_query = "select * from sandbox.met_data." + metdata
        df_solcast = spark.sql(solcast_query).toPandas()
        simdata_df = pd.DataFrame()
        simdata_df['PeriodStart'] = df_solcast['PeriodStart']
        simdata_df['temp_air'] = df_solcast['AirTemp']
        simdata_df['dhi'] = df_solcast['Dhi']
        simdata_df['dni'] = df_solcast['Dni']
        simdata_df['ghi'] = df_solcast['Ghi']
        simdata_df['wind_speed'] = df_solcast['WindSpeed10m']
        simdata_df['solar_azimuth'] = df_solcast['Azimuth']
        simdata_df['solar_zenith'] = df_solcast['Zenith']
        simdata_df['albedo'] = albedo
        simdata_df.set_index('PeriodStart')

        return simdata_df
    
    def get_mod_params(self):
        """
        Extract module parameters from GUI
        """
        # write code here
        return

    def get_temperature_model_parameters(self):
        """
        Get temperature_model_parameters depending on module properties
        """
        if self.bifacial:
            self.temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        else:
            self.temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']


    # "location" and "system" are parameters inherited from ModelChain. I am initialising them with the custom classes.
    def load_all(self):
        #step 1
        self.get_met_data()
        #step 2
        self.get_tilt_azimuth()
        #step 3
        self.get_temperature_model_parameters()
        #step 4
        self.system = Custom_System(surface_tilt=self.surface_tilt,
                                    surface_azimuth=self.surface_azimuth,
                                    albedo=self.albedo,
                                    module=self.module,
                                    module_type=self.module_type,
                                    module_parameters=self.module_parameters,
                                    temperature_model_parameters=self.temperature_model_parameters,
                                    modules_per_string=self.modules_per_string,
                                    strings_per_inverter=self.strings_per_inverter)
        self.location = Custom_Location()
        # self.mc = ModelChain(self.system,self.location)

        #step 5
        self.sim_data = self.sim_data.append(self.system.get_irradiance())

        #step 6
        self.sim_data["cell_temp_vmp"] = self.system.get_cell_temperature(self.sim_data['poa_global'], self.sim_data['temp_air'], self.sim_data['wind_speed'], model="sapm", effective_irradiance=None)

        #step 7
        



class Custom_System(PVSystem):
    """
    Overwrite a few of the pvlib.PVSystem class functions
    """
    def __init__(self):
        # self.surface_tilt = None
        # self.surface_azimuth = None
        # self.albedo
        # self.surface_type

        # self.module
        # self.module_type
        # self.module_parameters

        # self.temperature_model_parameters

        # self.modules_per_string
        # self.strings_per_inverter
        # self.racking_model
        return



    # def get_irradiance(self):
    #     '''
    #     Override get_irradiance method, use infinite_sheds.get_irradiance if modules are bifacial.

    #     Args:

    #     Returns:
    #         None
    #     '''
    #     return


# class Custom_Location(Location):
#     """
#     Overwrite a few of the pvlib.PVSystem class functions
#     """
#     def __init__(self):



# COMMAND ----------

pv_model = Custom_System()

# COMMAND ----------

import pvlib as pl
import pandas as pd
from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain


def get_met_data_sources():
    """
    Returns a list of table names in the data repository schema "sandbox.met_data"
    """
    return spark.sql("SHOW TABLES in sandbox.met_data").rdd.map(lambda x: x.tableName).collect()

def get_modules():
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
    CECMODS = pl.pvsystem.retrieve_sam(path="https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Modules.csv")
    module_list = []
    for col in CECMODS.columns:
        module_list.append(col)
    
    return module_list

def get_location(metdata):
    string = metdata.strip('solcast')
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

def create_simdata(metdata, albedo):
        solcast_query = "select * from sandbox.met_data." + metdata
        df_solcast = spark.sql(solcast_query).toPandas()
        simdata_df = pd.DataFrame()
        simdata_df['PeriodStart'] = df_solcast['PeriodStart']
        simdata_df['temp_air'] = df_solcast['AirTemp']
        simdata_df['dhi'] = df_solcast['Dhi']
        simdata_df['dni'] = df_solcast['Dni']
        simdata_df['ghi'] = df_solcast['Ghi']
        simdata_df['wind_speed'] = df_solcast['WindSpeed10m']
        simdata_df['solar_azimuth'] = df_solcast['Azimuth']
        simdata_df['solar_zenith'] = df_solcast['Zenith']
        simdata_df['albedo'] = df_solcast['AlbedoDaily']
        simdata_df.set_index('PeriodStart')

        return simdata_df


def pvlib_ui():
    CECMODS = pl.pvsystem.retrieve_sam(path="https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Modules.csv")
    output = widgets.Output()
    
    def create_new_system(simdata_df):
        # Initialise new wrapper class
        NewSystem = pvlib_wrapper()
        solcast_query = "select * from sandbox.met_data." + met_data_widget.value
        df_solcast = spark.sql(solcast_query).toPandas()

        # Load Generic Params
        NewSystem.met_data = df_solcast
        NewSystem.met_data_name = met_data_widget.value            
        NewSystem.sim_data = simdata_df
        NewSystem.array_type = array_type_widget.value
        NewSystem.modules_per_string = modules_per_string_widget.value
        NewSystem.strings_per_inverter = strings_per_inverter_widget.value
        NewSystem.racking_model = racking_model_widget.value
        NewSystem.albedo = albedo_widget.value
        NewSystem.model = model_widget.value

        # Load Module Params
        NewSystem.bifacial = bifacial_widget.value
        NewSystem.bifacial_factor = bifacial_factor_widget.value
        NewSystem.module_type = module_type_widget.value
        NewSystem.module_name = module_widget.value

        # SAT Params
        if NewSystem.array_type == "SAT":
            NewSystem.sat_axis_tilt = sat_axis_tilt_widget.value
            NewSystem.sat_axis_azimuth = sat_axis_azimuth_widget.value
            NewSystem.sat_max_angle = sat_max_angle_widget.value
            NewSystem.sat_backtrack = sat_backtrack_widget.value
            NewSystem.sat_pitch = sat_pitch_widget.value
            NewSystem.sat_height = sat_height_widget.value
            NewSystem.sat_mod_length = sat_mod_length_widget.value


        #MAV Params
        if NewSystem.array_type == "MAV":
            NewSystem.mav_tilt = mav_tilt_widget.value
            NewSystem.mav_azimuth = mav_azimuth_widget.value


        # Get Module Parameters
        NewSystem.module_parameters = CECMODS[module_widget.value]
        NewSystem.get_tilt_azimuth()

        # Get Temperature Model Parameters and append to simdata
        NewSystem.get_temperature_model_parameters()

        return NewSystem


    # Disables inputs based on Array Widget value
    def on_value_change(change):
        with output:
            output.clear_output()
            print(change.new)
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
            # Create sim data dataframe and system
            latitude, longitude = get_location(met_data_widget.value)
            simdata_df = create_simdata(met_data_widget.value, albedo_widget.value)
            System = create_new_system(simdata_df)
            PV_location = Location(latitude, longitude)
            inverter_database = pl.pvsystem.retrieve_sam(name='cecinverter')
            inverter = inverter_database.ABB__PVI_3_0_OUTD_S_US__208V_

            PV_system = PVSystem(surface_tilt=System.surface_tilt,
                                    surface_azimuth=System.surface_azimuth,
                                    albedo=System.albedo,
                                    module=System.module_name,
                                    module_type=System.module_type,
                                    module_parameters=System.module_parameters,
                                    temperature_model_parameters=System.temperature_model_parameters,
                                    modules_per_string=System.modules_per_string,
                                    strings_per_inverter=System.strings_per_inverter,
                                    inverter=inverter,
                                    inverter_parameters=inverter,
                                    racking_model=System.racking_model)
            
            simdata_df['surface_tilt'] = System.surface_tilt
            simdata_df['surface_azimuth'] = System.surface_azimuth

            
            print(System.module_parameters)
            # Get Irradiance
            # poa_irradiance = PV_system.get_irradiance(solar_zenith=simdata_df['solar_zenith'],solar_azimuth=simdata_df['solar_azimuth'],dni=simdata_df['dni'], ghi=simdata_df['ghi'], dhi=simdata_df['dhi'], airmass=1.5,albedo=simdata_df['albedo'])
            # df = simdata_df.join(poa_irradiance, how="outer")
            
            # Get Cell Temperature
            # cell_temps = []
            # for row in df.iterrows():
            #     poa_global = row[1][9]
            #     temp_air = row[1][1]
            #     wind_speed = row[1][5]
            #     model = System.model

            #     cell_temp = PV_system.get_cell_temperature(poa_global=poa_global, temp_air=temp_air,wind_speed=wind_speed,model=model)
            #     cell_temps.append(cell_temp)

            # df['Cell Temperature'] = cell_temps


            # Get Single Diode Equation


            System_ModelChain = ModelChain(PV_system, PV_location, aoi_model="no_loss")

    layout = widgets.Layout(width='auto')
    style = {'description_width': 'initial'}
    sources = get_met_data_sources()
    met_data_widget = widgets.Dropdown(options=sources, description='Met Data', value=None, style=style)
    
    # Create Common Param Widgets
    array_type_widget = widgets.RadioButtons(options=['SAT', 'MAV'], description='Array Type', value=None, layout=layout, style=style)
    array_type_widget.observe(on_value_change, 'value')
    modules_per_string_widget = widgets.IntText(value=1, description='Modules Per String', disabled=False, layout=layout, style=style)
    strings_per_inverter_widget = widgets.IntText(value=1, description='Strings Per Inverter', disabled=False, layout=layout, style=style)
    racking_model_widget = widgets.Dropdown(options=['open_rack', 'close_mount', 'insulated_back'], description='Racking Model', value=None, layout=layout, style=style)
    albedo_widget = widgets.FloatSlider(value=0.3, min=0.1,max=0.4,step=0.1, description='Albedo',
    disabled=False, continuous_update=False, orientation='horizontal', readout=True,readout_format='.1f')
    model_widget = widgets.Dropdown(options=['sapm', 'pvsyst', 'faiman', 'fuentes', 'noct_sam'], description='Model', value=None, layout=layout, style=style)

    # Create Module Param Widgets
    bifacial_widget = widgets.Checkbox(value=False, description='Bifacial', disabled=False,indent=False, layout=layout, style=style)
    bifacial_factor_widget = widgets.FloatText(value=0.85, description='Bifacial Factor', disabled=False, layout=layout, style=style)
    module_widget = widgets.Dropdown(options=get_modules(), description='Module Name', value=None)
    module_type_widget = widgets.RadioButtons(options=['glass_polymer', 'glass_glass'], description='Module Type', value=None, layout=layout, style=style)

    # Create SAT Param Widgets
    sat_axis_tilt_widget = widgets.FloatText(value=0, description='Axis Tilt', disabled=True, layout=layout, style=style)
    sat_axis_azimuth_widget = widgets.FloatText(value=0, description='Axis Azimuth', disabled=True, layout=layout, style=style)
    sat_max_angle_widget = widgets.FloatText(value=60, description='Max Angle', disabled=True, layout=layout, style=style)
    sat_backtrack_widget = widgets.Checkbox(value=False, description='Backtrack', disabled=True,indent=False, layout=layout, style=style)
    sat_pitch_widget = widgets.FloatText(value=5, description='Pitch', disabled=True, layout=layout, style=style)
    sat_height_widget = widgets.FloatText(value=1.5, description='Height', disabled=True, layout=layout, style=style)
    sat_mod_length_widget = widgets.FloatText(value=2.1, description='Module Length', disabled=True, layout=layout, style=style)

    # Create MAV Param Widgets
    mav_tilt_widget = widgets.FloatText(value=0, description='Tilt', disabled=True, layout=layout, style=style)
    mav_azimuth_widget = widgets.FloatText(value=0, description='Azimuth', disabled=True, layout=layout, style=style)

    # Create Common Tab
    array_box =  widgets.HBox([array_type_widget, albedo_widget])
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

pvlib_ui()

# COMMAND ----------


