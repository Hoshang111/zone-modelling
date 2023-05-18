# Databricks notebook source
# MAGIC %pip install pvlib

# COMMAND ----------

from IPython.display import display
from datetime import datetime
import ipywidgets as widgets
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location


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

        Last Step. Return a dataframe with the hourly Pmp and curve_info

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
        self.module_parameters = self.CECMODS[self.module_name]
        self.temperature_model_parameters = None

    def get_tilt_azimuth(self):
        """
        This function needs to return values for surface_tilt and surface_azimuth for every time step in the simulation data.
        """
        if self.array_type == "SAT":
            # load solar position and tracker orientation for use in pvsystem object
            sat_mount = pvsystem.SingleAxisTrackerMount(axis_tilt=self.sat_axis_tilt,  # flat array
                                                        axis_azimuth=self.sat_axis_azimuth,  # north-facing azimuth
                                                        max_angle=self.sat_max_angle,  # a common maximum rotation
                                                        backtrack=self.sat_backtrack,
                                                        gcr=self.sat_mod_length / self.sat_pitch)
            # created for use in pvfactors timeseries
            orientation = sat_mount.get_orientation(self.weather['solar_zenith'],self.weather['solar_azimuth'])
            self.surface_tilt = orientation['surface_tilt']
            self.surface_azimuth = orientation['surface_azimuth']

        else:
            self.surface_tilt = self.mav_tilt
            self.surface_azimuth = self.mav_azimuth

    def get_met_data(self):
        """
        function to extract relevant data out of Solcast dataframe.
        """
        # write code here.
        return
    
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
            self.temperature_module_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        else:
            self.temperature_module_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']


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
        self.sim_data = self.sim_data.append(self.system.get_irradiance())

        #step 7
        self.sim_data["cell_temp_vmp"] = self.system.get_cell_temperature(self.sim_data['poa_global'], self.sim_data['temp_air'], self.sim_data['wind_speed'], model="sapm", effective_irradiance=None)



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


    # HOSHANG ---  just coppied this code from an earlier attempt to build it as a class. Build the ui into the Model_Chain class, similar to how we built the scopti_tools ui into the scenario class.

    def init_ui(self):
        self.met_data_widget = widgets.Dropdown(options=self.get_met_data_sources(), description='Met Data', value=None)

        self.array_type_widget = widgets.RadioButtons(options=['SAT', 'MAV'], description='Array Type', value=None)
        self.array_type_widget.observe(self.on_value_change, 'value')

        self.load_button_widget = widgets.Button(description="Load")
        self.load_button_widget.on_click()

        self.output = widgets.Output()
        display(self.met_data_widget, self.array_type_widget, self.load_button_widget, self.output)


    def on_value_change(self, change):
        with self.output:
            output.clear_output()
            print(change.new)
        # attr_name = self.__fetch_attr_name(change.owner.description)
        # setattr(self, attr_name, change.new)

    # def on_load(self):
        


    def get_met_data_sources(self):
        """
        Returns a list of table names in the data repository schema "sandbox.met_data"
        """
        return spark.sql("SHOW TABLES in sandbox.met_data").rdd.map(lambda x: x.tableName).collect()

# COMMAND ----------

init_ui()

# COMMAND ----------


