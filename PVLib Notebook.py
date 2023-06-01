# Databricks notebook source
# DBTITLE 1,Install PVLib
# MAGIC %pip install pvlib

# COMMAND ----------

# DBTITLE 1,Custom System
from IPython.display import display
from datetime import datetime
from scenario_class import Scenario
from databricks.sdk.runtime import spark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pvlib as pl
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from IPython.display import display

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
        # module_list = []
        # for col in CECMODS.columns:
        #     module_list.append(col)

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
            # created for use in pvfactors timeseries
            orientation = sat_mount.get_orientation(self.sim_data['solar_zenith'],self.sim_data['solar_azimuth'])
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
        # CECMODS = pl.pvsystem.retrieve_sam(path="https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Modules.csv")
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
        
        yearly_df = system.sim_data.iloc[:105120]
        self.yearly_df = yearly_df

        IL, I0, Rs, Rsh, nNsVth = pl.pvsystem.calcparams_cec(effective_irradiance=yearly_df["poa_global"],
                        temp_cell=yearly_df['cell_temperature'], 
                        alpha_sc=alpha_sc,
                        I_L_ref=I_L_ref,
                        I_o_ref=I_o_ref,
                        R_s=R_s,
                        R_sh_ref=R_sh_ref,
                        a_ref=a_ref,
                        Adjust=Adjust)

        curve_info = pl.pvsystem.singlediode(photocurrent=IL,
                                saturation_current=I0,
                                resistance_series=Rs,
                                resistance_shunt=Rsh,
                                nNsVth=nNsVth,
                                ivcurve_pnts=15,
                                method='lambertw')

        self.yearly_df['p_mp'] = curve_info['p_mp']
        self.yearly_df['i_sc'] = curve_info['i_sc']
        self.yearly_df['v_oc'] = curve_info['v_oc']
        self.yearly_df['i_mp'] = curve_info['i_mp']
        self.yearly_df['v_mp'] = curve_info['v_mp']
        self.curves = curve_info

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
                print("Running, this may take a few seconds")
                self.get_met_data(met_data_widget.value)
                print("Met Data Loaded")

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

                self.get_module_parameters(module_widget.value)
                print("Module Parameters Loaded")
                self.get_temperature_model_parameters()
                print("Temperature Parameters Loaded")
                self.get_tilt_azimuth()
                print("Tilt Azimuth Loaded")
                latitude, longitude = self.get_location()

                self.create_pvsystem()
                print("PVSystem Created")
                self.get_irradiance()
                print("Fetched Irradiance")
                self.get_cell_temperature()
                print("Fetched Cell Temperature")
                self.get_iv_curve()
                print("Run Finished!")


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

# Code to output yearly_df using spark interface
# adds pandas datetime index back in as column
spark.createDataFrame(system.yearly_df.reset_index()).display()

# COMMAND ----------

# DBTITLE 1,Save pmp csv
pmp_df = system.yearly_df
outdir = '/dbfs/FileStore/pvlib_ui/'

pmp = outdir+ 'pmp_df' + '.csv'
pmp_df.to_csv(pmp, index=True, encoding="utf-8")

# COMMAND ----------

system.yearly_df

# COMMAND ----------

# DBTITLE 1,Scopti Tools UI
scenario = Scenario()

# COMMAND ----------

scenario.optimisation.df

# COMMAND ----------

newdf = pd.DataFrame(np.repeat(df.values, 12, axis=0))
newdf.columns = df.columns
newdf

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


