#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the CalipyProbModel base class that is useful for representing,
modifying, analyzing, and optimizing instrument models.

The classes are
    CalipyProbModel: Short for Calipy Probabilistic Model. Base class providing
    functionality for integrating instruments, effects, and data into one 
    CalipyProbModel object. Allows simulation, inference, and illustration of 
    deep instrument models.
  
The CalipyProbModel class provides a comprehensive representation of the interactions
between instruments and data. It contains several subobjects representing the
physical instrument, random and systematic effects originating from instrument
or environment, unknown parameters and variables, constraints, and the objective
function. All of these subobjects form a probabilistic model that can be sampled
and conditioned on measured data. For more information, see the separate
documentation entries the CalipyProbModel class, for the subobjects, or the tutorial.        
        

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    CalipyProbModel class ----------------------------------------------------
"""


import pyro

import numpy as np

from hh2e_setup_components import  PlantComponentCollector
from hh2e_setup_variables import VariableCollector
from hh2e_setup_constraints import ConstraintCollector
from hh2e_setup_dynamics import DynamicsCollector
from hh2e_setup_control import ControlCollector
from hh2e_setup_optimizer import OptimizationCollector
from hh2e_setup_illustrations import IllustrationCollector
from hh2e_setup_iterator import IterationCollector
from hh2e_setup_support_funs import SupportCollector, setup_info_string
import hh2e_setup_support_funs as sf 


# Plant class determines attributes and methods for summarizing and accessing
# components, variables, and system dynamics of the plant.
class HH2EPlant(pyro.nn.PyroModule):
    """ Collector class aggregating all information and methods related to a plant 
    into an object accessible via hh2e_plant. This objects attributes fullfill 
    several completely disjoint roles. They span from providing a summary of physical
    plant components or variables involved in the plant design and temporal evolution over
    constraints formalizing dynamical and legal requirements to plant control schemes
    and generating inllustrations. All these subobjects come with their own methods.

    :param design_params: Dictionary containing info necessary to specify hh2e_plant uniquely.
        Check out the :func:`hh2e_setup_support_funs.invoke_standard_params` function of the 
        :mod:`hh2e_setup_support_funs` module to see an example of a complete 
        set of design parameters and possible values
    :type design_params: dictionary
    :raise hh2e_module.InvalidDesignParamsError: If the design parameters are invalid.
    :param data_info: dictionary of info specifying data completely, including numerical info on
        the keys 'data_start' and 'data_end'. The source of the data is specified via the key
        'PG_data_source', more info can be found in the documentation of the function
        :func:`hh2e_setup_support_funs.invoke_standard_params`.
    :type data_info: dictionary
    :raise hh2e_module.InvalidDataInfoError: If the data_info dict parameters are invalid.
    :return: Instance of the HH2EPlant class built on the basis of information compiled
        in the design_params and data_info dictionaries and prior information.
    :rtype: HH2EPlant
    
    
    The subobjects of the HH2EPlant class are:

    .. list-table:: Subobjects
       :widths: 25 70 70
       :header-rows: 1
    
       * - Name
         - Interpretation
         - Attributes, Methods
       * - components
         - This module provides functionality for the basic setup of hh2e_plant components.
           The components represent central entities involved in the operation of the power
           plant and include physical objects like batteries and electrolysers and non-
           phyiscal objects like the sport market.
           Contains subsubobjects [gc, ppa, el, hts, bat_nas, bat_li, bat_bx, mkt, tfs]
         - * print_info() = print information about the subobject
         
           for each of the subsubobjects:
               
           * print_info() = print information about the subsubobject
           * print_params() = print all parameters associated to the subsubobject
           * params = dictionary of parameters of the subsubobject
       * - variables
         - This module provides functionality for the basic setup of hh2e_plant variables.
           The variables represent central numerical entities involved in the operation of
           the power plant. They include constants arising from plant design and technical
           constraints as well as dynamic quantities that arise from time-varying external
           influences or responses to system inputs.
           Contains subsubobjects [system_vars, external_vars, decision_vars, 
           constraint_vars, response_vars, state_vars] 
         - * print_info() = print information about the subobject
           * update_variables() = imprint variables from dictionaries into attributes
           * var_list = list of all subsubobjects
         
           for each of the subsubobjects:
               
           * print_info() = print information about the subsubobject
           * print_vars() = print all variables associated to the subsubobject
           * populate() = imprint dictionary into subsubobject attributes
           * variable_dict = dictionary of all variables including values and explanation
           * variable_list = list of all variables
           * nr_vars = nr of variables in subsubobject
           * each variable is an attribute
       * - constraints
         - This module provides functionality for the setup of classes and functions that
           represent the imposition and testing procedures for constraints. These constraints
           represent information available about internal system dynamics, physical impossibilities,
           balancing equations, and legal requirements. 
         - * print_info() = print information about the subobject
           * print_constraints() = print all constraints from the dictionary
           * check_all_true() = check if a statement holds for all involved variables
           * check_constraints() = check if all constraints are satisfied
           * constraint_list = list of all constraints
           * constraint_dict = dictionary of all constraints
       * - dynamics
         - This module provides functionality for the basic setup of hh2e_plant dynamics.
           The dynamics are represented by a set of dictionaries relating vector components
           to system states and methods for evolving the system state in time. The functions
           step_forward and run_plant can be called from the hh2e_plant class by inputting
           an initial state and actions resp. policies to perform a full simulation of a
           hh2e_plant object.
         - * print_info() = print information about the subobject
           * step_forward() = evolve the plant for one step based on current state and an action
           * run_plant() = evolve the plant over multiple timesteps based on a policy
           * Delta_T = length of a timestep [in h]
           * dim_state = dimension of the state vector
           * dim_action = dimension of the action vector
           * initial_state = the initial values of the state variables at first timestep 
           * state_index_dict = dictionary linking entries in a state vector and their meaning
           * action_index_dict = dictionary linking entries in an action vector and their meaning 
       * - control
         - This module provides functionality for the basic setup of hh2e_plant control.
           The control policies are represented by set of functions mapping system states
           to actions and methods for constructing policies according to optimization rules.
           Policies are collected in a dictionary, can be activated and passed as an 
           argument into a hh2e_plant.run_plant() function to explore its performance.
           Further methods allow for the manipulation of existing policies and the creation of
           new ones.
         - * print_info() = print information about the subobject
           * active_policy() = print the currently active policy
           * activate_policy() = activate a pre-implemented policy from the policy_list
           * policy_list = list of all pre-implemented policies
           * policy_dict = dictionary containing policy functions
       * - illustrations
         - This module provides plotting functionality for the HH2EPlant class. The illustration
           functions range from normal lineplots of decision variables through scatterplots 
           of multidimensional state variables to simple summary reports in textform.
           The functions are integrated into an IllustrationCollector object that can
           be accessed and made to plot illustrations via commands of the type 
           hh2e_plant.illustrations.plot_decision_vars().Further methods allow for the 
           printing and saving of standardized reports to evaluate plant designs and controls.
         - * plot_state_vars() = plot all state variables
           * plot_state_vars_joint() = plot huge panel of joint plots of state variables
           * plot decision_vars() = plot all decision variables
           * plot_decision_vars_joint() = plot huge panel of joint plots of decision variables
           * plot_var() = plot a specific variable
           * plot_vars_boxplot() = plot multiple variables in a boxplot
           * plot_vars_joint() = plot a jointplot of two variables
        
        
        
    The methods of the HH2EPlant class are:
        load_emhires: Load EMHIRES dataset already preprocessed to be compatible
            with numpy and pandas.
    
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        #
        # Imports and definitions ------------------------------------------------
        #
        # i) Imports
        import hh2e_module
        import numpy as np
        #
        # ii) Definitions 
        # # Overall design parameters
        design_params=dict()    # Dictionary of main parameters of hh2e plant design
        design_params['DELTA_T'] = [1, ' Length of a time step for simulation, [in h]']
        design_params['P_GC']=[100,'Maximum power intake (MPI) through the grid connection, [in MW]']
        design_params['NR_UNIT_EL'] = [3,'Number of installed Electrolysis units']
        design_params['P_PPA'] = [np.array([100,0,150]),('Distribution of PPA among installed capacities'
                                              ' of power sources [wind (onshore), wind (offshore), solar], [in MW]')]
        design_params['C_NAS'] = [40,'Installed capacity of NaS batteries in MWh']
        design_params['C_LI'] = [5,'Installed capacity of Li batteries in MWh']
        design_params['C_BX'] = [1,'Installed capacity of X batteries in MWh']
        design_params['DIM_HTS'] = [np.array([240,40]),'Installed capacity and maximum power outflow (MPO) of HTS in [MWh, MW]']
        design_params['P_OUT_HTS'] = [10,'Designated outflow of HTS power to industrial customers in MWh']
        #
        # # Data definition
        data_info = dict()
        data_info['data_start'] = 0
        data_info['data_end'] = 1000
        #
        # Invocation and exploration of plant -----------------------------------
        #    
        # i) Invoke plant  
        hh2e_plant=hh2e_module.HH2EPlant(design_params, data_info)
        #
        # ii) Look through the HH2EPlant class attributes
        # hh2e_plant view with object explorer
        # hh2e_plant.components.  <- see what goes there 
        hh2e_plant.components.el
        hh2e_plant.components.el.print_params()
        hh2e_plant.components.print_info()
        #
        # hh2e_plant.variables. <- see what goes there
        hh2e_plant.variables.system_vars.print_vars()
        hh2e_plant.variables.system_vars.P_EL
        hh2e_plant.variables.external_vars.print_vars()
        hh2e_plant.variables.decision_vars.print_vars()
        hh2e_plant.variables.decision_vars.x_in_el
        hh2e_plant.variables.constraint_vars.print_vars()
        hh2e_plant.variables.response_vars.print_vars()
        hh2e_plant.variables.response_vars.soe_bat_li
        #
        # hh2e_plant.constraints. <- see what goes there
        hh2e_plant.constraints.print_constraints()
        hh2e_plant.constraints.constraint_dict['bound_lb_x_buyfrom_market']
        _ = hh2e_plant.constraints.check_constraints()
        hh2e_plant.constraints.check_constraints()
        #
        #Setup control and run -------------------------------------------------
        #    
        # hh2e_plant.control. <- see what goes there
        hh2e_plant.control.policy_dict
        hh2e_plant.control.active_policy
        hh2e_plant.control.activate_policy('pseudocode_policy_1')
        hh2e_plant.control.active_policy
        #    
        # hh2e_plant.dynamics. <- see what goes there
        hh2e_plant.dynamics.initial_state
        hh2e_plant.dynamics.step_forward()
        hh2e_plant.dynamics.state_index_dict
        hh2e_plant.dynamics.state_action_dict
        #
        # Run the plant
        hh2e_plant.run_plant()
        hh2e_plant.variables.decision_vars.x_in_el
        #
        # Then run the definitions of the plotting functions
        hh2e_plant.illustrations.plot_decision_vars(hh2e_plant, save_name = 'DEMO')
        hh2e_plant.illustrations.plot_state_vars(hh2e_plant, save_name = 'DEMO')
    
    """
    
    
    # i) Initialization
    
    def __init__(self, design_params, data_info):
        super(HH2EPlant, self).__init__()
        self.__version__ = '1.0.0'
        self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(1,), dtype=np.float32)
        
        
        
        # Create plant components
        self.support = SupportCollector(self)
        data_info = sf.CompletedDataInfo(self, data_info,design_params)
        self.design_params = design_params
        self.data_info = data_info
        self.components = PlantComponentCollector(self)
        self.variables = VariableCollector(self)
        self.constraints = ConstraintCollector(self)
        self.dynamics = DynamicsCollector(self)
        self.optimizer = OptimizationCollector(self)
        self.control = ControlCollector(self )
        self.illustrations = IllustrationCollector(self )
        self.iterator = IterationCollector(self)
        self.dict_all_tags = dict()
        info_string = setup_info_string(self,context = "hh2e_plant")
        self.info = 'Collector object containing all info pertaining to the hh2e_plant' + info_string
        
    
    # ii) Dynamics functionality
    
    def run_plant(self):
        self.control.activate_current_policy()
        _ = self.dynamics.run_plant(self, self.dynamics.initial_state, self.control.active_policy[0])
        pass
        
    def step(self, action):
        pass


    def reset(self):
        pass
        

    # iii) Illustrative functionality
    
    def render(self, reward, mode='console'):
        pass        