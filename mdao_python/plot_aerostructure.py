# -*- coding: utf-8 -*-
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
MDO formulations for a toy example in aerostructure
===================================================
"""
from __future__ import division, unicode_literals

from copy import deepcopy

from gemseo.api import (
    configure_logger,
    create_discipline,
    create_scenario,
    generate_n2_plot,
)
from gemseo.problems.aerostructure.aerostructure_design_space import (
    AerostructureDesignSpace,
)

configure_logger()


algo_options = {
    "xtol_rel": 1e-8,
    "xtol_abs": 1e-8,
    "ftol_rel": 1e-8,
    "ftol_abs": 1e-8,
    "ineq_tolerance": 1e-5,
    "eq_tolerance": 1e-3,
}

#############################################################################
# Create discipline
# -----------------
# First, we create disciplines (aero, structure, mission) with dummy formulas
# using the :class:`.AnalyticDiscipline` class.

aero_formulas = {
    "drag": "0.1*((sweep/360)**2 + 200 + "
    + "thick_airfoils**2-thick_airfoils -4*displ)",
    "forces": "10*sweep + 0.2*thick_airfoils-0.2*displ",
    "lift": "(sweep + 0.2*thick_airfoils-2.*displ)/3000.",
}
aerodynamics = create_discipline(
    "AnalyticDiscipline", name="Aerodynamics", expressions_dict=aero_formulas
)
struc_formulas = {
    "mass": "4000*(sweep/360)**3 + 200000 + " + "100*thick_panels +200.0*forces",
    "reserve_fact": "-3*sweep " + "-6*thick_panels+0.1*forces+55",
    "displ": "2*sweep + 3*thick_panels-2.*forces",
}
structure = create_discipline(
    "AnalyticDiscipline", name="Structure", expressions_dict=struc_formulas
)
mission_formulas = {"range": "8e11*lift/(mass*drag)"}
mission = create_discipline(
    "AnalyticDiscipline", name="Mission", expressions_dict=mission_formulas
)

disciplines = [aerodynamics, structure, mission]

#############################################################################
# We can see that structure and aerodynamics are strongly coupled:
generate_n2_plot(disciplines, save=False, show=True)

#############################################################################
# Create an MDO scenario with MDF formulation
# -------------------------------------------
# Then, we create an MDO scenario based on the MDF formulation
design_space = AerostructureDesignSpace()
scenario = create_scenario(
    disciplines=disciplines,
    formulation="MDF",
    objective_name="range",
    design_space=design_space,
    maximize_objective=True,
)
scenario.add_constraint("reserve_fact", "ineq", value=0.5)
scenario.add_constraint("lift", "eq", value=0.5)
scenario.execute({"algo": "NLOPT_SLSQP", "max_iter": 10, "algo_options": algo_options})
scenario.post_process("OptHistoryView", save=False, show=True)

#############################################################################
# Create an MDO scenario with bilevel formulation
# -----------------------------------------------
# Then, we create an MDO scenario based on the bilevel formulation
sub_scenario_options = {
    "max_iter": 5,
    "algo": "NLOPT_SLSQP",
    "algo_options": algo_options,
}
design_space_ref = AerostructureDesignSpace()

##############################################################################
# Create the aeronautics sub-scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For this purpose, we create a first sub-scenario to maximize the range
# with respect to the thick airfoils, based on the aerodynamics discipline.
design_space_aero = deepcopy(design_space_ref).filter(["thick_airfoils"])
aero_scenario = create_scenario(
    disciplines=[aerodynamics, mission],
    formulation="DisciplinaryOpt",
    objective_name="range",
    design_space=design_space_aero,
    maximize_objective=True,
)
aero_scenario.default_inputs = sub_scenario_options

##############################################################################
# Create the structure sub-scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We create a second sub-scenario to maximize the range
# with respect to the thick panels, based on the structure discipline.
design_space_struct = deepcopy(design_space_ref).filter(["thick_panels"])
struct_scenario = create_scenario(
    disciplines=[structure, mission],
    formulation="DisciplinaryOpt",
    objective_name="range",
    design_space=design_space_struct,
    maximize_objective=True,
)
struct_scenario.default_inputs = sub_scenario_options

##############################################################################
# Create the system scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Lastly, we build a system scenario to maximize the range with respect to
# the sweep, which is a shared variable, based on the previous sub-scenarios.
design_space_system = deepcopy(design_space_ref).filter(["sweep"])
system_scenario = create_scenario(
    disciplines=[aero_scenario, struct_scenario, mission],
    formulation="BiLevel",
    objective_name="range",
    design_space=design_space_system,
    maximize_objective=True,
    mda_name="MDAJacobi",
    tolerance=1e-8,
)
system_scenario.add_constraint("reserve_fact", "ineq", value=0.5)
system_scenario.add_constraint("lift", "eq", value=0.5)
system_scenario.execute(
    {"algo": "NLOPT_COBYLA", "max_iter": 7, "algo_options": algo_options}
)
system_scenario.post_process("OptHistoryView", save=False, show=True)



"""class Stage1_beam_problem2(MDODiscipline):
    def __init__(self):
        super(Stage1_beam_problem2, self).__init__()
        # Initialize the grammars to define inputs and outputs
        #x = x local
        #z = [x shared, 1       x shared, 2]
        #y_1 = y1
        #y_2 = y2
        self.input_grammar.initialize_from_data_names(["L", "W"])
        self.output_grammar.initialize_from_data_names(["obj", "c_j_1", "c_j_2", "c_j_3", "c_b_1", "c_b_2", "c_b_3", "c_r"])
        # Default inputs define what data to use when the inputs are not
        # provided to the execute method
        self.default_inputs = {
            "L": 5,
            "W": 3.5,
        }

    def _run(self):
        # The run method defines what happens at execution
        # ie how outputs are computed from inputs
        L, W = self.get_inputs_by_name(["L", "W"])
        # The ouputs are stored here
        input = Input()
        joist_area = L * W
        joist_weight_per_length = input.LVL_density * joist_area

        beam = Beam(input.room_length, input.beam_area, input.beam_height, input.beam_weight_per_length, input.steel_E,
                    input.beam_I, input.beam_yield_strength, input.beam_shear_strength)
        joist = Joist(input.joist_height, input.joist_width, input.room_width, joist_weight_per_length, input.LVL_E,
                      input.LVL_yield_strength, input.LVL_shear_strength)
        total_load = input.total_load_per_area * input.room_length * input.room_width
        joist_applied_uniform_load = total_load / input.number_joists / input.joist.length
        applied_uniform_load = total_load / beam.length / 2
        self.local_data["obj"] = 1 / (L * W)
        # self.local.data["obj2"] = number_joists * joist_width * joist_height * W
        # self.local_data["c_j_1"] = W / max_joist_deflection - 720
        # self.local_data["c_j_2"] = max_beam_bending_stress_ratio - 0.5
        # self.local_data["c_j_3"] = max_beam_shear_stress_ratio - 1/3
        self.local_data["c_b_1"] = compute_SSbeam_min_deflection_rate(input.beam, applied_uniform_load) - 720
        max_beam_bending_stress_ratio, max_beam_shear_stress_ratio = compute_SSbeam_max_stress_ratios(input.beam,
                                                                                                      applied_uniform_load)
        self.local_data["c_b_2"] = max_beam_bending_stress_ratio - 0.5
        self.local_data["c_b_3"] = max_beam_shear_stress_ratio - 1/3
        self.local_data["c_r"] = L - W
"""
