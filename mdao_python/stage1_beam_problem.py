from __future__ import division, unicode_literals
from numpy import array, ones
from gemseo.algos.design_space import DesignSpace
from gemseo.api import configure_logger, create_scenario, generate_n2_plot
from gemseo.core.discipline import MDODiscipline
from functions import *

configure_logger()

class Beam:
    def __init__(self, length, cross_sectional_area, height, weight_per_length, E, I, yield_strength, shear_strength):
        self.length = length
        self.cross_sectional_area = cross_sectional_area
        self.height = height
        self.weight_per_length = weight_per_length
        self.E = E
        self.I = I
        self.yield_strength = yield_strength
        self.shear_strength = shear_strength


class Joist:
    def __init__(self, height, width, length, weight_per_length, E, yield_strength, shear_strength):
        self.height = height
        self.width = width
        self.length = length
        self.weight_per_length = weight_per_length
        self.E = E
        self.I = width*height**2/12
        self.yield_strength = yield_strength
        self.shear_strength = shear_strength

class Input:
    def __init__(self):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIXED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        self.number_joists = 10
        # applied load
        self.dead_load_psf = 20.0  # pound / square foot
        self.live_load_psf = 50.0  # pound / square foot
        self.psf_to_Pa = 47.88  # psf to N/m2
        self.total_load_per_area = (self.dead_load_psf + self.live_load_psf) * self.psf_to_Pa  # N/m2 or Pa
        # central beam: ASTM A992 Grade 50 steel W8x58
        self.in2_to_m2 = 0.00064516
        self.in_to_m = 0.0254
        self.steel_E = 200e9  # Pascals
        self.beam_I = 228 * self.in2_to_m2 * self.in2_to_m2  # meters^4
        self.beam_area = 17.1 * self.in2_to_m2  # meters^2
        self.beam_height = 8.75 * self.in_to_m  # meters
        self.beam_weight_per_length = 58 * 14.594  # N / m
        self.beam_yield_strength = 345e6  # Pascals
        self.beam_shear_strength = 345e6  # Pascals
        # joist: made of Laminated veneer lumber (LVL)
        self.LVL_E = 13e9  # Pascals (https://en.wikipedia.org/wiki/Laminated_veneer_lumber#Qualities)
        self.LVL_yield_strength = 19e6  # Pascals (https://en.wikipedia.org/wiki/Laminated_veneer_lumber#Qualities)
        self.LVL_density = 510  # kg/m3 (https://www.metsawood.com/global/Products/kerto/Pages/Kerto.aspx)
        self.LVL_shear_strength = 10e6  # absolute guess
        # ~~~~~~~~~~~~~~~~~ VARIABLE - to be determined through optimization ~~~~~~~~~~~~~~~~~ #
        self.room_length = 5  # meters
        self.room_width = 3.5  # meters
        self.joist_width = 2 * self.in_to_m  # meters
        self.joist_height = 6 * self.in_to_m  # meters


class Stage1_beam_problem1(MDODiscipline):
    def __init__(self):
        super(Stage1_beam_problem1, self).__init__()
        self.input_grammar.initialize_from_data_names(["L", "W"])
        self.output_grammar.initialize_from_data_names(["obj"])
        self.default_inputs = {
            "L": ones(1) * 5,
            "W": ones(1) * 3.5,
        }

    def _run(self):
        # The run method defines what happens at execution
        # ie how outputs are computed from inputs
        L, W = self.get_inputs_by_name(["L", "W"])
        # The ouputs are stored here
        self.local_data["obj"] = array([1 / (L[0] * W[0])])
        # self.local.data["obj2"] = number_joists * joist_width * joist_height * W
        # self.local_data["c_j_1"] = W / max_joist_deflection - 720
        # self.local_data["c_j_2"] = max_beam_bending_stress_ratio - 0.5
        # self.local_data["c_j_3"] = max_beam_shear_stress_ratio - 1/3

class Stage1_beam_problem2(MDODiscipline):
    def __init__(self):
        super(Stage1_beam_problem2, self).__init__()
        # Initialize the grammars to define inputs and outputs
        #x = x local
        #z = [x shared, 1       x shared, 2]
        #y_1 = y1
        #y_2 = y2
        self.input_grammar.initialize_from_data_names(["L", "W"])
        self.output_grammar.initialize_from_data_names(["c_b_1"])
        # Default inputs define what data to use when the inputs are not
        # provided to the execute method
        self.default_inputs = {
            "L": ones(1) * 5,
            "W": ones(1) * 3.5,
        }

    def _run(self):
        # The run method defines what happens at execution
        # ie how outputs are computed from inputs
        L, W = self.get_inputs_by_name(["L", "W"])
        # The ouputs are stored here
        input = Input()
        # joist_area = L[0] * W[0]
        joist_weight_per_length = input.LVL_density * input.beam_area
        beam = Beam(L[0], input.beam_area, input.beam_height, input.beam_weight_per_length, input.steel_E,
                    input.beam_I, input.beam_yield_strength, input.beam_shear_strength)
        total_load = input.total_load_per_area * L[0] * W[0]
        applied_uniform_load = total_load / beam.length / 2
        # self.local.data["obj2"] = number_joists * joist_width * joist_height * W
        # self.local_data["c_j_1"] = W / max_joist_deflection - 720
        # self.local_data["c_j_2"] = max_beam_bending_stress_ratio - 0.5
        # self.local_data["c_j_3"] = max_beam_shear_stress_ratio - 1/3
        #self.local_data["c_b_1"] = array([-L[0] + 720])
        self.local_data["c_b_1"] = array([-L[0] / (5 * (applied_uniform_load + beam.weight_per_length) * L[0]**4 / (384 * beam.E * beam.I)) + 720])
        #self.local_data["c_b_1"] = array([compute_SSbeam_min_deflection_rate(beam, applied_uniform_load) - 720])

class Stage1_beam_problem3(MDODiscipline):
    def __init__(self):
        super(Stage1_beam_problem3, self).__init__()
        # Initialize the grammars to define inputs and outputs
        #x = x local
        #z = [x shared, 1       x shared, 2]
        #y_1 = y1
        #y_2 = y2
        self.input_grammar.initialize_from_data_names(["L", "W"])
        self.output_grammar.initialize_from_data_names(["c_b_2", "c_b_3"])
        # Default inputs define what data to use when the inputs are not
        # provided to the execute method
        self.default_inputs = {
            "L": ones(1) * 5,
            "W": ones(1) * 3.5,
        }

    def _run(self):
        # The run method defines what happens at execution
        # ie how outputs are computed from inputs
        L, W = self.get_inputs_by_name(["L", "W"])
        # The ouputs are stored here
        input = Input()
        # joist_area = L * W
        joist_weight_per_length = input.LVL_density * input.beam_area

        beam = Beam(L[0], input.beam_area, input.beam_height, input.beam_weight_per_length, input.steel_E,
                    input.beam_I, input.beam_yield_strength, input.beam_shear_strength)
        total_load = input.total_load_per_area * L * W
        applied_uniform_load = total_load / beam.length / 2
        max_beam_bending_stress_ratio, max_beam_shear_stress_ratio = compute_SSbeam_max_stress_ratios(beam,
                                                                                                      applied_uniform_load)
        self.local_data["c_b_2"] = array([max_beam_bending_stress_ratio[0] - 0.5])
        self.local_data["c_b_3"] = array([max_beam_shear_stress_ratio[0] - 1/3])

class Stage1_beam_problem4(MDODiscipline):
    def __init__(self):
        super(Stage1_beam_problem4, self).__init__()
        self.input_grammar.initialize_from_data_names(["L", "W"])
        self.output_grammar.initialize_from_data_names(["c_r"])
        self.default_inputs = {
            "L": ones(1) * 5,
            "W": ones(1) * 3.5,
        }

    def _run(self):
        # The run method defines what happens at execution
        # ie how outputs are computed from inputs
        L, W = self.get_inputs_by_name(["L", "W"])
        self.local_data["c_r"] = array([W[0] - L[0]])





disciplines = [Stage1_beam_problem1(), Stage1_beam_problem2(), Stage1_beam_problem3(), Stage1_beam_problem4()]
design_space = DesignSpace()
design_space.add_variable("L", 1, l_b=1.0, u_b=15.0, value = ones(1)*5)
design_space.add_variable("W", 1, l_b=0.5, u_b=10.0, value = ones(1)*3.5)
scenario = create_scenario(disciplines, formulation="MDF", objective_name="obj", design_space=design_space)
scenario.add_constraint("c_b_1", "ineq")
scenario.add_constraint("c_b_2", "ineq")
scenario.add_constraint("c_b_3", "ineq")
scenario.add_constraint("c_r", "ineq")
scenario.set_differentiation_method("finite_differences", 1e-6)
scenario.execute(input_data={"max_iter": 50, "algo": "SLSQP"})
#scenario.post_process("OptHistoryView", save=False, show=True)
generate_n2_plot(disciplines, save = False, show = True)


#constraints might actually need to be with the objective function????