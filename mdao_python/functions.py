import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OVERALL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def compute_room_area(L, W):

    area = L*W     # Room area based on length and width
    return area


def compute_joist_volume(joist, number_joists):
    volume = number_joists * joist.height * joist.width * joist.length
    return volume


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sub-functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def compute_SSbeam_min_deflection_rate(beam, applied_uniform_load):

    L = beam.length
    E = beam.E
    I = beam.I
    q = applied_uniform_load + beam.weight_per_length

    dMax = 5*q*L**4 / (384*E*I)   # Max deflection on beam (L, EI) with uniform load q
    deflectionRate = L/dMax       # min deflection rate

    return deflectionRate


def compute_SSbeam_max_stress_ratios(beam, applied_uniform_load):

    L = beam.length
    A = beam.cross_sectional_area
    h = beam.height
    I = beam.I
    yield_strength = beam.yield_strength
    shear_strength = beam.shear_strength
    q = applied_uniform_load + beam.weight_per_length

    maxBending = q * L**2 / 8           # Magnitude of max bending moment at middle of uniformly loaded beam
    maxStressBending = (h/2)*maxBending/I    # sigma = -x2*M3(x1)/I1; maximum at x2=h/2 (top/bottom of beam)
    max_bending_stress_ratio = maxStressBending / yield_strength

    maxShear = q*L/2        # Magnitude of max shear force at either end of uniformly loaded beam
    maxStressShear = maxShear/A     # Max shear stress is max shear force / cross-sectional area
    max_shear_stress_ratio = maxStressShear / shear_strength

    return max_bending_stress_ratio, max_shear_stress_ratio


def compute_SSbeam_deflections(main_beam, applied_uniform_load, n):
    q = applied_uniform_load + main_beam.weight_per_length
    L = main_beam.length
    E = main_beam.E
    I = main_beam.I
    a = L/(n+1)                 # joist spacing
    loc = a * np.arange(n + 1)  # location where deltas must be computed (where joists meet the beam)

    # compute deflections: delta(x) = px(L^3-2Lx^2+x^3)/24EI
    deflections = (np.power(loc, 3) - 2*L*np.power(loc, 2)+L**3) * loc*q / (24*E*I)

    return deflections


def compute_joist_max_stress_ratios(joist, applied_uniform_load, J):
    # bending stress ratio
    q = applied_uniform_load + joist.weight_per_length
    max_joist_moment = q*joist.length**2/8 - J*joist.length/4
    max_joist_bending_stress = abs(joist.height/2 * max_joist_moment / joist.I)
    max_joist_bending_stress_ratio = max_joist_bending_stress / joist.yield_strength
    # shear stress ratio
    max_joist_shear_force = q*joist.length-J/2
    max_joist_shear_stress = abs(max_joist_shear_force / (joist.length*joist.width))
    max_joist_shear_stress_ratio = max_joist_shear_stress / joist.shear_strength

    return max_joist_bending_stress_ratio, max_joist_shear_stress_ratio


def compute_beam_max_stress_ratios(beam, J):
    n = len(J)                  # number of joists and J forces
    q = beam.weight_per_length  # uniform load due to beam weight
    L = beam.length             # beam length
    R = (q*L + sum(J))/2        # reaction at end of beam
    a = L / (len(J)+1)          # J spacing
    # bending stress ratio
    Jarms = np.zeros(len(J))+L/2 - a*np.arange(n)   # moment arms of J about center (L/2-ia for Li, i from 1 to n/2)
    Jmoments = J * Jarms
    max_beam_moment = R*L/2 - q*L**2/8 - sum(Jmoments[0:n//2])  # at center of beam
    max_beam_bending_stress = beam.height/2 * max_beam_moment / beam.I  # at top and bottom of cross section
    max_beam_bending_stress_ratio = abs(max_beam_bending_stress / beam.yield_strength)
    # shear stress ratio
    max_beam_shear_force = (beam.weight_per_length * beam.length + sum(J)) / 2
    max_beam_shear_stress = abs(max_beam_shear_force / beam.cross_sectional_area)
    max_beam_shear_stress_ratio = max_beam_shear_stress / beam.shear_strength

    return max_beam_bending_stress_ratio, max_beam_shear_stress_ratio


def compute_true_joist_deflections(joist, J, joist_applied_uniform_load):
    W = joist.length
    q = joist_applied_uniform_load + joist.weight_per_length
    E = joist.E
    I = joist.I

    joist_deltas = 5*q*W**4/(384*E*I) - W**3 / (48*E*I) *J  # J expected as a numpy array

    return joist_deltas


def compute_true_beam_deflections(beam, number_joists, J):
    L = beam.length
    n = number_joists
    a = L / (n+1)
    E = beam.E
    I = beam.I
    p = beam.weight_per_length
    R = (p*L + sum(J))/2
    Js = np.concatenate(([0], J))
    loc = a*np.arange(n+1)

    # compute deflections due to uniform load: delta(x) = px/24EI * (L^3-2Lx^2+x^3)
    delta_uniform = (np.power(loc, 3) - 2*L*np.power(loc, 2) + L**3)*loc*p/(24*E*I)

    # compute deflections due to Ji and add together in delta_Js
    delta_Js = np.zeros(n+1)
    for i in range(1, n+1):
        A = i*a
        B = L-A
        delta_Ji_left = Js[i]*B*loc * (L**2 - B**2 - np.power(loc, 2)) / (6*L*E*I)
        delta_Ji_right = Js[i]*B * (L/B*np.power(loc-A, 3) + (L**2-B**2)*loc - np.power(loc, 3)) / (6*L*E*I)
        delta_Ji = np.concatenate((delta_Ji_left[:i+1], delta_Ji_right[i+1:]))
        delta_Js = delta_Js + delta_Ji

    beam_deltas = delta_uniform + delta_Js

    return beam_deltas[1:]


def delta_residual(Js, beam, joist, number_joists, joist_applied_uniform_load):

    beam_deltas = compute_true_beam_deflections(beam, number_joists, Js)
    joist_deltas = compute_true_joist_deflections(joist, Js, joist_applied_uniform_load)
    residual = beam_deltas - joist_deltas
    return residual


def get_combined_problem_results(beam, joist, number_joists, joist_applied_uniform_load):
    #Solve for Js
    guessJ = np.zeros(number_joists) + joist_applied_uniform_load*joist.length/2
    sol = optimize.root(delta_residual, guessJ, args=(beam, joist, number_joists, joist_applied_uniform_load))
    solutionJs = sol.x

    # Get joist constraint values
    # deflection rate
    joist_deltas = compute_true_joist_deflections(joist, solutionJs, joist_applied_uniform_load)
    max_joist_deflection_rate = joist.length / max(joist_deltas)
    # bending stress and shear stress ratios
    max_joist_bending_stress_ratio, max_joist_shear_stress_ratio = compute_joist_max_stress_ratios(
        joist, joist_applied_uniform_load, min(solutionJs))
    # return
    joist_constraint_values = [max_joist_deflection_rate, max_joist_bending_stress_ratio, max_joist_shear_stress_ratio]

    # Get beam constraint values
    # deflection rate
    max_beam_deflection_rate = beam.length / max(joist_deltas)
    # bending stress ratio
    max_beam_bending_stress_ratio, max_beam_shear_stress_ratio = compute_beam_max_stress_ratios(beam, solutionJs)
    # return
    beam_constraint_values = [max_beam_deflection_rate, max_beam_bending_stress_ratio, max_beam_shear_stress_ratio]

    return beam_constraint_values, joist_constraint_values
