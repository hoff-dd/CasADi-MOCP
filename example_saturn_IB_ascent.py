"""
Saturn IB Ascent Example

Planar, nonrotating spherical Earth with an atmosphere.

Phases:
    Ascent Burn
    Staging Coast
    Insertion Burn

State per phase: position (rx, ry), velocity (vx, vy) and mass. The control is the thrust pitch angle measured from the
launch site vertical. The objective is the mass delivered to the parking orbit, with the dynamic pressure limit softened
by a slack variable.

Solved with scaled units to keep NLP conditioned.

"""

import math
import types
import casadi
from casadi_mocp import MultiPhaseOptimalControlProblem

EARTH_RADIUS = 6371.0e3             # m
MU = 3.986004418e14                 # m^3/s^2
G0 = 9.80665                        # m/s^2
AIR_DENSITY_SEA_LEVEL = 1.225       # kg/m^3
SCALE_HEIGHT = 8500.0               # m
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s (equatorial launch)

LIFTOFF_MASS = 590000.0          # kg
S_IB_PROPELLANT = 413100.0       # kg
S_IB_DRY_MASS = 41000.0          # kg
S_IB_THRUST = 7.58e6             # N   (8 x H-1, vacuum)
S_IB_ISP = 289.0                 # s
S_IVB_PROPELLANT = 104000.0      # kg
S_IVB_THRUST = 889600.0          # N   (J-2, vacuum)
S_IVB_ISP = 421.0                # s
DRAG_AREA = 34.2                 # m^2  (6.6 m diameter)
DRAG_COEFFICIENT = 0.3
STAGING_COAST_TIME = 6.0         # s

TARGET_ALTITUDE = 185.0e3        # m
MAX_DYNAMIC_PRESSURE = 35.0e3    # Pa

LENGTH_SCALE = EARTH_RADIUS
TIME_SCALE = math.sqrt(EARTH_RADIUS ** 3 / MU)
SPEED_SCALE = LENGTH_SCALE / TIME_SCALE
MASS_SCALE = LIFTOFF_MASS
FORCE_SCALE = MASS_SCALE * LENGTH_SCALE / TIME_SCALE ** 2
DENSITY_SCALE = MASS_SCALE / LENGTH_SCALE ** 3
PRESSURE_SCALE = FORCE_SCALE / LENGTH_SCALE ** 2
ROTATION_RATE = EARTH_ROTATION_RATE * TIME_SCALE

def add_vehicle_phase(mocp, phase_name, n_intervals, thrust, exhaust_speed,duration_guess, altitude_guess, speed_guess,
                      mass_guess, pitch_guess=0.0):
    duration = mocp.create_phase(phase_name, init=duration_guess, n_intervals=n_intervals)
    rx = mocp.add_trajectory(phase_name, 'rx', init=0.5 * speed_guess * duration_guess)
    ry = mocp.add_trajectory(phase_name, 'ry', init=1.0 + altitude_guess)
    vx = mocp.add_trajectory(phase_name, 'vx', init=speed_guess)
    vy = mocp.add_trajectory(phase_name, 'vy', init=0.2 * speed_guess)
    mass = mocp.add_trajectory(phase_name, 'mass', init=mass_guess)

    r = casadi.vertcat(rx, ry)
    v = casadi.vertcat(vx, vy)
    radius = casadi.sqrt(rx ** 2 + ry ** 2)
    altitude = radius - 1.0

    air_density = mocp.get_parameter('atmosphere_scale') * (AIR_DENSITY_SEA_LEVEL / DENSITY_SCALE) * casadi.exp(-altitude * LENGTH_SCALE / SCALE_HEIGHT)
    air_relative_velocity = v - ROTATION_RATE * casadi.vertcat(ry, -rx)
    air_speed = casadi.sqrt(casadi.sumsqr(air_relative_velocity) + 1e-8)
    dynamic_pressure = 0.5 * air_density * air_speed ** 2
    drag = dynamic_pressure * (DRAG_COEFFICIENT * DRAG_AREA / LENGTH_SCALE ** 2)

    acceleration = -r / radius ** 3 - (drag / mass) * (air_relative_velocity / air_speed)

    mocp.add_path_constraint(rx ** 2 + ry ** 2 > 1.0)

    pitch = None
    if thrust > 0.0:
        pitch = mocp.add_trajectory(phase_name, 'pitch', init=pitch_guess)
        mocp.add_path_constraint(pitch > -0.2)
        mocp.add_path_constraint(pitch < 0.5 * math.pi + 1.0)
        acceleration = acceleration + (thrust / mass) * casadi.vertcat(casadi.sin(pitch), casadi.cos(pitch))
        mocp.set_derivative(mass, -thrust / exhaust_speed)
    else:
        mocp.set_derivative(mass, 0.0)

    mocp.set_derivative(rx, vx)
    mocp.set_derivative(ry, vy)
    mocp.set_derivative(vx, acceleration[0])
    mocp.set_derivative(vy, acceleration[1])

    return types.SimpleNamespace(name=phase_name, duration=duration, r=r, v=v, rx=rx, ry=ry, vx=vx, vy=vy, mass=mass,
                                 pitch=pitch, altitude=altitude, dynamic_pressure=dynamic_pressure)

def build_problem(n_intervals=8):
    mocp = MultiPhaseOptimalControlProblem()

    mocp.add_parameter('atmosphere_scale', init=1.0)

    thrust_1 = S_IB_THRUST / FORCE_SCALE
    thrust_2 = S_IVB_THRUST / FORCE_SCALE
    exhaust_1 = S_IB_ISP * G0 / SPEED_SCALE
    exhaust_2 = S_IVB_ISP * G0 / SPEED_SCALE
    burn_time_1 = S_IB_PROPELLANT * S_IB_ISP * G0 / S_IB_THRUST / TIME_SCALE
    burn_time_2 = S_IVB_PROPELLANT * S_IVB_ISP * G0 / S_IVB_THRUST / TIME_SCALE
    staging_mass_loss = S_IB_DRY_MASS / MASS_SCALE

    boost = add_vehicle_phase(mocp, 'boost', n_intervals, thrust_1, exhaust_1,
                              burn_time_1, 30.0e3 / LENGTH_SCALE,
                              1200.0 / SPEED_SCALE, 0.7, pitch_guess=0.5)
    coast = add_vehicle_phase(mocp, 'coast', 1, 0.0, 0.0,
                              STAGING_COAST_TIME / TIME_SCALE, 70.0e3 / LENGTH_SCALE,
                              2400.0 / SPEED_SCALE, 0.23)
    insertion = add_vehicle_phase(mocp, 'insertion', n_intervals, thrust_2, exhaust_2,
                                  0.8 * burn_time_2, 150.0e3 / LENGTH_SCALE,
                                  5500.0 / SPEED_SCALE, 0.15, pitch_guess=1.4)

    # Launch
    mocp.add_constraint(mocp.start(boost.rx) == 0.0)
    mocp.add_constraint(mocp.start(boost.ry) == 1.0)
    mocp.add_constraint(mocp.start(boost.vx) == ROTATION_RATE)
    mocp.add_constraint(mocp.start(boost.vy) == 0.0)
    mocp.add_constraint(mocp.start(boost.pitch) == 0.0)
    mocp.add_constraint(mocp.start(boost.mass) == 1.0)

    # Ascent Burn to depletion
    mocp.add_constraint(boost.duration == burn_time_1)
    mocp.add_constraint(coast.duration == STAGING_COAST_TIME / TIME_SCALE)
    mocp.add_constraint(mocp.start(coast.mass) == mocp.end(boost.mass) - staging_mass_loss)
    mocp.add_constraint(mocp.start(insertion.mass) == mocp.end(coast.mass))
    mocp.add_constraint(insertion.duration > 0.05 * burn_time_2)
    mocp.add_constraint(insertion.duration < burn_time_2)

    for previous, following in ((boost, coast), (coast, insertion)):
        for name in ('rx', 'ry', 'vx', 'vy'):
            mocp.add_constraint(mocp.start(getattr(following, name)) == mocp.end(getattr(previous, name)))

    # Insertion Burn into circular parking orbit
    target_radius = 1.0 + TARGET_ALTITUDE / LENGTH_SCALE
    r_end = mocp.end(insertion.r)
    v_end = mocp.end(insertion.v)
    mocp.add_constraint(casadi.sumsqr(r_end) == target_radius ** 2)
    mocp.add_constraint(casadi.sumsqr(v_end) == 1.0 / target_radius)
    mocp.add_constraint(casadi.dot(r_end, v_end) == 0.0)

    # Aerodynamic load limit, softened by a slack variable
    slack_dynamic_pressure = mocp.add_variable('slack_dynamic_pressure', init=0.1)
    mocp.add_constraint(slack_dynamic_pressure > 0.0)
    mocp.add_objective(1.0 * slack_dynamic_pressure)
    mocp.add_path_constraint(boost.dynamic_pressure * (PRESSURE_SCALE / MAX_DYNAMIC_PRESSURE) < 1.0 + slack_dynamic_pressure)

    # Maximize Mass
    mocp.add_objective(-10.0 * mocp.end(insertion.mass))

    return mocp, (boost, coast, insertion)


def solve_ascent():
    mocp, phases = build_problem()
    atmosphere_scale = mocp.get_parameter('atmosphere_scale')

    mocp.set_value(atmosphere_scale, 0.0)
    mocp.solve()
    mocp.set_value(atmosphere_scale, 1.0)
    _, slack_values = mocp.solve()

    return mocp, phases, slack_values

def main():
    mocp, phases, slack_values = solve_ascent()
    boost, coast, insertion = phases

    print()
    print('S-IB burn time     {:9.1f} s'.format(mocp.get_value(boost.duration) * TIME_SCALE))
    print('S-IVB burn time    {:9.1f} s'.format(mocp.get_value(insertion.duration) * TIME_SCALE))
    print('propellant left    {:9.0f} kg'.format(S_IVB_PROPELLANT - mocp.get_value(insertion.duration) * TIME_SCALE
                                                 * S_IVB_THRUST / (S_IVB_ISP * G0)))
    print('mass into orbit    {:9.0f} kg'.format(mocp.get_value(mocp.end(insertion.mass)) * MASS_SCALE))

    for name, value in slack_values:
        print('slack {:24s} {:.3e}'.format(name, value))


if __name__ == '__main__':
    main()
