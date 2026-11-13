"""
Saturn IB Ascent Example

Planar, nonrotating spherical Earth with an atmosphere.

Phases:
    Ascent Burn
    Staging Coast
    Insertion Burn
"""

import math
import types
import casadi
from casadi_mocp import MultiPhaseOptimalControlProblem

EARTH_RADIUS = 6371.0e3             # m
MU = 3.986004418e14                 # m^3/s^2
AIR_DENSITY_SEA_LEVEL = 1.225       # kg/m^3
AIR_PRESSURE_SEA_LEVEL = 101325.0   # Pa
SCALE_HEIGHT = 7640.0               # m
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s
LAUNCH_LATITUDE = math.radians(28.56)  # Cape Canaveral SLC-40

S1_DRY_MASS = 25600.0            # kg
S1_PROPELLANT = 395700.0         # kg
S2_DRY_MASS = 3900.0             # kg
S2_PROPELLANT = 92670.0          # kg
S1_ENGINE_COUNT = 9
MERLIN_EXHAUST_SPEED = 3076.0    # m/s
MERLIN_EXIT_AREA = 0.7           # m^2
MERLIN_EXIT_PRESSURE = 60000.0   # Pa
MERLIN_MAX_MASS_FLOW = 287.0     # kg/s per engine
MVAC_EXHAUST_SPEED = 3413.0      # m/s (effective, vacuum)
MVAC_MAX_MASS_FLOW = 287.0       # kg/s
DRAG_AREA = 5.3                  # m^2  (Cd * A)
STAGING_COAST_TIME = 11.0        # s
PAYLOAD_GUESS = 15000.0          # kg

TARGET_ALTITUDE = 280.0e3               # m
TARGET_INCLINATION = math.radians(51.6)
MAX_DYNAMIC_PRESSURE = 25.0e3           # Pa

STACK_MASS = S1_DRY_MASS + S1_PROPELLANT + S2_DRY_MASS + S2_PROPELLANT
LENGTH_SCALE = EARTH_RADIUS
TIME_SCALE = math.sqrt(EARTH_RADIUS ** 3 / MU)
SPEED_SCALE = LENGTH_SCALE / TIME_SCALE
MASS_SCALE = STACK_MASS + PAYLOAD_GUESS
MASS_FLOW_SCALE = MASS_SCALE / TIME_SCALE
FORCE_SCALE = MASS_SCALE * LENGTH_SCALE / TIME_SCALE ** 2
DENSITY_SCALE = MASS_SCALE / LENGTH_SCALE ** 3
PRESSURE_SCALE = FORCE_SCALE / LENGTH_SCALE ** 2
ROTATION_RATE = EARTH_ROTATION_RATE * TIME_SCALE

LAUNCH_SITE = casadi.DM([math.cos(LAUNCH_LATITUDE), 0.0, math.sin(LAUNCH_LATITUDE)])
EARTH_SPIN = casadi.DM([0.0, 0.0, ROTATION_RATE])

def solve_launch_direction():
    target_radius = 1.0 + TARGET_ALTITUDE / LENGTH_SCALE
    target_angular_momentum = math.sqrt(target_radius)
    target_energy = -0.5 / target_radius

    mocp = MultiPhaseOptimalControlProblem()
    rx = mocp.add_variable('rx', init=float(LAUNCH_SITE[0]))
    ry = mocp.add_variable('ry', init=float(LAUNCH_SITE[1]))
    rz = mocp.add_variable('rz', init=float(LAUNCH_SITE[2]))
    vx = mocp.add_variable('vx', init=0.01)
    vy = mocp.add_variable('vy', init=0.01)
    vz = mocp.add_variable('vz', init=0.01)
    r = casadi.vertcat(rx, ry, rz)
    v = casadi.vertcat(vx, vy, vz)

    angular_momentum = casadi.cross(r, v)
    energy = 0.5 * casadi.sumsqr(v) - 1.0 / casadi.norm_2(r)

    mocp.add_objective(casadi.sumsqr(r - LAUNCH_SITE))
    mocp.add_objective((casadi.norm_2(angular_momentum) - target_angular_momentum) ** 2)
    mocp.add_objective((angular_momentum[2] - target_angular_momentum * math.cos(TARGET_INCLINATION)) ** 2)
    mocp.add_objective((energy - target_energy) ** 2)
    mocp.solve()

    velocity = casadi.DM([mocp.get_value(vx), mocp.get_value(vy), mocp.get_value(vz)])
    up = LAUNCH_SITE
    downrange = velocity - casadi.dot(velocity, up) * up
    downrange = downrange / casadi.norm_2(downrange)

    return types.SimpleNamespace(up=up, downrange=downrange, crossrange=casadi.cross(up, downrange))


def add_vehicle_phase(mocp, phase_name, n_intervals, frame, engine, duration_guess, altitude_guess, downrange_guess,
                      speed_guess, climb_angle_guess, mass_guess, pitch_guess=0.0):
    duration = mocp.create_phase(phase_name, init=duration_guess, n_intervals=n_intervals)

    position_guess = (1.0 + altitude_guess) * frame.up + downrange_guess * frame.downrange
    velocity_guess = speed_guess * (math.sin(climb_angle_guess) * frame.up + math.cos(climb_angle_guess) * frame.downrange)
    rx = mocp.add_trajectory(phase_name, 'rx', init=float(position_guess[0]))
    ry = mocp.add_trajectory(phase_name, 'ry', init=float(position_guess[1]))
    rz = mocp.add_trajectory(phase_name, 'rz', init=float(position_guess[2]))
    vx = mocp.add_trajectory(phase_name, 'vx', init=float(velocity_guess[0]))
    vy = mocp.add_trajectory(phase_name, 'vy', init=float(velocity_guess[1]))
    vz = mocp.add_trajectory(phase_name, 'vz', init=float(velocity_guess[2]))
    mass = mocp.add_trajectory(phase_name, 'mass', init=mass_guess)

    r = casadi.vertcat(rx, ry, rz)
    v = casadi.vertcat(vx, vy, vz)
    radius = casadi.sqrt(casadi.sumsqr(r))
    altitude = radius - 1.0

    atmosphere_scale = mocp.get_parameter('atmosphere_scale')
    atmosphere_fraction = atmosphere_scale * casadi.exp(-altitude * LENGTH_SCALE / SCALE_HEIGHT)
    air_density = (AIR_DENSITY_SEA_LEVEL / DENSITY_SCALE) * atmosphere_fraction
    air_pressure = (AIR_PRESSURE_SEA_LEVEL / PRESSURE_SCALE) * atmosphere_fraction
    air_relative_velocity = v - casadi.cross(EARTH_SPIN, r)
    air_speed = casadi.sqrt(casadi.sumsqr(air_relative_velocity) + 1e-8)
    dynamic_pressure = 0.5 * air_density * air_speed ** 2
    drag = dynamic_pressure * (DRAG_AREA / LENGTH_SCALE ** 2)

    acceleration = -r / radius ** 3 - (drag / mass) * (air_relative_velocity / air_speed)

    # Stay above the ground
    mocp.add_path_constraint(rx ** 2 + ry ** 2 + rz ** 2 > 1.0)

    pitch = yaw = None
    if engine is not None:
        pitch = mocp.add_trajectory(phase_name, 'pitch', init=pitch_guess)
        yaw = mocp.add_trajectory(phase_name, 'yaw', init=0.0)
        mocp.add_path_constraint(pitch > -0.2)
        mocp.add_path_constraint(pitch < 0.5 * math.pi + 1.0)
        mocp.add_path_constraint(yaw > -0.5)
        mocp.add_path_constraint(yaw < 0.5)

        mass_flow = engine.count * (engine.max_mass_flow / MASS_FLOW_SCALE)
        thrust = mass_flow * (engine.exhaust_speed / SPEED_SCALE) \
                 + engine.count * (engine.exit_area / LENGTH_SCALE ** 2) \
                 * (engine.exit_pressure / PRESSURE_SCALE - air_pressure)
        thrust_direction = casadi.cos(pitch) * frame.up + casadi.sin(pitch) * (casadi.cos(yaw) * frame.downrange + casadi.sin(yaw) * frame.crossrange)
        acceleration = acceleration + (thrust / mass) * thrust_direction
        mocp.set_derivative(mass, -mass_flow)
    else:
        mocp.set_derivative(mass, 0.0)

    mocp.set_derivative(rx, vx)
    mocp.set_derivative(ry, vy)
    mocp.set_derivative(rz, vz)
    mocp.set_derivative(vx, acceleration[0])
    mocp.set_derivative(vy, acceleration[1])
    mocp.set_derivative(vz, acceleration[2])

    return types.SimpleNamespace(name=phase_name, duration=duration, r=r, v=v, rx=rx, ry=ry, rz=rz,
                                 vx=vx, vy=vy, vz=vz, mass=mass, pitch=pitch, yaw=yaw,
                                 altitude=altitude, dynamic_pressure=dynamic_pressure)

def build_problem(frame, n_intervals=8):
    mocp = MultiPhaseOptimalControlProblem()

    mocp.add_parameter('atmosphere_scale', init=1.0)

    merlin = types.SimpleNamespace(count=S1_ENGINE_COUNT, exhaust_speed=MERLIN_EXHAUST_SPEED,
                                   max_mass_flow=MERLIN_MAX_MASS_FLOW, exit_area=MERLIN_EXIT_AREA,
                                   exit_pressure=MERLIN_EXIT_PRESSURE)
    merlin_vacuum = types.SimpleNamespace(count=1, exhaust_speed=MVAC_EXHAUST_SPEED, max_mass_flow=MVAC_MAX_MASS_FLOW,
                                          exit_area=0.0, exit_pressure=0.0)

    burn_time_1 = S1_PROPELLANT / (S1_ENGINE_COUNT * MERLIN_MAX_MASS_FLOW) / TIME_SCALE
    burn_time_2 = S2_PROPELLANT / MVAC_MAX_MASS_FLOW / TIME_SCALE
    coast_time = STAGING_COAST_TIME / TIME_SCALE
    upper_stage_guess = (S2_DRY_MASS + S2_PROPELLANT + PAYLOAD_GUESS) / MASS_SCALE

    boost = add_vehicle_phase(mocp, 'boost', n_intervals, frame, merlin,
                              burn_time_1, 30.0e3 / LENGTH_SCALE, 30.0e3 / LENGTH_SCALE,
                              1500.0 / SPEED_SCALE, 0.5, 0.6, pitch_guess=0.5)
    coast = add_vehicle_phase(mocp, 'coast', 1, frame, None,
                              coast_time, 70.0e3 / LENGTH_SCALE, 80.0e3 / LENGTH_SCALE,
                              2400.0 / SPEED_SCALE, 0.4, upper_stage_guess)
    insertion = add_vehicle_phase(mocp, 'insertion', n_intervals, frame, merlin_vacuum,
                                  burn_time_2, 180.0e3 / LENGTH_SCALE, 800.0e3 / LENGTH_SCALE,
                                  5500.0 / SPEED_SCALE, 0.1, 0.6 * upper_stage_guess, pitch_guess=1.4)

    payload = mocp.add_variable('payload', init=PAYLOAD_GUESS / MASS_SCALE)
    mocp.add_constraint(payload > 0.0)

    # Launch
    launch_velocity = casadi.cross(EARTH_SPIN, LAUNCH_SITE)
    for name, value in (('rx', LAUNCH_SITE[0]), ('ry', LAUNCH_SITE[1]), ('rz', LAUNCH_SITE[2]),
                        ('vx', launch_velocity[0]), ('vy', launch_velocity[1]),
                        ('vz', launch_velocity[2])):
        mocp.add_constraint(mocp.start(getattr(boost, name)) == float(value))
    mocp.add_constraint(mocp.start(boost.pitch) == 0.0)
    mocp.add_constraint(mocp.start(boost.mass) == STACK_MASS / MASS_SCALE + payload)

    # First stage
    upper_stage_mass = (S2_DRY_MASS + S2_PROPELLANT) / MASS_SCALE + payload
    mocp.add_constraint(mocp.end(boost.mass) > upper_stage_mass + S1_DRY_MASS / MASS_SCALE)
    mocp.add_constraint(mocp.start(coast.mass) == upper_stage_mass)
    mocp.add_constraint(boost.duration > 0.5 * burn_time_1)
    mocp.add_constraint(boost.duration < 2.0 * burn_time_1)
    mocp.add_constraint(coast.duration == coast_time)

    # Second stage
    mocp.add_constraint(mocp.start(insertion.mass) == mocp.end(coast.mass))
    mocp.add_constraint(mocp.end(insertion.mass) > S2_DRY_MASS / MASS_SCALE + payload)
    mocp.add_constraint(insertion.duration > 0.05 * burn_time_2)
    mocp.add_constraint(insertion.duration < 2.0 * burn_time_2)

    for previous, following in ((boost, coast), (coast, insertion)):
        for name in ('rx', 'ry', 'rz', 'vx', 'vy', 'vz'):
            mocp.add_constraint(mocp.start(getattr(following, name))
                                == mocp.end(getattr(previous, name)))

    # Insertion Burn into circular parking orbit
    target_radius = 1.0 + TARGET_ALTITUDE / LENGTH_SCALE
    r_end = mocp.end(insertion.r)
    v_end = mocp.end(insertion.v)
    angular_momentum = casadi.cross(r_end, v_end)
    mocp.add_constraint(casadi.sumsqr(r_end) == target_radius ** 2)
    mocp.add_constraint(casadi.sumsqr(v_end) == 1.0 / target_radius)
    mocp.add_constraint(casadi.dot(r_end, v_end) == 0.0)
    mocp.add_constraint(angular_momentum[2] == math.sqrt(target_radius) * math.cos(TARGET_INCLINATION))

    # Aerodynamic load limit, softened by a slack variable
    slack_dynamic_pressure = mocp.add_variable('slack_dynamic_pressure', init=0.1)
    mocp.add_constraint(slack_dynamic_pressure > 0.0)
    mocp.add_objective(1.0 * slack_dynamic_pressure)
    mocp.add_path_constraint(boost.dynamic_pressure * (PRESSURE_SCALE / MAX_DYNAMIC_PRESSURE) < 1.0 + slack_dynamic_pressure)
    for phase in (boost, coast):
        mocp.add_path_constraint(phase.dynamic_pressure * (PRESSURE_SCALE / MAX_DYNAMIC_PRESSURE) < 1.0 + slack_dynamic_pressure)

    # Maximize Mass
    mocp.add_objective(-10.0 * payload)

    return mocp, (boost, coast, insertion), payload

def solve_ascent():
    frame = solve_launch_direction()
    mocp, phases, payload = build_problem(frame)
    atmosphere_scale = mocp.get_parameter('atmosphere_scale')

    mocp.set_value(atmosphere_scale, 0.0)
    mocp.solve()
    mocp.set_value(atmosphere_scale, 1.0)

    _, slack_values = mocp.solve()

    return mocp, phases, payload, slack_values

def main():
    mocp, phases, payload, slack_values = solve_ascent()
    boost, coast, insertion = phases

    payload_mass = mocp.get_value(payload) * MASS_SCALE
    s1_propellant_used = (mocp.get_value(mocp.start(boost.mass))- mocp.get_value(mocp.end(boost.mass))) * MASS_SCALE
    s2_propellant_used = (mocp.get_value(mocp.start(insertion.mass)) - mocp.get_value(mocp.end(insertion.mass))) * MASS_SCALE

    print()
    print('first stage burn   {:9.1f} s'.format(mocp.get_value(boost.duration) * TIME_SCALE))
    print('second stage burn  {:9.1f} s'.format(mocp.get_value(insertion.duration) * TIME_SCALE))
    print('S1 propellant left {:9.0f} kg'.format(S1_PROPELLANT - s1_propellant_used))
    print('S2 propellant left {:9.0f} kg'.format(S2_PROPELLANT - s2_propellant_used))
    print('lift-off mass      {:9.0f} kg'.format(STACK_MASS + payload_mass))
    print('payload into orbit {:9.0f} kg'.format(payload_mass))
    for name, value in slack_values:
        print('slack {:24s} {:.3e}'.format(name, value))

if __name__ == '__main__':
    main()