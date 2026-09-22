"""
Multi-phase optimal control transcription using CasADi/IPOPT.
"""

import re
import collections
import casadi

N_NODES = 5
VariableInfo = collections.namedtuple('VariableInfo', ['flag', 'phase', 'name', 'index'])

def parse_variable_name(name):
    parts = name.split('/')
    assert len(parts) >= 3
    flag = parts[0]
    assert len(flag) == 1
    phase = parts[1] if parts[1] else None
    var_name = parts[2] if parts[2] else None
    index = parts[3] if (len(parts) > 3 and parts[3]) else None
    return VariableInfo(flag, phase, var_name, index)

def is_valid_name(s):
    return re.fullmatch(r'\w+', s) is not None


class Trajectory:
    def __init__(self, phase_name, trajectory_name, n_intervals, init_value, parent_phase):
        if isinstance(init_value, casadi.DM):
            init_value = float(init_value)
        assert isinstance(init_value, float)
        self.trajectory_interior = casadi.SX.sym('T/' + phase_name + '/' + trajectory_name)
        self.start = casadi.SX.sym('S/' + phase_name + '/' + trajectory_name)
        self.end = casadi.SX.sym('E/' + phase_name + '/' + trajectory_name)
        self.derivative = None
        self.values = [init_value] * (n_intervals * (N_NODES - 1) + 1)
        self.parent_phase = parent_phase

class Scalar:
    def __init__(self, sym_name, value):
        self.symbol = casadi.SX.sym(sym_name)
        self.value = float(value)


class Phase:
    def __init__(self, phase_name, n_intervals, duration_value, parent_mocp):
        assert is_valid_name(phase_name) and n_intervals >= 1
        self.phase_name = phase_name
        self.trajectories = dict()
        self.parent_mocp = parent_mocp
        self.n_intervals = n_intervals
        self.duration_symbol = casadi.SX.sym('D/' + phase_name + '/')
        self.duration_value = duration_value

    def add_trajectory(self, trajectory_name, init_value):
        assert is_valid_name(trajectory_name)
        assert trajectory_name not in self.trajectories
        self.trajectories[trajectory_name] = Trajectory(
            self.phase_name, trajectory_name, self.n_intervals, init_value, self)
        return self.trajectories[trajectory_name]

class MultiPhaseOptimalControlProblem:
    def __init__(self):
        self.phases = dict()
        self.variables = dict()
        self.parameters = dict()

    def create_phase(self, phase_name, **kwargs):
        assert phase_name not in self.phases
        init = kwargs.get('init', 1.0)
        n_intervals = kwargs.get('n_intervals', 2)
        self.phases[phase_name] = Phase(phase_name, n_intervals, init, self)
        return self.phases[phase_name].duration_symbol

    def add_trajectory(self, phase_name, trajectory_name, **kwargs):
        init = kwargs.get('init', 0.0)
        if phase_name not in self.phases:
            self.create_phase(phase_name)
        self.phases[phase_name].add_trajectory(trajectory_name, init)
        return self.phases[phase_name].trajectories[trajectory_name].trajectory_interior

    def get_phase_duration(self, phase_name):
        if phase_name not in self.phases:
            self.create_phase(phase_name)
        return self.phases[phase_name].duration_symbol

    def add_variable(self, name, **kwargs):
        assert is_valid_name(name) and name not in self.variables
        self.variables[name] = Scalar('V//' + name, kwargs.get('init', 0.0))
        return self.variables[name].symbol

    def add_parameter(self, name, **kwargs):
        assert is_valid_name(name) and name not in self.parameters
        self.parameters[name] = Scalar('P//' + name, kwargs.get('init', 0.0))
        return self.parameters[name].symbol

    def get_parameter(self, name):
        return self.parameters[name].symbol

    def set_derivative(self, x, dxdt_fn):
        if isinstance(dxdt_fn, float):
            dxdt_fn = casadi.SX(dxdt_fn)
        assert x.is_leaf() and x.numel() == 1 and dxdt_fn.numel() == 1
        info = parse_variable_name(x.name())
        assert info.flag == 'T'
        assert self.phases[info.phase].trajectories[info.name].derivative is None
        self.phases[info.phase].trajectories[info.name].derivative = dxdt_fn