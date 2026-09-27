"""
Multi-phase optimal control transcription using CasADi/IPOPT.
"""

import re
import inspect
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

def get_all_SX_leaves(e):
    result = []
    stack = [e]
    while stack:
        x = stack.pop()
        if x.is_leaf():
            if x.is_symbolic() and not any(casadi.is_equal(x, r) for r in result):
                result.append(x)
        else:
            for i in range(x.n_dep()):
                stack.append(x.dep(i))
    return result

def has_trajectory_variable(sx_list):
    return any(parse_variable_name(v.name()).flag == 'T' for v in sx_list)

def get_unique_phase_names(sx_list):
    names = [parse_variable_name(v.name()).phase for v in sx_list]
    return list(set(n for n in names if n is not None))

__generated_names_count = dict()

def make_name_from_caller_info(caller_info):
    filename = caller_info.filename.split('/')[-1]
    if filename.endswith('.py'):
        filename = filename[:-3]
    name = ''.join(caller_info.code_context or [''])
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.replace('<', 'LEQ').replace('>', 'GEQ').replace('==', 'EQ')
    name = re.sub(r'\W+', ' ', name).strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub('.*?add.*?constraint', '', name)
    name = re.sub('.*?add.*?objective', '', name)
    name = filename + '_L' + str(caller_info.lineno) + '_' + name
    name = name.replace('__', '_')

    if name in __generated_names_count:
        __generated_names_count[name] += 1
        name += '_' + str(__generated_names_count[name])
    else:
        __generated_names_count[name] = 1
    return name


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

class Constraint:
    def __int__(self, name, is_equation, is_path_constraint, g):
        assert isinstance(g, casadi.SX) and g.numel() == 1
        self.name = name
        self.is_equation = is_equation
        self.is_path_constraint = is_path_constraint
        self.g = g
        self.phase_name = None
        self.parent_phase = None

        leaves = get_all_SX_leaves(g)
        _has_traj = has_trajectory_variable(leaves)
        phase_names = get_unique_phase_names(leaves)
        if is_path_constraint:
            assert _has_traj, 'A path constraint must use a trajectory variable.'
            assert len(phase_names) == 1, 'A path constraint must apply to exactly one phase.'
            self.phase_name = phase_names[0]
        else:
            assert not _has_traj, 'A simple constraint must not use trajectory variables. Use a path constraint.'


class Phase:
    def __init__(self, phase_name, n_intervals, duration_value, parent_mocp):
        assert is_valid_name(phase_name) and n_intervals >= 1
        self.phase_name = phase_name
        self.trajectories = dict()
        self.parent_mocp = parent_mocp
        self.n_intervals = n_intervals
        self.duration_symbol = casadi.SX.sym('D/' + phase_name + '/')
        self.duration_value = duration_value
        self.constraints = dict()

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
        self.constraints = dict()
        self.objectives = dict()

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

    def _add_constraint_impl(self, constraint_expr, name, is_path_constraint):
        assert isinstance(constraint_expr, casadi.SX) and constraint_expr.numel() == 1
        op = constraint_expr.op()
        assert op in (casadi.OP_LE, casadi.OP_LT, casadi.OP_EQ)
        is_equation = (op == casadi.OP_EQ)
        g = constraint_expr.dep(0) - constraint_expr.dep(1)
        constraint = Constraint(name, is_equation, is_path_constraint, g)
        if is_path_constraint:
            assert name not in self.phases[constraint.phase_name].constraints, 'Duplicate name'
            constraint.parent_phase = self.phases[constraint.phase_name]
            self.phases[constraint.phase_name].constraints[name] = constraint
        else:
            assert name not in self.constraints, 'Duplicate name'
            self.constraints[name] = constraint

    def add_constraint(self, constraint_expr, **kwargs):
        name = kwargs.get('name') or make_name_from_caller_info(
            inspect.getouterframes(inspect.currentframe())[1])
        self._add_constraint_impl(constraint_expr, name, False)
        return name

    def add_path_constraint(self, constraint_expr, **kwargs):
        name = kwargs.get('name') or make_name_from_caller_info(
            inspect.getouterframes(inspect.currentframe())[1])
        self._add_constraint_impl(constraint_expr, name, True)
        return name

    def add_objective(self, f, **kwargs):
        name = kwargs.get('name') or make_name_from_caller_info(
            inspect.getouterframes(inspect.currentframe())[1])
        assert not has_trajectory_variable(get_all_SX_leaves(f)), \
            'Simple objectives may not contain a trajectory variable. Use a mean/integral objective.'
        assert name not in self.objectives, 'Duplicate name'
        self.objectives[name] = f
        return name

    def remove_objective(self, objective_name):
        del self.objectives[objective_name]