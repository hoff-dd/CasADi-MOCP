"""
Multi-phase optimal control transcription using CasADi/IPOPT.

Symbol naming:
    D/<phase>/          phase duration
    T/<phase>/<name>    trajectory "interior" placeholder (pre-transcription)
    S/<phase>/<name>    trajectory value at phase start
    E/<phase>/<name>    trajectory value at phase end
    N/<phase>/<name>/<i>  trajectory value at interior node i
    V//<name>           free scalar variable
    P//<name>           fixed parameter
"""

import re
import inspect
import collections
import casadi

# ---------------------------------------------------------------------------------------------------------------------
# Collocation constants: 5-point Lobatto (LGL) quadrature on [0, 1].
# Nodes are 0, (1 - sqrt(3/7))/2, 1/2, (1 + sqrt(3/7))/2, 1.
# Integration_Matrix row k integrates the degree-4 interpolating polynomial from node 0 to node k+1; the last row
# equals the LGL quadrature weights. Constants as in https://doi.org/10.21914/anziamj.v47i0.1033.
# ---------------------------------------------------------------------------------------------------------------------
N_NODES = 5
NODES = [0.0, 0.172673164646011, 0.5, 0.827326835353989, 1.0]
INTEGRATION_MATRIX = [
    [ 0.0677284321861569,  0.119744769343412,  -0.0217357218665581,   0.0106358242254155, -0.00370013924241453],
    [           0.040625,  0.303184183323043,    0.177777777777778,  -0.0309619611008206,             0.009375],
    [ 0.0537001392424145,  0.261586397996807,    0.377291277422114,    0.152477452878811,  -0.0177284321861569],
    [               0.05,  0.272222222222222,    0.355555555555556,    0.272222222222222,                 0.05],
]
INTEGRATION_WEIGHTS = [0.05, 0.272222222222222, 0.355555555555556, 0.272222222222222, 0.05]

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
    def __init__(self, name, is_equation, is_path_constraint, g):
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
        self.mean_objectives = dict()

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

    def inital(self, sx_trajectory): return self.start(sx_trajectory)
    def start(self, sx_trajectory): return self._get_boundary(sx_trajectory, 'start')
    def final(self, sx_trajectory): return self.end(sx_trajectory)
    def end(self, sx_trajectory): return self._get_boundary(sx_trajectory, 'end')

    def _get_boundary(self, sx, boundary_name):
        assert isinstance(sx, casadi.SX)
        if sx.numel() > 1:
            result = casadi.SX.zeros(sx.size1(), sx.size2())
            for i in range(sx.size1()):
                for j in range(sx.size2()):
                    result[i, j] = self._get_boundary(sx[i, j], boundary_name)
            return result
        if sx.is_leaf():
            assert sx.is_symbolic()
            info = parse_variable_name(sx.name())
            assert info.flag == 'T'
            t = self.phases[info.phase].trajectories[info.name]
            return t.start if boundary_name == 'start' else t.end
        leaves = [{'sym': e, 'info': parse_variable_name(e.name())} for e in get_all_SX_leaves(sx)]
        assert len(set(e['info'].phase for e in leaves if e['info'].flag == 'T')) == 1, 'Error, mixed phases'
        subs = [(e['sym'], self._get_boundary(e['sym'], boundary_name))
                for e in leaves if e['info'].flag == 'T']
        return casadi.substitute([sx], [s[0] for s in subs], [s[1] for s in subs])[0]

    def get_value(self, sx): return self.access_value(sx)

    def set_value(self, sx, value):
        assert isinstance(value, float)
        self.access_value(sx, value)

    def access_value(self, sx, new_value=None):
        assert isinstance(sx, casadi.SX) and sx.numel() == 1 and sx.is_symbolic()
        info = parse_variable_name(sx.name())
        if info.flag == 'T':
            raise RuntimeError('Paths are evaluated using Phase.interpolate()')
        assert info.flag in ('D', 'S', 'E', 'V', 'P')
        if info.flag == 'P':
            if isinstance(new_value, float): self.parameters[info.name].value = new_value
            return self.parameters[info.name].value
        if info.flag == 'V':
            if isinstance(new_value, float): self.variables[info.name].value = new_value
            return self.variables[info.name].value
        if info.flag == 'D':
            if isinstance(new_value, float): self.phases[info.phase].duration_value = new_value
            return self.phases[info.phase].duration_value
        if info.flag == 'S':
            if isinstance(new_value, float): self.phases[info.phase].trajectories[info.name].values[0] = new_value
            return self.phases[info.phase].trajectories[info.name].values[0]
        if info.flag == 'E':
            if isinstance(new_value, float): self.phases[info.phase].trajectories[info.name].values[-1] = new_value
            return self.phases[info.phase].trajectories[info.name].values[-1]
        raise RuntimeError('unreachable')

    def get_symbol_value_pairs(self):
        return \
            [(e.symbol, e.value) for e in self.variables.values()] + \
            [(e.symbol, e.value) for e in self.parameters.values()] + \
            [(e.duration_symbol, e.duration_value) for e in self.phases.values()] + \
            [(f.start, f.values[0]) for e in self.phases.values() for f in e.trajectories.values()] + \
            [(f.end, f.values[-1]) for e in self.phases.values() for f in e.trajectories.values()]

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
        name = kwargs.get('name') or make_name_from_caller_info(inspect.getouterframes(inspect.currentframe())[1])
        self._add_constraint_impl(constraint_expr, name, False)
        return name

    def add_path_constraint(self, constraint_expr, **kwargs):
        name = kwargs.get('name') or make_name_from_caller_info(inspect.getouterframes(inspect.currentframe())[1])
        self._add_constraint_impl(constraint_expr, name, True)
        return name

    def add_objective(self, f, **kwargs):
        name = kwargs.get('name') or make_name_from_caller_info(inspect.getouterframes(inspect.currentframe())[1])
        assert not has_trajectory_variable(get_all_SX_leaves(f)), \
            'Simple objectives may not contain a trajectory variable. Use a mean/integral objective.'
        assert name not in self.objectives, 'Duplicate name'
        self.objectives[name] = f
        return name

    def remove_objective(self, objective_name):
        del self.objectives[objective_name]

    def add_integral_objective(self, f, **kwargs):
        name = kwargs.get('name') or make_name_from_caller_info(inspect.getouterframes(inspect.currentframe())[1])
        leaves = get_all_SX_leaves(f)
        assert has_trajectory_variable(leaves), 'Integral objectives must use a trajectory variable.'
        phase_names = get_unique_phase_names(leaves)
        assert len(phase_names) == 1, 'Integral objectives must apply to exactly one phase.'
        phase = self.phases[phase_names[0]]
        assert name not in phase.mean_objectives, 'Duplicate name'
        phase.mean_objectives[name] = phase.duration_symbol * f
        return name

    def add_mean_objective(self, f, **kwargs):
        name = kwargs.get('name') or make_name_from_caller_info(inspect.getouterframes(inspect.currentframe())[1])
        leaves = get_all_SX_leaves(f)
        assert has_trajectory_variable(leaves), 'Mean objectives must use a trajectory variable.'
        phase_names = get_unique_phase_names(leaves)
        assert len(phase_names) == 1, 'Mean objectives must apply to exactly one phase.'
        phase = self.phases[phase_names[0]]
        assert name not in phase.mean_objectives, 'Duplicate name'
        phase.mean_objectives[name] = f
        return name

    def solve(self, **kwargs):
        transcriber = Transcriber(self)
        _, x_value = transcriber.pack_variables()
        _, p_value = transcriber.pack_parameters()
        solver = HomotopyNlpSolver(transcriber)
        result = solver.solver_loop(x_value, p_value, **kwargs)
        transcriber.unpack_variables(result[-1]['result']['x'])
        return result

BoxConstraint = collections.namedtuple('BoxConstraint', ['variable', 'bound', 'is_upper_bound'])

class Transcriber:
    def __init__(self, mocp):
        self.mocp = mocp
        self.phases = {name: TranscribedPhase(mocp.phases[name]) for name in mocp.phases}

        total_objective = casadi.SX(0.0)
        total_objective += sum(mocp.objectives.values())
        for tp in self.phases.values():
            total_objective += sum(tp.mean_objective_integrals.values())

        equality_constraints = []
        inequality_constraints = []
        for tp in self.phases.values():
            for tt in tp.trajectories.values():
                if tt.ode_node_defects is not None:
                    equality_constraints.append(tt.ode_node_defects)
            equality_constraints.extend(c.g_nodes for c in tp.constraints.values() if c.ocp_constraint.is_equation)
            inequality_constraints.extend(c.g_nodes for c in tp.constraints.values() if not c.ocp_constraint.is_equation)
        equality_constraints.extend(c.g for c in mocp.constraints.values() if c.is_equation)
        inequality_constraints.extend(c.g for c in mocp.constraints.values() if not c.is_equation)
        equality_constraints = casadi.vertcat(casadi.SX(0, 1), *equality_constraints)
        inequality_constraints = casadi.vertcat(casadi.SX(0, 1), *inequality_constraints)

        eq_box, eq_other, ineq_box, ineq_other = [], [], [], []
        for i in range(inequality_constraints.numel()):
            b = self.parse_box_constraint(inequality_constraints[i])
            (ineq_other if b is None else ineq_box).append(inequality_constraints[i] if b is None else b)
        for i in range(equality_constraints.numel()):
            b = self.parse_box_constraint(equality_constraints[i])
            (eq_other if b is None else eq_box).append(equality_constraints[i] if b is None else b)

        x_symbol, _ = self.pack_variables()
        x_names = [e.name() for e in casadi.vertsplit(x_symbol)]
        assert len(set(x_names)) == len(x_names), 'Duplicate variable name!'
        x_index = {n: i for i, n in enumerate(x_names)}
        lower_bound = [casadi.SX(-casadi.inf)] * x_symbol.numel()
        upper_bound = [casadi.SX(casadi.inf)] * x_symbol.numel()
        for b in ineq_box:
            i = x_index[b.variable.name()]
            if b.is_upper_bound:
                upper_bound[i] = casadi.fmin(upper_bound[i], b.bound)
            else:
                lower_bound[i] = casadi.fmax(lower_bound[i], b.bound)
        for b in eq_box:
            i = x_index[b.variable.name()]
            upper_bound[i] = casadi.fmin(upper_bound[i], b.bound)
            lower_bound[i] = casadi.fmax(lower_bound[i], b.bound)

        p_symbol, _ = self.pack_parameters()
        self.nlp_x = x_symbol
        self.nlp_p = p_symbol
        self.nlp_f = total_objective
        self.nlp_h = casadi.vertcat(casadi.SX(0, 1), *eq_other)
        self.nlp_g = casadi.vertcat(casadi.SX(0, 1), *ineq_other)
        self.nlp_lb = casadi.vertcat(*lower_bound) if lower_bound else casadi.SX(0, 1)
        self.nlp_ub = casadi.vertcat(*upper_bound) if upper_bound else casadi.SX(0, 1)
        self.nlp_fn_lb_ub = casadi.Function('nlp_fn_lb_ub',
                                            [self.nlp_p], [self.nlp_lb, self.nlp_ub], ['p'], ['lb', 'ub'])

    def parse_box_constraint(self, g):
        assert g.numel() == 1
        op = g.op()
        if (not g.is_symbolic()) and (op not in (casadi.OP_SUB, casadi.OP_NEG)):
            return None
        if op == casadi.OP_SUB:
            lhs, rhs = g.dep(0), g.dep(1)
        elif op == casadi.OP_NEG:
            lhs, rhs = casadi.SX(0.0), g.dep(0)
        else:
            lhs, rhs = g, casadi.SX(0.0)

        if not (lhs.is_leaf() and rhs.is_leaf()):
            return None
        is_lhs_variable = (not lhs.is_constant()) and (not lhs.name().startswith('P'))
        is_rhs_variable = (not rhs.is_constant()) and (not rhs.name().startswith('P'))
        if is_lhs_variable and is_rhs_variable:
            return None
        if (not is_lhs_variable) and (not is_rhs_variable):
            raise RuntimeError('A constraint without variables is non-sense')

        if is_lhs_variable:
            return BoxConstraint(variable=lhs, bound=rhs, is_upper_bound=True)
        return BoxConstraint(variable=rhs, bound=lhs, is_upper_bound=False)

    def pack_variables(self):
        symbols, values = [], []
        for phase_name in self.mocp.phases:
            phase = self.mocp.phases[phase_name]
            for trajectory_name in phase.trajectories:
                symbols.extend(self.phases[phase_name].trajectories[trajectory_name].node_symbols)
                values.extend(phase.trajectories[trajectory_name].values)
            symbols.append(phase.duration_symbol)
            values.append(phase.duration_value)
        for v in self.mocp.variables.values():
            symbols.append(v.symbol)
            values.append(v.value)
        return (casadi.vertcat(casadi.SX(0, 1), *symbols),
                casadi.vertcat(casadi.DM(0, 1), *[casadi.DM(v) for v in values]))

    def unpack_variables(self, x_value):
        j = 0
        for phase in self.mocp.phases.values():
            for t in phase.trajectories.values():
                for i in range(len(t.values)):
                    t.values[i] = float(x_value[j]); j += 1
            phase.duration_value = float(x_value[j]); j += 1
        for v in self.mocp.variables.values():
            v.value = float(x_value[j]); j += 1
        assert x_value.numel() == j

    def pack_parameters(self):
        symbols = [p.symbol for p in self.mocp.parameters.values()]
        values = [p.value for p in self.mocp.parameters.values()]
        return (casadi.vertcat(casadi.SX(0, 1), *symbols),
                casadi.vertcat(casadi.DM(0, 1), *[casadi.DM(v) for v in values]))


class TranscribedPhase:
    def __init__(self, ocp_phase):
        self.ocp_phase = ocp_phase
        self.mean_objective_integrals = dict()
        self.trajectories = {name: TranscribedTrajectory(ocp_phase.trajectories[name])
                             for name in ocp_phase.trajectories}

        n_phase_nodes = ocp_phase.n_intervals * (N_NODES - 1) + 1
        interiors = [t.trajectory_interior for t in ocp_phase.trajectories.values()]
        self.node_substitutions = \
            [(interiors, [self.trajectories[name].node_symbols[i] for name in ocp_phase.trajectories])
            for i in range(n_phase_nodes)]

        for tt in self.trajectories.values():
            tt.create_ode_constraints(self)

        self.constraints = {name: TranscribedPathConstraint(ocp_phase.constraints[name], self)
                            for name in ocp_phase.constraints}

        quadrature_weights = casadi.DM(INTEGRATION_WEIGHTS) / ocp_phase.n_intervals
        for objective_name, dF_dtau in ocp_phase.mean_objectives.items():
            dF_dtau_nodes = self.substitute_nodes(dF_dtau)
            interval_means = []
            for i in range(ocp_phase.n_intervals):
                idx = [i * (N_NODES - 1) + k for k in range(N_NODES)]
                interval_means.append(quadrature_weights.T @ casadi.vertcat(*[dF_dtau_nodes[k] for k in idx]))
            self.mean_objective_integrals[objective_name] = sum(interval_means)

    def substitute_nodes(self, f):
        return [casadi.substitute([f], sub[0], sub[1])[0] for sub in self.node_substitutions]


class TranscribedTrajectory:
    def __init__(self, ocp_trajectory):
        self.ocp_trajectory = ocp_trajectory
        n_phase_nodes = len(ocp_trajectory.values)
        info = parse_variable_name(ocp_trajectory.trajectory_interior.name())
        self.node_symbols = [None] * n_phase_nodes
        for i in range(1, n_phase_nodes - 1):
            self.node_symbols[i] = casadi.SX.sym('N/' + info.phase + '/' + info.name + '/' + str(i))
        self.node_symbols[0] = ocp_trajectory.start
        self.node_symbols[-1] = ocp_trajectory.end
        self.ode_node_defects = None

    def create_ode_constraints(self, transcribed_phase):
        if self.ocp_trajectory.derivative is None:
            return
        n_intervals = transcribed_phase.ocp_phase.n_intervals
        node_derivatives = transcribed_phase.substitute_nodes(self.ocp_trajectory.derivative)
        interval_duration = transcribed_phase.ocp_phase.duration_symbol / n_intervals
        defects = []
        for i in range(n_intervals):
            idx = [i * (N_NODES - 1) + k for k in range(N_NODES)]
            dxdt = [node_derivatives[j] for j in idx]
            x = [self.node_symbols[j] for j in idx]
            delta_x = [(xi - x[0]) for xi in x[1:]]
            dxdt_integrals = interval_duration * (casadi.DM(INTEGRATION_MATRIX) @ casadi.vertcat(*dxdt))
            defects.append(dxdt_integrals - casadi.vertcat(*delta_x))
        self.ode_node_defects = casadi.vertcat(*defects)


class TranscribedPathConstraint:
    def __init__(self, ocp_constraint, transcribed_phase):
        self.ocp_constraint = ocp_constraint
        self.g_nodes = casadi.vertcat(*transcribed_phase.substitute_nodes(ocp_constraint.g))

class HomotopyNlpSolver:
    IPOPT_OPTIONS = {
        'bound_consistency': False,
        'ipopt': {
            'max_iter': 2000,
            'max_cpu_time': 1800,
            'mu_strategy': 'adaptive',
            'nlp_scaling_method': 'gradient-based',
            'mumps_permuting_scaling': 7,
            'mumps_scaling': 8,
            'bound_frac': 1e-06,
            'bound_push': 1e-06,
            'slack_bound_push': 1e-06,
            'bound_relax_factor': 0.0,
            'honor_original_bounds': 'no',
            'warm_start_init_point': 'yes',
            'warm_start_mult_bound_push': 1e-06,
            'warm_start_bound_frac': 1e-06,
            'warm_start_bound_push': 1e-06,
            'warm_start_slack_bound_push': 1e-06,
            'warm_start_slack_bound_frac': 1e-06,
            'mu_max': 0.01,
            'linear_solver': 'mumps',
        }
    }

    def __init__(self, transcriber):
        self.transcriber = transcriber

        x_names = [n.name() for n in casadi.vertsplit(transcriber.nlp_x)]
        x_init_sym = casadi.vertcat(*[casadi.SX.sym('__init_' + n) for n in x_names])
        init_penalty_weight_sym = casadi.SX.sym('__initPenaltyWeight')
        init_penalty = init_penalty_weight_sym * casadi.sum1((transcriber.nlp_x - x_init_sym) ** 2)

        self.casadi_lbg = casadi.vertcat(casadi.DM.zeros(transcriber.nlp_h.numel()),
                                         (-casadi.inf) * casadi.DM.ones(transcriber.nlp_g.numel()))
        self.casadi_ubg = casadi.vertcat(casadi.DM.zeros(transcriber.nlp_h.numel()),
                                         casadi.DM.zeros(transcriber.nlp_g.numel()))
        self.casadi_g = casadi.vertcat(transcriber.nlp_h, transcriber.nlp_g)
        self.casadi_nlp = {'x': transcriber.nlp_x,
                           'p': casadi.vertcat(transcriber.nlp_p, init_penalty_weight_sym, x_init_sym),
                           'f': transcriber.nlp_f + init_penalty,
                           'g': self.casadi_g,
        }
        self.casadi_nlpsol = casadi.nlpsol('nlpsolver', 'ipopt', self.casadi_nlp, self.IPOPT_OPTIONS)

    def solver_step(self, x_value, p_value, init_penalty_weight, previous_result=None):
        lbxubx = self.transcriber.nlp_fn_lb_ub.call({'p': p_value})
        call_args = dict(x0=x_value, lbx=lbxubx['lb'], ubx=lbxubx['ub'], lbg=self.casadi_lbg, ubg=self.casadi_ubg,
                         p=casadi.vertcat(p_value, casadi.DM(init_penalty_weight), x_value))
        if previous_result is not None:
            call_args['lam_x0'] = previous_result['lam_x']
            call_args['lam_g0'] = previous_result['lam_g']
        result = self.casadi_nlpsol(**call_args)
        solver_stats = self.casadi_nlpsol.stats()
        assert solver_stats['success'], 'IPOPT did not converge: ' + str(solver_stats['return_status'])
        return {'result': result, 'solver_stats': solver_stats}

    def solver_loop(self, x_value, p_value, **kwargs):
        init_penalty_weight = kwargs.get('initPenaltyWeight', 100.0)
        beta = kwargs.get('beta', 0.1)
        steps = kwargs.get('penaltyWeightReductionSteps', 5)
        solve_original = kwargs.get('solveOriginalProblem', True)

        status = [{'result': None}]
        for _ in range(steps):
            status.append(self.solver_step(x_value, p_value, init_penalty_weight, status[-1]['result']))
            init_penalty_weight *= beta
        if solve_original:
            status.append(self.solver_step(x_value, p_value, 0.0, status[-1]['result']))
        return status
