'''
Different ATSP models to be implemented from Gurobi
'''
from gurobipy import *


def atsp_basic(funcion_subtours, num_nodes, distance_matrix):
    model = Model("atsp - " + funcion_subtours.__name__)
    x = {}
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            # if i != j:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name="x(%s,%s)" % (i, j))  # Variable Xij
    model.update()
    model.addConstr(quicksum(x[i, i] for i in range(1, num_nodes + 1)) == 0, "Diagonal")
    for i in range(1, num_nodes + 1):
        model.addConstr(quicksum(x[i, j] for j in range(1, num_nodes + 1) if i != j) == 1,
                        "Out(%s)" % i)  # Only one output
        model.addConstr(quicksum(x[j, i] for j in range(1, num_nodes + 1) if i != j) == 1,
                        "In(%s)" % i)  # Only one input

    model.setObjective(quicksum(distance_matrix[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)
    model = funcion_subtours(model, x, num_nodes)
    model.update()

    return model


def claus(model, x, n):
    """claus: Claus  model for the (asymmetric) traveling salesman problem
    (potential formulation)
    Parameters:
        - n - number of nodes
        - x[i,j] - 1 if arc (i,j) is used, 0 otherwise
        - model - model without elimination subtours formulation
    Returns a model, ready to be solved.
    """
    w = {}
    for k in range(2, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                w[k, i, j] = model.addVar(lb=0, vtype=GRB.BINARY, name="w(%s,%s,%s)" % (k, i, j))
    model.update()

    for k in range(2, n + 1):
        for i in range(2, n + 1):
            if i != k:
                model.addConstr(quicksum(w[k, i, j] for j in range(1, n + 1)) -
                                quicksum(w[k, j, i] for j in range(1, n + 1)) == 0,
                                "FlowCons(%s,%s)" % (k, i))  # 13
    for k in range(2, n + 1):
        model.addConstr(
            quicksum(w[k, 1, j] for j in range(2, n + 1)) - quicksum(w[k, j, 1] for j in range(2, n + 1)) == -1,
            'FlowIn(%s)' % k)  # 14

    for i in range(2, n + 1):
        model.addConstr(
            quicksum(w[i, i, j] for j in range(1, n + 1)) - quicksum(w[i, j, i] for j in range(1, n + 1)) == 1,
            'FlowOut(%s)' % i)  # 15

    for (k, i, j) in w:
        model.addConstr(w[k, i, j] <= x[i, j], "FlowUB(%s,%s,%s)" % (k, i, j))

    model.update()
    model.__data = x, w
    return model


def claus_gp(model, x, n):
    """claus: Equivalent version of Claus model  considered by Gouveia and Pires for the (asymmetric) traveling salesman problem
    (potential formulation)
    Parameters:
        - n - number of nodes
        - x[i,j] - 1 if arc (i,j) is used, 0 otherwise
        - model - model without elimination subtours formulation
    Returns a model, ready to be solved.
    """
    f, v = {}, {}
    for k in range(2, n + 1):
        for i in range(1, n + 1):
            if i >= 2:
                v[k, i] = model.addVar(vtype=GRB.BINARY, name="v(%s,%s)" % (k, i))
            for j in range(1, n + 1):
                f[k, i, j] = model.addVar(lb=0, vtype=GRB.BINARY, name="f(%s,%s,%s)" % (k, i, j))  # 44
    model.update()

    for k in range(2, n + 1):
        for j in range(2, n + 1):
            if j != k:
                model.addConstr(quicksum(f[k, j, i] for i in range(1, n + 1)) -
                                quicksum(f[k, i, j] for i in range(1, n + 1)) == 0,
                                "FlowCons(%s,%s)" % (j, k))  # 41

    for j in range(2, n + 1):
        model.addConstr(
            quicksum(f[j, j, i] for i in range(1, n + 1)) == 1,
            'FlowOut(%s)' % j)  # 42

    for (k, i, j) in f:
        model.addConstr(f[k, i, j] <= x[i, j], "FlowUB(%s,%s,%s)" % (k, i, j))  # 43

    for i in range(2, n + 1):
        for k in range(2, n + 1):
            model.addConstr(quicksum(f[k, i, j] for j in range(1, n + 1)) == v[k, i])  # 45

    model.update()
    model.__data = x, f, v
    model.params.LazyConstraints = 1

    return model


def mtz(model, x, n):
    """mtz: Miller-Tucker-Zemlin's model for the (asymmetric) traveling salesman problem
    (potential formulation)
    Parameters:
        - n - number of nodes
        - x[i,j] - 1 if arc (i,j) is used, 0 otherwise
        - model - model without elimination subtours formulation
    Returns a model, ready to be solved.
    """
    u = {}
    for i in range(2, n + 1):
        u[i] = model.addVar(lb=1, ub=n - 1, vtype="C", name="u(%s)" % i)  # Variable Ui
    model.update()

    for i in range(2, n + 1):
        for j in range(2, n + 1):
            model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, "MTZ(%s,%s)" % (i, j))

    model.updatce()
    model.__data = x, u
    return model


def dl(model, x, n):
    """dl: Desrochers and Laporte (MTZ strong) --> model for the (asymmetric) traveling salesman problem
    (potential formulation, adding stronger constraints)
    Parameters:
        - n - number of nodes
        - x[i,j] - 1 if arc (i,j) is used, 0 otherwise
        - model - model without elimination subtours formulation
    Returns a model, ready to be solved.
    """
    u = {}
    for i in range(1, n + 1):
        u[i] = model.addVar(lb=0, ub=n - 1, vtype="C", name="u(%s)" % i)

    model.update()

    for i in range(2, n + 1):
        for j in range(2, n + 1):
            model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] + (n - 3) * x[j, i] <= n - 2,
                            "LiftedMTZ(%s,%s)" % (i, j))

    for i in range(2, n + 1):
        model.addConstr(1 + (n - 3) * x[i, 1] + quicksum(x[j, i] for j in range(2, n + 1)) <= u[i],
                        name="LiftedLB(%s)" % i)
        model.addConstr(n - 1 - (n - 3) * x[1, i] - quicksum(x[i, j] for j in range(2, n + 1)) >= u[i],
                        name="LiftedUB(%s)" % i)

    model.update()
    model.__data = x, u
    return model


def sd(model, x, n):
    """sd: Sherali and Driscoll (DL strong) --> model for the (asymmetric) traveling salesman problem
    (potential formulation, adding other stronger constraints)
    Parameters:
        - n - number of nodes
        - x[i,j] - 1 if arc (i,j) is used, 0 otherwise
        - model - model without elimination subtours formulation
    Returns a model, ready to be solved.
    """
    u, y = {}, {}
    for i in range(2, n + 1):
        u[i] = model.addVar(lb=0, vtype="C", name="u(%s)" % i)
        for j in range(2, n + 1):
            # if i != j:
            y[i, j] = model.addVar(lb=0, vtype="C", name="y(%s,%s)" % (i, j))  # 21

    model.update()

    for i in range(2, n + 1):
        model.addConstr(quicksum(y[i, j] for j in range(2, n + 1)) + (n - 1) * x[i, 1] == u[i],
                        "EquivalenceUi(%s)" % i)  # 13
    for j in range(2, n + 1):
        model.addConstr(quicksum(y[i, j] for i in range(2, n + 1)) + 1 == u[j], "EquivalenceUj(%s)" % j)  # 14

    for i in range(2, n + 1):
        for j in range(2, n + 1):
            model.addConstr(x[i, j] <= y[i, j], "ArcOrderLB(%s,%s)" % (i, j))  # 15
            model.addConstr(y[i, j] <= (n - 2) * x[i, j], 'ArcOrderUB(%s,%s)' % (i, j))  # 16
            model.addConstr(
                u[j] + (n - 2) * x[i, j] - (n - 1) * (1 - x[j, i]) <= y[i, j] + y[j, i],
                'AdyacentArcsOrderLB(%s,%s)' % (i, j))  # 17
            model.addConstr(y[i, j] + y[j, i] <= u[j] - (1 - x[j, i]),
                            'AdyacentArcsOrderUB(%s,%s)' % (i, j))  # 18

    for j in range(2, n + 1):
        model.addConstr(1 + (1 - x[1, j]) + (n - 3) * x[j, 1] <= u[j], 'ULB(%s)' % j)  # 19
        model.addConstr(u[j] <= (n - 1) - (n - 3) * x[1, j] - (1 - x[j, 1]), 'UUB(%s)' % j)  # 20

    model.update()
    model.__data = x, u, y

    return model


def scf(model, x, n):
    """scf: single-commodity flow formulation for the (asymmetric) traveling salesman problem
    Parameters:
        - n - number of nodes
        - x[i,j] - 1 if arc (i,j) is used, 0 otherwise
        - model - model without elimination subtours formulation
    Returns a model, ready to be solved.
    """

    g = {}
    for i in range(1, n + 1):
        for j in range(2, n + 1):
            g[i, j] = model.addVar(lb=0, vtype='C', name="g(%s,%s)" % (i, j))
    model.update()

    for i in range(1, n + 1):
        if i > 1:
            model.addConstr(quicksum(g[j, i] for j in range(1, n + 1)) -
                            quicksum(g[i, j] for j in range(2, n + 1)) == 1, "FlowCons(%s)" % i)
        for j in range(2, n + 1):
            model.addConstr(g[i, j] <= (n - 1) * x[i, j], 'FlowUB(%s,%s)' % (i, j))

    model.update()
    model.__data = x, g
    return model


def mcf(model, x, n):
    """mcf: multi-commodity flow formulation for the (asymmetric) traveling salesman problem
    Parameters:
        - n - number of nodes
        - x[i,j] - 1 if arc (i,j) is used, 0 otherwise
        - model - model without elimination subtours formulation
    Returns a model, ready to be solved.
    """
    w = {}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(2, n + 1):
                l = 1
                w[i, j, k, l] = model.addVar(lb=0, vtype=GRB.BINARY, name="w(%s,%s,%s,%s)" % (i, j, k, l))
            for l in range(2, n + 1):
                k = 1
                w[i, j, k, l] = model.addVar(lb=0, vtype=GRB.BINARY, name="w(%s,%s,%s,%s)" % (i, j, k, l))

    model.update()

    for l in range(2, n + 1):
        for i in range(2, n + 1):
            if i != l:
                model.addConstr(quicksum(w[i, j, 1, l] for j in range(1, n + 1)) - quicksum(
                    w[j, i, 1, l] for j in range(1, n + 1)) == 0, "Flowwl(%s, %s)" % (i, l))
                # k como l
                model.addConstr(quicksum(w[i, j, l, 1] for j in range(1, n + 1)) - quicksum(
                    w[j, i, l, 1] for j in range(1, n + 1)) == 0, "Flowwk(%s, %s)" % (i, l))
        model.addConstr(quicksum(w[1, j, 1, l] for j in range(2, n + 1)) - quicksum(
            w[j, 1, 1, l] for j in range(2, n + 1)) == 1, "FlowInwl(%s)" % (l))
        # k como l
        model.addConstr(quicksum(w[1, j, l, 1] for j in range(2, n + 1)) - quicksum(
            w[j, 1, l, 1] for j in range(2, n + 1)) == -1, "FlowInwk(%s)" % (l))

    for i in range(2, n + 1):
        model.addConstr(
            quicksum(w[i, j, 1, i] for j in range(1, n + 1)) - quicksum(w[j, i, 1, i] for j in range(1, n + 1)) == -1,
            "FlowOutwl(%s)" % (i))
        # k como l
        model.addConstr(
            quicksum(w[i, j, i, 1] for j in range(1, n + 1)) - quicksum(w[j, i, i, 1] for j in range(1, n + 1)) == 1,
            "FlowOutwk(%s)" % (i))

    for (i, j, k, l) in w:
        model.addConstr(w[i, j, k, l] <= x[i, j], "FlowUBw(%s,%s,%s,%s)" % (i, j, k, l))

    model.update()
    model.__data = x, w
    return model


def dfj(model, x, n):
    model.params.LazyConstraints = 1
    model.__data = {0: x}

    return model