'''Code file for vehicle routing problem created for Advanced Algorithms
Spring 2020 at Olin College. These functions solve the vehicle routing problem
using an integer programming and then a local search approach. This code has
been adapted from functions written by Alice Paul.'''

import picos as pic
from picos import RealVariable
import numpy as np
from read_files import read_file_type_A, read_file_type_C
# import cplex

# Integer programming approach
def cvrp_ip(C,q,K,Q,obj=True):
    '''
    Solves the capacitated vehicle routing problem using an integer programming
    approach.

    C: matrix of edge costs, that represent distances between each node
    q: list of demands associated with each client node
    K: number of vehicles
    Q: capacity of each vehicle
    obj: whether to set objective (ignore unless you are doing local search)
    returns:
        objective_value: value of the minimum travel cost
        x: matrix representing number of routes that use each arc
    '''
    # TODO: add in destination node (same distances as source & demand = 0)
    # print(C)

    # This is depressingly complex :/ I'm probably being an idiot with numpy
    new_distances = np.zeros([C.shape[0] + 1, C.shape[1] + 1])
    new_down = np.zeros([C.shape[1] + 1, 1])
    for indx, x in enumerate(C):
        if indx < C.shape[0]:
            for indy, y in enumerate(x):
                new_distances[indx][indy] = y
            new_distances[indx][C.shape[1]] = x[0]
            new_down[indx] = x[0]
    # print(new_down)
    q = np.append(q,0)
    for indy, y in enumerate(new_down):
        new_distances[C.shape[0]][indy] = y
    # print(new_distances)

    num_nodes = new_distances.shape[0]

    # set up the picos problem
    prob = pic.Problem()

    x = prob.add_variable('x', new_distances.shape, vtype='binary')
    u = prob.add_variable('u', num_nodes, vtype='continuous', upper=Q, lower=q)
    vars = [x, u]
    # print(vars)

    #use Sum
    prob.add_constraint(sum(x[0,i] for i in range(num_nodes)) <= K)
    prob.add_constraint(sum(x[i,0] for i in range(num_nodes)) == 0) #Make sure nothing ends @ origin
    prob.add_constraint(sum(x[i,num_nodes-1] for i in range(num_nodes)) == sum(x[0,i] for i in range(num_nodes)))
    prob.add_list_of_constraints([sum([x[j,i] for i in range(num_nodes)])==1 for j in range(1, num_nodes - 1)])
    prob.add_list_of_constraints([sum([x[i,j] for i in range(num_nodes)])==1 for j in range(1, num_nodes - 1)])
    prob.add_list_of_constraints([(u[i]-u[j])+Q*x[i,j] <= Q-q[j] for i in range(num_nodes) for j in range(num_nodes)])
    prob.add_constraint(sum(x[i,i] for i in range(num_nodes)) == 0) #because a node has nothing to itself...

    prob.set_objective('min', pic.sum([new_distances[i, j]*x[i, j] for i in range(num_nodes) for j in range(num_nodes)]))

    # TODO: add variables, constraints, and objective function!

    solution = prob.solve(solver='cplex', verbose=True)

    print(solution)
    print(x)

    # if (not "integer optimal solution" == solution['status']):
    #     return 0, x

    objective_value = prob.obj_value()

    return objective_value, x

# Local search approach (OPTIONAL)
def local_search(C,q,K,Q):
    '''
    Solves the capacitated vehicle routing problem using a local search
    approach.

    C: matrix of edge costs, that represent distances between each node
    q: list of demands associated with each client node
    K: number of vehicles
    Q: capacity of each vehicle
    returns:
        bestval: value of the minimum travel cost
        bestx: matrix representing number of routes that use each arc
    '''
    bestx = []
    bestval = 0

    # TODO (OPTIONAL): implement local search to solve vehicle routing problem

    return bestval, bestx


if __name__ == "__main__":

    # an example call to test your integer programming implementation
    C,q,K,Q = read_file_type_A('data/A-n05-k04.xml')
    travel_cost, x = cvrp_ip(C,q,K,Q)
    print("Travel cost: " + str(travel_cost))
