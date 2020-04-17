import picos as pic
from picos import RealVariable
from copy import deepcopy
from heapq import *
import heapq as hq
import numpy as np
import itertools
import math
import logging
counter = itertools.count()

SOLVER = 'cvxopt'

class BBTreeNode():
    def __init__(self, vars = [], constraints = [], objective='', prob=None):
        self.vars = vars
        self.constraints = constraints
        self.objective = objective
        self.prob = prob

    def __deepcopy__(self, memo):
        '''
        Deepcopies the picos problem
        This overrides the system's deepcopy method bc it doesn't work on classes by itself
        '''
        newprob = pic.Problem.clone(self.prob)
        return BBTreeNode(self.vars, newprob.constraints, self.objective, newprob)

    def buildProblem(self):
        '''
        Bulids the initial Picos problem
        '''
        prob=pic.Problem()

        prob.add_list_of_constraints(self.constraints)

        prob.set_objective('max', self.objective)
        self.prob = prob
        return self.prob

    def is_integral(self):
        '''
        Checks if all variables (excluding the one we're maxing) are integers
        '''
        for v in self.vars[:-1]:
            logging.info(f'Var round is {round(v.value)}')
            logging.info(f'Var float is {float(v.value)}')

            if v.value == None or abs(round(v.value) - float(v.value)) > 1e-4 :
                return False
        return True

    def branch_floor(self, branch_var):
        '''
        Makes a child where xi <= floor(xi)
        '''
        n1 = deepcopy(self)
        n1.prob.add_constraint( branch_var <= math.floor(branch_var.value) ) # add in the new binary constraint

        return n1

    def branch_ceil(self, branch_var):
        '''
        Makes a child where xi >= ceiling(xi)
        '''
        n2 = deepcopy(self)
        n2.prob.add_constraint( branch_var >= math.ceil(branch_var.value) ) # add in the new binary constraint
        return n2


    def bbsolve(self):
        '''
        Use the branch and bound method to solve an integer program
        This function should return:
            return bestres, bestnode_vars

        where bestres = value of the maximized objective function
              bestnode_vars = the list of variables that create bestres
        '''

        # these lines build up the initial problem and adds it to a heap
        logging.info(f'TEST')
        root = self
        res = root.buildProblem() #.solve(solver=SOLVER)
        heap = [(next(counter), root)]
        bestres = -1e20 # a small arbitrary initial best objective value
        bestnode_vars = [] # [float(x.value) for x in root.vars] # initialize bestnode_vars to the root vars

        #TODO: fill this part in
        while len(heap) > 0:
            curr = heap.pop(0)
            curr[1].prob.solve(solver=SOLVER)
            logging.info(f'Current len of heap: {len(heap)}')
            logging.info(f'counter is: {curr[0]}')
            logging.info(f'node is: {curr[1]}')
            logging.info(f'node vals are {[float(x) for x in curr[1].vars]}')
            logging.info(f'node vals are {curr[1].vars}')

            if float(curr[1].objective) < bestres:
                logging.info(f'Bad objective of {float(curr[1].objective)}')
                continue

            if curr[1].is_integral():
                #save it as the best value if it is better than the bestres
                logging.info("Is integral")
                logging.info(f' vars are: {[float(x) for x in curr[1].vars]}')
                logging.info(f'Compared new bestval is: {curr[1].objective} \n compared to {bestres}')
                if curr[1].objective > bestres:
                    bestres = float(curr[1].vars[-1])
                    bestnode_vars = [float(x.value) for x in curr[1].vars]
                    logging.info(f'New bestres is {bestres}')
                    logging.info(f'New bestnode_vars are: {bestnode_vars}')
            else:
                #divide and conquer
                logging.info("Not integral.")

                # Pick a var [loop through]
                for v in curr[1].vars:
                    if abs(round(v.value) - float(v.value)) > 1e-4 :
                        var_to_round = v
                        break

                # Round up and round down
                try:
                    floor = curr[1].branch_floor(var_to_round)
                    logging.info(f'floor is {floor}')
                    logging.info(f'floor prob is {floor.prob}')
                    logging.info(f'floor vars are {[float(x) for x in floor.vars]}')
                    heap.append((next(counter), floor))
                    logging.info(f'newest counter val: {heap[-1][0]} \nHas objective {float(floor.objective)}')
                except Exception as e:
                    logging.info(f'No valid round down. Exception {e}')

                try:
                    ceil = curr[1].branch_ceil(var_to_round)
                    logging.info(f'ceil is: {ceil}')
                    logging.info(f'ceil prob is {ceil.prob}')
                    logging.info(f'ceil vars are {[float(x) for x in ceil.vars]}')
                    ceil_ans = ceil.prob.solve(solver=SOLVER)
                    heap.append((next(counter), ceil))
                    logging.info(f'newest counter val: {heap[-1][0]} \nHas objective {ceil.objective}')
                except Exception as e:
                    logging.info(f'No valid round up. Exception {e}')


        logging.info(f'Solved everything! bestres: {bestres} \n bestnode_vars: {bestnode_vars} \n\n\n\n\n\n\n\n')
        return bestres, bestnode_vars
