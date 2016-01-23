# =============================================================================
# HEPHAESTUS VALIDATION 1 - SCIPY OPTIMIZE
# =============================================================================

import numpy as np
from scipy.optimize import minimize

class Wing:
    
    def __init__(self):
        self.fun = 100000000
        self.con = [0,0,0]
    
    def instantiate(self, x):
        self.x = x
        self.fun = (x[0] - 1)**2 + (x[1] - 2.5)**2
        con1 = x[0] - 2 * x[1] + 2
        con2 = -x[0] - 2 * x[1] + 6
        con3 = -x[0] + 2 * x[1] + 2
        self.con = np.array([con1, con2, con3])
    
    def objFun(self):
        return self.fun
    
    def Const(self):
        return self.con
        
    def printState(self):
        test = 1
        #print('\nThe value of x is: [%4.4f , %4.4f]' % (self.x[0],self.x[1]))
        #print('The function value is: %4.4f' % (self.objFun()))
        
        
wingFun = Wing()
wingCon = Wing()
        
def funMin(x):
    wingFun.instantiate(x)
    wingFun.printState()
    return wingFun.objFun()
    
def con1(x):
    wingCon.instantiate(x)
    wingCon.printState()
    #print('The constraints value are:')
    #print(wingCon.con)
    return wingCon.con[0]
def con2(x):
    wingCon.instantiate(x)
    wingCon.printState()
    #print('The constraints value are:')
    #print(wingCon.con)
    return wingCon.con[1]
def con3(x):
    wingCon.instantiate(x)
    wingCon.printState()
    #print('The constraints value are:')
    #print(wingCon.con)
    return wingCon.con[2]


bnds = ((0, None), (0, None))
#cons = ({'type': 'ineq', 'fun': lambda x:  con1()})
cons = [lambda x:  con1(x),lambda x: con2(x),lambda x: con3(x)]
cons = ({'type': 'ineq', 'fun': lambda x:  con1(x)},
        {'type': 'ineq', 'fun': lambda x:  con2(x)},
        {'type': 'ineq', 'fun': lambda x:  con3(x)})
res = minimize(funMin, [2, 0], method='SLSQP', bounds=bnds, constraints=cons)
#res = fmin_slsqp(funMin, [2, 0], ieqcons = cons, bounds=bnds)
print(res)