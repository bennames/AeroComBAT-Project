#Aerodynamics.py
# =============================================================================
# HEPHAESTUS PYTHON MODULES
# =============================================================================

# =============================================================================
# IMPORT SCIPY MODULES
# =============================================================================
import numpy as np

class Airfoil:
    def __init__(self,c,**kwargs):
        '''
        Inputs:
        c - the chord length of the airfoil section
        Optional arguments:
        name - the NACA 4 series airfoil name
           ex, name = 'NACA0012'
        xu - the x-coordinates of the upper curve of the airfoil
        yu - the y-coordinates of the upper curve of the airfoil
        xl - the x-coordinates of the lower curve of the airfoil
        yl - the y-coordinates of the lower curve of the airfoil
        '''
        name = kwargs.pop('name','NACA0012')
        xu = kwargs.get('xu')
        yu = kwargs.get('yu')
        xl = kwargs.get('xl')
        yl = kwargs.get('yl')
        self.c = c
        #If xu, yu, xl, yl, don't generate a NACA airfoil
        if not ((xu==None) or (yu==None) or (xl==None) or (yl==None)):
            #TODO: Finish this section, will require curve fitting module, scipy.optimize
            test=1
        elif name=='box':
            pass
        else:
            self.t = float(name[-2:])/100
            self.p = float(name[-3])/10
            self.m = float(name[-4])/100
        self.name = name
    def points(self,x):
        #Inputs:
        #x, a non-dimensional chord length from the leading edge
        x = x*self.c
        # For more detail on the NACA 4 series airfoil,
        # see: https://en.wikipedia.org/wiki/NACA_airfoil
        #TODO: Comment this method more thuroughly
        if self.name=='box':
            return x,self.c*np.ones(len(x))/2,x,-self.c*np.ones(len(x))/2
        else:
            c = self.c
            t = self.t
            m = self.m
            p = self.p
            yt = 5*t*c*(0.2969*np.sqrt(x/c)-.126*(x/c)-.3516*(x/c)**2+.2843*(x/c)**3-.1015*(x/c)**4)
            
            xc0 = x[x<c*p]
            xc1 = x[x>=c*p]
            
            if len(xc0)>0:
                yc0 = (m*xc0/p**2)*(2*p-xc0/c)
                dyc0dx = (2*m/p**2)*(p-xc0/c)
            else:
                yc0 = []
                dyc0dx = []
            if len(xc1)>0:
                yc1 = m*((c-xc1)/(1-p)**2)*(1+xc1/c-2*p)
                dyx1dx = (2*m/(1-p)**2)*(p-xc1/c)
            else:
                yc1 = []
                dyx1dx = []
            yc = np.append(yc0,yc1)
            dycdx = np.append(dyc0dx,dyx1dx)
            
            th = np.arctan(dycdx)
            
            xu = x-yt*np.sin(th)
            yu = yc+yt*np.cos(th)
            xl = x+yt*np.sin(th)
            yl = yc-yt*np.cos(th)
            return xu,yu,xl,yl