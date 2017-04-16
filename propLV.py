########### Lotka Volterra propensities ############
import numpy as np
from propensities import Propensities


class PropLV(Propensities):
    def __init__(self,x,t,c1,c2,c3):
        Propensities.__init__(self,x,t)
  
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)

    def calculate(self) :
        h = np.array([[self.x[0]*self.c1],[self.c2*self.x[0]*self.x[1]],[self.c3*self.x[1]]])
        return h
