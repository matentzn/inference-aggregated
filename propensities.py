#==============================================================================
# propensities class for Gillespie 
#==============================================================================
class Propensities(object):
    def __init__(self,x,t):
        #self.c = c
        self.x = x
        self.t = t

##    def get_c(self):
##        return self.c

    def calculate(self):
        raise NotImplementedError


