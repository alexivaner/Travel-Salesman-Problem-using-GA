import numpy as np
import re


class Worker:
    """
    Worker
    """
    weightdata=None    

    @classmethod
    def evaluateFitnessPool(cls,x):
        tempintercity=[]
        for a in range (0,len(x)):

            if(a==len(x)-1):
                tempintercity.append(cls.weightdata[x[a]][x[0]])
            else:
                tempintercity.append(cls.weightdata[x[a]][x[a+1]])
        return sum(tempintercity)

