#
# Individual.py
#
#

import math
import collections
import numpy

#A simple 1-D Individual class
class Individual:
    """
    Individual
    """
    minSigma=1e-100
    maxSigma=1
    learningRate=None
    uniprng=None
    normprng=None
    fitFunc=None
    cityCoordinat=None
    startCity=None
    numberofCity=None
    weightdata=None
    populationSize=None
    var=None
    
        

    def __init__(self):
        self.x=CityPath()
        self.fit=self.__class__.fitFunc(self.x)
        self.sigma=self.uniprng.uniform(1/self.populationSize,1/self.numberofCity) #use "normalized" sigma
        
    def crossover(self, other):
        
        for a in range(0,len(self.x)):
            alpha=self.uniprng.uniform(0,1)
            probability=0.5
            
            if(alpha<probability):
                self.x[a], other.x[a] = other.x[a], self.x[a]
                dupli_x=self.list_duplicates(self.x)
                dupli_other=self.list_duplicates(other.x)
                
                if (len(dupli_x)!=0 and len(dupli_other)!=0 ):
                    self.x[dupli_x[0]], other.x[dupli_other[0]] = other.x[dupli_other[0]], self.x[dupli_x[0]]
        '''This is for 1 Point crossover
        #perform crossover "in-place"
        alpha=self.uniprng.randint(0,self.latticeLength-1)
        
        #Swap the list based on alpha
        self.x[alpha:], other.x[alpha:] = other.x[alpha:], self.x[alpha:]'''
        
        self.fit=None
        other.fit=None
        
    def list_duplicates(self,seq):
        seen = set()
        seen_add = seen.add
        return [idx for idx,item in enumerate(seq) if item in seen or seen_add(item)]

    
    def mutate(self):
        self.sigma=self.sigma*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.sigma < 1/self.populationSize: 
            self.sigma=(1/self.populationSize)
            #self.sigma=self.minSigma

        if self.sigma > 1/self.numberofCity: 
            self.sigma=(1/self.numberofCity)
            #self.sigma=self.maxSigma


        if self.uniprng.uniform(0,1)<self.sigma: #randomprobability of mutation
            if (self.startCity==0):
                mutationindex=self.uniprng.randint(0,self.numberofCity-1)
                self.x[mutationindex]=self.uniprng.randint(0,self.numberofCity-1)

            else:
                mutationindex=self.uniprng.randint(1,self.numberofCity-1)   
                self.x[mutationindex]=self.uniprng.randint(1,self.numberofCity-1)
            differ=self.diff(self.x)

            if len(differ)!=0:
                samevalindex=self.list_duplicates_of(self.x.x, self.x[mutationindex])
                samevalindex.remove(mutationindex)
                self.x[samevalindex[0]]=differ[0]
           
        self.fit=None
        
    def diff(self,second):
        first =set(list(range(0, self.numberofCity)))
        return [item for item in first if item not in second]    
        
    def list_duplicates_of(self,seq,item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item,start_at+1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs

    def evaluateFitness(self):
        if self.fit == None: self.fit=self.__class__.fitFunc(self.x)
        
        
    def __repr__(self):
        ar = numpy.array(self.x)
        return str(ar+1)+'\t'+str(self.fit)+'\t'+str(self.sigma)

    
class CityPath(Individual):
    def __init__(self):
        if (self.startCity==0):
            self.x=self.uniprng.sample(range(self.numberofCity), self.numberofCity)
            
        else:
            x=self.uniprng.sample(range(self.startCity,self.numberofCity), self.numberofCity-self.startCity)
            y=self.uniprng.sample(range(0,self.startCity-1),self.startCity-1)
            z=[self.startCity-1]
            z.append(y)
            z.append(x)
            flatlist=self.flatten(z)
            copy = flatlist[1:]
            self.uniprng.shuffle(copy)
            flatlist[1:] = copy # overwrite the original
            self.x=flatlist
    
    def flatten(self,x):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in self.flatten(i)]
        else:
            return [x]
            

    def interactionCity(self):
        tempintercity=[]
        for a in range (0,len(self.x)):

            if(a==len(self.x)-1):
                tempintercity.append(self.weightdata[self.x[a]][self.x[0]])
            else:
                tempintercity.append(self.weightdata[self.x[a]][self.x[a+1]])
        return sum(tempintercity)

              
    def __repr__(self):
        return str(self.x)
    
    def __getitem__(self, key):
        return self.x[key]
    
    def __len__(self):
        return len(self.x)
    
    def __setitem__(self, key, value):
        self.x[key] = value

    
