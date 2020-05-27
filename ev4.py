#
# ev3.py: An elitist (mu+mu) generational-with-overlap EA
#
#
# To run: python ev3.py --input ev3_example.cfg
#         python ev3.py --input my_params.cfg
#
# Basic features of ev3:
#   - Supports self-adaptive mutation
#   - Uses binary tournament selection for mating pool
#   - Uses elitist truncation selection for survivors
#

import optparse
import sys
import yaml
import math
from random import Random
from Population import *
import re
import numpy as np
import matplotlib.pyplot as plt



class EV3_Config:
    """
    EV3 configuration class
    """
    # class variables
    sectionName='EV3'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'startCity':(int,True),
             'learningRate':(float,True),
            'cityCoordinat':(str,True),
            'weightData':(str,True)
            }
     
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV3 section
        infile=open(inFileName,'r')
        ymlcfg=yaml.safe_load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))
         
        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]
 
                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                
                #Open the noisy_data.out directly
                if (opt=="cityCoordinat"):
                    with open(optval, 'r') as f:
                        x = f.read().splitlines()
                        optval=[re.split(r' +', i) for i in x]
                        optval=[[float(b) for b in a] for a in optval]
                        
                if (opt=="weightData"):
                    with open(optval, 'r') as f:
                        x = f.read().splitlines()
                        optval=[re.split(r' +', i) for i in x]
                        optval=[[float(b) for b in a] for a in optval]

                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
     
    #string representation for class data    
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))
    
def plot_result(new_list):
    plt.plot(*zip(*new_list), 'o')
    plt.show()
    
def fitnessFunc(x):
    #return -10.0-(0.04*x)**2+10.0*math.cos(0.04*math.pi*x)
    return x.interactionCity()

def printStats(pop,gen):
    print('Generation:',gen)
    avgval=0
    minval=pop[0].fit 
    sigma=pop[0].sigma
    for ind in pop:
        avgval+=ind.fit
        if ind.fit < minval:
            minval=ind.fit
            sigma=ind.sigma
        print(ind)

    print('Min fitness',minval)
    print('Sigma',sigma)
    print('Avg fitness',avgval/len(pop))
    print('')


def plotTSP(paths, points, num_iters=1):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    
    """

    # Unpack the primary TSP path and transform it into a list of ordered 
    # coordinates

    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    
    plt.plot(x, y, 'co')
    plt.title("Optimized Tour")
    a_scale = float(max(x))/float(100)
    
    for i in range(len(paths[0])):
        plt.annotate(i+1, (points[i][0], points[i][1]),fontsize=13,verticalalignment='top')
    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
            color ='g', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'g', length_includes_head = True)

    #Set axis too slitghtly larger than the set of x and y
#     plt.xlim(0, max(x)*1.1)
#     plt.ylim(0, max(y)*1.1)


def plotGraph(path1, cityCoordinat):
    res = [tuple(i) for i in cityCoordinat]
    # Pack the paths into a list
    paths = [path1]
    
    # Run the function
    plotTSP(paths, res)       
         

 #EV3:
#            
def ev3(cfg):

    #start random number generators
    uniprng=Random()
    uniprng.seed(cfg.randomSeed)
    normprng=Random()
    normprng.seed(cfg.randomSeed+101)

    #set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)

    Individual.fitFunc=fitnessFunc
    Individual.uniprng=uniprng
    Individual.normprng=normprng
    Individual.populationSize=cfg.populationSize
    Individual.cityCoordinat=cfg.cityCoordinat
    Individual.startCity=cfg.startCity
    Individual.numberofCity=len(cfg.weightData)
    Individual.weightdata=cfg.weightData
    Individual.learningRate=cfg.learningRate
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg.crossoverFraction
    Population.startCity=cfg.startCity

    #create initial Population (random initialization)
    population=Population(cfg.populationSize)
        
    #print initial pop stats    
    printStats(population,0)

    #evolution main loop
    for i in range(cfg.generationCount):
        #create initial offspring population by copying parent pop
        offspring=population.copy()
        
        #select mating pool
        offspring.conductTournament()

        #perform crossover
        offspring.crossover()
        
        #random mutation
        offspring.mutate()
        
        #update fitness values
        offspring.evaluateFitness()        
            
        #survivor selection: elitist truncation using parents+offspring
        population.combinePops(offspring)
        population.truncateSelect(cfg.populationSize)
        
        #print population stats    
        printStats(population,i+1)
              
        plotGraph(population.population[0].x,cfg.cityCoordinat)
        
                
        plt.pause(0.001)
        plt.clf()
     
    plt.show()
#
# Main entry point
#
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)
        
        #validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")
        
        #Get EV3 config params
        cfg=EV3_Config(options.inputFileName)
        
        
        plot_result(cfg.cityCoordinat)
        #print config params
        print(cfg)
                    
        #run EV3
        ev3(cfg)
        
        
        if not options.quietMode:                    
            print('EV3 Completed!')
    
    except Exception as info:
        raise
        # if 'options' in vars() and options.debugMode:
            # from traceback import print_exc
            # print_exc()
        # else:
            # print(info)
    

if __name__ == '__main__':
    main()
    
