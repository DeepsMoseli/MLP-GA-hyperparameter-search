# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:41:18 2020

@author: Deeps
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier as NN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore")
import pickle


##############################################################################
# Load the breast cancer data and split into test and training to use in NN  #
##############################################################################
data = load_breast_cancer()
x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,
                                                 test_size=0.2,random_state=42)


##############################################################################
##        Class to help automate generation of populationof neural nets     ##
##############################################################################
class MLPerceptronClass:
    """ this will take the following parameters for optimization:
    
    """
    
    def __init__(self,randomize=True,params={}):
        if randomize==True:
            self.params = self.Random_individual()
        else:
            self.params = params
        self.model = NN(hidden_layer_sizes=self.params["hidden_layer_sizes"],
                        activation=self.params["activation"],
                        solver=self.params["solver"],
                        alpha=self.params["alpha"],
                        learning_rate=self.params["learning_rate"],
                        learning_rate_init=self.params["learning_rate_init"],
                        max_iter=self.params["max_iter"])
        
        
    def hidden_layer_sizes(self,layers):
        return layers
    
    def activation(self,activation):
        switch = {0:"identity",1:"logistic",2:"tanh",3:"relu"}
        return switch[activation]
    
    def solver(self,solver_num):
        switch = {0:"sgd",1:"adam"}
        return switch[solver_num]

    
    def alpha(self,alpha_num): #penalty term
        if alpha_num is not None:
            return alpha_num
        else:
            return np.random.random()/100
    
    def learning_rate_structure(self,lrs):
        switch = {0:"constant",1:"invscaling",2:"adaptive"}
        return switch[lrs]
    
    def learning_rate_init(self,lri):
        if lri is not None:
            return lri
        else:
            return np.random.random()/200
        
    def max_iter(self):
        return np.random.randint(50,400)
    
    def Random_individual(self):
        #hiddend layer 
        params={}
        layer_size = np.random.randint(20,400)
        num_layers =  np.random.randint(1,200)
        params["hidden_layer_sizes"] = (layer_size,num_layers)
        
        #activation
        params["activation"] = self.activation(np.random.randint(0,4))
        
        #solver
        params["solver"] = self.solver(np.random.randint(0,2))
        
        #alpha(L2 regularizer)
        params["alpha"] = self.alpha(np.random.random()/100)
    
        #learning_rate_structure
        params["learning_rate"] = self.learning_rate_structure(np.random.randint(0,3))
        
        #learning_rate_init
        params["learning_rate_init"] = self.learning_rate_init(np.random.random()/200)
        
        #max_iter
        params["max_iter"] =self.max_iter()
        return params
        
    
class Genetic_Algorithm:
    def __init__(self,population_size,mutation_prob,elitism,crossover_prob):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.crossover_prob = crossover_prob
        self.population = []
        self.new_population = []
        self.fitness = {}
        self.target = 0.98
        self.max_gen = 100 
        self.decay_generations = 5
        
        
    def mutate(self):
        return random.choice(self.genes)
    
    def random_genome(self):
        Model_instance = MLPerceptronClass(True)
        return Model_instance.params
    
    def mating(self,parent1,parent2):
        param_len = len(parent1)
        assert param_len == len(parent2),"Both parents should have the same gene count"
        offspring = {}
        prob = random.random()
        for k in parent1:
            if prob<=(1-self.mutation_prob)/2:
                offspring[k] = parent1[k]
            elif prob<=(1-self.mutation_prob):
                offspring[k] = parent2[k]
            else:
                offspring[k] = self.random_genome()[k]
        return offspring
    
    def calc_fitness(self,individual):
        model_eval = MLPerceptronClass(False,individual)
        model_eval.model.fit(x_train,y_train)
        return accuracy_score(y_test,model_eval.model.predict(x_test))
    
    def adaptive_probs(self):
        self.mutation_prob -= 0.02
        self.crossover_prob -= 0.01 
        self.elitism += 0.012
    
    def Main(self):
        
        #population init
        Converged = False
        Generation = 1;
        for k in range(self.population_size):
            self.population.append(self.random_genome())
        
        #selection
        pbar = tqdm(range(self.max_gen))
        while(Converged==False):
            print("-----------Generation %s---------"%Generation)
            """calc fitness and do selection"""
            
            self.fitness["Generation %s"%Generation]= list(map(self.calc_fitness,self.population))
            sortedindexes =  list(np.flip(np.argsort(self.fitness["Generation %s"%Generation])))
            print("best fitness: ",pd.Series(self.fitness["Generation %s"%Generation])[sortedindexes[0]])
            print("Average fitness: ",np.mean(self.fitness["Generation %s"%Generation]))
            print("\n---------------------------------")
            
            """-elite top 10% straight to new generation
                -cross over for other 85%, only within top 50%
                -new entrants for last 5%
            """
            self.new_population.extend(list(pd.Series(self.population)[sortedindexes[:int(self.elitism*self.population_size)]]))
            self.new_population.extend([self.random_genome() for _ in range(int((1-self.crossover_prob-self.elitism)*self.population_size))])
            while(len(self.new_population)!=len(self.population)):
                """ Crossover from only the top 50% from previous population """
                p1=random.choice(list(pd.Series(self.population)[sortedindexes[:int(0.5*self.population_size)]]))
                p2=random.choice(list(pd.Series(self.population)[sortedindexes[:int(0.5*self.population_size)]]))
                self.new_population.append(self.mating(p1,p2))
            
            self.population=self.new_population
            self.new_population=[]
            pbar.update(1)
            if max(self.fitness["Generation %s"%Generation])>=self.target or Generation>=self.max_gen:
                Converged=True
            else:
                Generation+=1
                if Generation%self.decay_generations==0:
                    self.adaptive_probs()
        pbar.close()


class Analytics:
    def __init__(self,Evolved_GA):
        self.GA = Evolved_GA
        self.features ={"categorical":["activation","solver","learning_rate"],
                        "numeric":["hidden_layer_sizes","alpha","learning_rate_init","max_iter"]}
        self.fitness_plots()
    
    def fitness_plots(self):
        #find lowest, mean, max
        generations = list(range(len(self.GA.fitness)))
        weakest = [np.min(self.GA.fitness[k]) for k in self.GA.fitness]
        std_plus = [np.var(self.GA.fitness[k])+np.mean(self.GA.fitness[k]) for k in self.GA.fitness]
        std_minus = [-np.var(self.GA.fitness[k])+np.mean(self.GA.fitness[k]) for k in self.GA.fitness]
        average = [np.mean(self.GA.fitness[k]) for k in self.GA.fitness]
        fittest = [np.max(self.GA.fitness[k]) for k in self.GA.fitness]
        
        #plt.plot(generations, std_minus,'r',label="weakest")
        plt.plot(generations, average,'b',label="Average")
        #plt.plot(generations, std_plus,'g',label="Fittest")
        plt.fill_between(generations,std_plus,std_minus,facecolor='blue', alpha=0.2)
        plt.ylabel("Generation")
        plt.ylabel("Fitness (accuracy)")
        plt.show()
        
    def numeric_plot(self,feature):
        print(feature)
        
    def categorical(self):
        return 0
        
###############################################################################
#################################Time To Rumble################################
###############################################################################
        
"""parameters: population_size,mutation_prob,elitism,crossover_prob"""

initThis = Genetic_Algorithm(population_size=100,mutation_prob=0.25,elitism=0.05,crossover_prob=0.87)
initThis.Main() 
np.argmax(initThis.fitness["Generation %s"%56])
initThis.fitness["Generation %s"%56][99]
initThis.population[99]
initThis.population[17]        
#######################################################################################

file_pi = open('GA_Evolved_1.obj', 'wb') 
pickle.dump(initThis, file_pi,pickle.HIGHEST_PROTOCOL)

with open('GA_Evolved_1.obj', 'rb') as input:
    load_test = pickle.load(input)


results = Analytics(load_test)

