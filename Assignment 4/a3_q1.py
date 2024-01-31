
# Modify this code as needed 

import numpy as np
from scipy import stats


class Random_Variable: 
    
    def __init__(self, name, values, probability_distribution): 
        self.name = name 
        self.values = values 
        self.probability_distribution = probability_distribution
        if all(type(item) is np.int64 for item in values):
            self.type = 'numeric'
            self.rv = stats.rv_discrete(name = name, values = (values, probability_distribution))
        elif all(type(item) is str for item in values): 
            self.type = 'symbolic'
            self.rv = stats.rv_discrete(name = name, values = (np.arange(len(values)), probability_distribution))
            self.symbolic_values = values 
        else: 
            self.type = 'undefined'
            
    def sample(self,size): 
        if (self.type =='numeric'):
            return self.rv.rvs(size=size)
        elif (self.type == 'symbolic'): 
            numeric_samples = self.rv.rvs(size=size)
            mapped_samples = [self.values[x] for x in numeric_samples]
            return mapped_samples

    def get_name(self):
        return self.name
    
valuesA = np.array([2, 2, 4, 4, 9, 9])
valuesB = np.array([1, 1, 6, 6, 8, 8])
valuesC = np.array([3, 3, 5, 5, 7, 7])

# Define the probability distributions for the three dice
probabilities_Red = np.array([1/6., 1/6., 1/6., 1/6., 1/6., 1/6.])
probabilities_Green = np.array([1/6., 1/6., 1/6., 1/6., 1/6., 1/6.])
probabilities_Blue = np.array([1/6., 1/6., 1/6., 1/6., 1/6., 1/6.])



# Create instances of the Random_Variable class for each die
die_Red = Random_Variable('Red', valuesA, probabilities_Red)
die_Green = Random_Variable('Green', valuesB, probabilities_Green)
die_Blue = Random_Variable('Blue', valuesC, probabilities_Blue)

def dice_war(A,B, num_samples = 1000, output=True):
    # your code goes here 
    samples_A = A.sample(num_samples)
    samples_B = B.sample(num_samples)
    wins_A = 0
    for value_A, value_B in zip(samples_A, samples_B):
        if value_A > value_B:
            wins_A += 1

    
    # Calculate the probability of A winning
    prob = wins_A / num_samples
    res = prob > 0.5
    
    if output: 
        if res:
            print('{} beats {} with probability {}'.format(A.get_name(),
                                                           B.get_name(),
                                                           prob))
        else:
            print('{} beats {} with probability {:.2f}'.format(B.get_name(),
                                                               A.get_name(),
                                                               1.0-prob))
    return (res, prob)
        


dice_war(die_Red, die_Green)
dice_war(die_Green, die_Blue)
dice_war(die_Blue, die_Red)



