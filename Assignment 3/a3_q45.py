%matplotlib inline 
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from hmmlearn import hmm



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
        
    def probs(self): 
        return self.probability_distribution
    
    def vals(self): 
        print(self.type)
        return self.values 
            
def markov_chain(transmat, state, state_names, samples): 
    (rows, cols) = transmat.shape 
    rvs = [] 
    values = list(np.arange(0,rows))
    
    # create random variables for each row of transition matrix 
    for r in range(rows): 
        rv = Random_Variable("row" + str(r), values, transmat[r])
        rvs.append(rv)
    
    # start from initial state and then sample the appropriate 
    # random variable based on the state following the transitions 
    states = [] 
    for n in range(samples): 
        state = rvs[state].sample(1)[0]    
        states.append(state_names[state])
    return states


# YOUR CODE GOES HERE 
%matplotlib inline 
import matplotlib.pyplot as plt
import numpy as np 
from hmmlearn import hmm


states = ['CGD', 'CGS']
observations = ['A', 'C', 'G', 'T']
start_probs = np.array([0.5, 0.5])
transmat1 = np.array([[0.85, 0.15], [0.15, 0.85]])
transmat2 = np.array([[0.15, 0.35, 0.35, 0.15], [0.40, 0.10, 0.10, 0.40]])
model = hmm.CategoricalHMM(n_components=2)
model.startprob_ = start_probs
model.transmat_ = transmat1
model.emissionprob_ = transmat2
model.n_features = len(observations)

# Set the number of samples (n_samples)
n_samples = 1000
X, Z = model.sample(n_samples)

def plot_DNA(samples, state2color, title): 
    colors = [state2color[x] for x in samples]
    x = np.arange(0, len(colors))
    y = np.ones(len(colors))
    plt.figure(figsize=(10,1))
    plt.bar(x, y, color=colors, width=1)
    plt.title(title)

# Define the color mapping
state2color = {0: 'black', 1: 'white'}
obj2color = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue'}

# Plot states and observations
plot_DNA(Z, state2color, 'States')
plt.show()
plot_DNA(X.flatten(), obj2color, 'Observations')
plt.show()



# Generate samples from the model
num_samples = 10000
generated_samples, hidden_states = model.sample(num_samples)

# Train a new model
new_model = hmm.CategoricalHMM(n_components=2, n_iter=1000000).fit(generated_samples)

# Display Transition Matrices
print("Transition Matrices:")
print("Estimated Model:")
print(new_model.transmat_)
print("\nOriginal Model:")
print(model.transmat_)

# Display Emission Probabilities
print("\nEmission Probabilities:")
print("Estimated Model:")
print(new_model.emissionprob_)
print("\nOriginal Model:")
print(model.emissionprob_)
