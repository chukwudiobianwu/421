'''
Let's consider a scenario where we are dealing with the content preferences of users on a streaming platform. We have five hypotheses (H1 to H5) representing different user profiles based on the amount of time they spend watching two genres, drama (D) and comedy (C).

For each hypothesis, we model the user's preference with a continuous probability density function using a Normal distribution. The mean (μ) and variance (σ^2) of each distribution represent the average time spent and the variability in preferences for the respective genres.

Hypotheses:

H1: Users who prefer a balanced mix of drama and comedy.
H2: Users who strongly prefer drama.
H3: Users who strongly prefer comedy.
H4: Users who moderately prefer drama but also enjoy comedy.
H5: Users who moderately prefer comedy but also enjoy drama.
Now, we assign some parameters to these hypotheses:

H1: Normal distribution with μ=50 (balanced) and σ^2=25.
H2: Normal distribution with μ=70 (strongly prefers drama) and σ^2=20.
H3: Normal distribution with μ=30 (strongly prefers comedy) and σ^2=15.
H4: Normal distribution with μ=60 (moderately prefers drama) and σ^2=30.
H5: Normal distribution with μ=40 (moderately prefers comedy) and σ^2=18.
Assume that we collect data on the time users spend watching drama and comedy, and we obtain successive samples. We can use this data to perform maximum likelihood estimation (MLE), maximum a posteriori estimation (MAP), and Bayesian learning.

For simplicity, let's say we have collected data for a user and observed that they spent 55% of their time watching drama and 45% watching comedy.

We can use this data to update our beliefs about each hypothesis. The formulas for the Normal distribution and Bayesian updating are used in the calculations.

Maximum Likelihood Estimation (MLE):
For MLE, we maximize the likelihood function, which is the product of the probabilities of observing the given data under each hypothesis. In the case of a Normal distribution, it involves finding the mean and variance that maximize the likelihood.

Given a user's observed data (55% drama, 45% comedy), we can calculate the likelihood for each hypothesis and choose the one with the highest likelihood.

The likelihood for hypothesis H1 is the probability of observing the given data under the assumption that users prefer a balanced mix of drama and comedy, modeled by a Normal distribution with a mean (μ) of 50 and a variance (σ^2) of 25.

Similarly,we calculate the likelihood for H2, H3, H4, and H5. The hypothesis with the highest likelihood is the MLE estimate.

Maximum A Posteriori Estimation (MAP):
we incorporate prior beliefs about the hypotheses by multiplying the likelihood with the prior probability of each hypothesis. The prior represents our initial belief in the likelihood of each hypothesis before observing data.

The posterior probability for hypothesis H1 is proportional to the product of the prior probability for H1 and the likelihood of observing the given data under the assumption that users prefer a balanced mix of drama and comedy, 
represented by a Normal distribution with a mean (μ) of 50 and a variance (σ^2) of 25

Similarly, we calculate the posterior for H2, H3, H4, and H5. The hypothesis with the highest posterior is the MAP estimate.

Bayesian Learning:
For each new sample, we update the posterior using Bayes' theorem which is:
The updated posterior probability is proportional to the product of the likelihood, representing the probability of observing the given data under a specific hypothesis, 
and the prior probability, which reflects the initial belief in that hypothesis before considering the observed data.

Using the updated posterior as the new prior, we repeat this process for each successive sample. The final posterior represents our updated beliefs about the user's preferences based on all the observed data.

we also have to normalize the posterior probabilities so that they sum to 1 after each update.

and all these calculations are performed using the Normal distribution probability density function and Bayesian updating formulas
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define hypotheses
hypotheses = ['H1', 'H2', 'H3', 'H4', 'H5']
means = [50, 70, 30, 60, 40]
variances = [25, 20, 15, 30, 18]

# User's observed data
observed_drama_percentage = 0.55
observed_comedy_percentage = 0.45

# Plot the initial hypotheses
x = np.linspace(0, 100, 1000)
plt.figure(figsize=(10, 6))

for i in range(len(hypotheses)):
    y = norm.pdf(x, loc=means[i], scale=np.sqrt(variances[i]))
    plt.plot(x, y, label=hypotheses[i])

plt.title('Initial Hypotheses')
plt.xlabel('Time Spent on Drama')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Bayesian updating based on observed data
for i in range(len(hypotheses)):
    likelihood_drama = norm.pdf(observed_drama_percentage, loc=means[i]/100, scale=np.sqrt(variances[i])/100)
    likelihood_comedy = norm.pdf(observed_comedy_percentage, loc=(100-means[i])/100, scale=np.sqrt(variances[i])/100)

    # Assuming a uniform prior for simplicity
    prior = 1

    # Bayesian updating
    posterior = prior * likelihood_drama * likelihood_comedy
    plt.bar(hypotheses[i], posterior, alpha=0.5)

plt.title('Posterior Probabilities after Bayesian Updating')
plt.xlabel('Hypotheses')
plt.ylabel('Posterior Probability')
plt.show()
