import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Define activities and locations
activities = ['eat', 'study', 'exercise']
locations = ['Restaurant A', 'Restaurant B', 'University A', 'University B', 'Gym A', 'Gym B']

# Number of states and components
n_states = len(activities)
n_components = len(locations)

# Generate means and covariances for each GMM component
means = np.random.rand(n_states, n_components, 2)  # 2D coordinates
covariances = np.random.rand(n_states, n_components, 2, 2)  # 2x2 covariance matrices

# Create HMM model
model = hmm.GMMHMM(n_components=n_components, n_mix=n_states, covariance_type='full')

# Set initial parameters
model.means_ = means
model.covars_ = covariances

# Set transition probabilities (for simplicity, you can set them uniformly)
model.transmat_ = np.ones((n_components, n_components)) / n_components

# Plot locations and GMM components
plt.figure(figsize=(10, 8))

for i, activity in enumerate(activities):
    for j, location in enumerate(locations):
        mean = means[i, j]
        covariance = covariances[i, j]

        # Plot GMM components as circles
        v, w = np.linalg.eigh(covariance)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=f'C{i}', alpha=0.5)
        plt.gca().add_patch(ell)

        # Plot location points
        plt.scatter(mean[0], mean[1], color=f'C{i}', marker='o', label=f'{activity} - {location}')

plt.title('Locations and GMM Components')
plt.legend()
plt.show()