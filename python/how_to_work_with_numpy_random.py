## How to work with numpy.random

import numpy as np

# Randomly sample from a given distribution:
#   np.random.choice([A list of choices], p=[A list of probabilities that sum to 1])
# Example:
np.random.choice([1,2,4], p=[0.2,0.5,0.3])

# Randomly sample a float from [0,1)
# Example:
np.random.random()