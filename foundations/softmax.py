import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        x = 0
        y = max(z)
        for j in range(len(z)):
            x = x + np.exp(z[j]-y)
        for i in range(len(z)):
            z[i] = np.exp(z[i]-y)/x
            z[i] = np.round(z[i],4)
        return z
