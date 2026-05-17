import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxim = max(z)
        y = np.exp(z-maxim)
        for i in range(len(z)):
            z[i] = np.exp(z[i]-maxim)/sum(y)
        return np.round(z,4)
