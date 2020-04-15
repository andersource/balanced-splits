import numpy as np
from balanced_splits.split import optimized_split, score, VarType

X = np.vstack([
    np.random.normal(size=100),
    np.random.normal(loc=-5, size=100),
    np.random.choice(['f', 'm'], size=100).astype('O')
]).T

A, B = optimized_split(X)

print(score([A, B],
            [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL]))
