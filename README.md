# balanced-splits
A utility library for splitting datasets in a balanced manner, with regards to several features.

### Installation
`pip install balanced-splits`

### Usage
```python
import numpy as np
import pandas as pd
from balanced_splits.split import optimized_split

sample_size = 100
df = pd.DataFrame({
    'age': np.random.normal(loc=45, scale=7., size=sample_size),
    'skill': 1 - np.random.power(4, size=sample_size),
    'type': np.random.choice(['T1', 'T2', 'T3'], size=sample_size)
})

A, B = optimized_split(df)

print('Partition 1\n===========\n')
print(A.describe())
print(A['type'].value_counts())

print('\n\n')

print('Partition 2\n===========\n')
print(B.describe())
print(B['type'].value_counts())

```

Check out the "examples" section for more examples.
