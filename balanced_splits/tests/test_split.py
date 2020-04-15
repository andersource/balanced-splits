import numpy as np
import pandas as pd
from balanced_splits.split import optimized_split, score, score_var, guess_var_type, VarType


def test_guess_var_type_nonnumeric():
    assert guess_var_type([0, 1, 2, 3, 4, 'hello']) == VarType.CATEGORICAL


def test_guess_var_type_normal():
    assert guess_var_type(np.random.normal(size=20)) == VarType.CONTINUOUS


def test_guess_var_type_few_values():
    assert guess_var_type(np.random.choice([1, 2, 3, 4], size=30)) == VarType.CATEGORICAL


def test_guess_var_type_many_values():
    assert guess_var_type(np.arange(30)) == VarType.CONTINUOUS


def test_score_var_categorical():
    assert score_var([
        np.random.choice([1,2,3], size=20),
        np.random.choice([1,2,4], size=20)
    ], VarType.CATEGORICAL) < .1

    assert score_var([
        [1, 1, 1, 2, 1, 1, 3, 1, 3, 2, 2, 1, 3, 3, 3, 3, 1, 2, 1, 2, 3, 3, 3, 2, 3],
        [3, 2, 1, 3, 3, 1, 3, 1, 1, 1, 3, 2, 3, 2, 3, 3, 1, 2, 3, 3, 2, 2, 1, 2, 1]
    ], VarType.CATEGORICAL) > .9


def test_score_var_continuous():
    assert score_var([
        np.random.normal(size=100),
        np.random.normal(loc=1., scale=2., size=100)
    ], VarType.CONTINUOUS) < .1

    assert score_var([
        [0.217, 0.907, 1.78, 0.756, -1.171, 0.006, 1.048, 0.143, 0.16, -0.176, 0.552, -1.327, 1.123, -1.395, 0.931, 2.093, 0.334, -0.711, -0.226, -0.11],
        [-1.417, 1.758, -1.194, 0.908, 0.53, 0.12, -1.35, 1.026, 0.312, 2.071, 1.1, 0.195, 0.734, 0.137, -0.248, 0.884, -0.199, -0.733, -0.133, -0.016],
        [-0.003, 0.047, 0.052, -0.894, -0.248, -1.253, 0.595, 1.028, -1.384, 1.817, -0.138, 0.681, 0.114, 1.024, -0.06, 0.325, 0.949, 2.114, 1.13, -1.255]
    ], VarType.CONTINUOUS) > .9


def test_score_bad():
    A = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=-5, size=100),
        np.random.choice(['f', 'm'], size=100).astype('O')
    ]).T
    B = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=-5, size=100),
        np.random.choice(['f', 'm', 'f', 'f'], size=100).astype('O')
    ]).T

    assert score([A, B],
                [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL]) < .1

    A = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=-5, size=100),
        np.random.choice(['f', 'm'], size=100).astype('O')
    ]).T
    B = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=5, size=100),
        np.random.choice(['f', 'm'], size=100).astype('O')
    ]).T

    assert score([A, B],
                [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL]) < .1


def test_score_good():
    A = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=-5, size=100),
        np.random.choice(['f', 'm'], size=100).astype('O')
    ]).T
    B = A.copy()
    B[:, 0] += np.random.normal(scale=.1, size=B.shape[0])
    B[:, 1] += np.random.normal(scale=.2, size=B.shape[0])
    swap_ind = np.random.choice(np.arange(B.shape[0]))
    B[swap_ind, 2] = 'm' if B[swap_ind, 2] == 'f' else 'f'

    assert score([A, B],
                [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL]) > .9


def test_optimized_split_basic():
    X = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=-5, size=100),
        np.random.choice(['f', 'm'], size=100).astype('O')
    ]).T

    A, B = optimized_split(X)

    assert score([A, B],
                [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL]) > .95


def test_optimized_split_df():
    X = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=-5, size=100),
        np.random.choice(['f', 'm'], size=100).astype('O')
    ]).T

    df = pd.DataFrame(data=X, columns=['a', 'b', 'c'])

    A, B = optimized_split(df)

    assert score([A, B],
                [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL]) > .95


def test_optimized_split_multiple():
    np.random.seed(0)
    # In rare cases X is generated such that it's very difficult to split
    # to 6 parts in a balanced manner so we fix the seed

    X = np.vstack([
        np.random.normal(size=120),
        np.random.normal(loc=-5, size=120),
        np.random.choice(['f', 'm'], size=120).astype('O')
    ]).T

    parts = optimized_split(X, n_partitions=6, max_iter=3000, score_threshold=.9)

    assert score(parts,
                [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL]) >= .9


def test_optimized_split_more():
    binary_feature = np.concatenate([np.ones(18), np.zeros(102)])
    np.random.shuffle(binary_feature)

    X = np.vstack([
        np.random.normal(loc=45, scale=5, size=120),
        binary_feature
    ]).T

    parts = optimized_split(X, n_partitions=3, max_iter=2000)

    for part in parts:
        assert np.abs(45 - np.mean(part[:, 0])) < 1.
        assert part[:, 1].sum() == 6


def test_optimized_split_callback():
    X = np.vstack([
        np.random.normal(size=100),
        np.random.normal(loc=-5, size=100),
        np.random.choice(['f', 'm'], size=100).astype('O')
    ]).T

    scores = []
    def callback(partitions, score):
        scores.append(score)

    A, B = optimized_split(X, iter_callback=callback)

    assert len(scores) > 10 and scores[-1] == score([A, B],
                                                    [VarType.CONTINUOUS, VarType.CONTINUOUS, VarType.CATEGORICAL])
