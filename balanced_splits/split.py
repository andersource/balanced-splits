"""Balanced splitting utility"""

from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


class VarType(Enum):
    """Enum class describing the statistical type of a variable"""
    CATEGORICAL = 0
    CONTINUOUS = 1


def guess_var_type(x):
    """Use heuristics to guess at a variable's statistical type"""
    if type(x) == list:
        x = np.array(x)

    if x.dtype == 'O':
        try:
            x = x.astype(float)
        except ValueError:
            pass

    if not np.issubdtype(x.dtype, np.number):
        return VarType.CATEGORICAL

    if np.unique(x).shape[0] / x.shape[0] <= .2:
        return VarType.CATEGORICAL

    return VarType.CONTINUOUS


def score(partitions, var_types):
    """Score the balance of a particular split of a dataset"""
    return np.min([
        score_var([_get_accessor(partition)[:, i] for partition in partitions], var_types[i])
        for i in range(len(var_types))
    ])


def score_var(var_partitions, var_type):
    """Score the balance of a single variable in a certain split of a dataset"""
    if var_type == VarType.CATEGORICAL:
        unique_values = np.unique(np.concatenate(var_partitions))
        value_counts = count_values(var_partitions, unique_values)
        return chi2_contingency(value_counts)[1]

    pvalues = []
    for i in range(len(var_partitions)):
        other_partitions = [var_partitions[j]
            for j in range(len(var_partitions)) if j != i]
        pvalues.append(ks_2samp(var_partitions[i],
                                np.concatenate(other_partitions))[1])

    return np.min(pvalues)


def count_values(var_partitions, unique_values):
    """Count the number of appearances of each unique value in each list"""
    value2index = {v: k for k, v in dict(enumerate(unique_values)).items()}
    counts = np.zeros((len(var_partitions), len(unique_values)))
    for i in range(len(var_partitions)):
        for value in var_partitions[i]:
            counts[i, value2index[value]] += 1

    return counts


def optimized_split(X, n_partitions=2, t_start=1,
                    t_decay=.99, max_iter=1000,
                    score_threshold=.99, var_types=None,
                    iter_callback=None):
    """
    Perform an optimized split of a dataset using (basic) simulated annealing.

    Parameters
    ----------

    X : 2d-ndarray or DataFrame
        Dataset to split.

    n_partitions : Number of partitions to split the dataset to (default 2)

    t_start : Initial temperature of simulated annealing (default 1.)

    t_decay : Temperature decay rate of simulated annealing (default .99)

    max_iter : Maximum iterations of the simulated annealing (default 1000)

    score_threshold : Float in the range [0, 1) (default .99)
                      Desired split quality. If this quality is reached
                      before `max_iter` the function breaks and
                      returns the current best solution.

    var_types : None or list of VarType (default None)
                List of variable statistical types in the dataset.
                Used to calculate the quality of split alternatives.
                If None, variable types are guessed based on simple heuristics.

    iter_callback : Callable or None (default None)
                    If callable, it is called on every iteration with
                    two parameters: the current split and its score.

    Returns
    -------

    partitions : List of ndarrays or DataFrames
                 Split dataset.
    """
    X_shape = X.shape
    X = _get_accessor(X)

    if var_types is None:
        var_types = [guess_var_type(X[:, i]) for i in range(X_shape[1])]

    def _score(indices):
        partitions = [X[i] for i in indices]

        return score(partitions, var_types)

    def _neighbor(curr_indices):
        curr_indices = np.copy(curr_indices)

        part1, part2 = np.random.choice(np.arange(len(curr_indices)),
                        size=2, replace=False)
        part1_ind = np.random.choice(np.arange(curr_indices[part1].shape[0]))
        part2_ind = np.random.choice(np.arange(curr_indices[part2].shape[0]))
        temp = curr_indices[part1][part1_ind]
        curr_indices[part1][part1_ind] = curr_indices[part2][part2_ind]
        curr_indices[part2][part2_ind] = temp

        return curr_indices

    def _T(i):
        return t_start * np.power(t_decay, i)

    def _P(curr_score, new_score, t):
        if new_score >= curr_score:
            return 1

        if t == 0:
            return 0

        return np.exp(-(curr_score - new_score) / t)

    all_indices = np.arange(X_shape[0])
    np.random.shuffle(all_indices)
    indices = np.array_split(all_indices, n_partitions)

    best_score = _score(indices)
    for i in range(max_iter):
        new_indices = _neighbor(indices)
        new_indices_score = _score(new_indices)
        if (new_indices_score >= best_score or
            np.random.random() <= _P(best_score, new_indices_score, _T(i))):
            best_score = new_indices_score
            indices = new_indices

        if iter_callback is not None:
            iter_callback([X[i] for i in indices], best_score)

        if best_score >= score_threshold:
            break

    return [X[i] for i in indices]

def _get_accessor(X):
    """Get indexable object from X"""
    if type(X) == pd.DataFrame:
        return X.iloc

    return X
