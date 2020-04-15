import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt

def count_values(v, max_value):
    counts = np.zeros(max_value)
    for n in v.astype(int):
        counts[n] += 1

    return counts

S = 200

age = np.random.normal(loc=40, scale=5, size=S)
usage = np.random.power(4, size=S)
gender = np.random.choice(np.arange(2), size=S)
language = np.random.choice(np.arange(10),
                            size=S)

data = np.stack([age, usage, gender, language]).T

def score(A, B):
    pvalues = [
        ks_2samp(A[:, 0], B[:, 0])[1],
        ks_2samp(A[:, 1], B[:, 1])[1],
        chi2_contingency([count_values(A[:, 2], 2), count_values(B[:, 2], 2)])[1],
        chi2_contingency([count_values(A[:, 3], 10), count_values(B[:, 3], 10)])[1],
    ]

    return min(pvalues)

def sa_split(data):
    def _score(labels):
        A = data[np.where(labels == 0)]
        B = data[np.where(labels == 1)]
        return score(A, B)

    def neighbor(labels):
        labels = labels.copy()

        A_ind = np.where(labels == 0)[0]
        B_ind = np.where(labels == 1)[0]
        labels[np.random.choice(A_ind)] = 1
        labels[np.random.choice(B_ind)] = 0

        return labels

    def T(i):
        T_DECAY = .99
        T_START = 1
        return T_START * np.power(T_DECAY, i)

    def P(curr_score, new_score, t):
        if new_score >= curr_score:
            return 1

        if t == 0:
            return 0

        return np.exp(-(curr_score - new_score) / t)

    labels = np.concatenate([np.zeros(data.shape[0] // 2), np.ones(data.shape[0] // 2)])
    np.random.shuffle(labels)

    MAX_ITERS = 1000
    best_score = _score(labels)
    for i in range(MAX_ITERS):
        new_sol = neighbor(labels)
        new_sol_score = _score(new_sol)
        if new_sol_score >= best_score or np.random.random() <= P(best_score, new_sol_score, T(i)):
            best_score = new_sol_score
            labels = new_sol

    A = data[np.where(labels == 0)]
    B = data[np.where(labels == 1)]

    return A, B

def main():
    A, B = sa_split(data)
    print(score(A, B))

# def main():
#     scores = []
#     for i in range(10):
#         scores.append(score(*sa_split(data)))
#
#     plt.hist(scores)
#     plt.show()


if __name__ == '__main__':
    main()
