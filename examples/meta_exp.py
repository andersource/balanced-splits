import numpy as np
from scipy.stats import ttest_ind, ks_2samp, fisher_exact

def random_feature_effect_coef():
    if np.random.random() > .2:
        return 0

    if np.random.random() <= .5:
        return np.random.normal(loc=1., scale=.5)
    return np.random.normal(loc=-1., scale=.5)

def random_split(X):
    labels = np.concatenate([np.zeros(X.shape[0] // 2), np.ones(X.shape[0] // 2)])
    np.random.shuffle(labels)

    A = X[np.where(labels == 0)]
    B = X[np.where(labels == 1)]

    return A, B


def optimized_split(X):
    def _score(labels):
        A = X[np.where(labels == 0)]
        B = X[np.where(labels == 1)]

        return min([ks_2samp(A[:, feature], B[:, feature])[1] for feature in range(X.shape[1])])

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

    labels = np.concatenate([np.zeros(X.shape[0] // 2), np.ones(X.shape[0] // 2)])
    np.random.shuffle(labels)

    MAX_ITERS = 1000
    best_score = _score(labels)
    for i in range(MAX_ITERS):
        new_sol = neighbor(labels)
        new_sol_score = _score(new_sol)
        if new_sol_score >= best_score or np.random.random() <= P(best_score, new_sol_score, T(i)):
            best_score = new_sol_score
            labels = new_sol

    A = X[np.where(labels == 0)]
    B = X[np.where(labels == 1)]

    return A, B


def run_experiment(split_strategy):
    # returns tuple (ground_truth, conclusion)
    # 0 - treatment not effective, 1 - treatment effective
    sample_size = 2 * np.round(np.random.uniform(25, 100)).astype(int)
    n_features = np.round(np.random.uniform(3, 7)).astype(int)

    treatment_effect = np.random.normal(loc=1., scale=.5) if np.random.random() <= .5 else 0
    feature_effects = [random_feature_effect_coef() for i in range(n_features)]

    X = np.random.normal(size=(sample_size, n_features))
    control_group, treatment_group = split_strategy(X)

    control_results = (np.random.normal(size=control_group.shape[0]) +
                        np.dot(control_group, feature_effects))
    treatment_results = (np.random.normal(size=treatment_group.shape[0]) +
                            np.dot(treatment_group, feature_effects) +
                            treatment_effect)

    conclusion = 0
    if (np.mean(treatment_results) > np.mean(control_results) and
        ttest_ind(control_results, treatment_results).pvalue <= .05):
        conclusion = 1


    return int(treatment_effect != 0), conclusion



def main():
    N = 10000
    res = []

    for split_strategy in [random_split, optimized_split]:
        wrong_conclusions = 0
        false_positives = 0
        false_negatives = 0
        for i in range(N):
            ground_truth, conclusion = run_experiment(split_strategy)
            if ground_truth != conclusion:
                wrong_conclusions += 1
                if ground_truth and not conclusion:
                    false_negatives += 1
                else:
                    false_positives += 1

        print('%s: %d / %d conclusions were wrong (%d FP, %d FN)' % (split_strategy.__name__.ljust(20, ' '),
                                                      wrong_conclusions, N,
                                                      false_positives,
                                                      false_negatives))
        res.append(wrong_conclusions)

    print('Fisher\'s exact test pvalue - %.3f' % fisher_exact([[r, N - r] for r in res])[1])


if __name__ == '__main__':
    main()
