import numpy as np
from scipy.stats import ttest_ind, ks_2samp, fisher_exact

def random_feature_effect_coef():
    if np.random.random() <= .5:
        return np.random.normal(loc=1., scale=.5)
    return np.random.normal(loc=-1., scale=.5)

def random_split(X):
    labels = np.concatenate([np.zeros(X.shape[0] // 2), np.ones(X.shape[0] // 2)])
    np.random.shuffle(labels)

    A = X[np.where(labels == 0)]
    B = X[np.where(labels == 1)]

    return A, B


def run_experiment(feature_effect):
    # returns tuple (ground_truth, conclusion)
    # 0 - treatment not effective, 1 - treatment effective
    sample_size = 2 * np.round(np.random.uniform(50, 200)).astype(int)

    treatment_effect = np.random.normal(loc=1., scale=.5) if np.random.random() <= .5 else 0

    X = np.random.normal(size=(sample_size, 1))
    control_group, treatment_group = random_split(X)

    control_results = (np.random.normal(size=control_group.shape[0]) +
                        (control_group * feature_effect)[:, 0])
    treatment_results = (np.random.normal(size=treatment_group.shape[0]) +
                            (treatment_group * feature_effect)[:, 0]  +
                            treatment_effect)

    conclusion = 0

    if (np.mean(treatment_results) > np.mean(control_results) and
        ttest_ind(control_results, treatment_results).pvalue <= .05):
        conclusion = 1


    return int(treatment_effect != 0), conclusion



def main():
    N = 10000
    res = []

    for feature_strategy in ['no_effect', 'effect']:
        wrong_conclusions = 0
        false_positives = 0
        false_negatives = 0
        for i in range(N):
            ground_truth, conclusion = run_experiment(0 if feature_strategy == 'no_effect' else random_feature_effect_coef())
            if ground_truth != conclusion:
                wrong_conclusions += 1
                if ground_truth and not conclusion:
                    false_negatives += 1
                else:
                    false_positives += 1

        print('%s: %d / %d conclusions were wrong (%d FP, %d FN)' % (feature_strategy.ljust(20, ' '),
                                                      wrong_conclusions, N,
                                                      false_positives,
                                                      false_negatives))
        res.append(wrong_conclusions)

    print('Fisher\'s exact test pvalue - %.3f' % fisher_exact([[r, N - r] for r in res])[1])


if __name__ == '__main__':
    main()
