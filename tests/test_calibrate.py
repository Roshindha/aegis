from eval.calibrate import best_threshold, cross_validate, f1_scores, stratified_folds


def test_separable_scores_give_perfect_threshold():
    scores = [-9, -8, -7, 2, 3, 4]
    labels = [True, True, True, False, False, False]
    t = best_threshold(scores, labels)
    assert -7 < t < 2
    assert f1_scores(scores, labels, t)["f1"] == 1.0


def test_f1_counts():
    m = f1_scores([-1, -1, 1, 1], [True, False, True, False], 0)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_folds_partition_and_stratify():
    labels = [True] * 10 + [False] * 40
    folds = stratified_folds(labels, 5, seed=1)
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(50))
    assert all(sum(labels[i] for i in f) == 2 for f in folds)


def test_cross_validate_on_separable_data_is_perfect():
    scores = [-9 - i * 0.1 for i in range(10)] + [3 + i * 0.1 for i in range(40)]
    labels = [True] * 10 + [False] * 40
    assert cross_validate(scores, labels)["f1"] == 1.0
