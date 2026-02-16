from typing import List, Tuple
import random


def make_loso_folds(subject_ids: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Returns: [(test_subject, train_subjects), ...]
    """
    subject_ids = list(dict.fromkeys([str(s) for s in subject_ids]))
    folds = []
    for test_s in subject_ids:
        train_s = [s for s in subject_ids if s != test_s]
        folds.append((test_s, train_s))
    return folds


def split_train_val_subjects(train_subjects: List[str], n_val: int, seed: int) -> Tuple[List[str], List[str]]:
    """
    Subject-level split inside train pool.
    """
    assert 0 < n_val < len(train_subjects)
    rng = random.Random(seed)
    shuffled = train_subjects[:]
    rng.shuffle(shuffled)
    val_subjects = shuffled[:n_val]
    inner_train = shuffled[n_val:]
    return inner_train, val_subjects
