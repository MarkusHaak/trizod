from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import count_fasta


def test_pinned_testset_path_and_file_present():
    paths = resolve_paths()
    assert paths.pinned_testset.name == "TriZOD_test_set.fasta"
    assert paths.pinned_testset.parent.name == "pinned"
    assert paths.pinned_testset.exists(), "committed pinned test set is missing"
    assert count_fasta(paths.pinned_testset) == 342
