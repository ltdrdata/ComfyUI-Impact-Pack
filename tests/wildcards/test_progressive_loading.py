"""Directory scanning, progressive TXT loading and loaded-list semantics."""
from impact import wildcards


def test_early_termination(tmp_path):
    for i in range(10):
        (tmp_path / f"test{i}.txt").write_text("x" * 10)
    assert wildcards.calculate_directory_size(tmp_path) == 100
    assert 50 <= wildcards.calculate_directory_size(tmp_path, limit=50) <= 100


def test_metadata_scan(wildcard_loader):
    (wildcard_loader / "first.txt").write_text("one\ntwo\n")
    (wildcard_loader / "second.txt").write_text("three\n")
    (wildcard_loader / "palette.yaml").write_text("colors: [red, blue]\n")
    assert wildcards.scan_wildcard_metadata(wildcard_loader) == 3
    assert wildcards.available_wildcards == {
        name: str(wildcard_loader / filename) for name, filename in (
            ("first", "first.txt"), ("second", "second.txt"), ("palette", "palette.yaml"))
    }
    assert wildcards.loaded_wildcards == {}
    assert wildcards.get_wildcard_dict() == {}


def test_progressive_loading(wildcard_loader):
    for i in range(1, 4):
        (wildcard_loader / f"wildcard{i}.txt").write_text(f"option{i}a\noption{i}b\n")
    wildcards._on_demand_mode = True
    assert wildcards.scan_wildcard_metadata(wildcard_loader) == 3
    assert wildcards.get_loaded_wildcard_list() == []
    for i in range(1, 4):
        assert wildcards.get_wildcard_value(f"wildcard{i}") == [f"option{i}a", f"option{i}b"]
        assert set(wildcards.get_loaded_wildcard_list()) == {f"__wildcard{j}__" for j in range(1, i + 1)}
        # A cache hit must not reread the edited file or load another entry.
        (wildcard_loader / "wildcard1.txt").write_text("edited on disk\n")
        assert wildcards.get_wildcard_value("wildcard1") == ["option1a", "option1b"]
        assert len(wildcards.get_loaded_wildcard_list()) == i


def test_wildcard_list_functions(wildcard_loader):
    for i in range(5):
        (wildcard_loader / f"test{i}.txt").write_text(f"option{i}\n")
    wildcards._on_demand_mode = True
    assert wildcards.scan_wildcard_metadata(wildcard_loader) == 5
    assert set(wildcards.available_wildcards) == {f"test{i}" for i in range(5)}
    assert wildcards.get_wildcard_list() == wildcards.get_loaded_wildcard_list() == []
    for i in range(5):
        assert wildcards.get_wildcard_value(f"test{i}") == [f"option{i}"]
        expected = {f"__test{j}__" for j in range(i + 1)}
        assert set(wildcards.get_wildcard_list()) == set(wildcards.get_loaded_wildcard_list()) == expected
