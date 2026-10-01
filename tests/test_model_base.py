import pytest
from unittest.mock import patch
from aiod.models.base._base import _AiodModelPkg


class DummySinglePkg(_AiodModelPkg):
    _tags = {
        "pkg_id": "SingleModel",
        "pkg_obj": "reference",
        "pkg_pypi_name": "dummy_pkg",
    }
    _obj = "dummy_pkg.SingleModel"


class DummyMultiplePkg(_AiodModelPkg):
    _tags = {
        "pkg_id": "__multiple",
        "pkg_obj": "reference",
        "pkg_pypi_name": "dummy_pkg",
    }
    _obj_dict = {
        "ModelA": "dummy_pkg.ModelA",
        "ModelB": "dummy_pkg.ModelB",
    }


class DummyCodePkg(_AiodModelPkg):
    _tags = {
        "pkg_id": "CodeModel",
        "pkg_obj": "code",
        "pkg_pypi_name": "dummy_pkg",
    }
    _obj = "obj = 'executed_code_result'"


def test_contained_ids_single():
    assert DummySinglePkg.contained_ids() == ["SingleModel"]


def test_contained_ids_multiple():
    ids = DummyMultiplePkg.contained_ids()
    assert set(ids) == {"ModelA", "ModelB"}


def test_init_multiple_resolves_obj():
    pkg = DummyMultiplePkg(id="ModelB")
    assert pkg._obj == "dummy_pkg.ModelB"


def test_init_single_ignores_id_for_obj():
    pkg = DummySinglePkg(id="ignored_id")
    assert pkg._obj == "dummy_pkg.SingleModel"


@patch("skbase.utils.dependencies._safe_import")
@patch("aiod.base._base_pkg._check_estimator_deps")
def test_materialize_reference(mock_check_deps, mock_safe_import):
    mock_safe_import.return_value = "mocked_class"
    
    pkg = DummySinglePkg()
    result = pkg.materialize()
    
    mock_check_deps.assert_called_once_with(obj=pkg)
    mock_safe_import.assert_called_once_with("dummy_pkg.SingleModel", pkg_name="dummy_pkg")
    assert result == "mocked_class"


@patch("aiod.base._base_pkg._check_estimator_deps")
def test_materialize_code(mock_check_deps):
    pkg = DummyCodePkg()
    result = pkg.materialize()
    
    mock_check_deps.assert_called_once_with(obj=pkg)
    assert result == "executed_code_result"


@patch("aiod.base._base_pkg._check_estimator_deps")
def test_materialize_invalid_obj_none(mock_check_deps):
    class InvalidPkg(_AiodModelPkg):
        _tags = {"pkg_obj": "reference"}
        _obj = None

    pkg = InvalidPkg()
    with pytest.raises(ValueError, match="the _obj attribute must be not None"):
        pkg.materialize()


@patch("aiod.base._base_pkg._check_estimator_deps")
def test_materialize_invalid_pkg_obj_tag(mock_check_deps):
    class InvalidTagPkg(_AiodModelPkg):
        _tags = {"pkg_obj": "invalid_tag"}
        _obj = "some.obj"

    pkg = InvalidTagPkg()
    with pytest.raises(ValueError, match="must be one of"):
        pkg.materialize()
