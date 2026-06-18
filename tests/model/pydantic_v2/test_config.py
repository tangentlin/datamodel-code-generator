"""Tests for Pydantic v2 config generation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from datamodel_code_generator import AliasGenerator
from datamodel_code_generator.model import DataModelFieldBase, pydantic_v2
from datamodel_code_generator.model.msgspec import Constraints as MsgspecConstraints
from datamodel_code_generator.model.pydantic_base import PatternConstraints
from datamodel_code_generator.model.pydantic_v2.base_model import (
    _CONFIG_ITEMS_TEMPLATE_DATA_KEY,
    BaseModel,
    DataModelField,
    _alias_generator_name,
    _config_dict_items,
    _generate_alias,
)
from datamodel_code_generator.model.pydantic_v2.base_model import Constraints as PydanticV2Constraints
from datamodel_code_generator.model.pydantic_v2.dataclass import DataClass
from datamodel_code_generator.reference import Reference
from datamodel_code_generator.types import DataType


def _extra_template_data() -> defaultdict[str, dict[str, Any]]:
    return defaultdict(
        dict,
        {
            "Model": {
                "additionalProperties": False,
                "allow_population_by_field_name": True,
                "use_attribute_docstrings": True,
            }
        },
    )


def _field() -> DataModelFieldBase:
    return DataModelFieldBase(name="a", data_type=DataType(type="str"), required=True)


def _reference() -> Reference:
    return Reference(name="Model", path="Model")


@pytest.mark.allow_direct_assert
def test_config_dict_reexport_preserves_public_surface() -> None:
    """ConfigDict remains importable from the package for compatibility."""
    assert pydantic_v2.ConfigDict.__module__ == "datamodel_code_generator.model.pydantic_v2"
    assert "ConfigDict" not in pydantic_v2.__all__


@pytest.mark.allow_direct_assert
def test_base_model_config_key_order_with_multiple_shared_parameters() -> None:
    """BaseModel config generation keeps deterministic key ordering."""
    model = BaseModel(
        fields=[_field()],
        reference=_reference(),
        extra_template_data=_extra_template_data(),
    )

    config = model.extra_template_data["config"]
    assert isinstance(config, pydantic_v2.ConfigDict)
    assert list(config.dict(exclude_unset=True)) == ["extra", "populate_by_name", "use_attribute_docstrings"]
    assert model.extra_template_data[_CONFIG_ITEMS_TEMPLATE_DATA_KEY] == [
        ("extra", "'forbid'"),
        ("populate_by_name", True),
        ("use_attribute_docstrings", True),
    ]
    assert (
        "model_config = ConfigDict(\n"
        "        extra='forbid',\n"
        "        populate_by_name=True,\n"
        "        use_attribute_docstrings=True,\n"
        "    )"
    ) in model.render()


@pytest.mark.allow_direct_assert
def test_base_model_config_alias_generator_order_and_import() -> None:
    """BaseModel config places alias_generator after populate_by_name."""
    extra_template_data = defaultdict(
        dict,
        {
            "Model": {
                "allow_population_by_field_name": True,
                "alias_generator": "to_camel",
            }
        },
    )
    model = BaseModel(
        fields=[_field()],
        reference=_reference(),
        extra_template_data=extra_template_data,
    )

    config = model.extra_template_data["config"]
    assert isinstance(config, pydantic_v2.ConfigDict)
    assert list(config.dict(exclude_unset=True)) == ["populate_by_name", "alias_generator"]
    rendered = model.render()
    assert any(
        import_.from_ == "pydantic.alias_generators" and import_.import_ == "to_camel" for import_ in model.imports
    )
    assert (
        "model_config = ConfigDict(\n        populate_by_name=True,\n        alias_generator=to_camel,\n    )"
    ) in rendered


@pytest.mark.allow_direct_assert
def test_base_model_alias_generator_adds_field_import_for_mismatch() -> None:
    """Alias generator mismatch fields still import Field."""
    extra_template_data = defaultdict(dict, {"Model": {"alias_generator": "to_camel"}})
    model = BaseModel(
        fields=[
            DataModelField(
                name="foo_bar",
                original_name="foo_bar",
                data_type=DataType(type="str"),
                required=True,
            )
        ],
        reference=_reference(),
        extra_template_data=extra_template_data,
    )

    rendered = model.render()
    assert any(import_.from_ == "pydantic" and import_.import_ == "Field" for import_ in model.imports)
    assert "foo_bar: str = Field(..., alias='foo_bar')" in rendered


@pytest.mark.allow_direct_assert
def test_base_model_alias_generator_omits_matching_alias_field_import() -> None:
    """Alias generator matching fields do not import Field just for an alias."""
    extra_template_data = defaultdict(dict, {"Model": {"alias_generator": "to_camel"}})
    model = BaseModel(
        fields=[
            DataModelField(
                name="first_name",
                original_name="firstName",
                alias="firstName",
                data_type=DataType(type="str"),
                required=True,
            )
        ],
        reference=_reference(),
        extra_template_data=extra_template_data,
    )

    rendered = model.render()
    assert not any(import_.from_ == "pydantic" and import_.import_ == "Field" for import_ in model.imports)
    assert "first_name: str" in rendered
    assert "Field(" not in rendered


@pytest.mark.allow_direct_assert
def test_base_model_alias_generator_variants_omit_matching_aliases() -> None:
    """Alias generator variants omit matching per-field aliases."""
    cases = [
        ("to_pascal", "first_name", "FirstName"),
        ("to_snake", "firstName", "first_name"),
    ]
    for alias_generator, field_name, wire_name in cases:
        extra_template_data = defaultdict(dict, {"Model": {"alias_generator": alias_generator}})
        model = BaseModel(
            fields=[
                DataModelField(
                    name=field_name,
                    original_name=wire_name,
                    alias=wire_name,
                    data_type=DataType(type="str"),
                    required=True,
                )
            ],
            reference=_reference(),
            extra_template_data=extra_template_data,
        )

        rendered = model.render()
        assert not any(import_.from_ == "pydantic" and import_.import_ == "Field" for import_ in model.imports)
        assert f"{field_name}: str" in rendered
        assert "Field(" not in rendered


@pytest.mark.allow_direct_assert
def test_base_model_alias_generator_keeps_non_alias_early_returns() -> None:
    """Alias generator field processing handles ClassVar and unnamed wire fields."""
    extra_template_data = defaultdict(dict, {"Model": {"alias_generator": "to_camel"}})
    model = BaseModel(
        fields=[
            DataModelField(
                name="class_var_field",
                data_type=DataType(type="str"),
                extras={"x-is-classvar": True},
                required=True,
            ),
            DataModelField(
                name="plain_field",
                original_name=None,
                data_type=DataType(type="str"),
                required=True,
            ),
        ],
        reference=_reference(),
        extra_template_data=extra_template_data,
    )

    rendered = model.render()
    assert not any(import_.from_ == "pydantic" and import_.import_ == "Field" for import_ in model.imports)
    assert "class_var_field: ClassVar[str]" in rendered
    assert "plain_field: str" in rendered
    assert "Field(" not in rendered


@pytest.mark.allow_direct_assert
def test_alias_generator_helpers_handle_enum_and_unknown_values() -> None:
    """Alias generator helpers keep fallback behavior explicit."""
    field = DataModelField(name="field_name", data_type=DataType(type="str"), required=True)

    assert _alias_generator_name(AliasGenerator.ToCamel) == "to_camel"
    assert _alias_generator_name(object()) is None
    assert _generate_alias("custom_generator", "field_name") == "field_name"
    assert field._automatic_alias_disabled_for_alias_generator() is False


@pytest.mark.allow_direct_assert
def test_config_dict_items_accepts_supported_config_shapes() -> None:
    """Config rendering accepts model, dict, legacy dict method, and empty values."""

    class LegacyConfig:
        def dict(self, **_kwargs: Any) -> dict[str, Any]:
            return {"frozen": True}

    assert _config_dict_items(pydantic_v2.ConfigDict(extra="'allow'")) == [("extra", "'allow'")]
    assert _config_dict_items({"extra": "'forbid'"}) == [("extra", "'forbid'")]
    assert _config_dict_items(LegacyConfig()) == [("frozen", True)]
    assert _config_dict_items(None) == []
    assert _config_dict_items(object()) == []


@pytest.mark.allow_direct_assert
def test_dataclass_config_key_order_with_multiple_shared_parameters() -> None:
    """Dataclass config generation keeps deterministic key ordering."""
    model = DataClass(
        fields=[_field()],
        reference=_reference(),
        extra_template_data=_extra_template_data(),
    )

    config = model.extra_template_data["config"]
    assert isinstance(config, dict)
    assert list(config) == ["extra", "populate_by_name", "use_attribute_docstrings"]
    assert (
        "@dataclass(config=ConfigDict(extra='forbid', populate_by_name=True, use_attribute_docstrings=True))"
        in model.render()
    )


@pytest.mark.allow_direct_assert
def test_pattern_constraints_keep_leaf_specific_behavior() -> None:
    """Renamed pydantic constraints remain leaf-model specific."""
    assert list(PatternConstraints.model_fields)[-2:] == ["regex", "pattern"]
    assert list(PydanticV2Constraints.model_fields)[-2:] == ["regex", "pattern"]
    assert list(MsgspecConstraints.model_fields)[-2:] == ["regex", "pattern"]

    assert PydanticV2Constraints.model_validate({"minItems": 1}).model_dump(exclude_unset=True) == {"min_length": 1}
    assert MsgspecConstraints.model_validate({"minItems": 1}).model_dump(exclude_unset=True) == {"min_items": 1}
