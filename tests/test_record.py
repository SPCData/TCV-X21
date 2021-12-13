"""
Tests the data record object
"""
import pytest
from pathlib import Path
import numpy as np
import tcvx21
from tcvx21.file_io import read_from_json, read_struct_from_file
from tcvx21.record_c import (
    Record,
    observables_template,
    write_template_files,
    template_file,
    RecordWriter,
)
from tcvx21.dict_methods_m import recursive_assert_dictionaries_equal


def test_template_io(tmpdir):
    """Test whether a file can be written and read"""

    write_template_files(tmpdir)

    test_dict = read_from_json(tmpdir / "observables.json")
    ref_dict = observables_template()

    recursive_assert_dictionaries_equal(test_dict, ref_dict)


def test_template_json():
    """Tests whether the written JSON template file exactly matches the computed template"""

    test_dict = read_from_json(template_file)
    ref_dict = observables_template()

    recursive_assert_dictionaries_equal(test_dict, ref_dict)


def test_template_mat():
    """
    Tests whether the written .mat file matches the JSON template
    The exact keys won't match -- we've replaced '-' with '_' to make legal data entries
    """
    struct = read_struct_from_file(tcvx21.library_dir / "record_c/observables.mat")
    with pytest.raises(AssertionError):
        recursive_assert_dictionaries_equal(observables_template(), struct)

    for key in ["LFS-LP", "LFS-IR", "HFS-LP"]:
        struct[key] = struct.pop(key.replace("-", "_"))

    recursive_assert_dictionaries_equal(observables_template(), struct)


def test_record_construction(tcv_forward_data):
    """Tests whether a record can be made from an existing NetCDF file"""

    record = Record(tcv_forward_data)
    record = Record(tcv_forward_data, label="DEG", color="green", linestyle="--")
    m = record.get_observable("RDPA", "density")
    # Make sure that the attributes get copied in for fluctuations
    assert m.color == "green"
    assert m.linestyle == "--"
    assert m.label == "DEG"


def test_fluctuation_calculation(tcv_forward_data):
    """Tests whether the fluctuations can be computed"""
    record = Record(tcv_forward_data, label="ABC", color="purple", linestyle="..")
    m = record.get_observable("FHRP", "jsat_fluct")
    # Make sure that the attributes get copied in for fluctuations
    assert m.color == "purple"
    assert m.linestyle == ".."
    assert m.label == "ABC"


def test_valid_observable_check(tcv_forward_data):
    """Tests whether the Record will prevent you from asking for invalid data"""
    record = Record(tcv_forward_data, label="ABC", color="purple", linestyle="..")
    with pytest.raises(AssertionError):
        m = record.get_observable("FHRP", "I dont exist")


def test_writer(tmpdir):
    """Test if you can write a blank dictionary to file"""

    blank_dict = observables_template()

    additional_attributes = {"pizza_flavour": "Margherita"}

    output = Path(tmpdir) / "output.nc"
    writer = RecordWriter(
        output, descriptor="ABC", description="A long description of the data"
    )

    writer.write_data_dict(blank_dict, additional_attributes=additional_attributes)

    # Make sure that you can read the written file
    record = Record(output)

    with pytest.raises(FileExistsError):
        writer = RecordWriter(output, "ABC", "123", allow_overwrite=False)
    # Test that you can overwrite
    writer = RecordWriter(output, "ABC", "123", allow_overwrite=True)


def test_writer_w_data(tmpdir, tcv_forward_data):
    """Use the TCV data"""

    blank_dict = observables_template()

    record = Record(tcv_forward_data)

    for diagnostic, observable in record.keys():

        obs = blank_dict[diagnostic]["observables"][observable]
        m = record.get_observable(diagnostic, observable)

        if m.is_empty:
            continue

        obs["values"] = m._values
        obs["errors"] = m._errors

        obs["R"] = m._positions_rsep.magnitude * 0.0
        obs["Z"] = m._positions_rsep.magnitude * 1.0
        obs["R_units"] = "m"
        obs["Z_units"] = "m"

        if obs["dimensionality"] == 1:
            obs["Ru"] = m._positions_rsep.magnitude

        elif obs["dimensionality"] == 2:
            obs["Ru"] = m._positions_rsep.magnitude
            obs["Zx"] = m._positions_zx.magnitude

        obs["simulation_hierarchy"] = 10

    additional_attributes = {"pizza_flavour": "Margherita"}

    output = Path(tmpdir) / "output.nc"
    writer = RecordWriter(
        output, descriptor="ABC", description="A long description of the data"
    )

    writer.write_data_dict(blank_dict, additional_attributes=additional_attributes)

    # Make sure that you can read the written file
    record = Record(output)


def test_observables_arithmetic(tcv_forward_data):

    record = Record(tcv_forward_data)

    assert np.allclose(
        record.get_observable("FHRP", "potential").positions,
        record.get_observable("FHRP", "electron_temp").positions,
    )

    test = record.get_observable("FHRP", "potential") - record.get_observable(
        "FHRP", "electron_temp"
    ) * tcvx21.Quantity(3.0, "1/e")
    ref = record.get_observable("FHRP", "vfloat")

    mask = test.points_overlap(ref.positions)
    mapped_test = test.interpolate_onto_positions(ref.positions[mask])
    assert np.isclose((mapped_test / ref.trim_to_mask(mask)).values.mean(), 1.00133)


def test_zero_error(tcv_forward_data):

    record = Record(tcv_forward_data)

    record.set_error_to_zero()

    m = record.get_observable("FHRP", "density")

    assert np.allclose(m.errors, 0.0)
