"""
Interface to a standard NetCDF file
"""
import netCDF4
from pathlib import Path
import tcvx21
from tcvx21 import Quantity
from tcvx21.observable_c.observable_m import MissingDataError
from tcvx21.observable_c.observable_empty_m import EmptyObservable
from tcvx21.observable_c.observable_1d_m import Observable1D
from tcvx21.observable_c.observable_2d_m import Observable2D


class Record:
    def __init__(
        self,
        file_path: Path,
        label: str = None,
        color: str = "C0",
        linestyle: str = "-",
    ):
        """
        Adds a link to a standard NetCDF file

        label should be a short label for use in a legend

        color should be a colour code that will be used to represent observables from this dataset

        linestyle should correspond to a linestyle in code (see link below). Helpful for colour-blindness or
        black & white printing
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        """

        file_path = Path(file_path)
        assert file_path.exists() and file_path.suffix == ".nc"

        self.file_path = file_path
        self.dataset = netCDF4.Dataset(file_path)

        if label is None:
            self.label = self.dataset.descriptor
        else:
            self.label = label

        self.color = color
        self.linestyle = linestyle

        self.data_template = tcvx21.read_from_json(tcvx21.template_file)

        for diagnostic in list(self.data_template.keys()):
            for observable in list(
                self.data_template[diagnostic]["observables"].keys()
            ):
                self.make_observable(diagnostic, observable)

    @property
    def is_empty(self):
        return False

    def set_error_to_zero(self):
        """For all observables, overwrite the error with zeros"""
        for diagnostic, observable in self.keys():
            try:
                m = self.get_observable(diagnostic, observable)
                if not m.is_empty:
                    m._errors = 0.0 * m._errors
            except AssertionError:
                # Ignore non-standard data (i.e. the X-point lineout)
                pass

    def make_observable(self, diagnostic: str, observable: str):
        """
        Makes a 'observable' object to represent the data stored in the NetCDF
        """
        data = self.dataset[diagnostic]["observables"][observable]
        data_template = self.data_template[diagnostic]["observables"][observable]

        try:
            dimensionality = data.dimensionality
        except AttributeError:
            dimensionality = data_template["dimensionality"]

        try:
            # if dimensionality == 0:
            #     observable_ = Observable0D(data, diagnostic, observable,
            #                                  label=self.label, color=self.color, linestyle=self.linestyle)

            if dimensionality == 1:
                observable_ = Observable1D(
                    data,
                    diagnostic,
                    observable,
                    label=self.label,
                    color=self.color,
                    linestyle=self.linestyle,
                )

            elif dimensionality == 2:
                observable_ = Observable2D(
                    data,
                    diagnostic,
                    observable,
                    label=self.label,
                    color=self.color,
                    linestyle=self.linestyle,
                )

            else:
                raise NotImplementedError(
                    f"No implementation for dimensionality {dimensionality}"
                )

        except MissingDataError:
            observable_ = EmptyObservable()

        # Overwrite the entry in the observables dictionary with the observable object
        self.data_template[diagnostic]["observables"][observable] = observable_

    def keys(self):
        """
        Makes a (diagnostic, observable) iterator for iterating over all of the observables
        """
        for diagnostic in self.dataset.groups:
            for observable in self.dataset[diagnostic]["observables"].groups:
                yield diagnostic, observable

    def __getitem__(self, item) -> Quantity:
        """Index on the underlying dataset"""
        return self.dataset[item]

    def _check_legal_access(self, diagnostic: str = None, observable: str = None):
        """
        Makes sure that the requested data is a valid entry, and if not provide a helpful error
        """
        if diagnostic is None:
            return
        allowed_diagnostics = list(self.data_template.keys())
        assert (
            diagnostic in allowed_diagnostics
        ), f"diagnostic should be in {allowed_diagnostics} but was {diagnostic}"

        if observable is None:
            return
        allowed_observables = list(self.data_template[diagnostic]["observables"].keys())
        assert (
            observable in allowed_observables
        ), f"{diagnostic}:observable should be in {allowed_observables} but was {observable}"

    def get_observable(
        self, diagnostic: str, observable: str, with_check: bool = False
    ):
        """
        Safe access to observables

        Makes sure that you are accessing a valid observable, and then converts the observable into Quantity

        Note that you can also get attributes from the NetCDF via square-bracket indexing on the named keys
        """
        if observable.endswith("_fluct"):
            base_observable = observable.rstrip("fluct")[:-1]
            std = self.get_observable(diagnostic, f"{base_observable}_std", with_check)
            mean = self.get_observable(diagnostic, base_observable, with_check)
            if not (std.is_empty or mean.is_empty):
                fluct = std / mean
                fluct.name = f"Fluctuation of {mean.name}"
                if fluct.dimensionality == 1:
                    fluct.set_plot_limits(ymin=0.0, ymax=1.0)
                return fluct
            else:
                return EmptyObservable()

        self._check_legal_access(diagnostic, observable)

        observable_ = self.data_template[diagnostic]["observables"][observable]

        if with_check:
            assert isinstance(
                observable_, observable
            ), f"Requested observable but returned type {type(observable_)}"

        return observable_

    def get_nc_group(self, diagnostic: str = None, observable: str = None):
        """
        Returns the raw NetCDF group (useful when there is missing data)
        """
        self._check_legal_access(diagnostic, observable)

        if diagnostic is None:
            return self.dataset
        elif observable is None:
            return self.dataset[diagnostic]
        else:
            return self.dataset[diagnostic]["observables"][observable]


class EmptyRecord(Record):
    def __init__(self):
        pass

    def get_observable(self, *args, **kwargs):
        return EmptyObservable()

    @property
    def is_empty(self):
        return True
