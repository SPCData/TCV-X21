# Testing for the TCV-X21 Python analysis routines

Tests are a vital part of quality assurance for any code, including post-processing scripts!

To test the python library, you can run
```bash
pytest
```
from the `tcvx21` repository.

## Notes on testing

The notebooks in the tcvx21 validation repository are tested by converting the notebooks to a `.py` file and then executing the file.
This makes it easier to track changes in the notebooks, and the `.py` files can be tested easily.

The other tests here can be broadly defined as either *coverage* or *functional* tests.

Coverage tests aim to hit as many lines of code as possible with typical inputs, to make sure that the code *runs*.
These are quite easy to write. A slight improvement might be to pick a result of the run and to ensure that this
doesn't change.

Functional tests are more difficult to write. They essentially check that the computed result is equal to
an expected result. This usually requires an analytical result for a simple test. These have been written for
complex functionality like the vector analysis.

For plotting routines, the tests write sample outputs to ensure that the style is what you want. These can be found in `3.results/test_fig` (they will be updated each time that you run pytest).
