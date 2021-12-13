"""
Sets up the analysis routines for processing GRILLIX cases

If running this on a server, you can forward the dask dashboard to your
local machine via SSH forwarding. i.e. use

```bash
ssh -L dask_port:localhost:8787
```

where dask_port is the port used by `dask`. Then open `localhost:8787` in a web-browser.
"""
import click
from pathlib import Path
from tcvx21.grillix_post.work_file_writer_m import WorkFileWriter
from tcvx21.grillix_post.validation_writer_m import (
    convert_work_file_to_validation_netcdf,
)

from dask.distributed import Client, LocalCluster

run_directory = Path("/ptmp/tbody/runs_for_TCV_paper").absolute()
work_directory = Path("work_files").absolute()
output_directory = Path(".").absolute()


def process_grillix_run(
    file_path: Path, output: str, data_length: int, toroidal_field_direction: str
):
    """
    Processes a GRILLIX simulation and writes a NetCDF

    Arguments
    file_path: File path to main run directory
    output: Output file path
    data_length: How many time-points to average over?
    """

    with LocalCluster(processes=True) as cluster, Client(cluster) as client:
        print(repr(client))

        data = WorkFileWriter(
            file_path=file_path,
            work_file=work_directory / output,
            toroidal_field_direction=toroidal_field_direction,
            data_length=data_length,
        )

        data.fill_work_file()

        client.close()

    print("Done")


@click.command()
@click.argument("filepath", type=str)
@click.option("--data_length", default=500, help="How many snaps to iterate over?")
def cli(filepath: str, data_length: int):
    """
    Process a single run
    """

    runs = {
        "R2_divjpar": "GRILLIX_favourable_2mm_standard_fuelling.nc",
        "R2_dj_S0": "GRILLIX_favourable_2mm_reduced_fuelling.nc",
        "R2_dj_S1": "GRILLIX_favourable_2mm_increased_fuelling.nc",
        "R1_divjpar": "GRILLIX_favourable_1mm_standard_fuelling.nc",
        "T2_divjpar": "GRILLIX_unfavourable_2mm_standard_fuelling.nc",
        "T2_dj_S0": "GRILLIX_unfavourable_2mm_reduced_fuelling.nc",
        "T2_dj_S1": "GRILLIX_unfavourable_2mm_increased_fuelling.nc",
        "T1_divjpar": "GRILLIX_unfavourable_1mm_standard_fuelling.nc",
        "R05_divjpar": "GRILLIX_favourable_0.5mm_standard_fuelling.nc",
    }

    process_grillix_run(
        file_path=run_directory / filepath,
        output=runs[filepath],
        data_length=data_length,
        toroidal_field_direction=("forward" if "R" in filepath else "reverse"),
    )


if __name__ == "__main__":
    cli()
