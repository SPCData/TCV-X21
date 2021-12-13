from pathlib import Path
from .record_m import Record, EmptyRecord
from .template_writer_m import observables_template, write_template_files
from .record_writer_m import RecordWriter

template_file = (Path(__file__).parent / "observables.json").absolute()
