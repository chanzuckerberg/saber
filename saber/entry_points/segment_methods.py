from saber.entry_points.slurm import tomograms_slurm, micrographs_slurm
from saber.entry_points.run_tomogram_segment import slab, tomograms
from saber.entry_points.run_micrograph_segment import micrographs
import click

@click.group(name="segment")
def methods():
    """Segment Tomograms and Micrographs with SABER."""
    pass

methods.add_command(slab)
methods.add_command(micrographs)
methods.add_command(tomograms)

@click.group(name="segment")
def cli_methods():
    """Segment Tomograms and Micrographs with SABER with SLURM Submissions."""
    pass

cli_methods.add_command(micrographs_slurm)
cli_methods.add_command(tomograms_slurm)

if __name__ == "__main__":
    methods()


