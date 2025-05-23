from saber.entry_points.run_micrograph_segment import image, mrc_project, mrc_project_slurm
from saber.entry_points.run_tomogram_segment import slab, tomograms, tomograms_slurm
import click

@click.group(name="segment")
def methods():
    """Segment Tomograms and Micrographs with SABER."""
    pass

methods.add_command(image)
methods.add_command(slab)
methods.add_command(mrc_project)
methods.add_command(tomograms)
methods.add_command(tomograms_slurm)

@click.group(name="segment")
def cli_methods():
    """Segment Tomograms and Micrographs with SABER with SLURM Submissions."""
    pass

cli_methods.add_command(mrc_project_slurm)
cli_methods.add_command(tomograms_slurm)

if __name__ == "__main__":
    methods()


