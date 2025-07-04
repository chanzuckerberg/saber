from saber.entry_points.run_membrane_refinement import refine_membranes
from saber.entry_points.run_analysis import statistics
import click

@click.group(name='analysis')
@click.pass_context
def methods(ctx):
    """Post-processing analysis after segmentation."""
    pass

methods.add_command(refine_membranes)
methods.add_command(statistics)

@click.group()
@click.pass_context
def cli(ctx):
    pass


if __name__ == '__main__':
    methods()   