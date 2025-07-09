from saber.entry_points.segment_methods import methods as segment
from saber.classifier.cli import classifier_routines as classifier
from saber.entry_points.run_low_pass_filter import cli as filter3d
from saber.analysis.analysis_cli import methods as analysis
from saber.entry_points.run_analysis import cli as save
from saber.pretrained_weights import cli as download
from saber.utils.importers import cli as importers
import click
try:
    from saber.gui.zarr_gui import gui
    gui_available = True
except:
    gui_available = False

@click.group()
def routines():
    """SABER ⚔️ -- Segment Anything Based Electron tomography Recognition."""
    pass

# Add subcommands to the group
routines.add_command(analysis)
routines.add_command(download)
routines.add_command(filter3d)
routines.add_command(classifier)
if gui_available: routines.add_command(gui)
routines.add_command(segment)
routines.add_command(save)

## TODO: Add Routines for Slurm CLI. 
@click.group()
def slurm_routines():
    """Slurm CLI for SABER⚔️."""
    pass

if __name__ == "__main__":
    routines()
