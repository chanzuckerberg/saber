from saber.entry_points.segment_methods import methods as segment
from saber.classifier.cli import classifier_routines as classifier
from saber.entry_points.run_low_pass_filter import cli as filter3d
from saber.analysis.analysis_cli import methods as analysis
from saber.entry_points.run_analysis import cli as save
from saber.gui.web.main import main as web
from saber import cli_context
import click
try:
    from saber.gui.base.zarr_gui import gui
    gui_available = True
except Exception as e:
    gui_available = False

@click.group(context_settings=cli_context)
def routines():
    """SABER ⚔️ -- Segment Anything Based Electron tomography Recognition."""
    pass

# Add subcommands to the group
routines.add_command(analysis)
routines.add_command(filter3d)
routines.add_command(classifier)
if gui_available: 
    routines.add_command(gui)
routines.add_command(segment)
routines.add_command(save)
routines.add_command(web)

## TODO: Add Routines for Slurm CLI. 
@click.group(context_settings=cli_context)
def slurm_routines():
    """Slurm CLI for SABER⚔️."""
    pass

if __name__ == "__main__":
    routines()
