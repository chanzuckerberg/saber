from saber.entry_points.segment_methods import methods as segment
from saber.classifier.cli import classifier_routines as classifier
from saber.analysis.analysis_cli import methods as analysis
from saber.entry_points.run_analysis import cli as save
from saber.gui.run import web
from saber import cli_context
import rich_click as click
try:
    from saber.gui.run import gui
    gui_avail = True
except Exception as e:
    gui_avail = False

@click.group(context_settings=cli_context)
def routines():
    """SABER ⚔️ -- Segment Anything Based Expert Recognition."""
    pass

# Add subcommands to the group
routines.add_command(analysis)
routines.add_command(classifier)
if gui_avail: 
    routines.add_command(gui)
routines.add_command(segment)
routines.add_command(save)
routines.add_command(web)

## TODO: Add Routines for Slurm CLI. 
@click.group(context_settings=cli_context)
def slurm_routines():
    """Slurm CLI for SABER⚔️ -- Not Implemented Yet... Coming Soon!"""
    pass

if __name__ == "__main__":
    routines()
