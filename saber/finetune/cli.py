from saber.finetune.train import finetune
from saber.finetune.prep import main as prep
import click

@click.group(name="finetune")
def finetune_routines():
    """Routines for finetuning SAM2 on New Modalities."""
    pass

# Add subcommands to the group
finetune_routines.add_command(finetune)
finetune_routines.add_command(prep)
