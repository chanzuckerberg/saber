from saber.classifier.preprocess.slurm import prepare_tomogram_training_slurm, prepare_micrograph_training_slurm
from saber.classifier.preprocess.micrograph_training_prep import prepare_micrograph_training
from saber.classifier.preprocess.tomogram_training_prep import prepare_tomogram_training
from saber.classifier.preprocess.split_merge_data import split_data, merge_data
from saber.classifier.preprocess.training_data_info import class_info
from saber.classifier.inference import predict, predict_slurm
from saber.classifier.train import train, train_slurm
from saber.classifier.evaluator import evaluate
import click

@click.group(name="classifier")
def classifier_routines():
    """Routines for training and evaluating classifiers."""
    pass

# Add subcommands to the group
classifier_routines.add_command(split_data)
classifier_routines.add_command(merge_data)
classifier_routines.add_command(train)
classifier_routines.add_command(predict)
classifier_routines.add_command(prepare_tomogram_training)
classifier_routines.add_command(prepare_micrograph_training)
classifier_routines.add_command(evaluate)
classifier_routines.add_command(class_info)

@click.group(name="classifier")
def slurm_classifier_routines():
    """Routines for training and evaluating classifiers."""
    pass

# Add subcommands to the group
slurm_classifier_routines.add_command(train_slurm)
slurm_classifier_routines.add_command(predict_slurm)
slurm_classifier_routines.add_command(prepare_tomogram_training_slurm)
slurm_classifier_routines.add_command(prepare_micrograph_training_slurm)

if __name__ == "__main__":
    classifier_routines()