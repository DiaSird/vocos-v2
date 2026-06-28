import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger

from src.dataloaders.datamodule import AudioDataModule
from src.pretraining import AudioPLModule


# -----------------------------
# Logger factory
# -----------------------------
def create_logger(name: str):
    """Return Lightning logger."""
    if name == "mlflow":
        return MLFlowLogger(
            experiment_name="vocos",
            tracking_uri="file:./mlruns",
        )
    return True  # default Lightning logger


# -----------------------------
# CLI
# -----------------------------
def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logger",
        type=str,
        default="default",
        choices=["default", "mlflow"],
        help="Logger backend",
    )

    args, _ = parser.parse_known_args()
    logger = create_logger(args.logger)

    LightningCLI(
        AudioPLModule,
        AudioDataModule,
        subclass_mode_model=True,
        trainer_defaults={"logger": logger},
    )


if __name__ == "__main__":
    cli_main()
