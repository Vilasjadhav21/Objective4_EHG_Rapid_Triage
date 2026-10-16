"""
Configuration settings for Objective 4: Rapid Triage (1-Minute Algorithm).

Centralized configuration for dataset paths, PhysioNet TPEHGDB v1.0.1 facts,
rapid triage windowing, EMD decomposition targets, and model definitions.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Project Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
IMFS_DIR = OUTPUT_DIR / "imfs"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"


@dataclass
class DatasetConfig:
    """PhysioNet TPEHGDB v1.0.1 Dataset Specifications."""
    dataset_name: str = "PhysioNet TPEHGDB v1.0.1"
    num_records: int = 300
    num_pregnancies: int = 300
    sampling_frequency: float = 20.0  # Hz
    record_duration_minutes: float = 30.0  # minutes per record
    record_duration_seconds: float = 1800.0  # seconds per record
    num_ehg_channels: int = 3  # S1, S2, S3
    
    # 12 signal channels per record
    signal_channels: List[str] = field(default_factory=lambda: [
        "S1 raw", "S1 0.08-4 Hz", "S1 0.3-3 Hz", "S1 0.3-4 Hz",
        "S2 raw", "S2 0.08-4 Hz", "S2 0.3-3 Hz", "S2 0.3-4 Hz",
        "S3 raw", "S3 0.08-4 Hz", "S3 0.3-3 Hz", "S3 0.3-4 Hz"
    ])
    
    # Filtered signals boundary trimming rule
    trim_boundary_seconds: float = 180.0  # Ignore first and last 180 seconds for filtered signals
    
    # Metadata file
    metadata_filename: str = "tpehgdb.smr"


@dataclass
class TriagePipelineConfig:
    """Target Pipeline Parameters for Rapid Triage (1-Minute Algorithm)."""
    triage_window_seconds: float = 60.0  # 1-minute EHG recording
    target_imf: int = 1                  # IMF-1 component
    num_top_features: int = 5            # Top 5 EHG features
    classifier_model: str = "Fine Gaussian Kernel SVM"


@dataclass
class EMDConfig:
    """EMD Decomposition Parameters."""
    sampling_frequency: float = 20.0
    max_imfs: int = 10


@dataclass
class ProjectConfig:
    """Master configuration bundling dataset and pipeline settings."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    triage: TriagePipelineConfig = field(default_factory=TriagePipelineConfig)
    emd: EMDConfig = field(default_factory=EMDConfig)
    
    data_dir: Path = DATA_DIR
    plots_dir: Path = PLOTS_DIR
    imfs_dir: Path = IMFS_DIR
    features_dir: Path = FEATURES_DIR
    models_dir: Path = MODELS_DIR


# Legacy & pipeline compatibility aliases
PipelineConfig = ProjectConfig
SignalConfig = DatasetConfig
FeatureConfig = TriagePipelineConfig
ModelConfig = TriagePipelineConfig


def ensure_directories():
    """Create all required research project directories if they do not exist."""
    directories = [
        DATA_DIR,
        OUTPUT_DIR,
        PLOTS_DIR,
        IMFS_DIR,
        FEATURES_DIR,
        MODELS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    config = ProjectConfig()
    print(f"Loaded config for {config.dataset.dataset_name}")
    print(f"Sampling Frequency: {config.dataset.sampling_frequency} Hz")
    print(f"Signal Channels ({len(config.dataset.signal_channels)}): {config.dataset.signal_channels}")
    print(f"Trim Rule: Ignore first and last {config.dataset.trim_boundary_seconds}s for filtered signals")
    print(f"Target Classifier: {config.triage.classifier_model}")
