from typing import Literal

from torch.utils.data import Dataset

from .dataset import Dataset
from .dataset_precomputed_dino_features import DatasetDinoFeatures


def assert_kwargs_present(kwargs: dict, required_keys: list[str], dataset_type: str):
    for key in required_keys:
        assert key in kwargs, f"{key} must be provided for {dataset_type} dataset"


def get_datasets(
    dataset_type: Literal["Normal", "DinoFeatures", "WithDeformations"],
    splits,
    **kwargs,
) -> list[Dataset]:

    datasets = []

    assert_kwargs_present(
        kwargs,
        [
            "dataset_name",
            "paths_preprocessed",
            "paths_original",
            "split",
            "resolution_proc",
            "dim_proc",
            "n_classes",
        ],
        dataset_type=dataset_type,
    )

    if dataset_type == "DinoFeatures":
        assert_kwargs_present(
            kwargs,
            ["paths_preprocessed_dino", "hierarchy_level"],
            dataset_type=dataset_type,
        )

    if dataset_type == "WithDeformations":
        assert_kwargs_present(kwargs, ["deformation_params"], dataset_type=dataset_type)

    for split in splits:
        datasets.append(Dataset(split=split, **kwargs))

    return datasets
