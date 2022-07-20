from pathlib import Path

from torch.utils.data import DataLoader

from .imagename_dataset import ImagenameDataset

# def get_dataset_self(
#     root_dir: str = "dataset/scood_benchmark/",
#     benchmark: str = "cifar10",
#     num_classes: int = 10,
#     name: str = "cifar10",
#     stage: str = "test",
# ):
#     """get benchmark data set for testing.

#     Args:
#         root_dir (str, optional): Defaults to "dataset/scood_benchmark/".
#         benchmark (str, optional): benchmark base name. Defaults to "cifar10".
#         num_classes (int, optional): Defaults to 10.
#         name (str, optional): in-d/out-d name to test [cifar10, cifar100, lsun,
#         places365, svhn, texture, tin]. Defaults to "cifar10".
#         stage (str, optional): train or test. Defaults to "test".

#     Returns:
#         _type_: _description_
#     """
#     root_dir = Path(root_dir)
#     data_dir = root_dir / "images"
#     imglist_dir = root_dir / "imglist" / f"benchmark_{benchmark}"

#     return ImagenameDataset(
#         name=name,
#         stage=stage,
#         interpolation=None,
#         imglist=imglist_dir / f"{stage}_{name}.txt",
#         root=data_dir,
#         num_classes=num_classes,
#     )

def get_dataloader_self(
    root_dir: str = "../dataset/scood_benchmark/",
    benchmark: str = "cifar10",
    num_classes: int = 10,
    name: str = "cifar10",
    stage: str = "test",
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
):
    dataset = get_dataset(root_dir, benchmark, num_classes, name, stage)
    print(dataset.transform_image)
    print(dataset.transform_aux_image)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dataset(
    root_dir: str = "data",
    benchmark: str = "cifar10",
    num_classes: int = 10,
    name: str = "cifar10",
    stage: str = "train",
    interpolation: str = "bilinear",
):
    root_dir = Path(root_dir)
    data_dir = root_dir / "images"
    imglist_dir = root_dir / "imglist" / f"benchmark_{benchmark}"

    return ImagenameDataset(
        name=name,
        stage=stage,
        interpolation=interpolation,
        imglist=imglist_dir / f"{stage}_{name}.txt",
        root=data_dir,
        num_classes=num_classes,
    )


def get_dataloader(
    root_dir: str = "data",
    benchmark: str = "cifar10",
    num_classes: int = 10,
    name: str = "cifar10",
    stage: str = "train",
    interpolation: str = "bilinear",
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
):
    dataset = get_dataset(root_dir, benchmark, num_classes, name, stage, interpolation)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
