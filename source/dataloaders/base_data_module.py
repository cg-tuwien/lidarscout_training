import os
import typing
from abc import ABC, abstractmethod

from overrides import EnforceOverrides
import numpy as np
import torch.utils.data as torch_data
from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule, ABC):

    def __init__(self, use_ddp, workers, in_file, patches_per_shape: typing.Optional[int],
                 do_data_augmentation: bool, debug: bool, batch_size: int, seed: int):
        super(BaseDataModule, self).__init__()
        self.use_ddp = use_ddp
        self.workers = workers
        self.in_file = in_file
        self.train_set, self.val_set, self.test_set = get_set_files(in_file)
        self.patches_per_shape = patches_per_shape
        self.do_data_augmentation = do_data_augmentation
        self.debug = debug
        self.batch_size = batch_size
        self.seed = seed

    @abstractmethod
    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):
        pass

    def make_data_sampler(self, dataset, shuffle=False):
        from torch.cuda import device_count
        if bool(self.use_ddp) and device_count() > 1:
            from torch.utils.data.distributed import DistributedSampler
            data_sampler = DistributedSampler(
                dataset, num_replicas=None, rank=None,
                shuffle=shuffle, seed=0, drop_last=False)
        else:
            data_sampler = None
        return data_sampler

    def make_dataloader(self, dataset, data_sampler, batch_size: int, shuffle: bool = False):
        dataloader = torch_data.DataLoader(
            dataset,
            sampler=data_sampler,
            batch_size=batch_size,
            num_workers=int(self.workers),
            persistent_workers=True if int(self.workers) > 0 else False,
            pin_memory=True,
            worker_init_fn=seed_train_worker,
            shuffle=shuffle,
            collate_fn=collate_fn_custom)
        return dataloader

    def train_dataloader(self):
        dataset = self.make_dataset(in_file=self.train_set, reconstruction=False,
                                    patches_per_shape=self.patches_per_shape,
                                    do_data_augmentation=self.do_data_augmentation)
        data_sampler = self.make_data_sampler(dataset=dataset, shuffle=True)
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=self.batch_size, shuffle=data_sampler is None)
        return dataloader

    def val_dataloader(self):
        dataset = self.make_dataset(in_file=self.val_set, reconstruction=False,
                                    patches_per_shape=self.patches_per_shape,
                                    do_data_augmentation=False)
        data_sampler = self.make_data_sampler(dataset=dataset, shuffle=False)
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=self.batch_size)
        return dataloader

    def test_dataloader(self):
        batch_size = 1
        dataset = self.make_dataset(in_file=self.test_set, reconstruction=False,
                                    patches_per_shape=None, do_data_augmentation=False)
        data_sampler = None
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=batch_size)
        return dataloader

    def predict_dataloader(self):
        batch_size = 1
        dataset = self.make_dataset(in_file=self.test_set, reconstruction=True,
                                    patches_per_shape=None, do_data_augmentation=False)
        data_sampler = None
        dataloader = self.make_dataloader(dataset=dataset, data_sampler=data_sampler,
                                          batch_size=batch_size)
        return dataloader


class BaseDataset(torch_data.Dataset, EnforceOverrides):

    def __init__(self, in_file: str, seed, use_ddp: bool, load_gt: bool,
                 patches_per_shape: typing.Optional[int], do_data_augmentation, debug):

        super(BaseDataset, self).__init__()

        self.in_file = in_file
        self.seed = seed
        self.patches_per_shape = patches_per_shape
        self.do_data_augmentation = do_data_augmentation
        self.load_gt = load_gt
        self.use_ddp = use_ddp
        self.debug = debug
        self.shape_names = []

        # initialize rng for picking points in the local subsample of a patch
        if self.seed is None:
            self.seed = np.random.random_integers(-(2 ** 31 - 1), 2 ** 31, 1)[0]

        from torch.cuda import device_count
        if bool(self.use_ddp) and device_count() > 1:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
            self.seed += rank
        self.rng = np.random.RandomState(self.seed)
        self.get_shape_names(in_file=in_file)

    def get_shape_names(self, in_file: str):

        # get all shape names in the dataset
        if isinstance(self.in_file, str):
            # assume .txt files contain a list of shapes
            if os.path.splitext(self.in_file)[1].lower() == '.txt':
                self.shape_names = []
                with open(os.path.join(in_file)) as f:
                    self.shape_names = f.readlines()
                self.shape_names = [x.strip() for x in self.shape_names]
                self.shape_names = list(filter(None, self.shape_names))
            else:  # all other single files are just one shape to be reconstructed
                self.shape_names = [self.in_file]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.shape_names)

    def add_shape_data(self, shape_id, shape_data: dict) -> dict:
        return shape_data

    def visualize_shape_data(self, shape_id, shape_data: dict):
        return

    def add_gt_data(self, shape_id, shape_data: dict) -> dict:
        return shape_data

    def augment_data(self, shape_data: dict) -> dict:
        return shape_data

    def prepare_shape_data_for_cuda(self, shape_data: dict) -> dict:
        # remove data items that should not be moved to GPU
        # clean up data that is not needed anymore, e.g. non-augmented data

        # assuming augmented data has a '_aug' suffix, we move it to the non-augmented key
        for key in list(shape_data.keys()):
            if key.endswith('_aug'):
                key_no_aug = key[:-4]
                shape_data[key_no_aug] = shape_data[key]
                shape_data.pop(key)

        # remove keys with None values
        keys_to_remove = [key for key, val in shape_data.items() if val is None]
        for key in keys_to_remove:
            shape_data.pop(key)

        return shape_data

    @staticmethod
    def _make_shape_data_read_only(shape_data: dict) -> dict:
        for key, val in shape_data.items():
            if isinstance(val, np.ndarray):
                shape_data[key].flags.writeable = False
        return shape_data

    def __getitem__(self, shape_id):
        from source.base.container import dict_np_to_torch, dict_np_double_to_float

        shape_data = dict()
        shape_data = self.add_shape_data(shape_id, shape_data)

        shape_data = self.add_gt_data(shape_id, shape_data)

        if self.do_data_augmentation:
            shape_data = self._make_shape_data_read_only(shape_data)  # augmentation should copy and replace
            shape_data = self.augment_data(shape_data)

        shape_data = self._make_shape_data_read_only(shape_data)  # no modification after augmentation

        self.visualize_shape_data(shape_id, shape_data)

        shape_data = self.prepare_shape_data_for_cuda(shape_data)

        shape_data = dict_np_double_to_float(shape_data)  # must be before poco part
        shape_data = dict_np_to_torch(shape_data)

        return shape_data


def seed_train_worker(worker_id):
    import random
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def in_file_is_dataset(in_file: str):
    return os.path.splitext(in_file)[1].lower() == '.txt'


def get_dataset_dir(in_file: str):
    dataset_dir = os.path.dirname(in_file)
    return dataset_dir


def get_dataset_name(in_file: str):
    dataset_dir = get_dataset_dir(in_file)
    dataset_name = os.path.basename(dataset_dir)
    return dataset_name


def get_set_files(in_file: str):
    if in_file_is_dataset(in_file):
        # expect structure like [stage][_optional_name].txt
        in_file_name = os.path.splitext(os.path.basename(in_file))[0]
        name_parts = in_file_name.split('_')
        stage = name_parts[0] if len(name_parts) > 1 else in_file_name
        name = '_' + '_'.join(name_parts[1:]) if len(name_parts) > 1 else ''
        train_set = os.path.join(os.path.dirname(in_file), 'train{}.txt'.format(name)) if stage != 'train' else in_file
        val_set = os.path.join(os.path.dirname(in_file), 'val{}.txt'.format(name)) if stage != 'val' else in_file
        test_set = os.path.join(os.path.dirname(in_file), 'test{}.txt'.format(name)) if stage != 'test' else in_file

    else:
        train_set = in_file
        val_set = in_file
        test_set = in_file
    return train_set, val_set, test_set


def get_results_dir(out_dir: str, name: str, in_file: str):
    dataset_name = get_dataset_name(in_file)
    model_results_rec_dir = os.path.join(out_dir, name, dataset_name)
    return model_results_rec_dir


def read_shape_list(shape_list_file: str):
    with open(shape_list_file) as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))
    return shape_names


def load_pts(pts_file: str):
    # Supported file formats are:
    # - PLY, STL, OBJ and other mesh files loaded by [trimesh](https://github.com/mikedh/trimesh).
    # - XYZ as whitespace-separated text file, read by [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html).
    # Load first 3 columns as XYZ coordinates. All other columns will be ignored.
    # - NPY and NPZ, read by [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
    # NPZ assumes default key='arr_0'. All columns after the first 3 columns will be ignored.
    # - LAS and LAZ (version 1.0-1.4), COPC and CRS loaded by [Laspy](https://github.com/laspy/laspy).
    # You may want to sub-sample large point clouds to ~250k points to avoid speed and memory issues.
    # For detailed reconstruction, you'll need to extract parts of large point clouds.

    import os

    file_name, file_ext = os.path.splitext(pts_file)
    file_ext = file_ext.lower()
    if file_ext == '.npy':
        pts = np.load(pts_file)
    elif file_ext == '.npy':
        arrs = np.load(pts_file)
        pts = arrs['arr_0']
    elif file_ext == '.xyz':
        from source.base.point_cloud import load_xyz
        pts = load_xyz(pts_file)
    elif file_ext in ['.stl', '.ply', '.obj', 'gltf', '.glb', '.dae', '.off', '.ctm', '.3dxml']:
        import trimesh
        trimesh_obj: typing.Union[trimesh.Scene, trimesh.Trimesh] = trimesh.load_mesh(file_obj=pts_file)
        if isinstance(trimesh_obj, trimesh.Scene):
            mesh: trimesh.Trimesh = trimesh_obj.geometry.items()[0]
        elif isinstance(trimesh_obj, trimesh.Trimesh):
            mesh: trimesh.Trimesh = trimesh_obj
        elif isinstance(trimesh_obj, trimesh.PointCloud):
            mesh: trimesh.Trimesh = trimesh_obj
        else:
            raise ValueError('Unknown trimesh object type: {}'.format(type(trimesh_obj)))
        pts = np.array(mesh.vertices)
    elif file_ext in ['.las', '.laz', '.copc', '.crs']:
        import laspy
        las = laspy.read(pts_file)
        pts = las.xyz
    else:
        raise ValueError('Unknown point cloud type: {}'.format(pts_file))
    return pts


def collate_fn_custom(batch):
    import torch

    # from torch.utils.data import default_collate
    # batch_default = default_collate(batch)

    # TODO: add patch points again, add exception here and keep them in lists
    # concatenate tensors to merge the per-sample batch dim
    keys = batch[0].keys()
    batch_dict = {key: [] for key in keys}
    for key in keys:
        for item in batch:
            batch_dict[key].append(item[key])
        if isinstance(batch_dict[key][0], torch.Tensor):
            if batch_dict[key][0].dim() == 0:  # can't cat scalars
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)
            else:
                batch_dict[key] = torch.cat(batch_dict[key], dim=0)
    return batch_dict
