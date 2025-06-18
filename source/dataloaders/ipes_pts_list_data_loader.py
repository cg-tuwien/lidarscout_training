import typing

from overrides import override

from source.dataloaders.ipes_data_loader import IpesDataModule, IpesDataset


class IpesPtsListDataModule(IpesDataModule):

    def __init__(self,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 seed, in_file, workers, use_ddp,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, debug: bool, batch_size: int):
        super(IpesPtsListDataModule, self).__init__(
            context_radius_factor=context_radius_factor, hm_interp_size=hm_interp_size, hm_size=hm_size,
            meters_per_pixel=meters_per_pixel, dataset_step=dataset_step,
            use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation, batch_size=batch_size, debug=debug, seed=seed)

        if batch_size != 1:
            # there will be different number of points for each shape and tile
            # collate will therefore fail unless we keep batch size at 1
            raise ValueError('batch_size must be 1 for IpesPtsListDataModule')

    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):

        if reconstruction:
            patches_per_shape = None
            do_data_augmentation = False
            load_gt = False
        else:
            patches_per_shape = self.patches_per_shape
            do_data_augmentation = do_data_augmentation
            load_gt = True

        dataset = IpesPtsListDataset(
            in_file=in_file,
            context_radius_factor=self.context_radius_factor,
            hm_interp_size=self.hm_interp_size,
            hm_size=self.hm_size,
            seed=self.seed,
            patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation,
            use_ddp=self.use_ddp,
            meters_per_pixel=self.meters_per_pixel,
            dataset_step=self.dataset_step,
            load_gt=load_gt,
            debug=self.debug,
        )
        return dataset


class IpesPtsListDataset(IpesDataset):

    def __init__(self,
                 context_radius_factor: float, hm_interp_size: int, hm_size: int, meters_per_pixel: float,
                 dataset_step: int,
                 in_file, seed, use_ddp, load_gt: bool,
                 patches_per_shape: typing.Optional[int], do_data_augmentation, debug):

        super(IpesPtsListDataset, self).__init__(
            context_radius_factor=context_radius_factor, hm_interp_size=hm_interp_size, hm_size=hm_size,
            meters_per_pixel=meters_per_pixel, dataset_step=dataset_step,
            in_file=in_file, seed=seed, use_ddp=use_ddp,
            patches_per_shape=patches_per_shape, do_data_augmentation=do_data_augmentation,
            load_gt=load_gt, debug=debug)

    @override
    def prepare_shape_data_for_cuda(self, shape_data: dict) -> dict:
        # override to keep points. there will be different number of points for each shape and tile
        # collate will therefore fail unless we keep batch size at 1

        # shape_data = super(IpesDataset, self).prepare_shape_data_for_cuda(shape_data)
        # shape_data.pop('pts_ms')  # raw point cloud is large
        # shape_data.pop('pts_rgb')  # raw point cloud is large
        # shape_data.pop('pts_local_ms')  # would be collate of variable length
        # shape_data.pop('pts_local_ps')  # would be collate of variable length
        # shape_data.pop('pts_local_rgb')  # would be collate of variable length
        return shape_data
