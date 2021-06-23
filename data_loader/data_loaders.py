from torch.utils.data import DataLoader
# local modules
# .xxx 是“相对导入”，避免Python导入默认的path和package中名称冲突
from .dataset import DynamicH5Dataset, MemMapDataset, SequenceDataset
from utils.data import concatenate_subfolders, concatenate_datasets

class InferenceDataLoader(DataLoader):

    def __init__(self, data_path, num_workers=1, pin_memory=True, dataset_kwargs=None, ltype="H5"):
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if ltype == "H5":
            dataset = DynamicH5Dataset(data_path, **dataset_kwargs)
        elif ltype == "MMP":
            dataset = MemMapDataset(data_path, **dataset_kwargs)
        else:
            raise Exception("Unknown loader type {}".format(ltype))
        super().__init__(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


class HDF5DataLoader(DataLoader):
    """
    HDF5DataLoader继承了pytorch自带的Dataloader，使用了默认的iter方法(__next__)
    首先根据文件路径整理了数据，形成了dataset，再初始化了DataLoader，
    之后每次enumerate(HDF5DataLoader)时会调用DataLoader的__next__方法，产生Dataset数据

    而这个代码重写了Dataset类，调用了SequenceDataset()的__getitem__方法
    （作者的SequenceDataset()中，getitem时调用了self.dataset.__getitiem__方法，具体的注释跳转过去。
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        # print('====> in HDF5DataLoader __init__')
        # print('batch_size: ', batch_size, 'shuffle: ', shuffle, 'num_workers: ', num_workers, 'pin_memory: ', pin_memory)
        # print('data file: ', data_file, 'sequence_kwargs: ', sequence_kwargs)
        dataset = concatenate_datasets(data_file, SequenceDataset, sequence_kwargs)
        # print('=================== dataset =================== ')
        # print(dataset)                # torch.utils.data.dataset.ConcatDataset
        # print('=================== dataset =================== ')
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


class MemMapDataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, SequenceDataset, sequence_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
