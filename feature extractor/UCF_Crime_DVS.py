from typing import Callable, Dict, Optional, Tuple
import numpy as np
from spikingjelly import datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import time
import sys
# Add /usr/lib/python3/dist-packages/ to PYTHONPATH if the output of print(sys.path) does not mention it.
sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_core.event_io import EventsIterator
from tqdm import tqdm
from spikingjelly import configure
import torch
import sys
np_savez = np.savez_compressed if configure.save_datasets_compressed else np.savez


class UCF_Crime_DVS(sjds.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            segment_size: int =64,
    ) -> None:
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.segment_size = segment_size



    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is class_index of the target class.
    #     """
    #     path, target = self.samples[index]
    #     sample = self.loader(path)
    #
    #     # 将样本分成多个小段
    #     segments = self.split_sample(sample)
    #
    #     # 逐个小段进行预处理并返回
    #     all_segments = []
    #     for segment in segments:
    #         segment = self.resize(segment)
    #         if self.transform is not None:
    #             segment = self.transform(segment)
    #         all_segments.append(segment)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #     return all_segments, target
    #
    #
    # def split_sample(self, sample):
    #     """
    #     将样本分成多个小段
    #     """
    #     T, C, H, W = sample.shape
    #     num_segments = T // self.segment_size
    #     if T % self.segment_size != 0:
    #         num_segments += 1
    #
    #     for i in range(num_segments):
    #         start = i * self.segment_size
    #         end = min((i + 1) * self.segment_size, T)
    #         yield sample[start:end]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)



        # 在这里添加自定义的预处理逻辑
        sample = self.resize(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target ,path
        # """
        # Args:
        #     index (int): Index
        #
        # Returns:
        #     list: 包含一个或多个 (sample, target) 元组的列表,其中 sample 的时间维度小于等于 1000。
        # """
        # path, target = self.samples[index]
        # sample = self.loader(path)
        #
        # # 对样本进行 resize
        # split_samples = self.resize(sample)
        #
        # # 如果只有一个片段,直接返回
        # if len(split_samples) == 1:
        #     return split_samples[0], target
        #
        # # 如果有多个片段,将它们封装成一个列表返回
        # return_samples = []
        # for split_sample in split_samples:
        #     if self.transform is not None:
        #         split_sample = self.transform(split_sample)
        #     if self.target_transform is not None:
        #         target = self.target_transform(target)
        #     return_samples.append((split_sample, target))
        #
        # return return_samples

    def resize(self, sample):
        """
        在这里添加对样本进行resize的逻辑
        """
        # 假设sample的格式为 (T, C, H, W)
        T, C, H, W = sample.shape

        # 逐帧对样本进行resize
        resized_sample = []
        for t in range(T):
            frame = torch.from_numpy(sample[t])
            resized_frame = F.interpolate(frame.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            resized_sample.append(resized_frame.squeeze(0))

        # 将resize后的帧合并为新的样本
        resized_sample = torch.stack(resized_sample, dim=0)

        return resized_sample
        # if T > 1000:
        #     split_sizes = []
        #     while T > 1000:
        #         size = T // 2
        #         split_sizes.append(size)
        #         T = size
        #     split_sizes.append(T)
        #     split_samples = torch.split(torch.from_numpy(sample), split_sizes, dim=0)
        #
        #     # 对每个片段进行resize
        #     resized_samples = []
        #     for split_sample in split_samples:
        #         t, c, h, w = split_sample.shape
        #         resized_sample = []
        #         for i in range(t):
        #             frame = split_sample[i]
        #             resized_frame = F.interpolate(frame.unsqueeze(0), size=(128, 128), mode='bilinear',
        #                                           align_corners=False)
        #             resized_sample.append(resized_frame.squeeze(0))
        #         resized_sample = torch.stack(resized_sample, dim=0)
        #         resized_samples.append(resized_sample)
        #
        #     return resized_samples
        # else:
        #     # 如果T小于等于1000,直接对整个样本进行resize
        #     resized_sample = []
        #     for i in range(T):
        #         frame = torch.from_numpy(sample[i])
        #         resized_frame = F.interpolate(frame.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
        #         resized_sample.append(resized_frame.squeeze(0))
        #     resized_sample = torch.stack(resized_sample, dim=0)
        #     return [resized_sample]

    # def get_label_from_filename(filename):
    #     if 'Abuse' in filename:
    #         return 0
    #     elif 'Arrest' in filename:
    #         return 1
    #     elif 'Arson' in filename:
    #         return 2
    #     elif 'Assault' in filename:
    #         return 3
    #     elif 'Burglary' in filename:
    #         return 4
    #     elif 'Explosion' in filename:
    #         return 5
    #     elif 'Fighting' in filename:
    #         return 6
    #     elif 'Normal' in filename:
    #         return 7
    #     elif 'Road' in filename:
    #         return 8
    #     elif 'Robbery' in filename:
    #         return 9
    #     elif 'Shooting' in filename:
    #         return 10
    #     elif 'Shoplifting' in filename:
    #         return 11
    #     elif 'Stealing' in filename:
    #         return 12
    #     elif 'Vandalism' in filename:
    #         return 13
    #     else:
    #         return None

    def get_label_from_filename(filename):
        label_mapping = {
            'Abuse': 0,
            'Arrest': 1,
            'Arson': 2,
            'Assault': 3,
            'Burglary': 4,
            'Explosion': 5,
            'Fighting': 6,
            'Normal': 7,
            'Road': 8,
            'Robbery': 9,
            'Shooting': 10,
            'Shoplifting': 11,
            'Stealing': 12,
            'Vandalism': 13,
        }
        for keyword, label in label_mapping.items():
            if keyword in filename:
                return label
        return None

    @staticmethod
    def load_raw_files_to_np(args):

    # def load_raw_files_to_np(fname: str,raw_file:str,output_dir: str):
        fname, raw_file, output_dir = args
        print(f'Start to split [{raw_file}] to samples.')
        data = {'t': [], 'x': [], 'y': [], 'p': []}
        max_duration = None
        delta_t = 33333
        # 迭代遍历 EventsIterator
        mv_iterator = EventsIterator(raw_file, delta_t=delta_t,
                                     max_duration=max_duration)
        #file_name 按照读入的文件名，最后把.raw替换为.npz
        label = UCF_Crime_DVS.get_label_from_filename(fname)


        file_name = os.path.join(output_dir, str(label), f'{fname}.npz')
        for ev in tqdm(mv_iterator):
            # 将 ev 转换为字典
            for (x, y, p, t) in ev:
                data['t'].append(t)
                data['x'].append(x)
                data['y'].append(y)
                data['p'].append(p)
        data_np = {f'{k}': np.array(v) for k, v in data.items()}
        # Save data as .npy file
        np_savez(file_name, **data_np)


    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        raw_dir = os.path.join(extract_root, 'UCF_Crime_DVS')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in range(14):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        with open(os.path.join(raw_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(raw_dir, 'trials_to_test.txt')) as trials_to_test_txt,open('/media/qyb/My Passport/UCFC-DVS_raw/exist_name.txt') as exist_file:
            # use multi-thread to accelerate
            t_ckp = time.time()

            with multiprocessing.Pool(processes=2) as pool:
                print(f'Start the multiprocessing Pool with max workers = [{pool._processes}].')
                f_name = [f.strip() for f in exist_file.readlines()]

                args_list_train = [(os.path.splitext(fname.strip())[0], os.path.join(raw_dir, fname.strip()), train_dir) for fname in
                                   trials_to_train_txt.readlines() if len(fname.strip()) > 0 and fname.strip() not in f_name]
                pool.map(UCF_Crime_DVS.load_raw_files_to_np, args_list_train)

                args_list_test = [(os.path.splitext(fname.strip())[0], os.path.join(raw_dir, fname.strip()), test_dir) for fname in
                                  trials_to_test_txt.readlines() if len(fname.strip()) > 0 and fname.strip() not in f_name]
                pool.map(UCF_Crime_DVS.load_raw_files_to_np, args_list_test)
            # with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
            #     print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
            #
            #     for fname in trials_to_train_txt.readlines():
            #         fname = fname.strip()
            #         if fname.__len__() > 0:
            #             raw_file = os.path.join(raw_dir, fname)
            #             fname = os.path.splitext(fname)[0]
            #
            #             tpe.submit(UCF_Crime_DVS.load_raw_files_to_np, fname, raw_file, train_dir)
            #             # UCF_Crime_DVS.load_raw_files_to_np(fname, raw_file, train_dir)
            #
            #     for fname in trials_to_test_txt.readlines():
            #         fname = fname.strip()
            #         if fname.__len__() > 0:
            #             raw_file = os.path.join(raw_dir, fname)
            #             fname = os.path.splitext(fname)[0]
            #             tpe.submit(UCF_Crime_DVS.load_raw_files_to_np, fname, raw_file, test_dir)
            #             # UCF_Crime_DVS.load_raw_files_to_np(fname, raw_file, test_dir)


            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All raw files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 720, 1280

# if __name__ == '__main__':
#     train_set = UCF_Crime_DVS(root='/media/qyb/My Passport/UCFC-DVS_raw',train=True,data_type='frame',duration=533328)
#     data_loader = torch.utils.data.DataLoader(
#         dataset=train_set,
#         batch_size=16,
#         shuffle=True,
#         num_workers=4,
#         drop_last=True,
#         # sampler=train_sampler,
#         pin_memory=True)

