import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import random
import pickle

rng = np.random.RandomState(2020)



def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        # import pdb;pdb.set_trace()
        
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # if 'ped2' in self.dir and '12' not in video:
            #     continue
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        # videos = [videos[0]]
        for video in sorted(videos):
            # if 'ped2' in self.dir and '12' not in video:
            #     continue
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames               
            
        
    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0)
        
        
    def __len__(self):
        return len(self.samples)
class VideoDataLoader(data.Dataset):
    def __init__(self, video_folder, dataset_type, transform, resize_height, resize_width, time_step=4, segs=32, num_pred=1, batch_size=1):
        self.dir = video_folder
        self.dataset_type = dataset_type
        self.transform = transform
        self.videos = OrderedDict()
        self.video_names = []
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.num_segs = segs
        self.batch_size = batch_size
        
    def setup(self):
        train_folder = self.dir
        file_name = './data/frame_'+self.dataset_type+'.pickle'

        if os.path.exists(file_name):
            file = open(file_name,'rb')
            self.videos = pickle.load(file)
            for name in self.videos:
                self.video_names.append(name)
        else:
            videos = glob.glob(os.path.join(train_folder, '*'))
            
            for video in sorted(videos):
                video_name = video.split('/')[-1]
                self.video_names.append(video_name)
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = {}
        videos = glob.glob(os.path.join(self.dir, '*'))
        num = 0
        # videos = [videos[0]]
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            frames[video_name] = []
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames[video_name].append(self.videos[video_name]['frame'][i])
                num += 1
                           
        return frames, num
            
    
    def __getitem__(self, index):
        
        video_name = self.video_names[index]
        length = self.videos[video_name]['length']-self._time_step
        seg_ind = random.sample(range(0, self.num_segs), self.batch_size)
        frame_ind = random.sample(range(0, length//self.num_segs), 1)

        batch = []
        for j in range(self.batch_size):
            frame_name = seg_ind[j]*(length//self.num_segs)+frame_ind[0]
        
            for i in range(self._time_step+self._num_pred):
                image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
                if self.transform is not None:
                    batch.append(self.transform(image))
        return np.concatenate(batch, axis=0)
    
class MetaDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, task_size=2, segs=32, num_pred=1):
        if "UCF" in video_folder:
            self.dir = '/pcalab/tmp/UCF-Crime/UCF_Crimes/transed'
            self.pkl = '../ano_pred_cvpr2018/Data/UCF/normal_videos.pkl'
        else:
            self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self.video_names = []
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        # self.samples,_ = self.get_all_samples()
        self.task_size = task_size
        self.num_segs = segs
        # self.test()
        # import pdb;pdb.set_trace()

        
    def setup(self):
        # import pdb;pdb.set_trace()
        if "UCF" in self.dir:
            videos = glob.glob(os.path.join(self.dir, 'Nor*'))
            for video in sorted(videos):
                video_name = video.split('/')[-1]
                self.video_names.append(video_name)
                # self.videos[video_name] = {}
                # self.videos[video_name]['path'] = video
                # self.videos[video_name]['frame'] = glob.glob(os.path.join(video, 'img_*.jpg'))
                # self.videos[video_name]['frame'].sort()
                # self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            # import pdb;pdb.set_trace()
            # fw = open(self.pkl, "wb")
            # pickle.dump(self.videos, fw)
            fr = open(self.pkl, "rb")
            self.videos = pickle.load(fr)
            fr.close()
        else:
            videos = glob.glob(os.path.join(self.dir, '*'))
            
            for video in sorted(videos):
                video_name = video.split('/')[-1]
                self.video_names.append(video_name)
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = {}
        videos = glob.glob(os.path.join(self.dir, '*'))
        num = 0
        # videos = [videos[0]]
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            frames[video_name] = []
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames[video_name].append(self.videos[video_name]['frame'][i])
                num += 1
                           
        return frames, num
            
    
    def __getitem__(self, index):
        
        video_name = self.video_names[index]
        length = self.videos[video_name]['length']-4
        seg_ind = random.sample(range(0, self.num_segs), 1)
        frame_ind = random.sample(range(0, length//self.num_segs), self.task_size)

        batch = []
        for j in range(self.task_size):
            couple = []
            frame_name = seg_ind[0]*(length//self.num_segs)+frame_ind[j]
            for i in range(self._time_step+self._num_pred):
                image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
                # print(self.videos[video_name]['frame'][frame_name+i])
                if self.transform is not None:
                    couple.append(self.transform(image))
            batch.append(np.expand_dims(np.concatenate(couple, axis=0), axis=0))
        # import pdb;pdb.set_trace()
        return np.concatenate(batch, axis=0)

    def test(self, index=10):
        video_name = self.video_names[index]
        length = self.videos[video_name]['length']-4
        seg_ind = random.sample(range(0, self.num_segs), 1)
        frame_ind = random.sample(range(0, length//self.num_segs), self.task_size)

        batch = []
        for j in range(self.task_size):
            couple = []
            frame_name = seg_ind[0]*(length//self.num_segs)+frame_ind[j]
            for i in range(self._time_step+self._num_pred):
                image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width)
                # print(self.videos[video_name]['frame'][frame_name+i])
                if self.transform is not None:
                    couple.append(self.transform(image))
            # import pdb;pdb.set_trace()
            batch.append(np.expand_dims(np.concatenate(couple, axis=0), axis=0))
        import pdb;pdb.set_trace()
        return np.concatenate(batch, axis=0)
        
    def __len__(self):
        return len(self.video_names)
