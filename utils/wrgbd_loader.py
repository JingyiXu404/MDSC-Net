import fnmatch
import os

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset
import cv2
from utils import wrgbd51,ocid30,JHUIT50
from utils.basic_utils import RunSteps

"""
# WashingtonAll class is used to extract all cnn features at once without train/test splits.
# train/test splits are chosen later from the already saved/extracted files/features.
"""
def imsave(img, img_path):
    img_save = img.clone()
    # print(img_save[:,100,100:130])
    img_save = np.transpose(img_save.cpu().numpy(),(1,2,0))
    # print(img_save[100,100:130,:])
    img_save = np.uint8((img_save * 255.0).round())
    # print(img_save[100,100:130,:])
    cv2.imwrite(img_path, img_save)

class WashingtonAll(Dataset):
    def __init__(self, params, loader=None, transform=None):
        self.params = params
        self.loader = loader
        self.transform = transform
        self.data = self._init_dataset()
    def __getitem__(self, index):
        inp_path, out_path = self.data[index]
        datum = self.loader(inp_path, self.params)
        if self.transform is not None:
            datum = self.transform(datum)

        return datum, out_path

    def __len__(self):
        return len(self.data)

    def _init_dataset(self):
        data = []
        data_path = os.path.join(self.params.dataset_path, 'eval-set/')
        results_dir = self.params.dataset_path + self.params.features_root + self.params.proceed_step + '/' + \
                      self.params.net_model + '_results_' + self.params.data_type
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for category in sorted(os.listdir(data_path)):
            category_path = os.path.join(data_path, category)

            for instance in sorted(os.listdir(category_path)):
                instance_path = os.path.join(category_path, instance)

                data.extend(self.add_item(instance_path, results_dir))

        return data

    def add_item(self, instance_path, results_dir):
        indices = []
        suffix = '*_' + self.params.data_type + '.png'
        num_debug = 0

        for file in fnmatch.filter(sorted(os.listdir(instance_path)), suffix):
            path = os.path.join(instance_path, file)
            result_filename = results_dir + "/" + file + '.hdf5'
            item = (path, result_filename)
            indices.append(item)
            # get the first #debug_size (default=10) of sorted samples from each instance
            num_debug += 1
            if num_debug == self.params.debug_size and self.params.debug_mode:
                break
        return indices


class JHUIT50Dataset(Dataset):
    def __init__(self, params, phase, loader=None, transform=None):
        self.params = params
        self.phase = phase
        self.loader = loader
        self.transform = transform
        self.data = self._init_dataset()
    def __getitem__(self, index):
        if self.params.qloss:
            if self.params.data_type == 'rgbd':
                path1, path2, q_label, target = self.data[index]
                filename = path1[path1.rfind('/') + 1:]
                datum_RGB, datum_Depth = self.loader([path1, path2], self.params)
                # cv2.imwrite(filename+'_0.png', datum_RGB)
                # cv2.imwrite(filename+'_1.png', datum_Depth)
                if self.transform is not None:
                    datum_RGB = self.transform(datum_RGB)
                    datum_Depth = self.transform(datum_Depth)
                # imsave(datum_RGB,filename+'_2.png')
                # imsave(datum_Depth,filename+'_3.png')
                datum = torch.cat((datum_RGB, datum_Depth), dim=0)
            else:
                path, q_label, target = self.data[index]
                filename = path[path.rfind('/') + 1:]
                datum = self.loader([path], self.params)

                if self.transform is not None:
                    datum = self.transform(datum)

            return datum, q_label, target, filename
        else:
            if self.params.data_type == 'rgbd':
                path1,path2,target = self.data[index]
                filename = path1[path1.rfind('/') + 1:]
                datum_RGB, datum_Depth = self.loader([path1, path2], self.params)
                if self.transform is not None:
                    datum_RGB = self.transform(datum_RGB)
                    datum_Depth = self.transform(datum_Depth)
                datum = torch.cat((datum_RGB, datum_Depth), dim=0)
            else:
                path, target = self.data[index]
                filename = path[path.rfind('/') + 1:]
                datum = self.loader([path], self.params)
                # cv2.imwrite('1.png', datum)
                if self.transform is not None:
                    datum = self.transform(datum)
            return datum, target, filename

    def __len__(self):
        return len(self.data)

    def _init_dataset(self):
        data = []
        data_path = self.params.dataset_path
        for instance in sorted(os.listdir(data_path)):
            instance_path = os.path.join(data_path, instance)
            cat_ind = int(JHUIT50.class_name_to_id[instance])
            data.extend(self.add_item(instance_path, cat_ind))

        return data

    def add_item(self, instance_path, cat_ind):
        indices = []
        suffix = '*_rgbcrop.png'
        num_debug = 0
        if self.params.qloss:
            if self.params.cu:
                h = self.params.img_size // self.params.down_scale_encoder
                q_label = generate_Q(h=h, w=h, class_id=cat_ind, num_class=self.params.num_class,channel_per_class=self.params.M, cu=self.params.cu,times=self.params.down_time+1)
            else:
                h = self.params.img_size // self.params.down_scale_encoder
                q_label = generate_Q(h=h, w=h, class_id=cat_ind, num_class=self.params.num_class,channel_per_class=self.params.M, cu=self.params.cu,times=self.params.down_time+1)
            # print(q_label.shape)
        for file in fnmatch.filter(sorted(os.listdir(instance_path)), suffix):
            classname = JHUIT50.class_id_to_name[str(cat_ind)]
            setjudge = file[len(classname) + 1]
            if self.phase == 'train':
                if int(setjudge) >= 4:
                    continue
            else:
                if int(setjudge) < 4:
                    continue
            path = os.path.join(instance_path, file)
            if self.params.qloss:
                # RGBD
                if self.params.data_type == 'rgbd':
                    if self.params.depth_type == 3:  # 3 channel Depth
                        path2 = path.replace('_rgbcrop.png', '_depthsn.png')
                    else:  # 1 channel Depth
                        path2 = path.replace('_rgbcrop.png', '_depthcrop.png')
                    item = (path, path2, q_label, cat_ind)
                # Depth
                elif self.params.data_type == 'depthcrop':
                    if self.params.depth_type == 3:  # 3 channel Depth
                        path2 = path.replace('_depthcrop.png', '_depthsn.png')
                    else:  # 1 channel Depth
                        path2 = path
                    item = (path2, q_label, cat_ind)
                # RGB
                else:
                    item = (path, q_label, cat_ind)
            else:
                if self.params.data_type == 'rgbd':
                    if self.params.depth_type == 3:  # 3 channel Depth
                        path2 = path.replace('_rgbcrop.png', '_depthsn.png')
                    else:  # 1 channel Depth
                        path2 = path.replace('_rgbcrop.png', '_depthcrop.png')
                    item = (path, path2, cat_ind)
                elif self.params.data_type == 'depthcrop':
                    if self.params.depth_type == 3:  # 3 channel Depth
                        path2 = path.replace('_depthcrop.png', '_depthsn.png')
                    else:  # 1 channel Depth
                        path2 = path
                    item = (path2, cat_ind)
                else:
                    item = (path, cat_ind)
            indices.append(item)
            num_debug += 1
            if num_debug == self.params.debug_size and self.params.debug_mode:
                break
        return indices

class OCIDDataset(Dataset):
    def __init__(self, params, phase, loader=None, transform=None):
        self.params = params
        self.phase = phase
        self.loader = loader
        self.transform = transform
        self.data = self._init_dataset()
    def __getitem__(self, index):
        if self.params.qloss:
            if self.params.data_type == 'rgbd':
                path1, path2, q_label, target = self.data[index]
                filename = path1[path1.rfind('/') + 1:]
                datum_RGB, datum_Depth = self.loader([path1, path2], self.params)
                # cv2.imwrite(filename+'_0.png', datum_RGB)
                # cv2.imwrite(filename+'_1.png', datum_Depth)
                if self.transform is not None:
                    datum_RGB = self.transform(datum_RGB)
                    datum_Depth = self.transform(datum_Depth)
                # imsave(datum_RGB,filename+'_2.png')
                # imsave(datum_Depth,filename+'_3.png')
                datum = torch.cat((datum_RGB, datum_Depth), dim=0)
            else:
                path, q_label, target = self.data[index]
                filename = path[path.rfind('/') + 1:]
                datum = self.loader([path], self.params)

                if self.transform is not None:
                    datum = self.transform(datum)

            return datum, q_label, target, filename
        else:
            if self.params.data_type == 'rgbd':
                path1,path2,target = self.data[index]
                filename = path1[path1.rfind('/') + 1:]
                datum_RGB, datum_Depth = self.loader([path1, path2], self.params)
                if self.transform is not None:
                    datum_RGB = self.transform(datum_RGB)
                    datum_Depth = self.transform(datum_Depth)
                datum = torch.cat((datum_RGB, datum_Depth), dim=0)
            else:
                path, target = self.data[index]
                filename = path[path.rfind('/') + 1:]
                datum = self.loader([path], self.params)
                # cv2.imwrite('1.png', datum)
                if self.transform is not None:
                    datum = self.transform(datum)
            return datum, target, filename

    def __len__(self):
        return len(self.data)

    def _init_dataset(self):
        data = []
        # train_rgb_path = self.params.dataset_path + '/ARID20_crops/squared_rgb/'
        # train_depth_path = self.params.dataset_path + '/ARID20_crops/surfnorm++/'
        # test_rgb_path = self.params.dataset_path + '/ARID10_crops/squared_rgb/'
        # test_depth_path = self.params.dataset_path + '/ARID10_crops/surfnorm++/'

        if self.phase == 'train':
            data_path = os.path.join(self.params.dataset_path, 'ARID20_crops/')
        else:
            data_path = os.path.join(self.params.dataset_path, 'ARID10_crops/')
        for category in sorted(os.listdir(os.path.join(data_path,'squared_rgb/'))):
            #30 categories for training
            #28 categories for testing
            category_path = os.path.join(data_path, 'squared_rgb/', category)
            cat_ind = int(ocid30.class_name_to_id[category])
            for instance in sorted(os.listdir(category_path)):
                instance_path = os.path.join(category_path, instance)
                data.extend(self.add_item(instance_path, cat_ind))

        return data

    def add_item(self, instance_path, cat_ind):
        indices = []
        suffix = '*.png'
        num_debug = 0
        if self.params.qloss:
            if self.params.cu:
                h = self.params.img_size // self.params.down_scale_encoder
                q_label = generate_Q(h=h, w=h, class_id=cat_ind, num_class=self.params.num_class,channel_per_class=self.params.channel_per_class, cu=self.params.cu,times=self.params.down_time)
            else:
                h = self.params.img_size // self.params.down_scale_encoder
                q_label = generate_Q(h=h, w=h, class_id=cat_ind, num_class=self.params.num_class, channel_per_class=self.params.channel_per_class, cu = self.params.cu, times = self.params.down_time+1)
            # print(q_label.shape)

        for file in fnmatch.filter(sorted(os.listdir(instance_path)), suffix):
            path_rgb = os.path.join(instance_path, file)
            path_depth = path_rgb.replace('squared_rgb','surfnorm++')
            if not os.path.exists(path_rgb) or not os.path.exists(path_depth):
                continue
            if self.params.qloss:
                #RGBD
                if self.params.data_type == 'rgbd':
                    item = (path_rgb, path_depth, q_label, cat_ind)
                #Depth
                elif self.params.data_type == 'depthcrop':
                    pass
                # RGB
                else:
                    item = (path_rgb,q_label, cat_ind)
            else:
                if self.params.data_type == 'rgbd':
                    item = (path_rgb,path_depth,cat_ind)
                elif self.params.data_type == 'depthcrop':
                    pass
                else:
                    item = (path_rgb, cat_ind)
            indices.append(item)
            num_debug += 1
            if num_debug == self.params.debug_size and self.params.debug_mode:
                break
        return indices

class WashingtonDataset(Dataset):
    def __init__(self, params, phase, loader=None, transform=None):
        self.params = params
        self.phase = phase
        self.loader = loader
        self.transform = transform
        self.data = self._init_dataset()
    def __getitem__(self, index):
        if self.params.qloss:
            if self.params.data_type == 'rgbd':
                path1, path2, q_label, target = self.data[index]
                filename = path1[path1.rfind('/') + 1:]
                datum_RGB, datum_Depth = self.loader([path1, path2], self.params)
                # cv2.imwrite(str(index)+'_0.png', datum_RGB)
                # cv2.imwrite(str(index)+'_1.png', datum_Depth)
                if self.transform is not None:
                    datum_RGB = self.transform(datum_RGB)
                    datum_Depth = self.transform(datum_Depth)
                    # imsave(datum_RGB,str(index)+'_2.png')
                    # imsave(datum_Depth,str(index)+'_3.png')
                datum = torch.cat((datum_RGB, datum_Depth), dim=0)
            else:
                path, q_label, target = self.data[index]
                filename = path[path.rfind('/') + 1:]
                datum = self.loader([path], self.params)

                if self.transform is not None:
                    datum = self.transform(datum)

            return datum, q_label, target, filename
        else:
            if self.params.data_type == 'rgbd':
                path1,path2,target = self.data[index]
                filename = path1[path1.rfind('/') + 1:]
                datum_RGB, datum_Depth = self.loader([path1, path2], self.params)
                if self.transform is not None:
                    datum_RGB = self.transform(datum_RGB)
                    datum_Depth = self.transform(datum_Depth)
                datum = torch.cat((datum_RGB, datum_Depth), dim=0)
            else:
                path, target = self.data[index]
                filename = path[path.rfind('/') + 1:]
                datum = self.loader([path], self.params)
                # cv2.imwrite('1.png', datum)
                if self.transform is not None:
                    datum = self.transform(datum)
            return datum, target, filename

    def __len__(self):
        return len(self.data)

    def _init_dataset(self):
        data = []
        data_path = os.path.join(self.params.dataset_path, 'eval-set/')
        split_file = os.path.join(self.params.dataset_path, 'splits.mat')

        split_data = io.loadmat(split_file)['splits'].astype(np.uint8)
        test_instances = split_data[:, self.params.split_no - 1]
        # print('sorted(os.listdir(data_path))',len(sorted(os.listdir(data_path))))
        for category in sorted(os.listdir(data_path)):#51 categories

            category_path = os.path.join(data_path, category)
            cat_ind = int(wrgbd51.class_name_to_id[category])
            # print('sorted(os.listdir(category_path))',sorted(os.listdir(category_path)))
            for instance in sorted(os.listdir(category_path)):
                instance_path = os.path.join(category_path, instance)

                if self.phase == 'test':
                    if test_instances[cat_ind] == np.uint8(instance.split('_')[-1]):
                        data.extend(self.add_item(instance_path, cat_ind))
                else:
                    if test_instances[cat_ind] != np.uint8(instance.split('_')[-1]):
                        data.extend(self.add_item(instance_path, cat_ind))
                    # debug
                    # if test_instances[cat_ind] == np.uint8(instance.split('_')[-1]):
                    #     data.extend(self.add_item(instance_path, cat_ind))

        return data

    def add_item(self, instance_path, cat_ind):
        indices = []
        if self.params.data_type == 'rgbd':
            suffix = '*_crop.png'
        else:
            suffix = '*_' + self.params.data_type + '.png'
        num_debug = 0
        if self.params.qloss:
            if self.params.cu:
                h = self.params.img_size // self.params.down_scale_encoder
                q_label = generate_Q(h=h, w=h, class_id=cat_ind, num_class=self.params.num_class,channel_per_class=self.params.M, cu=self.params.cu,times=self.params.down_time+1)
            else:
                h = self.params.img_size // self.params.down_scale_encoder
                q_label = generate_Q(h=h, w=h, class_id=cat_ind, num_class=self.params.num_class, channel_per_class=self.params.M, cu = self.params.cu, times = self.params.down_time+1)
            # print(q_label.shape)
            #debug
            # q = np.transpose(q_label,(1, 2, 0))
            # for i in range(q.shape[2]):
            #     os.makedirs('q/'+str(cat_ind)+'/',exist_ok=True)
            #     print(cat_ind,i,q[0,0,i:i+1])
            #     cv2.imwrite('q/'+str(cat_ind)+'/'+str(i)+'.png',255.*q[:,:,i:i+1].numpy().astype(np.float32))
        for file in fnmatch.filter(sorted(os.listdir(instance_path)), suffix):
            path = os.path.join(instance_path, file)
            # print(file)#water_bottle_2_4_96_depthcrop.png
            # print(path)#/temp_disk2/zyt/XJY/dataset/wrgbd/eval-set/pliers/pliers_6/pliers_6_4_96_depthcrop.png
            if self.params.qloss:
                #RGBD
                if self.params.data_type == 'rgbd':
                    if self.params.depth_type == 3: #3 channel Depth
                        path2 = path.replace('_crop.png', '_depthsn.png')
                    else: #1 channel Depth
                        path2 = path.replace('_crop.png', '_depthcrop.png')
                    item = (path, path2, q_label, cat_ind)
                #Depth
                elif self.params.data_type == 'depthcrop':
                    if self.params.depth_type == 3: #3 channel Depth
                        path2 = path.replace('_depthcrop.png', '_depthsn.png')
                    else: #1 channel Depth
                        path2 = path
                    item = (path2, q_label, cat_ind)
                # RGB
                else:
                    item = (path,q_label, cat_ind)
            else:
                if self.params.data_type == 'rgbd':
                    if self.params.depth_type == 3: #3 channel Depth
                        path2 = path.replace('_crop.png', '_depthsn.png')
                    else: #1 channel Depth
                        path2 = path.replace('_crop.png', '_depthcrop.png')
                    item = (path,path2,cat_ind)
                elif self.params.data_type == 'depthcrop':
                    if self.params.depth_type == 3: #3 channel Depth
                        path2 = path.replace('_depthcrop.png', '_depthsn.png')
                    else: #1 channel Depth
                        path2 = path
                    item = (path2, cat_ind)
                else:
                    item = (path, cat_ind)
            indices.append(item)
            # get the first debug_size (default=10) of sorted samples from each instance
            num_debug += 1
            if num_debug == self.params.debug_size and self.params.debug_mode:
                break
        return indices
def generate_Q(h,w, class_id, num_class, channel_per_class, cu, times):
    if cu:
        C = num_class * channel_per_class
    else:
        C = num_class * channel_per_class
    one_begin = int(class_id) * channel_per_class
    one_end = (int(class_id) + 1) * channel_per_class
    # print(one_begin,one_end)

    Q = torch.zeros((h, w, C)).long()
    Q[:, :, one_begin:one_end] = torch.tensor([1]).long()
    Q = Q.permute(2, 0, 1)
    return Q
