# -*- coding: utf-8 -*-
"""
@author: ByteHong
@organization: SCUT
"""
import os
import torch
from torchvision import datasets, transforms

from loader.data_list import ImageList
from loader.sampler import BatchSampler

def load_data_for_UDA(root_dir, dataset, src_domain, tar_domain, phase, batch_size, net, label_sampler=None, is_extract=False, pre_type=1, args=None):
    crop_size = 224 if net != 'alexnet' else 227
    resize_size = 256
    if pre_type==1:
        transform_dict = {
            'train': transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ])}

    list_root_dir = os.path.join(root_dir, 'list')
    data_root_dir = os.path.join(root_dir)
    # data_list file name
    labeled_source_list = os.path.join(list_root_dir, '{}.txt'.format(src_domain))
    unlabeled_target_list = os.path.join(list_root_dir, '{}.txt'.format(tar_domain))
    
    if phase == 'pretrain':
        src_data_pretrain = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'])
        
        src_loader_pretrain = torch.utils.data.DataLoader(src_data_pretrain, batch_size=batch_size*2, shuffle=True, drop_last=True,
                                               num_workers=4)
        return src_loader_pretrain
    elif phase=='train':
        src_data_labeled = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'], type='labeled')
        tar_data_unlabeled = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['train'], type='unlabeled')
        src_batch_sampler_labeled = BatchSampler(src_data_labeled, label_sampler, batch_size )
        tar_batch_sampler_unlabeled = BatchSampler(tar_data_unlabeled, label_sampler, batch_size)
        src_loader_labeled = torch.utils.data.DataLoader(src_data_labeled, batch_sampler = src_batch_sampler_labeled, num_workers=4)
        tar_loader_unlabeled = torch.utils.data.DataLoader(tar_data_unlabeled, batch_sampler = tar_batch_sampler_unlabeled, num_workers=4)
        return src_loader_labeled, tar_loader_unlabeled
    else: 
        ## if phase=='test'
        src_data_test = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['test'])
        tar_data_test = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['test'], type='labeled')
        if is_extract == True:
            src_loader_test = torch.utils.data.DataLoader(src_data_test, batch_size=batch_size*2, shuffle=True, drop_last=False,
                                             num_workers=4)
            tar_loader_test = torch.utils.data.DataLoader(tar_data_test, batch_size=batch_size*2, shuffle=True, drop_last=False,
                                             num_workers=4)
        else:
            src_loader_test = torch.utils.data.DataLoader(src_data_test, batch_size=batch_size*2, shuffle=False, drop_last=False,
                                             num_workers=4)
            tar_loader_test = torch.utils.data.DataLoader(tar_data_test, batch_size=batch_size*2, shuffle=False, drop_last=False,
                                             num_workers=4)
        return src_loader_test, tar_loader_test

