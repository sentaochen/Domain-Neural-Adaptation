# -*- coding: utf-8 -*-
"""
@author: ByteHong
@organization: SCUT
"""
import torch
import torch.nn as nn

import time
import math

from utils.lr_schedule import inv_lr_scheduler, multi_step_lr_scheduler
from utils.eval import test, predict
from utils import globalvar as gl

def finetune_for_UDA(args, model, optimizer, dataloaders):
    DEVICE = gl.get_value('DEVICE')
    check_path = gl.get_value('check_path')
    record_file = gl.get_value('record_file')
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_time = time.time()
        step = 0
        model.train()
        total_loss, correct = 0, 0
        for inputs, labels in dataloaders['src_pretrain']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs,_ = model(inputs)
                loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data).item()
        
        epoch_loss = total_loss / len(dataloaders['src_pretrain'].dataset)
        epoch_acc = float(correct) / len(dataloaders['src_pretrain'].dataset)
        print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, args.epochs, 'src_pretrain', epoch_loss, epoch_acc))  
        print('one epoch train time: {:.1f}s'.format(time.time()-epoch_time))
        test_time = time.time()
        loss_tar, acc_tar = test(model, dataloaders['src_test'])
        print('one test time: {:.1f}s'.format(time.time()-test_time))
        print('record {}'.format(record_file))
        with open(record_file, 'a') as f:
            f.write('Epoch {} acc_tar {:.4f}\n'.format(epoch, acc_tar))
        seconds = time.time() - start_time
        print('Epoch {} cost time: {}h {}m {:.0f}s\n'.format(epoch, seconds//3600, seconds%3600//60, seconds%60))
    torch.save(model.state_dict(), '{}/best(PT)_{}_{}.pth'.format(check_path, args.net, args.source))
    print('final model save!')
    time_pass = time.time() - start_time
    print('Training {} step complete in {}h {}m {:.0f}s\n'.format(step, time_pass//3600, time_pass%3600//60, time_pass%60))




def train_for_UDA(args, model, optimizer, dataloaders):
    DEVICE = gl.get_value('DEVICE')
    check_path = gl.get_value('check_path')
    record_file = gl.get_value('record_file')
    _,acc_src = test(model, dataloaders['src_test'])
    _,acc_tar = test(model, dataloaders['tar_test'])
    print('Initial model: acc_src:{:.4f}, acc_tar:{:.4f}'.format(acc_src, acc_tar))
    init_acc = acc_tar
    with open(record_file, 'a') as f:
        f.write('Initial model: acc_src:{:.4f}, acc_tar:{:.4f}\n'.format(acc_src, acc_tar))
    print('train DNA ing......')
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    if args.early:
        best_acc = 0
        best_acc_step = 0
        counter = 0
    src_data_l = iter(dataloaders['src_train_l'])
    tar_data_ul = iter(dataloaders['tar_train_ul'])
    
    start_time = time.time()
    avg_div_loss = 0
    for step in range(1, args.steps + 1):
        model.train()
        current_lr = inv_lr_scheduler(optimizer, step, args.lr_mult, init_lr=args.lr)
        lambd = 2 / (1 + math.exp(-10 * step / args.lam_step)) - 1
        inputs_l, labels_l = next(src_data_l)
        inputs_ul, _ = next(tar_data_ul)
        optimizer.zero_grad()
        s_img, s_label = inputs_l.to(DEVICE), labels_l.to(DEVICE)
        t_img = inputs_ul.to(DEVICE)
        s_output, div_loss = model(s_img, t_img, s_label)
        cls_loss = criterion(s_output, s_label)
        loss = lambd * div_loss + cls_loss
        loss.backward()
        optimizer.step()
        avg_div_loss += div_loss
        if step > 0 and step % args.log_interval == 0:
            print('Learning rate: {:.8f}'.format(current_lr))
            print('Step: [{}/{}]: lambd:{}, div_loss:{:.4f}, cls_loss:{:.4f}, total_loss:{:.4f}'.format(step, args.lam_step, lambd, div_loss, cls_loss, loss))
        if step > 0 and step % args.save_interval == 0:
            print('{} step train time: {:.1f}s'.format(args.save_interval, time.time()-start_time))
            test_time = time.time()
            _,acc_tar = test(model, dataloaders['tar_test'])
            print('one test time: {:.1f}s'.format(time.time()-test_time))
            print('record {}'.format(record_file))
            with open(record_file, 'a') as f:
                f.write('step {} avg_div_loss {:.4f}  acc_tar {:.4f} \n'.format(step, avg_div_loss/args.save_interval, acc_tar))
                avg_div_loss = 0
        
            if step >= args.start_update_step and step % args.update_interval == 0 :
                pseudo_labels = predict(model, dataloaders['tar_test'])
                dataloaders['tar_train_ul'].dataset.update_pseudo_labels(pseudo_labels)
                dataloaders['src_train_l'].batch_sampler.label_sampler.set_batch_num(args.update_interval)
                src_data_l = iter(dataloaders['src_train_l'])
                tar_data_ul = iter(dataloaders['tar_train_ul'])
            if args.early:
                if acc_tar > best_acc:
                    best_acc = acc_tar
                    best_acc_step = step
                    counter = 0
                    if args.save_check : 
                        torch.save(model.state_dict(), '{}/best_DNA_{}_to_{}.pth'.format(check_path, args.source, args.target))
                        # torch.save(model.state_dict(), '{}/best_DNA_to_{}.pth'.format(check_path, args.target))
                else:
                    counter += 1
                    if counter > args.patience:
                        print('early stop! training_step:{}'.format(step))
                        break
            seconds = time.time() - start_time
            print('{} step cost time: {}h {}m {:.0f}s\n'.format(step, seconds//3600, seconds%3600//60, seconds%60))
    time_pass = time.time() - start_time
    with open(record_file, 'a') as f:
        f.write('Training {} step complete in {}h {}m {:.0f}s\n'.format(step, time_pass//3600, time_pass%3600//60, time_pass%60))
    print('Training {} step complete in {}h {}m {:.0f}s\n'.format(step, time_pass//3600, time_pass%3600//60, time_pass%60))
    print('Training_step:{}, acc_tar:{}'.format(step, acc_tar))

