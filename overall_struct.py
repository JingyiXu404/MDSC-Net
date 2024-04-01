import logging

import torch
from torch import nn
from torchvision import models

import utils.basic_utils as basic_utils
import recursive_nn
import utils.wrgbd51
from utils.basic_utils import Models, RunSteps, PrForm, OverallModes, DataTypes,classify
from utils.loader_utils import custom_loader
from utils.model_utils import get_data_transform
from resnet_models import ResNet
from vgg16_model import VGG16Net
from utils.wrgbd_loader import WashingtonDataset
import cv2
from cls_net import clsNet
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
from IPython import embed
import torch.nn.functional as F
import utils.freeze as freeze
from ptflops import get_model_complexity_info
matplotlib.use('agg')
warnings.filterwarnings(action='ignore')
def tensor2uint(img: torch.Tensor) -> np.ndarray:
    # print('1',img[0])
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    # print('2', img[0])
    # np.uint8((img * 255.0).round())
    return img
def process_data(img):
    img = img.detach().float().cpu().data.squeeze().float()
    return img
def imsave(img, img_path):
    img_save = img.clone()
    # print('1',img_save[0,:])
    img_save = np.squeeze(img_save.clamp_(0, 1).cpu().numpy())
    # print('2',img_save[0,:])
    img_save = np.uint8((img_save * 255.0).round())
    # print(img_path,img_save[0,:])
    cv2.imwrite(img_path, img_save)
def cal_l1(input):
    # print(np.absolute(input.numpy())[0,:])
    SUM = np.sum(np.absolute(input.numpy()))
    SUM =  round(SUM,2)
    # print(SUM)
    return SUM
def cal_l0(input):
    # print(input.shape,np.count_nonzero(input.numpy()))
    SUM = np.count_nonzero(input.numpy())
    return SUM

class_id_to_name = {
    "0": "apple",
    "1": "ball",
    "2": "banana",
    "3": "bell_pepper",
    "4": "binder",
    "5": "bowl",
    "6": "calculator",
    "7": "camera",
    "8": "cap",
    "9": "cell_phone",
    "10": "cereal_box",
    "11": "coffee_mug",
    "12": "comb",
    "13": "dry_battery",
    "14": "flashlight",
    "15": "food_bag",
    "16": "food_box",
    "17": "food_can",
    "18": "food_cup",
    "19": "food_jar",
    "20": "garlic",
    "21": "glue_stick",
    "22": "greens",
    "23": "hand_towel",
    "24": "instant_noodles",
    "25": "keyboard",
    "26": "kleenex",
    "27": "lemon",
    "28": "lightbulb",
    "29": "lime",
    "30": "marker",
    "31": "mushroom",
    "32": "notebook",
    "33": "onion",
    "34": "orange",
    "35": "peach",
    "36": "pear",
    "37": "pitcher",
    "38": "plate",
    "39": "pliers",
    "40": "potato",
    "41": "rubber_eraser",
    "42": "scissors",
    "43": "shampoo",
    "44": "soda_can",
    "45": "sponge",
    "46": "stapler",
    "47": "tomato",
    "48": "toothbrush",
    "49": "toothpaste",
    "50": "water_bottle"
}
class_name_to_id = {v: k for k, v in class_id_to_name.items()}
class_names = set(class_id_to_name.values())

def get_class_name(cls_id):
    cls_name = class_id_to_name[str(cls_id)]
    return np.asarray(cls_name)
def get_class_names(ids):
    names = []
    for cls_id in ids:
        cls_name = class_id_to_name[str(cls_id)]
        names.append(cls_name)
    return np.asarray(names)
def get_class_ids(cls_name):
    cls_id = class_name_to_id[cls_name]
    return np.asarray(cls_id)
def metricCLS(y, y_pred):

    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average="macro")
    f1 = f1_score(y, y_pred, average="macro")
    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    return acc, precision, recall, f1, kappa

def calculate_class_name(out_class,label):
    out_sort_list = np.argpartition(out_class, -5)
    out_id = out_sort_list[-5:]
    out_id = out_id[::-1]
    label_id = label
    out_class_name = get_class_names(out_id)
    label_name = get_class_names([label_id])
    return out_class_name,label_name

def save_class_txts_top5(filepath,outputs,labels,img_names):
    for j in os.listdir(filepath):
        delate_path = os.path.join(filepath, j)
        os.remove(delate_path)
    length = len(img_names)
    for i in range(length):
        output = outputs[i]
        label = labels[i]
        img_name = img_names[i]
        out_class_name,label_name = calculate_class_name(output,label)
        item = img_name.split('_')[0]
        if item in class_names:
            pass
        else:
            for class_name_i in class_names:
                if item in class_name_i:
                    item = class_name_i
        img_path = os.path.join(
            filepath,
            f"{item}.txt"
        )
        file = open(img_path,'a+')
        file.write(img_name+' ## Top-5: '+' '.join(out_class_name)+' ## label: '+''.join(label_name)+'\n')
        file.flush()

def save_class_txts_top1(filepath,outputs,labels,img_names):
    for j in os.listdir(filepath):
        delate_path = os.path.join(filepath, j)
        os.remove(delate_path)
    length = len(img_names)
    for i in range(length):
        output = outputs[i]
        label = labels[i]
        img_name = img_names[i]
        out_class_name = get_class_name(output)
        label_name = get_class_name(label)
        item = img_name.split('_')
        img_path = os.path.join(
            filepath,
            f"{item[0]}.txt"
        )

        file = open(img_path,'a+')
        file.write(img_name+' ## Top-1: '+' '.join(str(out_class_name))+' ## label: '+''.join(str(label_name))+'\n')
        file.flush()
def get_parameters(net):
    for name,params in net.named_parameters():
        print(name)
def get_freeze_encoder(model,params):
    if params.gpu:
        freeze.freeze_by_names(model.module.MMCSC,'encoder')
    else:
        freeze.freeze_by_names(model.modules.module.MMCSC,'encoder')
def get_freeze_encoder_lastfine(model,params):
    if params.gpu:
        freeze.freeze_by_names(model.module.MMCSC, 'encoder')
        freeze.unfreeze_by_names(model.module.MMCSC.encoder,'fine_c_encoder_end')
    else:
        freeze.freeze_by_names(model.modules.module.MMCSC, 'encoder')
        freeze.unfreeze_by_names(model.modules.module.MMCSC.encoder,'fine_c_encoder_end')
def get_unfreeze_encoder(model,params):
    if params.gpu:
        freeze.unfreeze_by_names(model.module.MMCSC,'encoder')
    else:
        freeze.unfreeze_by_names(model.modules.module.MMCSC,'encoder')
def process_networks(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that I force to use cuda here
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))
    model_ft = clsNet(classes_num=params.num_class, basic_model=params.net_model, pretrain=params.pretrained, need_gamma=params.qloss)

    EPOCH = params.EPOCH
    phase = params.phase
    split = params.split_no
    use_qloss = params.qloss
    # Set model to evaluation mode (without this, results will be completely different)
    # Remember that you must call model.eval() to set dropout and batch normalization layers
    # to evaluation mode before running inference.
    macs, parameters = get_model_complexity_info(model_ft, (6, 224, 224), as_strings=True, print_per_layer_stat=True,verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', parameters))
    if params.gpu:
        model_ft = nn.DataParallel(model_ft).cuda()
    model_ft = model_ft.to(device)

    if params.ablation:
        model_name = params.net_model + '_' + str(params.k_c) + '_' + str(params.k_f) + '_' + str(params.M) + '_' + str(params.J)
    else:
        model_name = params.net_model

    os.makedirs("./debug/split_" + str(split) + "/" + model_name + "/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/" + model_name + "/saved_models/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/" + model_name + "/test_metrics/", exist_ok=True)
    txtfile = "./debug/split_" + str(split) + "/" + model_name + "/test_metrics/results.txt"
    os.makedirs("./debug/split_" + str(split) + "/" + model_name + "/loss_curve/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/" + model_name + "/saved_top5/", exist_ok=True)
    filepath = "./debug/split_" + str(split) + "/" + model_name + "/saved_top5/"

    print("loading dataset-------------------------------------")
    data_form = get_data_transform(params.data_type)
    # print('-------------train_dataset------------------------')
    training_set = WashingtonDataset(params, phase='train', loader=custom_loader, transform=data_form)
    train_loader = torch.utils.data.DataLoader(training_set, params.batch_size, shuffle=True)
    # print('-------------test_dataset------------------------')
    test_set = WashingtonDataset(params, phase='test', loader=custom_loader, transform=data_form)
    test_loader = torch.utils.data.DataLoader(test_set, params.batch_size, shuffle=False)

    data_loaders = {'train': train_loader, 'test': test_loader}
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=params.lr)
    CE_criterion = nn.CrossEntropyLoss().cuda()
    best_OA = 0.80051  # 0

    if phase == 'train':
        if os.path.exists('./debug/split_' + str(split) + '/' + model_name + '/saved_models/best.pth'):
            print("loading best model-------------------------------------")
            model_ft.load_state_dict(torch.load('./debug/split_' + str(split) + '/' + model_name + '/saved_models/best.pth'))
        with open(txtfile, "a+") as file:
            plot_loss = []
            plot_batchAcc = []
            plot_ACC = []
            plot_Presision = []
            plot_Recall = []
            plot_F1 = []
            plot_Kappa = []
            for epoch in range(74,EPOCH+1):
                train_bar = tqdm(data_loaders[phase])
                if epoch % 100 == 0:
                    new_lr = params.lr / (2*(epoch//100))
                    for para_group in optimizer.param_groups:
                        para_group['lr'] = new_lr
                    print("Learning weight decays to %f"%(new_lr))
                epoch_loss = []
                batch_acc = []
                batch_ind = 0
                for inputs, labels, filenames in train_bar:
                    # print(filenames,inputs.shape)
                    batch_ind += 1
                    model_ft = model_ft.train()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    if params.qloss:
                        outputs, _ = model_ft(inputs)
                    else:
                        outputs = model_ft(inputs)
                    loss = CE_criterion(outputs, labels)*10000000
                    _, predicted = torch.max(outputs.data, 1)

                    loss.backward()
                    optimizer.step()
                    total = labels.size(0)
                    correct = predicted.eq(labels.data).cpu().sum()

                    epoch_loss.append(loss.item())
                    batch_acc.append(100.*correct/total)
                    train_bar.set_description(
                        desc=model_name + ' [%d/%d] ce_loss: %.4f  | batch_acc: %.4f' % (
                            epoch, EPOCH,
                            loss.item(),
                            100.*correct/total,
                        ))
                save = './debug/split_'+str(split)+'/'+model_name+ '/saved_models/latest.pth'
                torch.save(model_ft.state_dict(), save)


                fig1 = plt.figure()
                plot_loss.append(np.mean(epoch_loss))
                plt.plot(plot_loss)
                plt.savefig('./debug/split_'+str(split)+'/'+model_name+'/loss_curve/CE_loss.png')
                fig2 = plt.figure()
                plot_batchAcc.append(np.mean(batch_acc))
                plt.plot(plot_batchAcc)
                plt.savefig('./debug/split_'+str(split)+'/'+model_name+'/loss_curve/Acc_batchwise.png')

                # eval
                if epoch:
                    with torch.no_grad():
                        val_bar = tqdm(data_loaders['test'])
                        y = []
                        pred = []
                        output = []
                        filename = []
                        label = []
                        batch_ind = 0
                        for inputs, labels, filenames in val_bar:
                            batch_ind += 1
                            model_ft = model_ft.eval()
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            if params.qloss:
                                outputs, _ = model_ft(inputs)
                            else:
                                outputs = model_ft(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            y = np.append(y, labels.cpu().numpy())
                            pred = np.append(pred, predicted.cpu().numpy())
                            for o in outputs.data.cpu().numpy():
                                output.append(o)
                            for f in filenames:
                                filename.append(f)
                            for l in labels.cpu().numpy():
                                label.append(l)
                        save_class_txts_top5(filepath, output, label, filename)
                        acc, precision, recall, f1, kappa = metricCLS(y, pred)

                        print("Test: %s EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%%, trainAcc = %.3f%%" % (model_name, epoch, 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss), np.mean(batch_acc)))
                        file.write("Test: EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%%, trainAcc = %.3f%%" % (epoch , 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss), np.mean(batch_acc)))

                        file.write('\n')
                        file.flush()

                        if acc > best_OA:
                            best_OA = acc
                            print("saving best model.....")
                            save = './debug/split_' + str(split) +'/'+model_name+'/saved_models/best.pth'
                            torch.save(model_ft.state_dict(), save)

                        if epoch % 10 == 0:
                            print("saving model.....")
                            save= './debug/split_' + str(split) +'/'+model_name+ '/saved_models/epoch_'+str(epoch)+'.pth'
                            torch.save(model_ft.state_dict(), save)
                fig3 = plt.figure()
                plot_ACC.append(acc)
                plt.plot(plot_ACC)
                plt.savefig('./debug/split_'+str(split)+'/'+model_name+'/loss_curve/ACC.png')
                fig4 = plt.figure()
                plot_Presision.append(precision)
                plt.plot(plot_Presision)
                plt.savefig('./debug/split_'+str(split)+'/'+model_name+'/loss_curve/Presision.png')
                fig5 = plt.figure()
                plot_Recall.append(recall)
                plt.plot(plot_Recall)
                plt.savefig('./debug/split_'+str(split)+'/'+model_name+'/loss_curve/Recall.png')
                fig6 = plt.figure()
                plot_F1.append(f1)
                plt.plot(plot_F1)
                plt.savefig('./debug/split_'+str(split)+'/'+model_name+'/loss_curve/F1.png')
                fig7 = plt.figure()
                plot_Kappa.append(kappa)
                plt.plot(plot_Kappa)
                plt.savefig('./debug/split_'+str(split)+'/'+model_name+'/loss_curve/Kappa.png')
    elif phase == 'test':
        if os.path.exists('./debug/split_' + str(split) + '/' + model_name + '/saved_models/best.pth'):
            print("loading best model-------------------------------------")
            model_ft.load_state_dict(torch.load('./debug/split_' + str(split) + '/' + model_name + '/saved_models/best.pth'))
        with torch.no_grad():
            val_bar = tqdm(data_loaders['test'])
            y = []
            pred = []
            output = []
            filename = []
            label = []
            batch_ind = 0
            for inputs, labels, filenames in val_bar:
                batch_ind += 1
                # if batch_ind == 1:
                #     break
                model_ft = model_ft.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)
                if params.qloss:
                    outputs, _ = model_ft(inputs)
                else:
                    outputs = model_ft(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y = np.append(y, labels.cpu().numpy())
                pred = np.append(pred, predicted.cpu().numpy())
                for o in outputs.data.cpu().numpy():
                    output.append(o)
                for f in filenames:
                    filename.append(f)
                for l in labels.cpu().numpy():
                    label.append(l)
            save_class_txts_top5(filepath, output, label, filename)
            acc, precision, recall, f1, kappa = metricCLS(y, pred)

            print("Test: %s | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%" % (model_name,  100. * acc, 100. * precision, 100. * recall, 100. * f1, 100. * kappa))

def process_networks_qloss(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that I force to use cuda here
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))
    model_ft = clsNet(classes_num=params.num_class, basic_model=params.net_model, pretrain=params.pretrained, need_gamma=params.qloss,k_c=params.k_c,k_f=params.k_f,J=params.J,M=params.M)

    EPOCH = params.EPOCH
    phase = params.phase
    split = params.split_no
    use_qloss = params.qloss

    print(model_ft)
    macs, parameters = get_model_complexity_info(model_ft, (6, 224, 224), as_strings=True, print_per_layer_stat=True,
                                                 verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', parameters))
    # Set model to evaluation mode (without this, results will be completely different)
    # Remember that you must call model.eval() to set dropout and batch normalization layers
    # to evaluation mode before running inference.
    if params.gpu:
        model_ft = nn.DataParallel(model_ft).cuda()
    model_ft = model_ft.to(device)

    if params.ablation:
        model_name = params.net_model + '_' + str(params.k_c) + '_' + str(params.k_f) + '_' + str(params.M) + '_' + str(params.J)
    else:
        model_name = params.net_model

    os.makedirs("./debug/split_" + str(split) + "/qloss_" + model_name + "/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/qloss_" + model_name + "/saved_models/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/qloss_" + model_name + "/test_metrics/", exist_ok=True)
    txtfile = "./debug/split_" + str(split) + "/qloss_" + model_name + "/test_metrics/results.txt"
    os.makedirs("./debug/split_" + str(split) + "/qloss_" + model_name + "/loss_curve/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/qloss_" + model_name + "/saved_top5/", exist_ok=True)
    filepath = "./debug/split_" + str(split) + "/qloss_" + model_name + "/saved_top5/"

    print("loading dataset-------------------------------------")
    data_form = get_data_transform(params.data_type)
    # print('-------------train_dataset------------------------')
    training_set = WashingtonDataset(params, phase='train', loader=custom_loader, transform=data_form)
    train_loader = torch.utils.data.DataLoader(training_set, params.batch_size, shuffle=True)
    # print('-------------test_dataset------------------------')
    test_set = WashingtonDataset(params, phase='test', loader=custom_loader, transform=data_form)
    test_loader = torch.utils.data.DataLoader(test_set, params.batch_size, shuffle=False)

    data_loaders = {'train': train_loader, 'test': test_loader}
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=params.lr)
    KL_criterion = nn.KLDivLoss(reduction='mean').cuda()
    CE_criterion = nn.CrossEntropyLoss().cuda()
    best_OA = 0 # 0.77492

    if phase == 'train':
        if os.path.exists('./debug/split_' + str(split) + '/qloss_' + model_name + '/saved_models/best.pth'):
            print("loading best model-------------------------------------")
            model_ft.load_state_dict(torch.load('./debug/split_' + str(split) + '/qloss_' + model_name + '/saved_models/best.pth'))
        with open(txtfile, "a+") as file:
            plot_loss = []
            plot_loss1 = []
            plot_loss2 = []
            plot_batchAcc = []
            plot_ACC = []
            plot_Presision = []
            plot_Recall = []
            plot_F1 = []
            plot_Kappa = []
            for epoch in range(1,EPOCH+1):
                train_bar = tqdm(data_loaders[phase])
                if epoch % 100 == 0:
                    new_lr = params.lr / (2*(epoch//100))
                    for para_group in optimizer.param_groups:
                        para_group['lr'] = new_lr
                    print("Learning weight decays to %f"%(new_lr))
                epoch_loss = []
                epoch_loss1 = []
                epoch_loss2 = []
                batch_acc = []
                batch_ind = 0
                for inputs, q_label, labels, filenames in train_bar:
                    batch_ind += 1
                    model_ft = model_ft.train()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    q_label = q_label.to(device)
                    optimizer.zero_grad()
                    # freeze
                    # if epoch % 100 <= 50:
                    #     get_freeze_encoder(model_ft, params)
                    # else:
                    #     get_unfreeze_encoder(model_ft, params)
                    if params.qloss:
                        outputs, q_out = model_ft(inputs)
                    else:
                        outputs = model_ft(inputs)
                    if params.cu:
                        loss1 = 0
                        for j in range(len(q_out)):
                            loss1 = loss1 + KL_criterion(F.log_softmax(q_out[j]), q_label.float())
                    else:
                        loss1 = KL_criterion(F.log_softmax(q_out[-1]), q_label.float())
                    loss2 = CE_criterion(outputs, labels)
                    loss = loss1 + loss2
                    _, predicted = torch.max(outputs.data, 1)

                    loss.backward()
                    optimizer.step()
                    total = labels.size(0)
                    correct = predicted.eq(labels.data).cpu().sum()

                    epoch_loss.append(loss.item())
                    epoch_loss1.append(loss1.item())
                    epoch_loss2.append(loss2.item())
                    batch_acc.append(100.*correct/total)
                    train_bar.set_description(
                        desc=model_name + ' [%d/%d] loss=q_loss+h_loss: %.4f=%.4f+%.4f  | batch_acc: %.4f' % (
                            epoch, EPOCH,
                            loss.item(),loss1.item(),loss2.item(),
                            100.*correct/total,
                        ))
                save = './debug/split_'+str(split)+'/qloss_'+model_name+ '/saved_models/latest.pth'
                torch.save(model_ft.state_dict(), save)



                plot_loss.append(np.mean(epoch_loss))
                plot_loss1.append(np.mean(epoch_loss1))
                plot_loss2.append(np.mean(epoch_loss2))
                fig1 = plt.figure()
                plt.plot(plot_loss)
                plt.savefig('./debug/split_'+str(split)+'/qloss_'+model_name+'/loss_curve/loss.png')
                fig2 = plt.figure()
                plt.plot(plot_loss1)
                plt.savefig('./debug/split_' + str(split) + '/qloss_' + model_name + '/loss_curve/q_loss.png')
                fig3 = plt.figure()
                plt.plot(plot_loss2)
                plt.savefig('./debug/split_' + str(split) + '/qloss_' + model_name + '/loss_curve/h_loss.png')
                fig4 = plt.figure()
                plot_batchAcc.append(np.mean(batch_acc))
                plt.plot(plot_batchAcc)
                plt.savefig('./debug/split_'+str(split)+'/qloss_'+model_name+'/loss_curve/Acc_batchwise.png')

                # eval
                if epoch:
                    with torch.no_grad():
                        val_bar = tqdm(data_loaders['test'])
                        y = []
                        pred = []
                        output = []
                        filename = []
                        label = []
                        batch_ind = 0
                        for inputs, q_label, labels, filenames in val_bar:
                            batch_ind += 1
                            model_ft = model_ft.eval()
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            q_label = q_label.to(device)
                            if params.qloss:
                                outputs, q_out = model_ft(inputs)
                            else:
                                outputs = model_ft(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            y = np.append(y, labels.cpu().numpy())
                            pred = np.append(pred, predicted.cpu().numpy())
                            for o in outputs.data.cpu().numpy():
                                output.append(o)
                            for f in filenames:
                                filename.append(f)
                            for l in labels.cpu().numpy():
                                label.append(l)
                        save_class_txts_top5(filepath, output, label, filename)
                        acc, precision, recall, f1, kappa = metricCLS(y, pred)

                        print("Test: %s EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%% + %.6f%% = %.6f%%, trainAcc = %.3f%%" % (model_name, epoch, 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss),np.mean(epoch_loss1),np.mean(epoch_loss2), np.mean(batch_acc)))
                        file.write("Test: EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%% + %.6f%% = %.6f%%, trainAcc = %.3f%%" % (epoch , 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss),np.mean(epoch_loss1),np.mean(epoch_loss2), np.mean(batch_acc)))

                        file.write('\n')
                        file.flush()

                        if acc > best_OA:
                            best_OA = acc
                            print("saving best model.....")
                            save = './debug/split_' + str(split) +'/qloss_'+model_name+'/saved_models/best.pth'
                            torch.save(model_ft.state_dict(), save)

                        if epoch % 10 == 0:
                            print("saving model.....")
                            save= './debug/split_' + str(split) +'/qloss_'+model_name+ '/saved_models/epoch_'+str(epoch)+'.pth'
                            torch.save(model_ft.state_dict(), save)
                fig3 = plt.figure()
                plot_ACC.append(acc)
                plt.plot(plot_ACC)
                plt.savefig('./debug/split_'+str(split)+'/qloss_'+model_name+'/loss_curve/ACC.png')
                fig4 = plt.figure()
                plot_Presision.append(precision)
                plt.plot(plot_Presision)
                plt.savefig('./debug/split_'+str(split)+'/qloss_'+model_name+'/loss_curve/Presision.png')
                fig5 = plt.figure()
                plot_Recall.append(recall)
                plt.plot(plot_Recall)
                plt.savefig('./debug/split_'+str(split)+'/qloss_'+model_name+'/loss_curve/Recall.png')
                fig6 = plt.figure()
                plot_F1.append(f1)
                plt.plot(plot_F1)
                plt.savefig('./debug/split_'+str(split)+'/qloss_'+model_name+'/loss_curve/F1.png')
                fig7 = plt.figure()
                plot_Kappa.append(kappa)
                plt.plot(plot_Kappa)
                plt.savefig('./debug/split_'+str(split)+'/qloss_'+model_name+'/loss_curve/Kappa.png')
    elif phase == 'test':
        txtfile = "./debug/split_" + str(split) + "/qloss_" + model_name + "/test_metrics/"
        if os.path.exists('./debug/split_' + str(split) + '/qloss_' + model_name + '/saved_models/best.pth'):
            print("loading best model-------------------------------------")
            model_ft.load_state_dict(torch.load('./debug/split_' + str(split) + '/qloss_' + model_name + '/saved_models/best.pth'))
        with torch.no_grad():
            val_bar = tqdm(data_loaders['test'])
            y = []
            pred = []
            output = []
            output_q = []
            filename = []
            label = []
            batch_ind = 0
            avg_L0_channel = {}
            avg_L1_channel = {}
            avg_L0_image = {}
            avg_L1_image = {}
            name = ['X','Y','U1', 'V1', 'W2', 'W3', 'W4']#X,Y,U1,V1,W2,W3,...,WN
            for i in range(len(name)):
                for m in range(len(class_names)):
                    avg_L0_image[name[i] + '_' + list(class_names)[m]] = []
                    avg_L1_image[name[i] + '_' + list(class_names)[m]] = []
                    avg_L0_channel[name[i] + '_' + list(class_names)[m]] = []
                    avg_L1_channel[name[i] + '_' + list(class_names)[m]] = []
            for inputs, q_label, labels, filenames in val_bar:
                batch_ind += 1
                # if batch_ind == 10:
                #     break
                model_ft = model_ft.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)
                q_label = q_label.to(device)
                if params.qloss:
                    outputs, q_out = model_ft(inputs)
                else:
                    outputs = model_ft(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y = np.append(y, labels.cpu().numpy())
                pred = np.append(pred, predicted.cpu().numpy())
                for o in outputs.data.cpu().numpy():
                    output.append(o)
                for f in filenames:
                    filename.append(f)
                for l in labels.cpu().numpy():
                    label.append(l)

                #save_interpretable_intermidiate_features
                # with open(txtfile, "a+") as file:
                if params.interpret:
                    c = inputs.shape[1]
                    X, Y = inputs[:, :3, :, :], inputs[:, 3:c, :, :]
                    a = X.shape[0]
                    b = X.shape[1]
                    for m in range(a):# single image from batch
                        
                        X_m, Y_m = X[m:m+1,:,:,:], Y[m:m+1,:,:,:]
                        class_name = filenames[m].split('_')[0]
                        if filenames[m].split('_')[0] in class_names:
                            class_name = filenames[m].split('_')[0]
                        else:
                            for class_name_i in class_names:
                                if filenames[m].split('_')[0] in class_name_i:
                                    if filenames[m].split('_')[1] in class_name_i:
                                        class_name = class_name_i
                        L0_channel = []
                        L1_channel = []
                        L0_image = []
                        L1_image = []
                        for j in range(b):# each channel
                            X_m_j = X_m[:, j:j+1, :, :]
                            X_m_j = process_data(X_m_j)
                            l1_per_channel = cal_l1(X_m_j)
                            l0_per_channel = cal_l0(X_m_j)
                            L0_image.append(round(l0_per_channel,4))
                            L1_image.append(round(l1_per_channel,4))
                            L0_channel.append(str(round(l0_per_channel,4)))
                            L1_channel.append(str(round(l1_per_channel,4)))
                        avg_L0_channel[name[0] + '_' + class_name].append(L0_image)
                        avg_L1_channel[name[0] + '_' + class_name].append(L1_image)
                        avg_L0_image[name[0] + '_' + class_name].append(np.mean(L0_image))
                        avg_L1_image[name[0] + '_' + class_name].append(np.mean(L0_image))
                        L0_channel = []
                        L1_channel = []
                        L0_image = []
                        L1_image = []
                        for j in range(b):# each channel
                            Y_m_j = Y_m[:, j:j+1, :, :]
                            Y_m_j = process_data(Y_m_j)
                            l1_per_channel = cal_l1(Y_m_j)
                            l0_per_channel = cal_l0(Y_m_j)
                            L0_image.append(round(l0_per_channel,4))
                            L1_image.append(round(l1_per_channel,4))
                            L0_channel.append(str(round(l0_per_channel,4)))
                            L1_channel.append(str(round(l1_per_channel,4)))
                        avg_L0_channel[name[1] + '_' + class_name].append(L0_image)
                        avg_L1_channel[name[1] + '_' + class_name].append(L1_image)
                        avg_L0_image[name[1] + '_' + class_name].append(np.mean(L0_image))
                        avg_L1_image[name[1] + '_' + class_name].append(np.mean(L0_image))
                        # for i in range(2):
                        #     txtfile_l0_channel = txtfile + 'channel_l0_' + name[i] + '.txt'
                        #     txtfile_l1_channel = txtfile + 'channel_l1_' + name[i] + '.txt'
                        #     with open(txtfile_l0_channel, "a+") as file_l0:
                        #         file_l0.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], l0_write_channel))
                        #         file_l0.write('\n')
                        #         file_l0.flush()
                        #     with open(txtfile_l1_channel, "a+") as file_l1:
                        #         file_l1.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], l1_write_channel))
                        #         file_l1.write('\n')
                        #         file_l1.flush()
                        #     l0_write_image = np.mean(L0_image)
                        #     l1_write_image = np.mean(L1_image)
                        #     txtfile_l0 = txtfile + 'image_l0_' + name[i] + '.txt'
                        #     txtfile_l1 = txtfile + 'image_l1_' + name[i] + '.txt'
                        #     with open(txtfile_l0, "a+") as file_l0:
                        #         file_l0.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], str(l0_write_image)))
                        #         file_l0.write('\n')
                        #         file_l0.flush()
                        #     with open(txtfile_l1, "a+") as file_l1:
                        #         file_l1.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], str(l1_write_image)))
                        #         file_l1.write('\n')
                        #         file_l1.flush()

                    num = len(q_out)
                    for i in range(num):# type of intermidiate feature
                        q_out_i = q_out[i]
                        a = q_out_i.shape[0]
                        b = q_out_i.shape[1]

                        for m in range(a):# single image from batch
                            q_out_i_m = q_out_i[m:m+1,:,:,:]
                            class_name = filenames[m].split('_')[0]
                            if filenames[m].split('_')[0] in class_names:
                                class_name = filenames[m].split('_')[0]
                            else:
                                for class_name_i in class_names:
                                    if filenames[m].split('_')[0] in class_name_i:
                                        if filenames[m].split('_')[1] in class_name_i:
                                            class_name = class_name_i
                            os.makedirs('./debug/split_'+str(split)+'/qloss_'+model_name+'/save_interpret/'+class_name+'/'+filenames[m][:-4]+'/'+name[i+2]+'/', exist_ok=True)
                            L0_channel = []
                            L1_channel = []
                            L0_image = []
                            L1_image = []
                            for j in range(b):# each channel
                                q_out_i_m_j = q_out_i_m[:, j:j+1, :, :]
                                if i == num-1:
                                    q_out_i_m_j[q_out_i_m_j < 0] = 0
                                q_out_i_m_j = process_data(q_out_i_m_j)
                                imsave(q_out_i_m_j, './debug/split_'+str(split)+'/qloss_'+model_name+'/save_interpret/'+class_name+'/'+filenames[m][:-4]+'/'+name[i+2]+'/' + str(j) + '.png')
                                l1_per_channel = cal_l1(q_out_i_m_j)
                                l0_per_channel = cal_l0(q_out_i_m_j)
                                L0_image.append(round(l0_per_channel,4))
                                L1_image.append(round(l1_per_channel,4))
                                L0_channel.append(str(round(l0_per_channel,4)))
                                L1_channel.append(str(round(l1_per_channel,4)))

                            avg_L0_channel[name[i+2] + '_' + class_name].append(L0_image)
                            avg_L1_channel[name[i+2] + '_' + class_name].append(L1_image)


                            avg_L0_image[name[i+2] + '_' + class_name].append(np.mean(L0_image))
                            avg_L1_image[name[i+2] + '_' + class_name].append(np.mean(L0_image))

                            # l1_write_channel = ','.join(L1_channel)
                            # l0_write_channel = ','.join(L0_channel)
                            # txtfile_l0_channel = txtfile + 'channel_l0_' + name[i+2] + '.txt'
                            # txtfile_l1_channel = txtfile + 'channel_l1_' + name[i+2] + '.txt'
                            # with open(txtfile_l0_channel, "a+") as file_l0:
                            #     file_l0.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], l0_write_channel))
                            #     file_l0.write('\n')
                            #     file_l0.flush()
                            # with open(txtfile_l1_channel, "a+") as file_l1:
                            #     file_l1.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], l1_write_channel))
                            #     file_l1.write('\n')
                            #     file_l1.flush()
                            # l0_write_image = np.mean(L0_image)
                            # l1_write_image = np.mean(L1_image)
                            # txtfile_l0 = txtfile + 'image_l0_' + name[i+2] + '.txt'
                            # txtfile_l1 = txtfile + 'image_l1_' + name[i+2] + '.txt'
                            # with open(txtfile_l0, "a+") as file_l0:
                            #     file_l0.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], str(l0_write_image)))
                            #     file_l0.write('\n')
                            #     file_l0.flush()
                            # with open(txtfile_l1, "a+") as file_l1:
                            #     file_l1.write("Class=%s | Instance=%s | %s" % (class_name, filenames[m][:-4], str(l1_write_image)))
                            #     file_l1.write('\n')
                            #     file_l1.flush()
            for i in range(num+2):
                avg_txtfile_l0 = txtfile + 'avg_image_l0_' + name[i] + '.txt'
                avg_txtfile_l1 = txtfile + 'avg_image_l1_' + name[i] + '.txt'
                avg_txtfile_l0_channel = txtfile + 'avg_channel_l0_' + name[i] + '.txt'
                avg_txtfile_l1_channel = txtfile + 'avg_channel_l1_' + name[i] + '.txt'
                name_type = name[i]
                for m in range(len(class_names)):
                    name_class = list(class_names)[m]
                    name_key = name_type + '_' + name_class
                    with open(avg_txtfile_l0, "a+") as file_l0:
                        file_l0.write("Class=%s | %s" % (
                        name_class, str(np.mean(avg_L0_image[name_key]))))
                        file_l0.write('\n')
                        file_l0.flush()
                    with open(avg_txtfile_l1, "a+") as file_l1:
                        file_l1.write("Class=%s | %s" % (
                        name_class, str(np.mean(avg_L1_image[name_key]))))
                        file_l1.write('\n')
                        file_l1.flush()
                    a = np.array(avg_L0_channel[name_key])
                    b = np.array(avg_L1_channel[name_key])
                    # print(b)
                    # print(np.mean(b,axis=0))
                    # print(np.mean(b,axis=0).tolist())
                    # print(np.mean(b,axis=0).shape)
                    with open(avg_txtfile_l0_channel, "a+") as file_l0:
                        file_l0.write("Class=%s | %s" % (
                        name_class, str(np.around(np.mean(a,axis=0),3).tolist())))
                        file_l0.write('\n')
                        file_l0.flush()
                    with open(avg_txtfile_l1_channel, "a+") as file_l1:
                        file_l1.write("Class=%s | %s" % (
                        name_class, str(np.around(np.mean(b,axis=0),3).tolist())))
                        file_l1.write('\n')
                        file_l1.flush()

            save_class_txts_top5(filepath, output, label, filename)
            acc, precision, recall, f1, kappa = metricCLS(y, pred)
            print("Test: %s | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%" % (model_name, 100. * acc, 100. * precision, 100. * recall, 100. * f1, 100. * kappa))




@basic_utils.profile
def run_overall_steps(params):

    if params.net_model:
        if not params.qloss:
            process_networks(params)
        else:
            process_networks_qloss(params)

