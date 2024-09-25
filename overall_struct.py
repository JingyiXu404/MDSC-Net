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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
from IPython import embed
import torch.nn.functional as F
import time
import matplotlib.ticker as ticker
matplotlib.use('agg')
warnings.filterwarnings(action='ignore')
import utils.freeze as freeze
from ptflops import get_model_complexity_info
def get_filter(filters, name, layer,down):
    filters = np.where(filters > np.quantile(filters, 0.75), 0, filters)
    filters = np.where(filters < np.quantile(filters, 0.25), 0, filters)

    filters = np.transpose(filters, (1, 2, 3, 0))
    if not down:
        filters = np.transpose(filters, (3, 1, 2, 0))
    n_filters,h,w,channel = filters.shape #32,7,7,3
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    if layer == 0:
        h1 = 8
        w1 = 4
        return
    elif layer == 1:
        h1 = 8
        w1 = 8
        return
    elif layer == 2:
        h1 = 17
        w1 = 6
    elif layer == 3:
        h1 = 17
        w1 = 12
    else:
        h1 = 17
        w1 = 18
    img = np.zeros((int(h * h1 + h1-1), int(w * w1 + w1-1)))
    # print(layer,down,filters.shape)
    for channel_i in range(channel):
        for idx in range(n_filters):
            i = idx % w1 * (h + 1)
            j = idx // w1 * (w + 1)
            print(idx,j,i,filters[idx, :, :, channel_i].shape,img[j:j + h, i:i + w].shape)
            img[j:j + h, i:i + w] = filters[idx, :, :, channel_i]
        img = np.absolute(img)
        plt.imshow(img, cmap='Greys',vmin=img.min(),vmax=img.max())
        title = name + '_channel' + str(channel_i)
        plt.savefig(title + ".png", dpi=200, bbox_inches='tight',cmap='Greys')
        # plt.imsave(title + ".png",img,cmap='Greys',vmin=img.min(),vmax=img.max())
def plot_confusion_matrix(cm, savepath):
    plt.figure()#figsize=(80, 80), dpi=200
    np.set_printoptions(precision=2)  # 输出小数点的个数
    classes = list(class_names)
    classes.sort()
    ind_array = np.arange(len(classes))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1)
    plt.title('Confusion matrix', fontsize=12, pad=13, weight='bold', color= 'black')

    cb = plt.colorbar(fraction=0.03, pad=0.001)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.ax.tick_params(labelsize=10)
    cb.locator = tick_locator
    cb.update_ticks()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, size=6, rotation=90, weight='bold')
    plt.yticks(xlocations, classes, size=6,  weight='bold')
    plt.ylabel('Ground Truth label', fontsize=12, labelpad=6, weight='bold')
    plt.xlabel('Predicted label', fontsize=12, labelpad=6, weight='bold')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(savepath, dpi=1000, bbox_inches='tight')
def tensor2uint(img: torch.Tensor) -> np.ndarray:
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    return img
def process_data(img):
    img = img.detach().float().cpu().data.squeeze().float()
    return img
def imsave(img, img_path):
    img_save = img.clone()
    img_save = np.squeeze(img_save.clamp_(0, 1).cpu().numpy())
    img_save = np.uint8((img_save * 255.0).round())
    cv2.imwrite(img_path, img_save)
def cal_l1(input):
    SUM = np.sum(np.absolute(input.numpy()))
    SUM =  round(SUM,2)
    return SUM
def cal_l1_mean(input):
    SUM = np.mean(np.absolute(input.numpy()))
    SUM =  round(SUM,2)
    return SUM
def cal_l0(input):
    SUM = np.count_nonzero(input.numpy())
    return SUM
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
def get_freeze_encoder_coarse(model,params):
    if params.gpu:
        freeze.freeze_by_names(model.module,'MMCSC')
        freeze.unfreeze_by_names(model.module.MMCSC, 'encoder.coarse_encoder')
    else:
        freeze.freeze_by_names(model.modules.module,'MMCSC')
        freeze.unfreeze_by_names(model.modules.module.MMCSC, 'encoder.coarse_encoder')

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
def interpret_test(q_out,avg_L0_channel,avg_L1_channel,avg_L0_image,avg_L1_image,name,filenames,params):
    num = len(q_out)
    for i in range(num):  # type of intermidiate feature
        q_out_i = q_out[i]
        a = q_out_i.shape[0]
        b = q_out_i.shape[1]

        for m in range(a):  # single image from batch
            q_out_i_m = q_out_i[m:m + 1, :, :, :]
            if filenames[m].split('_')[0] in class_names:
                class_name = filenames[m].split('_')[0]
            else:
                for class_name_i in class_names:
                    if filenames[m].split('_')[0] in class_name_i:
                        if filenames[m].split('_')[1] in class_name_i:
                            class_name = class_name_i
            os.makedirs('./debug/split_'+str(params.split_no)+'/qloss_'+params.net_model+'/save_interpret/'+class_name+'/'+filenames[m][:-4]+'/'+name + str(i)+'/', exist_ok=True)

            if i == num - 1:
                L0_channel = []
                L1_channel = []
                for j in range(b):  # each channel
                    q_out_i_m_j = q_out_i_m[:, j:j + 1, :, :]
                    q_out_i_m_j[q_out_i_m_j < 0] = 0
                    q_out_i_m_j = process_data(q_out_i_m_j)
                    # save features for each channel
                    imsave(q_out_i_m_j, './debug/split_' + str(params.split_no) + '/qloss_' + params.net_model + '/save_interpret/' + class_name + '/' + filenames[m][:-4] + '/' + name + str(i) + '/' + str(j) + '.png')
                    l0_per_channel = round(cal_l0(q_out_i_m_j), 4)
                    l1_per_channel = round(cal_l1(q_out_i_m_j), 4)
                    L0_channel.append(l0_per_channel)
                    L1_channel.append(l1_per_channel)
                # save l0 and l1 norm for each channel
                avg_L0_channel[name + str(i) + '_' + class_name].append(L0_channel)
                avg_L1_channel[name + str(i) + '_' + class_name].append(L1_channel)
                # save l0 and l1 norm for each image
                avg_L0_image[name + str(i) + '_' + class_name].append(round(cal_l0(process_data(q_out_i_m)), 4))
                avg_L1_image[name + str(i) + '_' + class_name].append(round(cal_l1(process_data(q_out_i_m)), 4))
            else:
                for j in range(b):  # each channel
                    q_out_i_m_j = q_out_i_m[:, j:j + 1, :, :]
                    q_out_i_m_j[q_out_i_m_j < 0] = 0
                    q_out_i_m_j = process_data(q_out_i_m_j)
                    # save features for each channel
                    imsave(q_out_i_m_j, './debug/split_' + str(params.split_no) + '/qloss_' + params.net_model + '/save_interpret/' + class_name + '/' + filenames[m][:-4] + '/' + name + str(i) + '/' + str(j) + '.png')
                # save l0 and l1 norm for each image
                avg_L0_image[name + str(i) + '_' + class_name].append(round(cal_l0(process_data(q_out_i_m)), 4))
                avg_L1_image[name + str(i) + '_' + class_name].append(round(cal_l1(process_data(q_out_i_m)), 4))
    return avg_L0_channel,avg_L1_channel,avg_L0_image,avg_L1_image
def process_networks(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that I force to use cuda here
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))
    model_ft = clsNet(classes_num=params.num_class, basic_model=params.net_model, pretrain=params.pretrained, need_gamma=params.qloss)

    EPOCH = params.EPOCH
    phase = params.phase
    split = params.split_no

    if params.gpu:
        model_ft = nn.DataParallel(model_ft).cuda()
    model_ft = model_ft.to(device)

    os.makedirs("./debug/split_" + str(split) + "/" + params.net_model + "/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/" + params.net_model + "/saved_models/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/" + params.net_model + "/test_metrics/", exist_ok=True)
    txtfile = "./debug/split_" + str(split) + "/" + params.net_model + "/test_metrics/results.txt"
    os.makedirs("./debug/split_" + str(split) + "/" + params.net_model + "/loss_curve/", exist_ok=True)
    os.makedirs("./debug/split_" + str(split) + "/" + params.net_model + "/saved_top5/", exist_ok=True)
    filepath = "./debug/split_" + str(split) + "/" + params.net_model + "/saved_top5/"

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
    best_OA = 0  # 0

    if phase == 'train':
        if os.path.exists('./debug/split_' + str(split) + '/' + params.net_model + '/saved_models/latest.pth'):
            print("loading latest model-------------------------------------")
            model_ft.load_state_dict(torch.load('./debug/split_' + str(split) + '/' + params.net_model + '/saved_models/latest.pth'))
        with open(txtfile, "a+") as file:
            plot_loss = []
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
                batch_acc = []
                batch_ind = 0
                for inputs, labels, filenames in train_bar:
                    batch_ind += 1
                    model_ft = model_ft.train()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    # freeze
                    # if epoch%100<50:
                    #     get_freeze_encoder(model_ft,params)
                    # else:
                    #     get_unfreeze_encoder(model_ft, params)
                    if params.qloss:
                        outputs, _ = model_ft(inputs)
                    else:
                        outputs = model_ft(inputs)
                    loss = CE_criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)

                    loss.backward()
                    optimizer.step()
                    total = labels.size(0)
                    correct = predicted.eq(labels.data).cpu().sum()

                    epoch_loss.append(loss.item())
                    batch_acc.append(100.*correct/total)
                    train_bar.set_description(
                        desc=params.net_model + ' [%d/%d] ce_loss: %.4f  | batch_acc: %.4f' % (
                            epoch, EPOCH,
                            CE_criterion(outputs, labels).item(),
                            100.*correct/total,
                        ))
                save = './debug/split_'+str(split)+'/'+params.net_model+ '/saved_models/latest.pth'
                torch.save(model_ft.state_dict(), save)


                fig1 = plt.figure()
                plot_loss.append(np.mean(epoch_loss))
                plt.plot(plot_loss)
                plt.savefig('./debug/split_'+str(split)+'/'+params.net_model+'/loss_curve/CE_loss.png')
                fig2 = plt.figure()
                plot_batchAcc.append(np.mean(batch_acc))
                plt.plot(plot_batchAcc)
                plt.savefig('./debug/split_'+str(split)+'/'+params.net_model+'/loss_curve/Acc_batchwise.png')

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

                        print("Test: %s EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%%, trainAcc = %.3f%%" % (params.net_model, epoch, 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss), np.mean(batch_acc)))
                        file.write("Test: EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%%, trainAcc = %.3f%%" % (epoch , 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss), np.mean(batch_acc)))

                        file.write('\n')
                        file.flush()

                        if acc > best_OA:
                            best_OA = acc
                            print("saving best model.....")
                            save = './debug/split_' + str(split) +'/'+params.net_model+'/saved_models/best.pth'
                            torch.save(model_ft.state_dict(), save)

                        if epoch % 10 == 0:
                            print("saving model.....")
                            save= './debug/split_' + str(split) +'/'+params.net_model+ '/saved_models/epoch_'+str(epoch)+'.pth'
                            torch.save(model_ft.state_dict(), save)
                fig3 = plt.figure()
                plot_ACC.append(acc)
                plt.plot(plot_ACC)
                plt.savefig('./debug/split_'+str(split)+'/'+params.net_model+'/loss_curve/ACC.png')
                fig4 = plt.figure()
                plot_Presision.append(precision)
                plt.plot(plot_Presision)
                plt.savefig('./debug/split_'+str(split)+'/'+params.net_model+'/loss_curve/Presision.png')
                fig5 = plt.figure()
                plot_Recall.append(recall)
                plt.plot(plot_Recall)
                plt.savefig('./debug/split_'+str(split)+'/'+params.net_model+'/loss_curve/Recall.png')
                fig6 = plt.figure()
                plot_F1.append(f1)
                plt.plot(plot_F1)
                plt.savefig('./debug/split_'+str(split)+'/'+params.net_model+'/loss_curve/F1.png')
                fig7 = plt.figure()
                plot_Kappa.append(kappa)
                plt.plot(plot_Kappa)
                plt.savefig('./debug/split_'+str(split)+'/'+params.net_model+'/loss_curve/Kappa.png')
    elif phase == 'test':
        if os.path.exists('./debug/split_' + str(split) + '/' + params.net_model + '/saved_models/best.pth'):
            print("loading best model-------------------------------------")
            model_ft.load_state_dict(torch.load('./debug/split_' + str(split) + '/' + params.net_model + '/saved_models/best.pth'))
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
                # if batch_ind == 5:
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

            print("Test: %s | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%" % (params.net_model,  100. * acc, 100. * precision, 100. * recall, 100. * f1, 100. * kappa))

def process_networks_qloss(params):
    # "cuda" if torch.cuda.is_available() else "cpu" instead of that I force to use cuda here
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))
    model_ft = clsNet(classes_num=params.num_class, basic_model=params.net_model, pretrain=params.pretrained, need_gamma=params.qloss, interpret=params.interpret,k_c=params.k_c,k_f=params.k_f,J=params.J,M=params.M)

    EPOCH = params.EPOCH
    phase = params.phase
    split = params.split_no
    use_qloss = params.qloss

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
    best_OA = 0.74748 # 0.77492

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
                    new_lr = params.lr / (2*(1+epoch//100))
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
                        loss1 = KL_criterion(F.log_softmax(q_out[-1]), q_label.float())
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

                        print("Test: %s EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%% + %.6f%% = %.6f%%, trainAcc = %.3f%%" % (model_name, epoch, 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss1),np.mean(epoch_loss2),np.mean(epoch_loss), np.mean(batch_acc)))
                        file.write("Test: EPOCH=%03d | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%, trainloss = %.6f%% + %.6f%% = %.6f%%, trainAcc = %.3f%%" % (epoch , 100.*acc, 100.*precision, 100.*recall, 100.*f1, 100.*kappa, np.mean(epoch_loss1),np.mean(epoch_loss2), np.mean(epoch_loss),np.mean(batch_acc)))

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
            model_ft.load_state_dict(
                torch.load('./debug/split_' + str(split) + '/qloss_' + model_name + '/saved_models/best.pth'))
        if params.dict:
            model_ft = model_ft.eval()
            save_file = "./debug/split_" + str(split) + "/qloss_" + model_name + "/save_dict/"
            os.makedirs(os.path.join(save_file, 'down_conv_x'), exist_ok=True)
            os.makedirs(os.path.join(save_file, 'down_conv_y'), exist_ok=True)
            os.makedirs(os.path.join(save_file, 'up_conv_x'), exist_ok=True)
            os.makedirs(os.path.join(save_file, 'up_conv_y'), exist_ok=True)
            for layer in range(3):
                os.makedirs(os.path.join(save_file, 'down_conv_f', 'layer' + str(layer)), exist_ok=True)
                os.makedirs(os.path.join(save_file, 'up_conv_f', 'layer' + str(layer)), exist_ok=True)
            for ord in range(4):
                down_conv_x = model_ft.MMCSC.encoder.net1_x.layer_down[ord].weight.data.cpu().numpy()
                down_conv_y = model_ft.MMCSC.encoder.net1_y.layer_down[ord].weight.data.cpu().numpy()
                up_conv_x = model_ft.MMCSC.encoder.net1_x.layer_up[ord].weight.data.cpu().numpy()
                up_conv_y = model_ft.MMCSC.encoder.net1_y.layer_up[ord].weight.data.cpu().numpy()
                get_filter(down_conv_x, os.path.join(save_file, 'down_conv_x') + '/iter' + str(ord), layer=0, down=True)
                get_filter(down_conv_y, os.path.join(save_file, 'down_conv_y') + '/iter' + str(ord), layer=0, down=True)
                get_filter(up_conv_x, os.path.join(save_file, 'up_conv_x') + '/iter' + str(ord), layer=0, down=False)
                get_filter(up_conv_y, os.path.join(save_file, 'up_conv_y') + '/iter' + str(ord), layer=0, down=False)
                for layer in range(3):
                    down_conv_f = model_ft.MMCSC.encoder.net_fuse[layer].layer_down[ord].weight.data.cpu().numpy()
                    up_conv_f = model_ft.MMCSC.encoder.net_fuse[layer].layer_up[ord].weight.data.cpu().numpy()
                    get_filter(down_conv_f,
                               os.path.join(save_file, 'down_conv_f', 'layer' + str(layer)) + '/iter' + str(ord),
                               layer=layer + 1, down=True)
                    get_filter(up_conv_f,
                               os.path.join(save_file, 'up_conv_f', 'layer' + str(layer)) + '/iter' + str(ord),
                               layer=layer + 1, down=True)

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
            total_time = []
            coarse_time = []
            fine_time1 = []
            fine_time2 = []
            fine_time3 = []
            fc_time = []
            name = ['X', 'Y', 'U0', 'V0', 'C0', 'U1', 'V1', 'C1', 'U2', 'V2', 'C2', 'U3', 'V3', 'C3']
            for i in range(len(name)):
                for m in range(len(class_names)):
                    avg_L0_image[name[i] + '_' + list(class_names)[m]] = []
                    avg_L1_image[name[i] + '_' + list(class_names)[m]] = []
                    avg_L0_channel[name[i] + '_' + list(class_names)[m]] = []
                    avg_L1_channel[name[i] + '_' + list(class_names)[m]] = []
            for inputs, q_label, labels, filenames in val_bar:
                batch_ind += 1
                # if batch_ind == 5:
                #     break
                model_ft = model_ft.eval()
                inputs = inputs.to(device)
                labels = labels.to(device)
                q_label = q_label.to(device)
                if not params.interpret:
                    if params.qloss:
                        outputs, q_out = model_ft(inputs)
                    else:
                        outputs = model_ft(inputs)
                else:
                    outputs, q_out_C, q_out_U, q_out_V = model_ft(inputs)
                _, predicted = torch.max(outputs.data, 1)

                y = np.append(y, labels.cpu().numpy())
                pred = np.append(pred, predicted.cpu().numpy())
                for o in outputs.data.cpu().numpy():
                    output.append(o)
                for f in filenames:
                    filename.append(f)
                for l in labels.cpu().numpy():
                    label.append(l)

                # with open(txtfile, "a+") as file:
                if params.interpret:
                    c = inputs.shape[1]
                    X, Y = inputs[:, :3, :, :], inputs[:, 3:c, :, :]
                    a = X.shape[0]
                    b = X.shape[1]

                    # L0 and L1 norm of input X and Y, single image from each batch
                    for m in range(a):  # single image from batch
                        X_m, Y_m = X[m:m + 1, :, :, :], Y[m:m + 1, :, :, :]
                        # class_name = filenames[m].split('_')[0]
                        if filenames[m].split('_')[0] in class_names:
                            class_name = filenames[m].split('_')[0]
                        else:
                            for class_name_i in class_names:
                                if filenames[m].split('_')[0] in class_name_i:
                                    if filenames[m].split('_')[1] in class_name_i:
                                        class_name = class_name_i

                        avg_L0_image['X_' + class_name].append(round(cal_l0(process_data(X_m)), 4))
                        avg_L1_image['X_' + class_name].append(round(cal_l1(process_data(X_m)), 4))
                        avg_L0_image['Y_' + class_name].append(round(cal_l0(process_data(Y_m)), 4))
                        avg_L1_image['Y_' + class_name].append(round(cal_l1(process_data(Y_m)), 4))

                    avg_L0_channel, avg_L1_channel, avg_L0_image, avg_L1_image = interpret_test(q_out_C, avg_L0_channel,avg_L1_channel,avg_L0_image,avg_L1_image, name='C',filenames=filenames,params=params)
                    avg_L0_channel, avg_L1_channel, avg_L0_image, avg_L1_image = interpret_test(q_out_U, avg_L0_channel,
                                                                                                avg_L1_channel,
                                                                                                avg_L0_image,
                                                                                                avg_L1_image, name='U',
                                                                                                filenames=filenames,
                                                                                                params=params)
                    avg_L0_channel, avg_L1_channel, avg_L0_image, avg_L1_image = interpret_test(q_out_V, avg_L0_channel,
                                                                                                avg_L1_channel,
                                                                                                avg_L0_image,
                                                                                                avg_L1_image, name='V',
                                                                                                filenames=filenames,
                                                                                                params=params)
            if params.interpret:
                print(total_time)
                print(np.mean(total_time[1:]),np.mean(coarse_time[1:]),np.mean(fine_time1[1:]),np.mean(fine_time2[1:]),np.mean(fine_time3[1:]),np.mean(fc_time[1:]))
                print(avg_L1_image.keys)
                num = len(name)
                for i in range(num):
                    if i == num - 1:  # for C_N focus on channel
                        txtfile_l0_channel = txtfile + 'channel_l0_' + name[i] + '.txt'
                        txtfile_l1_channel = txtfile + 'channel_l1_' + name[i] + '.txt'
                        avg_txtfile_l0_channel = txtfile + 'avg_channel_l0_' + name[i] + '.txt'
                        avg_txtfile_l1_channel = txtfile + 'avg_channel_l1_' + name[i] + '.txt'
                    # focus on image
                    txtfile_l0 = txtfile + 'image_l0_' + name[i] + '.txt'
                    txtfile_l1 = txtfile + 'image_l1_' + name[i] + '.txt'
                    avg_txtfile_l0 = txtfile + 'avg_image_l0_' + name[i] + '.txt'
                    avg_txtfile_l1 = txtfile + 'avg_image_l1_' + name[i] + '.txt'
                    name_type = name[i]
                    for m in range(len(class_id_to_name.keys())):
                        name_class = class_id_to_name[str(m)]
                        name_key = name_type + '_' + name_class
                        if i == num - 1:  # for C_N
                            a = avg_L0_channel[name_key]
                            b = avg_L1_channel[name_key]
                            with open(txtfile_l0_channel, "a+") as file_l0:
                                for per_img in a:
                                    file_l0.write("Class=%s | %s" % (name_class, str(per_img)))
                                    file_l0.write('\n')
                                file_l0.flush()
                            with open(txtfile_l1_channel, "a+") as file_l1:
                                for per_img in b:
                                    file_l1.write("Class=%s | %s" % (name_class, str(per_img)))
                                    file_l1.write('\n')
                                file_l1.flush()
                            with open(avg_txtfile_l0_channel, "a+") as file_l0:
                                file_l0.write("Class=%s | %s" % (
                                    name_class, str(np.around(np.mean(a, axis=0), 3).tolist())))
                                file_l0.write('\n')
                                file_l0.flush()
                            with open(avg_txtfile_l1_channel, "a+") as file_l1:
                                file_l1.write("Class=%s | %s" % (
                                    name_class, str(np.around(np.mean(b, axis=0), 3).tolist())))
                                file_l1.write('\n')
                                file_l1.flush()

                        with open(txtfile_l0, "a+") as file_l0:
                            file_l0.write("Class=%s | %s" % (name_class, str(avg_L0_image[name_key])))
                            file_l0.write('\n')
                            file_l0.flush()
                        with open(txtfile_l1, "a+") as file_l1:
                            file_l1.write("Class=%s | %s" % (name_class, str(avg_L1_image[name_key])))
                            file_l1.write('\n')
                            file_l1.flush()
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

            ## confusion matrix
            os.makedirs('./debug/split_' + str(split) + '/qloss_' + model_name + '/save_confusion/',
                        exist_ok=True)
            cm = confusion_matrix(y, pred, normalize='true')
            cm_file = './debug/split_' + str(split) + '/qloss_' + model_name + '/save_confusion/cm.txt'
            with open(cm_file, "a+") as file_l1:
                for i in cm:
                    file_l1.write(str(i))
                    file_l1.write('\n')
                file_l1.flush()
            plot_confusion_matrix(cm, './debug/split_' + str(
                split) + '/qloss_' + model_name + '/save_confusion/cm.png')

            ## top5 predicted
            save_class_txts_top5(filepath, output, label, filename)
            acc, precision, recall, f1, kappa = metricCLS(y, pred)
            print("Test: %s | OA=%.3f%%, Precision=%.3f%%, Recall =%.3f%%, F1=%.3f%%, kappa=%.3f%%" % (
            model_name, 100. * acc, 100. * precision, 100. * recall, 100. * f1, 100. * kappa))


@basic_utils.profile
def run_overall_steps(params):

    if params.net_model:
        if not params.qloss:
            process_networks(params)
        else:
            process_networks_qloss(params)

