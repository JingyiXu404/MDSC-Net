import cv2
import os
import warnings
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('agg')
warnings.filterwarnings(action='ignore')
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
def sort_b_by_a(a,b):
    sorted_b = sorted(b, key=lambda x: a[b.index(x)])
    sorted_a = sorted(a)
    return sorted_a,sorted_b
def read_txt(filepath):
    f = open(filepath)
    data = f.read()
    data = data.split('\n')
    Name = []
    L1_norm = []
    for i in data:
        if len(i) == 0:
            continue
        part = i.split(' | ')
        name = part[0][6:]
        l1_norm = float(part[1])
        Name.append(name)
        L1_norm.append(l1_norm)
    L1_norm,Name = sort_b_by_a(L1_norm,Name)
    f.close()
    return Name, L1_norm
def write_new_txt_for_visualize(filepath):
    Name_X, L1_X = read_txt(filepath + 'avg_image_l1_X.txt')
    Name_Y, L1_Y = read_txt(filepath + 'avg_image_l1_Y.txt')
    Name_U0, L1_U0 = read_txt(filepath + 'avg_image_l1_U0.txt')
    Name_V0, L1_V0 = read_txt(filepath + 'avg_image_l1_V0.txt')
    Name_C0, L1_C0 = read_txt(filepath + 'avg_image_l1_C0.txt')
    Name_U1, L1_U1 = read_txt(filepath + 'avg_image_l1_U1.txt')
    Name_V1, L1_V1 = read_txt(filepath + 'avg_image_l1_V1.txt')
    Name_C1, L1_C1 = read_txt(filepath + 'avg_image_l1_C1.txt')
    Name_U2, L1_U2 = read_txt(filepath + 'avg_image_l1_U2.txt')
    Name_V2, L1_V2 = read_txt(filepath + 'avg_image_l1_V2.txt')
    Name_C2, L1_C2 = read_txt(filepath + 'avg_image_l1_C2.txt')
    Name_U3, L1_U3 = read_txt(filepath + 'avg_image_l1_U3.txt')
    Name_V3, L1_V3 = read_txt(filepath + 'avg_image_l1_V3.txt')
    Name_C3, L1_C3 = read_txt(filepath + 'avg_image_l1_C3.txt')
    with open(filepath + 'avg_l1_name.txt', "a+") as file_l0:
        file_l0.write('X ##### ')
        file_l0.write(str(Name_X))
        file_l0.write('\n')
        file_l0.write('Y ##### ')
        file_l0.write(str(Name_Y))
        file_l0.write('\n')
        file_l0.write('U0 ##### ')
        file_l0.write(str(Name_U0))
        file_l0.write('\n')
        file_l0.write('V0 ##### ')
        file_l0.write(str(Name_V0))
        file_l0.write('\n')
        file_l0.write('C0 ##### ')
        file_l0.write(str(Name_C0))
        file_l0.write('\n')
        file_l0.write('U1 ##### ')
        file_l0.write(str(Name_U1))
        file_l0.write('\n')
        file_l0.write('V1 ##### ')
        file_l0.write(str(Name_V1))
        file_l0.write('\n')
        file_l0.write('C1 ##### ')
        file_l0.write(str(Name_C1))
        file_l0.write('\n')
        file_l0.write('U2 ##### ')
        file_l0.write(str(Name_U2))
        file_l0.write('\n')
        file_l0.write('V2 ##### ')
        file_l0.write(str(Name_V2))
        file_l0.write('\n')
        file_l0.write('C2 ##### ')
        file_l0.write(str(Name_C2))
        file_l0.write('\n')
        file_l0.write('U3 ##### ')
        file_l0.write(str(Name_U3))
        file_l0.write('\n')
        file_l0.write('V3 ##### ')
        file_l0.write(str(Name_V3))
        file_l0.write('\n')
        file_l0.write('C3 ##### ')
        file_l0.write(str(Name_C3))
        file_l0.write('\n')
        file_l0.flush()
    with open(filepath + 'avg_l1_value.txt', "a+") as file_l0:
        file_l0.write('X ##### ')
        file_l0.write(str(L1_X))
        file_l0.write('\n')
        file_l0.write('Y ##### ')
        file_l0.write(str(L1_Y))
        file_l0.write('\n')
        file_l0.write('U0 ##### ')
        file_l0.write(str(L1_U0))
        file_l0.write('\n')
        file_l0.write('V0 ##### ')
        file_l0.write(str(L1_V0))
        file_l0.write('\n')
        file_l0.write('C0 ##### ')
        file_l0.write(str(L1_C0))
        file_l0.write('\n')
        file_l0.write('U1 ##### ')
        file_l0.write(str(L1_U1))
        file_l0.write('\n')
        file_l0.write('V1 ##### ')
        file_l0.write(str(L1_V1))
        file_l0.write('\n')
        file_l0.write('C1 ##### ')
        file_l0.write(str(L1_C1))
        file_l0.write('\n')
        file_l0.write('U2 ##### ')
        file_l0.write(str(L1_U2))
        file_l0.write('\n')
        file_l0.write('V2 ##### ')
        file_l0.write(str(L1_V2))
        file_l0.write('\n')
        file_l0.write('C2 ##### ')
        file_l0.write(str(L1_C2))
        file_l0.write('\n')
        file_l0.write('U3 ##### ')
        file_l0.write(str(L1_U3))
        file_l0.write('\n')
        file_l0.write('V3 ##### ')
        file_l0.write(str(L1_V3))
        file_l0.write('\n')
        file_l0.write('C3 ##### ')
        file_l0.write(str(L1_C3))
        file_l0.write('\n')
        file_l0.flush()
def write_new_txt_for_visualize_l0(filepath):
    Name_X, L1_X = read_txt(filepath + 'avg_image_l0_X.txt')
    Name_Y, L1_Y = read_txt(filepath + 'avg_image_l0_Y.txt')
    Name_U0, L1_U0 = read_txt(filepath + 'avg_image_l0_U0.txt')
    Name_V0, L1_V0 = read_txt(filepath + 'avg_image_l0_V0.txt')
    Name_C0, L1_C0 = read_txt(filepath + 'avg_image_l0_C0.txt')
    Name_U1, L1_U1 = read_txt(filepath + 'avg_image_l0_U1.txt')
    Name_V1, L1_V1 = read_txt(filepath + 'avg_image_l0_V1.txt')
    Name_C1, L1_C1 = read_txt(filepath + 'avg_image_l0_C1.txt')
    Name_U2, L1_U2 = read_txt(filepath + 'avg_image_l0_U2.txt')
    Name_V2, L1_V2 = read_txt(filepath + 'avg_image_l0_V2.txt')
    Name_C2, L1_C2 = read_txt(filepath + 'avg_image_l0_C2.txt')
    Name_U3, L1_U3 = read_txt(filepath + 'avg_image_l0_U3.txt')
    Name_V3, L1_V3 = read_txt(filepath + 'avg_image_l0_V3.txt')
    Name_C3, L1_C3 = read_txt(filepath + 'avg_image_l0_C3.txt')
    with open(filepath + 'avg_l0_name.txt', "a+") as file_l0:
        file_l0.write('X ##### ')
        file_l0.write(str(Name_X))
        file_l0.write('\n')
        file_l0.write('Y ##### ')
        file_l0.write(str(Name_Y))
        file_l0.write('\n')
        file_l0.write('U0 ##### ')
        file_l0.write(str(Name_U0))
        file_l0.write('\n')
        file_l0.write('V0 ##### ')
        file_l0.write(str(Name_V0))
        file_l0.write('\n')
        file_l0.write('C0 ##### ')
        file_l0.write(str(Name_C0))
        file_l0.write('\n')
        file_l0.write('U1 ##### ')
        file_l0.write(str(Name_U1))
        file_l0.write('\n')
        file_l0.write('V1 ##### ')
        file_l0.write(str(Name_V1))
        file_l0.write('\n')
        file_l0.write('C1 ##### ')
        file_l0.write(str(Name_C1))
        file_l0.write('\n')
        file_l0.write('U2 ##### ')
        file_l0.write(str(Name_U2))
        file_l0.write('\n')
        file_l0.write('V2 ##### ')
        file_l0.write(str(Name_V2))
        file_l0.write('\n')
        file_l0.write('C2 ##### ')
        file_l0.write(str(Name_C2))
        file_l0.write('\n')
        file_l0.write('U3 ##### ')
        file_l0.write(str(Name_U3))
        file_l0.write('\n')
        file_l0.write('V3 ##### ')
        file_l0.write(str(Name_V3))
        file_l0.write('\n')
        file_l0.write('C3 ##### ')
        file_l0.write(str(Name_C3))
        file_l0.write('\n')
        file_l0.flush()
    with open(filepath + 'avg_l0_value.txt', "a+") as file_l0:
        file_l0.write('X ##### ')
        file_l0.write(str(L1_X))
        file_l0.write('\n')
        file_l0.write('Y ##### ')
        file_l0.write(str(L1_Y))
        file_l0.write('\n')
        file_l0.write('U0 ##### ')
        file_l0.write(str(L1_U0))
        file_l0.write('\n')
        file_l0.write('V0 ##### ')
        file_l0.write(str(L1_V0))
        file_l0.write('\n')
        file_l0.write('C0 ##### ')
        file_l0.write(str(L1_C0))
        file_l0.write('\n')
        file_l0.write('U1 ##### ')
        file_l0.write(str(L1_U1))
        file_l0.write('\n')
        file_l0.write('V1 ##### ')
        file_l0.write(str(L1_V1))
        file_l0.write('\n')
        file_l0.write('C1 ##### ')
        file_l0.write(str(L1_C1))
        file_l0.write('\n')
        file_l0.write('U2 ##### ')
        file_l0.write(str(L1_U2))
        file_l0.write('\n')
        file_l0.write('V2 ##### ')
        file_l0.write(str(L1_V2))
        file_l0.write('\n')
        file_l0.write('C2 ##### ')
        file_l0.write(str(L1_C2))
        file_l0.write('\n')
        file_l0.write('U3 ##### ')
        file_l0.write(str(L1_U3))
        file_l0.write('\n')
        file_l0.write('V3 ##### ')
        file_l0.write(str(L1_V3))
        file_l0.write('\n')
        file_l0.write('C3 ##### ')
        file_l0.write(str(L1_C3))
        file_l0.write('\n')
        file_l0.flush()
if __name__ == '__main__':
    filepath = '/mnt/ssd1/XJY/MDCSC/src/debug/split_1/qloss_cu_mmcsc_n3_d8/test_metrics/'
    write_new_txt_for_visualize_l0(filepath)