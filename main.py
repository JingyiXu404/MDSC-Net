import argparse
import logging
import os
from datetime import datetime


from utils.basic_utils import RunSteps, PrForm, DataTypes, DepthTypes, Models, Pools, OverallModes
from overall_struct import run_overall_steps
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
def is_suitable_rgbd_fusion(params):
    if params.proceed_step in (RunSteps.FIX_RECURSIVE_NN, RunSteps.FINE_RECURSIVE_NN, RunSteps.OVERALL_RUN):
        confidence_scores_path = params.dataset_path + params.features_root + params.proceed_step + \
                                 '/svm_confidence_scores/'
        if not os.path.exists(confidence_scores_path):
            print('{}{}Failed to load the RGB/Depth scores! First, you need to run the system to create RGB/Depth '
                  'scores!{}'.format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
            return False
        else:
            """if not params.load_features:
                print('{}{}Param --load-features needs to be set!{}'.format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
                return False"""
            return True
    else:
        return False


def is_initial_params_suitable(params):
    is_suitable = is_model_available(params.net_model)

    if not os.path.exists(params.dataset_path):
        print('{}{}Dataset path error! Please verify the dataset path!'.
              format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
        is_suitable = False

    if params.data_type == DataTypes.RGBD:
        is_suitable = True #is_suitable_rgbd_fusion(params)

    if params.data_type not in DataTypes.ALL:
        print('{}{}Data type error! Please verify the data type!'.
              format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
        is_suitable = False

    if params.debug_mode not in (0, 1):
        print('{}{}Mode selection error! Please choose either debug (1) or prod (0) mode!'.
              format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
        is_suitable = False

    if params.debug_mode == 1 and params.debug_size < 3:
        print('{}{}Debug size should not be less than 3!'.
              format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
        is_suitable = False

    return is_suitable


def is_model_available(net_model):
    try:
        assert net_model in Models.ALL
        return True
    except AssertionError as e:
        print('{}{}Model param error: "{}{}{}{}{}{}{}"! The available models are "{}alexnet{}", '
              '"{}vgg16_bn{}", "{}resnet50{}", "{}resnet101{}", and "{}densenet121{}".{}'
              .format(PrForm.BOLD, PrForm.RED, PrForm.UNDERLINE, PrForm.MAGENTA, net_model, e, PrForm.END_FORMAT,
                      PrForm.BOLD, PrForm.RED, PrForm.GREEN, PrForm.RED, PrForm.GREEN, PrForm.RED, PrForm.GREEN,
                      PrForm.RED, PrForm.GREEN, PrForm.RED, PrForm.GREEN, PrForm.RED, PrForm.END_FORMAT))
        return False


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")


def init_logger(logfile_name, params):
    os.makedirs(os.path.dirname(logfile_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s',
                        filename=logfile_name, filemode='w')
    logging.info('Running params: {}'.format(params))
    logging.info('----------\n')


def get_initial_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-path", dest="dataset_path", default="/raid/xjy/dataset/wrgbd/colored_rgbd/",
                        help="Path to the data root")
    parser.add_argument("--features-root", dest="features_root", default="models-features",
                        help="Root folder for CNN features to load/save")
    parser.add_argument("--data-type", dest="data_type", default=DataTypes.RGBD, choices=DataTypes.ALL,
                        type=str.lower, help="Data type to process, crop for rgb, depthcrop for depth data")
    parser.add_argument("--depth-type", dest="depth_type", default=DepthTypes.RGB, choices=DepthTypes.ALL,
                        type=int, help="Depth type to process, 3 for colored depth data, 1 for original depth data")
    parser.add_argument("--net-model", dest="net_model", default=Models.cu_MMCSC_n3_d8_share4, choices=Models.ALL,
                        type=str.lower, help="Backbone CNN model to be employed as the feature extractor")
    parser.add_argument("--debug-mode", dest="debug_mode", default=0, type=int, choices=[0, 1])
    parser.add_argument("--debug-size", dest="debug_size", default=3, type=int, help="Debug size for each instance. ")
    parser.add_argument("--log-dir", dest="log_dir", default="../logs", help="Log directory")

    return parser


def init_save_dirs(params):
    annex = ''
    if params.debug_mode:
        annex += '[debug]'
    params.features_root += annex + '/'
    params.log_dir += annex + '/'

    return params


def get_overall_run_params():
    parser = get_initial_parser()
    parser.add_argument("--gpu", dest="gpu", default=False, type=str2bool, help="Use more than 1 gpu or not")
    parser.add_argument("--batch-size", dest="batch_size", default=128, type=int)
    parser.add_argument("--split-no", dest="split_no", default=1, type=int, choices=range(1, 11), help="Split number")
    parser.add_argument("--run-mode", dest="run_mode", default=OverallModes.FIX_PRETRAIN_MODEL, type=int,
                        choices=OverallModes.ALL)
    parser.add_argument("--num-rnn", dest="num_rnn", default=128, type=int, help="Number of RNN")
    parser.add_argument("--lr", dest="lr", default=1*1e-4, type=float, help="learning rate")
    parser.add_argument("--epoch", dest="EPOCH", default=500, type=int, help="epochs")
    parser.add_argument("--phase", dest="phase", default='train', type=str, help="train or test")
    parser.add_argument("--qloss", dest="qloss", default=False, type=str2bool, help="Use q loss or not")
    parser.add_argument("--cu", dest="cu", default=False, type=str2bool, help="Use cu or not")
    parser.add_argument("--down_time", dest="down_time", default=1, type=int, help="down sample times")
    parser.add_argument("--interpret", dest="interpret", default=False, type=str2bool, help="visualize interpret features")
    parser.add_argument("--img-size", dest="img_size", default=224, type=int, help="size of train and test dataset")
    parser.add_argument("--num-class", dest="num_class", default=51, type=int, help="Number of classes")
    parser.add_argument("--num-layer", dest="num_layer", default=1, type=int, help="Number of layers")
    parser.add_argument("--channel-per-class", dest="channel_per_class", default=8, type=int, help="Number of channel_per_class")

    parser.add_argument("--ablation", dest="ablation", default=False, type=str2bool)
    parser.add_argument("--M", dest="M", default=4, type=int, help="M")
    parser.add_argument("--J", dest="J", default=4, type=int, help="J")
    parser.add_argument("--k_c", dest="k_c", default=7, type=int, help="kc")
    parser.add_argument("--k_f", dest="k_f", default=4, type=int, help="kf")

    parser.add_argument("--down-scale-encoder", dest="down_scale_encoder", default=8, type=int, help="down scale times of encoder")
    parser.add_argument("--down-scale-classifier", dest="down_scale_classifier", default=8, type=int, help="down scale times of classifier")
    parser.add_argument("--pretrained", dest="pretrained", default=False, type=int, help="Load pretrained models or not")
    parser.add_argument("--save-features", dest="save_features", default=0, type=int, choices=[0, 1])
    parser.add_argument("--reuse-randoms", dest="reuse_randoms", default=1, choices=[0, 1], type=int,
                        help="Handles whether the random weights are gonna save/load or not")
    parser.add_argument("--pooling", dest="pooling", default=Pools.RANDOM, choices=Pools.ALL,
                        type=str.lower, help="Pooling type")
    parser.add_argument("--load-features", dest="load_features", default=0, type=int, choices=[0])
    parser.add_argument("--trial", dest="trial", default=0, type=int, help="For multiple runs")
    params = parser.parse_args()
    params.proceed_step = RunSteps.OVERALL_RUN
    # params.proceed_step = RunSteps.FIX_RECURSIVE_NN
    return params


def run_overall_pipeline():
    params = get_overall_run_params()
    params = init_save_dirs(params)
    if not is_initial_params_suitable(params):
        return

    logfile_name = params.log_dir + params.proceed_step + '/' + get_timestamp() + '_' + str(params.trial) + '-' + \
                   params.net_model + '_' + params.data_type + '_split_' + str(params.split_no) + '.log'
    init_logger(logfile_name, params)
    run_overall_steps(params)


def main():
    run_overall_pipeline()


if __name__ == '__main__':
    main()
