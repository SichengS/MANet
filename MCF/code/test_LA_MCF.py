import os
import argparse
import torch
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='Name of Experiment')  # todo change dataset path
parser.add_argument('--model', type=str,  default="baseline", help='model_name')                # todo change test model name
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--label_num', type=int, default=8, help='label data num')
FLAGS = parser.parse_args()

if FLAGS.model == 'baseline': 
    from networks.vnet import VNet
    from networks.ResNet34 import Resnet34
elif FLAGS.model == 'edge':
    from networks.vnet_edge import VNet
    from networks.ResNet_edge import Resnet34

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model + "_" + str(FLAGS.label_num) + '/'
test_save_path = "../model/"+FLAGS.model + "_" + str(FLAGS.label_num) + '/prediction/'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/test.list', 'r') as f:                                         # todo change test flod
    image_list = f.readlines()
image_list = [FLAGS.root_path + "2018LA_Seg_Training_Set/" +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]

def create_model(name='vnet'):
    # Network definition
    if name == 'vnet':
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
    if name == 'resnet34':
        net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()

    return model

def test_calculate_metric(epoch_num):
    vnet   = create_model(name='vnet')
    resnet = create_model(name='resnet34')

    v_save_mode_path = os.path.join(snapshot_path, 'vnet_iter_' + str(epoch_num) + '.pth')
    vnet.load_state_dict(torch.load(v_save_mode_path))
    print("init weight from {}".format(v_save_mode_path))
    vnet.eval()

    r_save_mode_path = os.path.join(snapshot_path, 'resnet_iter_' + str(epoch_num) + '.pth')
    resnet.load_state_dict(torch.load(r_save_mode_path))
    print("init weight from {}".format(r_save_mode_path))
    resnet.eval()

    avg_metric = test_all_case(vnet, resnet, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)
    
    with open(test_save_path+'/performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))

    return avg_metric

if __name__ == '__main__':
    iters = 6000
    metric = test_calculate_metric(iters)
    print(metric)
