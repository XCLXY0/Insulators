import argparse
import time
from process import post_process
import torch
from model.rtpose_vgg import get_model
import cv2

def construct_model(args):
    model = get_model(trunk='vgg19')
    state_dict = torch.load(args.model)['model']
    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda().float()
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='./picture/000033.JPG', help='input image')
    parser.add_argument('--model', type=str, default='./paf_800X800_6000_80_14_8_SGD_0.1.pth',
                        help='path to the weights file')
    parser.add_argument('--result-dir', type=str, default="./result/", help='test image save dir')

    args = parser.parse_args()
    input_image = args.image

    # load model
    model = construct_model(args)
    print('start processing image:  ' + args.image)
    points_lines_img, bbox_img = post_process(model, input_image, args)
    cv2.imwrite(args.result_dir + args.image[-10:-4] + ".jpg", points_lines_img)
    cv2.imwrite(args.result_dir + args.image[-10:-4] + "_bbox.jpg", bbox_img)
