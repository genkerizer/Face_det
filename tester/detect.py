# Author: Nguyen Y Hop
import os
import cv2
import glob
import time
import yaml
import math
import torch
import datetime
import numpy as np
import torch.optim as optim
from src.loader.processes.img_process import Preproc
from torch.utils.data import DataLoader

from src.utils.box_utils import decode_landm, decode
from src.utils.py_cpu_nms import py_cpu_nms
from src.loader.processes.generate_box import PriorBox
from src.losses.multibox_loss import MultiBoxLoss
from src.loader.facedet_loader import FaceDataLoader, detection_collate
from src.models.architectures.base_model import RetinaFace

class Tester:

    def __init__(self, config, **kwargs):
        self.global_config = config['Global']
        self.arch_config = config['Architecture']
        self.optim_config = config['Optimizer']
        self.criterion_config = config['Criterion']
        self.data_config = config['Dataloader']
        self.prior_config = config['PriorBox']
        self.save_config = config['SaveWeight']
        self.inference_config = config['Inference']
        self.convert_config = config['ConvertOnnx']
        self.device = torch.device("cpu" if self.inference_config['cpu'] else "cuda")
        
        self.resize = self.inference_config['resize']
        self.confidence_threshold = self.inference_config['confidence_threshold']
        self.top_k = self.inference_config['top_k']
        self.variance = self.inference_config['variance']
        self.keep_top_k = self.inference_config['keep_top_k']
        self.nms_threshold = self.inference_config['nms_threshold']
        self.vis_thres = self.inference_config['vis_thres']
        self.save_image = self.inference_config['save_image']
        self.save_dir = self.inference_config['save_dir']

        self.build_model()





    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}
    

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def build_model(self):
        pretrained_path = self.inference_config['pretrained_path']
        self.model = RetinaFace(self.arch_config)
        print('Loading pretrained model from {}'.format(pretrained_path))
        if self.inference_config['load_to_cpu']:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(self.model, pretrained_dict)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()

        if self.convert_config['convert']:
            img_size = self.convert_config['img_size']
            save_onnx = self.convert_config['save_onnx']
            os.makedirs(save_onnx, exist_ok=True)
            input_names = ["input0"]
            output_names = ["output0"]
            inputs = torch.randn(1, 3, img_size, img_size).to(self.device)

            torch_out = torch.onnx._export(self.model, inputs, os.path.join(save_onnx, "face_det.onnx"), export_params=True, verbose=False,
                                        input_names=input_names, output_names=output_names)
            exit()
        

    def build_prior_box(self):
        img_size = self.data_config['Preproc']['image_size']
        priorbox = PriorBox(self.prior_config, image_size=(img_size, img_size))
        with torch.no_grad():
            self.priors = priorbox.forward()
            self.priors = self.priors.cuda()
        
    
    def test(self):
        for img_path in glob.glob("DATATEST/image/*.*"):
            img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)

            tic = time.time()
            loc, conf, landms = self.model(img, training=False)  # forward pass
            print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(self.prior_config, image_size=(im_height, im_width))

            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.variance)
            boxes = boxes * scale / self.resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.variance)
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / self.resize
            landms = landms.cpu().numpy()


            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]
            landms = landms[:self.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # show image
            if self.save_image:
                for b in dets:
                    if b[4] < self.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                # save image

                name = img_path.split('/')[-1]
                cv2.imwrite(os.path.join(self.save_dir, name), img_raw)




        return None



if __name__ == '__main__':
    config = yaml.load(open('configs/cfg_retinaface.yml'))
    trainer = Tester(config)
    trainer.test()