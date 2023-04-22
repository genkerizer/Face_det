

import os
import cv2
from .utils import Timer
from face_det_torch.predict_model import FaceDet

if __name__ == '__main__':
    det = FaceDet()

    testset_folder = "DATASET/widerface/val/images/"
    testset_list = "DATASET/widerface/val/wider_val.txt"
    save_folder = "evaluation/outputs/widerface_evaluate/widerface_txt/"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()

    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}


    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name

        img = cv2.imread(image_path)
        preds = det.predict(img, visual_mode=False, evals=True)
        
        save_name = save_folder + img_name[:-4] + ".txt"

        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with open(save_name, "w") as fd:
            bboxs = preds[:, :5]
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
