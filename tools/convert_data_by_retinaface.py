
import os
import cv2
from tqdm import tqdm
from face_det_torch.predict_model import FaceDet

if __name__ == '__main__':
    det = FaceDet()
    input_dir = "DATATEST/cfp-dataset/Data/Images"
    output_dir = "DATATEST/cfp-dataset_convert"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for root, dirs, filenames in tqdm(os.walk(input_dir)):
        if len(dirs) != 0:
            continue
        
        id_dir = os.path.join(output_dir, "/".join(root.split('/')[2:]))
        if not os.path.exists(id_dir):
            os.makedirs(id_dir, exist_ok=True)
        for img_name in filenames:
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, None,fx=0.7, fy=0.7)
            out = det.predict(img, visual_mode=False, evals=False)
            if len(out) == 0:
                continue
            out = out[0][0]
            save_path = os.path.join(id_dir, img_name)
            cv2.imwrite(save_path, out)