# inference with detector, gdrn, and refiner
import csv
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)


from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import os

import cv2


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
def get_image_list(rgb_images_path, depth_images_path=None):
    image_names = []

    rgb_file_names = os.listdir(rgb_images_path)
    rgb_file_names.sort()
    for filename in rgb_file_names:
        apath = os.path.join(rgb_images_path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)

    if depth_images_path is not None:
        depth_file_names = os.listdir(depth_images_path)
        depth_file_names.sort()
        for i, filename in enumerate(depth_file_names):
            apath = os.path.join(depth_images_path, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names[i] = (image_names[i], apath)
                # depth_names.append(apath)

    else:
        for i, filename in enumerate(rgb_file_names):
            image_names[i] = (image_names[i], None)

    return image_names


if __name__ == "__main__":
    image_paths = get_image_list(osp.join(PROJ_ROOT,"datasets/debug_data/lmo/test/000002/rgb"), osp.join(PROJ_ROOT,"datasets/debug_data/lmo/test/000002/depth"))
    yolo_predictor = YoloPredictor(
                       exp_name="yolox-x",
                       config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_lmo_pbr_lmo_bop_test.py"),
                       ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_lmo_pbr_lmo_bop_test/model_final.pth"),
                       fuse=True,
                       fp16=False
                     )
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/model_final_wo_optim.pth"),
        camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/lmo/camera.json"),
        path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/lmo/models")
    )

    for rgb_img, depth_img in image_paths:
        rgb_img = cv2.imread(rgb_img)
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
        outputs = yolo_predictor.inference(image=rgb_img)
        data_dict = gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=depth_img)
        # if the data directory does not exist, create it
        if not os.path.exists(osp.join(PROJ_ROOT,"output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/gdrn_demo")):
            os.makedirs(osp.join(PROJ_ROOT,"output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/gdrn_demo"))
        # write the data dictionary to a csv file in the output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo directory using csv.DictWriter
        with open(osp.join(PROJ_ROOT,"output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/gdrn_demo/data_dict.csv"), 'w') as f:
            w = csv.writer(f)
            # if a key has no value remove the key from the data dictionary 
            for key in list(data_dict.keys()):
                if len(data_dict[key]) == 0:
                    del data_dict[key]
            key_list = list(data_dict.keys())
            limit = len(data_dict[key_list[0]].tolist())
            w.writerow(data_dict.keys())
            for i in range(limit):
                # if the key has an no value, skip it
                w.writerow([data_dict[x].tolist()[i] for x in key_list])
        out_dict = gdrn_predictor.inference(data_dict)
        # write the dictionary to a csv file in the output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo directory using csv.DictWriter
        with open(osp.join(PROJ_ROOT,"output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/gdrn_demo/output_dict.csv"), 'w') as f:
            w = csv.writer(f)
            # if a key has no value remove the key from the data dictionary 
            for key in list(out_dict.keys()):
                if len(out_dict[key]) == 0:
                    del out_dict[key]
            key_list = list(out_dict.keys())
            limit = len(out_dict[key_list[0]].tolist())
            w.writerow(out_dict.keys())
            for i in range(limit):
                # if the key has an no value, skip it
                w.writerow([out_dict[x].tolist()[i] for x in key_list])
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)
        # write the poses to a csv file in the output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo directory using csv.DictWriter
        with open(osp.join(PROJ_ROOT,"output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/gdrn_demo/poses.csv"), 'w') as f:
            w = csv.writer(f)
            # if a key has no value remove the key from the data dictionary 
            for key in list(poses.keys()):
                if len(poses[key]) == 0:
                    del poses[key]
            key_list = list(poses.keys())
            limit = len(poses[key_list[0]].tolist())
            w.writerow(poses.keys())
            for i in range(limit):
                # if the key has an no value, skip it
                w.writerow([poses[x].tolist()[i] for x in key_list]) 
        gdrn_predictor.gdrn_visualization(batch=data_dict, out_dict=out_dict, image=rgb_img)

