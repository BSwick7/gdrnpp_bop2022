# inference with detector, gdrn, and refiner
import cv2
import os
from predictor_gdrn import GdrnPredictor
from predictor_yolo import YoloPredictor
import csv
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)


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
    # given a dataset name like lmo, ycbv, or tless, load the correct config from the configs/yolox/bop_pbr folder in this repo
    dataset_name = "ycbv"
    camera_type = "camera_uw"
    scene_num = "000048"
    rgb_img_path = f"datasets/debug_data/{dataset_name}/test/{scene_num}/rgb"
    depth_img_path = f"datasets/debug_data/{dataset_name}/test/{scene_num}/depth"
    image_paths = get_image_list(
        osp.join(PROJ_ROOT, rgb_img_path), osp.join(PROJ_ROOT, depth_img_path))
    # if the dataset_name is hb load the config ends with primesense_bop19 instead
    if dataset_name == "hb":
        yolox_shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_test_primsense_bop19"
    else:
        yolox_shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_bop_test"
    yolox_config_file = f"configs/{yolox_shared_path}.py"
    yolox_ckpt_file = f"output/{yolox_shared_path}/model_final.pth"
    yolo_predictor = YoloPredictor(
        exp_name="yolox-x",
        config_file_path=osp.join(PROJ_ROOT, yolox_config_file),
        ckpt_file_path=osp.join(PROJ_ROOT, yolox_ckpt_file),
        fuse=True,
        fp16=False
    )
    # if the dataset_name is ycbv the gdrn dir is just the dataset name, otherwise the gdrn dir is the dataset name + _pbr
    if dataset_name == "ycbv":
        gdrn_dir = dataset_name
    else:
        gdrn_dir = f"{dataset_name}_pbr"

    # given a dataset gdrn directory like lmo_pbr, ycbv_pbr, or tless_pbr, load the correct config from the configs/gdrn folder in this repo
    gdrn_shared_path = f"gdrn/{gdrn_dir}/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_{dataset_name}"
    gdrn_config_file = f"configs/{gdrn_shared_path}.py"
    gdrn_ckpt_file = f"output/{gdrn_shared_path}/model_final_wo_optim.pth"
    camera_json = f"datasets/BOP_DATASETS/{dataset_name}/{camera_type}.json"
    model_path = f"datasets/BOP_DATASETS/{dataset_name}/models"
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT, gdrn_config_file),
        ckpt_file_path=osp.join(PROJ_ROOT, gdrn_ckpt_file),
        camera_json_path=osp.join(PROJ_ROOT, camera_json),
        path_to_obj_models=osp.join(PROJ_ROOT, model_path)
    )

    for rgb_img, depth_img in image_paths:
        rgb_img = cv2.imread(rgb_img)
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
        outputs = yolo_predictor.inference(image=rgb_img)
        data_dict = gdrn_predictor.preprocessing(
            outputs=outputs, image=rgb_img, depth_img=depth_img)
        # if the data directory does not exist, create it
        demo_output_dir = f"output/{gdrn_shared_path}/gdrn_demo/{scene_num}"
        if not os.path.exists(osp.join(PROJ_ROOT, demo_output_dir)):
            os.makedirs(osp.join(
                PROJ_ROOT, demo_output_dir))
        # write the data dictionary to a csv file in the output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo directory using csv.DictWriter
        with open(osp.join(PROJ_ROOT, demo_output_dir + "/data_dict.csv"), 'w') as f:
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
        with open(osp.join(PROJ_ROOT, demo_output_dir + "/output_dict.csv"), 'w') as f:
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
        with open(osp.join(PROJ_ROOT, demo_output_dir + "/poses.csv"), 'w') as f:
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
        gdrn_predictor.gdrn_visualization(
            batch=data_dict, out_dict=out_dict, image=rgb_img)
