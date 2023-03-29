# inference with detector, gdrn, and refiner
import cv2
import os
import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="lmo",
                    help="dataset name")
    ap.add_argument("-c", "--camera_type", default="camera",
                    help="camera type")
    ap.add_argument("-s", "--scene_num", default="000001",
                    help="scene number")
    args = vars(ap.parse_args())
    
    dataset_name = args["dataset_name"]
    camera_type = args["camera_type"]
    scene_num = args["scene_num"]

    if dataset_name not in ["hb", "icbin", "itodd", "lmo", "tless", "tless_real", "tudl", "tudl_real", "ycbv", "ycbv_real"]:
        print("Invalid dataset name")
        exit()
    if dataset_name == "hb" and camera_type not in ["camera_kinect", "camera_primesense"]:
        print("Invalid camera type for hb. must be camera_kinect or camera_primesense")
        exit()
    elif "tless" in dataset_name and camera_type != "camera_primesense":
        print("Invalid camera type for tless, must be camera_primesense")
        exit()
    elif "ycbv" in dataset_name and camera_type not in ["camera_cmu", "camera_uw"]:
        print("Invalid camera type for ycbv, must be camera_cmu or camera_uw")
        exit()
    elif dataset_name not in ["hb", "tless", "tless_real", "ycbv", "ycbv_real"] and camera_type != "camera":
        print("Invalid camera type, must be camera")
        exit()

    # if the dataset_name is ycbv the camera must be camera_uw for scene_num 000000-000059 and camera_cmu for scene_num 000060-000091
    if "ycbv" in dataset_name and camera_type == "camera_uw" and int(scene_num) > 59:
        print("Invalid scene number for camera_uw, must be 000060-000091")
        exit()
    # given a dataset name like lmo, ycbv, or tless, load the correct config from the configs/yolox/bop_pbr folder in this repo
    # if the dataset_name is hb load the config ends with primesense_bop19 instead
    if dataset_name == "hb":
        # check if the camera type is kinect or primesense
        hb_camera = camera_type.split("_")[1]
        if not os.path.isdir(osp.join( PROJ_ROOT, f"datasets/BOP_DATASETS/{dataset_name}/test_{hb_camera}/{scene_num}")):
            print(f"{dataset_name} dataset does not have a scene number {scene_num} for the {camera_type}")
            exit()
        yolox_shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_test_{hb_camera}_bop19"
        rgb_img_path = f"datasets/BOP_DATASETS/{dataset_name}/test_{hb_camera}/{scene_num}/rgb/000001.png"
        depth_img_path = f"datasets/BOP_DATASETS/{dataset_name}/test_{hb_camera}/{scene_num}/depth/000001.png"
    # if the dataset_name is itodd, the rgb_img_path ends with /gray instead of /rgb and the img type is .tif instead of .png
    elif dataset_name == "itodd":
        if not os.path.isdir(osp.join( PROJ_ROOT, f"datasets/BOP_DATASETS/{dataset_name}/test/{scene_num}")):
            print(f"{dataset_name} dataset does not have a scene number {scene_num}")
            exit()
        yolox_shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_bop_test"
        rgb_img_path = f"datasets/BOP_DATASETS/{dataset_name}/test/{scene_num}/gray/000001.tif"
        depth_img_path = f"datasets/BOP_DATASETS/{dataset_name}/test/{scene_num}/depth/000001.tif"
    elif "real" in dataset_name:
        if not os.path.isdir(osp.join( PROJ_ROOT, f"datasets/BOP_DATASETS/{dataset_name.split('_')[0]}/test/{scene_num}")):
            print(f"{dataset_name.split('_')[0]} dataset does not have a scene number {scene_num}")
            exit()
        yolox_shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name.split('_')[0]}_bop_test"
        rgb_img_path = f"datasets/BOP_DATASETS/{dataset_name.split('_')[0]}/test/{scene_num}/rgb/000001.png"
        depth_img_path = f"datasets/BOP_DATASETS/{dataset_name.split('_')[0]}/test/{scene_num}/depth/000001.png"
    else:
        if not os.path.isdir(osp.join( PROJ_ROOT, f"datasets/BOP_DATASETS/{dataset_name}/test/{scene_num}")):
            print(f"{dataset_name} dataset does not have a scene number {scene_num}")
            exit()
        yolox_shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_bop_test"
        rgb_img_path = f"datasets/BOP_DATASETS/{dataset_name}/test/{scene_num}/rgb/000001.png"
        depth_img_path = f"datasets/BOP_DATASETS/{dataset_name}/test/{scene_num}/depth/000001.png"

    # if the rgb_img path is a directory, get the list of images in the directory
    if os.path.isdir(osp.join(PROJ_ROOT, rgb_img_path)):
        image_paths = get_image_list(
            osp.join(PROJ_ROOT, rgb_img_path), osp.join(PROJ_ROOT, depth_img_path))
    # if the rgb_img path is a file, get the image path
    else:
        image_paths = [(osp.join(PROJ_ROOT, rgb_img_path), osp.join(PROJ_ROOT, depth_img_path))]
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
    if "real" in dataset_name:
        dataset_name = dataset_name.split("_")[0]
    if dataset_name in ["tless", "tudl", "ycbv"]:
        gdrn_dir = dataset_name
    else:
        gdrn_dir = f"{dataset_name}_pbr"

    # given a dataset gdrn directory like lmo_pbr, ycbv_pbr, or tless_pbr, load the correct config from the configs/gdrn folder in this repo
    gdrn_shared_path = f"gdrn/{gdrn_dir}/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_{dataset_name}"
    gdrn_config_file = f"configs/{gdrn_shared_path}.py"
    gdrn_ckpt_file = f"output/{gdrn_shared_path}/model_final_wo_optim.pth"
    camera_json = f"datasets/BOP_DATASETS/{dataset_name}/{camera_type}.json"
    if dataset_name == "tless":
        model_path = f"datasets/BOP_DATASETS/{dataset_name}/models_cad"
    else:
        model_path = f"datasets/BOP_DATASETS/{dataset_name}/models"
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT, gdrn_config_file),
        ckpt_file_path=osp.join(PROJ_ROOT, gdrn_ckpt_file),
        camera_json_path=osp.join(PROJ_ROOT, camera_json),
        path_to_obj_models=osp.join(PROJ_ROOT, model_path)
    )

    # if the dataset name is itodd, set the grayscale flag to true
    if dataset_name == "itodd":
        cv2_read_gray = 0
    else:
        cv2_read_gray = 1
    for rgb_img, depth_img in image_paths:
        rgb_img = cv2.imread(rgb_img, cv2_read_gray)
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
        
