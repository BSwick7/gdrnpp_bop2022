# the core predictor classes for gdrn
import csv
import glob
import os
import os.path as osp
import sys
from contextlib import ExitStack
from types import SimpleNamespace

import cv2
import torch
import torchvision
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.data.data_augment import ValTransform
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.engine.yolox_trainer import YOLOX_DefaultTrainer
from det.yolox.exp import get_exp
from det.yolox.utils import fuse_model, get_model_info, postprocess, vis
from lib.utils.time_utils import get_time_str
from loguru import logger
from setproctitle import setproctitle
from torch import nn

from detectron2.config import LazyConfig
from detectron2.evaluation import inference_context

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)


class YoloPredictor():

    def __init__(self, exp_name="yolox-x",
                 config_file_path=osp.join(
                     PROJ_ROOT, "configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_lmo_pbr_lmo_bop_test.py"),
                 ckpt_file_path=osp.join(
                     PROJ_ROOT, "output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_lmo_pbr_lmo_bop_test/model_final.pth"),
                 fuse=True,
                 fp16=False):
        self.exp = get_exp(None, exp_name)
        self.model = self.exp.get_model()
        logger.info("Model Summary: {}".format(
            get_model_info(self.model, self.exp.test_size)))
        self.model.cuda()

        logger.info("loading checkpoint")
        self.args = SimpleNamespace(ckpt_file=ckpt_file_path,
                                    config_file=config_file_path,
                                    eval_only=True,
                                    fuse=fuse,
                                    fp16=fp16
                                    )
        self.model = YOLOX_DefaultTrainer.build_model(self.setup())
        MyCheckpointer(self.model).resume_or_load(
            self.args.ckpt_file, resume=True
        )
        logger.info("loaded checkpoint done.")
        if self.args.fuse:
            logger.info("\tFusing model...")
            self.model = fuse_model(self.model)

        self.preproc = ValTransform(legacy=False)

    def setup(self):
        """Create configs and perform basic setups."""
        cfg = LazyConfig.load(self.args.config_file)

        default_yolox_setup(cfg, self.args)
        # register_datasets_in_cfg(cfg)
        setproctitle("{}.{}".format(cfg.train.exp_name, get_time_str()))
        self.cfg = cfg.test
        return cfg

    def visual_yolo(self, output, rgb_image, class_names, cls_conf=0.35):
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        if output is None:
            return rgb_image
        output = output.cpu()

        bboxes = output[:, 0:4]

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(rgb_image, bboxes, scores, cls, cls_conf, class_names)
        cv2.imshow('cam', vis_res)
        cv2.waitKey(0)

    def postprocess(self, det_preds, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False,
                    keep_single_instance=False):
        box_corner = det_preds.new(det_preds.shape)
        box_corner[:, :, 0] = det_preds[:, :, 0] - det_preds[:, :, 2] / 2
        box_corner[:, :, 1] = det_preds[:, :, 1] - det_preds[:, :, 3] / 2
        box_corner[:, :, 2] = det_preds[:, :, 0] + det_preds[:, :, 2] / 2
        box_corner[:, :, 3] = det_preds[:, :, 1] + det_preds[:, :, 3] / 2
        det_preds[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(det_preds))]
        for i, image_pred in enumerate(det_preds):

            # If none are remaining => process next image
            if not image_pred.size(0):
                # logger.warn(f"image_pred.size: {image_pred.size(0)}")
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] *
                         class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if keep_single_instance:
                instance_detections = torch.rand(num_classes, 7)
                for class_num in range(num_classes):
                    max_conf = 0
                    for detection in detections[detections[:, 6] == class_num]:
                        if detection[4] * detection[5] > max_conf:
                            instance_detections[class_num] = detection
                            max_conf = detection[4] * detection[5]
                detections = instance_detections
            if not detections.size(0):
                # logger.warn(f"detections.size(0) {detections.size(0)} num_classes: {num_classes} conf_thr: {conf_thre} nms_thr: {nms_thre}")
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            detections = detections[detections[:,
                                               6].argsort()].clone().detach()
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output

    def inference(self, image):
        """
        Preprocess input image, run inference and postprocess the output.
        Args:
            image: rgb image
        Returns:
            postprocessed output
        """
        img, _ = self.preproc(image, None, self.cfg.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        if self.args.fp16:
            img = img.float()
            img = img.type(torch.cuda.HalfTensor)

        with ExitStack() as stack:
            if isinstance(self.model, nn.Module):
                stack.enter_context(inference_context(self.model))
            stack.enter_context(torch.no_grad())
            if self.args.fp16:
                self.model = self.model.half()
            outputs = self.model(img, cfg=self.cfg)
            outputs = self.postprocess(outputs["det_preds"],
                                       self.cfg.num_classes,
                                       self.cfg.conf_thr,
                                       self.cfg.nms_thr,
                                       class_agnostic=True,
                                       keep_single_instance=True)

        return outputs


if __name__ == "__main__":
    dataset_name = "lmo"
    if dataset_name == "hb":
        shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_test_primsense_bop19"
    else:
        shared_path = f"yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_bop_test"
    config_file = f"configs/{shared_path}.py"
    ckpt_file = f"output/{shared_path}/model_final.pth"
    predictor = YoloPredictor(
        exp_name="yolox-x",
        config_file_path=osp.join(
            PROJ_ROOT, config_file),
        ckpt_file_path=osp.join(
            PROJ_ROOT, ckpt_file),
        fuse=True,
        fp16=False
    )
    rgb_path = f"datasets/BOP_DATASETS/{dataset_name}/test/000002/rgb/000003.png"
    img_path = osp.join(
        PROJ_ROOT, rgb_path)
    rgb_path_split = rgb_path.split('/')
    scene_id = int(rgb_path_split[4])
    image_id = int(rgb_path_split[6].split('.')[0])
    img = cv2.imread(img_path)
    result = predictor.inference(img)
    predictor_output_dir = f"output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_{dataset_name}_pbr_{dataset_name}_bop_test/scenes/"
    if not osp.exists(osp.join(PROJ_ROOT, predictor_output_dir + rgb_path_split[4])):
        os.makedirs(osp.join(
            PROJ_ROOT, predictor_output_dir + rgb_path_split[4]))
    with open(osp.join(PROJ_ROOT, predictor_output_dir + rgb_path_split[4] + '/yolox_predictor_output_' + rgb_path_split[6].split('.')[0] + '.csv'), 'w') as csv_output_file:
        writer = csv.writer(csv_output_file)
        writer.writerow(['scene id', 'image id', 'x1', 'y1', 'x2', 'y2',
                        'object confidence', 'class confidence', 'object prediction'])
        for item in result:
            for prediction_data in item:
                prediction_data = prediction_data.tolist()
                prediction_data[:0] = [scene_id, image_id]
                writer.writerow(prediction_data)
    # given the dataset name, create an array of class names with the {:06d} value from the obj_{:06d}.ply files in the datasets/BOP_DATASETS/{dataset_name}/models folder
    obj_files = glob.glob(
        f"datasets/BOP_DATASETS/{dataset_name}/models/obj_*.ply")
    cls_names = []
    for file in obj_files:
        # find the value between obj_ and .ply
        i = int(file.split('_')[1].split('.')[0])
        cls_names.append(f"cls_name_{i}")
    predictor.visual_yolo(result[0], img, cls_names)
