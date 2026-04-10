#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import contextlib
import io
import itertools
import json
import tempfile
import time

import numpy as np
import torch
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from exps.data.visdrone_class import VISDRONE_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
)


def per_class_mAP_table(coco_eval, class_names=VISDRONE_CLASSES, headers=["class", "AP"], colums=2):
    per_class_mAP = {}
    precisions = coco_eval.eval["precision"]
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_mAP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_mAP) * len(headers))
    result_pair = [x for pair in per_class_mAP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    return tabulate(
        row_pair,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=table_headers,
        numalign="left",
    )


class ONEX_VISDRONEEvaluator:
    """COCO-style evaluator for one-future VisDrone predictions."""

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        testdev=False,
        per_class_mAP=True,
    ):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_mAP = per_class_mAP

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        data_list = []
        progress_bar = tqdm if is_main_process() else iter
        inference_time = 0
        nms_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for output, img_h, img_w, img_id in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue

            output = output.cpu()
            bboxes = output[:, 0:4]
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                data_list.append(
                    {
                        "image_id": int(img_id),
                        "category_id": label,
                        "bbox": bboxes[ind].numpy().tolist(),
                        "score": scores[ind].numpy().item(),
                        "segmentation": [],
                    }
                )

        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(name, value)
                for name, value in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, a_infer_time + a_nms_time],
                )
            ]
        )
        info = time_info + "\n"

        if len(data_dict) == 0:
            return 0, 0, info

        coco_gt = self.dataloader.dataset.coco
        if self.testdev:
            json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
            coco_dt = coco_gt.loadRes("./yolox_testdev_2017.json")
        else:
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            coco_dt = coco_gt.loadRes(tmp)

        try:
            from yolox.layers import COCOeval_opt as COCOeval

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        except Exception as exc:
            logger.warning(
                "Falling back to pycocotools COCOeval because optimized COCOeval is unavailable: {}",
                exc,
            )
            from pycocotools.cocoeval import COCOeval

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        info += redirect_string.getvalue()
        if self.per_class_mAP:
            info += "per class mAP:\n" + per_class_mAP_table(coco_eval)
        return coco_eval.stats[0], coco_eval.stats[1], info
