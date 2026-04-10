import os
from collections import defaultdict

import cv2
import numpy as np
from pycocotools.coco import COCO

from yolox.data.datasets.datasets_wrapper import Dataset


class ONE_VISDRONEDataset(Dataset):
    """One-future training dataset for VisDrone MOT annotations."""

    def __init__(
        self,
        data_dir=r"E:\VOD-dataset\VisDrone_MOT_TransVOD",
        json_file="imagenet_vid_train.json",
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.annotation_file = os.path.join(self.data_dir, "annotations", self.json_file)
        self.image_root = os.path.join(self.data_dir, "Data", "VID")
        self.coco = COCO(self.annotation_file)
        self.ids = sorted(self.coco.getImgIds())
        self.images = {img["id"]: img for img in self.coco.dataset["images"]}
        self.class_ids = sorted(self.coco.getCatIds())
        self._classes = tuple(self.coco.cats[cat_id]["name"] for cat_id in self.class_ids)
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.preproc = preproc
        self.imgs = None

        self.support_ids, self.target_ids = self._build_sequence_links()
        self.annotations = self._load_coco_annotations()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        imgs = getattr(self, "imgs", None)
        if imgs is not None:
            del self.imgs

    def _build_sequence_links(self):
        support_ids = {}
        target_ids = {}
        frames_by_video = defaultdict(list)

        for image in self.coco.dataset["images"]:
            frames_by_video[image["video_id"]].append(image)

        for frames in frames_by_video.values():
            frames.sort(key=lambda image: (image["frame_id"], image["id"]))
            for index, image in enumerate(frames):
                image_id = image["id"]
                prev_image_id = frames[index - 1]["id"] if index > 0 else image_id
                next_image_id = frames[index + 1]["id"] if index + 1 < len(frames) else image_id
                support_ids[image_id] = prev_image_id
                target_ids[image_id] = next_image_id

        return support_ids, target_ids

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(image_id) for image_id in self.ids]

    def _load_boxes_for_image(self, image_id):
        image = self.images[image_id]
        width = image["width"]
        height = image["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        res = np.zeros((len(objs), 5), dtype=np.float32)
        for index, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[index, 0:4] = obj["clean_bbox"]
            res[index, 4] = cls

        resize_ratio = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= resize_ratio

        img_info = (height, width)
        resized_info = (int(height * resize_ratio), int(width * resize_ratio))
        return res, img_info, resized_info

    def _image_path(self, image_id):
        return os.path.normpath(os.path.join(self.image_root, self.images[image_id]["file_name"]))

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_anno_from_ids(self, image_id):
        support_image_id = self.support_ids[image_id]
        target_image_id = self.target_ids[image_id]

        res, img_info, resized_info = self._load_boxes_for_image(target_image_id)
        support_res, _, _ = self._load_boxes_for_image(image_id)

        file_name = self._image_path(image_id)
        support_file_name = self._image_path(support_image_id)

        return (
            res,
            support_res,
            img_info,
            resized_info,
            file_name,
            support_file_name,
            target_image_id,
        )

    def load_resized_img(self, index):
        img = self.load_image(index)
        resize_ratio = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        img = cv2.imread(self.annotations[index][4])
        assert img is not None
        return img

    def load_support_resized_img(self, index):
        img = self.load_support_image(index)
        resize_ratio = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_support_image(self, index):
        img = cv2.imread(self.annotations[index][5])
        assert img is not None
        return img

    def pull_item(self, index):
        res, support_res, img_info, resized_info, _, _, target_image_id = self.annotations[index]
        img = self.load_resized_img(index)
        support_img = self.load_support_resized_img(index)
        return (
            img,
            support_img,
            res.copy(),
            support_res.copy(),
            img_info,
            np.array([target_image_id], dtype=np.int64),
        )

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, support_img, target, support_target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, support_img, target, support_target = self.preproc(
                (img, support_img), (target, support_target), self.input_dim
            )

        return np.concatenate((img, support_img), axis=0), (target, support_target), img_info, img_id
