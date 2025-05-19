import cv2
import numpy as np
from pycocotools.coco import COCO
import os

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

class LOAFDataset(Dataset):
    """
    Dataset loader for LOAF COCO-format annotations.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="loaf_train.json",
        name="train",
        img_size=(512, 512),
        preproc=None,
    ):
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "LOAF_512")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.class_ids)
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_id) for _id in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        file_name = im_ann["file_name"]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            if obj.get("ignore", 0) == 1 or obj.get("iscrowd", 0) == 1:
                continue  # skip reflections, crowd, or ignored instances

            x1, y1, w, h = obj["bbox"]
            x2 = x1 + w
            y2 = y1 + h
            if obj["area"] > 0 and x2 > x1 and y2 > y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6))  # [x1, y1, x2, y2, class_id, track_id]

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])  # always 0 for person
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = -1  # No track ID in LOAF

        # only store height, width, and file_name
        img_info = (height, width, file_name)

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]
        res, img_info, file_name = self.annotations[index]

        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"Image not found at {img_file}"

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
