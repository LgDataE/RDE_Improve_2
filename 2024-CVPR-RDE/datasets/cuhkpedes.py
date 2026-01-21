import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class CUHKPEDES(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'CUHK-PEDES'

    def __init__(self, root='', verbose=True):
        super(CUHKPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = self._resolve_anno_path()
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        if annos:
            sample_fp = annos[0].get('file_path', '')
            if sample_fp:
                sample_img_path = op.join(self.img_dir, sample_fp)
                if not op.exists(sample_img_path):
                    raise RuntimeError(
                        "CUHK-PEDES annotation/image structure mismatch. "
                        f"First sample image path not found: '{sample_img_path}'. "
                        "Please ensure your dataset directory contains 'imgs/' with the same "
                        "subfolders referenced by 'caption_all.json'."
                    )
        for anno in annos:
            split = anno.get('split', None)
            if split is None:
                fp = str(anno.get('file_path', '')).lower()
                if fp.startswith('train') or 'train_query' in fp:
                    split = 'train'
                elif fp.startswith('test') or 'test_query' in fp:
                    split = 'test'
                elif fp.startswith('val') or fp.startswith('valid') or 'val' in fp:
                    split = 'val'
            if split is None:
                raise RuntimeError(
                    "CUHK-PEDES annotation format mismatch: expected key 'split' in each item "
                    "(train/test/val). Your 'caption_all.json' entries look like: "
                    f"{list(anno.keys())}. Please use the processed annotation required by this repo "
                    "or add 'split' field to each entry."
                )

            if split == 'train':
                train_annos.append(anno)
            elif split == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _resolve_anno_path(self):
        candidates = [
            'reid_raw_clean.json',
            'reid_raw.json',
            'caption_all_clean.json',
            'caption_all.json',
        ]
        for name in candidates:
            p = op.join(self.dataset_dir, name)
            if op.exists(p):
                return p
        raise RuntimeError(
            f"No CUHK-PEDES annotation json found under '{self.dataset_dir}'. Expected one of: {candidates}"
        )


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
