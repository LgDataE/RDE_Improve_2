from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import numpy as np
import os


def inject_noisy_correspondence(dataset, noisy_rate, noisy_file =None):
    logger = logging.getLogger("RDE.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions  = [i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i]== i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i],image_ids[i],images[i],captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))

    return dataset, np.array(real_correspondeces)

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("RDE.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = len(tokenizer.encoder) - 2
    eot_token = len(tokenizer.encoder) - 1
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


DEFAULT_PROMPT_TEMPLATES = [
    "{caption}",
    "a person. {caption}",
    "a photo of a person. {caption}",
]


def get_prompt_templates(args):
    if args is None:
        return []
    if hasattr(args, 'prompt_templates') and args.prompt_templates:
        return list(args.prompt_templates)
    if hasattr(args, 'prompt_template') and args.prompt_template:
        return [args.prompt_template]
    return []


def apply_prompt_template(caption: str, template: str):
    if template is None or template == "":
        return caption
    if "{caption}" in template:
        return template.replace("{caption}", caption)
    return template + " " + caption


def prepend_soft_prompt_tokens(text: str, args):
    if args is None:
        return text
    if getattr(args, 'soft_prompt', False) and getattr(args, 'soft_prompt_len', 0) > 0:
        prefix = ("<|mask|> " * int(args.soft_prompt_len)).strip()
        if prefix:
            return prefix + " " + text
    return text


def build_prompted_text(caption: str, args, template: str):
    text = apply_prompt_template(caption, template)
    text = prepend_soft_prompt_tokens(text, args)
    return text


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None, args=None):
        self.image_pids = image_pids
        self.transform = transform
        self.args = args
        
        # Apply occlusion path transformation if needed
        if args and hasattr(args, 'test_dt_type') and args.test_dt_type == 0:
            from .build import change_path
            print("Test Dataset: Occluded Dataset")
            self.img_paths = [change_path(args.dataset_name, path) for path in img_paths]
        else:
            print("Test Dataset: Holistic Dataset")
            self.img_paths = img_paths
        
    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        

        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True,
                 args=None):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.args = args
        self.tokenizer = SimpleTokenizer()
  
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        if self.args is not None and getattr(self.args, 'prompt_ensemble', False):
            templates = get_prompt_templates(self.args)
            if not templates:
                templates = DEFAULT_PROMPT_TEMPLATES
            caption_tokens = torch.stack([
                tokenize(build_prompted_text(caption, self.args, t), tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
                for t in templates
            ], dim=0)
        else:
            templates = get_prompt_templates(self.args)
            template = templates[0] if templates else ""
            caption_tokens = tokenize(build_prompted_text(caption, self.args, template), tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption_tokens


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
        self.args = args
       
        # Apply occlusion path transformation if needed
        if hasattr(args, 'train_dt_type') and args.train_dt_type == 0:
            from .build import change_path
            print("Train Dataset: Occluded Dataset")
            dataset_with_occlusion = []
            for item in dataset:
                pid, image_id, img_path, caption = item
                occlusion_path = change_path(args.dataset_name, img_path)
                dataset_with_occlusion.append((pid, image_id, occlusion_path, caption))
            self.dataset = dataset_with_occlusion
        else:
            print("Train Dataset: Holistic Dataset")
            
        self.dataset, self.real_correspondences = inject_noisy_correspondence(self.dataset,args.noisy_rate,args.noisy_file)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        templates = get_prompt_templates(self.args)
        template = ""
        if templates:
            if getattr(self.args, 'prompt_train_random', False) and len(templates) > 1:
                template = random.choice(templates)
            else:
                template = templates[0]
        caption_tokens = tokenize(build_prompted_text(caption, self.args, template), tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.txt_aug:
            caption_tokens = self.txt_data_aug(caption_tokens.cpu().numpy())
        
        ret = {
        'pids': pid,
        'image_ids': image_id,
        'images': img,
        'caption_ids': caption_tokens,
        'index':index,
        }

        return ret

    def txt_data_aug(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.20:
                    prob /= 0.20
                    # 50% randomly change token to mask token
                    if prob < 0.6:
                        aug_tokens.append(mask) 
                    # 20% randomly change token to random token
                    elif prob < 0.8:
                        aug_tokens.append(random.choice(token_range)) # -> rest 10% randomly keep current token
                    else:
                        None # # 30% randomly remove
                else:
                    # no masking token (will be ignored by loss function later)
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)