"""
    Module contains Dataset class, collate function for DataLoader and loader getter function.

    * ImageCaptionDataset loads data from pickle file and returns image embedding and caption.
    * cl_fn is used to process batch of data and return tensors.
    * get_loader returns DataLoader object.
"""

import os
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json

class ImageCaptionDataset(Dataset):
    '''
    Take in: 
    - img_emb_dir: root directory containing image embeddings for each image
    - caption_path: json file containing caption annotations for each image
    '''
    # TODO: 需要实现一个 ImageCaptionDataset
    def __init__(self, img_emb_dir, caption_path):
        self.img_emb = []
        self.captions = []
        self.img_path = []

        with open(caption_path, 'rb') as f:
            captions_data = json.load(f)

        for item in captions_data:
            image_id = item['image_id']
            caption = item['caption']
 
            image_id_padded = str(image_id).zfill(12)
            img_emb_path = os.path.join(img_emb_dir, f"COCO_train2014_{image_id_padded}.npy")
            img_path = f"COCO_train2014_{image_id_padded}.jpg"

            if os.path.exists(img_emb_path): # NOTE: misalignment with captions and images
                img_emb = np.load(img_emb_path)

                self.img_path.append(img_path)
                self.img_emb.append(img_emb)
                self.captions.append(caption)

        self.data = list(zip(self.img_path, self.img_emb, self.captions))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Return (img_name, img_emb, caption)
        '''
        img_path, img_emb, caption = self.data[idx]
        return img_path, torch.tensor(img_emb, dtype=torch.float32), caption


def cl_fn(batch, tokenizer):
    # TODO: 需要实现一个 collate function
    img_embs = []
    captions = []

    for _, img_emb, caption in batch:
        img_embs.append(img_emb)
        captions.append(caption)

    img_embs = torch.stack(img_embs)
    encodings = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    return img_embs, input_ids, attention_mask

from transformers import AutoTokenizer
def get_loader(dataset, bs_exp=5, shuffle=True, num_workers=0, pin_memory=False, name="gpt2"):
    print("Tokenizer name in dataloader: ", name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    return DataLoader(
        dataset,
        batch_size=2**bs_exp,
        collate_fn=lambda b: cl_fn(b, tokenizer),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
