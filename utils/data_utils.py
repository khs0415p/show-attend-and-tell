import torch

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, config, captions, tokenizer, transform):
        self.config = config
        self.captions = captions
        self.tokenizer = tokenizer

        self.max_length = config.max_length

        self.data = [[transform(Image.open(self.config.img_data_path+img)), cap] for img, cap in tqdm(zip(captions['image'], captions['caption']), total=len(captions), desc='Data loading..')]

    def __getitem__(self, index):
        image, caption = self.data[index]
        caption = [self.tokenizer.bos_token_id] + self.tokenizer.encode(caption)[:self.config.max_length - 2] + [self.tokenizer.eos_token_id]

        return {
            "image": image,
            "caption": torch.LongTensor(caption)
        }

    def __len__(self):
        return len(self.data)