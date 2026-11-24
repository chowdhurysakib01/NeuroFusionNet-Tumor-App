from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]


class BrainMRIDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = ROOT / row["path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        label = int(row["label_idx"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label
