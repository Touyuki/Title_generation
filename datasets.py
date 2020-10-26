# create a dataset with local images
import torch
import os, glob
import csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
from skimage import io

class title_m(Dataset):
    def __init__(self, root):
        super(title_m, self).__init__()
        self.root = root

        self.images, self.labels = self.load_csv("images.csv")

    def load_csv(self, filename):
        # check if there exits a cvs file
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in {"0"}:
                dis = self.root + "train/" + "*.png"
                images += glob.glob(dis)

            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:
                    name = img[16:23]
                    label = name
                    # write the dir and the label of the image into the csv file
                    writer.writerow([img, label])

        # if there exits a cvs file, read it
        images2, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                images2.append(img)
                labels.append(label)
        assert len(images2) == len(labels)
        templates = np.zeros((len(images2), 1, 64, 448), dtype=np.short)  # the size is 64*448
        ten_s = []
        for i in range(0, len(images2)):
            templates[i, 0, :, :] = io.imread(images2[i]).astype(np.short)
            ten_s.append(torch.from_numpy(templates[i]))
        return ten_s, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        return img, label