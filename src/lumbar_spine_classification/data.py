import logging
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


# Create the PyTorch dataset
class LumbarSpineDataset(Dataset):
    def __init__(self, df, labels_df, data_dir, transform=None):
        self.df = df
        self.labels_df = labels_df
        self.data_dir = data_dir
        self.transform = transform

        # Create a dictionary to map study_id to a list of (series_id, instance_number) pairs
        self.study_to_samples = {}
        for _, row in self.labels_df.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            instance_number = row["instance_number"]
            if study_id not in self.study_to_samples:
                self.study_to_samples[study_id] = []
            self.study_to_samples[study_id].append((series_id, instance_number))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["study_id"]

        # Load the DICOM images for the given study_id
        images = []
        for series_id, instance_number in self.study_to_samples[study_id]:
            dicom_path = os.path.join(
                self.data_dir,
                "train_images",
                study_id,
                series_id,
                f"{instance_number}.dcm",
            )
            dicom = pydicom.read_file(dicom_path)
            image = dicom.pixel_array
            images.append(image)

        # Stack the images into a 3D tensor
        image_tensor = torch.tensor(np.stack(images, axis=0)).float()

        # Get the labels
        labels = []
        for condition in [
            "spinal_canal_stenosis",
            "neural_foraminal_narrowing",
            "subarticular_stenosis",
        ]:
            for side in ["left", "right"]:
                for level in ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]:
                    label_col = f"{condition}_{side}_{level}"
                    if label_col in row:
                        label = row[label_col]
                        if label == "Normal/Mild":
                            label = 0
                        elif label == "Moderate":
                            label = 1
                        elif label == "Severe":
                            label = 2
                        else:
                            label = -1
                        labels.append(label)
                    else:
                        labels.append(-1)

        labels_tensor = torch.tensor(labels).long()

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, labels_tensor


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    base_dataset = torchvision.datasets.Caltech101(
        root=data_config["trainpath"],
        download=True,
        transform=input_transform,
    )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = len(base_dataset.categories)
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes
