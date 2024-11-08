import logging
import random
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from lumbar_spine_classification.utils import load_df
from custom_logging.logger import logger


class LumbarSpineDataset(Dataset):
    def __init__(
        self, description_df, labels_df, data_config, is_train, transform=None
    ):
        """
        Sagittal T2/STIR
        Sagittal T1
        Axial T2
        """
        logger.info("Initializing the LumbarSpineDataset")
        self.view_type = data_config["view_type"]
        self.data_dir = data_config["path"]
        self.transform = transform
        self.image_paths = {}
        self.labels = {}
        self.df = description_df.loc[
            description_df["series_description"] == self.view_type
        ]
        self.labels_df = labels_df
        if is_train:
            self.image_folder = data_config["train_path"]
        else:
            self.image_folder = data_config["test_path"]
        self.get_path_by_study_id_series_id()
        self.get_label_by_study_id()

    def __len__(self):
        return len(self.df)

    def get_path_by_study_id_series_id(self):
        for study_id, series_id in zip(self.df["study_id"], self.df["series_id"]):
            study_dir = os.path.join(self.data_dir, self.image_folder, str(study_id))
            series_dir = os.path.join(study_dir, str(series_id))
            images = os.listdir(series_dir)
            self.image_paths[(study_id, series_id)] = [
                os.path.join(series_dir, img) for img in images
            ]

    def get_label_by_study_id(self):
        for study_id in self.labels_df["study_id"].unique():
            label_row = self.labels_df.loc[self.labels_df["study_id"] == study_id]
            study_labels = []
            for label_col in self.labels_df.columns:
                if label_col != "study_id":
                    label = label_row[label_col].item()
                    logger.debug(label)
                    if label == "Normal/Mild":
                        study_labels.append([1, 0, 0])
                    elif label == "Moderate":
                        study_labels.append([0, 1, 0])
                    elif label == "Severe":
                        study_labels.append([1, 0, 1])
                    else:
                        study_labels.append([1, 0, 0])
            self.labels[study_id] = torch.tensor(study_labels).flatten().float()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        study_id = row["study_id"]
        series_id = row["series_id"]

        # Load the DICOM images for the given series
        series_image_paths = self.image_paths[(study_id, series_id)]
        num_images = len(series_image_paths)

        # Calculate the index of the center image and the two before and after
        center_index = num_images // 2
        start_index = max(0, center_index - 2)
        end_index = min(num_images, center_index + 2)

        dicom_images = []
        for i in range(start_index, end_index):
            image_path = series_image_paths[i]
            image = dicom.pixel_array(image_path)
            dicom_images.append(image)

        image = np.stack(dicom_images, axis=0)
        image_tensor = torch.tensor(image).float()
        if self.transform:
            image_tensor = self.transform(image_tensor)
        print(f"shape: {image_tensor.shape}")

        # Get the labels

        labels_tensor = self.labels[study_id]

        return image_tensor, labels_tensor


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")
    description_df = load_df("train_series_descriptions.csv")
    labels_df = load_df("train.csv")

    input_transform = transforms.Compose([transforms.Resize((256, 256))])

    base_dataset = LumbarSpineDataset(
        description_df=description_df,
        data_config=data_config,
        labels_df=labels_df,
        transform=input_transform,
        is_train=True,
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

    num_classes = len(base_dataset[0][1])
    input_size = tuple(base_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes
