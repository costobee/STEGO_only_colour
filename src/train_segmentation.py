# General imports
import os
import sys
import random
from datetime import datetime
from pathlib import Path

# Data manipulation and visualization
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import torchvision.transforms as T

# PyTorch and PyTorch Lightning
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

# Configuration and utilities
import hydra
from omegaconf import DictConfig, OmegaConf

# Custom modules
from utils import *
from modules import *
from data import *

torch.multiprocessing.set_sharing_strategy('file_system')

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))

def get_patches(img, patch_size):
    patches = []
    h, w, _ = img.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def compute_avg_rgb(patches):
    avg_rgb = [patch.mean(axis=(0, 1)) for patch in patches]
    return np.array(avg_rgb)

def color_based_clustering(image, patch_size, n_clusters):
    patches = get_patches(image, patch_size)
    avg_rgb = compute_avg_rgb(patches)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(avg_rgb)
    return kmeans.labels_
    
class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super()._init_()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        # color-based cluster probe
        self.train_cluster_probe = ClusterLookup(dim, n_classes)

        # Initializing cluster probes with an option for extra clusters if needed
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)

        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        # Color map for dataset visualization
        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()
    def load_from_checkpoint(cls, checkpoint_path, n_classes, cfg, **kwargs):
        # Override the method to include n_classes and cfg
        return super().load_from_checkpoint(checkpoint_path, n_classes=n_classes, cfg=cfg, **kwargs)

    def forward(self, x):
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # optimizers
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()
        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

        # Obtaining features and codes from the network
        feats, code = self.net(img)
        if self.cfg.correspondence_weight > 0:
            feats_pos, code_pos = self.net(img_pos)

        # Preparing signals based on true labels or features
        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            signal_pos = feats_pos

        loss = 0
        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)

        # Salience maps
        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        # Color-based clustering
        patch_size = 170
        n_clusters = self.n_classes  # Or any other number of clusters
        img_rgb = img.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC format
        color_clusters = color_based_clustering(img_rgb[0], patch_size, n_clusters)  # Only for first image in batch

        # Integrate color clusters into the segmentation pipeline
        # Example integration: adjust code with color clustering information or use it in a custom loss function
        # Placeholder: convert color_clusters to a tensor or use it to refine code

        if self.cfg.correspondence_weight > 0:
            pos_intra_loss, pos_intra_cd, pos_inter_loss, pos_inter_cd, neg_inter_loss, neg_inter_cd = self.contrastive_corr_loss_fn(
                signal, signal_pos, salience, salience_pos, code, code_pos)

            if should_log_hist:
                self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
                self.logger.experiment.add_histogram("inter_cd", pos_inter_cd, self.global_step)
                self.logger.experiment.add_histogram("neg_inter_cd", neg_inter_cd, self.global_step)

            intra_loss = (pos_intra_loss + pos_inter_loss) / 2
            loss = self.cfg.intra_weight * intra_loss + self.cfg.neg_inter_weight * neg_inter_loss

        feats_aug, code_aug = self.net(img_aug)
        seg_loss, confidence = self.crf_loss_fn(
            feats, feats_aug, coord_aug, salience, self.current_epoch + 1, None)

        net_optim.backward(seg_loss + loss)

        # Optimizing the cluster probe and the linear probe
        cluster_probe_optim.zero_grad()
        probe_feats = self.decoder(feats.detach())
        probe_loss = self.train_cluster_probe(probe_feats, confidence, ind)
        cluster_probe_optim.backward(probe_loss)
        cluster_probe_optim.step()

        linear_probe_optim.zero_grad()
        linear_probe_feats = self.linear_probe(feats.detach())
        linear_loss = self.linear_probe_loss_fn(linear_probe_feats, label)
        linear_probe_optim.backward(linear_loss)
        linear_probe_optim.step()

        # Logging metrics
        log_dict = {
            "train/loss": (seg_loss + loss).detach(),
            "train/cluster_probe_loss": probe_loss.detach(),
            "train/linear_probe_loss": linear_loss.detach()
        }
        self.log_dict(log_dict, on_step=True, on_epoch=False)
        net_optim.step()
        return (seg_loss + loss).detach()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

        feats, code = self.net(img)
        feats_aug, code_aug = self.net(img_aug)

        # Color-based clustering
        patch_size = 170
        n_clusters = self.n_classes  # Or any other number of clusters
        img_rgb = img.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC format
        color_clusters = color_based_clustering(img_rgb[0], patch_size, n_clusters)  # Only for first image in batch

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            signal_pos = feats_pos

        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        seg_loss, confidence = self.crf_loss_fn(
            feats, feats_aug, coord_aug, salience, self.current_epoch + 1, None)

        linear_probe_feats = self.linear_probe(feats)
        linear_loss = self.linear_probe_loss_fn(linear_probe_feats, label)

        log_dict = {
            "val/seg_loss": seg_loss.detach(),
            "val/linear_probe_loss": linear_loss.detach()
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True)
        return (seg_loss + linear_loss).detach()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

        feats, code = self.net(img)
        feats_aug, code_aug = self.net(img_aug)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            signal_pos = feats_pos

        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        # Color-based clustering
        patch_size = 170
        n_clusters = self.n_classes  # Or any other number of clusters
        img_rgb = img.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC format
        color_clusters = color_based_clustering(img_rgb[0], patch_size, n_clusters)  # Only for first image in batch

        seg_loss, confidence = self.crf_loss_fn(
            feats, feats_aug, coord_aug, salience, self.current_epoch + 1, None)

        linear_probe_feats = self.linear_probe(feats)
        linear_loss = self.linear_probe_loss_fn(linear_probe_feats, label)

        log_dict = {
            "test/seg_loss": seg_loss.detach(),
            "test/linear_probe_loss": linear_loss.detach()
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True)
        return (seg_loss + linear_loss).detach()

    def configure_optimizers(self):
        net_optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(self.linear_probe.parameters(), lr=self.cfg.lr)
        cluster_probe_optim = torch.optim.Adam(self.cluster_probe.parameters(), lr=self.cfg.lr)
        return [net_optim, linear_probe_optim, cluster_probe_optim]
@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = os.path.join(cfg.output_root, "data")
    log_dir = os.path.join(cfg.output_root, "logs")
    checkpoint_dir = os.path.join(cfg.output_root, "checkpoints")

    # Creating directories if they do not exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    prefix = f"{cfg.log_dir}/{cfg.dataset_name}_{cfg.experiment_name}"
    name = f"{prefix}_date_{datetime.now().strftime('%b%d_%H-%M-%S')}"
    cfg.full_name = prefix

    # Seed everything for reproducibility
    seed_everything(seed=0)

    print(f"Data directory: {data_dir}")
    print(f"Log directory: {log_dir}")

    # Define data transformations
    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    val_loader_crop = None if cfg.dataset_name == "voc" else "center"

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    val_batch_size = 16 if cfg.submitting_to_aml else cfg.batch_size
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    gpu_args = dict(gpus=1, val_check_interval=250) if cfg.submitting_to_aml else dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)
    if gpu_args["val_check_interval"] > len(train_loader):
        gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                every_n_train_steps=400,
                save_top_k=2,
                monitor="test/cluster/mIoU",
                mode="max",
            )
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    my_app()
