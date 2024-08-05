import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans

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


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
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

        self.train_cluster_probe = ClusterLookup(dim, n_classes)

        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)

        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]


    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        feats, code1 = self.net(imgs)
        feats = F.normalize(feats, dim=1)

        # Calculate RGB values of the patches
        patch_size = self.cfg.patch_size
        img_patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        avg_rgb_patches = img_patches.mean(dim=(-1, -2))

        # Reshape to (N, num_patches, 3)
        avg_rgb_patches = avg_rgb_patches.reshape(avg_rgb_patches.size(0), -1, 3).cpu().numpy()

        # Cluster using K-Means
        kmeans = KMeans(n_clusters=self.n_classes, random_state=0).fit(avg_rgb_patches.reshape(-1, 3))
        cluster_assignments = kmeans.predict(avg_rgb_patches.reshape(-1, 3))
        cluster_assignments = cluster_assignments.reshape(avg_rgb_patches.shape[0], avg_rgb_patches.shape[1])

        # Assign cluster labels to features
        cluster_labels = torch.tensor(cluster_assignments, device=feats.device).view(feats.size(0), 1, 1, feats.size(2), feats.size(3))
        cluster_labels = cluster_labels.expand(feats.size())

        # Loss calculation and backpropagation
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        loss = F.cross_entropy(feats, cluster_labels)
        self.manual_backward(loss)
        optimizer.step()

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        feats, code1 = self.net(imgs)
        feats = F.normalize(feats, dim=1)

        # Calculate RGB values of the patches
        patch_size = self.cfg.patch_size
        img_patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        avg_rgb_patches = img_patches.mean(dim=(-1, -2))

        # Reshape to (N, num_patches, 3)
        avg_rgb_patches = avg_rgb_patches.reshape(avg_rgb_patches.size(0), -1, 3).cpu().numpy()

        # Cluster using K-Means
        kmeans = KMeans(n_clusters=self.n_classes, random_state=0).fit(avg_rgb_patches.reshape(-1, 3))
        cluster_assignments = kmeans.predict(avg_rgb_patches.reshape(-1, 3))
        cluster_assignments = cluster_assignments.reshape(avg_rgb_patches.shape[0], avg_rgb_patches.shape[1])

        # Assign cluster labels to features
        cluster_labels = torch.tensor(cluster_assignments, device=feats.device).view(feats.size(0), 1, 1, feats.size(2), feats.size(3))
        cluster_labels = cluster_labels.expand(feats.size())

        # Validation loss calculation
        loss = F.cross_entropy(feats, cluster_labels)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_epoch_end(self):
        # Log the clustering results or any other metrics if needed
        self.log_cluster_centers()

    def log_cluster_centers(self):
        # Method to log cluster centers (for monitoring purposes)
        kmeans = KMeans(n_clusters=self.n_classes, random_state=0)
        cluster_centers = kmeans.cluster_centers_
        for i, center in enumerate(cluster_centers):
            self.logger.experiment.add_scalar(f'cluster_center_{i}', center.mean(), self.current_epoch)


    def training_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]

        # Forward pass
        feats, _ = self.net(img)

        # Clustering
        # Get patches
        patches = self.get_patches(img, patch_size=self.cfg.patch_size)

        # Calculate average RGB values for each patch
        avg_rgb_values = patches.mean(dim=(-2, -3))

        # KMeans clustering
        kmeans = KMeans(n_clusters=self.n_classes, random_state=0)
        cluster_labels = kmeans.fit_predict(avg_rgb_values.view(avg_rgb_values.size(0), -1).cpu().numpy())

        # Convert cluster_labels back to tensor
        cluster_labels = torch.tensor(cluster_labels, device=self.device, dtype=torch.long)

        # Calculate loss (e.g., CrossEntropyLoss with cluster_labels as targets)
        loss = F.cross_entropy(feats.view(-1, self.n_classes), cluster_labels.view(-1))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def get_patches(self, img, patch_size):
        # Method to extract patches from the image
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(img.size(0), img.size(1), -1, patch_size, patch_size)
        return patches


    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, _ = self.net(img)
            patches = self.get_patches(img, patch_size=self.cfg.patch_size)
            avg_rgb_values = patches.mean(dim=(-2, -3))
            kmeans = KMeans(n_clusters=self.n_classes, random_state=0)
            cluster_labels = kmeans.fit_predict(avg_rgb_values.view(avg_rgb_values.size(0), -1).cpu().numpy())
            cluster_labels = torch.tensor(cluster_labels, device=self.device, dtype=torch.long)

            loss = F.cross_entropy(feats.view(-1, self.n_classes), cluster_labels.view(-1))

            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

    def test_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, _ = self.net(img)
            patches = self.get_patches(img, patch_size=self.cfg.patch_size)
            avg_rgb_values = patches.mean(dim=(-2, -3))
            kmeans = KMeans(n_clusters=self.n_classes, random_state=0)
            cluster_labels = kmeans.fit_predict(avg_rgb_values.view(avg_rgb_values.size(0), -1).cpu().numpy())
            cluster_labels = torch.tensor(cluster_labels, device=self.device, dtype=torch.long)

            loss = F.cross_entropy(feats.view(-1, self.n_classes), cluster_labels.view(-1))

            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        net_optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        return net_optim


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

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

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

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

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    if cfg.submitting_to_aml:
        gpu_args = dict(gpus=1, val_check_interval=250)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    else:
        gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)

        if gpu_args["val_check_interval"] > len(train_loader) // 4:
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
                monitor="val_loss",
                mode="min",
            )
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()


