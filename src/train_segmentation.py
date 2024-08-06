import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

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
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        # Initialize model for color-based clustering
        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        else:
            # Remove this line or update to handle color-based clustering without DINO
            raise ValueError("Unknown arch {}".format(cfg.arch))

        # Color-based cluster probe
        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))
        self.cluster_metrics = UnsupervisedMetrics("test/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_cluster_metrics = UnsupervisedMetrics("final/cluster/", n_classes, cfg.extra_clusters, True)

        # Remove DINO-specific loss function
        self.automatic_optimization = False

        # Color map for dataset visualization
        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.save_hyperparameters()
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        # Initialize model for color-based clustering
        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        # Color-based cluster probe
        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))
        self.cluster_metrics = UnsupervisedMetrics("test/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_cluster_metrics = UnsupervisedMetrics("final/cluster/", n_classes, cfg.extra_clusters, True)

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # Optimizers
        net_optim, cluster_probe_optim = self.optimizers()
        net_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            img = batch["img"]
            label = batch["label"]

        # Obtaining features and codes from the network
        feats, code = self.net(img)
        
        # Color-based clustering
        patch_size = 16
        n_clusters = self.n_classes
        img_rgb = img.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC format
        color_clusters = color_based_clustering(img_rgb[0], patch_size, n_clusters)  # Only for first image in batch

        # Integrate color clusters if needed
        # Placeholder for using color_clusters

        cluster_loss, cluster_preds = self.cluster_probe(code, None)
        loss = cluster_loss
        self.log('loss/cluster', cluster_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss/total', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()

        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, code = self.net(img)
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            # Color-based clustering for validation
            patch_size = 16
            n_clusters = self.n_classes
            img_rgb = img.permute(0, 2, 3, 1).cpu().numpy() 
            color_clusters = color_based_clustering(img_rgb[0], patch_size, n_clusters) 

            # Integrate color clusters if needed
            # Placeholder for using color_clusters

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'cluster_preds': cluster_preds[:self.cfg.n_images].detach().cpu(),
                'label': label[:self.cfg.n_images].detach().cpu()
            }

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            tb_metrics = {
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:
                # Select a random output for visualization
                output_num = random.randint(0, len(outputs) - 1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}

                fig, ax = plt.subplots(3, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 3 * 3))
                for i in range(self.cfg.n_images):
                    ax[0, i].imshow(prep_for_plot(output["img"][i]))
                    ax[1, i].imshow(self.label_cmap[output["label"][i]])
                    ax[2, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                ax[0, 0].set_ylabel("Image", fontsize=16)
                ax[1, 0].set_ylabel("Label", fontsize=16)
                ax[2, 0].set_ylabel("Cluster Probe", fontsize=16)
                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.logger.experiment, "plot_labels", self.global_step)

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        # Main optimizer for network parameters
        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)

        # Separate optimizer for cluster probe
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        return [net_optim], [cluster_probe_optim]
import os
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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
