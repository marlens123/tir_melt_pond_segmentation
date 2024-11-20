import os
import argparse
import time
import warnings
import numpy as np
import json

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import JaccardIndex

from .utils.preprocess_helpers import get_preprocessing
import segmentation_models_pytorch as smp

from models.AutoSAM.loss_functions.dice_loss import SoftDiceLoss

from torch.utils.data import DataLoader
from .utils.data import Dataset
from .utils.train_helpers import compute_class_weights, set_seed
from models.smp.build_rs_models import create_model_rs

import wandb

wandb.login()

parser = argparse.ArgumentParser(description="PyTorch Unet Training")

# prefix
parser.add_argument(
    "--pref", default="default", type=str, help="used for wandb logging"
)

# hyperparameters
parser.add_argument("--path_to_config", default="configs/unetplusplus/aid.json", type=str, help="Path to config file that stores hyperparameter setting.")

# processing
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--seed", default=84, type=int, help="seed for initializing training."
)
parser.add_argument("--gpu", default=0, help="GPU id to use.")

# data
parser.add_argument(
    "--path_to_X_train", type=str, default="data/training/train_images.npy"
)
parser.add_argument(
    "--path_to_y_train", type=str, default="data/training/train_masks.npy"
)
parser.add_argument(
    "--path_to_X_test", type=str, default="data/training/test_images.npy"
)
parser.add_argument(
    "--path_to_y_test", type=str, default="data/training/test_masks.npy"
)
parser.add_argument(
    "--arch", type=str, default="UnetPlusPlus", choices=["Unet", "UnetPlusPlus", "MAnet", "Linknet", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "FPN", "PAN"]
)

# wandb monitoring
parser.add_argument(
    '--use_wandb', 
    default=True, 
    action='store_false', 
    help='use wandb for logging'
)

parser.add_argument(
    '--wandb_entity', 
    default='sea-ice', 
    type=str, 
    help='wandb entity name'
)

def main():
    args = parser.parse_args()

    set_seed(args.seed)

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s).")
        else:
            print("CUDA is not available.")

    with open(args.path_to_config) as f:
        config = json.load(f)

    main_worker(args, config)

def main_worker(args, config):

    cfg_model = config['model']
    cfg_training = config['training']

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if cfg_model["pretrain"] == "none":
        cfg_model["pretrain"] = None

    # create model
    if cfg_model["pretrain"] == "imagenet" or cfg_model["pretrain"] == None:
        model = smp.create_model(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            encoder_weights=cfg_model["pretrain"],
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
        print("Using smp pretrain weights")
    else:
        model = create_model_rs(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            pretrain=cfg_model["pretrain"],
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
        print("Using custom pretrain weights")
    #print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # freeze weights in the image_encoder
    if not cfg_model["encoder_freeze"]:
        for name, param in model.named_parameters():
            if param.requires_grad and "iou" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    else:
        for name, param in model.named_parameters():
            if param.requires_grad and "encoder" in name or "iou" in name:
                if "bn" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad:
            print(name, "requires grad")

    if cfg_training["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg_training["learning_rate"],
            weight_decay=cfg_training["weight_decay"],
        )
    elif cfg_training["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg_training["learning_rate"],
            momentum=0.9,
            weight_decay=cfg_training["weight_decay"],
        )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    cudnn.benchmark = True

    # compute class weights
    if cfg_training["use_class_weights"]:
        class_weights = compute_class_weights(args.path_to_y_train)
        # for dice loss component
        class_weights_np = class_weights
        # for cce loss component
        class_weights = torch.from_numpy(class_weights).float().cuda(args.gpu)
    else:
        class_weights = None
        class_weights_np = None

    # Data loading code
    train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))
    test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_training["batch_size"],
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.workers
    )

    save_dir = "runs/" + args.pref

    best_mp_iou = 0.30

    # wandb setup
    if args.use_wandb and args is not None:

        wandb.init(
            entity=args.wandb_entity,
            project='melt_pond',
            name=args.pref,
        )
        wandb.config.update(config)
        wandb.watch(model, log_freq=2)

    for epoch in range(cfg_training["num_epochs"]):
        print("EPOCH {}:".format(epoch + 1))

        # train for one epoch
        train(
            train_loader,
            class_weights,
            model,
            optimizer,
            epoch,
            cfg_model,
            args,
            class_weights_np=class_weights_np,
        )
        _, _, mp_iou, _, _, _, _ = validate(test_loader, model, epoch, scheduler, cfg_model, args)

        # save model weights
        if mp_iou > best_mp_iou:
            best_mp_iou = mp_iou
            save_path = os.path.join("runs/", args.pref, "weights/")
            os.makedirs(save_path, exist_ok = True)
            torch.save(
                model.state_dict(), os.path.join(save_path, "weights_{0}_epoch_{1}.pth".format(args.arch, epoch))
            )

        if epoch == 145:
            model_weights_path = os.path.join("models/", "weights/", args.arch)
            os.makedirs(model_weights_path, exist_ok = True)
            torch.save(
                model.state_dict(), os.path.join(model_weights_path, "{0}_epoch_{1}.pth".format(args.pref, epoch))
            )


def train(
    train_loader,
    class_weights,
    model,
    optimizer,
    epoch,
    cfg_model,
    args=None,
    class_weights_np=None,
):

    if args is None:
        gpu = 0
        use_wandb = False
    else:
        gpu = args.gpu
        use_wandb = args.use_wandb

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    dice_loss = SoftDiceLoss(
        batch_dice=True, do_bg=False, rebalance_weights=class_weights_np
    )
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

        if gpu is not None:
            img = tup[0].float().cuda(gpu, non_blocking=True)
            label = tup[1].long().cuda(gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1].long()

        # compute output
        start_time = time.time()
        mask = model.forward(img)
        end_time = time.time()
        print("Time for forward pass smp: ", end_time - start_time)

        pred_softmax = F.softmax(mask, dim=1)
        loss = ce_loss(mask, label.squeeze(1)) + dice_loss(
            pred_softmax, label.squeeze(1)
        )
        print("Time for loss calculation smp: ", time.time() - end_time)

        jaccard = JaccardIndex(task="multiclass", num_classes=cfg_model["num_classes"]).to(
            gpu
        )
        jac = jaccard(torch.argmax(mask, dim=1), label.squeeze(1))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if use_wandb:
        wandb.log({"epoch": epoch, "train_loss": loss})
        wandb.log({"epoch": epoch, "train_jac": jac})


def validate(val_loader, model, epoch, scheduler, cfg_model, args=None):
    loss_list = []
    jac_list_mp = []
    jac_list_si = []
    jac_list_oc = []
    jac_mean = []
    pred_coll = []
    label_coll = []
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    model.eval()

    if args is None:
        gpu = 0
        use_wandb = False
    else:
        gpu = args.gpu
        use_wandb = args.use_wandb

    with torch.no_grad():
        for i, tup in enumerate(val_loader):
            if gpu is not None:
                img = tup[0].float().cuda(gpu, non_blocking=True)
                label = tup[1].long().cuda(gpu, non_blocking=True)
            else:
                img = tup[0]
                label = tup[1]
            b, _, h, w = img.shape

            label_coll.append(label.squeeze(1).cpu())

            # compute output
            mask = model.forward(img)
            mask = mask.view(b, -1, h, w)

            pred_softmax = F.softmax(mask, dim=1)
            loss = dice_loss(
                pred_softmax, label.squeeze(1)
            )  # self.ce_loss(pred, target.squeeze())
            loss_list.append(loss.item())

            pred = torch.argmax(mask, dim=1)
            pred_coll.append(pred.cpu())

            jaccard = JaccardIndex(
                task="multiclass", num_classes=cfg_model["num_classes"], average=None
            ).to(gpu)
            jaccard_mean = JaccardIndex(
                task="multiclass", num_classes=cfg_model["num_classes"]
            ).to(gpu)

            jac = jaccard(pred_softmax, label.squeeze(1))
            jac_m = jaccard_mean(pred_softmax, label.squeeze(1))

            jac_list_mp.append(jac[0].item())
            jac_list_si.append(jac[1].item())
            jac_list_oc.append(jac[2].item())
            jac_mean.append(jac_m.item())

            if use_wandb:
                wandb.log({"epoch": epoch, "val_loss_{}".format(i): loss.item()})
                wandb.log({"epoch": epoch, "val_jac_mp_{}".format(i): jac[0].item()})
                wandb.log({"epoch": epoch, "val_jac_si_{}".format(i): jac[1].item()})
                wandb.log({"epoch": epoch, "val_jac_oc_{}".format(i): jac[2].item()})
                wandb.log({"epoch": epoch, "val_jac_{}".format(i): jac_m.item()})

    if use_wandb:
        wandb.log({"epoch": epoch, "val_loss": np.mean(loss_list)})
        wandb.log({"epoch": epoch, "val_jac_mp": np.mean(jac_list_mp)})
        wandb.log({"epoch": epoch, "val_jac_si": np.mean(jac_list_si)})
        wandb.log({"epoch": epoch, "val_jac_oc": np.mean(jac_list_oc)})
        wandb.log({"epoch": epoch, "val_jac": np.mean(jac_mean)})

    if epoch >= 10:
        scheduler.step(np.mean(loss_list))

    print(
        "Validating: Epoch: %2d Loss: %.4f"
        % (epoch, np.mean(loss_list))
    )
    return np.mean(loss_list), np.mean(jac_mean), np.mean(jac_list_mp), np.mean(jac_list_oc), np.mean(jac_list_si), label_coll, pred_coll


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    print("Time with GPU: ", end_time - start_time)