import wandb
import os
from sklearn.model_selection import KFold
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
from .utils.data import Dataset
from .utils.train_helpers import compute_class_weights, set_seed
from .run_smp_finetune import train as unet_torch_train
from .run_smp_finetune import validate as unet_torch_validate
import segmentation_models_pytorch as smp
import argparse
from .utils.preprocess_helpers import get_preprocessing
from sklearn.metrics import confusion_matrix
from models.smp.build_rs_models import create_model_rs

parser = argparse.ArgumentParser(description="PyTorch Unet Training")
parser.add_argument(
    "--arch", type=str, default="Unet"
)
parser.add_argument(
    "--final_sweep", default=False, action="store_true"
)
parser.add_argument(
    "--seed", type=int, default=84
)
parser.add_argument(
    "--config", type=str, default="configs/unet/imnet.json"
)

final_unet_sweep_configuration_imnet = {
    "name": "sweep_unet_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [1e-5]},
    },
}

final_unet_sweep_configuration_aid = {
    "name": "sweep_unet_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_unet_sweep_configuration_rsd = {
    "name": "sweep_unet_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_unet_sweep_configuration_no = {
    "name": "sweep_unet_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_unetplusplus_sweep_configuration_imnet = {
    "name": "sweep_unetplusplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_unetplusplus_sweep_configuration_aid = {
    "name": "sweep_unetplusplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_unetplusplus_sweep_configuration_rsd = {
    "name": "sweep_unetplusplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.00001]},
    },
}

final_unetplusplus_sweep_configuration_no = {
    "name": "sweep_unetplusplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.0005]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_psp_sweep_configuration_imnet = {
    "name": "sweep_psp_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
    },
}

final_psp_sweep_configuration_aid = {
    "name": "sweep_psp_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
    },
}

final_psp_sweep_configuration_rsd = {
    "name": "sweep_psp_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
    },
}

final_psp_sweep_configuration_no = {
    "name": "sweep_psp_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_deeplab_sweep_configuration_imnet = {
    "name": "sweep_deeplab_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
    },
}

final_deeplab_sweep_configuration_aid = {
    "name": "sweep_deeplab_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
    },
}

final_deeplabplus_sweep_configuration_imnet = {
    "name": "sweep_deeplabplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [2]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.001]},
    },
}

final_deeplabplus_sweep_configuration_aid = {
    "name": "sweep_deeplabplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.00001]},
    },
}

final_deeplabplus_sweep_configuration_rsd = {
    "name": "sweep_deeplabplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0.00001]},
    },
}

final_deeplabplus_sweep_configuration_no = {
    "name": "sweep_deeplabplus_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [0.01]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["SGD"]},
        "weight_decay": {"values": [0.001]},
    },
}

final_linknet_sweep_configuration = {
    "name": "sweep_linknet_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [0]},
    },
}

final_manet_sweep_configuration = {
    "name": "sweep_manet_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [5e-4]},
        "batch_size": {"values": [4]},
        "optimizer": {"values": ["Adam"]},
        "weight_decay": {"values": [1e-5]},
    },
}

smp_torch_sweep_configuration = {
    "name": "sweep_smp_torch",
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_melt_pond_iou"},
    "parameters": {
        "im_size": {"values": [480]},
        "learning_rate": {"values": [1e-4, 5e-4, 1e-3, 1e-2]},
        "batch_size": {"values": [1, 2, 4]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "weight_decay": {"values": [0, 1e-5, 1e-3]},
    },
}

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def train_smp_torch(num, args, sweep_id, sweep_run_name, config, train_loader, test_loader, class_weights):
    run_name = f'{sweep_run_name}-{num}'
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )

    with open(args.config) as f:
        hyper_config = json.load(f)

    cfg_model = hyper_config['model']
    cfg_training = hyper_config['training']

    if cfg_model["pretrain"] == "none":
        cfg_model["pretrain"] = None

    class_weights_np = class_weights
    class_weights = torch.from_numpy(class_weights).float().cuda(0)

    # create model
    if cfg_model["pretrain"] == "imagenet" or cfg_model["pretrain"] == None:
        model = smp.create_model(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            encoder_weights=cfg_model["pretrain"],
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
        print("using smp model")
    else:
        model = create_model_rs(
            arch=args.arch,
            encoder_name=cfg_model["backbone"],
            pretrain=cfg_model["pretrain"],
            in_channels=3,
            classes=cfg_model["num_classes"],
        )
        print("using custom model")

    torch.cuda.set_device(0)
    model = model.cuda(0)

    # freeze weights in the image_encoder
    if not cfg_model["encoder_freeze"]:
        for name, param in model.named_parameters():
            if param.requires_grad and "iou" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    else:
        for name, param in model.named_parameters():
            if param.requires_grad and "image_encoder" in name or "iou" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    elif config['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay'],
        )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=20)

    cudnn.benchmark = True

    for epoch in range(150):
        unet_torch_train(
            train_loader,
            class_weights,
            model,
            optimizer,
            epoch,
            cfg_model,
            class_weights_np=class_weights_np,
        )
    _, val_miou, val_mp_iou, val_oc_iou, val_si_iou, y_true, y_pred = unet_torch_validate(test_loader, model, epoch, scheduler, cfg_model)
    cm = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten(), normalize='true')

    run.log(dict(val_mean_iou=val_miou, val_melt_pond_iou=val_mp_iou, val_ocean_iou=val_oc_iou, val_sea_ice_iou=val_si_iou))
    run.finish()
    return val_miou, val_mp_iou, val_oc_iou, val_si_iou, cm


def cross_validate_smp_torch():
    args=parser.parse_args()

    num_folds = 3

    X_path = "data/training/all_images.npy"
    y_path = "data/training/all_masks.npy"

    with open(args.config) as f:
        hyper_config = json.load(f)

    cfg_training = hyper_config['training']
    cfg_model = hyper_config['model']
    
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=14)

    X, y = np.load(X_path), np.load(y_path)

    class_weights = compute_class_weights(y_path)

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f'{project_url}/groups/{sweep_id}'
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    cfg_model['im_size'] = sweep_run.config.im_size

    metrics_miou = []
    metrics_mp_iou = []
    metrics_oc_iou = []
    metrics_si_iou = []

    confusion_matrices = []

    for num, (train, test) in enumerate(kfold.split(X, y)):
        train_dataset = Dataset(cfg_model, cfg_training, mode="train", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[train], masks=y[train])
        test_dataset = Dataset(cfg_model, cfg_training, mode="test", args=args, preprocessing=get_preprocessing(pretraining=cfg_model["pretrain"]), images=X[test], masks=y[test])

        train_loader = DataLoader(
            train_dataset,
            batch_size=sweep_run.config.batch_size,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        )
        reset_wandb_env()
        val_miou, val_mp_iou, val_oc_iou, val_si_iou, cm = train_smp_torch(
            sweep_id=sweep_id,
            num=num,
            args=parser.parse_args(),
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
            train_loader=train_loader,
            test_loader=test_loader,
            class_weights=class_weights,
        )
        metrics_miou.append(val_miou)
        metrics_mp_iou.append(val_mp_iou)
        metrics_oc_iou.append(val_oc_iou)
        metrics_si_iou.append(val_si_iou)

        confusion_matrices.append(cm)

    # average confusion matrices
    confusion_matrix_avg = np.mean(np.stack(confusion_matrices, axis=0), axis=0)
    np.save(f"confusion_matrix_{sweep_run_name}.npy", confusion_matrix_avg)

    # resume the sweep run
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(dict(val_melt_pond_iou=sum(metrics_mp_iou) / len(metrics_mp_iou)))
    sweep_run.log(dict(val_mean_iou=sum(metrics_miou) / len(metrics_miou)))
    sweep_run.log(dict(val_ocean_iou=sum(metrics_oc_iou) / len(metrics_oc_iou)))
    sweep_run.log(dict(val_sea_ice_iou=sum(metrics_si_iou) / len(metrics_si_iou)))
    sweep_run.finish()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


def main():    
    wandb.login()
    args=parser.parse_args()

    set_seed(args.seed)

    count = 1

    if args.final_sweep and args.arch == "Unet":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_unet_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_unet_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_unet_sweep_configuration_no
        else:
            sweep_config = final_unet_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "UnetPlusPlus":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_unetplusplus_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_unetplusplus_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_unetplusplus_sweep_configuration_no
        else:
            sweep_config = final_unetplusplus_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "PSPNet":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_psp_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_psp_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_psp_sweep_configuration_no
        else:
            sweep_config = final_psp_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "DeepLabV3":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_deeplab_sweep_configuration_aid
        else:
            sweep_config = final_deeplab_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "DeepLabV3Plus":
        if "aid" in args.config:
            print("Using AID configuration")
            sweep_config = final_deeplabplus_sweep_configuration_aid
        elif "rsd" in args.config:
            print("Using RSD configuration")
            sweep_config = final_deeplabplus_sweep_configuration_rsd
        elif "no" in args.config:
            print("Using no pretraining configuration")
            sweep_config = final_deeplabplus_sweep_configuration_no
        else:
            sweep_config = final_deeplabplus_sweep_configuration_imnet
    elif args.final_sweep and args.arch == "Linknet":
        sweep_config = final_linknet_sweep_configuration
    elif args.final_sweep and args.arch == "MAnet":
        sweep_config = final_manet_sweep_configuration
    else:
        sweep_config = smp_torch_sweep_configuration
        count = 100
    sweep_id = wandb.sweep(sweep=sweep_config, project="sam", entity="sea-ice")
    wandb.agent(sweep_id, function=cross_validate_smp_torch, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
