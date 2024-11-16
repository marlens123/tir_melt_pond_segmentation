import torch
from segmentation_models_pytorch.encoders import encoders
from segmentation_models_pytorch.encoders import TimmUniversalEncoder
import torch.utils.model_zoo as model_zoo

def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    """
    Function adapted from segmentation_models_pytorch.encoders.get_encoder to allow for loading of custom weights
    """
    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs,
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError(
            "Wrong encoder name `{}`, supported encoders: {}".format(
                name, list(encoders.keys())
            )
        )

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights == "rsd46-whu":
        print("using rsd weights")
        weights = torch.load(r"pretraining_checkpoints/resnet34/{}/resnet34-epoch.19-val_acc.0.921.ckpt".format(weights))["state_dict"]
        for k in list(weights.keys()):
            weights[str(k)[4:]]=weights.pop(k)

        #encoder = ResNet("resnet34",46)
        print(weights.keys())
        encoder.load_state_dict(weights)

    elif weights == "aid":
        print("using aid weights")
        weights = torch.load(r"pretraining_checkpoints/resnet34/{}/resnet34_224-epoch.9-val_acc.0.966.ckpt".format(weights))["state_dict"]
        for k in list(weights.keys()):
            weights[str(k)[4:]]=weights.pop(k)

        #encoder = ResNet("resnet34",30)
        encoder.load_state_dict(weights)

    elif weights is not None:
        print("using imagenet weights")
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys())
                )
            )
        weights = model_zoo.load_url(settings["url"])
        print(weights.keys())
        encoder.load_state_dict(weights)
        encoder.set_in_channels(in_channels, pretrained=weights is not None)

    if output_stride != 32:
        encoder.make_dilated(output_stride)
    print(encoder)

    return encoder