"""
model.py — EfficientNet-B0 backbone (ImageNet-pretrained) with a fresh 5-class head.
timm swaps the classifier automatically when num_classes is passed.
"""

import timm
import torch.nn as nn


def build_model(num_classes=5, backbone="efficientnet_b0", pretrained=True, dropout=0.2):
    return timm.create_model(
        backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)


def from_config(cfg):
    m = cfg["model"]
    return build_model(
        num_classes=len(cfg["classes"]),
        backbone=m.get("backbone", "efficientnet_b0"),
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.2),
    )


def freeze_backbone(model, freeze=True):
    for p in model.parameters():
        p.requires_grad = not freeze
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    return model


def trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total


if __name__ == "__main__":
    import torch
    model = build_model()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    tr, tot = trainable_parameters(model)
    print("output:", tuple(y.shape), "(expect (2, 5))")
    print(f"trainable params: {tr:,} / {tot:,}")
    assert y.shape == (2, 5)
    print("OK")
