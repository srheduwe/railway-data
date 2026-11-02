import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtSmallClassifier(nn.Module):
    """
    ConvNeXtSmall image classifier.

    Args:
        freeze_backbone:    If True, keep all ConvNeXtSmall layers frozen to fine-tune only the head)
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()

        # Load the pretrained vanilla ConvNeXtSmall weights
        self.backbone = models.convnext_small(weights="ConvNeXt_Small_Weights.IMAGENET1K_V1")

        self.backbone.classifier[2] = nn.Linear(in_features=768, out_features=6, bias=True)

        if freeze_backbone:
            # Freeze all layers except the classification head
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits

if __name__ == "__main__":
    # Define a vanilla ConvNeXtSmall classifier for testing
    model = ConvNeXtSmallClassifier(
        freeze_backbone=False
        )
    print(model.backbone.fc)

    # Check input and output shapes
    BATCH_SIZE   = 8
    IMAGE_SHAPE  = (3, 256, 256)
    dummy = torch.randn(BATCH_SIZE, *IMAGE_SHAPE)
    logits = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Logits shape: {logits.shape}")