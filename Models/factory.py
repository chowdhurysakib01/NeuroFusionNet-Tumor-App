from Models.custom_cnn import CustomCNN
from Models.densenet169 import DenseNet169
from Models.mobilenetv3 import MobileNetV3
from Models.resnet50 import ResNet50
from Models.vgg19 import VGG19
from Models.densenet169_resnet50_fusion import DenseNet169ResNet50Fusion
from Models.neurofusionnet import NeuroFusionNet


MODEL_NAMES = [
    "custom_cnn",
    "densenet169",
    "mobilenetv3",
    "resnet50",
    "vgg19",
    "densenet169_resnet50_fusion",
    "neurofusionnet"
]


def build_model(model_name, num_classes=4, pretrained=True):

    name = model_name.lower().strip()

    model_classes = {
        "custom_cnn": CustomCNN,
        "densenet169": DenseNet169,
        "mobilenetv3": MobileNetV3,
        "resnet50": ResNet50,
        "vgg19": VGG19,
        "densenet169_resnet50_fusion": DenseNet169ResNet50Fusion,
        "neurofusionnet": NeuroFusionNet
    }

    if name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = model_classes[name]

    if name == "custom_cnn":
        return model_class(num_classes=num_classes)

    return model_class(
        num_classes=num_classes,
        pretrained=pretrained
    )
