from net_params import MODEL_CHOICE, NUM_CLASSES
from torchvision import models


def get_model():
    if MODEL_CHOICE == 'deeplab50':
        net = models.segmentation.deeplabv3_resnet50(num_classes=NUM_CLASSES)
    elif MODEL_CHOICE == 'deeplab101':
        net = models.segmentation.deeplabv3_resnet101(num_classes=NUM_CLASSES)
    elif MODEL_CHOICE == 'deeplab_mobile':
        net = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=NUM_CLASSES)
    elif MODEL_CHOICE == 'fcn50':
        net = models.segmentation.fcn_resnet50(num_classes=NUM_CLASSES)
    elif MODEL_CHOICE == 'fcn101':
        net = models.segmentation.fcn_resnet101(num_classes=NUM_CLASSES)
    elif MODEL_CHOICE == 'lraspp':
        net = models.segmentation.lraspp_mobilenet_v3_large(num_classes=NUM_CLASSES)
    else:
        print('Unsupported model {}... continuing with the default - fcn50'.format(MODEL_CHOICE))
        net = models.segmentation.fcn_resnet50(num_classes=NUM_CLASSES)

    return net