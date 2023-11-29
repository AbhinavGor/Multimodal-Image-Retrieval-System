# Functionality: This file has functions which enable extraction of avgpool, fc, layer 3 features from an image
from torchvision.models import resnet50, ResNet50_Weights
import torch

from database_connection import connect_to_mongo

avgpool_output = None
fc_output = None
layer3_output = None


def process_avgpool(avgpool):
    res = []
    avgpool = avgpool.detach().numpy().flatten()
    for i in range(0, len(avgpool) - 1, 2):
        res.append((avgpool[i] + avgpool[i+1])/2)
    return res


def process_layer3(layer3):
    res = []
    layer3 = layer3.detach().numpy()[0]
    for i in range(0, len(layer3)):
        res.append(sum(layer3[i].flatten())/(14*14))
    return res


def extract_from_resnet(img):
    img = (torch.as_tensor(img)).unsqueeze(0)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.fc.register_forward_hook(get_activation('fc'))
    model.layer3.register_forward_hook(get_activation('layer3'))

    output = model(img)

    processed_avgpool = process_avgpool(activation['avgpool'])
    processed_fc = activation['fc'].detach().numpy().tolist()[0]
    processed_layer3 = process_layer3(activation['layer3'])
    processed_features = {
        "avgpool": processed_avgpool,
        "layer3": processed_layer3,
        "fc": processed_fc
    }

    return processed_features
