from flower_fl.models.lenet_5 import LeNet5
from flower_fl.models.resnet import resnet20
from flower_fl.models.cnns import *
from flower_fl.models.masked_models import Mask8CNN, Mask6CNN, Mask4CNN


def load_model(params):
    model, optimizer = None, None

    if params.get('model').get('id') == 'LeNet':
        return LeNet5()
    elif params.get('model').get('id') == 'ResNet':
        return resnet20(params)
    elif params.get('model').get('id') == 'Conv8':
        if params.get('model').get('mode') == 'mask':
            return Mask8CNN()
    elif params.get('model').get('id') == 'Conv6':
        if params.get('model').get('mode') == 'mask':
            return Mask6CNN()
        elif params.get('model').get('mode') == 'dense':
            return Dense6CNN()
    elif params.get('model').get('id') == 'Conv4':
        if params.get('model').get('mode') == 'mask':
            return Mask4CNN()
        elif params.get('model').get('mode') == 'dense':
            return Dense4CNN()



