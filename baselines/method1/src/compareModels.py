import torch.nn as nn
import torch
from tasks.pvqa_model_adv import PVQAAdvModel
baseUrl = 'drive/MyDrive/PathVQA'
new_checkpoint_save_dir = baseUrl + \
    "/checkpoint_adv_LXRT_qi_2.pth"  # checkpint_new_LXRT
original_checkpoint_save_dir = baseUrl+"/checkpoint_original_model.pth"


def compareModelWeights(model_a, model_b):
    module_a = model_a._modules
    module_b = model_b._modules
    if len(list(module_a.keys())) != len(list(module_b.keys())):
        return False
    a_modules_names = list(module_a.keys())
    b_modules_names = list(module_b.keys())
    for i in range(len(a_modules_names)):
        layer_name_a = a_modules_names[i]
        layer_name_b = b_modules_names[i]
        if layer_name_a != layer_name_b:
            return False
        layer_a = module_a[layer_name_a]
        layer_b = module_b[layer_name_b]
        if (
            (type(layer_a) == nn.Module) or (type(layer_b) == nn.Module) or
            (type(layer_a) == nn.Sequential) or (
                type(layer_b) == nn.Sequential)
        ):
            if not compareModelWeights(layer_a, layer_b):
                return False
        if hasattr(layer_a, 'weight') and hasattr(layer_b, 'weight'):
            if not torch.equal(layer_a.weight.data, layer_b.weight.data):
                return False
    return True


def loadQAModel():
    PATH = original_checkpoint_save_dir
    checkpoint = torch.load(PATH)
    return checkpoint


if __name__ == '__main__':
    model1 = loadQAModel()['model_lxrt']
    model2 = PVQAAdvModel(4092).lxrt_encoder
    res = compareModelWeights(model1, model2)
    print(res)
