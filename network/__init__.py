import os
from network.modules import E_resnet,E_mobilenet
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import absl.flags as flags

FLAGS = flags.FLAGS

__models__ = {
    'ResNet18': lambda :E_resnet(resnet18(pretrained = True)),
	'ResNet34': lambda :E_resnet(resnet34(pretrained = True)),
	'ResNet50': lambda :E_resnet(resnet50(pretrained = True)),
	'ResNet101': lambda :E_resnet(resnet101(pretrained = True)),
	'ResNet152': lambda :E_resnet(resnet152(pretrained = True)),
}
def get_models():
    backbone = FLAGS.backbone
    if os.getenv('TORCH_HOME') != FLAGS.depth_pretrained_dir:
        os.environ['TORCH_HOME'] = FLAGS.depth_pretrained_dir
    else:
        pass
    return __models__[backbone]()

