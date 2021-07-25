import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
from parse_config import ConfigParser       # 自己定义的模块
from trainer import Trainer


# zhujun

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# TODO: 这个函数没有搞清楚在哪里被调用。
def load_model(args, checkpoint=None, config=None):
    """
    negative voxel indicates a model trained on negative voxels -.-
    """
    resume = checkpoint is not None
    if resume:
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')

    if args.legacy:
        config['arch']['type'] += '_legacy'
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    if resume:
        model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    if args.color:
        model = ColorNet(model)
        print('Loaded ColorNet')
    return model


def main(config):
    logger = config.get_logger('train')
     # 多次使用相同的名字调用 getLogger() 会一直返回相同的 Logger 对象的引用
    # setup data_loader instances
    model = config.init_obj('arch', module_arch)
    x = torch.rand(2,10,64,64)
    # y = model(x)

    # import pdb;pdb.set_trace()
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)
    # logger.info(model)
    loss_ftns = [getattr(module_loss, loss)(**kwargs) for loss, kwargs in config['loss_ftns'].items()]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())        
    # filter(func, iter), 保留可迭代器iter中满足func的对象
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    trainer = Trainer(model, loss_ftns, optimizer,
                    config=config,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler)
    trainer.train()
    # import pdb;pdb.set_trace()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/reconstruction_firenet.json', type=str, help='config file path (default: config/reconstruction.json)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('--limited_memory', default=False, action='store_true', help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".')
    # 不接参数 --limited_memory 则 args.parse_args().limited_memory = True 否则为False
    # --limited_memory   limited_memory是属性的名字， --代表可选，没有这两小横表示位置参数，必须指定

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs_test', 'flags type target')  # namedtuple: 利用别名访问tuple元素，依次是 flags/type/target
    # namedtuple是继承自tuple的子类。namedtuple创建一个和tuple类似的对象，而且对象拥有可访问的属性。
    # 原型：
    #       collections.namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)
    # typename 
    #   返回的是一个名为 typename 的元组子类。这个返回的子类用于创建类似元组的对象，这些对象具有可通过属性查找访问的字段以及可索引和可迭代的字段。
    # field_names
    #   field_names 是形如 [ 'x', 'y' ] 的字符串序列。或者为另外两种形式，以逗号或空格分隔的字符串，如 ‘x, y' 和 ’x y'。
    #   除下划线开头的名称外，任何有效的 Python 标识符均可用于 fieldname。有效标识符由字母，数字和下划线组成，
    #   但不能以数字或下划线开头，并且不能是诸如class，for，return，global，pass或raise之类的关键字。
    options = [ # 为啥用两个短横？？？？？？
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        # 有点像字典的元组：CustomArgs_test(flags=['--lr', '--learning_rate'], type=<class 'float'>, target='optimizer;args;lr')
        # options[0].flags
        # ['--lr', '--learning_rate']
        # options[0][0]
        # ['--lr', '--learning_rate']
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--rmb', '--reset_monitor_best'], type=bool, target='trainer;reset_monitor_best'),
        CustomArgs(['--vo', '--valid_only'], type=bool, target='trainer;valid_only')
    ]
    # config ConfigParser 的实例
    config = ConfigParser.from_args(args, options)
    if args.parse_args().limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
    # import pdb;pdb.set_trace()
    main(config)
