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
    print("=====> in 'main'")
    logger = config.get_logger('train')

    # setup data_loader instances
    print("=====> Init obj. data_loader")
    data_loader = config.init_obj('data_loader', module_data)
    '''
    在init一个module_data(data_loader.data_loaders)对象时，读取config文件中'data_loader'的相关配置进行初始化
    初始化时传入的可选参数和关键参数列表均为空。
    在init_obj函数内部，首先根据传入的名字('data_loader')从config中确定type为HDF5格式，并读取了相关配置参数为module_args
    之后对参数进行更新（及如果传入了关键参数，则更新；否则默认使用config中的全部配置）
    之后调用getattr()(args, module_args)时，getattr()获取了module的'hdf5'属性，attr(args, module_args)相当于进行了HDF5DataLoader的初始化(__init__)
    在HDF5DataLoader的__init__中进行了相关参数配置
    '''
    print("<===== Init obj. data_loader done")


    print("=====> Init obj. valid_data_loader")
    valid_data_loader = config.init_obj('valid_data_loader', module_data)
    print("<===== Init obj. valid_data_loader done")

    # build model architecture, then print to console
    # 用module_arch初始化 config 中的'arch'参数， 命名为model。module_arch是自己定义的
    print("=====> Init architecture.")
    print('module_arch: ', module_arch)
    model = config.init_obj('arch', module_arch)    # arch:architecture
    ''' 这里init_obj 初始化的是module
    module_name: E2VIDRecurrent 
    module_args: {'unet_kwargs': OrderedDict([('num_bins', 10), ('skip_type', 'sum'),
    ('recurrent_block_type', 'convlstm'), ('num_encoders', 3), ('base_num_channels', 32),
    ('num_residual_blocks', 2), ('use_upsample_conv', True), ('final_activation', ''), ('norm', 'none')])}
    最后同样，利用getattr()初始化一个E2VIDRecurrent。TODO: 目前初始化失败。
    '''
    # logger.info(model)
    print("<===== Init architecture done.")

    # init loss classes
    print("======>  init loss classes")
    loss_ftns = [getattr(module_loss, loss)(**kwargs) for loss, kwargs in config['loss_ftns'].items()]
    print("<======  init loss classes done")
    # 在这里getattr获取config中[loss_ftns]的属性值，即perceptual_loss和temporal_consistency_loss

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    print("======>  init optimizer")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())        # filter(func, iter), 保留可迭代器iter中满足func的对象
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    print("======>  init optimizer done")

    print("======>  init Trainer")
    trainer = Trainer(model, loss_ftns, optimizer,          # TODO:
                    config=config,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler)
    print("<======  init Trainer done")
    print("=====> start 'trainer.train()'")
    trainer.train()
    print("=================== traner.train() done  ===================")


if __name__ == '__main__':
    print("=====> in 'train'")
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/reconstruction.json', type=str, help='config file path (default: config/reconstruction.json)')
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
    # import pdb;pdb.set_trace()
    # c：（continue）继续执行，直到遇到下一条断点
    # s：（step）执行当前语句，如果本句是函数调用，则s会执行到函数的第一句
    # n：（next）执行当前语句，如果本句是函数调用，则完整执行函数，接着指向当前执行语句的下一句。
    # r：（return）执行当前运行函数到return,如果本句是函数调用，则s会执行到函数的第一句，如果在函数外且不是函数调用则相当于s
    # a：（args）列出当前执行函数的参数
    # l：（list）列出当前执行语句周围11条代码
    # b：（break）添加断点
    #     b 列出当前所有断点，和断点执行到统计次数
    #     b line_no(行数)：当前脚本的line_no行添加断点
    # tbreak：（temporary break）临时断点  在第一次执行到这个断点之后，就自动删除这个断点，用法和b一样
    # cl：（clear）清除断点
    #     cl lineno 清除当前脚本lineno行的断点，不太好用。。。
    config = ConfigParser.from_args(args, options)
    if args.parse_args().limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    main(config)
