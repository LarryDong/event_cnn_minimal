import os
import logging
from pathlib import Path        # pathlib: Object-oriented filesystem paths
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils.util import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        # （1）_xxx      " 单下划线 " 开始的成员变量叫做保护变量，意思是只有类实例和子类实例能访问到这些变量，
        # 需通过类提供的接口进行访问； 不能用'from module import *'导入
        # （2）__xxx    类中的私有变量/方法名 （Python的函数也是对象，所以成员方法称为成员变量也行得通。）,
        # " 双下划线 " 开始的是私有成员，意思是 只有类对象自己能访问，连子类对象也不能访问到这个数据。

        # 为了保证不能在class之外访问私有变量，Python会在类的内部自动的把我们定义的__spam私有变量的名字替换成为
        # _classname__spam(注意，classname前面是一个下划线，spam前是两个下划线)，因此，用户在外部访问__spam的时候就会
        # 提示找不到相应的变量。   python中的私有变量和私有方法仍然是可以访问的；访问方法如下：
        # 私有变量:实例._类名__变量名
        # 私有方法:实例._类名__方法名() 所以还是能够访问！！！

        self.resume = resume
        save_dir = Path(self.config['trainer']['save_dir'])
        # 就是说类前面的_符号只是标志，并不是名字的一部分self._config 和 self.config等价
        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')       # 接收间元组，并返回以可读字符串表示的当地时间，格式由参数format 决定。
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id
        exist_ok = run_id == '' # 如果当前目录存在是否ok(抛出异常)的问题，true就是ok，False就是不ok，抛出异常
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok) # parents=True 中间父文件夹不存在时创建
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        write_json(self.config, self.save_dir / 'config.json') # save updated config file to the checkpoint dir
        setup_logging(self.log_dir) # configure logging module
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod                
    # classmethod修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的cls参数，可以来调用类的属性，类的方法，实例化对象等。
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        1 将自定义的CustomArgs加入args中
        2 args.resume判断是否重新训练还是接着之前的结果训练，并返回一个ConfigParser实例
        """
        # cls 表示的就是类自身！！！
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type) 
            # 也就相当于train.py中的前几句。。。。但是没有用到target！！所以这也是区别！！！
            # CustomArgs_test(flags=['--lr', '--learning_rate'], type=<class 'float'>, target='optimizer;args;lr')
            # CustomArgs_test(flags=['--bs', '--batch_size'], type=<class 'int'>, target='data_loader;args;batch_size')
            # CustomArgs_test(flags=['--rmb', '--reset_monitor_best'], type=<class 'bool'>, target='trainer;reset_monitor_best')
            # CustomArgs_test(flags=['--vo', '--valid_only'], type=<class 'bool'>, target='trainer;valid_only')
            # opt 就是一个可以用属性访问的tuple子类，opt.flags为列表
        # import pdb;pdb.set_trace()
        if not isinstance(args, tuple):
            args = args.parse_args()
        # import pdb;pdb.set_trace()
        # args = Namespace(bs=None, config='config/reconstruction_firenet.json', device=None, limited_memory=False, lr=None, resume=None, rmb=None, vo=None)
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg # 这个用法有意思
            resume = None
            cfg_fname = Path(args.config)
        config = read_json(cfg_fname) # 将json文件加载为字典对象
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))
        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        # {'optimizer;args;lr': None, 'data_loader;args;batch_size': None, 'trainer;reset_monitor_best': None, 'trainer;valid_only': None}
        return cls(config, resume, modification) # 返回ConfigParser的一个实例

    def init_obj(self, name, module, *args, **kwargs):
        # *args表示任何多个无名参数，它是一个tuple；**kwargs表示关键字参数，它是一个dict
        # foo(*args, **kwargs)
        # foo('a', 1, None, a=1, b='2', c=3)  args =  ('a', 1, None) kwargs =  {'a': 1, 'c': 3, 'b': '2'}
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.
        `object = config.init_obj('name', module, a, b=1)` is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']  # 类名称
        module_args = dict(self[name]['args']) # 类参数
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        # all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。元素除了是 0、空、None、False 外都算 True。
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args) # getattr() 函数用于返回一个对象属性值。

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.
        `function = config.init_ftn('name', module, a, b=1)` is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name) 
        # 多次使用相同的名字调用 getLogger() 会一直返回相同的 Logger 对象的引用
        logger.setLevel(self.log_levels[verbosity]) # 设置日志级别，即只有日志级别大于等于设置的级别才会输出！！
        return logger

    # setting read-only attributes
    # 关于 property:  由于python进行属性的定义时，没办法设置私有属性，因此要通过@property的方法来进行设置。这样可以隐藏属性名，让用户进行使用的时候无法随意修改。
    # 访问时不需要加括号()，即可当作成员函数进行访问
    @property # 设置读getter属性，没有设置setter属性，因此为只读！不能更改！
    def config(self): 
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree) 
    # 这个操作可以
    # >>> d = {"a": {"b": {"c": 4}}}
    # >>> l = ["a","b","c"]
    # >>> from operator import getitem
    # >>> reduce(getitem, l, d)
    # 4
