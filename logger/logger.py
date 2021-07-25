import logging
import logging.config
from pathlib import Path
from utils.util import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        # import pdb;pdb.set_trace()
        for _, handler in config['handlers'].items(): # items()将key和value包装成元组，字典嵌套，因此handler(value)还是字典
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])  # 这样的修改能直接影响到config嘛？？？？还真能改！！！
                # '/home/zhujun/Documents/data/e2vid/model/log/reconstruction/0624_094834/info.log'
        logging.config.dictConfig(config) # 配置logging，之后就可以直接用logging.info写入日志了
        # import pdb;pdb.set_trace()
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
