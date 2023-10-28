import os
import time
import torch
import tool
import logging
from tensorboardX import SummaryWriter
import numpy as np

class Env(object):
    def __init__(self, args):
        """
        初始化环境
        :param args:
        """
        self.args = args
        self.ROOT_PATH = './data'

        # self.DATA_PATH = os.path.join(self.ROOT_PATH, 'data')
        self.DATA_PATH = os.path.join(self.ROOT_PATH, self.args.dataset)
        # self.DATA_PATH = os.path.join(self.DATA_PATH, 'cold')
        self.BASE_PATH = os.path.join(self.ROOT_PATH, 'exp_report')
        self.BASE_PATH = os.path.join(self.BASE_PATH, self.args.dataset)
        self.BOARD_PATH = os.path.join(self.BASE_PATH, 'tensorboard')
        self.CKPT_PATH = os.path.join(self.BASE_PATH, self.args.suffix)
        self.LOG_PATH = os.path.join(self.BASE_PATH, self.args.suffix)
        self.PIC_PATH = os.path.join(self.BASE_PATH, self.args.suffix)
        # self.reset(args)

    def reset(self, args):
        self.args = args
        self.time_stamp = time.strftime('%y-%m-%d-%H', time.localtime(time.time()))
        self._check_direcoty()
        self._init_device()
        self._set_seed(self.args.seed)

        if self.args.log:
            logging.shutdown()
            self._init_logger()

        if self.args.tensorboard:
            self._init_tensorboard()

    def _check_direcoty(self):
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH, exist_ok=True)
        if not os.path.exists(self.BOARD_PATH):
            os.makedirs(self.BOARD_PATH, exist_ok=True)
        if not os.path.exists(self.CKPT_PATH):
            os.makedirs(self.CKPT_PATH, exist_ok=True)
        if not os.path.exists(self.LOG_PATH):
            os.makedirs(self.LOG_PATH, exist_ok=True)
        if not os.path.exists(self.PIC_PATH):
            os.makedirs(self.PIC_PATH, exist_ok=True)

    def _init_device(self):
        if torch.cuda.is_available() and self.args.use_gpu:
            self.device = torch.device(self.args.device_id)
        else:
            self.device = 'cpu'
        tool.cprint(f'Code is running on {self.device}')

    def _init_logger(self):
        self.train_logger = tool.Log('train', os.path.join(self.LOG_PATH,
                                                              f'{self.time_stamp}_train_log_{self.args.suffix}.log'))
        self.val_logger = tool.Log('val',
                                      os.path.join(self.LOG_PATH, f'{self.time_stamp}_val_log_{self.args.suffix}.log'))
        self.test_logger = tool.Log('test', os.path.join(self.LOG_PATH,
                                                            f'{self.time_stamp}_test_log_{self.args.suffix}.log'))
        self.train_logger.info(self.args)
        self.val_logger.info(self.args)
        self.test_logger.info(self.args)
        tool.cprint(f'Init Logger')


    def _init_tensorboard(self):
        self.w = SummaryWriter(os.path.join(self.BOARD_PATH, self.time_stamp + "-" + self.args.suffix))
        tool.cprint(f'Init Tensorboard')

    def close_env(self):
        if self.args.log:
            logging.shutdown()

        if self.args.tensorboard:
            self.w.close()

    def _set_seed(self, seed):
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
