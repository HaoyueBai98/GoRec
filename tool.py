import numpy as np
import torch
import math
import time
import logging
import os
def get_logger(log_name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    vlog = logging.getLogger(log_name)
    vlog.setLevel(level)
    vlog.addHandler(fileHandler)
    return vlog

class Log(object):
    def __init__(self, log_name, log_file, level=logging.INFO):
        # 文件的命名
        logging.basicConfig()
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level)
        self.log_file = log_file
        self.logger.propagate = False
        # 日志输出格式
        self.formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
 
    def __console(self, level, message):
        # 创建一个FileHandler，用于写到本地
        fh = logging.FileHandler(self.log_file, 'a', encoding='utf-8')  # 这个是python3的
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)
 
        # 创建一个StreamHandler,用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)
 
        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        # 这两行代码是为了避免日志输出重复问题
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)
        # 关闭打开的文件
        fh.close()
 
    def debug(self, message):
        self.__console('debug', message)
 
    def info(self, message):
        self.__console('info', message)
 
    def warning(self, message):
        self.__console('warning', message)
 
    def error(self, message):
        self.__console('error', message)
 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.best_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.best_val = max(self.best_val, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cprint(words):
    print(f"\033[0;30;43m{words}\033[0m")
    # print(words)


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def del_tensor_ele(tensor, index):
    tensor_1 = tensor[0:index]
    tensor_2 = tensor[index + 1:]
    return torch.cat((tensor_1, tensor_2), dim=0)

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))

