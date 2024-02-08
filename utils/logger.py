import os

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str, verbose=True):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')

        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

        if verbose:
            print(str)


def set_log(config, is_train=True):

    data = config.dataset
    exp_name = config.diffusion.method

    log_folder_name = os.path.join(*[data, exp_name])
    root = 'logs_train' 
    if not(os.path.isdir(f'./{root}/{log_folder_name}')):
        os.makedirs(os.path.join(f'./{root}/{log_folder_name}'))
    log_dir = os.path.join(f'./{root}/{log_folder_name}/')

    print('-'*100)
    print("Make Directory {} in Logs".format(log_folder_name))

    return log_folder_name, log_dir


def check_log(log_folder_name, log_name):
    return os.path.isfile(f'./logs_sample/{log_folder_name}/{log_name}.log')


def data_log(logger, config):
    logger.log(f'[{config.dataset}]   seed={config.seed}')


def model_log(logger, config):
    config_m = config.model
    model_log = f'nhid={config_m.nhid} layers={config_m.num_layers} '\
                f'linears={config_m.num_linears}'
    logger.log(model_log)


def start_log(logger, config):
    logger.log('-'*100)
    data_log(logger, config)
    logger.log('-'*100)


def train_log(logger, config):
    logger.log(f'n_feat={config.data.nfeat} nlabel={config.data.nlabel} diffussion_method={config.diffusion.method} diffusion_steps={config.diffusion.step} diffusion_s={config.diffusion.s} | '
               f'nhid={config.model.nhid} cat={config.model.cat_mode} skip={config.model.skip} num_layers={config.model.num_layers} num_linears={config.model.num_linears} output_dir={config.model.output_dir} | '
               f'num_epochs={config.train.num_epochs} print_interval={config.train.print_interval} time_batch={config.train.time_batch} batch={config.train.batch} block_size={config.train.block_size} | '
               f'lr={config.train.lr} unweighted_MSE={config.train.unweighted_MSE}  schedule={config.train.lr_schedule} weight_decay={config.train.weight_decay} lr_decay={config.train.lr_decay} eps={config.train.eps} grad_norm={config.train.grad_norm}')
    model_log(logger, config)
    logger.log('-'*100)

