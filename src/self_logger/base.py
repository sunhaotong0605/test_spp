import logging
import os

rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))

def is_main_process():
    flag = rank == 0 and local_rank == 0
    dist_url = "tcp://%s:%s" % (os.environ.get('MASTER_ADDR', -1), os.environ.get('MASTER_PORT', -1))
    print('**************| distributed init (rank {}): {}, gpu {}; flag: {}; ********'.format(
        rank, dist_url, local_rank, flag))
    return flag


if is_main_process():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)d|pid(%(process)d)|%(threadName)s|%(filename)s:%(lineno)d|%(levelname)s|%(message)s',
    )
    logger = logging.getLogger("ConsoleGenomicsLLM")
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger("NoneLogGenomicsLLM")
    logger.setLevel(logging.ERROR)


def init_logger(svr_name, log_path=None, log_level=logging.INFO, svr_num=None):
    os.makedirs(log_path, exist_ok=True)
    global logger
    if is_main_process():
        if log_path is not None and rank == 0:
            logger.setLevel(log_level)
            file_handler = logging.FileHandler(f"{log_path}/{svr_name}.log")
            logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()

    return logger
