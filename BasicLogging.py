import logging

FORMAT = '%(threadName)s %(levelname)7s %(name)10s%(filename)16s:%(lineno)4d -%(funcName)5s %(asctime)s, %(msecs)s, %(message)s'
def start():
    logging.basicConfig(
            level = logging.DEBUG,
            format = FORMAT,
            datefmt = '%H:%M:%S',
            filename='sample.log',
            filemode='a',
    )

    logger = logging.getLogger('FirstLogger')

    logger.debug('logging message')
    logger.info('logging message')
    logger.warning('logging message')
    logger.error('logging message')
    logger.critical('logging message')
    
if __name__ == '__main__':
    start()
