import logging
import logging.handlers


LOG_FILENAME = 'LoggingDemo.log'
FORMAT = '%(levelname)7s%(name)10s%(filename)15s:%(lineno)4d -%(funcName)8s %(asctime)s, %(msecs)s, %(message)s'

class MyClass(object):
    def __init__(self):
        logging.basicConfig(
                    filename = LOG_FILENAME,
                    filemode = 'a',
                    level = logging.INFO,
                    format = FORMAT,
                    datefmt = '%H:%M:%S')

        self.logger = logging.getLogger('abcLogger')
        # Adding Console handler

        fhandler = logging.handlers.RotatingFileHandler(
               LOG_FILENAME, maxBytes=100, backupCount=3)
        self.logger.addHandler(fhandler)


        chandler = logging.StreamHandler()
        chandler.setFormatter(logging.Formatter(FORMAT))
        self.logger.addHandler(chandler)

    def process(self):
        self.logger.info('Processing started...')
        y = 10
        z = int(input('Enter Z value:'))
        self.logger.debug('Z value red from console {}'.format(z))
        if z < 0:
            self.logger.warning('Z cannot be -ve')
        try:
            x = y/z
        except Exception as e:
            self.logger.error('Exception received :{}'.format(type(e)))

        self.logger.info('Processing done!')

if __name__ == '__main__':

    mobj = MyClass()
    mobj.process()

