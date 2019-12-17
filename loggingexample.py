import logging

from django.http import HttpResponse

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'console': {
            'format': '%(name)-12s %(levelname)-8s %(message)s'
        },
        'file': {
            'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'filename': '/tmp/debug.log'
        }
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
})

# This retrieves a Python logging instance (or creates it)
logger = logging.getLogger(__name__)

def index(request):
    # Send the Test!! log message to standard out
    logger.error("Test!!")
    return HttpResponse("Hello logging world.")

index()













































#importing module
# import logging

# #Create and configure logger
# logging.basicConfig(filename="newfile.log",
# 					format='%(asctime)s %(message)s',
# 					filemode='w')
#
# #Creating an object
# logger=logging.getLogger()
#
# #Setting the threshold of logger to DEBUG
# logger.setLevel(logging.DEBUG)
#
# #Test messages
# logger.debug("Harmless debug Message")
# logger.info("Just an information")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")
# if __name__ == "__main__":
#     logging.debug("Harmless debug Message")
#     logging.info("Just an information")
#     logging.warning("Its a Warning")
#     logging.error("Did you try to divide by zero")
#     logging.critical("Internet is down")
     # logging.warning("I'm a warning!")
     # logging.info("Hello, Python!")
     # logging.debug("I'm a debug message!")