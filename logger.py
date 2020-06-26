
class colors:
    """
    Better for your eyes
    """
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    end = '\033[0m'


def info(msg):
    print(colors.green + 'INFO: ' + colors.end + msg)
def warn(msg):
    print(colors.yellow + 'WARN: ' + colors.end + msg)
