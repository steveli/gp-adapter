import os
import sys
import datetime
import time

log_dir = 'log'

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        time_str = (datetime.datetime.fromtimestamp(time.time())
                    .strftime('%m%d_%H%M%S'))
        log_file = '%s/%s_%s' % (log_dir, time_str, os.getpid())
        print 'log file:', log_file
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    sys.stdout = Logger()
    print 'test'
    for i in xrange(10):
        sys.stdout.write('.')
        sys.stdout.flush()
    print


if __name__ == '__main__':
    main()
