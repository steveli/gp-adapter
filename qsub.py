import argparse
import os
from subprocess import Popen, PIPE


def qsub(command, job_name=None, stdout=None, stderr=None, depend=None):
    """
    depend could be either a string or a list (or tuple, etc.)
    """
    args = ['qsub']
    if job_name:
        args.extend(['-N', job_name])
    if stderr:
        args.extend(['-e', stderr])
    if stdout:
        args.extend(['-o', stdout])
    if depend:
        # in python3, use isinstance(depend, str) instead.
        if not isinstance(depend, basestring):
            depend = ','.join(depend)
        args.extend(['-hold_jid', depend])
    out = Popen(args, stdin=PIPE, stdout=PIPE).communicate(command + '\n')[0]
    print out.rstrip()
    job_id = out.split()[2]
    return job_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', default='sge_log')
    args = parser.parse_args()

    working_dir = args.dir
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir, 0755)

    log_dir = working_dir + '/log'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir, 0755)

    job1_id = qsub('date; sleep 10', 'sleep10',
                   log_dir + '/job1_o.txt', log_dir + '/job1_e.txt')
    job2_id = qsub('date', 'date1',
                   log_dir + '/job2_o.txt', log_dir + '/job2_e.txt',
                   depend=job1_id)
    job3_id = qsub('date; sleep 20', 'sleep20',
                   log_dir + '/job3_o.txt', log_dir + '/job3_e.txt')
    job4_id = qsub('date', 'date2',
                   log_dir + '/job4_o.txt', log_dir + '/job4_e.txt',
                   depend=[job1_id, job3_id])

    print job1_id, job2_id, job3_id, job4_id


if __name__ == '__main__':
    main()
