import os
from qsub import qsub


def run(gamma=5, update_gp=False):
    if update_gp:
        params = 'hp_params.pkl'
        desc = 'meg-10-x-1k-%d-ag' % gamma
    else:
        params = 'params/B-UWaveGestureLibraryAll-10.pkl'
        desc = 'meg-10-x-1k-u-%d-ag' % gamma

    command = ('python meg_kernel_exact.py '
               '-f data/B-UWaveGestureLibraryAll-10.pkl '
               '-p %s '
               '-v .3 '
               '-x -l '
               '-m 1000 '
               '-r .1 -o adagrad '
               '--saveall model1/%s.pkl -a %d') % (params, desc, gamma)
    if not update_gp:
        command += ' -u'

    qsub(command,
         job_name=desc[10:-3],
         stdout='stdout/' + desc,
         stderr='stderr/' + desc)


def main():
    for gamma in [4, 2]:
        run(gamma, True)
        run(gamma, False)


if __name__ == '__main__':
    main()
