import os
from qsub import qsub


def main():
    #log_dir = 'log'
    #if not os.path.isdir(log_dir):
    #    os.mkdir(log_dir, 0755)

    script = 'gpnet_fast.py'

    for data in [
            #'data/UWaveGestureLibraryAll-10.pkl',
            'data/UWaveGestureLibraryAll-1.pkl',
            ]:
        data_id = data.rsplit('-', 1)[-1][:-4]
        for net_arch in ['logreg', 'cnn', 'mlp', 'lstm']:
            # Stochastic train
            job_id = '%s_%s_%s' % (net_arch[:3], data_id, 's')
            qsub('python %s -f %s -e 100 -n %s' %
                    (script, data, net_arch),
                 job_name=job_id,
                 stdout='stdout/' + job_id,
                 stderr='stderr/' + job_id)

            # Deterministic train
            job_id = '%s_%s_%s' % (net_arch[:3], data_id, 'd')
            qsub('python %s -f %s -e 100 -n %s -d' %
                    (script, data, net_arch),
                 job_name=job_id,
                 stdout='stdout/' + job_id,
                 stderr='stderr/' + job_id)


if __name__ == '__main__':
    main()
