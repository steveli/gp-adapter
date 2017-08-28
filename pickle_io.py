import cPickle as pickle


def pickle_load(filename):
    with open(filename, 'rb') as f:
        dat = pickle.load(f)
    #if len(dat) == 0:
    #    return dat[0]
    return dat


def pickle_save(filename, *args):
    with open(filename, 'wb') as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)

