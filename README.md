# Scalable GP Adapter for Time Series Classification

Implementation of the paper
["A scalable end-to-end Gaussian process adapter for irregularly sampled time series classification"](https://arxiv.org/abs/1606.04443).

## Requirements

* [Python 2.7](https://www.python.org/downloads/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Theano](http://www.deeplearning.net/software/theano/)
* [Lasagne](https://lasagne.readthedocs.io/)
* [Cython](http://cython.org/):
  run the command `python setup.py build_ext --inplace` to compile
  Cython modules.

## Examples

Train GP adapter with a convolutional net for 100 epochs on the sparsified
uWave data:

```
python gpnet_fast.py -f data/UWaveGestureLibraryAll-1.pkl -e 100 -n cnn
```

Run `python gpnet_fast.py -h` to learn the command line options.
