# predicure

A sketch-based semi-Markov learner with approximate duration estimation.

## Overview

This module implements an on-line (streaming) data structure for 'learning'
semi-Markov models. A Markov model is a stochastic model used to model randomly
changing systems where it is assumed that future states depend only on the
current state, and not on the events (states) that occurred before it [[wikipedia]](https://en.wikipedia.org/wiki/Markov_model). A semi-Markov process is one in which the probability of there being a change in state additionally depends on the amount of time that has elapsed since entry into the *current* state.

The `SemiMarkov` data structure within this module is designed to take in state change events (`state 1 -> state 2`) and their corresponding holding times (`duration`), and update a simple semi-Markov model. As each new event is added to the data structure, the data structure...

1. Updates `state 1` duration distribution, and
2. Updates the `state 1` transition `counter` and set of possible transition states `state 2`

There are tons of resources for computing Markov models, a quick Google will find lots of implementations.

## Installation

`predicure` has not yet been uploaded to [PyPi](https://pypi.python.org/pypi),
as we are currently at the 'pre-release' stage\*. Having said that you should be
able to install it via `pip` directly from the GitHub repository with:

```bash
pip install git+git://github.com/carsonfarmer/predicure.git
```

You can also install `predicure` by cloning the
[GitHub repository](https://github.com/carsonfarmer/predicure) and using the
setup script:

```bash
git clone https://github.com/carsonfarmer/predicure.git
cd addc
python setup.py install
```

Note that `predicure` is written for Python 3 only. While it may work in earlier
versions of Python, no attempt has been made to make it Python 2.x compatible.

\* *This means the API is not set, and subject to crazy changes at any time!*

## Testing

`predicure` comes with a <del>comprehensive</del> a very basic range
of tests. To run the tests, you can use [`py.test`](http://pytest.org/latest/)
(maybe also `nosetests`?), which can be installed via `pip` using the
`recommended.txt` file (note, this will also install some other stuff (`numpy`,
`scipy`, and `matplotlib`) which are all great and useful for
tests and examples):

```bash
pip install -r recommended.txt
py.test predicure
```

## Features

In the following examples we use the `random` module to generate data.

```python
from predicure import SemiMarkov
import random
```

### Basics

The simplest way to use a `SemiMarkov` data-structure is to initialize one
and then update it with data points (via the `update` method). In this first
example... and add them to an `SemiMarkov` object:

```python
ac = SemiMarkov().batch(data)  # Add paths all at once...
```

We can then add additional data, and start to query the data-structure. As transitions are added, the data-structure responds and updates
accordingly.

To illustrate the use of this data-structure, here is an example plot using
the `cluster_data` from above:

```python
import matplotlib.pyplot as plt

plt.figure()
mod.plot()
plt.show()
```
![](output.png)

## License

Copyright © 2016, [Carson J. Q. Farmer](http://carsonfarmer.com/)  
Licensed under the [BSD-2 License](https://opensource.org/licenses/BSD-2-Clause).  
