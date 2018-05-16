# phission

*phission:* **ph**asing via nuclear "f**ission**"

This package implements [haplotype phasing](https://en.wikipedia.org/wiki/Haplotype_estimation) by formulating it as a low-rank matrix completion problem. `phission` uses minimum nuclear norm as a convex relaxation of low-rank.

## Installation

Clone this repo

```
git clone https://github.com/captaincapsaicin/phission.git
```

From the project root, create a [virtual environment](https://virtualenv.pypa.io/en/stable/) using `virtualenv`

```
cd phission
virtualenv venv
```

Activate the virtual environment
```
source venv/bin/activate
```

Install `numpy` before `msprime` (msprime installation depends on it)
```
pip install numpy --no-cache-dir
pip install msprime --no-cache-dir
```

Then install the rest of the dependencies (this may take some time)
```
pip install -r requirements.txt
```

## Running the example

```
python run_phission.py --num-haps 100 --num-snps 60 --seed 1
```

For help with command line options, run:

```
python run_phission.py -h
```

## Running beagle for comparison

Make sure you have the beagle jar in your working directory

```
wget https://faculty.washington.edu/browning/beagle/beagle.27Jan18.7e1.jar
```

Then run the script
```
python run_beagle.py --num-haps 100 --num-snps 60 --seed 1
```

## Running a suite of experiments

To run a series of experiments (to collect some statistics on performance), run

```
python run_experiments.py --num-experiments 8 --recombination-rate 0.0 --haps-snps 10,10 20,20
```

Which writes the statistics to a pickle file, which you can then examine with:

```
import pickle

import numpy as np
from tabulate import tabulate


def get_stats_back(num_haps_snps_list):
    with open('beagle_stats.pkl', 'rb') as f:
        beagle_stats = pickle.load(f)
    with open('phission_stats.pkl', 'rb') as f:
        phission_stats = pickle.load(f)

    for haps_snps in num_haps_snps_list:
        print()
        headers = [str(haps_snps), 'rank reduction', 'percent switch error']
        data = [['phission', np.mean(phission_stats[haps_snps]['rank_true'] - phission_stats[haps_snps]['rank_phased']), np.mean(phission_stats[haps_snps]['switch_error'] / phission_stats[haps_snps]['positions_phased'])],
                ['beagle', np.mean(beagle_stats[haps_snps]['rank_true'] - beagle_stats[haps_snps]['rank_phased']), np.mean(beagle_stats[haps_snps]['switch_error'] / beagle_stats[haps_snps]['positions_phased'])]]
        print(tabulate(data, headers=headers))
```

Followed by

```
In [7]: num_haps_snps_list = [(10, 10), (20, 20)]

In [8]: get_stats_back(num_haps_snps_list)
(10, 10)      rank reduction    percent switch error
----------  ----------------  ----------------------
phission               0.625                0.34059
beagle                -0.375                0.165035

(20, 20)      rank reduction    percent switch error
----------  ----------------  ----------------------
phission               0.125                0.306674
beagle                -0.625                0.101885
```

## Running tests

```
py.test
```

## Future work

Parallelize optimization using ADMM, performing phasing over overlapping windows, and then sensibly combining their results.

## Advanced Notes

`phission` is compatible with Python 2 and 3.

If you want to put `ipdb` hooks in the code, remember to run `pytest` without swallowing output flag.

```
py.test -s
```
