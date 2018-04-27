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
