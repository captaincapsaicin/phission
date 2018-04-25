# phission

This repo does genotype phasing (conversion from genotype to haplotype) using a convex relaxation of the problem.

We formulate phasing as a matrix completion problem with constraints. We do matrix completion on overlapping windows, trying to minimize the nuclear norm, a convex proxy for rank, in the process.

Ideas for optimization:
- Optimize each of the windows separately using ADMM

## To run tests:

```
py.test
```

If you want to put `ipdb` hooks in the code, remember to run pytest without swallowing output flag.

```
py.test -s
```

## Running the example

```
python phission.py --num-haps 100 --num-snps 60
```

## To run beagle:

Make sure you have the beagle jar in your working directory

```
wget https://faculty.washington.edu/browning/beagle/beagle.27Jan18.7e1.jar
```

Then run the script
```
python beagle.py --num-haps 100 --num-snps 60
```

## A note on installation:

You may have to install numpy first (msprime installation depends on it)
```
pip install numpy --no-cache-dir
pip install msprime --no-cache-dir
```

Then install the reset of the dependencies
```
pip install -r requirements.txt
```
