# Phasing via Nuclear Norm

This repo does genotype phasing (conversion from genotype to haplotype) using a convex relaxation of the problem.

We formulate phasing as a matrix completion problem with constraints. We do matrix completion on overlapping windows, trying to minimize the nuclear norm, a convex proxy for rank, in the process.

Ideas for optimization:
- Optimize each of the windows separately using ADMM

To run tests:

```
py.test
```

If you want to put `ipdb` hooks in the code, remember to run pytest without swallowing output flag.

```
py.test -s
```


## Installation Note:

`msprime` requires `numpy` to be installed BEFORE it is installed. If you're getting warnings like


```
SystemError: Function not available without numpy
```

Then you may have to recreate your virtualenv from scratch, and run these commands *in order*:

```
pip install numpy --no-cache
pip install msprime --no-cache
```
