# PANDA
This is the implementation for the paper "Verification of Incomplete Graph Unlearning
through Adversarial Perturbations".

## Requirements
```
pip install -r requirements.txt
```

## Reproducibility

### Verificatoin Performance

#### Table 1

* Verification with PANDA
```
python experiment.py --task verify-ours -g 0 --dataset citeseer --num-trials 1
```
* Verification with baselines
```
python experiment.py --task baseline -g 0 --dataset citeseer --num-trails 1
```

#### Table 2
* Verification time
```
python experiment.py --task efficiency -g 0 --dataset citeseer --num-trials 1
```

### Factor Analysis
#### Figure 3

* Impact of $\alpha$ and $\beta$ ((a) and (b)).
```
python experiment.py --task empirical -g 0 --dataset citeseer
```
* Incompleteness ratio (c).
```
python experiment.py --task incomplete-ratio -g 0 --dataset citeseer --num-trials 1
```
* Perturbation budget (d).
```
python experiment.py --task num-edges -g 0 --dataset citeseer --num-trials 1

```

#### Table 3
```
python experiment.py --task composition -g 0 --dataset citeseer --num-trials 1
```

### Transferability

#### Table 4
```
python experiment.py --task unlearn -g 0 --dataset citeseer --num-trials 1
```

### Robustness of Verification
#### Table 5
```
python experiment.py --task detect -g 0 --dataset citeseer --num-trials 1
```