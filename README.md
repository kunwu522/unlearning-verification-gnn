# PANDA
This is the implementation for the paper "Probabilistic Verification of Untrusted Graph Unlearning".

## Requirements
```
pip install -r requirements.txt
```

## Reproducibility

### Verificatoin Performance

#### Table 2
```
python experiment.py --task empirical -g 0 --dataset citeseer
```

#### Table 3
```
python experiment.py --task estimate -g 0 --dataset citeseer
```

### Impact of Various Factors on Verification Performance

#### Figure 4
```
python experiment.py --task incomplete-ratio -g 0 --dataset citeseer --num-trials 1

python experiment.py --task num-edges -g 0 --dataset citeseer --num-trials 1

```

#### Table 4
```
python experiment.py --task surrogate -g 0 --dataset citeseer --num-trials 1
```

#### Table 5
```
python experiment.py --task unlearn -g 0 --dataset citeseer --num-trials 1
```

### Robustness of Verification
```
python experiment.py --task detect -g 0 --dataset citeseer --num-trials 1
```