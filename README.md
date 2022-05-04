# Safe-FinRL

This is the open source implementation of a few important multi-step deep RL algorithms
discussed in the [ICML 2021 paper](https://arxiv.org/abs/2103.00107). We implement these
algorithms by Pytorch and mainly in combination with [TD3](https://arxiv.org/abs/1802.09477) and [SAC](https://arxiv.org/abs/1801.01290), which are currently popular actor-critic algorithms for continuous control.
Moreover, we apply the multi-step deep RL algorithms to the Financial Environment inspired by
[FinRL](https://github.com/AI4Finance-Foundation/FinRL)

This code implements a few multi-step algorithms to reduce bias and variance, including

* [Peng's Q(lambda)](https://link.springer.com/content/pdf/10.1023/A:1018076709321.pdf)
* [Uncorrected n-step](https://arxiv.org/pdf/1710.02298.pdf)
* [Retrace](https://arxiv.org/abs/1606.02647)
* [Tree-backup](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs)
* Undamped Importance Sampling

## Installation


## Code structure


## Running the code

For a length of look back window as 10 (```l=10```),
to run Retrace-SAC, with n-step buffer with ```n=60```, run the following
```sh
python trace_main.py --trace_type='retrace' --lambda_=0.8 --nsteps=60 --look_back=10
```

To run Peng's Q(lambda) with the same setting we have
```sh
python trace_main.py --trace_type='qlambda' --lambda_=0.8 --nsteps=60 --look_back=10
```

To run Tree-backup, we have

```sh
python trace_main.py --trace_type='treebackup' --nsteps=60 --look_back=10
```

To run uncorrected n-step, we have
```sh
python trace_main.py --trace_type='qlambda' --lambda_=1.0 --nsteps=60 --look_back=10
```

To run pure Importance Sampling, we have
```sh
python trace_main.py --trace_type='IS' --lambda_=1.0 --nsteps=60 --look_back=10
```

To run 1-step SAC we have
```sh
python trace_main.py --nsteps=1 --look_back=10
```

To run 1-step TD3 we have
```sh
python trace_main.py --policy_type='Deterministic' --nsteps=1 --look_back=10
```

Some commonly used variables
```sh
python trace_main.py --policy_type [str] \
                     --trace_type  [str] \
                     --look_back   [int] \
                     --cuda        [bool] \
                     --nsteps      [int] \
                     --lambda_     [int] \
                     --lr_actor    [int] \
                     --lr_critic   [int] \
                     --lr_alpha    [int] \
                     --episodes    [int] \
                     --reps        [int]
```

Finally, we suggest using tensorboard to visualize the loss during training by
```sh
tensorboard --logdir=./runs
```
