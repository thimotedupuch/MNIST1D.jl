# MNIST1D.jl

MNIST1D.jl provides a small, reproducible and configurable Julia dataset for machine learning research. Each observation is a one-dimensional signal representing a transformed digit template. The default dataset has 4,000 training examples and 1,000 test examples, each with 40 features. Generation happens in memory when `dataset()` is called, so no data download or local cache is required.

## Installation

MNIST1D.jl is not registered. Install it directly from GitHub using Julia's package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/thimotedupuch/MNIST1D.jl")
```

Alternatively, press `]` in the Julia REPL and run:

```julia-repl
pkg> add https://github.com/thimotedupuch/MNIST1D.jl
```

## Usage

```julia
using MNIST1D

data = dataset()                          # defaults from the paper
train_x, train_y = data.train             # 4000×40, 4000 labels
test_x, test_y = data.test                # 1000×40, 1000 labels

data = dataset(num_samples=2_000,
               final_seq_length=64,
               max_translation=20,
               shuffle_seq=true,
               seed=123)
```

`data.x`, `data.y`, `data.x_test`, `data.y_test`, `data.t`, and `data.templates` are also available directly. Generation applies random padding, scaling, translation, correlated noise, independent noise, and shear. `DatasetOptions` can be used when passing options as an object:

```julia
options = DatasetOptions(num_samples=5_000, corr_noise_scale=0.1)
data = make_dataset(options)
```

The implementation uses only Julia's standard library (`Random` and `Statistics`). Labels are zero-based integers from 0 to 9, matching the reference implementation. `num_samples` is rounded down to a balanced number divisible by ten.

## Attribution

MNIST-1D was introduced in:

> Sam Greydanus and Dmitry Kobak. “Scaling down deep learning with MNIST-1D.” *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2011.14439

Please cite this paper when using MNIST-1D. The upstream transformation implementation credits Sam Greydanus and Peter Steinbach. This Julia package is an independent Julia implementation of the published dataset procedure.

```bibtex
@inproceedings{greydanus2024scaling,
  title={Scaling down deep learning with {MNIST}-{1D}},
  author={Greydanus, Sam and Kobak, Dmitry},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```

## License

Released under the Apache License, Version 2.0. See [LICENSE](LICENSE).
