import argparse
import itertools as iter
import copy

#TODO: Add hyperparameter presets, i.e, predefined sets of parameter values with a name
# (e.g. 'baseline', 'M1'...).
class ModelParams:

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._hyperparams = []

    def __repr__(self):
        return self._parser.format_usage()

    def add_param(self, name, *args, **kwargs):
        self._parser.add_argument(f'--{name}', *args, **kwargs)
        return self

    def add_hyperparam(self, name, *args, **kwargs):
        self._hyperparams.append(name)
        return self.add_param(name, *args, nargs='+', **kwargs)

    def parse_args(self, *args, **kwargs) -> 'ModelArgs':
        args = self._parser.parse_args(*args, **kwargs)
        return ModelArgs(args, self._hyperparams)

    def get_defaults(self) -> 'Args':
        return self._parser.parse_args([])

class Args(argparse.Namespace):
    pass

class ModelArgs:
    def __init__(self, args, hyperparams, combs = None):
        self._args = args
        self._hyperparams = hyperparams
        self._hyper_combs = combs

    def _compute_combs(self):
        hyperargs = []
        for hyperparam in self._hyperparams:
            hyperarg = getattr(self._args, hyperparam)
            hyperargs.append(hyperarg if isinstance(hyperarg,list) else [hyperarg])
        self._hyper_combs = list(iter.product(*hyperargs))

    def get_num_combs(self):
        if (self._hyper_combs == None):
            self._compute_combs()
        return len(self._hyper_combs)

    def get_comb(self, index) -> Args:
        if (self._hyper_combs == None):
            self._compute_combs()

        if (index >= self.get_num_combs()):
            return None

        args = copy.copy(self._args)
        comb = self._hyper_combs[index]

        for i, hyperparam in enumerate(self._hyperparams):
            setattr(args, hyperparam, comb[i])
        return args

    def get(self) -> Args:
        return self.get_comb(0)

    def get_comb_partition(self, index, num_partitions) -> 'ModelArgs':
        if (self._hyper_combs == None):
            self._compute_combs()
        if (index >= num_partitions):
            return None
        num_combs = self.get_num_combs()
        min_partition_size = num_combs // num_partitions
        remainder = num_combs - (min_partition_size*num_partitions)
        comb_start = index * min_partition_size + min(index, remainder)
        comb_end = comb_start + min_partition_size + (1 if remainder-index > 0 else 0)

        combs = self._hyper_combs[comb_start: comb_end]
        return ModelArgs(self._args, self._hyperparams, combs)




