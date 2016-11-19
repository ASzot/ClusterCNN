import numpy as np

class DiscriminatoryFilter(object):
    CUTOFF = 2000

    def __init__(self, min_variance, selection_percent):
        self.min_variance = min_variance
        self.selection_percent = selection_percent

    def filter_samples(self, samples):
        sample_variances = [(sample, np.var(sample)) for sample in samples]
        variances = [sample_variance[1] for sample_variance in sample_variances]
        sample_variances = [(sample, variance) for sample, variance in sample_variances if variance > self.min_variance]
        selection_count = int(len(sample_variances) * self.selection_percent)
        # Order by variance.
        # Sort with the highest values first.
        sample_variances = sorted(sample_variances, key = lambda x: -x[1])
        samples = [sample_variance[0] for sample_variance in sample_variances]
        samples = samples[0:selection_count]
        if selection_count > self.CUTOFF:
            print '-----Greater than the cutoff randomly sampling'
            selected_samples = []
            for i in np.arange(self.CUTOFF):
                select_index = np.random.randint(len(samples))
                selected_samples.append(samples[select_index])
                del samples[select_index]
            return selected_samples
        else:
            return samples
