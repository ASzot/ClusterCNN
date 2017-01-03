import numpy as np
from helpers.printhelper import PrintHelper as ph

class DiscriminatoryFilter(object):
    CUTOFF = None

    def __init__(self, min_variance = None, selection_percent = None):
        self.min_variance = min_variance
        self.selection_percent = selection_percent


    def filter_samples(self, samples):
        if self.min_variance is None or self.selection_percent is None:
            ph.disp('Skipping discriminatory filter', ph.FAIL)
            return samples

        ph.disp(('-' * 5) + 'Filtering input.')
        before_len = len(samples)

        overall_var = np.std(samples)
        overall_avg = np.mean(samples)

        # self.min_variance = overall_var / 4.0

        ph.disp('Var: %.5f, Avg: %.5f' % (overall_var, overall_avg), ph.OKGREEN)

        ph.disp('-----Min variance: %.5f, Select: %.5f%%' % (self.min_variance, (self.selection_percent * 100.)))
        ph.disp('-----Starting with %i samples' % len(samples))
        sample_variances = [(sample, np.std(sample)) for sample in samples]
        variances = [sample_variance[1] for sample_variance in sample_variances]
        prev_len = len(variances)
        sample_variances = [(sample, variance) for sample, variance in sample_variances if variance > self.min_variance]

        ph.disp('-----%i samples discarded from min variance' % (len(sample_variances) - prev_len))

        selection_count = int(len(sample_variances) * self.selection_percent)
        ph.disp('-----Trying to select %i samples' % selection_count)
        # Order by variance.
        # Sort with the highest values first.
        sample_variances = sorted(sample_variances, key = lambda x: -x[1])
        samples = [sample_variance[0] for sample_variance in sample_variances]
        samples = samples[0:selection_count]
        if (self.CUTOFF is not None) and selection_count > self.CUTOFF:
            print '-----Greater than the cutoff randomly sampling'
            selected_samples = []
            for i in np.arange(self.CUTOFF):
                select_index = np.random.randint(len(samples))
                selected_samples.append(samples[select_index])
                del samples[select_index]
            samples = selected_samples

        after_len = len(samples)
        ph.disp(('-' * 5) + '%i reduced to %i' % (before_len, after_len))

        return samples

