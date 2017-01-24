import numpy as np
from sklearn.feature_selection import VarianceThreshold
from helpers.printhelper import PrintHelper as ph

class DiscriminatoryFilter(object):
    CUTOFF = None

    def __init__(self, min_variance = None, selection_percent = None):
        self.min_variance = min_variance
        self.selection_percent = selection_percent

    def custom_filter(self, samples):
        ph.disp('Getting sample variances')
        sample_variances = [(sample, np.std(sample)) for sample in samples]
        variances = [sample_variance[1] for sample_variance in sample_variances]

        overall_var = np.std(samples)
        overall_avg = np.mean(samples)
        per_sample_var = np.std(variances)
        per_sample_avg = np.mean(variances)

        thresh_fact = 1.0
        self.min_variance = per_sample_avg + (thresh_fact * per_sample_var)
        self.min_variance = 0.0

        ph.disp('STD: %.5f, Avg: %.5f' % (overall_var, overall_avg), ph.OKGREEN)
        ph.disp('Per Sample STD: STD: %.5f, Avg: %.5f' % (per_sample_var, per_sample_avg), ph.OKGREEN)

        if self.min_variance is None or self.selection_percent is None:
            ph.disp('Skipping discriminatory filter', ph.FAIL)
            return samples

        ph.disp(('-' * 5) + 'Filtering input.')

        ph.disp('-----Min variance: %.5f, Select: %.5f%%' % (self.min_variance, (self.selection_percent * 100.)))
        ph.disp('-----Starting with %i samples' % len(samples))
        prev_len = len(variances)
        sample_variances = [(sample, variance) for sample, variance in sample_variances if variance > self.min_variance]

        ph.disp('-----%i samples discarded from min variance' % (prev_len - len(sample_variances)))

        selection_count = int(len(sample_variances) * self.selection_percent)
        ph.disp('-----Trying to select %i samples' % selection_count)
        # Order by variance.
        # Sort with the highest values first.
        sample_variances = sorted(sample_variances, key = lambda x: -x[1])
        samples = [sample_variance[0] for sample_variance in sample_variances]
        samples = samples[0:selection_count]
        #self.selection_percent = int(self.selection_percent)
        #samples = samples[0:self.selection_percent]

        if (self.CUTOFF is not None) and selection_count > self.CUTOFF:
            ph.disp('-----Greater than the cutoff randomly sampling')
            selected_samples = []
            for i in np.arange(self.CUTOFF):
                select_index = np.random.randint(len(samples))
                selected_samples.append(samples[select_index])
                del samples[select_index]
            samples = selected_samples

        return samples


    def filter_samples(self, samples):
        before_len = len(samples)

        use_custom = True;

        if use_custom:
            samples = self.custom_filter(samples)
        else:
            ph.disp('Threshold variance of %.6f' % self.min_variance)
            selector = VarianceThreshold(self.min_variance)
            samples = selector.fit_transform(samples)

        after_len = len(samples)
        ph.disp(('-' * 5) + '%i reduced to %i' % (before_len, after_len))

        return samples

