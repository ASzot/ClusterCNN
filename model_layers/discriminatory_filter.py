import numpy as np
import datetime
from helpers.printhelper import PrintHelper as ph

class DiscriminatoryFilter(object):
    CUTOFF = [5000, 176, 135, 176, 500]
    cur_layer = 2
    use_select_count = False

    def __init__(self, selection_percent = None):
        """
        Constructor

        :param selection_percent: floating point value in [0.0, 1.0]
        the percentage of elements sorted by variance to return.
        """
        self.selection_percent = selection_percent


    def custom_filter(self, samples):
        """
        Filter the samples based off of the selection percentage,
        an optional min_variance, and an optional CUTOFF to determine
        the max number of values that are selected.

        :param samples: The list of samples to be filtered.
        :return the selected samples
        """

        ph.disp('Getting sample variances')

        variances = np.var(samples, axis=1)
        sample_variances = list(zip(samples, variances))

        #overall_var = np.var(samples)
        #overall_avg = np.mean(samples)
        per_sample_std = np.std(variances)
        per_sample_avg = np.mean(variances)

        #thresh_fact = 0.0
        #min_variance = per_sample_avg + (thresh_fact * per_sample_var)
        min_variance = 0.0

        #ph.disp('STD: %.5f, Avg: %.5f' % (overall_var, overall_avg), ph.OKGREEN)
        #ph.disp('Per Sample STD: STD: %.5f, Avg: %.5f' % (per_sample_var, per_sample_avg), ph.OKGREEN)

        if self.selection_percent is None:
            ph.disp('Skipping discriminatory filter', ph.FAIL)
            return samples

        ph.disp(('-' * 5) + 'Filtering input.')

        if not self.use_select_count:
            ph.disp('-----Min variance: %.5f, Select: %.5f%%' % (min_variance, (self.selection_percent * 100.)))

        ph.disp('-----Starting with %i samples' % len(samples))

        prev_len = len(variances)

        if min_variance != 0.0:
            # Discard due to minimum variance.
            sample_variances = [(sample, variance) for sample, variance in sample_variances
                    if variance > min_variance]

            ph.disp('-----%i samples discarded from min variance' % (prev_len - len(sample_variances)))
            ph.disp('-----%i samples remain' % len(sample_variances))

        if not self.use_select_count:
            selection_count = int(len(sample_variances) * self.selection_percent)
        else:
            selection_count = int(self.selection_percent)

        ph.disp('-----Trying to select %i samples' % selection_count)

        # Order by variance.
        # Sort with the highest values first.
        ph.disp('-----Starting sort')
        start = datetime.datetime.now()

        toss_thresh = per_sample_avg + (per_sample_std * 2)
        sample_variances = [(sample, variance) for sample, variance in sample_variances if variance > toss_thresh]
        ph.disp('-----Filtered out values to make sorting easier')
        sample_variances = sorted(sample_variances, key = lambda x: -x[1])

        delta = datetime.datetime.now() - start
        seconds = delta.seconds
        microseconds = delta.microseconds % 100
        ph.disp('-----Sort ended (%s s : %s ms)' % (str(seconds), str(microseconds)))

        samples = [sample_variance[0] for sample_variance in sample_variances]
        samples = samples[0:selection_count]
        #self.selection_percent = int(self.selection_percent)
        #samples = samples[0:self.selection_percent]

        # An optional cutoff parameter to only select CUTOFF values.
        # For slower computers with not as much RAM and processing power.
        ph.disp('The current cutoff is ' + str(self.CUTOFF[self.cur_layer]))
        if (self.CUTOFF is not None) and (self.CUTOFF[self.cur_layer] is not None) and selection_count > self.CUTOFF[self.cur_layer]:
            ph.disp('-----Greater than the cutoff randomly sampling')
            selected_samples = []
            cur_cutoff = self.CUTOFF[self.cur_layer]

            for i in np.arange(cur_cutoff):
                select_index = np.random.randint(len(samples))
                selected_samples.append(samples[select_index])
                del samples[select_index]
            samples = selected_samples

        self.cur_layer += 1

        return samples


    def filter_samples(self, samples):
        """
        Wrapper for the custom_filter that outputs debug information
        """

        before_len = len(samples)

        samples = self.custom_filter(samples)

        after_len = len(samples)
        ph.disp(('-' * 5) + '%i reduced to %i' % (before_len, after_len))

        return samples

