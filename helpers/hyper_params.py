class HyperParamData:
    def __init__(self, input_shape, subsample, patches_subsample, filter_size, batch_size,
            nkerns, fc_sizes, n_epochs, selection_percentages, use_filters,
            activation_func, extra_path, should_set_weights, should_eval, remaining, cluster_count):
        self.input_shape           = input_shape
        self.subsample             = subsample
        self.patches_subsample     = patches_subsample
        self.filter_size           = filter_size
        self.batch_size            = batch_size
        self.nkerns                = nkerns
        self.fc_sizes              = fc_sizes
        self.n_epochs              = n_epochs
        self.selection_percentages = selection_percentages
        self.use_filters           = use_filters
        self.activation_func       = activation_func
        self.extra_path            = extra_path
        self.should_set_weights    = should_set_weights
        self.should_eval           = should_eval
        self.remaining             = remaining
        self.cluster_count         = cluster_count
