class HyperParamSearch:
    def __init__(self, model, eval_func_name, hyper_params_range):
       self.model = model
       self.eval_func_name = eval_func_name
       self.hyper_params_range = hyper_params_range


    def search(self):
        all_param_names = list(self.hyper_params_range.keys())
        return self.__recur_search(0, all_param_names)


    def __recur_search(self, cur_index, all_param_names):
        param_name = all_param_names[cur_index]

        accuracies = []

        for param_value in self.hyper_params_range[param_name]:
            self.model.set_hyperparam(param_name, param_value)
            if cur_index == (len(all_param_names) - 1):
                eval_func = self.__get_eval_func()
                accuracies.append(eval_func())
            else:
                accuracies.append(self.__recur_search(cur_index + 1, all_param_names))

        return accuracies


    def __get_eval_func(self):
        try:
            return getattr(self.model, self.eval_func_name)
        except AttributeError:
            print 'Could not find evaluation function'
            return None

