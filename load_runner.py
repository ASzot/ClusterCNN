import pickle

class LoadRunner:
    def __init__(self, callback):
        self.cb = callback

    def __run(self, filename):
        output = self.cb()
        with open(filename, 'wb') as f:
            pickle.dump(output, f)

        return output

    def run_or_load(self, filename, force_create = False):
        if not force_create:
            try:
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            except IOError:
                pass

        return self.__run(filename)
