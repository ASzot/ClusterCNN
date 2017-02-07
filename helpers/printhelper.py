class PrintHelper(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    DISP = True

    @staticmethod
    def disp(txt, txt_type = None):
        if not PrintHelper.DISP:
            return
        final_str = ""
        if txt_type is not None:
            final_str += txt_type

        final_str += str(txt)

        if txt_type is not None:
            final_str += PrintHelper.ENDC

        print(final_str)

    @staticmethod
    def linebreak():
        if not PrintHelper.DISP:
            return
        print('\n' * 2)
