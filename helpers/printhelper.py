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


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end='')
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end='')

    print('')

    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end='')

        cell_txt = ''

        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end='')

        print('')


