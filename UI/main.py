import argparse
import sys

from PyQt5.QtWidgets import QApplication

from tools.Extension import FFT
from tools.View import MainWindow
from tools.ViewModel import FileView3D


def getArgs():
    parser = argparse.ArgumentParser(description='Visualize a 3D image file')
    parser.add_argument('-f', '--file', dest='file', type=str, default=False,
                        help='Load image from a file')
    return parser.parse_args()


if __name__ == '__main__':
    args = getArgs()
    # if not args.file:
    #     raise Exception("No File")
    app = QApplication([])
    main = MainWindow(FFT)
    # main = MainWindow(view=FileView3D(args.file, (400, 400)))
    main.show()
    sys.exit(app.exec_())
