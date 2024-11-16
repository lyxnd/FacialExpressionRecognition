import sys
import os

from src.ui.ui import UI
from PyQt5 import QtWidgets
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


sys.path.append(os.path.dirname(__file__) + '/ui')



def load_cnn_model():
    """
    载入CNN模型
    :return:
    """
    from model import CNN3
    model1 = CNN3()  # 确保 CNN3 是训练时的相同模型架构
    model1.load_weights('./models/fer2013/cnn3_best.weights.h5')  # 完全加载
    # model1.load_weights('./models/ck+/cnn3_best.weights.h5')  # 完全加载
    # model1.load_weights('./models/jaffe/cnn3_best.weights.h5')  # 完全加载
    # model1.load_weights('./models/cnn3_best.keras')  # 完全加载
    return model1



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
