import numpy as np
import sys
from relu import ReLu
from conv2d import Conv2D
from fullyConnect import FullyConnect
from maxpool2d import MaxPooling2x2
from crossEntropy import CrossEntropy
from Dataset import DataSet


def main() -> int:
    # read data
    mnistDataSet = DataSet()

    # construct neural network
    conv1 = Conv2D(1, 5, 32)
    reLu1 = ReLu()
    pool1 = MaxPooling2x2()
    conv2 = Conv2D(32, 5, 64)
    reLu2 = ReLu()
    pool2 = MaxPooling2x2()
    fc1 = FullyConnect(7*7*64, 512)
    reLu3 = ReLu()
    fc2 = FullyConnect(512, 10)
    lossfunc = CrossEntropy()

    # train
    lr = 1e-2
    for epoch in range(10):
        for i in range(600):
            train_data, train_label = mnistDataSet.next_batch(100)

            # forward
            CNN = conv1.forward(train_data)
            CNN = reLu1.forward(A)
            CNN = pool1.forward(A)
            CNN = conv2.forward(A)
            CNN = reLu2.forward(A)
            CNN = pool2.forward(A)
            CNN = CNN.reshape(CNN.shape[0], 7*7*64)
            CNN = fc1.forward(CNN)
            CNN = reLu3.forward(CNN)
            CNN = fc2.forward(CNN)
            loss = lossfunc.forward(CNN, train_label)

            # backward
            grad = lossfunc.backward()
            grad = fc2.backward(grad)
            grad = reLu3.backward(grad)
            grad = fc1.backward(grad)
            grad = grad.reshape(grad.shape[0], 7, 7, 64)
            grad = pool2.backward(grad)
            grad = reLu2.backward(grad)
            grad = conv2.backward(grad)
            grad = grad.copy()
            grad = pool1.backward(grad)
            grad = reLu1.backward(grad)
            grad = conv1.backward(grad)

            # update parameters
            fc2.update_parameters(lr)
            fc1.update_parameters(lr)
            conv2.update_parameters(lr)
            conv1.update_parameters(lr)

            if (i + 1) % 100 == 0:
                test_index = 0
                sum_accu = 0
                for j in range(100):
                    test_data, test_label = mnistDataSet.test_data[test_index: test_index + 100], \
                        mnistDataSet.test_label[test_index: test_index + 100]
                    CNN = conv1.forward(test_data)
                    CNN = reLu1.forward(CNN)
                    CNN = pool1.forward(CNN)
                    CNN = conv2.forward(CNN)
                    CNN = reLu2.forward(CNN)
                    CNN = pool2.forward(CNN)
                    CNN = CNN.reshape(CNN.shape[0], 7 * 7 * 64)
                    CNN = fc1.forward(CNN)
                    CNN = reLu3.forward(CNN)
                    CNN = fc2.forward(CNN)
                    preds = lossfunc.cal_softmax(CNN)
                    preds = np.argmax(preds, axis=1)
                    sum_accu += np.mean(preds == test_label)
                    test_index += 100
                print("epoch{} train_number{} accuracy: {}%".format(
                    epoch+1, i+1, sum_accu))

    return 0


if __name__ == '__main__':
    sys.exit(main())
