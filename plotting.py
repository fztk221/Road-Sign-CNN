from matplotlib import pyplot as plt

def plotting(history, model, X_test, y_test):
    # plotting
    plt.figure(1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score:', score[0])
    print('Test Accuracy:', score[1])