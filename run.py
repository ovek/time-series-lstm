import os
import json
import matplotlib.pyplot as plt
from core.data_processor import DataProcessor
from core.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']) : os.makedirs(configs['model']['save_dir'])

    data = DataProcessor(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    model.train(x, y,
                epochs=configs['training']['epochs'],
                batch_size=configs['training']['batch_size'],
                save_dir=".")

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions_pointbypoint = model.predict_point_by_point(x_test)
    plot_results(predictions_pointbypoint, y_test)

    predictions_fullseq = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    plot_results(predictions_fullseq, y_test)


if __name__ == '__main__':
    main()