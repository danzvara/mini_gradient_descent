import argparse
import numpy as np
import matplotlib.pyplot as plt
import linear_layer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_x", type=str)
    parser.add_argument("train_y", type=str)
    parser.add_argument("test_x", type=str)
    parser.add_argument("test_y", type=str)
    return parser.parse_args()


def load_data(args):
    with open(args.train_x, "rb") as f:
        data_buffer = f.read()
    with open(args.train_y, "rb") as f:
        labels_buffer = f.read()
    with open(args.test_x, "rb") as f:
        test_data_buffer = f.read()
    with open(args.test_y, "rb") as f:
        test_labels_buffer = f.read()

    num_examples = int.from_bytes(labels_buffer[4:8])
    num_test_samples = int.from_bytes(test_labels_buffer[4:8])
    labels = np.frombuffer(labels_buffer, dtype=np.uint8, offset=8)
    test_labels = np.frombuffer(test_labels_buffer, dtype=np.uint8, offset=8)

    w, h = int.from_bytes(data_buffer[8:12]), int.from_bytes(data_buffer[12:16])
    raw_image_data = np.frombuffer(data_buffer, dtype=np.uint8, offset=16)
    raw_test_image_data = np.frombuffer(test_data_buffer, dtype=np.uint8, offset=16)

    image_data = raw_image_data.reshape(num_examples, w * h, 1)
    test_image_data = raw_test_image_data.reshape(num_test_samples, w * h, 1)

    return image_data, labels, test_image_data, test_labels


def loss(output_y, target_y):
    assert (
        output_y.shape == target_y.shape
    ), f"Shapes {output_y.shape} and {target_y.shape} do not match"

    mse = (1 / 2) * (np.square(output_y - target_y).mean())

    return mse


def loss_dy(output_y, target_y):
    return output_y - target_y


if __name__ == "__main__":
    args = parse_args()

    train_x, train_y, test_x, test_y = load_data(args)
    train_x, test_x = train_x / 255.0, test_x / 255.0
    whole_dataset = np.concatenate([train_x, test_x])
    mean = whole_dataset.mean()
    std = whole_dataset.std()

    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    shuffled_training_set = list(zip(train_x, train_y))
    np.random.shuffle(shuffled_training_set)

    input_size = 28 * 28

    layers = [
        linear_layer.LinearLayer(128, input_size),
        linear_layer.LinearLayer(30, 128),
        linear_layer.LinearLayer(10, 30, "sigmoid"),
    ]

    losses = []
    accuracies = []

    for i in range(5):
        for image, label in shuffled_training_set:
            target_y = np.zeros(10).reshape(10, 1)
            target_y[label] = 1.0

            # forward pass
            y = np.array(image).reshape(input_size, 1)
            for layer in layers:
                y = layer.forward(y)

            l = loss(y, target_y)
            l_dy = loss_dy(y, target_y)

            for layer in layers[::-1]:
                l_dy = layer.backprop(l_dy)

            losses.append(l)

    accuracy = 0
    for test_x_, test_y_ in zip(test_x, test_y):
        y = test_x_
        for layer in layers:
            y = layer.forward(y)
        accuracy += 1 if (y.argmax() == test_y_) else 0
        print(y.argmax())
    accuracy = accuracy / len(test_x)
    print("Accuracy:", accuracy)

    image, label = shuffled_training_set[1000]
    y = image
    for layer in layers:
        y = layer.forward(y)

    print("Prediction:", y.argmax(), "Label:", label)
    print(y)
    print(y.sum())

    plt.plot(losses)
    plt.show()
