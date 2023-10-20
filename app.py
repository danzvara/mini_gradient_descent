import argparse
import numpy as np
import matplotlib.pyplot as plt
import nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_x", type=str)
    parser.add_argument("train_y", type=str)
    return parser.parse_args()


def load_data(train_x_filename, train_y_filename):
    with open(train_x_filename, "rb") as f:
        data_buffer = f.read()
    with open(train_y_filename, "rb") as f:
        labels_buffer = f.read()

    num_examples = int.from_bytes(labels_buffer[4:8])
    labels = np.frombuffer(labels_buffer, dtype=np.uint8, offset=8)

    w, h = int.from_bytes(data_buffer[8:12]), int.from_bytes(data_buffer[12:16])
    raw_image_data = np.frombuffer(data_buffer, dtype=np.uint8, offset=16)

    image_data = raw_image_data.reshape(num_examples, w * h)

    images = np.zeros((num_examples, h, w))
    for i in range(num_examples):
        images[i] = image_data[i].reshape(h, w)

    return images, labels


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

    images, labels = load_data(args.train_x, args.train_y)
    shuffled_training_set = list(zip(images, labels))
    np.random.shuffle(shuffled_training_set)

    input_size = 28 * 28

    layers = [
        nn.LinearLayer(128, input_size),
        nn.LinearLayer(30, 128),
        nn.LinearLayer(10, 30),
    ]

    losses = []

    for i in range(10):
        for image, label in shuffled_training_set:
            standardized_image = np.array(image).reshape(input_size, 1)
            standardized_image = standardized_image / np.sum(standardized_image)

            target_y = np.zeros(10).reshape(10, 1)
            target_y[label] = 1.0

            # forward pass
            y = standardized_image
            for layer in layers:
                y = layer.forward(y)

            l = loss(y, target_y)
            l_dy = loss_dy(y, target_y)

            for layer in layers[::-1]:
                l_dy = layer.backprop(l_dy)

            losses.append(l)

    image, label = shuffled_training_set[1000]
    input_vector = image.reshape(input_size, 1)
    input_vector = input_vector / np.sum(input_vector)

    y = input_vector
    for layer in layers:
        y = layer.forward(y)

    print("Prediction:", y.argmax(), "Label:", label)
    print(y)

    plt.plot(losses)
    plt.show()
