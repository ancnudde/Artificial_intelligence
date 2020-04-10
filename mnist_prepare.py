import pickle
import numpy as np


def import_mnist(file_path):
    """
    Imports the mnist data.

    Args:
       file_path Path to the mnist pickle file.

    Returns:
        The training, validation and test sets from the mnist dataset.

        Each set contains:
        A list containing numpy arrays correspondign to the images in the
        irst position, and an array encoding for these images desired
        output in the last position.

    """
    with open(file_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'  # Encoding needed to read the dataset.
        # File contains the three sets.
        training, validation, test = u.load()
    return (training, validation, test)


def output_to_array(results_array):
    """
    Takes the expected output on the form of an integer and transforms it into
    an output array of size 10 with a 1 at the index corresponding to this
    integer and zeros everywhere else.

    Args:
        results_array: The array containing the pictures outputs.

    Returns:
        A list of output arrays.

    """
    arr_outputs = []
    for outp in results_array:
        # Generate array of 0.
        o_array = np.zeros((10, 1))
        # Place a 1 at the good position.
        o_array[outp] = 1
        arr_outputs.append(o_array)
    return (arr_outputs)


def parse_data(path, d_set='train'):
    """
    Parse the dataset in usable content.

    The parsed dataset contains tuples with in first position the pixels
    array of the image, and in second position the expected output array.

    Args:
       dataset: Set to be parsed.

    Returns:
        A list of tuples containing in first position the pixels array and in 
        second position the output array.

    """
    # Dict used to extract the wanted dataset.
    set_dict = {'train': 0, 'test': 1, 'validation': 2}
    # Data import.
    dataset = import_mnist(path)[set_dict[d_set]]
    images_set = []
    # Data extraction.
    output_array = output_to_array(dataset[1])
    for im, output in zip(dataset[0], output_array):
        images_set.append((np.reshape(im.flatten(), (-1, 1)),
                           output))
    return (images_set)


x = parse_data('mnist.pkl', 'train')
print((x[0]))
