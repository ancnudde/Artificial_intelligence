def read_unirep(file_path):
    """
    Parse a Unirep file.
    """
    data_matrix = []
    with open(file_path) as fp:
        data = fp.readlines()
    for i, line in enumerate(data):
        if line[0] == '>':
            data_matrix.append(np.array(
                data[i + 1].strip().split(sep=' '),
                dtype=np.float64))
    return np.array(data_matrix)

# Imports unirep files.
raw_cyt = read_unirep("cytoxplasmUniRef50(1).unirep")
raw_peri = read_unirep("periplasmUniRef50(1).unirep")
# Shuffle imported data.
np.random.shuffle(raw_cyt)
np.random.shuffle(raw_peri)
# Creates mixed dataset (from shuffled = random selection).
all_raw = np.concatenate((raw_cyt[0:3000], raw_peri[0:3000]))

# We use sklearn, which allows to get the scaling factor 
# and the mean used to scale the initial data.

# Scaler object init and fit to data.
scaler = StandardScaler().fit(all_raw)
# Apply scaler to the training set.
scaled_transfrom = scaler.transform(all_raw)
# Extracts and save scaling parameters.
sc = scaler.scale_
mn = scaler.mean_
np.save('scale_factor.npy' , sc)
np.save('scale_mean.npy', mn)

# We transform data back to see if it works as expected.
back_transform = scaled_transfrom * sc + mn
equality_check = [elt - tle for elt, tle in zip(back_transform[0], all_raw[0]) 
                  if elt != tle]
print(equality_check)
""" All comparisons considered as not equal are actually really close """

# Gives value 1 to cytoplasmic proteins, 0 to others.
labels = np.zeros(len(all_raw), dtype=int)
labels[0:3000] = 1

# One hot encoding for neural network.
def one_hot(labels):
    encoded_labels = []
    number_classes = len(set(labels))
    for item in labels:
        encoded = np.zeros(number_classes, dtype=int)
        encoded[item] = 1
        encoded_labels.append(encoded)
    return np.array(encoded_labels)

f_labels = one_hot(labels)

# Prepares input for neural network.
def nn_input_format(dataset, labels):
    encoded_data = []
    for data, lab in zip(dataset, labels):
        value = (data.reshape((-1,1)), lab.reshape((-1,1)))
        encoded_data.append(value)
    return encoded_data

x = nn_input_format(scaled_transfrom, f_labels)

# Network initialisation and training.
final_model = nn.ShallowNetwork([64, 22, 2])
final_model.stochastic_gradient_descent(x, 100, 8, 0.2, 0.5)

np.save('weights.npy' , final_model.weights)
np.save('biases.npy', final_model.biases)
