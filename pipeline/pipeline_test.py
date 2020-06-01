import sys
import pandas as pd
import subprocess
from tqdm import tqdm
import tensorflow as tf  # Make sure version 1.3.0 is installed
import numpy as np
import os
# Imports the mLSTM babbler model, for unirep vector generation.
from utils_2 import uniprotRetrieve
from unirep import babbler64 as babbler
# Imports the neural network, for classification.
import shallow_nn as nn
# Imports tools for sequence query and parsing.
from Bio import Entrez, SeqIO


def unirep_model():
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = "./64_weights"
    # Generates the model.
    batch_size = 12
    b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)
    return b

def load_classifier():
    """
    Loads the classifier model.
    """
    # Imports model weights and biases.
    weights = np.load('weights.npy', allow_pickle=True)
    biases = np.load('biases.npy', allow_pickle=True)
    # Generates a classifier with given weights and biases.
    classifier = nn.ShallowNetwork([64, 22, 2])
    classifier.weights = weights
    classifier.biases = biases
    return classifier

def scale_vector(vector):
    """
    Apply the scaling factor used on the training set in the laerning phase
    to the vector to be predicted.
    """
    # Standard deviation for each feature.
    scale_factor = np.load('scale_factor.npy', allow_pickle=True)
    # Mean for each feature
    mean_vector = np.load('scale_mean.npy', allow_pickle=True)
    # Scales the new vector using training parameters.
    return (vector - mean_vector) / scale_factor


def query_protein(query_terms, email=None):
    """
    Queries the protein database and returns the ID's of the hit.

    An interface allows to show all the hits from the given query, and to select
    one of them to get it's sequence.
    """
    # Query phase.
    # Email used to be contacted in case of abuse.
    Entrez.email = email
    # Terms to be searched.
    query = query_terms
    # Queries and parse the results. 
    handle = Entrez.esearch(db="protein", term=query, limit=10)
    records = Entrez.read(handle)
    # List of hits ID's
    id_list = records['IdList']
    handle.close()
    # Selection phase, user is invited to select the hit of interest.
    select = -2
    # Prints informations about the hits.
    for i, each_id in enumerate(id_list):
        fasta = Entrez.efetch(db="protein", id=each_id, rettype="fasta")
        fasta_record = SeqIO.read(fasta, "fasta")
        print(f'{i}: {each_id}| {fasta_record.description}')
    # Selection phase.
    # -2 means no correct input has been entered.
    while select == -2:
        # Asks user to give an integer input, and checks if this integer is
        # a valid index for the list of hits.
        select = int(input("Enter desired sequence number. Type -1 to leave. "))
        if select not in range(len(id_list)):
            # If value is -1, exit the program successfully.
            if select == -1:
                raise ValueError("End of query")
            # If value is not valid, prints an error message, and asks again.
            select = -2
            print("Wrong number")
    return id_list[select]


def get_sequence(prot_id):
    """
    Gets the amino-acid sequence of the protein of interest.
    """
    fasta = Entrez.efetch(db="protein", id=prot_id, rettype="fasta")
    fasta_record = SeqIO.read(fasta, "fasta")
    return fasta_record.seq

def unirep_vectorize(model, classifier, protein_id='556503394',
                     query_terms=False, email='ancnudde@ulb.ac.be'):
    """
    Queries protein database, then uses the mLSTM model to generate the unirep 
    vector.
    """
    translate_prediction = {0: 'Periplasmic', 1:'Cytoplasmic'}
    # If query term is entered, queries the database.
    if query_terms:
        query_id = query_protein(query_terms, email)
    # If protein ID is given, skips query.
    elif protein_id:
        query_id = protein_id
    # Retrieves the sequence from the ID.
    sequence = get_sequence(query_id)
    # Get UniRep vector from the sequence.
    vector = model.get_rep(sequence)[0]
    # Scales the vector.
    scaled_vector = scale_vector(vector).reshape(-1, 1)
    # Makes prediction.
    prediction = classifier.predict(scaled_vector)
    print(prediction)
    return translate_prediction[prediction]


