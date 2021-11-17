import pickle
"""File defining functions for pickling in and out data"""

def load_in_pickles(file_names, path):
    """Loads in Previously Pickled Dataset files"""
    data_sets = []
    for i in range(0, len(file_names)):
        pickle_in = open(path + file_names[i], "rb")
        data_sets.append(pickle.load(pickle_in))
    return data_sets


def save_pickles(file_names, data_sets, path):
    """Saves Data into Pickle files for later use"""
    for i in range(0, len(data_sets)):
        pickle_out = open(path + file_names[i], "wb")
        pickle.dump(data_sets[i], pickle_out)
        pickle_out.close()


def pickle_model(model_name, model):
    """Pickles trained model to "models" Directory with assigned name"""
    pickle_out = open("models/" + model_name + ".p", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


def pickle_model_history(model_name, history):
    """Pickles trained model to "models/history" Directory with assigned name"""
    pickle_out = open("models/history/" + model_name + "_history.p", "wb")
    pickle.dump(history, pickle_out)
    pickle_out.close()


def pickle_scan(scan_name, scan):
    """Pickles Hyper-parameter Scan to "models/scans" Directory with assigned name"""
    pickle_out = open("models/scans/" + scan_name + ".p", "wb")
    pickle.dump(scan, pickle_out)
    pickle_out.close()
