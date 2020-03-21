import numpy as np

def get_file_names(thefilename,use_random=True):
    import json
    with open(thefilename) as json_file:  
        data = json.load(json_file)
    thelist = list(data.keys())
    if use_random:
        np.random.shuffle(thelist)
    return data,thelist
