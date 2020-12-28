"""
Get relevant chunks from the dataset
(Not optimized by fast enough...)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
import tqdm

"""
A1: Major, minor, diminished: N (which means no chord), maj, min, dim;
A2: Major, minor, seventh, diminished: maj, min, dim, N, maj7, min7, 7, dim7;
A3: Major, minor, seventh, diminished, augmented, suspended: N, maj, min, maj7, min7, 7, dim,dim7, aug, sus;
"""

PITCH_LIST = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
MAIN_QUALITY_LIST = ["maj", "min", "dim"]
EXTRA_QUALITY_LIST = ["N", "maj7", "min7"]



def sentence_to_chunks(sentence, N_CHORDS=16):
    """ Convert a sentence into a list of chunks of N_CHORDS chords

    Parameters
    ----------
    sentence : list
        List of chords (str) from the realbook dataset
    N_CHORDS : int, optional
        Number of chords in a chunk 

    Returns
    -------
    list
        List of lists of N_CHORDS chords (str)
    """
    chunks = []
    for i in range(len(sentence) - N_CHORDS):
        chunks.append(sentence[i : i+N_CHORDS])
    return chunks



def accept_chunk(chunk):
    """ Accept / modify a chunk depending on the set of qualities
    Simplify complex colors (ex : D:maj(2,*3)/b7) to match an available chord (ex : D:maj)

    Parameters
    ----------
    chunk : list
        List of chords (str)

    Returns
    -------
    bool, list
        True, new_chunk : if the chunk is accepted, and new_chunk may be a modified version of the original chunk
        False, None : if the chunk is not accepted i.e. the quality of a chord is not in the set of considered qualities 
    
    """
    for i, chord in enumerate(chunk):
        pitch, quality = chord.split(":")

        if bool(re.search('aug|sus|hdim', quality)): # TODO : a prendre en compte un jour
            return False, None
        
        if quality not in QUALITY_LIST:
            # TODO : maj/7, min/7 : à considérer comme des maj7 / min7 ou des maj/min ?
            if "maj(7" in chord :
                chunk[i] = pitch + ":maj7"  
            elif "min(7" in chord :
                chunk[i] = pitch + ":min7"
            elif "dim" in chord:
                chunk[i] = pitch + ":dim"
            elif "7" in chord:
                chunk[i] = pitch + ":7"
            elif "min" in chord:
                chunk[i] = pitch + ":min"
            elif bool(re.search('maj|9|11', quality)):
                chunk[i] = pitch + ":maj"
            else:
                # print(chord)
                chunk[i] = pitch + ":maj"
                # return False, None

    return True, chunk        



def get_chunks():
    """ Get chunks of chords from the realbook dataset

    Returns
    -------
    list
        List of list of accepted chords (as str)
    """
    # Repeated chord accepted or not (i.e. if REPEAT==False: C:maj C:maj C:maj C:maj G:9 G:9 -> C:maj C:9)
    REPEAT = True

    # Number of chords in a chunk
    N_CHORDS = 16

    path = 'chord_sentences.txt' # the txt data source
    text = open(path).read()
    print('corpus length:', len(text))

    chord_seq = text.split(' ')
    chars = sorted(set(chord_seq))

    # Find total number of diff chords
    chord_indexes = dict((c, i) for i, c in enumerate(chars))
    list_chords_order = np.array([c for c, i in chord_indexes.items()])
    num_chars = len(chord_indexes)
    print('total diff chords:', num_chars)


    # Put sentences in list
    sentences = []
    last_chord = ""

    for i, chord in enumerate(chord_seq):
        if chord == "_START_":
            current_sentence = []
        elif chord == "_END_":
            sentences.append(current_sentence)
        else:
            # if chord != last_chord and REPEAT == False:
                current_sentence.append(chord) 
                last_chord = chord
    print('total number of sentences : ', len(sentences))

    print("."*50)

    
    # Convert sentences into chunks
    all_chunks = []
    for sentence in sentences:
        all_chunks += sentence_to_chunks(sentence, N_CHORDS)
    
    print("total number of chunks : ", len(all_chunks))


    # Accept of reject some chunks depending on the qualities chosen
    all_chunks_ok = []
    for chunk in all_chunks:
        is_accepted, transformed_chunk = accept_chunk(chunk)
        if is_accepted:
            all_chunks_ok.append(transformed_chunk)

    print('total number of accepted chunks : ', len(all_chunks_ok), ' (ratio : ', round(len(all_chunks_ok)/len(all_chunks), 2), ')')

    return all_chunks_ok



def plot_chord(tensor_chord):
    """ Plot a one-hot tensor

    Parameters
    ----------
    torch.tensor
        Tensor of size len(PITCH_LIST) x len(MAIN_QUALITY_LIST) x len(EXTRA_QUALITY_LIST)
        It must be full of zeros, except at one index

    
    """
    fig, ax = plt.subplots(1,1)

    tick_main = []
    for p in PITCH_LIST:
        for q in MAIN_QUALITY_LIST:
            tick_main.append(p + q)

    ax.imshow(tensor_chord.view(-1, len(EXTRA_QUALITY_LIST)))
    ax.set_yticks(np.arange(len(MAIN_QUALITY_LIST) * len(PITCH_LIST)))
    ax.set_xticks(np.arange(len(EXTRA_QUALITY_LIST)))
    ax.set_xticklabels(EXTRA_QUALITY_LIST)
    ax.set_yticklabels(tick_main)
    plt.xticks(rotation=45)
    plt.show()



def chunk_to_tensor(chunk):
    """ Convert a chunk into a one-hot tensor 

    Parameters
    ----------
    list
        List of accepted chords (as str)

    Returns
    -------
    torch.tensor
        Tensor of size N_CHORDS x N_PITCH x N_QUALITIES
        Each sub-tensor of size N_PITCH x N_QUALITIES is a one-hot tensor

    """
    one_hot_tensor = torch.zeros(len(chunk), len(PITCH_LIST), len(MAIN_QUALITY_LIST), len(EXTRA_QUALITY_LIST))
    
    for index_chord, chord in enumerate(chunk):
        pitch, main_quality, extra_quality = chord.split(":")
        index_pitch = PITCH_LIST.index(pitch)
        index_main_quality = MAIN_QUALITY_LIST.index(main_quality)
        all_extra_qualities = extra_quality.split(",")
        for extra in all_extra_qualities:
            index_extra_quality = EXTRA_QUALITY_LIST.index(extra)
            one_hot_tensor[index_chord, index_pitch, index_main_quality, index_extra_quality] = 1

    return one_hot_tensor



def tensor_to_chunk(one_hot_tensor):
    """ Convert a tensor into a chunk of chords

    Parameters
    ----------
    torch.tensor
        Tensor of size N_CHORDS x N_PITCH x N_QUALITIES
        Each sub-tensor of size N_PITCH x N_QUALITIES must be a one-hot tensor

    Returns
    -------
    list
        List of accepted chords (as str) 
    
    """
    chords = []
    index_chord, index_pitch, index_quality = torch.where(one_hot_tensor == 1)

    for i in index_chord:
        chords.append(PITCH_LIST[index_pitch[i]] + ":" + QUALITY_LIST[index_quality[i]])
    
    print(chords)

    return chords



def set_one_hot_dataset():
    """ Compute and export a dataset composed of chunks of chords

    Returns
    -------
    torch.tensor
        Tensor of size N_CHUNKS x N_CHORDS x N_PITCH x N_MAIN_QUALITIES x N_EXTRA_QUALITIES
        It contains chords from the realbook dataset as one-hot vectors
    """
    all_chunks = get_chunks()
    dataset_one_hot = torch.zeros(len(all_chunks), 16, len(PITCH_LIST), len(MAIN_QUALITY_LIST), len(EXTRA_QUALITY_LIST))

    print("Converting into tensor")
    pbar = tqdm.tqdm(total = len(all_chunks))
    for i, chunk in enumerate(all_chunks):
        dataset_one_hot[i,:,:,:,:] = chunk_to_tensor(chunk)
        pbar.update(1)
    
    pbar.close()
    torch.save(dataset_one_hot, 'dataset_chunks.pt')

    return dataset_one_hot



def import_dataset():
    """ Import the dataset

    Returns
    -------
    torch.tensor
        Tensor of size N_CHUNKS x N_CHORDS x N_PITCH x N_QUALITIES
        It contains chords from the realbook dataset as one-hot vectors
    
    """
    try:
        dataset_one_hot = torch.load("dataset_chunks.pt")
    except FileNotFoundError:
        print("Dataset not found.")
        print("Computing dataset...")
        print("."*50)
        dataset_one_hot = set_one_hot_dataset()

    print("Dataset loaded !")
    
    return dataset_one_hot
        
    


def main():
    import_dataset()
    



def test():
    print("=== TEST ===")
    t = chunk_to_tensor(["C:maj:maj7,min7"])
    plot_chord(t)
    print(t)
    # dataset_one_hot = import_dataset()
    # tensor_to_chunk(dataset_one_hot[0,:,:,:])
    # plot_chord(dataset_one_hot[0,5,:,:])
    print("=== END TEST ===")



if __name__ == '__main__':
    test()