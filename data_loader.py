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
A2: Major, minor, seventh, diminished: N, maj, min, maj7, min7, 7, dim, dim7;
A3: Major, minor, seventh, diminished, augmented, suspended: N, maj, min, maj7, min7, 7, dim,dim7, aug, sus;
"""

PITCH_LIST = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
QUALITY_LIST = ["maj", "min", "dim", "maj7", "min7", "7", "dim7"]


def sentence_to_chunks(sentence, N_CHORDS=16):
    chunks = []
    for i in range(len(sentence) - N_CHORDS):
        chunks.append(sentence[i : i+N_CHORDS])
    return chunks



def accept_chunk(chunk):

    QUALITY_LIST = ["maj", "min", "dim", "maj7", "min7", "7", "dim"]

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

    # Repeated chord accepted or not (i.e. if REPEAT==False: C:maj C:maj C:maj C:maj G:9 G:9 -> C:maj C:9)
    REPEAT = False 

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
            if chord != last_chord and REPEAT == False:
                current_sentence.append(chord) 
                last_chord = chord
    print('total number of sentences : ', len(sentences))

    print("."*50)

    
    all_chunks = []
    for sentence in sentences:
        all_chunks += sentence_to_chunks(sentence, N_CHORDS)
    
    print("total number of chunks : ", len(all_chunks))

    all_chunks_ok = []
    for chunk in all_chunks:
        is_accepted, transformed_chunk = accept_chunk(chunk)
        if is_accepted:
            all_chunks_ok.append(transformed_chunk)

    print('total number of accepted chunks : ', len(all_chunks_ok), ' (ratio : ', round(len(all_chunks_ok)/len(all_chunks), 2), ')')

    return all_chunks_ok



def plot_chord(tensor_chord):
    fig, ax = plt.subplots(1,1)

    ax.imshow(tensor_chord)
    ax.set_xticks(np.arange(len(QUALITY_LIST)))
    ax.set_yticks(np.arange(len(PITCH_LIST)))
    ax.set_xticklabels(QUALITY_LIST)
    ax.set_yticklabels(PITCH_LIST)
    plt.xticks(rotation=45)
    plt.show()



def chunk_to_tensor(chunk):
    one_hot_tensor = torch.zeros(len(chunk), len(PITCH_LIST), len(QUALITY_LIST))
    
    for index_chord, chord in enumerate(chunk):
        pitch, quality = chord.split(":")
        index_pitch = PITCH_LIST.index(pitch)
        index_quality = QUALITY_LIST.index(quality)

        one_hot_tensor[index_chord, index_pitch, index_quality] = 1

    # TODO : Soit on garde la size en N_CHORDS x N_PITCH x N_QUALITY ou bien il faut la reshape en N_CHORDS x (N_PITCH*N_QUALITY)

    return one_hot_tensor



def tensor_to_chunk(one_hot_tensor):
    chords = []
    index_chord, index_pitch, index_quality = torch.where(one_hot_tensor == 1)

    for i in index_chord:
        chords.append(PITCH_LIST[index_pitch[i]] + ":" + QUALITY_LIST[index_quality[i]])
    
    print(chords)



def set_one_hot_dataset():
    all_chunks = get_chunks()
    dataset_one_hot = torch.zeros(len(all_chunks), 16, len(PITCH_LIST), len(QUALITY_LIST))

    print("Converting into tensor")
    pbar = tqdm.tqdm(total = len(all_chunks))
    for i, chunk in enumerate(all_chunks):
        dataset_one_hot[i,:,:,:] = chunk_to_tensor(chunk)
        pbar.update(1)
    
    pbar.close()
    torch.save(dataset_one_hot, 'dataset_chunks.pt')

    return dataset_one_hot



def import_dataset():
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
    dataset_one_hot = import_dataset()
    tensor_to_chunk(dataset_one_hot[0,:,:,:])
    plot_chord(dataset_one_hot[0,5,:,:])
    print("=== END TEST ===")



if __name__ == '__main__':
    test()