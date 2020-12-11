"""
Get relevant chunks from the dataset
(Not optimized by fast enough...)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
A1: Major, minor, diminished: N (which means no chord), maj, min, dim;
A2: Major, minor, seventh, diminished: N, maj, min, maj7, min7, 7, dim, dim7;
A3: Major, minor, seventh, diminished, augmented, suspended: N, maj, min, maj7, min7, 7, dim,dim7, aug, sus;
"""


def sentence_to_chunks(sentence, N_CHORDS=16):
    chunks = []
    for i in range(len(sentence) - N_CHORDS):
        chunks.append(sentence[i : i+N_CHORDS])
    return chunks

def accept_chunk(chunk):
    AVAILABLE_QUALITY = ["maj", "min", "dim", "maj7", "min7", "7", "dim"] #dim7 ?

    for chord in chunk:
        pitch, color = chord.split(":")
        if color not in AVAILABLE_QUALITY:
            return False

    return True        

def main():

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
        if accept_chunk(chunk):
            all_chunks_ok.append(chunk)

    print('total number of accepted chunks : ', len(all_chunks_ok), ' (ratio : ', round(len(all_chunks_ok)/len(all_chunks), 2), ')')
if __name__ == '__main__':
    main()