""" 
    Transition matrix of the realbook dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def set_chord_transition_matrix(sentences, chord_indexes):
    transition_matrix = np.zeros((len(chord_indexes), len(chord_indexes)))
    for sentence in sentences:
        for i, current_chord in enumerate(sentence[:-1]):
            next_chord = sentence[i+1]
            index_current_chord = chord_indexes[current_chord]
            index_next_chord = chord_indexes[next_chord]
            transition_matrix[index_current_chord, index_next_chord] += 1 # y-axis : FROM ; x-axis : TO
    
    return transition_matrix


def get_transition_list(transition_matrix, chord_indexes, list_chords_order):
    """ Print list 
        CHORD_FROM --------------
            CHORD_TO : number of occurrences of transition CHORD_FROM -> CHORD_TO
    """
    width_print = 50
    for c, i in chord_indexes.items():
        print(c.ljust(width_print,'.'))

        to_chords_list = list((list_chords_order[index_non_zero], transition_matrix[i][index_non_zero]) for index_non_zero in np.nonzero(transition_matrix[i])[0])
        to_chords_list = sorted(to_chords_list, key= lambda chord_count : chord_count[1], reverse=True)

        for chord, count in to_chords_list:
            print(f"\t {chord} : {count}", end=' ')
            if(chord == c):
                print("(Repeated)", end='')
            print()



def main():
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
            current_sentence.append(chord) 
            last_chord = chord
    print('total number of sentences : ', len(sentences))

    transition_matrix = set_chord_transition_matrix(sentences, chord_indexes)
    get_transition_list(transition_matrix, chord_indexes, list_chords_order)

if __name__ == '__main__':
    main()