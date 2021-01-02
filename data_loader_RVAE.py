"""
Get relevant chunks from the dataset
(Not optimized but fast enough...)

Representation of the dataset : tensor of size N_CHUNKS x N_CHORDS x (N_PITCH * N_MAIN_QUALITIES + N_EXTRA_QUALITIES)
Representation of a chunk : tensor of size N_CHORDS x (N_PITCH * N_MAIN_QUALITIES + N_EXTRA_QUALITIES)
Representation of a chord : tensor of size (N_PITCH * N_MAIN_QUALITIES + N_EXTRA_QUALITIES)
    where there is  a 1 between 0 and N_PITCH * N_MAIN_QUALITIES (main chord)
                    a 1 between N_PITCH * N_MAIN_QUALITIES and -1 (extra colors) 

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
EXTRA_QUALITY_LIST = ["N", "maj7", "min7"] # "None", "+ major 7th", "+ minor 7th"

TOTAL_MAIN = len(PITCH_LIST) * len(MAIN_QUALITY_LIST)


def accept_sentence(sentence):
    """ Accept / modify a sentence depending on the set of qualities
    Simplify complex colors (ex : D:maj(2,*3)/b7) to match an available chord (ex : D:maj:N)

    Parameters
    ----------
    sentence : list
        List of chords (str)

    Returns
    -------
    bool, list
        True, new_sentence : if the sentence is accepted, and new_sentence may be a modified version of the original sentence
        False, None : if the sentence is not accepted i.e. the quality of a chord is not in the set of considered qualities 
    
    """

    for i, chord in enumerate(sentence[1:-1]):
        pitch, quality = chord.split(":")


        if bool(re.search('aug|sus|hdim', quality)): # TODO : a prendre en compte un jour
            return False, None
        
        if quality not in MAIN_QUALITY_LIST:
            # i.e not "min", "maj" or "dim"
            if quality == "7": # dominant 7th chord alone
                sentence[i+1] = pitch + ":maj:min7"
            elif bool(re.search('maj7|maj\(7', quality)): # maj7 alone OR maj(7 + other extra color)
                sentence[i+1] = pitch + ":maj:maj7"
            elif bool(re.search('min7|min\(7', quality)): # min7 alone OR min(7 + other extra color)
                sentence[i+1] = pitch + ":min:min7"
            elif bool(re.search('dim7', quality)):
                sentence[i+1] = pitch + ":dim:min7"
            elif bool(re.search('7\(|7/', quality)): # dominant 7th chord + other extra colors
                sentence[i+1] = pitch + ":maj:min7"

            elif quality == "9" or bool(re.search('9\(|9/', quality)): # dominant 7th chord + 9th (bypass 9) OR the main chord is a 9th chord
                sentence[i+1] = pitch + ":maj:min7"


            elif bool(re.search('min', quality)): # minor + other color
                sentence[i+1] = pitch + ":min:N"
            elif bool(re.search('maj|6\(', quality)): # Major + other color
                sentence[i+1] = pitch + ":maj:N"
            elif bool(re.search('dim', quality)): # dimished + other color
                sentence[i+1] = pitch + ":dim:N"
            else:
                return False, None
        else:
            sentence[i+1] += ":N"

    return True, sentence        



def get_sentences():
    """ Get sentences of chords from the realbook dataset

    Returns
    -------
    list
        List of list of accepted chords (as str)
    """

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

    for i, chord in enumerate(chord_seq):
        if chord == "_START_":
            current_sentence = ["START"]
        elif chord == "_END_":
            current_sentence.append("END")
            sentences.append(current_sentence)
            current_sentence = []
        else:
            current_sentence.append(chord) 
    print('total number of sentences : ', len(sentences))

    print("."*50)



    # Accept or reject some sentences depending on the qualities chosen
    all_sentences_ok = []
    len_sentences = []
    for sentence in sentences:
        is_accepted, transformed_sentence = accept_sentence(sentence)
        if is_accepted:
            all_sentences_ok.append(transformed_sentence)
            len_sentences.append(len(transformed_sentence))

    print('total number of accepted sentences : ', len(all_sentences_ok), ' (ratio : ', np.mean(len_sentences), ')')

    return all_sentences_ok, len_sentences



def sentence_to_tensor(sentence):
    """ Convert a sentence into a one-hot tensor 

    Parameters
    ----------
    list
        List of accepted chords (as str)

    Returns
    -------
    torch.tensor
        Tensor of size N_CHORDS x (N_PITCH * N_MAIN_QUALITIES + N_EXTRA_QUALITIES)

    """
    # To get a tensor of size N_CHORDS x N_PITCH x N_MAIN_QUALITIES x N_EXTRA_QUALITIES

    # one_hot_tensor = torch.zeros(len(sentence), len(PITCH_LIST), len(MAIN_QUALITY_LIST), len(EXTRA_QUALITY_LIST))
    
    # for index_chord, chord in enumerate(sentence):
    #     pitch, main_quality, extra_quality = chord.split(":")
    #     index_pitch = PITCH_LIST.index(pitch)
    #     index_main_quality = MAIN_QUALITY_LIST.index(main_quality)
    #     all_extra_qualities = extra_quality.split(",")
    #     for extra in all_extra_qualities:
    #         index_extra_quality = EXTRA_QUALITY_LIST.index(extra)
    #         one_hot_tensor[index_chord, index_pitch, index_main_quality, index_extra_quality] = 1

    one_hot_tensor = torch.zeros(len(sentence), len(PITCH_LIST)*len(MAIN_QUALITY_LIST) + len(EXTRA_QUALITY_LIST) + 2)
    
    
    one_hot_tensor[0, TOTAL_MAIN + len(EXTRA_QUALITY_LIST)] = 1
    for index_chord, chord in enumerate(sentence[1:-1]):
            
        pitch, main_quality, extra_quality = chord.split(":")
        index_pitch = PITCH_LIST.index(pitch)
        index_main_quality = MAIN_QUALITY_LIST.index(main_quality)

        one_hot_tensor[index_chord + 1, index_pitch*len(MAIN_QUALITY_LIST) + index_main_quality] = 1 # Set 1 to the main chord in the first part of the vector
        
        all_extra_qualities = extra_quality.split(",")
        for extra in all_extra_qualities:
            index_extra_quality = EXTRA_QUALITY_LIST.index(extra)
            one_hot_tensor[index_chord + 1, TOTAL_MAIN + index_extra_quality] = 1 # Set 1 to the extra color in the second part of the vector

    one_hot_tensor[-1, TOTAL_MAIN + len(EXTRA_QUALITY_LIST) + 1] = 1

    return one_hot_tensor



def tensor_to_sentence(one_hot_tensor,length_tensor):
    """ Convert a tensor into a sentence of chords

    Parameters
    ----------
    torch.tensor
        Tensor of size N_CHORDS x (N_PITCH * N_MAIN_QUALITIES + N_EXTRA_QUALITIES)

    Returns
    -------
    list
        List of accepted chords (as str) 
    
    """
    chords = []

    # From a tensor of size N_CHORDS x N_PITCH x N_MAIN_QUALITIES x N_EXTRA_QUALITIES
    # for chord_i in range(one_hot_tensor.shape[0]): 
    #     index_pitch, index_main_quality, index_extra_quality = torch.where(one_hot_tensor[chord_i,:,:,:] == 1)

    #     chord_str = PITCH_LIST[index_pitch[0]] + ":" + MAIN_QUALITY_LIST[index_main_quality[0]] + ":"

    #     for extra_i in index_extra_quality:
    #         chord_str += EXTRA_QUALITY_LIST[extra_i] + ","
        
    #     chords.append(chord_str[:-1])


    for chord_i in range(length_tensor):
        
        one_hot_tensor_extend = torch.cat((one_hot_tensor[chord_i,:TOTAL_MAIN],one_hot_tensor[chord_i,-2:]),0)
        index_main = torch.where(one_hot_tensor_extend == torch.max(one_hot_tensor_extend))[0]
        print(index_main)
        print()
        if index_main == TOTAL_MAIN:
            chords.append("START")
        elif index_main == TOTAL_MAIN + 1:
            chords.append("END")
        else:
            index_pitch = index_main // len(MAIN_QUALITY_LIST)
            index_main_quality = index_main % len(MAIN_QUALITY_LIST)

            chord_str = PITCH_LIST[index_pitch] + ":" + MAIN_QUALITY_LIST[index_main_quality] + ":"

            index_extra_quality = torch.where(one_hot_tensor[chord_i,TOTAL_MAIN:] == torch.max(one_hot_tensor[chord_i,TOTAL_MAIN:]))[0]
            for extra_i in index_extra_quality:
                chord_str += EXTRA_QUALITY_LIST[extra_i] + ","

            chords.append(chord_str[:-1])

    return chords




def set_one_hot_dataset():
    """ Compute and export a dataset composed of sentences of chords

    Returns
    -------
    torch.tensor
        Tensor of size N_SENTENCES x N_CHORDS x (N_PITCH * N_MAIN_QUALITIES + N_EXTRA_QUALITIES)
        It contains chords from the realbook dataset as one-hot vectors
    """
    all_sentences, len_sentences = get_sentences()
    max_sequence_length = max(len_sentences)
    # dataset_one_hot = torch.zeros(len(all_sentences), 16, len(PITCH_LIST), len(MAIN_QUALITY_LIST), len(EXTRA_QUALITY_LIST))
    dataset_one_hot = torch.zeros(len(all_sentences), max_sequence_length, len(PITCH_LIST) * len(MAIN_QUALITY_LIST) + len(EXTRA_QUALITY_LIST) + 2)

    print("Converting into tensor")
    pbar = tqdm.tqdm(total = len(all_sentences))
    for i, sentence in enumerate(all_sentences):
        dataset_one_hot[i,:len_sentences[i],:] = sentence_to_tensor(sentence)
        pbar.update(1)
    
    pbar.close()
    torch.save(dataset_one_hot, 'dataset_sentences.pt')
    np.save("len_sentences.npy", np.array(len_sentences))

    return dataset_one_hot



def import_dataset():
    """ Import the dataset

    Returns
    -------
    torch.tensor
        Tensor of size N_SENTENCES x N_CHORDS x (N_PITCH * N_MAIN_QUALITIES + N_EXTRA_QUALITIES)
        It contains chords from the realbook dataset as one-hot vectors
    """
    try:
        dataset_one_hot = torch.load("dataset_sentences.pt")
        len_sentences = np.load("len_sentences.npy")
    except FileNotFoundError:
        print("Dataset not found.")
        print("Computing dataset...")
        print("."*50)
        dataset_one_hot = set_one_hot_dataset()

    print("Dataset loaded !")
    
    return dataset_one_hot, len_sentences
        
    


def main():
    import_dataset()
    



def test():
    print("=== TEST ===")
    # chunk_test = ['C:maj:N', 'C:maj:N', 'C:maj:N', 'C:maj:N', 'C:maj:N', 'C:maj:N', 'C:maj:N', 'C:maj:N', 'G:maj:maj7', 'G:maj:min7', 'G:maj:min7', 'G:maj:min7', 'G:maj:min7', 'G:maj:min7', 'G#:dim:min7', 'G:maj:min7,maj7']
    # t = chunk_to_tensor(chunk_test)
    # print(t)
    # str_t = tensor_to_chunk(t)
    # plot_chord(t[8,:])
    # print(chunk_test)
    # print(str_t)

    dataset_one_hot, len_sentences = import_dataset()
    str_c = tensor_to_sentence(dataset_one_hot[40,:,:],len_sentences[40])
    print(str_c)
    
    print("=== END TEST ===")



if __name__ == '__main__':
    test()