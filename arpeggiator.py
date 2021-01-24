# -*- coding: utf-8 -*-
"""
    Arpeggiator ML Jazz
"""

from music21 import *
import random as rd
from PIL import Image
from music21.converter.subConverters import ConverterMusicXML

"""
https://web.mit.edu/music21/doc/moduleReference/moduleHarmony.html
symbols = ['', 'm', '+', 'dim', '7',
           'M7', 'm7', 'dim7', '7+', 'm7b5',  # half-diminished
           'mM7', '6', 'm6', '9', 'Maj9', 'm9',
           '11', 'Maj11', 'm11', '13',
           'Maj13', 'm13', 'sus2', 'sus4',
           'N6', 'It+6', 'Fr+6', 'Gr+6', 'pedal',
           'power', 'tristan', '/E', 'm7/E-', 'add2',
           '7omit3',]
"""

# A changer selon l'emplacement de MuseScore (utile seulement pour générer la partition)
environment.set("musescoreDirectPNGPath", "C:/Program Files/MuseScore 3/bin/MuseScore3.exe")
environment.set("musicxmlPath", "C:/Program Files/MuseScore 3/bin/MuseScore3.exe")



class Chord:
    """ Single chord

    Attributes
    ----------
    name : str
        Name of the chord
    harmony_chord : music21.harmony.ChordSymbol
        Music21 chord
    cardinality : int
        Number of notes in the chord
    list_notes : list
        List of notes (as music21.pitch.Pitch) in the chord 
    cardinality_ordered : int
        Number of notes in the ordered arpeggio
    list_notes_ordered : list
        List of notes (as music21.pitch.Pitch) in the arpeggio
    order : str
        Type of arpeggio 
            "up" : C E G
            "down" : G E C
            "incl" : C E G G E C
            "excl" : C E G E 
            "rand" : ok.
    duration : int
        Duration of the arpeggio (in crotchets)
    speed : float
        Duration of one note (in crotchets)
            4 : whole note
            2 : half note
            1 : crotchet
            0.6666666666666 : triplet quarter note
            0.5 : quaver
            0.3333333333333 : triplet eigth note
            0.25 : semi-quaver
    """

    def __init__(self, name, duration=1, order="up", speed=0.25):
        self.name = name
        self.harmony_chord = harmony.ChordSymbol(name)
        self.cardinality = 0
        self.list_notes = []
        self.cardinality_ordered = 0
        self.list_notes_ordered = []

        self.order = order
        self.duration = duration
        self.speed = speed

        self._get_notes()
        self._set_order_notes()

    def _get_notes(self):
        """ Compute list_notes and cardinality
        """
        self.list_notes = list(self.harmony_chord.pitches)
        self.cardinality = len(self.list_notes)

    def _set_order_notes(self):
        """ Compute the order of the notes in the arpeggio
        """
        order = self.order
        if order == "up":
            self.list_notes_ordered = self.list_notes
        elif order == "down":
            self.list_notes_ordered = list(reversed(self.list_notes))
        elif order == "incl":
            self.list_notes_ordered = self.list_notes + list(reversed(self.list_notes))
        elif order == "excl":
            self.list_notes_ordered = self.list_notes + list(reversed(self.list_notes[1:-1]))
        
        self.cardinality_ordered = len(self.list_notes_ordered)

    
    def _set_stream(self, stream_output=None):
        """ Compute the arpeggio
        """
    
        if stream_output == None:
            stream_output = stream.Stream()

        n_tot_notes = int(self.duration / self.speed)

        for i in range(n_tot_notes):
            if self.order == "rand":
                current_pitch = rd.choice(self.list_notes)
            else:
                current_pitch = self.list_notes_ordered[i % self.cardinality_ordered]
            
            current_duration = duration.Duration(self.speed) # TODO : Changer ici pour faire des rythmes funky

            current_note = note.Note(pitch=current_pitch, duration=current_duration)
            stream_output.append(current_note)

        return stream_output

    def get_stream(self, stream_output):
        return self._set_stream(stream_output)


    def print_params(self):
        """ Pretty print paramters
        """
        print(f"Name : {self.name}")
        print(f"RAW Cardinality : {self.cardinality}")
        print(f"List_notes : {[str(p) for p in self.list_notes]}")
        print(f"ORDERED Cardinality : {self.cardinality_ordered}")
        print(f"List_notes_ordered : {[str(p) for p in self.list_notes_ordered]}")
        print(f"Total number of notes : ", int(self.duration / self.speed))


    def visualize(self, PIANOROLL=False):
        """ Visualize the arpeggio under a pianoroll format or a sheetmusic format (Musescore needed)
        """
        stream_output = self._set_stream()

        if PIANOROLL:
            stream_output.plot()

        else:
            conv_musicxml = ConverterMusicXML()
            out_filepath = conv_musicxml.write(stream_output, 'musicxml', fp='./test.xml', subformats=['png'])
            image = Image.open('test-1.png')
            image.show()

    def output(self):
        """ Mega sound
        """
        stream_output = self._set_stream()
        stream_output.show('midi')

def convert_format(chord_str):
    return "".join(chord_str.split(":"))


def output_list_chords(list_chords_str):
    list_chords = []

    for chord_str in list_chords_str:
        chord_format = convert_format(chord_str)
        list_chords.append(Chord(
            name=chord_format,
            duration=4,
            order="excl",
            speed=0.33333333333333333
        ))

    total_stream = stream.Stream()

    for chord in list_chords:
        chord.print_params()
        print("*"*20)

        total_stream = chord.get_stream(total_stream)

    conv_musicxml = ConverterMusicXML()
    out_filepath = conv_musicxml.write(total_stream, 'musicxml', fp='./total.xml', subformats=['png'])
    image = Image.open('total-1.png')
    image.show()

    total_stream.show('midi')


def test():
    n_bar = 2
    c1 = Chord("Adim7", duration=4*n_bar, order="excl", speed=0.33333333333)

    c1.print_params()
    c1.visualize()
    c1.output()


def main():
    list_chords_str = ['G:maj7','B:dim','C:maj7','D#:min','D#:min7','C#:maj','D:maj','C:maj7','C:dim7','A:dim','G#:dim','C:maj7','F#:min7','F#:min','C:dim7','A#:maj']
    output_list_chords(list_chords_str)

if __name__ == '__main__':
    main()