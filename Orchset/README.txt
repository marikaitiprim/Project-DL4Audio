ORCHSET: A DATASET FOR MELODY EXTRACTION IN SYMPHONIC MUSIC RECORDINGS
====================================================

Description
-----------------
Orchset is intended to be used as a dataset for the development and evaluation of melody extraction algorithms. This collection contains 64 audio excerpts focused on symphonic music. with their corresponding annotation of the melody.
Melody is here defined as “the single (monophonic) pitch sequence that a listener might reproduce if asked to whistle or hum a piece of polyphonic music”.
The dataset creation comprised several tasks: excerpts selection, recording sessions of people singing along with the excerpts, analysis of the recordings and melody annotation. A complete description of the dataset and the creation methodology is presented in this paper:
Bosch, J., Marxer, R, Gomez, E., “Evaluation and Combination of Pitch Estimation Methods for Melody Extraction in Symphonic Classical Music”, Journal of New Music Research (2016)

Files included
-----------------

Audio
-----------------
This dataset consists of 64 audio excerpts (wav format, stereo and mono, sampled at 44.1kHz) from symphonies and symphonic poems, ballets suites and other musical forms interpreted by symphonic orchestras, mostly from the romantic period, as well as classical and 20th century pieces. The length of the excerpts ranges from 10 to 32 seconds. The number of excerpts per composer are: Beethoven (13), Brahms (4), Dvorak (4), Grieg (3), Haydn (3), Holst (4), Mussorgsky (9), Prokofiev (2), Ravel (3), Rimsky-Korsakov (10), Schubert (1), Smetana (2), Strauss (3), Tchaikovsky (2), Wagner (1).

Annotations
-----------------

For each excerpt we provide the annotation of the melody in MIDI format, and a text file with the sequence of melody pitches derived from the MIDI file, using a sampling period of 10 ms. If no melody pitch is annotated at a specific time, the frame is considered as unvoiced, otherwise it is consider as voiced. The annotations for an excerpt named: “excerptName.wav” are given in “excerptName.mel” and “excerptName.mid”.

93.69% of the frames of the dataset are labelled as voiced while 6.31% are unvoiced (in which case the pitch is set to be 0). The dataset then focuses on the evaluation of pitch estimation instead of voicing estimation (guessing if a certain frame contains melody or not). We also provide a csv file with annotations of the predominant melodic instrument section(s) playing the melody. Information is provided for each of the audio files, mentioning if the melody is played by strings, brass, woodwinds, a combination of them, or alternating sections.

Conditions of Use
-----------------
The Orchset dataset is offered free of charge for non-commercial use only.  You can not redistribute it nor modify it.
Dataset by Juan J. Bosch and Emilia Gómez, Music Technology Group - Universitat Pompeu Fabra (Barcelona)
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License


Please Acknowledge Orchset in Academic Research
---------------------------------------------------
When Orchset is used for academic research, we would highly appreciate if scientific publications of works partly based on the Orchset dataset quote the following publication:

J. Bosch, R Marxer, and E. Gomez. Evaluation and combination of pitch estimation methods for melody extraction in symphonic classical music. Journal of New Music Research, DOI:10.1080/09298215.2016.1182191, 2016.

The creation of this dataset was partially supported by the European Union Seventh Framework Programme FP7 / 2007-2013 through the PHENICX project (grant agreement no. 601166). It is also (partly) supported by the Spanish Ministry of Economy and Competitiveness under the Maria de Maeztu Units of Excellence Programme (MDM-2015-0502).

Feedback
------------------------

Problems, positive feedback, negative feedback, help to improve the annotations... it is all welcome! Send your feedback to: juan.bosch@upf.edu AND mtg@upf.edu
In case of a problem report please include as many details as possible.

