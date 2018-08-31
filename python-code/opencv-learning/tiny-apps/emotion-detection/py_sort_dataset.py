"""
Mani experimenting with facial information extraction.
This module is used to sort the CK+ dataset.
"""

import glob
from shutil import copyfile

# No need to modify this one as it is a helper script.
__version__ = "1.0, 01/04/2016"
__author__ = "Paul van Gent, 2016"

# Define emotion order
emotions = ["neutral", "anger", "contempt", "disgust",
            "fear", "happy", "sadness", "surprise"]

# Returns a list of all folders with participant numbers
participants = glob.glob("source_emotions\\*")

# i = 1;

for x in participants:

    # i += 1;
    # store current participant number
    part = "%s" % x[-4:]

    # Store list of sessions for current participant
    for sessions in glob.glob("%s\\*" % x):
        for files in glob.glob("%s\\*" % sessions):
            current_session = files[20:-30]
            file1 = open(files, 'r')

            # Emotions are encoded as a float, readline as float,
            # then convert to integer.
            emotion = int(float(file1.readline()))

            # get path for last image in sequence, which contains the emotion
            sourcefile_emotion = glob.glob(
                "source_images\\%s\\%s\\*" % (part, current_session))[-1]

            # do same for neutral image
            sourcefile_neutral = glob.glob(
                "source_images\\%s\\%s\\*" % (part, current_session))[0]

            # Generate path to put neutral image
            dest_neut = "sorted_set\\neutral\\%s" % sourcefile_neutral[25:]

            # Do same for emotion containing image
            dest_emot = "sorted_set\\%s\\%s" % (
                emotions[emotion], sourcefile_emotion[25:])

            # Copy file
            copyfile(sourcefile_neutral, dest_neut)
            copyfile(sourcefile_emotion, dest_emot)

    # if i == 10:
            # break;