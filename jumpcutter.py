import argparse
import math
import os
import re
import subprocess
from shutil import copyfile, rmtree

import numpy as np
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from loguru import logger
from pytube import YouTube
from scipy.io import wavfile


def download_file(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ', '_')
    os.rename(name, newname)
    return newname


def get_max_volume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv, -minv)


def copy_frame(input_frame, output_frame):
    # src = TEMP_FOLDER + "/frame{:06d}".format(inputFrame + 1) + ".jpg"
    frame_input: str = "frame{:06d}".format(input_frame + 1) + ".jpg"
    src = os.path.join(TEMP_FOLDER, frame_input)
    # dst = TEMP_FOLDER + "/newFrame{:06d}".format(outputFrame + 1) + ".jpg"
    frame_output: str = "newFrame{:06d}".format(output_frame + 1) + ".jpg"
    dst = os.path.join(TEMP_FOLDER, frame_output)
    # if not os.path.isfile(src):
    #     return False
    try:
        copyfile(src, dst)
    except FileNotFoundError:
        raise FileNotFoundError("Special case for last frame in video!")
    if output_frame % 20 == 19:
        print(str(output_frame + 1) + " time-altered frames saved.")


def input_to_output_filename(filename):
    dot_index = filename.rfind(".")
    return filename[:dot_index] + "_ALTERED" + filename[dot_index:]


def create_path(s):
    # assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."

    try:
        logger.info("attempting to create dir with path: " + s)
        os.mkdir(s)
    except OSError:
        assert False, "Creation of the directory %s failed. " \
                      "(The TEMP folder may already exist. Delete or rename it, and try again.)"


def delete_path(s):
    try:
        rmtree(s, ignore_errors=False)
    except OSError:
        print("Deletion of the directory %s failed" % s)
        print(OSError)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Modifies a video file to play at different speeds when there is sound vs. silence.')
    parser.add_argument('--input_file', type=str, help='the video file you want modified')
    parser.add_argument('--url', type=str, help='A youtube url to download and process')
    parser.add_argument('--output_file', type=str, default="",
                        help="the output file. (optional. if not included, it'll just modify the input file name)")
    parser.add_argument('--silent_threshold', type=float, default=0.03,
                        help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". "
                             "It ranges from 0 (silence) to 1 (max volume)")
    parser.add_argument('--sounded_speed', type=float, default=1.00,
                        help="the speed that sounded (spoken) frames should be played at. Typically 1.")
    parser.add_argument('--silent_speed', type=float, default=5.00,
                        help="the speed that silent frames should be played at. 999999 for jumpcutting.")
    parser.add_argument('--frame_margin', type=float, default=1,
                        help="some silent frames adjacent to sounded frames are included to provide context. "
                             "How many frames on either the side of speech should be included? That's this variable.")
    parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
    parser.add_argument('--frame_rate', type=float, default=30,
                        help="frame rate of the input and output videos. optional... "
                             "I try to find it out myself, but it doesn't always work.")
    parser.add_argument('--frame_quality', type=int, default=3,
                        help="quality of frames to be extracted from input video. "
                             "1 is highest, 31 is lowest, 3 is the default.")
    parser.add_argument('--folder_watcher_mode', type=bool, default=False,
                        help="Mark true if you want to run this script in watcher mode. "
                             "This mode will process all files in the specified --watched_dir directory.")
    parser.add_argument('--watched_dir', type=str, help='The directory to listen for new files to process.')
    parser.add_argument('--tmp_working_dir', type=str, default="",
                        help="Please specify a directory where all generated files will be temporarily stored. "
                             "This may be be helpful considering the large storage space this script needs to run on.")
    return parser.parse_args()


args = parse_arguments()

frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]

WATCHER_MODE: bool = args.folder_watcher_mode
WATCHED_DIR: str = args.watched_dir
TMP_WORKING_DIR: str = args.tmp_working_dir

if args.url is not None:
    INPUT_FILE = download_file(args.url)
else:
    INPUT_FILE = args.input_file
URL = args.url
FRAME_QUALITY = args.frame_quality

assert INPUT_FILE is not None and WATCHER_MODE is False, "why u put no input file, " \
                                                         "and did not specify watcher mode, one has to be set"

if len(args.output_file) >= 1:
    OUTPUT_FILE = args.output_file
else:
    OUTPUT_FILE = input_to_output_filename(INPUT_FILE)

TEMP_FOLDER = os.path.join(TMP_WORKING_DIR, "tmp")
# smooth out transition's audio by quickly fading in/out (arbitrary magic number whatever)
AUDIO_FADE_ENVELOPE_SIZE = 400

create_path(TEMP_FOLDER)

command = "ffmpeg -i " + INPUT_FILE + " -qscale:v " + str(
    FRAME_QUALITY) + " \"" + TEMP_FOLDER + "/frame%06d.jpg\" -hide_banner"
logger.info("Executing: " + command)
subprocess.call(command, shell=True)

command = "ffmpeg -i " + INPUT_FILE + " -ab 160k -ac 2 -ar " + str(
    SAMPLE_RATE) + " -vn \"" + TEMP_FOLDER + "/audio.wav\""
logger.info("Executing: " + command)
subprocess.call(command, shell=True)

command = "ffmpeg -i " + TEMP_FOLDER + "/input.mp4 2>&1"
f = open(TEMP_FOLDER + "/params.txt", "w")
subprocess.call(command, shell=True, stdout=f)

sampleRate, audioData = wavfile.read(os.path.join(TEMP_FOLDER, "audio.wav"))
audioSampleCount = audioData.shape[0]
maxAudioVolume = get_max_volume(audioData)

f = open(TEMP_FOLDER + "/params.txt", 'r+')
pre_params = f.read()
f.close()
params = pre_params.split('\n')
for line in params:
    m = re.search('Stream #.*Video.* ([0-9]*) fps', line)
    if m is not None:
        frameRate = float(m.group(1))

samplesPerFrame = sampleRate / frameRate

audioFrameCount = int(math.ceil(audioSampleCount / samplesPerFrame))

hasLoudAudio = np.zeros(audioFrameCount)

for i in range(audioFrameCount):
    start = int(i * samplesPerFrame)
    end = min(int((i + 1) * samplesPerFrame), audioSampleCount)
    audiochunks = audioData[start:end]
    maxchunksVolume = float(get_max_volume(audiochunks)) / maxAudioVolume
    if maxchunksVolume >= SILENT_THRESHOLD:
        hasLoudAudio[i] = 1

chunks = [[0, 0, 0]]
shouldIncludeFrame = np.zeros(audioFrameCount)
for i in range(audioFrameCount):
    start = int(max(0, i - FRAME_SPREADAGE))
    end = int(min(audioFrameCount, i + 1 + FRAME_SPREADAGE))
    shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
    if i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i - 1]:  # Did we flip?
        chunks.append([chunks[-1][1], i, shouldIncludeFrame[i - 1]])

chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i - 1]])
chunks = chunks[1:]

outputAudioData = np.zeros((0, audioData.shape[1]))
outputPointer = 0

lastExistingFrame = None
for chunk in chunks:
    audioChunk = audioData[int(chunk[0] * samplesPerFrame):int(chunk[1] * samplesPerFrame)]

    sFile = TEMP_FOLDER + "/tempStart.wav"
    eFile = TEMP_FOLDER + "/tempEnd.wav"
    wavfile.write(sFile, SAMPLE_RATE, audioChunk)
    with WavReader(sFile) as reader:
        with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
            tsm.run(reader, writer)
    _, alteredAudioData = wavfile.read(eFile)
    leng = alteredAudioData.shape[0]
    endPointer = outputPointer + leng
    outputAudioData = np.concatenate((outputAudioData, alteredAudioData / maxAudioVolume))

    # outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

    # smooth out transitiion's audio by quickly fading in/out

    if leng < AUDIO_FADE_ENVELOPE_SIZE:
        outputAudioData[outputPointer:endPointer] = 0  # audio is less than 0.01 sec, let's just remove it.
    else:
        premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE) / AUDIO_FADE_ENVELOPE_SIZE
        mask = np.repeat(premask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
        outputAudioData[outputPointer:outputPointer + AUDIO_FADE_ENVELOPE_SIZE] *= mask
        outputAudioData[endPointer - AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1 - mask

    startOutputFrame = int(math.ceil(outputPointer / samplesPerFrame))
    endOutputFrame = int(math.ceil(endPointer / samplesPerFrame))
    for outputFrame in range(startOutputFrame, endOutputFrame):
        inputFrame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame - startOutputFrame))
        try:
            copy_frame(inputFrame, outputFrame)
            lastExistingFrame = inputFrame
        except FileNotFoundError:
            copy_frame(lastExistingFrame, outputFrame)

    outputPointer = endPointer

wavfile.write(TEMP_FOLDER + "/audioNew.wav", SAMPLE_RATE, outputAudioData)

'''
outputFrame = math.ceil(outputPointer/samplesPerFrame)
for endGap in range(outputFrame,audioFrameCount):
    copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
'''

command = "ffmpeg -framerate " + str(
    frameRate) + " -i " + TEMP_FOLDER + "/newFrame%06d.jpg -i " + TEMP_FOLDER + "/audioNew.wav -strict -2 " + OUTPUT_FILE
subprocess.call(command, shell=True)

delete_path(TEMP_FOLDER)
