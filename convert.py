import sys
import glob
from pydub import AudioSegment
import os

#Path to all mp3 files
files_path = sys.argv[1] + '/*/*.wav'
outdir = 'speech_8k/'

for file in sorted(glob.iglob(files_path)):
    name = file.split('.wav')[0]
    sound=AudioSegment.from_wav(file)
    os.system('mkdir -p ' +outdir + os.path.dirname(name))
    sound.export(outdir+name + '.wav',format='wav', parameters=['-ar', '8k'])
