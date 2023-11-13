from dataclasses import astuple, dataclass

import os
import re


@dataclass
class AudioSample:
    id: str
    path: str
    reference: str

    def __iter__(self):
        return iter(astuple(self))


# ".../67/70970" as a directory for speaker 67 with passage 70970
SUBDIR_REGEX = "\d+\/\d+$"
# ".../67-70970-0000" audio for speaker 67 with passage 70970 part 0000
AUDIO_NAME_REGEX = '^\d+\\-\d+\-\d+'
# ".../67-70970-0000.wav" audio for speaker 67 with passage 70970 part 0000
AUDIO_WAV_REGEX = AUDIO_NAME_REGEX + '.wav$'
# ".../67-70970.trans.txt" audio for speaker 67 with passage 70970 part 0000
TRANS_TXT_FILE_REGEX = '^\d+\\-\d+\.trans\.txt$'


class AudioTraversal:
    def samples(self, samples_dir):
        samples = []
        for subdir, dirs, files in os.walk(samples_dir):
            # for each directory traverse through all audios and call model-inference
            if re.search(SUBDIR_REGEX, subdir):
                # collecting samples
                audio_ids = {}
                audio_samples_dict = {}
                for file in files:
                    filepath = os.path.join(subdir, file)
                    audio_id = self.__audio_wav_name(file)
                    if audio_id:
                        audio_samples_dict[audio_id] = filepath
                    else:
                        trans_txt_match = re.search(TRANS_TXT_FILE_REGEX, file)
                        if trans_txt_match:
                            audio_ids = self.__audios_dict(filepath)
                for id, trans in audio_ids.items():
                    samples.append(AudioSample(id, audio_samples_dict[id], trans))
        return samples
        # process_samples(nemo_model_name, model, subdir, samples)

    # Input file with trans.txt with audio-ids and translations
    # Return a dictionary: e.g. (key: '1089-134686-0013', value: 'IF EVER HE WAS IMPELLED')
    def __audios_dict(self, trans_txt_file):
        dict = {}
        with open(trans_txt_file, 'r', errors='replace') as file:
            lines = file.readlines()
            for line in lines:
                id, transcript = line.split(' ', 1)
                dict[id] = transcript[:-1]
        return dict

    # returns audio id: e.g. 1089-134686-0013.wav - return 1089-134686-0013
    def __audio_wav_name(self, file):
        audio_wav_match = re.search(AUDIO_WAV_REGEX, file)
        if audio_wav_match:
            audio_name_match = re.search(AUDIO_NAME_REGEX, audio_wav_match.group())
            if audio_name_match:
                return audio_name_match.group()
            return None

#%%
