import time

import librosa
import nemo.collections.asr as nemo_asr
from memory_profiler import memory_usage

from toolkit.statistics_reporter import StatisticsReporter


class NemoProcessor:
    def __init__(self, nemo_model_path, statistics_file_prefix=''):
        self.model = None
        self.model_load_time = None
        self.model_inference_time = None
        self.dataset_audio_time_length = None
        self.nemo_model_path = nemo_model_path
        self.statistics_file_prefix = statistics_file_prefix

    def process(self, samples):
        if len(samples) == 0:
            raise Exception("Samples are empty! Provide at least one sample in the samples")

        mem_usage_start = memory_usage()[0]

        self.__load_model()

        # Forming samples to inference
        paths = [x.path for x in samples]
        references = [x.reference for x in samples]

        self.dataset_audio_time_length = 0
        for audio_file in paths:
            audio_duration = librosa.get_duration(path=audio_file)
            self.dataset_audio_time_length += audio_duration

        # Model inference
        model_inference_start = time.time()
        transcribes = self.model.transcribe(paths2audio_files=paths, batch_size=6)

        model_inference_end = time.time()

        self.model_inference_time = model_inference_end - model_inference_start

        # Metrics
        hypotheses = [t.upper() for t in transcribes]

        mem_usage_stop = memory_usage()[0]

        reporter = StatisticsReporter(self.nemo_model_path, self.statistics_file_prefix)
        reporter.save_statistics(
            len(samples),
            mem_usage_stop - mem_usage_start,
            self.model_load_time,
            self.model_inference_time,
            self.dataset_audio_time_length,
            references,
            hypotheses
        )
        reporter.save_errors(samples, hypotheses)

    def __load_model(self):
        # I consider the time to restore an existing model as the time to load it
        model_load_time_start = time.time()
        self.model = nemo_asr.models.ASRModel.restore_from(restore_path=self.nemo_model_path)
        model_load_time_end = time.time()
        self.model_load_time = model_load_time_end - model_load_time_start
