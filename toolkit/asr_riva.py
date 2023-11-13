import time
import io
import IPython.display as ipd
import grpc
import librosa
import riva.client

from memory_profiler import memory_usage

from toolkit.statistics_reporter import StatisticsReporter


class RivaProcessor:
    def __init__(self, statistics_file_prefix=''):
        self.riva_asr = None
        self.mem_usage = None
        self.model_load_time = None
        self.model_inference_time = None
        self.dataset_audio_time_length = None
        self.statistics_file_prefix = statistics_file_prefix
        self.riva_asr_config = self.__setup_config()

    def process(self, samples):
        if len(samples) == 0:
            raise Exception("Samples are empty! Provide at least one sample in the samples")

        # Forming samples to inference
        audio_files = [x.path for x in samples]
        self.dataset_audio_time_length = sum(librosa.get_duration(path=file) for file in audio_files)

        references = [x.reference for x in samples]

        transcribes = self.__get_transcribes(audio_files)
        hypotheses = [t.upper() for t in transcribes]

        reporter = StatisticsReporter("citrinet_1024", self.statistics_file_prefix)
        reporter.save_statistics(
            len(samples),
            self.mem_usage,
            self.model_load_time,
            self.model_inference_time,
            self.dataset_audio_time_length,
            references,
            hypotheses
        )
        reporter.save_errors(samples, hypotheses)

    def __get_transcribes(self, audio_files):
        self.__load_model()

        transcribes = []
        self.mem_usage = 0
        self.model_inference_time = 0

        for audio_file in audio_files:
            transcript, mem_usage, model_inference_time = self.__get_transcribe(audio_file)
            transcribes.append(transcript)
            self.mem_usage += mem_usage
            self.model_inference_time += model_inference_time

        return transcribes

    def __get_transcribe(self, audio_file):
        with io.open(audio_file, 'rb') as fh:
            content = fh.read()

        mem_usage_start = memory_usage()[0]
        model_inference_start = time.time()

        response = self.riva_asr.offline_recognize(content, self.riva_asr_config)
        transcript = response.results[0].alternatives[0].transcript

        mem_usage_stop = memory_usage()[0]
        model_inference_end = time.time()

        return transcript, mem_usage_stop - mem_usage_start, model_inference_end - model_inference_start

    def __load_model(self):
        model_load_start = time.time()
        auth = riva.client.Auth(uri='localhost:50051')
        self.riva_asr = riva.client.ASRService(auth)
        model_load_end = time.time()
        elapsed = model_load_end - model_load_start
        self.model_load_time = elapsed

    @staticmethod
    def __setup_config():
        # Set up an offline/batch recognition request
        config = riva.client.RecognitionConfig(enable_automatic_punctuation=False)
        # req.config.encoding = ra.AudioEncoding.LINEAR_PCM    # Audio encoding can be detected from wav
        # req.config.sample_rate_hertz = 0                     # Sample rate can be detected from wav and resampled if needed
        config.language_code = "en-US"  # Language code of the audio clip
        config.max_alternatives = 1  # How many top-N hypotheses to return
        config.enable_automatic_punctuation = False  # Add punctuation when end of VAD detected
        config.audio_channel_count = 1  # Mono channel
        return config
