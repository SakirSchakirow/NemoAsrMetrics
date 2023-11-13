import time
import onnx
import onnxruntime
import tempfile
import librosa
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
import os
import torch
import yaml
from omegaconf import DictConfig
import json
import numpy as np
from memory_profiler import memory_usage

import nemo.collections.asr.metrics.wer
from onnxruntime import SessionOptions, ExecutionMode

from toolkit.statistics_reporter import StatisticsReporter


class OnnxProcessor:
    def __init__(self, onnx_model_path, config_model_path, statistics_file_prefix=''):
        self.mem_usage = None
        self.model_load_time = None
        self.model_inference_time = None
        self.dataset_audio_time_length = None
        self.onnx_model_path = onnx_model_path
        self.config_model_path = config_model_path
        self.statistics_file_prefix = statistics_file_prefix

    def process(self, samples):
        if len(samples) == 0:
            raise Exception("Samples are empty! Provide at least one sample in the samples")

        # Forming samples to inference
        audio_files = [x.path for x in samples]
        references = [x.reference for x in samples]

        with open(self.config_model_path) as f:
            params = yaml.safe_load(f)

        # NeMo creates preprocessor and decoder this inside EncDecCTCModel,
        # here we do it explicitly since we are going to use onnx and not EncDecCTCModel
        preprocessor_cfg = DictConfig(params).preprocessor
        preprocessor = EncDecCTCModel.from_config_dict(preprocessor_cfg)

        decoder_cfg = DictConfig(params).decoder
        decoding_cfg = DictConfig(params).decoding
        decoder = nemo.collections.asr.metrics.wer.CTCDecoding(decoding_cfg, decoder_cfg.vocabulary)

        labels = params['decoder']['vocabulary']  # "vocabulary" could be something else depending on model
        out_batch = []

        self.dataset_audio_time_length = 0
        # this part is just copy-pasted from NeMo library code
        with tempfile.TemporaryDirectory() as dataloader_tmpdir:
            with open(os.path.join(dataloader_tmpdir, 'manifest.json'), 'w') as fp:
                for audio_file in audio_files:
                    audio_duration = librosa.get_duration(path=audio_file)
                    self.dataset_audio_time_length += audio_duration
                    entry = {
                        'audio_filepath': audio_file,
                        'duration': audio_duration,
                        'text': 'nothing'
                    }
                    fp.write(json.dumps(entry) + '\n')

            config = {'paths2audio_files': audio_files, 'batch_size': 1, 'temp_dir': dataloader_tmpdir}
            temporary_datalayer = self.__get_nemo_dataset(config, labels)
            for test_batch in temporary_datalayer:
                out_batch.append(test_batch)

        transcribes = self.__get_transcribes(preprocessor, decoder, out_batch)

        # Metrics
        hypotheses = [t.upper() for t in transcribes]

        reporter = StatisticsReporter(self.onnx_model_path, self.statistics_file_prefix)
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

    def __get_transcribes(self, preprocessor, decoder, out_batch):
        mem_usage_start = memory_usage()[0]

        self.load_model()

        sess_opt = SessionOptions()
        sess_opt.execution_mode = ExecutionMode.ORT_SEQUENTIAL

        sess = onnxruntime.InferenceSession(self.onnx_model_path,
                                            sess_opt,
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        audio_signal_input_name = sess.get_inputs()[0].name
        length_input_name = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name

        transcribes = []

        self.model_inference_time = 0
        for batch in out_batch:
            processed_signal, processed_signal_length = preprocessor(input_signal=batch[0], length=batch[1], )
            model_inference_start = time.time()
            output_logits = sess.run(
                [label_name],
                {
                    audio_signal_input_name: processed_signal.cpu().numpy(),
                    length_input_name: processed_signal_length.cpu().numpy()
                }
            )
            model_inference_end = time.time()
            elapsed = model_inference_end - model_inference_start
            self.model_inference_time += elapsed

            array_logits = np.asarray(output_logits)
            logits = torch.from_numpy(array_logits[0])
            greedy_predictions = logits.argmax(dim=-1, keepdim=False)
            hypotheses = decoder.ctc_decoder_predictions_tensor(greedy_predictions)
            transcribes.append(hypotheses[0][0].replace('▁', ' ').strip())

        mem_usage_stop = memory_usage()[0]
        self.mem_usage = mem_usage_stop - mem_usage_start

        return transcribes

    def load_model(self):
        model_load_start = time.time()
        onnx.load(self.onnx_model_path)
        model_load_end = time.time()
        elapsed = model_load_end - model_load_start
        self.model_load_time = elapsed

    @staticmethod
    def __get_nemo_dataset(config, labels, sample_rate=16000):
        augmentor = None

        config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': sample_rate,
            'labels': labels,
            'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
            'trim_silence': True,
            'shuffle': False,
        }

        dataset = AudioToCharDataset(
            manifest_filepath=config['manifest_filepath'],
            labels=config['labels'],
            sample_rate=config['sample_rate'],
            int_values=config.get('int_values', False),
            augmentor=augmentor,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            max_utts=config.get('max_utts', 0),
            blank_index=config.get('blank_index', -1),
            unk_index=config.get('unk_index', -1),
            normalize=config.get('normalize_transcripts', True),
            trim=config.get('trim_silence', True),
            parser=config.get('parser', 'en'),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,  # config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config.get('shuffle', False),
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
