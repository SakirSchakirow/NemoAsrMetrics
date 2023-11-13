import os
from datetime import datetime
from datetime import timedelta
import csv
import jiwer

from toolkit.statistics import real_time_factor, round_percent, chop_microseconds

DATETIME_FORMAT = "%d-%m-%Y %H-%M-%S "

STATISTICS_CSV = 'model_statistics.csv'
STATISTICS_HEADERS = ['Model path',
                      'Model size (MB)',
                      'Memory usage (MB)',
                      'Model load time (sec)',
                      'Inference time (hh:mm:ss)',
                      'Dataset samples count',
                      'Dataset length (hh:mm:ss)',
                      'RTF (%)',
                      'WER (%)', 'MER (%)', 'WIL (%)', 'WIP (%)', 'CER (%)']

ERRORS_CSV = 'model_errors.csv'
ERRORS_HEADERS = ['id', 'path', 'reference', 'hypothesis']


class StatisticsReporter:
    def __init__(self, model_file_path, file_prefix=''):
        self.model_file_path = model_file_path
        self.file_prefix = file_prefix

    def save_statistics(self,
                        samples_count,
                        peak_memory_usage,
                        load_time_seconds,
                        inference_time_seconds,
                        dataset_audio_time_length_seconds,
                        references,
                        hypotheses
                        ):
        values = [self.model_file_path,
                  self.__get_file_size_or_none(),
                  round(peak_memory_usage, 2),
                  round(load_time_seconds, 3),
                  chop_microseconds(inference_time_seconds),
                  samples_count,
                  chop_microseconds(dataset_audio_time_length_seconds),
                  real_time_factor(inference_time_seconds, dataset_audio_time_length_seconds),
                  round_percent(jiwer.wer(references, hypotheses)),
                  round_percent(jiwer.mer(references, hypotheses)),
                  round_percent(jiwer.wil(references, hypotheses)),
                  round_percent(jiwer.wip(references, hypotheses)),
                  round_percent(jiwer.cer(references, hypotheses))
                  ]
        statistics_file = self.__create_statistics_csv()
        self.__write_csv_row(statistics_file, STATISTICS_HEADERS, values)

    def save_errors(self, samples, hypotheses):
        errors = []
        for sample, hypothesis in zip(samples, hypotheses):
            if jiwer.wer(sample.reference, hypothesis):
                values = [sample.id, sample.path, sample.reference, hypothesis]
                errors.append(values)

        errors_file = self.__create_errors_csv()
        self.__write_csv_rows(errors_file, ERRORS_HEADERS, errors)

    def __create_statistics_csv(self):
        return self.__create_csv(STATISTICS_CSV, STATISTICS_HEADERS)

    def __create_errors_csv(self):
        return self.__create_csv(ERRORS_CSV, ERRORS_HEADERS)

    @staticmethod
    def __write_csv_row(csv_file, headers, values):
        with open(csv_file, 'a', encoding='UTF8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            row = {headers[i]: values[i] for i in range(len(values))}
            writer.writerow(row)

        # writes multiple rows (each row is an array of values)

    @staticmethod
    def __write_csv_rows(csv_file, headers, rows):
        with open(csv_file, 'a', encoding='UTF8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            for values in rows:
                row = {headers[i]: values[i] for i in range(len(values))}
                writer.writerow(row)

    def __create_csv(self, csv_file_name: str, headers) -> str:
        filename = self.file_prefix + datetime.now().strftime(DATETIME_FORMAT) + csv_file_name
        with open(filename, 'x', encoding='UTF8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        return filename

    def __get_file_size_or_none(self):
        try:
            result = os.stat(self.model_file_path).st_size / (1024 * 1024)
            return result
        except Exception as e:
            return None  # Not a path to model
