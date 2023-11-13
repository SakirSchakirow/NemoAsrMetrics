from datetime import timedelta


def real_time_factor(processing_time, audio_length, decimals=4):
    """ Real-Time Factor (RTF) is defined as processing-time / length-of-audio. """

    rtf = (processing_time / audio_length)

    return round_percent(rtf)


def round_percent(ratio):
    return round(ratio * 100, 2)


def chop_microseconds(seconds):
    delta = timedelta(seconds=seconds)
    return delta - timedelta(microseconds=delta.microseconds)
