import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf


class NemoOnnxConverter:
    def __init__(self, nemo_model_path, onnx_model_path, config_model_path):
        self.nemo_model_path = nemo_model_path
        self.onnx_model_path = onnx_model_path
        self.config_model_path = config_model_path

    def produce_onnx(self):
        model_reco = nemo_asr.models.ASRModel.restore_from(restore_path=self.nemo_model_path)
        model_reco.export(self.onnx_model_path, onnx_opset_version=12)
        model_reco = nemo_asr.models.ASRModel.restore_from(
            restore_path=self.nemo_model_path,
            return_config=True
        )
        textfile = open(self.config_model_path, 'w')
        textfile.write(str(OmegaConf.to_yaml(model_reco)))
        textfile.close()
