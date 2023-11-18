# NemoAsrMetrics
This project has been made to test NVIDIA NeMo models and runtime on minicomputers

## Installation
Before using these project and its notebooks, check the machines section below to see if specific asr-runtime is available.
Overall, try to install NeMo library, then ONNX, or Riva as a last resort

## Jupyter notebooks to use
   - _[ASR Librispeech Wav Audios Preparation.ipynb](ASR%20Librispeech%20Wav%20Audios%20Preparatioin.ipynb)_ - unzipping a dataset in a
     Librispeech format and converting audio samples to a `.wav`-format suitable for
     NeMo-models. Put models in the `.nemo`/`.onnx` -models into `models`-folder
   - _[ASR NeMo-ONNX Conversion.ipynb](ASR%20NeMo-ONNX%20Conversion.ipynb)_ - converting an already downloaded .nemo-file
     to a `.onnx`-file with model to use in ONNX-runtime
   - _[ASR Librispeech Wav Audios Traversal.ipynb](ASR%20Librispeech%20Wav%20Audios%20Traversal.ipynb)_ - one cell with gathering audio samples
     from unzipped dataset of wav-s, one cell to feed these sample with 3 different runtimes: [ONNX](https://onnxruntime.ai/)/[NeMo](https://github.com/NVIDIA/NeMo)/[Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html)

## Minicomputers Journey
1) I have chosen one of the test sets of [LibriSpeech ASR corpus](https://www.openslr.org/12), namely `test-other.tar.gz` and written a python-code to unzip files and
   collect them into a set of samples to feed the model later
2) I have chosen one of the suggested [STT En Citrinet 1024](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_1024) model as a top one
   listed to use from NeMo-library
3) With `nemo-toolkit` I converted the model to the [ONNX](https://onnxruntime.ai/)-model to measure any
   changes in the model performance
4) Machines: one of the hardest part in my work was trying to create an environment
   to run my notebook for performance measuring on a variety of existing set of
   machines

## Machines and their specific details
   - **Raspberry Pi _3_ Model B** - _no success_

   Multiple errors during installation, [pip](https://pip.pypa.io/en/stable/getting-started/) cannot find the proper version if wheels to install needed libraries multiple incompatibility issues
   
   - **Raspberry Pi 4 Model B** - _**Partial Success**_

   NeMo library installed successfully, but running the NeMo-cell has not in jupyter wasn't successful. Although, ONNX-cell run successfully 

   - **Apple Silicon M1** - _no success_

   Created a condo environment, installed all packages successfully, but the code produces errors. I didn’t proceed to try, since these are not our target devices

   - **[Nvidia Jetson Nano Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)** - _no success_

   Created a condo environment, installed all packages successfully, but the code produces errors. I didn’t proceed to try, since these are not our target devices

   - **Intel Devices** - _**Success**_
   
   _(Mac Mini Intel Core i7, Intel NUC 12 wshi3000, AsRock, AsRock DeskMini h470)_

   - **[Nvidia Jetson Xavier NX](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit)** - _Success with Riva_
   
   NeMo-library couldn't be installed, instead Riva-api has been used to measure it

