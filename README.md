# Vision Transformer


- Teste dos VIT no dataset de Cachorro e Gato para teste de desempenho

## Run Code

- Instale a virtual environement.

```
python -m venv env
```

- Entre na virtural environment.

```
env/Scripts/activated
```

- Instale os pacotes requeridos no arquivo requirements.txt.

```
pip install -r requirements.txt
```

## Rodando Codigo do Tensorflow na GPU

- Verifique a versão do tensorflow e do cuda necessárias [https://www.tensorflow.org/install/source?hl=pt-br#tested_build_configurations](https://www.tensorflow.org/install/source?hl=pt-br#tested_build_configurations)

- Instale o CUDA, acessando o site da nvidia [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive).


- Rode os comando no runfile(local)
```
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

- Crie as variaveis de ambiente
```
nano ~/.bashrc
export PATH=$PATH:/urs/local/cuda/12.4/bin
export LD_LIBRARY_PATH=/urs/local/cuda-12.4/lib
source ~/.bashrc
```

- Verifique se o cuda foi instalado no WSL2

```
nvcc -V
```

- No WSL2 em uma distro linux baixe o tensorflow cuda.

```
pip install tensorflow[and-cuda]
```

- Verifique a versão do Cuda instalada por default no WSL2
```
nvidia-smi
watch -n 0.1 nvidia-smi
```

- Na versão 2.16.1 (26/04/2024) ele está bugada não reconhecendo a GPU, porém é a unica versão possivel de rodar transformer, então rode os seguintes codigos.

```
export NVIDIA_DIR=$(dirname $(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))
export LD_LIBRARY_PATH=$(echo ${NVIDIA_DIR}/*/lib/ | sed -r 's/\s+/:/g')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- Use o código abaixo para verificar se retorna a GPU da nvidia.

```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Rodando Codigo do Pytorch na GPU

- Instalar os pacotes do Pytorch

```
pip install torch torchvision
```

- Ter o cuda instalado igual do Tensorflow, e testar o codigo abaixo para saber se identifica a GPU

```
print(torch.cuda.is_available())
```

## Linguagens de Desenvolvimento

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="50px"/>


## Desenvolvimento ✏

**Feito por**: [Vinícius Henrique Giovanini](https://github.com/viniciushgiovanini)