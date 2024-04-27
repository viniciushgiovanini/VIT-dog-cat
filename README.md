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

- WSL ja vem com o cuda pre instalado, mas instale por completo.

```
sudo apt install nvidia-cuda-toolkit
```

- No WSL2 em uma distro linux baixe o tensorflow cuda.

```
pip install tensorflow[and-cuda]
```

- Verifique a versão do Cuda instalada por default no WSL2
```
nvidia-smi
watch -n 1 nvidia-smi
```

- Caso seja preciso instalar outra versão do Cuda por cima, tem que verificar pelo seguinte comando
```
nvcc -V
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


## Linguagens de Desenvolvimento

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="50px"/>&nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="50px"/>


## Desenvolvimento ✏

**Feito por**: [Vinícius Henrique Giovanini](https://github.com/viniciushgiovanini)