# Adaptive Compression in Federated Learning via Side Information
PyTorch implementation of the KLMS framework by the authors of the AISTATS 2024 paper "Adaptive Compression in Federated Learning via Side Information".

> [Adaptive Compression in Federated Learning via Side Information](https://arxiv.org/pdf/2306.12625.pdf) <br/>
>[Berivan Isik](https://sites.google.com/view/berivanisik), [Francesco Pase](https://sites.google.com/view/pasefrance), [Deniz Gunduz](https://www.imperial.ac.uk/people/d.gunduz),  [Sanmi Koyejo](https://cs.stanford.edu/people/sanmi/), [Tsachy Weissman](https://web.stanford.edu/~tsachy/), [Michele Zorzi](https://signet.dei.unipd.it/zorzi/) <br/>
> The 27th International Conference on Artificial Intelligence and Statistics (AISTATS), 2024. <br/>


## Environment setup:
Packages can be found in `federated_klms.yml`.

## Training:
Set the `params_path` in `main.py` to the the path of the `{}.yaml` file with the desired model and dataset. The default parameters can be found in the provided `{}.yaml` files. To train the model, run:

```
python3 main.py
```
