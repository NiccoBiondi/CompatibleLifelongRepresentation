# CL<sup>2</sup>R: Compatible Lifelong Learning Representations

This repo contains the code of "CL<sup>2</sup>R: Compatible Lifelong Learning Representations", _Niccolò Biondi, Federico Pernici, Matteo Bruni, Daniele Mugnai, and Alberto Del Bimbo_, ACM Transactions on Multimedia Computing, Communications, and Applications 

## Abstract

<p align="center">
  <img src="imgs/method.png" width="70%">
</p>

> In this article, we propose a method to partially mimic natural intelligence for the problem of lifelong learning representations that are compatible. We take the perspective of a learning agent that is interested in recognizing object instances in an open dynamic universe in a way in which any update to its internal feature representation does not render the features in the gallery unusable for visual search. We refer to this learning problem as Compatible Lifelong Learning Representations (CL<sup>2</sup>R), as it considers compatible representation learning within the lifelong learning paradigm. We identify stationarity as the property that the feature representation is required to hold to achieve compatibility and propose a novel training procedure that encourages local and global stationarity on the learned representation. Due to stationarity, the statistical properties of the learned features do not change over time, making them interoperable with previously learned features. Extensive experiments on standard benchmark datasets show that our CL<sup>2</sup>R training procedure outperforms alternative baselines and state-of-the-art methods. We also provide novel metrics to specifically evaluate compatible representation learning under catastrophic forgetting in various sequential learning tasks. Code is available at https://github.com/NiccoBiondi/CompatibleLifelongRepresentation.


## Install and Train
```sh
git clone https://github.com/NiccoBiondi/CompatibleLifelongRepresentation.git
cd CompatibleLifelongRepresentation
pip install -r requirements.txt
pip install -e .

# run experiment
cl2r --config_path config.yml
```

To modify hyperparameters please modify the [config](config.yml) file.

## Authors

- Niccolò Biondi <niccolo.biondi (at) unifi.it>[![GitHub](https://img.shields.io/badge/NiccoBiondi-%23121011.svg?style=plastic&logo=github&logoColor=white)](https://github.com/NiccoBiondi)
- Federico Pernici <federico.pernici (at) unifi.it> [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/FedPernici.svg?style=social&label=FedPernici)](https://twitter.com/FedPernici)
- Matteo Bruni <matteo.bruni (at) unifi.it>[![GitHub](https://img.shields.io/badge/matteo--bruni-%23121011.svg?style=plastic&logo=github&logoColor=white)](https://github.com/matteo-bruni)
- Daniele Mugnai <mugnai.mugnai (at) gmail.com>
- Alberto Del Bimbo <alberto.delbimbo (at) unifi.it>

## Citation

If you find this code useful for your research, please cite our paper:

```
@article{biondi2023cl2r,
  title={CL2R: Compatible Lifelong Learning Representations},
  author={Biondi, Niccolo and Pernici, Federico and Bruni, Matteo and Mugnai, Daniele and Bimbo, Alberto Del},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={18},
  number={2s},
  pages={1--22},
  year={2023},
  publisher={ACM New York, NY}
}
```


# TODO
- [ ] add creation pairs (at runtime for valiation)
- [ ] add model 
- [ ] add evaluation 
- [ ] add checkpoints saving
- [ ] add train function 
- [ ] add correct args  

