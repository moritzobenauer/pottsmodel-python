







## To-Do




![raccoon](./figures/logo-wide.png)
# pottsmodel-python

 [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


**Performing the *Wang Landau Algorithm* for the *Q-State Potts 
Model* on a two-dimensional square lattice.**


## Theoretical Background

> [!NOTE]
> The **Wang Landau algorithm** estimates the density of states (DOS) by performing random moves, updating probabilities based on energy changes, and iteratively flattening a histogram. It allows uniform energy space sampling, facilitating accurate thermodynamic property calculations over various temperatures, overcoming limitations of traditional *Monte Carlo* methods dependent on specific temperatures. The **Potts model** is a generalization of the Ising model in statistical mechanics. It describes interacting spins on a lattice, where each spin can be in one of [0,Q) states. The model is used to study phase transitions, critical phenomena, and various problems in condensed matter physics and materials science.

## How to use

```bash
python main.py -g 10 -f example -z 0.8 -m 0.001 -n 100 -q 2
```

| Parameter | Default       |   Description |
| :---      | :---          | :---          |
| -g        | 10      | gridsize                            |
| -f        | WLA-RUN       | directory name               |
| -z        | 0.8          | WLA histogram flatness                                      |
| -m        | 0.000001         | Final ln(f) value |
| -n        | 100          | number of bins  |
| -q        | 2          | number of possible q states |




## Thermodynamic Results

### Ising Model (Q=2)
For the $Q=2$ case a second order phase tranisition can be observed. The vertical line indicates the analytical *Onsager* solution.
![ising_lnge](./figures/ising_lnge.png)
![ising_lnge](./figures/ising_c.png)
![ising_lnge](./figures/ising_s.png)

### Higher Order (Q=8)
For the $Q=8$ case a first order phase tranisition can be observed.
![ising_lnge](./figures/q8_lnge.png)
![ising_lnge](./figures/q8_c.png)
![ising_lnge](./figures/q8_s.png)

## Known bugs and To-Do's

> [!WARNING]
> - Implement proper energy boundaries (upper and lower energy limits for proper sampling). This currently leads to a small inconsistency at $E=-1.0$ for the $Q=8$ case.
>
>- Parallelization
>- Include calculations for order parameter depending on the temperature


