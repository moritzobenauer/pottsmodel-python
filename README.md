# pottsmodel-python

Performing the *Wang Landau Algorithm* for the *Q-State Potts Model* on a two-dimensional square lattice.

## Theoretical Background

> 
> The Wang Landau algorithm estimates the density of states (DOS) by performing random moves, updating probabilities based on energy changes, and iteratively flattening a histogram. It allows uniform energy space sampling, facilitating accurate thermodynamic property calculations over various temperatures, overcoming limitations of traditional *Monte Carlo* methods dependent on specific temperatures.

> 
> The Potts model is a generalization of the Ising model in statistical mechanics. It describes interacting spins on a lattice, where each spin can be in one of [0,Q) states. The model is used to study phase transitions, critical phenomena, and various problems in condensed matter physics and materials science.

## Usage

## Thermodynamic Results

## To-Do

- Implement proper energy boundaries (upper and lower energy limits for proper sampling)

- Parallelization
- Include calculations for order parameter depending on the temperature
