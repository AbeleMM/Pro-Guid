# Pro-Guid

Codebase is based on [DeFoG](https://github.com/manuelmlmadeira/DeFoG), which serves as the backbone model for the test-time extrapolation task.

With [Conda](https://conda-forge.org/) installed, use `conda env create -p .env -f environment.yaml` to set up the project dependencies.

To run experiments, use the following command format with `src` as the working directory:

```sh
python main.py ++general.wandb=disabled +experiment=$D dataset=$D ++general.test_only=$CHECKPOINT_DIR  ++sample.sampler=$S ++sample.num_nodes=$N
```

The datasets `D` showcased in the paper are `planar`, `tree`, and `sbm`.

DeFoG authors make checkpoints for the different datasets [available online](https://drive.switch.ch/index.php/s/MG7y2EZoithAywE).

Samplers `S` showcased in the paper are: `default` (original), `grad-guid`, and `pro-guid`.

Extrapolation using the `grad-guid` sampler requires additionally setting the guidance weight `++general.guidance_weight=$W`. All experiments in the paper have `W=5`.

Extrapolation using the `pro-guid` sampler requires additionally setting the guidance weight and anchor size `++general.guidance_weight=$W ++++sample.n_anc=$A`. Values for the two parameters in the paper experiments:

| Dataset | W  | A  |
| :------ | :- | :- |
| Planar  | 7  | 3  |
| Tree    | 8  | 4  |
| SBM     | 4  | 20 |
