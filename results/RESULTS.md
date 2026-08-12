# Avellaneda-Stoikov Monte Carlo Replication — Results

## Simulation Parameters

- S0 = 100.0, T = 1.0, sigma = 2.0
- dt = 0.005, N = 200
- A = 140.0, k = 1.5
- n_paths = 1000, master_seed = 42
- gammas = [0.01, 0.1, 0.5]

## Experiment 1: Full Time-Varying Spread

### Research Hypotheses Tested
1. Inventory-aware quoting reduces std(Profit) vs symmetric benchmark
2. Inventory-aware quoting reduces std(Final q) vs symmetric benchmark
3. At low gamma (0.01), strategies converge behaviorally
4. At high gamma (0.5), stronger profit-risk trade-off

### Summary Table (Experiment 1)

|   gamma | strategy       |   mean_profit |   std_profit |   mean_final_q |   std_final_q |
|--------:|:---------------|--------------:|-------------:|---------------:|--------------:|
|    0.01 | inventory_full |       68.451  |      8.94044 |          0.029 |       5.07227 |
|    0.01 | symmetric_full |       69.2594 |     13.5467  |          0.256 |       8.85929 |
|    0.1  | inventory_full |       64.7544 |      6.38173 |          0.097 |       2.94249 |
|    0.1  | symmetric_full |       67.6508 |     13.3196  |         -0.357 |       8.46186 |
|    0.5  | inventory_full |       48.5389 |      5.92983 |         -0.014 |       1.93534 |
|    0.5  | symmetric_full |       58.9502 |     11.4489  |          0.321 |       7.24544 |

## Experiment 2: Constant Spread (Table-Faithful)

### Summary Table (Experiment 2)

|   gamma | strategy        |   constant_spread |   mean_profit |   std_profit |   mean_final_q |   std_final_q |
|--------:|:----------------|------------------:|--------------:|-------------:|---------------:|--------------:|
|    0.01 | inventory_const |           1.32891 |       67.8192 |      8.90781 |         -0.111 |       5.07961 |
|    0.01 | symmetric_const |           1.32891 |       68.8027 |     13.7658  |          0.212 |       8.73208 |
|    0.1  | inventory_const |           1.29077 |       63.9907 |      6.08237 |         -0.002 |       2.88727 |
|    0.1  | symmetric_const |           1.29077 |       68.4266 |     13.5892  |          0.359 |       8.71034 |
|    0.5  | inventory_const |           1.15073 |       24.3018 |      5.05884 |          0.024 |       1.93989 |
|    0.5  | symmetric_const |           1.15073 |       68.5691 |     13.5272  |         -0.476 |       9.41084 |

## Key Findings

### Experiment 1
- gamma=0.01: Inventory strategy std(Profit)=8.9404 (lower than symmetric=13.5467); std(q)=5.0723 (lower than 8.8593)
- gamma=0.1: Inventory strategy std(Profit)=6.3817 (lower than symmetric=13.3196); std(q)=2.9425 (lower than 8.4619)
- gamma=0.5: Inventory strategy std(Profit)=5.9298 (lower than symmetric=11.4489); std(q)=1.9353 (lower than 7.2454)

### Experiment 2
- gamma=0.01: Inventory strategy std(Profit)=8.9078 (lower than symmetric=13.7658); std(q)=5.0796 (lower than 8.7321)
- gamma=0.1: Inventory strategy std(Profit)=6.0824 (lower than symmetric=13.5892); std(q)=2.8873 (lower than 8.7103)
- gamma=0.5: Inventory strategy std(Profit)=5.0588 (lower than symmetric=13.5272); std(q)=1.9399 (lower than 9.4108)

## Figures Generated

### Experiment 1
- `exp1_path_gamma*_inventory_full.png`: Representative path plots
- `exp1_path_gamma*_symmetric_full.png`: Representative path plots
- `exp1_histograms_gamma*.png`: Terminal profit/inventory histograms
- `exp1_summary_comparison.png`: Bar charts of summary statistics

### Experiment 2
- `exp2_path_gamma*_inventory_const.png`: Representative path plots
- `exp2_path_gamma*_symmetric_const.png`: Representative path plots
- `exp2_histograms_gamma*.png`: Terminal profit/inventory histograms
- `exp2_summary_comparison.png`: Bar charts of summary statistics