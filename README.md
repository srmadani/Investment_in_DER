
# Investment in Vehicle-to-Grid and Distributed Energy Resources 🏠🚗⚡

This repository contains the implementation of the research paper "Investment in vehicle-to-grid and distributed energy resources: Distributor versus prosumer perspectives and the impact of rate structures". The paper presents a cost-benefit analysis of various investment scenarios for a distributor and a prosumer, considering the impact of different rate structures.

## Project Description 📖

The main objectives of this project are:

- To evaluate the profitability of investments in distributed energy resources (DER), vehicle-to-grid (V2G) technology, and energy management systems (EMS) from the perspectives of both the distributor and the prosumer.
- To analyze the impact of different rate structures on the viability of these investment scenarios.

The study utilizes real-world data from four single-detached houses in Vermont, USA, to model the electricity demand, solar PV generation, and the behavior of the distributor and prosumer under various investment and rate structure scenarios.

## Implementation 💻

The implementation of the proposed models and analysis is provided in a Jupyter Notebook called `Main.ipynb`. This notebook includes the following steps:

1. **Data Loading and Preprocessing**: The electricity consumption and solar PV generation data for the four houses are loaded and preprocessed.
2. **Modeling and Optimization**: The distributor and prosumer models are implemented using the Gurobi optimization solver in Python. The models consider different investment scenarios and rate structures to evaluate their profitability.
3. **Visualization and Analysis**: The results of the optimization models are visualized and analyzed, including the net present value (NPV), internal rate of return (IRR), energy flows, and peak loads for each scenario.
4. **Sensitivity Analysis**: The sensitivity of the results to changes in load patterns, battery capacity, and solar panel system size is investigated.

### Required Libraries 📚

The following Python libraries are used in the implementation:

- `numpy`: for numerical operations
- `pandas`: for data manipulation and analysis
- `matplotlib`: for static data visualization
- `plotly`: for interactive data visualization
- `gurobipy`: for optimization modeling and solving
- `os`: for file path operations

To run the Jupyter Notebook, you will need to have Gurobi Optimizer installed and a valid Gurobi license. You can obtain a free academic license from the [Gurobi website](https://www.gurobi.com).

## How to Use 🚀

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/srmadani/Investment_in_DER.git
   ```

2. Install the required libraries, including Gurobi Optimizer.
3. Open the `Main.ipynb` Jupyter Notebook and run the cells to reproduce the analysis and results.

## Contribution 🤝

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. Contributions are welcome!

## Acknowledgments 🙏

The authors would like to acknowledge the support from the Fonds de Recherche du Quebec – Nature et Technologies (FRQNT) [300234]. We also thank Eric Rondeau for his help and dcbel and a utility operating in the Northeast United States for their collaboration.

## References 📚

Madani, S., & Pineau, P.-O. (2024). Investment in vehicle-to-grid and distributed energy resources: Distributor versus prosumer perspectives and the impact of rate structures. Utilities Policy, 88, 101736. [https://doi.org/10.1016/j.jup.2024.101736](https://doi.org/10.1016/j.jup.2024.101736)
