# ReasonMotion Editor

This directory contains the code and scripts for the ReasonMotion Editor's visualization and operations.

## Generating Visualizations

To visualize the batch text grid, you can use the `visualize_batch_text_grid.py` script. 

### Usage Example

```bash
CUDA_VISIBLE_DEVICES=4 python visualize_batch_text_grid.py \
    --run_dir "your run dir" \
    --res_pk  "your .pk file" \
    --batch_start your batch start \
    --batch_end your batch end \
    --step your step \
    --seed your seed \
    --slidewindow your slidewindow
```

### Key Arguments

* **`--run_dir`**: The directory containing the model runs / checkpoints.
* **`--res_pk`**: Path to the `.pk` file containing the result poses or trajectory data.
* **`--batch_start`**, **`--batch_end`**, **`--step`**: Controls the range and step size for batch processing.
* **`--seed`**: Random seed for reproducibility.
* **`--slidewindow`**: The size of the sliding window used during visualization.
