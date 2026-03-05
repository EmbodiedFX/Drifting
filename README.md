# Generative Modeling via Drifting (Toy Example)

A minimal **Drifting generative modeling** toy example implemented in pure NumPy, with all code contained in a single Jupyter notebook: `toy.ipynb`. The goal is to help you understand the training dynamics and generation behavior of Drifting, rather than to provide any SOTA results or production-ready code.

中文说明见：[README-CH.md](./README-CH.md)

---

## Experiment setup

- **Real data distribution**  
  - A 4D Gaussian mixture with two modes;  
  - The two mode means are approximately around (1, 0, 0, 0) and (0, 1, 0, 0), with isotropic covariance $\sigma^2 I$;  
  - A fixed set of data points is generated at the beginning of the notebook and kept unchanged throughout training.

- **Noise and generator**  
  - Noise is drawn from a standard normal distribution $\varepsilon \sim \mathcal{N}(0, I)$, with dimension comparable to the data;  
  - The generator is a two-hidden-layer MLP with `tanh` activations, initialized with small Gaussian weights and zero biases;  
  - At inference time, generation is a single forward pass: `x = f_theta(eps)`.

- **Drifting field**  
  - Positive samples: the fixed real data points;  
  - Negative samples: the current batch of generated samples;  
  - An RBF kernel $k(x, y) = \exp(-\|x - y\|^2 / \tau)$ is used for mean-shift, defining  
    $V(x) = m_{\text{pos}}(x) - m_{\text{neg}}(x)$;  
  - In each training iteration, generated samples are first drifted to $x_{\text{drift}} = x + V(x)$, and then the generator is trained with a mean squared error loss to fit these drifted targets.

---

## Expected results

With default hyperparameters and training from scratch, you can roughly observe:

- In early iterations, generated samples form some random cloud near the data space but far from the two real modes;  
- As training progresses, the Drifting field gradually pushes generated samples toward the two data modes, and the model distribution starts to show a “two-cluster” structure;  
- The positive mean-shift \(m_{\text{pos}}(x)\) pulls the point cloud toward high-density regions of the real data;  
- The negative mean-shift \(m_{\text{neg}}(x)\) provides a repulsive signal within the model distribution, helping to alleviate mode collapse where all samples fall onto a single mode.

---

## Steps to Follow

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-name>/Drifting.git
   cd Drifting
   ```

2. **Install dependencies (if not already installed)**

    It is recommended to use a Python virtual environment and install at least:

    ```bash
    pip install numpy matplotlib jupyter
    ```

3. Launch Jupyter and open the notebook

    ```bash
    jupyter notebook
    ```
    In your browser, open `toy.ipynb` and run all cells from top to bottom.
    
After running the notebook, you will see numerical outputs and visualizations of the training process, which should give you an intuitive understanding of the Drifting generative modeling paradigm.

---

## More possibilities

The notebook `toy.ipynb` is deliberately structured so that key components can be easily swapped or tweaked to gain more insights:

- **Data distribution**  
  The code that constructs the real data can be modified to:
  - Use different mode locations (e.g., rings, more modes);  
  - Change covariance structures (anisotropic, higher-dimensional, more sparse, etc.);  
  - Change the data dimension.  

- **Generator architecture and hyperparameters**  
  You can experiment with:
  - Different noise / data dimensions;  
  - Different hidden sizes, number of hidden layers, or activation functions (e.g., ReLU instead of `tanh`);  
  - Learning rate, number of training iterations, and other optimization hyperparameters.  

- **Drifting-related parameters**  
  - Kernel temperature \(\tau\) (more local vs more global behavior);  
  - `drift_scale` (large vs small step size per iteration);  
  - `batch_size`, which affects the statistical stability of the negative mean-shift.  

- **Logging and visualization**  
  - Whether to print detailed intermediate states (per-sample values, mean-shifts, loss, etc.);  
  - Whether to redirect logs to a text file;  
  - How to draw scatter plots and metric curves.  

By tweaking these components, you can systematically observe how different settings affect Drifting’s ability to alleviate mode collapse and how the geometry of the learned model distribution changes.