# Interactive Motion Model Visualizer

Two-window OpenCV app to **build intuition for mobile robot motion models**.  
- **Motion PDF**: sampled end-poses (colored by final heading).  
- **Control Space**: move the mouse to set commands/odometry terms.

Noise is sampled from a chosen distribution (default: gaussian) with tunable \(\alpha\) parameters.

---
# How to Run
# Prerequisites
```
pip install numpy opencv-python
```
## Cloning
```
git clone https://github.com/miggyval/metr4202_motion_demo.git
cd metr4202_motion_demo
```
## Running the Demo
```
python demo.py
```

---
# Velocity Motion Model
Recall that the motion model is given by:

$$
\begin{align*}
x_{1} &= x_{0} - \frac{v}{\omega}\sin{\theta_{0}} + \frac{v}{\omega}\sin{(\theta_{0}+\omega\Delta t)} \\
y_{1} &= y_{0} - \frac{v}{\omega}\cos{\theta_{0}} - \frac{v}{\omega}\cos{(\theta_{0}+\omega\Delta t)} \\
\theta_{1} &= \theta_{0} + \omega\Delta t+\gamma \Delta t
\end{align*}
$$


---




---

## Features
- Velocity and Odometry motion models
- Dark/light theme toggle
- Adjustable noise via \(\alpha\)-parameters (mantissa/exponent stepping)
- Distributions: triangular (default), Gaussian (Uniform present in code)

---
