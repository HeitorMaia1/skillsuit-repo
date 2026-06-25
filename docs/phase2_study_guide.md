# SkillSuit — Phase 2 Study Guide

The science you own. This guide is built for self-study: paste any section into Claude chat and ask
it to derive the math with you, step by step. It assumes you are strong in physics and calculus and
**new to the ML and sensor side** — so every ML/sensor term is defined inline at first use. Physics
terms (Hamiltonian, phase space, Lagrangian) are treated as known but refreshed briefly.

Goal of Phase 2: be able to *defend* the scientific spine of SkillSuit — why a **pHNN**
(Port-Hamiltonian Neural Network) is the right model for human motor data, how orientation is
recovered from raw sensors, and why no existing system fills our gap. You make the Phase 5 modeling
calls and the 9.8 pivot judgment; this is the understanding behind them.

---

## How to use this guide

1. Read in the order below — each part depends on the one before it.
2. For each part there is a **"Derive with Claude"** prompt. Paste it into Claude chat (claude.ai)
   and work the derivation by hand alongside it. Do not accept a result you cannot re-derive.
3. After each part, answer the **"You should be able to…"** checklist out loud. If you can't, you
   haven't finished that part.
4. Keep a running list of the equations you've derived; those become the Methods section of the
   paper and the spec for the Phase 5 code.

**The through-line:**
Classical Hamiltonian mechanics → make $H$ a neural network (**HNN**) → an alternative using the
Lagrangian (**LNN**) → add integration + control (**Symplectic ODE-Net**) → add *dissipation* and
external forcing (**pHNN** — our core). Separately: how to turn raw sensor noise into clean
orientation (**Madgwick**), the body we're measuring (**biomechanics**), and the **gap** we fill.

---

## Reading order (dependency graph)

```
Part 1  Hamiltonian mechanics  (foundation — pure physics)
   └─> Part 2  HNN (Greydanus 2019)        the base idea
          ├─> Part 3  LNN (Cranmer 2020)   the coordinate-free sibling
          ├─> Part 4  Symplectic ODE-Net   integration + control
          └─> Part 5  pHNN (Desai 2021)    ★ THE CORE — dissipation + ports
Part 6  Madgwick (2010)        orientation from IMU  (independent track)
Part 7  Arm biomechanics       what we're measuring  (independent track)
Part 8  The gap (prior art)    why this is novel     (read last)
```

---

## Part 1 — Hamiltonian mechanics (the foundation)

You know this from physics; this fixes notation everything else uses.

A mechanical system is described by **generalized coordinates** $q$ (e.g. joint angles) and their
**conjugate momenta** $p$ (a *conjugate momentum* is the quantity $p_i = \partial L/\partial \dot q_i$;
for a particle it reduces to mass × velocity). The pair $(q, p)$ is a point in **phase space** — the
space of all positions-and-momenta the system can occupy.

The **Hamiltonian** $H(q, p)$ is, for the systems we care about, the total energy (kinetic +
potential). The dynamics are **Hamilton's equations**:

$$\dot q = \frac{\partial H}{\partial p}, \qquad \dot p = -\frac{\partial H}{\partial q}.$$

Two structural facts matter for everything below:

1. **Energy conservation** for autonomous (no external forcing) systems: $\dfrac{dH}{dt} = 0$ along
   trajectories. (Derive this — it falls straight out of the two equations.)
2. **Symplectic structure**: the flow preserves phase-space volume (Liouville's theorem).
   "*Symplectic*" = the geometry-preserving property of Hamiltonian flow; a *symplectic integrator*
   is a numerical scheme that respects it and so doesn't artificially gain/lose energy over long
   rollouts.

**Derive with Claude:** "Starting from Hamilton's equations, show $dH/dt = 0$ for a
time-independent Hamiltonian. Then derive Hamilton's equations from the Lagrangian via the Legendre
transform, defining conjugate momentum along the way."

**You should be able to…** write Hamilton's equations from memory; explain why $(q,p)$ beats
$(q,\dot q)$ for conservation; state what "symplectic" means in one sentence.

---

## Part 2 — Hamiltonian Neural Networks (Greydanus, Dzamba & Yosinski, 2019)

arXiv: **1906.01563**. The base idea everything else extends.

**The problem it solves.** A plain neural network trained to predict the time-derivative
$(\dot q, \dot p)$ directly will drift — it has no reason to conserve energy, so rollouts blow up or
decay. (A *rollout* = repeatedly feeding the model's prediction back in to simulate forward in time.)

**The idea.** Don't learn the derivatives. Learn the *energy*. Parameterize the Hamiltonian itself as
a neural network $H_\theta(q,p)$ — concretely an **MLP** (*multilayer perceptron*: the basic
fully-connected neural network, a stack of linear layers with nonlinearities). Then *compute* the
dynamics from it using Hamilton's equations, getting the needed partial derivatives by **autograd**
(*automatic differentiation*: the framework, e.g. PyTorch, differentiates the network's output with
respect to its inputs exactly, for free). Train so those computed derivatives match the data:

$$\mathcal{L}_{\text{HNN}} = \left\lVert \frac{\partial H_\theta}{\partial p} - \dot q \right\rVert^2
\;+\; \left\lVert -\frac{\partial H_\theta}{\partial q} - \dot p \right\rVert^2 .$$

Because the dynamics are *forced through Hamilton's equations*, the learned system conserves a
quantity (the learned $H_\theta$) by construction. That's the whole trick: **structure as an
inductive bias** — bake the physics into the architecture, not the loss alone.

**Why it matters to SkillSuit.** This is the sanity-check rung (task 5.3: train it on an ideal
pendulum, confirm energy is conserved in rollouts) before we trust the bigger models on real motion.

**Derive with Claude:** "Write the HNN loss and explain how $\partial H_\theta/\partial p$ is obtained
by autograd. Why does this conserve energy where a vanilla MLP predicting $(\dot q,\dot p)$ does not?"

**You should be able to…** explain "learn $H$, not $\dot x$"; write the loss; explain why the
inductive bias gives conservation; define MLP and autograd.

---

## Part 3 — Lagrangian Neural Networks (Cranmer et al., 2020)

arXiv: **2003.04630**. The coordinate-free sibling of the HNN.

**The limitation it removes.** HNNs need **canonical coordinates** — you must have $(q,p)$ with $p$
the true conjugate momentum. For real data you often have positions and velocities $(q, \dot q)$, and
computing $p$ requires already knowing the dynamics. LNNs sidestep this.

**The idea.** Learn the **Lagrangian** $L_\theta(q,\dot q)$ (kinetic minus potential energy) and use
the **Euler–Lagrange equation** $\frac{d}{dt}\frac{\partial L}{\partial \dot q} = \frac{\partial L}{\partial q}$.
Solving it for the acceleration:

$$\ddot q = \left(\nabla_{\dot q}\nabla_{\dot q}^{\top} L\right)^{-1}
\left[\nabla_q L - \left(\nabla_q \nabla_{\dot q}^{\top} L\right)\dot q\right].$$

The $\nabla_{\dot q}\nabla_{\dot q}^\top L$ term is a **Hessian** (matrix of second derivatives), got
again by autograd; it must be inverted, which is the practical cost.

**Why it matters to SkillSuit.** Human joint data is naturally $(q,\dot q)$ (angles + angular
velocity), so LNN framing is the more natural fit than raw HNN — context for why pHNN (Part 5) is
built to work without clean canonical momenta.

**Derive with Claude:** "Derive the LNN acceleration formula from the Euler–Lagrange equation by
expanding the total time derivative and solving for $\ddot q$."

**You should be able to…** state why HNN's canonical-coordinate requirement is a problem for real
data; explain what the Hessian inversion buys you.

---

## Part 4 — Symplectic ODE-Net (Zhong, Dey & Chakraborty, 2020)

arXiv: **1909.12077**. Adds two things HNN lacks: principled integration and **control**.

**Idea 1 — integrate symplectically.** HNN matches instantaneous derivatives. Symplectic ODE-Net
embeds the learned Hamiltonian field inside a **Neural ODE** (*a model that defines $\dot x = f_\theta(x)$
and integrates it with a numerical ODE solver, backpropagating through the solver*) using a
**symplectic integrator**, so long rollouts stay energy-stable.

**Idea 2 — control inputs.** Real systems are *actuated*. They model
$\dot x = f_\theta(x) + g_\theta(x)\,u$, where $u$ is the control (e.g. muscle torque) and
$g_\theta(x)$ maps it into the state. This "$+\,g(x)u$" input term is the bridge to the
**port** idea in Part 5.

**Why it matters to SkillSuit.** A moving arm is driven by muscle torques — not autonomous. We need a
framework that admits inputs. This paper is where that enters cleanly.

**Derive with Claude:** "Explain how a Neural ODE backpropagates through an ODE solver, and why a
symplectic integrator preserves energy over long rollouts. Where does the control term $g(x)u$ enter?"

**You should be able to…** define a Neural ODE; explain why symplectic integration matters for
rollouts; identify the control term and what it represents physically.

---

## Part 5 — Port-Hamiltonian Neural Networks (Desai et al., 2021) ★ THE CORE

arXiv: **2107.08024** (Phys. Rev. E 104, 034312). This is the model SkillSuit is built on. Spend the
most time here.

**Why HNN isn't enough for human motion.** Real movement **dissipates energy** — muscles, tendons,
and soft tissue lose energy as heat; the prep and settle phases of a motion are damped, not
conservative. A pure HNN conserves energy and *cannot represent* this. We need a model that has both
a conservative part and an explicit **dissipative** part.

**The Port-Hamiltonian formulation.** A port-Hamiltonian system is written:

$$\dot x = \big(J(x) - R(x)\big)\,\nabla H(x) \;+\; g(x)\,u,$$

term by term (this decomposition is the whole point — learn each piece):

- $\nabla H(x)$ — gradient of the learned energy (the conservative driving force).
- $J(x)$ — the **interconnection matrix**, *skew-symmetric* ($J = -J^{\top}$). Skew-symmetry means it
  only *routes* energy between coordinates without creating or destroying it — the conservative
  structure, generalizing the $\begin{psmallmatrix}0&1\\-1&0\end{psmallmatrix}$ of Hamilton's equations.
- $R(x)$ — the **dissipation matrix**, *symmetric positive semidefinite* ($R = R^{\top}\succeq 0$).
  This is energy *leaving* the system (damping/friction). PSD guarantees energy can only be lost,
  never spontaneously created. In code you enforce PSD by learning a factor $L_\theta$ and setting
  $R = L_\theta L_\theta^{\top}$.
- $g(x)\,u$ — the **port**: external input power entering the system (the control, as in Part 4).

The **energy balance** (derive it) is:

$$\dot H = -\,\nabla H^{\top} R\, \nabla H \;+\; \nabla H^{\top} g\, u,$$

i.e. energy change = (dissipation, always ≤ 0) + (input power through the port). The $J$ term
contributes nothing — that's what skew-symmetry guarantees.

**What the network learns.** $H_\theta(x)$ (an MLP for the energy) **and** $R_\theta(x)$ (an MLP
producing the dissipation matrix via the $L L^\top$ trick). Optionally $g_\theta$. Loss =
trajectory-prediction error + an energy-balance residual.

**The core result we're after (task 5.9).** Plot the *learned dissipation magnitude vs. time* across
motion classes. The physical prediction: dissipation should be ≈ 0 during the ballistic **active**
phase (energy conserved, like a thrown projectile) and concentrate in the **prep** and **settle**
phases (muscles braking/damping). If the learned $R$ does that on real data, the model has recovered
real physics — that's the scientific payoff and what makes the exported data *physically grounded*
rather than just trajectories. If it doesn't (diverges or gives nonsense $R$), that's the **9.8
pivot**: narrow the claim to sensor-fusion + data-format.

**Derive with Claude:** "Derive the port-Hamiltonian energy balance $\dot H = -\nabla H^\top R\nabla H
+ \nabla H^\top g u$ from $\dot x=(J-R)\nabla H + gu$, using $J=-J^\top$ and $R=R^\top\succeq0$.
Explain why $R=LL^\top$ enforces positive semidefiniteness."

**You should be able to…** write the PH equation and name every term; derive the energy balance;
explain *physically* why human motion needs $R$ and what the dissipation-vs-time plot should show;
state the pivot condition.

---

## Part 6 — Madgwick orientation filter (2010)

Madgwick, "An efficient orientation filter for inertial/magnetic sensor arrays" (U. Bristol report;
also IEEE ICORR 2011). This is the **sensor-fusion** track — independent of the HNN line.

**The raw problem.** An **IMU** (*Inertial Measurement Unit* — the chip on the sleeve) gives two
noisy signals per sample: a **gyroscope** (measures angular velocity $\omega$, in deg/s — accurate
short-term but its integral *drifts* over time as small errors accumulate) and an **accelerometer**
(measures acceleration including gravity, in $g$ — gives an absolute "down" direction, no drift, but
noisy and corrupted by motion). Neither alone gives stable orientation. **Sensor fusion** combines
them so the gyro handles fast motion and the accelerometer corrects long-term drift.

**Orientation as a quaternion.** A **quaternion** $q = [q_w, q_x, q_y, q_z]$ is a 4-number encoding of
a 3D rotation that avoids *gimbal lock* (the singularity Euler angles suffer) and composes cleanly.
Gyro integration in quaternion form:

$$\dot q_\omega = \tfrac{1}{2}\, q \otimes {}^S\!\omega,$$

where $\otimes$ is quaternion multiplication and ${}^S\omega$ is the gyro reading as a pure
quaternion $[0,\omega_x,\omega_y,\omega_z]$.

**The Madgwick correction (gradient descent).** Define an error: the gravity direction predicted by
the current orientation should match the measured accelerometer. Write it as an objective
$f(q,{}^S\!a)$ and take its gradient $\nabla f = J^{\top} f$ (here $J$ is the Jacobian of $f$). Step
the gyro estimate *down* that gradient with gain $\beta$:

$$q_t = q_{t-1} + \left(\dot q_\omega - \beta\,\frac{\nabla f}{\lVert \nabla f\rVert}\right)\Delta t .$$

$\beta$ trades the two error sources: large $\beta$ trusts the accelerometer (kills drift, adds
noise); small $\beta$ trusts the gyro (smooth, drifts). It's cheap enough to run on the ESP32 in
real time — the reason we pick it over a heavier Kalman filter for the embedded side.

**Why it matters to SkillSuit.** This is task 4.1 — we implement it from scratch and validate it
(still → gravity-aligned; 90° rotations recovered; <0.5° error on noiseless synthetic). Joint angles
come from the *relative* orientation of two adjacent IMUs, so getting each IMU's quaternion right is
the whole game.

**Derive with Claude:** "Derive the quaternion derivative $\dot q = \tfrac12 q\otimes\omega$ from the
definition of angular velocity. Then set up the Madgwick accelerometer objective $f$ and its Jacobian,
and explain the role of the gain $\beta$."

**You should be able to…** explain why gyro+accelerometer must be fused; define quaternion and why
not Euler angles; write the gyro-integration and the gradient-correction steps; explain $\beta$.

---

## Part 7 — Arm biomechanics (multi-motion)

No single canonical paper — this is background to extract from biomechanics texts/reviews.

**The kinematic chain.** The arm is a linked set of rigid segments with joints of differing
**DOF** (*degrees of freedom* — independent ways a joint can move): shoulder ≈ 3-DOF (ball joint),
elbow ≈ 1-DOF flexion + forearm pronation/supination, wrist ≈ 2–3 DOF. We instrument segments
(scapula S0, upper arm S2, forearm S4, wrist S5), **not** joints — a joint angle is recovered from
the relative orientation of the two segments flanking it (shoulder from S0↔S2, elbow from S2↔S4).

**Motion phases.** Most arm motions decompose into **prep → active → settle**. The active phase is
often near-ballistic (briefly conservative); prep and settle are damped (where dissipation lives —
tie this back to Part 5's $R$). We label phases per timestep; the label is a first-class data field.

**Why speed matters (D2).** Distal segments move fastest — the wrist can exceed several thousand
deg/s in fast gestures, beyond our ±2000°/s gyro range, where the sensor **saturates** (pins at its
max; reading is wrong). We document this envelope rather than buy it away. Understanding which
motions saturate is a biomechanics question.

**You should be able to…** sketch the arm as a DOF chain; explain "joint angle from relative segment
orientation"; map prep/active/settle onto where dissipation should appear; explain which motions
saturate and why.

---

## Part 8 — The gap (prior art) — read last

This is the claim the whole project rests on, so you must be able to state it precisely and know its
weaknesses. (I am running the prior-art scan for this — task 2.7 — and will hand you a draft; your
job is to pressure-test it.)

**How motor data for AI is captured today, and the limitation of each:**
- **Optical motion capture** (Vicon/OptiTrack): sub-mm accurate but lab-bound, marker-based,
  expensive; gives kinematics, no internal dynamics/forces.
- **Teleoperation demos** (e.g. ALOHA-style robot arms): captures action data, but on the *robot's*
  body, not the human expert's, and only for tasks you can teleoperate.
- **Video / vision** (pose estimation): cheap and scalable but no force, no reliable 3D dynamics,
  occlusion-prone.
- **Existing wearable IMU suits** (e.g. Xsens): give body kinematics, but export *orientation/pose*,
  not a learning-ready, physically-grounded (energy/dissipation-structured) representation.

**The gap we claim:** there is no *cheap wearable* that captures *general* human motor data and
exports it as **AI-training-ready data with a physics-grounded (HNN/pHNN) dynamical model** — i.e.
data that carries the energy + dissipation structure of the motion, not just trajectories.

**Your job:** confirm this gap actually holds (search hard for counterexamples) and find the *closest*
prior work, because reviewers and any advisor will. If a system already does this, we narrow the
claim. (Keep this scientific — the business framing stays out of the public repo, per the privacy
split.)

**You should be able to…** state the gap in one sentence; name the four capture methods and each
one's limitation; name the closest prior work and how we differ.

---

## Synthesis — how it assembles into SkillSuit

```
 sleeve IMUs ──(raw gyro+accel)──> Madgwick fusion (Part 6) ──> per-segment quaternions
        │                                                            │
        └──────────────> relative orientation ──> joint angles q, q̇ (Part 7)
                                                            │
                                          SkillData v1 (labeled, calibrated dataset)
                                                            │
                                  pHNN (Part 5): learn H(q,p) + dissipation R
                                                            │
                            physically-grounded model  ──>  AI-training-ready data
```

HNN (Part 2) is the proof-of-concept rung; LNN (Part 3) and Symplectic ODE-Net (Part 4) explain the
design choices (coordinate-freeness, control inputs) that pHNN inherits; pHNN (Part 5) is the
delivered science. Madgwick (Part 6) and biomechanics (Part 7) are how raw sensors become $q,\dot q$.
Part 8 is why anyone should care.

---

## Copy-paste prompts for Claude chat

- **Foundations:** "Act as a first-principles physics tutor. Derive Hamilton's equations from the
  Lagrangian via the Legendre transform, defining conjugate momentum, then prove energy conservation
  for a time-independent $H$. Use LaTeX, no skipped steps."
- **HNN → pHNN arc:** "Walk me from a vanilla neural ODE to an HNN to a Port-Hamiltonian NN. At each
  step state exactly what structural constraint is added and what physical phenomenon it captures.
  End with the pHNN energy-balance derivation."
- **Madgwick:** "Derive the Madgwick filter: quaternion kinematics from angular velocity, the
  accelerometer objective and its Jacobian, the gradient-descent correction, and the role of $\beta$."
- **Red-team the gap:** "Here is my novelty claim: [paste Part 8]. Find the strongest existing work
  that undermines it and tell me how to narrow the claim if needed."

## Glossary (quick reference)

- **IMU** — Inertial Measurement Unit; chip measuring its own acceleration + angular velocity.
- **Gyroscope / accelerometer** — angular-velocity sensor (drifts) / acceleration+gravity sensor (noisy, no drift).
- **Sensor fusion** — combining sensors so each covers the other's weakness.
- **Quaternion** — 4-number 3D-rotation encoding; avoids gimbal lock.
- **DOF** — degrees of freedom; independent directions a joint can move.
- **Phase space** — the space of all $(q,p)$ states.
- **Hamiltonian $H$ / Lagrangian $L$** — energy functions; $H(q,p)$ total energy, $L(q,\dot q)$ kinetic − potential.
- **Conjugate momentum** — $p_i = \partial L/\partial \dot q_i$.
- **Symplectic** — the volume/geometry-preserving structure of Hamiltonian flow.
- **MLP** — multilayer perceptron; basic fully-connected neural network.
- **Autograd** — automatic differentiation; exact gradients computed by the framework.
- **Neural ODE** — model defining $\dot x=f_\theta(x)$, integrated by an ODE solver with backprop through it.
- **Skew-symmetric** ($J=-J^\top$) / **positive semidefinite** ($R=R^\top\succeq0$) — the structural constraints making $J$ conserve and $R$ only dissipate energy.
- **Dissipation** — energy leaving the system (damping/friction).
- **Saturation** — sensor pinned at its max range; reading invalid.
- **Rollout** — simulating forward by feeding predictions back in.
