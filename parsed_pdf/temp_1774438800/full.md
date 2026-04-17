# Optimal Gradient-based Algorithms for Non-concave Bandit Optimization

Baihe Huang1∗, Kaixuan Huang2∗, Sham M. Kakade3,4∗, Jason D. Lee2∗ Qi Lei2∗, Runzhe Wang2∗, Jiaqi Yang5*

1Peking University 2Princeton University 3University of Washington 4Microsoft Research 5Tsinghua University

July 12, 2021

# Abstract

Bandit problems with linear or concave reward have been extensively studied, but relatively few works have studied bandits with non-concave reward. This work considers a large family of bandit problems where the unknown underlying reward function is non-concave, including the low-rank generalized linear bandit problems and two-layer neural network with polynomial activation bandit problem. For the low-rank generalized linear bandit problem, we provide a minimax-optimal algorithm in the dimension, refuting both conjectures in [LMT21, JWWN19]. Our algorithms are based on a unified zeroth-order optimization paradigm that applies in great generality and attains optimal rates in several structured polynomial settings (in the dimension). We further demonstrate the applicability of our algorithms in RL in the generative model setting, resulting in improved sample complexity over prior approaches. Finally, we show that the standard optimistic algorithms (e.g., UCB) are sub-optimal by dimension factors. In the neural net setting (with polynomial activation functions) with noiseless reward, we provide a bandit algorithm with sample complexity equal to the intrinsic algebraic dimension. Again, we show that optimistic approaches have worse sample complexity, polynomial in the extrinsic dimension (which could be exponentially worse in the polynomial degree).

# 1 Introduction

Bandits [LS20] are a class of online decision-making problems where an agent interacts with the environment, only receives a scalar reward, and aims to maximize the reward. In

many real-world applications, bandit and RL problems are characterized by large or continuous action space. To encode the reward information associated with the action, function approximation for the reward function is typically used, such as linear bandits [DHK08]. Stochastic linear bandits assume the mean reward to be the inner product between the unknown model parameter and the feature vector associated with the action. This setting has been extensively studied, and algorithms with optimal regret are known [DHK08, LS20, AYPS11, BCB12].

However, linear bandits suffer from limited representation power unless the feature dimension is prohibitively large. A comprehensive empirical study [RTS18] found that real-world problems required non-linear models and thus non-concave rewards to attain good performance on a testbed of bandit problems. To take a step beyond the linear setting, it becomes more challenging to design optimal algorithms. Unlike linear bandits, more sophisticated algorithms beyond optimism are necessary. For instance, a natural first step is to look at quadratic $[ \mathsf { K K S } ^ { + } 1 7 ]$ and higher-order polynomial [HZWS20] reward. In the context of phase retrieval, which is a special case for the quadratic bandit, people have derived algorithms that achieve minimax risks in the statistical learning setting [ACCD12, LM13, $\mathbf { C L M ^ { + } l 6 } ]$ . However, the straightforward adaptation of these algorithms results in sub-optimal dimension dependency.

In the bandit domain, existing analysis on the nonlinear setting includes eluder dimension [RVR13], subspace elimination [LMT21, JWWN19], etc. Their results also suffer from a larger dimension dependency than the best known lower bound in many settings. (See Table 1 and Section 3 for a detailed discussion of these results.) Therefore in this paper, we are interested in investigating the following question:

What is the optimal regret for non-concave bandit problems, including structured polynomials (low-rank etc.)? Can we design algorithms with optimal dimension dependency?

Contributions: In this paper, we answer the questions and close the gap (in problem dimension) for various non-linear bandit problems.

1. First, we design stochastic zeroth-order gradient-like1 ascent algorithms to attain minimax regret for a large class of structured polynomials. The class of structured polynomials contains bilinear and low-rank linear bandits and symmetric and asymmetric higher-order homogeneous polynomial bandits with action dimension $d$ . Though the reward is non-concave, we combine techniques from two bodies of work, nonconvex optimization and numerical linear algebra, to design robust gradient-based algorithms that converge to global maxima. Our algorithms are also computationally efficient, practical, and easily implementable.

In all cases, our algorithms attain the optimal dependence on dimension $d$ , which was not previously attainable using existing optimism techniques. As a byproduct, our algorithm refutes2 the conjecture from [JWWN19] on the bilinear bandit, and the conjecture from [LMT21] on low-rank linear bandit by giving an algorithm that attains the optimal dimension dependence.

2. We demonstrate that our techniques for non-concave bandits extend to RL in the generative setting, improving upon existing optimism techniques.

3. When the reward is a general polynomial without noise, we prove that solving polynomial equations achieves regret equal to the intrinsic algebraic dimension of the underlying polynomial class, which is often linear in $d$ for interesting cases. In general, this complexity cannot be further improved.

4. Furthermore, we provide a lower bound showing that all UCB algorithms have a sample complexity of $\Omega ( d ^ { p } )$ , where $p$ is the degree of the polynomial. The dimension of all homogeneous polynomials of degree $p$ in dimension $d$ is $d ^ { p }$ , showing that UCB is oblivious to the polynomial class and highly sub-optimal even in the noiseless setting.

# 1.1 Related Work

Linear Bandits. Linear bandit problems and their variants are studied in [DHK08, LS20, AYBM14, BCB12, AYPS11, DHK08, RT10, HLW20, HLSW21]. The matching upper bound and minimax lower bound achieves $O ( d \sqrt { T } )$ regret. Structured linear bandits, including sparse linear bandit [AYPS12] which developed an online-to-confidence-set technique. This technique yields the optimal $O ( { \sqrt { s d T } } )$ rate for sparse linear bandit. However [LMT21] employed the same technique for low-rank linear bandits giving an algorithm with regret $\hat { O ( } \sqrt { d ^ { 3 } \mathrm { p o l y } ( k ) T } )$ which we improve to $O ( \sqrt { d ^ { 2 } \mathrm { p o l y } ( k ) T } )$ , which meets the lower bound given in [LMT21].

Eluder Dimension. [RVR13] proposed the eluder dimension as a general complexity measure for nonlinear bandits. However, the eluder dimension is only known to give nontrivial bounds for linear function classes and monotone functions of linear function classes. For structured polynomial classes, the eluder dimension simply embeds into an ambient linear space of dimension $d ^ { p }$ , where $d$ is the dimension and $p$ is the degree. This parallels the linearization/NTK line in supervised learning [WLLM19, GMMM19, AZL19] which show that linearization also incurs a similarly large penalty of $d ^ { p }$ sample complexity, and more advanced algorithm design is need to circumvent linearization [BL20, $\mathrm { C B L } ^ { + } 2 0 $ , FLYZ20, WGL $^ + 1 9$ , ${ \mathrm { G C L } } ^ { + } 1 9$ , $\mathrm { N G L ^ { + } } 1 9$ , GLM18, MGW+20, HWLM20, WWL $^ { + } 2 0$ , DML21].

Neural Kernel Bandits. $[ \mathrm { V K M ^ { + } } 1 3 ]$ initiated the study of kernelized linear bandits, showing regret dependent on the information gain. [XWZG20] specialized this to the Neural Tangent Kernel (NTK) [LL18, $\mathrm { D L L ^ { + } } 1 9$ , JHG18, ${ \mathrm { G C L } } ^ { + } 1 9$ , BL20], where the algorithm utilizes gradient descent but remains close to initialization and thus remains a kernel class. Furthermore NTK methods require $d ^ { p }$ samples to express a degree $p$ polynomial in $d$ dimensions [GMMM21], similar to eluder dimension of polynomials, and so lack the inductive biases necessary for real-world applications of decision-making problems [RTS18].

Concave Bandits. There has been a rich line of work on concave bandits starting with [FKM04, Kle04]. $[ \mathrm { A F H ^ { + } } 1 1 ]$ attained the first $\sqrt { T }$ regret algorithm for concave bandits though with a large poly(d) dependence. In the adversarial setting, a line of work [HL16, BLE17, Lat20] have attained polynomial-time algorithms with $\sqrt { T }$ regret with increasingly improved dimension dependence. The sharp dimension dependence remains unknown.

Non-concave Bandits. To our knowledge, there is no general study of non-concave bandits, likely due to the difficulty of globally maximizing non-concave functions. A natural starting point of studying the non-concave setting are quadratic rewards such as the Rayleigh quotient, or namely bandit PCA [KN19, GHM15]. In the bilinear setting [JWWN19] and the low-rank linear setting [LMT21] can be made in the low rank (rank $k$ ) case to achieve $\tilde { O } ( \sqrt { d ^ { 3 } \mathrm { p o l y } ( k ) T } )$ regret. Other literature consider related but different settings that are not comparable to our results $[ \mathsf { K K S } ^ { + } 1 7$ , JSB16, GMZ16, LAAH19]. We note that the regret of all previous work is at least $O ( \sqrt { d ^ { 3 } T } )$ . This includes the subspace exploration and refinement algorithms from [JWWN19, LMT21], or from eluder dimension [RVR13]. Recently [HZWS20] considers online problem with a low-rank tensor associated with axis-aligned set of arms, which corresponds to finding the largest entry of the tensor. Finally, [KN19] study the bandit PCA problem in the adversarial setting, attaining regret of $O ( \sqrt { d ^ { 3 } T } )$ . We leave adapting our results to the advesarial setting as an open problem.

In the noiseless setting, there is some investigation in phase retrieval borrowing the tools from algebraic geometry (see e.g. [WX19]). In this paper, we will study the bandit problem with more general reward functions: neural nets with polynomial activation (structured polynomials). [KTB19] study similar structured polynomials, also using tools from algebraic geometry, but they only study the expressivity of those polynomials and do not consider the learning problems. [DYM21] study noiseless bandits, but only attain local optimality.

In concurrent work, [LH21] address the phase retrieval bandit problem which is equivalent to a symmetric rank 1 variant of the bilinear bandit of [JWWN19] and attain $\widetilde { O } ( \sqrt { d ^ { 2 } T } )$ regret. Our work in Section 3.1 specialized to the rank 1 case attains the same regret.

Matrix/Tensor Power Method. Our analysis stems from noisy power methods for matrix/tensor decomposition problems. Robust power method, subspace iteration, and tensor decomposition that tolerate noise first appeared in [HP14, $\mathrm { A G H ^ { + } 1 4 } ]$ . Follow-up work attained the optimal rate for both gap-dependence and gap-free settings for matrix decom-

position [MM15, AZL17]. An improvement on the problem dimension for tensor power method is established in [WA16]. [SBRL19] considers the convergence of tensor power method in the non-orthogonal case.

# 2 Preliminaries

# 2.1 Setup: Structured Polynomial Bandit

We study structured polynomial bandit problems where the reward function is from a class of structured polynomials (the precise settings and structures are discussed below). A player plays the bandit for $T$ rounds, and at each round $t \in [ T ]$ , the player chooses one action $\mathbf { } \mathbf { a } _ { t }$ from the feasible action set $A$ and receives the reward $r _ { t }$ afterward.

We consider both the stochastic case where $r _ { t } = f _ { \pmb { \theta } } ( \pmb { a } _ { t } ) + \eta _ { t }$ where $\eta _ { t }$ is the random noise, and the noiseless case $r _ { t } = f _ { \theta } ( a _ { t } )$ . Specifically the function $f _ { \theta }$ is unknown to the player, but lies in a known function class $\mathcal { F }$ . We use the notation $\mathrm { v e c } ( M )$ to denote the vectorization of a matrix or a tensor $M$ , and $\pmb { v } ^ { \otimes p }$ to denote the $p$ -order tensor product of a vector $\pmb { v }$ . For vectors we use $\| \cdot \| _ { 2 }$ or $\| \cdot \|$ to denote its $\ell _ { 2 }$ norm. For matrices $\| \cdot \| _ { 2 }$ or $\| \cdot \|$ stands for its spectral norm, and $\Vert \cdot \Vert _ { F }$ is Frobenius norm. For integer $n$ , $[ n ]$ denotes set $\{ 1 , 2 , \cdots n \}$ . For cleaner presentation, in the main paper we use $\widetilde O$ , $\widetilde { \Theta }$ or $\widetilde \Omega$ to hide universal constants, polynomial factors in $p$ and polylog factors in dimension $d$ , error $\varepsilon$ , eigengap $\Delta$ , total round number $T$ or failure rate $\delta$ .

We now present the outline with the settings considered in the paper:

The stochastic bandit eigenvector case, $\mathcal { F } _ { \mathrm { E V } }$ , considers action set $\pmb { \mathcal { A } } = \{ \pmb { a } \in \mathbb { R } ^ { d }$ : $\| \pmb { a } \| _ { 2 } \le 1 \}$ as shown in Section 3.1, and

$$
\mathcal {F} _ {\mathrm {E V}} = \left\{ \begin{array}{l} f _ {\boldsymbol {\theta}} (\boldsymbol {a}) = \boldsymbol {a} ^ {T} \boldsymbol {M} \boldsymbol {a}, \boldsymbol {M} = \sum_ {j = 1} ^ {k} \lambda_ {j} \boldsymbol {v} _ {j} \boldsymbol {v} _ {j} ^ {\top}, \text {f o r o r t h o n o r m a l} \boldsymbol {v} _ {j} \\ \boldsymbol {M} \in \mathbb {R} ^ {d \times d}, 1 \geq \lambda_ {1} \geq | \lambda_ {2} | \geq \dots \geq | \lambda_ {k} | \end{array} \right\}.
$$

The stochastic low-rank linear reward case, $\mathcal { F } _ { \mathrm { L R } }$ considers action sets on bounded matrices $\mathcal { A } = \{ \pmb { A } \in \mathbb { R } ^ { d \times d } : \lVert \pmb { A } \rVert _ { \mathrm { F } } \leq 1 \}$ in Section 3.2, and

$$
\mathcal {F} _ {\mathrm {L R}} = \left\{ \begin{array}{l l} f _ {\boldsymbol {\theta}} (\boldsymbol {A}) = \langle \boldsymbol {M}, \boldsymbol {A} \rangle = \operatorname {v e c} (\boldsymbol {M}) ^ {\top} \operatorname {v e c} (\boldsymbol {A}), \\ \boldsymbol {M} \in \mathbb {R} ^ {d \times d}, \operatorname {r a n k} (\boldsymbol {M}) = k, \boldsymbol {M} = \boldsymbol {M} ^ {\top}, \| \boldsymbol {M} \| _ {\mathrm {F}} \leq 1 \end{array} \right\}.
$$

We illustrate how to apply the established bandit oracles to attain a better sample complexity for RL problems with the simulator in Section 3.2.1.

The stochastic homogeneous polynomial reward case is presented in Section 3.3. For the symmetric case in Section 3.3.1, the action sets are $\mathcal { A } = \{ \pmb { a } \in \mathbb { R } ^ { d } : \| \pmb { a } \| _ { 2 } \leq 1 \}$ , and

$$
\mathcal {F} _ {\mathrm {S Y M}} = \left\{ \begin{array}{l} f _ {\boldsymbol {\theta}} (\boldsymbol {a}) = \sum_ {j = 1} ^ {k} \lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {p} \text {f o r o r t h o n o r m a l} \boldsymbol {v} _ {j}, \\ 1 \geq r ^ {*} = \lambda_ {1} > | \lambda_ {2} | \geq \dots \geq | \lambda_ {k} | \end{array} \right\};
$$

in the asymmetric case in Section 3.3.3, the action sets are

$$
\mathcal {A} = \left\{\boldsymbol {a} = \boldsymbol {a} (1) \otimes \boldsymbol {a} (2) \otimes \dots \otimes \boldsymbol {a} (p) \in \mathbb {R} ^ {d ^ {p}}: \forall q \in [ p ], \| \boldsymbol {a} (q) \| _ {2} \leq 1 \right\},
$$

$$
\mathcal {F} _ {\mathrm {A S Y M}} = \left\{ \begin{array}{l} f _ {\boldsymbol {\theta}} (\boldsymbol {a}) = \sum_ {j = 1} ^ {k} \lambda_ {j} \prod_ {q = 1} ^ {p} (\boldsymbol {v} _ {j} (q) ^ {\top} \boldsymbol {a} (q)) \text {f o r o r t h o n o r m a l} \boldsymbol {v} _ {j} (q) \text {f o r e a c h} q, \\ 1 \geq r ^ {*} = | \lambda_ {1} | \geq | \lambda_ {2} | \geq \dots \geq | \lambda_ {k} | \end{array} \right\}.
$$

In the above settings, there is stochastic noise on the observed rewards. We also consider noiseless settings as below:

The noiseless polynomial reward case is presented in Section 3.4. The action sets $A$ are subsets of $\mathbb { R } ^ { d }$ , and

$$
\mathcal {F} _ {\mathrm {P}} = \left\{f _ {\boldsymbol {\theta}} (\boldsymbol {a}) = \langle \boldsymbol {\theta}, \widetilde {\boldsymbol {a}} ^ {\otimes p} \rangle : \boldsymbol {\theta} \in \mathcal {V}, \widetilde {\boldsymbol {a}} = [ 1, \boldsymbol {a} ^ {\top} ] ^ {\top}, \mathcal {V} \subseteq (\mathbb {R} ^ {d + 1}) ^ {\otimes p} \text {i s a n a l g e b r a i c v a r i e t y} \right\}.
$$

Additionally, $\mathcal { F }$ needs to be admissible (Definition 3.25). This class includes two-layer neural networks with polynomial activations (i.e. structured polynomials). We study the fundamental limits of all UCB algorithms in Section 3.4.2 as they are $\Omega ( d ^ { p - 1 } )$ worse than our algorithm presented in Section 3.4.1.

In the above settings, we are concerned with the cumulative regret $\Re ( T )$ for $T$ rounds. Let $f _ { \pmb \theta } ^ { * } = \operatorname* { s u p } _ { \pmb { a } \in \mathcal { A } } f _ { \pmb \theta } ( \pmb a ) .$ ,

$$
\Re_ {\boldsymbol {\theta}} (T) := \sum_ {t = 1} ^ {T} \left(f _ {\boldsymbol {\theta}} ^ {*} - f _ {\boldsymbol {\theta}} \left(\boldsymbol {a} _ {t}\right)\right)
$$

And since the parameters can be chosen adversarially, we are bounding $\Re ( T ) = \operatorname* { s u p } _ { \theta } \Re _ { \theta } ( T )$ in this paper.

In all the stochastic settings above, we make the standard assumption on stochasticity that $\eta _ { t }$ is conditionally zero-mean 1-sub-Gaussian random variable regarding the randomness before $t$ .

# 2.2 Warm-up: Adapting Existing Algorithms

In all of the above settings, the function class can be viewed as a generalized linear function in kernel spaces. Namely, there is fixed feature maps $\psi , \phi$ so that $f _ { \pmb \theta } ( \mathbf { \boldsymbol { a } } ) = \psi ( \pmb \theta ) ^ { T } \phi ( \mathbf { \boldsymbol { a } } )$ . Thus it is straightforward to adapt linear bandit algorithms like the renowned LinUCB [LCLS10] to our settings. Furthermore, another baseline is given by the eluder dimension argument [RVR13][WSY20] which gives explicit upper bounds for general function classes. We present the best upper bound by adapting these methods as a baseline in Table 1, together with our newly-derived lower bound and upper bound in this paper.

The best-known statistical rates are based on the following result.

Theorem 2.1 (Proposition 4 in [RVR13]). With $\alpha = O ( T ^ { - 2 } )$ appropriately small, given the $\alpha$ -covering-number $N$ (under $\| \cdot \| _ { \infty } )$ and the $\alpha$ -eluder-dimension $d _ { E }$ of the function class F, Eluder UCB (Algorithm 4) achieves regret $\widetilde { \cal O } ( \sqrt { d _ { E } T \log N } )$ .

In the first row of Table 1, we further elaborate on the best results obtained from Theorem 2.1 in individual settings. More details can be found in Appendix A.

<table><tr><td colspan="3">Regret</td><td>\( \mathcal{F}_{\text{SYM}} \)</td><td>\( \mathcal{F}_{\text{ASYM}} \)</td><td>\( \mathcal{F}_{\text{EV}} \)</td><td>\( \mathcal{F}_{\text{LR}} \)</td></tr><tr><td colspan="3">LinUCB/eluder</td><td>\( \sqrt{dp+1kT} \)</td><td>\( \sqrt{dp+1kT} \)</td><td>\( \sqrt{d^3kT} \)</td><td>\( \sqrt{d^3kT} \)</td></tr><tr><td rowspan="3">Our Results</td><td rowspan="2">NPM</td><td>Gap</td><td>N/A</td><td>N/A</td><td>\( \sqrt{\kappa^3d^2T} \)</td><td>\( \sqrt{d^2k\lambda_k^{-2}T} \)</td></tr><tr><td>Gap-free</td><td>\( \sqrt{dpkT} \)</td><td>\( \sqrt{k^pdpT} \)</td><td>\( k^{4/3}(dT)^{2/3} \)</td><td>\( (dkT)^{2/3} \)</td></tr><tr><td colspan="2">Lower Bound</td><td>\( \sqrt{dpT} \)</td><td>\( \sqrt{dpT} \)</td><td>\( \sqrt{d^2T} \)</td><td>\( \sqrt{d^2k^2T} (*) \)</td></tr></table>

Table 1: Baselines and our main results (for stochastic settings). Eigengap $\Delta = \lambda _ { 1 } - | \lambda _ { 2 } |$ , condition number $\kappa = \lambda _ { 1 } / \Delta$ . The result with $^ *$ is from [LMT21]. For simplicity, in this table, we treat $p$ and $r ^ { * } : = \operatorname* { m a x } _ { \mathbf { \alpha } } _ { \mathbf { \alpha } } { } _ { a \in \mathcal { A } } f _ { \theta } ( \mathbf { \alpha } a )$ as constant and ignore all poly log factors.

# 3 Main results

We now present our main results. We consider four different stochastic settings (see Table 1) and one noiseless setting with structured polynomials.

In the cases of stochastic reward, all our algorithms can be unified as gradient-based optimization. At each stage with a candidate action $^ { a }$ , we define the estimator $G _ { n } ( { \pmb a } ) : =$ $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( f _ { \pmb { \theta } } \big ( \big ( 1 - \zeta \big ) \pmb { a } + \zeta \pmb { z } _ { i } \big ) + \eta _ { i } \big ) z _ { i } } \end{array}$ , with $\pmb { z } _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \pmb { I } _ { d } )$ and proper step-size $\zeta$ [FKM04]. Therefore $\mathbb { E } _ { z } [ G ( a ) ] = \sigma ^ { 2 } \zeta \nabla f _ { \theta } ( ( 1 - \zeta ) a ) + O ( \zeta ^ { 2 } ) = \zeta ( 1 - \zeta ) ^ { p - 1 } \sigma ^ { 2 } \nabla f _ { \theta } ( a ) + O ( \zeta ^ { 2 } )$ for $p$ -th order homogeneous polynomials. Therefore with enough samples, we are able to implement noisy gradient ascent with bias.

In the noiseless setting, our algorithm solves for the parameter $\pmb \theta$ with randomly sampled actions $\left\{ \pmb { a } _ { t } \right\}$ and the noiseless reward $\{ f _ { \pmb { \theta } } ( \pmb { a } _ { t } ) \}$ , and then determines the optimal action by computing arg maxa $f _ { \pmb \theta } ( \pmb a )$ .

# 3.1 Stochastic Eigenvalue Reward $( \mathcal { F } _ { \bf E V } )$

Now consider bandits with stochastic reward $\boldsymbol { r } ( \mathbf { a } ) = \mathbf { a } ^ { \top } \mathbf  \} \mathbf { M } \mathbf { a } + \boldsymbol { \eta }$ with action set ${ \mathcal { A } } =$ $\{ a | \| a \| _ { 2 } \leq 1 \}$ , where $\begin{array} { r } { \pmb { M } = \sum _ { i = 1 } ^ { r } \lambda _ { i } \pmb { v } _ { i } \pmb { v } _ { i } ^ { \top } } \end{array}$ is symmetric and satisfies $r ^ { * } = \lambda _ { 1 } > | \lambda _ { 2 } | \geq$ $| \lambda _ { 3 } | \cdot \cdot \cdot \geq | \lambda _ { r } | , \eta \sim { \mathcal { N } } ( 0 , 1 )$ .

Denote by $\ b { a } ^ { * }$ the optimal action $( \pm v _ { 1 } )$ , the leading eigenvector of $M$ , $( { \pmb a } ^ { * } ) ^ { \top } { \pmb M } { \pmb a } ^ { * } = \lambda _ { 1 }$ Let $\Delta = \lambda _ { 1 } - | \lambda _ { 2 } | > 0$ be the eigengap and $\kappa : = \lambda _ { 1 } / \Delta$ be the condition number.

Remark 3.1 (Negative leading eigenvalue). For a symmetric matrix $M$ , we will conduct noisy power method to recover its leading eigenvector, and therefore we require its leading eigenvalue $\lambda _ { 1 }$ to be positive. It is straightforward to extend to the setting where the nonzero eigenvalues satisfy: $r ^ { * } \equiv \lambda _ { 1 } > \lambda _ { 2 } \geq \lambda _ { l } > 0 > \lambda _ { l + 1 } \cdot \cdot \cdot \geq \lambda _ { k }$ , and $| \lambda _ { k } | > \lambda _ { 1 }$ . For this problem, we can shift $M$ to get $M + | \lambda _ { k } | I$ and the eigen-spectrum now becomes $\lambda _ { 1 } + \vert \lambda _ { k } \vert , \lambda _ { 2 } + \vert \lambda _ { k } \vert , \cdot \cdot \cdot 0$ ; therefore, we can still recover the optimal action with dependence on the new condition number $( \lambda _ { 1 } + | \lambda _ { k } | ) / ( \lambda _ { 1 } - \lambda _ { 2 } )$ .

Remark 3.2 (Asymmetric matrix). Our algorithm naturally extends to the asymmetric setting: $f ( \pmb { a } _ { 1 } , \pmb { a } _ { 2 } ) = \pmb { a } _ { 1 } ^ { \top } \widetilde { \pmb { M } } \pmb { a } _ { 2 }$ , where $\widetilde { M } = U \Sigma V ^ { \top }$ . This setting can be reduced to the

symmetric case via defining

$$
\boldsymbol {M} = \left[ \begin{array}{c c} 0 & \widetilde {\boldsymbol {M}} ^ {\top} \\ \widetilde {\boldsymbol {M}} & 0 \end{array} \right] = \frac {1}{2} \left[ \begin{array}{c c} \boldsymbol {V} & \boldsymbol {V} \\ \boldsymbol {U} & - \boldsymbol {U} \end{array} \right] \left[ \begin{array}{c c} \boldsymbol {\Sigma} & 0 \\ 0 & - \boldsymbol {\Sigma} \end{array} \right] \left[ \begin{array}{c c} \boldsymbol {V} & \boldsymbol {V} \\ \boldsymbol {U} & - \boldsymbol {U} \end{array} \right] ^ {\top},
$$

which is a symmetric matrix, and its eigenvalues are $\pm \sigma _ { i } ( \widetilde { M } )$ , the singular values of $\widetilde { M }$ . Therefore our analysis on symmetric matrices also applies to the asymmetric setting and will equivalently depend on the gap between the top singular values of $\widetilde { M }$ . A formal asymmetric to symmetric conversion algorithm is presented in Algorithm $I$ in [GHM15].

Algorithm. We note that by conducting zeroth-order gradient estimate $\textstyle 1 / n \sum _ { i = 1 } ^ { n } ( f ( { \pmb a } / 2 +$ $z _ { i } / 2 ) + \eta _ { i } ) z _ { i }$ with step-size 1/2 [FKM04] and sample size $n$ , we get an estimate for $\begin{array} { r } { \mathbb { E } _ { \eta , z } [ ( f ( { \pmb a } / 2 + z / 2 ) + \eta ) z ] = \frac { \sigma ^ { 2 } } { 2 } M { \pmb a } } \end{array}$ when $z \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . Therefore we are able to use noisy power method to recover the top eigenvector. We present a more general result with Tensor power method in Algorithm 1 and attain a gap-dependent risk bound:

# Algorithm 1 Noisy power method for bandit eigenvalue problem.

1: Input: Quadratic function $f : { \mathcal { A } }  \mathbb { R }$ with noisy reward, failure probability $\delta$ , error ε.   
2: Initialization: Initial action $\pmb { a } _ { 0 } \in \mathbb { R } ^ { d }$ randomly sampled on the unit sphere $\mathbb { S } ^ { d - 1 }$ . We set $\alpha ~ = ~ | \lambda _ { 2 } / \lambda _ { 1 } |$ , sample size per iteration $n = C _ { n } d ^ { 2 } \log ( d / \delta ) ( \lambda _ { 1 } \alpha ) ^ { - 2 } \varepsilon ^ { - 2 }$ , sample variance $m = C _ { m } d \log ( n / \delta )$ , total iteration $L = \lfloor C _ { L } \log ( d / \varepsilon ) \rfloor + 1$ .   
3: for Iteration $l$ from 1 to $L$ do   
4: Sample $z _ { i } \sim \mathcal { N } ( 0 , 1 / m I _ { d } ) , i = 1 , 2 , \cdot \cdot \cdot n$ . (Re-sample the whole batch if exists $z _ { i }$ with norm greater than 1.)   
5: Noisy power method:   
6: Take actions $\begin{array} { r } { \overline { { \widetilde { \mathbf { a } } _ { i } } } = \frac { \mathbf { a } _ { l - 1 } + \pmb { z } _ { i } } { 2 } } \end{array}$ and observe $r _ { i } = f ( \pmb { a } _ { i } ) + \eta _ { i } , \forall i \in [ n ]$   
7: e 2Update normalized action $\begin{array} { r } { \pmb { a } _ { l }  \frac { m } { n } \sum _ { i = 1 } ^ { n } r _ { i } \pmb { z } _ { i } } \end{array}$ , and normalize $\pmb { a } _ { l }  \pmb { a } _ { l } / \| \pmb { a } _ { l } \| _ { 2 }$   
8: Output: ${ \pmb a } _ { L }$

Theorem 3.3 (Regret bound for noisy power method (NPM)). In Algorithm 1, we set $\varepsilon \in$ $( 0 , 1 / 2 )$ , $\delta = 0 . 1 / ( L _ { 0 } S )$ and let $C _ { L } , C _ { S } , C _ { n } , C _ { m }$ be large enough universal constants. Then with high probability 0.9 we have: the output ${ \pmb a } _ { L }$ satisfies tan $\theta ( { \pmb a } ^ { * } , { \pmb a } _ { L } ) \le \varepsilon$ and yields $r ^ { * } \varepsilon ^ { 2 }$ -optimal reward; and the total number of actions we take is $\begin{array} { r } { \widetilde { \cal O } ( \frac { \kappa d ^ { 2 } } { \Delta ^ { 2 } \varepsilon ^ { 2 } } ) } \end{array}$ . By explore-thencommit (ETC) the cumulative regret is at most $\widetilde { O } ( \sqrt { \kappa ^ { 3 } d ^ { 2 } T } )$ .

All proofs in this subsection are in Appendix B.1.

Remark 3.4 (Intuition of [JWWN19] and how to overcome the conjectured lower bound via the design of adaptive algorithms). Let us consider the rank 1 case of $r ( \mathbf { a } ) = ( \mathbf { a } ^ { T } \pmb { \theta } ^ { * } ) ^ { 2 } +$ η. A random action $\pmb { a } \sim U n i f ( \mathbb { S } ^ { d - 1 } )$ has $f ( \pmb { a } ) \asymp 1 / d ^ { 2 }$ , and the noise has standard deviation

$O ( 1 )$ . Thus the signal-to-noise-ratio is $O ( 1 / d ^ { 2 } )$ and the optimal action $\theta ^ { * }$ requires d bits to encode. If we were to play non-adaptively, this would require $O ( d ^ { 3 } )$ queries and result in regret $\sqrt { d ^ { 3 } T }$ which matches the result of [JWWN19].

To go beyond this, we must design algorithms that are adaptive, meaning the information in $f ( \pmb { a } ) + \eta$ is strictly larger than $\textstyle { \frac { 1 } { d ^ { 2 } } }$ . As an illustration of why this is possible, consider batching the time-steps into $d$ stages so that each stage decode 1 bit of $\pmb { \theta } ^ { * }$ . At the first stage, random exploration $\pmb { a } \sim U n i f ( \mathbb { S } ^ { d - 1 } )$ gives signal-to-noise-ratio $O ( 1 / d ^ { 2 } )$ . Suppose $k$ bits of θ are decoded at $k$ -th stage by $\widehat { \pmb { \theta } }$ , adaptive algorithms can boost the signal-to-noise-ratio to $O ( k / d ^ { 2 } )$ by using $\widehat { \pmb { \theta } }$ as bootstrap (e.g. exploring with $\widehat { \pm } a$ where a is random exploration in the unexplored subspace). In this way adaptive algorithms only need $d ^ { 2 } / k$ queries in $( k + 1 )$ -th stage and so the total number of queries sums up to $\textstyle \sum _ { k = 1 } ^ { d } d ^ { 2 } / k \approx \dot { d ^ { 2 } } \log d$ .

Gradient descent and power method offer a computationally efficient and seamless way to implement the above intuition. For every iterate action $^ { a }$ , we estimate M a from noisy observations and take it as our next action $\pmb { a } ^ { + }$ . With $d ^ { 2 } / ( \Delta ^ { 2 } \varepsilon ^ { 2 } )$ samples, noisy power method enjoys linear progress tan $\theta ( { \pmb a } ^ { + } , { \pmb a } ^ { * } ) \le \operatorname* { m a x } \{ c \tan ( { \pmb a } , { \pmb a } ^ { * } ) , \varepsilon \}$ , where $c < 1$ is a constant that depends on $\lambda _ { 1 } , \lambda _ { 2 }$ , and $\varepsilon$ . Therefore even though every step costs $d ^ { 2 }$ samples, overall we only need logarithmic $( i n d , \varepsilon , \lambda _ { 1 } , \lambda _ { 2 } )$ iterations to find an $\varepsilon$ -optimal action.

Remark 3.5 (Connection to phase retrieval and eluder dimension). For rank-1 case $M =$ $\pmb { x } \pmb { x } ^ { \top }$ , the bilinear bandits can be viewed as phase retrieval, where one observes $y _ { r } =$ $( { \pmb a } _ { r } ^ { \top } { \pmb x } ) ^ { 2 } = { \pmb a } _ { r } ^ { \top } M { \pmb a } _ { r }$ plus some noise $\eta _ { r } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . The optimal (among non-adaptive algorithms) sample complexity to recover $_ { x }$ is $\sigma ^ { 2 } d / \epsilon ^ { 2 }$ [CLS15, $C L M ^ { + } l 6 ]$ where they play $^ { a }$ from random Gaussian $\mathcal { N } ( 0 , \pmb { I } )$ . However, in bandit, we need to set the variance of a to at most $1 / d$ to ensure $\| \mathbfcal { a } \| \leq 1$ . Our problem is equivalent to observing $y _ { r } / d$ where their $\pmb { a } / \sqrt { d } \sim \mathcal { N } ( 0 , 1 / d \pmb { I } )$ and noise level $\eta _ { r } / d \sim \mathcal { N } ( 0 , 1 )$ , i.e., $\sigma ^ { 2 } = d ^ { 2 }$ . Therefore one gets $d ^ { 3 } / \epsilon ^ { 2 }$ even for the rank-1 problems. On the other hand, for all rank-1 M the condition number $\kappa ~ = ~ 1$ ; and thus our results match the lower bound (see Section 3.3.4) up to logarithmic factors, and also have fundamental improvements for phase retrieval problems by leveraging adaptivity.

For UCB algorithms based on eluder dimension, the regret upper bound is $O ( \sqrt { d _ { E } \log ( N ) T } ) =$ $O ( \sqrt { d ^ { 3 } T } )$ as presented in Theorem 2.1, where the dependence on $d$ is consistent with [JWWN19, LMT21] and is non-optimal.

The previous result depends on the eigen-gap. When the matrix is ill-conditioned, i.e., $\lambda _ { 1 }$ is very close to $\lambda _ { 2 }$ , we can obtain gap-free versions with a modification: The first idea stems from finding higher reward instead of recovering the optimal action. Therefore when $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are very close (gap being smaller than desired accuracy $\varepsilon ^ { \cdot }$ ), it is acceptable to find any direction in the span of $( v _ { 1 } , v _ { 2 } )$ . More formally, we care about the convergence speed of identifying any action in the space spanned by any top eigenvectors (whose associated eigenvalues are higher than $\lambda _ { 1 } - \varepsilon )$ . Therefore the convergence speed will depend on $\varepsilon$ instead of $\lambda _ { 1 } - \lambda _ { 2 }$ .

Table 2: Summary of results for quadratic reward. All red expressions are our results. LB, UB, NPM stands for lower bound, upper bound, and noisy power method respectively. $\Delta = \lambda _ { 1 } - | \lambda _ { 2 } |$ is the eigengap, and $\kappa = \lambda _ { 1 } / \Delta$ , $\widetilde { \kappa } = \lambda _ { 1 } / | \lambda _ { k } |$ are the condition numbers. The result with $^ *$ is not computationally tractable. For low-rank setting in this table, we treat $\| \boldsymbol { M } \| _ { F }$ as a constant for simplicity and leave its dependence in the theorems. Our upper bounds match the lower bound in terms of dimension and substantially improve over existing algorithms that are computationally efficient.   

<table><tr><td>FEV</td><td>LB (k=1)</td><td>[JWWN19]</td><td>NPM</td><td>Gap-free NPM</td><td>Subspace Iteration</td></tr><tr><td>Regret</td><td>√d2T</td><td>√d3kλk-2T</td><td>√κ3d2T</td><td>d2/5T4/5</td><td>min(k4/3(dT)2/3, k1/3(κdT)2/3)</td></tr><tr><td>FLR</td><td>LB ([LMT21])</td><td colspan="2">UB ([LMT21])</td><td colspan="2">Subspace Iteration</td></tr><tr><td>Regret</td><td>Ω(√d2k2T)</td><td colspan="2">√d3kT* or √d3kλk-2T</td><td colspan="2">min(√d2kλk-2T, (dkT)2/3)</td></tr></table>

Corollary 3.6 (Gap-free regret bound). For positive semi-definite matrix $M$ , by setting $\alpha = 1 - \varepsilon ^ { 2 } / 2$ in Algorithm 1 and performing ETC afterwards, one can obtain cumulative regret of $\widetilde { O } ( \lambda _ { 1 } ^ { 3 / 5 } d ^ { 2 / 5 } T ^ { 4 / 5 } )$ .

Again, the PSD assumption is not essential. For general symmetric matrices with $\lambda _ { 1 } \geq$ $\lambda _ { 2 } \geq \cdots > 0 > \cdots \lambda _ { k }$ . We can still conduct shifted power method on $M - \lambda _ { k } I$ , yielding a cumulative regret of (λ1 + |λk|)3/5d2/5T 4/5. $( \lambda _ { 1 } + | \lambda _ { k } | ) ^ { 3 / 5 } d ^ { 2 / 5 } T ^ { 4 / 5 }$

Another novel gap-free algorithm requires to identify any top eigenspace $V _ { 1 : l } , l \in [ k ]$ : $V _ { 1 : l }$ is the column span of $\{ \pmb { v } _ { 1 } , \pmb { v } _ { 2 } , \cdots \pmb { v } _ { l } \}$ . Notice that in traditional subspace iteration, the convergence rate of recovering $V _ { l }$ depends on the eigengap $\Delta _ { l } : = | \lambda _ { l } | - | \lambda _ { l + 1 } |$ . Meanwhile, since $\begin{array} { r } { \sum _ { l = 1 } ^ { k } \Delta _ { l } = \lambda _ { 1 } } \end{array}$ , at least one eigengap is larger or equal to $\lambda _ { 1 } / k$ . Suppose $\Delta _ { l ^ { * } } \geq \lambda _ { 1 } / k$ ; we can, therefore, set $\alpha = 1 { - } 1 / k$ and recover the top $l ^ { * }$ subspace up to $\lambda _ { 1 } \epsilon$ error, which will give an $\epsilon$ -optimal reward in the end. We don’t know $l ^ { * }$ beforehand and will try recovering the top subspace $V _ { 1 } , V _ { 2 } , \cdots V _ { k }$ respectively, which will only lose a $k$ factor. With the existence of $l ^ { * }$ , at least one trial (on recovering $V _ { l ^ { * } }$ ) will be successful with the parameters of our choice, and we simply output the best action among all trials.

Theorem 3.7 (Informal statement: (gap-free) subspace iteration). By running subspace iteration (Algorithm 5) with proper choices of parameters, we attain a cumulative regret of $\widetilde { \cal O } ( \lambda _ { 1 } ^ { 1 / 3 } k ^ { 4 / 3 } ( d T ) ^ { 2 / 3 } )$ . Algorithm 5) with another set of parameters can also recover the whole eigenspace, and achieve cumulative regret of $\widetilde { O } ( ( \lambda _ { 1 } k ) ^ { 1 / 3 } ( \widetilde { \kappa } d T ) ^ { 2 / 3 } )$ , where $\widetilde { \kappa } = \lambda _ { 1 } / | \lambda _ { k } |$ .

# 3.2 Stochastic Low-rank Linear Bandits $( \mathcal { F } _ { \mathbf { L R } } )$

In the low-rank linear bandit, the reward function is $f ( A ) = \langle A , M \rangle$ , with noisy observations $r _ { t } = f ( A ) + \eta _ { t }$ , and the action space is $\{ A \in \mathbb { R } ^ { d \times d } : \| A \| _ { F } \leq 1 \}$ . Without loss of generality we assume $k$ , the rank of $M$ satisfies $\begin{array} { r } { k \leq \frac { d } { 2 } } \end{array}$ , since when $k$ is of the same order

as $d$ , the known upper and lower bound are both $\sqrt { d ^ { 2 } k ^ { 2 } T } = \Theta ( \sqrt { d ^ { 4 } T } )$ [LMT21] and there is no room for improvement. We write $r ^ { * } = \lVert \boldsymbol { M } \rVert _ { F } \leq 1$ , therefore the optimal action is $A ^ { * } = M / r ^ { * }$ . In this section, we write $\pmb { X } ( s )$ to be the $s$ -th column of any matrix $\pmb { X }$ .

As presented in Algorithm 2, we conduct noisy subspace iteration to estimate the right eigenspace of $M$ . Subspace iteration requires calculating $M X _ { t }$ at every step. This can be done by considering a change of variable of $g ( X ) : = f ( X X ^ { \top } ) = \langle X X ^ { \top } , M \rangle$ whose gradient3 is $\nabla g ( { \pmb X } ) = 2 M { \pmb X }$ . The zeroth-order gradient estimator can then be employed to stochastically estimate $_ { M X }$ . We instantiate the analysis with symmetric $M$ while extending to asymmetric setting is straightforward since the problem can be reduced to symmetric setting (suggested in Remark 3.2).

With stochastic observations and randomly sampled actions, we achieve the next iterate $\mathbf { { Y } } _ { l }$ that satisfies $M X _ { l - 1 } \equiv \mathbb { E } [ Y _ { l } ]$ in Algorithm 2. Let $M = V \Sigma V ^ { \top }$ . With proper concentration bounds presented in the appendix, we can apply the analysis of noisy power method [HP14] and get:

Algorithm 2 Subspace Iteration Exploration for Low-rank Linear Reward.   
1: Input: Quadratic function $f: \mathcal{A} \to \mathbb{R}$ with noisy reward, failure probability $\delta$ , error $\varepsilon$ .  
2: Initialization: Set $k' = 2k$ . Initial candidate matrix $X_0 \in \mathbb{R}^{d \times k'}$ , $X_0(j) \in \mathbb{R}^d$ , $j = 1, 2, \dots, k'$ is the $j$ -th column of $X_0$ and are i.i.d sampled on the unit sphere $\mathbb{S}^{d-1}$ uniformly. Sample variance $m, \#$ sample per iteration $n$ , total iteration $L$ .  
3: for Iteration $l$ from 1 to $L$ do  
4: Sample $z_i \sim \mathcal{N}(0, 1/mI_d)$ , $i = 1, 2, \dots, n$ .  
5: for $s$ from 1 to $k'$ do  
6: Noisy subspace iteration:  
7: Calculate tentative rank-1 actions $\widetilde{A}_i = X_{l-1}(s)\boldsymbol{z}_i^\top$ .  
8: Conduct estimation $Y_l(s) \gets m/n\sum_{i=1}^n(\langle M, \widetilde{A}_i\rangle + \eta_{i,s})\boldsymbol{z}_i$ . ( $Y_l \in \mathbb{R}^{d \times k'}$ )  
9: Let $Y_l = X_lR_l$ be a QR-factorization of $Y_l$ 10: Update target action $A_l \gets Y_lX_l^\top$ .  
11: Output: $\widehat{A} = A_L / \|A_L\|_F$

Theorem 3.8 (Informal statement; sample complexity for low-rank linear reward). With Algorithm 2, $X _ { L }$ satisfies $\| ( I - { \mathbf { } } { \mathbf { } } _ { L } { \mathbf { } } _ { L } ^ { \top } ) V \| \leq \varepsilon / 4 \}$ , and output $\widehat { A }$ satisfies $\lVert \hat { A } - \dot { A } ^ { * } \rVert _ { F } \leq$ $\varepsilon \lVert M \rVert _ { F }$ with sample size $\widetilde { O } ( d ^ { 2 } k \lambda _ { k } ^ { - 2 } \varepsilon ^ { - 2 } )$ .

We defer the proofs for low-rank linear reward in Appendix B.2.

Corollary 3.9 (Regret bound for low-rank linear reward). We first call Algorithm 2 with $\varepsilon ^ { 4 } = \widetilde { \Theta } ( \dot { d } ^ { 2 } k \lambda _ { k } ^ { - 2 } { T } ^ { - 1 } )$ to obtain $\widehat { A }$ ; we then play $\widehat { A }$ for the remaining steps. The cumulative

regret satisfies

$$
\Re (T) \leq \widetilde {O} (\sqrt {d ^ {2} k (r ^ {*}) ^ {2} \lambda_ {k} ^ {- 2} T}),
$$

with high probability 0.9.

To be more precise, we need $T \geq \widetilde { \Theta } ( d ^ { 2 } k / \lambda _ { k } ^ { 2 } )$ for Algorithm 2 to take sufficient actions; however, the conclusion still holds for smaller $T$ . Since simply playing 0 for all $T$ actions will give a sharper bound of $\Re ( T ) \le r ^ { * } T \le \widetilde { \cal O } ( r ^ { * } \sqrt { d ^ { 2 } k \lambda _ { k } ^ { - 2 } T } ) \}$ ). For cleaner presentation, we won’t stress this for every statement.

Proof. The corollary uses a special property of the strongly convex action set that ensures: $A ^ { * } = M / r ^ { * }$ . With $\widehat { A }$ that satisfies $\hat { \| } \hat { A } \| _ { F } \overset { \cdot } { = } 1$ , we have

$$
\begin{array}{l} r ^ {*} - f _ {M} (\boldsymbol {A}) = r ^ {*} - \langle \widehat {\boldsymbol {A}}, \boldsymbol {M} \rangle = r ^ {*} - \langle \widehat {\boldsymbol {A}}, r ^ {*} \boldsymbol {A} ^ {*} \rangle \\ = \frac {r ^ {*}}{2} (2 - 2 \langle \widehat {\boldsymbol {A}}, \boldsymbol {A} ^ {*} \rangle) = \frac {r ^ {*}}{2} (\| \widehat {\boldsymbol {A}} \| _ {F} ^ {2} + \| \boldsymbol {A} ^ {*} \| _ {F} ^ {2} - \langle \widehat {\boldsymbol {A}}, \boldsymbol {A} ^ {*} \rangle) \\ = \frac {r ^ {*}}{2} \| \widehat {\boldsymbol {A}} - \boldsymbol {A} ^ {*} \| _ {F} ^ {2} \leq \frac {r ^ {*} \varepsilon^ {2}}{2} \tag {1} \\ \end{array}
$$

Therefore, with first ${ \cal T } _ { 1 } ~ = ~ \widetilde O ( d ^ { 2 } k \lambda _ { k } ^ { - 2 } \varepsilon ^ { - 2 } )$ exploratory samples we get $r ^ { * } - f ( { \widehat { A } } ) ~ \leq$ $\begin{array} { r } { r ^ { * } \varepsilon ^ { 2 } / 2 = r ^ { * } \sqrt { \frac { d ^ { 2 } k } { \lambda _ { k } ^ { 2 } T } } = \sqrt { \frac { ( r ^ { * } ) ^ { 2 } d ^ { 2 } k } { \lambda _ { k } ^ { 2 } T } } } \end{array}$ . Together we have:

$$
\begin{array}{l} \Re (T) = \sum_ {t = 1} ^ {T _ {1}} r ^ {*} - f (\pmb {A} _ {t}) + \sum_ {t = T _ {1} + 1} ^ {T} r ^ {*} - f (\widehat {\pmb {A}}) \\ <   r ^ {*} T _ {1} + T r ^ {*} \varepsilon^ {2} \\ \leq \widetilde {O} (\sqrt {d ^ {2} k (r ^ {*}) ^ {2} \lambda_ {k} ^ {- 2} T}). \\ \end{array}
$$

![](images/0bddfd6634d6663c6573c0792ab887d774d59402cbbe1ddada2030e8ffade151.jpg)

Notice that $k \leq \| M \| _ { F } ^ { 2 } / \lambda _ { k } ^ { 2 } \leq k \widetilde { \kappa } ^ { 2 }$ is order $k$ . Thus for well-conditioned matrices $M$ , our upper bound of $\widetilde { \cal O } ( \sqrt { d ^ { 2 } k \| M \| _ { F } ^ { 2 } / \lambda _ { k } ^ { 2 } T } )$ matches the lower bound $\sqrt { d ^ { 2 } k ^ { 2 } T }$ except for logarithmic factors.

In the previous setting with the bandit eigenvalue problem, estimating $M$ up to an $\epsilon$ error (measured by operator norm) gives us an $\epsilon$ -optimal reward. Therefore the sample complexity for eigenvalue reward with similar subspace iteration is $\| M \| _ { 2 } ^ { 2 } d ^ { 2 } k / ( \lambda _ { k } ^ { 2 } \epsilon ^ { 2 } )$ . In this section, on the other hand, we need Frobenius norm bound $\| A _ { L } - A ^ { * } \| _ { F } \leq \epsilon$ ; naturally the complexity becomes $\| M \| _ { F } ^ { 2 } d ^ { 2 } k / ( \lambda _ { k } ^ { 2 } \epsilon ^ { 2 } )$ .

Theorem 3.10 (Regret bound for low-rank linear reward: gap-free case). Set $\begin{array} { r } { \varepsilon ^ { 6 } = \Theta \bigl ( \frac { d ^ { 2 } k ^ { 2 } } { ( r ^ { * } ) ^ { 2 } T } \bigr ) } \end{array}$ (  (r ∗ )2 T ) , $\begin{array} { r } { n = \widetilde { \Theta } ( \frac { d ^ { 2 } k ^ { 2 } } { ( r ^ { * } ) ^ { 2 } \varepsilon ^ { 4 } } ) , L = \Theta ( \log ( d / \varepsilon ) ) } \end{array}$ and $k ^ { \prime } = 2 k$ in Algorithm 2 and get $\widehat { A }$ . Then we play it for the remaining steps, the cumulative regret satisfies:

$$
\Re (T) \leq \widetilde {O} ((d k T) ^ {2 / 3} (r ^ {*}) ^ {1 / 3}).
$$

We summarize all our results and prior work for quadratic reward in Table 2.

# 3.2.1 RL with Simulator: $Q$ -function is Quadratic and Bellman Complete

In this section we demonstrate how our results for non-concave bandits also apply to reinforcement learning. Let $\mathcal { T } _ { h }$ be the Bellman operator applied to the Q-function $Q _ { h + 1 }$ defined as:

$$
\mathcal {T} _ {h} (Q _ {h + 1}) (s, a) = r _ {h} (s, a) + \mathbb {E} _ {s ^ {\prime} \sim \mathbb {P} (\cdot | s, a)} [ \max  _ {a ^ {\prime}} Q _ {h + 1} (s ^ {\prime}, a ^ {\prime}) ].
$$

Definition 3.11 (Bellman complete). Given MDP $\boldsymbol { \mathcal { M } } = ( \boldsymbol { \mathcal { S } } , \boldsymbol { \mathcal { A } } , \mathbb { P } , \boldsymbol { r } , \boldsymbol { H } )$ , function class $\mathcal { F } _ { h } : S \times \mathcal { A } \mapsto \mathbb { R } , h \in [ H ]$ is called Bellman complete if for all $h \in [ H ]$ and $Q _ { h + 1 } \in \mathcal { F } _ { h + 1 }$ , $\mathcal { T } _ { h } ( Q _ { h + 1 } ) \in \mathcal { F } _ { h }$ .

Assumption 3.12 (Bellman complete for low-rank quadratic reward). We assume the function class $\mathcal { F } _ { h } = \{ f _ { M } : f _ { M } ( s , a ) = \phi ( s , a ) ^ { \top } M \phi ( s , a ) , \mathrm { r a n k }$ $\mathrm { r a n k } ( M ) \leq k$ , and $0 < \lambda _ { 1 } ( M ) / \lambda _ { \mathrm { m i n } } ( M ) \leq$ $\widetilde { \kappa } . \}$ is a class of quadratic function and the MDP is Bellman complete. Here $f _ { M } ( \phi ( s , a ) ) =$ ${ \textstyle \sum _ { j = 1 } ^ { k } \lambda _ { j } ( \pmb { v } _ { j } ^ { \top } \phi ( s , a ) ) ^ { 2 } }$ when $\begin{array} { r } { \pmb { M } = \sum _ { j = 1 } ^ { k } \lambda _ { j } \pmb { v } _ { j } \pmb { v } _ { j } ^ { \top } } \end{array}$ . Write $Q _ { h } ^ { \ast } = f _ { M _ { h } ^ { \ast } } \in \mathcal { F } _ { h }$ .

Observation: When querying $s _ { h - 1 } , a _ { h - 1 }$ , we observe $s _ { h } ^ { \prime } \sim \mathbb { P } ( \cdot | s _ { h - 1 } , a _ { h - 1 } )$ and reward $r _ { h - 1 } \big ( s _ { h - 1 } , a _ { h - 1 } \big )$ .

Oracle to recover parameter $\widehat { M }$ : Given $n \geq \widetilde { \Theta } ( d ^ { 2 } k ^ { 2 } \widetilde { \kappa } ^ { 2 } ( M ) / \varepsilon ^ { 2 } )$ , if one can play $\geq n$ samples $\mathbf { a } _ { i }$ and observe $y _ { i } \sim \mathbf { a } _ { i } ^ { \top } M \mathbf { a } _ { i } + \eta _ { i } , i \in [ n ]$ with 1-sub-gaussian and mean-zero noise $\eta$ , we can recover $\widehat { \pmb { M } } = \widehat { \pmb { M } } ( \{ ( \mathbf { a } _ { i } , y _ { i } ) \} )$ such that $\| \widehat { M } - M \| _ { 2 } \leq \varepsilon$ . This oracle is implemented via our analysis from the bandit setting.

With the oracle, at time step $H$ , we can estimate $\widehat { M } _ { H }$ that is $\epsilon / H$ close to $M _ { H } ^ { * }$ in spectral norm through noisy observations from the reward function with ${ \widetilde O } ( \widetilde { \kappa } ^ { 2 } d ^ { 2 } H ^ { 2 } / { \epsilon } ^ { 2 } )$ samples. Next, for each time step $h = H { - } 1 , H { - } 1 , \cdots , 1$ , sample $s _ { i } ^ { \prime } \sim \mathbb { P } ( \cdot | s , a )$ , we define $\begin{array} { r } { \eta _ { i } = \underset { . } { \operatorname* { m a x } } _ { a ^ { \prime } } f _ { \widehat { M } _ { h + 1 } } ( s _ { i } ^ { \prime } , a ^ { \prime } ) - \underset { . } { \mathbb { E } } _ { s ^ { \prime } \sim \mathbb { P } ( \cdot \vert s , a ) } \underset { . } { \operatorname* { m a x } } _ { a ^ { \prime } } f _ { \widehat { M } _ { h + 1 } } ( s ^ { \prime } , a ^ { \prime } ) } \end{array}$ . $\eta _ { i }$ is mean-zero and $O ( 1 )$ -subgaussian since it is bounded. Denote $M _ { h }$ as the matrix that satisfies $f _ { M _ { h } } : = T f _ { \widehat { M } _ { h + 1 } }$ , which is well-defined due to Bellman completeness. We estimate $\widehat { M } _ { h }$ from the noisy observations $y _ { i } = r _ { h } ( s , a ) + \operatorname* { m a x } _ { a ^ { \prime } } f _ { \widehat M _ { h + 1 } } ( s _ { i } ^ { \prime } , a ^ { \prime } ) = \mathcal { T } f _ { \widehat M _ { h + 1 } } + \eta _ { i } = : f _ { M _ { h } } + \eta _ { i }$ . Therefore with the oracle, we can estimate $\widehat { M } _ { h }$ such that $\| \widehat { M } _ { h } - M _ { h } \| _ { 2 } \leq \epsilon / H$ with $\Theta ( \widetilde { \kappa } ^ { 2 } d ^ { 2 } k ^ { 2 } H ^ { 2 } / \epsilon ^ { 2 } )$ bandits. More details are deferred to Algorithm 6 and the Appendix. We state the theorem on sample complexity of finding $\epsilon$ -optimal policy here:

Theorem 3.13. Suppose $\mathcal { F }$ is Bellman complete associated with parameter $\widetilde { \kappa }$ . With probability $1 - \delta$ , Algorithm 6 learns an ǫ-optimal policy $\pi$ with $\widetilde { \Theta } ( d ^ { 2 } \dot { k } ^ { 2 } \widetilde { \kappa } ^ { 2 } H ^ { 3 } / \epsilon ^ { 2 } )$ samples.

Existing approaches require $O ( d ^ { 3 } H ^ { 2 } / \epsilon ^ { 2 } )$ trajectories, or equivalently $O ( d ^ { 3 } H ^ { 3 } / \epsilon ^ { 2 } )$ samples, though they operate in the online RL setting [ZLKB20, $\mathrm { D K L } ^ { + } 2 1$ , JLM21], which is worse by a factor of $d$ .

It is an open problem on how to attain $d ^ { 2 }$ sample complexity in the online RL setting. The quadratic Bellman complete setting can also be easily extended to any of the polynomial settings of Section 3.3.

# 3.3 Stochastic High-order Homogeneous Polynomial Reward

Next we move on to homogeneous high-order polynomials.

# 3.3.1 The symmetric setting

Let reward function be a $p$ -th order stochastic polynomial function $f : { \mathcal { A } }  \mathbb { R }$ , where the action set $\mathcal { A } : = \{ B _ { 1 } ^ { d } = \{ \pmb { a } \in \mathbb { R } ^ { d } , \| \pmb { a } \| \leq 1 . \}$ . $f ( \pmb { a } ) = \pmb { T } ( \pmb { a } ^ { \otimes p } ) , r _ { t } = f ( \pmb { a } ) + \eta _ { t }$ , where $\begin{array} { r } { \pmb { T } = \sum _ { j = 1 } ^ { k } \lambda _ { j } \pmb { v } _ { j } ^ { \otimes \tilde { p } } } \end{array}$ is an orthogonally decomposable rank- $k$ tensor. $\{ \pmb { v } _ { 1 } , \cdots \pmb { v } _ { k } \}$ form an orthonormal basis. Optimal reward $r ^ { * }$ satisfies $1 \geq r ^ { * } = \lambda _ { 1 } \geq | \lambda _ { 2 } | \cdot \cdot \cdot \geq | \lambda _ { k } |$ . Noise $\eta _ { t } \sim \mathcal { N } ( 0 , 1 )$ .

In this setting, the problem is fundamentally more challenging than quadratic reward functions. On the one hand, it has a higher noise-to-signal ratio with larger $p$ . One can tell from the rank-1 setting where $\pmb { T } = \lambda _ { 1 } \pmb { v } _ { 1 } ^ { \otimes p }$ . For a randomly generated action $^ { a }$ on the unit ball, $\mathbb { E } [ \| \pmb { a } ^ { \top } \pmb { v } _ { 1 } \| ^ { 2 } ] = 1 / d$ . Therefore on average the signal strength is only $( { \pmb a } ^ { \top } { \pmb v } _ { 1 } ) ^ { p } \sim d ^ { - p / 2 }$ , much smaller than the noise level 1. Intuitively this demonstrates why higher complexity is needed for high-order polynomials. On the other hand, it is also technically more challenging. Unlike the matrix case, the expected zeroth-order update is no longer equal to any tensor product. Therefore existing tensor decomposition arguments do not apply. Fortunately, we prove that zeroth-order optimization still pushes the iterated actions toward the optimal action with linear convergence, given a good initialization. We show the bandit optimization procedure in Algorithm 3 and present the result in Theorem 3.14:

Theorem 3.14 (Staged progress). For each stage s, with high probability $\mathcal { A } _ { s }$ is not empty; and at least one action $\pmb { a } \in \mathcal { A } _ { s }$ satisfies: $\tan \theta ( { \pmb a } , { \pmb a } ^ { * } ) \leq \widetilde { \varepsilon } _ { s } = 2 ^ { - s }$ .

We defer the proofs together with formal statements to Appendix C.1; we will also present the proof sketch in the next subsection 3.3.2.

Remark 3.15 (Choice of step-size). Here we choose step-size $\zeta = 1 / 2 p$ . Note that $\mathbb { E } [ \pmb { y } ] =$ $\zeta ( 1 - \zeta ) ^ { p } \sigma ^ { 2 } \nabla _ { a } f ( { \pmb a } )$ . The scaling $( 1 - \zeta ) ^ { p } \geq 1 / \sqrt { e }$ ensures the signal to noise ratio not too small. The choice of weighted action is a delicate balance between making progress in optimization and controlling the noise to signal ratio.

Corollary 3.16 (Regret bound for tensors). Algorithm 3 yields an regret of: $\Re ( T ) \ \leq$ $\widetilde { O } \left( \sqrt { k d ^ { p } T } \right)$ , with high probability.

Corollary 3.17 (Regret bound with burn-in period). In Algorithm 3, we first set $\varepsilon =$ $1 / p , n _ { s } = \widetilde { \Theta } ( d ^ { p } / \lambda _ { 1 } ^ { 2 } )$ . We can first estimate the action $^ { a }$ such that ${ \pmb v } _ { 1 } ^ { \top } { \pmb a } \ge 1 - 1 / p$ with $\widetilde { O } ( k d ^ { p } / \lambda _ { 1 } ^ { 2 } )$ samples. Next we change $\varepsilon = \widetilde { \Theta } ( k ^ { 1 / 4 } d ^ { 1 / 2 } \lambda _ { 1 } ^ { - 1 / 2 } T ^ { - 1 / 4 } )$ , and $n _ { s } = \widetilde \Theta ( d ^ { 2 } \varepsilon ^ { - 2 } \lambda _ { 1 } ^ { - 2 } )$ in Algorithm 3. This procedure suffices to find $\lambda _ { 1 } \varepsilon ^ { 2 }$ -optimal reward with $\widetilde { O } ( k d ^ { 2 } / \lambda _ { 1 } ^ { 2 } \varepsilon ^ { 2 } )$ samples in the candidate set with size at most $\widetilde O ( k )$ . Finally with the UCB algorithm altogether we have a regret bound of:

$$
\widetilde {O} \left(\frac {k d ^ {p}}{\lambda_ {1}} + \sqrt {k d ^ {2} T}\right).
$$

Algorithm 3 Phased elimination with zeroth order exploration.   
1: Input: Function $f: \mathcal{A} \to \mathbb{R}$ of polynomial degree $p$ generating noisy reward, failure probability $\delta$ , error $\varepsilon$ .  
2: Initialization: $L_0 = C_L k \log(1/\delta)$ ; Total number of stages $S = C_S [\log(1/\varepsilon)] + 1$ , $A_0 = \{a_0^{(1)}, a_0^{(2)}, \dots, a_0^{(L_0)}\}$ where each $a_0^{(l)}$ is uniformly sampled on the unit sphere $\mathbb{S}^{d-1}$ . $\widetilde{\varepsilon}_0 = 1$ .  
3: for $s$ from 1 to $S$ do  
4: $\widetilde{\varepsilon}_s \gets \widetilde{\varepsilon}_{s-1}/2$ , $n_s \gets C_n d^p \log(d/\delta)/\lambda_1^2 \widetilde{\varepsilon}_s^2$ , $n_s \gets n_s \cdot \log^3(n_s/\delta)$ , $m_s \gets C_m d \log(n_s/\delta)$ , $A_s = \emptyset$ .  
5: for $l$ from 1 to $L_{s-1}$ do  
6: Zeroth-order optimization:  
7: Locate current action $\widetilde{\boldsymbol{a}} = \boldsymbol{a}_{s-1}^{(l)}$ .  
8: for $[(1/(1-\alpha)) \log(2d)]$ times do  
9: Sample $z_i \sim \mathcal{N}(0, 1/m_s I_d)$ , $i = 1, 2, \dots, n_s$ .  
10: Take actions $a_i = (1 - \frac{1}{2p})\widetilde{\boldsymbol{a}} + \frac{1}{2p}z_i$ and observe $r_i = T(a_i) + \eta_i, i \in [n_s]$ ; Take actions $\frac{1}{2p}z_i$ and observe $r_i' = T(\frac{1}{2p}z_i) + \eta_i'$ , $i \in [n_s]$ .  
11: Conduct estimation $y \gets 1/n_s \sum_{i=1}^{n_s} (r_i - r_i')z_i$ .  
12: Update the current action $\widetilde{\boldsymbol{a}} \gets y / \|y\|$ .  
13: Estimate the expected reward for $\widetilde{\boldsymbol{a}}$ through $n_s$ samples: $r_n(\widetilde{\boldsymbol{a}}) = 1/n_s \sum_{i=1}^{n_s} (T(\widetilde{\boldsymbol{a}}) + \eta_i)$ .  
14: Candidate Elimination:  
if $r_n \geq \lambda_1 (1 - p\widetilde{\varepsilon}_s^2)$ then  
Keep the action $A_s \gets A_s \cup \{\widetilde{\boldsymbol{a}}\}$ Label the actions: $L_s = |\mathcal{A}_s|, A_s =: \{a_s^{(1)}, \dots, a_s^{(L_s)}\}$ .  
Run UCB (Algorithm 7) with the candidate set $A_S$ .

In Section 3.3.4 we will also demonstrate the necessity of the sample complexity in this burn-in period.

# 3.3.2 Proof Sketch of Theorem 3.14

Definition 3.18 (Zeroth order gradient function). For some scalar $m$ , we define an empirical operator $G _ { n } : { \mathcal { A } }  { \mathcal { A } }$ that is similar to the zeroth-order gradient of $f$ through $n$ samples:

$$
G _ {n} (\pmb {a}) := \frac {m}{n} \sum_ {i = 1} ^ {n} \left(T \left(\left((1 - \frac {1}{2 p}) \pmb {a} + \frac {1}{2 p} \pmb {z} _ {i}\right) ^ {\otimes p}\right) - T (\frac {1}{2 p} \pmb {z} _ {i})\right) \pmb {z} _ {i} + (\eta_ {i} - \eta_ {i} ^ {\prime}) \pmb {z} _ {i}.
$$

where $\begin{array} { r } { z _ { i } \sim \mathcal { N } ( 0 , \frac { 1 } { m } \pmb { I } ) } \end{array}$ and $\eta _ { i } , \eta _ { i } ^ { \prime }$ are independent zero-mean 1-sub-Gaussian noise. Therefore we have:

$$
\mathbb {E} \left[ G _ {n} (\boldsymbol {a}) \right] = m \mathbb {E} \left[ \sum_ {l = 0} ^ {p - 1} \binom {p} {l} \boldsymbol {T} \left(\left(1 - \frac {1}{2 p}\right) ^ {p - l} \boldsymbol {a} ^ {\otimes (p - l)} \otimes \left(\frac {1}{2 p}\right) ^ {l} \boldsymbol {z} ^ {\otimes l}\right) \boldsymbol {z} \right]
$$

(Due to symmetry of Gaussian only for odd $l = : 2 s + 1$ expectation is nonzero)

$$
= (1 - \frac {1}{2 p}) ^ {p - 2 s - 1} (\frac {1}{2 p}) ^ {2 s + 1} [ \sum_ {s = 0} ^ {\lfloor p / 2 - 1 \rfloor} m ^ {- s} \binom {p} {2 s + 1} T (\boldsymbol {a} ^ {\otimes (p - 2 s - 1)} \otimes \boldsymbol {I} ^ {\otimes s + 1}) ]
$$

Note that for even $p$ the last term (when $s = p / 2 - 1 )$ is $\begin{array} { r } { \pmb { T } ( \pmb { a } \otimes \pmb { I } ^ { \otimes p / 2 } ) = \sum _ { j = 1 } ^ { k } \lambda _ { j } ( \pmb { a } ^ { \top } \pmb { v } _ { j } ) \pmb { v } _ { j } } \end{array}$ While all other terms will push the iterate towards the optimal action at a superlinear speed, the last term perform a matrix multiplication and the convergence speed will depend on the eigengap. Therefore for $p \geq 4$ we will remove the extra bias in the last term that is orthogonal to ${ \pmb v } _ { 1 }$ and will treat it as noise. (Notice for quadratic function $s = 0 = p / 2 - 1$ is the only term in $\mathbb { E } [ G _ { n } ( \pmb { a } ) ]$ . This is the distinction between $p = 2$ and larger $p$ , and why its convergence depends on eigengap.)

We further define $G ( \pmb { a } )$ as the population version of $G _ { n } ( { \pmb a } )$ by removing this undesirable bias term that will be treated as noise:

$$
\begin{array}{l} G (\boldsymbol {a}) = \left\{ \begin{array}{l l} \mathbb {E} [ G _ {n} ] - \frac {(\frac {1}{2 p}) ^ {p - 1} (1 - \frac {1}{2 p}) p}{m ^ {p / 2 - 1}} \sum_ {j = 2} ^ {k} \lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) \boldsymbol {v} _ {j}, & \text {w h e n p i s e v e n} \\ \mathbb {E} [ G _ {n} ], & \text {w h e n p i s o d d}. \end{array} \right. \\ = \sum_ {s = 0} ^ {\lfloor (p - 3) / 2 \rfloor} \frac {\left(\frac {1}{2 p}\right) ^ {2 s + 1}}{m ^ {s}} \binom {p} {2 s + 1} T \left(I ^ {\otimes s + 1} \otimes \left((1 - \frac {1}{2 p}) a\right) ^ {\otimes p - 2 s - 1}\right) \\ = \frac {1}{2} \left(1 - \frac {1}{2 p}\right) ^ {p - 1} \boldsymbol {T} (\boldsymbol {I}, \boldsymbol {a} ^ {\otimes p - 1}) + O (1 / m). \\ \end{array}
$$

We define $G ( \pmb { a } )$ to push the action $^ { a }$ towards the ${ \pmb v } _ { 1 }$ direction with at least linear convergence rate. More precisely, their angle $\tan \theta ( G ( \pmb { a } ) , \pmb { v } _ { 1 } )$ will converge linearly to 0 for proper initialization with the dynamics $\mathbf { \pmb { a } }  G ( \pmb { a } )$ . An easy way to see that is when $p = 2$ or 3, $G$ is conducting (3-order tensor) power iteration. For higher-order problems, this operation $G$ is equivalent to the summation of $p , p - 2 , p - 4 , \cdots$ -th order tensor product and hence the linear convergence.

The estimation error $G _ { n } ( { \pmb a } ) - G ( { \pmb a } )$ will be treated as noise (which is not mean zero when $p$ is even but will be small enough: $O ( ( 2 p ) ^ { - p } m ^ { - ( p - 1 ) / 2 } ) ;$ ). Therefore the iterative algorithm with ${ \pmb a }  G _ { n } ( { \pmb a } )$ will converge to a small neighborhood of ${ \pmb v } _ { 1 }$ depending on the estimation error. This estimation error is controlled by the choice of sample size $n$ in each iteration. We now provide the proof sketch:

Lemma 3.19 (Initialization for $p \geq 3$ ; Corollary C.1 from [WA16] ). For any $\eta \in ( 0 , 1 / 2 )$ , with $L = \Theta ( k \log ( 1 / \eta ) )$ samples $\mathcal { A } = \{ \pmb { a } ^ { ( 1 ) } , \pmb { a } ^ { ( 2 ) } , \cdot \cdot \cdot \pmb { a } ^ { ( L ) } \}$ where each $\mathbf { \Delta } \mathbf { a } ^ { ( l ) }$ is sampled

uniformly on the sphere $\mathbb { S } ^ { d - 1 }$ . At least one sample ${ \pmb a } \in \mathcal { A }$ satisfies

$$
\max  _ {j \neq 1} | \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a} | \leq 0. 5 | \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} |, a n d | \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} | \geq 1 / \sqrt {d}. \tag {2}
$$

with probability at least $1 - \eta$

Lemma 3.20 (Iterative progress). Let $\alpha = 1 / 2$ for $p \geq 3$ in Algorithm 3. Consider noisy operation ${ \pmb a } ^ { + }  G ( { \pmb a } ) + { \pmb g }$ . If the error term $\textbf {  { g } }$ satisfies:

$$
\begin{array}{l} \left\| \boldsymbol {g} \right\| \leq \min  \left\{\frac {0 . 0 2 5}{p} \lambda_ {1} \left(\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}\right) ^ {p - 2}, 0. 1 \lambda_ {1} \widetilde {\varepsilon} \right\} \\ + 0. 0 3 \lambda_ {1} | \sin \theta (\boldsymbol {v} _ {1}, \boldsymbol {a}) | (\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}) ^ {p - 2}, \\ | \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {g} | \leq 0. 0 5 \lambda_ {1} (\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}) ^ {p - 1}. \\ \end{array}
$$

Suppose a satisfies $0 . 5 | \pmb { v } _ { 1 } ^ { \top } \pmb { a } | \geq \operatorname* { m a x } _ { j \geq 2 } | \pmb { v } _ { j } ^ { \top } \pmb { a } |$ , we have:

$$
\tan \theta (\pmb {a} ^ {+}, \pmb {v} _ {1}) \leq 0. 8 \tan \theta (\pmb {a}, \pmb {v} _ {1}) + \widetilde {\varepsilon}.
$$

We can also bound $\textbf {  { g } }$ by standard concentration plus an additional small bias term.

Lemma 3.21 (Estimation error bound for $G$ ). For fixed value $\delta \in ( 0 , 1 )$ and large enough universal constant $c _ { 1 } , c _ { 2 } , c _ { m } , c _ { n } ,$ , when $m = c _ { m } d \log ( n / \delta ) , n \geq c _ { n } d \log ( d / \delta ) ,$ , we have

$$
\begin{array}{l} \| \pmb {g} \| \equiv \| G _ {n} (\pmb {a}) - G (\pmb {a}) \| \leq c _ {1} \sqrt {\frac {d ^ {2} \log^ {3} (n / \delta) \log (d / \delta)}{n}} + e \lambda_ {2} | \sin \theta (\pmb {a}, \pmb {v} _ {1}) |, \\ | \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {g} | \equiv | \boldsymbol {v} _ {1} ^ {\top} G _ {n} (\boldsymbol {a}) - \boldsymbol {v} _ {1} ^ {\top} G (\boldsymbol {a}) | \leq c _ {2} \sqrt {\frac {d \log^ {3} (n / \delta) \log (d / \delta)}{n}}. \\ \end{array}
$$

with probability 1 − δ. $e = 0$ for odd $p$ and $e = ( 2 p ) ^ { - ( p - 1 ) } m ^ { - ( p / 2 - 1 ) }$ for even p.

Together we are able to prove Theorem 3.14:

Proof of Theorem 3.14. Initially with high probability there exists an ${ \pmb a } _ { 0 } \in \mathcal { A } _ { 0 }$ such that Eqn. (2) holds, i.e., ${ \pmb v } _ { 1 } ^ { \top } { \pmb a } _ { 0 } \ge 1 / \sqrt { d }$ and ${ \pmb v } _ { 1 } ^ { \top } { \pmb a } _ { 0 } \geq 2 | { \pmb v } _ { j } ^ { \top } { \pmb a } _ { 0 } | , \forall j \geq 2$ .

Next, from Lemma 3.21, the extra bias term is bounded by $e \lambda _ { 2 } | \sin \theta ( { \pmb a } , { \pmb v } _ { 1 } ) |$ $\leq 0 . 0 3 \lambda _ { 1 } ( { \pmb v } _ { 1 } ^ { \top } { \pmb a } ) ^ { p - 2 } | \sin \theta ( { \pmb a } , { \pmb v } _ { 1 } ) |$ since $e = ( 2 p ) ^ { - p + 1 } m _ { s } ^ { - p / 2 + 1 }$ and with our choice of variance $m _ { s } \geq d \geq ( v _ { 1 } ^ { \top } a ) ^ { - 2 }$ , plus $p \geq 3$ . Next with our setting of $n _ { s } = \widetilde { \Theta } ( d ^ { p } / ( \lambda _ { 1 } ^ { 2 } \widetilde { \varepsilon } _ { t } ^ { 2 } ) )$ , the error term $\| \mathbb { E } [ G ( { \pmb a } ) ] - G _ { n } ( { \pmb a } ) \|$ is upper bounded by $\widetilde { \cal O } ( \sqrt { \frac { d ^ { 2 } } { n } } ) \le 0 . 0 2 5 \lambda _ { 1 } d ^ { - ( p - 2 ) / 2 } \widetilde { \varepsilon } _ { s } / p +$ $0 . 1 \lambda _ { 1 } \widetilde { \varepsilon } _ { s }$ . Meanwhile $\begin{array} { r } { | \pmb { v } _ { 1 } ^ { \top } \pmb { g } | \leq \widetilde { O } ( \sqrt { \frac { d } { n _ { s } } } ) \leq 0 . 0 5 \lambda _ { 1 } ( \pmb { v } _ { 1 } ^ { \top } \pmb { a } ) ^ { p - 1 } , } \end{array}$

This meets the requirements for Theorem 3.20 and therefore $\tan \theta ( G _ { n } ( { \pmb a } _ { 0 } ) , { \pmb v } _ { 1 } ) \ \leq$ $0 . 8 \tan \theta ( { \pmb a } _ { 0 } , { \pmb v } _ { 1 } ) + 0 . 1 \lambda _ { 1 } \widetilde { \varepsilon } _ { s }$ . Therefore after $l$ steps will have

$$
\begin{array}{l} \tan \theta (G _ {n} ^ {l} (\pmb {a} _ {0}), \pmb {v} _ {1}) \leq 0. 8 ^ {l} \tan \theta (\pmb {a} _ {0}, \pmb {v} _ {1}) + \sum_ {i = 1} ^ {l} 0. 8 ^ {i} \cdot 0. 1 \widetilde {\varepsilon_ {s}} \\ \leq 0. 8 ^ {l} \tan \theta (\boldsymbol {a} _ {0}, \boldsymbol {v} _ {1}) + 0. 5 \widetilde {\varepsilon_ {s}}. \\ \end{array}
$$

Notice initially tan $1 \theta ( { \pmb a } _ { 0 } , { \pmb v } _ { 1 } ) \leq 1 / ( { \pmb v } _ { 1 } ^ { \top } { \pmb a } _ { 0 } ) \leq \sqrt { d }$ . Therefore after at most $l = O ( \log _ { 2 } ( \tan \theta ( { \pmb a } _ { 0 } , { \pmb v } _ { 1 } ) ) ) \leq O ( \log _ { 2 } ( d ) )$ steps, we will have $\tan ( G _ { n } ^ { l } ( { \pmb a } _ { 0 } ) , { \pmb v } _ { 1 } ) \leq \widetilde { \varepsilon } _ { 0 } / 2 =$ $\widetilde { \varepsilon } _ { 1 }$ . With the same argument, the progress also holds for $s > 0$ with even smaller $l$ . □

# 3.3.3 The asymmetric setting

Now we consider the asymmetric tensor problem with reward $f : { \mathcal { A } }  \mathbb { R }$ . The input space $\mathcal { A }$ consists of $p$ vectors in a unit ball: ${ \vec { a } } = ( \mathbf { { \boldsymbol { a } } } ( 1 ) , \mathbf { { \boldsymbol { a } } } ( 2 ) , \cdots { \boldsymbol { a } } ( p ) ) \in \mathcal { A } , \| \mathbf { { \boldsymbol { a } } } ( s ) \| \leq 1 , \forall s \in [ p ]$ . $f ( { \vec { \pmb { a } } } ) = \pmb { T } ( \otimes _ { s = 1 } ^ { p } \pmb { a } ( s ) ) + \eta$ . Tensor $\begin{array} { r } { \pmb { T } = \sum _ { j = 1 } ^ { k } \lambda _ { j } \pmb { v } _ { j } ( 1 ) \otimes \pmb { v } _ { j } ( 2 ) \cdots \otimes \pmb { v } _ { j } ( p ) } \end{array}$ . For each $s \in [ p ]$ , $\{ \pmb { v } _ { 1 } ( s ) , \pmb { v } _ { 2 } ( s ) , \pmb { \cdot } \cdot \pmb { \cdot } \pmb { v } _ { k } ( s ) \}$ are orthonormal vectors. We order the eigenvalues such that $\lambda _ { 1 } \geq | \lambda _ { 2 } | \cdot \cdot \cdot \geq | \lambda _ { k } |$ . Therefore the optimal reward is $\lambda _ { 1 }$ and can be achieved by $\pmb { a } ^ { * } ( s ) = \pmb { v } _ { 1 } ( s ) , s \in [ p ]$ . In this section we only consider $p \geq 3$ and leave the quadratic and low-rank matrix setting to the next section.

Theorem 3.22. For $p \ \geq \ 3 .$ , by conducting alternating power iteration, one can get a $\varepsilon$ -optimal reward with a total $\widetilde { \cal O } \left( ( 2 k ) ^ { p } \log ^ { \bar { p } } ( p / \delta ) d ^ { p } \lambda _ { 1 } ^ { - 1 } \bar { \varepsilon } ^ { - 1 } \right)$ actions; therefore the regret bound is at most $\widetilde { O } ( \sqrt { k ^ { p } d ^ { p } T } )$ .

This setting is actually much easier than the symmetric setting. Notice by replacing one slice of $\vec { \pmb { a } }$ by random Gaussian $z _ { i } \sim \mathcal { N } ( 0 , 2 / d \log ( d / \delta ) )$ , one directly gets $\pmb { T } ( \pmb { a } ( 1 ) , \cdots \pmb { a } ( s -$ $1 ) , I , \pmb { a } ( s + 1 ) , \cdot \cdot \cdot \pmb { a } ( p ) )$ on each slice with $1 / n \sum _ { i } f ( \pmb { a } ( 1 ) , \cdot \cdot \cdot \pmb { a } ( s - 1 ) , z _ { i } , \pmb { a } ( s + 1 ) , \cdot \cdot \cdot \pmb { a } ( p ) ) z _ { i }$ which is tensor product. We defer the proof to Appendix C.4.

# 3.3.4 Lower Bounds for Stochastic Polynomial Bandits

In this section we show lower bound for stochastic polynomial bandits.

$$
f (\boldsymbol {a}) = \prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a}\right) + \eta , \text {w h e r e} \eta \sim N (0, 1), \| \boldsymbol {a} \| _ {2} \leq 1 \text {a n d} f (\boldsymbol {a}) \leq 1. \tag {3}
$$

Theorem 3.23. Define minimax regret as follow

$$
\Re (d, p, T) = \inf _ {\pi} \sup _ {(\pmb {\theta} _ {1}, \dots , \pmb {\theta} _ {p})} \mathbb {E} _ {(\pmb {\theta} _ {1}, \dots , \pmb {\theta} _ {p})} \left[ T \max _ {\pmb {a}} \prod_ {i = 1} ^ {p} (\pmb {\theta} _ {i} ^ {\top} \pmb {a}) - \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\pmb {\theta} _ {i} ^ {\top} \pmb {a} ^ {(t)}) \right].
$$

For all algorithms $A$ that adaptively interact with bandit (Eq (3)) for $T$ rounds, we have $\Re ( d , p , T ) \ge \Omega ( \sqrt { d ^ { p } T } / p ^ { p } )$ .

From the theorem, we can see even when the problem is rank-1, any algorithm incurs at least $\Omega ( \sqrt { d ^ { p } T } / p ^ { p } )$ regret. This further implies algorithm requires sample complexity of $\Omega ( ( d / p ^ { 2 } ) ^ { p } / \epsilon ^ { 2 } )$ to attain $\epsilon$ -optimal reward. This means our regret upper bound obtained in Corollary 3.16 is optimal in terms of dependence on $d$ . In Appendix F we also show a $\Omega ( \sqrt { d ^ { p } T } )$ lower bound for asymmetric actions setting, which our upper bound up to polylogarithmic factors.

We note that with burn-in period, our algorithm also obtains a cumulative regret of $k d ^ { p } / \lambda _ { 1 } + \sqrt { k d ^ { 2 } T }$ , as shown in Corollary 3.17. Here for a fixed $\lambda _ { 1 }$ and very large $T$ , this is better result than the previous upper bound. We note that there is no contradiction with the lower bound above, since this worst case is achieved with a specific relation between $T$ and $\lambda _ { 1 }$ .

The burn-in period requires $k d ^ { p } / \lambda _ { 1 } ^ { 2 }$ samples to get a constant of $r ^ { * }$ . We want to investigate whether our dependence on $\lambda _ { 1 } \equiv r ^ { * }$ is optimal. Next we show a gap-dependent lower bound for finding an arm that is close to the optimal arm by a constant factor.

Theorem 3.24. For all algorithms $A$ that adaptively interact with bandit (Eq (3)) for $T$ rounds and output a vector $\pmb { a } ^ { ( T ) } \in \mathbb { R } ^ { d }$ , it requires at least $T = \Omega ( d ^ { p } / \lVert \pmb { \theta } \rVert ^ { 2 p } )$ rounds to find an arm $\pmb { a } ^ { ( T ) } \in \mathbb { R } ^ { d }$ such that

$$
\prod_ {i = 1} ^ {p} (\pmb {\theta} _ {i} ^ {\top} \pmb {a} ^ {(T)}) \geq \frac {3}{4} \cdot \max _ {\pmb {a}} \prod_ {i = 1} ^ {p} (\pmb {\theta} _ {i} ^ {\top} \pmb {a}).
$$

In the rank-1 setting $r ^ { * } = \| \pmb \theta \| ^ { p }$ and the lower bound for achieving a constant approximation for the optimal reward is $\Omega ( d ^ { p } / ( r ^ { * } ) ^ { 2 } )$ . Therefore our burn-in sample complexity is also optimal in the dependence on $d$ and $r ^ { * }$ .

# 3.4 Noiseless Polynomial Reward

In this subsection, we study the regret bounds for learning bandits with noiseless polynomial rewards. First we present the definition of admissible polynomial families.

Definition 3.25 (Admissible Polynomial Family). For $\pmb { a } \in \mathbb { R } ^ { d }$ , define $\widetilde { \pmb { a } } = [ 1 , \pmb { a } ^ { \top } ] ^ { \top }$ . For a algebraic variety $\mathcal { V } \subseteq ( \mathbb { R } ^ { d + 1 } ) ^ { \otimes p }$ , define $\mathcal { R } _ { \mathcal { V } } : = \{ r _ { \theta } ( \boldsymbol { \mathsf { a } } ) = \langle \pmb { \theta } , \widetilde { \pmb { a } } ^ { \otimes p } \rangle : \pmb { \theta } \in \mathcal { V } \}$ as the polynomial family with parameters in $\nu$ . We define the dimension of the family $\mathcal { R } _ { \mathcal { V } }$ as the algebraic dimension of $\nu$ . Next, define $\mathcal { X } : = \{ \widetilde { \pmb { a } } ^ { \otimes p } : \pmb { a } \in \mathbb { R } ^ { d } \}$ . An polynomial family $\mathcal { R } _ { \mathcal { V } }$ is said to be admissible4 w.r.t. $\mathcal { X }$ if for any $\pmb \theta \in \mathcal V$ , $\dim ( { \mathcal { X } } \cap { \bar { \ O ( X \in { \mathcal { X } } : \langle X , \theta \rangle = 0 \rangle } } \} ) \ <$ < $\dim ( { \mathcal { X } } ) = d$ .

# 3.4.1 Upper Bounds via Solving Polynomial Equations

We show that if the action set $\mathcal { A }$ is of positive measure with respect to the Lebesgue measure $\mu$ , then by playing actions randomly, we can uniquely solve for the ground-truth reward function $r _ { \pmb \theta } ( \pmb a )$ almost surely with samples of size that scales with the intrinsic algebraic dimension of $\nu$ , provided that $\nu$ is an admissible algebraic variety.

Theorem 3.26. Assume that the reward function class is an admissible polynomial family $\mathcal { R } _ { \mathcal { V } }$ , and the maximum reward is upper bounded by 1. If $\mu ( \mathcal { A } ) > 0$ , where µ is the Lebesgue

meaure, then by randomly sample actions $\mathbf { } a _ { 1 } , \ldots , \mathbf { } a _ { T }$ from $\mathbb { P } _ { \pmb { a } \sim \mathcal { N } ( 0 , I _ { d } ) } ( \cdot | \pmb { a } \in \mathcal { A } )$ , when $T \geq 2 \mathrm { d i m } ( \mathcal { V } ) .$ , we can uniquely solve for the ground-truth $\pmb \theta$ and thus determine the optimal action almost surely. Therefore, the cumulative regret at round $T$ can be bounded as

$$
\Re (T) \leq \min  \{T, 2 \dim (\mathcal {V}) \}.
$$

We state two important examples of admissible polynomial families with $O ( d )$ dimensions.

Example 3.27 (low-rank polynomials). The function class $\mathcal { R } _ { \mathcal { V } }$ of possibly inhomogeneous degree- $p$ polynomials with $k$ summands $\begin{array} { r } { \mathcal { R } _ { \mathcal { V } } = \{ r ( \pmb { a } ) = \sum _ { i = 1 } ^ { k } \lambda _ { i } \langle \pmb { v } _ { i } , \pmb { a } \rangle ^ { p _ { i } } \ | \ \lambda _ { i } \in \mathbb { R } , \pmb { v } _ { i } \in \mathbb { } } \end{array}$ $\mathbb { R } ^ { d } \}$ is admissible with $\mathrm { d i m } ( \mathcal { R } _ { \mathcal { V } } ) \leq d k$ , where $p = \operatorname* { m a x } \{ p _ { i } \}$ . Neural network with monomial/polynomial activation functions are low-rank polynomials.

Example 3.28 ([CM20]). The function class $\mathcal { R } _ { \mathcal { V } } = \left\{ r ( \pmb { a } ) = q ( \pmb { U } \pmb { a } ) ~ | ~ \pmb { U } \in \mathbb { R } ^ { k \times d } , \deg q ( \cdot ) \le \right.$ $p \}$ is admissible with $\dim ( \mathcal { V } ) \leq d k + ( k + 1 ) ^ { p }$ .

# 3.4.2 Lower Bounds with UCB Algorithms

In this subsection, we construct a hard bandit problem where the rewards are noiseless degree- $p$ polynomial, and show that any UCB algorithm needs at least $\Omega ( d ^ { p } )$ actions to learn the optimal action. On the contrary, Theorem 3.26 shows that by playing actions randomly, we only need $2 ( d k + ( p + 1 ) ^ { p } ) = O ( d )$ actions.

Hard Case Construction Let $e _ { i }$ denotes the $i$ -th standard orthonormal basis of $\mathbb { R } ^ { d }$ , i.e., $e _ { i }$ has only one 1 at the $i$ -th entry and 0’s for other entries. We define a $p$ -th multi-indices set $\Lambda$ as $\Lambda = \{ ( \alpha _ { 1 } , \ldots , \alpha _ { p } ) | 1 \le \alpha _ { 1 } < \cdots < \alpha _ { p } \le d \}$ . For an $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { p } ) \in \Lambda$ , denote $M _ { \alpha } = e _ { \alpha _ { 1 } } \otimes \cdot \cdot \cdot \otimes e _ { \alpha _ { p } } ,$ . Then the model space $\mathcal { M }$ is defined as $\mathcal { M } = \left\{ M _ { \alpha } | \alpha \in \Lambda \right\}$ , which is a subset of rank-1 $p$ -th order tensors. The action set $\mathcal { A }$ is defined as $\mathcal { A } = \mathrm { c o n v } ( \{ e _ { \alpha _ { 1 } } +$ $\begin{array} { r } { \cdots + e _ { \alpha _ { p } } | \alpha \in \Lambda \} , } \end{array}$ ). Assume that the ground-truth parameter is $M ^ { \ast } = M _ { \alpha ^ { \ast } } \in \mathcal { M }$ . The noiseless reward $\begin{array} { r } { r _ { t } = r ( M ^ { * } , \pmb { a } _ { t } ) = \langle M ^ { * } , ( \pmb { a } _ { t } ) ^ { \otimes p } \rangle = \prod _ { i = 1 } ^ { p } \langle e _ { \alpha _ { i } ^ { * } } , \pmb { a } _ { t } \rangle } \end{array}$ is a polynomial of $\mathbf { } \mathbf { a } _ { t }$ and falls into the case of Example 3.28.

UCB Algorithms The UCB algorithms sequentially maintain a confidence set $\mathcal { C } _ { t }$ after playing actions $\mathbf { } a _ { 1 } , \ldots , \mathbf { } a _ { t }$ . Then UCB algorithms play $\mathbf { \delta } \mathbf { a } _ { t + 1 } \in \arg \operatorname* { m a x } _ { \mathbf { \alpha } _ { a \in \mathcal { A } } } \mathrm { U C B } _ { t } ( \mathbf { \alpha } \mathbf { a } )$ where $\operatorname { U C B } _ { t } ( \pmb { a } ) = \operatorname* { m a x } _ { M \in \mathcal { C } _ { t } } \langle M , ( \pmb { a } ) ^ { \otimes p } \rangle$ .

Theorem 3.29. Assume that for each $t \geq 0$ , the confidence set $\mathcal { C } _ { t }$ contains the groundtruth model, i.e., $M ^ { \ast } \in \mathcal { C } _ { t }$ . Then for the noiseless degree-p polynomial bandits, any UCB algorithm needs to play at least $\binom { d } { p } - 1$ actions to distinguish models in M. Furthermore, the worst-case cumulative regret at round $T$ can be lower bounded by

$$
\mathfrak {R} (T) \geq \min  \{T, \left( \begin{array}{c} d \\ p \end{array} \right) - 1 \}.
$$

Theorem 3.29 shows the failure of the optimistic mechanism, which forbids the algorithm to play an informative action that is known to be of low reward for all models in the confidence set. On the contrary, the reward function class falls into the form of $q ( U \mathbf { a } )$ , therefore, by playing actions randomly5, we only need $O ( d )$ actions as Theorem 3.26 suggests.

# 4 Conclusion

In this paper, we design minimax-optimal algorithms for a broad class of bandit problems with non-concave rewards. For the stochastic setting, our algorithms and analysis cover the low-rank linear reward setting, bandit eigenvector problem, and homogeneous polynomial reward functions. We improve the best-known regret from prior work and attain the optimal dependence on problem dimension $d$ . Our techniques naturally extend to RL in the generative model settings. Furthermore, we obtain the optimal regret, dependent on the intrinsic algebraic dimension, for general polynomial reward without noise. Our regret bound demonstrates the fundamental limits of UCB algorithms, being $\Omega ( d ^ { p - 1 } )$ worse than our result for cases of interest.

We leave to future work several directions. First, the gap-free algorithms for the lowrank linear and bandit eigenvector problem do not attain $\bar { \sqrt { T } }$ regret. We believe $\sqrt { d ^ { 2 } T }$ regret without any dependence on gap is impossible, but do not have a lower bound. Secondly, our study of degree $p$ polynomials only covers the noiseless setting or orthogonal tensors with noise. We believe entirely new algorithms are needed for general polynomial bandits to surpass eluder dimension. As a first step, designing optimal algorithms would require understanding the stability of algebraic varieties under noise, so we leave this as a difficult future problem. Finally, we conjecture our techniques can be used to design optimal algorithms for representation learning in bandits and MDPs [YHLD20, $\mathrm { H C J ^ { + } } 2 1 ]$ .

# Acknowledgment

JDL acknowledges support of the ARO under MURI Award W911NF-11-1-0303, the Sloan Research Fellowship, NSF CCF 2002272, and an ONR Young Investigator Award. QL is supported by NSF 2030859 and the Computing Research Association for the CIFellows Project. SK acknowledges funding from the NSF Award CCF-1703574 and the ONR award N00014-18-1-2247. The authors would like to thank Qian Yu for numerous conversations regarding the lower bound in Theorem 3.23. JDL would like to thank Yuxin Chen and Anru Zhang for several conversations on tensor power iteration, Simon S. Du and Yangyi Lu for explaining to him to the papers of [JWWN19, LMT21], and Max Simchowitz and Chao Gao for several conversations regarding adaptive lower bounds.

# References

[ACCD12] Ery Arias-Castro, Emmanuel J Candes, and Mark A Davenport. On the fundamental limits of adaptive sensing. IEEE Transactions on Information Theory, 59(1):472–481, 2012.   
$[ \mathrm { A F H ^ { + } } 1 1 ]$ Alekh Agarwal, Dean P Foster, Daniel Hsu, Sham M Kakade, and Alexander Rakhlin. Stochastic convex optimization with bandit feedback. arXiv preprint arXiv:1107.1744, 2011.   
$[ \mathrm { A G H ^ { + } } 1 4 ]$ Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. Journal of machine learning research, 15:2773–2832, 2014.   
[AJKS19] Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep, 2019.   
[AYBM14] Yasin Abbasi-Yadkori, Peter L Bartlett, and Alan Malek. Linear programming for large-scale Markov decision problems. arXiv preprint arXiv:1402.6763, 2014.   
[AYPS11] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. In NIPS, volume 11, pages 2312–2320, 2011.   
[AYPS12] Yasin Abbasi-Yadkori, David Pal, and Csaba Szepesvari. Online-toconfidence-set conversions and application to sparse stochastic bandits. In Artificial Intelligence and Statistics, pages 1–9. PMLR, 2012.   
[AZL17] Zeyuan Allen-Zhu and Yuanzhi Li. First efficient convergence for streaming k-pca: a global, gap-free, and near-optimal rate. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pages 487–492. IEEE, 2017.   
[AZL19] Zeyuan Allen-Zhu and Yuanzhi Li. What can resnet learn efficiently, going beyond kernels? arXiv preprint arXiv:1905.10337, 2019.   
[BCB12] Sebastien Bubeck and Nicolo Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends in Machine Learning, 5(1), 2012.   
[BCR13] Jacek Bochnak, Michel Coste, and Marie-Françoise Roy. Real algebraic geometry, volume 36. Springer Science & Business Media, 2013.

[BDWY16] Maria-Florina Balcan, Simon Shaolei Du, Yining Wang, and Adams Wei Yu. An improved gap-dependency analysis of the noisy power method. In Conference on Learning Theory, pages 284–309. PMLR, 2016.   
[BL20] Yu Bai and Jason D. Lee. Beyond linearization: On quadratic and higherorder approximation of wide neural networks. In International Conference on Learning Representations, 2020.   
[BLE17] Sébastien Bubeck, Yin Tat Lee, and Ronen Eldan. Kernel-based methods for bandit convex optimization. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing, pages 72–85, 2017.   
$[ \mathrm { C B L ^ { + } } 2 0 ]$ Minshuo Chen, Yu Bai, Jason D Lee, Tuo Zhao, Huan Wang, Caiming Xiong, and Richard Socher. Towards understanding hierarchical learning: Benefits of neural representations. Neural Information Processing Systems (NeurIPS), 2020.   
$[ \mathbf { C L M ^ { + } } 1 6 ]$ T Tony Cai, Xiaodong Li, Zongming Ma, et al. Optimal rates of convergence for noisy sparse phase retrieval via thresholded wirtinger flow. The Annals of Statistics, 44(5):2221–2251, 2016.   
[CLS15] Emmanuel J Candes, Xiaodong Li, and Mahdi Soltanolkotabi. Phase retrieval via wirtinger flow: Theory and algorithms. IEEE Transactions on Information Theory, 61(4):1985–2007, 2015.   
[CM20] Sitan Chen and Raghu Meka. Learning polynomials in few relevant dimensions. In Conference on Learning Theory, pages 1161–1227. PMLR, 2020.   
[DHK08] Varsha Dani, Thomas P Hayes, and Sham M Kakade. Stochastic linear optimization under bandit feedback. The 21st Annual Conference on Learning Theory, 2008.   
$[ \mathrm { D H K } ^ { + } 2 0 ]$ Simon S Du, Wei Hu, Sham M Kakade, Jason D Lee, and Qi Lei. Fewshot learning via learning the representation, provably. arXiv preprint arXiv:2002.09434, 2020.   
[DKL+21] Simon S Du, Sham M Kakade, Jason D Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in RL. arXiv preprint arXiv:2103.10897, 2021.   
[DLL+19] Simon S Du, Jason D Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In International Conference on Machine Learning, pages 1675–1685, 2019.   
[DML21] Alex Damian, Tengyu Ma, and Jason Lee. Label noise sgd provably prefers flat global minimizers. arXiv preprint arXiv:2106.06530, 2021.

[DYM21] Kefan Dong, Jiaqi Yang, and Tengyu Ma. Provable model-based nonlinear bandit and reinforcement learning: Shelve optimism, embrace virtual curvature. arXiv preprint arXiv:2102.04168, 2021.   
[FKM04] Abraham D Flaxman, Adam Tauman Kalai, and H Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. arXiv preprint cs/0408007, 2004.   
[FLYZ20] Cong Fang, Jason D Lee, Pengkun Yang, and Tong Zhang. Modeling from features: a mean-field framework for over-parameterized deep neural networks. arXiv preprint arXiv:2007.01452, 2020.   
[GCL+19] Ruiqi Gao, Tianle Cai, Haochuan Li, Liwei Wang, Cho-Jui Hsieh, and Jason D Lee. Convergence of adversarial training in overparametrized networks. Neural Information Processing Systems (NeurIPS), 2019.   
[GHM15] Dan Garber, Elad Hazan, and Tengyu Ma. Online learning of eigenvectors. In International Conference on Machine Learning, pages 560–568. PMLR, 2015.   
[GLM18] Rong Ge, Jason D Lee, and Tengyu Ma. Learning one-hidden-layer neural networks with landscape design. International Conference on Learning Representations (ICLR), 2018.   
[GMMM19] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. arXiv preprint arXiv:1904.12191, 2019.   
[GMMM21] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. The Annals of Statistics, 49(2):1029–1054, 2021.   
[GMZ16] Aditya Gopalan, Odalric-Ambrym Maillard, and Mohammadi Zaki. Lowrank bandits with latent mixtures. arXiv preprint arXiv:1609.01508, 2016.   
$[ \mathrm { H C J ^ { + } } 2 1 ]$ Jiachen Hu, Xiaoyu Chen, Chi Jin, Lihong Li, and Liwei Wang. Near-optimal representation learning for linear bandits and linear rl, 2021.   
[HL16] Elad Hazan and Yuanzhi Li. An optimal algorithm for bandit convex optimization. arXiv preprint arXiv:1603.04350, 2016.   
[HLSW21] Botao Hao, Tor Lattimore, Csaba Szepesvári, and Mengdi Wang. Online sparse reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 316–324. PMLR, 2021.   
[HLW20] Botao Hao, Tor Lattimore, and Mengdi Wang. High-dimensional sparse linear bandits. arXiv preprint arXiv:2011.04020, 2020.

[HP14] Moritz Hardt and Eric Price. The noisy power method: A meta algorithm with applications. Advances in neural information processing systems, 27:2861–2869, 2014.   
[HWLM20] Jeff Z. HaoChen, Colin Wei, Jason D. Lee, and Tengyu Ma. Shape matters: Understanding the implicit bias of the noise covariance. arXiv preprint arXiv:2006.08680, 2020.   
[HZWS20] Botao Hao, Jie Zhou, Zheng Wen, and Will Wei Sun. Low-rank tensor bandits. arXiv preprint arXiv:2007.15788, 2020.   
[JHG18] Arthur Jacot, Clément Hongler, and Franck Gabriel. Neural tangent kernel: Convergence and generalization in neural networks. In NeurIPS, 2018.   
[JLM21] Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. arXiv preprint arXiv:2102.00815, 2021.   
[JSB16] Nicholas Johnson, Vidyashankar Sivakumar, and Arindam Banerjee. Structured stochastic linear bandits. arXiv preprint arXiv:1606.05693, 2016.   
[JWWN19] Kwang-Sung Jun, Rebecca Willett, Stephen Wright, and Robert Nowak. Bilinear bandits with low-rank structure. In International Conference on Machine Learning, pages 3163–3172. PMLR, 2019.   
$[ \mathsf { K K S } ^ { + } 1 7 ]$ Sumeet Katariya, Branislav Kveton, Csaba Szepesvari, Claire Vernade, and Zheng Wen. Stochastic rank-1 bandits. In Artificial Intelligence and Statistics, pages 392–401. PMLR, 2017.   
[Kle04] Robert Kleinberg. Nearly tight bounds for the continuum-armed bandit problem. Advances in Neural Information Processing Systems, 17:697–704, 2004.   
[KN19] Wojciech Kotłowski and Gergely Neu. Bandit principal component analysis. In Conference On Learning Theory, pages 1994–2024. PMLR, 2019.   
[KTB19] Joe Kileel, Matthew Trager, and Joan Bruna. On the expressive power of deep polynomial neural networks. Advances in Neural Information Processing Systems, 32, 2019.   
[LAAH19] Sahin Lale, Kamyar Azizzadenesheli, Anima Anandkumar, and Babak Hassibi. Stochastic linear bandits with hidden low rank structure. arXiv preprint arXiv:1901.09490, 2019.   
[Lat20] Tor Lattimore. Improved regret for zeroth-order adversarial bandit convex optimisation, 2020.

[LCLS10] Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextualbandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pages 661–670, 2010.   
[LH21] Tor Lattimore and Botao Hao. Bandit phase retrieval. arXiv preprint arXiv:2106.01660, 2021.   
[LL18] Yuanzhi Li and Yingyu Liang. Learning overparameterized neural networks via stochastic gradient descent on structured data. In Advances in Neural Information Processing Systems, pages 8157–8166, 2018.   
[LM13] Guillaume Lecué and Shahar Mendelson. Minimax rate of convergence and the performance of erm in phase recovery. arXiv preprint arXiv:1311.5024, 2013.   
[LMT21] Yangyi Lu, Amirhossein Meisami, and Ambuj Tewari. Low-rank generalized linear bandit problems. In International Conference on Artificial Intelligence and Statistics, pages 460–468. PMLR, 2021.   
[LS20] Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.   
[MGW+20] Edward Moroshko, Suriya Gunasekar, Blake Woodworth, Jason D Lee, Nathan Srebro, and Daniel Soudry. Implicit bias in deep linear classification: Initialization scale vs training accuracy. Neural Information Processing Systems (NeurIPS), 2020.   
[Mil17] James S. Milne. Algebraic geometry (v6.02), 2017. Available at www.jmilne.org/math/.   
[MM15] Cameron Musco and Christopher Musco. Randomized block krylov methods for stronger and faster approximate singular value decomposition. arXiv preprint arXiv:1504.05477, 2015.   
$[ \mathrm { N G L ^ { + } } 1 9 ]$ Mor Shpigel Nacson, Suriya Gunasekar, Jason Lee, Nathan Srebro, and Daniel Soudry. Lexicographic and depth-sensitive margins in homogeneous and non-homogeneous deep models. In International Conference on Machine Learning, pages 4683–4692. PMLR, 2019.   
[RT10] Paat Rusmevichientong and John N Tsitsiklis. Linearly parameterized bandits. Mathematics of Operations Research, 35(2):395–411, 2010.   
[RTS18] Carlos Riquelme, George Tucker, and Jasper Snoek. Deep bayesian bandits showdown: An empirical comparison of bayesian deep networks for thompson sampling, 2018.

[RVR13] Dan Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. In Advances in Neural Information Processing Systems, pages 2256–2264, 2013.   
[SBRL19] Maziar Sanjabi, Sina Baharlouei, Meisam Razaviyayn, and Jason D Lee. When does non-orthogonal tensor decomposition have no spurious local minima? arXiv preprint arXiv:1911.09815, 2019.   
[Sha13] Igor R Shafarevich. Basic Algebraic Geometry 1: Varieties in Projective Space. Springer Science & Business Media, 2013.   
[Tro12] Joel A Tropp. User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics, 12(4):389–434, 2012.   
$[ \mathrm { V K M ^ { + } } 1 3 ]$ Michal Valko, Nathaniel Korda, Rémi Munos, Ilias Flaounas, and Nelo Cristianini. Finite-time analysis of kernelised contextual bandits. arXiv preprint arXiv:1309.6869, 2013.   
[WA16] Yining Wang and Animashree Anandkumar. Online and differentially-private tensor decomposition. arXiv preprint arXiv:1606.06237, 2016.   
[WGL+19] Blake Woodworth, Suriya Genesekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Kernel and deep regimes in overparametrized models. In Conference on Learning Theory (COLT), 2019.   
[WLLM19] Colin Wei, Jason D Lee, Qiang Liu, and Tengyu Ma. Regularization matters: Generalization and optimization of neural nets vs their induced kernel. In Advances in Neural Information Processing Systems, pages 9709–9721, 2019.   
[WSY20] Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems, 33:6123–6135, 2020.   
[WWL+20] Xiang Wang, Chenwei Wu, Jason D Lee, Tengyu Ma, and Rong Ge. Beyond lazy training for over-parameterized tensor decomposition. Neural Information Processing Systems (NeurIPS), 2020.   
[WX19] Yang Wang and Zhiqiang Xu. Generalized phase retrieval: measurement number, matrix recovery and beyond. Applied and Computational Harmonic Analysis, 47(2):423–446, 2019.   
[XWZG20] Pan Xu, Zheng Wen, Handong Zhao, and Quanquan Gu. Neural contextual bandits with deep representation and shallow exploration. arXiv preprint arXiv:2012.01780, 2020.

[YHLD20] Jiaqi Yang, Wei Hu, Jason D Lee, and Simon S Du. Provable benefits of representation learning in linear bandits. arXiv preprint arXiv:2010.06531, 2020.   
[ZLKB20] Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent Bellman error. In International Conference on Machine Learning, 2020.

# A Additional Preliminaries

In this section we show that adapting the eluder UCB algorithms from [RVR13] would yield the sample complexity in Theorem 2.1. Especially we give the rates in Table 1 for our stochastic settings.

Algorithm 4 Eluder UCB   
1: Input: Function class $\mathcal{F}$ , failure probability $\delta$ , parameters $\alpha, N, C$ .  
2: Initialization: $\mathcal{F}_0 \gets \mathcal{F}$ .  
3: for $t$ from 1 to $T$ do  
4: Select Action:  
5: $a_t \in \arg \max_{a \in \mathcal{A}} \sup_{f_\theta \in \mathcal{F}_{t-1}} f_\theta(a)$ 6: Play action $a_t$ and observe reward $r_t$ 7: Update Statistics:  
8: $\widehat{\pmb{\theta}}_t \in \arg \min_{\pmb{\theta}} \sum_{s=1}^t (f_\theta(\pmb{a}_s) - r_s)^2$ 9: $\beta_t \gets 8 \log(N/\delta) + 2\alpha t(8C + \sqrt{8 \ln(4t^2/\delta)})$ 10: $\mathcal{F}_t \gets \{f_\theta : \sum_{s=1}^t (f_\theta - f_{\widehat{\theta}_t})^2(\pmb{a}_s) \leq \beta_t\}$

The algorithm [RVR13] consider Algorithm 4 for the stochastic generalized linear bandit problem. Assume that $\pmb { \theta } ^ { * }$ is the true parameter of the reward model. The reward is $r _ { t } = f _ { \theta ^ { * } } ( \pmb { a } _ { t } ) + \eta _ { t }$ for $f _ { \theta ^ { \ast } } \in \mathcal { F }$ . Let $N$ be the $\alpha$ -covering-number (under $\| \cdot \| _ { \infty } )$ of $\mathcal { F }$ , $d _ { E }$ be the $\alpha$ -eluder-dimension of $\mathcal { F }$ (see Definition 3,4 in [RVR13]). Let $C = \operatorname* { s u p } _ { \substack { c \sim r \sim c } } | f ( a ) |$ . We f∈F,a∈A set $\begin{array} { r } { \alpha = \frac { 1 } { T ^ { 2 } } } \end{array}$ in the algorithm.

The regret analysis Choosing $\alpha = 1 / T ^ { 2 }$ , proposition 4 in [RVR13] state that with probability $1 - \delta$ , for some universal constant $C$ , the total regret $\Re ( T ) \leq { \frac { 1 } { T } } + C \operatorname* { m i n } \{ d _ { E } , T \} +$ $4 \sqrt { d _ { E } \beta _ { T } T } \le 1 + C \sqrt { d _ { E } T } + 4 \sqrt { d _ { E } \beta _ { T } T } = O ( \sqrt { d _ { E } ( 1 + \beta _ { T } ) T } ) .$ . In our settings with $\alpha =$ $1 / T ^ { 2 }$ $\cdot \cdot \cdot \cdot \langle T ^ { 2 } , \beta _ { T } = 8 \log ( N / \delta ) + 2 ( 8 C + \sqrt { 8 \ln ( 4 T ^ { 2 } / \delta ) } ) / T = O ( \log ( N / \delta ) )$ ) where $\log ( N ) = \Omega ( 1 )$ for our action sets, and thus

$$
\mathfrak {R} (T) = \widetilde {O} (\sqrt {d _ {E} T \log N}).
$$

Applications in our settings We show that in our settings Theorem 2.1 will obtain the rates listed in Table 1.

# The covering numbers

Lemma A.1. The log-covering-number (of radius $\alpha$ with $\alpha \ll 1$ , under $\| \cdot \| _ { \infty } )$ of the function classes are: $\begin{array} { r } { \log N ( \mathcal { F } _ { S Y M } ) = O ( d k \log \frac { k } { \alpha } ) } \end{array}$ , $\begin{array} { r } { \log N ( \mathcal { F } _ { A S Y M } ) = O ( d k \log \frac { k } { \alpha } ) } \end{array}$ , $\log N ( \mathcal { F } _ { E V } ) =$ $\begin{array} { r } { O ( d k \log { \frac { k } { \alpha } } ) } \end{array}$ , and $\begin{array} { r } { \log N ( \mathcal { F } _ { L R } ) = O ( d k \log \frac { \bar { k } } { \alpha } ) } \end{array}$ .

Proof. Let $S _ { \xi } ^ { d }$ denote a minimal $\xi$ -covering of $\mathbb { S } ^ { d - 1 }$ (under $\left\| \cdot \right\| _ { 2 } )$ for $0 < \xi < \frac { 1 } { 1 0 }$ , and $| S _ { \xi } ^ { d } | = O ( d \log 1 / \xi )$ (see for example [RVR13]). Then we can construct the coverings in our settings from $S _ { \xi } ^ { d }$ :

• $\mathcal { F } _ { \mathrm { S Y M } }$ : let $\xi ~ = ~ { \frac { \alpha } { k p } }$ , and for $k$ copies of $S _ { \xi } ^ { d }$ , we can construct a covering of $\mathcal { F } _ { \mathrm { S Y M } }$ with size $| S _ { \xi } ^ { d } | ^ { k }$ . Specifically, let the covering be $\begin{array} { r } { S _ { \mathrm { { S Y M } } } = \{ g ( \pmb { a } ) = \sum _ { j = 1 } ^ { k } \lambda _ { j } ( \pmb { u } _ { j } ^ { \top } \pmb { a } ) ^ { p } } \end{array}$ $( { \pmb u } _ { 1 } , { \pmb u } _ { 2 } , \cdot \cdot \cdot , { \pmb u } _ { k } ) \in S _ { \xi } ^ { d } \times S _ { \xi } ^ { d } \times \cdot \cdot \cdot \times S _ { \xi } ^ { d } \}$ , then for each $\begin{array} { r } { f ( \pmb { a } ) = \sum _ { j = 1 } ^ { k } \lambda _ { j } ( \pmb { v } _ { j } ^ { \top } \pmb { a } ) ^ { p } \in } \end{array}$ $\mathcal { F } _ { \mathrm { S Y M } }$ , as we can find ${ \pmb u } _ { j } \in S _ { \xi } ^ { d }$ that $\| \pmb { u } _ { j } - \pmb { v } _ { j } \| _ { 2 } \le \xi$ ,

$$
\sup  _ {\boldsymbol {a}} [ f (\boldsymbol {a}) - g (\boldsymbol {a}) ] \leq \sup  _ {\boldsymbol {a}} [ \sum_ {j = 1} ^ {k} | \lambda_ {j} | | \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {a} - \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a} | | \sum_ {q = 0} ^ {p - 1} (\boldsymbol {u} _ {j} ^ {\top} \boldsymbol {a}) ^ {q} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {p - q - 1} | ] \leq p k \xi = \alpha ;
$$

• $\mathcal { F } _ { \mathrm { A S Y M } }$ : let $\begin{array} { r } { \xi = \frac { \alpha } { k p } } \end{array}$ , and for $k p$ copies of $S _ { \xi } ^ { d }$ , let the covering be $S _ { \mathrm { A S Y M } } = \{ g ( \pmb { a } ) =$ $\begin{array} { r } { \underset { - j = 1 } { \overset {  } {  } } \lambda _ { j } \prod _ { q = 1 } ^ { p } ( \boldsymbol { u } _ { j } ( q ) ^ { \top } \boldsymbol { a } ( q ) ) : ( \boldsymbol { u } _ { 1 } ( 1 ) , \boldsymbol { u } _ { 1 } ( 2 ) , \cdot \cdot \cdot , \boldsymbol { u } _ { 1 } ( p ) , \boldsymbol { u } _ { 2 } ( 1 ) , \cdot \cdot \cdot , \boldsymbol { u } _ { k } ( p ) ) \in S _ { \xi } ^ { d } \times } \end{array}$ $S _ { \xi } ^ { d } { \times } \cdots { \times } S _ { \xi } ^ { d }  \}$ with size $| S _ { \xi } ^ { d } | ^ { k p }$ . Then for each $\begin{array} { r } { f ( \pmb { a } ) = \sum _ { j = 1 } ^ { k } \lambda _ { j } \prod _ { q = 1 } ^ { p } ( \pmb { v } _ { j } ( q ) ^ { \top } \pmb { a } ( q ) ) \in } \end{array}$ $\mathcal { F } _ { \mathrm { A S Y M } }$ , as we can find ${ \pmb u } _ { j } ( q ) \in S _ { \xi } ^ { d }$ that $\begin{array} { r } { \| \pmb { u } _ { j } ( \boldsymbol { q } ) - \pmb { v } _ { j } ( \boldsymbol { q } ) \| _ { 2 } \le \xi } \end{array}$ ,

$$
\begin{array}{l} \sup  _ {\boldsymbol {a}} [ f (\boldsymbol {a}) - g (\boldsymbol {a}) ] \leq \sup  _ {\boldsymbol {a}} \left[ \sum_ {j = 1} ^ {k} | \lambda_ {j} | \sum_ {q = 1} ^ {p} | \boldsymbol {u} _ {j} (q) ^ {\top} \boldsymbol {a} - \boldsymbol {v} _ {j} (q) ^ {\top} \boldsymbol {a} \right| \cdot \\ | \prod_ {r <   q} (\boldsymbol {u} _ {j} (r) ^ {\top} \boldsymbol {a}) \prod_ {r > q} (\boldsymbol {v} _ {j} (r) ^ {\top} \boldsymbol {a}) | ] \\ \leq p k \xi = \alpha ; \\ \end{array}
$$

• $\mathcal { F } _ { \mathrm { E V } }$ : the construction follows that of $\mathcal { F } _ { \mathrm { S Y M } }$ by taking $p = 2$   
• $\mathcal { F } _ { \mathrm { L R } }$ : taking the construction of $\mathcal { F } _ { \mathrm { S Y M } }$ with $p = 2$ and $\xi = { \frac { \alpha } { 2 k } }$ , for $\begin{array} { r } { \pmb { N } = \sum _ { j = 1 } ^ { k } \lambda _ { j } \pmb { u } _ { j } \pmb { u } _ { j } ^ { \top } } \end{array}$ and $\begin{array} { r } { \pmb { M } = \sum _ { j = 1 } ^ { k } \lambda _ { j } \pmb { v } _ { j } \pmb { v } _ { j } ^ { \top } } \end{array}$ with $\| \pmb { u } _ { j } - \pmb { v } _ { j } \| _ { 2 } \le \xi$ , we know $\begin{array} { r } { \| \pmb { N } - \pmb { M } \| _ { \mathrm { F } } \leq \left\| \pmb { N } - \sum _ { j = 1 } ^ { k } \lambda _ { j } \pmb { u } _ { j } \pmb { v } _ { j } ^ { \top } \right\| _ { \mathrm { F } } + } \end{array}$ $\begin{array} { r } { \left\| \sum _ { j = 1 } ^ { k } \lambda _ { j } \pmb { u } _ { j } \pmb { v } _ { j } ^ { \top } - \pmb { M } \right\| _ { \mathrm { F } } \leq \sum _ { j = 1 } ^ { k } 2 | \lambda _ { j } | \xi \leq \alpha . } \end{array}$ . Then $\mathrm { s u p } _ { A } [ f _ { M } ( A ) - f _ { N } ( A ) ] \leq$ $\operatorname* { s u p } _ { A } \left\| M - N \right\| _ { \mathrm { F } } \cdot \left\| \bar { A ^ { } } \right\| _ { \mathrm { F } } \leq \alpha .$ .

Then we can bound the covering numbers in Theorem 2.1. Notice that in the settings the log-covering numbers are only different by constant factors. □

# The eluder dimensions

Lemma A.2. The ǫ-eluder-dimension $\epsilon < 1$ ) $d _ { E }$ of the function classes are: $d _ { E } ( \mathcal { F } _ { S Y M } ) =$ $\widetilde { \Theta } ( d ^ { p } )$ (for $k \geq p ,$ ), $d _ { E } ( \mathcal { F } _ { A S Y M } ) = \widetilde { \Theta } ( d ^ { p } )$ , $d _ { E } ( \mathcal { F } _ { E V } ) = \widetilde { \Theta } ( d ^ { 2 } )$ , and $d _ { E } ( \mathcal { F } _ { L R } ) = \widetilde { \Theta } ( d ^ { 2 } )$ . In the settings WLOG we assume the top eigenvalue is $r ^ { * } = \lambda _ { 1 } = 1$ as we are mostly interested in the cases where $r ^ { * } > \epsilon$ .

Proof. The upper bounds for the eluder dimension can be given by the linear argument. [RVR13] show that the $d$ -dimension linear model $\{ f _ { \pmb \theta } ( \pmb a ) = \pmb \theta ^ { \top } \pmb a \}$ has $\epsilon$ -eluder-dimension

$O ( d \log \frac { 1 } { \epsilon } )$ . In all of these settings, we can find feature maps $\phi$ and $\psi$ so that ${ \mathcal F } =$ $\{ f _ { \theta } ( \pmb { a } ) , \mathsf { \check { f } } _ { \pmb { \theta } } ( \pmb { a } ) = \phi ( \pmb { \theta } ) ^ { \top } \psi ( \pmb { a } ) , \lVert \phi ( \pmb { \theta } ) \rVert _ { 2 } \le k , \lVert \psi ( \pmb { a } ) \rVert _ { 2 } \le k \}$ . Then the eluder dimensions will be bounded by the corresponding linear dimension as an original $\epsilon$ -independent sequence $\left\{ { { a } _ { i } } \right\}$ will induce an $\epsilon$ -independent sequence $\{ \psi ( { \pmb a } _ { i } ) \}$ in the linear model. Therefore for matrices ( $\mathcal { F } _ { \mathrm { L R } }$ and $\mathcal { F } _ { \mathrm { E V , } }$ ) the eluder dimension is $O ( d ^ { 2 } \log \frac { k } { \epsilon } )$ and for the tensors $( \mathcal { F } _ { \mathrm { S Y M } }$ and $\mathcal { F } _ { \mathrm { A S Y M } } )$ it is $O ( d ^ { p } \log \frac { k } { \epsilon } )$ .

Then we consider the lower bounds. We provide the following example of $O ( 1 )$ - independent sequences to bound the eluder dimension in our settings up to a log factor.

• $\mathcal { F } _ { \mathrm { S Y M } }$ : the sequence is $\{ \pmb { a } _ { i } = ( \pmb { e } _ { i _ { 1 } } , \pmb { e } _ { i _ { 2 } } , \cdot \cdot \cdot , \pmb { e } _ { i _ { p } } ) : i = ( i _ { 1 } , i _ { 2 } , \cdot \cdot \cdot , i _ { p } ) \in [ d ] ^ { p } \}$ . For $\begin{array} { r } { f _ { j } ( \pmb { a } ) = \prod _ { q = 1 } ^ { p } e _ { j _ { q } } ^ { \top } \pmb { a } ( q ) , f _ { j } ( \pmb { a } _ { i } ) } \end{array}$ is only 1 when $i = j$ and 0 otherwise. Then each $\mathbf { a } _ { i }$ is 1-independent to the predecessors on $f _ { i }$ and zero, and thus the eluder dimension is lower bounded by $d ^ { p }$ .   
• $\mathcal { F } _ { \mathrm { A S Y M } }$ : for $p \leq d$ and $k \geq p$ , the sequence is $\begin{array} { r } { \{ \pmb { a } _ { i } = \frac { 1 } { \sqrt { p } } ( \pmb { e } _ { i _ { 1 } } + \pmb { e } _ { i _ { 2 } } + \cdot \cdot \cdot + \pmb { e } _ { i _ { p } } ) : i = } \end{array}$ $( i _ { 1 } , i _ { 2 } , \cdot \cdot \cdot , i _ { p } ) \in [ d ] ^ { p } , i _ { 1 } < i _ { 2 } < \cdot \cdot \cdot < i _ { p } \}$ . There are tensors $f _ { j }$ and $g _ { j }$ of CP-rank $k$ that $\begin{array} { r } { ( f _ { j } - g _ { j } ) ( \pmb { a } ) = \prod _ { q = 1 } ^ { p } ( \pmb { e } _ { j _ { q } } ^ { \top } \pmb { a } ) } \end{array}$ where $j _ { 1 } < j _ { 2 } < \cdots < j _ { p }$ , $( f _ { j } - g _ { j } ) ( { \pmb a } _ { i } )$ is only 1 when $i = j$ and 0 otherwise. Then each $\mathbf { a } _ { i }$ is 1-independent to the predecessors on $f _ { i }$ and $g _ { i }$ , and thus the eluder dimension is lower bounded by $\binom { d } { p }$ .   
• $\mathcal { F } _ { \mathrm { E V } }$ : the sequence is $\begin{array} { r } { \{ \pmb { a } _ { i } = \frac { 1 } { \sqrt { 2 } } ( \pmb { e } _ { i _ { 1 } } + \pmb { e } _ { i _ { 2 } } ) : i = ( i _ { 1 } , i _ { 2 } ) \in [ d ] ^ { 2 } , i _ { 1 } \le i _ { 2 } \} . } \end{array}$ . For $\begin{array} { r } { f _ { j } ( \pmb { a } ) = \frac { 1 } { 2 } \pmb { a } ^ { \top } ( \pmb { e } _ { j _ { 1 } } + \pmb { e } _ { j _ { 2 } } ) ( \pmb { e } _ { j _ { 1 } } + \pmb { \dot { e } } _ { j _ { 2 } } ) ^ { \top } \pmb { a } } \end{array}$ and $\begin{array} { r } { g _ { j } ( \pmb { a } ) = \frac { 1 } { 2 } \pmb { a } ^ { \top } ( \pmb { e } _ { j _ { 1 } } - \pmb { e } _ { j _ { 2 } } ) ( \pmb { e } _ { j _ { 1 } } - \pmb { e } _ { j _ { 2 } } ) ^ { \top } \pmb { a } } \end{array}$ with $j _ { 1 } \ \le \ j _ { 2 }$ , $( f _ { j } - g _ { j } ) ( { \pmb a } _ { i } )$ is only 1 when $i = j$ and 0 otherwise. Then each $\mathbf { a } _ { i }$ is 1-independent to the predecessors on $f _ { i }$ and $g _ { i }$ , and thus the eluder dimension is lower bounded by $\binom { d } { 2 }$ .   
• $\mathcal { F } _ { \mathrm { L R } }$ : the sequence is $\begin{array} { r } { \{ A _ { i } = \frac { 1 } { 2 } e _ { i _ { 1 } } e _ { i _ { 2 } } ^ { T } + e _ { i _ { 1 } } e _ { i _ { 2 } } ^ { T } : i = ( i _ { 1 } , i _ { 2 } ) \in [ d ] ^ { 2 } , i _ { 1 } \le i _ { 2 } \} } \end{array}$ . For $f _ { j } ( A ) = \langle { \textstyle { \frac { 1 } { 2 } } } ( e _ { j _ { 1 } } e _ { j _ { 2 } } ^ { T } + e _ { j _ { 2 } } e _ { j _ { 1 } } ^ { T } ) , { \bf \bar { A } } \rangle$ with $j _ { 1 } \le j _ { 2 }$ , $f _ { j } ( A _ { i } )$ is only 1 when $i = j$ and 0 otherwise. Then each $A _ { i }$ is 1-independent to the predecessors on $f _ { i }$ and zero, and thus the eluder dimension is lower bounded by $\binom { d } { 2 }$ .

![](images/f50a886a6467f78569b25d338fbe94ff42fe232df4003d25baebce1277e5e0a6.jpg)

Then we are all set for the results in the first line of 1. Notice that when we choose $\alpha = O ( 1 / T ^ { 2 } )$ and $\epsilon = O ( 1 / T ^ { 2 } )$ in our analysis of Algorithm 4, the regret upper bound would only expand by $\log ( T )$ factors.

# B Omitted Proofs for Quadratic Reward

In this section we include all the omitted proof of the theorems presented in the main paper.

# B.1 Omitted Proofs of Main Results for Stochastic Bandit Eigenvector Problem

Proof of Theorem 3.3. Notice in Algorithm 1, for each iterate $^ { a }$ , its next iterate $\textbf {  { y } }$ satisfies

$$
\begin{array}{l} \boldsymbol {y} = \frac {1}{n _ {s}} \sum_ {i = 1} ^ {n _ {s}} \left(\boldsymbol {a} / 2 + \boldsymbol {z} _ {i} / 2\right) ^ {\top} \boldsymbol {M} \left(\boldsymbol {a} / 2 + \boldsymbol {z} _ {i} / 2\right) \boldsymbol {z} _ {i} + \eta_ {i} \boldsymbol {z} _ {i} \\ = \frac {m _ {s}}{n _ {s}} \sum_ {i = 1} ^ {n _ {s}} \left(\frac {1}{4} \boldsymbol {a} ^ {\top} \boldsymbol {M} \boldsymbol {a} + \frac {1}{2} \boldsymbol {a} ^ {\top} \boldsymbol {M} \boldsymbol {z} _ {i} + \eta_ {i}\right) \boldsymbol {z} _ {i}. \\ \end{array}
$$

Therefore $\mathbb { E } [ { \pmb y } ] = \frac { 1 } { 2 } M { \pmb a }$ . We can write $2 y = M a + g$ where $\begin{array} { r } { \pmb { g } : = \frac { m _ { s } } { n _ { s } } \sum _ { i = 1 } ^ { n _ { s } } ( \frac { 1 } { 2 } \pmb { a } ^ { \top } \pmb { M } \pmb { a } + } \end{array}$ ns $2 \eta _ { i } ) z _ { i }$ . With Claim C.8 and Claim C.7 we get that $\begin{array} { r } { \| \pmb { g } \| \le C \sqrt { \frac { m _ { s } \log ^ { 2 } ( n / \delta ) \log ( d / \delta ) d } { n _ { s } } } } \end{array}$ . Therefore with our choice of $\begin{array} { r } { n _ { s } \ge \widetilde \Theta \big ( \frac { d ^ { 2 } } { \varepsilon _ { s } ^ { 2 } ( \lambda _ { 1 } - | \lambda _ { 2 } | ) ^ { 2 } } \big ) } \end{array}$ λ2 )2 ) we guarantee $\| { \pmb g } \| \le \varepsilon _ { s } ( \lambda _ { 1 } - | \lambda _ { 2 } | )$ . Therefore it satisfies the requirements for noisy power method, and by applying Corollary B.4, we have with $L = O ( \kappa \log ( d / \varepsilon ) )$ iterations we will be able to find $\| { \widehat { \pmb a } } - { \pmb a } ^ { * } \| \leq \varepsilon$ . By setting $\delta <$ $0 . 1 / L$ in the algorithm we can guarantee the whole process succeed with high probability. Altogether it is sufficient to take $L n _ { s } = { \widetilde O } ( \kappa d ^ { 2 } / ( \varepsilon \Delta ) ^ { 2 } )$ actions to get an $\varepsilon$ -optimal arm.

Finally to get the cumulative regret bound, we apply Claim C.3 with $\begin{array} { r } { A = \frac { d ^ { 2 } \kappa } { \Delta ^ { 2 } } } \end{array}$ and $a = 2$ Therefore we set $\begin{array} { r } { \varepsilon = A ^ { 1 / 4 } T ^ { - 1 / 4 } = \frac { d ^ { 1 / 2 } \kappa ^ { 1 / 4 } } { \Delta ^ { 1 / 2 } T ^ { 1 / 4 } } } \end{array}$ ∆1/2T 1/4 and get: d1/2κ1/4

$$
\operatorname {R e g} (T) \lesssim T ^ {1 / 2} A ^ {1 / 2} r ^ {*} = \sqrt {\frac {d ^ {2} \kappa}{\Delta^ {2}} T} r ^ {*} = \sqrt {d ^ {2} \kappa^ {3} T}.
$$

![](images/01ee482a1df0c38d17beb930075e68265cca3f8cb92537f8fe1972cb71c451be.jpg)

Corollary B.1 (Formal statement for Corollary 3.6). In Algorithm 1, by setting $\alpha = 1 -$ $\varepsilon ^ { 2 } / 2$ , one can get $\varepsilon$ -optimal reward with a total of $\widetilde { O } ( d ^ { 2 } \lambda _ { 1 } ^ { 2 } / \bar { \varepsilon } ^ { 4 } )$ total samples to get a such that $r ^ { * } - f ( \pmb { a } ) \leq \varepsilon$ . Therefore one can get an accumulative regret of $\widetilde { O } ( \lambda _ { 1 } ^ { 3 / 5 } d ^ { 2 / 5 } T ^ { 4 / 5 } )$ .

Proof of Lemma 3.6. In order to find an arm with $\lambda _ { 1 } \varepsilon ^ { 2 }$ -optimal reward, one will want to recover an arm that is $\varepsilon / 2$ -close (meaning to find an $^ { a }$ such that $\tan \theta ( V _ { l } , \boldsymbol { a } ) \leq \varepsilon / 2 )$ to the top eigenspace span $( \pmb { v } _ { 1 } , \cdot \cdot \cdot \pmb { v } _ { l } )$ , where $l$ satisfies $\lambda _ { l } \geq \lambda _ { 1 } - \widetilde { \varepsilon }$ and $\lambda _ { l + 1 } \leq \lambda _ { 1 } - \widetilde { \varepsilon }$ . Here we set $\widetilde { \varepsilon } : = \lambda _ { 1 } \varepsilon ^ { 2 } / 2$ . We first show 1) this is sufficient to get an $\lambda _ { 1 } \varepsilon$ -optimal reward, and next show 2) how to set parameter to achieve this.

To get 1), we write $V _ { l } = [ \pmb { v } _ { 1 } , \cdot \cdot \cdot \pmb { v } _ { l } ] \in \mathbb { R } ^ { d \times l }$ and $V _ { l } ^ { \perp } = [ \pmb { v } _ { l + 1 } , \cdot \cdot \cdot \pmb { v } _ { k } ]$ . When tan $\mathbf { \nabla } \theta ( { \bf V } _ { l } , { \pmb a } _ { T } ) =$ $\| V ^ { \perp } \pmb { a } \| / \| V \pmb { a } \| \le \varepsilon / 2$ , from the proof of Claim C.2, we get $r ^ { * } - f ( { \pmb a } ) \leq \mathrm { m i n } \{ \lambda _ { 1 } , \lambda _ { 1 } 2 ( \varepsilon / 2 ) ^ { 2 } +$ $\widetilde { \varepsilon } \} = \lambda _ { 1 } \varepsilon ^ { 2 }$ .

Now to get 2), we note that in each iteration we try to conduct the power iteration to find an action $\tan \theta ( V _ { l } , \widehat { \mathbf { a } } ) \leq \varepsilon / 2$ and with eigengap $\ge \widetilde { \varepsilon } : = \lambda _ { 1 } \varepsilon ^ { 2 } / 2$ . Therefore it is sufficient to let $\| \pmb { g } \| \le 0 . 1 \widetilde { \varepsilon } \varepsilon$ and $| \pmb { v } _ { 1 } ^ { \top } \pmb { g } | \le 0 . 1 \widetilde { \epsilon } \frac { 1 } { \sqrt { d } }$ , and thus $\begin{array} { r } { n _ { s } \geq \widetilde \Theta ( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \widetilde \varepsilon ^ { 2 } } ) \leq \widetilde \Theta ( d ^ { 2 } / \lambda _ { 1 } ^ { 2 } \varepsilon ^ { 6 } ) } \end{array}$ . Together we need $\lambda _ { 1 } / \widetilde { \epsilon } \log ( 2 d / \varepsilon ) n _ { s } = \widetilde { \Theta } ( d ^ { 2 } / \lambda _ { 1 } ^ { 2 } \varepsilon ^ { 8 } )$ samples to get an $\lambda _ { 1 } \varepsilon ^ { 2 }$ -optimal reward. Namely we get ${ \widetilde { \varepsilon } } .$ -optimal reward with $\widetilde { O } ( d ^ { 2 } \lambda _ { 1 } ^ { 2 } / \widetilde { \varepsilon } ^ { 4 } )$ samples.

Finally by applying Claim C.3 we get:

$$
\Re (T) \lesssim (d ^ {2} \lambda_ {1} ^ {2}) ^ {\frac {1}{5}} T ^ {\frac {4}{5}} \lambda_ {1} ^ {\frac {1}{5}} \leq \widetilde {O} (\lambda_ {1} ^ {3 / 5} d ^ {2 / 5} T ^ {4 / 5}).
$$

![](images/e7d059a02f7c1950271ae3a06aa29fe20f7720feb2eefad769bbea315459f8b8.jpg)

# Algorithm 5 Gap-free Subspace Iteration for Bilinear Bandit

1: Input: Quadratic reward $f : \mathcal { X }  \mathbb { R }$ generating noisy reward, failure probability $\delta$ error $\varepsilon$ .   
2: Initialization: Set $k ^ { \prime } = 2 k$ . Initial candidate matrix $\pmb { X } _ { 0 } \in \mathbb { R } ^ { d \times k ^ { \prime } }$ , $\pmb { X } _ { 0 } ( j ) \in \mathbb { R } ^ { d } , j =$ $1 , 2 , \cdots k ^ { \prime }$ is the $j$ -th column of $X _ { 0 }$ and are i.i.d sampled on the unit sphere $\mathbb { S } ^ { d - 1 }$ uniformly. Sample variance $m$ , # sample per iteration $n$ , total iteration $L$ .   
3: for Iteration $l$ from 1 to $L$ do   
4: for $s$ from 1 to $k ^ { \prime }$ do   
5: Noisy subspace iteration:   
6: Sample $z _ { i } \sim \mathcal { N } ( 0 , 1 / m I _ { d } ) , i = 1 , 2 , \cdot \cdot \cdot n _ { s }$   
7: Calculate tentative rank-1 arms $\begin{array} { r } { \widetilde { \pmb { a } } _ { i } = \frac { 1 } { 2 } ( { \pmb X } _ { l - 1 } ( s ) + { \pmb z } _ { i } ) } \end{array}$ .   
8: Conduct estimation $\begin{array} { r } { \pmb { Y } _ { l } ( s )  4 m / n \sum _ { i = 1 } ^ { n } ( f ( \widetilde { \pmb { a } } _ { i } ) + \eta _ { i } ) z _ { i } } \end{array}$ . (Yl ∈ Rd×k′ )   
9: Let $Y _ { l } = X _ { l } R _ { l }$ be a QR-factorization of $\mathbf { { \cal Y } } _ { l }$   
10: Update target arm $\begin{array} { r } { \pmb { a } _ { l }  \mathrm { a r g m a x } _ { \parallel \pmb { a } \parallel = 1 } \pmb { a } ^ { \top } \pmb { Y } _ { l } \pmb { X } _ { l - 1 } ^ { \top } \pmb { a } . } \end{array}$   
11: Output: ${ \pmb a } _ { L }$

Theorem B.2 (Formal statement of Theorem 3.7). In Algorithm 5, if we set $\begin{array} { r } { n = \widetilde { \Theta } ( \frac { d ^ { 2 } \lambda _ { 1 } ^ { 2 } } { \varepsilon ^ { 2 } \lambda _ { k } ^ { 2 } } ) , m = } \end{array}$ (  ε2λ2 ), m = d2 λ21 $d \log ( n / \delta ) , L = \Theta ( \log ( d / \varepsilon ) ) , \delta = 0 . 1 / L$ , we will be able to identify an action $\widehat { \mathbf { a } }$ that yield at most $\varepsilon$ -regret with probability 0.9. Therefore by applying the standard PAC to regret conversion as discussed in Claim C.3 we get a cumulative regret of $\widetilde { \cal O } ( \lambda _ { 1 } ^ { 1 / 3 } k ^ { 1 / 3 } ( \widetilde { \kappa } d T ) ^ { \top 2 / 3 } )$ for large enough $T$ , where $\widetilde { \kappa } = \lambda _ { 1 } / | \lambda _ { k } |$ .

eOn the other hand, we set $\begin{array} { r } { n = \widetilde { \Theta } \big ( \frac { d ^ { 2 } k ^ { 2 } } { \varepsilon ^ { 2 } } \big ) } \end{array}$ and keep the other parameters. If we play Algorithm $5 ~ k$ times by setting $k ^ { \prime } = 2 , 4 , 6 , \cdot \cdot \cdot 2 k$ and select the best output among them, we can get a gap-free cumulative regret of $\widetilde { \cal O } ( \lambda _ { 1 } ^ { 1 / 3 } k ^ { 4 / 3 } ( d T ) ^ { 2 / 3 } )$ for large enough $T$ with high probability.

Proof of Theorem 3.7. First we show the first setting identify an $\varepsilon$ -optimal reward with $\widetilde { \cal O } ( \widetilde { \kappa } ^ { 2 } d ^ { 2 } k \epsilon ^ { - 2 } )$ samples.

Similarly as Theorem 3.8, when setting $n \geq \widetilde { \Theta } ( d ^ { 2 } / ( \sigma _ { k } ^ { 2 } \widetilde { \epsilon } ^ { 2 } ) )$ , we can find $X _ { L }$ that satisfies $\| ( \pmb { X } _ { L } \pmb { X } _ { L } ^ { \top } - I ) \pmb { U } \| \le \widetilde \epsilon$ , and therefore we recover an $Y _ { L } = M X _ { L - 1 } + G _ { L } $ with $\| G _ { L } \| \leq$ $\sigma _ { k } \widetilde { \varepsilon }$ and $\begin{array} { r } { \| \mathbf { Y } _ { L } \mathbf { X } _ { L - 1 } ^ { \top } - \mathbf { M } \| _ { 2 } = \| M \mathbf { X } _ { L - 1 } \mathbf { X } _ { L - 1 } ^ { \top } - M + G _ { L } \mathbf { X } _ { L - 1 } ^ { \top } \| _ { 2 } \leq ( \lambda _ { 1 } + | \lambda _ { k } | ) \widehat { \varepsilon } } \end{array}$ .  Therefore by definition o $\begin{array} { r } { \mathsf { f } a _ { L } , \boldsymbol { a } _ { L } ^ { \top } Y _ { L } \boldsymbol { X } _ { L - 1 } ^ { \top } \boldsymbol { a } _ { L } = \operatorname* { m a x } _ { \| \boldsymbol { a } \| = 1 } \boldsymbol { a } ^ { \top } ( M \boldsymbol { X } _ { L - 1 } \boldsymbol { X } _ { L - 1 } ^ { \top } + G _ { L } \boldsymbol { X } _ { L - 1 } ^ { \top } ) \boldsymbol { a } \ge \lambda _ { 1 } - \frac { 1 } { L } \| \boldsymbol { A } _ { L - 1 } \| ^ { 2 } , } \end{array}$ $( \lambda _ { 1 } + | \lambda _ { k } | ) \widetilde { \varepsilon }$ . Therefore $\begin{array} { r } { \pmb { a } _ { L } ^ { \top } \pmb { M } \pmb { a } _ { L } \geq \lambda _ { 1 } - 2 ( \lambda _ { 1 } + | \lambda _ { k } | ) \hat { \varepsilon } } \end{array}$ . Therefore we set $2 ( \lambda _ { 1 } + | \lambda _ { k } | ) \widetilde { \varepsilon } = \epsilon$

i.e., $\widetilde { \varepsilon } = 0 . 5 \epsilon / ( \lambda _ { 1 } + \vert \lambda _ { k } \vert )$ which will get a total sample of $T = \widetilde \Theta ( k n ) = \widetilde \Theta ( d ^ { 2 } \widetilde \kappa ^ { 2 } k \varepsilon ^ { - 2 } )$ . Then by applying Claim C.3 we get the cumulative regret bound.

Next we show how to estimate the action with $\widetilde { \cal O } ( d ^ { 2 } k ^ { 4 } \varepsilon ^ { - 2 } )$ samples. To achieve this result, we need to slightly alter Algorithm 5 where we respectively set $k ^ { \prime } = 2 , 4 , 6 , \cdot \cdot \cdot 2 k$ and keep the best arm among the $k$ outputs. We argue that among all the choices of $k ^ { \prime }$ , at least for one $l \in [ k ] , k ^ { \prime } = 2 l$ , we have $| \lambda _ { l } | - | \lambda _ { l + 1 } | \ge \lambda _ { 1 } / k$ . Notice with similar argument as above, when we set $n = \widetilde { \Theta } ( d ^ { 2 } \lambda _ { l } ^ { - 2 } \widetilde { \varepsilon } ^ { - 2 } ) \le \widetilde { \Theta } ( d ^ { 2 } k ^ { 2 } \lambda _ { 1 } ^ { - 2 } \widetilde { \varepsilon } ^ { - 2 } )$ we can get $\| G \| \le \widetilde { \varepsilon } \lambda _ { l }$ as required by Corollary B.4, the total number of iterations $L = O ( \sigma _ { l } / ( \sigma _ { l } - \sigma _ { l + 1 } ) \log ( 2 d / \epsilon ) =$ ${ \widetilde O } ( k )$ . Finally by setting $\widetilde { \varepsilon } = \epsilon / ( 4 \lambda _ { 1 } )$ we get the overall samples we required is ${ \widetilde { O } } ( k ^ { 2 } n ) =$ $\widetilde { \cal O } ( d ^ { 2 } k ^ { 4 } \epsilon ^ { - 2 } )$ .

For both settings, directly applying our arguments in the PAC to regret conversion: Claim C.3 will finish the proof. □

# B.2 Omitted Proofs of Main Results of Low-Rank Linear Reward

Theorem B.3 (Formal statement of Theorem 3.8). In Algorithm 2, for large enough constants $C _ { n } , C _ { L } , C _ { m }$ , let $n = C _ { n } d ^ { 2 } \log ^ { 2 } ( d / \delta ) \sigma _ { k } ^ { - 2 } \varepsilon ^ { - 2 }$ , $m = C _ { m } d \log ( n / \delta )$ , and $L = C _ { L } \log ( d / \varepsilon ) ,$ , $X _ { L }$ satisfies $\| ( I - X _ { L } X _ { L } ^ { \top } ) V \| \leq \varepsilon / 4$ , and the output $\widehat { A }$ satisfies $\| \widehat { A } - A ^ { * } \| _ { F } \leq \| M \| _ { F } \varepsilon$ . Altogether to get an $\varepsilon$ -optimal action, it is sufficient to have total sample complexity of $T \le \breve { O } ( d ^ { 2 } k \lambda _ { k } ^ { - 2 } \varepsilon ^ { - 2 } )$ .

Proof of Theorem 3.8 . Let $M = V \Sigma V ^ { \top }$ . From Claim B.6 we get that for each noisy subspace iteration step we get $Y _ { l } = M X _ { l } + G _ { l }$ with $5 \| \pmb { G } _ { l } \| ~ \le ~ \varepsilon \sigma _ { k }$ and $\| \mathbfcal { V } ^ { \top } G \| \leq$ $\sigma _ { k } \sqrt { k } / 3 \sqrt { d } \ \leq \ \sigma _ { k } ( \sqrt { 2 k } - \sqrt { k } ) / 2 \sqrt { d }$ . Therefore we can apply Corollary B.4, and get $\| V ( X _ { L } X _ { L } ^ { \top } - I ) \| \le \varepsilon / 4$ with $O ( \log 2 d / \epsilon )$ steps. Therefore we have:

$$
\begin{array}{l} \left\| \boldsymbol {A} _ {L} - \boldsymbol {M} \right\| _ {F} = \left\| \left(\boldsymbol {M} \boldsymbol {X} _ {L} + \boldsymbol {G} _ {L}\right) \boldsymbol {X} _ {L} ^ {\top} - \boldsymbol {M} \right\| _ {F} = \left\| \boldsymbol {V} ^ {\top} \boldsymbol {\Sigma} \boldsymbol {V} \left(\boldsymbol {X} _ {L} \boldsymbol {X} _ {L} ^ {\top} - \boldsymbol {I}\right)\right) + \left. \boldsymbol {G} _ {L} \boldsymbol {X} _ {L} ^ {\top} \right\| _ {F} \\ \leq \| \boldsymbol {M} \| _ {F} \| \boldsymbol {V} (\boldsymbol {X} _ {L} \boldsymbol {X} _ {L} ^ {\top} - \boldsymbol {I}) \| + \| \boldsymbol {G} _ {L} \| \| \boldsymbol {X} _ {L} \| _ {F} \\ \leq (\| \boldsymbol {M} \| _ {F} + \sigma_ {k}) \varepsilon / 4 <   \| \boldsymbol {M} \| _ {F} \varepsilon / 2. \\ \end{array}
$$

Meanwhile, notice $\| A ^ { * } \| _ { F } = 1 , \| M \| _ { F } = r ^ { * }$ and $\| \widehat { \cal A } \| _ { F } = 1$ . $\| A _ { L } / r ^ { * } - A ^ { * } \| _ { F } \leq \varepsilon / 2$ . $\| \ b { \tilde { A } } - \pmb { A ^ { * } } \| _ { F } = \| \pmb { A _ { L } } / \| \pmb { A _ { L } } \| _ { F } - \pmb { A ^ { * } } \| _ { F } = \| \mathrm { v e c } ( \pmb { A _ { L } } ) / \| \mathrm { v e c } ( \pmb { A _ { L } } ) \| _ { 2 } - \mathrm { v e c } ( \pmb { A ^ { * } } ) \| _ { 2 }$ .

Write $\theta _ { A } : = \theta ( \mathrm { v e c } ( A _ { L } ) , \mathrm { v e c } ( A ^ { * } )$ . The worst case that makes $\| \mathrm { v e c } ( \widehat { A } ) - \mathrm { v e c } ( A ^ { * } ) \|$ to be larger than $\left\| \mathrm { v e c } ( A _ { L } / r ^ { * } ) - \mathrm { v e c } ( A ^ { * } ) \right\|$ is when $\| \mathrm { v e c } ( A _ { L } / r ^ { * } ) - \mathrm { v e c } ( A ^ { * } ) \| = \sin \theta _ { A }$ and $\| \mathrm { v e c } ( \widehat { A } ) - \mathrm { v e c } ( A ^ { * } ) \|$ is always $2 \sin ( \theta _ { A } / 2 )$ . Notice trivially $2 \sin ( \theta _ { A } / 2 ) \leq 2 \sin ( \theta _ { A } )$ Therefore we could get $\Vert \widehat { \pmb { A } } - \pmb { A } ^ { * } \Vert _ { F } \leq 2 \Vert \pmb { A } _ { L } / r ^ { * } - \pmb { A } ^ { * } \Vert _ { F } \leq \varepsilon$ .

![](images/268b0c18a31dd4e8064e3d3b5913fc6204b7d720e779be34335c30e71fb8ac03.jpg)

Proof of Theorem 3.10. We find an $l$ to be the smallest integer such that $\begin{array} { r } { \sum _ { i = l + 1 } ^ { k } \sigma _ { i } ^ { 2 } \ \le \ } \end{array}$ $\epsilon ^ { 2 } \lVert \boldsymbol { M } \rVert _ { F } ^ { 2 }$ . Then we have $\sigma _ { l } \geq \epsilon / \sqrt { k - l } > \epsilon / \sqrt { k }$ .

Notice that in Algorithm 2, we set $\begin{array} { r } { n \geq \widetilde { \Theta } \big ( \frac { d ^ { 2 } k } { ( r ^ { \ast } ) ^ { 2 } \varepsilon ^ { 4 } } \big ) } \end{array}$ (  d k(r ∗ )2 ε 4 ) large enough such that $\| G \| _ { 2 } ~ \leq$ $O ( \| M \| _ { F } \epsilon ^ { 2 } / \sqrt { k } ) \lesssim \epsilon ( \sigma _ { l } - 0 )$ and $\begin{array} { r } { \| \pmb { U } ^ { \top } \pmb { G } \| _ { 2 } \le \| \pmb { M } \| _ { F } \epsilon / \sqrt { k } \frac { \sqrt { k ^ { \prime } } - \sqrt { k - 1 } } { 2 \sqrt { d } } } \end{array}$ . (This comes from the argument proved in Claim B.6.)

Therefore by conducting noisy power method we get with $\begin{array} { r } { O ( n k ) = \widetilde { O } \big ( \frac { d ^ { 2 } k ^ { 2 } } { ( r ^ { * } ) ^ { 2 } \varepsilon ^ { 4 } } \big ) } \end{array}$ ( d2k2(r∗)2ε4 ) samples we can get an action $\widehat { A }$ that satisfies:

$$
\| \boldsymbol {M} - \boldsymbol {X} _ {L} \boldsymbol {X} _ {L} ^ {\top} \boldsymbol {M} \| _ {F} ^ {2} \leq \sum_ {i = l + 1} ^ {k} \sigma_ {i} ^ {2} + l \epsilon^ {2} \sigma_ {l} ^ {2} \leq 2 \| \boldsymbol {M} \| _ {F} ^ {2} \epsilon^ {2}.
$$

Therefore we could get $\| A ^ { * } - { \widehat { A } } \| \leq 2 \epsilon$ , and with similar argument as (1) we have $r ^ { * } -$ $f ( \widehat { A } ) \leq \| M \| _ { F } \epsilon ^ { 2 }$ .

Therefore if we want to take a total of $T$ actions, we will set $\begin{array} { r } { \epsilon ^ { 6 } = \widetilde { \Theta } \big ( \frac { d ^ { 2 } k ^ { 2 } } { ( r ^ { * } ) ^ { 2 } T } \big ) } \end{array}$ d2 k 2 and we get:

$$
\begin{array}{l} \Re (T) = \sum_ {t = 1} ^ {T _ {1}} r ^ {*} - f (\pmb {A} _ {t}) + \sum_ {t = T _ {1} + 1} ^ {T} r ^ {*} - f (\widehat {\pmb {A}}) \\ <   r ^ {*} T _ {1} + T r ^ {*} \varepsilon^ {2} \\ \leq \widetilde {O} (d ^ {2 / 3} k ^ {2 / 3} (r ^ {*}) ^ {1 / 3} T ^ {2 / 3}). \\ \end{array}
$$

![](images/c0e1957b1627e3406561faa5523dd41ed06956d78c79b2c9424f129910a0fdf2.jpg)

# B.3 Technical Details for Quadratic Reward

# Noisy Power Method.

Corollary B.4 (Adapted from Corollary 1.1 from [HP14]). Let $k ^ { \prime } \geq l .$ . Let $U \in \mathbb { R } ^ { d \times l }$ represent the top l singular vectors of M and let $\sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { k } > 0$ denote its singular values. Suppose $X _ { 0 }$ is an orthonormal basis of a random $k ^ { \prime }$ -dimensional subspace. Further suppose that at every step of NPM we have

$$
\begin{array}{l} 5 \| \boldsymbol {G} \| \leq \epsilon (\sigma_ {l} - \sigma_ {l + 1}), \\ a n d 5 \| \boldsymbol {U} ^ {\top} \boldsymbol {G} \| \leq (\sigma_ {l} - \sigma_ {l + 1}) \frac {\sqrt {k ^ {\prime}} - \sqrt {l - 1}}{2 \sqrt {d}} \\ \end{array}
$$

for some fixed parameter $\epsilon < 1 / 2$ . Then with all but $2 ^ { - \Omega ( k ^ { \prime } + 1 - l ) } + e ^ { \Omega ( d ) }$ probability, there exists an $\begin{array} { r } { L = O ( \frac { \sigma _ { l } } { \sigma _ { l } - \sigma _ { l + 1 } } \log ( 2 d / \epsilon ) ) } \end{array}$ so that after $L$ steps we have that $\| ( I - { \mathbf { } } X _ { L } { \mathbf { } } X _ { L } ^ { \top } ) U \| \leq$ σl −σl+1 ǫ.

Theorem B.5 (Adapted from Theorem 2.2 from [BDWY16]). Let $U _ { l } \in \mathbb { R } ^ { d \times l }$ represent the top l singular vectors of M and let $\sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { k } > 0$ denote its singular values. Naturally $l \leq k$ . Suppose $X _ { 0 }$ is an orthonormal basis of a random $k ^ { \prime }$ -dimensional subspace where $k ^ { \prime } \geq k$ . Further suppose that at every step of NPM we have

$$
\begin{array}{l} \| \boldsymbol {G} \| \leq O (\epsilon \sigma_ {l}), \\ a n d \| \pmb {U} _ {k} ^ {\top} \pmb {G} \| _ {2} \leq O (\sigma_ {l} \frac {\sqrt {k ^ {\prime}} - \sqrt {k - 1}}{2 \sqrt {d}}) \\ \end{array}
$$

for small enough ǫ. Then with all but $2 ^ { - \Omega ( k ^ { \prime } + 1 - k ) } + e ^ { \Omega ( d ) }$ probability, there exists an $L =$ $O ( \log ( 2 d / \epsilon ) )$ so that after $L$ steps we have that $\begin{array} { r } { \| ( \pmb { I } - \pmb { X } _ { L } \pmb { X } _ { L } ^ { \top } ) \pmb { U } _ { l } \| \le \epsilon . } \end{array}$ . Furthermore:

$$
\| \boldsymbol {M} - \boldsymbol {X} _ {L} \boldsymbol {X} _ {L} ^ {\top} \boldsymbol {M} \| _ {F} ^ {2} \leq \sum_ {i = l + 1} ^ {k} \sigma_ {i} ^ {2} + l \sigma^ {2} \sigma_ {l} ^ {2}.
$$

# Concentration Bounds.

Claim B.6. Write the eigendecomposition for $M$ as $M = U \Sigma U ^ { \top }$ . In Algorithm 2, when $n \geq \widetilde { \Theta } ( d ^ { 2 } / ( \lambda _ { k } ^ { 2 } \varepsilon ^ { 2 } ) )$ , the noisy subspace iteration step can be written as: $Y _ { l } = M X _ { l - 1 } + G _ { l } ,$ , where the noise term satisfies:

$$
5 \| \left| \boldsymbol {G} _ {l} \right| \| \leq \varepsilon | \lambda_ {k} |
$$

$$
5 \| \boldsymbol {U} ^ {\top} \boldsymbol {G} _ {l} \| \leq \varepsilon | \lambda_ {k} | \frac {\sqrt {k}}{3 \sqrt {d}}.
$$

with high probability for our choice of n.

Proof. For compact notation, write vector $\pmb { \eta } _ { i } : = [ \eta _ { i , 1 } , \eta _ { i , 2 } , \bot \cdot \cdot \eta _ { i , k ^ { \prime } } ] ^ { \top } \in \mathbb { R } ^ { k ^ { \prime } }$ . We have:

$$
\boldsymbol {G} _ {l} (s) = \frac {m}{n} \sum_ {i = 1} ^ {n} \left(\boldsymbol {z} _ {i} ^ {\top} \boldsymbol {M} \boldsymbol {X} _ {l} (s)\right) \boldsymbol {z} _ {i} + \frac {m}{n} \sum_ {i = 1} ^ {n} \eta_ {i, s} \boldsymbol {z} _ {i} - \boldsymbol {M} \boldsymbol {X} _ {l} (s), \text {t h e r e f o r e}
$$

$$
\boldsymbol {G} _ {l} = \left(\frac {m}{n} \sum_ {i = 1} ^ {n} \left[ \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} \right] - I\right) \boldsymbol {M} \boldsymbol {X} _ {l} + \frac {m}{n} \sum_ {i = 1} ^ {n} \boldsymbol {z} _ {i} \boldsymbol {\eta} _ {i} ^ {\top}.
$$

First note that for orthogonal matrix $X _ { l } , \| M X _ { l } \| \ \leq \ \lambda _ { 1 }$ $X _ { l }$ , and $\begin{array} { r } { \| \frac { m } { n } \sum _ { i = 1 } ^ { n } [ z _ { i } z _ { i } ^ { \top } ] - I \| \ \leq } \end{array}$ $O ( \sqrt { \frac { d + \log ( 1 / \delta ) } { n } } )$ d+log(1/δ)n ). The bottleneck is from the second term and we will use Matrix Bernstein t. Write $\begin{array} { r } { S _ { i } = \frac { m } { n } z _ { i } \eta _ { i } ^ { \top } } \end{array}$ . We have $\begin{array} { r } { \| \boldsymbol { S } _ { i } \| \le O ( \frac { \sqrt { m k ^ { \prime } } \log ( n / \delta ) } { n } ) } \end{array}$ with probability $1 - \delta$ and $\begin{array} { r } { \mathbb { E } [ \sum _ { i } S _ { i } S _ { i } ^ { \top } ] = \frac { m k ^ { \prime } } { n } I _ { d } } \end{array}$ and $\begin{array} { r } { \mathbb { E } [ \sum _ { i } S _ { i } ^ { \top } S _ { i } ] = \frac { m d } { n } I _ { k ^ { \prime } } } \end{array}$ . Therefore with matrix Bernstein we can get that $\begin{array} { r } { \| \sum _ { i } S _ { i } \| _ { i } \le O ( \sqrt { \frac { m d } { n } } \log ( d / \delta ) ) } \end{array}$ with probability $1 - \delta$ .

Therefore for $n \geq \widetilde \Omega ( d ^ { 2 } / ( \lambda _ { k } ^ { 2 } \varepsilon ^ { 2 } )$ , we can get that $5 \| G _ { l } \| \le \varepsilon \| \lambda _ { k } \|$

Similarly since $\begin{array} { r } { U ^ { \top } z _ { i } \sim { \mathcal N } ( 0 , \frac { 1 } { m } I _ { k ^ { \prime } } ) } \end{array}$ , with the same argument one can easily get that $\begin{array} { r } { \| U ^ { \top } \pmb { G } _ { l } \| \le O ( \sqrt { \frac { m k ^ { \prime } } { n } } \log ( d / \delta ) ) } \end{array}$ . Therefore with the same lower bound for $n$ one can get $\begin{array} { r } { 1 5 \| \pmb { U } ^ { \top } \pmb { G } _ { l } \| \leq \varepsilon \| \lambda _ { k } \| \sqrt { \frac { k } { d } } , } \end{array}$ . □

# B.4 Omitted Proof for RL with Quadratic Q function

Proof of Theorem 3.13. With the oracle, at horizon $H$ , we can estimate $\widehat { M } _ { H }$ that is $\epsilon / H$ close to $M _ { H } ^ { * }$ in spectral norm through noisy observations from the reward function with

Algorithm 6 Learn policy complete polynomial with simulator.   
1: Initialize: Set $n = \widetilde{\Theta}(\widetilde{\kappa}^2 d^2 H^3 / \varepsilon^2)$ , Oracle to estimate $\widehat{T}_h$ from noisy observations.  
2: for $h = H, \ldots, 1$ do  
3: Sample $\phi(s_h^i, a_h^i)$ , $i \in [n]$ from standard Gaussian $N(0, I_d)$ 4: for $i \in [n]$ do  
5: Query $(s_h^i, a_h^i)$ and use $\pi_{h+1}, \ldots, \pi_H$ as the roll-out to get estimation $\widehat{Q}_h^{\pi_{h+1}, \ldots, \pi_H}(s_h^i, a_h^i)$ 6: Retrieve $\widehat{M}_h$ from estimation $\widehat{Q}_h^{\pi_{h+1}, \ldots, \pi_H}(s_h^i, a_h^i)$ , $i \in [n]$ 7: Set $\widehat{Q}_h(s, a) \gets f_{\widehat{T}_h}$ 8: Set $\pi_h(s) \gets \arg \max_{a \in S} \widehat{Q}_h(s, a)$ 9: Return $\pi_1, \ldots, \pi_H$

$\widetilde { \cal O } ( \widetilde { \kappa } ^ { 2 } d ^ { 2 } H ^ { 2 } / \varepsilon ^ { 2 } )$ samples. Next, for each horizon $h = H - 1 , H - 1 , \cdots , 1$ , sample $s _ { i } ^ { \prime } \sim$ $\mathbb { P } ( \cdot | s , a )$ , we define $\begin{array} { r } { \eta _ { i } = \operatorname* { m a x } _ { a ^ { \prime } } f _ { \widehat { M } _ { h + 1 } } ( s _ { i } ^ { \prime } , a ^ { \prime } ) - { \mathbb E } _ { s ^ { \prime } \sim { \mathbb P } ( \cdot \vert s , a ) } \operatorname* { m a x } _ { a ^ { \prime } } f _ { \widehat { M } _ { h + 1 } } ( s ^ { \prime } , a ^ { \prime } ) } \end{array}$ . $\eta _ { i }$ is meanzero and $O ( 1 )$ -sub-gaussian since it is bounded. Denote $M _ { h }$ as the matrix that satisfies $f _ { M _ { h } } : = \tau f _ { \widehat { M } _ { h + 1 } }$ , which is well-defined due to Bellman completeness. We estimate $\widehat { M } _ { h }$ from the noisy observations $\begin{array} { r } { y _ { i } = r _ { h } ( s , a ) + \operatorname* { m a x } _ { a ^ { \prime } } f _ { \widehat M _ { h + 1 } } ( s _ { i } ^ { \prime } , a ^ { \prime } ) = \mathcal { T } f _ { \widehat M _ { h + 1 } } + \eta _ { i } = : f _ { M _ { h } } + \eta _ { i } . } \end{array}$ Therefore with the oracle, we can estimate $\widehat { M } _ { h }$ such that $\| \widehat { M } _ { h } - M _ { h } \| _ { 2 } \le \epsilon / H$ with $\Theta ( \widetilde { \kappa } ^ { 2 } d ^ { 2 } k ^ { 2 } H ^ { 2 } / \epsilon ^ { 2 } )$ bandits. Together we have:

$$
\begin{array}{l} \left\| f _ {\widehat {\boldsymbol {M}} _ {h}} - f _ {\widehat {\boldsymbol {M}} _ {h} ^ {*}} \right\| _ {\infty} = \left\| \widehat {\boldsymbol {M}} _ {h} - \boldsymbol {M} _ {h} ^ {*} \right\| \\ \leq \left\| \widehat {\boldsymbol {M}} _ {h} - \boldsymbol {M} _ {h} \right\| + \left\| \boldsymbol {M} _ {h} - \boldsymbol {M} _ {h} ^ {*} \right\| \\ \leq \epsilon / H + \| \mathcal {T} f _ {\widehat {M} _ {h + 1}} - \mathcal {T} f _ {M _ {h + 1} ^ {*}} \| _ {\infty} \\ \leq \epsilon / H + \| f _ {\widehat {M} _ {h + 1}} - f _ {M _ {h + 1} ^ {*}} \| _ {\infty} \\ \leq 2 \epsilon / H + \| f _ {\widehat {M} _ {h + 2}} - f _ {M _ {h + 2} ^ {*}} \| _ {\infty} \\ \leq \dots \\ \leq (H - h) \epsilon / H. \\ \end{array}
$$

Finally for $h = 1$ we have $\| \widehat { M } _ { 1 } - M ^ { * } \| \leq \epsilon$ if we sample $n = \widetilde { \Theta } ( \widetilde { \kappa } ^ { 2 } d ^ { 2 } k ^ { 2 } H ^ { 2 } / \epsilon ^ { 2 } )$ for each $h \in [ H ]$ . Therefore for all the $H$ timesteps we need $\Theta ( \widetilde { \kappa } ^ { 2 } d ^ { 2 } k ^ { 2 } H ^ { 3 } / \epsilon ^ { 2 } )$ .

![](images/4992445e38c6a5013508f3b070f111fcd3025678b195de29124ca46393661cd4.jpg)

# C Technical details for General Tensor Reward

# C.1 Technical Details for Symmetric Setting

Lemma C.1 (Zeroth order optimization for noiseless setting). For $p \geq 3$ , suppose $0 . 5 a ^ { \top } v _ { 1 } >$ $| \mathbf { \boldsymbol { a } } ^ { \top } \mathbf { \boldsymbol { v } } _ { j } |$ for all $j \geq 2$ , we have:

$$
\tan \theta (G (\boldsymbol {a}), \boldsymbol {v} _ {1}) \leq \frac {1}{2} \tan \theta (\boldsymbol {a}, \boldsymbol {v} _ {1}).
$$

Proof. We first simplify $\begin{array} { r } { G ( \pmb { a } ) = \sum _ { j = 1 } ^ { r } \lambda _ { j } \pmb { v } _ { j } \cdot \boldsymbol { S } _ { j } } \end{array}$ , where

$$
\begin{array}{l} G(\boldsymbol {a}) = \sum_{s = 0}^{\lfloor (p - 3) / 2\rfloor}\frac{(1 - \frac{1}{2p})^{p - 2s - 1}(\frac{1}{2p})^{2s + 1}}{m^{s}}\binom {p}{2s + 1}T(I^{\otimes s + 1}\otimes \boldsymbol{a}^{\otimes p - 2s - 1}) \\ = \sum_ {s = 0} ^ {\lfloor (p - 3) / 2 \rfloor} \frac {(1 - \frac {1}{2 p}) ^ {p - 2 s - 1} (\frac {1}{2 p}) ^ {2 s + 1}}{m ^ {s}} \binom {p} {2 s + 1} \sum_ {j = 1} ^ {k} \lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {p - 2 j - 1} \boldsymbol {v} _ {j} \\ = \sum_ {j = 1} ^ {k} v _ {j} \cdot \overbrace {\lambda_ {j} \sum_ {s = 0} ^ {\lfloor (p - 3) / 2 \rfloor} \frac {(1 - \frac {1}{2 p}) ^ {p - 2 s - 1} (\frac {1}{2 p}) ^ {2 s + 1}}{m ^ {s}} \binom {p} {2 s + 1} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {p - 2 s - 1}} ^ {S _ {j} :=} \\ = \sum_ {j = 1} ^ {k} S _ {j} \boldsymbol {v} _ {j}. \\ \end{array}
$$

Notice for even $p$ ,

$$
\begin{array}{l} S _ {j} = \lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {3} \cdot \sum_ {s = 0} ^ {p / 2 - 2} \frac {(1 - \frac {1}{2 p}) ^ {p - 2 s - 1} (\frac {1}{2 p}) ^ {2 s + 1}}{m ^ {s}} \binom {p} {2 s + 1} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {p - 2 s - 4} \\ = \lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {3} \cdot \sum_ {r = 0} ^ {p / 2 - 2} \frac {(1 - \frac {1}{2 p}) ^ {2 r + 3} (\frac {1}{2 p}) ^ {p - 3 - 2 r}}{m ^ {p / 2 - 2 - r}} \binom {p} {p - 2 r - 3} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {2 r}. \\ (\text {l e t} 2 r = p - 4 - 2 s) \\ \end{array}
$$

$$
\frac {S _ {j}}{\lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {3}} = \sum_ {r = 0} ^ {p / 2 - 2} \frac {(1 - \frac {1}{2 p}) ^ {2 r + 3} (\frac {1}{2 p}) ^ {p - 3 - 2 r}}{m ^ {p / 2 - 2 - r}} \binom {p} {p - 2 r - 3} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {2 r}
$$

(Divide both sides by $\lambda _ { j } ( \pmb { v } _ { j } ^ { \top } \pmb { a } ) ^ { 3 } )$ )

$$
\leq \sum_ {r = 0} ^ {p / 2 - 2} \frac {(1 - \frac {1}{2 p}) ^ {2 r + 3} (\frac {1}{2 p}) ^ {p - 3 - 2 r}}{m ^ {p / 2 - 2 - r}} \binom {p} {p - 2 r - 3} (\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}) ^ {2 r}.
$$

(Since the first term is constant and $| \pmb { v } _ { j } ^ { \top } \pmb { a } | \leq \pmb { v } _ { 1 } ^ { \top } \pmb { a }$ for $r \geq 1 \AA$ )

$$
= \frac {S _ {1}}{\lambda_ {1} \left(\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}\right) ^ {3}}.
$$

Therefore for even $p \geq 4$ :

$$
\left| S _ {j} \right| \leq \frac {\left| \lambda_ {j} \right|}{\lambda_ {1}} \frac {\left| \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a} \right| ^ {3}}{\left| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} \right| ^ {3}} S _ {1} \leq \frac {1}{4} \frac {\left| \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a} \right|}{\left| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} \right|} S _ {1}, \forall j \geq 2. \tag {4}
$$

Similarly for odd $p$ , we have:

$$
S _ {j} = \lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {2} \cdot \sum_ {s = 0} ^ {(p - 3) / 2} \frac {(1 - \frac {1}{2 p}) ^ {p - 2 s - 1} (\frac {1}{2 p}) ^ {2 s + 1}}{m ^ {s}} \binom {p} {2 s + 1} (v _ {j} ^ {\top} a) ^ {p - 2 s - 3}
$$

$$
= \lambda_ {j} (\pmb {v} _ {j} ^ {\top} \pmb {a}) ^ {2} \cdot \sum_ {r = 0} ^ {(p - 3) / 2} \frac {(1 - \frac {1}{2 p}) ^ {2 r + 2} (\frac {1}{2 p}) ^ {p - 2 - 2 r}}{m ^ {(p - 3) / 2 - r}} \binom {p} {p - 2 - 2 r} (\pmb {v} _ {j} ^ {\top} \pmb {a}) ^ {2 r},
$$

(L $\mathfrak { a } \ : r = ( p - 3 ) / 2 - s )$

$$
\frac {S _ {j}}{\lambda_ {j} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {2}} = \sum_ {r = 0} ^ {(p - 3) / 2} \frac {(1 - \frac {1}{2 p}) ^ {2 r + 2} (\frac {1}{2 p}) ^ {p - 2 - 2 r}}{m ^ {(p - 3) / 2 - r}} \binom {p} {p - 2 - 2 r} (\boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a}) ^ {2 r}
$$

(Divide both sides by $\lambda _ { j } ( \pmb { v } _ { j } ^ { \top } \pmb { a } ) ^ { 2 } )$

$$
\leq \sum_ {r = 0} ^ {(p - 3) / 2} \frac {(1 - \frac {1}{2 p}) ^ {2 r + 2} (\frac {1}{2 p}) ^ {p - 2 - 2 r}}{m ^ {(p - 3) / 2 - r}} \binom {p} {p - 2 - 2 r} (\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}) ^ {2 r}
$$

(Since the first term is constant and $| \pmb { v } _ { j } ^ { \top } \pmb { a } | \leq \pmb { v } _ { 1 } ^ { \top } \pmb { a }$ for $r \geq 1 \AA$

$$
= \frac {S _ {1}}{\lambda_ {1} (\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}) ^ {2}}.
$$

Therefore for odd $p$ we have:

$$
\left| S _ {j} \right| \leq \frac {\left| \lambda_ {j} \right|}{\lambda_ {1}} \frac {\left| \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a} \right| ^ {2}}{\left| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} \right| ^ {2}} S _ {1} \leq \frac {1}{2} \frac {\left| \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a} \right|}{\left| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} \right|} S _ {1}, \forall j \geq 2. \tag {5}
$$

Write $V = [ \pmb { v } _ { 2 } , \pmb { v } _ { 3 } , \pmb { \cdot } \cdot \pmb { \cdot } , \pmb { v } _ { k } ] \in \mathbb { R } ^ { d \times k }$ be the complement for ${ \pmb v } _ { 1 }$ . Therefore for any $_ { x }$ without normalization, one can conveniently represent $| \tan \theta ( { \pmb x } , { \pmb v } _ { 1 } ) |$ as $\| \ b { V } ^ { \top } \pmb { x } \| _ { 2 } / | \pmb { v } _ { 1 } ^ { \top } \ b { x } |$ .

$$
\begin{array}{l} \left\| \boldsymbol {V} ^ {\top} G (\boldsymbol {a}) \right\| ^ {2} = \sum_ {j = 2} ^ {k} S _ {j} ^ {2} (6) \\ \leq \sum_ {j = 2} ^ {k} \frac {\left| \boldsymbol {v} _ {j} ^ {\top} \boldsymbol {a} \right| ^ {2}}{4 \left| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} \right| ^ {2}} S _ {1} ^ {2} \quad (\text {f r o m (5) , (4)}) \\ = \frac {1}{4} \tan^ {2} \theta (\boldsymbol {v} _ {1}, \boldsymbol {a}) (\boldsymbol {v} _ {1} ^ {\top} G (\boldsymbol {a})) ^ {2}. (7) \\ \end{array}
$$

Therefore for $p \geq 3$ , $\begin{array} { r } { \tan \theta ( G ( \pmb { a } ) , \pmb { v } _ { 1 } ) \leq \frac { 1 } { 2 } \tan \theta ( \pmb { a } , \pmb { v } _ { 1 } ) . } \end{array}$

Proof of Lemma 3.21. We first estimate $G _ { n } ( \pmb { a } ) - \mathbb { E } [ G _ { n } ( \pmb { a } ) ]$ , which is want we want for even $p$ . For odd $p$ we will need to analyze an extra bias term that is orthogonal to ${ \pmb v } _ { 1 }$ , $\begin{array} { r } { e : = \frac { ( \frac { 1 } { 2 p } ) ^ { p - 1 } ( 1 - \frac { 1 } { 2 p } ) p } { m ^ { p / 2 - 1 } } \sum _ { j = 2 } ^ { k } \lambda _ { j } ( \pmb { v } _ { j } ^ { \top } \pmb { a } ) \pmb { v } _ { j } } \end{array}$ mp/2−1 ; and we have $G _ { n } ( \pmb { a } ) - \mathbb { E } [ G _ { n } ( \pmb { a } ) ] = G _ { n } ( \pmb { a } ) - G ( \pmb { a } ) + e$

We decompose $G _ { n } ( { \pmb a } )$ as $\begin{array} { r } { G _ { n } ( \pmb { a } ) = \sum _ { s = 1 } ^ { k } G _ { n } ^ { ( s ) } + N } \end{array}$ , where $\begin{array} { r } { G _ { n } ^ { ( s ) } : = \frac { m } { n } \sum _ { i = 1 } ^ { n } \binom { p } { s } T ( ( 1 - \Biggr . } \end{array}$ $0 . 5 / p ) { \pmb a } ) ^ { \otimes p - s } \otimes ( { \pmb z } _ { i } / ( 2 p ) ) ^ { \otimes s } ) { \pmb z } _ { i }$ . The noise term $\begin{array} { r } { N : = \frac { m } { n } \sum \epsilon _ { i } z _ { i } } \end{array}$ .

$$
\begin{array}{l} G _ {n} ^ {(s)} := \frac {m}{n} \sum_ {i = 1} ^ {n} \binom {p} {s} T ((1 - 0. 5 / p) \boldsymbol {a}) ^ {\otimes p - s} \otimes (\boldsymbol {z} _ {i} / (2 p)) ^ {\otimes s}) \boldsymbol {z} _ {i} \\ = \frac {m}{n} (1 - \frac {1}{2 p}) ^ {p - s} (\frac {1}{2 p}) ^ {s} {\binom {p} {s}} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \lambda_ {j} ({\pmb a} ^ {\top} {\pmb v} _ {j}) ^ {p - s} ({\pmb z} _ {i} ^ {\top} {\pmb v} _ {j}) ^ {s} {\pmb z} _ {i}. \\ \end{array}
$$

$$
\begin{array}{l} \mathbb {E} \left[ G _ {n} ^ {(s)} \right] = m (1 - \frac {1}{2 p}) ^ {p - s} (\frac {1}{2 p}) ^ {s} \binom {p} {s} \sum_ {j = 1} ^ {k} \lambda_ {j} \left(\boldsymbol {a} ^ {\top} \boldsymbol {v} _ {j}\right) ^ {p - s} \mathbb {E} \left[ \left(\boldsymbol {z} ^ {\top} \boldsymbol {v} _ {j}\right) ^ {s} \boldsymbol {z} \right] \\ = \left\{ \begin{array}{l l} (1 - \frac {1}{2 p}) ^ {p - s} (\frac {1}{2 p}) ^ {s} m \binom {p} {s} \sum_ {j = 1} ^ {k} \lambda_ {j} (\boldsymbol {a} ^ {\top} \boldsymbol {v} _ {j}) ^ {p - s} \frac {1}{m ^ {(s + 1) / 2}} (s)!! \boldsymbol {v} _ {j}, & \text {f o r o d d s ,} \\ 0, & \text {f o r e v e n s} \end{array} \right. \\ = \left\{ \begin{array}{l l} (1 - \frac {1}{2 p}) ^ {p - s} (\frac {1}{2 p}) ^ {s} \frac {s ! !}{m ^ {(s - 1) / 2}} \binom {p} {s} \sum_ {j = 1} ^ {k} \lambda_ {j} (\boldsymbol {a} ^ {\top} \boldsymbol {v} _ {j}) ^ {p - s} \boldsymbol {v} _ {j}, & \text {f o r o d d s ,} \\ 0, & \text {f o r e v e n s} \end{array} \right. \\ \end{array}
$$

$$
G _ {n} ^ {(s)} - \mathbb {E} [ G _ {n} ^ {(s)} ] = m (1 - \frac {1}{2 p}) ^ {p - s} (\frac {1}{2 p}) ^ {s} {\binom {p} {s}} \sum_ {j = 1} ^ {k} \lambda_ {j} (\pmb {a} ^ {\top} \pmb {v} _ {j}) ^ {p - s} \pmb {g} _ {n, s} (j),
$$

where $\begin{array} { r } { \pmb { g } _ { n , s } ( j ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \pmb { z } _ { i } ^ { \top } \pmb { v } _ { j } ) ^ { s } \pmb { z } _ { i } - \mathbb { E } [ ( \pmb { z } ^ { \top } \pmb { v } _ { j } ) ^ { s } \pmb { z } ] } \end{array}$

Notice the scaling in each $G _ { n } ^ { ( s ) }$ is $\begin{array} { r } { ( 1 - \frac 1 { 2 p } ) ^ { p - s } ( \frac 1 { 2 p } ) ^ { s } \binom { p } { s } \leq ( \frac 1 { 2 p } ) ^ { s } p ^ { s } / ( s ! ) < 2 ^ { - s } } \end{array}$ decays exponentially. In Claim C.8 we give bounds for $g _ { n , s } ( j )$ . We note the bound for each $g _ { n , s }$ also decays with $s$ .

Therefore the bottleneck of the upper bound mostly depend on $\mathbf { \mathcal { g } } _ { n , 0 }$ and ${ \pmb v } _ { 1 } ^ { \top } { \pmb g } _ { n , 0 }$ , and we get:

$$
\begin{array}{l} \| G _ {n} - \mathbb {E} [ G _ {n} ] \| \leq C _ {1} \lambda_ {1} \sqrt {\frac {(d + \log (1 / \delta)) d \log (n / \delta)}{n}} + N, \\ | \pmb {v} _ {1} ^ {\top} G _ {n} - \mathbb {E} [ \pmb {v} _ {1} ^ {\top} G _ {n} ] | \leq C _ {2} \lambda_ {1} \sqrt {\frac {d \log (n / \delta) (1 + \log (1 / \delta))}{n}} + \pmb {v} _ {1} ^ {\top} N. \\ \end{array}
$$

Next from Claim C.7, the noise term

$$
\begin{array}{l} N \leq C _ {3} \sqrt {\frac {m \log (n / \delta) (d + \log (n / \delta)) \log (d / \delta)}{n}}, \\ | v _ {1} ^ {\top} N | \leq C _ {4} \sqrt {m \frac {\log^ {2} (n / \delta) \log (d / \delta)}{n}}, \\ \end{array}
$$

Finally $e$ is very small: $\begin{array} { r } { \| e \| \le \frac { 1 } { m ^ { p / 2 - 1 } ( 2 p ) ^ { ( } p - 1 ) } \lambda _ { 2 } \| V ^ { \top } { \pmb a } \| = \lambda _ { 2 } \frac { 1 } { m ^ { p / 2 - 1 } ( 2 p ) ^ { ( } p - 1 ) } \sin \theta ( { \pmb a } , { \pmb v } _ { 1 } ) . } \end{array}$ $| e ^ { \top } { \pmb v } _ { 1 } | = 0$ .

Together we can bound $G _ { n } ( { \pmb a } ) - G ( { \pmb a } )$ and finish the proof.

Proof of Lemma 3.20. From Lemma C.1 we have: $| \tan \theta ( G ( \pmb { a } ) , \pmb { v } _ { 1 } ) | \leq 1 / 2 | \tan \theta ( \pmb { a } , \pmb { v } _ { 1 } ) | .$ Let $V = \left[ \pmb { v } _ { 2 } , \cdot \cdot \cdot \pmb { v } _ { k } \right]$ . For any $p \geq 2$ , we have:

$$
\begin{array}{l} | \tan \theta (\boldsymbol {a} ^ {+}, \boldsymbol {v} _ {1}) | = \frac {\left\| \boldsymbol {V} ^ {\top} \boldsymbol {a} ^ {+} \right\| _ {2}}{\left| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} ^ {+} \right|} \\ = \frac {\left\| \boldsymbol {V} ^ {\top} (\boldsymbol {G} (\boldsymbol {a}) + \boldsymbol {g}) \right\|}{\left| \boldsymbol {v} _ {1} ^ {\top} (\boldsymbol {G} (\boldsymbol {a}) + \boldsymbol {g}) \right|} \\ \leq \frac {\| \boldsymbol {V} ^ {\top} G (\boldsymbol {a}) \| + \| \boldsymbol {V} ^ {\top} \boldsymbol {g} \|}{| \boldsymbol {v} _ {1} ^ {\top} G (\boldsymbol {a}) | - | \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {g} |} \\ \leq \frac {1 / 2 | \tan \theta (\pmb {a} , \pmb {v} _ {1}) | | \pmb {v} _ {1} ^ {\top} G (\pmb {a}) | + \| \pmb {g} \|}{| \pmb {v} _ {1} ^ {\top} G (\pmb {a}) | - | \pmb {v} _ {1} ^ {\top} \pmb {g} |} \\ = \alpha | \tan \theta (\boldsymbol {a}, \boldsymbol {v} _ {1}) | \frac {S _ {1}}{S _ {1} - \| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {g} \|} + \frac {\| \boldsymbol {g} \|}{S _ {1} - \| \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {g} \|}, \\ \end{array}
$$

where S1 := v⊤1 G(a) = v⊤1 G(a) = λ1 P⌊(p−3)/2⌋s=0 (1− 12p )p−2s−1( 12p )2sms $\begin{array} { r } { \widetilde { \gamma } _ { 1 } : = v _ { 1 } ^ { \top } G ( { \pmb a } ) = v _ { 1 } ^ { \top } G ( { \pmb a } ) = \lambda _ { 1 } \sum _ { s = 0 } ^ { \lfloor ( p - 3 ) / 2 \rfloor } \frac { ( 1 - \frac { 1 } { 2 p } ) ^ { p - 2 s - 1 } ( \frac { 1 } { 2 p } ) ^ { 2 s + 1 } } { m ^ { s } } \binom { p } { 2 s + 1 } ( v _ { 1 } ^ { \top } { \pmb a } ) ^ { p - 2 s - 1 } \geq 0 , } \end{array}$ $\begin{array} { r } { \lambda _ { 1 } ( 1 - \frac { 1 } { 2 p } ) ^ { p - 1 } ( \frac { 1 } { 2 p } ) p ( { \pmb v } _ { 1 } ^ { \top } { \pmb a } ) ^ { p - 1 } \geq \frac { \lambda _ { 1 } } { 4 } ( { \pmb v } _ { 1 } ^ { \top } { \pmb a } ) ^ { p - 1 } , } \end{array}$ . The inequality comes from keeping only the first term where $s ~ = ~ 0$ . With the assumption that $| { \pmb v } _ { 1 } ^ { \top } { \pmb g } | \le 0 . 0 5 \lambda _ { 1 } ( { \pmb v } _ { 1 } ^ { \top } { \pmb a } ) ^ { p - 1 }$ , we have $| { \pmb v } _ { 1 } ^ { \top } { \pmb g } | \le 0 . 2 S _ { 1 }$ . Therefore

$$
\begin{array}{l} | \tan \theta (\pmb {a} ^ {+}, \pmb {v} _ {1}) | \leq 1. 2 5 / 2 | \tan \theta (\pmb {a}, \pmb {v} _ {1}) | + \frac {\| \pmb {g} \|}{S _ {1} - | \pmb {v} _ {1} ^ {\top} \pmb {g} |} \\ \leq 1. 2 5 / 2 \left| \tan \theta (\boldsymbol {a}, \boldsymbol {v} _ {1}) \right| + 5 / 4 \frac {\| \boldsymbol {g} \|}{S _ {1}} \\ \leq 1. 2 5 / 2 | \tan \theta (\boldsymbol {a}, \boldsymbol {v} _ {1}) | + 5 \frac {\| \boldsymbol {g} \|}{\lambda_ {1} (\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}) ^ {p - 1}}. \\ \end{array}
$$

Notice when 5 kgkλ1(v⊤a)p−1 ≤ max{0.125| tan θ(a, v1)|, eǫ}, which will ensure | tan θ(G(a), v1)| ≤ $\begin{array} { r } { 5 \frac { \lVert g \rVert } { \lambda _ { 1 } ( v _ { 1 } ^ { \top } { \pmb a } ) ^ { p - 1 } } \le \operatorname* { m a x } \{ 0 . 1 2 5 | \tan \theta ( { \pmb a } , { \pmb v } _ { 1 } ) | , \widetilde \epsilon \} } \end{array}$ $| \tan \theta ( G ( { \pmb a } ) , { \pmb v } _ { 1 } ) | \leq$ $( 1 . 2 5 / 2 + 0 . 1 2 5 ) | \tan \theta ( G ( { \pmb a } ) , { \pmb v } _ { 1 } ) | + \widetilde { \epsilon }$ . (We will prove this condition is satisfied when $\begin{array} { r } { \| \pmb { g } \| \le \operatorname* { m i n } \{ \frac { 0 . 0 2 5 } { p } \lambda _ { 1 } ( \pmb { v } _ { 1 } ^ { \top } \pmb { a } ) ^ { p - 2 } , 0 . 1 \lambda _ { 1 } \widetilde { \varepsilon } \} } \end{array}$ e. We will handle the additional term in the upper bound of $\| g \|$ later.) We divide this requirement into the following two cases. On one hand, when $| \pmb { v } _ { 1 } ^ { \top } \pmb { a } | \le 1 - 1 / ( p - 1 )$ $- 1 / ( p - 1 ) , \| \dot { \mathbf { V } ^ { \top } } \pmb { a } \| \geq \sqrt { 1 - ( 1 - 1 / ( p - \hat { 1 } ) ) ^ { 2 } } > 1 / p$ , therefore $| \tan \theta ( { \mathbf a } , { \mathbf v } _ { 1 } ) | \geq 1 / p | { \mathbf v } _ { 1 } ^ { \top } { \mathbf a } |$ . Therefore

$$
\begin{array}{l} 5 \frac {\| \pmb {g} \|}{\lambda_ {1} (\pmb {v} _ {1} ^ {\top} \pmb {a}) ^ {p - 1}} \leq 0. 1 2 5 | \tan \theta (\pmb {a}, \pmb {v} _ {1}) | \\ \Leftarrow 5 \frac {\| \boldsymbol {g} \|}{\lambda_ {1} \left(\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}\right) ^ {p - 1}} \leq 0. 1 2 5 / \left(p \mid \boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a} \mid\right) \\ \Leftrightarrow \| \boldsymbol {g} \| \leq 0. 0 2 5 \lambda_ {1} (\boldsymbol {v} _ {1} ^ {\top} \boldsymbol {a}) ^ {p - 2} / p. \\ \end{array}
$$

On the other hand, when $| \pmb { v } _ { 1 } ^ { \top } \pmb { a } | \geq 1 - 1 / ( p - 1 )$ , $| \pmb { v } _ { 1 } ^ { a } | ^ { p - 1 } \geq 1 / 4$ when $p = 3$ . Therefore $\begin{array} { r } { 5 \frac { \| g \| } { \lambda _ { 1 } ( \pmb { v } _ { 1 } ^ { \top } \pmb { a } ) ^ { p - 1 } } \leq 2 0 \| g \| / \lambda _ { 1 } } \end{array}$ . Therefore we will need $\| \pmb { g } \| \le 0 . 0 5 \lambda _ { 1 } \widetilde { \epsilon }$ , and then the requirement that $5 \frac { | | g | | } { \lambda _ { 1 } ( v _ { 1 } ^ { \top } a ) ^ { p - 1 } } \leq \widetilde { \epsilon }$ 5 λ (v⊤a)p−1 ≤ eǫ is satisfied. kgk

Altogether in both cases we have: $| \tan \theta ( G ( \mathbf { a } ) , v _ { 1 } ) | \leq 0 . 7 5 | \tan \theta ( G ( \mathbf { a } ) , v _ { 1 } ) | + \widetilde \epsilon .$ Finally if we additionally increase $\| g \|$ by $0 . 0 5 \lambda _ { 1 } ( { \pmb v } _ { 1 } ^ { \top } { \pmb a } ) ^ { p - 1 }$ we will have: $| \tan \theta ( G ( { \pmb a } ) , { \pmb v } _ { 1 } ) | \leq$ 0 $) . 8 | \tan \theta ( G ( \mathbf { 0 } ) , v _ { 1 } ) | + \widetilde \epsilon .$ □

Proof of Corollary 3.16 . As shown in Theorem 3.14 at least one action $^ { a }$ in $A _ { S } , | A _ { S } | \leq$ $\widetilde O ( k )$ satisfies $\tan \theta ( { \pmb a } , { \pmb a } ^ { * } ) \leq \varepsilon$ with a total of $\begin{array} { r } { \widetilde { O } ( \frac { d ^ { p } k } { \lambda _ { 1 } ^ { 2 } \varepsilon ^ { 2 } } ) } \end{array}$ steps. Therefore with Claim C.2 we have to get ${ \widetilde { \varepsilon } } .$ -optimal reward we need $\begin{array} { r } { \widetilde { O } ( \frac { d ^ { p } k } { \lambda _ { 1 } { \widetilde { \varepsilon } } } ) } \end{array}$ steps. Notice the eluder dimension for symmetric polynomials is $d ^ { p }$ and the size of $\mathcal { A } _ { S }$ is at most $\widetilde O ( k )$ . Then by applying Corollary C.5 we get that the total regret is at most $\widetilde { \cal O } ( \sqrt { d ^ { p } k T } + \sqrt { | { \cal A } _ { S } | T } ) = \widetilde { \cal O } ( \sqrt { d ^ { p } k T } )$ . □

# C.2 PAC to Regret Bound Relation.

Claim C.2 (Connecting angle to regret). When $0 < \tan \theta ( { \pmb a } , { \pmb v } _ { 1 } ) \leq \zeta$ , we have regret $r ^ { * } - r ( { \pmb a } ) \leq r ^ { * } \operatorname* { m i n } \{ 2 , p \zeta ^ { 2 } \}$ .

Proof.

$$
\begin{array}{l} | \cos \theta (\boldsymbol {a}, \boldsymbol {v} _ {1}) | = | \boldsymbol {a} ^ {\top} \boldsymbol {v} _ {1} | =: b, \\ | \tan \theta (\pmb {a}, \pmb {v} _ {1}) | = \frac {\sqrt {1 - b ^ {2}}}{b} \leq \zeta \Leftrightarrow b \geq \frac {1}{\sqrt {\zeta^ {2} + 1}}. \\ \Rightarrow r ^ {*} - r (\boldsymbol {a}) \leq \lambda_ {1} - \lambda_ {1} b ^ {p} \\ \leq \lambda_ {1} \left(1 - \left(\zeta^ {2} + 1\right) ^ {- p / 2}\right) \\ = \lambda_ {1} \frac {\left(\zeta^ {2} + 1\right) ^ {p / 2} - 1}{\left(\zeta^ {2} + 1\right) ^ {p / 2}} \\ \leq \lambda_ {1} \left(\left(\zeta^ {2} + 1\right) ^ {p / 2} - 1\right) \quad \text {(s i n c e d e n o m i n a t o r} \left(\zeta^ {2} + 1\right) ^ {p / 2} \geq 1) \\ \leq \lambda_ {1} p \zeta^ {2}, \text {w h e n} \zeta^ {2} \leq 1 / p. \\ \end{array}
$$

Additionally by definition $r ^ { * } - r ( { \pmb a } ) \leq \lambda _ { 1 } - ( - \lambda _ { 1 } ) = 2 \lambda _ { 1 }$ and thus $r ^ { * } - r ( a ) \leq \lambda _ { 1 } \operatorname* { m i n } \{ 2 , p \zeta ^ { 2 } \}$ We now derive the last inequality. When $\zeta \geq 1 / p$ it is trivially true. When $\zeta \leq 1 / p$ , we have $( 1 + \zeta ^ { 2 } ) ^ { p / 2 } \leq 1 + p \zeta ^ { 2 }$ for any $p \geq 2$ . Since the LHS is a convex function for $\zeta$ when $p \geq 2$ and when $\zeta = 0$ LHS=RHS and when $\zeta ^ { 2 } = 1 / p$ LHS is always smaller than RHS $( = 2 )$ .

Notice the argument is straightforward to extend to the setting where the angle is between $^ { a }$ and subspace $V _ { 1 }$ that satisfies $\forall v \in V _ { 1 } , T ( v ) \geq \lambda _ { 1 } - \epsilon$ , then one also get $r ^ { * } - r ( { \pmb a } ) \le \lambda _ { 1 } - ( \lambda _ { 1 } - \epsilon ) b ^ { p } \le \operatorname* { m i n } \{ \lambda _ { 1 } , \lambda _ { 1 } p \zeta ^ { 2 } + \epsilon b ^ { p } \} \le \operatorname* { m i n } \{ \lambda _ { 1 } , \lambda _ { 1 } p \zeta ^ { 2 } + \epsilon \} .$ . □

Claim C.3 (Connecting PAC to Cumulative Regret). Suppose we have an algorithm $a l g ( \zeta )$ that finds $\zeta$ -optimal action $\widehat { \mathbf { a } }$ that satisfies $0 < \tan \theta ( { \pmb a } , { \pmb v } _ { 1 } ) \leq \zeta$ by taking $A \zeta ^ { - a }$ actions.

Here A can depend on any parameters such as $d , \lambda _ { 1 }$ , probability error $\delta$ , etc., that are not $\zeta$ . Then for large enough T , by calling alg with $\zeta = A ^ { \frac { 1 } { a + 2 } } T ^ { - { \frac { 1 } { a + 2 } } } p ^ { - { \frac { 1 } { a + 2 } } }$ and playing its output action a for the remaining actions, one can get a cumulative regret of:

$$
\Re (T) \lesssim T ^ {\frac {a}{a + 2}} p ^ {\frac {a}{a + 2}} A ^ {\frac {2}{a + 2}} r ^ {*}.
$$

Similarly, if an oracle finds $\varepsilon$ -optimal action $\widehat { \mathbf { a } }$ that satisfies $r ^ { * } - r ( \pmb { a } ) \leq \varepsilon$ with $B \varepsilon ^ { - b }$ samples, then by setting $\varepsilon = ( B r ^ { * } / T ) ^ { \frac { 1 } { 1 + b } }$ , and playing the output arm for the remaining actions, one can get cumulative regret of:

$$
\Re (T) \lesssim B ^ {\frac {1}{1 + b}} T ^ {\frac {b}{1 + b}} r ^ {\frac {1}{1 + b}}.
$$

Proof. For the chosen $\zeta$ , write $T _ { 1 } = A \zeta ^ { - a }$ be the number of actions that finds $\zeta$ -optimal action. Therefore $T _ { 1 } ~ = ~ A ^ { \frac { 2 } { a + 2 } } T ^ { \frac { a } { a + 2 } } p ^ { \frac { a } { a + 2 } }$ . First, when $T \geq A p ^ { a / 2 }$ , $\zeta ^ { 2 } \ \leq \ 1 / p$ , namely $r ^ { * } - r ( { \pmb a } ) \leq r ^ { * } p \zeta ^ { 2 }$ . We have:

$$
\begin{array}{l} \Re (T) \leq \sum_ {t = 1} ^ {T _ {1}} 2 r ^ {*} + \sum_ {t = T _ {1} + 1} ^ {T} r ^ {*} p \zeta^ {2} \\ \leq 2 r ^ {*} T _ {1} + T r ^ {*} p \zeta^ {2} \\ \leq 3 T ^ {\frac {a}{a + 2}} p ^ {\frac {a}{a + 2}} A ^ {\frac {2}{a + 2}} r ^ {*}. \\ \end{array}
$$

When $T < A p ^ { a / 2 }$ , it trivially holds that $\Re ( T ) \leq 2 r ^ { * } T < 2 T ^ { { \frac { a } { a + 2 } } } p ^ { { \frac { a } { a + 2 } } } A ^ { { \frac { 2 } { a + 2 } } } r ^ { * } .$ .

![](images/8b9f1d604cce7562270834eb1cd42341eb363f09fe203e22163462c1a6d0b79e.jpg)

# Algorithm 7 UCB (Algorithm 1 in Section 5 of [AJKS19])

1: Input: Stochastic reward function $f$ , failure probability $\delta$ , action set $A$ with finite size $\overline { { K } }$ .   
2: for $t$ from 1 to $T - 1 - K$ do   
3: Execute arm $\begin{array} { r } { I _ { t } ~ = ~ \mathrm { a r g } \operatorname* { m a x } _ { i \in [ K ] } \Big ( \widehat { \mu } ^ { t } ( i ) + \sqrt { \frac { \log ( T K / \delta ) } { N ^ { t } ( i ) } } \Big ) } \end{array}$ . Here $N ^ { t } ( { \pmb a } ) ~ = ~ 1 +$ $\textstyle \sum _ { i = 1 } ^ { t } \mathbf { 1 } \{ I _ { i } = \mathbf { a } \}$ ; and $\begin{array} { r } { \widehat { \mu } ^ { t } ( \pmb { a } ) = \frac { 1 } { N ^ { t } ( \pmb { a } ) } \left( r _ { a } + \sum _ { i = 1 } ^ { t } \pmb { 1 } \{ I _ { i } = \pmb { a } \} r _ { i } \right) } \end{array}$ .   
4: Observe $r _ { I _ { t } }$

Theorem C.4 (Theorem 5.1 from [AJKS19]). With UCB algorithm on action set with size $K$ , we have with probability $1 - \delta$ ,

$$
\Re (T) = \widetilde {O} (\min \{\sqrt {K T} \} + K).
$$

Corollary C.5. With the same setting of Claim C.3, except that now the algorithm $a l g ( \varepsilon )$ finds a set $\mathcal { A }$ of size $S$ where at least one action $\mathbf { \pmb { a } } \in \mathcal { A }$ satisfies $r ^ { * } - f ( \pmb { a } ) \leq \varepsilon$ . Then all argument in Claim C.3 still hold by adding $\widetilde { O } ( \sqrt { S T } )$ on the RHS of each regret bound.

Proof. Suppose we run $a l g$ for $T _ { 1 }$ steps and achieve $\varepsilon$ -optimal reward.

Let $r _ { \varepsilon } : = \operatorname* { m a x } _ { \mathbf { \alpha } } _ { a \in \mathcal { A } } f ( \mathbf { \alpha } \mathbf { a } )$ . Th fore with UCB on utiarm bandit we have: $\scriptstyle \sum _ { t = T _ { 1 } + 1 } ^ { T } r _ { \varepsilon } -$ $f ( \mathbf { } \mathbf { } a _ { t } ) \leq \widetilde { O } ( \sqrt { S T } )$ by Theorem C.4.

From the statement $r _ { \varepsilon } \ge r ^ { * } - \zeta$ . Therefore $\begin{array} { r } { \sum _ { t = T _ { 1 } + 1 } ^ { T } r ^ { * } - f ( \pmb { a } _ { t } ) \leq \widetilde { O } ( \sqrt { S T } ) + \varepsilon ( T - T _ { 1 } ) . } \end{array}$ Therefore

$$
\Re (T) \leq \sum_ {t = 1} ^ {T _ {1}} 2 r ^ {*} + \varepsilon (T - T _ {1}) + \widetilde {O} (\sqrt {S T}).
$$

With the same choices of $T _ { 1 }$ in Claim C.3, the same conclusion still holds with an additional term of $\widetilde { O } ( \sqrt { S T } )$ .

For symmetric tensor problems the set size is $\widetilde O ( k )$ and therefore we will have an additional $\sqrt { k T }$ term which will be subsumed in our regret bound.

# C.3 Variance and Noise Concentration

Lemma C.6 (Vector Bernstein; adapted from Theorem 7.3.1 in [Tro12]). Consider a finite sequence $\{ \pmb { x } _ { k } \} _ { k = 1 } ^ { n }$ be i.i.d randomly generated samples, $\boldsymbol { x } _ { k } \in \mathbb { R } ^ { d }$ , and assume that $\mathbb { E } [ { \pmb x } _ { k } ] =$ 0 $, \| \pmb { x } _ { k } \| \le L ,$ and covariance matrix of $x _ { k }$ is $\Sigma$ . Then it satisfies that when $n \geq \log d / \delta$ , we have:

$$
\left\| \frac {\sum_ {i = 1} ^ {n} \boldsymbol {x} _ {i}}{n} \right\| \leq C \sqrt {\frac {(\| \Sigma \| + L ^ {2}) \log d / \delta}{n}},
$$

with probability $1 - \delta$ .

Claim C.7 (Noise concentration). Let independent samples $ { \boldsymbol { z } } _ { i } \sim \mathcal { N } ( 0 , 1 / m I _ { d } )$ and $\epsilon _ { i } \sim$ $\mathcal { N } ( 0 , 1 )$ . With probability $1 - \delta , \delta \in ( 0 , 1 )$ :

$$
\left\| \frac {m}{n} \sum_ {i = 1} ^ {n} \epsilon_ {i} \pmb {z} _ {i} \right\| \leq C \sqrt {\frac {m \log (n / \delta) (d + \log (n / \delta)) \log (d / \delta)}{n}}
$$

$$
\left| \frac {m}{n} \sum_ {i = 1} ^ {n} \epsilon_ {i} \boldsymbol {z} _ {i} ^ {\top} \boldsymbol {v} _ {1} \right| \leq C ^ {\prime} \sqrt {\frac {m \log^ {2} (n / \delta) \log (d / \delta)}{n}}.
$$

Proof. We use the Vector Bernstein Lemma C.6. The covariance matrix for ${ \pmb x } _ { i } = \epsilon _ { i } { \pmb z } _ { i }$ satisfies $\mathbb { E } [ \pmb { x } _ { i } \pmb { x } _ { i } ^ { \top } ] = 1 / m I _ { d }$ . $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ is mean zero. $\| \epsilon _ { i } \pmb { z } _ { i } \| ^ { 2 } = \epsilon _ { i } ^ { 2 } \| \pmb { z } _ { i } \| ^ { 2 }$ . Notice $\epsilon _ { i } ^ { 2 } \sim \chi ( 1 ) \lesssim$ $1 + \log ( 1 / \delta )$ and $m z _ { i } ^ { \top } z _ { i } \sim \chi ( d ) \lesssim d + \log ( 1 / \delta )$ . Therefore by directly applying Vector Bernstein $\begin{array} { r } { \| \epsilon _ { i } z _ { i } \| \le c \sqrt { \frac { ( 1 + \log ( 1 / \delta ) ) ( d + \log ( 1 / \delta ) ) } { m } } } \end{array}$ with probability $1 - \delta$ . By union bound we have: for all i, $\begin{array} { r } { \| \epsilon _ { i } z _ { i } \| \le c \sqrt { \frac { \log ( n / \delta ) ( d + \log ( n / \delta ) ) } { m } } } \end{array}$ log(n/δ)(d+log(n/δ)) with probability 1 − δ. Therefore $1 - \delta$

$$
\left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \epsilon_ {i} \pmb {z} _ {i} \right\| \leq C \sqrt {\frac {\log (n / \delta) (d + \log (n / \delta)) \log (d / \delta)}{m n}},
$$

with probability $1 - \delta$ . Similarly

$$
\begin{array}{l} \left| \frac {1}{n} \sum_ {i = 1} ^ {n} \epsilon_ {i} \boldsymbol {z} _ {i} ^ {\top} \boldsymbol {v} _ {1} \right| \leq C \sqrt {\frac {\log (n / \delta) (1 + \log (n / \delta)) \log (d / \delta)}{m n}} \\ = C ^ {\prime} \sqrt {\frac {\log^ {2} (n / \delta) \log (d / \delta)}{m n}}, \\ \end{array}
$$

![](images/eb04cfed1f9a9daaff45f616b8a0242f539c308c5e739844c439599147eac43b.jpg)

Claim C.8. Let $\{ z _ { i } \} _ { i = 1 } ^ { n }$ be i.i.d samples from $\mathcal { N } ( 0 , 1 / m I _ { d } )$ . Let $\begin{array} { r } { g _ { n , s } ( j ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( z _ { i } ^ { \top } v _ { j } ) ^ { s } z _ { i } - } \end{array}$ $\mathbb { E } [ ( z ^ { \top } { \pmb v } _ { j } ) ^ { s } z ]$ . We have:

$$
\begin{array}{l} \| g _ {n, 0} (j) \| \lesssim \sqrt {\frac {d + \log (1 / \delta)}{n m}}, \\ | \boldsymbol {v} _ {1} ^ {\top} g _ {n, 0} (j) | \lesssim \sqrt {\frac {1 + \log (1 / \delta)}{n m}}, \\ | \boldsymbol {v} _ {1} ^ {\top} g _ {n, 1} (j) | \leq \| g _ {n, 1} (j) \| \lesssim \sqrt {\frac {d + \log (1 / \delta)}{m ^ {2} n}}, w h e n n \geq d \log (1 / \delta), \\ | \boldsymbol {v} _ {1} ^ {\top} g _ {n, s} (j) | \leq \| g _ {n, s} (j) \| \lesssim \sqrt {\frac {\log (d / \delta)}{d ^ {s} n}}, \text {w h e n} n \geq \log (d / \delta), m \geq c _ {0} d \log (n / \delta), s \geq 2. \\ \end{array}
$$

For any $j \in [ k ]$ .

We mostly care about the correct concentration for smaller $s$ . For larger $s$ a very loose bound will already suffice our requirement.

Proof of Claim C.8. For $s = 0$ $\begin{array} { r } { , n m \| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \| ^ { 2 } \sim \chi ( d ) } \end{array}$ , therefore $\begin{array} { r } { \| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \| \lesssim \sqrt { \frac { d + \log ( 1 / \delta ) } { n m } } . } \end{array}$ $\begin{array} { r } { n m ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } ^ { \top } \pmb { v } _ { 1 } ) ^ { 2 } \sim \chi ( 1 ) } \end{array}$ . Therefore $\begin{array} { r } { | \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } ^ { \top } \pmb { v } _ { 1 } | \lesssim \sqrt { \frac { 1 + \log ( 1 / \delta ) } { n m } } } \end{array}$ .

For $s \ = \ 1$ , due to standard concentration for covariance matrices (see e.g. [Tro12, $\mathrm { D H K } ^ { + } 2 0 ]$ ), we have:

$$
m \| (\frac {1}{n} \sum_ {i = 1} ^ {n} \boldsymbol {z} _ {i} \boldsymbol {z} _ {i} ^ {\top} - \mathbb {E} [ \boldsymbol {z} \boldsymbol {z} ^ {\top} ]) \| \leq \max  \{\sqrt {\frac {d + \log (2 / \delta)}{n}}, \frac {d + \log (2 / \delta)}{n} \}.
$$

Therefore when $n \geq d \log ( 1 / \delta )$ , both results

$$
\begin{array}{l} \| g _ {n, 1} (j) \| \lesssim \sqrt {\frac {d + \log (1 / \delta)}{m ^ {2} n}} \| \boldsymbol {v} _ {j} \|, \\ = \sqrt {\frac {d + \log (1 / \delta)}{m ^ {2} n}}, \text {a n d} \\ \end{array}
$$

$$
\begin{array}{l} \| \boldsymbol {v} _ {1} ^ {\top} g _ {n, 1} (j) \| \lesssim \sqrt {\frac {d + \log (1 / \delta)}{m ^ {2} n}} \| \boldsymbol {v} _ {1} \| \| \boldsymbol {v} _ {j} \| \\ = \sqrt {\frac {d + \log (1 / \delta)}{m ^ {2} n}} \\ \end{array}
$$

hold.

For larger $s \geq 2$ , with probability $1 - \delta$ , $| z _ { i } ^ { \top } v _ { j } | \leq C \sqrt { \log ( n / \delta ) / m } = C c _ { 0 } / \sqrt { d } \leq$ $1 / { \sqrt { d } }$ . When $m \geq c _ { 0 } d \log ( n / \delta )$ , for small enough $c _ { 0 }$ we have $| z _ { i } ^ { \top } v _ { j } | \leq 1 / \sqrt { d }$ and $\| z _ { i } \| \leq 1$ for all $i \in [ n ]$ . Therefore $\| ( \pmb { z } _ { i } ^ { \top } \pmb { v } _ { j } ) ^ { s } \pmb { z } _ { i } \| \le d ^ { - s / 2 }$ We can use vector Bernstein, i.e., Lemma C.6 to get:

$$
\left\| g _ {n, s} (j) \right\| \leq C _ {1} \sqrt {\frac {\log (d / \delta)}{d ^ {s} n}}.
$$

Therefore we have:

$$
| g _ {n, s} (j) ^ {\top} \pmb {v} _ {1} | \leq C _ {1} \sqrt {\frac {\log (d / \delta)}{d ^ {s} n}}.
$$

□

# C.4 Omitted Details for Asymmetric Tensors

Lemma C.9 (Asymmetric Tensor Initialization). With probability $1 - \delta _ { : }$ , with $L = \widetilde { \Theta } ( ( 2 k ) ^ { p } \log ^ { p } ( p / \delta ) )$ random initializations ${ \mathcal A } _ { 0 } = \{ { \pmb a } _ { 0 } ^ { ( 0 ) } , { \pmb a } _ { 0 } ^ { ( 1 ) } , \cdot \cdot \cdot { \pmb a } _ { 0 } ^ { ( L ) } \}$ a(L)0 }, there exists an initialization a0 ∈ A0 ${ \pmb a } _ { 0 } \in \mathcal { A } _ { 0 }$ that satisfies:

$$
\begin{array}{l} \alpha \boldsymbol {a} _ {0} (s) ^ {\top} \boldsymbol {v} _ {1} ^ {(s)} \geq | \boldsymbol {a} _ {0} (s) ^ {\top} \boldsymbol {v} _ {j} ^ {(s)} |, \forall j \geq 2 \& j \in [ k ], \forall s \in [ p ], \tag {8} \\ \boldsymbol {a} _ {0} (s) ^ {\top} \boldsymbol {v} _ {1} ^ {(s)} \geq 1 / \sqrt {d}. \\ \end{array}
$$

with some constant $\alpha < 1$ .

Proof. This lemma simply comes from applying Lemma 3.19 for $p$ times and we need $\geq 2 k \log _ { 2 } ( p \delta )$ to ensure the condition for each $\pmb { a } _ { 0 } ( s ) , s \in [ p ]$ holds. Therefore together we will need $( 2 k \log _ { 2 } ( p / \delta ) ) ^ { p }$ samples. □

Lemma C.10 (Asymmetric tensor progress). For each a that satisfies Eqn. (8) with constant $\alpha < 1$ , we have:

$$
\tan \theta (\pmb {T} (\pmb {a} (1), \dots \pmb {a} (s - 1), \pmb {I}, \pmb {a} (s + 1), \dots \pmb {a} (p)), \pmb {v} _ {1} ^ {(s)}) \leq \alpha \tan \theta (\pmb {a} _ {j}, \pmb {v} _ {1} ^ {(j)}),
$$

for any $j$ that is in $[ p ]$ but is not s. When $n \geq \Theta ( d ^ { p } \log ( d / \delta ) \log ^ { 3 } ( n / \delta ) / { \widetilde \varepsilon } ^ { 2 } )$ and $m =$ $\Theta ( d \log ( n / \delta ) )$ ), we have:

$$
\begin{array}{l} \tan \theta (\boldsymbol {T} (\boldsymbol {a} (1), \dots \boldsymbol {a} (s - 1), \boldsymbol {I}, \boldsymbol {a} (s + 1), \dots \boldsymbol {a} (p)), \boldsymbol {v} _ {1} ^ {(s)}) \\ \leq (1 + \alpha) / 2 \tan \theta (\pmb {a} _ {j}, \pmb {v} _ {1} ^ {(j)}) + \widetilde {\varepsilon}, \forall j \in [ p ] \& j \neq s. \\ \end{array}
$$

The remaining proof is a simpler version for the symmetric tensor setting on conducting noisy power method with the good initialization and iterative progress.

Finally due to the good initialization that satisfies (8) and together with Lemma C.10 we can finish the proof for Theorem 3.22.

Algorithm 8 Phased elimination with alternating tensor product.   
1: Input: Stochastic reward $r:(B_1^d)^{\otimes p}\to \mathbb{R}$ of polynomial degree $p$ , failure probability $\delta$ error $\varepsilon$ 2: Initialization: $L_{0} = C_{L}k\log (1 / \delta)$ Total number of stages $S = C_S[\log (1 / \varepsilon)] + 1$ $\mathcal{A}_0 = \{\pmb {a}_0^{(1)},\pmb {a}_0^{(2)},\dots \pmb {a}_0^{(L_0)}\} \subset (B_1^d)^{\otimes p}$ where each $\pmb {a}_0^{(l)}(j),j\in [p]$ is uniformly sampled on the unit sphere $\mathbb{S}^{d - 1}$ $\widetilde{\varepsilon}_0 = 1$ 3: for s from 1 to S do   
4: $\widetilde{\varepsilon}_{s}\gets \widetilde{\varepsilon}_{s - 1} / 2,n_{s}\gets C_{n}d^{p}\log (d / \delta) / \widetilde{\varepsilon}_{s}^{2},n_{s}\gets n_{s}\cdot \log^{3}(n_{s} / \delta),m_{s}\gets C_{m}d\log (n / \delta),$ $\mathcal{A}_s = \emptyset$ 5: for l from 1 to $L_{s - 1}$ do   
6: Tensor product update:   
7: Locate current arm $\widetilde{\pmb{a}} = \pmb{a}_{s - 1}^{(l)}$ 8: for $[(\lambda_1 / \Delta)\log (2d)]$ times do   
9: for j from 1 to p do   
10: Sample $z_{i}\sim \mathcal{N}(0,1 / m_{s}I_{d}),i = 1,2,\dots n_{s}$ 11: Calculate tentative arm $a_i\gets \widetilde{a},a_i(j) = (1 - \widetilde{\varepsilon}_s)\widetilde{a} (j) + \widetilde{\varepsilon}_sz_i$ 12: Conduct estimation $\pmb {y}\gets 1 / n_s\sum_{i = 1}^{n_s}r_{\epsilon_i}(\pmb {a}_i)\pmb {z}_i$ 13: Update the current arm $\widetilde{\pmb{a}} (j)\gets \pmb {y} / \| \pmb {y}\|$ 14: Estimate the expected reward for a through ns samples: $r_n =$ $1 / n_{s}\sum_{i = 1}^{n_{s}}r_{\epsilon_{i}}(\widetilde{\boldsymbol{a}}).$ 15: Candidate Elimination: if $r_n\geq \lambda_1(1 - p\widetilde{\varepsilon}_s)$ then   
17: Keep the arm $\mathcal{A}_s\gets \mathcal{A}_s\cup \{\widetilde{\boldsymbol{a}}\}$ 18: Label the arms: $L_{s} = |\mathcal{A}_{s}|,\mathcal{A}_{t} = :|\mathbf{a}_{s}^{(1)},\dots \mathbf{a}_{s}^{(L_{s})}\rangle .$ 19: Run Algorithm 7 with $\mathcal{A}_S$

# D Proof of Theorem 3.26

# D.1 Additional Notations

Here, we briefly introduce complex and real algebraic geometry. This section is based on [Mil17, Sha13, BCR13, WX19].

An (affine) algebraic variety is the common zero loci of a set of polynomials, defined as $V = Z ( S ) = \{ \pmb { x } \in \mathbb { C } ^ { n } : f ( \pmb { x } ) = 0 , \forall f \in S \} \subseteq \mathbb { A } ^ { n } = \mathbb { C } ^ { n }$ for some $S \subseteq \mathbb { C } [ x _ { 1 } , \cdot \cdot \cdot , x _ { n } ]$ . A projective variety $U$ is a subset of $\mathbb { P } ^ { n } = ( \mathbb { C } ^ { n + 1 } \setminus \{ 0 \} ) / \sim$ , where $( x _ { 0 } , \cdots , x _ { n } ) \ \sim$ $k ( x _ { 0 } , \cdots , x _ { n } )$ for $k \neq 0$ and $S$ is a set of homogeneous polynomials of $( n + 1 )$ variables.

For an affine variety $V$ , its projectivization is the variety $\mathbb { P } ( V ) = \{ [ { \pmb x } ] : x \in V \} \subseteq$ $\mathbb P ^ { n - 1 }$ , where $[ x ]$ is the line corresponding to $_ { x }$ .

The Zariski topology is the topology generated by taking all varieties to be the closed sets.

A set is irreducible if it is not the union of two proper closed subsets.

A variety is irreducible if and only if it is irreducible under the Zariski topology.

The algebraic dimension $d = \dim V$ of a variety $V$ is defined as the length of the longest chain $V _ { 0 } \subset V _ { 1 } \subset \cdots \subset V _ { d } = V $ , such that each $V _ { i }$ is irreducible.

A variety $V$ is said to be admissible to a set of linear functions $\{ \ell _ { \alpha } : \mathbb { C } ^ { d } \to \mathbb { C } \} _ { \alpha \in I } \}$ , if for every $\ell _ { \alpha }$ , we have $\dim ( V \cap \{ \pmb { x } \in \mathbb { C } ^ { d } : \ell _ { \alpha } ( \pmb { x } ) = 0 \} ) < \dim V .$ .

A map $f = ( f _ { 1 } , \cdot \cdot \cdot , f _ { m } ) : \mathbb { A } ^ { n } \to \mathbb { A } ^ { m }$ is regular if each $f _ { i }$ is a polynomial.

A algebraic set is the common real zero loci of a set of polynomials.

For a complex variety $V \subseteq \mathbb { A } ^ { n }$ , its real points form a algebraic set $V _ { \mathbb { R } }$

For an algebraic set $V _ { \mathbb { R } }$ , its real dimension $d = \dim _ { \mathbb { R } } V _ { \mathbb { R } }$ is the maximum number $d$ such that $V _ { \mathbb { R } }$ is locally semi-algebraically homeomorphic to the unit cube $( 0 , 1 ) ^ { d }$ , details can be found in [BCR13].

# D.2 Proof of Sample Complexity

Lemma D.1 ([WX19], Theorem 3.2). For $i = 1 , \dots , T ,$ , let $L _ { i } : \mathbb { C } ^ { n } \times \mathbb { C } ^ { m } \to \mathbb { C }$ be bilinear functions and $V _ { i }$ be varieties given by homogeneous polynomials in $\mathbb { C } ^ { n }$ . Let $V =$ $V _ { 1 } \times \cdots \times V _ { T } \subseteq ( \mathbb { C } ^ { n } ) ^ { N }$ . Let $W \subseteq \mathbb { C } ^ { m }$ be a variety given by homogeneous polynomials. In addition, we assume $V _ { i }$ is admissible with respect to the linear functions $\{ f ^ { w } ( \cdot ) =$ $L _ { i } ( \cdot , \pmb { w } ) : \pmb { w } \in W \setminus \{ 0 \} \}$ . When $T \geq \dim W$ , let $\delta = T - \dim W + 1 \geq 1$ . Then there exists a subvariety $Z \subseteq V w i t h \dim Z \leq \dim V - \delta$ such that for any $( { \pmb x } _ { 1 } , . . . , { \pmb x } _ { T } ) \in V \backslash Z$ and $\pmb { w } \in W$ , if $L _ { 1 } ( { \pmb x } _ { 1 } , { \pmb w } ) = \dots = L _ { T } ( { \pmb x } _ { T } , { \pmb w } ) = 0 .$ , then ${ \pmb w } = 0$ .

Lemma D.2 ([WX19], Lemma 3.1). Let $V$ be an algebraic variety in $\mathbb { C } ^ { d }$ . Then $\dim _ { \mathbb { R } } V _ { \mathbb { R } } \leq$ $\dim V$ .

Lemma D.3. Let $W$ be a vector space. For vectors $\pmb { x } _ { 1 } , \cdots , \pmb { x } _ { T }$ , if the map $f : w \mapsto \mapsto$ $( \langle \pmb { x } _ { 1 } , \pmb { w } \rangle , \dots , \langle \pmb { x } _ { T } , \pmb { w } \rangle )$ is not injective over $W - W : = \{ w _ { 1 } - w _ { 2 } : w _ { 1 } , w _ { 2 } \in W \} _ { } \nonumber$ , then there exists $v \in W$ such that $f ( \pmb { v } ) = 0$ .

Proof. Suppose $f ( { \pmb w } _ { 1 } ) = f ( { \pmb w } _ { 2 } )$ . Let $\pmb { v } = \pmb { w } _ { 1 } - \pmb { w } _ { 2 }$ . Then ${ \pmb v } \in { \cal W } - { \cal W }$ and $f ( \pmb { v } ) =$ $f ( { \pmb w } _ { 1 } ) - f ( { \pmb w } _ { 2 } ) = 0$ .

Definition D.4 (Tensorization). Let $f$ be a polynomial of $x _ { 1 } , \cdots , x _ { d }$ with degree $\deg f \leq p$ Then every $p$ -tensor ${ \pmb W } _ { f }$ satisfying $\langle { \pmb { \mathsf { W } } } _ { f } , { \pmb { \mathsf { X } } } _ { \pmb { x } } \rangle = f ( { \pmb { x } } )$ is said to be a tensorization of the polynomial $f$ , where ${ \pmb X } _ { x }$ is the tensorization of $_ { x }$ itself:

$$
\mathbf {X} _ {\boldsymbol {x}} = \binom {1} {\boldsymbol {x}} ^ {\otimes p}. \tag {9}
$$

Let $\mathcal { F }$ be a class of polynomials. A variety of tensorization of $\mathcal { F }$ is defined to be an irredicuble closed variety defined by homogeneous polynomials $W$ , such that for every $f \in { \mathcal { F } }$ , there is a tensorization ${ \pmb W } _ { f }$ of $f$ , such that $W \ni \pmb { W } _ { f }$ contains its tensorization. Note that neither tensorization of $f$ nor variety of tensorization of $\mathcal { F }$ is unique.

We define the variety of tensorization of $_ { x }$ as follows. (Note that this is uniquely defined.) Consider the regular map

$$
\varphi_ {1}: \mathbb {C} ^ {d} \rightarrow \mathbb {C} ^ {(d + 1) ^ {p}}, \quad \boldsymbol {x} \mapsto \binom {1} {\boldsymbol {x}} ^ {\otimes p}, \tag {10}
$$

the tensorization of $_ { x }$ is defined as $V _ { i } = \mathbb { P } ( \overline { { \mathrm { I m } \varphi _ { 1 } } } )$ .

Note that $V _ { i }$ is irreducible because $\varphi _ { 1 }$ is regular and $\mathbb { C } ^ { d }$ is irreducible. By [Mil17, Theorem 9.9], its dimension is given by

$$
\dim V _ {i} \leq \dim \overline {{\operatorname {I m} \varphi_ {1}}} + 1 \leq \dim \mathbb {C} ^ {d} + 1 = d + 1. \tag {11}
$$

Lemma D.5. For any non-zero polynomial $f \neq 0$ with $\deg f \leq p$ . Let $W _ { f }$ be a tensorization of $f$ . Then $V _ { i }$ is admissible with respect to $\{ L _ { i } ( \cdot ) = \langle \cdot , W _ { f } \rangle \}$ .

Proof. Since $V _ { i }$ is irredicuble and $L _ { i }$ is a linear function, it suffices to verify that $\langle \pmb { \mathrm { X } } _ { \pmb { x } } , W _ { f } \rangle \neq$ 0 [WX19]. But according to Definition D.4, $\langle \pmb { \mathrm { X } } _ { x } , W _ { f } \rangle \neq 0$ is equivalent to

$$
f (\boldsymbol {x}) = \left\langle \mathbf {W} _ {f}, \binom {1} {\boldsymbol {x}} ^ {\otimes p} \right\rangle \neq 0. \tag {12}
$$

Since $f \neq 0$ , we must have $f ( { \pmb x } ) \neq 0$ for some $_ { x }$ , which gives a non-zero ${ \pmb X } _ { { \pmb x } } \neq 0$ for the above equation: $\langle \pmb { \mathrm { X } } _ { x } , W _ { f } \rangle \neq 0$ , and we conclude that $V _ { i }$ is admissible. □

Lemma D.6. Let $V \subset \mathbb { C } ^ { n }$ be a (Zariski) closed proper subset, $V \neq \mathbb { C } ^ { n }$ . Then $V$ is a null set, i.e. it has (Lebesgue) measure zero.

Proof. Suppose $V = Z ( S )$ is the vanishing set for some $S \subseteq \mathbb { C } [ x _ { 1 } , \cdot \cdot \cdot , x _ { n } ]$ . Since $V \neq$ $\mathbb { C } ^ { n }$ , let $f \in S$ , we have $V \subseteq Z ( f )$ , so it suffices to show $\operatorname { L e b } ( Z ( f ) ) = 0$ , which is because $Z ( f ) = f ^ { - 1 } ( 0 ) , { \mathrm { L e b } } ( \{ 0 \} ) = 0 , f$ is a continuous function (under Euclidean topology), and $\operatorname { L e b } ( \{ { \pmb x } : \nabla f ( { \pmb x } ) = 0 \} ) = 0$ . □

Theorem D.7. Assume that the reward function class is a class of polynomials $\mathcal { F }$ . Let $W$ be (one of) its variety of tensorization. If we sample $T \geq \dim W$ times, and the sample points satisfying $( \pmb { x } _ { 1 } , \cdot \cdot \cdot , \pmb { x } _ { T } ) \in ( \mathbb { C } ^ { d } ) ^ { T } \setminus Z$ for some null set $Z$ . Then we can uniquely determine the reward function $f$ from the observed rewards $( f ( \pmb { x } _ { 1 } ) , \cdot \cdot \cdot , f ( \pmb { x } _ { T } ) )$ .

Proof. Let $n = m = ( d + 1 ) ^ { p }$ $n = m = ( d + 1 ) ^ { p } , L _ { i } ( { \pmb x } , { \pmb w } ) = \langle { \pmb x } , { \pmb w } \rangle , V = V _ { 1 } \times \dots \times V _ { T } .$ , where $V _ { i }$ is as in Definition D.4. By [Sha13, Example 1.33], we have $\dim V \leq ( d + 1 ) T$ . Since $W$ is a vareity of tensorization, by Lemma D.5, $V _ { i }$ is admissible with respect to $\{ L _ { i } ( \cdot , \pmb { \mathsf { W } } ) : \pmb { \mathsf { W } } \in \qquad $ $W \}$ .

We are now ready to apply Lemma D.1, which gives that when $T \geq \dim W$ , there exists subvariety $Z \subset V$ with $\dim Z < \dim V \leq r T .$ , and for any $( { \pmb X } _ { 1 } , \cdot \cdot \cdot , { \pmb X } _ { T } ) \in V \setminus Z$ and any $\pmb { W } \in W$ , if $\langle { \pmb { \mathsf { X } } } _ { 1 } , { \pmb { \mathsf { W } } } \rangle = \cdots = \langle { \pmb { \mathsf { X } } } _ { T } , { \pmb { \mathsf { W } } } \rangle = 0$ , then ${ \pmb W } = 0$ . By Lemma D.3, we have

for every $( { \pmb X } _ { 1 } , \cdot \cdot \cdot , { \pmb X } _ { T } ) \in V \setminus Z$ , the map ${ \pmb W } \mapsto ( \langle { \pmb X } _ { 1 } , { \pmb W } \rangle , \allowbreak , \allowbreak \cdot \cdot \mathrm { ~ , ~ } \langle { \pmb X } _ { T } , { \pmb W } \rangle )$ is injective, so ${ \pmb W } _ { f }$ and thus $f$ can be uniquely recovered from the observed rewards.

Finally, we show that $( \varphi _ { 1 } ^ { - 1 } \times \cdot \cdot \cdot \times \varphi _ { 1 } ^ { - 1 } ) ( Z )$ is a null set, where $\varphi _ { 1 }$ is as in (10). According to the proof of Lemma D.1 by [WX19], we find that $Z$ is also defined by homogeneous polynomials. We take the slice $Z ^ { \prime } = \{ \pmb { x } \in Z : x _ { 1 1 } = \cdots = x _ { T 1 } = 1 \} , V ^ { \prime } = \{ \pmb { x } \in \nonumber$ $V : x _ { 1 1 } = \cdot \cdot \cdot = x _ { T 1 } = 1 \}$ , (here $x _ { i j }$ is the $j$ -th coordinate of $\mathbf { \Delta } \mathbf { x } _ { i }$ ), then $Z ^ { \prime } , V ^ { \prime }$ are varieties. Since $\dim Z < \dim V$ , we have dim $Z ^ { \prime } = \dim Z - T < \dim V - T = \dim V ^ { \prime }$ and $Z ^ { \prime } \subset V ^ { \prime }$ .

Now consider the regular map $\varphi _ { 1 } ^ { \prime } : V ^ { \prime } \to ( \mathbb { C } ^ { d } ) ^ { T }$ ,

$$
\left(\left( \begin{array}{l} 1 \\ \boldsymbol {x} _ {1} \end{array} \right) ^ {\otimes p}, \dots , \left( \begin{array}{l} 1 \\ \boldsymbol {x} _ {1} \end{array} \right) ^ {\otimes p}\right) \mapsto (\boldsymbol {x} _ {1}, \dots , \boldsymbol {x} _ {T}). \tag {13}
$$

Then $\varphi _ { 1 } ^ { \prime } ( Z ^ { \prime } ) , \varphi _ { 1 } ^ { \prime } ( V ^ { \prime } )$ are both varieties. By [Mil17, Lemma 9.9], we have $\dim \overline { { \varphi _ { 1 } ^ { \prime } ( Z ^ { \prime } ) } } \leq$ $\mathrm { d i m } Z$ . Since $\varphi _ { 1 } ^ { \prime } ( V ) = ( \mathbb { C } ^ { d } ) ^ { T }$ and $\dim \varphi _ { 1 } ^ { \prime } ( V ^ { \prime } ) \leq \dim V ^ { \prime } = \dim V - T \leq ( d + 1 ) T - T$ , we have $\dim V = \dim \varphi _ { 1 } ^ { \prime } ( V ) = d T$ and as a result, $\dim { \overline { { \varphi _ { 1 } ^ { \prime } ( Z ) } } } \leq \dim Z < \dim V =$ $d T$ . By Lemma D.6, $\overline { { \varphi _ { 1 } ^ { \prime } ( Z ) } }$ is a null set. Since $( { \pmb x } _ { 1 } , \cdot \cdot \cdot , { \pmb x } _ { T } ) \not \in \overline { { { \varphi } _ { 4 } ^ { \prime } ( Z ) } }$ implies that $( \varphi _ { 1 } ( \pmb { x } _ { 1 } ) , \cdot \cdot \cdot , \varphi _ { 1 } ( \pmb { x } _ { T } ) ) \notin Z$ , we conclude the proof.

Theorem D.7 is stated for complex sample points. Next we extend it to the real case.

Lemma D.8. In Lemma D.1, if we assume in addition that $\dim _ { \mathbb { R } } V _ { \mathbb { R } } = \dim V$ , then the conclusion can be enhanced to ensure that $Z$ is a real subvariety and $\dim _ { \mathbb { R } } Z < \dim _ { \mathbb { R } } V _ { \mathbb { R } }$ .

Lemma D.9. Let $V \subset \mathbb { R } ^ { n }$ be a (Zariski) closed proper subset, $V \neq \mathbb { R } ^ { n }$ . Then $V$ is a null set.

The proof of Lemma D.9 is the same as that of Lemma D.6.

Theorem D.10. We can additionally assume $\pmb { x } _ { i } \in \mathbb { R } ^ { d }$ in Theorem D.7.

Proof. We verify that $\dim V = \dim _ { \mathbb { R } } V _ { \mathbb { R } }$ , where $V$ is defined in the proof of Theorem D.7, but this follows clearly by [BCR13, Corollary 2.8.2]. We conclude the proof by applying Lemma D.8. □

Finally, we apply Theorem D.10 to two concrete classes of polynomials, namely Examples 3.27 and 3.28. For Example 3.27, we construct its variety of tensorization of $\mathcal { R } _ { \mathcal { V } }$ as follows. We first construct the tensorization of each polynomial. We define

$$
\mathbf {W} _ {f} = \sum_ {i = 1} ^ {r} a _ {i} \binom {1} {\boldsymbol {w} _ {i}} ^ {\otimes p _ {i}} \otimes \binom {1} {0} ^ {\otimes (p - p _ {i})}. \tag {14}
$$

Next we construct the variety of tensorization W . Consider the map $\varphi _ { 2 } : ( \mathbb { C } ^ { d } ) ^ { r } \to \mathbb { C } ^ { ( d + 1 ) ^ { p } }$

$$
\varphi_ {2} \left(\boldsymbol {w} _ {1}, \dots , \boldsymbol {w} _ {r}\right) = \sum_ {i = 1} ^ {r} \binom {1} {\boldsymbol {w} _ {i}} ^ {\otimes p _ {i}} \otimes \binom {1} {0} ^ {\otimes (p - p _ {i})}, \tag {15}
$$

and let $Y = \mathbb { P } ( \overline { { \mathrm { I m } \varphi _ { 2 } } } )$ . Similar to $V _ { i }$ , we can prove that $Y$ is an irredicuble closed variety defined by homogeneous polynomials with $\mathrm { d i m } Y \leq d r + 1 \quad$ . Next consider the map $\varphi _ { 2 } ^ { \prime }$ : $( \mathbb { C } ^ { d } ) ^ { 2 r } \to \mathbb { C } ^ { ( d + 1 ) ^ { p } }$ ,

$$
\varphi_ {2} ^ {\prime} \left(\boldsymbol {w} _ {1}, \dots , \boldsymbol {w} _ {2 r}\right) = \varphi_ {2} \left(\boldsymbol {w} _ {1}, \dots , \boldsymbol {w} _ {r}\right) - \varphi_ {2} \left(\boldsymbol {w} _ {r + 1}, \dots , \boldsymbol {w} _ {2 r}\right) \tag {16}
$$

and let $W = \mathbb { P } ( \overline { { \mathrm { I m } \varphi _ { 2 } ^ { \prime } } } )$ . Similar to $Y$ , we can prove that $W$ is an irredicuble closed variety defined by homogeneous polynomials with $\dim W \leq 2 d r + 1$ . Together with Theorem D.10, we can conlude that the optimal action for Example 3.27 can be uniquely determined using at most $2 d r + 1$ samples.

For Example 3.28, we construct $W$ as follows. Let

$$
\boldsymbol {U} = \left( \begin{array}{c c c} \boldsymbol {w} _ {1} & \dots & \boldsymbol {w} _ {k} \end{array} \right), \qquad q = \sum_ {I \subseteq [ k ]: | I | \leq p} a _ {I} x ^ {I},
$$

then we construct the tensorization of each polynomial by

$$
\mathbf {W} _ {f} = \sum_ {I \subseteq [ k ]: | I | \leq p} a _ {I} \bigotimes_ {i \in I} \binom {1} {\boldsymbol {w} _ {i}} \otimes \binom {1} {0} ^ {\otimes (p - | I |)}. \tag {17}
$$

Then we have $f ( \pmb { x } ) = \langle \pmb { \mathsf { W } } _ { f } , \pmb { \mathsf { X } } _ { \pmb { x } } \rangle$ . To reduce the dimension of $W$ and get better sample complexity bound, we construct in a manner slightly different from what we did for Example 3.27. Consider the map $\varphi _ { 3 } : ( \mathbb { C } ^ { d } ) ^ { k } \times \mathbb { C } ^ { ( k + 1 ) ^ { p } } \to \mathbb { C } ^ { ( d + 1 ) ^ { p } }$ ,

$$
\left(\boldsymbol {w} _ {1}, \dots , \boldsymbol {w} _ {k}\right) \times \left(a _ {I}: I \subseteq [ k ], | I | \leq p\right) \mapsto \boldsymbol {W} _ {f}, \tag {18}
$$

where ${ \pmb W } _ { f }$ is as defined in (17). Let $Y = \mathbb { P } ( \overline { { \mathrm { I m } \varphi _ { 3 } } } )$ and $W = \mathbb { P } ( \overline { { \mathrm { I m } \varphi _ { 3 } - \mathrm { I m } \varphi _ { 3 } } } )$ . We end up with $\dim Y \ = \leq \ d k + ( k + 1 ) ^ { p } + 1 , \dim W \ \leq \ 2 ( d k + ( k + 1 ) ^ { p } ) + 1$ . So we conlude that the optimal action for Example 3.28 can be uniquely determined using at most $2 d k + 2 ( k + 1 ) ^ { p } + 1$ samples.

# E Omitted Proof for Lower Bounds with UCB Algorithms

In this section, we provide the proof for the lower bounds for learning with UCB algorithms in Subsection 3.4.2.

Notation Recall that we use $\Lambda$ to denote the subset of the $p$ -th multi-indices $\Lambda = \{ ( \alpha _ { 1 } , \ldots , \alpha _ { p } ) | 1 \leq$ $\alpha _ { 1 } < \cdots < \alpha _ { p } \leq d \}$ . For an $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { p } ) \in \Lambda$ , denote $M _ { \alpha } = e _ { \alpha _ { 1 } } \otimes \cdot \cdot \cdot \otimes e _ { \alpha _ { p } }$ , $A _ { \alpha } = ( e _ { \alpha _ { 1 } } + \cdot \cdot \cdot + e _ { \alpha _ { p } } ) ^ { \otimes p }$ . The model space $\mathcal { M }$ is a subset of rank-1 $p$ -th order tensors, which is defined as $\mathcal { M } \ = \ \Big \{ M _ { \alpha } | \alpha \ \in \ \Lambda \Big \}$ . We define the core action set $\mathcal { A } _ { 0 }$ as $\mathcal { A } _ { 0 } = \{ e _ { \alpha _ { 1 } } + \cdot \cdot \cdot + e _ { \alpha _ { p } } | \alpha \in \Lambda \}$ . The action set $A$ is the convex hull of $\mathcal { A } _ { 0 }$ : $\mathscr { A } = \mathrm { c o n v } ( \mathscr { A } _ { 0 } )$ . Assume that the ground-truth parameter is $M ^ { * } = M _ { \alpha ^ { * } } \in \mathcal { M }$ . At round $t$ , the algorithm chooses an action $\mathbf { \Omega } _ { \mathbf { { a } _ { t } } } ~ \in ~ \mathcal { A }$ , and gets the noiseless reward $r _ { t } ~ = ~ r ( M ^ { * } , { \pmb a } _ { t } ) ~ =$ $\begin{array} { r } { \langle M ^ { * } , ( { \pmb a } _ { t } ) ^ { \otimes p } \rangle = \prod _ { i = 1 } ^ { p } \langle \pmb { e } _ { \alpha _ { i } ^ { * } } , \pmb { a } _ { t } \rangle } \end{array}$ .

# E.1 Proof for Theorem 3.29

We introduce a lemma showing that if the action set is restricted to the core action set $\mathcal { A } _ { 0 }$ , then at least $\begin{array} { r } { | \mathcal { A } _ { 0 } | - 1 = \binom { d } { p } - \mathbf { \bar { 1 } } } \end{array}$ actions are needed to identify the ground-truth.

Lemma E.1. If the actions are restricted to $\mathcal { A } _ { 0 }$ , then for the noiseless degree-p polynomial bandits, any algorithm needs to play at least $\binom { d } { p } - 1$ actions to determine $M ^ { * }$ in the worst case. Furthermore, the worst-case cumulative regret at round $T$ can be lower bounded by

$$
\mathfrak {R} (T) \geq \min  \{T, \left( \begin{array}{c} d \\ p \end{array} \right) - 1 \}.
$$

proof of Lemma E.1. For any $\alpha$ and $\alpha ^ { \prime }$ , the reward of playing $\pmb { e } _ { \alpha _ { 1 } } + \cdots + \pmb { e } _ { \alpha _ { p } }$ when the ground-truth model is $M _ { \alpha } ^ { \prime }$ is

$$
\begin{array}{l} \left\langle \boldsymbol {M} _ {\alpha} ^ {\prime}, \left(\boldsymbol {e} _ {\alpha_ {1}} + \dots + \boldsymbol {e} _ {\alpha_ {p}}\right) ^ {\otimes p} \right\rangle = \prod_ {i = 1} ^ {p} \left\langle \boldsymbol {e} _ {\alpha_ {i} ^ {\prime}}, \boldsymbol {e} _ {\alpha_ {1}} + \dots + \boldsymbol {e} _ {\alpha_ {p}} \right\rangle \\ = \prod_ {i = 1} ^ {p} \mathbb {I} \left\{\alpha_ {i} ^ {\prime} \in \alpha \right\} \\ = \left\{ \begin{array}{l l} 1, & \text {i f} \alpha = \alpha^ {\prime} \\ 0, & \text {o t h e r w i s e}  . \end{array} \right. \\ \end{array}
$$

Hence, no matter how the algorithm adaptively chooses the actions, in the worst case $\binom { d } { p } - 1$ actions are needed to determine $M ^ { * }$ . Also notice that the reward for $e _ { \alpha _ { 1 } } + \cdot \cdot \cdot + e _ { \alpha _ { p } }$ is zero if $\alpha \neq \alpha ^ { * }$ . Therefore the regret lower bound follows.

Next, we show that even when the action set is unrestricted, any UCB algorithm fails to explore in an unrestricted way. This is because the optimistic mechanism forbids the algorithm to play an informative action that is known to be low reward for all models in the confidence set. We first recall the definition of UCB algorithms.

UCB Algorithms The UCB algorithms sequentially maintain a confidence set $\mathcal { C } _ { t }$ after playing actions $\mathbf { } a _ { 1 } , \ldots , \mathbf { } a _ { t }$ . Then UCB algorithms play

$$
\boldsymbol{a}_{t + 1}\in \operatorname *{arg  max}_{\boldsymbol {a}\in \mathcal{A}}\mathrm{UCB}_{t}(\boldsymbol {a}),
$$

where

$$
\mathrm {U C B} _ {t} (\pmb {a}) = \max _ {\pmb {M} \in \mathcal {C} _ {t}} \langle \pmb {M}, (\pmb {a}) ^ {\otimes p} \rangle .
$$

proof of Theorem 3.29. We prove that even if the action set is unrestrcited, the optimistic mechanism in the UCB algorithm above forces it to choose actions in the restricted action set $\mathcal { A } _ { 0 }$ .

Assume $M ^ { * } = M _ { \alpha ^ { * } }$ . Next we show that for all $\pmb { a } \in \mathcal { A } - \mathcal { A } _ { 0 }$ (where the minus sign should be understood as set difference), we have

$$
\mathrm {U C B} _ {t} (\boldsymbol {a}) <   1.
$$

For all ${ \pmb a } \in \mathcal { A }$ , since $\mathscr { A } = \mathrm { c o n v } ( \mathscr { A } _ { 0 } )$ , we can write

$$
\boldsymbol {a} = \sum_ {\alpha \in \Lambda} p _ {\alpha} \left(\boldsymbol {e} _ {\alpha_ {1}} + \dots + \boldsymbol {e} _ {\alpha_ {p}}\right),
$$

where $\textstyle \sum _ { \alpha \in \Lambda } p _ { \alpha } = 1$ and $p _ { \alpha } \geq 0$ . Therefore,

$$
\begin{array}{l} \mathrm {U C B} _ {t} (\boldsymbol {a}) = \max  _ {\boldsymbol {M} \in \mathcal {C} _ {t}} \langle \boldsymbol {M}, (\boldsymbol {a}) ^ {\otimes p} \rangle \\ \leq \max  _ {\boldsymbol {M} \in \mathcal {M}} \left\langle \boldsymbol {M}, (\boldsymbol {a}) ^ {\otimes p} \right\rangle \\ = \max  _ {\alpha^ {\prime}} \langle \boldsymbol {M} _ {\alpha^ {\prime}}, (\boldsymbol {a}) ^ {\otimes p} \rangle \\ = \max  _ {\alpha^ {\prime}} \prod_ {i = 1} ^ {p} \left\langle \boldsymbol {e} _ {\alpha_ {i} ^ {\prime}}, \boldsymbol {a} \right\rangle . \\ \end{array}
$$

Plug in the expression of $^ { a }$ , we have

$$
\begin{array}{l} \langle \boldsymbol {e} _ {\alpha_ {i} ^ {\prime}}, \boldsymbol {a} \rangle = \sum_ {\alpha} p _ {\alpha} \langle \boldsymbol {e} _ {\alpha_ {i} ^ {\prime}}, \boldsymbol {e} _ {\alpha_ {1}} + \dots + \boldsymbol {e} _ {\alpha_ {p}} \rangle \\ = \sum_ {\alpha} p _ {\alpha} \mathbb {I} \left\{\alpha_ {i} ^ {\prime} \in \alpha \right\} \\ \leq \sum_ {\alpha} p _ {\alpha} = 1. \\ \end{array}
$$

Therefore, for any fixed $\alpha ^ { \prime } = ( \alpha _ { 1 } ^ { \prime } , \ldots , \alpha _ { p } ^ { \prime } )$ ,

$$
\begin{array}{l} \prod_ {i = 1} ^ {p} \langle \boldsymbol {e} _ {\alpha_ {i} ^ {\prime}}, \boldsymbol {a} \rangle = \left(\sum_ {\alpha} p _ {\alpha} \mathbb {I} \{\alpha_ {1} ^ {\prime} \in \alpha \}\right) \dots \left(\sum_ {\alpha} p _ {\alpha} \mathbb {I} \{\alpha_ {p} ^ {\prime} \in \alpha \}\right) \\ \leq 1, \\ \end{array}
$$

where the equality holds if and only if for any $p _ { \alpha } > 0$ , $\alpha = \alpha ^ { \prime }$ , which is equivalent to $\pmb { a } = \pmb { e } _ { \alpha _ { 1 } ^ { \prime } } + \cdot \cdot \cdot + \pmb { e } _ { \alpha _ { p } ^ { \prime } }$ . Therefore, if $\pmb { a } \in \mathcal { A } - \mathcal { A } _ { 0 }$ , for any $\alpha ^ { \prime } \in \Lambda$ , we have $\begin{array} { r } { \prod _ { i = 1 } ^ { p } \langle e _ { \alpha _ { i } ^ { \prime } } , \pmb { a } \rangle < 1 } \end{array}$ . This means

$$
\mathrm {U C B} _ {t} (\boldsymbol {a}) <   1.
$$

Meanwhile, we can see that for the action $\pmb { a } ^ { * } = \pmb { e } _ { \alpha _ { 1 } ^ { * } } + \cdot \cdot \cdot + \pmb { e } _ { \alpha _ { p } ^ { * } } \in \mathcal { A } _ { 0 } ,$ ,

$$
\begin{array}{l} \mathrm {U C B} _ {t} \left(\boldsymbol {a} ^ {*}\right) = \max  _ {\boldsymbol {M} \in \mathcal {C} _ {t}} \langle \boldsymbol {M}, \left(\boldsymbol {a} ^ {*}\right) ^ {\otimes p} \rangle \\ \geq \langle \boldsymbol {M} ^ {*}, (\boldsymbol {a} ^ {*}) ^ {\otimes p} \rangle \quad (\boldsymbol {M} ^ {*} \in \mathcal {C} _ {t}) \\ = \left\langle \boldsymbol {M} ^ {*}, \boldsymbol {A} _ {\alpha^ {*}} \right\rangle = 1. \\ \end{array}
$$

Therefore, we see that $( \mathcal { A } - \mathcal { A } _ { 0 } ) \cap \arg \operatorname* { m a x } _ { { \pmb a } \in \mathcal { A } } \mathrm { U C B } _ { t } ( { \pmb a } ) = \emptyset$ , which means $\mathbf { \pmb { a } } _ { t + 1 } \in \mathcal { A } _ { 0 }$ for all $t \geq 0$ . Therefore, by Lemma E.1, the theorem holds.

# E.2 $O ( d )$ Actions via Solving Polynomial Equations

Firstly, we verify that the model falls into the category of Example 3.28 with $k = p$ . For every $\alpha \in \Lambda$ , the reward of playing $^ { a }$ when the ground-truth model is $M _ { \alpha }$ is

$$
\langle \boldsymbol {M} _ {\alpha}, (\boldsymbol {a}) ^ {\otimes p} \rangle = \prod_ {i = 1} ^ {p} \langle \boldsymbol {e} _ {\alpha_ {i}}, \boldsymbol {a} \rangle ,
$$

which can be written as $q _ { 0 } ( U _ { \alpha } { \pmb a } )$ , where $q _ { 0 } ( x _ { 1 } , \ldots , x _ { p } ) = x _ { 1 } x _ { 2 } \cdot \cdot \cdot x _ { p }$ and $U _ { \alpha } \in \mathbb { R } ^ { p \times d }$ is a matrix with $e _ { \alpha _ { i } }$ as the $i$ -th row.

Secondly, we show that since the ground-truth model is $p$ -homogenous, we can extend the action set to $\mathrm { c o n v } ( \mathcal { A } , \mathbf { 0 } )$ . This is because for every action of the form $^ { c a }$ , where $0 \leq$ $c \leq 1$ and ${ \pmb a } \in \mathcal { A }$ , the reward is $c ^ { p }$ times the reward at $^ { a }$ . Therefore, to get the reward at $c \pmb { a }$ we only need to play at $^ { a }$ and multiply the reward by $c ^ { p }$ .

Notice that $\mathrm { c o n v } ( \mathcal { A } , \mathbf { 0 } )$ is of positive Lebesgue measure. By Theorem 3.26, we know that only $2 ( d k + ( p + 1 ) ^ { p } ) = { \cal O } ( d )$ actions are needed to determine the optimal action almost surely.

# F Proof of Section 3.3.4

We present the proof of Theorem 3.23 in the following.

Proof. We overload the notation and use $[ d ]$ to denote the set $\{ e _ { 1 } , e _ { 2 } , \ldots , e _ { d } \}$ . The hard instances are chosen in $\Delta \cdot [ d ] ^ { p }$ , i.e. $( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { p } ) = \Delta \cdot ( \widehat { \pmb { \theta } } _ { 1 } , \dots , \widehat { \pmb { \theta } } _ { p } )$ where $( { \widehat { \pmb { \theta } } } _ { 1 } , \dots , { \widehat { \pmb { \theta } } } _ { p } ) \in$ $[ d ] ^ { p }$ . For a group of vectors $\pmb { \theta } _ { 1 } , \ldots , \pmb { \theta } _ { p } \in [ d ]$ , we use

$$
\operatorname {s u p p} (\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) := (\max  _ {i \in [ p ]} (\boldsymbol {\theta} _ {i}) _ {1}, \dots , \max  _ {i \in [ p ]} (\boldsymbol {\theta} _ {i}) _ {d}) \in \{0, 1 \} ^ {d}
$$

to denote the support of these vectors. We use $\pmb { a } ^ { ( t ) } \in \mathbb { R } ^ { d }$ to denote the action in $t$ -th episode.

We use $\mathbb { P } _ { ( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { p } ) }$ to denote the measure on outcomes induced by the interaction of the fixed policy and the bandit paramterised by $\begin{array} { r } { r = \prod _ { i = 1 } ^ { p } ( \pmb { \theta } _ { i } ^ { \top } \pmb { a } ) + \epsilon } \end{array}$ . Specifically, We use $\mathbb { P } _ { 0 }$ to denote the measure on outcomes induced by the interaction of the fixed policy and the

pure noise bandit $r = \epsilon$ .

$$
\begin{array}{l} \Re (d, p, T) \\ \geq \frac {1}{d ^ {p}} \sum_ {(\pmb {\theta} _ {1}, \dots , \pmb {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \mathbb {E} _ {(\pmb {\theta} _ {1}, \dots , \pmb {\theta} _ {p})} \left[ T \Delta^ {p} / p ^ {p / 2} - \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\pmb {\theta} _ {i} ^ {\top} \pmb {a} ^ {(t)}) \right] \\ = \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T / p ^ {p / 2} - \mathbb {E} _ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p})} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) \right]\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T / p ^ {p / 2} - \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) \right] - T \| \mathbb {P} _ {0} - \mathbb {P} _ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p})} \| _ {\mathrm {T V}}\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T / p ^ {p / 2} - \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left(\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}\right) \right] - T \sqrt {D _ {\mathrm {K L}} \left(\mathbb {P} _ {0} | | \mathbb {P} _ {\left(\boldsymbol {\theta} _ {1} , \dots , \boldsymbol {\theta} _ {p}\right)}\right)}\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T / p ^ {p / 2} - \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) \right] - T \sqrt {\Delta^ {2 p} \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) ^ {2} \right]}\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \left(\frac {d ^ {p} T}{p ^ {\frac {p}{2}}} - \mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1}, \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) \right] - T d ^ {\frac {p}{2}} \Delta^ {p} \sqrt {\mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1} , \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) ^ {2} \right]}\right) \\ \end{array}
$$

where the first step comes from

$$
\operatorname {R e g r e t} \geq \mathbb {E} _ {\left(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}\right)} \left[ T \Delta^ {p} / p ^ {p / 2} - \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}\right) \right]
$$

(the optimal action in hindsight is $\pmb { a } = \mathrm { s u p p } ( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { p } ) / \sqrt { p } )$ ; the second step comes from $( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { p } ) = \Delta { \cdot } ( \widehat { \pmb { \theta } } _ { 1 } , \dots , \widehat { \pmb { \theta } } _ { p } )$ and algebra; the third step comes from $\begin{array} { r } { \left| \sum _ { t = 1 } ^ { T } \prod _ { i = 1 } ^ { p } ( \widehat { \pmb { \theta } _ { i } ^ { \top } \pmb { a } ^ { ( t ) } } ) \right| \leq } \end{array}$ $T$ ; the fourth step comes from Pinsker’s inequality; the fifth step comes from

$$
\begin{array}{l} D _ {\mathrm {K L}} \left(\mathbb {P} _ {0} | | \mathbb {P} _ {\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}}\right) = \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} D _ {\mathrm {K L}} \left(N (0, 1) | | N \left(\prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}\right), 1\right)\right) \right] \\ = \Delta^ {2 p} \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left(\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}\right) ^ {2} \right] \\ \end{array}
$$

and the final step comes from Jensen’s inequality and algebra.

Notice that

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1}, \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) \right] = \mathbb {E} _ {0} \left[ \sum_ {(j _ {1}, \dots , j _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\boldsymbol {a} _ {j _ {i}} ^ {(t)}) \right] \\ = \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left(\sum_ {j = 1} ^ {d} \boldsymbol {a} _ {j} ^ {(t)}\right) \right] \\ \leq \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \| \boldsymbol {a} ^ {(t)} \| _ {1} \right] \\ \leq d ^ {p / 2} T \\ \end{array}
$$

and

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1}, \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}) ^ {2} \right] = \mathbb {E} _ {0} \left[ \sum_ {(j _ {1}, \dots , j _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\boldsymbol {a} _ {j _ {i}} ^ {(t)}) ^ {2} \right] \\ = \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \| \boldsymbol {a} ^ {(t)} \| _ {2} ^ {2} \right] \\ \leq T \\ \end{array}
$$

where we used $\| \pmb { a } ^ { ( t ) } \| _ { 2 } \le 1 , \forall t \in [ T ]$ . Therefore plugging back we have

$$
\Re (d, p, T) \geq \frac {\Delta^ {p}}{d ^ {p}} \left(\frac {d ^ {p} T}{p ^ {\frac {p}{2}}} - d ^ {\frac {p}{2}} T - T d ^ {\frac {p}{2}} \Delta^ {p} \sqrt {T}\right)
$$

and finally letting $\begin{array} { r } { \Delta ^ { p } = \sqrt { \frac { d ^ { p } } { 4 T p ^ { p } } } } \end{array}$ leads to

$$
\Re (d, p, T) \geq O (\sqrt {d ^ {p} T} / p ^ {p}).
$$

![](images/fc3f0ecd857d3c95782d28acb56d77a6304308dce75bd461ebee80680e151140.jpg)

Remark F.1. Better result $O ( { \sqrt { d ^ { p } T } } )$ holds for bandits $\begin{array} { r } { r = \prod _ { i = 1 } ^ { p } ( \pmb { \theta } _ { i } ^ { \top } \pmb { a } _ { i } ) + \epsilon } \end{array}$ where ${ \mathbf { } } a _ { i } \in \mathrm { \Sigma }$ $\mathbb { R } ^ { d } , \| \pmb { a } _ { i } \| _ { 2 } \le 1$ .

For completeness, we show the proof of the above remark.

Proof. We overload the notation and use $[ d ]$ to denote the set $\{ e _ { 1 } , e _ { 2 } , \ldots , e _ { d } \}$ . The hard instances are chosen in $\Delta \cdot [ d ] ^ { p }$ , i.e. $( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { p } ) = \Delta \cdot ( \widehat { \pmb { \theta } } _ { 1 } , \dots , \widehat { \pmb { \theta } } _ { p } )$ where $( { \widehat { \pmb { \theta } } } _ { 1 } , \dots , { \widehat { \pmb { \theta } } } _ { p } ) \in$ $[ d ] ^ { p }$ . We use $\pmb { a } _ { i } ^ { ( t ) } \in \mathbb { R } ^ { d }$ to denote the $i$ -th action in $t$ -th episode, where $i \in [ p ] , t \in [ T ]$ .

We use $\mathbb { P } _ { ( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { p } ) }$ to indicate the measure on outcomes induced by the interaction of the fixed policy and the bandit paramterised by $\begin{array} { r } { r = \prod _ { i = 1 } ^ { p } ( \pmb { \theta } _ { i } ^ { \top } \pmb { a } _ { i } ) + \epsilon } \end{array}$ . Specifically, We use $\mathbb { P } _ { 0 }$

to indicate the measure on outcomes induced by the interaction of the fixed policy and the pure noise bandit $r = \epsilon$ .

$$
\begin{array}{l} \Re (d, p, T) \\ \geq \frac {1}{d ^ {p}} \sum_ {(\pmb {\theta} _ {1}, \dots , \pmb {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \mathbb {E} _ {(\pmb {\theta} _ {1}, \dots , \pmb {\theta} _ {p})} \left[ T \Delta^ {p} - \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\pmb {\theta} _ {i} ^ {\top} \pmb {a} _ {i} ^ {(t)}) \right] \\ = \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T - \mathbb {E} _ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p})} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) \right]\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T - \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) \right] - T \| \mathbb {P} _ {0} - \mathbb {P} _ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p})} \| _ {\mathrm {T V}}\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T - \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) \right] - T \sqrt {D _ {\mathrm {K L}} (\mathbb {P} _ {0} | | \mathbb {P} _ {(\boldsymbol {\theta} _ {1} , \dots , \boldsymbol {\theta} _ {p})})}\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \sum_ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}) \in \Delta \cdot [ d ] ^ {p}} \left(T - \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) \right] - T \sqrt {\Delta^ {2 p} \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) ^ {2} \right]}\right) \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \left(d ^ {p} T - \mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1}, \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) \right] - T d ^ {\frac {p}{2}} \Delta^ {p} \sqrt {\mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1} , \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) ^ {2} \right]}\right) \\ \end{array}
$$

where the first step comes from

$$
\mathrm {R e g r e t} \geq \mathbb {E} _ {(\pmb {\theta} _ {1}, \dots , \pmb {\theta} _ {p})} \left[ T \Delta^ {p} - \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\pmb {\theta} _ {i} ^ {\top} \pmb {a} _ {i} ^ {(t)}) \right]
$$

(the optimal action in hindsight is $\mathbf { a } _ { i } = { \widehat { \pmb { \theta } } } _ { i } .$ ); the second step comes from $( \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { p } ) =$ $\Delta \cdot ( \widehat { \pmb { \theta } } _ { 1 } , \dots , \widehat { \pmb { \theta } } _ { p } )$ and algebra; the third step comes from $\begin{array} { r } { \left| \sum _ { t = 1 } ^ { T } \prod _ { i = 1 } ^ { p } ( \widehat { { \pmb \theta } _ { i } ^ { \top } } { \pmb a } _ { i } ^ { ( t ) } ) \right| \leq T } \end{array}$ ; the fourth step comes from Pinsker’s inequality; the fifth step comes from

$$
\begin{array}{l} D _ {\mathrm {K L}} \left(\mathbb {P} _ {0} | | \mathbb {P} _ {\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}}\right) = \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} D _ {\mathrm {K L}} \left(N (0, 1) | | N \left(\prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}\right), 1\right)\right) \right] \\ = \Delta^ {2 p} \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\pmb {\theta}} _ {i} ^ {\top} \pmb {a} _ {i} ^ {(t)}) ^ {2} \right] \\ \end{array}
$$

and the final step comes from Jensen’s inequality and algebra.

Notice that

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1}, \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} (\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}) \right] = \mathbb {E} _ {0} \left[ \sum_ {(j _ {1}, \dots , j _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left((\boldsymbol {a} _ {i} ^ {(t)}) _ {j _ {i}}\right) \right] \\ = \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left(\sum_ {j = 1} ^ {d} \left(\boldsymbol {a} _ {i} ^ {(t)}\right) _ {j}\right) \right] \\ \leq \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \| \boldsymbol {a} _ {i} ^ {(t)} \| _ {1} \right] \\ \leq d ^ {p / 2} T \\ \end{array}
$$

and

$$
\begin{array}{l} \mathbb {E} _ {0} \left[ \sum_ {(\widehat {\boldsymbol {\theta}} _ {1}, \dots , \widehat {\boldsymbol {\theta}} _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left(\widehat {\boldsymbol {\theta}} _ {i} ^ {\top} \boldsymbol {a} _ {i} ^ {(t)}\right) ^ {2} \right] = \mathbb {E} _ {0} \left[ \sum_ {(j _ {1}, \dots , j _ {p}) \in [ d ] ^ {p}} \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \left(\left(\boldsymbol {a} _ {i} ^ {(t)}\right) _ {j _ {i}}\right) ^ {2} \right] \\ = \mathbb {E} _ {0} \left[ \sum_ {t = 1} ^ {T} \prod_ {i = 1} ^ {p} \| \boldsymbol {a} _ {i} ^ {(t)} \| _ {2} ^ {2} \right] \\ \leq T \\ \end{array}
$$

where we used $\lVert \mathbf { \boldsymbol { a } } _ { i } ^ { ( t ) } \rVert _ { 2 } \leq 1 , \forall t \in [ T ]$ . Therefore plugging back we have

$$
\Re (d, p, T) \geq \frac {\Delta^ {p}}{d ^ {p}} \left(d ^ {p} T - d ^ {\frac {p}{2}} T - T d ^ {\frac {p}{2}} \Delta^ {p} \sqrt {T}\right)
$$

and finally letting $\begin{array} { r } { \Delta ^ { p } = \sqrt { \frac { d ^ { p } } { 4 T } } } \end{array}$ 4T leads to

$$
\Re (d, p, T) \geq O (\sqrt {d ^ {p} T}).
$$

![](images/2530034bcb5300db5311bc569ae853b832f7d58044b0cd5c16c4b77b05a560b1.jpg)

We present the proof of Theorem 3.24 in the following.

Proof. Denote the optimal action in hindsight as ${ \pmb a } ^ { * } = \mathrm { s u p p } ( \pmb { \theta } _ { 1 } , \ldots , \pmb { \theta } _ { p } ) / \sqrt { p }$ . From the

proof of Theorem 3.23 we know that if $\begin{array} { r } { T \le \frac { 1 } { 4 p ^ { p } } \cdot \frac { d ^ { p } } { \Delta ^ { 2 p } } } \end{array}$ , then

$$
\begin{array}{l} \frac {1}{d ^ {p}} \sum_ {\left(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}\right) \in \Delta \cdot [ d ] ^ {p}} \mathbb {E} _ {\left(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}\right)} \left[ \prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} ^ {*}\right) - \prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}\right) \right] \\ \geq \frac {\Delta^ {p}}{d ^ {p}} \left(\frac {d ^ {p}}{p ^ {\frac {p}{2}}} - d ^ {\frac {p}{2}} - d ^ {\frac {p}{2}} \Delta^ {p} \sqrt {T}\right) \\ \geq \frac {\Delta^ {p}}{4 p ^ {\frac {p}{2}}} \\ \geq \frac {1}{4} \cdot \frac {1}{d ^ {p}} \sum_ {\left(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}\right) \in \Delta \cdot [ d ] ^ {p}} \mathbb {E} _ {\left(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p}\right)} \left[ \prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} ^ {*}\right) \right] \\ \end{array}
$$

which indicates the following

$$
\inf  _ {\pi} \sup  _ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p})} \mathbb {E} _ {(\boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {p})} \left[ \frac {3}{4} \cdot \prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} ^ {*}\right) - \prod_ {i = 1} ^ {p} \left(\boldsymbol {\theta} _ {i} ^ {\top} \boldsymbol {a} ^ {(t)}\right) \right] \geq 0.
$$

![](images/60a33c641a3241ee182a2c100dcb2a7aa1f0a44e040fdd890d40526c0a66a21e.jpg)