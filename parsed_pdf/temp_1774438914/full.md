# Distributed Principal Component Analysis with Limited Communication

Foivos Alimisis

Department of Mathematics University of Geneva

Peter Davies

Department of Computer Science University of Surrey

Bart Vandereycken

Department of Mathematics University of Geneva

Dan Alistarh

IST Austria &

Neural Magic, Inc.

# Abstract

We study efficient distributed algorithms for the fundamental problem of principal component analysis and leading eigenvector computation on the sphere, when the data are randomly distributed among a set of computational nodes. We propose a new quantized variant of Riemannian gradient descent to solve this problem, and prove that the algorithm converges with high probability under a set of necessary spherical-convexity properties. We give bounds on the number of bits transmitted by the algorithm under common initialization schemes, and investigate the dependency on the problem dimension in each case.

# 1 Introduction

The need to process ever increasing quantities of data has motivated significant research interest into efficient distributed algorithms for standard tasks in optimization and data analysis; see, e.g., [16, 27, 11]. In this context, one of the fundamental metrics of interest is communication cost, measured as the number of bits sent and received by the processors throughout computation. Communicationefficient variants are known for several classical tasks and algorithms, from basic ones such as mean estimation [28, 20, 8] to variants of gradient descent [19, 12] or even higher-order optimization [2, 15].

By contrast, much less is known concerning the communication cost for classical problems in data analysis, such as Principal Component Analysis (PCA) [13, 5], or spectral clustering [23]. In this paper, we will focus on the closely-related problem of computing the leading eigenvector of the data covariance matrix, i.e., determining the first principal component, in the setting where $d$ -dimensional data comes from an unknown distribution, and is split among $n$ machines.

Given the fundamental nature of PCA, there is a rich body of work on solving sequential variants, both in the Euclidean domain [24, 25, 26] and, more recently, based on Riemannian optimization. This is because maximizing the Rayleigh quotient can be re-interpreted in terms of a hyper-sphere, leading to an unconstrained Riemannian optimization problem [1, 32, 31]. These references have exclusively focused on the sequential setting, where the data can be stored and processed by a single node. However, in many settings the large volume of data requires processing by multiple nodes, which has inspired significant work on reducing the distribution overheads of PCA, e.g. [4, 11], specifically on obtaining efficient algorithms in terms of round complexity.

More precisely, an impressive line of work considered the deterministic setting [17, 4, 6, 10], where the input is partitioned arbitrarily between machines, for which both round-efficient and communication-efficient solutions are now known. The general approach is to first perform a careful low-rank approximation of the local data and then communicate iteratively to reconstruct a faithful

estimate of the covariance matrix. Due to the worst-case setting, the number of communication rounds scales polynomially in the eigengap $\delta$ and in the precision $\varepsilon$ , making them costly in practice.

Thus, on the practical side, it is common to assume a setting with stochastic rather than worst-case data partitioning. Specifically, Garber et al. [11] consider this setting, and note that standard solutions based on distributed Power Method or Lanczos would require a number of communication rounds which increases with the sample size, which is impractical at scale. They circumvent this issue by replacing explicit Power Method iterations with a set of convex optimization problems, which can be solved in a round-efficient manner. Huang and Pan [14] proposed a round-efficient solution, by leveraging the connection to Riemannian optimization. Their proposal relies on a procedure which allows the update of node-local variables via a surrogate gradient, whose deviation from the global gradient should be bounded. Thus, communication could only be used to reach consensus on the local variables among the nodes. Unfortunately, upon close inspection, we have noted significant technical gaps in their analysis, which we discuss in detail in the next section.

The line of work in the stochastic setting focused primarily on latency cost, i.e., the number of communication rounds for solving this problem. For highly-dimensional data and/or highly-distributed settings, it is known that bandwidth cost, i.e., the number of bits which need to be transmitted per round, can be a significant bottleneck, and an important amount of work deals with it for other classic problems, e.g. [28, 12, 15]. By contrast, much less is known concerning the communication complexity of distributed PCA.

Contribution. In this paper, we provide a new algorithm for distributed leading eigenvector computation, which specifically minimizes the total number of bits sent and received by the nodes. Our approach leverages the connection between PCA and Riemannian optimization; specifically, we propose the first distributed variant of Riemannian gradient descent, which supports quantized communication. We prove convergence of this algorithm utilizing a set of necessary convexity-type properties, and bound its communication complexity under different initializations.

At the technical level, our result is based on a new sequential variant of Riemannian gradient descent on the sphere. This may be of general interest, and the algorithm can be generalized to Riemannian manifolds of bounded sectional curvatures using more sophisticated geometric bounds. We strengthen the gradient dominance property proved in [31], by proving that our objective is weakly-quasi-convex in a ball of some minimizer and combining that with a known quadratic-growth condition. We leverage these properties by adapting results of [7] in the Riemannian setting and choosing carefully the learning-rate in order to guarantee convergence for the single-node version.

This algorithm is specifically-designed to be amenable to distribution: specifically, we port it to the distributed setting, and add communication-compression in the tangent spaces of the sphere, using a variant of the lattice quantization technique of [8], which we port from the context of standard gradient descent. We provide a non-trivial analysis for the communication complexity of the resulting algorithm under different initializations. We show that, under random initialization, a solution can be reached with total communication that is quadratic in the dimension $d$ , and linear in $1 / \delta$ , where $\delta$ is the gap between the two biggest eigenvalues; when the initialization is done to the leading eigenvector of one of the local covariance matrices, this dependency can be reduced to linear in $d$ .

# 2 Setting and Related Work

Setting. We assume to be given $m$ total samples coming from some distribution $\mathcal { D }$ , organized as a global $m \times d$ data matrix $M$ , which is partitioned row-wise among $n$ processors, with node $i$ being assigned the matrix $M _ { i }$ , consisting of $m _ { i }$ consecutive rows, such that $\bar { \sum _ { i = 1 } ^ { n } } m _ { i } = m$ . As is common, let $\begin{array} { r } { A : = M ^ { T } M = \sum _ { i = 1 } ^ { d } M _ { i } ^ { T } M _ { i } } \end{array}$ be the global covariance matrix, and $A _ { i } : = M _ { i } ^ { T } M _ { i }$ the local covariance matrix owned by node $i$ . We denote by $\lambda _ { 1 } , \lambda _ { 2 } , . . . , \lambda _ { d }$ the eigenvalues of $A$ in descending order and by $v _ { 1 } , v _ { 2 } , . . . , v _ { d }$ the corresponding eigenvectors. Then, it is well-known that, if the gap $\delta = \lambda _ { 1 } - \lambda _ { 2 }$ between the two leading eigenvalues of $A$ is positive, we can approximate the leading eigenvector by solving the following empirical risk minimization problem up to accuracy $\varepsilon$ :

$$
x ^ {\star} = \operatorname {a r g m i n} _ {x \in \mathbb {R} ^ {d}} \left(- \frac {x ^ {T} A x}{\| x \| ^ {2}}\right) = \operatorname {a r g m i n} _ {x \in \mathbb {R} ^ {d}, \| x \| _ {2} = 1} \left(- x ^ {T} A x\right) = \operatorname {a r g m i n} _ {x \in \mathbb {S} ^ {d - 1}} \left(- x ^ {T} A x\right), \tag {1}
$$

where $\mathbb { S } ^ { d - 1 }$ is the $( d - 1 )$ -dimensional sphere.

We define $f : \mathbb { S } ^ { d - 1 }  \mathbb { R }$ , with $f ( x ) = - x ^ { T } A x$ and $f _ { i } : \mathbb { S } ^ { d - 1 } \to \mathbb { R }$ , with $f _ { i } ( x ) = - x ^ { T } A _ { i } x$ . Since the inner product is bilinear, we can write the global cost as the sum of the local costs: $\begin{array} { r } { f ( x ) = \dot { \sum _ { i = 1 } ^ { n } { f _ { i } ( x ) } } } \end{array}$ .

In our analysis we use notions from the geometry of the sphere, which we recall in Appendix A.

Related Work. As discussed, there has been a significant amount of research on efficient variants of PCA and related problems [5, 24, 25, 26, 1, 32, 31]. Due to space constraints, we focus on related work on communication-efficient algorithms. In particular, we discuss the relationship to previous round-efficient algorithms; to our knowledge, this is the first work to specifically focus on the bit complexity of this problem in the setting where data is randomly partitioned. More precisely, previous work on this variant implicitly assumes that algorithms are able to transmit real numbers at unit cost.

The straightforward approach to solve the minimization problem (1) in a distributed setting, where the data rows are partitioned, would be to use a distributed version of the Power Method, Riemannian gradient descent (RGD), or the Lanczos algorithm. In order to achieve an $\varepsilon$ -approximation of the minimizer $x ^ { \star }$ , the latter two algorithms require $\tilde { \mathcal { O } } ( \lambda _ { 1 } \log ( 1 / \varepsilon ) / \delta )$ rounds, where the $\tilde { \mathcal { O } }$ notation hides poly-logarithmic factors in $d$ . Distributed Lanczos and accelerated RGD would improve this by an $\mathcal { O } ( \sqrt { \lambda _ { 1 } / \delta } )$ factor. However, Garber et al. [11] point out that, when $\delta$ is small, e.g. $\delta = \Theta ( 1 / \sqrt { K n } )$ , then unfortunately the number of communication rounds would increase with the sample size, which renders these algorithms non-scalable.

Standard distributed convex approaches, e.g. [16, 27], do not directly extend to our setting due to non-convexity and the unit-norm constraint. Garber et al. [11] proposed a variant of the Power Method called Distributed Shift-and-Invert (DSI) which converges in roughly $\begin{array} { r } { \mathcal { O } \left( \sqrt { \frac { b } { \delta \sqrt { n } } } \log ^ { 2 } ( 1 / \varepsilon ) \log ( 1 / \delta ) \right) } \end{array}$ rounds, where $b$ is a bound on the squared $\ell _ { 2 }$ -norm of the data. Huang and Pan [14] aimed to improve the dependency of the algorithm on $\varepsilon$ and $\delta$ , and proposed an algorithm called Communication-Efficient Distributed Riemannian Eigensolver (CEDRE). This algorithm is shown to have round√ complexity $\begin{array} { r } { \mathcal { O } ( \frac { b } { \delta \sqrt { n } } \log ( 1 / \varepsilon ) ) } \end{array}$ , which does not scale with the sample size for $\delta = \Omega ( 1 / \sqrt { K n } )$ , and has only logarithmic dependency on the accuracy $\varepsilon$ .

Technical issues in [14]. Huang and Pan [14] proposed an interesting approach, which could provide the most round-efficient distributed algorithm to date. Despite the fact that we find the main idea of this paper very creative, we have unfortunately identified a significant gap in their analysis, which we now outline.

Specifically, one of their main results, Theorem 3, uses the local gradient dominance shown in [31]; yet, the application of this result is invalid, as it is done without knowing in advance that the iterates of the algorithms continue to remain in the ball of initialization. This is compounded by another error on the constant of gradient dominance (Lemma 2), which we believe is caused by a typo in the last part of the proof of Theorem 4 in [31]. This typo is unfortunately propagated into their proof. More precisely, the objective is indeed gradient dominated, but with a constant which vanishes when we approach the equator, in contrast with the $2 / \delta$ which is claimed globally. (We reprove this result independently in Proposition 4). Thus, starting from a point lying in some ball of the minimizer can lead to a new point where the objective is more poorly gradient dominated.

This is a non-trivial technical problem which, in the case of gradient descent, can be addressed by a choice of the learning rate depending on the initialization. Given these issues, we perform a new and formally-correct analysis for gradient descent based on new convexity properties which guarantee convergence directly in terms of iterates, and not just function values. We would like however to note that our focus in this paper is on bit and not round complexity.

# 3 Step 1: Computing the Leading Eigenvector in One Node

# 3.1 Convexity-like Properties and Smoothness

We first analyze the case that all the matrices $A _ { i }$ are simultaneously known to all machines. This practically means that we can run gradient descent in one of the machines (let it be the master), since we have all the data stored there. All the proofs for this section are in Appendix B.

Our problem reads

$$
\min  _ {x \in \mathbb {S} ^ {d - 1}} - x ^ {T} A x
$$

where $A = M ^ { T } M$ is the global covariance matrix. If $\delta = \lambda _ { 1 } - \lambda _ { 2 } > 0$ , this problem has exactly two global minima: $v _ { 1 }$ and $- v _ { 1 }$ . Let $x \in \mathbb { S } ^ { d - 1 }$ be an arbitrary point. Then $x$ can be written in the form $\begin{array} { r } { x = \sum _ { i = 1 } ^ { d } \alpha _ { i } v _ { i } } \end{array}$ . Fixing the minimizer $v _ { 1 }$ , we have that a ball in $\mathbb { S } ^ { d - 1 }$ around $v _ { 1 }$ is of the form

$$
B _ {a} = \{x \in \mathbb {S} ^ {d - 1} \mid \alpha_ {1} \geq a \} = \left\{x \in \mathbb {S} ^ {d - 1} \mid \langle x, v _ {1} \rangle \geq a \right\} \tag {2}
$$

for some $a$ . Without loss of generality, we may assume $a > 0$ (otherwise, consider the ball around $- v _ { 1 }$ to establish convergence to $- v _ { 1 } ,$ ). In this case, the region $\mathcal { A }$ as defined in Def. 11 is identical to $B _ { a }$ and $x ^ { * } : = v _ { 1 }$ .

We begin by investigating the convexity properties of the function $- x ^ { T } A x$ . In particular, we prove that this function is weakly-quasi-convex (see Def. 13) with constant $2 a$ in the ball $B _ { a }$ . See also Appendix A for a definition of the Riemannian gradient grad $f$ and the logarithm log.

Proposition 1. The function $f ( x ) = - x ^ { T } A x$ satisfies

$$
2 a (f (x) - f ^ {*}) \leq \langle \operatorname {g r a d} f (x), - \log_ {x} (x ^ {*}) \rangle
$$

for any $x \in B _ { a }$ with $a > 0$ .

We continue by providing a quadratic growth condition for our cost function, which can also be found in [30, Lemma 2] in a slightly different form. Here dist is the intrinsic distance in the sphere, that we also define in Appendix A.

Proposition 2. The function $f ( x ) = - x ^ { T } A x$ satisfies

$$
f (x) - f ^ {*} \geq \frac {\delta}{4} \mathrm {d i s t} ^ {2} (x, x ^ {*})
$$

for any $x \in B _ { a }$ with $a > 0$ .

According to Def. 14, this means that $f$ satisfies quadratic growth with constant $\mu = \delta / 2$ .

Next, we prove that quadratic growth and weak-quasi-convexity imply a “strong-weak-convexity” property. This is similar to [7, Proposition 3.1].

Proposition 3. A function $f \colon \mathbb { S } ^ { d - 1 } \to \mathbb { R }$ which satisfies quadratic growth with constant $\mu > 0$ and is 2a-weakly-quasi-convex with respect to a minimizer $x ^ { * }$ in a ball of this minimizer, satisfies for any x in the same ball the inequality

$$
f (x) - f ^ {*} \leq \frac {1}{a} \left\langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \right\rangle - \frac {\mu}{2} \operatorname {d i s t} ^ {2} \left(x, x ^ {*}\right).
$$

Interestingly, using Proposition 3, we can recover the gradient dominance property proved in [31, Theorem 4].

Proposition 4. The function $f ( x ) = - x ^ { T } A x$ satisfies

$$
\left\| \operatorname {g r a d} f (x) \right\| ^ {2} \geq \delta a ^ {2} (f (x) - f ^ {*})
$$

for any $x \in B _ { a }$ with $a > 0$ .

The proof of Proposition 4 can be done independently based only on Proposition 3 and [7, Lemma 3.2]. This verifies that our results until now are optimal regarding the quality of initialization since gradient dominance proved in [31] is optimal in that sense (i.e. the inequalities in the proof are tight).

Proposition 5. The function $f ( x ) = - x ^ { T } A x$ is geodesically $2 \lambda _ { 1 }$ -smooth on the sphere.

Thus, the smoothness constant of $f$ , as defined in Def. 12, equals $\gamma = 2 \lambda _ { 1 }$ . Similarly, let $\gamma _ { i }$ denote the smoothness constant of $f _ { i }$ , which equals twice the largest eigenvalue of $A _ { i }$ . In order to estimate $\gamma$ in a distributed fashion using only the local data matrices $A _ { i }$ , we shall use the over-approximation $\gamma \leq n \operatorname* { m a x } _ { i = 1 , \dots , n } \gamma _ { i }$ .

# 3.2 Convergence

We now consider Riemannian gradient descent with learning rate $\eta > 0$ starting from a point $x ^ { ( 0 ) } \in B _ { a }$ :

$$
x ^ {(t + 1)} = \exp_ {x ^ {(t)}} (- \eta \operatorname {g r a d} f (x ^ {(t)})),
$$

where exp is the exponential map of the sphere, defined in Appendix A.

Using Proposition 3 and a proper choice of $\eta$ , we can establish a convergence rate for the instrinsic distance of the iterates to the minimizer.

Proposition 6. An iterate of Riemannian gradient descent for $f ( x ) = - x ^ { T } A x$ starting from a point $x ^ { ( t ) } \in B _ { a }$ and with step-size $\eta \leq a / \gamma$ where $\gamma \geq 2 \lambda _ { 1 }$ , produces a point $x ^ { ( t + 1 ) }$ that satisfies

$$
\operatorname {d i s t} ^ {2} \left(x ^ {(t + 1)}, x ^ {*}\right) \leq (1 - a \mu \eta) \operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right) \quad f o r \mu = \delta / 2.
$$

Note that this result implies directly that if our initialization $x ^ { ( 0 ) }$ lies in the ball $B _ { a }$ , the distance of $x ^ { ( 1 ) }$ to the center $x ^ { * }$ decreases and thus all subsequent iterates continue being in the initialization ball. This is essential since it guarantees that the convexity-like properties for $f$ continue to hold during the whole optimization process. The proof of Proposition 6 is in Appendix B.

This result is also valid in a more general setting when applied to an arbitrary $f \colon \mathbb { S } ^ { d - 1 } \to \mathbb { R }$ that is $2 a$ -weakly-quasi convex and $\gamma$ -smooth, and that satisfies quadratic growth with constant $\mu$ . The Euclidean analogue of this result can be found in [7, Lemma 4.4].

# 4 Step 2: Computing the Leading Eigenvector in Several Nodes

We now present our version of distributed gradient descent for leading eigenvector computation and measure its bit complexity until reaching accuracy $\epsilon$ in terms of the intrinsic distance of an iterate $x ^ { ( T ) }$ from the minimizer $x ^ { * }$ $x ^ { * }$ is the leading eigenvector closest to the initialization point).

Lattice Quantization. For estimating the Riemannian gradient in a distributed manner with limited communication, we use a quantization procedure developed in [8]. The original quantization scheme involves randomness, but we use a deterministic version of it, by picking up the closest point to the vector that we want to encode. This is similar to the quantization scheme used by [3] and has the following properties.

Proposition 7. [8, 3] Denoting by b the number of bits that each machine uses to communicate, there exists a quantization function

$$
Q \colon \mathbb {R} ^ {d - 1} \times \mathbb {R} ^ {d - 1} \times \mathbb {R} _ {+} \times \mathbb {R} _ {+} \to \mathbb {R} ^ {d - 1},
$$

which, for each $w , y > 0$ , consists of an encoding function enc $\boldsymbol { w } , \boldsymbol { y } : \mathbb { R } ^ { d - 1 }  \{ 0 , 1 \} ^ { b }$ and a decoding one $\mathrm { d e c } _ { w , y } : \{ 0 , 1 \} ^ { b } \times \mathbb { R } ^ { d - 1 } \to \mathbb { R } ^ { d - 1 }$ , such that, for all $x , x ^ { \prime } \in \mathbb { R } ^ { d - 1 }$ ,

• decw,y(encw,y(x), x0) = Q(x, x0, y, w), if kx − x0k ≤ y.   
• $\begin{array} { r } { \| Q ( x , x ^ { \prime } , y , w ) - x \| \leq w , i f \| x - x ^ { \prime } \| \leq y . } \end{array}$   
• $I f y / w > 1$ , the cost of the quantization procedure in number of bits satisfies

$$
b = \mathcal {O} ((d - 1) \log_ {2} \left( \begin{array}{c} y \\ w \end{array} \right)) = \mathcal {O} (d \log \left( \begin{array}{c} y \\ w \end{array} \right)).
$$

In the following, the quantization takes place in the tangent space of each iterate $T _ { x ^ { ( t ) } } \mathbb { S } ^ { d - 1 }$ , which is linearly isomorphic to $\mathbb { R } ^ { d - 1 }$ . We denote by $Q _ { x }$ the specification of the function $Q$ at $T _ { x } \mathbb { S } ^ { d - 1 }$ . The vector inputs of the function $Q _ { x }$ are represented in the local coordinate system of the tangent space that the quantization takes place at each step. For decoding at $t > 0$ , we use information obtained in the previous step, that we need to translate to the same tangent space. We do that using parallel transport $\Gamma$ , which we define in Appendix A.

# Algorithm

We present now our main algorithm, which is inspired by quantized gradient descent firstly designed by [22], and its similar version in [3].

1. Choose an arbitrary machine to be the master node, let it be $i _ { 0 }$   
2. Choose $x ^ { ( 0 ) } \in \mathbb { S } ^ { d - 1 }$ (we analyze later specific ways to do that).   
3. Consider the following parameters

$$
\sigma := 1 - \cos (D) \mu \eta , K := \frac {2}{\sqrt {\sigma}}, \theta := \frac {\sqrt {\sigma} (1 - \sqrt {\sigma})}{4},
$$

$$
\sqrt {\xi} := \theta K + \sqrt {\sigma}, R ^ {(t)} = \gamma K (\sqrt {\xi}) ^ {t} D
$$

where $D$ is an over-approximation for $\mathrm { d i s t } ( x ^ { ( 0 ) } , x ^ { * } )$ (the way to practically choose $D$ is discussed later).

We assume that $\begin{array} { r } { \cos ( D ) \mu \eta \le \frac 1 2 } \end{array}$ , otherwise we run the algorithm with $\begin{array} { r } { \sigma = \frac { 1 } { 2 } } \end{array}$

In $T _ { x ^ { ( 0 ) } } \mathbb { S } ^ { d - 1 }$ :

4. Compute the local Riemannian gradient grad $f _ { i } ( x ^ { ( 0 ) } )$ at $x ^ { ( 0 ) }$ in each node.   
5. Encode grad $f _ { i } ( x ^ { ( 0 ) } )$ in each node and decode in the master node using its local information:

$$
q _ {i} ^ {(0)} = Q _ {x ^ {(0)}} \left(\operatorname {g r a d} f _ {i} (x ^ {(0)}), \operatorname {g r a d} f _ {i _ {0}} (x ^ {(0)}), 2 \gamma \pi , \frac {\theta R ^ {(0)}}{2 n}\right).
$$

6. Sum the decoded vectors in the master node:

$$
r ^ {(0)} = \sum_ {i = 1} ^ {n} \operatorname {g r a d} f _ {i} (x ^ {(0)}).
$$

7. Encode the sum in the master and decode in each machine $i$ using its local information:

$$
q ^ {(0)} = Q _ {x ^ {(0)}} \left(r ^ {(0)}, \operatorname {g r a d} f _ {i} \left(x ^ {(0)}\right), \frac {\theta R ^ {(0)}}{2} + 2 \gamma \pi , \frac {\theta R ^ {(0)}}{2}\right).
$$

For $t \geq 0$

8. Take a gradient step using the exponential map:

$$
x ^ {(t + 1)} = \exp_ {x ^ {(t)}} (- \eta q ^ {(t)})
$$

with step-size $\eta$ (the step-size is discussed later).

In $T _ { x ^ { ( t + 1 ) } } \mathbb { S } ^ { d - 1 }$ :

9. Compute the local Riemannian gradient grad $f _ { i } ( x ^ { ( t + 1 ) } )$ at $x ^ { ( t + 1 ) }$ in each node.

10. Encode grad $f _ { i } ( x ^ { ( t + 1 ) } )$ in each node and decode in the master node using its (parallelly transported) local information from the previous step:

$$
q _ {i} ^ {(t + 1)} = Q _ {x ^ {(t + 1)}} \left(\operatorname {g r a d} f _ {i} (x ^ {(t + 1)}), \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} q _ {i} ^ {(t)}, \frac {R ^ {(t + 1)}}{n}, \frac {\theta R ^ {(t + 1)}}{2 n}\right).
$$

11. Sum the decoded vectors in the master node:

$$
r ^ {(t + 1)} = \sum_ {i = 1} ^ {n} \operatorname {g r a d} f _ {i} (x ^ {(t + 1)}).
$$

12. Encode the sum in the master and decode in each machine using its local information in the previous step after parallel transport:

$$
q ^ {(t + 1)} = Q _ {x ^ {(t + 1)}} \left(r ^ {(t + 1)}, \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} q ^ {(t)}, \left(1 + \frac {\theta}{2}\right) R ^ {(t + 1)}, \frac {\theta R ^ {(t + 1)}}{2}\right).
$$

Convergence We first control the convergence of iterates simultaneously with the convergence of quantized gradients.

Note that

$$
\sqrt {\xi} = \frac {1 - \sqrt {\sigma}}{2} + \sqrt {\sigma} = \frac {1 + \sqrt {\sigma}}{2} \leq \frac {\sqrt {2} \sqrt {1 + \sigma}}{2} = \sqrt {\frac {1 + \sigma}{2}}.
$$

Lemma 8. If $\begin{array} { r } { \eta \le \frac { \cos ( D ) } { \gamma } } \end{array}$ , the previous quantized gradient descent algorithm produces iterates $x ^ { ( t ) }$ and quantized gradients $q ^ { ( t ) }$ that satisfy

$$
\operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right) \leq \xi^ {t} D ^ {2}, \quad \left\| q _ {i} ^ {(t)} - \operatorname {g r a d} f _ {i} \left(x ^ {(t)}\right) \right\| \leq \frac {\theta R ^ {(t)}}{2 n}, \quad \left\| q ^ {(t)} - \operatorname {g r a d} f \left(x ^ {(t)}\right) \right\| \leq \theta R ^ {(t)}.
$$

The proof is a Riemannian adaptation of the similar one in [22] and [3] and is presented in Appendix C. We recall that since the sphere is positively curved, it provides a landscape easier for optimization. It is quite direct to derive a general Riemannian method for manifolds of bounded curvature using more advanced geometric bounds, however this exceeds the scope of this work which focuses on leading eigenvector computation.

We now move to our main complexity result.

Theorem 9. Let $\begin{array} { r } { \eta \le \frac { \cos ( D ) } { \gamma } } \end{array}$ . Then, the previous quantized gradient descent algorithm needs at most

$$
b = \mathcal {O} \left(n d \frac {1}{\cos (D) \delta \eta} \log \left(\frac {n}{\cos (D) \delta \eta}\right) \log \left(\frac {D}{\epsilon}\right)\right)
$$

bits in total to estimate the leading eigenvector with an accuracy  measured in intrinsic distance.

The proof is based on the previous Lemma 8 in order to count the number of steps that the algorithm needs to estimate the minimizer with accuracy $\epsilon$ and Proposition 7 to count the quantization cost in each round. We present the proof again in Appendix C.

# 5 Step 3: Dependence on Initialization

# 5.1 Uniformly random initialization

The cheapest choice to initialize quantized gradient descent is a point in the sphere chosen uniformly at random. According to Theorem 4 in [31], such a random point $x ^ { ( 0 ) }$ will lie, with probability at least $1 - p$ , in a ball $B _ { a }$ (see Def. 2) where

$$
a \geq c \frac {p}{\sqrt {d}} \Longleftrightarrow \operatorname {d i s t} \left(x ^ {(0)}, x ^ {*}\right) \leq \arccos  \left(c \frac {p}{\sqrt {d}}\right). \tag {3}
$$

Here, $c$ is a universal constant (we estimated numerically that $c$ is lower bounded by 1, see Appendix D, thus we can use $c = 1$ ). By choosing the step-size $\eta$ as

$$
\eta = \frac {c p}{\sqrt {d} \gamma}
$$

and the parameter $D$ as

$$
D = \arccos \left(c \frac {p}{\sqrt {d}}\right),
$$

we are guaranteed that $\begin{array} { r } { \eta = \frac { \cos ( D ) } { \gamma } } \end{array}$ , and $\mathrm { d i s t } ( x ^ { ( 0 ) } , x ^ { * } ) \le D$ with probability at least $1 - p$ . Our analysis above therefore applies (up to probability $1 - p )$ and the general communication complexity result becomes

$$
\mathcal {O} \left(\frac {n d}{\cos (D) \delta \eta} \log \frac {n}{\cos (D) \delta \eta} \log \frac {D}{\epsilon}\right) = \mathcal {O} \left(\frac {n d}{\eta \gamma \delta \eta} \log \frac {n}{\eta \gamma \delta \eta} \log \frac {D}{\epsilon}\right) = \mathcal {O} \left(\frac {n d}{\eta^ {2} \gamma \delta} \log \frac {n}{\eta^ {2} \gamma \delta} \log \frac {D}{\epsilon}\right).
$$

Substituting $\begin{array} { r } { \eta ^ { 2 } = \frac { p ^ { 2 } c ^ { 2 } } { d \gamma ^ { 2 } } } \end{array}$ dγ2 , the number of bits satisfies (up to probability $1 - p )$ the upper bound

$$
b = \mathcal {O} \left(n \frac {d ^ {2}}{p ^ {2}} \frac {\gamma}{\delta} \log \frac {n d \gamma}{p \delta} \log \frac {D}{\epsilon}\right) = \tilde {\mathcal {O}} \left(n \frac {d ^ {2}}{p ^ {2}} \frac {\gamma}{\delta}\right).
$$

# 5.2 Warm Start

A reasonable strategy to get a more accurate initialization is to perform an eigenvalue decomposition to one of the local covariance matrices, for instance $A _ { i _ { 0 } }$ (in the master node $i _ { 0 }$ ), and compute its leading eigenvector, let it be $v ^ { i _ { 0 } }$ . Then we communicate this vector to all machines in order to use its quantized approximation $x ^ { ( 0 ) }$ as initialization. We define:

$$
\tilde {x} ^ {(0)} = Q \left(v ^ {i _ {0}}, v ^ {i}, \| v ^ {i _ {0}} - v ^ {i} \|, \frac {\langle v ^ {i _ {0}} , x ^ {*} \rangle}{2 (\sqrt {2} + 2)}\right)
$$

$$
x ^ {(0)} = \frac {\tilde {x} ^ {(0)}}{\| \tilde {x} ^ {(0)} \|}
$$

where $Q$ is the lattice quantization scheme in $\mathbb { R } ^ { d }$ (i.e. we quantize the leading eigenvector of the master node as a vector in $\mathbb { R } ^ { d }$ and then project back to the sphere). The input and output variance in this quantization can be bounded by constants that we can practically estimate (see Appendix D).

Proposition 10. Assume that our data are i.i.d. and sampled from a distribution $\mathcal { D }$ bounded in $\ell _ { 2 }$ norm by a constant h. Given that the eigengap δ and the number of machines n satisfy

$$
\delta \geq \Omega \left(\sqrt {n} \sqrt {\log \frac {d}{p}}\right), \tag {4}
$$

we have that the previous quantization costs $\mathcal { O } ( n d )$ many bits and $\langle x ^ { ( 0 ) } , x ^ { * } \rangle$ is lower bounded by a constant with probability at least $1 - p$ .

Thus, if bound 4 is satisfied, then the communication complexity becomes

$$
b = \mathcal {O} \left(n d \frac {\gamma}{\delta} \log \frac {n \gamma}{\delta} \log \frac {D}{\epsilon}\right) = \tilde {\mathcal {O}} \left(n d \frac {\gamma}{\delta}\right)
$$

many bits in total with probability at least $1 - p$ (notice that this can be further simplified using bound 4). This is because $D$ in Theorem 9 is upper bounded by a constant and the communication cost of quantizing $v ^ { i _ { 0 } }$ does not affect the total communication cost. If we can estimate the specific relation in bound 4 (the constant hidden inside $\Omega$ ), then we can compute estimations of the quantization parameters in the definition of x(0). $x ^ { ( 0 ) }$

Condition 4 is quite typical in this literature; see [14] and references therein (beginning of page 2), as√ we also briefly discussed in the introduction. Notice that $\sqrt { n }$ appears in the numerator and not the denominator, only because we deal with the sum of local covariance matrices and not the average, thus our eigengap is $n$ times larger than the eigengap of the normalized covariance matrix.

# 6 Experiments

We evaluate our approach experimentally, comparing the proposed method of Riemannian gradient quantization against three other benchmark methods:

• Full-precision Riemannian gradient descent: Riemannian gradient descent, as described in Section 3.2, is performed with the vectors communicated at full (64-bit) precision.   
• Euclidean gradient difference quantization: the ‘naïve’ approach to quantizing Riemannian gradient descent. Euclidean gradients are quantized and averaged before being projected to Riemannian gradients and used to take a step. To improve performance, rather than quantizing Euclidean gradients directly, we quantize the difference between the current local gradient and the previous local gradient, at each node. Since these differences are generally smaller than the gradients themselves, we expect this quantization to introduce lower error.   
• Quantized power iteration: we also use as a benchmark a quantized version of power iteration, a common method for leading-eigenvector computation given by the update rule $\begin{array} { r } { x ^ { ( t + 1 ) }  \frac { A x ^ { ( t ) } } { \| A x ^ { ( t ) } \| } } \end{array}$ . $A x ^ { ( t ) }$ can be computed in distributed fashion by communicating and summing the vectors $A _ { i } x ^ { ( t ) } , i \leq n$ . It is these vectors that we quantize.

![](images/6d998e120bf8860448c5101315720fa2d789501e5ff5ac2220f02aa7861d9625.jpg)  
Figure 1 Convergence results on real datasets

![](images/6dd0842e6f4d2e11ba5b28e98def436000d8841c1c1c9a352b8b84cff74bf8a4.jpg)

![](images/b2c75ba147bb643cae4fc060fd739af113d9ae76638b27bacc00b56136e9cd34.jpg)

![](images/12bbca833816a003e6e8bd9309ca86fc9935ac1a1e768078e7347e0366fbf3fc.jpg)

We show convergence results (Figure 1) for the methods on four real datasets: Human Activity from the MATLAB Statistics and Machine Learning Toolbox, and Mice Protein Expression, Spambase, and Libras Movement from the UCI Machine Learning Repository [9]. All results are averages over 10 runs of the cost function $- x ^ { T } A x$ (for each method, the iterates $x$ are normalized to lie in the 1-ball, so a lower value of $- x ^ { T } A x$ corresponds to being closer to the principal eigenvector).

All four datasets display similar behavior: our approach of Riemannian gradient quantization outperforms naïve Euclidean gradient quantization, and essentially matches the performance of the full-precision method while communicating only 4 bits per coordinate, for a $1 6 \times$ compression. Power iteration converges slightly faster (as would be expected from the theoretical convergence guarantees), but is much more adversely affected by quantization, and reaches a significantly suboptimal result.

We also tested other compression levels, though we do not present figures here due to space constraints: for fewer bits per coordinate (1-3), the performance differences between the four methods are in the same order but more pronounced; however, their convergence curves become more erratic due to increased variance from quantization. For increased number of bits per coordinate, the performance of the quantized methods reaches that of the full-precision methods at between 8-10 bits per coordinate depending on input, and the observed effect of quantized power iteration converging to a suboptimal result gradually diminishes. Our code is publicly available 1.

# 7 Discussion and Future Work

We proposed a communication efficient algorithm to compute the leading eigenvector of a covariance data matrix in a distributed fashion using only limited amount of bits. To the best of our knowledge, this is the first algorithm for this problem with bounded bit complexity. We focused on gradient descent since it provides robust convergence in terms of iterates, and can circumvent issues that arise due to a cost function that only has local convexity-like properties; second, it known to be amenable to quantization.

We would like briefly to discuss the round complexity of our method, emphasizing inherent round complexity trade-offs due to quantization. Indeed, the round complexity of our quantized RGD algo-

rithm for accuracy $\epsilon$ is $\tilde { \mathcal { O } } ( \frac { \gamma } { a ^ { 2 } \delta } )$ , where $\epsilon$ appears inside a logarithm. Specifically, if the initialization√ $x ^ { ( 0 ) }$ is uniformly random in the sphere (section 5.1), then $a = \mathcal { O } ( 1 / \sqrt { d } )$ . If we have warm start (section 5.2), then $a = \mathcal { O } ( 1 )$ . Both statements hold with high probability.

For random initialization, the round complexity of distributed power method is $\begin{array} { r } { \tilde { \mathcal { O } } ( \frac { \lambda _ { 1 } } { \delta } ) = \tilde { \mathcal { O } } ( \frac { \gamma } { \delta } ) } \end{array}$ (see the table at the beginning of page 4 in [11]). This is because the convergence rate of power method is $\tan ( \operatorname { d i s t } ( x _ { k } , x ^ { * } ) ) \stackrel { - } { \leq } ( \stackrel { - } 1 - \lambda _ { 1 } / \delta ) ^ { k } \tan ( \operatorname { d i s t } ( x _ { 0 } , x ^ { * } ) )$ . The dimension is again present in that rate, because $\tan ( \operatorname { d i s t } ( x _ { 0 } , x ^ { * } ) ) = \Theta ( { \sqrt { d } } )$ , but it appears inside a logarithm in the final iteration complexity.

Since we want to insert quantization, the standard way to do so is by starting from a rate which is contracting (see e.g. [22]). It may in theory be possible to add quantization in a non-contracting algorithm, but we are unaware of such a technique. Thus, we firstly prove that RGD with step-size $\frac { a } { \gamma }$ is contracting (based on weak-convexity and quadratic growth) and then we add quantization, paying the price of having an extra $\mathcal { O } ( 1 / a ^ { 2 } ) = \mathcal { O } ( d )$ in our round complexity. Thus, our round complexity is slightly worse than distributed power method, but this is in some sense inherent due to the introduction of quantization. Despite this, our algorithm can achieve low bit complexity and it is more robust in perturbations caused by bit limitations, as our experiments suggest.

We conclude with some interesting points for future work.

First, our choice of step-size is quite conservative since the local convexity properties of the cost function improve when approaching the optimum. Thus, instead of setting the step-size to be $\eta = a / \gamma$ , where $a$ is a lower bound for $\langle x ^ { ( 0 ) } , x ^ { * } \rangle$ , we can re-run the whole analysis with an adaptive step-size $\eta = \langle x ^ { ( t ) } , x ^ { \ast } \rangle / \gamma$ based on an improved value for $a$ from $\langle x ^ { ( t ) } , x ^ { \ast } \rangle$ . This should improve the dependency on the dimension of our complexity results.

Second, the version of Riemannian gradient descent that we have chosen is expensive, since one needs to calculate the exponential map in each step using sin and cos calculations. Practitioners usually use cheaper retraction-based versions, and it would be of interest to study how our analysis can be modified if the exponential map is substituted by a retraction (i.e., any first-order approximation).

Third, it would be interesting to see whether the ideas and techniques presented in this work can also be used for other data science problems in a distributed fashion through communication-efficient Riemannian methods. This could be, for example, PCA for the top $k$ eigenvectors using optimization on the Stiefel or Grassmann manifold [1] and low-rank matrix completion on the fixed-rank manifold [29].

# Acknowledgments and Disclosure of Funding

We would like to thank the anonymous reviewers for helpful comments and suggestions. We also thank Aurelien Lucchi and Antonio Orvieto for fruitful discussions at an early stage of this work. FA is partially supported by the SNSF under research project No. 192363 and conducted part of this work while at IST Austria under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 805223 ScaleML). PD partly conducted this work while at IST Austria and was supported by the European Union’s Horizon 2020 programme under the Marie Skłodowska-Curie grant agreement No. 754411.

# References

[1] P-A Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds. Princeton University Press, 2009.   
[2] Foivos Alimisis, Peter Davies, and Dan Alistarh. Communication-efficient distributed optimization with quantized preconditioners. In International Conference on Machine Learning. PMLR, 2021.   
[3] Dan Alistarh and Janne H. Korhonen. Improved communication lower bounds for distributed optimisation. arXiv preprint arXiv:2010.08222, 2020.   
[4] Maria-Florina Balcan, Vandana Kanchanapally, Yingyu Liang, and David Woodruff. Improved distributed principal component analysis. arXiv preprint arXiv:1408.5823, 2014.

[5] Christopher M Bishop. Pattern recognition and machine learning. Springer, 2006.   
[6] Christos Boutsidis, David P Woodruff, and Peilin Zhong. Optimal principal component analysis in distributed and streaming models. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pages 236–249, 2016.   
[7] Jingjing Bu and Mehran Mesbahi. A note on nesterov’s accelerated method in nonconvex optimization: a weak estimate sequence approach, 2020.   
[8] Peter Davies, Vijaykrishna Gurunathan, Niusha Moshrefi, Saleh Ashkboos, and Dan Alistarh. Distributed variance reduction with optimal communication. In International Conference on Learning Representations, 2021.   
[9] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.   
[10] Jianqing Fan, Dong Wang, Kaizheng Wang, and Ziwei Zhu. Distributed estimation of principal eigenspaces. Annals of statistics, 47(6):3009, 2019.   
[11] Dan Garber, Ohad Shamir, and Nathan Srebro. Communication-efficient algorithms for distributed stochastic principal component analysis. In International Conference on Machine Learning, pages 1203–1212. PMLR, 2017.   
[12] Samuel Horváth, Dmitry Kovalev, Konstantin Mishchenko, Sebastian Stich, and Peter Richtárik. Stochastic distributed learning with gradient quantization and variance reduction. arXiv preprint arXiv:1904.05115, 2019.   
[13] Harold Hotelling. Analysis of a complex of statistical variables into principal components. Journal of educational psychology, 24(6):417, 1933.   
[14] Long-Kai Huang and Sinno Pan. Communication-efficient distributed PCA by Riemannian optimization. In International Conference on Machine Learning, pages 4465–4474. PMLR, 2020.   
[15] Rustem Islamov, Xun Qian, and Peter Richtárik. Distributed second order methods with fast rates and compressed communication. arXiv preprint arXiv:2102.07158, 2021.   
[16] Martin Jaggi, Virginia Smith, Martin Takác, Jonathan Terhorst, Sanjay Krishnan, Thomas ˇ Hofmann, and Michael I Jordan. Communication-efficient distributed dual coordinate ascent. arXiv preprint arXiv:1409.1458, 2014.   
[17] Ravi Kannan, Santosh Vempala, and David Woodruff. Principal component analysis and higher correlations for distributed data. In Conference on Learning Theory, pages 1040–1057. PMLR, 2014.   
[18] Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition, 2020.   
[19] Jakub Konecnˇ y, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, ` and Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.   
[20] Jakub Konecnˇ y and Peter Richtárik. Randomized distributed mean estimation: Accuracy vs. ` communication. Frontiers in Applied Mathematics and Statistics, 4:62, 2018.   
[21] S. Li. Concise Formulas for the Area and Volume of a Hyperspherical Cap. Asian Journal of Mathematics & Statistics, 4(1):66–70, 2011.   
[22] Sindri Magnússon, Hossein Shokri-Ghadikolaei, and Na Li. On maintaining linear convergence of distributed learning and optimization under limited communication. IEEE Transactions on Signal Processing, 68:6101–6116, 2020.   
[23] Andrew Ng, Michael Jordan, and Yair Weiss. On spectral clustering: Analysis and an algorithm. Advances in neural information processing systems, 14:849–856, 2001.   
[24] Erkki Oja and Juha Karhunen. On stochastic approximation of the eigenvectors and eigenvalues of the expectation of a random matrix. Journal of mathematical analysis and applications, 106(1):69–84, 1985.   
[25] Ohad Shamir. A stochastic pca and svd algorithm with an exponential convergence rate. In International Conference on Machine Learning, pages 144–152. PMLR, 2015.   
[26] Ohad Shamir. Fast stochastic algorithms for svd and pca: Convergence properties and convexity. In International Conference on Machine Learning, pages 248–256. PMLR, 2016.

[27] Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization using an approximate newton-type method. In International conference on machine learning, pages 1000–1008. PMLR, 2014.   
[28] Ananda Theertha Suresh, X Yu Felix, Sanjiv Kumar, and H Brendan McMahan. Distributed mean estimation with limited communication. In International Conference on Machine Learning, pages 3329–3337. PMLR, 2017.   
[29] Bart Vandereycken. Low-rank matrix completion by riemannian optimization. SIAM Journal on Optimization, 23(2):1214–1236, 2013.   
[30] Zhiqiang Xu, Xin Cao, and Xin Gao. Convergence analysis of gradient descent for eigenvector computation. International Joint Conferences on Artificial Intelligence, 2018.   
[31] Hongyi Zhang, Sashank J Reddi, and Suvrit Sra. Riemannian svrg: Fast stochastic optimization on riemannian manifolds. arXiv preprint arXiv:1605.07147, 2016.   
[32] Hongyi Zhang and Suvrit Sra. First-order methods for geodesically convex optimization. In Conference on Learning Theory, pages 1617–1638. PMLR, 2016.

# A Geometry of the Sphere

We analyze here briefly some basic notions of the geometry of the sphere that we use in our algorithm and convergence analysis. We refer the reader to [1, p. 73–76] for a more comprehensive presentation.

Tangent Space: The tangent space of the $r$ -dimensional sphere $\mathbb { S } ^ { r }$ at a point $p$ is an $r$ -dimensional vector space, which generalizes the notion of tangent plane in two dimensions. We denote it by $T _ { p } \mathbb { S } ^ { r }$ and a vector $v$ belongs in it, if and only if, it can be written as $\dot { \alpha } ( 0 )$ , where $\alpha \colon ( - \varepsilon , \varepsilon )  { \bar { \mathbb { S } } } ^ { r }$ (for some $\varepsilon > 0$ ) is a smooth curve with $\alpha ( 0 ) = p$ . The tangent space at $p$ can be given also in an explicit way, as the set of all vectors in $\mathbb { R } ^ { r + 1 }$ orthogonal to $p$ with respect to the usual inner product. Given a vector $w \in \mathbb { R } ^ { r + 1 }$ , we can always project it orthogonally in any tangent space of $\mathbb { S } ^ { r }$ . Taking all vectors to be column vectors, the orthogonal projection in $T _ { p } \mathbb { S } ^ { r }$ satisfies

$$
P _ {p} (w) = (I - p p ^ {T}) w.
$$

Geodesics: Geodesics on high-dimensional surfaces are defined to be locally length-minimizing curves. On the $d$ -dimensional sphere, they coincide with great circles. These can be computed explicitly and give rise to the exponential and logarithmic maps. The exponential map at a point $p$ is defined as $\exp _ { p } : T _ { p } \mathbb { S } ^ { r }  \mathbb { S } ^ { r }$ with $\begin{array} { r } { \exp _ { p } ( v ) = c ( 1 ) } \end{array}$ , with $c$ being the (unique) geodesic (i.e., length minimizing part of great circle) satisfying $c ( 0 ) = p$ and $\dot { c } ( 0 ) = v$ . Defining the exponential map in this way makes it invertible with inverse $\log _ { p }$ . The exponential and logarithmic map are given by well-known explicit formulas:

$$
\exp_ {p} (v) = \cos (\| v \|) p + \sin (\| v \|) \frac {v}{\| v \|}, \quad \log_ {p} (q) = \operatorname {a r c c o s} (\langle q, p \rangle) \frac {P _ {p} (q - p)}{\| P _ {p} (q - p) \|}. \tag {5}
$$

The distance between points $p$ and $q$ measured intrinsically on the sphere is

$$
\operatorname {d i s t} (p, q) = \left\| \log_ {p} (q) \right\| = \operatorname {a r c c o s} (\langle q, p \rangle). \tag {6}
$$

Notice that $\langle q , p \rangle = \| q \| \| p \| \cos ( \angle ( p , q ) ) = \cos ( \angle ( p , q ) )$ , thus the distance of $p$ and $q$ is actually the angle between them.

The inner product inherited by the ambient Euclidean space $\mathbb { R } ^ { r + 1 }$ provides a way to transport vectors from a tangent space to another one, parallelly to the geodesics. This operation is called parallel transport and constitutes an orthogonal isomorphism between the two tangent spaces. We denote $\Gamma _ { p } ^ { q }$ for the parallel transport from $T _ { p } M$ to $T _ { q } M$ along the geodesic connecting $p$ and $q$ . If $q = \exp _ { p } ( t v )$ , then parallel transport is given by the formula

$$
\Gamma_ {p} ^ {q} u = \left(I _ {d} + \cos (t \| v \| - 1) \frac {v v ^ {T}}{\| v \| ^ {2}} - \sin (t \| v \|) \frac {p v ^ {t}}{\| v \|}\right) u.
$$

Riemannian Gradient: Given a function $f \colon { \mathbb { S } } ^ { r }  { \mathbb { R } }$ , we can define its differential at a point $p$ in a tangent direction $v \in T _ { p } \mathbb { S } ^ { r }$ as

$$
d f (p) v = \lim _ {t \to 0} \frac {f (\alpha (t)) - f (p)}{t},
$$

where $\alpha \colon ( - \varepsilon , \varepsilon ) \to \mathbb { S } ^ { r }$ is a smooth curve defined such that $\alpha ( 0 ) = p$ and $\dot { \alpha } ( 0 ) = v$ . Using this intrinsically defined Riemannian differential, we can define the Riemannian gradient at $p$ as a vector gra $1 f ( p ) \in T _ { p } \mathbb { S } ^ { r }$ , such that

$$
\langle \operatorname {g r a d} f (p), v \rangle = d f (p) v,
$$

for any $v \in T _ { p } \mathbb { S } ^ { r }$ . In the case of the sphere, we can also compute the Riemannian gradient by orthogonally projecting the Euclidean gradient $\nabla f ( \boldsymbol { p } )$ computed in the ambient space into the tangent space of $p$ :

$$
\operatorname {g r a d} f (p) = P _ {p} (\nabla f (p)).
$$

Geodesic Convexity and Smoothness: In this paragraph we define convexity-like properties in the context of the sphere, which we employ in our analysis.

Definition 11. A subset ${ \mathcal { A } } \subseteq { \mathbb { S } } ^ { r }$ is called geodesically uniquely convex, if every two points in A are connected by a unique geodesic.

Open hemispheres satisfy this definition, thus they are geodesically uniquely convex. Actually, they are the biggest possible subsets of the sphere with this property.

Definition 12. A smooth function $f \colon A \to \mathbb { R }$ is called geodesically $\gamma$ -smooth if

$$
\left\| \operatorname {g r a d} f (p) - \Gamma_ {q} ^ {p} \operatorname {g r a d} f (q) \right\| \leq \gamma \operatorname {d i s t} (p, q)
$$

for any $p , q \in { \mathcal { A } }$ .

Definition 13. A smooth function $f \colon { \mathcal { A } }  \mathbb { R }$ defined in a geodesically uniquely convex subset $A \subset \mathbb { S } ^ { r }$ is called geodesically $a$ -weakly-quasi-convex (for some $a > 0$ ) with respect to a point $p \in A$ $i f$

$$
a (f (x) - f (p)) \leq \langle \operatorname {g r a d} f (x), - \log_ {x} (p) \rangle \quad \text {f o r a l l} x \in \mathcal {A}.
$$

Observe that weak-quasi-convexity implies that any local minimum of $f$ lying in the interior of $\mathcal { A }$ is also a global one.

Definition 14. A function $f \colon A \to \mathbb { R }$ is said to satisfy quadratic growth condition with constant $\mu ,$ if

$$
f (x) - f ^ {*} \geq \frac {\mu}{2} \mathrm {d i s t} ^ {2} (x, x ^ {*}) \quad \text {f o r a l l} x \in \mathcal {A},
$$

where $f ^ { * }$ is the minimum of $f$ and $x ^ { * }$ the global minimizer of $f$ closest to x.

Quadratic growth condition is slightly weaker than the so-called gradient dominance one:

Definition 15. A function $f \colon { \mathcal { A } } \to \mathbb { R }$ is said to be $\mu$ -gradient dominated if

$$
f (x) - f ^ {*} \leq \frac {1}{2 \mu} \| \operatorname {g r a d} f (x) \| ^ {2} \quad \text {f o r a l l} x \in \mathcal {A}.
$$

Gradient dominance with constant $\mu$ implies quadratic growth with the same constant. The proof can be done by a direct Riemannian adaptation of the argument from [18, p. 13–14].

Tangent Space Distortion: The only non-trivial geometric result we use in this work is that the geodesics of the sphere spread more slowly than in a flat space. This is a direct application of the Rauch comparison theorem since the sphere has constant positive curvature.

Proposition 16. Let ∆abc a geodesic triangle on the sphere. Then we have

$$
\operatorname {d i s t} (a, b) \leq \| \log_ {c} (a) - \log_ {c} (b) \|.
$$

Notice that in Euclidean space we have equality in this bound.

# B Properties of the Cost and Convergence of Single-Node Gradient Descent

In this appendix, we will prove the convergence of Riemannian gradient descent applied to

$$
\min  _ {x \in \mathbb {S} ^ {d - 1}} f (x) \quad \text {w i t h} f (x) = - x ^ {T} A x
$$

and $A \in \mathbb { R } ^ { d \times d }$ a symmetric and positive definite matrix. In the following, we will denote the eigenvalues of $A$ by $\lambda _ { 1 } > \lambda _ { 2 } \geq \dots \geq \lambda _ { d } > 0$ and their corresponding orthonormal eigenvectors by $v _ { 1 } , v _ { 2 } , \ldots , v _ { d }$ . Denote the spectral gap by $\delta = \lambda _ { 1 } - \lambda _ { 2 } > 0$ .

Let $f ^ { * } = - \lambda _ { 1 }$ denote the minimum of $f$ on $\mathbb { S } ^ { d - 1 }$ which is attained at $x ^ { * } = v _ { 1 }$

Proposition 1. The function $f ( x ) = - x ^ { T } A x$ satisfies

$$
2 a \left(f (x) - f ^ {*}\right) \leq \langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \rangle
$$

for any $x \in B _ { a }$ with $a > 0$ .

Proof. For any $x \in B _ { a }$ , we can write

$$
x = \sum_ {i = 1} ^ {d} \alpha_ {i} v _ {i}, \quad A x = \sum_ {i = 1} ^ {d} \lambda_ {i} \alpha_ {i} v _ {i} \tag {7}
$$

for some scalars $\alpha _ { i }$ . Recall that $\alpha _ { 1 } \geq a > 0$ by definition (2) of $B _ { a }$ .

With the orthogonal projector $P _ { x } = I - x x ^ { T }$ onto the tangent space $T _ { x } \mathbb { S } ^ { d - 1 }$ , we get from (5) that

$$
\begin{array}{l} \langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \rangle = \left\langle P _ {x} \nabla f (x), \frac {\operatorname {d i s t} \left(x , x ^ {*}\right)}{\| P _ {x} \left(x - x ^ {*}\right) \|} P _ {x} \left(x - x ^ {*}\right) \right\rangle \\ = \frac {\operatorname {d i s t} \left(x , x ^ {*}\right)}{\| P _ {x} \left(x - x ^ {*}\right) \|} \langle P _ {x} \nabla f (x), x - x ^ {*} \rangle . \\ \end{array}
$$

because $P _ { x } ^ { 2 } = P _ { x }$

Direct calculation now gives

$$
\begin{array}{l} \left\langle P _ {x} \nabla f (x), x - x ^ {*} \right\rangle = - 2 x ^ {T} A x + 2 \left\langle A x, x ^ {*} \right\rangle - 2 f (x) \| x \| ^ {2} + 2 f (x) \left\langle x, x ^ {*} \right\rangle \\ = 2 f (x) + 2 \lambda_ {1} \alpha_ {1} - 2 f (x) + 2 f (x) \alpha_ {1} \\ = 2 \alpha_ {1} (f (x) + \lambda_ {1}) = 2 \alpha_ {1} (f (x) - f ^ {*}) \geq 0. \\ \end{array}
$$

It is easy to verify that $\mathrm { d i s t } ( x , x ^ { * } ) \geq \| P _ { x } ( x - x ^ { * } ) \|$ . We thus obtain

$$
\langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \rangle \geq 2 \alpha_ {1} \left(f (x) - f ^ {*}\right),
$$

which gives the desired result since $\alpha _ { 1 } \geq a$

![](images/6f25d46ee29d6b495289764c1500678af877473aca57005af9500ed078692f2a.jpg)

Proposition 2. The function $f ( x ) = - x ^ { T } A x$ satisfies

$$
f (x) - f ^ {*} \geq \frac {\delta}{4} \mathrm {d i s t} ^ {2} (x, x ^ {*})
$$

for any $x \in B _ { a }$ with $a > 0$ .

Proof. The proof follows the one in [30, Lemma 2]. Using the expansions in (7), we get

$$
x ^ {T} A x = \sum_ {i = 1} ^ {d} \lambda_ {i} \alpha_ {i} ^ {2} = \lambda_ {1} \alpha_ {1} ^ {2} + \sum_ {i = 2} ^ {d} \lambda_ {i} \alpha_ {i} ^ {2} \leq \lambda_ {1} \alpha_ {1} ^ {2} + \lambda_ {2} (1 - \alpha_ {1} ^ {2})
$$

since $\begin{array} { r } { \| x \| ^ { 2 } = 1 = \sum _ { i = 1 } ^ { d } \alpha _ { i } ^ { 2 } } \end{array}$ . From (6), we have that $\alpha _ { 1 } = \cos ( \mathrm { { d i s t } } ( x , x ^ { * } ) )$ and so

$$
x ^ {T} A x \leq \lambda_ {1} \cos^ {2} \left(\operatorname {d i s t} \left(x, x ^ {*}\right)\right) + \lambda_ {2} \sin^ {2} \left(\operatorname {d i s t} \left(x, x ^ {*}\right)\right).
$$

Direct calculation now shows

$$
\begin{array}{l} f (x) - f ^ {*} = - x ^ {T} A x + \lambda_ {1} \geq \lambda_ {1} - \lambda_ {1} \cos^ {2} \left(\operatorname {d i s t} \left(x, x ^ {*}\right)\right) - \lambda_ {2} \sin^ {2} \left(\operatorname {d i s t} \left(x, x ^ {*}\right)\right) \\ = \lambda_ {1} \sin^ {2} \left(\operatorname {d i s t} \left(x, x ^ {*}\right)\right) - \lambda_ {2} \sin^ {2} \left(\operatorname {d i s t} \left(x, x ^ {*}\right)\right) = \delta \sin^ {2} \left(\operatorname {d i s t} \left(x, x ^ {*}\right)\right). \\ \end{array}
$$

Since $x \in B _ { a }$ with $a > 0$ , we have that $x$ and $x ^ { * }$ are in the same hemisphere and thus $d =$ $\operatorname { d i s t } ( x , x ^ { * } ) \leq \pi / 2$ . The desired result follows using $\sin ( \phi ) \geq \phi / 2$ for $0 \le \phi \le \pi / 2$ .

Proposition 3. A function $f \colon \mathbb { S } ^ { d - 1 } \to \mathbb { R }$ which satisfies quadratic growth with constant $\mu > 0$ and is $2 a$ -weakly-quasi-convex with respect to a minimizer $x ^ { * }$ in a ball of this minimizer, satisfies for any $x$ in the same ball the inequality

$$
f (x) - f ^ {*} \leq \frac {1}{a} \left\langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \right\rangle - \frac {\mu}{2} \operatorname {d i s t} ^ {2} \left(x, x ^ {*}\right).
$$

Proof. From quadratic growth and weak-quasi-convexity, we have

$$
\frac {\mu}{2} \operatorname {d i s t} ^ {2} (x, x ^ {*}) \leq f (x) - f ^ {*} \leq \frac {1}{2 a} \langle \operatorname {g r a d} f (x), - \log_ {x} (x ^ {*}) \rangle .
$$

Now, again by weak-quasi-convexity

$$
\begin{array}{l} f (x) - f ^ {*} \leq \frac {1}{2 a} \left\langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \right\rangle + \frac {\mu}{2} \operatorname {d i s t} ^ {2} \left(x, x ^ {*}\right) - \frac {\mu}{2} \operatorname {d i s t} ^ {2} \left(x, x ^ {*}\right) \\ \leq \frac {1}{a} \left\langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \right\rangle - \frac {\mu}{2} \operatorname {d i s t} ^ {2} \left(x, x ^ {*}\right) \\ \end{array}
$$

by substituting the previous inequality.

![](images/78608007e7e33025596ca78846692a7e8ccb03193b3978fb253917f6ee3964ea.jpg)

Proposition 4. The function $f ( x ) = - x ^ { T } A x$ satisfies

$$
\left\| \operatorname {g r a d} f (x) \right\| ^ {2} \geq \delta a ^ {2} (f (x) - f ^ {*})
$$

for any $x \in B _ { a }$ with $a > 0$ .

Proof. By Proposition 3, we have

$$
f (x) - f ^ {*} \leq \frac {1}{a} \left\langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \right\rangle - \frac {\delta}{4} \operatorname {d i s t} ^ {2} \left(x, x ^ {*}\right)
$$

since, in our case, $\mu = \delta / 2$ . Using $\begin{array} { r } { \langle x , y \rangle \leq \frac { 1 } { 2 } ( \| x \| ^ { 2 } + \| y \| ^ { 2 } ) } \end{array}$ for all $x , y \in \mathbb { R } ^ { d }$ , we can write for any positive $\rho$ that

$$
\langle \operatorname {g r a d} f (x), - \log_ {x} \left(x ^ {*}\right) \rangle \leq \frac {\rho}{2} \| \operatorname {g r a d} f (x) \| ^ {2} + \frac {1}{2 \rho} \| \log_ {x} \left(x ^ {*}\right) \| ^ {2}.
$$

Combining with $\textstyle \rho = { \frac { 2 } { a \delta } }$ and using (6), we get

$$
f (x) - f ^ {*} \leq \frac {1}{a} \frac {1}{a \delta} \| \operatorname {g r a d} f (x) \| ^ {2} + \frac {1}{a} \frac {a \delta}{4} \operatorname {d i s t} ^ {2} (x, x ^ {*}) - \frac {\delta}{4} \operatorname {d i s t} ^ {2} (x, x ^ {*}) = \frac {1}{a ^ {2} \delta} \| \operatorname {g r a d} f (x) \| ^ {2}.
$$

Proposition 5. The function $f ( x ) = - x ^ { T } A x$ is geodesically $2 \lambda _ { 1 }$ -smooth on the sphere.

Proof. The proof can be found also in [14, Lemma 1]. For $p \in \mathbb { S } ^ { d - 1 }$ and $v \in T _ { p } \mathbb { S } ^ { d - 1 }$ with $\lVert \boldsymbol { v } \rVert = 1$ we have that the Riemannian Hessian of $f$ satisfies

$$
\langle v, \nabla^ {2} f (p) v \rangle = \langle v, - (I - p p ^ {T}) 2 A v + p ^ {T} 2 A p v \rangle = - 2 v ^ {T} A v + 2 p ^ {T} A p \leq 2 \lambda_ {1}
$$

because $v ^ { T } A v \geq 0$ (since $A$ is symmetric and positive semi-definite) and $p ^ { T } A p \leq \lambda _ { 1 }$ , by the definition of eigenvalues and $\left\| p \right\| = 1$ . We have also used that $\lVert \boldsymbol { v } \rVert = 1$ and $\langle p , v \rangle = 0$ . Notice now that the largest eigenvalue of the Hessian is the maximum of $\langle \dot { v } , \nabla ^ { 2 } f ( p ) v \rangle$ over all $v \in T _ { p } \mathbb { S } ^ { r }$ with $\lVert \boldsymbol { v } \rVert = 1$ , thus less or equal than $2 \lambda _ { 1 }$ . This result easily implies Def. 12, since the Riemannian Hessian is the covariant derivative of the Riemannian gradient.

Proposition 6. An iterate of Riemannian gradient descent for $f ( x ) = - x ^ { T } A x$ starting from a point $x ^ { ( t ) } \in B _ { a }$ and with step-size $\eta \leq a / \gamma$ where $\gamma \geq 2 \lambda _ { 1 }$ , produces a point $x ^ { ( t + 1 ) }$ that satisfies

$$
\operatorname {d i s t} ^ {2} \left(x ^ {(t + 1)}, x ^ {*}\right) \leq (1 - a \mu \eta) \operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right) \quad f o r \mu = \delta / 2.
$$

Proof. By definition of $x ^ { ( t + 1 ) }$ , we have $\log _ { x ^ { ( t ) } } ( x ^ { ( t + 1 ) } ) = - \eta \operatorname { g r a d } f ( x ^ { ( t ) } )$ . Applying Proposition 16, we can thus write

$$
\begin{array}{l} \left. \operatorname {d i s t} ^ {2} \left(x ^ {(t + 1)}, x ^ {*}\right) \leq \| - \eta \operatorname {g r a d} f \left(x ^ {(t)}\right) - \log_ {x ^ {(t)}} \left(x ^ {*}\right) \| ^ {2} \right. \\ = \eta^ {2} \| \operatorname {g r a d} f (x ^ {(t)}) \| ^ {2} + \| \log_ {x ^ {(t)}} (x ^ {*}) \| ^ {2} + 2 \eta \langle \operatorname {g r a d} f (x ^ {(t)}), \log_ {x ^ {(t)}} (x ^ {*}) \rangle . \\ \end{array}
$$

By Proposition 3 and 5, we have

$$
\begin{array}{l} \frac {1}{a} \left\langle \operatorname {g r a d} f \left(x ^ {(t)}\right), \log_ {x ^ {(t)}} \left(x ^ {*}\right) \right\rangle \leq f ^ {*} - f \left(x ^ {(t)}\right) - \frac {\mu}{2} \operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right) \\ \leq - \frac {1}{2 \gamma} \| \operatorname {g r a d} f (x ^ {(t)}) \| ^ {2} - \frac {\mu}{2} \operatorname {d i s t} ^ {2} (x ^ {(t)}, x ^ {*}). \\ \end{array}
$$

Multiplying with $2 \eta a$ and using $\eta \leq a / \gamma$ , we get

$$
\begin{array}{l} 2 \eta \left\langle \operatorname {g r a d} f \left(x ^ {(t)}\right), \log_ {x ^ {(t)}} \left(x ^ {*}\right) \right\rangle \leq - \frac {\eta a}{\gamma} \| \operatorname {g r a d} f \left(x ^ {(t)}\right) \| ^ {2} - \mu \eta a \operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right) \\ \leq - \eta^ {2} \| \operatorname {g r a d} f (x ^ {(t)}) \| ^ {2} - \mu \eta a \operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right). \\ \end{array}
$$

Substituting to the first inequality, we get the desired result.

# C Convergence of Distributed Gradient Descent on the Sphere

Lemma 8. If $\begin{array} { r } { \eta \le \frac { \cos ( D ) } { \gamma } } \end{array}$ , the previous quantized gradient descent algorithm produces iterates $x ^ { ( t ) }$ γ and quantized gradients $q ^ { ( t ) }$ that satisfy

$$
\operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right) \leq \xi^ {t} D ^ {2}, \quad \left\| q _ {i} ^ {(t)} - \operatorname {g r a d} f _ {i} \left(x ^ {(t)}\right) \right\| \leq \frac {\theta R ^ {(t)}}{2 n}, \quad \left\| q ^ {(t)} - \operatorname {g r a d} f \left(x ^ {(t)}\right) \right\| \leq \theta R ^ {(t)}.
$$

Proof. We do the proof by induction. We start from the case that $t = 0$ . The first inequality is direct by the definition of $D$ .

For the second one we have

$$
\left\| \operatorname {g r a d} f _ {i} \left(x ^ {(0)}\right) - \operatorname {g r a d} f _ {i _ {0}} \left(x ^ {(0)}\right) \right\| \leq \left\| \operatorname {g r a d} f _ {i} \left(x ^ {(0)}\right) \right\| + \left\| \operatorname {g r a d} f _ {i _ {0}} \left(x ^ {(0)}\right) \right\| \leq 2 \gamma \pi
$$

and by the definition of quantization, we get

$$
\left\| \operatorname {g r a d} f _ {i} \left(x ^ {(0)}\right) - q _ {i} ^ {(0)} \right\| \leq \frac {\theta R ^ {(0)}}{2 n}.
$$

Similarly for the third one, we have

$$
\left\| \operatorname {g r a d} f \left(x ^ {(0)}\right) - r ^ {(0)} \right\| \leq \sum_ {i = 1} ^ {n} \left\| \operatorname {g r a d} f _ {i} \left(x ^ {(0)}\right) - q _ {i} ^ {(0)} \right\| \leq \frac {\theta R ^ {(0)}}{2}.
$$

Then,

$$
\| r ^ {(0)} - \operatorname {g r a d} f _ {i} (x ^ {(0)}) \| \leq \| r ^ {(0)} - \operatorname {g r a d} f (x ^ {(0)}) \| + \| \operatorname {g r a d} f (x ^ {(0)}) - \operatorname {g r a d} f _ {i} (x ^ {(0)}) \| \leq \frac {\theta R ^ {(0)}}{2} + 2 \pi \gamma .
$$

By the definition of the quantization, we have

$$
\| q ^ {(0)} - \operatorname {g r a d} f (x ^ {(0)}) \| \leq \frac {\theta R ^ {(0)}}{2} \leq \theta R ^ {(0)}.
$$

We assume now that the inequalities hold for $t$ and we wish to prove that they continue to hold for $t + 1$ .

We start with the first one and denote by $\tilde { x } ^ { ( t + 1 ) }$ the iteration of exact gradient descent starting from $x ^ { ( t ) }$ . Since di $\mathrm { s t } ( x ^ { ( t ) } , x ^ { * } ) \leq D$ , we have that $x ^ { ( t ) } \in B _ { a }$ with $a = \cos ( D )$ .

We have

$$
\operatorname {d i s t} \left(x ^ {(t + 1)}, x ^ {*}\right) \leq \operatorname {d i s t} \left(x ^ {(t + 1)}, \tilde {x} ^ {(t + 1)}\right) + \operatorname {d i s t} \left(\tilde {x} ^ {(t + 1)}, x ^ {*}\right) \leq \| \eta \operatorname {g r a d} f \left(x ^ {(t)}\right) - \eta q ^ {(t)} \| + \sqrt {\sigma} \operatorname {d i s t} \left(x ^ {(t)}, x ^ {*}\right).
$$

We have the last inequality, because $\mathrm { d i s t } ( \tilde { x } ^ { ( t + 1 ) } , x ^ { * } ) \le \sqrt { \sigma } \mathrm { d i s t } ( x ^ { ( t ) } , x ^ { * } )$ by Proposition 6 and $\begin{array} { r } { \mathrm { d i s t } ( x ^ { ( t + 1 ) } , \tilde { x } ^ { ( t + 1 ) } ) \leq \| \log _ { x ^ { ( t ) } } ( x ^ { ( t + 1 ) } ) - \log _ { x ^ { ( t ) } } ( \tilde { x } ^ { ( t + 1 ) } ) \| = \| \eta \mathrm { g r a d } f ( x ^ { ( t ) } ) - \eta q ^ { ( t ) } \| \| } \end{array}$ by Proposition 16.

Thus

$$
\operatorname {d i s t} \left(x ^ {(t + 1)}, x ^ {*}\right) \leq \frac {a}{\gamma} \theta R ^ {(t)} + \sqrt {\sigma} (\sqrt {\xi}) ^ {t} D \leq \theta K (\sqrt {\xi}) ^ {t} D + \sqrt {\sigma} (\sqrt {\xi}) ^ {t} D \leq (\theta K + \sqrt {\sigma}) (\sqrt {\xi}) ^ {t} D \leq (\sqrt {\xi}) ^ {t + 1} D
$$

which concludes the induction for the first inequality.

For the second inequality, we have

$$
\begin{array}{l} \left\| \operatorname {g r a d} f _ {i} \left(x ^ {(t + 1)}\right) - \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} q _ {i} ^ {(t)} \right\| \leq \left\| \operatorname {g r a d} f _ {i} \left(x ^ {(t + 1)}\right) - \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} \operatorname {g r a d} f _ {i} \left(x ^ {(t)}\right) \right\| + \left\| \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} \operatorname {g r a d} f _ {i} \left(x ^ {(t)}\right) - \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} q _ {i} ^ {(t)} \right\| \\ \leq \gamma_ {i} \operatorname {d i s t} \left(x ^ {(t + 1)}, x ^ {(t)}\right) + \| \operatorname {g r a d} f _ {i} \left(x ^ {(t)}\right) - q _ {i} ^ {(t)} \| \\ \leq 2 \frac {\gamma}{n} (\sqrt {\xi}) ^ {t} D + \theta \frac {R ^ {(t)}}{n} = 2 \frac {\gamma}{n} (\sqrt {\xi}) ^ {t} D + \theta \gamma K (\sqrt {\xi}) ^ {t} D / n \\ = (2 / K + \theta) K \gamma (\sqrt {\xi}) ^ {t} D / n \leq (\sqrt {\sigma} + \theta K) K \gamma (\sqrt {\xi}) ^ {t} D / n = \frac {R ^ {(t + 1)}}{n} \\ \end{array}
$$

and by the definition of the quantization scheme, we have

$$
\left\| \operatorname {g r a d} f _ {i} \left(x ^ {(t + 1)}\right) - q _ {i} ^ {(t + 1)} \right\| \leq \frac {\theta R ^ {(t + 1)}}{2 n}.
$$

For the third inequality, we have

$$
\| r ^ {(t + 1)} - \operatorname {g r a d} f (x ^ {(t + 1)}) \| \leq \sum_ {i = 1} ^ {n} \| q _ {i} ^ {(t + 1)} - \operatorname {g r a d} f _ {i} (x ^ {(t + 1)}) \| \leq \frac {\theta R ^ {(t + 1)}}{2}
$$

and

$$
\begin{array}{l} \left\| r ^ {(t + 1)} - \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} q ^ {(t)} \right\| \leq \left\| r ^ {(t + 1)} - \operatorname {g r a d} f (x ^ {(t + 1)}) \right\| + \left\| \operatorname {g r a d} f (x ^ {(t + 1)}) - \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} \operatorname {g r a d} f (x ^ {(t)}) \right\| \\ + \left\| \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} \operatorname {g r a d} f (x ^ {(t)}) - \Gamma_ {x ^ {(t)}} ^ {x ^ {(t + 1)}} q ^ {(t)} \right\| \\ \leq \frac {\theta R ^ {(t + 1)}}{2} + \gamma \operatorname {d i s t} \left(x ^ {(t + 1)}, x ^ {(t)}\right) + \theta R ^ {(t)} \\ \leq \frac {\theta R ^ {(t + 1)}}{2} + R ^ {(t + 1)} = \left(1 + \frac {\theta}{2}\right) R ^ {(t + 1)} \\ \end{array}
$$

by using again the argument for deriving the second inequality. The last inequality implies that

$$
\left\| \operatorname {g r a d} f \left(x ^ {(t + 1)}\right) - q ^ {(t + 1)} \right\| \leq \frac {\theta R ^ {(t + 1)}}{2}
$$

by the definition of quantization. Summing the two last inequalities completes the induction.

![](images/35d779c063e1605015f328f6552c38d4d3ff379178da8ec07619809af106814b.jpg)

Theorem 9. Let $\begin{array} { r } { \eta \le \frac { \cos ( D ) } { \gamma } } \end{array}$ . Then, the previous quantized gradient descent algorithm needs at most

$$
b = \mathcal {O} \left(n d \frac {1}{\cos (D) \delta \eta} \log \left(\frac {n}{\cos (D) \delta \eta}\right) \log \left(\frac {D}{\epsilon}\right)\right)
$$

bits in total to estimate the leading eigenvector with an accuracy  measured in intrinsic distance.

Proof. For computing the cost of quantization at each step, we use Proposition 7.

The communication cost of encoding each grad $f _ { i }$ at $t = 0$

$$
\mathcal {O} \left(d \log \frac {2 \gamma \pi}{\frac {\theta R ^ {(0)}}{2 n}}\right) = \mathcal {O} \left(d \log \frac {4 n \gamma \pi}{\theta \gamma K \pi}\right) \leq \mathcal {O} \left(d \log \frac {2 n}{\theta}\right).
$$

Now we use that $\begin{array} { r } { \sigma \geq \frac { 1 } { 2 } } \end{array}$ and have

$$
\frac {1}{\theta} = \frac {4}{\sqrt {\sigma} (1 - \sqrt {\sigma})} \leq \frac {1 2}{1 - \sigma} = \frac {1 2}{\cos (D) \eta \mu}.
$$

Thus, the previous cost becomes

$$
\mathcal {O} \left(d \log \frac {2 \gamma \pi}{\frac {\theta R ^ {(0)}}{2 n}}\right) = \mathcal {O} \left(d \log \frac {n}{\cos (D) \eta \mu}\right).
$$

The communication cost of deconding each $q _ { i } ^ { ( 0 ) }$ in the master node is

$$
\mathcal {O} \left(d \log \frac {2 \gamma \pi + \frac {\theta R ^ {(0)}}{2}}{\frac {\theta R ^ {(0)}}{2}}\right) \leq \mathcal {O} \left(d \log \frac {2 \gamma \pi}{\frac {\theta R ^ {(0)}}{2}}\right) = \mathcal {O} \left(d \log \frac {1}{\cos (D) \eta \mu}\right).
$$

This is because $2 \gamma \pi \geq \frac { \theta R ^ { ( 0 ) } } { 2 }$ θR(0)

Thus, the total communication cost at $t = 0$ is

$$
\mathcal {O} \left(n d \log \frac {n}{\cos (D) \eta \mu}\right).
$$

For $t > 0$ , the cost of encoding grad $f _ { i }$ ’s is

$$
\mathcal {O} \left(n d \log \frac {R ^ {(t + 1)} / n}{\theta R ^ {(t + 1)} / 2 n}\right) = \mathcal {O} \left(n d \log \frac {2}{\theta}\right) = \mathcal {O} \left(n d \log \frac {1}{\cos (D) \eta \mu}\right).
$$

as before.

The cost of decoding in the master node is

$$
\mathcal {O} \left(n d \log \frac {(1 + \theta / 2) R ^ {(t + 1)}}{\theta R ^ {(t + 1)} / 2}\right) = \mathcal {O} \left(n d \log \frac {1}{\theta}\right) = \mathcal {O} \left(n d \log \frac {1}{\cos (D) \eta \mu}\right).
$$

because $\theta / 2 \leq 1$ .

Thus, the cost in each round of communication is in general bounded by

$$
\mathcal {O} \left(n d \log \frac {(1 + \theta / 2) R ^ {(t + 1)}}{\theta R ^ {(t + 1)} / 2}\right) = \mathcal {O} \left(n d \log \frac {n}{\cos (D) \eta \mu}\right).
$$

Our algorithm reaches accuracy $a$ in function values if

$$
\operatorname {d i s t} \left(x ^ {(t)}, x ^ {*}\right) \leq \epsilon .
$$

We can now write

$$
\operatorname {d i s t} ^ {2} \left(x ^ {(t)}, x ^ {*}\right) \leq \xi^ {t} D ^ {2} \leq e ^ {- (1 - \xi) t} D ^ {2}.
$$

Thus, we need to run our algorithm for

$$
\mathcal {O} \left(\frac {1}{1 - \xi} \log \frac {D}{\epsilon}\right) \leq \mathcal {O} \left(\frac {1}{\cos (D) \mu \eta} \log \frac {D}{\epsilon}\right)
$$

many iterates to reach accuracy $a$ .

The total communication cost for doing that is

$$
\mathcal {O} \left(\frac {1}{\cos (D) \mu \eta} \log \frac {D}{\epsilon} n d \log \frac {n}{\cos (D) \eta \mu}\right) = \mathcal {O} \left(n d \frac {1}{\cos (D) \mu \eta} \log \frac {n}{\cos (D) \mu \eta} \log \frac {D}{\epsilon}\right)
$$

Substituting

$$
\mu = \frac {\delta}{2}
$$

by Proposition 2, we get

$$
\mathcal {O} \left(n d \frac {1}{\cos (D) \delta \eta} \log \frac {n}{\cos (D) \delta \eta} \log \frac {D}{\epsilon}\right)
$$

many bits in total.

![](images/8d7482c12eda16c33e3e456f672c78fc20d1fbde33ab90e785db77c8919876b9.jpg)

# D Initialization

# D.1 Uniformly Random Initialization

We estimate numerically the constant $c$ that is used in (3) of Section 5.1.

Let $x ^ { ( 0 ) }$ be chosen from a uniform distribution on the sphere $\mathbb { S } ^ { d - 1 }$ . We are interested in $\alpha _ { 1 } = v _ { 1 } ^ { T } x ^ { ( 0 ) }$ for some fixed $v _ { 1 } \in \mathbb { S } ^ { d - 1 }$ . By spherical symmetry, $\alpha _ { 1 }$ is distributed in the same way as the first component of $x ^ { ( 0 ) }$ . Let $A _ { d } ( h )$ be the surface of the hyperspherical cap of $\mathbb { S } ^ { d - 1 }$ with height $h \in [ 0 , 1 ]$ . Then it is obvious that

$$
\mathbb {P} (| \alpha_ {1} | \geq a) = A _ {d} (1 - a) / A _ {d} (1) = I _ {1 - a ^ {2}} (\frac {d - 1}{2}, \frac {1}{2}),
$$

where we used the well-known formula for $A _ { d } ( h )$ in terms of the regularized incomplete Beta function $I _ { x } ( a , b )$ ; see, e.g., [21]. Solving the above expression2 for $a$ when it equals a given probability $1 - p$ ,

we can calculate the interval $[ - 1 , - a ] \cup [ a , 1 ]$ in which $\alpha _ { 1 }$ will lie for a random $x ^ { ( 0 ) }$ up to probability $1 - p$ .

In the figure below, we have plotted these values of $a$ divided by $p / \sqrt { d }$ for $p \quad =$ $1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 }$ . Numerically, there is strong evidence that $a \geq c p / \sqrt { d }$ with $c = 1 . 2 5$ .

![](images/cdb00735966859588bff3fc536d6a11c5ae23f265abfaa8ad694d1f12cd03681.jpg)

# D.2 Initialization in the Leading Eigenvector of the Master

Proposition 10. Assume that our data are i.i.d. and sampled from a distribution $\mathcal { D }$ bounded in $\ell _ { 2 }$ norm by a constant h. Given that the eigengap δ and the number of machines n satisfy

$$
\delta \geq \Omega \left(\sqrt {n} \sqrt {\log \frac {d}{p}}\right), \tag {4}
$$

we have that the previous quantization costs $\mathcal { O } ( n d )$ many bits and $\langle x ^ { ( 0 ) } , x ^ { * } \rangle$ is lower bounded by a constant with probability at least $1 - p$ .

Proof. By Lemma 3 in [14] we have that

$$
\left\| A _ {i _ {0}} - \frac {1}{n} \sum_ {i = 1} ^ {n} A _ {i} \right\| ^ {2} \leq \frac {3 2 \log \left(\frac {d}{p}\right) h ^ {2}}{n}
$$

which implies that

$$
\left\| n A _ {i _ {0}} - \sum_ {i = 1} ^ {n} A _ {i} \right\| ^ {2} \leq 3 2 n \log \left(\frac {d}{p}\right) h ^ {2}
$$

with probability at least $1 - p$ . Of course $\textstyle \sum _ { i = 1 } ^ { n } A _ { i } = A$

From this bound we can derive a bound for the distance between the eigenvectors of the two matrices. Indeed, using Lemmas 5 and 8 in [14], we can derive

$$
1 - \langle v ^ {i _ {0}}, x ^ {*} \rangle \leq \frac {\sqrt {1 2 8 n \log \left(\frac {d}{p}\right)} h}{\delta}
$$

and

$$
\langle v ^ {i _ {0}}, x ^ {*} \rangle \geq 1 - \frac {\sqrt {1 2 8 n \log \left(\frac {d}{p}\right)} h}{\delta}
$$

with probability at least $1 - p$ (note that the leading eigenvector of $A _ { i _ { 0 } }$ is equal to the leading eigenvector $n A _ { i _ { 0 } }$ ). This is because $\left. v ^ { i _ { 0 } } , x ^ { * } \right. \leq 1$ , which implies that $\langle v ^ { i _ { 0 } } , x ^ { * } \rangle ^ { 2 } \overset { \cdot } { \leq } \langle v ^ { i _ { 0 } } , x ^ { * } \rangle$ .

We notice that the squared distance of $v ^ { i _ { 0 } }$ and $x ^ { * }$ is

$$
\| v ^ {i _ {0}} - x ^ {*} \| ^ {2} = \| v ^ {i _ {0}} \| ^ {2} + \| x ^ {*} \| ^ {2} - 2 \langle v ^ {i _ {0}}, x ^ {*} \rangle = 2 (1 - \langle v ^ {i _ {0}}, x ^ {*} \rangle) \leq 2 \frac {\sqrt {1 2 8 n \log \left(\frac {d}{p}\right)} h}{\delta}
$$

which is upper bounded by a constant by Assumption 4. The same holds for $\lVert \boldsymbol { v } ^ { i } - \boldsymbol { x } ^ { * } \rVert$ , thus, by triangle inequality we have an upper bound on $\| v ^ { \bar { i } _ { 0 } } - v ^ { i } \|$ to be at most double of the upper bound for $\lVert \bar { \boldsymbol { v } } ^ { i _ { 0 } } - \bar { \boldsymbol { x } ^ { * } } \rVert$ , thus it is still upper bounded by a constant. Since $\langle v ^ { i _ { 0 } } , x ^ { * } \rangle$ is lower bounded by a constant, again by Assumption 4, we have that the ratio of the input to the output variance in the quantization of $\dot { v } ^ { i _ { 0 } }$ is upper bounded by a constant. Thus, the total communication cost of this quantization is $\mathcal { O } ( n d )$ .

By the definition of the quantization scheme, we get

$$
\left\| \tilde {x} ^ {(0)} - v ^ {i _ {0}} \right\| \leq \frac {\left\langle v ^ {i _ {0}} , x ^ {*} \right\rangle}{2 (\sqrt {2} + 2)} =: \zeta .
$$

For the projected vector $x ^ { ( 0 ) }$ , we have

$$
\left\| x ^ {(0)} - v ^ {i _ {0}} \right\| \leq \left\| \tilde {x} ^ {(0)} - v ^ {i _ {0}} \right\| + \left\| \tilde {x} ^ {(0)} - x ^ {(0)} \right\| \leq 2 \left\| \tilde {x} ^ {(0)} - v ^ {i _ {0}} \right\| \leq 2 \zeta
$$

because $x ^ { ( 0 ) }$ is the closest point to $\tilde { x } ^ { ( 0 ) }$ belonging to the sphere and $v ^ { i _ { 0 } }$ belongs also to the sphere.

By the triangle inequality, we have

$$
\left\| x ^ {(0)} - x ^ {*} \right\| \leq \left\| v ^ {i _ {0}} - x ^ {*} \right\| + \left\| x ^ {(0)} - v ^ {i _ {0}} \right\|
$$

which is equivalent to

$$
\sqrt {2 \left(1 - \langle x ^ {(0)} , x ^ {*} \rangle\right)} \leq \sqrt {2 \left(1 - \langle v ^ {i _ {0}} , x ^ {*} \rangle\right)} + 2 \zeta .
$$

Thus

$$
\begin{array}{l} \langle x ^ {(0)}, x ^ {*} \rangle \geq \langle v ^ {i _ {0}}, x ^ {*} \rangle - \sqrt {2 (1 - \langle v ^ {i _ {0}} , x ^ {*} \rangle)} \zeta - 2 \zeta^ {2} \geq \langle v ^ {i _ {0}}, x ^ {*} \rangle - \sqrt {2} \zeta - 2 \zeta \\ = \langle v ^ {i _ {0}}, x ^ {*} \rangle - (\sqrt {2} + 2) \zeta = \langle v ^ {i _ {0}}, x ^ {*} \rangle - (\sqrt {2} + 2) \frac {\langle v ^ {i _ {0}} , x ^ {*} \rangle}{2 (\sqrt {2} + 2)} = \frac {\langle v ^ {i _ {0}} , x ^ {*} \rangle}{2} \\ \end{array}
$$

with probability at least $1 - p$ .

Since $\langle v ^ { i _ { 0 } } , x ^ { * } \rangle$ is lower bounded by a constant, $\langle x ^ { ( 0 ) } , x ^ { * } \rangle$ is also lower bounded by a constant and we get the desired result.