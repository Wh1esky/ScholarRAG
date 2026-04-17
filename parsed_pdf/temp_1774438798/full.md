# A Geometric Structure of Acceleration and Its Role in Making Gradients Small Fast

Jongmin Lee

Seoul National University dlwhd2000@snu.ac.kr

Chanwoo Park

Seoul National University chanwoo.park@snu.ac.kr

Ernest K. Ryu

Seoul National University ernestryu@snu.ac.kr

# Abstract

Since Nesterov’s seminal 1983 work, many accelerated first-order optimization methods have been proposed, but their analyses lacks a common unifying structure. In this work, we identify a geometric structure satisfied by a wide range of firstorder accelerated methods. Using this geometric insight, we present several novel generalizations of accelerated methods. Most interesting among them is a method that reduces the squared gradient norm with $\mathcal { O } ( 1 / K ^ { 4 } )$ rate in the prox-grad setup, faster than the $\mathcal { O } ( \bar { 1 } / K ^ { 3 } )$ rates of Nesterov’s FGM or Kim and Fessler’s FPGM-m.

# 1 Introduction

Since Nesterov’s seminal 1983 work [57], accelerated methods in optimization have been widely used in large-scale optimization and machine learning. However, the many accelerated methods have been developed and analyzed with disparate techniques, without a unified framework.

In this work, we identify geometric structures, which we call the parallel and collinear structures, satisfied by a wide range of gradient- and prox-based accelerated methods including: Nesterov’s FGM [57], OGM [47], OGM-G [47], Nesterov’s FGM in the strongly convex setup (SC-FGM) [60, (2.2.22)], SC-OGM [63], TMM [78], non-stationary SC-FGM [19, $\ S 4 . 5 ]$ , ITEM [73], geometric descent [13], Güler’s first and second accelerated proximal methods [37], and FISTA [11] Using the insight provided by these geometric structures, we present several novel generalizations of accelerated methods.

Among these novel generalizations, the most interesting is FISTA-G, which reduces the squared gradient norm with $\bar { \mathcal { O } } ( 1 / K ^ { 4 } )$ rate when combined with FISTA in the prox-grad (composite minimization) setup. The rate is optimal as it matches the $\Omega ( 1 / K ^ { 4 } )$ complexity lower bound [55, 56] and is faster than the $\mathcal { O } ( 1 / K ^ { 3 } )$ rates achieved by of Nesterov’s AGM [46, 69] in the smooth convex setup or Kim and Fessler’s FPGM-m [45] in the prox-grad setup.

# 1.1 Preliminaries and notations

For $L > 0$ , $f \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } }$ is $L$ -smooth if $f$ is differentiable and

$$
\| \nabla f (x) - \nabla f (y) \| \leq L \| x - y \| \quad \forall x, y \in \mathbb {R} ^ {n}.
$$

For $\mu > 0$ , $g \colon \mathbb { R } ^ { n }  \mathbb { R } \cup \{ \infty \}$ is $\mu$ -strongly convex if $g ( x ) - ( \mu / 2 ) \| x \| ^ { 2 }$ is convex. Let $L$ and $\mu$ respectively denote the smoothness and strong convexity parameters of $f$ . Write the gradient steps with stepsizes $1 / L$ and $1 / \mu$ as

$$
x ^ {+} = x - \frac {1}{L} \nabla f (x), \quad x ^ {+ +} = x - \frac {1}{\mu} \nabla f (x).
$$

For $\lambda > 0$ , the proximal operator [53, 62] $\operatorname { P r o x } _ { \lambda g }$ $\operatorname { P r o x } _ { \lambda g } \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ is defined as

$$
x ^ {\circ} = \operatorname {P r o x} _ {\lambda g} (x) = \underset {y \in \mathbb {R} ^ {n}} {\arg \min } \left\{g (y) + \frac {1}{2 \lambda} \| y - x \| ^ {2} \right\}.
$$

When $g$ is closed, convex, and proper [65], the proximal operator is well-defined i.e., the arg min exists and is unique [52]. Define the prox-grad step as

$$
x ^ {\oplus} = \underset {y \in \mathbb {R} ^ {n}} {\arg \min } \left\{f (x) + \langle \nabla f (x), y - x \rangle + g (y) + \frac {L}{2} \| y - x \| ^ {2} \right\}.
$$

If $g = 0$ , then $x ^ { \oplus } = x ^ { + }$ . If $f = 0$ and $1 / \lambda = L$ , then $x ^ { \oplus } = x ^ { \circ }$ . Furthermore, write

$$
\tilde {\nabla} _ {L} F (x) := - L (x ^ {\oplus} - x), \qquad \tilde {\nabla} _ {1 / \lambda} g (x) := - \frac {1}{\lambda} (x ^ {\circ} - x).
$$

$u \in \mathbb { R } ^ { n }$ is a subgradient of convex function $g$ at $x$ if

$$
g (x) + \langle u, y - x \rangle \leq g (y) \quad \forall y \in \mathbb {R} ^ {n}.
$$

Subdifferential of $g$ at $x$ , denoted $\partial g ( x )$ , is the set of subgradients of $g$ at $x$ . Throughout this paper, we consider the problem

$$
\underset {x \in \mathbb {R} ^ {n}} {\text {m i n i m i z e}} F (x) := f (x) + g (x), \tag {P}
$$

where $f \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } }$ is convex and $L$ -smooth and $g \colon \mathbb { R } ^ { n }  \mathbb { R } \cup \{ \infty \}$ is closed, convex, and proper. We say we are in the smooth convex setup if $f \neq 0$ and $g = 0$ , the proximal-point setup if $f = 0$ and $g \neq 0$ , and the prox-grad setup if $f \neq 0$ and $g \neq 0$ . Write $f _ { \star }$ , $g _ { \star }$ , and $F _ { \star }$ to respectively denote the infima of $f , g$ , and $F$ . We also informally assume $\operatorname { P r o x } _ { \lambda g }$ is efficient to evaluate [17].

# 1.2 Prior works

In convex optimization and machine learning, the classical goal of algorithms is to reduce the function value efficiently. In the smooth convex setup, Nesterov’s celebrated fast gradient method (FGM) [57] achieves an accelerated rate, and the optimized gradient method (OGM) [44] improves this rate by a factor of 2, which is in fact exactly optimal [27]. In the smooth strongly convex setup, the strongly convex fast gradient method (SC-FGM) [60, (2.2.22)], strongly convex optimized gradient method (SC-OGM) [63], and non-stationary SC-FGM [19, $\ S 4 . 5 ]$ achieves an accelerated rate, and the triple momentum method (TMM) [78] and information theoretic exact method (ITEM) [73] achieve exact optimal rates [28]. In the proximal-point setup, there are Güler’s first and second accelerated proximal methods [37]. There are also other variants of accelerated proximal methods [51, 49, 39, 77].

The study of algorithms for reducing the gradient magnitude, which can help us understand nonconvex optimization better and design faster non-convex machine learning algorithms, was initiated by Nesterov [58]. For smooth nonconvex minimization, gradient descent (GD) achieves $\mathcal { O } ( ( f ( x _ { 0 } ) - f _ { \star } ) / K )$ rate [54, Proposition 3.3.1]. In the smooth convex setup, combining $K$ iterations of FGM with $K$ iterations of GD achieves $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 3 } )$ rate [58]. Adding a strongly convex regularization and then using SC-FGM achieves $\tilde { \mathcal { O } } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 4 } )$ rate, where $\tilde { \mathcal { O } }$ ignores logarithmic factors [58]. Finally, OGM-G [47] achieved $\mathcal { O } ( ( f ( x _ { 0 } ) - f _ { \star } ) / K ^ { 2 } )$ rate and $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { \zeta } / K ^ { 4 } )$ rate is achieved by combining FGM with OGM-G [61, Remark 2.1]. This rate is optimal as it matches the $\Omega ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 4 } )$ lower bound [55, 56]. Making gradients small have also been studied in the setup with stochastic gradients [2, 3, 33, 79, 34], for composite problems with strong convexity and non-Euclidean norms [23], and for convex-concave saddle-point problems [22, 24, 80].

In the prox-grad setup, iterative shrinkage-thresholding algorithm (ISTA) [12, 64, 20, 18] achieves $\mathcal { O } ( \| x _ { 0 } ^ { \cdot } - x _ { \star } \| ^ { 2 } / K )$ rate on function-value suboptimality and fast iterative shrinkage-thresholding algorithm (FISTA) [11] accelerates this rate to $\mathcal { O } ( \bar { \| } x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 2 } )$ . On the squared gradient magnitude, FPGM-m achieves $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 3 } )$ rate [45].

The performance estimation problem (PEP) is a computer-assisted proof methodology based on semidefinite programming [29, 75, 74]. Extensions and variations of the PEP have been utilized to obtain analyses and algorithms that would be difficult to obtain without the assistance of computers [67, 72, 10, 76, 21, 36, 50, 26, 43]. In particular, OGM [29, 44], OGM-G [47], and ITEM [73] were obtained with the PEP. Integral quadratic constraints is a related computer-assisted approach based on control-theoretic ideas [48, 38, 78, 31].

A Lyapunov analysis constructs a nonincreasing quantity, and many modern analyses of accelerated first-order methods are based on this technique [11, 70, 9, 72, 1, 6, 7, 8, 63]. In fact, Nesterov’s original presentation of FGM was a Lyapunov analysis [57].

Finally, we mention some closely related prior work. The scaled relative graph (SRG) analyzes optimization algorithms via Euclidean geometry [66, 40, 41, 68], but the SRG has not been used to analyze accelerated methods. Linear coupling [4, 5] interprets acceleration as a coupling between gradient descent and mirror descent to efficiently reduce function values. Geometric descent [13, 16, 42] is an accelerated algorithm designed expressly based on geometric principles, and quadratic averaging [30] is an equivalent algorithm with an alternate interpretation. The method of similar triangles (MST) generalizes FGM with geometric notions [32, 60, 1]. The parallel and collinear structures of this work, in our view, expand these prior notions and concretely articulate ideas that had been utilized implicitly. In particular, the prior interpretations, as is, are insufficient for analyzing OGM-G and deriving our newly proposed method FISTA-G. In Section H of the appendix, we further discuss the relationship of our contributions with these prior works.

# 1.3 Contribution

This paper presents two major contributions, one conceptual and one concrete. The first is the identification of the parallel and collinear geometric structures, which are observed in a wide variety of accelerated first-order methods. This geometric structure provides us with valuable insight into the mechanism of acceleration and enables us to obtain results that would be otherwise difficult to discover, including FISTA-G. The second major contribution is the novel method FISTA-G and FISTA $^ +$ FISTA-G. To the best of our knowledge, FISTA $+$ FISTA-G is the first method to achieve the rate $\mathcal { O } ( 1 / K ^ { 4 } )$ on the squared gradient norm in the prox-grad setup.

In addition, we use the geometric structures to find G-FISTA-G, G-FGM-G, G-Güler-G, Proximal-TMM, and Proximal-ITEM, novel variants of accelerated first-order methods, and present them as minor contributions. (G-FGM-G is the abbreviation for Generalized-FGM-(for reducing Gradients).)

# 2 Making gradients small at rate $\mathcal { O } ( 1 / K ^ { 4 } )$ in the prox-grad setup

In this section, we present a novel method FISTA-G, which reduces the squared gradient norm with $\mathcal { O } ( 1 / K ^ { 4 } )$ rate when combined with FISTA in the prox-grad setup. Our analysis uses a novel and unusual Lyapunov function, whose discovery crucially relied on the geometric insights presented later in Section 3. Here, we provide a self-contained analysis that uses, but does not explain this Lyapunov function.

Smooth convex setup. Kim and Fessler’s (OGM-G) [47] is

$$
x _ {k + 1} = x _ {k} ^ {+} + \frac {\left(\theta_ {k} - 1\right) \left(2 \theta_ {k + 1} - 1\right)}{\theta_ {k} \left(2 \theta_ {k} - 1\right)} \left(x _ {k} ^ {+} - x _ {k - 1} ^ {+}\right) + \frac {2 \theta_ {k + 1} - 1}{2 \theta_ {k} - 1} \left(x _ {k} ^ {+} - x _ {k}\right) \quad \text {f o r} k = 0, 1, \dots , K - 1,
$$

whereNote, $x _ { - 1 } ^ { + } : = x _ { 0 }$ $\theta _ { K } = 1$ $\begin{array} { r } { \theta _ { k } = \frac { 1 + \sqrt { 1 + 4 \theta _ { k + 1 } ^ { 2 } } } { 2 } } \end{array}$ $k = 1 , 2 , \ldots , K - 1$ , and ction $\begin{array} { r } { \theta _ { 0 } = \frac { 1 + \sqrt { 1 + 8 \theta _ { 1 } ^ { 2 } } } { 2 } } \end{array}$ . $K$ $\theta _ { 0 } = \mathcal { O } ( K )$ $\theta _ { 0 }$ $K$ the first method to achieve the accelerated rate $\mathcal { O } ( ( f ( x _ { 0 } ) - f _ { \star } ) / K ^ { 2 } )$ on the squared gradient norm, and the method was originally obtained through a computer-assisted methodology. However, the computer-generated analysis is verifiable but arguably difficult to understand.

We characterize the convergence of OGM-G with the following novel Lyapunov function

$$
\begin{array}{l} U _ {k} = \frac {1}{\theta_ {k} ^ {2}} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2} + f (x _ {k}) - f (x _ {K}) - \left\langle \nabla f (x _ {k}), x _ {k} - x _ {k - 1} ^ {+} \right\rangle\right) \\ + \frac {L}{\theta_ {k} ^ {4}} \left\langle z _ {k} - x _ {k - 1} ^ {+}, z _ {k} - x _ {K} ^ {+} \right\rangle , \qquad \text {f o r} k = 1, 2, \dots , K, \\ \end{array}
$$

where $\begin{array} { r } { z _ { k } = x _ { k } + \frac { ( \theta _ { k } - 1 ) ^ { 2 } } { 2 \theta _ { k } - 1 } ( x _ { k } - x _ { k - 1 } ^ { + } ) } \end{array}$ for $k = 1 , 2 , \ldots , K$ , (note $z _ { K } = x _ { K }$ ) through the steps

$$
\frac {1}{L} \| \nabla f (x _ {K}) \| ^ {2} = U _ {K} \leq U _ {K - 1} \leq \dots \leq U _ {1} \leq \frac {2}{\theta_ {0} ^ {2}} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + f (x _ {0}) - f (x _ {K})\right).
$$

Section 3 explains the geometric insights behind the discovery of this Lyapunov function and the proof that $\{ \hat { U } _ { k } \} _ { k = 1 } ^ { K }$ is nonincreasing. As related work, Diakonikolas and Wang [24] presented a human-understandable analysis of OGM-G, but their analysis is not a Lyapunov analysis since (as they acknowledge) they do not establish a nonincreasing quantity. In any case, our main contribution of this section is the following generalization of the method and analysis to the prox-grad setup.

# 2.1 Accelerated $\mathcal { O } ( ( F ( x _ { 0 } ) - F _ { \star } ) / K ^ { 2 } )$ rate with FISTA-G

We now present the novel method (FISTA-G):

$$
x _ {k + 1} = x _ {k} ^ {\oplus} + \frac {\varphi_ {k + 1} - \varphi_ {k + 2}}{\varphi_ {k} - \varphi_ {k + 1}} (x _ {k} ^ {\oplus} - x _ {k - 1} ^ {\oplus}) \qquad \mathrm {f o r} k = 0, 1, \ldots , K - 1,
$$

where $x _ { - 1 } ^ { \oplus } : = x _ { 0 }$ , $\varphi _ { K + 1 } = 0$ , $\varphi _ { K } = 1$ , and

$$
\varphi_ {k} = \frac {\varphi_ {k + 2} ^ {2} - \varphi_ {k + 1} \varphi_ {k + 2} + 2 \varphi_ {k + 1} ^ {2} + (\varphi_ {k + 1} - \varphi_ {k + 2}) \sqrt {\varphi_ {k + 2} ^ {2} + 3 \varphi_ {k + 1} ^ {2}}}{\varphi_ {k + 1} + \varphi_ {k + 2}} \quad \mathrm {f o r} k = - 1, 0, \ldots , K - 1.
$$

Note that $K$ is the total number of iterations.

Theorem 1. Consider (P). FISTA-G’s final iterate $x _ { K }$ exhibits the rate

$$
\min  \left\| \partial F (x _ {K} ^ {\oplus}) \right\| ^ {2} \leq 4 \left\| \tilde {\nabla} _ {L} F (x _ {K}) \right\| ^ {2} \leq \frac {2 6 4 L}{(K + 2) ^ {2}} \left(F (x _ {0}) - F _ {\star}\right).
$$

We clarify that min $\| \partial F ( x ) \| ^ { 2 } = \operatorname* { m i n } \{ \| u \| ^ { 2 } | u \in \partial F ( x ) \}$ for $x \in \mathbb { R } ^ { n }$

Proof outline. Define $z _ { 0 } = x _ { 0 }$ , $\begin{array} { r } { z _ { k } = \frac { \varphi _ { k } } { \varphi _ { k } - \varphi _ { k + 1 } } x _ { k } - \frac { \varphi _ { k + 1 } } { \varphi _ { k } - \varphi _ { k + 1 } } x _ { k - 1 } ^ { \oplus } } \end{array}$ xk for $k = 0 , 1 , \ldots , K$ , and

$$
\begin{array}{l} U _ {k} = \frac {2 \varphi_ {k - 1}}{(\varphi_ {k - 1} - \varphi_ {k}) ^ {2}} \left(\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {k}) \right\| ^ {2} + F (x _ {k} ^ {\oplus}) - F (x _ {K} ^ {\oplus}) - \left\langle \tilde {\nabla} _ {L} F (x _ {k}), x _ {k} - x _ {k - 1} ^ {\oplus} \right\rangle\right) \\ + \frac {L}{\varphi_ {k}} \left\langle z _ {k} - x _ {k - 1} ^ {\oplus}, z _ {k} - x _ {K} ^ {\oplus} \right\rangle \quad \text {f o r} k = 0, 1, \dots , K. \\ \end{array}
$$

(Note that $z _ { K } = x _ { K }$ .) We can show that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. Then

$$
\begin{array}{l} \frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {K}) \right\| ^ {2} = U _ {K} \leq \dots \leq U _ {0} = \frac {2 \varphi_ {- 1}}{(\varphi_ {- 1} - \varphi_ {0}) ^ {2}} \left(\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {0}) \right\| ^ {2} + F (x _ {0} ^ {\oplus}) - F (x _ {K} ^ {\oplus})\right) \\ \leq \frac {2 \varphi_ {- 1}}{(\varphi_ {- 1} - \varphi_ {0}) ^ {2}} \left(F (x _ {0}) - F (x _ {K} ^ {\oplus})\right) \leq \frac {3 3 L}{(K + 2) ^ {2}} \left(F (x _ {0}) - F _ {\star}\right), \\ \end{array}
$$

where we used $\begin{array} { r } { \frac { 1 } { 2 L } \left\| \tilde { \nabla } _ { L } F ( x _ { 0 } ) \right\| ^ { 2 } \leq F ( x _ { 0 } ) - F ( x _ { 0 } ^ { \oplus } ) } \end{array}$ , stated as Lemma 9 in the appendix.

![](images/30f757565bfce6055b62086aa0483dfa454b23a01d3323cc7dc81437bab8f0dd.jpg)

The complete proof is presented in Section E of the appendix. In fact, FISTA-G is the best instance within the family of methods G-FISTA-G presented in Section E of the appendix, in the sense that FISTA-G is the instance for which we could obtain the smallest constant in the bound.

# 2.2 Accelerated $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 4 } )$ rate with FISTA+FISTA-G

Define the method (FISTA+FISTA-G) as: from a starting point $x _ { 0 }$ run $K$ iterations of FISTA [11] and then from the output of FISTA start FISTA-G and run $K$ iterations. ( $2 K$ iterations total.) We denote the final iterate of this method as $x _ { 2 K }$ .

Corollary 1. Consider (P). Assume $F$ has a minimizer $x _ { \star }$ . FISTA+FISTA-G’s final iterate $x _ { 2 K }$ exhibits the rate1

$$
\min  \left\| \partial F (x _ {2 K} ^ {\oplus}) \right\| ^ {2} \leq 4 \left\| \tilde {\nabla} _ {L} F (x _ {2 K}) \right\| ^ {2} \leq \frac {5 2 8 L ^ {2}}{(K + 2) ^ {4}} \| x _ {0} - x _ {\star} \| ^ {2}.
$$

Proof. Combining the $F ( x _ { K } ^ { \oplus } ) - F _ { \star } \leq 2 L \| x _ { 0 } - x _ { \star } \| ^ { 2 } / ( K + 2 ) ^ { 2 }$ rate of FISTA [11, Theorem 4.4] with Theorem 1, we get the rate of FISTA+FISTA-G:

$$
\min \left\| \partial F (x _ {2 K} ^ {\oplus}) \right\| ^ {2} \leq 4 \left\| \tilde {\nabla} _ {L} F (x _ {2 K}) \right\| ^ {2} \leq \frac {2 6 4 L}{(K + 2) ^ {2}} (F (x _ {K} ^ {\oplus}) - F _ {\star}) \leq \frac {5 2 8 L ^ {2}}{(K + 2) ^ {4}} \| x _ {0} - x _ {\star} \| ^ {2}.
$$

![](images/c0ccbd4b702c0488ad6d08b44217a5f2724d2530de2f52e79931225bf463ac2d.jpg)

FISTA+FISTA-G was inspired by the FGM+OGM-G method, which has $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 4 } )$ rate on the squared gradient norm in the smooth convex setup [61, Remark 2.1]. Nesterov FGM by itself only achieves $\bar { \mathcal { O } } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 3 } )$ rate [75, 71, 46, 69, 24]. In the prox-grad setup, FPGM-m [45] held the prior state-of-the-art rate of $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 3 } )$ . Our $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 4 } )$ rate matches the known complexity lower bound [55, 56] and therefore is optimal in the case $g = 0$ .

# 3 Parallel structure for the convex setup

In this section, we present the parallel structure, a geometric structure observed in a wide range of accelerated first-order methods. We then utilize the parallel structure to obtain novel variants of accelerated first-order methods and, in particular, obtain the Lyapunov analysis presented in Section 2.

# 3.1 Parallel structure of acceleration

Nesterov’s (FGM) [57] has the form:

$$
\begin{array}{l} x _ {k + 1} = x _ {k} ^ {+} + \frac {\theta_ {k} - 1}{\theta_ {k + 1}} \left(x _ {k} ^ {+} - x _ {k - 1} ^ {+}\right) \\ z _ {k} = x _ {k} + \left(\theta_ {k} - 1\right) \left(x _ {k} - x _ {k - 1} ^ {+}\right) \quad \text {f o r} k = 0, 1, \dots , \\ \end{array}
$$

where x+−1 := x0, x0 = z0, θ0 = 1, and θk+1 = 1+ 1+4θ2k2 $x _ { - 1 } ^ { + } : = x _ { 0 }$ $x _ { 0 } = z _ { 0 }$ $\theta _ { 0 } = 1$ $\begin{array} { r } { \theta _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 \theta _ { k } ^ { 2 } } } { 2 } } \end{array}$ for $k = 0 , 1 , \ldots$ The $z _ { k }$ -iterates are known as the auxiliary sequence, and they play a key role in the Lyapunov analysis of FGM [57]. Figure 1 (left) depicts $\dot { x } _ { k - 1 } ^ { + }$ , xk , $x _ { k } ^ { + }$ , $x _ { k + 1 }$ , zk, and $z _ { k + 1 }$ . These points in $\mathbb { R } ^ { n }$ lie on a 2D-plane, which we call the plane of iteration. Observe that the line segments representing $x _ { k } ^ { + } - x _ { k }$ and $z _ { k + 1 } - z _ { k }$ are parallel. (We prove observations 1 and 2 in Section B of the appendix.)

Observation 1. In FGM, $x _ { k } ^ { + } - x _ { k }$ and $z _ { k + 1 } - z _ { k }$ are parallel2.

![](images/f2400a021c81ef84dd1d2bb2334998ca52524f02e26d00fb54ca6e58e909de10.jpg)

![](images/91b0f4e52ec62ebd9b33df097007ccc073a970587f8594d200be6fb69c38d074.jpg)

![](images/94a9dce0e0a5c6d49b55235bec1c2595fcbec48c6890d5d7d73a3c97517945bd.jpg)  
Figure 1: Plane of iteration of FGM (left), OGM (middle), and FISTA-G (right)

Drori, Teboulle, Kim, and Fessler’s (OGM) [29, 44] has the form:

$$
\begin{array}{l} x _ {k + 1} = x _ {k} ^ {+} + \frac {\theta_ {k} - 1}{\theta_ {k + 1}} \left(x _ {k} ^ {+} - x _ {k - 1} ^ {+}\right) + \frac {\theta_ {k}}{\theta_ {k + 1}} \left(x _ {k} ^ {+} - x _ {k}\right) \\ z _ {k} = x _ {k} + (\theta_ {k} - 1) (x _ {k} - x _ {k - 1} ^ {+}) \qquad \mathrm {f o r} k = 0, 1, \ldots K - 1, \\ \end{array}
$$

where $x _ { - 1 } ^ { + } : = x _ { 0 }$ , $z _ { 0 } ~ = ~ x _ { 0 }$ , $\theta _ { 0 } = 1$ , $\begin{array} { r } { \theta _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 \theta _ { k } ^ { 2 } } } { 2 } } \end{array}$ for $k = 0 , 1 , \ldots , K - 1$ , and $\theta _ { K } =$ 1+ 1+8θ2K−12 . OGM improves upon the rate of FGM [44]. Interestingly, OGM also exhibits a similar $\frac { 1 + \sqrt { 1 + 8 \theta _ { K - 1 } ^ { 2 } } } { 2 }$ geometric structure as depicted in Figure 1 (middle), and the auxiliary sequence, the $z _ { k }$ -iterates, plays a similar role in the Lyapunov analysis [63].

Observation 2. In OGM, $x _ { k } ^ { + } - x _ { k }$ and $z _ { k + 1 } - z _ { k }$ are parallel.

We refer to this geometric structure as the parallel structure. Given an algorithm expressed with momentum and correction terms, one can define the $z _ { k }$ -iterates to exhibit the parallel structure:

$$
x _ {k + 1} = x _ {k} ^ {+} + a _ {k + 1} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) + b _ {k + 1} (x _ {k} ^ {+} - x _ {k}) \quad \Longrightarrow \begin{array}{c} x _ {k} = c _ {k} x _ {k - 1} ^ {+} + (1 - c _ {k}) z _ {k} \\ z _ {k + 1} = z _ {k} - d _ {k} \nabla f (x _ {k}) \end{array}
$$

where $a _ { k + 1 }$ , $b _ { k + 1 }$ , $c _ { k }$ , and $d _ { k }$ satisfy an appropriate relationship as shown in Section B of the appendix. The $z _ { k }$ -iterates of FGM and OGM have this form. The parallel structure also holds for Güler’s accelerated methods [37] for the proximal-point setup and FISTA [11] for the prox-grad setup with analogous definitions of the $z _ { k }$ -iterates. The table of Section A.1 of the appendix presents the precise forms.

# 3.2 Parallel structure for OGM-G and FISTA-G

We equivalently write (OGM-G) as

$$
x _ {k} = \frac {\theta_ {k + 1} ^ {4}}{\theta_ {k} ^ {4}} x _ {k - 1} ^ {+} + \left(1 - \frac {\theta_ {k + 1} ^ {4}}{\theta_ {k} ^ {4}}\right) z _ {k}
$$

$$
z _ {k + 1} = z _ {k} - \frac {\theta_ {k}}{L} \nabla f (x _ {k}) \qquad \text {f o r} k = 1, 2, \dots , K,
$$

where $z _ { 0 } = x _ { 0 }$ , $\begin{array} { r } { z _ { 1 } = z _ { 0 } - \frac { \theta _ { 0 } + 1 } { 2 L } \nabla f ( x _ { 0 } ) , \theta _ { K - } } \end{array}$ ${ \theta } _ { K + 1 } = 0$ , and $\begin{array} { r } { \theta _ { k } = \frac { 1 + \sqrt { 1 + 4 \theta _ { k + 1 } ^ { 2 } } } { 2 } } \end{array}$ 1+ 1+4θ2k+1 for k = 1, 2, . . . , K , 2 $k = 1 , 2 , \dots , K$ and θ = 1+ 1+8θ21 . $\begin{array} { r } { \theta _ { 0 } = \frac { 1 + \sqrt { 1 + 8 \theta _ { 1 } ^ { 2 } } } { 2 } } \end{array}$ The key point is that the auxiliary $z _ { k }$ -sequence is defined to exhibit the parallel structure. As a comparison, [47, Equation (43)] provides a different auxiliary sequence for OGM-G that does not exhibit the parallel structure, but this sequence does not lead to a Lyapunov analysis.

Here, we show how the Lyapunov function $U _ { k }$ is obtained from the parallel structure of OGM-G. For $k \geq 1$ , combine the cocoercivity inequalities between $( x _ { k + 1 } , x _ { k } )$ and $( x _ { k } , x _ { K } )$ to get

$$
\begin{array}{l} 0 \geq \frac {1}{\theta_ {k + 1} ^ {2}} \left(f (x _ {k + 1}) - f (x _ {k}) - \langle \nabla f (x _ {k + 1}), x _ {k + 1} - x _ {k} \rangle + \frac {1}{2 L} \| \nabla f (x _ {k + 1}) - \nabla f (x _ {k}) \| ^ {2}\right) \\ \left. + \left(\frac {1}{\theta_ {k + 1} ^ {2}} - \frac {1}{\theta_ {k} ^ {2}}\right) \left(f (x _ {k}) - f (x _ {K}) - \langle \nabla f (x _ {k}), x _ {k} - x _ {K} \rangle + \frac {1}{2 L} \| \nabla f (x _ {k}) - \nabla f (x _ {K}) \| ^ {2}\right) \right. \\ = \frac {1}{\theta_ {k + 1} ^ {2}} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {k + 1}) \| ^ {2} + f (x _ {k + 1}) - f (x _ {K}) - \left\langle \nabla f (x _ {k + 1}), x _ {k + 1} - x _ {k} ^ {+} \right\rangle\right) \\ \left. - \frac {1}{\theta_ {k} ^ {2}} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2} + f (x _ {k}) - f (x _ {K}) - \left\langle \nabla f (x _ {k}), x _ {k} - x _ {k - 1} ^ {+} \right\rangle\right) \right. \\ \underbrace {- \left\langle \nabla f (x _ {k}) , \theta_ {k + 1} ^ {- 2} x _ {k} ^ {+} - \theta_ {k} ^ {- 2} x _ {k - 1} ^ {+} - \left(\theta_ {k + 1} ^ {- 2} - \theta_ {k} ^ {- 2}\right) x _ {K} ^ {+} \right\rangle .} _ {:= T} \\ \end{array}
$$

With the following geometric arguments, we analyze the term $T$ and conclude $0 \geq U _ { k + 1 } - U _ { k }$ Let $t \in \mathbb { R } ^ { n }$ be the (orthogonal) projection of $x _ { K } ^ { + }$ onto the plane of iteration. For $u , v \in \mathbb { R } ^ { n }$ , define $\vec { u \mathcal { b } } = v - u$ . Then

$$
\frac {1}{L} T \stackrel {\mathrm {(i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {+}}, \left(\theta_ {k + 1} ^ {- 2} - \theta_ {k} ^ {- 2}\right) \overrightarrow {t x _ {k} ^ {+}} + \theta_ {k} ^ {- 2} \overrightarrow {x _ {k - 1} ^ {+} x _ {k} ^ {+}} \right\rangle
$$

$$
\begin{array}{l} \stackrel {\mathrm {(i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\uparrow}}, \left(\theta_ {k + 1} ^ {- 2} - \theta_ {k} ^ {- 2}\right) \left(\overrightarrow {t z _ {k + 1}} - \overrightarrow {z _ {k} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {\uparrow}}\right) \right. \\ + \theta_ {k} ^ {- 2} (\overrightarrow {x _ {k - 1} ^ {+} x _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {+}}) \Bigg > \\ \end{array}
$$

$$
\begin{array}{l} \stackrel {\mathrm {(i i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {+}}, \left(\theta_ {k + 1} ^ {- 2} - \theta_ {k} ^ {- 2}\right) \overrightarrow {t z _ {k + 1}} - \left(\theta_ {k + 1} ^ {- 2} - \theta_ {k} ^ {- 2}\right) \left(\theta_ {k} - 1\right) \overrightarrow {x _ {k} x _ {k} ^ {+}} \right. \\ \left. - \left(\theta_ {k + 1} ^ {- 2} - \theta_ {k} ^ {- 2}\right) \overrightarrow {x _ {k} z _ {k}} + (2 \theta_ {k} - 1) \theta_ {k + 1} ^ {- 4} \overrightarrow {x _ {k} z _ {k}} + \theta_ {k} ^ {- 2} \overrightarrow {x _ {k} x _ {k} ^ {+}} \right\rangle \\ \end{array}
$$

$$
\begin{array}{l} \stackrel {\text {(i v)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {+}}, \left(\theta_ {k + 1} ^ {- 2} - \theta_ {k} ^ {- 2}\right) \overrightarrow {t z _ {k + 1}} + \theta_ {k + 1} ^ {- 2} \left(\theta_ {k} - 1\right) ^ {- 1} \overrightarrow {x _ {k} z _ {k}} \right\rangle \\ \stackrel {(\mathrm {v})} {=} \theta_ {k + 1} ^ {- 4} \left\langle \overrightarrow {x _ {k} ^ {+} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}}, \overrightarrow {t z _ {k + 1}} \right\rangle + \theta_ {k + 1} ^ {- 4} \left\langle \overrightarrow {t z _ {k + 1}} - \overrightarrow {t z _ {k}}, \overrightarrow {x _ {k} z _ {k}} \right\rangle \\ \stackrel {\mathrm {(v i)}} {=} \theta_ {k + 1} ^ {- 4} \left\langle z _ {k + 1} - x _ {k} ^ {+}, z _ {k + 1} - x _ {K} ^ {+} \right\rangle - \theta_ {k} ^ {- 4} \left\langle z _ {k} - x _ {k - 1} ^ {+}, z _ {k} - x _ {K} ^ {+} \right\rangle , \\ \end{array}
$$

where (i) follows from the definition of $t$ and the fact that we can replace $x _ { K } ^ { + }$ with $t$ , the projection of $x _ { K }$ onto the plane of iteration, without affecting the inner products, (ii) from vector addition,

![](images/f03429fc1dc9f732a35b97d8bc5ee799789d370e446158f22153814e6a3fbd81.jpg)

(iii) from the fact that $\overrightarrow { z _ { k } z _ { k + 1 } }$ and $\xrightarrow [ { x _ { k } x _ { k } ^ { + } } ] { }$ are parallel and their lengths satisfy ${ \overrightarrow { z _ { k } z _ { k + 1 } } } = \theta _ { k } { \overrightarrow { x _ { k } x _ { k } ^ { + } } }$ andthe $\overrightarrow { x _ { k - 1 } ^ { + } x _ { k } }$ $\overrightarrow { x _ { k } z _ { k } }$ eir l and $\begin{array} { r } { \overrightarrow { { x _ { k - 1 } ^ { + } } { x _ { k } } } = \frac { \theta _ { k } ^ { 2 } ( 2 \theta _ { k } - 1 ) } { \theta _ { k + 1 } ^ { 4 } } \overrightarrow { { x _ { k } } { z _ { k } } } } \end{array}$ θ2k(2θk−1)θ4 −−→xkzk, (iv) from , $( \theta _ { k + 1 } ^ { - 2 } - \theta _ { k } ^ { - 2 } ) ( \theta _ { k } - 1 ) = \theta _ { k } ^ { - 2 }$ $( 2 \theta _ { k } - 1 ) \theta _ { k + 1 } ^ { - 4 } - ( \theta _ { k + 1 } ^ { - 2 } - \theta _ { k } ^ { - 2 } ) = \theta _ { k + 1 } ^ { - 2 } ( \theta _ { k } - 1 ) ^ { - 1 }$ (v) from distributing the product and substituting $\overrightarrow { x _ { k } x _ { k } ^ { + } } = ( \theta _ { k } - 1 ) ^ { - 1 } ( \overrightarrow { x _ { k } ^ { + } z _ { k + 1 } } - \overrightarrow { x _ { k } z _ { k } } )$ (which follows from−−−→ $\overrightarrow { x _ { k } x _ { k } ^ { + } } = \overrightarrow { x _ { k } z _ { k } ^ { + } } + \overrightarrow { z _ { k } z _ { k + 1 } } - \overrightarrow { x _ { k } ^ { + } z _ { k + 1 } } = \overrightarrow { x _ { k } z _ { k } ^ { + } } + \theta _ { k } \overrightarrow { x _ { k } x _ { k } ^ { + } } - \overrightarrow { x _ { k } ^ { + } z _ { k + 1 } } )$ into the first term and x k x +k $\overrightarrow { x _ { k } x _ { k } ^ { + } } = \theta _ { k } ^ { - 1 } \overrightarrow { z _ { k } z _ { k + 1 } } = \theta _ { k } ^ { - 1 } \left( \overrightarrow { t z _ { k + 1 } } - \overrightarrow { t z _ { k } } \right)$ 1 = θ − 1k  tzk into the second term, and the identity $( \theta _ { k + 1 } ^ { - 2 } -$ $\theta _ { k } ^ { - 2 } ) ( \theta _ { k } - 1 ) ^ { - 1 } = \theta _ { k + 1 } ^ { - 2 } ( \theta _ { k } - 1 ) ^ { - 1 } \theta _ { k } ^ { - 1 } = \theta _ { k + 1 } ^ { - 4 }$ , and (vi) from cancelling out the cross terms, using $\theta _ { k } ^ { - 4 } \overrightarrow { x _ { k - 1 } ^ { + } z _ { k } } = \theta _ { k + 1 } ^ { - 4 } \overrightarrow { x _ { k } z _ { k } }$ , and by replacing aining few detail $t$ with of th $x _ { K } ^ { + }$ in the inner products. At this point, the proof isalysis of OGM-G are presented in Section C of the appendix.

This geometric reasoning naturally extends to the prox-grad setup. Using analogous Lyapunov functions and proof structure, we obtain convergence rates of FISTA-G and the other generalizations presented in Section 3.3. Figure 1 (right) illustrates the parallel structure of FISTA-G with the $z _ { k }$ -iterates defined as in Section A.2 of the appendix.

# 3.3 Other generalizations

Using the parallel structure, we find several novel variants of accelerated first-order methods: G-FISTA-G, G-FGM-G, and G-Güler-G. We discuss them in detail in Sections A.2, E, and F of the appendix. Here, we briefly highlight the two special cases that we found most interesting.

We present (FGM-G) for the smooth convex setup:

$$
x _ {k + 1} = x _ {k} ^ {+} + \frac {\varphi_ {k + 1} - \varphi_ {k + 2}}{\varphi_ {k} - \varphi_ {k + 1}} \left(x _ {k} ^ {+} - x _ {k - 1} ^ {+}\right) \quad \text {f o r} k = 0, 1, \dots , K - 1,
$$

where x+−1 = x0 and {ϕk}K+1k=0 i $x _ { - 1 } ^ { + } = x _ { 0 }$ $\{ \varphi _ { k } \} _ { k = 0 } ^ { K + 1 }$ s the same as the {ϕk}K+1k=0 $\{ \varphi _ { k } \} _ { k = 0 } ^ { K + 1 }$ of FISTA-G. We can view FGM-G as a special case of FISTA-G with $g = 0$ or as a special case of G-FGM-G. In any case, FGM-G’s final iterate $x _ { K }$ exhibits the rate

$$
\left\| \nabla f \left(x _ {K}\right) \right\| ^ {2} \leq \frac {6 6 L}{(K + 2) ^ {2}} \left(f \left(x _ {0}\right) - f _ {\star}\right).
$$

defined separately from OGM-G uses the “correction term” 2θk+1−12θk−1 ( . F $\frac { 2 \theta _ { k + 1 } - 1 } { 2 \theta _ { k } - 1 } \big ( x _ { k } ^ { + } - x _ { k } \big )$ and the “first-step modification”, i.e., tes that these features are not necessar $\theta _ { 0 }$ isor $\{ \theta _ { k } \} _ { k = 1 } ^ { K - 1 }$ achieving the accelerated $\mathcal { O } ( ( f ( x _ { 0 } ) - f _ { \star } ) / K ^ { 2 } )$ rate. However, the simplification does come at the cost of a worse constant.

We present (Güler-G) for the proximal-point setup:

$$
x _ {k} = \frac {\theta_ {k + 1} ^ {4}}{\theta_ {k} ^ {4}} x _ {k - 1} ^ {\circ} + \left(1 - \frac {\theta_ {k + 1} ^ {4}}{\theta_ {k} ^ {4}}\right) z _ {k}
$$

$$
z _ {k + 1} = z _ {k} - \theta_ {k} \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) \qquad \text {f o r} k = 0, 1, \ldots , K,
$$

where $z _ { 0 } = x _ { 0 }$ , ${ \theta } _ { K + 1 } = 0$ , and $\begin{array} { r } { \theta _ { k } = \frac { 1 + \sqrt { 1 + 4 \theta _ { k + 1 } ^ { 2 } } } { 2 } } \end{array}$ for $k = 0 , 1 , \ldots , K$ .

Theorem 2. Consider (P) with $f = 0$ . Güler-G’s final iterate $x _ { K }$ exhibits the rate

$$
\left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {K}) \right\| ^ {2} \leq \frac {4}{\lambda (K + 2) ^ {2}} (g (x _ {0}) - g _ {\star}).
$$

Furthermore assume $g$ has a minimizer $x _ { \star }$ and define the method Güler+Güler-G as: from a starting point $x _ { 0 }$ run $K$ iterations of Güler second method $I 3 7 J$ and then from the output start Güler-G and run $K$ iterations. Then the final iterate $x _ { 2 K }$ exhibits the rate

$$
\left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {2 K}) \right\| ^ {2} \leq \frac {4}{\lambda^ {2} (K + 2) ^ {4}} \| x _ {0} - x _ {\star} \| ^ {2}.
$$

# 4 Collinear structure for the strongly convex setup

In this section, we present the collinear structure, a geometric structure observed in a wide range of accelerated first-order methods for the strongly convex setup. We then utilize this structure to obtain novel variants of accelerated first-order methods. We specifically consider two strongly convex setups: $f$ is $L$ -smooth and $\mu$ -strongly convex and $g = 0$ in Section 4.1, while $f = 0$ and $g$ is $\mu$ -strongly convex in Section 4.3.

# 4.1 Collinear structure

Nesterov’s (SC-FGM) [60] has the form:

$$
x _ {k + 1} = x _ {k} ^ {+} + \frac {\sqrt {\kappa} - 1}{\sqrt {\kappa} + 1} \left(x _ {k} ^ {+} - x _ {k - 1} ^ {+}\right)
$$

$$
z _ {k + 1} = x _ {k + 1} + \sqrt {\kappa} \left(x _ {k + 1} - x _ {k} ^ {+}\right) \quad \text {f o r} k = 0, 1, \dots ,
$$

where $\begin{array} { r } { \kappa = \frac { L } { \mu } } \end{array}$ and $x _ { - 1 } ^ { + } : = x _ { 0 }$ . Again, the auxiliary $z _ { k }$ -iterates reveal the geometric structure and play a key role in the Lyapuno nalysis [9, $\ S 5 . 5 ]$ . Figure 2 (left) depicts $x _ { k - 1 } ^ { + }$ , xk , $x _ { k } ^ { + }$ , $x _ { k } ^ { + + }$ , $x _ { k + 1 }$ , zk , and $z _ { k + 1 }$ . These points in $\mathbb { R } ^ { n }$ lie on a 2D-plane, which we again call the plane of iteration. Note that $z _ { k }$ , $z _ { k + 1 }$ , and $x _ { k } ^ { + + }$ are collinear, i.e., there is a line in $\mathbb { R } ^ { n }$ intersecting the three points. (We prove observations 3 and 4 in Section B of the appendix.)

Observation 3. In SC-FGM, $z _ { k }$ , $z _ { k + 1 }$ , and $x _ { k } ^ { + + }$ are collinear3.

![](images/5b4abfc9c13eb290033e127767387afde3d76ab73f3c5ee2b59fd15e143ff6a4.jpg)

![](images/594cbb31ac9c0d914b40d0012c98cae6c1638e41e88e6c96b558e0b6b83c8680.jpg)

![](images/520a89c2cfba773a0f4e0104087a246d00e658b6869d7f6993625b9a0c548fe7.jpg)  
Figure 2: Plane of iteration of SC-FGM (left), TMM (middle), and Proxmal-TMM (right)

Van Scoy, Freeman, and Lynch’s (TMM) [78], which improves upon SC-FGM, has the form:

$$
x _ {k + 1} = x _ {k} ^ {+} + \frac {(\sqrt {\kappa} - 1) ^ {2}}{\sqrt {\kappa} (\sqrt {\kappa} + 1)} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) + \frac {\sqrt {\kappa} - 1}{\sqrt {\kappa}} (x _ {k} ^ {+} - x _ {k})
$$

$$
z _ {k + 1} = x _ {k + 1} + \frac {\sqrt {\kappa} - 1}{2} \left(x _ {k + 1} - x _ {k} ^ {+}\right) \quad \text {f o r} k = 0, 1, \dots ,
$$

where $x _ { - 1 } ^ { + } : = x _ { 0 }$ and $\begin{array} { r } { \kappa = \frac { L } { \mu } } \end{array}$ . TMM also exhibits a similar geometric structure as depicted in Figure 2 (middle), and the $z _ { k }$ -iterates play a similar role in the Lyapunov analysis [19, Theorem 4.19].

Observation 4. In TMM, $z _ { k }$ , $z _ { k + 1 }$ , and $x _ { k } ^ { + + }$ are collinear.

We refer to this geometric structure as the collinear structure. Given an algorithm expressed with momentum and correction terms, one can define the $z _ { k }$ -iterates to exhibit the collinear structure:

$$
x _ {k + 1} = x _ {k} ^ {+} + a _ {k} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) + b _ {k} (x _ {k} ^ {+} - x _ {k}) \quad \Longrightarrow \quad \begin{array}{c} x _ {k} = c _ {k} x _ {k - 1} ^ {+} + (1 - c _ {k}) z _ {k} \\ z _ {k + 1} = d _ {k} x _ {k} ^ {+ +} + (1 - d _ {k}) z _ {k} \end{array}
$$

where $a _ { k }$ , $b _ { k }$ , $c _ { k }$ , and $d _ { k }$ satisfy an appropriate relationship as shown in Section B of the appendix. The collinear structure also holds for non-stationary SC-FGM [19, §4.5], SC-OGM [63], ITEM [73], and geometric descent [13, 30]. The table of Section A.1 presents the precise forms.

# 4.2 Collinear to parallel structure as $\mu \to 0$

We briefly discuss how the parallel structure arises as the limit of the collinear structure as $\mu \to 0$ . When $\mu = 0$ , the collinear structure is undefined, as $x _ { k } ^ { + + } = x _ { k } - ( 1 / \mu ) \nabla f ( x _ { k } )$ ++ is undefined. In the limit $\mu \to 0$ , however, $x _ { k } ^ { + + }$ diverges to a point of infinity in the plane of iteration, and the collinear structure becomes the parallel structure since $z _ { k }$ , $z _ { k + 1 }$ , and $x _ { k } ^ { + + }$ are collinear and $x _ { k }$ , $x _ { k } ^ { + }$ , and $x _ { k } ^ { + + }$ are collinear.

There are two possible scenarios. In the first degenerate case, $z _ { k }$ and $z _ { k + 1 }$ diverges to infinity. This is the case with SC-FGM, SC-OGM, and TMM. In the second scenario, $z _ { k }$ and $z _ { k + 1 }$ stay bounded and $x _ { k } ^ { + } - x _ { k }$ x+k and $z _ { k + 1 } - z _ { k }$ become parallel. This is the case with non-stationary4 SC-FGM and ITEM, which respectively converge to FGM and OGM.

![](images/e63230944f655be8e1633f8af007801299da9ec17bb43c28c1a4d193f1d992be.jpg)  
Figure 3: Collinear to parallel structure as $\mu \to 0$ (ITEM (left) OGM (right))

# 4.3 Other generalizations

Using the collinear structure, we find two novel methods: Proximal-TMM and Proximal-ITEM. These methods can be viewed as proximal versions of TMM [78] and ITEM [73], and they improve upon the accelerated proximal point method [19, 10]. We provide the proofs in Section G.

We present (Proximal-TMM) for the strongly convex proximal-point setup:

$$
x _ {k} = \frac {1 - \sqrt {q}}{1 + \sqrt {q}} x _ {k - 1} ^ {\circ} + \left(1 - \frac {1 - \sqrt {q}}{1 + \sqrt {q}}\right) z _ {k}
$$

$$
z _ {k + 1} = \sqrt {q} x _ {k} ^ {\circ \circ} + (1 - \sqrt {q}) z _ {k} \quad \text {f o r} k = 0, 1, \dots ,
$$

where $\begin{array} { r } { q = \frac { \lambda \mu } { \lambda \mu + 1 } } \end{array}$ , $\begin{array} { r } { x _ { k } ^ { \circ \circ } = x _ { k } - \left( \lambda + \frac { 1 } { \mu } \right) \tilde { \nabla } _ { 1 / \lambda } g ( x _ { k } ) } \end{array}$ , and $x _ { - 1 } ^ { \circ } = x _ { 0 } = z _ { 0 }$

Theorem 3. Consider (P). Assume $g$ is $\mu$ -strongly convex, $g$ has a minimizer $x _ { \star }$ , and $f ~ = ~ 0$ . Proximal-TMM’s $z _ { k }$ -iterates exhibits the rate

$$
\left\| z _ {k} - x _ {\star} \right\| ^ {2} \leq \frac {2}{\mu} (1 - \sqrt {q}) ^ {2 k} (g (x _ {0}) - g (x _ {\star})).
$$

We present (Proximal-ITEM) for the strongly convex proximal-point setup:

$$
x _ {k} = \gamma_ {k} x _ {k - 1} ^ {\circ} + (1 - \gamma_ {k}) z _ {k}
$$

$$
z _ {k + 1} = q \delta_ {k} x _ {k} ^ {\circ \circ} + (1 - q \delta_ {k}) z _ {k} \quad \text {f o r} k = 0, 1, \dots ,
$$

where $\begin{array} { r } { q \ = \ \frac { \lambda \mu } { \lambda \mu + 1 } } \end{array}$ , x◦◦k = xk − λ + 1µ  ∇˜ 1/λg(xk), x◦−1 = x0 = z0, $A _ { 0 } ~ = ~ 0$ , $A _ { k + 1 } ~ =$ $\frac { ( 1 + q ) A _ { k } + 2 \left( 1 + { \sqrt { ( 1 + A _ { k } ) ( 1 + q A _ { k } ) } } \right) } { ( 1 - q ) ^ { 2 } }$ , $\begin{array} { r } { \gamma _ { k } = \frac { A _ { k } } { ( 1 - q ) A _ { k + 1 } } } \end{array}$ , and $\begin{array} { r } { \delta _ { k } = \frac { ( 1 - q ) ^ { 2 } A _ { k + 1 } - ( 1 + q ) A _ { k } } { 2 ( 1 + q + q A _ { k } ) } } \end{array}$ for $k = 0 , 1 , \ldots$

Theorem 4. Consider (P). Assume $g$ is $\mu$ -strongly convex, $g$ has a minimizer $x _ { \star }$ , and $f ~ = ~ 0$ . Proximal-ITEM’s $z _ { k }$ -iterates exhibits the rate

$$
\left\| z _ {k} - x _ {\star} \right\| ^ {2} \leq \frac {(1 - \sqrt {q}) ^ {2 k}}{(1 - \sqrt {q}) ^ {2 k} + q} \left\| z _ {0} - x _ {\star} \right\| ^ {2}.
$$

# 5 Experiments

We consider the compressed sensing [25, 15, 14] problems

$$
\underset {x \in \mathbb {R} ^ {n}} {\text {m i n i m i z e}} \| A x - b \| ^ {2} + \lambda \| x \| _ {1}, \quad \underset {x \in \mathbb {R} ^ {n}} {\text {m i n i m i z e}} \| A x - b \| ^ {2} + \lambda \| x \| _ {\text {n u c}},
$$

where $\lambda > 0$ , $\| \cdot \| _ { 1 }$ is the $\ell _ { 1 }$ -norm, $\| \cdot \| _ { \mathrm { n u c } }$ is the nuclear norm, and the data $A \in \mathbb { R } ^ { m \times n }$ and $b \in \mathbb { R } ^ { m }$ are generated synthetically. We describe the data generation in Section I of the appendix. Figure 4 presents the comparison of ISTA, FISTA, FPGM-m, FISTA, FISTA-G, and FISTA+FISTA-G. The results indicate that FISTA $^ +$ FISTA-G is indeed the most effective at reducing the gradient magnitude.

![](images/5b49e4d4a46bf5ae5cf0b3539f63de29f5abddb2f7215f92e35c3151dccfd4b9.jpg)

![](images/b4e25266accc4b5d9490d512127d2b738f48d44c3ded2b65bf18b6919b40a5d9.jpg)  
Figure 4: Minimizing the gradient magnitude for compressed sensing problems with $\ell _ { 1 }$ regularizer (left) and nuclear norm regularizer (right).

# 6 Conclusion

In this work, we identified geometric structures of accelerated first-order methods and utilized them to find novel variants. Specifically, we found that appropriate auxiliary $z _ { k }$ -iterates reveal parallel and collinear structures and that these iterates are crucial for the Lyapunov analyses. Among the new methods, we highlight FISTA-G, which, combined with FISTA, achieves a $\mathcal { O } ( \bar { \| } x _ { 0 } - x _ { \star } \| ^ { 2 } \big / K ^ { 4 } )$ rate on the squared gradient norm in the prox-grad setup.

FISTA-G and FISTA+FISTA-G have last-iterate rates, i.e., the bound is on $\| \tilde { \nabla } _ { L } F ( x _ { K } ) \| ^ { 2 }$ rather than $\begin{array} { r } { \operatorname* { m i n } _ { k = 0 , \ldots , K } \| \tilde { \nabla } _ { L } F ( x _ { k } ) \| ^ { 2 } } \end{array}$ , but are not anytime algorithms, i.e., the total iteration count $K$ must be known in advance and intermediate iterates have no guarantees. In contrast, the FGM achieves acceleration on function-value suboptimality with a last-iterate rate as an anytime algorithm. Interestingly, all known algorithms for the smooth convex setup or the prox-grad setup with rate $\mathcal { O } ( \| x _ { 0 } - \bar { x } _ { \star } \| ^ { 2 } / K ^ { 3 } )$ or better do not have this property. For example, OGM+OGM-G is not an anytime algorithm but has a last-iterate bound, while FGM is an anytime algorithm with a best-iterate, not a lastiterate, bound. In the prox-grad setup, FPGM-m [45] is not an anytime algorithm but has a last-iterate rate. Whether an anytime algorithm can achieve a last-iterate rate of $\mathcal { O } ( \| x _ { 0 } - x _ { \star } \| ^ { 2 } / K ^ { 3 } )$ or better is an open problem. On a related note, Diakonikolas and Wang conjecture that a $\mathcal { O } ( ( f ( x _ { 0 } ) - f _ { \star } ) / K ^ { 2 } )$ rate is impossible to achieve with an anytime algorithm in the smooth convex setup [24, Conjecture 1].

# Acknowledgements and Disclosure of Funding

JL and EKR were supported by the National Research Foundation of Korea (NRF) Grant funded by the Korean Government (MSIP) [No. 2020R1F1A1A01072877], the National Research Foundation of Korea (NRF) Grant funded by the Korean Government (MSIP) [No. 2017R1A5A1015626], and by the Samsung Science and Technology Foundation (Project Number SSTF-BA2101-02). CP was supported by an undergraduate research internship in the first half of the 2021 Seoul National University College of Natural Sciences. We thank Jaewook Suh, Jisun Park, and TaeHo Yoon for providing valuable feedback. We also thank anonymous reviewers for giving thoughtful comments.

# References

[1] K. Ahn and S. Sra. From Nesterov’s estimate sequence to Riemannian acceleration. COLT, 2020.   
[2] Z. Allen-Zhu. How to make the gradients small stochastically: Even faster convex and nonconvex SGD. NeurIPS, 2018.   
[3] Z. Allen-Zhu and Y. Li. NEON2: Finding local minima via first-order oracles. NeurIPS, 2018.   
[4] Z. Allen-Zhu and L. Orecchia. Linear coupling: An ultimate unification of gradient and mirror descent. ITCS, 2017.   
[5] Z. Allen-Zhu, Z. Qu, P. Richtárik, and Y. Yuan. Even faster accelerated coordinate descent using non-uniform sampling. ICML, 2016.   
[6] J.-F. Aujol and C. Dossal. Optimal rate of convergence of an ODE associated to the fast gradient descent schemes for $b > 0$ . HAL Archives Ouvertes, 2017.   
[7] J.-F. Aujol, C. Dossal, G. Fort, and É. Moulines. Rates of convergence of perturbed FISTA-based algorithms. HAL Archives Ouvertes, 2019.   
[8] J.-F. Aujol, C. Dossal, and A. Rondepierre. Optimal convergence rates for Nesterov acceleration. SIAM Journal on Optimization, 29(4):3131–3153, 2019.   
[9] N. Bansal and A. Gupta. Potential-function proofs for gradient methods. Theory of Computing, 15(4):1–32, 2019.   
[10] M. Barré, A. Taylor, and F. Bach. Principled analyses and design of first-order methods with inexact proximal operators. arXiv preprint arXiv:2006.06041, 2020.   
[11] A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1):183–202, 2009.   
[12] R. E. Bruck. On the weak convergence of an ergodic iteration for the solution of variational inequalities for monotone operators in Hilbert space. Journal of Mathematical Analysis and Applications, 61(1):159–164, 1977.   
[13] S. Bubeck. Convex optimization: Algorithms and complexity. Foundations and Trends® in Machine Learning, 8(3-4):231–357, 2015.   
[14] E. J. Candès, J. Romberg, and T. Tao. Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. IEEE Transactions on Information Theory, 52(2):489–509, 2006.   
[15] E. J. Candès and T. Tao. Near-optimal signal recovery from random projections: Universal encoding strategies? IEEE Transactions on Information Theory, 52(12):5406–5425, 2006.   
[16] S. Chen, S. Ma, and W. Liu. Geometric descent method for convex composite minimization. NeurIPS, 2017.   
[17] P. L. Combettes and J.-C. Pesquet. Proximal splitting methods in signal processing. In H. H. Bauschke, R. S. Burachik, P. L. Combettes, V. Elser, D. R. Luke, and H. Wolkowicz, editors, Fixed-Point Algorithms for Inverse Problems in Science and Engineering, pages 185–212. 2011.   
[18] P. L. Combettes and V. R. Wajs. Signal recovery by proximal forward-backward splitting. Multiscale Modeling and Simulation, 4(4):1168–1200, 2005.   
[19] A. d’Aspremont, D. Scieur, and A. Taylor. Acceleration methods. arXiv preprint arXiv:2101.09545, 2021.   
[20] I. Daubechies, M. Defrise, and C. De Mol. An iterative thresholding algorithm for linear inverse problems with a sparsity constraint. Communications on Pure and Applied Mathematics, 57(11):1413–1457, 2004.   
[21] E. De Klerk, F. Glineur, and A. B. Taylor. Worst-case convergence analysis of inexact gradient and newton methods through semidefinite programming performance estimation. SIAM Journal on Optimization, 30(3):2053–2082, 2020.   
[22] J. Diakonikolas. Halpern iteration for near-optimal and parameter-free monotone inclusion and strong solutions to variational inequalities. COLT, 2020.

[23] J. Diakonikolas and C. Guzmán. Complementary composite minimization, small gradients in general norms, and applications to regression problems. arXiv preprint arXiv:2101.11041, 2021.   
[24] J. Diakonikolas and P. Wang. Potential function-based framework for making the gradients small in convex and min-max optimization. arXiv preprint arXiv:2101.12101, 2021.   
[25] D. L. Donoho. Compressed sensing. IEEE Transactions on Information Theory, 52(4):1289– 1306, 2006.   
[26] R.-A. Dragomir, A. B. Taylor, A. d’Aspremont, and J. Bolte. Optimal complexity and certification of Bregman first-order methods. Mathematical Programming, 2021.   
[27] Y. Drori. The exact information-based complexity of smooth convex minimization. Journal of Complexity, 39:1–16, 2017.   
[28] Y. Drori and A. Taylor. On the oracle complexity of smooth strongly convex minimization. arXiv preprint arXiv:2101.09740, 2021.   
[29] Y. Drori and M. Teboulle. Performance of first-order methods for smooth convex minimization: a novel approach. Mathematical Programming, 145(1):451–482, 2014.   
[30] D. Drusvyatskiy, M. Fazel, and S. Roy. An optimal first order method based on optimal quadratic averaging. SIAM Journal on Optimization, 28(1):251–271, 2018.   
[31] M. Fazlyab, A. Ribeiro, M. Morari, and V. M. Preciado. Analysis of optimization algorithms via integral quadratic constraints: Nonstrongly convex problems. SIAM Journal on Optimization, 28(3):2654–2689, 2018.   
[32] A. V. Gasnikov and Y. E. Nesterov. Universal method for stochastic composite optimization problems. Computational Mathematics and Mathematical Physics, 58(1):48–64, 2018.   
[33] R. Ge, F. Huang, C. Jin, and Y. Yuan. Escaping from saddle points — online stochastic gradient for tensor decomposition. COLT, 2015.   
[34] S. Ghadimi and G. Lan. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming, 156(1-2):59–99, 2016.   
[35] N. Golowich, S. Pattathil, C. Daskalakis, and A. Ozdaglar. Last iterate is slower than averaged iterate in smooth convex-concave saddle point problems. COLT, 2020.   
[36] G. Gu and J. Yang. Tight sublinear convergence rate of the proximal point algorithm for maximal monotone inclusion problems. SIAM Journal on Optimization, 30(3):1905–1921, 2020.   
[37] O. Güler. New proximal point algorithms for convex minimization. SIAM Journal on Optimization, 2(4):649–664, 1992.   
[38] B. Hu and L. Lessard. Dissipativity theory for Nesterov’s accelerated method. ICML, 2017.   
[39] W. Huang and K. Wei. Riemannian proximal gradient methods. Mathematical Programming, 2021.   
[40] X. Huang, E. K. Ryu, and W. Yin. Scaled relative graph of normal matrices. arXiv preprint arXiv:2001.02061, 2019.   
[41] X. Huang, E. K. Ryu, and W. Yin. Tight coefficients of averaged operators via scaled relative graph. Journal of Mathematical Analysis and Applications, 490(1):124211, 2020.   
[42] S. Karimi and S. Vavasis. A single potential governing convergence of conjugate gradient, accelerated gradient and geometric descent. arXiv preprint arXiv:1712.09498, 2017.   
[43] D. Kim. Accelerated proximal point method for maximally monotone operators. Mathematical Programming, 2021.   
[44] D. Kim and J. A. Fessler. Optimized first-order methods for smooth convex minimization. Mathematical Programming, 159(1):81–107, 2016.   
[45] D. Kim and J. A. Fessler. Another look at the fast iterative shrinkage/thresholding algorithm (FISTA). SIAM Journal on Optimization, 28(1):223–250, 2018.   
[46] D. Kim and J. A. Fessler. Generalizing the optimized gradient method for smooth convex minimization. SIAM Journal on Optimization, 28(2):1920–1950, 2018.

[47] D. Kim and J. A. Fessler. Optimizing the efficiency of first-order methods for decreasing the gradient of smooth convex functions. Journal of Optimization Theory and Applications, 188(1):192–219, 2021.   
[48] L. Lessard, B. Recht, and A. Packard. Analysis and design of optimization algorithms via integral quadratic constraints. SIAM Journal on Optimization, 26(1):57–95, 2016.   
[49] H. Li and Z. Lin. Accelerated proximal gradient methods for nonconvex programming. NeurIPS, 2015.   
[50] F. Lieder. On the convergence rate of the Halpern-iteration. Optimization Letters, 15(2):405–418, 2021.   
[51] Q. Lin, Z. Lu, and L. Xiao. An accelerated proximal coordinate gradient method. NeurIPS, 2014.   
[52] J. J. Moreau. Fonctions convexes duales et points proximaux dans un espace hilbertien. Comptes rendus hebdomadaires des séances de l’Académie des sciences, 255:2897–2899, 1962.   
[53] J.-J. Moreau. Proximité et dualité dans un espace hilbertien. Bulletin de la Société mathématique de France, 93:273–299, 1965.   
[54] A. S. Nemirovski. Optimization II: Numerical methods for nonlinear continuous optimization. lecture note, 1999.   
[55] A. S. Nemirovsky. On optimality of Krylov’s information when solving linear operator equations. Journal of Complexity, 7(2):121–130, 1991.   
[56] A. S. Nemirovsky. Information-based complexity of linear operator equations. Journal of Complexity, 8(2):153–175, 1992.   
[57] Y. Nesterov. A method for unconstrained convex minimization problem with the rate of convergence $\mathcal { O } ( 1 / k ^ { 2 } )$ . Proceedings of the USSR Academy of Sciences, 269:543–547, 1983.   
[58] Y. Nesterov. How to make the gradients small. Optima. Mathematical Optimization Society Newsletter, (88):10–11, 2012.   
[59] Y. Nesterov. Gradient methods for minimizing composite functions. Mathematical Programming, 140(1):125–161, 2013.   
[60] Y. Nesterov. Lectures on Convex Optimization. 2nd edition, 2018.   
[61] Y. Nesterov, A. Gasnikov, S. Guminov, and P. Dvurechensky. Primal–dual accelerated gradient methods with small-dimensional relaxation oracle. Optimization Methods and Software, pages 1–38, 2020.   
[62] N. Parikh and S. Boyd. Proximal algorithms. Foundations and Trends in Optimization, 1(3):127– 239, 2014.   
[63] C. Park, J. Park, and E. K. Ryu. Factor- $\sqrt { 2 }$ acceleration of accelerated gradient methods. arXiv preprint arXiv:2102.07366, 2021.   
[64] G. B. Passty. Ergodic convergence to a zero of the sum of monotone operators in Hilbert space. Journal of Mathematical Analysis and Applications, 72(2):383–390, 1979.   
[65] R. T. Rockafellar. Convex Analysis. 1970.   
[66] E. K. Ryu, R. Hannah, and W. Yin. Scaled relative graph: Nonexpansive operators via 2d euclidean geometry. Mathematical Programming, 2021.   
[67] E. K. Ryu, A. B. Taylor, C. Bergeling, and P. Giselsson. Operator splitting performance estimation: Tight contraction factors and optimal parameter selection. SIAM Journal on Optimization, 30(3):2251–2271, 2020.   
[68] E. K. Ryu and W. Yin. Large-Scale Convex Optimization via Monotone Operators. Draft, 2021.   
[69] B. Shi, S. S. Du, M. I. Jordan, and W. J. Su. Understanding the acceleration phenomenon via high-resolution differential equations. Mathematical Programming, 2021.   
[70] W. Su, S. Boyd, and E. Candes. A differential equation for modeling Nesterov’s accelerated gradient method: Theory and insights. NeurIPS, 2014.   
[71] A. B. Taylor. Convex interpolation and performance estimation of first-order methods for convex optimization. PhD thesis, Catholic University of Louvain, Louvain-la-Neuve, Belgium, 2017.

[72] A. B. Taylor and F. Bach. Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions. COLT, 2019.   
[73] A. B. Taylor and Y. Drori. An optimal gradient method for smooth (possibly strongly) convex minimization. arXiv preprint arXiv:2101.09741, 2021.   
[74] A. B. Taylor, J. M. Hendrickx, and F. Glineur. Exact worst-case performance of first-order methods for composite convex optimization. SIAM Journal on Optimization, 27(3):1283–1313, 2017.   
[75] A. B. Taylor, J. M. Hendrickx, and F. Glineur. Smooth strongly convex interpolation and exact worst-case performance of first-order methods. Mathematical Programming, 161(1-2):307–345, 2017.   
[76] A. B. Taylor, J. M. Hendrickx, and F. Glineur. Exact worst-case convergence rates of the proximal gradient method for composite convex minimization. Journal of Optimization Theory and Applications, 178(2):455–476, 2018.   
[77] P. Tseng. On accelerated proximal gradient methods for convex-concave optimization. submitted to SIAM Journal on Optimization, 2008.   
[78] B. Van Scoy, R. A. Freeman, and K. M. Lynch. The fastest known globally convergent first-order method for minimizing strongly convex functions. IEEE Control Systems Letters, 2(1):49–54, 2017.   
[79] Y. Xu, R. Jin, and T. Yang. First-order stochastic algorithms for escaping from saddle points in almost linear time. NeurIPS, 2018.   
[80] T. Yoon and E. K. Ryu. Accelerated algorithms for smooth convex-concave minimax problems with $\mathcal { O } ( 1 / k ^ { 2 } )$ rate on squared gradient norm. ICML, 2021.

# A Method reference

In this section, we list the methods considered in this work in two forms: a form with momentum and correction terms and a form with the auxiliary iterates. The momentum and correction terms of an iteration are loosely defined as

$$
x _ {k + 1} = x _ {k} ^ {+} + \underbrace {a _ {k} (x _ {k} ^ {+} - x _ {k - 1} ^ {+})} _ {\text {m o m e n t u m t e r m}} + \underbrace {b _ {k} (x _ {k} ^ {+} - x _ {k})} _ {\text {c o r r e c t i o n t e r m}}.
$$

In the proximal-point and prox-grad setup, similar definitions are made with the $x _ { k } ^ { \circ }$ and $x _ { k } ^ { \oplus }$ terms. One of our main points is that the form with the auxiliary iterates has the advantage of better revealing the parallel and collinear structure, although the form with momentum and correction terms is more commonly presented in the accelerated methods literature. We separate the tables into existing methods and the novel methods we present.

# A.1 Existing Methods

<table><tr><td>Method name</td><td>With momentum</td><td>With auxiliary iterates</td></tr><tr><td>FGM [57]</td><td>xk+1=xk+θk-1/θk+1(xk+1-xk-1) for k=0,1,..., where x-1:=x0, θ0=1, and θk+1=1+√1+4θk2/2 for k=0,1,...</td><td>xk=θk-1/θk2xk-1+(1-θk-1/θk2)zk
zk+1=zk-θk1/L∇f(xk)
for k=0,1,..., where z0=x0 and θ-1=0</td></tr><tr><td>OGM [29, 44]</td><td>xk+1=xk+θk-1/θk+1(xk+1-xk-1) + θk/θk+1(xk+1-xk) for k=0,1,..., K-1, where x-1:=x0, θ0=1, θk+1=1+√1+4θk2/2 for k=0,1,..., K-1, and θK=1+√1+8θK-1/2</td><td>xk=θk-1/θk2xk-1+(1-θk-1/θk2)zk
zk+1=zk-2θk1/L∇f(xk)
for k=0,1,..., K, where z0=x0 and θ-1=0</td></tr><tr><td>OGM-G [47]</td><td>xk+1=xk+θk-1/θk(2θk-1)/2+2θk+1-1/2θk-1(xk+1-xk) for k=0,1,..., K-1, where x-1:=x0, θK=1, θk=1+√1+4θk2/2 for k=1,2,..., K-1, and θ0=1+√1+8θ1/2</td><td>xk=θk+1/θk4xk-1+(1-θk+1/θk4)zk
zk+1=zk-θk1/L∇f(xk)
for k=0,1,..., K-1, where z0=x0 and z1=z0-θk+1/2 L∇f(x0)</td></tr><tr><td>SC-FGM [60, (2.2.22)]</td><td>xk+1=xk+√κ-1/√κ+1(xk+1-xk-1) for k=0,1,..., where κ=L/μ and x-1:=x0</td><td>xk=√κ/√κ+1xk-1+1/√κ+1zk
zk+1=1/√κxk++√κ-1/√κzk
for k=0,1,..., where z0=x0</td></tr><tr><td>non-stationary SC-FGM [19, §4.5]</td><td>xk+1=xk+αk(xk+1-xk-1) where κ=L/μ, x+1:=x0, A0=0, A1=(1-κ-1)-1, Ak+2=2Ak+2+1+√4Ak+1+4κ-1A2k+1/2(1-κ-1), and αk=(Ak+2-Ak+1)(Ak+1(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+2-Ak+1)(Ak+1(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+2-Ak+1)(Ak+1(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+2-Ak+1)(Ak+1)(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+1-Ak)(Ak+1)(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+1-Ak)(Ak+1)(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+1-Ak)(Ak+1)(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+1-Ak)(Ak+1)(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+1-Ak)(Ak+1)(1-κ-1)-Ak-1)/2(1-κ-1), and αk=(Ak+1-Ak)(Ak+1)(1-κ-1)-Ak-1)/2(1-κ-1),</td><td>xk=(1-γk)xk-1+γzk
zk+1=κ-1δkxk++(1-κ-1δk)zk
for k=0,1,..., where z0=x0,
γk=(Ak+1-Ak)(1+κ-1Ak) / (Ak+1+2κ-1AkAkk+1-κ-1Ak2), and δk=Ak+1-Ak/1+κ-1Ak+1
for k=0,1,...</td></tr><tr><td>SC-OGM [63]</td><td>x_{k+1}=x_{k}^{+}+\frac{\kappa-1}{\sqrt{8\kappa+1}+2+\kappa}(x_{k}^{+}-x_{k-1}^{+})+ \frac{\kappa-1}{\sqrt{8\kappa+1}+2+\kappa}(x_{k}^{+}-x_{k})for k=0,1,..., where \kappa=\frac{L}{\mu} and x_{-1}^{+}:=x_{0}</td><td>x_{k}=\frac{\sqrt{8\kappa+1}+3}{2(\sqrt{8\kappa+1}+2+\kappa)}x_{k-1}^{+}+\frac{\sqrt{8\kappa+1}+1+2\kappa}{2(\sqrt{8\kappa+1}+2+\kappa)}z_{k}\\ z_{k+1}=\frac{\sqrt{1+8\kappa}+5-2\kappa}{\sqrt{1+8\kappa}+3}x_{k}^{++}+\frac{2\kappa-2}{\sqrt{1+8\kappa}+3}z_{k}\\ for k=0,1,..., where z_{0}=x_{0}</td></tr><tr><td>TMM [78]</td><td>x_{k+1}=x_{k}^{+}+\frac{(\sqrt{\kappa}-1)^{2}}{\sqrt{\kappa}(\sqrt{\kappa}+1)}(x_{k}^{+}-x_{k-1}^{+})+\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}}(x_{k}^{+}-x_{k})for k=0,1,..., where x_{-1}^{+}:=x_{0} and \kappa=\frac{L}{\mu}</td><td>x_{k}=\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}x_{k-1}^{+}+\frac{2}{\sqrt{\kappa}+1}z_{k}\\ z_{k+1}=\frac{1}{\sqrt{\kappa}}x_{k}^{++}+\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}}z_{k}\\ for k=0,1,..., where z_{0}=x_{0}</td></tr><tr><td>Geometric descent [13, 30]</td><td colspan="2">z_{0}=x_{0}^{++},R_{0}^{2}=\left(1-\frac{1}{\kappa}\right)\frac{\|\nabla f(x_{0})\|^{2}}{\mu^{2}}\lambda_{k+1}=\arg\min_{\lambda\in\mathbb{R}}f((1-\lambda)c_{t}+\lambda x_{k}^{+})\\ x_{k+1}=(1-\lambda_{k+1})z_{k}+\lambda_{k+1}x_{k}^{+}\\ If\frac{|\nabla f(x_{k})|^{2}}{\mu^{2}}&lt;\frac{R_{k}^{2}}{2},\\ z_{k+1}=x_{k+1}^{++}\\ R_{k+1}^{2}=\frac{\|\nabla f(x_{k+1})\|^{2}/\mu^{2}}{1-\kappa^{-1}}.\\ If\frac{|\nabla f(x_{k})|^{2}}{\mu^{2}}\geq\frac{R_{k}^{2}}{2},\\ z_{k+1}=(1-\frac{R_{k}^{2}+\|x_{k+1}-z_{k}\|^{2}}{2\|x_{k+1}^{++}-z_{k}\|^{2}})z_{k}+\frac{R_{k}^{2}+\|x_{k+1}-z_{k}\|^{2}}{2\|x_{k+1}^{++}-z_{k}\|^{2}}x_{k+1}^{++}\\ R_{k+1}^{2}=R_{k}^{2}-\frac{\|\nabla f(x_{k})\|^{2}}{\mu^{2}\kappa}-\left(\frac{R_{k}^{2}+\|x_{k+1}-z_{k}\|^{2}}{2\|x_{k+1}^{++}-z_{k}\|^{2}}\right)^{2}</td></tr><tr><td>ITEM [73]</td><td>x_{k+1}=x_{k}^{+}+\alpha_{k}(x_{k}^{+}-x_{k-1}^{+})+\beta_{k}(x_{k}^{+}-x_{k})k=0,1,..., where \kappa=\frac{L}{\mu},x_{-1}^{+}:=x_{0},A_{0}=0,A_{1}=(1-\kappa^{-1})^{-1},\\ A_{k+2}=\frac{(1+\kappa^{-1})A_{k+1}+2(1+\sqrt{(1+A_{k+1})(1+\kappa^{-1}A_{k+1})}}{(1-\kappa^{-1})^{2}},\\ \alpha_{k}=\frac{(2(1+\kappa^{-1})+\kappa^{-1}(3+\kappa^{-1})A_{k}+(1-\kappa^{-1})^{2}\kappa^{-1}A_{k+1})((1-\kappa^{-1})A_{k+2}-A_{k+1})A_{k}}{2(1-\kappa^{-1})(1+\kappa^{-1}+\kappa^{-1}A_{k})(1-\kappa^{-1})A_{k+1}-A_{k})A_{k+2}},\\ \text{and}\beta_{k}=\frac{(\kappa^{-1}A_{k}^{2}+2(1-\kappa^{-1})A_{k+1}+(1-\kappa^{-1})\kappa^{-1}A_{k}A_{k+1})((1-\kappa^{-1})A_{k+2}-A_{k+1})}{2(1+\kappa^{-1}+\kappa^{-1}A_{k})(1-\kappa^{-1})A_{k+1}-A_{k})A_{k+2}}\text{for }k=0,1,...</td><td>x_{k}=\gamma_{k}x_{k-1}^{+}+(1-\gamma_{k})z_{k}\\ z_{k+1}=\kappa^{-1}\delta_{k}x_{k}^{++}+(1-\kappa^{-1}\delta_{k})z_{k}\\ \text{for }k=0,1,..., where z_{0}=x_{0},A_{0}=0,\kappa=\frac{L}{\mu},\\ \gamma_{k}=\frac{A_{k}}{(1-\kappa^{-1})A_{k+1}}\text{and}\delta_{k}=\frac{(1-\kappa^{-1})^{2}A_{k+1}-(1+\kappa^{-1})A_{k}}{1+\kappa^{-1}+\kappa^{-1}A_{k}}\\ \text{for }k=0,1,...</td></tr><tr><td>ISTA [20]</td><td colspan="2">x_{k+1}=x_{k}^{\oplus}\quad\text{for }k=0,1,...</td></tr><tr><td>FISTA [11]</td><td>xk+1=xk+θk-1/θk+1(xk-1) for k=0,1,..., where x-1:=x0, θ0=1, and θk+1=1+√1+4θk2/2 for k=0,1,...</td><td>xk=θk-1/θk2xk-1+(1-θk-1/θk2)zk=zk-θkL/√L∇LF(xk) for k=0,1,..., where z0=x0</td></tr><tr><td>FPGM-m [45]</td><td>xk+1=xk+θk-1/θk+1(xk-1) for 0≤k≤m-1 xk+1=xkfor m≤k≤K where x-1:=x0, θ0=1, and θk+1=1+√1+4θk2/2 for k=0,1,..., m-1</td><td>xk=θk-1/θk2xk-1+(1-θk-1/θk2)zk=zk-θk∇1/λg(xk) for k=0,1,..., where z0=x0 and θ-1=0</td></tr><tr><td>Güler 1 [37]</td><td>xk+1=xk+θk-1/θk+1(xk-1) for k=0,1,..., where x-1:=x0, θ0=1, and θk+1=1+√1+4θk2/2 for k=0,1,...</td><td>xk=θk-1/θk2xk-1+(1-θk-1/θk2)zk=zk+1=zk-θk∇1/λg(xk) for k=0,1,..., where z0=x0 and θ-1=0</td></tr><tr><td>Güler 2 [37]</td><td>xk+1=xk+θk-1/θk+1(xk-1) for k=0,1,..., where x-1:=x0, θ0=1, and θk+1=1+√1+4θk2/2 for k=0,1,...</td><td>xk=θk-1/θk2xk-1+(1-θk-1/θk)zk=zk+1=zk-2θk∇1/λg(xk) for k=0,1,..., where z0=x0 and θ-1=0</td></tr></table>

# A.2 Novel methods

<table><tr><td>Method name</td><td>With momentum</td><td>With auxiliary iterates</td></tr><tr><td>FISTA-G</td><td>x_{k+1}=x_{k}^{\oplus}+\frac{\varphi_{k+1}-\varphi_{k+2}}{\varphi_{k}-\varphi_{k+1}}(x_{k}^{\oplus}-x_{k-1}^{\oplus}) for k=0,1,...,K-1, where x_{-1}^{\oplus}:=x_{0}, \varphi_{K+1}=0, \varphi_{K}=x_{k+1}+\frac{x_{k+1}^{\oplus}+(x_{k+1}-\varphi_{k+2})\sqrt{x_{k+1}^{\oplus}+3x_{k+1}^{\oplus}}}{x_{k+1}+\varphi_{k+2}} for k=0,1,...,K-1</td><td>x_{k}=\frac{\varphi_{k+1}}{\varphi_{k}}x_{k-1}^{\oplus}+\left(1-\frac{\varphi_{k+1}}{\varphi_{k}}\right)z_{k} z_{k+1}=z_{k}-\frac{\varphi_{k}}{\varphi_{k}-\varphi_{k+1}}\frac{1}{L}\tilde{\nabla}_{L}F(x_{k}) for k=0,1,...,K, where z_{0}=x_{0}</td></tr><tr><td>G-FISTA-G</td><td>x_{k+1}=x_{k}^{\oplus}+\frac{\varphi_{k+1}-\varphi_{k+2}}{\varphi_{k}-\varphi_{k+1}}(x_{k}^{\oplus}-x_{k-1}^{\oplus}) +\frac{\varphi_{k+1}-\varphi_{k+2}}{\varphi_{k+1}}(\tau_{k}\varphi_{k}-\tau_{k+1}\varphi_{k+1}-\frac{\varphi_{k}}{\varphi_{k}-\varphi_{k+1}})(x_{k}^{\oplus}-x_{k}) for k=0,1,...,K-1, where x_{-1}^{\oplus}:=x_{0}, \tau_{K}=\varphi_{K}=1,\varphi_{K+1}=0, and \{\varphi_{k}\}_{k=0}^{K-1} and the nondecreasing nonnegative sequence \{\tau_{k}\}_{k=0}^{K-1}\text{ satisfying } \tau_{k}\varphi_{k}-\tau_{k+1}\varphi_{k+1}=\varphi_{k+1}(\tau_{k+1}-\tau_{k})+1, and (\tau_{k}\varphi_{k}-\tau_{k+1}\varphi_{k+1})(\tau_{k+1}-\tau_{k})-\frac{\tau_{k+1}}{2}\leq0 for k=0,1,...,K-1</td><td>x_{k}=\frac{\varphi_{k+1}}{\varphi_{k}}x_{k-1}^{\oplus}+\left(1-\frac{\varphi_{k+1}}{\varphi_{k}}\right)z_{k} z_{k+1}=z_{k}-(\tau_{k}\varphi_{k}-\tau_{k+1}\varphi_{k+1})\frac{1}{L}\tilde{\nabla}_{L}F(x_{k}) for k=0,1,...,K, where z_{0}=x_{0}</td></tr><tr><td>FGM-G</td><td>x_{k+1}=x_{k}^{\oplus}+\frac{\varphi_{k+1}-\varphi_{k+2}}{\varphi_{k}-\varphi_{k+1}}(x_{k}^{\oplus}-x_{k-1}^{\oplus}) for k=0,1,...,K-1, where x_{-1}^{\oplus}:=x_{0}, \boldsymbol{\varphi}_{K+1}=0, \varphi_{K}=1, and \varphi_{k}=\frac{\varphi_{k+2}^{\oplus}-\varphi_{k+1}\varphi_{k+2}+2\varphi_{k+1}^{\oplus}+(\varphi_{k+1}-\varphi_{k+2})\sqrt{\varphi_{k+2}^{\oplus}+3\varphi_{k+1}^{\oplus}}}{\varphi_{k+1}+\varphi_{k+2}} for k=0,1,...,K-1</td><td>x_{k}=\frac{\varphi_{k+1}}{\varphi_{k}}x_{k-1}^{\oplus}+\left(1-\frac{\varphi_{k+1}}{\varphi_{k}}\right)z_{k} z_{k+1}=z_{k}-\frac{\varphi_{K}}{\varphi_{K}-\varphi_{K+1}}\frac{1}{L}\nabla f(x_{k}) for k=0,1,...,K, where z_{0}=x_{0}</td></tr><tr><td>G-FGM-G</td><td>x_{k+1}=x_{k}^{\oplus}+\frac{\varphi_{k+1}-\varphi_{k+2}}{\varphi_{k}-\varphi_{k+1}}(x_{k}^{\oplus}-x_{k-1}^{\oplus}) +\frac{\varphi_{k+1}-\varphi_{k+2}}{\varphi_{k+1}}(\tau_ {k}\varphi_{k}-\tau_{k+1}\varphi_{k+1}-\frac{\varphi_{k}}{\varphi_{k}-\varphi_{k+1}})(x_{k}^{\oplus}-x_{k}) for k=0,1,...,K-1, where x_{-1}^{\oplus}:=x_{0}, \tau_{K}=\varphi_{K}=1, \varphi_{K+1}=0, and \{\varphi_{k}\}_{k=0}^{K-1} and the nondecreasing nonnegative sequence \{\tau_{k}\}_{k=0}^{K-1} satisfying \tau_{k}\varphi_{k}-\tau_{k+1}\varphi_{k+1}=\varphi_{k+1}(\tau_{k+1}-\tau_{k})+1 and (\tau_{k}\varphi_{k}-\tau_{k+1}\varphi_{k+1})(\tau_{k+1}-\tau_{k})-\tau_{k+1}\leq0 for k=0,1,...,K-1</td><td>x_{k}=\frac{\varphi_{k+1}}{\varphi_{k}}x_{k-1}^{\oplus}+\left(1-\frac{\varphi_{k+1}}{\varphi_{k}}\right)z_{k} z_{k+1}=z_{k}-(\tau_{k}\varphi_{k}-\tau_{k+ 1}\varphi_{k+1})\frac{1}{L}\nabla f(x_{k}) for k=0,1,...,K, where z_{0}=x_{0}</td></tr><tr><td>Güler-G</td><td>x_{k+1}=x_{k}^{\circ}+\frac{(\theta_{k}-1)(2\theta_{k+1}-1)}{\theta_{k}(2\theta_{k}-1)}(x_{k}^{\circ}-x_{k-1}^{\circ}) +\frac{2\theta_{k+1}-1}{2\theta_{k}-1}(x_{k}^{\circ}-x_{k}) for k=0,1,...,K-1, where x_{-1}^{\circ}:=x_{0}, \theta_{K}=1, and \theta_{k}=\frac{1+\sqrt{1+4\theta_{k+1}^{2}}}{2} for k=0,1,...,K-1</td><td>x_{k}=\frac{\theta_{k+1}^{4}}{\theta_{k}^{4}}x_{k-1}^{\circ}+\left(1-\frac{\theta_{k+1}^{4}}{\theta_{k}^{4}}\right)z_{k} z_{k+1}=z_{k}-\theta_{k}\tilde{\nabla}_{1/\lambda}g(x_{k}) for k=0,1,...,K, where z_{0}=x_{0} and \theta_{K+1}=0</td></tr><tr><td>G-Güler-G</td><td>xk+1=x0+φk+1-φk+2/φk-φk+1(xk-φk-1)+φk+1-φk+2/φk+1(τkφk-τk+1φk+1-φk/φk+1)(xk-φk) for k=0,1,...,K-1 where x-1:=x0, τK=φK=1, φK+1=0, and {φk}K-1 and the nondecreasing nonnegative sequence {τk}K-1 satisfying τkφk-τk+1φk+1=φk+1(τk+1-τk) + 1 and (τkφk-τk+1φk+1)(τk+1-τk)-τk+1≤0 for k=0,1,...,K-1</td><td>xk=φk+1/φk-1+xk-1+(1-φk/φk)zk
zk+1=zk-(τkφk-τk+1φk+1)1/Lˆ∇1/λg(xk)
for k=0,1,...,K, where z0=x0</td></tr><tr><td>Proximal-TMM</td><td>xk+1=x0+√q-1/√q+1(xk-φk-1)+(1-√q)(xk-φk) for k=0,1,...,where x-1:=x0 and q=λμ/λμ+1</td><td>xk=1-√q/1+√qxk-1+(1-1-√q)/1+√qzk
zk+1=√qxk+1(1-√q)zk
for k=0,1,...,where xk+1=xdk-(λ+1/μ)ˆ∇1/λg(xk), and x-1=xdk=0,1,...</td></tr><tr><td>Proximal-ITEM</td><td>xk+1=x0+αk(xk-φk-1)+βk(xk-φk) for k=0,1,...,where q=λμ/λμ+1, x-1:=x0, A0=0, A1=(1-q)^{-1}, A(k+2)=(1+q)A(k+1)+2(1+√(1+A(k+1)(1+qA(k+1)/(1-q)^2), αk=(2(1+q)+q(3+q)A(k+(1-q)^2qA(k+1)((1-q)A(k+2-A(k+1)A(k)/2(1-q)(1+q+qA(k)((1-q)A(k+1-A(k)A(k+2), and βk=(qA(k^2+2(1-q)A(k+1+(1-q)qA(kA(k+1)((1-q)A(k+2-A(k+1)/2(1+q+qA(k)((1-q)A(k+1-A(k)A(k+2) for k=0,1,...</td><td>xk=γkx0-1+(1-γk)zk
zk+1=qδkxA0+1-(1-qδk)zk
for k=0,1,...,where z0=xdk=0, xk+1=Adk/(1-q)A(k+1), and δk=(1-q^2A(k+1-(1+q)A(k)/2(1+q+qA(k)) for k=0,1,...</td></tr></table>

# B Omitted proofs of geometric observation and form of algorithm

In this section, we formally establish the basic geometric claims made in the main body.

First, we state parallel lemma (left) and Menelaus’s lemma (right), which are classical results in Euclidean geometry:

![](images/0ee36c34047e15e5f88130992c73545f2bc0984470a4f4d148f88b929c78bb15.jpg)

$\overline { { B C } } \parallel \overline { { B ^ { \prime } C ^ { \prime } } }$ if and only if $\begin{array} { r } { \frac { \overline { { A B } } } { \overline { { B B ^ { \prime } } } } = \frac { \overline { { A C } } } { \overline { { C C ^ { \prime } } } } } \end{array}$

![](images/f95d5e8844eb42470304f524baf8145d06f8d522571b3237095fac258621d2e4.jpg)

$A ^ { \prime } , B ^ { \prime } , C ^ { \prime }$ is on line if and only if ${ \frac { \overline { { A ^ { \prime } B } } } { \overline { { A A ^ { \prime } } } } } \cdot { \frac { \overline { { B ^ { \prime } C } } } { \overline { { B B ^ { \prime } } } } } \cdot { \frac { \overline { { C ^ { \prime } A } } } { \overline { { C C ^ { \prime } } } } } = 1$

# B.1 Omitted proofs of observations

Proof of Observation 1. Figure 1 (left) depicts the plane of iteration of FGM. In the plane of iteration of FGM,

$$
\frac {\| x _ {k} - x _ {k - 1} ^ {+} \|}{\| z _ {k} - x _ {k} \|} = \frac {1}{\theta_ {k} - 1} = \frac {1}{\frac {\theta_ {k} - 1}{\theta_ {k + 1}} + \frac {(\theta_ {k} - 1) (\theta_ {k + 1} - 1)}{\theta_ {k + 1}}} = \frac {\| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \|}{\| x _ {k + 1} - x _ {k} ^ {+} \| + \| z _ {k + 1} - x _ {k + 1} \|} = \frac {\| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \|}{\| z _ {k + 1} - x _ {k} ^ {+} \|}
$$

by definition of $z _ { k } , x _ { k + 1 } , z _ { k + 1 }$ . Then the result comes from parallel lemma.

Proof of Observation 2. Figure 1 (middle) depicts the plane of iteration of OGM. By extending $\overline { { x _ { k - 1 } ^ { + } x _ { k } ^ { + } } }$ and defining new point $B$ that meets with $\stackrel { \longleftrightarrow } { z _ { k } } { z _ { k + 1 } }$ , observation can also be shown by parallel lemma. □

Proof of Observation 3. Figure 2 (left) depicts the plane of iteration of SC-FGM. Apply Menelaus’s lemma for $\triangle x _ { k - 1 } ^ { + } x _ { k } x _ { k } ^ { + }$ and $\overline { { z _ { k } z _ { k + 1 } x _ { k } ^ { + + } } }$ , that

$$
\frac {\| z _ {k} - x _ {k - 1} ^ {+} \|}{\| z _ {k} - x _ {k} \|} \cdot \frac {\| z _ {k + 1} - x _ {k} ^ {+} \|}{\| z _ {k + 1} - x _ {k - 1} ^ {+} \|} \cdot \frac {\| x _ {k} ^ {+ +} - x _ {k} \|}{\| x _ {k} ^ {+ +} - x _ {k} ^ {+} \|} = \frac {\sqrt {\kappa} + 1}{\sqrt {\kappa}} \cdot \frac {\sqrt {\kappa} - 1}{\sqrt {\kappa}} \cdot \frac {\frac {1}{\mu}}{\frac {1}{\mu} - \frac {1}{L}} = 1.
$$

Proof of Observation 4. Figure 2 (middle) depicts the plane of iteration of TMM. By extending $\overline { { x _ { k } ^ { + } x _ { k + 1 } } }$ and defining new point $Q$ that meets with $\overleftrightarrow { x _ { k - 1 } ^ { + } x _ { k } }$ , Observation 4 can be shown by Menelaus’s lemma for $\triangle Q x _ { k } x _ { k } ^ { + }$ and $\overline { { z _ { k } z _ { k + 1 } x _ { k } ^ { + + } } }$ . □

# B.2 Parallel structure from momentum-based iteration

Lemma 1. An iteration of the form

$$
x _ {k + 1} = x _ {k} ^ {+} + \frac {a _ {k} - 1}{a _ {k + 1}} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) + b _ {k + 1} (x _ {k} ^ {+} - x _ {k})
$$

![](images/95d7bb52fb9a4e5841c63e88b7f062ebe7c1614b24978aa05966738ee3303863.jpg)

![](images/5c59e243eb3323c7bb095512d3b71937d3714d478ff7f2b1c684aeaa656a72a3.jpg)  
Figure 5: Lemma 1 and Lemma 2

for $k = 0 , 1 , \ldots , K - 1$ , where $1 \leq a _ { 0 }$ , $1 < a _ { k }$ for $k = 1 , 2 , \ldots , K - 1$ , and $1 \leq a _ { K }$ , can be equivalently expressed as

$$
x _ {k} = \frac {\varphi_ {k - 1}}{\varphi_ {k}} x _ {k - 1} ^ {+} + \left(1 - \frac {\varphi_ {k - 1}}{\varphi_ {k}}\right) z _ {k}
$$

$$
z _ {k + 1} = z _ {k} - \frac {a _ {k} + a _ {k + 1} b _ {k + 1}}{L} \nabla f (x _ {k})
$$

for $k = 0 , 1 , \ldots , K$ , where 0 = ϕ−1, $0 < \varphi _ { k }$ , $\begin{array} { r } { \frac { a _ { k } - 1 } { a _ { k } } \ : = \ : \frac { \varphi _ { k - 1 } } { \varphi _ { k } } } \end{array}$ for $k = 1 , 2 , \ldots , K ,$ , and $\varphi _ { K } \leq \infty$ . (If ak $\varphi _ { K } = \infty$ , we define $\varphi _ { K - 1 } / \varphi _ { K } = 0 .$ .)

Proof. First, suppose $x _ { k }$ is not a minimizer which implies $\nabla f ( x _ { k } ) \neq 0$ and neither $x _ { k - 1 } ^ { + }$ , xk, zk are not the sathe firstthat orith. Let with aon the ry itthat ow . L $x _ { k - 1 } ^ { + } , x _ { k } , z _ { k }$ $A$ $\overleftrightarrow { x _ { k - 1 } ^ { + } x _ { k } ^ { + } }$ $\longleftrightarrow$ $z _ { k + 1 } : = \overleftrightarrow { z _ { k } B } \cap \overleftrightarrow { x _ { k } ^ { + } x _ { k + 1 } }$ the t $\overline { { A x _ { k + 1 } } } \parallel \overline { { x _ { k } x _ { k } ^ { + } } } .$ $B$ $x _ { k } ^ { + } A$ $\overline { { z _ { k } B } } \parallel \overline { { x _ { k } x _ { k } ^ { + } } }$ + Then, the condition for parallel term style is satisfied. We will show that the formula above also holds.

Since $\overline { { x _ { k } x _ { k } ^ { + } } } \parallel \overline { { z _ { k } B } }$ , parallel lemma indicates that

$$
\frac {\left\| z _ {k} - x _ {k} \right\|}{\left\| x _ {k} - x _ {k - 1} ^ {+} \right\|} = \frac {\left\| B - x _ {k} ^ {+} \right\|}{\left\| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \right\|} = \frac {\varphi_ {k - 1}}{\varphi_ {k} - \varphi_ {k - 1}}
$$

Since $\overline { { A x _ { k + 1 } } } \parallel \overline { { B z _ { k + 1 } } }$ , parallel lemma indicates that

$$
\frac {\left\| z _ {k + 1} - x _ {k} ^ {+} \right\|}{\left\| x _ {k + 1} - x _ {k} ^ {+} \right\|} = \frac {\left\| B - x _ {k} ^ {+} \right\|}{\left\| A - x _ {k} ^ {+} \right\|} = \frac {\varphi_ {k + 1}}{\varphi_ {k + 1} - \varphi_ {k}}
$$

Then,

$$
\frac {\left| \left| A - x _ {k} ^ {+} \right| \right|}{\left| \left| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \right| \right|} = \frac {a _ {k} - 1}{a _ {k + 1}} = \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} \cdot \frac {\varphi_ {k - 1}}{\varphi_ {k} - \varphi_ {k - 1}}
$$

and this relation holds if $\begin{array} { r } { a _ { k + 1 } = \frac { \varphi _ { k + 1 } } { \varphi _ { k + 1 } - \varphi _ { k } } \Longleftrightarrow \frac { a _ { k + 1 } - 1 } { a _ { k + 1 } } = \frac { \varphi _ { k } } { \varphi _ { k + 1 } } } \end{array}$ (This strong condition is for easy Lyapunov analysis).

Lastly, by parallel lemma and previous condition,

$$
z _ {k + 1} - B = \frac {\left\| z _ {k + 1} - x _ {k} ^ {+} \right\|}{\left\| x _ {k + 1} - x _ {k} ^ {+} \right\|} (x _ {k + 1} - A) = a _ {k + 1} b _ {k + 1} (x _ {k} ^ {+} - x _ {k})
$$

since $\| x _ { k + 1 } - A \| = b _ { k + 1 } \left\| x _ { k } ^ { + } - x _ { k } \right\|$ and

$$
B - z _ {k} = \frac {\left\| z _ {k} - x _ {k - 1} ^ {+} \right\|}{\left\| x _ {k} - x _ {k - 1} ^ {+} \right\|} (x _ {k} ^ {+} - x _ {k}) = a _ {k} (x _ {k} ^ {+} - x _ {k}),
$$

which indicates

$$
z _ {k + 1} = z _ {k} - \left(a _ {k} + a _ {k + 1} b _ {k + 1}\right) \frac {1}{L} \nabla f (x _ {k}).
$$

If $x _ { k }$ is a minimizer which implies $\nabla f ( x _ { k } ) = 0$ and $z _ { k + 1 } - z _ { k } = x _ { k } ^ { + } - x _ { k } = 0$ , this is degenerate case. In this case, proof is trivial.

Lemma 2. An iteration of the form

$$
x _ {k} = \frac {\varphi_ {k - 1}}{\varphi_ {k}} x _ {k - 1} ^ {+} + \left(1 - \frac {\varphi_ {k - 1}}{\varphi_ {k}}\right) z _ {k}
$$

$$
z _ {k + 1} = z _ {k} - \frac {\phi_ {k}}{L} \nabla f (x _ {k})
$$

for $k = 0 , 1 , \ldots , K ,$ , where $\{ \varphi _ { k } \} _ { k = - 1 } ^ { K }$ is a nonnegative increasing sequence, can be equivalently expressed as

$$
x _ {k + 1} = x _ {k} ^ {+} + \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} \cdot \frac {\varphi_ {k - 1}}{\varphi_ {k} - \varphi_ {k - 1}} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) + \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} \left(\phi_ {k} - \frac {\varphi_ {k}}{\varphi_ {k} - \varphi_ {k - 1}}\right) (x _ {k} ^ {+} - x _ {k})
$$

for $k = 0 , 1 , \ldots , K$ .

Proof. Suppose $x _ { k }$ is not a minimizer which implies $\nabla f ( x _ { k } ) \neq 0$ . Set $A$ on the $\overleftrightarrow { x _ { k - 1 } ^ { + } x _ { k } ^ { + } }$ that $\overline { { A x _ { k + 1 } } } \parallel \overline { { x _ { k } x _ { k } ^ { + } } }$ x k x +k . Let $B$ on the $\overleftrightarrow { x _ { k - 1 } ^ { + } x _ { k } ^ { + } } \cap \overline { { z _ { k } z _ { k + 1 } } }$ . Since $\overline { { x _ { k } x _ { k } ^ { + } } } \parallel \overline { { z _ { k } B } }$ and $\overline { { A x _ { k + 1 } } } \parallel \overline { { B z _ { k + 1 } } }$ ,

$$
\begin{array}{l} A - x _ {k} ^ {+} = \left(B - x _ {k} ^ {+}\right) - (B - A) = \left(B - x _ {k} ^ {+}\right) - \frac {\varphi_ {k}}{\varphi_ {k + 1}} \left(B - x _ {k} ^ {+}\right) \\ = \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} (B - x _ {k} ^ {+}) = \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} \cdot \frac {\varphi_ {k - 1}}{\varphi_ {k} - \varphi_ {k - 1}} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}). \\ \end{array}
$$

In addition, since $\overline { { x _ { k } x _ { k } ^ { + } } } \parallel \overline { { z _ { k } B } }$ and $\overline { { A x _ { k + 1 } } } \parallel \overline { { B z _ { k + 1 } } }$ ,

$$
\begin{array}{l} x _ {k + 1} - A = \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} (z _ {k + 1} - B) = \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} ((z _ {k + 1} - z _ {k}) - (B - z _ {k})) \\ = \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} \left(\phi_ {k} \left(x _ {k} ^ {+} - x _ {k}\right) - \frac {\varphi_ {k}}{\varphi_ {k} - \varphi_ {k - 1}} \left(x _ {k} ^ {+} - x _ {k}\right)\right) \\ = \frac {\varphi_ {k + 1} - \varphi_ {k}}{\varphi_ {k + 1}} \left(\phi_ {k} - \frac {\varphi_ {k}}{\varphi_ {k} - \varphi_ {k - 1}}\right) (x _ {k} ^ {+} - x _ {k}). \\ \end{array}
$$

If $x _ { k }$ is a minimizer which implies $\nabla f ( x _ { k } ) = 0$ and $z _ { k + 1 } - z _ { k } = x _ { k } ^ { + } = x _ { k } = 0$ , this is degenerate case. In this case, proof is trivial.

By Lemmas 1 and 2, there is a correspondence between the two algorithm forms.

![](images/75ff9f640b77f8a9dccaec8fe876f9c48f0735c73f2228a8e0d2a3432a283e80.jpg)

![](images/80b9dc3d4a31d3394d0fd4c89680691254de540f23a3acc111d7de52a7f9b5aa.jpg)  
Figure 6: Lemma 3 and Lemma 4

# B.3 Collinear structure from momentum-based iteration

Lemma 3. An iteration of the form

$$
x _ {k + 1} = x _ {k} ^ {+} + a _ {k} \left(x _ {k} ^ {+} - x _ {k - 1} ^ {+}\right) + b _ {k} \left(x _ {k} ^ {+} - x _ {k}\right),
$$

where $0 < a _ { k }$ and $0 \leq b _ { k }$ , for $k = 0 , 1 , \ldots$ . can be equivalently expressed as

$$
x _ {k} = (1 - \varphi_ {k}) x _ {k - 1} ^ {+} + \varphi_ {k} z _ {k}
$$

$$
z _ {k + 1} = \left(1 - \frac {a _ {k} \varphi_ {k}}{(1 - \varphi_ {k}) \varphi_ {k + 1}}\right) x _ {k} ^ {+ +} + \frac {a _ {k} \varphi_ {k}}{(1 - \varphi_ {k}) \varphi_ {k + 1}} z _ {k}
$$

for $k = 0 , 1 , \ldots$ , where $\begin{array} { r } { \varphi _ { k + 1 } = \left( a _ { k } + b _ { k } \right) \cdot \frac { \mu } { L - \mu } + \frac { a _ { k } \varphi _ { k } } { 1 - \varphi _ { k } } \cdot \frac { L } { L - \mu } } \end{array}$ , provided that $1 > \varphi _ { k } > 0$ for $k = 0 , 1 , \ldots$

k−1From first iteration of algorithm with auxiliary iterates, we know x+k−1, xk, zk are collinear. We inductively Proof. Suppose $x _ { k }$ is not a minimizer which implies $\nabla f ( x _ { k } ) \neq 0$ $x _ { k - 1 } ^ { + } , x _ { k } , z _ { k }$ and neither $x _ { k - 1 } ^ { + }$ , xk, zk are not the same. $z _ { k }$ $Q : = \overleftrightarrow { x _ { k - 1 } ^ { + } x _ { k } } \cap \overleftrightarrow { x _ { k } ^ { + } x _ { k + 1 } }$ $z _ { k + 1 } : = \stackrel { \setminus } { x } _ { k } ^ { + } x _ { k + 1 } \stackrel { \prime } { \cap } \stackrel { \setminus } { x } _ { k } ^ { + + } z _ { k } ^ { ' }$ ← → → $A$ $\overleftrightarrow { x _ { k - 1 } ^ { + } x _ { k } ^ { + } }$ $\overline { { A x _ { k + 1 } } }$ is parallel to $\overline { { x _ { k } x _ { k } ^ { + } } }$ . Set $R : = \dot { x } _ { k - 1 } ^ { + } z _ { k } \cap \overline { { A x _ { k + 1 } } }$ . Set $P$ on the $\stackrel { \cdot } { x _ { k - 1 } ^ { + } } { z _ { k } }$ that $\overline { { P z _ { k + 1 } } }$ is parallel to $\overline { { x _ { k } x _ { k } ^ { + } } }$ .

By parallel lemma,

$$
\frac {\| x _ {k} ^ {+} - x _ {k} \|}{\| A - R \|} = \frac {\| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \|}{\| A - x _ {k - 1} ^ {+} \|} = \frac {1}{1 + a _ {k}}
$$

and

$$
\frac {\left\| x _ {k} ^ {+} - x _ {k} \right\|}{\left\| x _ {k + 1} - R \right\|} = \frac {\left\| x _ {k} ^ {+} - x _ {k} \right\|}{\left\| x _ {k + 1} - A \right\| + \left\| A - R \right\|} = \frac {1}{1 + a _ {k} + b _ {k}}.
$$

Then, we have

$$
\frac {\| x _ {k} - Q \|}{\| R - x _ {k} \|} = \frac {\| x _ {k} ^ {+} - x _ {k} \|}{\| R - x _ {k + 1} \| - \| x _ {k} ^ {+} - x _ {k} \|} = \frac {1}{a _ {k} + b _ {k}}
$$

and

$$
\frac {\| x _ {k} - x _ {k - 1} ^ {+} \|}{\| x _ {k} - Q \|} = \frac {\| x _ {k} - x _ {k - 1} ^ {+} \|}{\| R - x _ {k} \|} \cdot \frac {\| R - x _ {k} \|}{\| x _ {k} - Q \|} = \frac {\| x _ {k} - x _ {k - 1} ^ {+} \|}{\| R - x _ {k} \|} \cdot \frac {\| A - x _ {k} ^ {+} \|}{\| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \|} = \frac {a _ {k} + b _ {k}}{a _ {k}}.
$$

Also parallel lemma implies

$$
\frac {\left\| R - x _ {k} \right\|}{\left\| P - R \right\|} = \frac {\left\| x _ {k + 1} - x _ {k} ^ {+} \right\|}{\left\| z _ {k + 1} - x _ {k + 1} \right\|} = \frac {\varphi_ {k + 1}}{1 - \varphi_ {k + 1}}.
$$

Applying Menelaus’s lemma to $\triangle Q x _ { k } x _ { k } ^ { + }$ and $\overline { { z _ { k } z _ { k + 1 } x _ { k } ^ { + + } } }$

$$
\frac {\left\| z _ {k + 1} - x _ {k} ^ {+} \right\|}{\left\| z _ {k + 1} - Q \right\|} \cdot \frac {\left\| x _ {k} ^ {+ +} - x _ {k} \right\|}{\left\| x _ {k} ^ {+ +} - x _ {k} ^ {+} \right\|} \cdot \frac {\left\| z _ {k} - Q \right\|}{\left\| z _ {k} - x _ {k} \right\|} = 1.
$$

Using $\begin{array} { r } { \frac { \| z _ { k + 1 } - x _ { k } ^ { + } \| } { \| z _ { k + 1 } - Q \| } = \frac { \| P - x _ { k } \| } { \| P - Q \| } } \end{array}$ , $\begin{array} { r } { \frac { \| x _ { k } ^ { + + } - z _ { k + 1 } \| } { \| z _ { k + 1 } - z _ { k } \| } = \frac { \| z _ { k } - P \| } { \| P - x _ { k } \| } } \end{array}$ , and previous formula, we get

$$
\varphi_ {k + 1} = (a _ {k} + b _ {k}) \cdot \frac {\mu}{L - \mu} + \frac {a _ {k} \varphi_ {k}}{1 - \varphi_ {k}} \cdot \frac {L}{L - \mu}.
$$

Furthermore, parallel lemma and $\begin{array} { r } { \frac { \| z _ { k } - x _ { k } \| } { \| x _ { k } - x _ { k - 1 } ^ { + } \| } = \frac { 1 - \varphi _ { k } } { \varphi _ { k } } } \end{array}$ 1−ϕk implies

$$
\frac {\| x _ {k} - x _ {k - 1} ^ {+} \|}{\| R - x _ {k} \|} = \frac {\| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \|}{\| A - x _ {k} ^ {+} \|} = \frac {1}{a _ {k}}
$$

and

$$
\frac {\| P - R \|}{\| z _ {k} - P \|} = \frac {a _ {k} \frac {1 - \varphi_ {k + 1}}{\varphi_ {k + 1}}}{\frac {1 - \varphi_ {k}}{\varphi_ {k}} - a _ {k} \left(\frac {1 - \varphi_ {k + 1}}{\varphi_ {k + 1}} + 1\right)}.
$$

Therefore, we have

$$
\frac {\| x _ {k} ^ {+ +} - z _ {k + 1} \|}{\| z _ {k + 1} - z _ {k} \|} = \frac {\| P - x _ {k} \|}{\| z _ {k} - P \|} = \frac {\frac {a _ {k}}{\varphi_ {k + 1}}}{\frac {1 - \varphi_ {k}}{\varphi_ {k}} - \frac {a _ {k}}{\varphi_ {k + 1}}} = \frac {a _ {k} \varphi_ {k}}{(1 - \varphi_ {k}) \varphi_ {k + 1} - a _ {k} \varphi_ {k}}.
$$

If $x _ { k }$ is a minimizer which implies $\nabla f ( x _ { k } ) = 0$ and $z _ { k + 1 } = z _ { k } = x _ { k } ^ { + + }$ , this is degenerate case. In this case, proof is trivial.

Lemma 4. An iteration of the form

$$
x _ {k} = (1 - \varphi_ {k}) x _ {k - 1} ^ {+} + \varphi_ {k} z _ {k}
$$

$$
z _ {k + 1} = \left(1 - \phi_ {k}\right) x _ {k} ^ {+ +} + \phi_ {k} z _ {k},
$$

where $0 < \varphi _ { k }$ and $0 < \phi _ { k }$ , for $k = 0 , 1 , \ldots$ . can be equivalently expressed as

$$
x _ {k + 1} = \frac {(1 - \varphi_ {k}) \varphi_ {k + 1} \phi_ {k}}{\varphi_ {k}} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) + \frac {\varphi_ {k + 1} ((\kappa - 1) (1 - \phi_ {k}) \varphi_ {k} - \phi_ {k})}{\varphi_ {k}} (x _ {k} ^ {+} - x _ {k})
$$

for $k = 0 , 1 , \ldots$ .

Proof. Suppose $x _ { k }$ is not a minimizer which implies $\nabla f ( x _ { k } ) \neq 0 .$ . Set $A$ on the $\stackrel {  } { x _ { k - 1 } ^ { + } x _ { k } ^ { + } }$ that $\overline { { A x _ { k + 1 } } } \parallel \overline { { x _ { k } x _ { k } ^ { + } } }$ x k x +k . Set $P$ on the $\overleftrightarrow { x _ { k - 1 } ^ { + } z _ { k } }$ that $\overline { { P z _ { k + 1 } } } \parallel \overline { { x _ { k } x _ { k } ^ { + } } }$ . Set $N : = \overrightarrow { x _ { k } ^ { + } A } \cap \overleftrightarrow { P z _ { k + 1 } }$ . Lastly, set $R : = \overleftrightarrow { x _ { k - 1 } ^ { + } z _ { k } } \cap \overleftrightarrow { A x _ { k + 1 } }$ → .

By parallel lemma, we have

$$
\frac {\| P - x _ {k} \|}{\| z _ {k} - P \|} = \frac {\| x _ {k} ^ {+ +} - z _ {k + 1} \|}{\| z _ {k + 1} - z _ {k} \|} = \frac {\phi_ {k}}{1 - \phi_ {k}}
$$

and

$$
\frac {\left\| R - x _ {k} \right\|}{\left\| P - R \right\|} = \frac {\left\| x _ {k + 1} - x _ {k} ^ {+} \right\|}{\left\| z _ {k + 1} - x _ {k + 1} \right\|} = \frac {\varphi_ {k + 1}}{1 - \varphi_ {k + 1}}.
$$

Also $\begin{array} { r } { \frac { \| x _ { k } - x _ { k - 1 } ^ { + } \| } { \| z _ { k } - x _ { k } \| } = \frac { \varphi _ { k } } { 1 - \varphi _ { k } } } \end{array}$ and previous formula implies 1−ϕk

$$
\frac {\left\| x _ {k} - x _ {k - 1} ^ {+} \right\|}{\left\| P - x _ {k} \right\|} = \frac {\varphi_ {k}}{\phi_ {k} (1 - \varphi_ {k})}
$$

and

$$
\frac {\| x _ {k} - x _ {k - 1} ^ {+} \|}{\| R - x _ {k} \|} = \frac {\varphi_ {k}}{(1 - \varphi_ {k}) \varphi_ {k + 1} \phi_ {k}}
$$

Furthermore, we get

$$
A - x _ {k} ^ {+} = \frac {\| A - x _ {k} ^ {+} \|}{\| x _ {k} ^ {+} - x _ {k - 1} ^ {+} \|} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) = \frac {\| R - x _ {k} \|}{\| x _ {k} - x _ {k - 1} ^ {+} \|} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}) = \frac {(1 - \varphi_ {k}) \varphi_ {k + 1} \phi_ {k}}{\varphi_ {k}} (x _ {k} ^ {+} - x _ {k - 1} ^ {+}).
$$

By parallel lemma, we have

$$
\frac {\| x _ {k} ^ {+ +} - x _ {k} \|}{\| z _ {k + 1} - P \|} = \frac {\| x _ {k} ^ {+ +} - z _ {k} \|}{\| z _ {k + 1} - z _ {k} \|} = \frac {1}{1 - \phi_ {k}}
$$

and

$$
\frac {\| N - P \|}{\| x _ {k} ^ {+} - x _ {k} \|} = \frac {\| P - x _ {k - 1} ^ {+} \|}{\| x _ {k} - x _ {k - 1} ^ {+} \|} = \frac {\varphi_ {k} + (1 - \varphi_ {k}) \phi_ {k}}{\varphi_ {k}}.
$$

Using $\begin{array} { r } { \frac { \| \boldsymbol { x } _ { k } ^ { + } - \boldsymbol { x } _ { k } \| } { \| \boldsymbol { x } _ { k } ^ { + + } - \boldsymbol { x } _ { k } \| } = \frac { 1 } { \kappa } } \end{array}$ kx+k κ ,

$$
\frac {\left\| z _ {k + 1} - P \right\|}{\left\| x _ {k} ^ {+} - x _ {k} \right\|} = \kappa (1 - \phi_ {k})
$$

and previous formula implies

$$
\frac {\| z _ {k + 1} - N \|}{\| x _ {k} ^ {+} - x _ {k} \|} = \frac {\| z _ {k + 1} - P \| - \| N - P \|}{\| x _ {k} ^ {+} - x _ {k} \|} = \frac {(\kappa - 1) (1 - \phi_ {k}) \varphi_ {k} - \phi_ {k}}{\varphi_ {k}}.
$$

Finally, we get

$$
x _ {k + 1} - A = \frac {\| z _ {k + 1} - N \|}{\| x _ {k} ^ {+} - x _ {k} \|} \frac {\| x _ {k + 1} - A \|}{\| z _ {k + 1} - N \|} (x _ {k} ^ {+} - x _ {k}) = \frac {\varphi_ {k + 1} ((\kappa - 1) (1 - \phi_ {k}) \varphi_ {k} - \phi_ {k})}{\varphi_ {k}} (x _ {k} ^ {+} - x _ {k}).
$$

If $x _ { k }$ is a minimizer which implies $\nabla f ( x _ { k } ) = 0$ and $z _ { k + 1 } = z _ { k } = x _ { k } ^ { + + }$ , this is degenerate case. In this case, proof is trivial.

By Lemmas 3 and 4, there is a correspondence between the two algorithm forms.

# C OGM-G analysis

Using Lemma 1, we can write OGM-G [47] as

$$
x _ {k} = \frac {\theta_ {k + 1} ^ {4}}{\theta_ {k} ^ {4}} x _ {k - 1} ^ {+} + \left(1 - \frac {\theta_ {k + 1} ^ {4}}{\theta_ {k} ^ {4}}\right) z _ {k}
$$

$$
z _ {k + 1} = z _ {k} - \frac {\theta_ {k}}{L} \nabla f (x _ {k}),
$$

where $z _ { 0 } = x _ { 0 }$ and $\begin{array} { r } { z _ { 1 } = z _ { 0 } - \frac { \theta _ { 0 } + 1 } { 2 L } \nabla f ( x _ { 0 } ) } \end{array}$ for $k = 1 , 2 , \dots K$

Theorem 5. Consider (P) with $g = 0$ . OGM-G’s $x _ { K }$ exhibits the rate

$$
\| \nabla f (x _ {K}) \| ^ {2} \leq \frac {2 L}{\theta_ {0} ^ {2}} (f (x _ {0}) - f _ {\star}) \leq \frac {4 L}{(K + 1) ^ {2}} (f (x _ {0}) - f _ {\star}).
$$

Proof. For $k = 1 , 2 , \dots , K$ , define

$$
\begin{array}{l} U _ {k} = \frac {1}{\theta_ {k} ^ {2}} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2} + f (x _ {k}) - f (x _ {K}) - \left\langle \nabla f (x _ {k}), x _ {k} - x _ {k - 1} ^ {+} \right\rangle\right) \\ + \frac {L}{\theta_ {k} ^ {4}} \left\langle z _ {k} - x _ {k - 1} ^ {+}, z _ {k} - x _ {K} ^ {+} \right\rangle \\ \end{array}
$$

and

$$
U _ {0} = \frac {2}{\theta_ {0} ^ {2}} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + f (x _ {0}) - f (x _ {K})\right).
$$

We can show that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. Using $\begin{array} { r } { \frac { 1 } { 2 L } \| \nabla f ( x _ { K } ) \| ^ { 2 } \leq f ( x _ { K } ) - f ( x _ { K } ^ { + } ) \leq f ( x _ { K } ) - f _ { \star } , } \end{array}$ which follows from $L$ -smoothness, we conclude with the rate

$$
\frac {1}{L} \| \nabla f (x _ {K}) \| ^ {2} = U _ {K} \leq U _ {0} \leq \frac {2}{\theta_ {0} ^ {2}} \left(f (x _ {0}) - f _ {\star}\right)
$$

and the bound $\theta _ { 0 } \geq \frac { \kappa + 1 } { \sqrt { 2 } }$ [47, Theorem 6.1]. Now, we complete the proof by showing that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. As we already showed $U _ { 1 } \geq U _ { 2 } \geq \cdot \cdot \cdot \geq U _ { K }$ in Section 3.2, all that remains is to show $U _ { 0 } \geq U _ { 1 }$ :

$$
\begin{array}{l} U _ {0} - U _ {1} \\ = - \frac {1}{\theta_ {1} ^ {2}} f \left(x _ {1}\right) + \frac {2}{\theta_ {0} ^ {2}} f \left(x _ {0}\right) + \left(\frac {1}{\theta_ {1} ^ {2}} - \frac {2}{\theta_ {0} ^ {2}}\right) f \left(x _ {K}\right) - \frac {1}{\theta_ {1} ^ {2}} \frac {1}{2 L} \| \nabla f \left(x _ {1}\right) \| ^ {2} - \left(\frac {1}{\theta_ {1} ^ {2}} - \frac {2}{\theta_ {0} ^ {2}}\right) \frac {1}{2 L} \| \nabla f \left(x _ {K}\right) \| ^ {2} \\ + \frac {1}{\theta_ {1} ^ {2}} \left\langle \nabla f (x _ {1}), x _ {1} - x _ {0} ^ {+} \right\rangle - \frac {L}{\theta_ {1} ^ {4}} \left\langle z _ {1} - x _ {0} ^ {+}, z _ {1} - x _ {K} ^ {+} \right\rangle \\ = - \frac {1}{\theta_ {1} ^ {2}} \left(f (x _ {1}) - f (x _ {0}) - \left\langle \nabla f (x _ {1}), x _ {1} - x _ {0} ^ {+} \right\rangle + \frac {1}{2 L} \| \nabla f (x _ {1}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {0}) \| ^ {2}\right) \\ \left. - \left(\frac {1}{\theta_ {1} ^ {2}} - \frac {2}{\theta_ {0} ^ {2}}\right) \left(f (x _ {0}) - f (x _ {K}) - \left\langle \nabla f (x _ {0}), x _ {0} - x _ {K} ^ {+} \right\rangle + \frac {1}{2 L} \| \nabla f (x _ {0}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2}\right) \right. \\ + \left(\frac {1}{\theta_ {1} ^ {2}} - \frac {1}{\theta_ {0} ^ {2}}\right) \frac {1}{L} \| \nabla f (x _ {0}) \| ^ {2} - \left(\frac {1}{\theta_ {1} ^ {2}} - \frac {2}{\theta_ {0} ^ {2}}\right) \left\langle \nabla f (x _ {0}), x _ {0} - x _ {K} ^ {+} \right\rangle - \frac {L}{\theta_ {1} ^ {4}} \left\langle z _ {1} - x _ {0} ^ {+}, z _ {1} - x _ {K} ^ {+} \right\rangle \\ \geq \left(\frac {1}{\theta_ {1} ^ {2}} - \frac {1}{\theta_ {0} ^ {2}}\right) \frac {1}{L} \| \nabla f (x _ {0}) \| ^ {2} - \left(\frac {1}{\theta_ {1} ^ {2}} - \frac {2}{\theta_ {0} ^ {2}}\right) \left\langle \nabla f (x _ {0}), x _ {0} - x _ {K} ^ {+} \right\rangle - \frac {L}{\theta_ {1} ^ {4}} \left\langle z _ {1} - x _ {0} ^ {+}, z _ {1} - x _ {K} ^ {+} \right\rangle \\ = \frac {\theta_ {0} + 1}{\theta_ {0} ^ {2} (\theta_ {0} - 1)} \frac {1}{L} \| \nabla f (x _ {0}) \| ^ {2} - \frac {1}{\theta_ {1} ^ {2} \theta_ {0}} \left\langle \nabla f (x _ {0}), x _ {0} - x _ {K} ^ {+} \right\rangle - \frac {L}{\theta_ {1} ^ {4}} \left\langle z _ {1} - x _ {0} ^ {+}, z _ {1} - x _ {K} ^ {+} \right\rangle \\ = 0 \\ \end{array}
$$

where the inequality follows from the cocoercivity inequalities.

# D Several preliminary inequalities

Lemma 5 ([60, (2.1.11)]). If $f \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } }$ is convex and $L$ -smooth, then

$$
f (x) - f (y) + \langle \nabla f (x), y - x \rangle + \frac {1}{2 L} \| \nabla f (x) - \nabla f (y) \| ^ {2} \leq 0 \quad \forall x, y \in \mathbb {R} ^ {n},
$$

$$
f (y) \leq f (x) + \langle \nabla f (x), y - x \rangle + \frac {L}{2} \| x - y \| ^ {2} \quad \forall x, y \in \mathbb {R} ^ {n}.
$$

Lemma 6. If $g \colon \mathbb { R } ^ { n }  \mathbb { R } \cup \{ \infty \}$ is µ-strongly convex, then for all $u \in \partial g ( x )$ ,

$$
g (x) + \langle u, y - x \rangle + \frac {\mu}{2} \| x - y \| ^ {2} \leq g (y) \quad \forall x, y \in \mathbb {R} ^ {n}.
$$

Lemma 7 ([11, lemma 2.2]). Consider (P) in the prox-grad setup. Then for some $u \in \partial g ( x ^ { \oplus } )$ ,

$$
\tilde {\nabla} _ {L} F (x) = \nabla f (x) + u \quad \forall x \in \mathbb {R} ^ {n}.
$$

Proof. Optimality condition for strongly convex function implies that there exist $u \in \partial g ( x ^ { \oplus } )$ such that $\nabla f ( x ) + { \dot { u } } + L ( { \dot { x } } ^ { \oplus } - x ) = 0$ .

Lemma 8 ([45, (2.8)]). Consider (P) in the prox-grad setup. Then for some $v \in \partial { \cal F } ( x ^ { \oplus } )$ ,

$$
\left\| v \right\| \leq 2 \left\| \tilde {\nabla} _ {L} F (x) \right\| \quad \forall x \in \mathbb {R} ^ {n}.
$$

Proof. By Lemma 7, $\tilde { \nabla } _ { L } F ( x ) = \nabla f ( x ) + u$ for some $u \in \partial g ( x ^ { \oplus } )$ . And there exist $v \in \partial { \cal F } ( x ^ { \oplus } )$ such that $v = \mathrm { \nabla } f ( \dot { \boldsymbol { x } } ^ { \oplus } ) + u$ . Thus we have

$$
\begin{array}{l} \| v \| \leq \| \nabla f (x ^ {\oplus}) - \nabla f (x) \| + \| \nabla f (x) + u \| (1) \\ \leq \| L (x - x ^ {\oplus}) \| + \| \tilde {\nabla} _ {L} F (x) \| (2) \\ = 2 \left\| \tilde {\nabla} _ {L} F (x) \right\|. (3) \\ \end{array}
$$

(1) follows from triangle inequality and (2) follows from $L$ -smootheness of $f$ , and (3) follows from the definition of $\tilde { \nabla } _ { L } F ( x )$ . □

Lemma 9 ([59, Theorem 1]). Consider (P) in the prox-grad setup. Then

$$
\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x) \right\| ^ {2} \leq F (x) - F (x ^ {\oplus}) \quad \forall x \in \mathbb {R} ^ {n}.
$$

Proof. By Lemma 7, for some $u \in \partial g ( x ^ { \oplus } )$ , we have

$$
\begin{array}{l} F \left(x ^ {\oplus}\right) \leq f (x) + \left\langle \nabla f (x), x ^ {\oplus} - x \right\rangle + \frac {L}{2} \left\| x ^ {\oplus} - x \right\| ^ {2} + g \left(x ^ {\oplus}\right) (4) \\ \leq f (x) + \left\langle L \left(x - x ^ {\oplus}\right) - u, x ^ {\oplus} - x \right\rangle + \frac {L}{2} \| x ^ {\oplus} - x \| ^ {2} + g \left(x ^ {\oplus}\right) (5) \\ = f (x) + g \left(x ^ {\oplus}\right) + \left\langle u, x - x ^ {\oplus} \right\rangle - \frac {L}{2} \left\| x ^ {\oplus} - x \right\| ^ {2} \\ \leq F (x) - \frac {L}{2} \| x ^ {\oplus} - x \| ^ {2} = F (x) - \frac {1}{2 L} \| \tilde {\nabla} _ {L} F (x) \| ^ {2}. (6) \\ \end{array}
$$

(4) follows from $L$ -smootheness of $f$ , (5) follows from the definition of $\tilde { \nabla } _ { L } F ( x )$ , and (6) follows from convexity of $g$ .

Lemma 10 ([11, lemma 2.3]). Consider (P) in the prox-grad setup. Then

$$
\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (y) \right\| ^ {2} - \left\langle y - x, \tilde {\nabla} _ {L} F (y) \right\rangle \leq F (x) - F (y ^ {\oplus}) \quad \forall x, y \in \mathbb {R} ^ {n}.
$$

Proof. By $L$ -smootheness of $f$ , we have

$$
F (y ^ {\oplus}) \leq f (y) + \left\langle \nabla f (y), y ^ {\oplus} - y \right\rangle + \frac {L}{2} \left\| y ^ {\oplus} - y \right\| ^ {2} + g (y ^ {\oplus}).
$$

Using convexity of $f$ and $g$ and Lemma 7, for some $u \in \partial g ( y ^ { \oplus } )$ , we have

$$
f (y) + \langle \nabla f (y), x - y \rangle \leq f (x)
$$

$$
g \left(y ^ {\oplus}\right) + \left\langle u, x - y ^ {\oplus} \right\rangle \leq g (x).
$$

Summing above three inequality, we obtain the lemma.

![](images/3e27545ed02765cc008eda47f0011b3d4b455ebaab16129a1034d2a67cf5e7c1.jpg)

# E Omitted proofs of Section 2

We present (G-FISTA-G) for the prox-grad setup:

$$
x _ {k} = \frac {\varphi_ {k + 1}}{\varphi_ {k}} x _ {k - 1} ^ {\oplus} + \left(1 - \frac {\varphi_ {k + 1}}{\varphi_ {k}}\right) z _ {k}
$$

$$
z _ {k + 1} = z _ {k} - \frac {\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}}{L} \tilde {\nabla} _ {L} F (x _ {k})
$$

for $k = 0 , 1 , \ldots , K$ , where $z _ { 0 } = x _ { 0 }$ , $L$ is smoothness constant of $f$ , and the nonnegative sequence $\left\{ \varphi _ { k } \right\} _ { k = 0 } ^ { K + 1 }$ and the nondecreasing nonnegative sequence $\left\{ \tau _ { k } \right\} _ { k = 0 } ^ { K }$ satisfy $\varphi _ { K + 1 } = 0$ , $\varphi _ { K } = \tau _ { K } = 1$ , and

$$
\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} (\tau_ {k + 1} - \tau_ {k}) + 1, \quad \left(\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}\right) \left(\tau_ {k + 1} - \tau_ {k}\right) \leq \frac {\tau_ {k + 1}}{2}
$$

for $k = 0 , 1 , \ldots , K - 1$ .

Theorem 6. Consider (P). G-FISTA-G’s $x _ { K }$ exhibits the rate

$$
\left\| \tilde {\nabla} _ {L} F (x _ {K}) \right\| ^ {2} \leq 2 L \tau_ {0} \left(F (x _ {0}) - F _ {\star}\right).
$$

Proof. For $k = 0 , 1 , \ldots , K$ , define

$$
\begin{array}{l} U _ {k} = \tau_ {k} \left(\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {k}) \right\| ^ {2} + F \left(x _ {k} ^ {\oplus}\right) - F \left(x _ {K} ^ {\oplus}\right) - \left\langle \tilde {\nabla} _ {L} F (x _ {k}), x _ {k} - x _ {k - 1} ^ {\oplus} \right\rangle\right) \\ + \frac {L}{\varphi_ {k}} \left<   z _ {k} - x _ {k - 1} ^ {\oplus}, z _ {k} - x _ {K} ^ {\oplus} \right > . \\ \end{array}
$$

(Note that $z _ { K } = x _ { K }$ .) By plugging in the definitions and performing direct calculations, we get

$$
U _ {K} = \frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {k}) \right\| ^ {2} \quad \text {a n d} \quad U _ {0} = \tau_ {0} \left(\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {0}) \right\| ^ {2} + F (x _ {0} ^ {\oplus}) - F (x _ {K} ^ {\oplus})\right).
$$

We can show that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. Using Lemma 9, we conclude the rate with

$$
\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {k}) \right\| ^ {2} = U _ {K} \leq U _ {0} \leq \tau_ {0} \left(F (x _ {0}) - F (x _ {K} ^ {\oplus})\right) \leq \tau_ {0} \left(F (x _ {0}) - F _ {\star}\right).
$$

Now we complete the proof by showing that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. For $k = 0 , 1 , \ldots , K - 1$ , we have

$$
\begin{array}{l} 0 \geq \tau_ {k + 1} \left(F \left(x _ {k + 1} ^ {\oplus}\right) - F \left(x _ {k} ^ {\oplus}\right) - \left\langle \tilde {\nabla} _ {L} F \left(x _ {k + 1}\right), x _ {k + 1} - x _ {k} ^ {\oplus} \right\rangle + \frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F \left(x _ {k + 1}\right) \right\| ^ {2}\right) \\ + \left(\tau_ {k + 1} - \tau_ {k}\right) \left(F \left(x _ {k} ^ {\oplus}\right) - F \left(x _ {K} ^ {\oplus}\right) - \left\langle \tilde {\nabla} _ {L} F (x _ {k}), x _ {k} - x _ {K} ^ {\oplus} \right\rangle + \frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {k}) \right\| ^ {2}\right) \\ = \tau_ {k + 1} \left(\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {k + 1}) \right\| ^ {2} + F \left(x _ {k + 1} ^ {\oplus}\right) - F \left(x _ {K} ^ {\oplus}\right) - \left\langle \tilde {\nabla} _ {L} F (x _ {k + 1}), x _ {k + 1} - x _ {k} ^ {\oplus} \right\rangle\right) \\ \left. - \tau_ {k} \left(\frac {1}{2 L} \left\| \tilde {\nabla} _ {L} F (x _ {k}) \right\| ^ {2} + F (x _ {k} ^ {\oplus}) - F (x _ {K} ^ {\oplus}) - \left\langle \tilde {\nabla} _ {L} F (x _ {k}), x _ {k} - x _ {k - 1} ^ {\oplus} \right\rangle\right) \right. \\ \underbrace{-\left\langle\tilde{\nabla}_{L}F(x_{k}),\tau_{k + 1}x_{k}^{\oplus} - \tau_{k}x_{k - 1}^{\oplus} - (\tau_{k + 1} - \tau_{k})x_{K}^{\oplus}\right\rangle - \tau_{k + 1}\frac{1}{2L}\left\| \tilde{\nabla}_{L}F(x_{k})\right\|^{2}}_{:= T}, \\ \end{array}
$$

where the inequality follows from the Lemma 10. Finally, we analyze $T$ with the following geometric argument. Let $t \in \mathbb { R } ^ { n }$ be the projection of $x _ { K } ^ { \oplus }$ onto the plane of iteration. Then,

![](images/c653aaba187b41819257d457b6d7762140d80d7457173ef852729a44ea9a813d.jpg)  
Figure 7: Plane of iteration of G-FISTA-G

$$
\frac {1}{L} T \stackrel {\text {(i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\oplus}}, (\tau_ {k + 1} - \tau_ {k}) \overrightarrow {t x _ {k} ^ {\oplus}} + \tau_ {k} \overrightarrow {x _ {k - 1} ^ {\oplus} x _ {k} ^ {\oplus}} - \frac {\tau_ {k + 1}}{2} \overrightarrow {x _ {k} x _ {k} ^ {\oplus}} \right\rangle
$$

$$
\begin{array}{l} \stackrel {\text {(i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\oplus}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \left(\overrightarrow {t z _ {k + 1}} - \overrightarrow {z _ {k} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {\oplus}}\right) + \tau_ {k} \left(\overrightarrow {x _ {k - 1} ^ {\oplus} x _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {\oplus}}\right) - \frac {\tau_ {k + 1}}{2} \overrightarrow {x _ {k} x _ {k} ^ {\oplus}} \right\rangle \\ \stackrel {\text {(i i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\oplus}}, (\tau_ {k + 1} - \tau_ {k}) \overrightarrow {t z _ {k + 1}} - (\tau_ {k + 1} - \tau_ {k}) (\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} - 1) \overrightarrow {x _ {k} x _ {k} ^ {\oplus}} \right. \\ + \tau_ {k} \overrightarrow {x _ {k} x _ {k} ^ {\oplus}} - \frac {\tau_ {k + 1}}{2} \overrightarrow {x _ {k} x _ {k} ^ {\oplus}} - (\tau_ {k + 1} - \tau_ {k}) \overrightarrow {x _ {k} z _ {k} ^ {\prime}} + \tau_ {k} \left(\frac {\varphi_ {k}}{\varphi_ {k + 1}} - 1\right) \overrightarrow {x _ {k} z _ {k} ^ {\prime}} \Bigg \rangle \\ \end{array}
$$

$$
\begin{array}{l} \stackrel {\text {(i v)}} {\geq} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\oplus}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \overrightarrow {t z _ {k + 1}} + \frac {\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}}{\varphi_ {k + 1}} \overrightarrow {x _ {k} z _ {k}} \right\rangle \\ \stackrel {\text {(v)}} {=} \frac {1}{\varphi_ {k + 1}} \left\langle \overrightarrow {x _ {k} ^ {\oplus} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}}, \overrightarrow {t z _ {k + 1}} \right\rangle + \frac {1}{\varphi_ {k + 1}} \left\langle \overrightarrow {t z _ {k + 1}} - \overrightarrow {t z _ {k}}, \overrightarrow {x _ {k} z _ {k}} \right\rangle \\ \stackrel {(\mathrm {v i})} {=} \frac {1}{\varphi_ {k + 1}} \left\langle z _ {k + 1} - x _ {k} ^ {\oplus}, z _ {k + 1} - x _ {K} ^ {\oplus} \right\rangle - \frac {1}{\varphi_ {k}} \left\langle z _ {k} - x _ {k - 1} ^ {\oplus}, z _ {k} - x _ {K} ^ {\oplus} \right\rangle \\ \end{array}
$$

where (i) follows from the definition of $t$ and the fact that we can replace $x _ { K } ^ { \oplus }$ with $t$ , the projection of $x _ { K }$ onto the plane of iteration, without affecting the inner products, (ii) from vector addition, (iii) from the fact that xkx⊕k $\xrightarrow [ { x _ { k } x _ { k } ^ { \oplus } } ] { }$ and $\overrightarrow { z _ { k } z _ { k + 1 } }$ are parallel and their lengths satisfy $( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } ) \overrightarrow { x _ { k } x _ { k } ^ { \oplus } } = \overrightarrow { z _ { k } z _ { k + 1 } }$ and $\overrightarrow { x _ { k - 1 } ^ { \oplus } x _ { k } }$ and $\overrightarrow { x _ { k } z _ { k } }$ are parallel and their lengths satisfy $\begin{array} { r } { \left( \frac { \varphi _ { k } } { \varphi _ { k + 1 } } - 1 \right) \overrightarrow { x _ { k } z _ { k } } = \overrightarrow { x _ { k - 1 } ^ { \oplus } x _ { k } } } \end{array}$ ϕk+1 , (iv) from vector addition and

$$
\frac {\tau_ {k + 1}}{2} - \left(\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}\right) \left(\tau_ {k + 1} - \tau_ {k}\right) \geq 0, \tag {7}
$$

(v) from distributing the product and substituting $\overline { { x _ { k } x _ { k } ^ { \oplus } } } = \left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } - 1 \right) ^ { - 1 } \left( \overrightarrow { x _ { k } ^ { \oplus } z _ { k + 1 } } - \overrightarrow { x _ { k } z _ { k } } \right) =$ (ϕk+1(τk+1 − τk))−1  $\left( \varphi _ { k + 1 } ( \tau _ { k + 1 } - \tau _ { k } ) \right) ^ { - 1 } \left( { \overrightarrow { x _ { k } ^ { \oplus } z _ { k + 1 } } } - { \overrightarrow { x _ { k } z _ { k } } } \right)$ into the first term and $\overline { { x _ { k } x _ { k } ^ { \oplus } } } = \left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } \right) ^ { - 1 } \overrightarrow { z _ { k } z _ { k + 1 } } =$ $( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } ) ^ { - 1 } \left( \overrightarrow { t z _ { k + 1 } } - \overrightarrow { t z _ { k } } \right)$ into the second term, and (vi) from cancelling out the cross terms, using $\varphi _ { k } ^ { - 1 } { \overrightarrow { x _ { k - 1 } ^ { \oplus } z _ { k } } } = \varphi _ { k + 1 } ^ { - 1 } { \overrightarrow { x _ { k } z _ { k } } }$ , and by replacing $t$ with $x _ { K } ^ { \oplus }$ in the inner products. In (v), we also used

$$
\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} \left(\tau_ {k + 1} - \tau_ {k}\right) + 1. \tag {8}
$$

Thus we conclude $U _ { k + 1 } \leq U _ { k }$ for $k = 0 , 1 , 2 , \ldots , K - 1$ .

Proof of Theorem 1. The conclusion of Theorem 1 follows from plugging FISTA-G’s $\varphi _ { k }$ and $\tau _ { k }$ into Theorem 6. If τk = $\begin{array} { r } { \tau _ { k } = \frac { 2 \varphi _ { k - 1 } } { ( \varphi _ { k - 1 } - \varphi _ { k } ) ^ { 2 } } } \end{array}$ (ϕk−1−ϕk)2 , we can check condition (8), condition (7), and 2ϕk−1

$$
\varphi_ {k} \tau_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} (\tau_ {k + 1} - \tau_ {k}) + 1 = \frac {\varphi_ {k}}{\varphi_ {k} - \varphi_ {k + 1}}.
$$

Then using Lemma 2, we get the iteration of the form of FISTA-G. Furthermore, by Lemma 8, $\| v \| \leq$ $2 \left\| \tilde { \nabla } _ { L } F ( x ) \right\|$ for some $v \in \partial { \cal F } ( x ^ { \oplus } )$ . Thus we have

$$
\min  \left\| \partial F (x _ {K} ^ {\oplus}) \right\| ^ {2} \leq 4 \left\| \tilde {\nabla} _ {L} F (x _ {K}) \right\| ^ {2} \leq \frac {2 6 4 L}{(K + 2) ^ {2}} \left(F (x _ {0}) - F _ {\star}\right).
$$

Finally, it remains to show $\begin{array} { r } { \tau _ { 0 } \leq \frac { 3 3 } { ( K + 2 ) ^ { 2 } } } \end{array}$

First, $\tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } = \varphi _ { k + 1 } \left( \tau _ { k + 1 } - \tau _ { k } \right) + 1$ and $\begin{array} { r } { ( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } ) ( \tau _ { k + 1 } - \tau _ { k } ) - \frac { \tau _ { k + 1 } } { 2 } = 0 } \end{array}$ implies

$$
\varphi_ {k + 1} = \frac {1}{\tau_ {k + 1} - \tau_ {k}} \left(\frac {\tau_ {k + 1}}{2 (\tau_ {k + 1} - \tau_ {k})} - 1\right).
$$

By substitution and direct calculation, we get

$$
a _ {k + 1} = a _ {k} + \frac {\left(a _ {k} - a _ {k - 1}\right) a _ {k}}{\sqrt {a _ {k} ^ {2} - a _ {k} a _ {k - 1} + a _ {k - 1} ^ {2}}},
$$

where $\begin{array} { r } { \tau _ { k } = \frac { 1 } { a _ { K - k } } } \end{array}$ . This is equivalent to

$$
\frac {a _ {k + 1}}{a _ {k}} = 1 + \frac {(a _ {k} - a _ {k - 1})}{\sqrt {(a _ {k} - a _ {k - 1}) ^ {2} + a _ {k} a _ {k - 1}}} \iff \frac {a _ {k + 1}}{a _ {k}} = 1 + \frac {1}{\sqrt {1 + \frac {\frac {1}{a _ {k}}}{\frac {a _ {k}}{a _ {k - 1}} + \frac {a _ {k - 1}}{a _ {k}}} - 2}}.
$$

Let $\begin{array} { r } { b _ { k } = \frac { a _ { k + 1 } } { a _ { k } } } \end{array}$ . Then, $b _ { 0 } = 1 + { \frac { 1 } { \sqrt { 3 } } }$ by $\tau _ { K } = \varphi _ { K } = 1$ , and

$$
b _ {k} = 1 + \frac {1}{\sqrt {1 + \frac {1}{\left(\sqrt {b _ {k - 1}} - \sqrt {\frac {1}{b _ {k - 1}}}\right) ^ {2}}}} \iff \frac {1}{(b _ {k} - 1) ^ {2}} = 1 + \frac {1}{(b _ {k - 1} - 1)} + \frac {1}{(b _ {k - 1} - 1) ^ {2}}.
$$

Let $\begin{array} { r } { c _ { k } = \frac { 1 } { b _ { k } - 1 } } \end{array}$ . Then

$$
c _ {k} ^ {2} = c _ {k - 1} ^ {2} + c _ {k - 1} + 1
$$

where $c _ { 0 } = { \sqrt { 3 } }$ . Also, by definition,

$$
a _ {k + 1} = b _ {k} b _ {k - 1} \dots b _ {0} = \left(1 + \frac {1}{c _ {k}}\right) \left(1 + \frac {1}{c _ {k - 1}}\right) \dots \left(1 + \frac {1}{c _ {0}}\right).
$$

Using $\begin{array} { r } { c _ { k } ^ { 2 } = c _ { k - 1 } ^ { 2 } + c _ { k - 1 } + 1 \iff \frac { c _ { k } ^ { 2 } - 1 } { c _ { k - 1 } ^ { 2 } } = 1 + \frac { 1 } { c _ { k - 1 } } } \end{array}$ , we have

$$
\begin{array}{l} \left(\frac {c _ {k} + 1}{c _ {k}}\right) \left(\frac {c _ {k - 1} + 1}{c _ {k - 1}}\right) \dots \left(\frac {c _ {0} + 1}{c _ {0}}\right) = \frac {c _ {k + 1} ^ {2} - 1}{c _ {k} ^ {2}} \frac {c _ {k} ^ {2} - 1}{c _ {k - 1} ^ {2}} \dots \frac {c _ {1} ^ {2} - 1}{c _ {0} ^ {2}} \\ = \left(\frac {c _ {k + 1} + 1}{c _ {k}}\right) \left(\frac {c _ {k} + 1}{c _ {k - 1}}\right) \ldots \left(\frac {c _ {1} + 1}{c _ {0}}\right) \left(\frac {c _ {k + 1} - 1}{c _ {k}}\right) \left(\frac {c _ {k} - 1}{c _ {k - 1}}\right) \ldots \left(\frac {c _ {1} - 1}{c _ {0}}\right). \\ \end{array}
$$

And after reduction of fraction, we get

$$
\begin{array}{l} \left(\frac {c _ {k + 1} + 1}{c _ {0} + 1}\right) \left(\frac {c _ {k + 1} - 1}{c _ {k}}\right) \left(\frac {c _ {k} - 1}{c _ {k - 1}}\right) \dots \left(\frac {c _ {1} - 1}{c _ {0}}\right) = 1 \\ \Longleftrightarrow c _ {k + 1} ^ {2} - 1 = (c _ {0} ^ {2} - 1) \left(\frac {c _ {k}}{c _ {k} - 1}\right) \left(\frac {c _ {k - 1}}{c _ {k - 1} - 1}\right) \ldots \left(\frac {c _ {0}}{c _ {0} - 1}\right). \\ \end{array}
$$

$\begin{array} { r } { \frac { c _ { k - 2 } + 1 } { c _ { k } } \geq \frac { c _ { k - 2 } } { c _ { k } - 1 } \iff c _ { k } \geq c _ { k - 2 } + 1 } \end{array}$ since $\begin{array} { r } { c _ { k } ^ { 2 } = ( c _ { k - 1 } + \frac { 1 } { 2 } ) ^ { 2 } + \frac { 3 } { 4 } } \end{array}$ implies $\begin{array} { r } { c _ { k } \geq c _ { k - 1 } + \frac { 1 } { 2 } } \end{array}$ . Therefore,

$$
\left(\frac {c _ {k} + 1}{c _ {k}}\right) \left(\frac {c _ {k - 1} + 1}{c _ {k - 1}}\right) \dots \left(\frac {c _ {0} + 1}{c _ {0}}\right) \geq \frac {(c _ {k} + 1) (c _ {k - 1} + 1)}{c _ {k} c _ {k - 1}} \frac {(c _ {1} - 1) (c _ {0} - 1)}{c _ {1} c _ {0} (c _ {0} ^ {2} - 1)} (c _ {k + 1} ^ {2} - 1).
$$

Furthermore, we can show $c _ { k } \geq \frac { k + 3 } { 2 }$ by induction. $\begin{array} { r } { c _ { 0 } = \sqrt { 3 } \geq \frac { 3 } { 2 } } \end{array}$ and if $c _ { k } \geq \frac { k + 3 } { 2 }$ , $\begin{array} { r } { c _ { k + 1 } \geq c _ { k } + \frac { 1 } { 2 } \geq \frac { k + 4 } { 2 } } \end{array}$ k+4 .) Finally

$$
a _ {k + 1} \geq \frac {(c _ {1} - 1) (c _ {0} - 1)}{c _ {1} c _ {0} (c _ {0} ^ {2} - 1)} (c _ {k + 1} ^ {2} - 1) \geq \frac {1}{3 3} (k + 3) ^ {2}.
$$

For $k = K - 1$ , $\begin{array} { r } { \frac { 1 } { a _ { K } } = \tau _ { 0 } } \end{array}$ and we get wanted result.

![](images/7fa92c2ae8934a195dfb4c0d85a8b89e7e6c90c6e57b7b58a1ebfe665da397e6.jpg)

# F Omitted proofs of Section 3

We present (G-FGM-G) for the smooth convex setup

$$
\begin{array}{l} x _ {k} = \frac {\varphi_ {k + 1}}{\varphi_ {k}} x _ {k - 1} ^ {+} + \left(1 - \frac {\varphi_ {k + 1}}{\varphi_ {k}}\right) z _ {k} \\ z _ {k + 1} = z _ {k} - \frac {\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}}{L} \nabla f (x _ {k}) \\ \end{array}
$$

for $k = 0 , 1 , \ldots , K$ where $z _ { 0 } = x _ { 0 }$ , $L$ is smootheness constant of $f$ , and the nonnegative sequence $\{ \varphi _ { k } \} _ { k = 0 } ^ { K + 1 }$ and the nondecreasing nonnegative sequence $\left\{ \tau _ { k } \right\} _ { k = 0 } ^ { K }$ satisfy $\varphi _ { K + 1 } = 0$ , $\varphi _ { K } = \tau _ { K } = 1$ , and

$$
\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} (\tau_ {k + 1} - \tau_ {k}) + 1, \quad \left(\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}\right) \left(\tau_ {k + 1} - \tau_ {k}\right) \leq \tau_ {k + 1}.
$$

for $k = 0 , 1 , \ldots , K - 1$ .

Note that G-FISTA-G had the parameter requirement $\leq \frac { \tau _ { k + 1 } } { 2 }$ while G-FGM-G (and later G-Güler G) has $\leq \tau _ { k + 1 }$ . The parameter requirements are otherwise identical.

Theorem 7. Consider (P) with $g = 0$ . G-FGM-G’s $x _ { K }$ exhibits the rate

$$
\left\| \nabla f (x _ {K}) \right\| ^ {2} \leq 2 L \tau_ {0} (f (x _ {0}) - f _ {\star}).
$$

Proof. For $k = 0 , 1 , \ldots , K$ , define

$$
\begin{array}{l} U _ {k} = \tau_ {k} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2} + f (x _ {k}) - f (x _ {K}) - \langle \nabla f (x _ {k}), x _ {k} - x _ {k - 1} ^ {+} \rangle\right) \\ + \frac {L}{\varphi_ {k}} \left<   z _ {k} - x _ {k - 1} ^ {+}, z _ {k} - x _ {K} ^ {+} \right > . \\ \end{array}
$$

(Note that $z _ { K } = x _ { K }$ .) By plugging in the definitions and performing direct calculations, we get

$$
U _ {K} = \frac {1}{L} \| \nabla f (x _ {k}) \| ^ {2} \qquad \text {a n d} \qquad U _ {0} = \tau_ {0} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {0}) \| ^ {2} + f (x _ {0}) - f (x _ {K})\right).
$$

We can show that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. Using $\begin{array} { r } { \frac { 1 } { 2 L } \| \nabla f ( x _ { K } ) \| ^ { 2 } \leq f ( x _ { K } ) - f ( x _ { K } ^ { + } ) \leq f ( x _ { K } ) - f ( x _ { \star } ) } \end{array}$ and $\begin{array} { r } { \frac { 1 } { 2 L } \| \nabla f ( x _ { 0 } ) \| ^ { 2 } \leq f ( x _ { 0 } ) - f ( x _ { 0 } ^ { + } ) \leq f ( x _ { 0 } ) - f _ { \star } } \end{array}$ , which follows from $L$ -smoothness, we conclude the rate with

$$
\frac {1}{L} \| \nabla f (x _ {K}) \| ^ {2} = U _ {K} \leq U _ {0} \leq 2 \tau_ {0} \left(f (x _ {0}) - f _ {\star}\right).
$$

Now we complete the proof by showing that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. For $k = 0 , 1 , \ldots K - 1$ , we have

$$
\begin{array}{l} 0 \geq \tau_ {k + 1} \left(f (x _ {k + 1}) - f (x _ {k}) - \langle \nabla f (x _ {k + 1}), x _ {k + 1} - x _ {k} \rangle + \frac {1}{2 L} \left\| \nabla f (x _ {k + 1}) - \nabla f (x _ {k}) \right\| ^ {2}\right) \\ + \left(\tau_ {k + 1} - \tau_ {k}\right) \left(f (x _ {k}) - f (x _ {K}) - \langle \nabla f (x _ {k}), x _ {k} - x _ {K} \rangle + \frac {1}{2 L} \left\| \nabla f (x _ {k}) - \nabla f (x _ {K}) \right\| ^ {2}\right) \\ = \tau_ {k + 1} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {k + 1}) \| ^ {2} + f (x _ {k + 1}) - f (x _ {K}) - \left\langle \nabla f (x _ {k + 1}), x _ {k + 1} - x _ {k} ^ {+} \right\rangle\right) \\ \left. - \tau_ {k} \left(\frac {1}{2 L} \| \nabla f (x _ {K}) \| ^ {2} + \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2} + f (x _ {k}) - f (x _ {K}) - \left\langle \nabla f (x _ {k}), x _ {k} - x _ {k - 1} ^ {+} \right\rangle\right) \right. \\ \underbrace {- \left\langle \nabla f (x _ {k}) , \tau_ {k + 1} x _ {k} ^ {+} - \tau_ {k} x _ {k - 1} ^ {+} - (\tau_ {k + 1} - \tau_ {k})   x _ {K} ^ {+} \right\rangle} _ {:= T}, \\ \end{array}
$$

where the inequality follows from the cocoercivity inequalities. Finally, we analyze $T$ with the following geometric argument. Let $t \in \mathbb { R } ^ { n }$ be the projection of $x _ { K } ^ { + }$ onto the plane of iteration. Then,

![](images/152267fb1861fa4a00abf24f6c391c960942f1715c95b4caaeb44ff6d9ef218b.jpg)  
Figure 8: Plane of iteration of G-FGM-G

$$
\begin{array}{l} \frac {1}{L} T \stackrel {\mathrm {(i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\uparrow}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \overrightarrow {t x _ {k} ^ {\uparrow}} + \tau_ {k} \overrightarrow {x _ {k - 1} ^ {+} x _ {k} ^ {\uparrow}} \right\rangle \\ \stackrel {\text {(i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {+}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \left(\overrightarrow {t z _ {k + 1}} - \overrightarrow {z _ {k} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {+}}\right) + \tau_ {k} \left(\overrightarrow {x _ {k - 1} ^ {+} x _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {+}}\right) \right\rangle \\ \stackrel {\mathrm {(i i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {+}}, (\tau_ {k + 1} - \tau_ {k}) \overrightarrow {t z _ {k + 1}} - (\tau_ {k + 1} - \tau_ {k}) (\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} - 1) \overrightarrow {x _ {k} x _ {k} ^ {+}} \right. \\ + \tau_ {k} \overrightarrow {x _ {k} x _ {k} ^ {+}} - (\tau_ {k + 1} - \tau_ {k}) \overrightarrow {x _ {k} z _ {k} ^ {\prime}} + \tau_ {k} \left(\frac {\varphi_ {k}}{\varphi_ {k + 1}} - 1\right) \overrightarrow {x _ {k} z _ {k} ^ {\prime}} \Bigg \rangle \\ \geq \stackrel {\text {(i v)}} {\left\langle \overrightarrow {x _ {k} x _ {k} ^ {+}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \overrightarrow {t z _ {k + 1}} + \frac {\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}}{\varphi_ {k + 1}} \overrightarrow {x _ {k} z _ {k}} \right\rangle} \\ \stackrel {\text {(v)}} {=} \frac {1}{\varphi_ {k + 1}} \left\langle \overrightarrow {x _ {k} ^ {+} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}}, \overrightarrow {t z _ {k + 1}} \right\rangle + \frac {1}{\varphi_ {k + 1}} \left\langle \overrightarrow {t z _ {k + 1}} - \overrightarrow {t z _ {k}}, \overrightarrow {x _ {k} z _ {k}} \right\rangle \\ \stackrel {(\mathrm {v i})} {=} \frac {1}{\varphi_ {k + 1}} \left\langle z _ {k + 1} - x _ {k} ^ {+}, z _ {k + 1} - x _ {K} ^ {+} \right\rangle - \frac {1}{\varphi_ {k}} \left\langle z _ {k} - x _ {k - 1} ^ {+}, z _ {k} - x _ {K} ^ {+} \right\rangle , \\ \end{array}
$$

where (i) follows from the definition of $t$ and the fact that we can replace $x _ { K } ^ { + }$ with $t$ , the projection of $x _ { K }$ onto the plane of iteration, without affecting the inner products, (ii) from vector addition, (iii) from the fact that $\xrightarrow [ { x _ { k } x _ { k } ^ { + } } ] { }$ and $\overrightarrow { z _ { k } z _ { k + 1 } }$ are parallel and their lengths satisfy $( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } ) \overrightarrow { x _ { k } x _ { k } ^ { + } } = \overrightarrow { z _ { k } z _ { k + 1 } }$ and $\overrightarrow { x _ { k - 1 } ^ { + } x _ { k } }$ and $\overrightarrow { x _ { k } z _ { k } }$ are parallel and their lengths satisfy $\begin{array} { r } { \left( \frac { \varphi _ { k } } { \varphi _ { k + 1 } } - 1 \right) \overrightarrow { x _ { k } z _ { k } } = \overrightarrow { x _ { k - 1 } ^ { + } x _ { k } } } \end{array}$ ϕk+1 , (iv) from vector addition and

$$
\tau_ {k + 1} - \left(\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}\right) \left(\tau_ {k + 1} - \tau_ {k}\right) \geq 0, \tag {9}
$$

(v) from distributing the product and substituting $\overline { { x _ { k } x _ { k } ^ { + } } } = \left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } - 1 \right) ^ { - 1 } \left( \overrightarrow { x _ { k } ^ { + } z _ { k + 1 } } - \overrightarrow { x _ { k } z _ { k } } \right) =$ $\left( \varphi _ { k + 1 } ( \tau _ { k + 1 } - \tau _ { k } ) \right) ^ { - 1 } \left( \overrightarrow { x _ { k } ^ { + } z _ { k + 1 } } - \overrightarrow { x _ { k } z _ { k } } \right)$ into the first term and $x _ { k } x _ { k } ^ { + } = \left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } \right) ^ { - 1 } { \overrightarrow { z _ { k } z _ { k + 1 } } } =$ $\left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } \right) ^ { - 1 } \left( \overrightarrow { t z _ { k + 1 } } - \overrightarrow { t z _ { k } } \right)$ into the second term, and (vi) from cancelling out the cross terms, using $\varphi _ { k } ^ { - 1 } { \overrightarrow { x _ { k - 1 } ^ { + } z _ { k } } } = \varphi _ { k + 1 } ^ { - 1 } { \overrightarrow { x _ { k } z _ { k } } }$ , and by replacing $t$ with $x _ { K } ^ { + }$ in the inner products.In (v), we also used

$$
\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} \left(\tau_ {k + 1} - \tau_ {k}\right) + 1. \tag {10}
$$

Thus we conclude $U _ { k + 1 } \leq U _ { k }$ for $k = 0 , 1 , \ldots , K - 1$ .

Theorem 8. Consider (P) with $g = 0$ . FGM-G’s $x _ { K }$ exhibits the rate

$$
\left\| \nabla f (x _ {K}) \right\| ^ {2} \leq \frac {6 6 L}{(K + 2) ^ {2}} \left(f (x _ {0}) - f _ {\star}\right).
$$

Proof. This follows from plugging FGM-G’s $\varphi _ { k }$ and (ϕk−1−ϕk)2 into Theorem 7’s ϕk and τk. We can check $\frac { 2 \varphi _ { k - 1 } } { ( \varphi _ { k - 1 } - \varphi _ { k } ) ^ { 2 } }$ 2ϕk−1 $\varphi _ { k }$ $\tau _ { k }$ condition (9), condition (10), and

$$
\varphi_ {k} \tau_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} \left(\tau_ {k + 1} - \tau_ {k}\right) + 1 = \frac {\varphi_ {k}}{\varphi_ {k} - \varphi_ {k + 1}}.
$$

Then using Lemma 2, we get the iteration of the form of FGM-G

![](images/71dc987d0083ac59692613369934dfcfc3f131b842fd125d3f24bbe67fdac86d.jpg)

We present (G-Güler-G) for the proximal-point setup:

$$
x _ {k} = \frac {\varphi_ {k + 1}}{\varphi_ {k}} x _ {k - 1} ^ {\circ} + \left(1 - \frac {\varphi_ {k + 1}}{\varphi_ {k}}\right) z _ {k}
$$

$$
z _ {k + 1} = z _ {k} - \left(\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}\right) \lambda \tilde {\nabla} _ {1 / \lambda} g (x _ {k})
$$

for $k = 0 , 1 , \ldots , K$ where $z _ { 0 } = x _ { 0 }$ and the nonnegative sequence $\{ \varphi _ { k } \} _ { k = 0 } ^ { K + 1 }$ and the nondecreasing nonnegative sequence $\left\{ \tau _ { k } \right\} _ { k = 0 } ^ { K }$ satisfy $\varphi _ { K + 1 } = 0$ , $\varphi _ { K } = \tau _ { K } = 1$ , and

$$
\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} \left(\tau_ {k + 1} - \tau_ {k}\right) + 1, \quad \left(\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}\right) \left(\tau_ {k + 1} - \tau_ {k}\right) \leq \tau_ {k + 1}.
$$

for $k = 0 , 1 , \ldots , K - 1$

Theorem 9. Consdier (P) with $f = 0$ . $G$ -Güler- $G$ ’s $x _ { K }$ exhibits the rate

$$
\left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {K}) \right\| ^ {2} \leq \frac {\tau_ {0}}{\lambda} \left(g (x _ {0}) - g _ {\star}\right)
$$

Proof. For $k = 0 , 1 , \ldots , K$ , define

$$
U _ {k} = \tau_ {k} \left(\lambda \left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) \right\| ^ {2} + g (x _ {k} ^ {\circ}) - g (x _ {K} ^ {\circ}) - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) \cdot (x _ {k} - x _ {k - 1} ^ {\circ})\right) + \frac {1}{\lambda \varphi_ {k}} \left\langle z _ {k} - x _ {k - 1} ^ {\circ}, z _ {k} - x _ {K} ^ {\circ} \right\rangle .
$$

(Note that $z _ { K } = x _ { K }$ .) By plugging in the definitions and performing direct calculations, we get

$$
U _ {K} = \lambda \left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {K}) \right\| ^ {2} \quad \text {a n d} \quad U _ {0} = \tau_ {0} \left(\lambda \left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {0}) \right\| ^ {2} + g (x _ {0} ^ {\circ}) - g (x _ {K} ^ {\circ})\right).
$$

We can show that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. Using $\lambda \left\| \tilde { \nabla } _ { 1 / \lambda } g ( x _ { 0 } ) \right\| ^ { 2 } \leq g ( x _ { 0 } ) - g ( x _ { 0 } ^ { \circ } )$ , we conclude the rate with

$$
\lambda \left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {K}) \right\| ^ {2} = U _ {K} \leq U _ {0} \leq \tau_ {0} (g (x _ {0}) - g (x _ {K} ^ {\circ})) \leq \tau_ {0} (g (x _ {0}) - g _ {\star}).
$$

Now we complete the proof by showing that $\{ U _ { k } \} _ { k = 0 } ^ { K }$ is nonincreasing. For $k = 0 , 1 , \ldots , K - 1$ , we have

$$
\begin{array}{l} 0 \geq \tau_ {k + 1} \left(g \left(x _ {k + 1} ^ {\circ}\right) - g \left(x _ {k} ^ {\circ}\right) - \left\langle \tilde {\nabla} _ {1 / \lambda} g \left(x _ {k + 1}\right), x _ {k + 1} - x _ {k} ^ {\circ} \right\rangle + \lambda \left\| \tilde {\nabla} _ {1 / \lambda} g \left(x _ {k + 1}\right) \right\| ^ {2}\right) \\ \left. + \left(\tau_ {k + 1} - \tau_ {k}\right) \left(g \left(x _ {k} ^ {\circ}\right) - g \left(x _ {K} ^ {\circ}\right) - \left\langle \tilde {\nabla} _ {1 / \lambda} g (x _ {k}), x _ {k} - x _ {K} ^ {\circ} \right\rangle + \lambda \left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) \right\| ^ {2}\right) \right. \\ = \tau_ {k + 1} \left(\lambda \left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {k + 1}) \right\| ^ {2} + g \left(x _ {k + 1} ^ {\circ}\right) - g \left(x _ {K} ^ {\circ}\right) - \tilde {\nabla} _ {1 / \lambda} g \left(x _ {k + 1} ^ {\circ}\right) \cdot \left(x _ {k + 1} - x _ {k} ^ {\circ}\right)\right) \\ - \tau_ {k} \left(\lambda \left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) \right\| ^ {2} + g \left(x _ {k} ^ {\circ}\right) - g \left(x _ {K} ^ {\circ}\right) - \tilde {\nabla} _ {1 / \lambda} g \left(x _ {k} ^ {\circ}\right) \cdot \left(x _ {k} - x _ {k - 1} ^ {\circ}\right)\right) \\ \underbrace {- \left\langle \tilde {\nabla} _ {1 / \lambda} g (x _ {k} ^ {\circ}) , \tau_ {k + 1} x _ {k} ^ {\circ} - \tau_ {k} x _ {k - 1} ^ {\circ} - (\tau_ {k + 1} - \tau_ {k})   x _ {K} ^ {\circ} \right\rangle} _ {:= T}, \\ \end{array}
$$

![](images/ec473d081e585f7a049cd192438c89bfe938cec89dd5cd7368d3c74bdb3d3e81.jpg)  
Figure 9: Plane of iteration of G-Güler-G

where the inequality follows from the convexity inequalities. Finally, we analyze $T$ with the following geometric argument. Let $t \in \mathbb { R } ^ { n }$ be the projection of $x _ { K } ^ { \circ }$ onto the plane of iteration. Then

$$
\begin{array}{l} \frac {1}{L} T \stackrel {\text {(i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\diamond}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \overrightarrow {t x _ {k} ^ {\diamond}} + \tau_ {k} \overrightarrow {x _ {k - 1} ^ {\circ} x _ {k} ^ {\diamond}} \right\rangle \\ \stackrel {\text {(i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\diamond}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \left(\overrightarrow {t z _ {k + 1}} - \overrightarrow {z _ {k} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {\diamond}}\right) + \tau_ {k} \left(\overrightarrow {x _ {k - 1} ^ {\circ} x _ {k}} + \overrightarrow {x _ {k} x _ {k} ^ {\diamond}}\right) \right\rangle \\ \stackrel {\text {(i i i)}} {=} \left\langle \overrightarrow {x _ {k} x _ {k} ^ {\diamond}}, (\tau_ {k + 1} - \tau_ {k}) \overrightarrow {t z _ {k + 1}} - (\tau_ {k + 1} - \tau_ {k}) (\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} - 1) \overrightarrow {x _ {k} x _ {k} ^ {\diamond}} \right. \\ + \tau_ {k} \overrightarrow {x _ {k} x _ {k} ^ {\prime}} - (\tau_ {k + 1} - \tau_ {k}) \overrightarrow {x _ {k} z _ {k}} + \tau_ {k} \left(\frac {\varphi_ {k}}{\varphi_ {k + 1}} - 1\right) \overrightarrow {x _ {k} z _ {k}} \Bigg \rangle \\ \geq \stackrel {\text {(i v)}} {\left\langle \overrightarrow {x _ {k} x _ {k} ^ {\diamond}}, \left(\tau_ {k + 1} - \tau_ {k}\right) \overrightarrow {t z _ {k + 1}} + \frac {\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}}{\varphi_ {k + 1}} \overrightarrow {x _ {k} z _ {k}} \right\rangle} \\ \stackrel {(\mathrm {v})} {=} \frac {1}{\varphi_ {k + 1}} \left\langle \overrightarrow {x _ {k} ^ {\circ} z _ {k + 1}} - \overrightarrow {x _ {k} z _ {k}}, \overrightarrow {t z _ {k + 1}} \right\rangle + \frac {1}{\varphi_ {k + 1}} \left\langle \overrightarrow {t z _ {k + 1}} - \overrightarrow {t z _ {k}}, \overrightarrow {x _ {k} z _ {k}} \right\rangle \\ \stackrel {(\mathrm {v i})} {=} \frac {1}{\varphi_ {k + 1}} \left\langle z _ {k + 1} - x _ {k} ^ {\circ}, z _ {k + 1} - x _ {K} ^ {\circ} \right\rangle - \frac {1}{\varphi_ {k}} \left\langle z _ {k} - x _ {k - 1} ^ {\circ}, z _ {k} - x _ {K} ^ {\circ} \right\rangle , \\ \end{array}
$$

where (i) follows from the definition of $t$ and the fact that we can replace $x _ { K } ^ { \circ }$ with $t$ , the projection of $x _ { K }$ onto the plane of iteration, without affecting the inner products, (ii) from vector addition, (iii) from the fact that $\xrightarrow [ x _ { k } x _ { k } ^ { \delta } ] { }$ and $\overrightarrow { z _ { k } z _ { k + 1 } }$ are parallel and their lengths satisfy $( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } ) \overrightarrow { x _ { k } x _ { k } ^ { \delta } } = \overrightarrow { z _ { k } z _ { k + 1 } }$ and $\xrightarrow [ x _ { k - 1 } ^ { \circ } x _ { k }$ and $\overrightarrow { x _ { k } z _ { k } }$ are parallel and their lengths satisfy $\begin{array} { r } { \left( \frac { \varphi _ { k } } { \varphi _ { k + 1 } } - 1 \right) \overrightarrow { x _ { k } z _ { k } } = \overrightarrow { x _ { k - 1 } ^ { \circ } x _ { k } } } \end{array}$ ϕk+1 , (iv) from vector addition and

$$
\tau_ {k + 1} - \left(\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1}\right) \left(\tau_ {k + 1} - \tau_ {k}\right) \geq 0, \tag {11}
$$

(v) from distributing the product and substituting $\overline { { x _ { k } x _ { k } ^ { \circ } } } = \left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } - 1 \right) ^ { - 1 } \left( \overrightarrow { x _ { k } ^ { \circ } z _ { k + 1 } } - \overrightarrow { x _ { k } z _ { k } } \right) =$ $\left( \varphi _ { k + 1 } ( \tau _ { k + 1 } - \tau _ { k } ) \right) ^ { - 1 } \left( \overrightarrow { x _ { k } ^ { \circ } z _ { k + 1 } } - \overrightarrow { x _ { k } z _ { k } } \right)$ into the first term and $\overline { { x _ { k } x _ { k } ^ { \circ } } } = \left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } \right) ^ { - 1 } \overrightarrow { z _ { k } z _ { k + 1 } } =$ $\left( \tau _ { k } \varphi _ { k } - \tau _ { k + 1 } \varphi _ { k + 1 } \right) ^ { - 1 } \left( \overrightarrow { t z _ { k + 1 } } - \overrightarrow { t z _ { k } } \right)$ into the second term, and (vi) from cancelling out the cross terms, using $\varphi _ { k } ^ { - 1 } { \overrightarrow { x _ { k - 1 } ^ { \circ } z _ { k } } } = \varphi _ { k + 1 } ^ { - 1 } { \overrightarrow { x _ { k } z _ { k } } }$ , and by replacing $t$ with $x _ { K } ^ { + }$ in the inner products.

$$
\tau_ {k} \varphi_ {k} - \tau_ {k + 1} \varphi_ {k + 1} = \varphi_ {k + 1} \left(\tau_ {k + 1} - \tau_ {k}\right) + 1. \tag {12}
$$

Thus we conclude $U _ { k + 1 } \leq U _ { k }$ for $k = 0 , 1 , \ldots , K - 1$ .

Proof of Theorem 2. The conclusion of Theorem 2 follows from plugging Güler-G’s $\varphi _ { k }$ and $\tau _ { k }$ into Theorem 9. If $\tau _ { k } = \theta _ { k } ^ { - 2 }$ and $\varphi _ { k } = \theta _ { k } ^ { 4 }$ , we can check conditions (11) and (12). Then using Lemma 2, we get the iteration of the form of Güler-G. Combining the $g ( x _ { K } ^ { \circ } ) - g _ { \star } \leq \| x _ { 0 } - x _ { \star } \| ^ { 2 } / \lambda ( K + 2 ) ^ { 2 }$ rate of Güler’s second method [37, Theorem 6.1] with rate of Güler-G, we get the rate of Güler+Güler-G:

$$
\left\| \tilde {\nabla} _ {1 / \lambda} g (x _ {2 K}) \right\| ^ {2} \leq \frac {4}{\lambda (K + 2) ^ {2}} \big (g (x _ {K} ^ {\circ}) - g _ {\star} \big) \leq \frac {4}{\lambda^ {2} (K + 2) ^ {4}} \| x _ {0} - x _ {\star} \| ^ {2}.
$$

# G Omitted proofs of Section 4

Proof of Theorem 3. In the setup of Theorem 3, define

$$
U _ {k} = g \left(x _ {k - 1} ^ {\circ}\right) - g _ {\star} - \frac {\mu}{2} \left\| x _ {k - 1} ^ {\circ} - x _ {\star} \right\| ^ {2} + \mu \left\| z _ {k} - x _ {\star} \right\| ^ {2}
$$

for $k = 0 , 1 , \ldots$ By plugging in the definitions and performing direct calculations, we get

$$
U _ {0} = g (x _ {0}) - g _ {\star} + \frac {\mu}{2} \| x _ {0} - x _ {\star} \| ^ {2}.
$$

We can show that $\begin{array} { r } { U _ { k + 1 } \le \left( 1 - \frac { 1 } { \sqrt { q } } \right) ^ { 2 } U _ { k } } \end{array}$ for $k = - 1 , 0 , \ldots .$ Using strong convexity, we conclude the rate with

$$
\mu \| z _ {k} - x _ {\star} \| ^ {2} \leq U _ {k} \leq U _ {0} \leq 2 (g (x _ {0}) - g _ {\star}).
$$

Now we complete the proof by showing that $\begin{array} { r } { U _ { k + 1 } \le \left( 1 - \frac { 1 } { \sqrt { q } } \right) ^ { 2 } U _ { k } } \end{array}$ for $k = - 1 , 0 , \ldots { \mathrm { F o r } } k = 0 , 1 , \ldots ,$ we have

$$
\begin{array}{l} U _ {k + 1} - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} U _ {k} \\ = \left(g \left(x _ {k} ^ {\circ}\right) - g _ {\star} - \frac {\mu}{2} \left\| x _ {k} ^ {\circ} - x _ {\star} \right\| ^ {2}\right) - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left(g \left(x _ {k - 1} ^ {\circ}\right) - g _ {\star} - \frac {\mu}{2} \left\| x _ {k - 1} ^ {\circ} - x _ {\star} \right\| ^ {2}\right) \\ + \mu \| z _ {k + 1} - x _ {\star} \| ^ {2} - \mu \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \| z _ {k} - x _ {\star} \| ^ {2}. \\ \end{array}
$$

For calculating the last term of difference, we use $( q - 1 ) x _ { k } - ( 1 - \sqrt { q } ) ^ { 2 } x _ { k - 1 } ^ { \circ } = 2 ( \sqrt { q } - 1 ) z _ { k } .$ . Since

$$
\begin{array}{l} \mu \| z _ {k + 1} - x _ {\star} \| ^ {2} = \mu \left\| \frac {1}{\sqrt {q}} \left(x _ {k} - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k})\right) + \left(1 - \frac {1}{\sqrt {q}}\right) z _ {k} - x _ {\star} \right\| ^ {2} \\ = \frac {\mu}{q} \left\| x _ {k} - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star} \right\| ^ {2} + \mu \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \| z _ {k} - x _ {\star} \| ^ {2} \\ + 2 \left(1 - \frac {1}{\sqrt {q}}\right) \frac {\mu}{\sqrt {q}} \left\langle x _ {k} - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star}, z _ {k} - x _ {\star} \right\rangle , \\ \end{array}
$$

we get

$$
\mu \| z _ {k + 1} - x _ {\star} \| ^ {2} - \mu \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \| z _ {k} - x _ {\star} \| ^ {2}
$$

$$
\begin{array}{l} = \frac {\mu}{q} \left\| x _ {k} - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star} \right\| ^ {2} \\ + \frac {\mu}{q} \left\langle x _ {k} - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star}, (q - 1) \left(x _ {k} - x _ {\star}\right) - (1 - \sqrt {q}) ^ {2} \left(x _ {k - 1} ^ {\circ} - x _ {\star}\right) \right\rangle \\ = \frac {\mu}{q} \left\langle x _ {k} - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star}, - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) + q (x _ {k} - x _ {\star}) - (1 - \sqrt {q}) ^ {2} \left(x _ {k - 1} ^ {\circ} - x _ {\star}\right) \right\rangle \\ = \frac {\mu}{q} \left\langle x _ {k} - \left(\frac {1}{\mu} + \lambda\right) \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star}, q (x _ {k} ^ {\circ} - x _ {\star}) - (1 - \sqrt {q}) ^ {2} (x _ {k - 1} ^ {\circ} - x _ {\star}) \right\rangle \\ = \mu \left\langle x _ {k} ^ {\circ} - \frac {1}{\mu} \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star}, x _ {k} ^ {\circ} - x _ {\star} \right\rangle - \mu \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left\langle x _ {k} ^ {\circ} - \frac {1}{\mu} \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - x _ {\star}, x _ {k - 1} ^ {\circ} - x _ {\star} \right\rangle \\ = \left(1 - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2}\right) \left\langle \mu x _ {k} ^ {\circ} - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - \mu x _ {\star}, x _ {k} ^ {\circ} - x _ {\star} \right\rangle + \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left\langle \mu x _ {k} ^ {\circ} - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - \mu x _ {\star}, x _ {k} ^ {\circ} - x _ {k - 1} ^ {\circ} \right\rangle . \\ \end{array}
$$

Therefore, we can write difference of $U _ { k + 1 }$ and $\begin{array} { r } { { \bigg ( } 1 - \frac { 1 } { \sqrt { q } } { \bigg ) } ^ { 2 } U _ { k } } \end{array}$ as

$$
\begin{array}{l} U _ {k + 1} - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} U _ {k} \\ = \left(g \left(x _ {k} ^ {\circ}\right) - g _ {\star} - \frac {\mu}{2} \left\| x _ {k} ^ {\circ} - x _ {\star} \right\| ^ {2}\right) - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left(g \left(x _ {k - 1} ^ {\circ}\right) - g _ {\star} - \frac {\mu}{2} \left\| x _ {k - 1} ^ {\circ} - x _ {\star} \right\| ^ {2}\right) \\ + \left(1 - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2}\right) \left\langle \mu x _ {k} ^ {\circ} - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - \mu x _ {\star}, x _ {k} ^ {\circ} - x _ {\star} \right\rangle + \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left\langle \mu x _ {k} ^ {\circ} - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - \mu x _ {\star}, x _ {k} ^ {\circ} - x _ {k - 1} ^ {\circ} \right\rangle \\ = \left(g \left(x _ {k} ^ {\circ}\right) - g _ {\star}\right) - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left(g \left(x _ {k - 1} ^ {\circ}\right) - g _ {\star} - \frac {\mu}{2} \| x _ {k - 1} ^ {\circ} - x _ {\star} \| ^ {2} + \frac {\mu}{2} \| x _ {k} ^ {\circ} - x _ {\star} \| ^ {2}\right) \\ \left. - \left(1 - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2}\right) \frac {\mu}{2} \| x _ {k} ^ {\circ} - x _ {\star} \| ^ {2} + \left(1 - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2}\right) \left\langle \mu x _ {k} ^ {\circ} - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - \mu x _ {\star}, x _ {k} ^ {\circ} - x _ {\star} \right\rangle \right. \\ + \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left\langle \mu x _ {k} ^ {\circ} - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) - \mu x _ {\star}, x _ {k} ^ {\circ} - x _ {k - 1} ^ {\circ} \right\rangle \\ = \left(g \left(x _ {k} ^ {\circ}\right) - g _ {\star}\right) - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left(g \left(x _ {k - 1} ^ {\circ}\right) - g _ {\star}\right) + \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \frac {\mu}{2} \left\langle x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ}, x _ {k - 1} ^ {\circ} + x _ {k} ^ {\circ} - 2 x _ {\star} \right\rangle \\ + \left(1 - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2}\right) \left\langle \frac {\mu}{2} \left(x _ {k} ^ {\circ} - x _ {\star}\right) - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}), x _ {k} ^ {\circ} - x _ {\star} \right\rangle \\ + \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left\langle - \mu x _ {k} ^ {\circ} + \tilde {\nabla} _ {1 / \lambda} g (x _ {k}) + \mu x _ {\star}, x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ} \right\rangle \\ = \left(g \left(x _ {k} ^ {\circ}\right) - g _ {\star}\right) - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left(g \left(x _ {k - 1} ^ {\circ}\right) - g _ {\star}\right) + \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left\langle \frac {\mu}{2} \left(x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ}\right) + \tilde {\nabla} _ {1 / \lambda} g (x _ {k}), x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ} \right\rangle \\ + \left(1 - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2}\right) \left\langle \frac {\mu}{2} \left(x _ {k} ^ {\circ} - x _ {\star}\right) - \tilde {\nabla} _ {1 / \lambda} g (x _ {k}), x _ {k} ^ {\circ} - x _ {\star} \right\rangle \\ \end{array}
$$

$$
\begin{array}{l} = - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2} \left(g \left(x _ {k - 1} ^ {\circ}\right) - g \left(x _ {k} ^ {\circ}\right) - \left\langle \tilde {\nabla} _ {1 / \lambda} g (x _ {k}), x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ} \right\rangle - \frac {\mu}{2} \| x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ} \| ^ {2}\right) \\ \left. - \left(1 - \left(1 - \frac {1}{\sqrt {q}}\right) ^ {2}\right) \left(g _ {\star} - g \left(x _ {k} ^ {\circ}\right) - \left\langle \tilde {\nabla} _ {1 / \lambda} g (x _ {k}), x _ {\star} - x _ {k} ^ {\circ} \right\rangle - \frac {\mu}{2} \| x _ {\star} - x _ {k} ^ {\circ} \| ^ {2}\right) \right. \\ \leq 0, \\ \end{array}
$$

where the inequality follows from strong convexity inequalities.

The case $\begin{array} { r } { U _ { 1 } \le \left( 1 - \frac { 1 } { \sqrt { q } } \right) ^ { 2 } U _ { 0 } } \end{array}$ follows from the same argument with $x _ { - 1 } ^ { \circ } = x _ { 0 }$ . Thus $\begin{array} { r } { U _ { k + 1 } \le \left( 1 - \frac { 1 } { \sqrt { q } } \right) ^ { 2 } U _ { k } } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \big \vert } \end{array}$ for $k = - 1 , 0 , \ldots$ .

Following proof is a close adaptation of the convergence analysis of ITEM [19, Theorem 3].

Proof of Theorem 4. In the setup of Theorem 4, define

$$
U _ {k} = A _ {k} \left(g \left(x _ {k - 1} ^ {\circ}\right) - g _ {\star} - \frac {\mu}{2} \left\| x _ {k - 1} ^ {\circ} - x _ {\star} \right\| ^ {2}\right) + \left(A _ {k} \mu + \mu + \frac {1}{\lambda}\right) \left\| z _ {k} - x _ {\star} \right\| ^ {2}
$$

for $k = 0 , 1 , \ldots$ By plugging in the definitions and performing direct calculations, we get

$$
U _ {0} = \left(\mu + \frac {1}{\lambda}\right) \| x _ {0} - x _ {\star} \| ^ {2}.
$$

We can show that $\{ U _ { k } \} _ { k = 0 } ^ { \infty }$ is nonincreasing. Using strong convexity, we conclude the rate with

$$
\left(A _ {k} \mu + \mu + \frac {1}{\lambda}\right) \| z _ {k} - x _ {\star} \| ^ {2} \leq U _ {k} \leq U _ {0} = \left(\mu + \frac {1}{\lambda}\right) \| z _ {0} - x _ {\star} \| ^ {2}.
$$

And by

$$
A _ {k} = \frac (1 + q) A _ {k - 1} + 2 (1 + \sqrt {(1 + A _ {k - 1}) (1 + q A _ {k - 1}))}{(1 - q) ^ {2}} \geq \frac {(1 + q) A _ {k - 1} + 2 \sqrt {q A _ {k - 1} ^ {2}}}{(1 - q) ^ {2}} = \frac {A _ {k - 1}}{(1 - \sqrt {q}) ^ {2}},
$$

we get theorem through direct calculation.

Now we complete the proof by showing that $\{ U _ { k } \} _ { k = 0 } ^ { \infty }$ is nonincreasing. For $k = 1 , 2 , \dots$ , we have

$$
U _ {k + 1} - U _ {k}
$$

$$
\begin{array}{l} = 4 \frac {\lambda}{1 - q} K _ {2} P \left(A _ {k + 1}, A _ {k}\right) \left\| (1 - q) A _ {k + 1} \tilde {\nabla} _ {1 / \lambda} g \left(x _ {k}\right) - \frac {\mu}{1 + \lambda \mu} A _ {k} \left(x _ {k - 1} ^ {\circ} - x _ {\star}\right) + \frac {\mu}{1 + \lambda \mu} K _ {3} \left(z _ {k} - x _ {\star}\right) \right\| ^ {2} \\ - \frac {1}{\lambda (1 - q)} K _ {1} P \left(A _ {k + 1}, A _ {k}\right) \| z _ {k} - x _ {\star} \| ^ {2} \\ + A _ {k} \left(g \left(x _ {k} ^ {\circ}\right) - g \left(x _ {k - 1} ^ {\circ}\right) + \left\langle \tilde {\nabla} _ {1 / \lambda} g (x _ {k}), x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ} \right\rangle + \frac {\mu}{2} \| x _ {k - 1} ^ {\circ} - x _ {k} ^ {\circ} \| ^ {2}\right) \\ \left. + \left(A _ {k + 1} - A _ {k}\right) \left(g \left(x _ {k} ^ {\circ}\right) - g _ {\star} + \left\langle \tilde {\nabla} _ {1 / \lambda} g \left(x _ {k}\right), x _ {\star} - x _ {k} ^ {\circ} \right\rangle + \frac {\mu}{2} \| x _ {\star} - x _ {k} ^ {\circ} \| ^ {2}\right) \right. \\ \end{array}
$$

$$
\leq 0
$$

where inequality follows from the strong convexity inequality,

$$
K _ {1} = \frac {q ^ {2}}{(1 + q) ^ {2} + (1 - q) ^ {2} q A _ {k + 1}}
$$

$$
K _ {2} = \frac {(1 + q) ^ {2} + (1 - q) ^ {2} q A _ {k + 1}}{(1 - q) ^ {2} (1 + q + q A _ {k}) ^ {2} A _ {k + 1} ^ {2}}
$$

$$
K _ {3} = (1 + q) \frac {(1 + q) A _ {k} - (1 - q) (2 + q A _ {k}) A _ {k + 1}}{(1 + q) ^ {2} + (1 - q) ^ {2} q A _ {k + 1}},
$$

$P ( x , y ) = ( y - ( 1 - q ) x ) ^ { 2 } - 4 x ( 1 + q y )$ and $P ( A _ { k + 1 } , A _ { k } ) = 0$ by condition, and equality follows from direct calculation.

The case $U _ { 1 } \leq U _ { 0 }$ follows from the same argument with $x _ { - 1 } ^ { \circ } = x _ { 0 }$ . Thus $U _ { k + 1 } \leq U _ { k }$ for $k = 0 , 1 , \ldots$ .

# H Other geometric and non-geometric views of acceleration

Geometric descent is an accelerated method designed expressly based on a geometric principle of shrinking balls for the smooth strongly convex setup [13, 16, 42]. Quadratic averaging is equivalent to geometric descent but has an interpretation of averaging quadratic lower bounds [30]. Both methods implicitly induce the collinear structure through steps equivalent to defining $z _ { k + 1 }$ as a convex combination of $z _ { k }$ and $x _ { k } ^ { + + }$ (In fact, our $x _ { k } ^ { + + }$ k notation comes from the geometric descent paper [13].) However, this line of work does not establish a rate faster than FGM or its corresponding proximal version, nor does it extend the geometric principle to the non-strongly convex setup.

The method of similar triangles (MST) is an accelerated method [32, 60, 1] with iterates forming similar triangles analogous to our illustration of FGM in Figure 1. One can also interpret acceleration as an approximate proximal point method with alternating upper and lower bounds and obtain the structure of similar triangles as a consequence [1]. The parallel structure we present generalizes the structure of similar triangles; the illustration of OGM and OGM-G in Figure 1 exhibits the parallel structure but not the similar triangles structure. To the best of our knowledge, the parallel structure we present is a geometric structure of acceleration that has not been considered, explicitly or implicitly, in prior works.

Linear coupling [4] interprets acceleration as a unification of gradient descent and mirror descent. The auxiliary iterates of our setup are referred to as the mirror descent iterates in the linear coupling viewpoint. However, the primary motivation of linear coupling is to unify gradient descent, which reduces the function value much when the gradient is large, with mirror descent, which reduces the function value much when the gradient is small. This motivation does not seem to be applicable to the problem setup of minimizing gradient magnitudes, the setup of OGM-G and FISTA-G.

The scaled relative graph (SRG) is another geometric framework for analyzing optimization algorithms; it establishes a correspondence between algebraic operations on nonlinear operators with geometric operations on subsets of the 2D plane [66, 40, 41, 68]. The SRG demonstrated that geometry can serve as a powerful tool for the analysis of optimization algorithms. However, there is no direct connection as the SRG has not been used to analyze accelerated optimization algorithms.

# I Experiment

For scientific reproducibility, we include code for generating the synthetic data of the experiments. We furthermore clarify that since FPGM-m, FISTA, and FISTA+FISTA-G are not anytime algorithms (i.e., since the total iteration count $K$ must be known in advance), the points in the plot of Figure 4 were generated with

a separate iteration. In other words, the plots for ISTA and FISTA were generated each with a single for-loop, while the plots for FPGM-m, FISTA, and FISTA+FISTA-G were generated with nested double for-loops.

import numpy as np   
np.random.seed(419)   
#11 norm problem data   
m, n, k = 60, 100, 20 # dimensions   
lamb $= 0.1$ # lasso penalty constant   
L = 324   
x_true = np.zeros(n)   
x_true[:k] = np.random.randint(k)   
np.randomshuffle(x_true)   
[U,_] = np.linalg.qr(np.random.randint(m,m))   
[V,_] = np.linalg.qr(np.random.randint(n,n))   
Sigma = np.zeros((m,n))   
np fillDIagonal (Sigma,np.abs(np.random.randint(m)))   
np fillDIagonal (Sigma[m-3:m,m-3:m],np.sqrt(L))   
A = U @ Sigma @ V.T   
b = A@x_true + 0.01 * np.random.randint(m)   
#nuclear problem data   
m, n, k = 60, 20, 20 # dimensions   
lamb $= 0.1$ # nuclear norm penalty constant   
L = 400   
n2 = int(n*(n+1)/2)   
x_true = np.zeros(n2)   
x_true[:k] = np.random.randint(k)   
np.randomshuffle(x_true)   
[U,_] = np.linalg.qr(np.random.randint(m,m))   
[V,_] = np.linalg.qr(np.random.randint(n2,n2))   
Sigma = np.zeros((m,n2))   
np fillDIagonal (Sigma,np.abs(np.random.randint(m)))   
np fillDIagonal (Sigma[m-3:m,m-3:m],np.sqrt(L))   
A = U @ Sigma @ V.T   
b = A@x_true + 0.01 * np.random.randint(m)