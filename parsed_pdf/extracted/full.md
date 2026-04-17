# Oneshot Differentially Private Top- $k$ Selection

Gang Qiao 1 Weijie J. Su 2 Li Zhang 3

# Abstract

Being able to efficiently and accurately select the top- $k$ elements with differential privacy is an integral component of various private data analysis tasks. In this paper, we present the oneshot Laplace mechanism, which generalizes the well-known Report Noisy Max (Dwork & Roth, 2014) mechanism to reporting noisy top- $k$ elements. We show that the oneshot Laplace mechanism with a noise level of $\tilde { O } ( \sqrt { k } / \varepsilon )$ is approximately differentially private. Compared to the previous peeling approach of running Report Noisy Max $k$ times, the oneshot Laplace mechanism only adds noises and computes the top $k$ elements once, hence much more efficient for large $k$ . In addition, our proof of privacy relies on a novel coupling technique that bypasses the use of composition theorems. Finally, we present a novel application of efficient top- $k$ selection in the classical problem of ranking from pairwise comparisons.

# 1. Introduction

Modern statistical analyses have increasingly relied on sensitive data from individuals and, accordingly, there is a growing recognition that privacy constraints should be incorporated into consideration in data analysis. In response, a mathematically rigorous framework called differential privacy (Dwork et al., 2006a;b) was introduced for privacy-preserving data analysis. Roughly speaking, a differentially private procedure ensures that the released information is not influenced significantly by any individual record in the dataset. As a consequence, the privacy of the individuals will not be revealed based on the released information.

1Department of Statistics, University of Michigan, Ann Arbor, MI, USA. 2The Wharton School, University of Pennsylvania, Philadelphia, PA, USA. 3Google Research, Mountain View, CA, USA. Correspondence to: <qiaogang@umich.edu, suw@wharton.upenn.edu, liqzhang@google.com>.

Proceedings of the ${ 3 } { { \boldsymbol { \mathit { 8 } } } ^ { t h } }$ International Conference on Machine Learning, PMLR 139, 2021. Copyright 2021 by the author(s).

This paper is concerned with the top- $k$ problem, one of the most important primitives in differential privacy: reporting $k$ items with (approximately) the maximum values among $m$ given values. The problem of privately reporting the $k$ largest elements is an essential building block in many machine learning tasks and has gained continued popularity in the literature (McSherry & Mironov, 2009; Friedman & Schuster, 2010; Banerjee et al., 2012; McSherry & Mironov, 2013; Shen & Jin, 2014; Qin et al., 2016; Bafna & Ullman, 2017; Steinke & Ullman, 2017; Dwork et al., 2018; Durfee & Rogers, 2019). The common peeling solution Hardt & Roth (2013) and Dwork et al. (2018) is by iteratively applying the Report Noisy Max algorithm and then resorting to the composition theorem for computing the privacy loss. In general, this results in the noise level of $O ( k / \varepsilon )$ for $\varepsilon$ pure privacy and $\widetilde { O } ( \sqrt { k } / \varepsilon ) ^ { 1 }$ f o r $( \varepsilon , \delta )$ privacy loss. While the peeling algorithm has good privacy guarantee, it requires to run Report Noisy Max $k$ times, hence incurring a high computational cost for large $k$ .

In this paper, we show that by adapting Report Noisy Max to reporting noisy top- $k$ items, we can still achieve comparable utility but with a much more efficient procedure as we only need to run the selection once. We call the resulted algorithm the oneshot Laplace mechanism. More precisely, in the oneshot Laplace mechanism, we add the Laplace noise to each count and then report the set of items with the top- $k$ noisy counts. In this paper, we show that the oneshot Laplace mechanism can achieve utility comparable to those obtained from the peeling procedure (see Theorems 2.1 and 2.2 for the precise statements).

It is relatively straightforward to show that by adding Laplace noise of level $k / \varepsilon$ , the mechanism is $\varepsilon$ purely differentially private (Theorem 2.1).2 However, it turns out to be much more challenging to show that with $\widetilde { O } ( \sqrt { k } / \varepsilon )$ noise, it is $( \varepsilon , \delta )$ -differentially private. Indeed, the proof of this fact is the main contribution of our paper (Theorem 2.2).

Our proof directly bounds the privacy loss without the help

of the composition theorems. The difficulty of this approach is in untangling the complex distribution dependence of the $k$ selected items, as opposed to the conditional dependence in other existing mechanisms, which allows us to use the advanced composition theorem (Dwork et al. (2010), see also Kairouz et al. (2017)). To deal with this difficulty, we introduce a novel theoretical technique that, in effect, reduces the oneshot problem to a multinomial distribution problem to bypass the use of composition theorems. To shed light on our proof, consider the case of many similar or equal values. This is the case where the true top- $k$ set can be extremely sensitive to the change of input values. In order to privately report the top- $k$ set in this case, we add independent Laplace noises centered at zero to these values, which yields an approximately equal chance that the noisy values of two adjacent inputs will “go up” or “go down”, leading to the cancellation of certain first-order terms in the (logarithms of) the probabilities of events and hence a tight control between their ratio.

Circumventing composition theorems, however, may have its advantage. Since it relies on the direct analysis so may avoid the slackness introduced by the generic composition theorems. Indeed, there have been recent work on exploring the special properties of the privacy mechanisms to improve upon the generic composition theorem (Abadi et al., 2016; Bun & Steinke, 2016; Dong et al., 2021).

One closely related previous work is the oneshot Gumbel mechanism proposed in Durfee & Rogers (2019). In that paper, the authors show that adding Gumbel noise and reporting the top- $k$ items is equivalent to the peeling procedure of the exponential mechanism for reporting the maximum item (Dwork & Roth, 2014). This elegant connection immediately provides privacy guarantees through the well understood composition of exponential mechanisms and can benefit from any improvement of the composition property (Dong et al., 2019). In addition, their mechanism reports the noisy rank of the top- $k$ selections. However, it is important to note that their analysis goes through the composition theorem, whereas our paper takes an entirely different angle by employing a composition-free analysis. The comparison between these two approaches, both in theory and in practice, remains an interesting future work.

# 1.1. Preliminaries

Before continuing, we pause to revisit some basic concepts in differential privacy.

Definition 1.1. Data sets $D , D ^ { \prime }$ are said to be neighbors, or adjacent, if one is obtained by removing or adding a single data item.

Differential privacy, sometimes called pure differential privacy now, was first defined and constructed in Dwork et al.

(2006b). The relaxation of pure differential privacy defined next is sometimes referred to as approximate differential privacy or $( \varepsilon , \delta )$ -differential privacy.

Definition 1.2 (Differential privacy (Dwork et al., 2006b)). A randomized mechanism $\mathcal { M }$ is $( \varepsilon , \delta )$ -differentially private if for all adjacent $D , D ^ { \prime }$ , and for any subset of possible outputs $S$ : $\mathbb { P } ( \mathcal { M } ( D ) \in S ) \le \mathtt { e } ^ { \varepsilon } \mathbb { P } ( \mathcal { M } ( D ^ { \prime } ) \in S ) + \delta$ . Pure differential privacy is the special case of approximate differential privacy in which $\delta = 0$ .

In differential privacy problems, privacy law protects individually identifiable data, and the parameters $\varepsilon$ and $\delta$ in the definition above measure the degree of privacy protected. Let $\begin{array} { r } { f \ = \ f ( D ) } \end{array}$ be a statistic on a database $D$ . In any randomized $( \varepsilon , \delta )$ -differentially private mechanism $\mathcal { M }$ , the perturbed response $f ( D ) + Z$ is reported instead of the true answer $f ( D )$ , where $Z$ is the random noise to guarantee indistinguishability between two datasets. The sensitivity of the statistic, or query function $f$ , is the largest change in its output when we change a single data item and is defined below.

Definition 1.3 (Sensitivity). Let $f = ( f _ { 1 } , \cdot \cdot \cdot , f _ { m } )$ be $m$ real valued functions that take a database as input. The sensitivity of $f$ , denoted as $s$ , is defined as

$$
s _ {f} = \max  _ {D, D ^ {\prime}} \max  _ {1 \leq i \leq m} | f _ {i} (D) - f _ {i} (D ^ {\prime}) |,
$$

where the maximum is taken over any adjacent databases $D$ and $D ^ { \prime }$ .

In the Laplace Mechanism, the output $f ( D )$ is perturbed with noise generated from the Laplace distribution $\mathrm { L a p } ( \lambda )$ with probabilitywhere the scale ensity function: should be calib $\begin{array} { r } { f _ { \mathrm { L a p } ( \lambda ) } ( z ) = \frac { 1 } { 2 \lambda } \mathrm { e } ^ { - | z | / \lambda } } \end{array}$ $\lambda$ the statistics $f$ .

# 2. The Oneshot Laplace Mechanism

In this section, we introduce the oneshot Laplace mechanism in full detail, along with its privacy guarantees. Consider the problem of privately reporting the minimum $k$ locations of $m$ values $x _ { 1 } , \ldots , x _ { m }$ and their estimated values. Here two input values $( x _ { 1 } , \ldots , x _ { m } )$ and $( x _ { 1 } ^ { \prime } , \ldots , x _ { m } ^ { \prime } )$ are called adjacent if $\| x - x ^ { \prime } \| _ { \infty } \leq 1$ , i.e., $| x _ { i } - x _ { i } ^ { \prime } | \leq 1$ for all $1 \leq i \leq m$ . In this definition, $x$ can be considered as the counts of each of $m$ -attributes of the population in some database $D$ and, therefore, changing any individual in $D$ may in the worst case change each count $x _ { i }$ by 1.

As a special case when $k \ = \ 1$ , the solution relies on the Report Noisy Min algorithm (Dwork & Roth, 2014; Dwork et al., 2018), which takes as input a function $f$ , database $D$ , and privacy parameter $\varepsilon$ , and outputs the index of the minimum element and its estimated value. The Report Noisy Min algorithm adds independently sampled

$\mathrm { L a p } ( 2 s _ { f } / \varepsilon )$ noise to each element of $f ( D )$ and reports the index $i ^ { * }$ of the minimum noisy count. The algorithm further reports its estimated value by adding noise freshly sampled from $\mathrm { L a p } ( 2 s _ { f } / \varepsilon )$ to $f _ { i ^ { * } } ( D )$ . In Dwork & Roth (2014); Dwork et al. (2018), the Report Noisy Min algorithm is proved to be $( \varepsilon , 0 )$ -differentially private. Notably, in order to avoid violation of differential privacy, we shall not report the minimum noisy element as its estimated value. Hence, we need to add fresh random noise to $f _ { i ^ { * } } ( D )$ in the last step.

To efficiently solve the top- $k$ problem where $k$ can be larger than 1, we introduce the oneshot Laplace mechanism $\mathcal { M } ^ { \mathrm { o s } }$ , which is one of our main contributions in this paper. In $\mathcal { M } ^ { \mathrm { o s } }$ , we add noise $\mathrm { L a p } ( \lambda )$ once to each value and report the indices and approximations of the minimum $k$ noisy values (Algorithm 1).

Algorithm 1 The Oneshot Laplace Mechanism $\mathcal { M } ^ { \mathrm { o s } }$ for Privately Reporting Minimum $k$ Elements

Input: database $D$ , functions ${ \overline { { f = ( f _ { 1 } , \cdots , f _ { m } ) } } }$ with sensitivity $s _ { f }$ , parameter $k$ , and the noise scale $\lambda$

Output: indices $i _ { 1 } , \dots , i _ { k }$ and approximations to $f _ { i _ { 1 } } ( D ) , \cdots , f _ { i _ { k } } ( D )$

1: for $i = 1$ to $m$ do   
2: set $y _ { i } = f _ { i } ( D ) + g _ { i }$ where $g _ { i }$ is sampled i.i.d. from $\mathrm { L a p } ( \lambda )$   
3: end for   
4: sort $y _ { 1 } , \ldots , y _ { m }$ from low to high, $y _ { i _ { 1 } } \leq y _ { i _ { 2 } } \leq \cdot \cdot \cdot \leq$ $y _ { i _ { m } }$   
5: return the set $\{ i _ { 1 } , \ldots , i _ { k } \}$ and $f _ { i _ { j } } ( D ) + g _ { i _ { j } } ^ { \prime }$ , where $1 \leq j \leq k$ and $g _ { i _ { j } } ^ { \prime }$ are fresh independent random noise sampled from Lap(λ)

We provide the following theorem for pure differential privacy of the oneshot Laplace mechanism. Its proof uses a standard coupling argument and it is given in the supplementary material.

Theorem 2.1. The oneshot Laplace mechanism is $( \varepsilon , 0 )$ - differentially private if we set $\lambda = 2 k s _ { f } / \varepsilon$ or larger.

However, it is surprisingly challenging to prove the privacy guarantees for the oneshot Laplace mechanism in the approximate differential privacy framework. Here we state the theorem and leave the technical sketches and intuition to Section 4 and the complete proof to the supplementary material.

Theorem 2.2 (Privacy guarantees). Given $\varepsilon \le 0 . 2 , \delta \le$ 0.05 and $m \geq 2$ , the oneshot Laplace mechanism is $( \varepsilon , \delta )$ - differentially private if we set $\begin{array} { r } { \lambda _ { \mathrm { o n e s h o t } } = \frac { 8 s _ { f } \sqrt { k \log ( m / \delta ) } } { \varepsilon } o r } \end{array}$ larger.

We remark that $\varepsilon$ can be set up to ${ \cal O } ( \log ( m / \delta ) )$ (Dwork et al., 2015), and the constant in $\lambda _ { \mathrm { o n e s h o t } }$ is for ease

of analysis. We also point out that our result includes a higher multiplicative factor of $O ( \sqrt { \log ( m / \delta ) } )$ compared to $O ( \sqrt { { 1 0 } \mathrm { g } ( { 1 / \delta } ) } )$ in the other results, which only incurs a small constant factor since $\delta$ is typically required to be $o ( 1 / m )$ . The next result is concerned with the utility of the oneshot Laplace mechanism.

Theorem 2.3 (Utility). Let $f _ { ( 1 ) } ( D ) \leq f _ { ( 2 ) } ( D ) \leq \cdots \leq$ $f _ { ( m ) } ( D )$ denote the order statistics of the counts. Write $\begin{array} { r } { \Delta \ : = \ \operatorname* { m i n } _ { 1 \leq i \leq m - 1 } \left\{ f _ { ( i + 1 ) } ( D ) - f _ { ( i ) } ( D ) \right\} } \end{array}$ . Then with probability at least

$$
p (\Delta) = \max  \left\{0, 1 - \frac {(m - 1) (2 \lambda + \Delta) \mathrm {e} ^ {- \Delta / \lambda}}{4 \lambda} \right\},
$$

the oneshot Laplace mechanism returns the index set of the true top-k elements.

The proof of Theorem 2.3 is mainly based on the application of Bonferroni bound and is left to the supplementary material. The form of $p ( \Delta )$ guarantees that when the gaps of $f _ { i } ( D )$ ’s are significantly large, the oneshot Laplace mechanism can return the true index set of top-$k$ elements almost surely. Specifically, when $\Delta \ge 2 0 \lambda$ and $m \leq 8 \times 1 0 ^ { 6 }$ , then with probability at least $p ( \Delta ) > 0 . 9 9$ the oneshot Laplace mechanism correctly returns the index set of the true top- $k$ elements.

# 3. Application to Differentially Private Pairwise Comparison

Our work was first motivated by and used for the private false discovery rate control mechanism (Dwork et al., 2018). Here we present another application in ranking $n$ objects from partial binary comparisons, a problem with many important applications in Statistics and Computer Science.

Given a large collection of $m$ items, and only part of the comparisons $\{ X _ { i j } \} _ { 1 \leq i \neq j \leq m }$ between pairs of the $m$ items are revealed. Our goal is to privately recover the set of $k$ items with the highest ranks through the information released by pairwise comparison. One of the most widely used parametric models for pairwise comparison discovered is the Bradley-Terry-Luce (BTL) model (Bradley & Terry, 1952; Luce, 2012).

The BTL model was introduced to derive a full ranking when one only has access to pairwise comparison information. The basic idea of the Bradley-Terry-Luce parametric model is to assume that there exists a latent preference score $\omega _ { i } ^ { * } ( i = 1 , \cdots , m )$ assigned to the $m$ items of interest, and given a pair of items $( i , j )$ from the population, one can estimate the winning probability of item $j$ over item i

in the pairwise comparison as

$$
P _ {j i} = \mathbb {P} \{\text {i t e m} j \text {i s p r e f e r r e d o v e r i t e m} i \} = \frac {\omega_ {j} ^ {*}}{\omega_ {i} ^ {*} + \omega_ {j} ^ {*}}.
$$

To define a comparison graph $\mathcal { G } = ( \nu , E )$ for the Bradley-Terry-Luce model, we define the vertices set $\nu = [ m ]$ of the graph $\mathcal { G }$ to represent the $m$ items we aim to compare, and each edge $( i , j )$ included in the edge set $E$ indicates that items $i$ and $j$ are compared and the comparison information is included in $\textbf {  { y } }$ . We further assume that the comparison graph $\mathcal { G }$ is drawn from the Erdös-Rényi random graph (Erdos & Rényi, 1960), such that each edge between any two vertices is present independently with some probability that captures the fraction of paired items being compared. For each edge $( i , j ) \in E$ , we obtain $L$ independent paired comparison sampled from items $i$ and $j$ , and for the lth comparison y(l)i,j $y _ { i , j } ^ { ( l ) }$ , where $1 \le l \le L$ , we build the pairwise comparison model by assigning

$$
y _ {i, j} ^ {(l)} = \left\{ \begin{array}{l l} 1, & \text {w i t h p r o b a b i l i t y} \frac {\omega_ {j} ^ {*}}{\omega_ {i} ^ {*} + \omega_ {j} ^ {*}}, \\ 0, & \text {o t h e r w i s e}, \end{array} \right. \tag {3.1}
$$

and y j, i $y _ { j , i } ^ { ( l ) } = 1 - y _ { i , j } ^ { ( l ) }$ yi,j for all $( i , j ) \in E$ . The sufficient statistics of this model are given by

$$
\pmb {y} := \{y _ {i, j} | (i, j) \in E \},
$$

where yi,j := 1L P1 l L y(l)i,j $\begin{array} { r } { y _ { i , j } } & { : = ~ \frac { 1 } { L } \sum _ { 1 \leq l \leq L } y _ { i , j } ^ { ( l ) } } \end{array}$ . We remark that in the regime of differential privacy, the comparison graphs of two adjacent datasets only differ in one data item, i.e., only one sample of a specific edge in the adjacent datasets is different.

Two algorithms tailored to the Bradley-Terry-Luce model that attract most attention are the spectral method (rank centrality) and the maximum likelihood estimator method (Chen et al., 2017), the former of which we will focus on due to the applicability of the oneshot Laplace mechanism. To adopt the spectral method, we make use of the pairwise comparison information $\textbf {  { y } }$ to establish a random walk over the graph $\mathcal { G }$ by defining its time-independent transition matrix $P _ { m \times m } = [ P _ { i j } ]$ , where $P _ { i j } = \mathbb { P } ( X _ { t + 1 } = j | X _ { t } = i )$ is defined as

$$
P _ {i j} = \left\{ \begin{array}{l l} \frac {1}{d} y _ {i, j} & \text {i f} (i, j) \in E, \\ 1 - \frac {1}{d} \sum_ {k: (i, k) \in E} y _ {i, k} & \text {i f} i = j, \\ 0 & \text {o t h e r w i s e .} \end{array} \right. \tag {3.2}
$$

Here $d > 0$ is some given normalization factor which is on the same order of the maximum vertex degree of graph $\mathcal { G }$ , and in general, we can assume that the normalization factor $d$ becomes larger as the number of vertices in graph $\mathcal { G }$ grows. The spectral method is summarized in Algorithm 2.

# Algorithm 2 The spectral method for pairwise comparison

Input: comparison graph $G = ( [ m ] , E )$ , sufficient statistics $\textbf {  { y } }$ and the normalization factor $d > 0$

Output: the rank of $\{ \pi ( i ) \} _ { i \in [ m ] }$

1: define the defined transition matrix $_ { r }$ as in (3.2)   
2: compute the stationary distribution $\pi$ of $_ { P }$   
3: sort $\pi _ { 1 } , \cdots , \pi _ { m }$ from low to high, $\pi _ { ( 1 ) } ~ \leq ~ \pi _ { ( 2 ) } ~ \leq$ · · · ≤ π(m)

The intuition behind the spectral method is based on the fact that, assuming the sample size is sufficiently large, the stationary distribution $\pi$ of the transition matrix $_ { r }$ defined in (3.2) is a reliable estimate of the preference scores $[ \omega _ { 1 } ^ { * } , \omega _ { 2 } ^ { * } , \cdots , \omega _ { m } ^ { * } ]$ up to some scaling (Chen et al., 2017). We notice that the result derived from the spectral method of pairwise comparison only takes advantage of the stationary distribution and, therefore, the oneshot Laplace mechanism can be applied to report top- $k$ elements via pairwise comparison information privately. We pause to introduce some definitions and a lemma to find the sensitivity of the statistic that maps the pairwise information to the stationary distribution. Throughout this paper, the $\infty$ -norm $\| \boldsymbol { P } \| _ { \infty }$ of a matrix $_ { r }$ is its maximum absolute row sum.

Definition 3.1 (Ergodicity coefficient of a stochastic matrix (Ipsen & Selee, 2011)). For a $m \times m$ stochastic matrix $\pmb { A }$ , the ergodicity coefficient $\tau _ { 1 } ( A )$ of matrix $\pmb { A }$ is defined as

$$
\tau_{1}(\boldsymbol {A})\equiv \sup_{\substack{\| \boldsymbol {v}\|_{1} = 1\\ \boldsymbol{v}^{\top}e = 0}}\| \boldsymbol{v}^{\top}\boldsymbol {A}\|_{1},
$$

where $e$ is the vector of all ones.

Definition 3.2 (Conditional number of a Markov Chain (Cho & Meyer, 2001)). Let $_ { r }$ denote the transition probability matrix of an $m$ state Markov chain $\mathcal { C }$ , and $\pi$ denotes the stationary distribution vector. The perturbed matrix $\tilde { P }$ is the transition probability matrix of another $n$ state Markov chain $\tilde { \mathcal { C } }$ with stationary distribution vector $\tilde { \pi }$ . The conditional number $\kappa$ of a Markov chain $\mathcal { C }$ is defined by the following perturbation bound $\| \pi - \tilde { \pi } \| _ { \infty } \leq \kappa \| P - \tilde { P } \| _ { \infty }$ .

In Ipsen & Selee (2011), the authors stated the result that for every transition matrix $_ { r }$ , the ergodicity coefficient of $_ { r }$ always falls between 0 and 1, and $\tau _ { 1 } ( P ) = 1$ if and only if the rank of matrix $_ { r }$ equals 1. There is also a vast literature on exploring the form of the conditional number $\kappa$ (Cho & Meyer, 2001). With all these preparations, we will build our private spectral method based on the following conclusion from Seneta (1988) and Cho & Meyer (2001).

Lemma 3.3 (Sensitivity of stationary distribution (Seneta, 1988; Cho & Meyer, 2001)). Suppose $_ { r }$ and $\tilde { P }$ are $m \times m$ transition matrices with unique stationary distributions $\pi ^ { \top }$

and $\tilde { \pi } ^ { \top }$ . If the ergodicity coefficient of transition matrix $_ { P }$ satisfies $\tau _ { 1 } ( P ) < 1$ , then $\begin{array} { r } { \| \pi ^ { \top } - \tilde { \pi } ^ { \top } \| _ { \infty } \leq \frac { 1 } { 1 - \tau _ { 1 } ( P ) } \| \tilde { P } - } \end{array}$ $P \| _ { \infty }$ .

Motivated by Lemma 3.3, the mapping $f$ has a bound sensitivity when the ergodicity coefficient of the transition matrix $_ { r }$ is upper bounded by a constant $\rho < 1$ . In light of this observation, we can build our oneshot algorithm based on the following definition.

Definition 3.4 $\overset { \cdot } { \rho }$ -constrained comparison graph). A comparison graph $\mathcal { G } = ( \nu , E )$ is said to be $\rho$ -constrained if: (1) the transition matrix $_ { r }$ of the Markov Chain defined as in (3.2) has unique stationary distribution $\pi ^ { \top }$ . (2) there exists a constant $\rho < 1$ such that $\tau _ { 1 } ( P ) \leq \rho$ .

Definition 3.4 implies that if the comparison graph $\mathcal { G }$ defined by the database $D$ is $\rho$ -constrained, then the mapping $f : P \to \pi$ has a sensitivity bounded by $( 1 - \rho ) ^ { - 1 }$ . Making use of this fact, the oneshot Laplace mechanism for privately reporting the maximum $k$ elements through pairwise comparison information is stated in Algorithm 3.

Algorithm 3 The oneshot differentially private spectral method   
Input: $\rho$ -constrained comparison graph $\mathcal{G} = ([m], E)$ , sufficient statistics $\pmb{y}$ , parameter $L$ , normalization factor $d > 0$ , $k \geq 1$ and privacy parameters $\varepsilon, \delta$

Now we establish the differential privacy of Algorithm 3 in Theorem 3.5 stated below, and the proof is left to the supplementary material.

Theorem 3.5. Given $\varepsilon ~ \leq ~ 0 . 2$ and $\delta ~ \leq ~ 0 . 0 5$ , assume that the comparison graph $\mathcal { G } = ( [ m ] , E )$ is $\rho$ -constrained, then Algorithm 3 is $( \varepsilon , \delta )$ -differentially private if $\lambda \ =$ $\frac { 8 s \sqrt { k \log ( m / \delta ) } } { \varepsilon }$ or larger, where the sensitivity s = 2dL(1 ρ) . $\begin{array} { r } { s = \frac { 2 } { d L ( 1 - \rho ) } } \end{array}$

# 4. Proofs and Intuition

In this section, we introduce a novel technique to prove the privacy of the oneshot Laplace mechanism. At a high level, the proof proceeds by considering the “bad” events, which have a large probability bias between two neighboring inputs. We show that those “bad” events happen when the sum of some dependent random variables deviates from its mean. We first partition the event space to remove the dependence between the random variables and, therefore, we

can apply a concentration bound directly. Furthermore, we apply a coupling technique to pair up the partitions for the two neighboring inputs. For each pair, we apply a concentration inequality to bound the probability of “bad” events. The technical tools developed for proving the privacy of the oneshot top- $k$ algorithm in this section can be applied in many other settings. We provide the proof sketch to our main result and leave most of the technical details to the supplementary material.

Our goal is to reduce the dependence on $k$ to $\sqrt { k }$ for $( \varepsilon , \delta )$ - differential privacy in the oneshot Laplace mechanism. We note that in the oneshot Laplace mechanism $\mathcal { M } ^ { \mathrm { o s } }$ , only a subset of $k$ elements, but not their ordering, is returned. The privacy proof of the oneshot Laplace mechanism crucially depends on this fact. We remark that one can further obtain the relative ranks and scores by running a second ranking phase in either mechanism to the reported $k$ elements. For example, by utilizing the Gaussian mechanism, one can publish more accurate scores, and hence their relative ranks, with the maximum noise of $O ( { \sqrt { k \log k } } )$ by paying slightly more privacy cost.

We start by providing the following lemma that directly establishes the privacy part of the oneshot Laplace mechanism in Theorem 2.2.

Lemma 4.1. For any $k$ -subset $S$ of $\{ 1 , \cdots , m \}$ and any adjacent $x , x ^ { \prime }$ , we have

$$
\mathbb {P} \left(\mathcal {M} ^ {\mathrm {o s}} (x) \in S\right) \leq \mathrm {e} ^ {\varepsilon} \mathbb {P} \left(\mathcal {M} ^ {\mathrm {o s}} \left(x ^ {\prime}\right) \in S\right) + \delta .
$$

The key idea of the proof is to divide the event space by fixing the kth smallest noisy element $j$ together with the noise value $g _ { j }$ . For each partition, whether an element $i \neq j$ is selected by $\mathcal { M } ^ { \mathrm { o s } }$ only depends on whether $x _ { i } + g _ { i } \leq x _ { j } + g _ { j }$ , which happens with probability $q _ { i } = G ( ( x _ { j } + g _ { j } - x _ { i } ) / \lambda )$ . Here $G$ denotes the cumulative distribution function of the standard Laplace distribution. As a result, we consider the following mechanism $\mathcal { M }$ instead: given $( q _ { 1 } , \dots , q _ { m } )$ where $0 < q _ { i } < 1$ , output a subset of indices where each index $i$ is included in the subset with probability $q _ { i }$ . In the following proof, we will first understand the sensitivity of $q _ { i }$ dependent on the change of $x _ { i }$ and then show that $\mathcal { M }$ is “private” with respect to the corresponding sensitivity on $q$ . In order to prove Lemma 4.1, we present the definition of $\tau$ -closeness for vectors and two other lemmata we shall also use.

Definition 4.2 $\tau$ -closeness for vectors). For two vectors $q = ( q _ { 1 } , \dots , q _ { m } )$ and $q ^ { \prime } = ( q _ { 1 } ^ { \prime } , \ldots , q _ { m } ^ { \prime } ) $ , we say $q$ is $\tau$ - close with respect to $q ^ { \prime }$ if for each $1 \leq i \leq m$ , $| q _ { i } - q _ { i } ^ { \prime } | \leq$ $\tau q _ { i } ( 1 - q _ { i } )$ .

Lemma 4.3. For any $z , z ^ { \prime }$ , we have

$$
\left| G \left(z ^ {\prime}\right) - G (z) \right| \leq 2 \mathrm {e} ^ {\left| z ^ {\prime} - z \right|} \left| z ^ {\prime} - z \right| G (z) \left(1 - G (z)\right),
$$

here $G$ denotes the cumulative distribution function of the standard Laplace distribution.

The following lemma is the key step that constitutes the privacy guarantee of the mechanism $\mathcal { M }$ with respect to the sensitivity on $q$ , and the proof of Lemma 4.1 is largely based on this result combined with Lemma 4.3.

Lemma 4.4. Assume $C _ { 0 } ~ = ~ 3 . 9 ^ { 2 }$ , $C _ { 1 } ~ = ~ 1 . 9 5$ . Under the conditions $\varepsilon ~ \leq ~ 0 . 2$ , $\delta ~ \leq ~ 0 . 0 5$ , $m \ \geq \ 2$ and $k \geq C _ { 0 } \log ( m / \delta )$ , and if $q$ is $\tau$ -close with respect to $q ^ { \prime }$ with $\tau \leq \frac { \varepsilon } { C _ { 1 } \sqrt { k \log ( m / \delta ) } }$ , then for any set $S$ of $k$ -subsets of $\{ 1 , \cdots , m \}$ , we have

$$
\mathbb {P} (\mathcal {M} (q) \in S) \leq \mathrm {e} ^ {\varepsilon} \mathbb {P} (\mathcal {M} (q ^ {\prime}) \in S) + \delta / m.
$$

The proof of Lemma 4.3 is relatively straightforward and is relegated to the supplementary material. To prove Lemma 4.4, notice that if $k \le C _ { 0 } \log ( m / \delta )$ , then according to Theorem 2.1, the mechanism is $( \varepsilon , 0 )$ -private. Thus we assume $k \geq C _ { 0 } \log ( m / \delta )$ . Since $S$ consists of $k$ -sets, we first show that if $\textstyle \sum _ { i } q _ { i } \gg k$ , then $\mathbb { P } ( \mathcal { M } ( q ) \in S )$ is small. This can be done by applying the standard concentration bound in the following lemma.

Lemma 4.5. Let $Z _ { 1 } , \ldots , Z _ { m }$ be m independent Bernoulli random variables with $\mathbb { P } ( Z _ { i } ~ = ~ 1 ) ~ = ~ q _ { i }$ . Suppose $\textstyle \sum _ { i = 1 } ^ { m } q _ { i } \geq ( 1 + t ) k$ for any $t > 0$ . Then $\begin{array} { r } { \mathbb { P } \left( \sum _ { i } Z _ { i } \leq k \right) \leq } \end{array}$ $\begin{array} { r } { \exp \left( - ( 1 + t ) k h \left( \frac { t } { t + 1 } \right) \right) } \end{array}$ where $h ( u ) = ( 1 + u ) \log ( 1 +$ $u ) - u .$ . Specifically, by setting $K = ( 1 + c \sqrt { \log ( m / \delta ) / k } ) k$ and $c = 1 . 9$ , if we have $\textstyle \sum _ { i = 1 } ^ { m } q _ { i } \geq K$ , then

$$
\mathbb {P} \left(\sum_ {i = 1} ^ {m} Z _ {i} \leq k\right) \leq \frac {\delta}{m}.
$$

The proof of Lemma 4.5 is based on the classical Bennett’s inequality and is left to the supplementary material. By Lemma 4.5, we only need to consider the case of $\textstyle \sum _ { i } q _ { i } \leq$ $K$ , which is more difficult than the case above. We first represent a set $S \subseteq \{ 1 , \dots , m \}$ by a binary vector $z \in \{ 0 , 1 \} ^ { m }$ and write $\mathbb { P } _ { q } ( z )$ as $\begin{array} { r } { \mathbb { P } _ { q } ( z ) = \prod _ { i : z _ { i } = 1 } q _ { i } \prod _ { i : z _ { i } = 0 } ( 1 - q _ { i } ) } \end{array}$ . Our goal is to show that for any $S$ consisting of weight $k$ vectors in $\{ 0 , 1 \} ^ { m }$ ,

$$
\sum_ {z \in S} \mathbb {P} _ {q} (z) \leq \mathrm {e} ^ {\varepsilon} \sum_ {z \in S} \mathbb {P} _ {q ^ {\prime}} (z) + \frac {\delta}{m}.
$$

By defining the set $S ^ { * } = \{ z \ : \ \mathbb { P } _ { q } ( z ) \geq \mathrm { e } ^ { \varepsilon } \mathbb { P } _ { q ^ { \prime } } ( z ) \}$ , to prove Lemma 4.4, it suffices to show that $\begin{array} { r } { \sum _ { z \in S ^ { * } } \mathbb { P } _ { q } ( z ) \le } \end{array}$ $\begin{array} { l } { { \frac { \delta } { m } } } \end{array}$ . By the form of $\mathbb { P } _ { q } ( z )$ , $z \in S ^ { * }$ ∈holds if and only if

$$
\prod_ {i: z _ {i} = 1} q _ {i} \prod_ {i: z _ {i} = 0} (1 - q _ {i}) \geq \mathrm {e} ^ {\varepsilon} \prod_ {i: z _ {i} = 1} q _ {i} ^ {\prime} \prod_ {i: z _ {i} = 0} (1 - q _ {i} ^ {\prime}). \tag {4.1}
$$

Let $\Delta _ { i } = q _ { i } ^ { \prime } - q _ { i }$ . The $\tau$ -closeness of $q$ with respect to $q ^ { \prime }$ implies $| \Delta _ { i } | \leq \tau q _ { i } ( 1 - q _ { i } )$ . Taking the logarithm of both sides of (4.1) and rearranging gives

$$
\sum_ {i: z _ {i} = 1} \log (1 + \Delta_ {i} / q _ {i}) + \sum_ {i: z _ {i} = 0} \log (1 - \Delta_ {i} / (1 - q _ {i})) \leq - \varepsilon .
$$

To bound $\textstyle \sum _ { z \in S ^ { * } } \mathbb { P } _ { q } ( z )$ , we consider indepen-∈dent Bernoulli random variables $Z _ { 1 } , \ldots , Z _ { m }$ , where for each i, $\begin{array} { r l r l } { Z _ { i } } & { { } = } & { 1 } \end{array}$ with probability $q _ { i }$ and $\begin{array} { r l r } { Z _ { i } } & { { } = } & { 0 } \end{array}$ with probability $1 \ - \ q _ { i }$ . We set $\zeta _ { i } = Z _ { i } \log ( 1 + \Delta _ { i } / q _ { i } ) + ( 1 - Z _ { i } ) \log ( 1 - \Delta _ { i } / ( 1 - q _ { i } ) )$ . Note that

$$
\sum_ {z \in S ^ {*}} \mathbb {P} _ {q} (z) = \sum_ {z} \mathbf {1} _ {\{\sum \zeta_ {i} \leq - \varepsilon \}} \mathbb {P} _ {q} (z) = \mathbb {P} \left(\sum_ {i} \zeta_ {i} \leq - \varepsilon\right),
$$

where $\mathbf { 1 } _ { \{ \cdot \} }$ denotes the indicator function and the last probability is over the distribution of $Z _ { 1 } , \ldots , Z _ { m }$ . Combine this with previous arguments, we need to prove that $\begin{array} { l l l l } { \mathbb { P } ( \sum _ { i } \zeta _ { i } } & { \leq } & { - \varepsilon ) } & { \leq } & { \delta / m } \end{array}$ . It is easy to check that $\zeta _ { 1 } ~ + ~ . ~ . ~ + ~ \zeta _ { m }$ has mean i=1 and variance $\textstyle \sum _ { i = 1 } ^ { m } \big ( q _ { i } \log ( 1 + \Delta _ { i } / q _ { i } ) + ( 1 - q _ { i } ) \log ( 1 - \Delta _ { i } / ( 1 - q _ { i } ) ) \big )$ $\begin{array} { r l r } { \sigma ^ { 2 } } & { { } = } & { \sum _ { i = 1 } ^ { m } q _ { i } ( 1 - q _ { i } ) \log ^ { 2 } \frac { 1 + \Delta _ { i } / q _ { i } } { 1 - \Delta _ { i } / ( 1 - q _ { i } ) } } \end{array}$ . the ranges of the centered random variables $\tilde { \zeta } _ { i } : = \zeta _ { i } -$ $q _ { i } \log ( 1 { + } \Delta _ { i } / q _ { i } ) - ( 1 { - } q _ { i } ) \log ( 1 { - } \Delta _ { i } / ( 1 { - } q _ { i } ) )$ are bounded in absolute value by max1≤i≤m log 1 ∆ /(1 q ) . $\begin{array} { r } { \operatorname* { m a x } _ { 1 \leq i \leq m } \left. \log \frac { 1 + \Delta _ { i } / q _ { i } } { 1 - \Delta _ { i } / ( 1 - q _ { i } ) } \right. } \end{array}$ 1+∆i/qi To see why this is true, observe that

$$
\begin{array}{l} \left| \tilde {\zeta} _ {i} \right| = \left| \left(Z _ {i} - q _ {i}\right) \log \frac {1 + \Delta_ {i} / q _ {i}}{1 - \Delta_ {i} / \left(1 - q _ {i}\right)} \right| \\ \leq \max  _ {1 \leq i \leq m} \left| \log \frac {1 + \Delta_ {i} / q _ {i}}{1 - \Delta_ {i} / (1 - q _ {i})} \right|. \\ \end{array}
$$

Therefore, according to Bennett’s inequality we can assert that for any $t \geq 0$ ,

$$
\sum_ {i = 1} ^ {m} \zeta_ {i} \geq \sum_ {i = 1} ^ {m} \left(q _ {i} \log \left(1 + \frac {\Delta_ {i}}{q _ {i}}\right) + (1 - q _ {i}) \log \left(1 - \frac {\Delta_ {i}}{1 - q _ {i}}\right)\right) - t
$$

with probability at least $\begin{array} { r } { 1 - \exp \left( - \frac { \sigma ^ { 2 } h ( A t / \sigma ^ { 2 } ) } { A ^ { 2 } } \right) } \end{array}$

$$
A = \max  _ {1 \leq i \leq m} | \log \frac {1 + \Delta_ {i} / q _ {i}}{1 - \Delta_ {i} / (1 - q _ {i})} |.
$$

Furthermore, by taking $\begin{array} { r l r l r l } { t } & { { } \quad } & { = \quad } & { { } } & { \varepsilon } & { { } + } \end{array}$ $\textstyle \sum _ { i = 1 } ^ { m } \big ( q _ { i } \log ( 1 + \Delta _ { i } / q _ { i } ) + ( 1 - q _ { i } ) \log ( 1 - \Delta _ { i } / ( 1 - q _ { i } ) ) \big )$ Bennett’s inequality implies that $\begin{array} { r } { \mathbb { P } ( \sum _ { i } \zeta _ { i } ~ \le ~ - \varepsilon ) ~ \le ~ } \end{array}$ $\exp \left( - \sigma ^ { 2 } h ( A \bar { t } / \sigma ^ { 2 } ) / A ^ { 2 } \right)$ . Hence, the case of $\textstyle \sum _ { i } q _ { i } \leq K$ can be established by proving

$$
\frac {\sigma^ {2} h \left(A t / \sigma^ {2}\right)}{A ^ {2}} \geq \log \frac {m}{\delta}. \tag {4.2}
$$

Now we seek to bound $t , \varepsilon , A$ and $\sigma$ using the fact that $| \Delta _ { i } | \le \tau q _ { i } ( 1 - q _ { i } )$ . We start with exploring the relationship between $t$ and $\varepsilon$ by applying the standard results that $\log ( 1 + u ) \leq u$ , and when $\begin{array} { r } { | u | \le \frac { 1 } { 2 } } \end{array}$ , $\log ( 1 + u ) \geq u - u ^ { 2 }$ . Notice that

$$
\begin{array}{l} \max  \left\{\left| \frac {\Delta_ {i}}{q _ {i}} \right|, \left| \frac {\Delta_ {i}}{1 - q _ {i}} \right| \right\} \leq \tau \leq \frac {\varepsilon}{C _ {1} \sqrt {k \log (m / \delta)}} \\ \leq \frac {\varepsilon}{C _ {1} \sqrt {C _ {0}} \log (m / \delta)} \leq \frac {0 . 2}{1 . 9 5 \times 3 . 9 \times \log (2 / 0 . 0 5)} <   \frac {1}{2}. \\ \end{array}
$$

We distinguish two cases. When $\Delta _ { i } ~ \geq ~ 0$ , we see that $q _ { i } \log ( 1 { + } \Delta _ { i } / q _ { i } ) > 0$ and $( 1 - q _ { i } ) \log ( 1 - \Delta _ { i } / ( 1 - q _ { i } ) ) < 0$ . If $| q _ { i } \log ( 1 + \Delta _ { i } / q _ { i } ) | \geq | ( 1 - q _ { i } ) \log ( 1 - \Delta _ { i } / ( 1 - q _ { i } ) ) |$ , these relations yield that

$$
\begin{array}{l} \left. \left| q _ {i} \log \left(1 + \Delta_ {i} / q _ {i}\right) + \left(1 - q _ {i}\right) \log \left(1 - \Delta_ {i} / \left(1 - q _ {i}\right)\right) \right| \right. \\ \leq \left| q _ {i} \left(\Delta_ {i} / q _ {i} + \left(\Delta_ {i} / q _ {i}\right) ^ {2}\right) + \left(1 - q _ {i}\right) \left(- \Delta_ {i} / \left(1 - q _ {i}\right) \right. \right. \\ + \left. \left(\Delta_ {i} / \left(1 - q _ {i}\right)\right) ^ {2}\right) | \\ = q _ {i} \left(\Delta_ {i} / q _ {i}\right) ^ {2} + \left(1 - q _ {i}\right) \left(\Delta_ {i} / \left(1 - q _ {i}\right)\right) ^ {2} \\ \leq \tau^ {2} \left(q _ {i} (1 - q _ {i}) ^ {2} + q _ {i} ^ {2} (1 - q _ {i})\right) \\ \leq \tau^ {2} q _ {i}. \\ \end{array}
$$

Similarly, in the case that $\begin{array} { r l } { | q _ { i } \log ( 1 + \Delta _ { i } / q _ { i } ) | } & { { } < } \end{array}$ $| ( 1 - q _ { i } ) \log ( 1 - \Delta _ { i } / ( 1 - q _ { i } ) ) |$ , it follows that

$$
\begin{array}{l} \left. \left| q _ {i} \log \left(1 + \Delta_ {i} / q _ {i}\right) + \left(1 - q _ {i}\right) \log \left(1 - \Delta_ {i} / \left(1 - q _ {i}\right)\right) \right| \right. \\ \leq \left| q _ {i} \left(\Delta_ {i} / q _ {i} - \left(\Delta_ {i} / q _ {i}\right) ^ {2}\right) + \left(1 - q _ {i}\right) \left(- \Delta_ {i} / \left(1 - q _ {i}\right) \right. \right. \\ - \left. \left(\Delta_ {i} / (1 - q _ {i})\right) ^ {2}\right) | \\ = q _ {i} \left(\Delta_ {i} / q _ {i}\right) ^ {2} + \left(1 - q _ {i}\right) \left(\Delta_ {i} / \left(1 - q _ {i}\right)\right) ^ {2} \\ \leq \tau^ {2} \left(q _ {i} (1 - q _ {i}) ^ {2} + q _ {i} ^ {2} (1 - q _ {i})\right) \\ \leq \tau^ {2} q _ {i}. \\ \end{array}
$$

To proceed, note that $\begin{array} { r } { \sum _ { i } q _ { i } \le K \le ( 1 + \frac { c } { \sqrt { C _ { 0 } } } ) k } \end{array}$ , and thus

$$
\begin{array}{l} \left| \sum_ {i = 1} ^ {m} \left(q _ {i} \log \left(1 + \Delta_ {i} / q _ {i}\right) + \left(1 - q _ {i}\right) \log \left(1 - \Delta_ {i} / \left(1 - q _ {i}\right)\right)\right) \right| \\ \leq \sum_ {i = 1} ^ {m} \left| q _ {i} \log \left(1 + \Delta_ {i} / q _ {i}\right) + (1 - q _ {i}) \log \left(1 - \Delta_ {i} / (1 - q _ {i})\right) \right| \\ \leq \tau^ {2} \sum_ {i = 1} ^ {m} q _ {i} \leq \left(1 + \frac {c}{\sqrt {C _ {0}}}\right) \tau^ {2} k. \\ \end{array}
$$

When $\Delta _ { i } < 0$ , by using the same arguments, we can also obtain

$$
\begin{array}{l} \left| \sum_ {i = 1} ^ {m} \left(q _ {i} \log \left(1 + \Delta_ {i} / q _ {i}\right) + \left(1 - q _ {i}\right) \log \left(1 - \Delta_ {i} / \left(1 - q _ {i}\right)\right)\right) \right| \\ \leq \left(1 + \frac {c}{\sqrt {C _ {0}}}\right) \tau^ {2} k. \\ \end{array}
$$

Making use of the assumption $\tau \leq \varepsilon / ( C _ { 1 } \sqrt { k \log ( m / \delta ) } )$ , we observe

$$
\begin{array}{l} \left| t - \varepsilon \right| \\ = \left| \sum_ {i = 1} ^ {m} \left(q _ {i} \log \left(1 + \frac {\Delta_ {i}}{q _ {i}}\right) + (1 - q _ {i}) \log \left(1 - \frac {\Delta_ {i}}{1 - q _ {i}}\right)\right) \right| \\ \leq \frac {1 + \frac {c}{\sqrt {C _ {0}}}}{C _ {1} ^ {2}} \frac {\varepsilon^ {2}}{\log (m / \delta)} \leq \frac {1 + \frac {1 . 9}{3 . 9}}{1 . 9 5 ^ {2}} \times \frac {0 . 2 \varepsilon}{\log (2 / 0 . 0 5)} \\ <   0. 0 2 1 3 \varepsilon . \\ \end{array}
$$

Rearranging the inequality above gives

$$
0. 9 7 8 7 \leq \frac {t}{\varepsilon} \leq 1. 0 2 1 3.
$$

Furthermore, note that

$$
\begin{array}{l} \tau \leq \frac {\varepsilon}{C _ {1} \sqrt {k \log (m / \delta)}} \leq \frac {\varepsilon}{C _ {1} \sqrt {C _ {0}} \log (m / \delta)} \\ \leq \frac {0 . 2}{1 . 9 5 \times 3 . 9 \times \log (2 / 0 . 0 5)} <   0. 0 0 7 2. \\ \end{array}
$$

Hence,

$$
\begin{array}{l} \left| \log \frac {1 + \Delta_ {i} / q _ {i}}{1 - \Delta_ {i} / (1 - q _ {i})} \right| \leq \left| \log \frac {1 + \tau (1 - q _ {i})}{1 - \tau q _ {i}} \right| \\ \leq \frac {\tau}{1 - \tau q _ {i}} \leq \frac {\tau}{1 - 0 . 0 0 7 2} <   1. 0 0 7 3 \tau , \\ \end{array}
$$

which implies

$$
A \leq 1. 0 0 7 3 \tau . \tag {4.3}
$$

Combining the relations above implies that

$$
\begin{array}{l} \sigma^ {2} = \sum_ {i = 1} ^ {m} q _ {i} (1 - q _ {i}) \log^ {2} \frac {1 + \Delta_ {i} / q _ {i}}{1 - \Delta_ {i} / (1 - q _ {i})} \\ \leq 1. 0 0 7 3 ^ {2} \sum_ {i = 1} ^ {m} q _ {i} (1 - q _ {i}) \tau^ {2} \\ \leq 1. 5 0 9 \tau^ {2} k. \tag {4.4} \\ \end{array}
$$

Since $u h ( a / u )$ is a decreasing function in $u$ , from (4.4) it follows that

$$
\sigma^ {2} h \left(A t / \sigma^ {2}\right) \geq 1. 5 0 9 \tau^ {2} k h \left(A t / \left(1. 5 0 9 \tau^ {2} k\right)\right),
$$

we set $\tau ~ = ~ \varepsilon / ( C _ { 1 } \sqrt { k \log ( m / \delta ) } )$ . Recognizing $k \_ \geq$ $C _ { 0 } \log ( m / \delta )$ , it is clear that

$$
\begin{array}{l} \frac {A t}{1 . 5 0 9 \tau^ {2} k} \leq \frac {1 . 0 0 7 3}{1 . 5 0 9} \cdot \frac {t}{\tau k} = \frac {1 . 0 0 7 3 \times 1 . 9 5}{1 . 5 0 9} \frac {t}{\varepsilon} \sqrt {\frac {\log (m / \delta)}{k}} \\ \leq \frac {1 . 0 0 7 3 \times 1 . 9 5}{1 . 5 0 9} \times 1. 0 2 1 3 \times \frac {1}{3 . 9} \leq 0. 3 4 0 9. \\ \end{array}
$$

Finally, by taking advantage of the fact that $h ( u ) / u ^ { 2 }$ is a decreasing function in $u$ , we see that

$$
\begin{array}{l} \frac {\sigma^ {2} h (A t / \sigma^ {2})}{A ^ {2}} \\ \geq \frac {1 . 5 0 9 \tau^ {2} k h (A t / (1 . 5 0 9 \tau^ {2} k))}{A ^ {2}} \\ \geq \frac {1 . 5 0 9 \tau^ {2} k}{A ^ {2}} \cdot \frac {h (0 . 3 4 0 9)}{0 . 3 4 0 9 ^ {2}} \cdot \left(\frac {A t}{1 . 5 0 9 \tau^ {2} k}\right) ^ {2} \\ \geq 1. 5 0 9 \cdot \frac {h (0 . 3 4 0 9)}{0 . 3 4 0 9 ^ {2}} \cdot \frac {1}{1 . 5 0 9 ^ {2}} \cdot 1. 9 5 ^ {2} \cdot \frac {t ^ {2} \log (m / \delta)}{\varepsilon^ {2}} \\ \geq 1. 5 0 9 \cdot \frac {h (0 . 3 4 0 9)}{0 . 3 4 0 9 ^ {2}} \cdot \frac {1}{1 . 5 0 9 ^ {2}} \cdot 1. 9 5 ^ {2} \cdot (0. 9 7 8 7) ^ {2} \log \left(\frac {m}{\delta}\right) \\ \geq 1. 0 8 \log \left(\frac {m}{\delta}\right) \\ > \log \left(\frac {m}{\delta}\right). \\ \end{array}
$$

This completes the proof of inequality (4.2), and therefore, also completes the proof of Lemma 4.4. We now prove Lemma 4.1. The proof follows from Lemma 4.3 combined with Lemma 4.4.

Proof of Lemma 4.1. Without loss of generality, we assume $s _ { f } = 1$ . First, notice that if $k < C _ { 0 } \log ( m / \delta )$ ,

$$
\lambda > \frac {8 \sqrt {k \log (m / \delta)}}{\varepsilon} \geq \frac {8}{\sqrt {C _ {0}}} \cdot \frac {k}{\varepsilon} > \frac {2 k}{\varepsilon}.
$$

Therefore, Theorem 2.1 immediately implies the mechanism is $( \varepsilon , 0 )$ -private. Now we assume $k \geq C _ { 0 } \log ( m / \delta )$ . Throughout we use $J _ { k }$ to denote the random variable of the index of the kth smallest element in terms of the noisy count $x$ , and we use $g _ { j }$ to denote the noise added to the $j$ th element of $x$ . We also define $J _ { k } ^ { \prime }$ and $g _ { j } ^ { \prime }$ in terms of $x ^ { \prime }$ , respectively. For any given $J _ { k } = j$ and the noise $g _ { j } = g$ , we have

$$
\mathbb {P} (i \in \mathcal {M} ^ {\mathrm {o s}} (x)) = G ((x _ {j} + g - x _ {i}) / \lambda) := q _ {i}.
$$

Setting $q = q ( g ) = ( q _ { i } )$ for $1 \leq i \leq m$ and $i \neq j$ , and $S _ { j } = \{ s / \{ j \} \ : \ s \in S$ and $j \in s \}$ , then we have

$$
\mathbb {P} (\mathcal {M} ^ {\mathrm {o s}} (x) \in S, J _ {k} = j | g _ {j} = g) = \mathbb {P} (\mathcal {M} (q) \in S _ {j}).
$$

Making use of the fact that $\| x - x ^ { \prime } \| _ { \infty } \leq 1$ , we conclude that for any $i$ ,

$$
\left| \frac {x _ {j} + g - x _ {i}}{\lambda} - \frac {x _ {j} ^ {\prime} + g - x _ {i} ^ {\prime}}{\lambda} \right| \leq \frac {2}{\lambda}.
$$

By Lemma 4.3, this implies

$$
\begin{array}{l} | q - q ^ {\prime} | \\ \leq 2 q (1 - q) \mathrm {e} ^ {\left| \frac {x _ {j} + g - x _ {i}}{\lambda} - \frac {x _ {j} ^ {\prime} + g - x _ {i} ^ {\prime}}{\lambda} \right|}. \\ \left| \frac {x _ {j} + g - x _ {i}}{\lambda} - \frac {x _ {j} ^ {\prime} + g - x _ {i} ^ {\prime}}{\lambda} \right| \\ \leq 2 \mathrm {e} ^ {\left| \frac {2}{\lambda} \right|} \left| \frac {2}{\lambda} \right| q (1 - q) \leq \frac {4}{\lambda} \mathrm {e} ^ {2 \varepsilon / 8 \sqrt {k \log (m / \delta)}} q (1 - q) \\ \leq \frac {4}{\lambda} \mathrm {e} ^ {\varepsilon / 4 \sqrt {C _ {0}} \log (m / \delta)} q (1 - q) \\ <   \frac {4 . 0 1 4}{\lambda} q (1 - q). \\ \end{array}
$$

Hence $q$ is $4 . 0 1 4 / \lambda$ -close with respect to $q ^ { \prime }$ . We also notice that

$$
\lambda \geq \frac {8 \sqrt {k \log (m / \delta)}}{\varepsilon} = \frac {C _ {1} \sqrt {k \log (m / \delta)}}{1 . 9 5 \varepsilon / 8},
$$

which implies that $q$ is $\frac { 0 . 9 7 8 5 \varepsilon } { C _ { 1 } \sqrt { k \log ( m / \delta ) } }$ -close with respect to $q ^ { \prime }$ . We write $\mathbb { P } ( \mathcal { M } ^ { \mathrm { o s } } ( x ) \ \in \ S , J _ { k } \ = \ j | g _ { j } \ = \ g )$ and $\mathbb { P } ( \mathcal { M } ^ { \mathrm { o s } } ( x ^ { \prime } ) \in S , J _ { k } ^ { \prime } = j | g _ { j } ^ { \prime } = g )$ as $\mathbb { P } _ { x }$ and $\mathbb { P } _ { x ^ { \prime } }$ respectively. Applying Lemma 4.4 to $q$ and $q ^ { \prime }$ with parameters $\varepsilon$ and $\delta$ , we have

$$
\mathbb {P} _ {x} \leq \mathrm {e} ^ {0. 9 7 8 5 \varepsilon} \mathbb {P} _ {x ^ {\prime}} + \delta / m.
$$

Let $l _ { g _ { j } } ( g )$ stands for the probability when the $j$ th noise is taking value of $g$ . Noting that

$$
\lambda \geq \frac {8 \sqrt {C _ {0}} \log (m / \delta)}{\varepsilon} \geq \frac {8 \times 3 . 9 \times \log (2 / 0 . 0 5)}{\varepsilon} > \frac {1 1 5}{\varepsilon}.
$$

The conclusion is now one step away. To show that the algorithm is $( \varepsilon , \delta )$ -differentially private, note that

$$
\begin{array}{l} \mathbb {P} \left(\mathcal {M} ^ {\mathrm {o s}} (x) \in S\right) = \int \sum_ {j = 1} ^ {m} l _ {g _ {j}} (g) \mathbb {P} _ {x} d g \\ \leq \int \sum_ {j = 1} ^ {m} l _ {g _ {j}} (g) \left[ \mathrm {e} ^ {0. 9 7 8 5 \varepsilon} \mathbb {P} _ {x ^ {\prime}} + \delta / m \right] d g \\ \leq \mathrm {e} ^ {0. 9 7 8 5 \varepsilon} \int \sum_ {j = 1} ^ {m} \left(\frac {l _ {g _ {j}} (g)}{l _ {g _ {j} ^ {\prime}} (g)}\right) \cdot l _ {g _ {j} ^ {\prime}} (g) \cdot \mathbb {P} _ {x ^ {\prime}} d g + \delta \\ = \mathrm {e} ^ {0. 9 7 8 5 \varepsilon} \int \sum_ {j = 1} ^ {m} \left(e ^ {\frac {| g - x _ {j} ^ {\prime} | - | g - x _ {j} |}{\lambda}}\right) \cdot l _ {g _ {j} ^ {\prime}} (g) \cdot \mathbb {P} _ {x ^ {\prime}} d g + \delta \\ \leq \mathrm {e} ^ {0. 9 7 8 5 \varepsilon + \frac {1}{\lambda}} \int \sum_ {j = 1} ^ {m} l _ {g _ {j} ^ {\prime}} (g) \mathbb {P} _ {x ^ {\prime}} d g + \delta \\ \leq \mathrm {e} ^ {0. 9 7 8 5 \varepsilon + \frac {\varepsilon}{1 1 5}} \int \sum_ {j = 1} ^ {m} l _ {g _ {j} ^ {\prime}} (g) \mathbb {P} _ {x ^ {\prime}} d g + \delta \\ <   \mathrm {e} ^ {0. 9 9 \varepsilon} \mathbb {P} \left(\mathcal {M} ^ {\mathrm {o s}} \left(x ^ {\prime}\right) \in S\right) + \delta . \\ \end{array}
$$

Therefore, Theorem 2.2 is proved given the completion of the proof of Lemma 4.1.

# 5. Discussion

In this paper, we provide a theoretical study of the classical top- $k$ problem in the regime of differential privacy. We propose a fast, low-distortion, statistically accurate, and differentially private algorithm to tackle this question, which we refer to as the oneshot Laplace mechanism. We provide a novel coupling technique in proving its privacy without taking advantage of the advanced composition theorems, thereby circumventing the linear dependence on $k$ in the privacy loss compared to the existing results in the literature. We further provide the applications of the oneshot Laplace mechanism in multiple hypothesis testing and pairwise comparison. Our contributions in the theoretical framework have the potential to impact many essential areas in machine learning in both theory and practice.

This study leaves a number of open questions that we hope will inspire further work. Through the proof of differential privacy on the oneshot Laplace mechanism, there is nothing to prevent us from achieving better bounds for $\lambda$ . From a different angle, we wonder if the coupling technique would lead to tighter privacy analysis using other notions of privacy such as concentrated differential privacy, Rényi differential privacy, and Gaussian differential privacy (Dwork & Rothblum, 2016; Bun & Steinke, 2016; Mironov, 2017; Dong et al., 2021; Bu et al., 2020). Finally, an important direction for further research is to obtain sharp asymptotic properties of the oneshot mechanism and use the results to give a more comprehensive comparison between the oneshot Laplace mechanism and existing approaches in the literature.

# Acknowledgments

We are very grateful to Cynthia Dwork for encouragement and discussions on an early version of the manuscript. This work was supported in part by NSF through CAREER DMS-1847415 and CCF-1763314, a Facebook Faculty Research Award, and an Alfred Sloan Research Fellowship.

# References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, pp. 308–318. ACM, 2016.   
Bafna, M. and Ullman, J. The price of selection in differential privacy. arXiv preprint arXiv:1702.02970, 2017.   
Banerjee, S., Hegde, N., and Massoulié, L. The price of privacy in untrusted recommendation engines. In 2012 50th Annual Allerton Conference on Communication,

Control, and Computing (Allerton), pp. 920–927. IEEE, 2012.   
Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.   
Bu, Z., Dong, J., Long, Q., and Su, W. J. Deep learning with gaussian differential privacy. Harvard Data Science Review, 2020(23), 2020.   
Bun, M. and Steinke, T. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, pp. 635–658. Springer, 2016.   
Chen, Y., Fan, J., Ma, C., and Wang, K. Spectral method and regularized MLE are both optimal for top- $k$ ranking. arXiv preprint arXiv:1707.09971, 2017.   
Cho, G. E. and Meyer, C. D. Comparison of perturbation bounds for the stationary distribution of a markov chain. Linear Algebra and its Applications, 335(1-3):137–150, 2001.   
Dong, J., Durfee, D., and Rogers, R. Optimal differential privacy composition for exponential mechanisms and the cost of adaptivity. arXiv preprint arXiv:1909.13830, 2019.   
Dong, J., Roth, A., and Su, W. J. Gaussian differential privacy. Journal of the Royal Statistical Society, Series B, 2021. to appear.   
Durfee, D. and Rogers, R. M. Practical differentially private top-k selection with pay-what-you-get composition. In Advances in Neural Information Processing Systems, pp. 3527–3537, 2019.   
Dwork, C. and Roth, A. The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science. Now Publishers, 2014.   
Dwork, C. and Rothblum, G. N. Concentrated differential privacy. arXiv preprint arXiv:1603.01887, 2016.   
Dwork, C., Kenthapadi, K., McSherry, F., Mironov, I., and Naor, M. Our data, ourselves: Privacy via distributed noise generation. In Annual International Conference on the Theory and Applications of Cryptographic Techniques, pp. 486–503. Springer, 2006a.   
Dwork, C., McSherry, F., Nissim, K., and Smith, A. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pp. 265–284. Springer, 2006b.

Dwork, C., Rothblum, G. N., and Vadhan, S. Boosting and differential privacy. In Foundations of Computer Science (FOCS), 2010 51st Annual IEEE Symposium on, pp. 51– 60. IEEE, 2010.   
Dwork, C., Su, W., and Zhang, L. Private false discovery rate control. arXiv preprint arXiv:1511.03803, 2015.   
Dwork, C., Su, W. J., and Zhang, L. Differentially private false discovery rate control. arXiv preprint arXiv:1807.04209, 2018.   
Erdos, P. and Rényi, A. On the evolution of random graphs. Publ. Math. Inst. Hung. Acad. Sci, 5(1):17–60, 1960.   
Friedman, A. and Schuster, A. Data mining with differential privacy. In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 493–502. ACM, 2010.   
Hardt, M. and Roth, A. Beyond worst-case analysis in private singular vector computation. In Proceedings of the forty-fifth annual ACM symposium on Theory of computing, pp. 331–340. ACM, 2013.   
Ipsen, I. C. and Selee, T. M. Ergodicity coefficients defined by vector norms. SIAM Journal on Matrix Analysis and Applications, 32(1):153–200, 2011.   
Kairouz, P., Oh, S., and Viswanath, P. The composition theorem for differential privacy. IEEE Transactions on Information Theory, 63(6):4037–4049, 2017.   
Luce, R. D. Individual choice behavior: A theoretical analysis. Courier Corporation, 2012.   
McSherry, F. and Mironov, I. Differentially private recommender systems: Building privacy into the netflix prize contenders. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 627–636. ACM, 2009.   
McSherry, F. D. and Mironov, I. Differential privacy preserving recommendation, December 31 2013. US Patent 8,619,984.   
Mironov, I. Rényi differential privacy. In 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pp. 263–275. IEEE, 2017.   
Qin, Z., Yang, Y., Yu, T., Khalil, I., Xiao, X., and Ren, K. Heavy hitter estimation over set-valued data with local differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, pp. 192–203. ACM, 2016.   
Seneta, E. Perturbation of the stationary distribution measured by ergodicity coefficients. Advances in Applied Probability, 20(1):228–230, 1988.

Shen, Y. and Jin, H. Privacy-preserving personalized recommendation: An instance-based approach via differential privacy. In 2014 IEEE International Conference on Data Mining, pp. 540–549. IEEE, 2014.   
Steinke, T. and Ullman, J. Tight lower bounds for differentially private selection. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), pp. 552–563. IEEE, 2017.

# A. Supplementary Material

# A.1. Proof of Theorem 2.1

Proof of Theorem 2.1. We prove this theorem with a standard coupling argument. Without loss of generality, we assume $\begin{array} { r l r } { s _ { f } } & { { } = } & { 1 } \end{array}$ . Let $\boldsymbol { x } ~ = ~ \left( x _ { 1 } , \ldots , x _ { m } \right)$ and $\begin{array} { r l } { x ^ { \prime } } & { { } = } \end{array}$ $( x _ { 1 } ^ { \prime } , \ldots , x _ { m } ^ { \prime } )$ be adjacent, i.e., $\| x - x ^ { \prime } \| _ { \infty } \leq 1$ . Let $S$ be an arbitrary $k$ -subset of $\{ 1 , \ldots , m \}$ and $\mathcal { G }$ consist of all $\left( g _ { 1 } , \ldots , g _ { m } \right)$ such that $\mathcal { M } ^ { \mathrm { o s } }$ with input $x$ reports $S$ . Similarly we have $\mathcal { G } ^ { \prime }$ for $x ^ { \prime }$ . It is clear that $\mathcal { G } - 2 { \cdot } \mathbf { 1 } _ { S } \subset \mathcal { G } ^ { \prime }$ , where $\mathbf { 1 } _ { S } \in \{ 0 , 1 \} ^ { m }$ satisfies $\mathbf { 1 } _ { S } ( i ) = 1$ if and only if $i \in S$ . Here $\{ 0 , 1 \} ^ { m }$ stands for the set of $m$ -elements sets which only contain 0 and 1 as elements. Therefore, the standard coupling argument gives

$$
\begin{array}{l} \mathbb {P} (\mathcal {M} ^ {\mathrm {o s}} (x) = S) = \int_ {\mathcal {G}} \frac {1}{2 ^ {m} \lambda^ {m}} \mathrm {e} ^ {- \frac {\| g \| _ {1}}{\lambda}} d g \\ \geq \int_ {\mathcal {G} ^ {\prime} + 2 \cdot \mathbf {1} _ {S}} \frac {1}{2 ^ {m} \lambda^ {m}} \mathrm {e} ^ {- \frac {\| g \| _ {1}}{\lambda}} d g = \int_ {\mathcal {G} ^ {\prime}} \frac {1}{2 ^ {m} \lambda^ {m}} \mathrm {e} ^ {- \frac {\| g + 2 \cdot \mathbf {1} _ {S} \| _ {1}}{\lambda}} d g \\ \geq \int_ {\mathcal {G} ^ {\prime}} \frac {\mathrm {e} ^ {- 2 k / \lambda}}{2 ^ {m} \lambda^ {m}} \mathrm {e} ^ {- \frac {\| g \| _ {1}}{\lambda}} d g = \mathrm {e} ^ {- \varepsilon} \mathbb {P} (\mathcal {M} ^ {\mathrm {o s}} (x ^ {\prime}) = S). \\ \end{array}
$$

On the opposite side, we have $\mathbb { P } ( \mathcal { M } ^ { \mathrm { o s } } ( x ) ~ = ~ S ) ~ \leq$ $\mathrm { e } ^ { \varepsilon } \mathbb { P } ( \mathcal { M } ^ { \mathrm { o s } } ( x ^ { \prime } ) = S )$ , which completes the proof since $S$ is arbitrary. □

# A.2. Proof of Theorem 2.3

We begin by proving the following lemma.

Lemma A.1. Let $X$ and $Y$ be independent identically $\mathrm { L a p } ( \lambda )$ distributions and $Z = X - Y$ , then the density function of random variable $Z$ has the form

$$
f _ {Z} (z) = \frac {\lambda + | z |}{4 \lambda^ {2}} \cdot \exp \left(- \frac {| z |}{\lambda}\right).
$$

Proof of Lemma A.1. Notice that

$$
f _ {Z} (z) = \int_ {- \infty} ^ {\infty} f _ {X} (x) f _ {Y} (x - z) d x,
$$

by applying the convolution formula above, when $z \geq 0$ we have

$$
\begin{array}{l} 4 \lambda^ {2} f _ {Z} (z) \\ = \int_ {- \infty} ^ {\infty} \exp \left(- \frac {| x |}{\lambda}\right) \exp \left(- \frac {| x - z |}{\lambda}\right) d x \\ = \int_ {- \infty} ^ {0} \exp \left(\frac {x}{\lambda}\right) \exp \left(\frac {x - z}{\lambda}\right) d x + \\ \int_ {0} ^ {z} \exp \left(- \frac {x}{\lambda}\right) \exp \left(\frac {x - z}{\lambda}\right) d x + \\ \int_ {z} ^ {\infty} \exp \left(- \frac {x}{\lambda}\right) \exp \left(- \frac {x - z}{\lambda}\right) d x + \\ = (\lambda + z) \exp \left(- \frac {z}{\lambda}\right). \\ \end{array}
$$

Therefore, with $z \geq 0$ , we have

$$
f _ {Z} (z) = \frac {\lambda + z}{4 \lambda^ {2}} \exp \left(- \frac {z}{\lambda}\right).
$$

Since the pdfs of random variables $X$ and $Y$ are symmetric around the origin, the pdf of $Z$ must be symmetric around the origin. From this we get that

$$
f _ {Z} (z) = \frac {\lambda + | z |}{4 \lambda^ {2}} \exp \left(- \frac {| z |}{\lambda}\right).
$$

![](images/0b3fccc59511db6d59bac6fc2c2a14ae9cd00130ed1ee273ca4af64b17ff6dde.jpg)

With the lemma above, we start to prove Theorem 2.3. Let $g _ { 1 } , g _ { 2 } , \cdots , g _ { m }$ be i.i.d. $\mathrm { L a p } ( \lambda )$ distributions. We define the event

$A = \{ \mathcal { M } _ { o n e s h o t }$ reports the index set of true top- $k$ elements}

and consider the extreme case when $y _ { i }$ ’s follow the exact same order as $f _ { i } ( D )$ ’s, i.e.,

$$
f _ {(1)} (D) + g _ {1} \leq f _ {(2)} (D) + g _ {2} \leq \dots \leq f _ {(m)} (D) + g _ {m}.
$$

Thus we can lower bound $P ( A )$ by the extreme case above,

$$
\begin{array}{l} \mathbb {P} (A) \geq \mathbb {P} \left(f _ {(1)} (D) + g _ {1} \leq \dots \leq f _ {(m)} (D) + g _ {m}\right) \\ = \mathbb {P} (f _ {(1)} (D) + g _ {1} \leq f _ {(2)} (D) + g _ {2}, \\ f _ {(2)} (D) + g _ {2} \leq f _ {(3)} (D) + g _ {3}, \dots , \\ f _ {(m - 1)} (D) + g _ {m - 1} \leq f _ {(m)} (D) + g _ {m}) \\ = \mathbb {P} \left(g _ {1} - g _ {2} \leq f _ {(2)} (D) - f _ {(1)} (D), \right. \\ g _ {2} - g _ {3} \leq f _ {(3)} (D) - f _ {(2)} (D), \dots , \\ g _ {m - 1} - g _ {m} \leq f _ {(m)} (D) - f _ {(m - 1)} (D)) \\ \geq \mathbb {P} (g _ {1} - g _ {2} \leq \Delta , g _ {2} - g _ {3} \leq \Delta , \dots , \\ \left. g _ {m - 1} - g _ {m} \leq \Delta\right). \\ \end{array}
$$

Recall the Bonferroni lower bound that for events $X _ { 1 } , \cdots , X _ { n }$ , we have

$$
\mathbb {P} \left(X _ {1}, \dots , X _ {n}\right) \geq \max  \left\{0, 1 - \sum_ {i = 1} ^ {m} \left(1 - \mathbb {P} \left(X _ {i}\right)\right) \right\}.
$$

Combining the calculation above, we have

$$
\begin{array}{l} \mathbb {P} (A) \geq 1 - \sum_ {i = 1} ^ {m - 1} \left(1 - \mathbb {P} \left(g _ {i} - g _ {i + 1} \leq \Delta\right)\right) \\ = 1 - \sum_ {i = 1} ^ {m - 1} \mathbb {P} \left(g _ {i} - g _ {i + 1} \leq - \Delta\right) \\ = 1 - \sum_ {i = 1} ^ {m - 1} \int_ {- \infty} ^ {- \Delta} f _ {Z} (z) d z \\ = 1 - \frac {(m - 1) (2 \lambda + \Delta) \mathrm {e} ^ {- \Delta / \lambda}}{4 \lambda}, \\ \end{array}
$$

this completes the proof of Theorem 2.3.

# A.3. Proof of Theorem 3.5

Proof of Theorem 3.5. We only need to prove the sensitivity part $\begin{array} { r } { s = \frac { 2 } { d \underline { { L } } ( 1 , - \underline { { \rho } } ) } } \end{array}$ . For two adjacent databases $D$ , $D ^ { \prime }$ where they only differ in only one data item, we assume their corresponding sufficient statistics are $\textbf {  { y } }$ and $\tilde { y }$ . To capture the definition of adjacent databases, we assume one sample yi0,j0 $y _ { i _ { 0 } , j _ { 0 } } ^ { ( l _ { 0 } ) }$ of the edge $( i _ { 0 } , j _ { 0 } )$ in database $D$ and $D ^ { \prime }$ is different. Without loss of generality, we assume that yi0,j0 $y _ { i _ { 0 } , j _ { 0 } } ^ { ( l _ { 0 } ) } = 0$ in $D$ and y˜i0,j0 $\tilde { y } _ { i _ { 0 } , j _ { 0 } } ^ { ( l _ { 0 } ) } = 1$ in $D ^ { \prime }$ . Note that

$$
P _ {i j} = \left\{ \begin{array}{l l} \frac {1}{d} y _ {i, j} & \text {i f} (i, j) \in E, \\ 1 - \frac {1}{d} \sum_ {k: (i, k) \in E} y _ {i, k} & \text {i f} i = j, \\ 0 & \text {o t h e r w i s e}, \end{array} \right.
$$

two transition matrices $_ { r }$ and $\tilde { P }$ only differ in four elements in positions $( i _ { 0 } , j _ { 0 } ) , ( j _ { 0 } , i _ { 0 } ) , ( i _ { 0 } , i _ { 0 } )$ $( i _ { 0 } , j _ { 0 } )$ $( i _ { 0 } , i _ { 0 } )$ and $( j _ { 0 } , j _ { 0 } )$ . To find the maximum possible value of $| | \boldsymbol { P } - \tilde { \boldsymbol { P } } | | _ { \infty }$ , for $i \neq i _ { 0 }$ and $i \neq j _ { 0 }$ ,

$$
\sum_ {j = 1} ^ {m} \left| P _ {i j} - \tilde {P} _ {i j} \right| = 0.
$$

In the case when $i = i _ { 0 }$ ,

$$
\begin{array}{l} \sum_ {j = 1} ^ {m} \left| P _ {i j} - \tilde {P} _ {i j} \right| \\ = \frac {\left| y _ {i _ {0} , j _ {0}} - \tilde {y} _ {i _ {0} , j _ {0}} \right|}{d} + \left| P _ {i _ {0} i _ {0}} - \tilde {P} _ {i _ {0} i _ {0}} \right| \\ = \frac {\left| y _ {i _ {0} , j _ {0}} - \tilde {y} _ {i _ {0} , j _ {0}} \right|}{d} + \\ \left| \left(1 - \frac {1}{d} \sum_ {k: (i _ {0}, k) \in E} y _ {i _ {0}, k}\right) - \left(1 - \frac {1}{d} \sum_ {k: (i _ {0}, k) \in E} \tilde {y} _ {i _ {0}, k}\right) \right| \\ = \frac {\left| y _ {i _ {0} , j _ {0}} - \tilde {y} _ {i _ {0} , j _ {0}} \right|}{d} + \frac {1}{d} \left| y _ {i _ {0}, j _ {0}} - \tilde {y} _ {i _ {0}, j _ {0}} \right| \\ = \frac {2}{d} | y _ {i _ {0}, j _ {0}} - \tilde {y} _ {i _ {0}, j _ {0}} | \\ = \frac {2}{d L}. \\ \end{array}
$$

get When $\begin{array} { r } { \sum _ { j = 1 } ^ { m } \left| \boldsymbol { \bar { P } } _ { i j } - \boldsymbol { \tilde { P } } _ { i j } \right| = \frac { 2 } { d L } } \end{array}$ $i = j _ { 0 }$ , we can follow the exact same calculation and . Therefore,

$$
\left| \left| \boldsymbol {P} - \tilde {\boldsymbol {P}} \right| \right| _ {\infty} = \max  _ {1 \leq i \leq m} \sum_ {j = 1} ^ {m} \left| P _ {i j} - \tilde {P} _ {i j} \right| = \frac {2}{d L}.
$$

By the definition of sensitivity, we have s = 2dL(1 ρ) . $\begin{array} { r } { s = \frac { 2 } { d L \left( 1 - \rho \right) } } \end{array}$

# A.4. Proof of Lemma 4.3

Proof of Lemma 4.3. By mean value theorem, there exists a point $\tilde { z }$ between $z$ and $z ^ { \prime }$ such that

$$
| g (\tilde {z}) | = \left| \frac {G (z ^ {\prime}) - G (z)}{z ^ {\prime} - z} \right|,
$$

where $g$ denotes the density function of the standard Laplace distribution. Hence, we only need to prove

$$
\frac {| g (\tilde {z}) |}{G (z) (1 - G (z))} \leq 2 \mathrm {e} ^ {| z ^ {\prime} - z |}.
$$

Now we prove this inequality in different cases.

Case 1: when $\operatorname* { m a x } ( z , z ^ { \prime } ) \leq 0$ . In this case,

$$
\frac {| g (\tilde {z}) |}{G (z) (1 - G (z))} = \frac {\mathrm {e} ^ {\tilde {z} - z}}{1 - \frac {1}{2} \mathrm {e} ^ {z}} \leq 2 \mathrm {e} ^ {| \tilde {z} - z |} \leq 2 \mathrm {e} ^ {| z ^ {\prime} - z |}.
$$

Case 2: when $\operatorname* { m i n } ( z , z ^ { \prime } ) \ge 0$ . With a similar argument in Case 1, we have

$$
\frac {| g (\tilde {z}) |}{G (z) (1 - G (z))} = \frac {\mathrm {e} ^ {z - \tilde {z}}}{1 - \frac {1}{2} \mathrm {e} ^ {- z}} \leq 2 \mathrm {e} ^ {| z - \tilde {z} |} \leq 2 \mathrm {e} ^ {| z ^ {\prime} - z |}.
$$

Case 3: when $\operatorname* { m i n } ( z , z ^ { \prime } ) < 0 < \operatorname* { m a x } ( z , z ^ { \prime } )$ . The triangle inequality gives

$$
\frac {| g (\tilde {z}) |}{G (z) (1 - G (z))} = \frac {\mathrm {e} ^ {| z | - | \tilde {z} |}}{1 - \frac {1}{2} \mathrm {e} ^ {- | z |}} \leq 2 \mathrm {e} ^ {| z - \tilde {z} |} \leq 2 \mathrm {e} ^ {| z ^ {\prime} - z |}.
$$

In summary, combining all the cases above gives the lemma. □

# A.5. Proof of Lemma 4.5

The proof of the Lemma 4.5 is based on the classical Bennett’s inequality stated below.

Bennett’s inequality: Let $Z _ { 1 } \ldots , Z _ { n }$ be independent random variables with all means being zero. In addition, assume $| Z _ { i } | ~ \le ~ a$ almost surely for all $i$ . Denoting by $\begin{array} { r } { \sigma ^ { 2 } = \sum _ { i = 1 } ^ { n } { \mathrm { V a r } } ( Z _ { i } ) } \end{array}$ , we have

$$
\mathbb {P} \left(\sum_ {i = 1} ^ {n} Z _ {i} > t\right) \leq \exp \left(- \frac {\sigma^ {2} h (a t / \sigma^ {2})}{a ^ {2}}\right)
$$

for any $t \geq 0$ , where $h ( u ) = ( 1 + u ) \log ( 1 + u ) - u .$ .

Proof of Lemma 4.5. By the Bennett’s inequality stated above, we have

$$
\begin{array}{l} \mathbb {P} (\sum_ {i} Z _ {i} \leq k) \\ = \mathbb {P} \left(\sum_ {i} Z _ {i} - \sum_ {i} q _ {i} \leq - t k\right) \leq \mathrm {e} ^ {- \sigma^ {2} h \left(t k / \sigma^ {2}\right)}, \\ \end{array}
$$

Note thaspect to $t ) k$ i=1 i. In this case, we have $\textstyle \sum _ { i = 1 } ^ { m } q _ { i }$ $\mathbb { P } ( \sum _ { i } Z _ { i } \leq k )$ is a decreasing we can assume $\begin{array} { r } { \sigma ^ { 2 } = \sum _ { i = 1 } ^ { m } q _ { i } ( 1 - \overline { { q _ { i } } } ) \leq ( 1 + t ) k } \end{array}$ $\textstyle \sum _ { i = 1 } ^ { m } q _ { i } = ( 1 +$

Making use of the fact that $\sigma ^ { 2 } h ( t k / \sigma ^ { 2 } )$ is a monotonically decreasing function with respect to $\sigma ^ { 2 }$ gives

$$
\begin{array}{l} \mathbb {P} \left(\sum_ {i} Z _ {i} \leq k\right) \leq \mathrm {e} ^ {- \sigma^ {2} h \left(t k / \sigma^ {2}\right)} \\ \leq \exp \left(- (1 + t) k h \left(\frac {t}{t + 1}\right)\right). \\ \end{array}
$$

Hence, the first part of Lemma 4.5 is proved. In order to prove the second part of the lemma, we need to take advantage of the conclusion from the first part. Note that $h ( u ) / u ^ { 2 }$ is a decreasing function of $u$ , by setting $t = c \sqrt { \frac { \log ( m / \delta ) } { k } } \leq$ √cC0 , we get $\frac { c } { \sqrt { C _ { 0 } } }$

$$
\begin{array}{l} (1 + t) k h \left(\frac {t}{t + 1}\right) \\ \geq (1 + t) k \left(\frac {t}{t + 1}\right) ^ {2} \frac {h (\frac {c}{c + \sqrt {C _ {0}}})}{(\frac {c}{c + \sqrt {C _ {0}}}) ^ {2}} \\ \geq k \left(c \sqrt {\frac {\log (m / \delta)}{k}}\right) ^ {2} \frac {1}{1 + \frac {c}{\sqrt {C _ {0}}}} \frac {h (\frac {c}{c + \sqrt {C _ {0}}})}{(\frac {c}{c + \sqrt {C _ {0}}}) ^ {2}} \\ \geq 1. 0 9 9 \log \left(\frac {m}{\delta}\right) \\ > \log \left(\frac {m}{\delta}\right). \\ \end{array}
$$

Therefore, when $\textstyle \sum _ { i = 1 } ^ { m } q _ { i } \geq ( 1 + t ) k$ , we have

$$
\mathbb {P} (\sum_ {i} Z _ {i} \leq k) \leq \exp \left(- (1 + t) k h \left(\frac {t}{t + 1}\right)\right) \leq \frac {\delta}{m}.
$$

![](images/a92a1a6128786fbf3d7246c29e09fc580707dc02969edacf6c1d8a995fe0ff72.jpg)