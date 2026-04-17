# BATCH NORMALIZATION ORTHOGONALIZES REPRESENTATIONS IN DEEP RANDOM NETWORKS

HADI DANESHMAND, AMIR JOUDAKI, AND FRANCIS BACH

Abstract. This paper underlines a subtle property of batch-normalization (BN): Successive batch normalizations with random linear transformations make hidden representations increasingly orthogonal across layers of a deep neural network. We establish a non-asymptotic characterization of the interplay between depth, width, and the orthogonality of deep representations. More precisely, under a mild assumption, we prove that the deviation of the representations from orthogonality rapidly decays with depth up to a term inversely proportional to the network width. This result has two main implications: 1) Theoretically, as the depth grows, the distribution of the representation –after the linear layers– contracts to a Wasserstein-2 ball around an isotropic Gaussian distribution. Furthermore, the radius of this Wasserstein ball shrinks with the width of the network. 2) In practice, the orthogonality of the representations directly influences the performance of stochastic gradient descent (SGD). When representations are initially aligned, we observe SGD wastes many iterations to orthogonalize representations before the classification. Nevertheless, we experimentally show that starting optimization from orthogonal representations is sufficient to accelerate SGD, with no need for BN.

# 1. Introduction

Batch Normalization (BN) (Ioffe and Szegedy, 2015) enhances training across a wide range of deep network architectures and experimental setups (He et al., 2016; Huang et al., 2017; Silver et al., 2017). The practical success of BN has inspired research into the underlying benefits of BN (Santurkar et al., 2018; Karakida et al., 2019; Arora et al., 2019b; Bjorck et al., 2018). In particular, it shown that BN influences first-order optimization methods by avoiding the rank collapse in deep representation (Daneshmand et al., 2020), direction-length decoupling of optimization (Kohler et al., 2018), influencing the learning rate of the steepest descent (Arora et al., 2019b; Bjorck et al., 2018), and smoothing the optimization objective function (Santurkar et al., 2018; Karakida et al., 2019). However, the benefits of BN go beyond its critical role in optimization. For example, Frankle et al. (2020) shows that BN networks with random weights also achieve surprisingly high performance after only minor adjustments of their weights. This striking result motivates us to study the representational power of random networks with BN.

We study hidden representations across layers of a laboratory random BN with linear activations. Consider a batch of samples passing through consecutive BN and linear layers with Gaussian weights. The representations of these samples are perturbed by each random linear transformation, followed by a non-linear BN. At first glance, the deep representations appear unpredictable after many stochastic and non-linear transformations. Yet, we show that these transformations orthogonalize the representations. To prove this statement, we introduce the notion of “orthogonality gap”, defined in Section 3, to quantify the deviation of representations from a perfectly orthogonal representation. Then, we prove that the orthogonality gap decays exponentially with the network

depth and stabilizes around a term inversely related to the network width. More precisely, we prove

$$
\mathbb {E} \left[ \mathrm {o r t h o g o n a l i t y g a p} \right] = \mathcal {O} \left((1 - \alpha) ^ {\mathrm {d e p t h}} + \frac {\mathrm {b a t c h s i z e}}{\alpha \sqrt {\mathrm {w i d t h}}}\right)
$$

holds for $\alpha > 0$ that is an absolute constant under a mild assumption. In probability theoretic terms, we prove stochastic stability of the Markov chain of hidden representations (Kushner, 1967; Kushner and Yin, 2003; Khasminskii, 2011). The orthogonality of deep representations allows us to prove that the distribution of the representations after linear layers contracts to a Wasserstein-2 ball around isotropic Gaussian distribution as the network depth grows. Moreover, the radius of the ball is inversely proportional to the network width. Omitting details, we prove the following bound holds:

$$
\mathrm {W a s s e r s t e i n} _ {2} (\mathrm {r e p r e s e n t a t i o n s , G a u s s i a n}) ^ {2} = \mathcal {O} \left((1 - \alpha) ^ {\mathrm {d e p t h}} (\mathrm {b a t c h s i z e}) + \frac {\mathrm {b a t c h s i z e}}{\alpha \sqrt {\mathrm {w i d t h}}}\right).
$$

The above equation shows how depth, width, and batch size, interact with the law of the representations. Since the established rate is exponential with depth, the distribution of the representations stays in a Wasserstein ball around isotropic Gaussian distribution after a few layers. Thus, BN not only stabilizes the distribution of the representations, which is its main promise (Ioffe and Szegedy, 2015), but also enforces Gaussian isotropic distribution in deep layers.

There is growing interest in bridging the gap between neural networks, as the most successful parametric methods for learning, and Gaussian processes and kernel methods, as well-understood classical models for learning (Jacot et al., 2018; Matthews et al., 2018; Lee et al., 2019; Bietti and Mairal, 2019; Huang et al., 2014). This link is inspired by studying random neural networks in the asymptotic regime of infinite width. The seminal work by Neal (1996) sparks a singlelayer network resembles a Gaussian process as its width goes to infinity. However, increasing the depth may significantly shift the distribution of the representations away from Gaussian (Ioffe and Szegedy, 2015). This distributional shift breaks the link between Gaussian processes and deep neural networks. To ensure Gaussian representations, Matthews et al. (2018) suggests increasing the width of the network proportional to the network depth. Here, we show that BN ensures Gaussian representations even for deep networks with finite width. This result bridges the link between deep neural networks and Gaussian process in the regime of finite width. Many studies rely on deep Gaussian representations in an infinite width setting (Yang et al., 2019; Schoenholz et al., 2017; Pennington et al., 2018; Klambauer et al., 2017; De Palma et al., 2019). Our nonasymptotic Gaussian approximation can be incorporated into their analysis to extend these results to the regime of finite width.

Since training starts from random networks, representations in these networks directly influence training. Hence, recent theoretical studies has investigated the interplay between initial hidden representations and training (Daneshmand et al., 2020; Bjorck et al., 2018; Frankle et al., 2020; Schoenholz et al., 2017; Saxe et al., 2014; Bahri et al., 2020). In particular, it is shown that hidden representations in random networks without BN become correlated as the network grows in depth, thereby drastically slows training (Daneshmand et al., 2020; He et al., 2016; Bjorck et al., 2018; Saxe et al., 2014). On the contrary, we prove that deep representations in networks with $B N$ are almost orthogonal. We experimentally validate that initial orthogonal representations can save training time that would otherwise be needed to orthogonalize them. By proposing a novel initialization scheme, we ensures the orthogonality of hidden representations; then, we show that such an initialization effectively avoids the training slowdown with depth for vanilla networks, with no need for BN. This observation further motivates studying the inner workings of BN to replace or improve it in deep neural networks.

Theoretically, we made the following contributions:

(1) For MLPs with batch normalization, linear activation, and Gaussian i.i.d. weights, we prove that representations across layers become increasingly orthogonal up to a constant inversely proportional to the network width.   
(2) Leveraging the orthogonality, we prove that the distribution of the representations contracts to a Wasserstein ball around a Gaussian distribution as the depth grows. Up to the best of our knowledge, this is the first non-asymptotic Gaussian approximation for deep neural networks with finite width.

Experimentally, we made the following contribution1:

(3) Inspired by our theoretical understanding, we propose a novel weight initialization for standard neural networks that ensure orthogonal representations without BN. Experimentally, we show that this initialization effectively avoids training slowdown with depth in the absence of BN.

# 2. Preliminaries

2.1. Notations. Akin to Daneshmand et al. (2020), we focus on a Multi-Layer Perceptron (MLP) with batch normalization and linear activation. Theoretical studies of linear networks is a growing research area (Saxe et al., 2014; Daneshmand et al., 2020; Bartlett et al., 2019; Arora et al., 2019a). When weights initialized randomly, linear and non-linear networks share similar properties (Daneshmand et al., 2020). For ease of analysis, we assume activations are linear. Yet, we will argue that our findings extend to non-linear in Appendix F.

We use $n$ to denote batch size, and $d$ to denote the width across all layers, which we further assume is larger than the batch size $d \geq n$ . Let $H _ { \ell } \in \mathbb { R } ^ { d \times n }$ denote $n$ sample representations in layer $\ell$ , with $H _ { 0 } \in \mathbb { R } ^ { d \times n }$ corresponding to $n$ input samples in the batch with $d$ features. Successive representations are connected by Gaussian weight matrices $W _ { \ell } \sim { \mathcal { N } } ( 0 , I _ { d } / d )$ as

$$
H _ {\ell + 1} = \frac {1}{\sqrt {d}} B N \left(W _ {\ell} H _ {\ell}\right), \quad B N (M) = \operatorname {d i a g} \left(M M ^ {\top}\right) ^ {- 1 / 2} M, \tag {1}
$$

where $\mathrm { d i a g } ( M )$ zeros out off-diagonal elements of its input matrix, and the scaling factor $1 / \sqrt { d }$ ensures that all matrices $\{ H _ { k } \} _ { k }$ have unit Frobenius norm (see Appendix 2). The BN function in Eq. (1) differs slightly from the commonly used definition for BN as the mean correction is omitted. However, Daneshmand et al. (2020) showes this difference does not change the network properties qualitatively. Readers can find all the notations in Appendix Table 2.

2.2. The linear independence of hidden representations. Daneshmand et al. (2020) observe that if inputs are linearly independent, then their hidden representations remain linearly independent in all layers as long as $d = \Omega ( n ^ { 2 } )$ . Under technical assumptions, Daneshmand et al. (2020) establishes a lower-bound on the average of the rank of hidden representations over infinite layers. Based on this study, we assume that the linear independence holds and build our analysis upon that. This avoids restating the technical assumptions of Daneshmand et al. (2020) and also further technical refinements of their theorems. The next assumption presents the formal statement of the linear independence property.

Assumption $A _ { 1 } ( \alpha , \ell )$ . There exists an absolute positive constant $\alpha$ such that the minimum singular value of $H _ { k }$ is greater than (or equal to) $\alpha$ for all $k = 1 , \dots , \ell$ .

The linear independence of the representations is a shared property across all layers. However, the representations constantly change when passing through random layers. In this paper, we mathematically characterize the dynamics of the representations across layers.

# 3. Orthogonality of deep representations

3.1. A warm-up observation. To illustrate the difference between BN and vanilla networks, we compare hidden representations of two input samples across the layers of these networks. Figure 1 plots the absolute value of cosine similarity of these samples across layers. This plot shows a stark contrast between vanilla and BN networks: While representations become increasingly orthogonal across layers of a BN network, they become increasingly aligned in a vanilla network. We used highly correlated inputs for the BN network and perfectly orthogonal inputs for the Vanilla network to contrast their differences. While the hidden representation of vanilla networks has been theoretically studied (Daneshmand et al., 2020; Bjorck et al., 2018; Saxe et al., 2014), up to the best of our knowledge, there is no theoretical analysis of BN networks. In the following section, we formalize and prove this orthogonalizing property for BN networks.

![](images/8dbad6d2e9c3b629d2c3fcade13e5f0ded5b8698d2c5b514e32512e7509f01e2.jpg)  
Figure 1. Orthogonality: BN vs. vanilla networks. The horizontal axis shows the number of layers, and the vertical axis shows the absolute value of cosine similarity between two samples across the layers ( $d = 3 2$ ). Mean and 95% confidence intervals of 20 independent runs.

3.2. Theoretical analysis. The notion of orthogonality gap plays a central role in our analysis. Given the hidden representation $H \in \mathbb { R } ^ { d \times n }$ , matrix $H ^ { \mid } H$ constitutes inner products between representations of different samples. Note that $H ^ { \mathrm { ~ l ~ } } H \in \mathbb { R } ^ { n \times n }$ is different form the covariance matrix $H H ^ { \top } / n \in \mathbb { R } ^ { d \times d }$ . The orthogonality gap of $H$ is defined as the deviations of $H ^ { \mid } H$ from identity matrix, after proper scaling. More precisely, define $V : \mathbb { R } ^ { d \times n } \backslash \mathbf { 0 }  \mathbb { R } _ { + }$ as

$$
V (H) := \left\| \left(\frac {1}{\| H \| _ {F} ^ {2}}\right) H ^ {\top} H - \left(\frac {1}{\| I _ {n} \| _ {F} ^ {2}}\right) I _ {n} \right\| _ {F}. \tag {2}
$$

The following theorem establishes a bound on the orthogonality of representation in layer $\ell$ .

Theorem 1. Under Assumption $ { \mathcal { A } } _ { 1 } ( \alpha , \ell )$ , the following holds:

$$
\mathbb {E} \left[ V \left(H _ {\ell + 1}\right) \right] \leq 2 \left(1 - \frac {2}{3} \alpha\right) ^ {\ell} + \frac {3 n}{\alpha \sqrt {d}}. \tag {3}
$$

Assumption $\mathcal { A } _ { 1 }$ is studied by Daneshmand et al. (2020) who note that $\mathcal { A } _ { 1 } ( \alpha , \infty )$ holds as long as $d = \Omega ( n ^ { 2 } )$ . If $\mathcal { A } _ { 1 }$ does not hold, one can still prove that there is a function of representations that decays with depth up to a constant (see Appendix C).

The above result implies that BN is an approximation algorithm for orthogonalizing the hidden representations. If we replace $\mathrm { d i a g } ( M ) ^ { - 1 }$ by $( M ) ^ { - 1 }$ in BN formula, in Eq. (1), then all the hidden representation will become exactly orthogonal. However, computing the inverse of a non-diagonal $d \times d$ matrix is computationally expensive, which must repeat for all layers throughout training,

and differentiation must propagate back through this inversion. The diagonal approximation in BN significantly reduces the computational complexity of the matrix inversion. Since the orthogonality gap decays at an exponential rate with depth, the approximate orthogonality is met after only a few layers. Interestingly, this yields a desirable cost-accuracy trade-off: For a larger width,√ the orthogonality is more accurate, due to term $1 / \sqrt { d }$ in the orthogonality gap, and also the computational gain is more significant.

From a different angle, Theorem 1 proves the stochastic stability of the Markov chain of hidden representations. In expectation, stochastic processes may obey an inherent Lyapunov-type of stability (Kushner, 1967; Kushner and Yin, 2003; Khasminskii, 2011). One can analyze the mixing and hitting times of Markov chains based on the stochastic stability of Markov chains,(Kemeny and Snell, 1976; Eberle, 2009). In our analysis, the orthogonality gap is a Lyapunov function characterizing the stability of the chain of hidden representations. This stability opens the door to more theoretical analysis of this chain, such as studying mixing and hitting times. Since these theoretical properties may not be of interest to the machine learning community, we focus on the implications of these results for understanding the underpinnings of neural networks.

It is helpful to compare the orthogonality gap in BN networks to studies on vanilla networks (Bjorck et al., 2018; Daneshmand et al., 2020; Saxe et al., 2014). The function implemented by a vanilla linear network is a linear transformation as the product of random weight matrices. The spectral properties of the product of i.i.d. random matrices are the subject of extensive studies in probability theory (Bougerol et al., 2012). Invoking these results, one can readily check that the orthogonality gap of the hidden representations in vanilla networks rapidly increases since the representations converge to a rank one matrix after a proper scaling (Daneshmand et al., 2020).

3.3. Experimental validations. Our experiments presented in Fig. 2a validate the exponential decay rate of $V$ with depth. In this plot, we see that $\log ( V _ { \ell } )$ linearly decreases for $\ell = 1 , \ldots , 2 0$ , then it wiggles around a small constant. Our experiments in Fig. 2b suggest that the $\mathcal { O } ( 1 / \sqrt { d } )$ dependency on width is almost tight. Since $V ( H _ { \ell } )$ rapidly converges to a ball, the average of $V ( H _ { \ell } )$ over layers estimates the radius of this ball. This plot shows how the average of $V ( H _ { \ell } )$ , over 500 layers, changes with the network width. Since the scale is logarithmic in this plot, slope is close to√ $- 1 / 2$ in logarithmic scale, validating the $\mathcal { O } ( 1 / \sqrt { d } )$ dependency implied by Theorem 1.

![](images/f0237b5fde742b8a0b6737fd329fa67a7112a4923779ca4420aaeb34b330d298.jpg)  
(a) Orthogonality vs. depth

![](images/5c6d62debdc17cde759b2ee385f2ffd1aec76d54c06c8c8c1672df1cbeab1e20.jpg)  
(b) Orthogonality vs. width   
horizontally . Right: Figure 2. Orthogonality gap vs. depth and width. Left: $\begin{array} { r } { \log ( \frac { 1 } { 5 0 0 } \sum _ { \ell = 1 } ^ { 5 0 0 } V ( H _ { \ell } ) ) } \end{array}$ vertically versus $d$ horizontally. The slop of the right $\log ( V ( H _ { \ell } ) )$ vertically versus $\ell$ plot is $- 0 . 4 6$ . Mean and $9 5 \%$ confidence interval of 20 independent runs.

# 4. Gaussian approximation

4.1. Orthogonality yields Gaussian approximation. The established result on the orthogonality gap allows us to show that the representations after linear layers are approximately Gaussian. If $H _ { \ell }$ is orthogonal, the random matrix $W _ { \ell } H _ { \ell }$ is equal in distribution to a standard Gaussian matrix due to the invariance of standard Gaussian distribution under linear orthogonal transformations. We can formalize this notion of Gaussianity by bounding the Wasserstein-2 distance, denoted by $\mathcal { W } _ { 2 }$ , between the distribution of $W _ { \ell } H _ { \ell }$ and standard Gaussian distribution. The next lemma formally establishes the link between the orthogonality and the distribution of the representations.

Lemma 2. Given $G \in \mathbb { R } ^ { d \times n }$ with i.i.d. zero-mean $1 / d$ -variance Gaussian elements, the following Gaussian approximation holds:

$$
\mathcal {W} _ {2} \left(\operatorname {l a w} \left(W _ {\ell} H _ {\ell}\right), \operatorname {l a w} (G / \sqrt {n})\right) ^ {2} \leq 2 n \mathbb {E} \left[ V \left(H _ {\ell}\right) \right]. \tag {4}
$$

Combining the above result with Theorem 1 yields the result presented in the next corollary.

Corollary 3. For $G \in \mathbb { R } ^ { d \times n }$ with i.i.d. zero-mean $1 / d$ -variance Gaussian elements,

$$
\mathcal {W} _ {2} \left(\operatorname {l a w} \left(W _ {\ell} H _ {\ell}\right), \operatorname {l a w} (G / \sqrt {n})\right) ^ {2} \leq 4 n \left(1 - \frac {2}{3} \alpha\right) ^ {\ell} + \frac {6 n ^ {2}}{\alpha \sqrt {d}} \tag {5}
$$

holds under Assumption $A _ { 1 } ( \alpha , \ell )$ .

In other words, the distribution of the representations contracts to a Wasserstein 2 ball around an isotropic Gaussian distribution as the depth grows. The radius of the Wasserstein 2 ball is at√ most $\mathcal { O } ( 1 / \sqrt { \mathrm { w i d t h } } )$ . As noted in the last section, $\mathcal { A } _ { 1 }$ is extensively studied by Daneshmand et al. (2020) where it is shown that $\mathcal { A } _ { 1 } ( \alpha > 0 , \infty )$ holds as long as $d = \Omega ( n ^ { 2 } )$ .

4.2. Deep neural networks as Gaussian processes. Leveraging BN, Corollary 3 establishes the first non-asymptotic Gaussian approximation for deep random neural networks. For vanilla networks, the Gaussianity is guaranteed only in the asymptotic regime of infinite width (Garriga-Alonso et al., 2019; Neal, 1996; Lee et al., 2019; Neal, 1996; Hazan and Jaakkola, 2015). Particularly, Matthews et al. (2018) links vanilla networks to Gaussian processes when their width is infinite and grows in successive layers, while our Gaussian approximation holds for networks with finite width across layers. Table 1 briefly compares Gaussian approximation for vanilla and BN networks.

Table 1. Distribution of representations in random vanilla and BN networks. For the convolutional network, the width refers to the number of channels. Results for Vanilla MLPs and Vanilla convolution networks are establish by Matthews et al. (2018), and Garriga-Alonso et al. (2019), respectively. Remarkably, Corollary 3 holds for MLP with linear activations.   

<table><tr><td>Network</td><td>Width</td><td>Depth</td><td>Distribution of Outputs</td></tr><tr><td>Vanilla MLP</td><td>infinite</td><td>finite</td><td>Converges to Gaussian as width → ∞</td></tr><tr><td>Vanilla Convnet</td><td>infinite</td><td>finite</td><td>Converges to Gaussian as width → ∞</td></tr><tr><td>BN MLP (Cor. 3)</td><td>(in)finite</td><td>(in)finite</td><td>In a O(width-1/4)-W2 ball around Gaussian</td></tr></table>

The link between Gaussian processes and infinite-width neural networks has inspired several studies to rely on Gaussian representations in deep random networks (Klambauer et al., 2017; De Palma et al., 2019; Pennington et al., 2018; Schoenholz et al., 2017; Yang et al., 2019). Assuming the representations are Gaussian, Klambauer et al. (2017) designed novel activation functions that improve the optimization performance, De Palma et al. (2019) studies the sensitivity of random networks, Pennington et al. (2018) highlights the spectral universality in random networks, Schoenholz et al. (2017) studies information propagating through the network layers, and Yang

et al. (2019) studies gradients propagation through the depth. Indeed, our analysis implies that including BN imposes the Gaussian representations required for these analyses. Although our result is established for linear activations, we conjecture a similar result holds for non-linear MLPs (see Appendix F).

# 5. The orthogonality and optimization

In the preceding sections, we elaborated on the theoretical properties of BN networks in controlled settings. In this section, we demonstrate the practical applications of our findings. In the first part, we focus on the relationship between depth and orthogonality. Increasing depth drastically slows the training of neural networks with BN. Furthermore, we observe that as depth grows, the training slowdown highly correlates with the orthogonality gap. This observation suggests that SGD needs to orthogonalize deep representations in order to start classification. This intuition leads us to the following question: If orthogonalization is a prerequisite for training, can we save optimization time by starting from orthogonal representations? To test this experimentally, we devised a weight initialization that guarantees orthogonality of representations. Surprisingly, even in a network without BN, our experiments showed that this initialization avoids the training slow down, affirmatively answering the question.

Throughout the experiments, we use vanilla MLP (without BN) with a width of 800 across all hidden layers, ReLU activation, and used Xavier’s method for weights intialization (Glorot and Bengio, 2010). We use SGD with stepsize 0.01 and batch size 500 and for training. The learning task is classification with cross entropy loss for CIFAR10 dataset (Krizhevsky et al., 2009, MIT license). We use PyTorch (Paszke et al., 2019, BSD license) and Google Colaboratory platform with a single Tesla-P100 GPU with 16GB memory in all the experiments. The reported orthogonality gap is the average of the orthogonality gap of representation in the last layer.

5.1. Orthogonality correlates with optimization performance. In the first experiment, we show that the orthogonality of representations at the initialization correlates with optimization speed. For networks with 15, 30, 45, 60, and 75 widths, we register training loss after 30 epochs and compare it with the initial orthogonality gap. Figure 3a shows the training loss (blue) and the initial orthogonality gap (red) as a function of depth. We observe that representations are more entangled, i.e., orthogonal, when we increase depth, coinciding with the training slowdown. Intuitively, the slowdown is due to the additional time SGD must spend to orthogonalize the representations before classification. In the second experiment, we validate this intuitive argument by tracking the orthogonality gap during training. Figure 3b plots the orthogonality gap of output and training loss for a network with 20 layers. We observe that SGD updates are iteratively orthogonalizing representations, marked by the reduction in the orthogonality gap.

5.2. Learning with initial orthogonal representations. We have seen that the slowdown in SGD for deeper networks correlates with the orthogonality gap before training. Here we show that by preemptively orthogonalizing representations, we avoid the slowdown with depth. While in MPL with linear activations, hidden representations remain orthogonal simply by taking orthogonal matrices as weights (Pennington et al., 2018; Saxe et al., 2014), the same does not hold for networks with non-linear activations, such as ReLU. To enforce the orthogonality in the absence of BN, we introduce a dependency between weights of successive layers that ensures deep representations remain orthogonal. More specifically, we incorporate the SVD decomposition of the hidden representation of each layer into the initialization of the subsequent layer. To emphasis this dependency between layers and to distinguish it from purely orthogonal weight initialization, we refer to this as iterative orthogonalization.

We take a large batch of samples $n \geq d$ , as the input batch for initialization. Let us assume that weights are initialized up to layer $W _ { \ell - 1 }$ . To initialize $W _ { \ell }$ , we compute SVD decomposition of the

![](images/6f223eda082cb83fbaea8a8966eb540ff4e5e6023d2a9a05eccb7e1ef8f9935b.jpg)  
(a) gap and loss vs. depth

![](images/61a38b943081aa278ffd592ad915fae3e460b93369568b5744fd687341e1b9fd.jpg)  
(b) gap and loss during training   
Figure 3. Orthogonality and Optimization Left: the orthogonality gap at initialization (red, left axis) and the training loss after 30 epochs (blue, right axis) with depth. Right: the orthogonality gap (red, left axis) and the training loss in each epoch (blue, right axis). Mean and 95% confidence interval of 4 independent runs.

representations $H _ { \ell } = U _ { \ell } \Sigma _ { \ell } V _ { \ell } ^ { \top }$ where matrices $U _ { \ell } \in \mathbb { R } ^ { d \times d }$ and $V _ { \ell } \in \mathbb { R } ^ { n \times d }$ are orthogonal. Given this decomposition, we initialize $W _ { \ell }$ by

$$
W _ {\ell} = \frac {1}{\left\| \Sigma_ {\ell} ^ {1 / 2} \right\| _ {F}} V _ {\ell} ^ {\prime} \Sigma_ {\ell} ^ {- 1 / 2} U _ {\ell} ^ {\top}, \tag {6}
$$

where $V _ { \ell } ^ { \prime } \in \mathbb { R } ^ { d \times d }$ is an orthogonal matrix obtained by slicing $V _ { \ell } \in \mathbb { R } ^ { n \times d }$ . Notably, the inverse in the above formula exists when $n$ is sufficiently larger than d 2. It is easy to check that $V ( W _ { \ell } H _ { \ell } ) < V ( H _ { \ell } )$ holds for the above initialization (see Appendix E), similar to BN. By enforcing the orthogonality, this initialization significantly alleviates the slow down of training with depth (see Fig. 4), with no need for BN. This initialization is not limited to MLPs. In Appendix G, we propose a similar SVD-based initialization for convolutional networks that effective accelerates training of deep convolutional networks.

![](images/ba404d48d1d6e42ab609d4b46cf2dfab303a6f17a32227b9c77e7bb5ffa70a9b.jpg)  
Figure 4. Iterative orthogonalization. Horizontal axis: depth. Vertical axis: the training loss after 30 epochs for Xavier’s initialization (blue), our initialization (red). Mean and 95% confidence interval of 4 independent runs.

# 6. Discussion

To recap, we proved the recurrence of random linear transformations and BN orthogonalizes samples. Our experiments underline practical applications of this theoretical finding: starting from orthogonal representations effectively avoids the training slowdown with depth for MLPs. Based on our experimental observations in Appendix G, we believe that this result extends to standard convolution networks used in practice. In other words, a proper initialization ensuring the orthogonality of hidden representations may replace BN in neural architectures. This future research direction has the potentials to boost the training of deep neural networks and change benchmarks in deep learning.

Although our theoretical bounds hold for MLPs with linear activations, our experiments confirm similar orthogonal stability for various neural architectures. Appendix Figure 5 demonstrates this stability for MLPs with ReLU activations and a convolutional networks. This plot compares the evolution of the orthogonality gap, through layers, for BN networks with vanilla networks. For vanilla networks, we observe that the gap increases with depth. On the contrary, the gap decreases by adding BN layers and stabilizes at a term that is constant with regard to depth. Based on our observations, we conjecture that hidden representations of modern BN networks obey similar stability. A more formal statement of the conjecture is presented in Appendix F.

# Acknowledgements

We thank Gideon Dresdner, Vincent Fortuin, and Ragnar Groot Koerkamp for their helpful comments and discussions. This work was funded in part by the French government under management of Agence Nationale de la Recherche as part of the “Investissements d’avenir” program, reference ANR-19-P3IA-0001(PRAIRIE 3IA Institute). We also acknowledge support from the European Research Council (grant SEQUOIA 724063).

# References

Arora, S., Cohen, N., Golowich, N., and Hu, W. (2019a). A convergence analysis of gradient descent for deep linear neural networks. International Conference on Learning Representations.   
Arora, S., Li, Z., and Lyu, K. (2019b). Theoretical analysis of auto rate-tuning by batch normalization. International Conference on Learning Representations.   
Bahri, Y., Kadmon, J., Pennington, J., Schoenholz, S. S., Sohl-Dickstein, J., and Ganguli, S. (2020). Statistical mechanics of deep learning. Annual Review of Condensed Matter Physics.   
Bartlett, P. L., Helmbold, D. P., and Long, P. M. (2019). Gradient descent with identity initialization efficiently learns positive-definite linear transformations by deep residual networks. Neural computation.   
Bietti, A. and Mairal, J. (2019). On the inductive bias of neural tangent kernels. Advances in Neural Information Processing Systems.   
Bjorck, J., Gomes, C., Selman, B., and Weinberger, K. Q. (2018). Understanding batch normalization. Advances in Neural Information Processing Systems.   
Bougerol, P. et al. (2012). Products of random matrices with applications to Schr¨odinger operators. Springer Science & Business Media.   
Daneshmand, H., Kohler, J., Bach, F., Hofmann, T., and Lucchi, A. (2020). Batch normalization provably avoids rank collapse for randomly initialised deep networks. Advances in Neural Information Processing Systems.   
De Palma, G., Kiani, B. T., and Lloyd, S. (2019). Random deep neural networks are biased towards simple functions. Advances in Neural Information Processing Systems.   
Eberle, A. (2009). Markov processes. Lecture Notes at University of Bonn.

Frankle, J., Schwab, D. J., and Morcos, A. S. (2020). Training batchnorm and only batchnorm: On the expressive power of random features in cnns. arXiv preprint arXiv:2003.00152.   
Garriga-Alonso, A., Rasmussen, C. E., and Aitchison, L. (2019). Deep convolutional networks as shallow gaussian processes. International Conference on Learning Representations.   
Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In International Conference on Artificial Intelligence and Statistics.   
Hazan, T. and Jaakkola, T. (2015). Steps toward deep kernel methods from infinite neural networks. arXiv preprint arXiv:1508.05133.   
He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition.   
Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition.   
Huang, P.-S., Avron, H., Sainath, T. N., Sindhwani, V., and Ramabhadran, B. (2014). Kernel methods match deep neural networks on timit. In ICASSP.   
Ioffe, S. and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML.   
Jacot, A., Gabriel, F., and Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. Advances in Neural Information Processing Systems.   
Karakida, R., Akaho, S., and Amari, S.-i. (2019). The normalization method for alleviating pathological sharpness in wide neural networks. In Advances in Neural Information Processing Systems.   
Kemeny, J. G. and Snell, J. L. (1976). Markov chains. Springer-Verlag, New York.   
Khasminskii, R. (2011). Stochastic stability of differential equations, volume 66. Springer Science & Business Media.   
Klambauer, G., Unterthiner, T., Mayr, A., and Hochreiter, S. (2017). Self-normalizing neural networks. Advances in Neural Information Processing Systems.   
Kohler, J., Daneshmand, H., Lucchi, A., Zhou, M., Neymeyr, K., and Hofmann, T. (2018). Exponential convergence rates for batch normalization: The power of length-direction decoupling in non-convex optimization. arXiv preprint arXiv:1805.10694.   
Krizhevsky, A., Hinton, G., et al. (2009). Learning multiple layers of features from tiny images.   
Kushner, H. and Yin, G. G. (2003). Stochastic approximation and recursive algorithms and applications. Springer Science & Business Media.   
Kushner, H. J. (1967). Stochastic stability and control. Technical report.   
Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., and Sohl-Dickstein, J. (2019). Deep neural networks as gaussian processes. International Conference on Learning Representations.   
Matthews, A. G. d. G., Rowland, M., Hron, J., Turner, R. E., and Ghahramani, Z. (2018). Gaussian process behaviour in wide deep neural networks. arXiv preprint arXiv:1804.11271.   
Neal, R. M. (1996). Bayesian learning for neural networks. Springer Science & Business Media.   
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems.   
Pennington, J., Schoenholz, S., and Ganguli, S. (2018). The emergence of spectral universality in deep networks. In International Conference on Artificial Intelligence and Statistics.   
Santurkar, S., Tsipras, D., Ilyas, A., and Madry, A. (2018). How does batch normalization help optimization?(no, it is not about internal covariate shift). Advances in Neural Information Processing Systems.

Sawa, T. (1972). Finite-sample properties of the k-class estimators. Econometrica: Journal of the Econometric Society, pages 653–680.   
Saxe, A. M., McClelland, J. L., and Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. International Conference on Learning Representations.   
Schoenholz, S. S., Gilmer, J., Ganguli, S., and Sohl-Dickstein, J. (2017). Deep information propagation. International Conference on Learning Representations.   
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. nature.   
Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research.   
Yang, G., Pennington, J., Rao, V., Sohl-Dickstein, J., and Schoenholz, S. S. (2019). A mean field theory of batch normalization. International Conference on Learning Representations.

# Appendix

# Appendix A. Preliminaries

A.1. Notations. Let $v , w \in \mathbb { R } ^ { k }$ then $v \odot w \in \mathbb { R } ^ { n }$ with coordinates

(7)

$$
[ v \odot w ] _ {i} = v _ {i} w _ {i}
$$

Furthermore $\boldsymbol { v } ^ { \otimes 2 } \in \mathbb { R } ^ { k \times k }$ with entities

(8)

$$
[ v ^ {\otimes 2} ] _ {i j} = v _ {i} v _ {j}.
$$

In Table 2, we summarize notations introduced previously.

Table 2. Notations   

<table><tr><td>Notation</td><td>Type</td><td>Definition</td></tr><tr><td>l</td><td>integer</td><td>number of layers</td></tr><tr><td>n</td><td>integer</td><td>batch size</td></tr><tr><td>d</td><td>integer</td><td>width of the network</td></tr><tr><td>k</td><td>integer</td><td>output dimension</td></tr><tr><td>X</td><td>Rd×n</td><td>input matrix</td></tr><tr><td>Hl</td><td>Rd×n</td><td>hidden representations at l (obeying Eq. (4))</td></tr><tr><td>BN</td><td>Rd×n → Rd×n</td><td>batch normalization layer (defined in Eq.(4))</td></tr><tr><td>Law(X)</td><td></td><td>the law of random matrix X</td></tr><tr><td>σi(M)</td><td>Rk1×k2 → R+</td><td>theith largest singular value of matrix M</td></tr><tr><td>Ik</td><td>Rk×k</td><td>Identity matrix of size k</td></tr><tr><td>1k</td><td>Rn</td><td>all-ones vector</td></tr></table>

A.2. The Markov chain of hidden representations. Recall the chain of the hidden representation, denoted by $\{ H _ { \ell } \in \mathbb { R } ^ { d \times n } \}$ , obeys the following recurrence:

$$
H _ {\ell + 1} = \frac {1}{\sqrt {d}} B N \left(W _ {\ell} H _ {\ell}\right), \quad B N (M) = \left(\operatorname {d i a g} \left(M M ^ {\top}\right)\right) ^ {- 1 / 2} M, \tag {9}
$$

where $W _ { \ell } \in \mathbb { R } ^ { d \times n }$ are random weight matrices with i.i.d. zero-mean Gaussian elements. It is easy to check that the Frobenius norm of $H _ { \ell }$ is one due to the row-wise normalization:

$$
\operatorname {T r} \left(B N (H) B N (H) ^ {\top}\right) = \operatorname {T r} \left(\operatorname {d i a g} \left(H H ^ {\top}\right) ^ {- 1 / 2} H H ^ {\top} \operatorname {d i a g} \left(H H ^ {\top}\right) ^ {- 1 / 2}\right)
$$

$$
\begin{array}{l} = \operatorname {T r} \left(\operatorname {d i a g} \left(H H ^ {\top}\right) ^ {- 1} H H ^ {\top}\right) \tag {10} \\ = d. \\ \end{array}
$$

A.3. A Lyapunov function for the orthogonal stability. We establish stochastic stability of the chain of the hidden representations. Our analysis is in terms of Lyapunov function $\widehat { V } : \mathbb { R } ^ { d \times n }$ defined as

$$
\widehat {V} (H) = \frac {1}{n} - (\sigma_ {n} (H)) ^ {2}, \tag {11}
$$

where $\sigma _ { n } ( H )$ is the minimum singular value of matrix $H$ . Next lemma proves $\widehat { V } ( H _ { \ell } )$ bounds the orthogonality gap.

Lemma 4. For all hidden representations $H _ { \ell }$ , the following holds:

$$
V (H _ {\ell}) \leq 2 n \widehat {V} (H _ {\ell}).
$$

Proof. Let $\sigma _ { 1 } , \ldots , \sigma _ { n }$ be the singular values of $H _ { \ell }$ . Given these singular values, one can compute $V ( H _ { \ell } )$ as

$$
(V (H _ {\ell})) ^ {2} = \sum_ {i = 1} ^ {n} \left(\sigma_ {i} ^ {2} - \frac {1}{n}\right) ^ {2}.
$$

According to Eq. (10), $\textstyle \sum _ { i = 1 } ^ { 2 } \sigma _ { i } ^ { 2 } = 1$ holds. The proof is an immediate consequence of this propery.

$$
\begin{array}{l} V^{2}(H_{\ell}) = \sum_{i = 1}^{n}\sigma_{i}^{4} - 2\underbrace{\left(\sum_{i = 1}^{n}\sigma_{i}^{2}\right)}_{= 1}\frac{1}{n} +\frac{1}{n} \\ = \sum_ {i = 1} ^ {n} \sigma_ {i} ^ {4} - \frac {1}{n}. \\ \end{array}
$$

Fixing $\sigma _ { n }$ , the maximum of $\textstyle \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 4 } - { \frac { 1 } { n } }$ subject to $\textstyle \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } = 1$ is met when $\sigma _ { 1 } ^ { 2 } = 1 - ( n - 1 ) \sigma _ { n } ^ { 2 }$ and $\sigma _ { 2 } = \cdots = \sigma _ { n }$ . Therefore,

$$
V ^ {2} (H _ {\ell}) \leq 2 (n - 1) ^ {2} \underbrace {\left(\frac {1}{n} - \sigma_ {n} ^ {2}\right) ^ {2}} _ {= \widehat {V} ^ {2} (H _ {\ell} (X))}
$$

holds true. Taking the square root of both sides concludes the proof.

![](images/7166aaff9295699ca76c8b10cf308358ee05d226fc129f1d762a149a47792441.jpg)

# Appendix B. Proof of Theorem 1

The proof of Theorem 1 relies on the following Theorem that characterizes the change of $\widehat { V }$ in consecutive layers.

Theorem 5. Sequence $\{ H _ { \ell } \}$ obeys

$$
\mathbb {E} \left[ \widehat {V} (H _ {\ell + 1}) | H _ {\ell} \right] \leq \left(1 - \frac {2}{3} \left(\frac {1}{n} - \widehat {V} (H _ {\ell})\right)\right) \widehat {V} (H _ {\ell}) + \frac {1}{\sqrt {d}}.
$$

Notably, the above result does not rely on Assumption $\mathcal { A } _ { 1 }$ . Assuming that Assumption $\mathcal { A } _ { 1 }$ holds, we complete the proof of Theorem 1. Combining the last Theorem by this assumption, we get

$$
\mathbb {E} \left[ \widehat {V} (H _ {\ell + 1}) \right] \leq \left(1 - \frac {2}{3} (\alpha)\right) \mathbb {E} \left[ \widehat {V} (H _ {\ell}) \right] + \frac {1}{\sqrt {d}}
$$

Induction over $\ell$ yields

$$
\begin{array}{l} \mathbb {E} \left[ \widehat {V} \left(H _ {\ell + 1} (X)\right) \right] \leq \left(1 - \frac {2}{3} \alpha\right) ^ {\ell} \mathbb {E} \left[ \widehat {V} \left(H _ {1}\right) \right] + \left(\sum_ {k = 1} ^ {\ell} \left(1 - \frac {2}{3} \alpha\right) ^ {k}\right) \frac {1}{\sqrt {d}} \\ \leq \left(1 - \frac {2}{3} \alpha\right) ^ {\ell} \mathbb {E} \left[ \widehat {V} (H _ {1}) \right] + \frac {3}{2 \alpha \sqrt {d}} \\ \end{array}
$$

An application of Lemma 4 completes the proof:

$$
\begin{array}{l} \mathbb {E} \left[ V \left(H _ {\ell + 1}\right) \right] \leq 2 n \mathbb {E} \left[ \widehat {V} \left(H _ {\ell + 1}\right) \right] \\ \leq 2 \left(1 - \frac {2}{3} \alpha\right) ^ {\ell} + \frac {3 n}{2 \alpha \sqrt {d}} \\ \end{array}
$$

B.1. Stability analysis without $\mathcal { A } _ { 1 }$ . Using Theorem 5, we can prove stability of the chain $\{ H _ { \ell } \}$ without Assumption $\mathcal { A } _ { 1 }$ . After rearrangement of terms in Theorem 5, we get

$$
\mathbb {E} \left[ \widehat {V} (H _ {\ell + 1}) | H _ {\ell} \right] - \widehat {V} (H _ {\ell}) \leq - \frac {2}{3} \widehat {V} (H _ {\ell}) \left(\frac {1}{n} - \widehat {V} (H _ {\ell})\right) + \frac {2}{\sqrt {d}}
$$

Taking the expectation over $H _ { \ell }$ and average over $\ell$ yields

$$
\mathbb {E} \left[ \frac {1}{\ell} \sum_ {k = 1} ^ {\ell} \widehat {V} (H _ {k}) \left(\frac {1}{n} - \widehat {V} (H _ {k})\right) \right] \leq \left(\frac {3 \mathbb {E} \left[ \widehat {V} (H _ {0}) \right]}{2 \ell}\right) + \frac {3}{2 \sqrt {d}}
$$

# Appendix C. Proof of Theorem 5

C.1. Spectral decomposition. Consider the SVD decomposition of $H _ { \ell }$ as $H _ { \ell } = U \mathrm { d i a g } ( \sigma ) V ^ { \ell }$ where $U$ and $V$ are orthogonal matrices. Given this decomposition, we get

$$
W _ {\ell} H _ {\ell} = \underbrace {W _ {\ell} U} _ {W} \operatorname {d i a g} (\sigma) V ^ {\top} \tag {12}
$$

Since $W _ { \ell }$ is Gaussian and $U$ is orthogonal, entities of matrix $W$ are also i.i.d. standard normal. This decomposition will be repeatedly used.

C.2. Concentration. Consider matrix $C _ { \ell + 1 } : = H _ { \ell + 1 } ^ { \scriptscriptstyle | } H _ { \ell + 1 }$ whose eigenvalues are $\sigma _ { 1 } ^ { 2 } , \ldots , \sigma _ { n } ^ { 2 }$ . T he SVD decomposition of $H _ { \ell }$ allows writing $C _ { \ell + 1 }$ as the average of vectors

$$
C _ {\ell + 1} = \frac {1}{d} \sum_ {i = 1} ^ {d} \left(\frac {w _ {i} \odot \sigma}{\| w _ {i} \odot \sigma \| _ {2}}\right) ^ {\otimes 2}
$$

where $w _ { i } \in \mathbb { R } ^ { n }$ are rows of matrix $W$ in Eq.(12) and $\sigma \in \mathbb { R } ^ { n }$ is the vector of singular values of $H _ { \ell }$ Thus, conditioned on $\sigma$ , $C _ { \ell + 1 }$ is an empirical average of i.i.d. random vectors. This allows us to prove that this empirical average is concentrated around its expectation. The next lemma states this concentration.

Lemma 6. The following concentration always holds

$$
\mathbb {E} _ {W _ {\ell}} \left\| C _ {\ell + 1} - \mathbb {E} _ {W _ {\ell}} \left[ C _ {\ell + 1} \right] \right\| ^ {2} \leq 1 / d
$$

where

$$
\mathbb {E} _ {W _ {\ell}} \left[ C _ {\ell + 1} \right] = \operatorname {d i a g} \left(p _ {1} (\sigma), \dots , p _ {n} (\sigma)\right), \quad p _ {i} (\sigma) := \mathbb {E} \left[ \frac {\sigma_ {n} ^ {2} w _ {n} ^ {2}}{\sum_ {k = 1} ^ {n} \sigma_ {k} ^ {2} w _ {k} ^ {2}} \right], \quad w _ {i} \stackrel {{i. i. d.}} {{\sim}} \mathcal {N} (0, 1) \tag {13}
$$

The concentration of $C _ { \ell + 1 }$ allows us to prove that the Lyapunov function $\widehat V ( H _ { \ell + 1 } )$ is concentrated around $1 / n - p _ { n } ( \sigma )$ .

Lemma 7. The following holds

$$
\mathbb {E} _ {W _ {\ell}} \left[ \left(\widehat {V} (H _ {\ell + 1}) - (1 / n - p _ {n} (\sigma))\right) ^ {2} \right] \leq \frac {1}{d}
$$

The last lemma allows us to predict the value of random variable $\widehat V ( H _ { \ell + 1 } )$ by deterministic term $1 / n - p _ { n } ( \sigma )$ .

C.3. Contraction. The decay in $\widehat V ( H _ { \ell + 1 } )$ with $\ell$ is due to term $1 / n - p _ { n } ( \sigma )$ in the last lemma. This term is less than (or equal to) $V ( H _ { \ell } )$ .

Lemma 8. For $p _ { n } ( \sigma )$ defined in Eq. (13), the following holds:

$$
\left(\frac {1}{n} - p _ {n} (\sigma)\right) \leq \left(1 - \frac {2}{3} \left(\frac {1}{n} - \widehat {V} (H _ {\ell})\right)\right) \widehat {V} (H _ {\ell}).
$$

Combining the last lemma by Lemma 7 concludes the proof of Theorem 5:

$$
\mathbb {E} _ {W _ {\ell}} \left[ \widehat {V} \left(H _ {\ell + 1}\right) \right] \leq \left(1 - \frac {2}{3} \left(\frac {1}{n} - V \left(H _ {\ell}\right)\right)\right) \widehat {V} \left(H _ {\ell}\right) + \frac {1}{\sqrt {d}}. \tag {14}
$$

To complete the proof, we prove Lemmas 6, 7, and 8.

C.4. Proof of Lemma 6. Given the spectral decomposition of $H _ { \ell }$ in Eq. (12), we compute element $_ { i j }$ of $C _ { \ell + 1 }$ , which is denoted by $[ C _ { \ell + 1 } ] _ { i j }$ :

$$
[ C _ {\ell + 1} ] _ {i j} = [ A ^ {\top} A ] _ {i j} = \frac {1}{d} \sum_ {k = 1} ^ {d} A _ {k i} A _ {k j}, \quad A _ {k i} = W _ {k i} \sigma_ {i} / \sqrt {v _ {k}}
$$

$\begin{array} { r } { v _ { k } = \sum _ { m = 1 } ^ { n } W _ { k m } ^ { 2 } \sigma _ { m } ^ { 2 } } \end{array}$ $W _ { k m }$

$$
\mathbb {E} \left[ C _ {\ell + 1} \right] _ {i j} = 0
$$

$$
\begin{array}{l} \mathbb {E} [ C _ {\ell + 1} ] _ {i j} ^ {2} = \frac {1}{d ^ {2}} \sum_ {k = 1} ^ {d} A _ {k i} ^ {2} A _ {k j} ^ {2} + \frac {1}{d ^ {2}} \sum_ {k, k ^ {\prime}} ^ {d} \underbrace {\mathbb {E} \left[ A _ {k i} ^ {2} A _ {k j} ^ {2} A _ {k ^ {\prime} i} ^ {2} A _ {k ^ {\prime} j} ^ {2} \right]} _ {= 0} \\ = \frac {1}{d ^ {2}} \sum_ {k = 1} ^ {d} A _ {k i} ^ {2} A _ {k j} ^ {2} \\ \end{array}
$$

holds for $i \neq j$ . By summing up over $i \neq j$ , we get

$$
\sum_ {i \neq j} \mathbb {E} [ C _ {\ell + 1} ] _ {i j} = \frac {1}{d ^ {2}} \left(\sum_ {k} \left(\sum_ {i} A _ {k i} ^ {2}\right) ^ {2} - \sum_ {i k} A _ {k i} ^ {4}\right)
$$

For the diagonal elements, we get

$$
\begin{array}{l} \mathbb {E} \left[ C _ {\ell + 1} \right] _ {i i} = \frac {1}{d} \sum_ {k = 1} ^ {d} \mathbb {E} A _ {k i} ^ {2} \\ = \frac {1}{d} \sum_ {k = 1} ^ {d} \mathbb {E} A _ {k i} ^ {2} \\ = \underbrace {\frac {1}{d} \sum_ {k = 1} ^ {d} \mathbb {E} \left[ \frac {W _ {k i} ^ {2} \sigma_ {i} ^ {2}}{\sum_ {j = 1} ^ {n} W _ {k j} ^ {2} \sigma_ {j} ^ {2}} \right]} _ {p _ {i} (\sigma)} \\ \end{array}
$$

The variance of $[ C _ { \ell + 1 } ] _ { i i }$ is bounded as

$$
\begin{array}{l} \operatorname {v a r} \left([ C _ {\ell + 1} ] _ {i i}\right) = \mathbb {E} \left(\frac {1}{d} \sum_ {k = 1} ^ {d} \left(A _ {k i} ^ {2} - p _ {i} (\sigma)\right)\right) ^ {2} \\ = \frac {1}{d ^ {2}} \sum_ {k = 1} ^ {d} \left(A _ {k i} ^ {2} - p _ {i} (\sigma)\right) ^ {2} \\ \leq \frac {1}{d ^ {2}} \sum_ {k = 1} ^ {d} A _ {k i} ^ {4} \\ \end{array}
$$

Combining results for diagonal and off-diagonal elements yields

$$
\begin{array}{l} \mathbb {E} \| C _ {\ell + 1} - \mathbb {E} _ {W _ {\ell}} \left[ C _ {\ell + 1} \right] \| _ {F} ^ {2} = \sum_ {i j} \operatorname {v a r} \left(\left[ C _ {\ell + 1} \right] _ {i j}\right) \\ \leq \frac {1}{d ^ {2}} \left(\sum_ {i} A _ {k i} ^ {2}\right) ^ {2} = 1 / d \\ \end{array}
$$

C.5. Proof of Lemma 7. Notably, the eigenvalues of $C _ { \ell + 1 }$ are squared singular values of $H _ { \ell + 1 }$ Let $\lambda _ { n } ( C )$ denote the $_ { n }$ th largest eigenvalue of matrix $C$ .

$$
\begin{array}{l} \mathbb {E} _ {W _ {\ell}} \left[ \left(V (C _ {\ell + 1}) - \left(\frac {1}{n} - p _ {n} (\sigma)\right)\right) ^ {2} \right] = \mathbb {E} _ {W _ {\ell}} \left[ \left(\lambda_ {n} \left(C _ {\ell + 1}\right) - \lambda_ {n} \left(\mathbb {E} _ {W _ {\ell}} [ C _ {\ell + 1} ]\right)\right) ^ {2} \right] \\ \leq \mathbb {E} _ {W _ {\ell}} \left[ \| C _ {\ell + 1} - \mathbb {E} _ {W _ {\ell}} \left[ C _ {\ell + 1} \right] \| _ {F} ^ {2} \right] \\ \leq \frac {1}{d} \\ \end{array}
$$

where the last inequality relies on Lemma 6.

C.6. Proof of Lemma 8. The proof is based an application of moment generating function that allows us to compute expectations of ratios of random variables.

Lemma 9 (Lemma 1 in (Sawa, 1972)). Let $X _ { 1 }$ be a random variable that is positive with probability one and $X _ { 2 }$ be an arbitrary random variable. Suppose that there exists a joint moment generating function of $X _ { 1 }$ and $X _ { 2 }$ :

$$
\phi \left(\theta_ {1}, \theta_ {2}\right) = \mathbb {E} \left[ \exp \left(\theta_ {1} X _ {1} + \theta_ {2} X _ {2}\right) \right]
$$

for $\theta _ { 1 } \leq \epsilon$ and $| \theta _ { 2 } | < \epsilon$ where  is some positive constant. Then

$$
\mathbb {E} \left[ \frac {X _ {2}}{X _ {1}} \right] = \int_ {- \infty} ^ {0} \left[ \frac {\partial \phi (\theta_ {1} , \theta_ {2})}{\partial \theta_ {2}} \right] _ {\theta_ {2} = 0} d \theta_ {1}
$$

To estimate $p _ { n } ( \sigma )$ , we set $X _ { 2 } : = \sigma _ { i } ^ { 2 } w _ { i } ^ { 2 }$ and $\begin{array} { r } { X _ { 1 } = \sum _ { j } \sigma _ { j } ^ { 2 } w _ { j } ^ { 2 } } \end{array}$ , which obtains

$$
\begin{array}{l} \phi \left(\theta_ {1}, \theta_ {2}\right) = \mathbb {E} \left[ \exp \left(\theta_ {1} X _ {1} + \theta_ {2} X _ {2}\right) \right] \\ = (2 \pi) ^ {- n / 2} \int_ {- \infty} ^ {\infty} \exp ((\theta_ {1} + \theta_ {2}) \sigma_ {i} ^ {2} w _ {i} ^ {2} + \sum_ {j \neq i} \theta_ {1} \sigma_ {j} ^ {2} w _ {j} ^ {2}) \exp (- \sum_ {k} w _ {k} ^ {2} / 2) d w \\ = (2 \pi) ^ {- n / 2} \int_ {- \infty} ^ {\infty} \exp \left(\left(- 0. 5 + \left(\theta_ {1} + \theta_ {2}\right) \sigma_ {i} ^ {2}\right) w _ {i} ^ {2}\right) d w _ {i} \left(\prod_ {j \neq i} \int_ {- \infty} ^ {\infty} \exp \left(\left(- 0. 5 + \theta_ {1} \sigma_ {j} ^ {2}\right) w _ {j} ^ {2}\right) d w _ {j}\right) \\ = \frac {1}{\sqrt {1 - 2 \left(\theta_ {1} + \theta_ {2}\right) \sigma_ {i} ^ {2}}} \left(\prod_ {j \neq i} \frac {1}{\sqrt {1 - 2 \theta_ {1} \sigma_ {j} ^ {2}}}\right). \\ \end{array}
$$

Taking derivative with respect to $\theta _ { 2 }$ yields

$$
\frac {\partial \phi}{\theta_ {2}} (\theta_ {1}, 0) = \frac {\sigma_ {i} ^ {2}}{(1 - 2 \theta_ {1} \sigma_ {i} ^ {2}) ^ {3 / 2}} \left(\prod_ {j \neq i} \frac {1}{\sqrt {1 - 2 \theta_ {1} \sigma_ {j} ^ {2}}}\right)
$$

Using the result of the last lemma, we get

$$
p _ {i} (\sigma) = \int_ {- \infty} ^ {0} \frac {\sigma_ {i} ^ {2}}{(1 - 2 \theta \sigma_ {i} ^ {2})} \left(\prod_ {j} \frac {1}{\sqrt {1 - 2 \theta \sigma_ {j} ^ {2}}}\right) d \theta
$$

Therefore,

$$
p _ {n} (\sigma) = \sigma_ {n} ^ {2} f _ {n} (\sigma), \quad f _ {n} (\sigma) := \int_ {- \infty} ^ {0} \frac {1}{(1 - 2 \theta \sigma_ {n} ^ {2}) ^ {3 / 2}} \prod_ {j \neq n} (1 - 2 \theta \sigma_ {j} ^ {2}) ^ {1 / 2} \tag {15}
$$

Since $\textstyle \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } = 1$ (see Eq. (10)), $f _ { n } ( \sigma )$ in the above formulation is minimized when the $\sigma _ { j } ^ { 2 }$ s are all equal for all $j \neq n$ . Let $\sigma _ { n } ^ { 2 } : = 1 / n - \delta$ and $\sigma _ { j } ^ { 2 } : = 1 / n + \delta / ( n - 1 )$ for all $j \neq i$ . This allows us to establish a lowerbound on $f _ { n } ( \sigma )$ as

$$
f _ {n} (\sigma) \geq \int_ {0} ^ {\infty} \underbrace {\left(1 + 2 \theta \left(\frac {1}{n} - \delta\right)\right) ^ {- \frac {3}{2}} \left(1 + 2 \theta \left(\frac {1}{n} + \frac {\delta}{n - 1}\right)\right) ^ {- \frac {n - 1}{2}}} _ {g (\delta) :=} \tag {16}
$$

Next lemma proves that $g ( \delta )$ is a convex function for $\delta \in [ 0 , 1 / n ]$

Lemma 10. The function $g ( \delta )$ , which is defined in Eq. (16), is a convex function on domain $\delta \in [ 0 , 1 / n ]$ .

The convexity of $g ( \delta )$ yields

$$
g ^ {\prime \prime} (\delta) \geq 0 \forall \delta \Rightarrow g (\delta) \geq g (0) + \delta g ^ {\prime} (0)
$$

The above bound allow us to bound $f _ { n } ( \sigma )$ as

$$
\begin{array}{l} f _ {n} (\sigma) \geq \int_ {0} ^ {\infty} g (0) + \delta \int_ {0} ^ {\infty} 2 \theta \left(1 + \frac {2 \theta}{n}\right) ^ {- \frac {n}{2} - 2} \\ = 1 + \delta \frac {2 n}{n + 2} \\ \end{array}
$$

Recall $\begin{array} { r } { \delta = \frac { 1 } { n } - \sigma _ { n } ^ { 2 } } \end{array}$ . Combining the above inequality by Eq. (15) concludes the proof of the Lemma 8:

$$
\frac {1}{n} - p _ {n} (\sigma) \leq \left(1 - \frac {2 n}{(n + 2)} \sigma_ {n} ^ {2}\right) \left(\frac {1}{n} - \sigma_ {n} ^ {2}\right)
$$

C.7. Proof of Lemma 10. We can show convexity of $g ( \delta )$ by showing $g ^ { \prime \prime } ( \delta ) \geq 0$ for all $\delta \in ( 0 , 1 )$ To this end, we define the helper function $h _ { 1 } ( \delta )$ (for the compactness of notations) as

$$
h _ {1} (\delta) := (1 + 2 \theta (\frac {1}{n} - \delta)) ^ {- 5 / 2} (1 + 2 \theta (\frac {1}{n} + \frac {\delta}{n - 1})) ^ {- \frac {n - 1}{2} - 1}.
$$

Given $h _ { 1 }$ , the derivative of $g$ reads as

$$
\begin{array}{l} g ^ {\prime} (\delta) = h _ {1} (\delta) \left((- \frac {3}{2}) (- 2 \theta) \left(1 + 2 \theta \left(\frac {1}{n} + \frac {\delta}{n - 1}\right)\right) - \frac {n - 1}{2} \frac {2 \theta}{n - 1} \left(1 + 2 \theta \left(\frac {1}{n} - \delta\right)\right)\right) \\ = \theta h _ {1} (\delta) \left(3 + \theta \frac {6}{n} + \theta \delta \frac {6}{n - 1} - 1 - \theta \frac {2}{n} + 2 \theta \delta\right) \\ = \theta h _ {1} (\delta) \underbrace {\left(2 + \theta \frac {4}{n} + \theta \delta \frac {2 n + 4}{n - 1}\right)} _ {h _ {2} (\delta) :=} \\ = \theta h _ {1} (\delta) h _ {2} (\delta). \\ \end{array}
$$

One can readily check that $h _ { 2 } ^ { \prime } ( \delta ) \geq 0$ . Hence, $h _ { 1 } ^ { \prime } ( \delta ) \geq 0$ ensures the convexity of $g ( \delta )$ . Consider the following helper

$$
h _ {3} (\delta) := \left(1 + 2 \theta \left(\frac {1}{n} - \delta\right)\right) ^ {- 7 / 2} \left(1 + 2 \theta \left(\frac {1}{n} + \frac {\delta}{n - 1}\right)\right) ^ {- \frac {n - 1}{2} - 2}.
$$

Given $h _ { 3 }$ , we compute $h _ { 1 } ^ { \prime }$ as

$$
\begin{array}{l} h _ {1} ^ {\prime} (\delta) = h _ {3} (\delta) \left((- \frac {5}{2}) (- 2 \theta) (1 + 2 \theta (\frac {1}{n} + \frac {\delta}{n - 1})) + (- \frac {n - 3}{2}) \frac {2 \theta}{n - 1} (1 + 2 \theta (\frac {1}{n} - \delta))\right) \\ = h _ {3} (\delta) \theta \left(5 \left(1 + 2 \theta / n + \frac {2 \theta \delta}{n - 1}\right) - \frac {n - 3}{n - 1} \left(1 + 2 \theta / n - 2 \theta \delta\right)\right) \\ = h _ {3} (\delta) \theta \underbrace {\left(\frac {4 n - 2}{n - 1} + 2 \theta \frac {4 n - 2}{n - 1} + 2 \theta \delta \frac {n + 2}{n - 1}\right)} _ {h _ {4} (\delta) :=} \\ = \theta h _ {3} (\delta) h _ {4} (\delta). \\ \end{array}
$$

Clearly we have $h _ { 3 } ( \delta ) , h _ { 4 } ( \delta ) \geq 0$ for all valid choices of $\delta$ which finishes the proof.

# Appendix D. Proof of Lemma 2

The main idea is based on a particular coupling of random matrices $W _ { \ell } H _ { \ell }$ and $G$ . Consider the truncated SVD decomposition of $H _ { \ell }$ as $H _ { \ell } = U \mathrm { d i a g } ( \sigma ) V ^ { \top }$ where $U \in \mathbb { R } ^ { d \times n }$ and $V \in \mathbb { R } ^ { n \times n }$ are orthogonal matrices. Due to the orthogonality, the law of $W _ { \ell } U$ is the same as those of $G V$ . By

coupling $W _ { \ell } U = G V$ , we get

$$
\begin{array}{l} \left(\mathcal {W} _ {2} (\mu (W _ {\ell} H _ {\ell}), \mu (G / \sqrt {n}))\right) ^ {2} = \inf _ {\mathrm {a l l t h e c o u p l i n g s}} \mathbb {E} \| W _ {\ell} U \mathrm {d i a g} (\sigma) V ^ {\top} - G V V ^ {\top} / \sqrt {n} \| _ {F} ^ {2} \\ \leq \mathbb {E} \| G V (\operatorname {d i a g} (\sigma) - I / \sqrt {n}) V ^ {\top} \| _ {F} ^ {2} \\ = \mathbb {E} \operatorname {T r} (G V (\operatorname {d i a g} (\sigma) - I / \sqrt {n}) V ^ {\top} V (\operatorname {d i a g} (\sigma) - I / \sqrt {n}) V ^ {\top} G ^ {\top}) \\ = \operatorname {T r} (V (\operatorname {d i a g} (\sigma) - I / \sqrt {n}) (\operatorname {d i a g} (\sigma) - I / \sqrt {n}) V ^ {\top} \mathbb {E} [ G ^ {\top} G ]) \\ = \mathbb {E} \operatorname {T r} \left(\left(\operatorname {d i a g} (\sigma) - I / \sqrt {n}\right) \left(\operatorname {d i a g} (\sigma) - I / \sqrt {n}\right) V ^ {\top} V\right) \\ = \mathbb {E} \| (\operatorname {d i a g} (\sigma) - I / \sqrt {n}) \| _ {F} ^ {2} \\ = \mathbb {E} \left[ \sum_ {i = 1} ^ {n} \left(\sigma_ {i} - 1 / \sqrt {n}\right) ^ {2} \right] \\ = \mathbb {E} \left[ \sum_ {i = 1} ^ {n} \left(\sigma_ {i} ^ {2} - 1 / n\right) ^ {2} / \left(\sigma_ {i} + 1 / \sqrt {n}\right) ^ {2} \right] \\ \leq n \mathbb {E} \left[ \sum_ {i = 1} ^ {n} \left(\sigma_ {i} ^ {2} - 1 / n\right) ^ {2} \right] \\ \leq n \mathbb {E} \left[ V ^ {2} \left(H _ {\ell}\right) \right] \\ \leq 2 n \mathbb {E} \left[ V \left(H _ {\ell}\right) \right]. \\ \end{array}
$$

# Appendix E. Orthogonality gap for the iterative initialization

Recall the proposed initialization for weights based on SVD decomposition $H _ { \ell } = U _ { \ell } \Sigma _ { \ell } V _ { \ell } ^ { \top }$

$$
W _ {\ell} = \frac {1}{\| \Sigma_ {\ell} ^ {1 / 2} \| _ {F}} V _ {\ell} ^ {\prime} \Sigma_ {\ell} ^ {- 1 / 2} U _ {\ell} ^ {\top}.
$$

Here, we show that

$$
V \left(H _ {\ell}\right) > V \left(W _ {\ell} H _ {\ell}\right) \tag {17}
$$

holds as long as $V ( H _ { \ell } ) \neq 0$ and $A _ { 1 } ( \alpha , \ell )$ holds. Given singular values $H _ { \ell }$ , we get

$$
\begin{array}{l} V \left(H _ {\ell}\right) = \sum_ {i = 1} ^ {n} \left(\sigma_ {i} ^ {2} - \frac {1}{n}\right) ^ {2} \tag {18} \\ = \sum_ {i} \sigma_ {i} ^ {4} - 1 / n, \\ \end{array}
$$

where we used $\textstyle \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } = 1$ (see Eq. (2)). Now, we compute $V ( H _ { \ell } )$ using the singular values.

$$
W _ {\ell} H _ {\ell} = \frac {1}{\| \Sigma_ {\ell} ^ {1 / 2} \| _ {F}} V _ {\ell} ^ {\prime} \Sigma_ {\ell} ^ {1 / 2} V _ {\ell}
$$

Hence, the following holds

$$
\begin{array}{l} V (W _ {\ell} H _ {\ell}) = \sum_ {i = 1} ^ {n} \left(\frac {\sigma_ {i}}{\sum_ {j} \sigma_ {j}} - \frac {1}{n}\right) ^ {2} \\ = \frac {1}{\left(\sum_ {i = 1} ^ {n} \sigma_ {i}\right) ^ {2}} - 1 / n \\ \end{array}
$$

Combining with Eq. (18), we get

$$
V (H _ {\ell}) - V (W _ {\ell} H _ {\ell}) = \sum_ {i = 1} ^ {n} \sigma_ {i} ^ {4} - \frac {1}{(\sum_ {i = 1} ^ {n} \sigma_ {i}) ^ {2}}.
$$

To show that the right side of the above equation is positive, we need to prove

$$
\sum_ {i} \sigma_ {i} ^ {4} \left(\sum_ {i = 1} ^ {n} \sigma_ {i}\right) ^ {2} > 1 = \left(\sum_ {i} \sigma_ {i} ^ {2}\right) ^ {4}
$$

holds. Using Cauchy-Schwarz inequality, we get

$$
\begin{array}{l} \left(\sum_ {i} \sigma_ {i} ^ {2}\right) ^ {4} = \left(\sum_ {i} \sigma_ {i} ^ {3 / 2} \sigma_ {i} ^ {1 / 2}\right) ^ {4} \\ \leq \left(\sum_ {i} \sigma_ {i}\right) ^ {2} \left(\sum_ {i} \sigma_ {i} ^ {2} \sigma_ {i}\right) ^ {2} \\ \leq \left(\sum_ {i} \sigma_ {i}\right) ^ {2} \left(\sum_ {i} \sigma_ {i} ^ {4}\right) ^ {2}, \\ \end{array}
$$

where the equality in the above inequality is met only for $\begin{array} { r } { \sigma _ { i } ^ { 2 } = \frac { 1 } { n } } \end{array}$ (so $V ( H _ { \ell } ) = 0$ ) under $\mathcal { A } _ { 1 }$ .

# Appendix F. Activation functions and the orthogonality

We have shown hidden representations become increasingly orthogonal through layers of BN MLPs with linear activations. We observed a similar results for MLP with ReLU activations and also a simple convolution network (see Figure 5). Here, we elaborate more on the role activation functions in our analysis.

![](images/7dd239893acae9989a74a90b5cc69b75099e4e75054c20f1ef8b05893bd1daac.jpg)  
(a) MLP

![](images/ce32f014b0d5cef075fff876e8c59b978b0ed012a12bf86b2fff8104a984938a.jpg)  
(b) ConvNet   
Figure 5. Orthogonality gap of layers for BatchNorm vs vanilla network with ReLU, for MLP and a basic convolutional network on CIFAR10. Batch size was set to 8, the number of hidden layers for both networks was set to 30, with 100 width for MLP and 100 channels for ConvNet in all hidden layers. For the ConvNet, the kernel size was set to 3, across all hidden layers, with zero-padding to make convolutional feature maps equal sized. The standard deviations were computed for 50 different randomly sampled batches.

For an arbitrary activation function $F$ , the hidden representations make the following Markov chain:

$$
Q _ {\ell + 1} = B N (W _ {\ell} F (Q _ {\ell})).
$$

F.1. Conjecture. Suppose that $\{ Q _ { \ell } \}$ is ergodict (Eberle, 2009), and admits a unique invariant distribution denoted by $\nu$ . We define function $L : \mathbb { R } ^ { d \times n } \to \mathbb { R } _ { + }$ as

$$
L \left(Q _ {\ell}\right) = \left\| Q _ {\ell} ^ {\top} Q _ {\ell} - \mathbb {E} _ {Q \sim \nu} \left[ Q ^ {\top} Q \right] \right\| _ {F}. \tag {19}
$$

We conjecture that there exits an integer $k$ such that for $\ell > k$ the following holds:

$$
\mathbb {E} \left[ L \left(Q _ {\ell}\right) \right] = \mathcal {O} \left(\frac {1}{\sqrt {d}}\right). \tag {20}
$$

Figure 6 experimentally validates that the above result holds for standard activations such as ReLU, sigmoid, and tanh.

![](images/458c7f62151fe436355c9fd3e47edbc2128fba36dd729953d122ebd5a0a67e90.jpg)  
Figure 6. Validating the conjecture. Horizontal axis: $\log ( d )$ ; vertical axis: log( 11000 P1000k=1 $\begin{array} { r } { \log ( \frac { 1 } { 1 0 0 0 } \sum _ { k = 1 } ^ { 1 0 0 0 } L ( Q _ { k } ) ) } \end{array}$ . The difference between plots for the activation functions is negligible, so not visible.

F.2. Odd activation. We observe that Theorem 1 holds for odd activations such as sin and tanh. For these activation result of Theorem 1.

$$
\mathbb {E} \left[ V (H _ {\ell + 1}) \right] = \mathcal {O} \left((1 - \alpha) ^ {\ell} + \frac {1}{\sqrt {d}}\right)
$$

holds under $A _ { 1 } ( \alpha , \ell )$ . This also aligns with our conjecture for all activations as we observed that $V = Q$ for these activations.

# Appendix G. Experiments for convolutional networks

The introduced initialization scheme in Section 5 extends to convnets. In convolutional networks, hidden representation are in the tensor form $H _ { \ell } \in \mathbb { R } ^ { d \times m \times m \times n }$ where $d$ is the number of filters and $k$ is the image dimensions. Let matrix $W _ { \ell } ^ { d \times k ^ { 2 } }$ are weights of convolutions layer $\ell$ with kernel size $k$ and $d$ filters. A 2D convolution is a matrix multiplication combined by projections as:

$$
\operatorname {c o n v 2 d} \left(W _ {\ell}, H _ {\ell}\right) = \operatorname {f o l d} \left(W _ {\ell} \underbrace {\operatorname {u n f o l d} \left(H _ {\ell}\right)} _ {H _ {\ell} ^ {\prime}}\right) \tag {21}
$$

![](images/c808399861107e0c9e708269165c7101e779c9483e0a34f832eb473f9bed9865.jpg)  
20 layers

![](images/e15398245c2fcdf46bb8c2f7759e283ccb35f7a91a367f94610d815f32a8435f.jpg)  
80 layers   
Figure 7. Training convolutional networks. Convergence of SGD for Xavier initialization (Glorot and Bengio, 2010) (in blue) and also the proposed initialization method that ensures the orthogonality of hidden representations (in red). Mean and 95% confidence interval of 4 independent runs.

where the unfolding extract batches of $k \times k$ from images in tensor $H _ { \ell }$ , folding operation combines the computed convolution for the extracted batches. We use the SVD decomposition $H _ { \ell } ^ { \prime } = U _ { \ell } \Sigma _ { \ell } V _ { \ell }$ to initialize weights $W _ { \ell }$ , exactly the same as Eq. (6):

$$
W _ {\ell} = \frac {1}{\| \Sigma_ {\ell} ^ {1 / 2} \| _ {F}} V _ {\ell} ^ {\prime} \Sigma_ {\ell} ^ {- 1 / 2} U _ {\ell} ^ {\top},
$$

where $V _ { \ell } ^ { \prime }$ is a slice of $V _ { \ell }$ . We experimentally validate the performance of this initialization for two convolution networks with ReLU activations consist of 20 and 80 convolutions. Table 3 outlines the details of these neural architectures.

Table 3. Convolutional networks used in the experiment. These neural architectures are obtained from adding layers to AlexNet (Srivastava et al., 2014). The kernel size is 3 for all the convolutional layers.   

<table><tr><td>The network with 20 layers</td><td>The network with 80 layers</td></tr><tr><td>Conv2d(3, 64) + MaxPool2d(2) + RELU</td><td>Conv2d(3, 64) + MaxPool2d(2) + RELU</td></tr><tr><td>Conv2d(64, 192) + MaxPool2d(2) + ReLU</td><td>Conv2d(64, 192) + MaxPool2d(2) + ReLU</td></tr><tr><td>Conv2d(192, 256) + Conv2d(256, 256) +ReLU</td><td>Conv2d(192, 256) Conv2d(256, 256) + ReLU</td></tr><tr><td>{Conv2d(256, 256) +ReLU} × 16</td><td>{Conv2d(256, 256) +ReLU} × 76</td></tr><tr><td>MaxPool2d(2) + ReLU</td><td>MaxPool2d(2) + ReLU</td></tr><tr><td>Dropout(0.5) + Linear(1024,4096) + ReLU</td><td>Dropout(0.5) + Linear(1024,4096) + ReLU</td></tr><tr><td>Dropout(0.5) + Linear(4096,4096) + ReLU</td><td>Dropout(0.5) + Linear(4096,4096) + ReLU</td></tr><tr><td>Linear(4096,10)</td><td>Linear(4096,10)</td></tr></table>

Figure 7 shows the performance of SGD (with stepsize 0.001 and batchsize 30) for the proposed initialization. This initialization slows SGD for the shallow network with 20 compared to standard Xavier’s initialization Glorot and Bengio (2010). However, this initialization outperforms Xavier’s initialization when the depth is significantly large. This result substantiates the role of the orthogonality in training and shows that the experimental result of Figure 4 extends to convolutional neural networks.