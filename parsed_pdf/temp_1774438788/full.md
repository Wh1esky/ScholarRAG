# Discrete-Valued Neural Communication

Dianbo Liu* Mila

Alex Lamb* Mila

Kenji Kawaguchi Harvard University

Anirudh Goyal Mila

Chen Sun Mila

Michael Curtis Mozer Brain and University of Colorado

Yoshua Bengio Mila

* co-first author Emails:liudianbo@gmail.com, alex6200@gmail.com, kkawaguchi@fas.harvard.edu

# Abstract

Deep learning has advanced from fully connected architectures to structured models organized into components, e.g., the transformer composed of positional elements, modular architectures divided into slots, and graph neural nets made up of nodes. In structured models, an interesting question is how to conduct dynamic and possibly sparse communication among the separate components. Here, we explore the hypothesis that restricting the transmitted information among components to discrete representations is a beneficial bottleneck. The motivating intuition is human language in which communication occurs through discrete symbols. Even though individuals have different understandings of what a “cat” is based on their specific experiences, the shared discrete token makes it possible for communication among individuals to be unimpeded by individual differences in internal representation. To discretize the values of concepts dynamically communicated among specialist components, we extend the quantization mechanism from the Vector-Quantized Variational Autoencoder to multi-headed discretization with shared codebooks and use it for discrete-valued neural communication (DVNC). Our experiments show that DVNC substantially improves systematic generalization in a variety of architectures—transformers, modular architectures, and graph neural networks. We also show that the DVNC is robust to the choice of hyperparameters, making the method very useful in practice. Moreover, we establish a theoretical justification of our discretization process, proving that it has the ability to increase noise robustness and reduce the underlying dimensionality of the model.

# 1 Introduction

In AI, there has long been a tension between subsymbolic and symbolic architectures. Subsymbolic architectures, like neural networks, utilize continuous representations and statistical computation. Symbolic architectures, like production systems (Laird et al., 1986) and traditional expert systems, use discrete, structured representations and logical computation. Each architecture has its strengths: subsymbolic computation is useful for perception and control, symbolic computation for higher level, abstract reasoning. A challenge in integrating these approaches is developing unified learning procedures.

As a step toward bridging the gap, recent work in deep learning has focused on constructing structured architectures with multiple components that interact with one another. For instance, graph neural networks are composed of distinct nodes (Kipf et al., 2019; Scarselli et al., 2008; Kipf et al., 2018; Santoro et al., 2017; Raposo et al., 2017; Bronstein et al., 2017; Gilmer et al., 2017; Tacchetti et al.,

2018; Van Steenkiste et al., 2018), transformers are composed of positional elements (Bahdanau et al., 2014; Vaswani et al., 2017), and modular models are divided into slots or modules with bandwidth limited communication (Jacobs et al., 1991; Bottou and Gallinari, 1991; Goyal and Bengio, 2020; Ronco et al., 1997; Reed and De Freitas, 2015; Lamb et al., 2021; Andreas et al., 2016; Rosenbaum et al., 2017; Fernando et al., 2017; Shazeer et al., 2017; Rosenbaum et al., 2019).

Although these structured models exploit the discreteness in their architectural components, the present work extends these models to leverage discreteness of representations, which is an essential property of symbols. We propose to learn a common codebook that is shared by all components for inter-component communication. The codebook permits only a discrete set of communicable values. We hypothesize that this communication based on the use and reuse of discrete symbols will provide us with two benefits:

• The use of discrete symbols limits the bandwidth of representations whose meaning needs to be learned and synchronized across modules. It may therefore serve as a common language for interaction, and make it easier to learn.   
• The use of shared discrete symbols will promote systematic generalization by allowing for the reuse of previously encountered symbols in new situations. This makes it easier to hot-swap one component for another when new out-of-distribution (OOD) settings arise that require combining existing components in novel ways.

Our work is inspired by cognitive science, neuroscience, and mathematical considerations. From the cognitive science perspective, we can consider different components of structured neural architectures to be analogous to autonomous agents in a distributed system whose ability to communicate stems from sharing the same language. If each agent speaks a different language, learning to communicate would be slow and past experience would be of little use when the need to communicate with a new agent arises. If all agents learn the same language, each benefits from this arrangement. To encourage a common language, we limit the expressivity of the vocabulary to discrete symbols that can be combined combinatorially. From the neuroscience perspective, we note that various areas in the brain, including the hippocampus (Sun et al., 2020; Quiroga et al., 2005; Wills et al., 2005), the prefrontal cortex (Fujii and Graybiel, 2003), and sensory cortical areas (Tsao et al., 2006) are tuned to discrete variables (concepts, actions, and objects), suggesting the evolutionary advantage of such encoding, and its contribution to the capacity for generalization in the brain. From a theoretical perspective, we present analyses suggesting that multi-head discretization of inter-component communication increases model sensitivity and reduces underlying dimensions (Section 2). These sources of inspiration lead us to the proposed method of Discrete-Valued Neural Communication (DVNC).

Architectures like graph neural networks (GNNs), transformers, and slot-based or modular neural networks consist of articulated specialist components, for instance, nodes in GNNs, positions in transformers, and slots/modules for modular models. We evaluate the efficacy of DVNC in GNNs, transformers, and in a modular recurrent architecture called RIMs. For each of these structured architectures, we keep the original architecture and all of its specialist components the same. The only change is that we impose discretization in the communication between components (Figure 1).

Our work is organized as follows. First, we introduce DVNC and present theoretical analyses showing that DVNC improves sensitivity and reduces underlying dimensionality of models (the logarithm of the covering number). Then we explain how DVNC can be incorporated into different model architectures. And finally we report experimental results showing improved OOD generalization with DVNC.

# 2 Discrete-Value Neural Communication and Theoretical Analysis

In this section, we begin with the introduction of Discrete-Value Neural Communication (DVNC) and proceed by conducting a theoretical analysis of DVNC affects the sensitivity and underlying dimensionality of models. We then explain how DVNC can be used within several different architectures.

Discrete-Value Neural Communication (DVNC) The process of converting data with continuous attributes into data with discrete attributes is called discretization (Chmielewski and Grzymala-Busse,

![](images/c0c091fff6b7c795a1a478b825531a971487ec7d0679dd4983d6e30cb760d51d.jpg)  
(a) DVNC in Graph Neural Network

![](images/d029ac0df005f3865dc6b5fbee7b2686f679ceb6a832cbc49e88ab3f91020d19.jpg)  
(b) DVNC in Transformer

![](images/6b18d120523a028974b3241a3a5d8fc2930b7ff99cad7bad17e754a58506db36.jpg)  
(c) DVNC in Modular Recurrent Mechanisms (RIMs)   
Figure 1: Communication among different components in neural net models is discretized via a shared codebook. In modular recurrent neural networks and transformers, values of results of attention are discretized. In graph neural networks, communication from edges is discretized.

1996). In this study, we use discrete latent variables to quantize information communicated among different modules in a similar manner as in Vector Quantized Variational AutoEncoder (VQ-VAE) (Oord et al., 2017). Similar to VQ-VAE, we introduce a discrete latent space vector $e \in \mathbb { R } ^ { L \times m }$ where $L$ is the size of the discrete latent space (i.e., an $L$ -way categorical variable), and $m$ is the dimension of each latent embedding vector $e _ { j }$ . Here, $L$ and $m$ are both hyperparameters. In addition, by dividing each target vector into $G$ segments or discretization heads, we separately quantize each head and concatenate the results (Figure 2). More concretely, the discretization process for each vector $h \in \mathcal { H } \subset \mathbb { R } ^ { m }$ is described as follows. First, we divide a vector $h$ into $G$ segments $s _ { 1 } , s _ { 2 } , \ldots , s _ { G }$ with $h = \mathrm { C O N C A T E N A T E } \left( s _ { 1 } , s _ { 2 } , \ldots , s _ { G } \right)$ $( s _ { 1 } , s _ { 2 } , \ldots , s _ { G } )$ , where each segment $s _ { i } \in \mathbb { R } ^ { m / G }$ with $\frac { m } { G } \in \mathbb { N } ^ { + }$ . Second, we discretize each segment $s _ { i }$ separately:

$$
e _ {o _ {i}} = \text {D I S C R E T I Z E} (s _ {i}), \quad \text {w h e r e} o _ {i} = \underset {j \in \{1, \dots , L \}} {\operatorname {a r g m i n}} | | s _ {i} - e _ {j} | |.
$$

Finally, we concatenate the discretized results to obtain the final discretized vector $Z$ as

$$
Z = \text {C O N C A T E N A T} (\text {D I S C R E T I Z E} (s _ {1}), \text {D I S C R E T I Z E} (s _ {2}), \dots , \text {D I S C R E T I Z E} (s _ {G})).
$$

The multiple steps described above can be summarized by $Z = q ( h , L , G )$ , where $q ( \cdot )$ is the whole discretization process with the codebook, $L$ is the codebook size, and $G$ is number of segments per vector. It is worth emphasizing that the codebook $e$ is shared across all communication vectors and heads, and is trained together with other parts of the model.

The overall loss for model training is:

$$
\mathcal {L} = \mathcal {L} _ {\text {t a s k}} + \frac {1}{G} \left(\sum_ {i} ^ {G} \left| \left| \operatorname {s g} \left(s _ {i}\right) - e _ {o _ {i}} \right| \right| _ {2} ^ {2} + \beta \sum_ {i} ^ {G} \left| \left| s _ {i} - \operatorname {s g} \left(e _ {o _ {i}}\right) \right| \right| _ {2} ^ {2}\right) \tag {1}
$$

where $\mathcal { L } _ { \mathrm { t a s k } }$ is the loss for specific task, e.g., cross entropy loss for classification or mean square error loss for regression, sg refers to a stop-gradient operation that blocks gradients from flowing into its argument, and $\beta$ is a hyperparameter which controls the reluctance to change the code. The second term $\begin{array} { r } { \sum _ { i } ^ { G } | | \operatorname { s g } ( s _ { i } ) - e _ { o _ { i } } | | _ { 2 } ^ { 2 } } \end{array}$ is the codebook loss, which only applies to the discrete latent vector and brings the selected selected $e _ { o _ { i } }$ close to the output segment $s _ { i }$ . The third term $\begin{array} { r } { \sum _ { i } ^ { G } | | s _ { i } - \mathrm { s g } ( e _ { o _ { i } } ) | | _ { 2 } ^ { 2 } } \end{array}$ is the commitment loss, which only applies to the target segment $s _ { i }$ and trains the module that outputs $s _ { i }$ to make $s _ { i }$ stay close to the chosen discrete latent vector $e _ { o _ { i } }$ . We picked $\beta = 0 . 2 5$ as in the original VQ-VAE paper (Oord et al., 2017). We initialized $e$ using $k$ -means clustering on vectors $h$ with $k = L$ and trained the codebook together with other parts of the model by gradient decent. When there were multiple $h$ vectors to discretize in a model, the mean of the codebook and commitment loss across all $h$ vectors was used. Unpacking this equation, it can be seen that we adapted the vanilla VQ-VAE loss to directly suit our discrete communication method (Oord et al., 2017). In particular, the VQ-VAE loss was adapted to handle multi-headed discretization by summing over all the separate discretization heads.

In the next subsection, we use the following additional notation. The function $\phi$ is arbitrary and thus can refer to the composition of an evaluation criterion and the rest of the network following discretization. Given any function $\phi : \mathbb { R } ^ { m }  \mathbb { R }$ and any family of sets $S = \{ S _ { 1 } , \ldots , S _ { K } \}$ with $S _ { 1 } , \dots , S _ { K } \subseteq \mathcal { H }$ , we define the corresponding function $\phi _ { k } ^ { S }$ by $\begin{array} { r } { \phi _ { k } ^ { S } ( h ) = \mathbb { 1 } \{ h \stackrel { \cdot } { \in } S _ { k } \} \phi ( h ) } \end{array}$ for all $k \in [ K ]$ , where $[ K ] = \{ 1 , 2 , \dots , K \}$ . Let $e \in E \subset \mathbb { R } ^ { L \times m }$ be fixed and we denote by $( Q _ { k } ) _ { k \in [ L ^ { G } ] }$ all the possible values after the discretization process: i.e., $q ( h , L , G ) \in \cup _ { k \in [ L ^ { G } ] } \{ Q _ { k } \}$ for all $h \in \mathcal H$ .

![](images/c70327a5f18bf0eb13065231ccd3c3901d01dfeff0c09e89eb68ea03b6f8a66c.jpg)  
Figure 2: In structured architectures, communication is typically vectorized. In DVNC, this communication vector is first divided into discretization heads. Each head is discretized separately to the nearest neighbor of a collection of latent codebook vectors which is shared across all the heads. The discretization heads are then concatenated back into the same shape as the original vector.

Table 1: Communication with discretized values achieves a noise-sensitivity bound that is independent of the number of dimensions $m$ and network lipschitz constant $\bar { \varsigma _ { k } }$ and only depends on the number of discretization heads $G$ and codebook size $L$ .   

<table><tr><td>Communication Type</td><td>Example</td><td>Sensitivity Bounds (Thm 1, 2)</td></tr><tr><td>Communication with continuous signals is expressive but can take a huge range of novel values, leading to poor systematic generalization</td><td>m ~ 105</td><td>O(√mln(4√nm)+ln(2/δ)/2n + ŷkRμ/√n)</td></tr><tr><td>Communication with multi-head discrete-values is both expressive and sample efficient</td><td>“John owns a car”G=15L=30</td><td>O(√Gln(L)+ln(2/δ)/2n)</td></tr></table>

Theoretical Analysis This subsection shows that adding the discretization process has two potential advantages: (1) it improves noise robustness and (2) it reduces underlying dimensionality. These are proved in Theorems 1–2, illustrated by examples (Table 1), and explored in analytical experiments using Gaussian-distributed vectors (Figure 3).

To understand the advantage on noise robustness, we note that there is an additional error incurred√ by noise without discretization, i.e., the second term $\bar { \varsigma } _ { k } R _ { \mathcal { H } } / \sqrt { n } \geq 0$ in the bound of Theorem 2 ( $\bar { \varsigma } _ { k }$ and $R _ { \mathcal { H } }$ are defined in Theorem 2). This error due to noise disappears with discretization in the bound of Theorem 1 as the discretization process reduces the sensitivity to noise. This is because the discretization process lets the communication become invariant to noise within the same category; e.g., the communication is invariant to different notions of “cats”.

To understand the advantage of discretization on dimensionality, we can see that it reduces the underlying dimensionality of $m \ln ( 4 { \sqrt { n m } } )$ without discretization (in Theorem 2) to that of $G \ln ( L )$ with discretization (in Theorem 1). As a result, the size of the codebook $L$ affects the underlying dimension in a weak (logarithmic) fashion, while the number of dimensions $m$ and the number of discretization heads $G$ scale the underlying dimension in a linear way. Thus, the discretization process√ successfully lowers the underlying dimensionality for any $n \geq 1$ as long as $G \ln ( L ) < m \ln \bar { ( } 4 \sqrt { m } )$ . This is nearly always the case as the number of discretization heads $G$ is almost always much smaller than the number of units $m$ . Intuitively, a discrete language has combinatorial expressiveness, making it able to model complex phenomena, but still lives in a much smaller space than the world of unbounded continuous-valued signals (as $G$ can be much smaller than $m$ ).

Theorem 1. (with discretization) Let $S _ { k } = \{ Q _ { k } \}$ for all $k \in [ L ^ { G } ]$ . Then, for any $\delta > 0$ , with probability at least $1 - \delta$ over an iid draw of $n$ examples $( h _ { i } ) _ { i = 1 } ^ { n }$ , the following holds for any $\phi : \mathbb { R } ^ { m }  \mathbb { R }$ and all $k \in [ L ^ { G } ]$ $k \in [ L ^ { G } ] \colon i f | \phi _ { k } ^ { S } ( h ) | \leq \alpha$ for all $h \in \mathcal H$ , then

$$
\left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (q (h, L, G)) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} (q \left(h _ {i}, L, G\right)) \right| = \mathcal {O} \left(\alpha \sqrt {\frac {G \ln (L) + \ln (2 / \delta)}{2 n}}\right), \tag {2}
$$

where no constant is hidden in $\mathcal { O }$ .

Theorem 2. (without discretization) Assume that $\| h \| _ { 2 } \le R _ { \mathcal { H } }$ for all $h \in \mathcal { H } \subset \mathbb { R } ^ { m }$ . Fix √ $\textit { c } \in$ argminC¯{|C| ¯ : C ⊆¯ Rm, $\mathcal { H } \subseteq \cup _ { c \in \bar { c } } B [ c ] \}$ where $\mathcal { B } [ c ] = \{ x \in \mathbb { R } ^ { m } : \| x - c \| _ { 2 } \leq R _ { \mathcal { H } } / ( 2 \sqrt { n } ) \}$ . Let $S _ { k } = B [ c _ { k } ]$ for all $k \in [ | \mathcal { C } | ]$ where $c _ { k } \in \mathcal { C }$ and $\cup _ { k } \{ c _ { k } \} = \mathcal { C }$ . Then, for any $\delta > 0$ , with probability at least $1 - \delta$ over an iid draw of $n$ examples $( h _ { i } ) _ { i = 1 } ^ { n }$ , the following holds for any $\phi : \mathbb { R } ^ { m }  \mathbb { R }$ and all $k \in [ | \mathcal { C } | ]$ : $f | \phi _ { k } ^ { S } ( h ) | \leq \alpha$ for all $h \in \mathcal H$ and $| \phi _ { k } ^ { S } ( h ) - \phi _ { k } ^ { S } ( h ^ { \prime } ) | \leq \varsigma _ { k } \| h - h ^ { \prime } \| _ { 2 }$ for all $h , h ^ { \prime } \in S _ { k }$ , then

$$
\left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(h _ {i}\right) \right| = \mathcal {O} \left(\alpha \sqrt {\frac {m \ln \left(4 \sqrt {n m}\right) + \ln \left(2 / \delta\right)}{2 n}} + \frac {\bar {\varsigma} _ {k} R _ {\mathcal {H}}}{\sqrt {n}}\right), \tag {3}
$$

![](images/14f2cee361ddeed9571bfedc58b32cd7b8f7d42706915e5bf225b691dfa630d2.jpg)

![](images/657e4261b37e07ff0ad6a24081dfb2a8453d7e4d471337062ae660ea8e99b453.jpg)

![](images/20b3553f2465e68c1faea4dde3e72d70cc8f44fe24d8f6c98dfc248fb0439f56.jpg)

![](images/1c3a69948ff90f965289c6dfdea3fd9872fff4db9f44c9fa5a83a786701b06d4.jpg)  
Figure 3: We perform empirical analysis on Gaussian vectors to build intuition for our theoretical analysis. Expressiveness scales much faster as we increase discretization heads than as we increase the size of the codebook. This can be seen when we measure the variance of a collection of Gaussian vectors following discretization (left), and can also be seen when we plot a vector field of the effect of discretization (center). Discretizing the values from an attention layer trained to select a fixed Gaussian vector makes it more robust to novel Gaussian distractors (right). For more details, see Appendix B.

where no constant is hidden in $\mathcal { O }$ and $\begin{array} { r } { \bar { \varsigma } _ { k } = \varsigma _ { k } \big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ h _ { i } \in \mathcal { B } [ c _ { k } ] \} \big ) . } \end{array}$

The two proofs of these theorems use the same steps and are equally tight as shown in Appendix D. Equation (3) is also as tight as that of related work as discussed in Appendix C.2. The set $S$ is chosen to cover the original continuous space $\mathcal { H }$ in Theorem 2 (via the $\epsilon$ -covering $\mathcal { C }$ of $\mathcal { H }$ ), and its discretized space in Theorem 1. Equations (2)–(3) hold for all functions $\phi : \mathbb { R } ^ { m }  \mathbb { R }$ , including the maps that depend on the samples $( h _ { i } ) _ { i = 1 } ^ { n }$ via any learning processes. For example, we can set $\phi$ to be an evaluation criterion of the latent space $h$ or the composition of an evaluation criterion and any neural network layers that are learned with the samples $( h _ { i } ) _ { i = 1 } ^ { n }$ . In Appendix A, we present additional theorems, Theorems 3–4, where we analyze the effects of learning the map $x \mapsto h$ and the codebook e via input-target pair samples $( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { n }$ .

Intuitively, the proof shows that we achieve the improvement in sample efficiency when $G$ and $L$ are small, with the dependency on $G$ being significantly stronger (details in Appendix). Moreover, the dependency of the bound on the Lipschitz constant $\varsigma _ { k }$ is eliminated by using discretization. Our theorems 1–2 are applicable to all of our models for recurrent neural networks, transformers, and graph neural networks (since the function $\phi$ is arbitrary) in the following subsections.

Communication along edges of Graph Neural Network One model where relational information is naturally used is in a graph neural network for modelling object interactions and predicting the next time frame. We denote node representations by $\zeta _ { i } ^ { t }$ , edge representations by $\epsilon _ { i , j } ^ { t }$ , and actions applied to each node by $a _ { i } ^ { t }$ . Without DVNC, the changes of each nodes after each time step is computed by $\zeta _ { i } ^ { t + 1 } = \zeta _ { i } ^ { t } + \Delta \zeta _ { i } ^ { t }$ , where $\begin{array} { r } { \Delta \zeta _ { i } ^ { t } = f _ { n o d e } ( \zeta _ { i } ^ { t } , a _ { t } ^ { i } , \sum _ { j \neq i } \epsilon _ { t } ^ { i , j } ) } \end{array}$ and $\epsilon _ { i , j } ^ { t } = f _ { e d g e } ( \zeta _ { i } ^ { t } , z _ { j } ^ { t } )$ .

In this present work, we discretize the sum of all edges connected to each node with DVNC, as so: $\begin{array} { r } { \Delta \zeta _ { i } ^ { t } = \bar { f } _ { n o d e } ( \zeta _ { i } ^ { t } , a _ { t } ^ { i } , q ( \sum _ { j \neq i } \epsilon _ { t } ^ { i , j } , L , G ) ) } \end{array}$ .

Communication Between Positions in Transformers In a transformer model without DVNC, at each layer, the scaled dot product multi-head soft attention is applied to allow the model to jointly attend to information from different representation subspaces at different positions (Vaswani et al., 2017) as:

Output = residual + MULTIHEADATTENTION $( B , K , V )$

where MULTIHEADATTENTION $\mathrm { \Delta } _ { ! } ( B , K , V ) = \mathrm { C o N C A T E N A T E } ( h e a d _ { 1 } , h e a d _ { 2 } , . . . . h e a d _ { n } ) { \cal W } ^ { O }$ and $h e a d _ { i } = \mathrm { S O F T A T T E N T I O N } \big ( B W _ { i } ^ { B } , K W _ { i } ^ { K } , V W _ { i } ^ { V } \big )$ . Here, $W ^ { O } , W ^ { B } , W ^ { K }$ , and $W ^ { V }$ are projection matrices, and $B$ , $K$ , and $V$ are queries, keys, and values respectively.

In this present work, we applied the DVNC process to the results of the attention in the last two layers of transformer model, as so:

$$
\text {O u t p u t} = \text {r e s i d u a l} + q (\text {M u l t i H e a d A T T E N T I O N} (B, K, V), L, G).
$$

Communication with Modular Recurrent Neural Networks There have been many efforts to introduce modularity into RNN. Recurrent independent mechanisms (RIMs) activated different

modules at different time step based on inputs (Goyal et al., 2019). In RIMs, outputs from different moduleswe have $\hat { z } _ { i } ^ { t + 1 } = \mathbf { R } \mathbf { N } \mathbf { N } ( z _ { i } ^ { t } , x ^ { t } )$ ach other via soft attentio for active modules, and $\hat { z } _ { i ^ { \prime } } ^ { t + 1 } = z _ { i ^ { \prime } } ^ { t }$ m. In tfor in e original RIMs methoctive modules, where $t$ is the time step, $i$ is index of the module, and $x _ { t }$ is the input at time step $t$ . Then, the dot product query-key soft attention is used to communication output from all modules $i \in \{ 1 , \ldots , M \}$ such that $\bar { h } _ { i } ^ { t + 1 } = \bar { \mathrm { S O F T A T T E N T I O N } } ( \hat { z } _ { 1 } ^ { t + 1 } , \hat { z } _ { 2 } ^ { t + 1 } , . . . . \hat { z } _ { M } ^ { t + 1 } )$ , ... .zˆt+1M ).

In this present work, we applied the DVNC process to the output of the soft attention, like so: zt+1 = $\boldsymbol { z } _ { i } ^ { t + 1 } = \dot { \boldsymbol { z } } _ { i } ^ { t + 1 } + \boldsymbol { q } ( h _ { i } ^ { t + 1 } , L , \dot { G } )$ . Appendix E presents the pseudocode for RIMs with discretization.

# 3 Related Works

The Society of Specialists Our work is related to the theoretical nature of intelligence proposed by Minsky (1988) and others (Braitenberg, 1986; Fodor, 1983), in which the authors suggest that an intelligent mind can be built from many little specialist parts, each mindless by itself. Dividing model architecture into different specialists been the subject of a number of research directions, including neural module networks (Andreas et al., 2016)), multi-agent reinforcement learning (Zhang et al., 2019) and many others (Jacobs et al., 1991; Reed and De Freitas, 2015; Rosenbaum et al., 2019). Specialists for different computation processes have been introduced in many models such as RNNs and transformers (Goyal et al., 2019; Lamb et al., 2021; Goyal et al., 2021b). Specialists for entities or objects in fields including computer vision (Kipf et al., 2019). Methods have also been proposed to taking both entity and computational process into consideration (Goyal et al., 2020). In a more recent work, in addition to entities and computations, rules were considered when designing specialists (Goyal et al., 2021a). Our Discrete-Valued Neural Communication method can be seen as introducing specialists of representation into machine learning model.

Communication among Specialists Efficient communication among different specialist components in a model requires compatible representations and synchronized messages. In recent years, attention mechanisms are widely used for selectively communication of information among specialist components in machine learning modes (Goyal et al., 2019, 2021b,a) and transformers (Vaswani et al., 2017; Lamb et al., 2021).collective memory and shared RNN parameters have also been used for multi-agent communication (Garland and Alterman, 1996; Pesce and Montana, 2020). Graph-based models haven been widely used in the context of relational reasoning, dynamical system simulation , multi-agent systems and many other fields. In graph neural networks, communication among different nodes were through edge attributes that a learned from the nodes the edge is connecting together with other information (Kipf et al., 2019; Scarselli et al., 2008; Bronstein et al., 2017; Watters et al., 2017; Van Steenkiste et al., 2018; Kipf et al., 2018; Battaglia et al., 2018; Tacchetti et al., 2018; Veerapaneni et al., 2019).Graph based method represent entities, relations, rules and other elements as node, edge and their attributes in a graph (Koller and Friedman, 2009; Battaglia et al., 2018). In graph architectures, the inductive bias is assuming the system to be learnt can be represented as a graph. In this study, our DVNC introduces inductive bias that forces inter-component communication to be discrete and share the same codebook. The combinatorial properties come from different combinations of latent vectors in each head and different combination of heads in each representation vector.While most of inter-specialist communication mechanisms operates in a pairwise symmetric manner, Goyal et al. (2021b) introduced a bandwidth limited communication channel to allow information from a limited number of modules to be broadcast globally to all modules, inspired by Global workspace theory (Baars, 2019, 1993).our proposed method selectively choose what information can communicated from each module. We argue these two methods are complimentary to each other and can be used together, which we like to investigate in future studies.

# 4 Experiments

In this study we have two hypothesis: 1) The use of discrete symbols limits the bandwidth of communication. 2) The use of shared discrete symbols will promote systematic generalization.Theoretical results obtained in section 2, agree with hypothesis 1. In order to verify hypothesis 2, in this section, we design and conduct empirical experiments to show that discretization of inter-component communication improves OOD generalization and model performance.

Table 2: Performance of Transformer Models with Discretized Communication on the Sort-of-Clevr Visual Reasoning Task.   

<table><tr><td>Method</td><td>Ternary Accuracy</td><td>Binary Accuracy</td><td>Unary Accuracy</td></tr><tr><td>Transformer baseline</td><td>57.25 ± 1.30</td><td>76.00 ± 1.41</td><td>97.75 ± 0.83</td></tr><tr><td>Discretized transformer (G=16)</td><td>61.33 ± 2.62</td><td>84.00 ± 2.94</td><td>98.00 ± 0.89</td></tr><tr><td>Discretized transformer (G=8)</td><td>62.67 ± 1.70</td><td>88.00 ± 0.82</td><td>98.75 ± 0.43</td></tr><tr><td>Discretized transformer (G=1)</td><td>58.50 ± 4.72</td><td>80.50 ± 7.53</td><td>98.50 ± 0.50</td></tr></table>

![](images/3e81ae5b6440f0dc81dd3789c7e275a11e0f6f479196dfcf0c2919b1d9162d40.jpg)  
(a) 2D Shapes

![](images/ea4deaf83374d49870f678d27826e741dd8e937f23cba1b0eb18d4b51c408244.jpg)  
(b) 3D Shapes

![](images/43c8e555ae4f08eb0c9f36d21e2d176c2d50ab46d2d8a4b70419865c5b84e018.jpg)  
(c) Three-Body

![](images/1cbee34c575327ccd34235d5f0e65fdd8121098665df99a54ee67e25a0cbcea3.jpg)  
(d) Adding

![](images/775d8e2e6c568e079e05ea693ea2f1dd547c7e055aac7bf4cb5fa07b0614fe93.jpg)  
(e) Seq. MNIST   
Figure 4: GNN models (a,b and c) and RIMs with DVNC (d and e) show improved OOD generalization. Test data in OOD 1 to 3 are increasingly different from training distribution.

# 4.1 Communication along Edges of Graph Neural Networks

We first turn our attention to visual reasoning task using GNNs and show that GNNs with DVNC have improved OOD generalization. In the tasks, the sequences of image frames in test set have different distribution from data used for training. We set up the model in a identical manner as in Kipf et al. 2019 (Kipf et al., 2019) except the introduction of DVNC. A CNN is used to to extract different objects from images. Objects are represented as nodes and relations between pairs of objects are represented as edges in the graph. The changes of nodes and edges can be learned by message passing in GNN. The information passed by all edges connected to a node is discretized by DVNC (see details in Section 2).

The following principles are followed when choosing and designing GNN visual reasoning tasks as well as all other OOD generalization tasks in subsequent sections: 1) Communication among different components are important for the tasks. 2) meaning of information communicated among components can be described in a discrete manner. 3) In OOD setting,distributions of information communicated among specialist components in test data also deviate from that in training data.

Object Movement Prediction in Grid World We begin with the tasks to predict object movement in grid world environments. Objects in the environments are interacting with each other and their movements are manipulated by actions applied to each object that give a push to the object in a randomly selected direction . Machine learning models are tasked to predict next positions of all objects in the next time step $t + 1$ given their position and actions in time step t. We can think of this task as detecting whether an object can be pushed toward a direction. If object is blocked by the environment or another object in certain direction, then it can not be moved in that direction. Positions of objects are captures by nodes in GNN and the relative positions among different objects can be communicated in a discrete manner via message passing through edges.

We adapted and modified the original 2D shapes and 3D shapes movement tasks from Kipf et al. (2019) by introducing different number of objects in training or testing environment. In both 2D and 3D shapes tasks ,five objects are available in training data, three objects are available in OOD-1 and only two objects are available in OOD-2. Our experimental results suggest that DVNC in GNN improved OOD generalization in a statistically significant manner (Figure 4). In addition, the improvement is robust across different hyperparameters $G$ and $L$ (Figure 5). Details of the visual reasoning tasks set up and model hyperparameters can be found in Appendix.

Three-body Physics Object Movement Prediction Next, we turn our attention to a three-bodyphysics environment in which three balls interacting with each other and move according to physical laws in classic mechanics in a 2D environment. There are no external actions applied on the objects. We adapted and modified the three-body-physics environment from (Kipf et al., 2019). In OOD experiment, the sizes of the balls are different from the original training data. Details of the

Table 3: Graph Neural Networks benefit from discretized communication on OOD generalization in predicting movement in Atari games.   

<table><tr><td>Game</td><td>GNN (Baseline)</td><td>GNN (Discretized)</td><td>P-Value (vs. baseline)</td><td>Game</td><td>GNN (Baseline)</td><td>GNN (Discretized)</td><td>P-Value (vs. baseline)</td></tr><tr><td>Alien</td><td>0.1991 ± 0.0786</td><td>0.2876 ± 0.0782</td><td>0.00019</td><td>DoubleDunk</td><td>0.8680 ± 0.0281</td><td>0.8793 ± 0.0243</td><td>0.04444</td></tr><tr><td>BankHeist</td><td>0.8224 ± 0.0323</td><td>0.8459 ± 0.0490</td><td>0.00002</td><td>MsPacman</td><td>0.2005 ± 0.0362</td><td>0.2325 ± 0.0648</td><td>0.05220</td></tr><tr><td>Berzerk</td><td>0.6077 ± 0.0472</td><td>0.6233 ± 0.0509</td><td>0.06628</td><td>Pong</td><td>0.1440 ± 0.0845</td><td>0.2965 ± 0.1131</td><td>0.00041</td></tr><tr><td>Boxing</td><td>0.9228 ± 0.0806</td><td>0.9502 ± 0.0314</td><td>0.69409</td><td>SpaceInvaders</td><td>0.0460 ± 0.0225</td><td>0.0820 ± 0.0239</td><td>0.00960</td></tr></table>

experimental set up can be found in Appendix. Our experimental results show that GNN with DVNC improved OOD generalization (Figure 4 ).

Movement Prediction in Atari Games Similarly, we design OOD movement prediction tasks for 8 Atari games. Changes of each image frame depends on previous image frame and actions applied upon different objects. A different starting frame number is used to make the testing set OOD. GNN with DVNC showed statistically significant improvement in 6 out of 8 games and marginally significant improvement in the other games (Table 3).

# 4.2 Communication Between Positions in Transformers

In transformer models, attention mechanism is used to communicate information among different position. We design and conduct two visual reasoning tasks to understand if discretizing results of attention in transformer models will help improve the performance (Section 2). In the tasks, transformers take sequence of pixel as input and detect relations among different objects which are communicated through attention mechanisms.

We experimented with the Sort-of-CLEVR visual relational reasoning task, where the model is tasked with answering questions about certain properties of various objects and their relations with other objects (Santoro et al., 2017). Each image in Sort-of-CLEVR contains randomly placed geometrical shapes of different colors and shapes. Each image comes with 10 relational questions and 10 nonrelational questions. Nonrelational questions only consider properties of individual objects. On the other hand, relational questions consider relations among multiple objects. The input to the model consists of the image and the corresponding question. Each image and question come with a finite number of possible answers and hence this task is to classify and pick the correct the answer(Goyal et al., 2021b). Transformer models with DVNC show significant improvement (Table 2).

# 4.3 Communication with Modular Recurrent Neural Networks

Recurrent Independent Mechanisms(RIMs) are RNN with modular structures. In RIMs, units communicate with each other using attention mechanisms at each time step. We discretize results of inter-unit attention in RIMs (Section 2). We conducted a numeric reasoning task and a visual reasoning task to understand if DVNC applied to RIMs improves OOD generalization.

We considered a synthetic adding task in which the model is trained to compute the sum of a sequence of numbers followed by certain number of dummy gap tokens(Goyal et al., 2019). In OOD settings of the task, the number of gap tokens after the target sequence is different in test set from training data. Our results show that DVNC makes significnat improvement in the OOD task(Figure 4).

For further evidence that RIMs with DVNC can better achieve OOD generlization, we consider the task of classifying MNIST digits as sequences of pixels (Krueger et al., 2016) and assay generalization to images of resolutions different from those seen during training. Our results suggest that RIMs with DVNC have moderately better OOD generalization than the baseline especially when the test images are very different from original training data (Figure 4).

# 4.4 Analysis and Ablations

Discretizing Communication Results is Better than Discretizing other Parts of the Model: The intuition is that discretizing the results of communication with a shared codebook encourages more reliable and independent processing by different specialists. We experimentally tested this key hypothesis on the source of the DVNC’s success by experimenting with discretizing other parts of the model. For example, in RIMs, we tried discretizing the updates to the recurrent state and

![](images/f80b62c678d9480a9751263e7120346074d9d3d91b2d58dc0cd8f6a3a490c8f6.jpg)  
(a) 2D Shapes (G)

![](images/9a005ff225ec33c51be9fa52904ef38206b29688fa43e7c7a7425e7b7f033427.jpg)  
(b) 2D Shapes (L)

![](images/9535fdaeb1944d2116585cd3195f89198daac911d2441b07469f6b736522544a.jpg)  
(c) 3D Shapes (G)

![](images/82dcb4dbff50b69b22fef6c95f0553f471a31de71501aa6c09b75c04305e9417.jpg)  
(d) 3D Shapes (L)

![](images/af30f4928c4b4313cbd3f1256e15155ca47e90d102dcddfb9d2d3c6902f00674.jpg)  
(e) 2D Shapes(GNN)

![](images/4ccf560edaae15b2f229d0f07e2091f43f93c534af4291ec489204cebacb9c98.jpg)  
(f) 3D Shapes(GNN)

![](images/0db06d0ee751a64b559db59449cdd660eecb305f645902372908af90c6751c85.jpg)  
(g) Adding Task(RIM)   
Figure 5: Upper Row: Models with DVNC have improved OOD generalization in wide range of hyperparameter settings. Red dots are performances of models with DVNC with different codebook size (L) and number of heads (G). Black dashed lines are performance of baseline methods without discretization. Lower row: (e) and (f) compare OOD generalization in HITs (higher is better) between GNNs with results of communication discretized vs. edge states discretized. (g) Compares RIMs model with results of communication discretized vs. other vectors discretized.

tried discretizing the inputs. On the adding task, this led to improved results over the baseline, but performed much worse than discretizing communication. For GNNs we tried discretizing the input to the communication (the edge hidden states) instead of the result of communication, and found that it led to significantly worse results and had very high variance between different trials. These results are in Figure 5.

VQ-VAE Discretization Outperforms Gumbel-Softmax: The main goal of our work was to demonstrate the benefits of communication with discrete values, and this discretization could potentially be done through a number of different mechanisms. Experimentally, we found that the nearest-neighbor and straight-through estimator based discretization technique,similar to method used in VQ-VAE, outperformed the use of a Gumbel-Softmax to select the discrete tokens (Figure 6 in Appendix). These positive results led us to focus on the VQ-VAE discretization technique, but in principle other mechanisms could also be recruited to accomplish our discrete-valued neural communication framework. We envision that DVCN should work complementarily with future advances in learning discrete representations.

# 5 Discussion

With the evolution of deep architectures from the monolithic MLP to complex architectures with specialized components, we are faced with the issue of how to ensure that subsystems can communicate and coordinate. Communication via continuous, high-dimensional signals is a natural choice given the history of deep learning, but our work argues that discretized communication results in more robust, generalizable learning. Discrete-Valued Neural Communication (DVNC) achieves a much lower noise-sensitivity bound while allowing high expressiveness through the use of multiple discretization heads. This technique is simple and easy-to-use in practice and improves out-ofdistribution generalization. DVNC is applicable to all structured architectures that we have examined where inter-component communication is important and the information to be communicated can be discretized by its nature.

Limitations The proposed method has two major limitations. First, DVNC can only improve performance if communication among specialists is important for the task. If the different components do not have good specialization, then DVNC’s motivation is less applicable. Another limitation is that the discretization process can reduce the expressivity of the function class, although this can be mitigated by using a large value for $G$ and $L$ and can be partially monitored by the quantity of training data (e.g., training loss) similarly to the principle of structural minimization. Hence future work could examine how to combine discrete communication and continuous communication.

Social Impact Research conducted in this study is purely technical. The authors expect no direct negative nor positive social impact.

# References

Andreas, J., Rohrbach, M., Darrell, T., and Klein, D. (2016). Neural module networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 39–48.   
Baars, B. J. (1993). A cognitive theory of consciousness. Cambridge University Press.   
Baars, B. J. (2019). On Consciousness: Science & Subjectivity. Nautilus Press.   
Bahdanau, D., Cho, K., and Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.   
Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261.   
Bottou, L. and Gallinari, P. (1991). A framework for the cooperation of learning algorithms. In Advances in neural information processing systems, pages 781–788.   
Braitenberg, V. (1986). Vehicles: Experiments in synthetic psychology. MIT press.   
Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., and Vandergheynst, P. (2017). Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 34(4):18–42.   
Chmielewski, M. R. and Grzymala-Busse, J. W. (1996). Global discretization of continuous attributes as preprocessing for machine learning. International journal of approximate reasoning, 15(4):319– 331.   
Fernando, C., Banarse, D., Blundell, C., Zwols, Y., Ha, D., Rusu, A. A., Pritzel, A., and Wierstra, D. (2017). Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint arXiv:1701.08734.   
Fodor, J. A. (1983). The modularity of mind. MIT press.   
Fujii, N. and Graybiel, A. M. (2003). Representation of action sequence boundaries by macaque prefrontal cortical neurons. Science, 301(5637):1246–1249.   
Garland, A. and Alterman, R. (1996). Multiagent learning through collective memory. In Adaptation, Coevolution and Learning in Multiagent Systems: Papers from the 1996 AAAI Spring Symposium, pages 33–38.   
Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. (2017). Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1263–1272. JMLR. org.   
Goyal, A. and Bengio, Y. (2020). Inductive biases for deep learning of higher-level cognition. arXiv preprint arXiv:2011.15091.   
Goyal, A., Didolkar, A., Ke, N. R., Blundell, C., Beaudoin, P., Heess, N., Mozer, M., and Bengio, Y. (2021a). Neural production systems. arXiv preprint arXiv:2103.01937.   
Goyal, A., Didolkar, A., Lamb, A., Badola, K., Ke, N. R., Rahaman, N., Binas, J., Blundell, C., Mozer, M., and Bengio, Y. (2021b). Coordination among neural modules through a shared global workspace. arXiv preprint arXiv:2103.01197.   
Goyal, A., Lamb, A., Gampa, P., Beaudoin, P., Levine, S., Blundell, C., Bengio, Y., and Mozer, M. (2020). Object files and schemata: Factorizing declarative and procedural knowledge in dynamical systems. arXiv preprint arXiv:2006.16225.   
Goyal, A., Lamb, A., Hoffmann, J., Sodhani, S., Levine, S., Bengio, Y., and Schölkopf, B. (2019). Recurrent independent mechanisms. arXiv preprint arXiv:1909.10893.   
Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. (1991). Adaptive mixtures of local experts. Neural computation, 3(1):79–87.

Kipf, T., Fetaya, E., Wang, K.-C., Welling, M., and Zemel, R. (2018). Neural relational inference for interacting systems. arXiv preprint arXiv:1802.04687.   
Kipf, T., van der Pol, E., and Welling, M. (2019). Contrastive learning of structured world models. arXiv preprint arXiv:1911.12247.   
Koller, D. and Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT press.   
Krueger, D., Maharaj, T., Kramár, J., Pezeshki, M., Ballas, N., Ke, N. R., Goyal, A., Bengio, Y., Courville, A., and Pal, C. (2016). Zoneout: Regularizing rnns by randomly preserving hidden activations. arXiv preprint arXiv:1606.01305.   
Laird, J. E., Rosenbloom, P. S., and Newell, A. (1986). Chunking in soar: The anatomy of a general learning mechanism. Machine learning, 1(1):11–46.   
Lamb, A., He, D., Goyal, A., Ke, G., Liao, C.-F., Ravanelli, M., and Bengio, Y. (2021). Transformers with competitive ensembles of independent mechanisms. arXiv preprint arXiv:2103.00336.   
Minsky, M. (1988). Society of mind. Simon and Schuster.   
Oord, A. v. d., Vinyals, O., and Kavukcuoglu, K. (2017). Neural discrete representation learning. arXiv preprint arXiv:1711.00937.   
Pesce, E. and Montana, G. (2020). Improving coordination in small-scale multi-agent deep reinforcement learning through memory-driven communication. Machine Learning, pages 1–21.   
Quiroga, R., Reddy, L., Kreiman, G., Koch, C., and Fried, I. (2005). Invariant visual representation by single neurons in the human brain. Nature, 435(7045):1102–1107.   
Raposo, D., Santoro, A., Barrett, D., Pascanu, R., Lillicrap, T., and Battaglia, P. (2017). Discovering objects and their relations from entangled scene representations. arXiv preprint arXiv:1702.05068.   
Reed, S. and De Freitas, N. (2015). Neural programmer-interpreters. arXiv preprint arXiv:1511.06279.   
Ronco, E., Gollee, H., and Gawthrop, P. J. (1997). Modular neural networks and self-decomposition. Technical Report CSC-96012.   
Rosenbaum, C., Cases, I., Riemer, M., and Klinger, T. (2019). Routing networks and the challenges of modular and compositional computation. arXiv preprint arXiv:1904.12774.   
Rosenbaum, C., Klinger, T., and Riemer, M. (2017). Routing networks: Adaptive selection of non-linear functions for multi-task learning. arXiv preprint arXiv:1711.01239.   
Santoro, A., Raposo, D., Barrett, D. G., Malinowski, M., Pascanu, R., Battaglia, P., and Lillicrap, T. (2017). A simple neural network module for relational reasoning. In Advances in neural information processing systems, pages 4967–4976.   
Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., and Monfardini, G. (2008). The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80.   
Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538.   
Sokolic, J., Giryes, R., Sapiro, G., and Rodrigues, M. (2017a). Generalization error of invariant classifiers. In Artificial Intelligence and Statistics, pages 1094–1103.   
Sokolic, J., Giryes, R., Sapiro, G., and Rodrigues, M. R. (2017b). Robust large margin deep neural networks. IEEE Transactions on Signal Processing.   
Sun, C., Yang, W., Martin, J., and Tonegawa, S. (2020). Hippocampal neurons represent events as transferable units of experience. Nature Neuroscience, 23(5):651–663.

Tacchetti, A., Song, H. F., Mediano, P. A., Zambaldi, V., Rabinowitz, N. C., Graepel, T., Botvinick, M., and Battaglia, P. W. (2018). Relational forward models for multi-agent learning. arXiv preprint arXiv:1809.11044.   
Tsao, D. Y., Freiwald, W. A., Tootell, R. B. H., and Livingstone, M. S. (2006). A cortical region consisting entirely of face-selective cells. Science, 311(5761):670–674.   
van der Vaart, A. W. and Wellner, J. A. (1996). Weak Convergence and Empirical Processes. Springer New York.   
Van Steenkiste, S., Chang, M., Greff, K., and Schmidhuber, J. (2018). Relational neural expectation maximization: Unsupervised discovery of objects and their interactions. arXiv preprint arXiv:1802.10353.   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems, pages 5998–6008.   
Veerapaneni, R., Co-Reyes, J. D., Chang, M., Janner, M., Finn, C., Wu, J., Tenenbaum, J. B., and Levine, S. (2019). Entity abstraction in visual model-based reinforcement learning. CoRR, abs/1910.12827.   
Watters, N., Zoran, D., Weber, T., Battaglia, P., Pascanu, R., and Tacchetti, A. (2017). Visual interaction networks: Learning a physics simulator from video. In Advances in neural information processing systems, pages 4539–4547.   
Wills, T., Lever, C., Cacucci, F., Burgess, N., and O’Keefe, J. (2005). Attractor dynamics in the hippocampal representation of the local environment. Science, 308(5723):873–876.   
Xu, H. and Mannor, S. (2012). Robustness and generalization. Machine learning, 86(3):391–423.   
Zhang, K., Yang, Z., and Ba¸sar, T. (2019). Multi-agent reinforcement learning: A selective overview of theories and algorithms. arXiv preprint arXiv:1911.10635.

# Checklist

The checklist follows the references. Please read the checklist guidelines carefully for information on how to answer these questions. For each question, change the default [TODO] to [Yes] , [No] , or [N/A] . You are strongly encouraged to include a justification to your answer, either by referencing the appropriate section of your paper or providing a brief inline description. For example:

• Did you include the license to the code and datasets? [Yes] See Section 2 (‘General formatting instructions’) of the tex template (‘Formatting Instructions For NeurIPS 2021’).   
• Did you include the license to the code and datasets? [No] The code and the data are proprietary.   
• Did you include the license to the code and datasets? [N/A]

Please do not modify the questions and only use the provided macros for your answers. Note that the Checklist section does not count towards the page limit. In your paper, please delete this instructions block and only keep the Checklist section heading above along with the questions/answers below.

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] Abstract and introduction directly reflect the paper’s contribution and scope   
(b) Did you describe the limitations of your work? [Yes] see last paragraph of section 6 of the manuscript   
(c) Did you discuss any potential negative societal impacts of your work? [N/A] This study is purely technical and do not involve any social aspects   
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] See section 2 and related appendix   
(b) Did you include complete proofs of all theoretical results? [Yes] see appendix

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] see section 2, URL available after double-blind review   
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] see appendix   
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] see figures with bar and box plot, error bars and p-val provided   
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] See appendix

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes] all cited   
(b) Did you mention the license of the assets? [Yes]   
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes] new dataset released together with code on github after double-blind review   
(d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [No] publicly available data used   
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A] no personally identifiable information included

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]   
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]   
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

# A Additional theorems for theoretical motivations

In this appendix, as a complementary to Theorems 1–2, we provide additional theorems, Theorems 3–4, which further illustrate the two advantages of the discretization process by considering an abstract model with the discretization bottleneck. For the advantage on the sensitivity, the error due to potential noise and perturbation without discretization — the third term $\xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d ) > 0$ in Theorem 4 — is shown to be minimized to zero with discretization in Theorems 3. For the second advantage, the underlying dimensionality of $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) + \ln ( \mathcal { N } _ { ( \mathcal { M } , d ) } ( r , \Theta ) / \delta )$ without discretization (in the bound of Theorem 4) is proven to be reduced to the typically much smaller underlying dimensionality of $L ^ { G } + \mathrm { l n } ( \mathcal { N } _ { ( \mathcal { M } , d ) } \mathsf { \bar { ( } } r , E \times \Theta )$ with discretization in Theorems 3. Here, for any metric space $( \mathcal { M } , d )$ and subset $M \subseteq { \mathcal { M } }$ , the $r$ -converging number of $M$ is defined by $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , M ) = \operatorname* { m i n } \left\{ | \mathcal { C } | : \mathcal { C } \subseteq \mathcal { M } , M \subseteq \cup _ { c \in \mathcal { C } } \mathcal { B } _ { ( \mathcal { M } , d ) } [ c , r ] \right\}$ where the (closed) ball of radius $r$ at centered at $c$ is denoted by $\mathcal { B } _ { ( \mathcal { M } , d ) } [ c , r ] = \{ x \in \mathcal { M } : d ( x , c ) \leq r \}$ . See Appendix C.1 for a simple comparison between the bound of Theorem 3 and that of Theorem 4 when the metric spaces $( \mathcal { M } , d )$ and $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ are chosen to be Euclidean spaces.

We now introduce the notation used in Theorems 3–4. Let $q _ { e } ( h ) : = q ( h , L , G )$ . The models are defined by $\tilde { f } ( x ) : = \tilde { f } ( x , w , \theta ) : = ( \varphi _ { w } \circ h _ { \theta } ) ( x )$ without the discretization and $f ( x ) : = f ( x , w , e , \theta ) : =$ $( \varphi _ { w } \circ q _ { e } \circ h _ { \theta } ) ( x )$ with the discretization. Here, $\varphi _ { w }$ represents a deep neural network with weight parameters $w \in \mathcal { W } \subset \mathbb { R } ^ { D }$ , $q _ { e }$ is the discretization process with the codebook $e \in E \subset \mathbb { R } ^ { L \times m }$ , and $h _ { \theta }$ represents a deep neural network with parameters $\theta \in \Theta \subset \mathbb { R } ^ { \zeta }$ . Thus, the tuple of all learnable parameters are $( w , e , \theta )$ . For the codebook space, $E = E _ { 1 } \times E _ { 2 }$ with $E _ { 1 } \subset \bar { \mathbb { R } } ^ { L }$ and $E _ { 2 } \subset \mathbb { R } ^ { m }$ . Moreover, let $J : ( f ( x ) , y ) \mapsto J ( f ( x ) , y ) \in \bar { \mathbb { R } }$ be an arbitrary (fixed) function, $h _ { \theta } ( x ) \in \mathcal { H } \subset \mathbb { R } ^ { m }$ , $x \in \mathcal { X }$ , and $\boldsymbol { y } \in \mathcal { V } = \{ \boldsymbol { y } ^ { ( 1 ) } , \boldsymbol { y } ^ { ( 2 ) } \}$ for some $y ^ { ( 1 ) }$ and $y ^ { ( 2 ) }$ .

Theorem 3. (with discretization) Let $C _ { J } ( w )$ be the smallest real number such that $| J ( \varphi _ { w } ( \eta ) , y ) | \le$ $C _ { J } ( w )$ for all $( \eta , y ) \in E _ { 2 } \times \mathcal { Y }$ . Let $\rho \in \mathbb { N } ^ { + }$ and $( \mathcal { M } , d )$ be a matric space such that $E \times \Theta \subseteq { \mathcal { M } }$ . Then, for any $\delta > 0$ , with probability at least $1 - \delta$ over an iid draw of n examples $( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { n }$ , the following holds: for any $( w , e , \theta ) \in \mathcal { W } \times E \times \Theta$ ,

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ J (f (x, w, e, \theta), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x _ {i}, w, e, \theta), y _ {i}) \right| \\ \leq C _ {J} (w) \sqrt {\frac {4 L ^ {G} \ln 2 + 2 \ln (\mathcal {N} _ {(\mathcal {M} , d)} (r , E \times \Theta) / \delta)}{n}} + \sqrt {\frac {\mathcal {L} _ {d} (w) ^ {2 / \rho}}{n}}, \\ \end{array}
$$

where $r ~ = ~ \mathcal { L } _ { d } ( w ) ^ { 1 / \rho - 1 } \sqrt { \frac { 1 } { n } }$ and $\mathcal { L } _ { d } ( w ) ~ \geq ~ 0$ is the smallest real number such that for all $( e , \theta )$ and $( e ^ { \prime } , \theta ^ { \prime } )$ in $E \times \Theta$ , $| \psi _ { w } ( e , \theta ) - \psi _ { w } ( e ^ { \prime } , \theta ^ { \prime } ) | \leq \mathcal { L } _ { d } ( w ) d ( ( e , \theta ) , ( e ^ { \prime } , \theta ^ { \prime } ) )$ with $\psi _ { w } ( e , \theta ) =$ $\begin{array} { r } { \mathbb { E } _ { x , y } [ J ( f ( x ) , y ) ] - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } J ( f ( x _ { i } ) , y _ { i } ) } \end{array}$

Theorem 4. (without discretization) Let ${ \tilde { C } } _ { J } ( w )$ be the smallest real number such that $| J ( \varphi _ { w } \circ$ $h _ { \theta } ) ( x ) , y ) | \leq \tilde { C } _ { J } ( w )$ for all $( \theta , x , y ) \in \Theta \times \mathcal { X } \times \mathcal { Y }$ . Let $\rho ~ \in ~ \mathbb { N } ^ { + }$ and $( \mathcal { M } , d )$ be a matric space such that $\Theta \subseteq { \mathcal { M } }$ . Let $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ be a matric space such that $\mathcal { H } \subseteq { \mathcal { M } } ^ { \prime }$ . Fix $r ^ { \prime } > 0$ and $\bar { \mathscr { z } } _ { r ^ { \prime } , d ^ { \prime } } \in \mathrm { a r g m i n } _ { \mathscr { C } } \{ | \mathscr { C } | : \mathscr { C } \subseteq \mathscr { M } ^ { \prime } , \mathscr { H } \subseteq \cup _ { c \in \mathscr { C } } \mathscr { B } _ { ( M ^ { \prime } , d ^ { \prime } ) } [ c , r ^ { \prime } ] \}$ }. Assume that for any $c \in \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } }$ , we have $| ( J ( \varphi _ { w } ( h ) , y ) - ( J ( \varphi _ { w } ( h ^ { \prime } ) , y ) | \leq \xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d ) .$ for any $h , h ^ { \prime } \in \mathcal { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c , r ^ { \prime } ]$ and $y \in \mathcal { V }$ . Then, for any $\delta > 0$ , with probability at least $1 - \delta$ over an iid draw of $n$ examples $( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { n }$ , the following holds: for any $( w , \theta ) \in \dot { \mathcal { W } } \times \Theta$ ,

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ J (\tilde {f} (x, w, \theta), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x _ {i}, w, \theta), y _ {i}) \right| \\ \leq \tilde {C} _ {J} (w) \sqrt {\frac {4 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} (r ^ {\prime} , \mathcal {H}) \ln 2 + 2 \ln (\mathcal {N} _ {(\mathcal {M} , d)} (r , \Theta) / \delta)}{n}} + \sqrt {\frac {\tilde {\mathcal {L}} _ {d} (w) ^ {2 / \rho}}{n}} + \xi (w, r ^ {\prime}, \mathcal {M} ^ {\prime}, d), \\ \end{array}
$$

where $r = \tilde { \mathcal { L } } _ { d } ( w ) ^ { 1 / \rho - 1 } \sqrt { \frac { 1 } { n } }$ and $\tilde { \mathcal { L } } _ { d } ( w ) \geq 0$ is the smallest real number such that for all $\theta$ and $\theta ^ { \prime }$ in $\begin{array} { r } { \Theta , | \tilde { \psi } _ { w } ( \theta ) - \tilde { \psi } _ { w } ( \theta ^ { \prime } ) | \overset { } { \le } \tilde { \mathcal { L } } _ { d } ( w ) d ( \theta , \theta ^ { \prime } ) { \mathrm { ~ } } w i t h \ \tilde { \psi } _ { w } ( \theta ) = \mathbb { E } _ { x , y } [ J ( \tilde { f } ( x ) , y ) ] - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } J ( \tilde { f } ( x _ { i } ) , y _ { i } ) . } \end{array}$ $\begin{array} { r } { \Im , | \tilde { \psi } _ { w } ( \theta ) - \tilde { \psi } _ { w } ( \theta ^ { \prime } ) | \le \tilde { \mathcal { L } } _ { d } ( w ) d ( \theta , \theta ^ { \prime } ) } \end{array}$

Note that we have $C _ { J } ( w ) \leq \tilde { C } _ { J } ( w )$ and $\mathcal { L } _ { d } ( w ) \approx \tilde { \mathcal { L } } _ { d } ( w )$ by their definition. For example, if we set $J$ to be a loss criterion, the bound in Theorem 4 becomes in the same order as and comparable to the

![](images/5830de670932d346b8f7f61acdbebf88cf75e5b4fe71db034f1040b276920730.jpg)  
Figure 6: Performance on adding task (RIMs) with no discretization, Gumbel-Softmax discretization, or VQ-VAE style discretization (ours). Test length $1 { = } 5 0 0$ is in-distribution test result and test length $\scriptstyle 1 0 0 0$ is out-of-distribution results.

generalization bound via the algorithmic robustness approach proposed by the previous papers (Xu and Mannor, 2012; Sokolic et al., 2017a,b), as we show in Appendix C.2.

# B Additional Experiments

# C Additional discussions on theoretical motivations

# C.1 Simple comparison of Theorems 3 and 4 with Euclidean space

For the purpose of of the comparison, we will now consider the simple worst case with no additional structure with the Euclidean space to instantiate $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) , \mathcal { N } _ { ( \mathcal { M } , d ) } ( r , \Theta )$ , and $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , E \times$ $\Theta$ ). It should be obvious that we can improve the bounds via considering metric spaces with additional structures. For example, we can consider a lower dimensional manifold $\mathcal { H }$ in the ambient space of $\mathbb { R } ^ { m }$ to reduce $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } )$ . Similar ideas can be applied for $\Theta$ and $E \times \Theta$ . Furthermore, the invariance as well as margin were used to reduce the bound on $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { X } )$ in previous works (Sokolic et al., 2017a,b) and similar ideas can be applied for $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) , \mathcal { N } _ { ( \mathcal { M } , d ) } ( r , \Theta )$ , and $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , E \times \Theta )$ . In this regard, the discretization can be viewed as a method to minimize $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } )$ to easily controllable $L ^ { G }$ while minimizing the sensitivity term $\xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d )$ to zero at the same time in Theorems 3 and 4.

Suppose that for any $y ~ \in ~ \mathcal { V }$ , the function $\begin{array} { r l } { h } & { { } \mapsto \quad J ( \varphi _ { w } ( h ) , y ) } \end{array}$ is Lipschitz continuous as $| ( \bar { J } ( \varphi _ { w } ( h ) , y ) - ( \bar { J } ( \varphi _ { w } ( h ^ { \prime } ) , y ) | \leq \varsigma ( w ) d ( h , h ^ { \prime } )$ . Then, we can set $\xi ( w , \bar { r } ^ { \prime } , \mathcal { M } ^ { \prime } , d ) = 2 \varsigma ( w ) r ^ { \prime }$ since $d ( h , h ^ { \prime } ) \leq 2 r ^ { \prime }$ for any $h , h ^ { \prime } \in \mathcal { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c , r ^ { \prime } ]$ .

As an simple example, let us choose the metric space $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ to be the Euclidean space $\mathbb { R } ^ { m }$ with the Euclidean metric and $\mathcal { H } \subset \mathbb { R } ^ { m }$ such that $\| v \| _ { 2 } \le R _ { \mathcal { H } }$ for all $v \in \mathcal H$ . Then, we have $\mathcal { N } _ { ( { \mathcal { M } } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) \leq ( 2 R _ { { \mathcal { H } } } \sqrt { m } / r ^ { \prime } ) ^ { m }$ and we can set $\xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d ) = 2 \varsigma ( w ) r ^ { \prime }$ . Thus, by setting $r ^ { \prime } = R _ { \mathcal { H } } / 2$ , we can replace $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } )$ by $( 4 \sqrt { m } ) ^ { m }$ and set $\xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d ) = \varsigma ( w ) R _ { \mathcal { H } }$ .

Similarly, let us choose the metric space $( \mathcal { M } , d )$ to be the Euclidean space with the Euclidean metric and $E \subset \mathbb { R } ^ { L m }$ and $\Theta \subset \mathbb { R } ^ { \zeta }$ such that $\| v \| _ { 2 } \le R _ { E }$ for all $v \in E$ and $\lVert \boldsymbol { v } \rVert _ { 2 } \leq R _ { \Theta }$ for all

$v \in \Theta$ . This implies that $\| ( v _ { E } , v _ { \theta } ) \| _ { 2 } \leq \sqrt { R _ { E } ^ { 2 } + R _ { \Theta } ^ { 2 } }$ . Thus, we have $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , \Theta ) \leq ( 2 R _ { \Theta } \sqrt { \zeta } / r ) ^ { \zeta }$ and $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , E \times \Theta ) \ : \le \ : ( 2 \sqrt { R _ { E } ^ { 2 } + R _ { \Theta } ^ { 2 } } \sqrt { L m + \zeta } / r ) ^ { L m + \zeta }$ . Since $r = \tilde { \mathcal { L } } _ { d } ( w ) ^ { 1 / \rho - 1 } \sqrt { \frac { 1 } { n } }$ and $r = \mathcal { L } _ { d } ( w ) ^ { 1 / \rho - 1 } \sqrt { \frac { 1 } { n } }$ , we can replace $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , \Theta )$ by $( 2 R _ { \Theta } \tilde { \mathcal { L } } _ { d } ( w ) ^ { 1 - 1 / \rho } \sqrt { \zeta n } ) ^ { \zeta }$ and $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , E \times$ $\Theta )$ by $( 2 \mathcal { L } _ { d } ( w ) ^ { 1 - 1 / \rho } \sqrt { R _ { E } ^ { 2 } + R _ { \Theta } ^ { 2 } } \sqrt { ( L m + \zeta ) n } ) ^ { L m + \zeta }$ . By summarizing these and ignoring the logarithmic dependency as in the standard $\tilde { O }$ notation, we have the following bounds for Theorems 3 and 4:

$$
\text {(w i t h d e s r e t i z a t i o n)} \quad C _ {J} (w) \sqrt {\frac {4 L ^ {G} + 2 L m + 2 \zeta + 2 \ln (1 / \delta)}{n}} + \sqrt {\frac {\mathcal {L} _ {d} (w) ^ {2 / \rho}}{n}},
$$

and

$$
\text {(w i t h o u t d i s c r e t i z a t i o n)} \quad \tilde {C} _ {J} (w) \sqrt {\frac {4 (4 \sqrt {m}) ^ {m} + 2 \zeta + 2 \ln (1 / \delta)}{n}} + \sqrt {\frac {\tilde {\mathcal {L}} _ {d} (w) ^ {2 / \rho}}{n}} + \varsigma (w) R _ {\mathcal {H}},
$$

where we used the fact that $\ln ( x / y ) = \ln ( x ) + \ln ( 1 / y )$ . Here, we can more easily see that the discretization process has the benefits in the two aspects:

1. The discretization process improves sensitivity against noise and perturbations: i.e., it reduces the sensitivity term $\varsigma ( \bar { w } ) R _ { \mathcal { H } }$ to be zero.   
2. The discretization process reduces underlying dimensionality: i.e., it reduce the term√ √ of $4 ( 4 \sqrt { m } ) ^ { m }$ to the term of $4 L ^ { G } + 2 L m$ . In practice, we typically have $4 ( 4 \sqrt { m } ) ^ { m } \gg$ $4 L ^ { G } + 2 L m$ . This shows that using the discretization process withe codebook of size $L \times m$ , we can successfully reduce the exponential dependency on $m$ to the linear dependency on $m$ . This is a significant improvement.

# C.2 On the comparison of Theorem 4 and algorithmic robustness

If we assume that the function $x \mapsto \ell ( { \tilde { f } } ( x ) , y )$ is Lipschitz for all $y \in \mathcal { V }$ with Lipschitz constant $\varsigma _ { x } ( w )$ similarly to our assumption in Theorem 4, the bound via the algorithmic robustness in the previous paper $\mathrm { { X u } }$ and Mannor, 2012) becomes the following: for any $\delta > 0$ , with probability at least $1 - \delta$ over an iid draw of $n$ examples $( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { n }$ , for any $( w , \theta ) \in \mathcal { W } \times \Theta$ ,

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ \ell (\tilde {f} (x, w, \theta), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} [ \ell (\tilde {f} (x _ {i}, w, \theta), y _ {i}) ] \right| \tag {4} \\ \le \hat {C} _ {J} \sqrt {\frac {4 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} (r ^ {\prime} , \mathcal {X}) \ln 2 + 2 \ln \frac {1}{\delta}}{n}} + 2 \varsigma_ {x} (w) r ^ {\prime}, \\ \end{array}
$$

where $\hat { C } _ { J } \geq \tilde { C } _ { J } ( w )$ for all $w \in \mathcal { W }$ and $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ is a metric space such that $\mathcal { X } \subseteq \mathcal { M } ^ { \prime }$ . See Appendix C.3. for more details on the algorithmic robustness bounds.

Thus, we can see that the dominant term $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } )$ in Theorem 4 is comparable to the dominant term $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { X } )$ in the previous study. Whereas the previous bound measures the robustness in the input space $\mathcal { X }$ , the bound in Theorem 4 measures the robustness in the bottleneck layer space $\mathcal { H }$ . When compared to the input space $\mathcal { X }$ , if the bottleneck layer space $\mathcal { H }$ is smaller or has more structures, then we can have $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) < \mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { X } )$ and Theorem 4 can be advantageous over the previous bound. However, Theorem 4 is not our main result as we have much tighter bounds for the discretization process in Theorem 3 as well as Theorem 1.

# C.3 On algorithmic robustness

In the previous paper, algorithmic robustness is defined to be the measure of how much the loss value can vary with respect to the perturbations of values data points $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ . More precisely, an algorithm $\mathcal { A }$ is said to be $( | \Omega | , \varrho ( \cdot ) )$ -robust if $\mathcal { X } \times \mathcal { V }$ can be partitioned into $| \Omega |$ disjoint sets $\Omega _ { 1 } , \dots , \Omega _ { | \Omega | }$ such that for any dataset $\ b { S } \in ( \ b { \mathcal { X } } \times \ b { \mathcal { Y } } ) ^ { m }$ , all $( x , y ) \in S$ , all $( x ^ { \prime } , y ^ { \prime } ) \in \mathcal { X } \times \mathcal { Y }$ , and all $i \in \{ 1 , \ldots , | \Omega | \}$ , if $( x , y ) , ( x ^ { \prime } , y ^ { \prime } ) \in \Omega _ { i }$ , then

$$
\left| \ell (\tilde {f} (x), y) - \ell (\tilde {f} \left(x ^ {\prime}\right), y ^ {\prime}) \right| \leq \varrho (S).
$$

If algorithm $\mathcal { A }$ is $( \Omega , \varrho ( \cdot ) )$ -robust and the codomain of $\ell$ is upper-bounded by $M$ , then given a dataset $S$ , we have (Xu and Mannor, 2012) that for any $\delta > 0$ , with probability at least $1 - \delta$ ,

$$
\left| \mathbb {E} _ {x, y} [ \ell (\tilde {f} (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} [ \ell (\tilde {f} (x _ {i}), y _ {i}) ] \right| \leq M \sqrt {\frac {2 | \Omega | \ln 2 + 2 \ln \frac {1}{\delta}}{n}} + \varrho (S).
$$

The previous paper (Xu and Mannor, 2012) further shows concrete examples of this bound for a case where the function $( x , y ) \mapsto \ell ( { \tilde { f } } ( x ) , y )$ is Lipschitz with Lipschitz constant $\varsigma _ { x , y } ( w )$ ,

$$
\left| \mathbb {E} _ {x, y} [ \ell (\tilde {f} (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} [ \ell (\tilde {f} (x _ {i}), y _ {i}) ] \right| \leq M \sqrt {\frac {2 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} (r ^ {\prime} , \mathcal {X} \times \mathcal {Y}) \ln 2 + 2 \ln \frac {1}{\delta}}{n}} + 2 \varsigma_ {x, y} (w) r ^ {\prime},
$$

where $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ is a metric space such that $\mathcal { X } \times \mathcal { Y } \subseteq \mathcal { M } ^ { \prime }$ . Note that the Lipschitz assumption on t he function $( x , y ) \mapsto \ell ( { \tilde { f } } ( x ) , y )$ does not typically hold for the 0-1 loss on classification. For classification, we can assume that the function $x \mapsto \ell ( \tilde { f } ( x ) , y )$ is Lipschitz instead, yielding equation (4).

# D Proofs

We use the notation of $q _ { e } ( h ) : = q ( h , L , G )$ in the proofs.

# D.1 Proof of Theorem 1

Proof of Theorem 1. Let ${ \mathcal { T } } _ { k } = \left\{ i \in [ n ] : q _ { e } ( h _ { i } ) = Q _ { k } \right\}$ . By using the following equality,

$$
\mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} \left(q _ {e} (h)\right) \right] = \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} \left(q _ {e} (h)\right) \mid q _ {e} (h) = Q _ {k} \right] \Pr \left(q _ {e} (h) = Q _ {k}\right) = \phi \left(Q _ {k}\right) \Pr \left(q _ {e} (h) = Q _ {k}\right),
$$

we first decompose the difference into two terms as

$$
\begin{array}{l} \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} \left(q _ {e} (h)\right) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(q _ {e} \left(h _ {i}\right)\right) \tag {5} \\ = \phi (Q _ {k}) \left(\Pr (q _ {e} (h) = Q _ {k}) - \frac {| \mathcal {I} _ {k} |}{n}\right) + \left(\phi (Q _ {k}) \frac {| \mathcal {I} _ {k} |}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} (q _ {e} (h _ {i}))\right). \\ \end{array}
$$

The second term in the right-hand side of (5) is further simplified by using

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} (q _ {e} (h _ {i})) = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \phi (q _ {e} (h _ {i})),
$$

and

$$
\phi (Q _ {k}) \frac {| \mathcal {I} _ {k} |}{n} = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \phi (q _ {e} (h _ {i})),
$$

as

$$
\phi (Q _ {k}) \frac {| \mathcal {I} _ {k} |}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} (q _ {e} (h _ {i})) = 0.
$$

Substituting these into equation (5) yields

$$
\begin{array}{l} \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} \left(q _ {e} (h)\right) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(q _ {e} \left(h _ {i}\right)\right) \right| = \left| \phi \left(Q _ {k}\right) \left(\Pr \left(q _ {e} (h) = Q _ {k}\right) - \frac {\left| \mathcal {I} _ {k} \right|}{n}\right) \right| \\ \leq | \phi (Q _ {k}) | \left| \Pr \left(q _ {e} (h) = Q _ {k}\right) - \frac {| \mathcal {I} _ {k} |}{n} \right|. \tag {6} \\ \end{array}
$$

Let the $p _ { k } = \mathrm { P r } ( q _ { e } ( h ) = Q _ { k } )$ and f the $\begin{array} { r } { \hat { p } = \frac { | \mathcal { I } _ { k } | } { n } } \end{array}$ . Consid variable the random varunder the map $X _ { i } = \mathbb { 1 } \{ q _ { e } ( h _ { i } ) = Q _ { k } \}$ withe, we $h _ { i }$ $h _ { i } \mapsto \mathbb { 1 } \{ q _ { e } ( h _ { i } ) = Q _ { k } \}$

have that $X _ { i } \in \{ 0 , 1 \} \subset [ 0 , 1 ]$ . Since $e$ is fixed and $h _ { 1 } , \ldots , h _ { n }$ are assumed to be iid, the Hoeffding’s inequality implies the following: for each fixed $k \in [ L ^ { G } ]$ ,

$$
\operatorname * {P r} (| p _ {k} - \hat {p} _ {k} | \geq t) \leq 2 \exp \left(- 2 n t ^ {2}\right).
$$

By solving $\delta ^ { \prime } = 2 \exp \left( - 2 n t ^ { 2 } \right)$ , this implies that for each fixed $k \in [ L ^ { G } ]$ , for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ ,

$$
| p _ {k} - \hat {p} _ {k} | \leq \sqrt {\frac {\ln (2 / \delta^ {\prime})}{2 n}}.
$$

By taking union bounds over $k \in [ L ^ { G } ]$ with $\begin{array} { r } { \delta ^ { \prime } = \frac { \delta } { L ^ { G } } } \end{array}$ , we have that for any $\delta > 0$ , with probability at least $1 - \delta$ , the following holds for all $k \in [ L ^ { G } ]$ :

$$
\left| p _ {k} - \hat {p} _ {k} \right| \leq \sqrt {\frac {\ln \left(2 L ^ {G} / \delta\right)}{2 n}}.
$$

Substituting this into equation (6) yields that for any $\delta > 0$ , with probability at least $1 - \delta$ , the following holds for all $\boldsymbol { k } \in [ L ^ { G } ]$ :

$$
\left| \mathbb {E} _ {h} [ \phi_ {k} ^ {S} (q _ {e} (h)) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} (q _ {e} (h _ {i})) \right| \leq | \phi (Q _ {k}) | \sqrt {\frac {\ln (2 L ^ {G} / \delta)}{2 n}} = | \phi (Q _ {k}) | \sqrt {\frac {G \ln (L) + \ln (2 / \delta)}{2 n}}.
$$

# D.2 Proof of Theorem 2

Proof of Theorem 2. Let $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ be a matric space such that $\mathcal { H } \subseteq \mathcal { M } ^ { \prime }$ . Fix $r ^ { \prime } > 0$ and $\bar { \mathcal { C } } \in \mathbf { \Xi }$ $\operatorname { a r g m i n } _ { \mathcal { C } } \{ | \mathcal { C } | : \mathcal { C } \subseteq \mathcal { M } ^ { \prime } , \dot { \mathcal { H } } \subseteq \cup _ { c \in \mathcal { C } } \mathcal { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c , \bar { r ^ { \prime } } ] \}$ such that $| \bar { \mathcal { C } } | < \infty$ . Fix an arbitrary ordering and define $c _ { k } \in \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } }$ to be the $k$ -the element in the ordered version of $\bar { \mathcal { C } }$ in that fixed ordering (i.e., $\cup _ { k } \{ c _ { k } \} = \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } )$ . Let $B [ c ] = B _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c , r ^ { \prime } ]$ and $S = \{ \boldsymbol { B } [ c _ { 1 } ] , \boldsymbol { B } [ c _ { 2 } ] , \ldots , \boldsymbol { B } [ c _ { | \bar { \mathcal { C } } | } ] \}$ . Suppose that $\left| \phi _ { k } ^ { S } ( h ) - \phi _ { k } ^ { S } ( h ^ { \prime } ) \right| \leq \xi _ { k } ( r ^ { \prime } , \mathcal { M } ^ { \prime } , d )$ for any $h , h ^ { \prime } \in B [ c _ { k } ]$ and $k \in [ | \bar { \mathcal { C } } | ]$ , which is shown to be satisfied later in this proof. Let ${ \mathcal { T } } _ { k } = \left\{ i \in [ n ] : h _ { i } \in B [ c _ { k } ] \right\}$ for all $k \in [ | \bar { C } | ]$ . By using the following equality,

$$
\mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \right] = \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) | h \in \mathcal {B} [ c _ {k} ] \right] \Pr \left(h \in \mathcal {B} [ c _ {k} ]\right),
$$

we first decompose the difference into two terms as

$$
\begin{array}{l} \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(h _ {i}\right) \right| \tag {7} \\ \leq \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) | h \in \mathcal {B} [ c _ {k} ] \right] \left(\Pr \left(h \in \mathcal {B} [ c _ {k} ]\right) - \frac {\left| \mathcal {I} _ {k} \right|}{n}\right) \right| + \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) | h \in \mathcal {B} [ c _ {k} ] \right] \frac {\left| \mathcal {I} _ {k} \right|}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(h _ {i}\right) \right| \\ \end{array}
$$

The second term in the right-hand side of (7) is further simplified by using

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} (h _ {i}) = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \phi_ {k} ^ {S} (h _ {i}),
$$

and

$$
\mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) | h \in \mathcal {B} [ c _ {k} ] \right] \frac {| \mathcal {I} _ {k} |}{n} = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) | h \in \mathcal {B} [ c _ {k} ] \right],
$$

as

$$
\begin{array}{l} \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \mid h \in \mathcal {B} \left[ c _ {k} \right] \right] \frac {\left| \mathcal {I} _ {k} \right|}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(h _ {i}\right) \right| \\ = \left| \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \left(\mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \mid h \in \mathcal {B} \left[ c _ {k} \right] \right] - \phi_ {k} ^ {S} \left(h _ {i}\right)\right) \right| \\ \end{array}
$$

$$
\leq \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k}} \sup  _ {h \in \mathcal {B} [ c _ {k} ]} \left| \phi_ {k} ^ {S} (h) - \phi_ {k} ^ {S} (h _ {i}) \right| \leq \frac {| \mathcal {I} _ {k} |}{n} \xi_ {k} \left(r ^ {\prime}, \mathcal {M} ^ {\prime}, d\right).
$$

Substituting these into equation (7) yields

$$
\begin{array}{l} \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(h _ {i}\right) \right| \\ \leq \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) | h \in \mathcal {B} [ c _ {k} ] \right] \left(\Pr (h \in \mathcal {B} [ c _ {k} ]) - \frac {| \mathcal {I} _ {k} |}{n}\right) \right| + \frac {| \mathcal {I} _ {k} |}{n} \xi_ {k} \left(r ^ {\prime}, \mathcal {M} ^ {\prime}, d\right) \\ \leq \left| \mathbb {E} _ {h} [ \phi (h) | h \in \mathcal {B} [ c _ {k} ] ] \right| \left| \left(\Pr (h \in \mathcal {B} [ c _ {k} ]) - \frac {\left| \mathcal {I} _ {k} \right|}{n}\right) \right| + \frac {\left| \mathcal {I} _ {k} \right|}{n} \xi_ {k} \left(r ^ {\prime}, \mathcal {M} ^ {\prime}, d\right), \tag {8} \\ \end{array}
$$

Let pus $p _ { k } = \operatorname* { P r } ( h \in { \cal B } [ c _ { k } ] )$ and  the $\begin{array} { r } { \hat { p } = \frac { | \mathcal { I } _ { k } | } { n } } \end{array}$ . Consiariable r the random vunder the map $X _ { i } = \mathbb { 1 } \{ h \in B [ c _ { k } ] \}$ with the we have $h _ { i }$ $h _ { i } \mapsto \mathbb { 1 } \{ h \in \bar { \mathcal { B } } [ c _ { k } ] \}$ that $X _ { i } \in \{ 0 , 1 \} \subset [ 0 , 1 ]$ . Since $B [ c _ { k } ]$ is fixed and $h _ { 1 } , \ldots , h _ { n }$ are assumed to be iid, the Hoeffding’s inequality implies the following: for each fixed $k \in [ | \bar { C } | ]$ ,

$$
\Pr \left(\left| p _ {k} - \hat {p} _ {k} \right| \geq t\right) \leq 2 \exp \left(- 2 n t ^ {2}\right).
$$

By solving $\delta ^ { \prime } = 2 \exp \left( - 2 n t ^ { 2 } \right)$ , this implies that for each fixed $k \in [ | \bar { C } | ]$ , for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ ,

$$
\left| p _ {k} - \hat {p} _ {k} \right| \leq \sqrt {\frac {\ln \left(2 / \delta^ {\prime}\right)}{2 n}}.
$$

By taking union bounds over $k \in [ | \bar { C } | ]$ with $\begin{array} { r } { \delta ^ { \prime } = \frac { \delta } { | \bar { c } | } } \end{array}$ δ|C| ¯ , we have that for any δ > 0, with probability at $\delta > 0$ least $1 - \delta$ , the following holds for all $k \in [ | \bar { C } | ]$ :

$$
\left| p _ {k} - \hat {p} _ {k} \right| \leq \sqrt {\frac {\ln (2 | \bar {\mathcal {C}} | / \delta)}{2 n}} = \sqrt {\frac {\ln (| \bar {\mathcal {C}} |) + \ln (2 / \delta)}{2 n}}.
$$

Substituting this into equation (8) yields that for any $\delta > 0$ , with probability at least $1 - \delta$ , the following holds for all $\dot { k ^ { \mathrm { ~ } } } \in [ | \bar { \mathcal { C } } | ]$ :

$$
\begin{array}{l} \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(h _ {i}\right) \right| \\ \leq | \mathbb {E} _ {h} [ \phi (h) | h \in \mathcal {B} [ c _ {k} ] ] | \sqrt {\frac {\ln (\mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} (r ^ {\prime} , \mathcal {H})) + \ln (2 / \delta)}{2 n}} \\ + \xi_ {k} (r ^ {\prime}, \mathcal {M} ^ {\prime}, d) \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {1} \{h _ {i} \in \mathcal {B} [ c _ {k} ] \}\right), \\ \end{array}
$$

where we used Euclidean spac $\begin{array} { r } { \left| \mathcal { T } _ { k } \right| = \sum _ { i = 1 } ^ { n } \mathbb { 1 } \left\{ h _ { i } \in \mathcal { B } [ c _ { k } ] \right\} } \end{array}$ . Let us etric and se the metrsuch that $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ toll $\mathbb { R } ^ { m }$ $\mathcal { H } \subset \mathbb { R } ^ { m }$ $\| v \| _ { 2 } \le R _ { \mathcal { H } }$ $v \in \mathcal { H }$ Then, we have $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) \leq ( 2 R _ { \mathcal { H } } \sqrt { m } / r ^ { \prime } ) ^ { m }$ and we can set $\xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d ) = 2 \varsigma _ { k } r ^ { \prime }$ . This is because that the function $h \mapsto \phi _ { k } ^ { S } ( h )$ is Lipschitz continuous as $| \phi _ { k } ^ { S } ( h ) - \phi _ { k } ^ { S } ( h ^ { \prime } ) | \leq \varsigma _ { k } d ( h , h ^ { \prime } )$ , and because $d ( h , h ^ { \prime } ) \leq 2 r ^ { \prime }$ for any $\ddot { h } , \dot { h ^ { \prime } } \in \dot { \mathcal { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } } [ c _ { k } , r ^ { \prime } ]$ . Thus, by setting $r ^ { \prime } = \dot { R } _ { \mathcal { H } } / ( 2 \sqrt { n } )$ , we can replace $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } )$ by $( 4 { \sqrt { n m } } ) ^ { m }$ and set $\xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d ) = \varsigma _ { k } R _ { \mathcal { H } } / \sqrt { n }$ .

This yields

$$
\begin{array}{l} \left| \mathbb {E} _ {h} \left[ \phi_ {k} ^ {S} (h) \right] - \frac {1}{n} \sum_ {i = 1} ^ {n} \phi_ {k} ^ {S} \left(h _ {i}\right) \right| \\ \leq | \mathbb {E} _ {h} [ \phi (h) | h \in \mathcal {B} [ c _ {k} ] ] | \sqrt {\frac {m \ln (4 \sqrt {n m}) + \ln (2 / \delta)}{2 n}} + \frac {\varsigma_ {k} R _ {\mathcal {H}}}{\sqrt {n}} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {1} \{h _ {i} \in \mathcal {B} [ c _ {k} ] \}\right). \\ \end{array}
$$

![](images/9a2ce617125c684bd379586c248a60fc94f5ce313e87e66c439c6110418ff5ea.jpg)

# D.3 Proof of Theorem 3

In the proof of Theorem 3, we write $f ( x ) : = f ( x , w , e , \theta )$ when the dependency on $( w , e , \theta )$ is clear from the context.

Lemma 1. Fix $\theta \in \Theta$ and $e \in E$ . Then for any $\delta > 0$ , with probability at least $1 - \delta$ over an iid draw of n examples $( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { n }$ , the following holds for any $w \in \mathcal { W }$ :

$$
\left| \mathbb {E} _ {x, y} [ J (f (x, w, e, \theta), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x, w, e, \theta), y _ {i}) \right| \leq C _ {J} (w) \sqrt {\frac {4 L ^ {G} \ln 2 + 2 \ln (1 / \delta)}{n}}.
$$

Proof of Lemma 1. Let ${ \mathcal { T } } _ { k , y } = \{ i \in [ n ] : ( q _ { e } \circ h _ { \theta } ) ( x _ { i } ) = Q _ { k } , y _ { i } = y \}$ . Using $\mathbb { E } _ { x , y } [ J ( f ( x ) , y ) ] =$ $\begin{array} { r } { \sum _ { k = 1 } ^ { L ^ { \infty } } \sum _ { y ^ { \prime } \in \mathcal { Y } } \mathbb { E } _ { x , y } [ J ( f ( x ) , y ) | ( q _ { e } \circ h _ { \theta } ) ( x ) = Q _ { k } , y = y ^ { \prime } ] \operatorname* { P r } ( ( q _ { e } \circ h _ { \theta } ) ( x ) = Q _ { k } \wedge y = y ^ { \prime } ) } \end{array}$ 0 ), we first decompose the difference into two terms as

$$
\begin{array}{l} \mathbb {E} _ {x, y} [ J (f (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J \left(f \left(x _ {i}\right), y _ {i}\right) \tag {9} \\ = \sum_ {k = 1} ^ {L ^ {G}} \sum_ {y ^ {\prime} \in \mathcal {Y}} \mathbb {E} _ {x, y} [ J (f (x), y) | (q _ {e} \circ h _ {\theta}) (x) = Q _ {k}, y = y ^ {\prime} ] \left(\Pr ((q _ {e} \circ h _ {\theta}) (x) = Q _ {k} \wedge y = y ^ {\prime}) - \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n}\right) \\ + \left(\sum_ {k = 1} ^ {L ^ {G}} \sum_ {y ^ {\prime} \in \mathcal {Y}} \mathbb {E} _ {x, y} [ J (f (x), y) | (q _ {e} \circ h _ {\theta}) (x) = Q _ {k}, y = y ^ {\prime} ] \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x _ {i}), y _ {i})\right). \\ \end{array}
$$

The second term in the right-hand side of (9) is further simplified by using

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x), y) = \frac {1}{n} \sum_ {k = 1} ^ {L ^ {G}} \sum_ {y ^ {\prime} \in \mathcal {Y}} \sum_ {i \in \mathcal {I} _ {k, y ^ {\prime}}} J (f (x _ {i}), y _ {i}),
$$

and

$$
\begin{array}{l} \mathbb {E} _ {x, y} [ J (f (x), y) | (q _ {e} \circ h _ {\theta}) (x) \\ = Q _ {k}, y = y ^ {\prime} ] \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n} = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k, y ^ {\prime}}} \mathbb {E} _ {x, y} [ J (f (x), y) | (q _ {e} \circ h _ {\theta}) (x) = Q _ {k}, y = y ^ {\prime} ], \\ \end{array}
$$

as

$$
\begin{array}{l} \sum_ {k = 1} ^ {L ^ {G}} \sum_ {y ^ {\prime} \in \mathcal {Y}} \mathbb {E} _ {x, y} [ J (f (x), y) | (q _ {e} \circ h _ {\theta}) (x) = Q _ {k}, y = y ^ {\prime} ] \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x _ {i}), y _ {i}) \\ = \frac {1}{n} \sum_ {k = 1} ^ {L ^ {G}} \sum_ {y ^ {\prime} \in \mathcal {Y}} \sum_ {i \in \mathcal {I} _ {k, y ^ {\prime}}} (\mathbb {E} _ {x, y} [ J (f (x), y) | (q _ {e} \circ h _ {\theta}) (x) = Q _ {k}, y = y ^ {\prime} ] - J (f (x _ {i}), y _ {i})) \\ = \frac {1}{n} \sum_ {k = 1} ^ {L ^ {G}} \sum_ {y ^ {\prime} \in \mathcal {Y}} \sum_ {i \in \mathcal {I} _ {k, y ^ {\prime}}} \left(J \left(\varphi_ {w} \left(Q _ {k}\right), y ^ {\prime}\right) - \left(J \left(\varphi_ {w} \left(Q _ {k}\right), y ^ {\prime}\right)\right) = 0 \right. \\ \end{array}
$$

Substituting these into equation (9) yields

$$
\begin{array}{l} \mathbb {E} _ {x, y} [ J (f (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x _ {i}), y _ {i}) \\ = \sum_ {k = 1} ^ {L ^ {G}} \sum_ {y ^ {\prime} \in \mathcal {Y}} J \left(\varphi_ {w} \left(Q _ {k}\right), y ^ {\prime}\right) \left(\Pr \left(\left(q _ {e} \circ h _ {\theta}\right) (x) = Q _ {k} \wedge y = y ^ {\prime}\right) - \frac {\left| \mathcal {I} _ {k , y ^ {\prime}} \right|}{n}\right) \\ = \sum_ {k = 1} ^ {2 L ^ {G}} J (v _ {k}) \left(\Pr \left(\left(\left(q _ {e} \circ h _ {\theta}\right) (x), y\right) = v _ {k}\right) - \frac {\left| \mathcal {I} _ {k} \right|}{n}\right), \\ \end{array}
$$

where the last line uses the fact that $\mathcal { V } = \{ y ^ { ( 1 ) } , y ^ { ( 2 ) } \}$ for some $( y ^ { ( 1 ) } , y ^ { ( 2 ) } )$ , along with the additional notation ${ \mathcal { T } } _ { k } = \{ i \in [ n ] : ( ( q _ { e } \circ h _ { \theta } ) ( x _ { i } ) , y _ { i } ) = v _ { k } \}$ . Here, $v _ { k }$ is defined as $v _ { k } = ( \varphi _ { w } ( Q _ { k } ) , y ^ { ( 1 ) } )$ for all $k \in [ L ^ { G } ]$ and $v _ { k } = ( \varphi _ { w } ( e _ { k - L ^ { G } } ) , y ^ { ( 2 ) } )$ for all $k \in \{ L ^ { G } + 1 , \ldots , 2 L ^ { G } \}$ .

By using the bound of $| J ( \varphi _ { w } ( \eta ) , y ) | \leq C _ { J } ( w )$ ,

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ J (f (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x _ {i}), y _ {i}) \right| \\ = \left| \sum_ {k = 1} ^ {2 L ^ {G}} J (v _ {k}) \left(\Pr \left(\left(\left(q _ {e} \circ h _ {\theta}\right) (x), y\right) = v _ {k}\right) - \frac {\left| \mathcal {I} _ {k} \right|}{n}\right) \right| \\ \leq C _ {J} (w) \sum_ {k = 1} ^ {2 L ^ {G}} \left| \Pr \left(\left(\left(q _ {e} \circ h _ {\theta}\right) (x), y\right) = v _ {k}\right) - \frac {\left| \mathcal {I} _ {k} \right|}{n} \right|. \\ \end{array}
$$

Since $\begin{array} { r } { | \mathcal { T } _ { k } | = \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ ( ( q _ { e } \circ h _ { \theta } ) ( x _ { i } ) , y _ { i } ) = v _ { k } \} } \end{array}$ and $( \theta , e )$ is fixed, the vector $( | \mathcal { T } _ { 1 } | , \dots , | \mathcal { T } _ { 2 L ^ { G } } | )$ follows a multinomial distribution with parameters $n$ and $p = ( p _ { 1 } , . . . , p _ { 2 L ^ { G } } )$ , where $p _ { k } = \operatorname* { P r } ( ( q _ { e } \circ$ $h _ { \theta } ) ( x ) , y ) = v _ { k } )$ for $k = 1 , \dots , 2 L ^ { G }$ . Thus, by using the Bretagnolle-Huber-Carol inequality (van der Vaart and Wellner, 1996, A6.6 Proposition), we have that with probability at least $1 - \delta$ ,

$$
\left| \mathbb {E} _ {x, y} [ J (f (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (f (x _ {i}), y _ {i}) \right| \leq C _ {J} (w) \sqrt {\frac {4 L ^ {G} \ln 2 + 2 \ln (1 / \delta)}{n}}.
$$

![](images/771201a03110de59770320ea23d92d831b3808be55b326d7a8efb093d01357f3.jpg)

Proof of Theorem 3. Let $\hat { \mathcal { C } } _ { r , d } \in \operatorname { a r g m i n } _ { \mathcal { C } } \left\{ | \mathcal { C } | : \mathcal { C } \subseteq \mathcal { M } , E \times \Theta \subseteq \cup _ { c \in \mathcal { C } } \mathcal { B } _ { ( \mathcal { M } , d ) } [ c , r ] \right\}$ . Note that if $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , E \times \Theta ) = \infty$ , the bound in the statement of the theorem vacuously holds. Thus, we focus on the case of $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , E \times \Theta ) = | \hat { \mathcal { C } } _ { r , d } | < \infty$ . For any $( w , e , \theta ) \in \mathcal { W } \times E \times \Theta$ , the following holds: for any $( \hat { e } , \hat { \theta } ) \in \hat { \mathcal { C } } _ { r , d }$ ,

$$
\begin{array}{l} | \psi_ {w} (e, \theta) | = \left| \psi_ {w} (\hat {e}, \hat {\theta}) + \psi_ {w} (e, \theta) - \psi_ {w} (\hat {e}, \hat {\theta}) \right| \\ \leq \left| \psi_ {w} (\hat {e}, \hat {\theta}) \right| + \left| \psi_ {w} (e, \theta) - \psi_ {w} (\hat {e}, \hat {\theta}) \right|. \tag {10} \\ \end{array}
$$

For the first term in the right-hand side of (10), by using Lemma 1 with $\delta = \delta ^ { \prime } / N _ { ( { \mathcal { M } } , d ) } ( r , E \times \Theta )$ and taking union bounds, we have that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ , the following holds for all $( \hat { e } , \hat { \theta } ) \in \hat { C } _ { r , d }$ ,

$$
\left| \psi_ {w} (\hat {e}, \hat {\theta}) \right| \leq C _ {J} (w) \sqrt {\frac {4 L ^ {G} \ln 2 + 2 \ln \left(\mathcal {N} _ {(\mathcal {M} , d)} (r , E \times \Theta) / \delta^ {\prime}\right)}{n}}. \tag {11}
$$

By combining equations (10) and (11), we have that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ , the following holds for any $( w , e , \theta ) \in \mathcal { W } \times E \times \Theta$ and any $( \hat { e } , \hat { \theta } ) \in \hat { C } _ { r , d }$ :

$$
| \psi_ {w} (e, \theta) | \leq C _ {J} (w) \sqrt {\frac {4 L ^ {G} \ln 2 + 2 \ln (\mathcal {N} _ {(\mathcal {M} , d)} (r , E \times \Theta) / \delta^ {\prime})}{n}} + \left| \psi_ {w} (e, \theta) - \psi_ {w} (\hat {e}, \hat {\theta}) \right|.
$$

This implies that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ , the following holds for any $( w , e , \theta ) ^ { - } \in \mathcal { W } \times E \times \Theta$ :

$$
| \psi_ {w} (e, \theta) | \leq C _ {J} (w) \sqrt {\frac {4 L ^ {G} \ln 2 + 2 \ln \left(\mathcal {N} _ {(\mathcal {M} , d)} (r , E \times \Theta) / \delta^ {\prime}\right)}{n}} + \min  _ {(\hat {e}, \hat {\theta}) \in \hat {\mathcal {C}} _ {r, d}} \left| \psi_ {w} (e, \theta) - \psi_ {w} (\hat {e}, \hat {\theta}) \right|. \tag {12}
$$

For the second term in the right-hand side of (12), we have that for any $( w , e , \theta ) \in \mathcal { W } \times E \times \Theta$ ,

$$
\min  _ {(\hat {e}, \hat {\theta}) \in \hat {\mathcal {C}} _ {r, d}} \left| \psi_ {w} (e, \theta) - \psi_ {w} (\hat {e}, \hat {\theta}) \right| \leq \mathcal {L} _ {d} (w) \min  _ {(\hat {e}, \hat {\theta}) \in \hat {\mathcal {C}} _ {r, d}} d ((e, \theta), (\hat {e}, \hat {\theta})) \leq \mathcal {L} _ {d} (w) r.
$$

Thus, by using $r = \mathcal { L } _ { d } ( w ) ^ { 1 / \rho - 1 } \sqrt { \frac { 1 } { n } }$ , we have that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ the following holds for any $( w , e , \dot { \theta } ) \in \mathcal { W } \times E \times \Theta$ :

$$
| \psi_ {w} (e, \theta) | \leq C _ {J} (w) \sqrt {\frac {4 L ^ {G} \ln 2 + 2 \ln (\mathcal {N} _ {(\mathcal {M} , d)} (r , E \times \Theta) / \delta^ {\prime})}{n}} + \sqrt {\frac {\mathcal {L} _ {d} (w) ^ {2 / \rho}}{n}}. \tag {13}
$$

Since this statement holds for any $\delta ^ { \prime } > 0$ , this implies the statement of this theorem.

![](images/da56c27503c8c2a880a6ac56e9006dd74ea7e92c13848a5e1619a55a3719ca91.jpg)

# D.4 Proof of Theorem 4

In the proof of Theorem 3, we write $\tilde { f } ( x ) : = \tilde { f } ( x , w , \theta )$ when the dependency on $( w , \theta )$ is clear from the context.

Lemma 2. Fix $\theta \in \Theta$ . Let $( \mathcal { M } ^ { \prime } , d ^ { \prime } )$ be a matric space such that $\mathcal { H } \subseteq \mathcal { M } ^ { \prime }$ . Fix $r ^ { \prime } > 0$ and $\bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } \in \operatorname { a r g m i n } _ { \mathcal { C } } \{ | \mathcal { C } | : \mathcal { C } \subseteq \mathcal { M } ^ { \prime } , \mathcal { H } \subseteq \cup _ { c \in \mathcal { C } } \mathcal { B } _ { ( { \mathcal { M } ^ { \prime } } , d ^ { \prime } ) } [ c , r ^ { \prime } ] \}$ . Assume that for any $c \in \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } }$ , we have $| ( J ( \varphi _ { w } ( h ) , y ) - ( J ( \varphi _ { w } ( h ^ { \prime } ) , y ) | \leq \xi ( w , r ^ { \prime } , \mathcal { M } ^ { \prime } , d ) \mathcal { I }$ for any $h , h ^ { \prime } \in \mathcal { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c , r ^ { \prime } ]$ and $y \in \mathcal { V }$ . Then for any $\delta > 0$ , with probability at least $1 - \delta$ over an iid draw of n examples $( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { n }$ , the following holds for any $w \in \mathcal { W }$ :

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ J (\tilde {f} (x, w, \theta), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x, w, \theta), y _ {i}) \right| \\ \leq \tilde {C} _ {J} (w) \sqrt {\frac {4 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} (r ^ {\prime} , \mathcal {H}) \ln 2 + 2 \ln (1 / \delta)}{n}} + \xi (w, r ^ {\prime}, \mathcal {M} ^ {\prime}, d). \\ \end{array}
$$

Proof of Lemma 2. Note that if $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) = \infty$ , the bound in the statement of the theorem vacuously holds. Thus, we focus on the case of $\mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } ) = | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | < \infty$ . Fix an arbitrary ordering and define $c _ { k } \in \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } }$ to be the $k$ -the element in the ordered version of $\bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } }$ in that fixed ordering (i.e., $\cup _ { k } \{ c _ { k } \} = \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } )$ ).

$\begin{array} { r } { \mathcal { T } _ { k , y } = \{ i \in [ n ] : h _ { \theta } ( x _ { i } ) \in \mathcal { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c _ { k } , r ^ { \prime } ] , y _ { i } = y \} } \end{array}$ $k \times y \in [ | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | ] \times \mathcal { Y }$ sing(x) ∈ $\begin{array} { r } { \mathbb { E } _ { x , y } [ J ( \tilde { f } ( x ) , y ) ] = \sum _ { k = 1 } ^ { | \mathcal { C } _ { \tau ^ { \prime } , d ^ { \prime } } | } \sum _ { y ^ { \prime } \in \mathcal { V } } \mathbb { E } _ { x , y } [ J ( \tilde { f } ( x ) , y ) | h _ { \theta } ( x ) \in \mathcal { B } _ { ( M ^ { \prime } , d ^ { \prime } ) } [ { c } _ { k } , r ^ { \prime } ] , y = y ^ { \prime } ] \operatorname* { P r } ( h _ { \theta } ( x ) , r ^ { \prime } ) } \end{array}$ $B _ { ( { \mathcal { M } } ^ { \prime } , d ^ { \prime } ) } [ c _ { k } , r ^ { \prime } ] \wedge y = y ^ { \prime } )$

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x _ {i}), y _ {i}) \right| \tag {14} \\ = \left| \sum_ {k = 1} ^ {| \tilde {\mathcal {C}} _ {r ^ {\prime}, d ^ {\prime}} |} \sum_ {y ^ {\prime} \in \mathcal {Y}} \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) | h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ], y = y ^ {\prime} ] \left(\operatorname * {P r} (h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ] \wedge y = y ^ {\prime}) - \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n}\right) \right| \\ + \left| \sum_ {k = 1} ^ {| \bar {\mathcal {C}} _ {r ^ {\prime}, d ^ {\prime}} |} \sum_ {y ^ {\prime} \in \mathcal {Y}} \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) | h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ], y = y ^ {\prime} ] \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x _ {i}), y _ {i}). \right| \\ \end{array}
$$

The second term in the right-hand side of (14) is further simplified by using

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x), y) = \frac {1}{n} \sum_ {k = 1} ^ {| \bar {\mathcal {C}} _ {r ^ {\prime}, d ^ {\prime}} |} \sum_ {y ^ {\prime} \in \mathcal {Y}} \sum_ {i \in \mathcal {I} _ {k, y ^ {\prime}}} J (\tilde {f} (x _ {i}), y _ {i}),
$$

and

$$
\begin{array}{l} \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) | h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ], y = y ^ {\prime} ] \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n} \\ = \frac {1}{n} \sum_ {i \in \mathcal {I} _ {k, y ^ {\prime}}} \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) | h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ], y = y ^ {\prime} ], \\ \end{array}
$$

as

$$
\left| \sum_ {k = 1} ^ {| \bar {\mathcal {C}} _ {r ^ {\prime}, d ^ {\prime}} |} \sum_ {y ^ {\prime} \in \mathcal {Y}} \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) | h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ], y = y ^ {\prime} ] \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n} - \frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x _ {i}), y _ {i}) \right|
$$

$$
\begin{array}{l} = \left| \frac {1}{n} \sum_ {k = 1} ^ {| \bar {\mathcal {C}} _ {r ^ {\prime}, d ^ {\prime}} |} \sum_ {y ^ {\prime} \in \mathcal {Y}} \sum_ {i \in \mathcal {T} _ {k, y ^ {\prime}}} \left(\mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) | h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ], y = y ^ {\prime} ] - J (\tilde {f} (x _ {i}), y _ {i})\right) \right| \\ \leq \frac {1}{n} \sum_ {k = 1} ^ {| \bar {c} _ {r ^ {\prime}, d ^ {\prime}} |} \sum_ {y ^ {\prime} \in \mathcal {Y}} \sum_ {i \in \mathcal {I} _ {k, y ^ {\prime}}} \sup  _ {h \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ]} | J (\varphi_ {w} (h, y ^ {\prime}) - J (\varphi_ {w} (h _ {\theta} (x _ {i})), y ^ {\prime}) | \leq \xi (w). \\ \end{array}
$$

Substituting these into equation (14) yields

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x _ {i}), y _ {i}) \right| \\ \leq \left| \sum_ {k = 1} ^ {| \mathcal {C} _ {r ^ {\prime}, d ^ {\prime}} |} \sum_ {y ^ {\prime} \in \mathcal {Y}} \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y ^ {\prime}) | h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ] ] \left(\Pr (h _ {\theta} (x) \in \mathcal {B} _ {(\mathcal {M} ^ {\prime}, d ^ {\prime})} [ c _ {k}, r ^ {\prime} ] \wedge y = y ^ {\prime}) - \frac {| \mathcal {I} _ {k , y ^ {\prime}} |}{n}\right) \right| + \xi (w) \\ \leq \tilde {C} _ {J} (w) \sum_ {k = 1} ^ {2 | \bar {\mathcal {C}} _ {r ^ {\prime}, d ^ {\prime}} |} \left| \left(\Pr \left(\left(h _ {\theta} (x), y\right) \in v _ {k}\right) - \frac {| \mathcal {I} _ {k} |}{n}\right) \right| + \xi (w), \\ \end{array}
$$

where the last line uses the fact that $\mathcal { V } = \{ y ^ { ( 1 ) } , y ^ { ( 2 ) } \}$ for some $( y ^ { ( 1 ) } , y ^ { ( 2 ) } )$ , along with the additional notation $\mathcal { T } _ { k } = \{ i \in [ n ] : ( h _ { \theta } ( x _ { i } ) , y _ { i } ) \in v _ { k } \}$ . Here, $v _ { k }$ is defined as $v _ { k } = \dot { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c _ { k } , r ^ { \prime } ] \times \{ y ^ { ( 1 ) } \}$ for all $k \in [ | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | ]$ and $v _ { k } = \mathcal { B } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } [ c _ { k - | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | } , r ^ { \prime } ] \times \{ y ^ { ( 2 ) } \}$ for all $k \in \{ | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | + 1 , \dots , 2 | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | \}$ .

Since $\textstyle | { \mathcal { T } } _ { k } | = \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ ( h _ { \theta } ( x ) , y ) \in v _ { k } \}$ and $\theta$ is fixed, the vector $\big ( | \mathcal { T } _ { 1 } | , \dots , | \mathcal { T } _ { 2 | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | } | \big )$ follows a multinomial distribution with parameters $n$ and $p = ( p _ { 1 } , . . . , p _ { 2 | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | } )$ , where $p _ { k } = \mathrm { P r } ( ( h _ { \theta } ( x ) , y ) \in$ $v _ { k } )$ for $k = 1 , \dots , 2 | \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } |$ . Thus, by noticing $| \bar { \mathcal { C } } _ { r ^ { \prime } , d ^ { \prime } } | = \mathcal { N } _ { ( \mathcal { M } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \mathcal { H } )$ and by using the Bretagnolle-Huber-Carol inequality (van der Vaart and Wellner, 1996, A6.6 Proposition), we have that with probability at least $1 - \delta$ ,

$$
\begin{array}{l} \left| \mathbb {E} _ {x, y} [ J (\tilde {f} (x), y) ] - \frac {1}{n} \sum_ {i = 1} ^ {n} J (\tilde {f} (x _ {i}), y _ {i}) \right| \\ \leq \tilde {C} _ {J} (w) \sqrt {\frac {4 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} (r ^ {\prime} , \mathcal {H}) \ln 2 + 2 \ln (1 / \delta)}{n}} + \xi (w). \\ \end{array}
$$

![](images/b170dc3724cf5ab5dc290c8a79243c1387acac2ad3b6056ad0b3cfb61f9becdf.jpg)

Proof of Theorem 4. Let $\hat { \mathcal { C } } _ { r , d } ~ \in ~ \operatorname { a r g m i n } _ { \mathcal { C } } \left\{ | \mathcal { C } | : \mathcal { C } \subseteq \mathcal { M } , \Theta \subseteq \cup _ { c \in \mathcal { C } } \mathcal { B } _ { ( \mathcal { M } , d ) } [ c , r ] \right\}$ . Note that if $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , \Theta ) = \infty$ , the bound in the statement of the theorem vacuously holds. Thus, we focus on the case of $\mathcal { N } _ { ( \mathcal { M } , d ) } ( r , \Theta ) = | \hat { \mathcal { C } } _ { r , d } | < \infty$ . For any $( w , \theta ) \in \mathcal { W } \times \Theta$ , the following holds: for any $\hat { \theta } \in \hat { \mathcal { C } } _ { r , d }$ ,

$$
\begin{array}{l} \left| \tilde {\psi} _ {w} (\theta) \right| = \left| \tilde {\psi} _ {w} (\hat {\theta}) + \tilde {\psi} _ {w} (\theta) - \tilde {\psi} _ {w} (\hat {\theta}) \right| \\ \leq \left| \tilde {\psi} _ {w} (\hat {\theta}) \right| + \left| \tilde {\psi} _ {w} (\theta) - \tilde {\psi} _ {w} (\hat {\theta}) \right|. \tag {15} \\ \end{array}
$$

For the first term in the right-hand side of (15), by using Lemma 2 with $\delta = \delta ^ { \prime } / N _ { ( { \mathcal { M } } ^ { \prime } , d ^ { \prime } ) } ( r ^ { \prime } , \Theta )$ and taking union bounds, we have that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ , the following holds for all $\hat { \theta } \in \hat { \mathcal { C } } _ { r , d }$ ,

$$
\left| \tilde {\psi} _ {w} (\hat {\theta}) \right| \leq \tilde {C} _ {J} (w) \sqrt {\frac {4 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} \left(r ^ {\prime} , \mathcal {H}\right) \ln 2 + 2 \ln \left(\mathcal {N} _ {(\mathcal {M} , d)} (r , \Theta) / \delta^ {\prime}\right)}{n}} + \xi (w, r ^ {\prime}, \mathcal {M} ^ {\prime}, d). \tag {16}
$$

By combining equations (15) and (16), we have that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ , the following holds for any $( w , \theta ) \in \mathcal { W } \times \Theta$ and any $\hat { \theta } \in \hat { \mathcal { C } } _ { r , d }$ :

$$
\left| \tilde {\psi} _ {w} (\theta) \right|
$$

$$
\leq \tilde {C} _ {J} (w) \sqrt {\frac {4 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} (r ^ {\prime} , \mathcal {H}) \ln 2 + 2 \ln (\mathcal {N} _ {(\mathcal {M} , d)} (r , \Theta) / \delta^ {\prime})}{n}} + \left| \tilde {\psi} _ {w} (\theta) - \tilde {\psi} _ {w} (\hat {\theta}) \right| + \xi (w, r ^ {\prime}, \mathcal {M} ^ {\prime}, d).
$$

This implies that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ , the following holds for any $( w , \theta ) \in \mathcal { W } \times \Theta$ :

$$
\begin{array}{l} \left| \tilde {\psi} _ {w} (\theta) \right| \leq C _ {J} (w) \sqrt {\frac {4 \mathcal {N} _ {(\mathcal {M} ^ {\prime} , d ^ {\prime})} \left(r ^ {\prime} , \mathcal {H}\right) \ln 2 + 2 \ln \left(\mathcal {N} _ {(\mathcal {M} , d)} (r , \Theta) / \delta^ {\prime}\right)}{n}} \tag {17} \\ + \min _ {\hat {\theta} \in \hat {\mathcal {C}} _ {r, d}} \left| \tilde {\psi} _ {w} (\theta) - \tilde {\psi} _ {w} (\hat {\theta}) \right| + \xi (w, r ^ {\prime}, \mathcal {M} ^ {\prime}, d). \\ \end{array}
$$

For the second term in the right-hand side of (17), we have that for any $( w , \theta ) \in \mathcal { W } \times \Theta$ ,

$$
\min  _ {\hat {\theta} \in \hat {\mathcal {C}} _ {r, d}} \left| \tilde {\psi} _ {w} (\theta) - \tilde {\psi} _ {w} (\hat {\theta}) \right| \leq \tilde {\mathcal {L}} _ {d} (w) \min  _ {\hat {\theta} \in \hat {\mathcal {C}} _ {r, d}} d (\theta , \hat {\theta}) \leq \mathcal {L} _ {d} (w) r.
$$

Thus, by using $r = \tilde { \mathcal { L } } _ { d } ( w ) ^ { 1 / \rho - 1 } \sqrt { \frac { 1 } { n } }$ , we have that for any $\delta ^ { \prime } > 0$ , with probability at least $1 - \delta ^ { \prime }$ , the following holds for any $( w , \theta ) \in \mathcal { W } \times \Theta$ :

$$
\begin{array}{l} \left| \tilde {\psi} _ {w} (\theta) \right| \leq \tilde {C} _ {J} (w) \sqrt {\frac {4 \mathcal {N} _ {\left(\mathcal {M} ^ {\prime} , d ^ {\prime}\right)} \left(r ^ {\prime} , \mathcal {H}\right) \ln 2 + 2 \ln \left(\mathcal {N} _ {\left(\mathcal {M} , d\right)} \left(r , \Theta\right) / \delta^ {\prime}\right)}{n}} \tag {18} \\ + \sqrt {\frac {\tilde {\mathcal {L}} _ {d} (w) ^ {2 / \rho}}{n}} + \xi (w, r ^ {\prime}, \mathcal {M} ^ {\prime}, d). \\ \end{array}
$$

Since this statement holds for any $\delta ^ { \prime } > 0$ , this implies the statement of this theorem.

![](images/d4f2f6a57bc12ecdfe78c38588f00df3bd943ee7301d4dbef47f56087fedfae3.jpg)

# E Method Details

Algorithm 1: Discretization of inter-module communication in RIM   
N is sample size,T is total time step, M is number of modules in the RIM model   
initialization;   
for i in 1..M do initialize $z_i^0$ .   
end   
Training;   
for n in 1..N do   
for t in 1..T do INPUTATTENTION $\equiv$ SOFTATTENTION $(z_1^t,z_2^t,\dots,z_M^t,x^t)$ if i in top K of INPUTATTENTION then $\hat{z}_i^{t + 1} = \mathrm{RNN}(z_i^t,x^t);$ else $\hat{z}_{i'}^{t + 1} = z_{i'}^t$ end for i in 1..M do Discretization; $h_i^{t + 1} =$ SOFTATTENTION( $\hat{z}_1^{t + 1},\hat{z}_2^{t + 1},\dots.\hat{z}_M^{t + 1})$ $z_{i}^{t + 1} = \hat{z}_{i}^{t + 1} + q(h_{i}^{t + 1},L,G)$ end Calculate task loss, codebook loss and commitment loss according to equation 1 Update model parameter $\Theta$ together with discrete latent vectors in codebook $e\in R^{LXD}$ .   
end

# E.1 Task Details

2D shape environment is a 5X5 grid world with different objects of different shapes and colors placed at random positions.Each location can only be occupied by one object.The underlying environment

dynamics of 3D shapes are the same as in the 2D dateset, and only the rendering component was changed (Kipf et al., 2019). In OOD setting, the total number of objects are changed for each environment. We used number of objects of 4 (validation), 3 (OOD-1) and 2 (OOD-2). We did not put in more than 5 objects because the environment will be too packed and the objects can hardly move.

The 3-body physics simulation environment is an interacting system that evolves according to physical laws.There are no actions applied onto any objects and movement of objects only depend on interaction among objects. This environment is adapted from Kipf et al. (2019). In the training environment, the radius of each ball is 3.In OOD settings, we changed the radius to 4 ( validation) and 2 (OOD test).

In all the 8 Atari games belong to the same collections of 2600 games from Atari Corporation.We used the games adapted to OpenAI gym environment. There are several versions of the same game available in OpenAI gym. We used version "Deterministic-v0" starting at warm start frame 50 for each game for training. Version "Frameskip-v0" starting at frame 250 as OOD validation and "Frameskip-v4" starting at frame 150 at OOD test.

In all the GNN compositional reasoning experiments. HITS at RANK K $\mathrm { K } { = } 1$ in this study) was used as as the metrics for performance.This binary score is 1 for a particular example if the predicted state representation is in the k-nearest neighbor set around the true observation. Otherwise this score is 0. MEAN RECIPROCAL RANK (MRR) is also used as a performance metrics, which is defined as M RR = 1N P n=1 1Rankn $\begin{array} { r } { M R R = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { 1 } { R a n k _ { n } } } \end{array}$ where $r a n k _ { n }$ is the rank of the n-th sample (Kipf et al. (2019)).

In adding task, gap length of 500 was used for training and gap length of 200 (OOD validation) and 1000 (OOD testing) are used for OOD settings. In sequential MNIST experiment , model was trained at 14X14 resolution and tested in different resolutions (Goyal et al. (2019)). Sort-of-Clevr experiments are conducted in the same way as Goyal et al. (2021b)

# E.2 Model Architecture, Hyperparameters and Training Details

DVNC implementation details In DVNC, codebook $e \in \mathbb { R } ^ { L \times m }$ was initialized by applying Kmeans clustering on training data points (s) where the number of clusters is $L$ . The nearest $e _ { j }$ ,by Euclidean distance ,is assigned to each $s _ { i }$ . The commitment loss $\begin{array} { r } { \beta \sum _ { i } ^ { G } | | s _ { i } - \mathrm { s g } ( e _ { o _ { i } } ) | | _ { 2 } ^ { 2 } } \end{array}$ , which encourages $s _ { i }$ stay close to the chosen codebook vector, and the task loss are back-propagated to each of the model components that send information in the inter-component communication process.The gradient of task loss are back-propagated to each of the components that send information using straight-through gradient estimator. The codebook loss $\begin{array} { r } { \sum _ { i } ^ { G } | | \operatorname { s g } ( s _ { i } ) - e _ { o _ { i } } | | ^ { 2 } } \end{array}$ that encourages the selected codebook vector to stay close to $s _ { i }$ is back-propagated to the selected codebook vector. Task loss is not backpropagated to codebook vectors. Only the task loss is back-propagated to the model component that receives the information. It is worth pointing out that in this study, we train the codebook vectors directly using gradient descent instead of using exponential moving average updates as in Oord et al. (2017).

Model architecture, hyperparameters and training settings of GNN used in this study are same as in Kipf et al. (2019), where encoder dimension is 4 and number of object slot is 3..Model architecture, hyperparameters and training settings of RIMs used in this study are identify to Goyal et al. (2019),where 6 RIM units and ${ \bf k } { = } 4$ are used. Model architecture, hyperparameters and training settings of transformer models are the same as in Goyal et al. (2021b), except that we did not include shared workspace. Hyperparameters of GNN and RIM models are summarized in table 4. Hyperparameters of transformers with various settings can be found in Goyal et al. (2021b). In all the models mentioned above,we include discretization of communication in DVNC and keep other parts of the model unchanged.

Data are split into training set, validation set and test set, the ratio varies among different tasks depending on data availability.For in-distribution performance, validation set has the same distribution as training set. In OOD task, one of the OOD setting,eg. certain number of blocks in 2D shape experiment, is used as validation set. The OOD setting used for validation was not included in test set.

![](images/01e88c1558e2fce32765278f96c3e0911288e940d5ab806699f0353f1bb8886b.jpg)  
(a) 2D Shapes

![](images/92a4919556a7a08c0f405a57e0c562e3685f3283ad979b0caea27f7a738fbc6a.jpg)  
(b) Three-body

![](images/1b78b53837b7786215b83b3cec27ee8758748f2fbac75302cf115ffca93046bc.jpg)

![](images/9641c0644b92dbf376b876a523df63341d15ecd8cc7d1dc41949dcdf9791494d.jpg)

![](images/df658e186ba990b0e50450dceabf2736cfceb3904bed15f9c75db4804bb5f636.jpg)

![](images/2a59e6eac81e4274edd8da5b9d32b4d2c5b38876f3b7059574bb7ec32a6a50de.jpg)

![](images/e60ffeeaca994ecd1f4ea3fdee2149e93479511b7157a46fd5f973bf857d3ffc.jpg)

![](images/9663d35c5d3f304b2c98e02a77b035c1d7188aaf6609fc8cd6c9e04e5d53b85d.jpg)

![](images/c22a98102e06fdab65bcec9bdd558d2fe580a92743c0524aa67177a773492982.jpg)

![](images/4375b5b8a1cfc0d3e2158d1737c574320638c628e35539f52b2b2cb620eb1ca5.jpg)  
(c) Atari Games

![](images/debe128a08bd5a60d4af8cd2fb750eeb27a3026c5f30407803805cac5b8c4c35.jpg)  
Figure 7: Examples of different task environments. Atari game screen shots are obtained from OpenAI gym platform. Sort-of-Clevr example was adapted from Goyal et al. (2021b) with permission

# Relational questions:

1.What is the shape of the object closest to the red object? ⇒ square   
2What is the shape of the object furthest to the orange object? $\Rightarrow$ circle   
3How many objects have same shape with the blue object? ⇒ 3

# Non-relational questions:

1.What is the shape of the red object $\Rightarrow$ Circle   
2. Is green object placd on the left side of the image? ⇒ yes   
3. Is orange object placed on the upside of the image? = no

# (d) Sort-of-Clevr

Table 4: Hyperparameters used for GNN and RIMs   

<table><tr><td>GNN model</td><td></td><td></td><td>RIMs model</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hyperparameters</td><td>Values</td><td></td><td>Hyperparameters</td><td>Values</td></tr><tr><td>Batch size</td><td>1024</td><td></td><td>Batch size</td><td>64</td></tr><tr><td>hidden dim</td><td>512</td><td></td><td>hidden dim</td><td>300</td></tr><tr><td>embedding-dim</td><td>512/G</td><td></td><td>embedding-dim</td><td>300/G</td></tr><tr><td>codebook_loss_weight</td><td>1</td><td></td><td>codebook_loss_weight</td><td>0.25</td></tr><tr><td>Max. number of epochs</td><td>200</td><td></td><td>Max. number of epochs</td><td>100</td></tr><tr><td>Number of slots(objects)</td><td>5</td><td></td><td>learning-rate</td><td>0.001</td></tr><tr><td>learning-rate</td><td>5.00E-04</td><td></td><td>Optimizer</td><td>Adam</td></tr><tr><td>Optimizer</td><td>Adam</td><td></td><td>Number of Units (RIMs)</td><td>6</td></tr><tr><td></td><td></td><td></td><td>Number of active RIMs</td><td>4</td></tr><tr><td></td><td></td><td></td><td>RIM unit type</td><td>LSTM</td></tr><tr><td></td><td></td><td></td><td>dropout</td><td>0.5</td></tr><tr><td></td><td></td><td></td><td>gradient clipping</td><td>1</td></tr></table>

# E.3 Computational resources

GPU nodes on university cluster are used. GNN training takes 3 hrs for each task with each hyperparameter setting on Tesla GPU. Training of RIMs and transformers take about 12 hours on the same GPU for each task. In total, the whole training progress of all models, all tasks, all hyperparameter settings takes approximately 800 hours on GPU nodes.