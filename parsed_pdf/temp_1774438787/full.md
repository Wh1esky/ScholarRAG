# BAYESIMP: Uncertainty Quantification for Causal Data Fusion

Siu Lun Chau∗

University of Oxford

Jean-François Ton∗

University of Oxford

Javier González

Microsoft Research Cambridge

Yee Whye Teh

University of Oxford

Dino Sejdinovic

University of Oxford

# Abstract

While causal models are becoming one of the mainstays of machine learning, the problem of uncertainty quantification in causal inference remains challenging. In this paper, we study the causal data fusion problem, where datasets pertaining to multiple causal graphs are combined to estimate the average treatment effect of a target variable. As data arises from multiple sources and can vary in quality and quantity, principled uncertainty quantification becomes essential. To that end, we introduce Bayesian Interventional Mean Processes, a framework which combines ideas from probabilistic integration and kernel mean embeddings to represent interventional distributions in the reproducing kernel Hilbert space, while taking into account the uncertainty within each causal graph. To demonstrate the utility of our uncertainty estimation, we apply our method to the Causal Bayesian Optimisation task and show improvements over state-of-the-art methods.

# 1 Introduction

Causal inference has seen a significant surge of research interest in areas such as healthcare [1], ecology [2], and optimisation [3]. However, data fusion, the problem of merging information from multiple data sources, has received limited attention in the context of causal modelling, yet presents significant potential benefits for practical situations [4, 5]. In this work, we consider a causal data fusion problem where two causal graphs are combined for the purposes of inference of a target variable (see Fig.1). In particular, our goal is to quantify the uncertainty under such a setup and determine the level of confidence in our treatment effect estimates.

![](images/d4241b0f04831c80b8c80bd9b79f3b54532c6b76c9f7e62610e4d86cb9d125c5.jpg)

![](images/7994d7e50d6986730e17046c940e8069517d6be3b7c427e4e76b389af6664183.jpg)

![](images/f30e51e4ec8667d22b8348031aaa20f55277ac8e036f403a748477ddea072426.jpg)  
Figure 1: Example problem setup: Causal graphs collected in two separate medical studies i.e. [6] and [7]. (Left) $\mathcal { D } _ { 1 }$ : Data describing the causal relationships between statin level and Prostate Specific Antigen (PSA). (Right) $\mathcal { D } _ { 2 }$ : Data from a prostate cancer study for patients about to receive a radical prostatectomy. Goal: Model E[Cancer Volume|do(Statin)] while also quantifying its uncertainty.

Target Variable

Mediating Variable

Treatment Variable

Confounding Variable

Let us consider the motivating example in Fig.1, where a medical practitioner is investigating how prostate cancer volume is affected by a statin drug dosage. We consider the case where the doctor

Preprint. Under review.

only has access to two separate medical studies describing the quantities of interest. On one hand we have observational data, from one medical study $\mathcal { D } _ { 1 }$ [1], describing the causal relationship between statin level and prostate specific antigen (PSA), and on the other hand we have observational data, from a second study $\mathcal { D } _ { 2 }$ [7], that looked into the link between PSA level and prostate cancer volume. The goal is to model the interventional effect between our target variable (cancer volume) and the treatment variable (statin). This problem setting is different from the standard observational scenario as it comes with the following challenges:

• Unmatched data: Our goal is to estimate E[cancer volume|do(statin)] but the observed cancer volume is not paired with statin dosage. Instead, they are related via a mediating variable PSA.   
• Uncertainty quantification: The two studies may be of different data quantity/quality. Furthermore, a covariate shift in the mediating variable, i.e. a difference between its distributions in two datasets, may cause inaccurate extrapolation. Hence, we need to account for uncertainty in both datasets.

Formally, let $X$ be the treatment (Statin), $Y$ be the mediating variable (PSA) and $T$ our target (cancer volume), and our aim is to estimate $\mathbb { E } [ T | d o ( X ) ]$ . The problem of unmatched data in a similar context has been previously considered by [5] using a two-staged regression approach $X  Y$ and $Y  T$ ). However, uncertainty quantification, despite being essential if our estimates of interventional effects will guide decision-making, has not been previously explored. In particular, it is crucial to quantify the uncertainty in both stages as this takes into account the lack of data in specific parts of the space. Given that we are using different datasets for each stage, there are also two sources of epistemic uncertainties (due to lack of data) as well as two sources of aleatoric uncertainties (due to inherent randomness in $Y$ and $T$ ) [8] . It is thus natural to consider regression models based on Gaussian Processes (GP) [9], as they are able to model both types of uncertainties. However, as GPs, or any other standard regression models, are designed to model conditional expectations only and will fail to capture the underlying distributions of interest (e.g. if there is multimodality in $Y$ as discussed in [10]). This is undesirable since, as we will see, interventional effect estimation requires accurate estimates of distributions. While one could in principle resort to density estimation methods, this becomes challenging since we typically deal with a number of conditional/ interventional densities.

In this paper, we introduce the framework of Bayesian Interventional Mean Processes (BAYESIMP) to circumvent the challenges in the causal data fusion setting described above. BAYESIMP considers kernel mean embeddings [11] for representing distributions in a reproducing kernel Hilbert space (RKHS), in which the whole arsenal of kernel methods can be extended to probabilistic inference (e.g. kernel Bayes rule [12], hypothesis testing [13], distribution regression [14]). Specifically, BAYES-IMP uses kernel mean embeddings to represent the interventional distributions and to analytically marginalise out $Y$ , hence accounting for aleatoric uncertainties. Further, BAYESIMP uses GPs to estimate the required kernel mean embeddings from data in a Bayesian manner, which allows to quantify the epistemic uncertainties when representing the interventional distributions. To illustrate the quality of our uncertainty estimates, we apply BAYESIMP to Causal Bayesian Optimisation [15], an efficient heuristic to optimise objective functions of the form $\begin{array} { r } { x ^ { * } = \arg \operatorname* { m i n } _ { x \in \mathcal { X } } \mathbb { E } [ T | d o ( X ) = x ] } \end{array}$ . Our contributions are summarised below:

1. We propose a novel Bayesian Learning of Conditional Mean Embedding (BAYESCME) that allows us to estimate conditional mean embeddings in a Bayesian framework.   
2. Using BAYESCME, we propose a novel Bayesian Interventional Mean Process (BAYESIMP) that allows us to model interventional effect across causal graphs without explicit density estimation, while obtaining uncertainty estimates for ${ \mathbb E } [ T | d o ( X ) = x ]$ .   
3. We apply BAYESIMP to Causal Bayesian Optimisation, a problem introduced in [15] and show significant improvements over existing state-of-the-art methods.

Note that [16] also considered a causal fusion problem but with a different objective. They focused on extrapolating experimental findings across treatment domains, i.e. inferring $\mathbb { E } [ Y | d o ( { \dot { X } } ) ]$ when only data from $p ( Y | d o ( S ) )$ is observed, where $S$ is some other treatment variable. In contrast, we focus on modelling combined causal graphs, with a strong emphasis on uncertainty quantification. While [17] considered mapping interventional distributions in the RKHS to model quantities such as $\mathbb { E } [ T | d o ( X ) ]$ , they only considered a frequentist approach, which does not account for epistemic uncertainties.

Notations. We denote $X , Y , Z$ as random variables taking values in the non-empty sets $\mathcal { X } , \mathcal { y }$ and $\mathcal { Z }$ respectively. Let $k _ { x } : \mathcal { X } \times \mathcal { X } \to \mathbb { R }$ be positive definite kernels on $X$ with an associated RKHS $\mathcal { H } _ { k _ { x } }$ . The corresponding canonical feature map $k _ { x } ( x ^ { \prime } , \cdot )$ is denoted as $\phi _ { x } ( x ^ { \prime } )$ . Analogously for $Y$ and $Z$ .

![](images/59a0a31928cf0315682b02aa3752820bcb808693d6a63a692928b7ecaab13df0.jpg)  
Figure 2: A general two stage causal learning setup.

In the simplest setting, we observe i.i.d samples $\begin{array} { r l } { \mathcal { D } _ { 1 } } & { { } = } \end{array}$ $\{ x _ { i } , y _ { i } , z _ { i } \} _ { i = 1 } ^ { \hat { N } }$ from joint distribution $\mathbb { P } _ { X Y Z }$ which we con-

catenate into vectors $\mathbf { x } : = [ x _ { 1 } , . . . , x _ { N } ] ^ { \top }$ . Similarly for y and z. For this work, $X$ is referred as treatment variable, $Y$ as mediating variable and $Z$ as adjustment variables accounting for confounding effects. With an abuse of notation, features matrices are defined by stacking feature maps along the columns, i.e $\Phi _ { \mathbf { x } } : = [ \phi _ { x } ( x _ { 1 } ) , . . . , \phi _ { x } ( x _ { N } ) ]$ . We denote the Gram matrix as $K _ { \mathbf { x } \mathbf { x } } : = \Phi _ { \mathbf { x } } ^ { \top } \Phi _ { \mathbf { x } } ^ { \phantom { \top } }$ and the vector of evaluations $k _ { x \mathbf { x } }$ as $[ k _ { x } ( x , x _ { 1 } ) , . . . , \bar { k } _ { x } ( x , x _ { N } ) ]$ . We define $\Phi _ { \mathbf { y } } , \Phi _ { \mathbf { z } }$ analogously for y and $\mathbf { z }$ .

Lastly, we denote $T = f ( Y ) + \epsilon$ as our target variable, which is modelled as some noisy evaluation of a function $f : \mathcal { V } \to \mathcal { T }$ on $Y$ while $\epsilon$ being some random noise. For our problem setup we observe a second dataset of i.i.d realisations $\mathcal { D } _ { 2 } = \{ \breve { y } _ { j } , t _ { j } \} _ { j = 1 } ^ { M }$ from the joint $\mathbb { P } _ { Y T }$ independent of $\mathcal { D } _ { 1 }$ . Again, we define $\tilde { \mathbf { y } } : = [ \tilde { y } _ { 1 } , . . . , \tilde { y } _ { M } ] ^ { \top }$ and $\mathbf { t } : = [ t _ { 1 } , . . . , t _ { M } ] ^ { \top }$ just like for $\mathcal { D } _ { 1 }$ . See Fig.2 for illustration.

# 2 Background

Representing interventional distributions in an RKHS has been explored in different contexts [18, 17, 19]. In particular, when the treatment is continuous, [17] introduced the Interventional Mean Embeddings (IMEs) to model densities in an RKHS by utilising smoothness across treatments. Given that IME is an important building block to our contribution, we give it a detailed review by first introducing the key concepts of $d o$ -calculus [20] and conditional mean embeddings [21].

# 2.1 Interventional distribution and do-calculus

In this work, we consider the structural causal model [20] (SCM) framework, where a causal directed acyclic graph (DAG) $\mathcal { G }$ is given and encodes knowledge of existing causal mechanisms amongst the variables in terms of conditional independencies. Given random variables $X$ and $Y$ , a central question in interventional inference [20] is to estimate the distribution $p ( Y | d o ( X ) = x )$ ), where $\{ d o ( X ) = x \}$ represents an intervention on $X$ whose value is set to $x$ . Note that this quantity is not directly observed given that we are usually only given observational data, i.e, data sampled from the conditional $p \bar { ( \boldsymbol { Y } | \boldsymbol { X } ) }$ but not from the interventional density $p ( Y | d o ( X ) )$ . However, Pearl [20] developed do-calculus which allows us to estimate interventional distributions from purely observational distributions under the identifiability assumption. Here we present the backdoor and front-door adjustments, which are the fundamental components of DAG based causal inference.

The backdoor adjustment is applicable when there are observed confounding variables $Z$ between the cause $X$ and the effect $Y$ (see Fig. 3 (Top)). In order to correct for this confounding bias we can use the following equation, adjusting for $Z$ as $\begin{array} { r } { p ( Y | d o ( X ) \stackrel { - } { = } x ) = \int _ { \mathcal { Z } } p ( Y | X = x , z ) p ( \bar { z } ) \dot { d z } } \end{array}$ .

The front-door adjustment applies to cases when confounders are unobserved (see Fig. 3 (Bottom)). Given a set of front-door adjustment variables $Z$ , we can again correct the estimate for the causal effect from $X$ to $Y$ with $\begin{array} { r } { p ( Y | d o ( X ) = x ) = \int _ { \mathcal { Z } } \int _ { \mathcal { X } } p ( Y | x ^ { \prime } , z ) p ( z | X = x ) p ( x ^ { \prime } ) d x ^ { \prime } d z } \end{array}$ .

We rewrite the above formulae in a more general form as we show below. For the remainder of the paper we will opt for this notation:

$$
p (Y | d o (X) = x) = \mathbb {E} _ {\Omega_ {x}} [ p (Y | \Omega_ {x}) ] = \int p (Y | \Omega_ {x}) p (\Omega_ {x}) d \Omega_ {x} \tag {1}
$$

![](images/c7c75b466949ad48113e08bfb496ac566301008db6332b59306400e2ba87107e.jpg)  
Figure 3: (Top) Backdoor adjustment (Bottom) Front-door adjustment, dashed edges denote unobserved confounders.

For backdoor we have $\Omega _ { x } = \{ X = x , Z \}$ and $p ( \Omega _ { x } ) = \delta _ { x } p ( Z )$ where $\delta _ { x }$ is the Dirac measure at $X = x$ . For front-door, $\Omega _ { x } = \{ X ^ { \prime } , Z \}$ and $p ( \Omega _ { x } ) = p ( X ^ { \prime } ) p ( Z | X = x )$ .

# 2.2 Conditional Mean Embeddings

Kernel mean embeddings of distributions provide a powerful framework for representing probability distributions [11, 21] in an RKHS. In particular, we work with conditional mean embeddings (CMEs) in this paper. Given random variables $X , Y$ with joint distribution $\mathbb { P } _ { X Y }$ , the conditional mean embedding with respect to the conditional density $p ( Y | X = x )$ , is defined as:

$$
\mu_ {Y \mid X = x} := \mathbb {E} _ {Y \mid X = x} [ \phi_ {y} (Y) ] = \int_ {\mathcal {Y}} \phi_ {y} (y) p (y \mid X = x) d y \tag {2}
$$

CMEs allow us to represent the distribution $p ( Y | X = x )$ as an element $\mu _ { Y \mid X = x }$ in the RKHS $\mathcal { H } _ { k _ { y } }$ without having to model the densities explicitly. Following [21], CMES can be associated with a Hilbert-Schmidt operator $\mathcal { C } _ { Y | X } : \mathcal { H } _ { k _ { x } }  \mathcal { H } _ { k _ { y } }$ , known as the conditional mean embedding operator, which satisfies $\mu _ { Y | X = x } = \mathcal { C } _ { Y | X } \phi _ { x } ( x )$ where $\mathcal { C } _ { Y | X } : = \mathcal { C } _ { Y X } \mathcal { C } _ { X X } ^ { - 1 }$ with $\mathcal { C } _ { Y X } : = \mathbb { E } _ { Y , X } [ \phi _ { y } ( Y ) \otimes$ $\phi _ { x } ( X ) ]$ and $\mathcal { C } _ { X X } : = \mathbb { E } _ { X , X } [ \phi _ { x } ( X ) \otimes \phi _ { x } ( X ) ]$ being the covariance operators. As a result, the finite sample estimator of ${ \mathcal { C } } _ { Y \mid X }$ based on the dataset $\{ \mathbf { x } , \mathbf { y } \}$ can be written as:

$$
\hat {\mathcal {C}} _ {Y \mid X} = \Phi_ {\mathbf {y}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} \Phi_ {\mathbf {x}} ^ {T} \tag {3}
$$

where $\lambda > 0$ is a regularization parameter. Note that from Eq.3, [22] showed that the CME can be interpret as a vector-valued kernel ridge regressor (V-KRR) i.e. $\phi _ { x } ( x )$ is regressed to an element in $\mathcal { H } _ { k _ { y } }$ . This is crucial as CMEs allow us to turn the integration, in Eq.2, into a regression task and hence remove the need for explicit density estimation. This insight is important as it allows us to derive analytic forms for our algorithms. Furthermore, the regression formalism of CMEs motivated us to derive a Bayesian version of CME using vector-valued Gaussian Processes (V-GP), see Sec.3.

# 2.3 Interventional Mean Embeddings

Interventional Mean Embeddings (IME) [17] combine the above ideas to represent interventional distributions in RKHSs. We derive the front-door adjustment embedding here but the backdoor adjustment follows analogously. Denote $\mu _ { Y | d o ( X ) = x }$ as the IME corresponding to the interventional distribution $p ( Y | d o ( X ) = x )$ ), which can be written as:

$$
\mu_ {Y | d o (X) = x} := \int_ {\mathcal {Y}} \phi_ {y} (y) p (y | d o (X) = x) d y = \int_ {\mathcal {X}} \int_ {\mathcal {Z}} \underbrace {\left(\int_ {\mathcal {Y}} \phi_ {y} (y) p (y | x ^ {\prime} , z) d y\right)} _ {\text {C M E} \mu_ {Y | X = x, Z = z}} p (z | x) p (x ^ {\prime}) d z d x ^ {\prime}
$$

using the front-door formula with adjustment variable $Z$ , and rearranging the integrals. By definition of CME $\begin{array} { r } { \int \phi _ { y } ( y ) p ( y | x ^ { \prime } , z ) d y = C _ { Y | X , Z } ( \phi _ { x } ( x ^ { \prime } ) \otimes \phi _ { z } ( z ) ) } \end{array}$ and linearity of integration, we have

$$
= C _ {Y | X, Z} \left(\underbrace {\int_ {\mathcal {X}} \phi_ {x} \left(x ^ {\prime}\right) p \left(x ^ {\prime}\right) d x ^ {\prime}} _ {= \mu_ {X}} \otimes \underbrace {\int_ {\mathcal {Z}} \phi_ {z} (z) p (z | x) d z} _ {= \mu_ {Z | X = x}}\right) = C _ {Y | X, Z} \left(\mu_ {X} \otimes \mu_ {Z | X = x}\right)
$$

Using notations from Sec.2.1, embedding interventional distributions into an RKHS is as follows.

Proposition 1. Given an identifiable do-density of the form $p ( Y | d o ( X ) = x ) = \mathbb { E } _ { \Omega _ { x } } [ p ( Y | \Omega _ { x } ) ]$ , the general form of the empirical interventional mean embedding is given by,

$$
\hat {\mu} _ {Y \mid d o (X) = x} = \Phi_ {Y} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} (x) ^ {\top} \tag {4}
$$

where $K _ { \Omega _ { x } } = K _ { X X } \odot K _ { Z Z }$ and $\Phi _ { \Omega _ { x } } ( x )$ is derived depending on $p ( \Omega _ { x } )$ . In particular, for backdoor adjustments, $\Phi _ { \Omega _ { x } } ^ { ( b d ) } ( x ) = \Phi _ { X } ^ { \top } k _ { X } ( x , \cdot ) \odot \Phi _ { Z } ^ { \top } \hat { \mu } _ { z }$ and for front-door $\Phi _ { \Omega _ { x } } ^ { ( f d ) } ( x ) = \Phi _ { X } ^ { \top } \hat { \mu } _ { X } \odot \Phi _ { Z } ^ { \top } \hat { \mu } _ { Z | X = x }$ .

# 3 Our Proposed Method

Two-staged Causal Learning. Given two independent datasets $\mathcal { D } _ { 1 } = \{ ( x _ { i } , z _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ and $\mathcal { D } _ { 2 } =$ $\{ ( \tilde { y } _ { j } , t _ { j } ) \} _ { j = 1 } ^ { M }$ , our goal is to model the average treatment effect in $T$ i=1  when intervening on variable $X$ , i.e model $g ( x ) = \mathbb { E } [ T | d o ( X ) = x ]$ . Note that the target variable $T$ and the treatment variable $X$ are never jointly observed. Rather, they are linked via a mediating variable $Y$ observed in both

![](images/e28395c4b97ec262ed204de7059ac751c40b2abf64790f76457437558c4425f2.jpg)  
Figure 4: Two-staged causal learning problem

Table 1: Summary of our proposed methods   

<table><tr><td>METHODS</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>IME [17]</td><td>KRR</td><td>KRR</td></tr><tr><td>IMP (Ours)</td><td>KRR</td><td>GP</td></tr><tr><td>BAYESIME (Ours)</td><td>GP</td><td>KRR</td></tr><tr><td>BAYESIMP (Ours)</td><td>GP</td><td>GP</td></tr></table>

datasets. In our problem setting, we make the following two assumptions: (A1) The treatment only affects the target through the mediating variable, i.e $T$ ⊥⊥ $d o ( X ) | Y$ and (A2) Function $f$ given by $f ( y ) = \mathbb { E } [ T | \bar { Y } = y ]$ belongs to an RKHS $\mathcal { H } _ { k _ { y } }$ .

We can thus express the average treatment effect as:

$$
\begin{array}{l} g (x) = \mathbb {E} [ T | d o (X) = x ] = \int_ {\mathcal {Y}} \underbrace {\mathbb {E} [ T | d o (X) = x , Y = y ]} _ {= \mathbb {E} [ T | Y = y ], \text {s i n c e} T \bot d o (X) | Y} p (y | d o (X) = x) d y (5) \\ = \int_ {\mathcal {Y}} f (y) p (y | d o (X) = x) d y = \langle f, \mu_ {Y \mid d o (X) = x} \rangle \mathcal {H} _ {k _ {y}}. (6) \\ \end{array}
$$

The final expression decomposes the problem of estimating $g$ into that of estimating the IME $\mu _ { Y \mid d o ( X ) }$ (which can be done using $\mathcal { D } _ { 1 }$ ) and that of estimating the integrand $f : \mathcal { y }  \mathcal { T }$ (which can be done using $\mathcal { D } _ { 2 }$ ). Each of these two components can either be estimated using a GP or KRR approach (See Table 1). Furthermore, the reformulation as an RKHS inner product is crucial, as it circumvents the need for density estimation as well as the need for subsequent integration in Eq.6. Rather, the main parts of the task can now be viewed as two instances of regression (recall that mean embeddings can be viewed as vector-valued regression).

To model $g$ and quantify its uncertainty, we propose 3 GP-based approaches. While the first 2 methods, Interventional Mean Process (IMP) and Bayesian Interventional Mean Embedding (BAYESIME) are novel derivations that allow us to quantify uncertainty from either one of the datasets, we treat them as intermediate yet necessary steps to derive our main algorithm, Bayesian Interventional Mean Process (BAYESIMP), which allows us to quantify uncertainty from both sources in a principled way. For a summary of the methods, see Fig.4 and Table 1. All derivations are included in the appendix.

Interventional Mean Process: Firstly, we train $f$ as a GP using $\mathcal { D } _ { 2 }$ and model $\mu _ { Y | d o ( X ) = x }$ as V-KRR using $\mathcal { D } _ { 1 }$ . By drawing parallels to Bayesian quadrature [23] and conditional mean process introduced in [24], the integral of interest $\begin{array} { r } { g ( x ) = \int f ( y ) p ( y | d o ( X ) = x ) d y } \end{array}$ will be a GP indexed by the treatment variable $X$ . We can then use the empirical embedding ${ \hat { \mu } } _ { Y \mid d o ( X ) }$ learnt in $\mathcal { D } _ { 1 }$ to obtain an analytic mean and covariance of $g$ .

Bayesian Interventional Mean Embedding: Next, to account for the uncertainty from $\mathcal { D } _ { 1 }$ , we model $f$ as a KRR and $\mu _ { Y | d o ( X ) = x }$ using a V-GP. We introduce our novel Bayesian Learning of Conditional Mean Embeddings (BAYESCME), which uses a nuclear dominant kernel [25] construction, similar to [26], to ensure that the inner product $\langle f , \mu _ { Y \mid d o ( X ) = x } \rangle$ is well-defined. As the embedding is a GP, the resulting inner product is also a GP and hence takes into account the uncertainty in $\mathcal { D } _ { 1 }$ .(See Prop. 4).

Bayesian Interventional Mean Process: Lastly, in order to account for uncertainties coming from both $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ , we combine ideas from the above IMP and BAYESIME. We place GPs on both $f$ and $\mu _ { Y \mid d o ( X ) }$ and use their inner product to model $\mathbb { E } [ T | d o ( X ) ]$ . Interestingly, the resulting uncertainty can be interpreted as the sum of uncertainties coming from IMP and BAYESIME with an additional interaction term (See Prop.5).

# 3.1 Interventional Mean Process

Firstly, we consider the case where $f$ is modelled using a GP and $\mu _ { Y | d o ( X ) = x }$ using a V-KRR. This allows us to take into account the uncertainty from $\mathcal { D } _ { 2 }$ by modelling the relationship between $Y$ and $T$ using in a GP. Drawing parallels to Bayesian quadrature [23] where integrating $f$ with respect to a marginal measure results into a Gaussian random variable, we integrate $f$ with respect to a conditional measure, thus resulting in a GP indexed by the conditioning variable. Note that [24] studied this GP in a non-causal setting, for a very specific downscaling problem. In this work, we

extend their approach to model uncertainty in the causal setting. The resulting mean and covariance are then estimated analytically, i.e without integrals, using the empirical IME ${ \hat { \mu } } _ { Y \mid d o ( X ) }$ learnt from $\mathcal { D } _ { 1 }$ , see Prop.2.

Proposition 2 (IMP). Given dataset $D _ { 1 } = \{ ( x _ { i } , y _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N }$ and ${ \cal D } _ { 2 } = \{ ( \tilde { y } _ { j } , t _ { j } ) \} _ { j = 1 } ^ { M }$ , if $f$ is the posterior GP learnt from $\mathcal { D } _ { 2 }$ , then $\begin{array} { r } { g = \int f ( y ) p ( y | d o ( X ) ) d y } \end{array}$ is a GP $\mathcal { G P } ( m _ { 1 } , \kappa _ { 1 } )$ defined on the treatment variable $X$ with the following mean and covariance estimated using ${ \hat { \mu } } _ { Y \mid d o ( X ) }$ ,

$$
\begin{array}{l} m _ {1} (x) = \left\langle \hat {\mu} _ {Y | d o (x)}, m _ {f} \right\rangle_ {\mathcal {H} _ {k _ {y}}} = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y} \tilde {\mathbf {y}}} \left(K _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} \mathbf {t} (7) \\ \kappa_ {1} (x, x ^ {\prime}) = \hat {\mu} _ {Y | d o (x)} ^ {\top} \hat {\mu} _ {Y | d o \left(x ^ {\prime}\right)} - \hat {\mu} _ {Y | d o (x)} ^ {\top} \Phi_ {\tilde {\mathbf {y}}} \left(K _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda I\right) ^ {- 1} \Phi_ {\tilde {\mathbf {y}}} ^ {\top} \hat {\mu} _ {Y | d o \left(x ^ {\prime}\right)} (8) \\ = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \tilde {K} _ {\mathbf {y y}} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) (9) \\ \end{array}
$$

where $\hat { \mu } _ { Y | d o ( x ) } = \hat { \mu } _ { Y | d o ( X ) = x } , K _ { \tilde { \mathbf { y } } \mathbf { y } } = \Phi _ { \tilde { \mathbf { y } } } ^ { \top } \Phi _ { \mathbf { y } }$ , $m _ { f }$ and $\tilde { K } _ { \mathbf { y } \mathbf { y } }$ are the posterior mean function and covariance of $f$ evaluated at y respectively. $\lambda > 0$ is the regularisation of the CME. $\lambda _ { f } > 0$ is the noise term for GP $f$ . $\Omega _ { x }$ is the set of variables as specified in Prop.1.

Summary: The posterior covariance between $x$ and $x ^ { \prime }$ in IMP can be interpreted as the similarity between their corresponding empirical IMES $\hat { \mu } _ { Y \mid d o ( X ) = x }$ and ${ \hat { \mu } } _ { Y \mid d o ( X ) = x ^ { \prime } }$ weighted by the posterior covariance $\tilde { K } _ { \mathbf { y } \mathbf { y } }$ , where the latter corresponds to the uncertainty when modelling $f$ as a GP in $\mathcal { D } _ { 2 }$ However, since $f$ only considers uncertainty in $\mathcal { D } _ { 2 }$ , we need to develop a method that allows us to quantify uncertainty when learning the IME from $\mathcal { D } _ { 1 }$ . In the next section, we introduce a Bayesian version of CME, which then lead to BAYESIME, a remedy to this problem.

# 3.2 Bayesian Interventional Mean Embedding

To account for the uncertainty in $\mathcal { D } _ { 1 }$ when estimating $\mu _ { Y \mid d o ( X ) }$ , we consider a GP model for CME, and later extend to the interventional embedding IME. We note that Bayesian formulation of CMEs has also been considered in [27], but with a specific focus on discrete target spaces.

Bayesian learning of conditional mean embeddings with V-GP. As mentioned in Sec.2, CMEs have a clear "feature-to-feature" regression perspective, i.e $\mathbb { E } [ \phi _ { y } ( Y ) | X = x ]$ is the result of regressing $\phi _ { y } ( Y )$ onto $\phi _ { x } ( X )$ . Hence, we consider a vector-valued GP construction to estimate the CME.

Let $\mu _ { g p } ( x , y )$ be a GP that models $\mu _ { Y \mid X = x } ( y )$ . Given that $f \in \mathcal { H } _ { k _ { y } }$ , for $\langle f , \mu _ { g p } ( x , \cdot ) \rangle _ { \mathcal { H } _ { k _ { y } } }$ to be well defined, we need to ensure $\mu _ { g p } ( x , \cdot )$ is also restricted to $\mathcal { H } _ { k _ { y } }$ for any fixed $x$ . Consequently, we cannot define a $\mathcal { G P } ( 0 , k _ { x } \otimes k _ { y } )$ prior on $\mu _ { g p }$ as usual, as draws from such prior will almost surely fall outside $\mathcal { H } _ { k _ { x } } \otimes \mathcal { H } _ { k _ { y } }$ [25]. Instead we define a prior over $\mu _ { g p } \sim \mathcal G P ( 0 , k _ { x } \otimes r _ { y } )$ , where $r _ { y }$ is a nuclear dominant kernel [25] over $k _ { y }$ , which ensures that samples paths of $\mu _ { g p }$ live in $\mathcal { \breve { H } } _ { k _ { x } } \otimes \mathcal { H } _ { k _ { y } }$ almost surely. In particular, we follow a similar construction as in [26] and model $r _ { y }$ as $\begin{array} { r } { r _ { y } ( y _ { i } , y _ { j } ) = \int k _ { y } ( y _ { i } , u ) k _ { y } ( u , y _ { j } ) \nu ( d u ) } \end{array}$ where $\nu$ is some finite measure on $Y$ . Hence we can now setup a vector-valued regression in $\mathcal { H } _ { k _ { y } }$ as follows:

$$
\phi_ {y} \left(y _ {i}\right) = \mu_ {g p} \left(x _ {i}, \cdot\right) + \lambda^ {\frac {1}{2}} \epsilon_ {i} \tag {10}
$$

where $\epsilon _ { i } \sim \mathcal { G P } ( 0 , r )$ are independent noise functions. By taking the inner product with $\phi _ { y } ( y ^ { \prime } )$ on both sides, we then obtain $k _ { y } ( y _ { i } , y ^ { \prime } ) = \mu _ { g p } ( x _ { i } , y ^ { \prime } ) + \lambda ^ { \frac { 1 } { 2 } } \epsilon _ { i } ( y ^ { \prime } )$ . Hence, we can treat $k ( y _ { i } , y _ { j } )$ as noisy evaluations of $\mu _ { g p } ( x _ { i } , \stackrel { } { y _ { j } } )$ and obtain the following posterior mean and covariance for $\mu _ { g p }$ .

Proposition 3 (BAYESCME). The posterior GP of $\mu _ { g p }$ given observations $\{ \mathbf { x } , \mathbf { y } \}$ has the following mean and covariance:

$$
m _ {\mu} ((x, y)) = k _ {x \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y} y} \tag {11}
$$

$$
\kappa_ {\mu} \left(\left(x, y\right), \left(x ^ {\prime}, y ^ {\prime}\right)\right) = k _ {x x ^ {\prime}} r _ {y, y ^ {\prime}} - k _ {x \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} k _ {\mathbf {x} x ^ {\prime}} r _ {y \mathbf {y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y} y ^ {\prime}} \tag {12}
$$

In addition, the following marginal likelihood can be used for hyperparameter optimisation,

$$
- \frac {N}{2} \left(\log | K _ {\mathbf {x x}} + \lambda I | + \log | R |\right) - \frac {1}{2} \operatorname {T r} \left((K _ {\mathbf {x x}} + \lambda I) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} K _ {\mathbf {y y}}\right) \tag {13}
$$

Note that in practice we fix the lengthscale of $k _ { y }$ and $r _ { y }$ when optimising the above likelihood. This is to avoid trivial solutions for the vector-valued regression problem as discussed in [10]. The Bayesian version of the IME is derived analogously and we refer the reader to appendix due to limited space.

Finally, with V-GPS on embeddings defined, we can model $g ( x )$ as $\langle f , \mu _ { g p } ( x , \cdot ) \rangle _ { { \cal { H } } _ { k _ { y } } }$ , which due to the linearity of the inner product, is itself a GP. Here, we first considered the case where $f$ is a KRR learnt from $\mathcal { D } _ { 2 }$ and call the model BAYESIME.

Proposition 4 (BAYESIME). Given dataset $D _ { 1 } = \{ ( x _ { i } , y _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N }$ and $D _ { 2 } = \{ ( \tilde { y } _ { j } , t _ { j } ) \} _ { j = 1 } ^ { M }$ , if f is a KRR learnt from $\mathcal { D } _ { 2 }$ and $\mu _ { Y \mid d o ( X ) }$ modelled as a V-GP using $\mathcal { D } _ { 1 }$ , then $g = \langle f , \mu _ { Y | d o ( X ) } \rangle \sim$ $\mathcal { G P } ( m _ { 2 } , \kappa _ { 2 } )$ where,

$$
m _ {2} (x) = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} R _ {\mathbf {y} \tilde {\mathbf {y}}} A \tag {14}
$$

$$
\kappa_ {2} (x, x ^ {\prime}) = B \Phi_ {\Omega_ {x}} (x) ^ {\top} \Phi_ {\Omega_ {x}} (x) - C \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) \tag {15}
$$

where $\mathbf { \nabla } \cdot A = ( K _ { \mathbf { \tilde { y } \tilde { y } } } + \lambda _ { f } I ) ^ { - 1 } \mathbf { t } , B = A ^ { \top } R _ { \mathbf { \tilde { y } \tilde { y } } } A a n d C = A ^ { \top } R _ { \mathbf { \tilde { y } \tilde { y } } } R _ { \mathbf { y } \mathbf { \tilde { y } } } ^ { - 1 } R _ { \mathbf { y } \mathbf { \tilde { y } } } A$ $A = ( K _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } + \lambda _ { f } I ) ^ { - 1 } \mathbf { t }$ $B = A ^ { \top } R _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } A$

Summary: Constants $B$ and $C$ in $\kappa _ { 2 }$ can be interpreted as different estimation of $| | f | | _ { \mathcal { H } _ { k _ { y } } }$ , i.e the RKHS norm of $f$ . As a result, for problems that are “harder” to learn in $\mathcal { D } _ { 2 }$ , i.e. corresponding to larger magnitude of $| | f | | _ { \mathcal { H } _ { k _ { y } } }$ , will result into larger values of $B$ and $C$ . Therefore the covariance $\kappa _ { 2 }$ can be interpreted as uncertainty in $\mathcal { D } _ { 1 }$ scaled by the difficulty of the problem to learn in $\mathcal { D } _ { 2 }$ .

# 3.3 Bayesian Interventional Mean Process

To incorporate both uncertainties in $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ , we combine ideas from IMP and BAYESIME to estimate $g = \langle f , \mu _ { Y \mid d o ( X ) } \rangle$ by placing GPs on both $f$ and $\mu _ { Y \mid d o ( X ) }$ . Again as before, a nuclear dominant kernel $r _ { y }$ was used to ensure the GP $f$ is supported on $\mathcal { H } _ { k _ { y } }$ . For ease of computation, we consider a finite dimensional approximation of the GPs $f$ and $\mu _ { Y \mid d o ( X ) }$ and estimate $g$ as the RKHS inner product between them. In the following we collate $\mathbf { y }$ and $\tilde { \mathbf { y } }$ into a single set of points $\hat { \mathbf { y } }$ , which can be seen as landmark points for the finite approximation [28]. We justify this in the Appendix.

Proposition 5 (BAYESIMP). Let $f$ and $\mu _ { Y \mid d o ( X ) }$ be GPs learnt as above. Denote $\tilde { f }$ and ${ \tilde { \mu } } _ { Y \mid d o ( X ) }$ as the finite dimensional approximation of $f$ and $\mu _ { Y \mid d o ( X ) }$ respectively. Then $\tilde { g } = \langle \tilde { f } , \tilde { \mu } _ { Y | d o ( X ) } \rangle$ has the following mean and covariance:

$$
m _ {3} (x) = E _ {x} K _ {\mathbf {y} \hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} R _ {\hat {\mathbf {y}} \tilde {\mathbf {y}}} \left(R _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} \mathbf {t} \tag {16}
$$

$$
\kappa_ {3} \left(x, x ^ {\prime}\right) = \underbrace {E _ {x} \Theta_ {1} ^ {\top} \tilde {R} _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} \Theta_ {1} E _ {x ^ {\prime}} ^ {\top}} _ {\text {U n c e r t a i n t y f r o m} \mathcal {D} _ {1}} + \underbrace {\Theta_ {2} ^ {(a)} F _ {x x ^ {\prime}} - \Theta_ {2} ^ {(b)} G _ {x x ^ {\prime}}} _ {\text {U n c e r t a i n t y f r o m} \mathcal {D} _ {2}} + \underbrace {\Theta_ {3} ^ {(a)} F _ {x x ^ {\prime}} - \Theta_ {3} ^ {(b)} G _ {x x ^ {\prime}}} _ {\text {U n c e r t a i n t y f r o m I t e r a c t i o n}} \tag {17}
$$

where $E _ { x } \ = \ \Phi _ { \Omega _ { x } } ( x ) ^ { \top } ( K _ { \Omega _ { x } } + \lambda I ) ^ { - 1 } , F _ { x x ^ { \prime } } \ = \ \Phi _ { \Omega _ { x } } ( x ) ^ { \top } \Phi _ { \Omega _ { x } } ( x ^ { \prime } ) , G _ { x x ^ { \prime } } \ = \ \Phi _ { \Omega _ { x } } ( x ) ^ { \top } ( K _ { \Omega _ { x } } + K _ { \Omega _ { x } } ) ,$ $\lambda I ) ^ { - 1 } \Phi _ { \Omega _ { x } } ( x ^ { \prime } )$ , and $\Theta _ { 1 } = K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } R _ { \hat { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } K _ { \mathbf { y } \mathbf { y } }$ , $\Theta _ { 2 } ^ { ( a ) } = \Theta _ { 4 } ^ { \top } R _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } \Theta _ { 4 } , \Theta _ { 2 } ^ { ( b ) } = \Theta _ { 4 } ^ { \top } R _ { \hat { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } R _ { \mathbf { y } \hat { \mathbf { y } } } \Theta _ { 4 }$ and $\begin{array} { r c l r } { \Theta _ { 3 } ^ { ( a ) } } & { = } & { t r ( K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } R _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } \bar { R } _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ) , \Theta _ { 3 } ^ { ( b ) } } & { = } & { t r ( R _ { \hat { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } R _ { \mathbf { y } \hat { \mathbf { y } } } K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } \bar { R } _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } ) } \end{array}$ and $\begin{array} { r l } { \Theta _ { 4 } } & { { } = } \end{array}$ $K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } R _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ( K _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } + \lambda _ { f } ) ^ { - 1 } \mathbf { t } .$ $\bar { R } _ { \hat { \mathbf { y } } \hat { \mathbf { y } } }$ is the posterior covariance of $f$ evaluated at $\hat { \mathbf { y } }$

Summary: While the first two terms in $\kappa _ { 3 }$ resemble the uncertainty estimates from IMP and BAYESIME, the last term acts as an extra interaction between the two uncertainties from $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ . We note that unlike IMP and BAYESIME, $\tilde { g }$ from Prop.5 is not a GP as inner products between Gaussian vectors are not Gaussian. Nonetheless, the mean and covariance can be estimated.

# 4 Experiments

In this section, we first present an ablation studies on how our methods would perform under settings where we have missing data parts at different regions of the two datasets. We then demonstrate BAYESIMP’s proficiency in the Causal Bayesian Optimisation setting.

In particular, we compare our methods against the sampling approach considered in [15]. [15] start by modelling $f : Y \to T$ as GP and estimate the density $p ( Y | d o ( X ) )$ using a GP along with do-calculus. Then givThe emp $x$ we obtain standard d $L$ samples of iation of th $y _ { l }$ and samp $R$ $f _ { r }$ m their posterior GPs.can now be taken to $\{ f _ { r } ( y _ { l } ) \} _ { l = 1 , r = 1 } ^ { L , R }$ ${ \mathbb E } [ T | d o ( X ) = x ]$ estimation requires repeated sampling and is thus inefficient compared to our approaches, where we explicitly model the uncertainty as covariance function.

![](images/0428755e7b37551a2b3e0d44f8c5fc18425d447d6fa7d71fe2b18585b0e319eb.jpg)

![](images/45d4e0aeec04794ab606704b1e5a4c893767888f9c28e5ef8665033d2d474cb8.jpg)

![](images/e39be29c47478033ce5707e0ed214dc6f83a6c6803605d905c54d28e1403bb27.jpg)

![](images/2e492a349a5610abd8f426efe113beeaec267bf86ba2c24d6bc5e51e11234d7a.jpg)

![](images/a831c3d50b6336072f6996e8039446a79e265ee5405c23ccdb29f12723ca5364.jpg)  
Figure 5: Ablation studies of various methods in estimating uncertainties for an illustrative experiment. $^ *$ indicates our methods. $N = M = 1 0 0$ data points are used. Uncertainty from sampling gives a uniform estimate of uncertainty and IME does not come with uncertainty estimates. We see IMP and BAYESIME covering different regions of uncertainty while BAYESIMP takes the best of both worlds.

Ablation study. In order to get a better intuition into our methods, we will start off with a preliminary example, where we investigate the uncertainty estimates in a toy case. We assume two simple causal graphs $X  Y$ for $\mathcal { D } _ { 1 }$ and $Y  T$ for $\mathcal { D } _ { 2 }$ and the goal is to estimate ${ \mathbb E } [ T | d o ( X ) = x ]$ (generating process given in the appendix). We compare our methods from Sec.3 with the sampling-based uncertainty estimation approach described above. In Fig.5 we plot the mean and the $9 5 \%$ credible interval of the resulting GP models for ${ \mathbb E } [ T | d o ( X ) = x ]$ . On the $x$ -axis we also plotted a histogram of the treatment variable $x$ to illustrate its density.

From Fig.5(a), we see that the uncertainty for sampling is rather uniform across the ranges of $x$ despite the fact we have more data around $x = 0$ . This is contrary to our methods, which show a reduction of uncertainty at high $x$ density regions. In particular, $x = - 5$ corresponds to an extrapolation of data, where $x$ gets mapped to a region of $y$ where there is no data in $\mathcal { D } _ { 2 }$ . This fact is nicely captured by the spike of credible interval in Fig.5(c) since IMP utilises uncertainty from $\mathcal { D } _ { 2 }$ directly. Nonetheless, IMP failed to capture the uncertainty stemming from $\mathcal { D } _ { 1 }$ , as seen from the fact that the credible interval did not increase as we have less data in the region $| x | > 5$ . In contrast, BAYESIME (Fig.5(d)) gives higher uncertainty around low $x$ density regions but failed to capture the extrapolation phenomenon. Finally, BAYESIMP Fig.5(e) seems to inherit the desirable characteristics from both IMP and BAYESIME, due to taking into account uncertainties from both $\mathcal { D } _ { 1 }$ , $\mathcal { D } _ { 2 }$ . Hence, in the our experiments, we focus on BAYESIMP and refer the reader to the appendix for the remaining methods.

BayesIMP for Bayesian Optimisation (BO). We now demonstrate, on both synthetic and real-world data, the usefulness of the uncertainty estimates obtained using our methods in BO tasks. Our goal is to utilise the uncertainty estimates to direct the search for the optimal value of $\mathbb { E } [ T | d o ( X ) \bar { = } x ]$ by querying as few values of the treatment variable $X$ as possible, i.e. we want to optimize for the following two datasets: from Prop.5 is not a GP as i $\begin{array} { r } { x ^ { * } = \arg \operatorname* { m i n } _ { x \in \mathcal { X } } \mathbb { E } [ T | d o ( X ) = x ] } \end{array}$ $\mathcal { \bar { D } } _ { 1 } = \bar { \{ } x _ { i } , u _ { i } , z _ { i } , y _ { i } \} _ { i = 1 } ^ { \bar { N } }$ . For the first synthetic experiment (see Fig.6 (Top)), we will use and auss $\mathcal { D } _ { 2 } = \{ \tilde { y } _ { j } , t _ { j } \} _ { j = 1 } ^ { M }$ . Note that BAYESIMP Gaussian. Nonetheless, with the mean and covariance estimated, we will use moment matching to construct a GP out of BAYESIMP for posterior inference. At the start, we are given $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ , where these observations are used to construct a GP prior for the interventional effect of $X$ on $T$ , i.e $\mathbb { E } [ T | d o ( X ) = x ]$ , to provide a “warm” start for the BO.

Again we compare BAYESIMP with the sampling-based estimation of $\mathbb { E } [ T | d o ( X ) ]$ and its uncertainty, which is exactly the formulation used in the Causal Bayesian Optimisation algorithm (CBO) [15]. In order to demonstrate how BAYESIMP performs in the multimodal setting, we will be considering the case where we have the following distribution on $Y$ i.e. $p ( y | u , z ) = \pi p _ { 1 } ( y | u , z ) + ( 1 - \pi ) p _ { 2 } ( y | u , z )$ where $Y$ is a mixture and $\pi \in [ 0 , 1 ]$ . These scenarios might arise when there is an unobserved binary variable which induces a switching between two regimes on how $Y$ depends on $( U , Z )$ . In this case, the GP model of [15] would only capture the conditional expectation of $Y$ with an inflated variance, leading to slower convergence

and higher variance in the estimates of the prior as we will show in our experiments. Throughout our experiments, similarly to [15], we will be using the expected improvement (EI) acquisition function to select the next point to query.

![](images/9281e3a00601038b06e482949dd1ae3b0f81005b7a246d2c323d82cec182657c.jpg)  
Figure 6: Illustration of synthetic data experiments.

Synthetic data experiments. We compare BAYESIMP to CBO as well as to a simple GP with no learnt prior as baseline. We will be using $N = 1 0 0$ datapoints for $\mathcal { D } _ { 1 }$ and $M = 5 0$ datapoints $\mathcal { D } _ { 2 }$ . We ran each method 10 times and plot the resulting standard deviation for each iteration in the figures below. The data generation and details were added in the Appendix. We see from the Fig.7 that

![](images/97816f8d58754dcf672b94b7c88949de3dc4a29bda1cbe9dae3e5dd424d84169.jpg)

![](images/d56f5c5a26acf4acc1dc8c8ccc7e6a8bd85c6f4ab8f7d318da198c9782f2933c.jpg)

![](images/6442bda22d9b07cb7b33a9e5e81ee5024382b911405601c2c402de3a99eae497.jpg)  
Figure 7: We are interested in finding the maximal value of $\mathbb { E } [ T | d o ( X ) = x ]$ with as few BO iterations as possible. We ran experiments with multimodality in $Y$ . (Left) Using front-door adjustment (Middle) Using backdoor adjustment (Right) Using backdoor adjustment (unimodal Y )

BAYESIMP is able to find the maxima much faster and with smaller standard deviations, than the current state-of-the-art method, CBO, using both front-door and backdoor adjustments (Fig.7(Right, Middle)). Given that our method uses more flexible representations of conditional distributions, we are able to circumvent the multimodality problem in $Y$ . In addition, we also consider the unimodal version, i.e. $\pi = 0$ (see right Fig.7). We see that the performance of CBO improves in the unimodal setting, however BAYESIMP still converges faster than CBO even in this scenario.

Next, we consider a harder causal graph (see Fig.6 (Bottom)), previously considered in [15]. We again introduce multimodality in the $Y$ variable in order to explore the case of more challenging conditional densities. We see from Fig.8 (Left, Middle), that BAYESIMP again converges much faster to the true optima than CBO [15] and the standard GP prior baseline. We note that the fast convergence of BAYESIMP throughout our experiments is not due to simplicity of the underlying BO problems. Indeed, the BO with a standard GP prior requires significantly more iterations. It is rather the availability of the observational data, allowing us to construct a more appropriate prior, which leads to a “warm” start of the BO procedure.

![](images/1edf1c8811e05d1ab8e98849eeb072a701f0649eb45c8acd3715eb8f764f7d89.jpg)

![](images/41b4e45ed545c85d8e7ffb683e410ff8efbbd2b4e382898b2fc170d7f8351282.jpg)

![](images/954da9f0cc0ef43bbfcf2239cd2c463185b6555b6a961086edbf43022581c407.jpg)  
Figure 8: (Left) Experiments where we are interested in $\mathbb { E } [ T | d o ( D ) = d ]$ with multimodal $Y$ , (Middle) Experiments where we are interested in $\mathbb { E } [ T | d o ( \dot { E } ) \dot { = } e ]$ with multimodal $Y$ , (Right) Experiments on healthcare data where we are interested in E[Cancer Volume|do(Statin)].

Healthcare experiments. We conclude with a healthcare dataset corresponding to our motivating medical example in Fig.1. The causal mechanism graph, also considered in the CBO paper [15], studies the effect of certain drugs (Aspirin/Statin) on Prostate-Specific Antigen (PSA) levels [6]. In our case, we modify statin to be continuous, in order to optimize for the correct drug dosage. However, in contrast to [15], we consider a second experimental dataset, arising from a different medical study, which looks into the connection between PSA levels and cancer volume amount in patients [7]. Similar to the original CBO paper [15], given that interventional data is hard to obtain, we construct data generators based on the true data collected in [7]. This is done by firstly fitting a GP on the data and then sampling from the posterior (see Appendix for more details). Hence this is the perfect testbed for our model where we are interested in $\mathbb { E } [ C a n c e r ~ V o l u m e | d o ( S t a t i n ) ]$ . We see from Fig.8 (Right) that BAYESIMP again converges to the true optima faster than CBO hence allowing us to find the fastest ways of optimizng cancer volume by requesting much less interventional data. This could be critical as interventional data in real-life situations can be very expensive to obtain.

# 5 Discussion and Conclusion

In this paper we propose BAYESIMP for quantifying uncertainty in the setting of causal data fusion. In particular, our proposed method BAYESIMP allows us to represent interventional densities in the RKHS without explicit density estimation, while still accounting for epistemic and aleatoric uncertainties. We demonstrated the quality of the uncertainty estimates in a variety of Bayesian optimization experiments, in both synthetic and real-world healthcare datasets, and achieve significant improvement over current SOTA in terms of convergence speed. However, we emphasize that BAYESIMP is not designed to replace CBO but rather an alternative model for interventional effects.

In the future, we would like to improve BAYESIMP over several limitations. As in [15], we assumed full knowledge of the underlying causal graph, which might be limiting in practice. Furthermore, as the current formulation of BAYESIMP only allows combination of two causal graphs, we hope to generalise the algorithm into arbitrary number of graphs in the future. Causal graphs with recurrent structure will be an interesting direction to explore.

# 6 Acknowledgements

The authors would like to thank Bobby He, Robert Hu, Kaspar Martens and Jake Fawkes for helpful comments. SLC and JFT are supported by the EPSRC and MRC through the OxWaSP CDT programme EP/L016710/1. YWT and DS are supported in part by Tencent AI Lab and DS is supported in part by the Alan Turing Institute (EP/N510129/1). YWT’s research leading to these results has received funding from the European Research Council under the European Union’s Seventh Framework Programme (FP7/2007-2013) ERC grant agreement no. 617071.

# References

[1] Clay Thompson. Causal graph analysis with the causalgraph procedure. In Proceedings of SAS Global Forum, 2019.   
[2] Travis A Courtney, Mario Lebrato, Nicholas R Bates, Andrew Collins, Samantha J De Putron, Rebecca Garley, Rod Johnson, Juan-Carlos Molinero, Timothy J Noyes, Christopher L Sabine, et al. Environmental controls on modern scleractinian coral and reef-scale calcification. Science advances, 3(11):e1701356, 2017.   
[3] Virginia Aglietti, Theodoros Damoulas, Mauricio Álvarez, and Javier González. Multi-task causal learning with gaussian processes. Advances in Neural Information Processing Systems, 33, 2020.   
[4] Tong Meng, Xuyang Jing, Zheng Yan, and Witold Pedrycz. A survey on machine learning for data fusion. Information Fusion, 57:115–129, 2020.   
[5] Rahul Singh, Maneesh Sahani, and Arthur Gretton. Kernel instrumental variable regression. arXiv preprint arXiv:1906.00232, 2019.   
[6] Ana Ferro, Francisco Pina, Milton Severo, Pedro Dias, Francisco Botelho, and Nuno Lunet. Use of statins and serum levels of prostate specific antigen. Acta Urológica Portuguesa, 32(2):71–77, 2015.   
[7] Thomas A Stamey, John N Kabalin, John E McNeal, Iain M Johnstone, Fuad Freiha, Elise A Redwine, and Norman Yang. Prostate specific antigen in the diagnosis and treatment of adenocarcinoma of the prostate. ii. radical prostatectomy treated patients. The Journal of urology, 141(5):1076–1083, 1989.   
[8] Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Machine Learning, 110(3):457–506, 2021.   
[9] C Rasmussen and C Williams. Gaussian Processes for Machine Learning, 2005.   
[10] Jean-Francois Ton, Lucian Chan, Yee Whye Teh, and Dino Sejdinovic. Noise contrastive meta-learning for conditional density estimation using kernel mean embeddings. In Arindam

Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 1099–1107. PMLR, 13–15 Apr 2021.   
[11] Krikamol Muandet, Kenji Fukumizu, Bharath Sriperumbudur, and Bernhard Schölkopf. Kernel mean embedding of distributions: A review and beyond. Foundations and Trends® in Machine Learning, 10(1-2):1–141, 2017.   
[12] Kenji Fukumizu, Le Song, and Arthur Gretton. Kernel bayes’ rule. arXiv preprint arXiv:1009.5736, 2010.   
[13] Qinyi Zhang, Sarah Filippi, Arthur Gretton, and Dino Sejdinovic. Large-scale kernel methods for independence testing. Statistics and Computing, 28(1):113–130, 2018.   
[14] Ho Chung Leon Law, Dougal Sutherland, Dino Sejdinovic, and Seth Flaxman. Bayesian approaches to distribution regression. In International Conference on Artificial Intelligence and Statistics, pages 1167–1176. PMLR, 2018.   
[15] Virginia Aglietti, Xiaoyu Lu, Andrei Paleyes, and Javier González. Causal bayesian optimization. In International Conference on Artificial Intelligence and Statistics, pages 3155–3164. PMLR, 2020.   
[16] Elias Bareinboim and Judea Pearl. Causal inference and the data-fusion problem. Proceedings of the National Academy of Sciences, 113(27):7345–7352, 2016.   
[17] Rahul Singh, Liyuan Xu, and Arthur Gretton. Kernel methods for policy evaluation: Treatment effects, mediation analysis, and off-policy planning. arXiv preprint arXiv:2010.04855, 2020.   
[18] Krikamol Muandet, Motonobu Kanagawa, Sorawit Saengkyongam, and Sanparith Marukatat. Counterfactual mean embeddings. arXiv preprint arXiv:1805.08845, 2018.   
[19] Jovana Mitrovic, Dino Sejdinovic, and Yee Whye Teh. Causal inference via kernel deviance measures. arXiv preprint arXiv:1804.04622, 2018.   
[20] Judea Pearl. Causal diagrams for empirical research. Biometrika, 82(4):669–688, 1995.   
[21] Le Song, Kenji Fukumizu, and Arthur Gretton. Kernel embeddings of conditional distributions: A unified kernel framework for nonparametric inference in graphical models. IEEE Signal Processing Magazine, 30(4):98–111, 2013.   
[22] Steffen Grünewälder, Guy Lever, Luca Baldassarre, Sam Patterson, Arthur Gretton, and Massimilano Pontil. Conditional mean embeddings as regressors-supplementary. arXiv preprint arXiv:1205.4656, 2012.   
[23] François-Xavier Briol, Chris J Oates, Mark Girolami, Michael A Osborne, Dino Sejdinovic, et al. Probabilistic integration: A role in statistical computation? Statistical Science, 34(1):1–22, 2019.   
[24] Siu Lun Chau, Shahine Bouabid, and Dino Sejdinovic. Deconditional downscaling with gaussian processes. arXiv preprint arXiv:2105.12909, 2021.   
[25] Milan Lukic and Jay Beder. Stochastic processes with sample paths in reproducing kernel ´ hilbert spaces. Transactions of the American Mathematical Society, 353(10):3945–3969, 2001.   
[26] Seth Flaxman, Dino Sejdinovic, John P Cunningham, and Sarah Filippi. Bayesian learning of kernel embeddings. arXiv preprint arXiv:1603.02160, 2016.   
[27] Kelvin Hsu, Richard Nock, and Fabio Ramos. Hyperparameter learning for conditional kernel mean embeddings with rademacher complexity bounds. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 227–242. Springer, 2018.   
[28] Giancarlo Ferrari Trecate, Christopher KI Williams, and Manfred Opper. Finite-dimensional approximation of gaussian processes. In Proceedings of the 1998 conference on Advances in neural information processing systems II, pages 218–224, 1999.

# Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes]   
(b) Did you describe the limitations of your work? [Yes] See section Discussion   
(c) Did you discuss any potential negative societal impacts of your work? [Yes] See below   
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] See Appendix   
(b) Did you include complete proofs of all theoretical results? [Yes] See Appendix

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] See Supp material   
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] See Appendix   
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] See Experiment section   
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] See Appendix

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes] See Experiment section and Appendix   
(b) Did you mention the license of the assets? [N/A]   
(c) Did you include any new assets either in the supplemental material or as a URL? [N/A]   
(d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [N/A]   
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]   
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]   
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

# Societal Impact

In this paper we propose a general framework for embedding interventional distributions into a RKHS while accounting for uncertainty in causal datasets. We believe that this is a crucial problem that has not gotten much attention yet, but is nonetheless important for the future of causal inference research. In particular, given that our proposed method allows us to combine datasets from different studies, we envision that this could potentially be used in a variety of scientific areas such as healthcare, drug discovery etc. Finally, the only potential negative impact, would be, when using biased data. Our method relies on knowing the correct causal graph and hence could be misinterpreted when this is not the case.

# A Additional background on backdoor/front-door adjustments

In causal inference, we are often times interested in the interventional distributions i.e $p ( \boldsymbol { y } | d o ( \boldsymbol { x } ) )$ rather than $p ( y | x )$ , as the former allows us to account for confounding effects. In order to obtain the interventional density $p ( \boldsymbol { y } | d o ( \boldsymbol { x } ) )$ , we resort to do-calculus [20]. Here below we write out the definition for the 2 most crucial formulaes; the front-door and backdoor adjustments, with which we are able to recover the interventional density using only the conditional ones.

# A.1 Back-door Adjustment

The key intuition of back-door adjustments is to find/adjust a set of confounders that are unaffected by the treatment. We can then study the effect of the treatment has to the target.

Definition 1 (Back-Door). A set of variables $Z$ satisfies the backdoor criterion relative to an ordered pair of variables $X _ { i } , X _ { j }$ in a DAG G if:

1. no node in $Z$ is a descendant of $X _ { i }$ ; and   
2. Z blocks every path between $X _ { i }$ and $X _ { j }$ that contains an arrow into $X _ { i }$

Similarly, if $X$ and $Y$ are two disjoint subsets of nodes in $G$ , then $Z$ is said satisfy the back-door criterion relative to $( X , Y )$ if it satisfies the criterion relative to any pair $( X _ { i } , X _ { j } )$ such that $X _ { i } \in X$ and $X _ { j } \in Y$

Now with a given set $Z$ that satisfies the back-door criterion, we apply the backdoor adjustment,

Theorem 1 (Back-Door Adjustment). If a set of variables $Z$ satisfies the back-door criterion relative to $( X , Y )$ , then the causal effect of $X$ on $Y$ is identifiable and is given by the formula

$$
P (y | d o (X) = x) = \int_ {z} p (y | x, z) p (z) d z \tag {18}
$$

# A.2 Front-door Adjustment

Front-door adjustment deals with the case where confounders are unobserved and hence the backdoor adjustment is not applicable.

Definition 2 (Front-door). A set of variables $Z$ is said to satisfy the front-door criterion relative to an ordered pair of variables $( X , Y )$ if:

1. Z intercepts all directed paths from $X$ to $Y$ ;   
2. there is no back-door path from $X$ to $Z$ ; and   
3. all back-door paths from $Z$ to $Y$ are blocked by $X$

Again, with an appropriate front-door adjustment set $Z$ , we can identify the do density using the front-door adjustment formula.

Theorem 2 (Front-Door Adjustment). If $Z$ satisfies the front-door criterion relative to $( X , Y )$ and if $P ( x , z ) > 0$ , then the causal effect of $X$ on $Y$ is identifiable and is given by the formula:

$$
p (y | d o (X) = x) = \int_ {z} p (z | x) \int_ {x ^ {\prime}} p (y | x ^ {\prime}, z) p \left(x ^ {\prime}\right) d x ^ {\prime} d z \tag {19}
$$

# B Derivations

# B.1 CMP Derivation

Proposition 2. Given dataset $D _ { 1 } = \{ ( x _ { i } , y _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N }$ and ${ \cal D } _ { 2 } = \{ ( \tilde { y } _ { j } , t _ { j } ) \} _ { j = 1 } ^ { M }$ , if $f$ is the posterior GP learnt from $\mathcal { D } _ { 2 }$ , then $\begin{array} { r } { g = \int f ( y ) p ( y | d o ( X ) ) d y } \end{array}$ is a GP $\mathcal { G P } ( m _ { 1 } , \kappa _ { 1 } )$ defined on the treatment variable $X$ with the following mean and covariance estimated using ${ \hat { \mu } } _ { Y \mid d o ( X ) }$ ,

$$
\begin{array}{l} m _ {1} (x) = \left\langle \hat {\mu} _ {Y | d o (x)}, m _ {f} \right\rangle_ {\mathcal {H} _ {k _ {y}}} = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y} \tilde {\mathbf {y}}} \left(K _ {\bar {\mathbf {y}} \bar {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} \mathbf {t} (20) \\ \kappa_ {1} (x, x ^ {\prime}) = \hat {\mu} _ {Y | d o (x)} ^ {\top} \hat {\mu} _ {Y | d o \left(x ^ {\prime}\right)} - \hat {\mu} _ {Y | d o (x)} ^ {\top} \Phi_ {\tilde {\mathbf {y}}} \left(K _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda I\right) ^ {- 1} \Phi_ {\tilde {\mathbf {y}}} ^ {\top} \hat {\mu} _ {Y | d o \left(x ^ {\prime}\right)} (21) \\ = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \tilde {K} _ {\mathbf {y y}} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) (22) \\ \end{array}
$$

where $\hat { \mu } _ { Y | d o ( x ) } = \hat { \mu } _ { Y | d o ( X ) = x } , K _ { \tilde { \mathbf { y } } \mathbf { y } } = \Phi _ { \tilde { \mathbf { y } } } ^ { \top } \Phi _ { \mathbf { y } }$ , $m _ { f }$ and $\tilde { K } _ { \mathbf { y } \mathbf { y } }$ are the posterior mean function and covariance of $f$ evaluated at y respectively. $\lambda > 0$ is the regularisation of the CME. $\lambda _ { f } > 0$ is the noise term for GP $f$ . $\Omega _ { x }$ is the set of variables as specified in Prop.1.

Proof for Proposition 2. Integral operator preserves Gaussianity under mild conditions (see conditions [24]), therefore

$$
g (x) = \int f (y) d P (y | d o (X) = x) \tag {23}
$$

is also a Gaussian. For a standard GP prior $f \sim G P ( 0 , k _ { y } )$ and data ${ D } _ { E } = \{ ( \tilde { y } _ { j } , t _ { j } ) \} _ { j = 1 } ^ { M }$ , standard conjugacy results for GPs lead to the posterior GP with mean $\bar { m } ( y ) = k _ { y \tilde { \bf y } } ( K _ { \tilde { \bf y } \tilde { \bf y } } + \lambda _ { f } I ) ^ { - 1 } { \bf t }$ and covariance $\bar { k _ { y } } ( y , y ^ { \prime } ) = k _ { y } ( y , y ^ { \prime } ) - k _ { y \tilde { \bf y } } ( K _ { \tilde { \bf y } \tilde { \bf y } } + \lambda _ { f } I ) ^ { - 1 } k _ { \tilde { \bf y } y }$ . Similar to [23], repeated application of Fubini’s theorem yields:

$$
\begin{array}{l} \mathbb {E} _ {f} [ g (x) ] = \mathbb {E} _ {f} \left[ \int f (y) d P (y | d o (X) = x \right] = \int \mathbb {E} _ {f} [ f (y) ] d P (y | d o (X) = x) (24) \\ = \int \bar {m} (y) d P (y | d o (X) = x) = \langle \bar {m}, \hat {\mu} _ {Y \mid d o (X) = x} \rangle (25) \\ \end{array}
$$

$$
\begin{array}{l} c o v (g (x), g \left(x ^ {\prime}\right)) = \int \int c o v (f (y), f \left(y ^ {\prime}\right)) d P (y | d o (X) = x) d P \left(y ^ {\prime} \mid d o (X) = x\right) (26) \\ = \iint \bar {k} _ {y} \left(y, y ^ {\prime}\right) d P \left(y \mid d o (x)\right) d P \left(y ^ {\prime} \mid d o \left(x ^ {\prime}\right)\right) (27) \\ = \left\langle \mu_ {Y \mid d o (x)}, \mu_ {Y \mid d o \left(x ^ {\prime}\right)} \right\rangle - \hat {\mu} _ {Y \mid d o (x)} ^ {\top} \Phi_ {\tilde {\mathbf {y}}} \left(K _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda I\right) ^ {- 1} \Phi_ {\tilde {\mathbf {y}}} ^ {\top} \hat {\mu} _ {Y \mid d o \left(x ^ {\prime}\right)} (28) \\ = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \tilde {K} _ {\mathbf {y y}} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) (29) \\ \end{array}
$$

# B.2 Choice of Nuclear Dominant Kernel

Recall in section 3.2, we introduced the nuclear dominant kernel $r _ { y }$ to ensure samples of $\mu _ { g p } \sim$ $G P ( 0 , k _ { x } \otimes r _ { y } )$ are supported in $\mathcal { H } _ { k _ { x } } \otimes \mathcal { H } _ { k _ { y } }$ with probability 1. In the following we will present the analytic form of the nuclear dominant kernel we used in this paper, which is the same as the formulation introduced in Appendix A.2 and A.3 of [26]. Pick $k _ { y }$ as the RBF kernel, i.e

$$
k _ {y} \left(y, y ^ {\prime}\right) = \exp \left(- \frac {1}{2} \left(y - y ^ {\prime}\right) ^ {\top} \Sigma_ {\theta} \left(y - y ^ {\prime}\right)\right) \tag {30}
$$

where $\Sigma _ { \theta }$ is covariance matrix for the kernel $k _ { y }$ . The nuclear dominant kernel construction from [26] then yield the following expression:

$$
r _ {y} \left(y, y ^ {\prime}\right) = \int k _ {y} (y, u) k _ {y} \left(u, y ^ {\prime}\right) \nu (d u) \tag {31}
$$

where $\nu$ is some finite measure. If we pick $\begin{array} { r } { \nu ( d u ) = \exp ( \frac { | | u | | _ { 2 } ^ { 2 } } { 2 \eta ^ { 2 } } ) d u } \end{array}$ , then we have

$$
\begin{array}{l} r _ {y} \left(y, y ^ {\prime}\right) = \left(2 \pi\right) ^ {D / 2} \left| 2 \Sigma_ {\theta} ^ {- 1} + \eta^ {- 2} I \right| ^ {- 1 / 2} \exp \left(- \frac {1}{2} \left(y - y ^ {\prime}\right) ^ {\top} \left(2 \Sigma_ {\theta}\right) ^ {- 1} \left(y - y ^ {\prime}\right)\right) (32) \\ \times \exp \left(- \frac {1}{2} \left(\frac {y + y ^ {\prime}}{2}\right) ^ {\top} \left(\frac {1}{2} \Sigma_ {\theta} + \eta^ {2} I\right) ^ {- 1} \left(\frac {y + y ^ {\prime}}{2}\right)\right) (33) \\ \end{array}
$$

# B.3 BayesCME derivations

Proposition 3. The posterior GP of $\mu _ { g p }$ given observations $\{ \mathbf { x } , \mathbf { y } \}$ has the following mean and covariance:

$$
m _ {\mu} ((x, y)) = k _ {x \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y y}} \tag {34}
$$

$$
\kappa_ {\mu} \left(\left(x, y\right), \left(x ^ {\prime}, y ^ {\prime}\right)\right) = k _ {x x ^ {\prime}} r _ {y, y ^ {\prime}} - k _ {x \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} k _ {\mathbf {x} x ^ {\prime}} r _ {y \mathbf {y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y} y ^ {\prime}} \tag {35}
$$

In addition, the following marginal likelihood can be used for hyperparameter optimisation,

$$
- \frac {N}{2} \left(\log | K _ {\mathbf {x x}} + \lambda I | + \log | R |\right) - \frac {1}{2} \operatorname {t r} \left((K _ {\mathbf {x x}} + \lambda I) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} K _ {\mathbf {y y}}\right) \tag {36}
$$

Proof of Proposition 3. Recall the Bayesian formulation of CME corresponds to the following model,

$$
\mu_ {g p} \sim G P (0, k _ {x} \otimes r _ {y}),
$$

$$
k _ {y} \left(y _ {i}, y ^ {\prime}\right) = \mu_ {g p} \left(x _ {i}, y ^ {\prime}\right) + \lambda^ {1 / 2} \epsilon_ {i} \left(y ^ {\prime}\right)
$$

with $\epsilon _ { i } \sim G P ( 0 , r _ { y } )$ independently across $i$ . Now consider $k _ { y } ( y _ { i } , y _ { j } )$ as noisy evaluations of $\mu _ { g p } ( x _ { i } , y _ { j } )$ , we have the predictive posterior mean as

$$
\begin{array}{l} \operatorname {v e c} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}}\right) ^ {\top} \left(K _ {\mathbf {x x}} \otimes R _ {\mathbf {y y}} + \lambda I \otimes R _ {\mathbf {y y}}\right) ^ {- 1} \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) = \operatorname {v e c} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}}\right) ^ {\top} \left(\left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} \otimes R _ {\mathbf {y y}} ^ {- 1}\right) \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) \\ = \operatorname {v e c} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}}\right) ^ {\top} \operatorname {v e c} \left(R _ {\mathbf {y} \mathbf {y}} ^ {- 1} K _ {\mathbf {y} \mathbf {y}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1}\right) \\ = \operatorname {t r} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1}\right) \\ = k _ {x \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y y}}. \\ \end{array}
$$

And the covariance is,

$$
\begin{array}{l} \kappa \left(\left(x, y\right), \left(x ^ {\prime}, y ^ {\prime}\right)\right) = k \left(x, x ^ {\prime}\right) r \left(y, y ^ {\prime}\right) - \operatorname {v e c} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}}\right) ^ {\top} \left(K _ {\mathbf {x x}} \otimes R _ {\mathbf {y y}} + \lambda I \otimes R _ {\mathbf {y y}}\right) ^ {- 1} \operatorname {v e c} \left(r _ {\mathbf {y} y ^ {\prime}} k _ {x ^ {\prime} \mathbf {x}}\right) \\ = k (x, x ^ {\prime}) r (y, y ^ {\prime}) - \operatorname {v e c} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}}\right) ^ {\top} \left(\left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} \otimes R _ {\mathbf {y y}} ^ {- 1}\right) \operatorname {v e c} \left(r _ {\mathbf {y} y ^ {\prime}} k _ {x ^ {\prime} \mathbf {x}}\right) \\ = k (x, x ^ {\prime}) r (y, y ^ {\prime}) - \operatorname {v e c} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}}\right) ^ {\top} \operatorname {v e c} \left(R _ {\mathbf {y} \mathbf {y}} ^ {- 1} r _ {\mathbf {y} y ^ {\prime}} k _ {x ^ {\prime} \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1}\right) \\ = k (x, x ^ {\prime}) r (y, y ^ {\prime}) - \operatorname {t r} \left(r _ {\mathbf {y} y} k _ {x \mathbf {x}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} k _ {\mathbf {x} x ^ {\prime}} r _ {y ^ {\prime} \mathbf {y}} R _ {\mathbf {y y}} ^ {- 1}\right) \\ = k (x, x ^ {\prime}) r \left(y, y ^ {\prime}\right) - k _ {x \mathbf {x}} \left(K _ {\mathbf {x} \mathbf {x}} + \lambda I\right) ^ {- 1} k _ {\mathbf {x} x ^ {\prime}} r _ {y ^ {\prime} \mathbf {y}} R _ {\mathbf {y} \mathbf {y}} ^ {- 1} r _ {\mathbf {y} y}. \\ \end{array}
$$

To compute the log likelihood, note that it contains the following two terms:

$$
\begin{array}{l} \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) ^ {\top} \left(K _ {\mathbf {x x}} \otimes R _ {\mathbf {y y}} + \lambda I \otimes R _ {\mathbf {y y}}\right) ^ {- 1} \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) = \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) ^ {\top} \left(\left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} \otimes R _ {\mathbf {y y}} ^ {- 1}\right) \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) \\ = \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) ^ {\top} \operatorname {v e c} \left(R _ {\mathbf {y y}} ^ {- 1} K _ {\mathbf {y y}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1}\right) \\ = \operatorname {t r} \left(K _ {\mathbf {y y}} \left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1}\right) \\ \end{array}
$$

and

$$
\begin{array}{l} - \frac {1}{2} \left(\log | (K _ {\mathbf {x x}} + \lambda I) \otimes R _ {\mathbf {y y}} |\right) = - \frac {1}{2} \log \left(| (K _ {\mathbf {x x}} + \lambda I) | ^ {N} | R | ^ {N}\right) \\ = - \frac {N}{2} \left(\log | K _ {\mathbf {x x}} + \lambda I | + \log | R |\right) \\ \end{array}
$$

where we used the fact that determinant of Kronecker product of two $N \times N$ matrices $A , B$ is: $| A \otimes B | = | A | ^ { N } | B | ^ { N }$ .

Therefore the log likelihood can be expressed as

$$
- \frac {N}{2} \left(\log | K _ {\mathbf {x x}} + \lambda I | + \log | R |\right) - \frac {1}{2} \operatorname {t r} \left(\left(K _ {\mathbf {x x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} K _ {\mathbf {y y}}\right) \tag {37}
$$

# B.4 Causal BayesCME derivations

The following proposition extend BAYESCME to the causal setting.

Proposition C.1 (Causal BayesCME). Denote $\mu _ { g p } ^ { d o }$ as the GP modelling $\mu _ { Y \mid d o ( X ) }$ . Then using the $\Omega$ notations introduced in proposition 1, the posterior GP of $\mu _ { g p } ^ { d o }$ given observations $\{ \mathbf { x } , \mathbf { z } , \mathbf { y } \}$ has the following mean and covariance:

$$
m _ {\mu} ^ {d o} ((x, y)) = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y y}} \tag {38}
$$

$$
\kappa_ {\mu} ^ {d o} \left(\left(x, y\right), \left(x ^ {\prime}, y ^ {\prime}\right)\right) = \Phi_ {\Omega_ {x}} (x) ^ {\top} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) r _ {y, y ^ {\prime}} - \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) r _ {y y} R _ {y y} ^ {- 1} r _ {y y ^ {\prime}} \tag {39}
$$

In addition, the following marginal likelihood can be used for hyperparameter optimisation,

$$
- \frac {N}{2} \left(\log | K _ {\Omega_ {x}} + \lambda I | + \log | R |\right) - \frac {1}{2} \operatorname {t r} \left((K _ {\Omega_ {x}} + \lambda I) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} K _ {\mathbf {y y}}\right) \tag {40}
$$

Proof of Proposition C.1. In the following we will assume $Z$ is the backdoor adjustment variable. Front-door and general cases follow analogously. Denote $\mu _ { g p } ( ( x , z ) , y )$ as the BAYESCME model for $\scriptstyle \mu _ { Y \mid X = x , Z = z } ( y )$ . As we have

$$
\begin{array}{l} \mu_ {Y \mid d o (X) = x} = \iint \phi_ {y} (y) p (y | x, z) p (z) d z d y (41) \\ = \int \mu_ {Y \mid X = x, Z = z} p (z) d z (42) \\ = \mathbb {E} _ {Z} \left[ \mu_ {Y \mid X = x, Z} \right] (43) \\ \end{array}
$$

It is thus natural to define $\mu _ { g p } ^ { d o }$ as the induced GP when we replace $\scriptstyle \mu _ { Y \mid X = x , Z = z }$ with $\mu _ { g p } ( ( x , z ) , \cdot )$ ,

$$
\mu_ {g p} ^ {d o} (x, \cdot) = \mathbb {E} _ {Z} [ \mu_ {g p} ((x, Z), \cdot) ] \tag {44}
$$

Now we can compute the mean of $\mu _ { g p } ^ { d o }$

$$
\begin{array}{l} m _ {\mu} ^ {d o} (x, y) = \mathbb {E} _ {\mu_ {g p}} \mathbb {E} _ {Z} [ \mu_ {g p} (x, Z, y) ] (45) \\ = \mathbb {E} _ {Z} \left(\left(k _ {x x} \odot k _ {z} (Z, \mathbf {z})\right) \left(K _ {\mathbf {x x}} \odot K _ {\mathbf {z z}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y} y}\right) (46) \\ = \left(\left(k _ {x x} \odot \mu_ {z} ^ {\top} \Phi_ {z}\right) \left(K _ {x x} \odot K _ {z z} + \lambda I\right) ^ {- 1} K _ {y y} R _ {y y} ^ {- 1} r _ {y y}\right) (47) \\ = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} r _ {\mathbf {y y}} (48) \\ \end{array}
$$

Similarly for covariance, we have,

$$
\kappa_ {\mu} ^ {d o} \left(\left(x, y\right), \left(x ^ {\prime}, y ^ {\prime}\right)\right) = \mathbb {E} _ {Z, Z ^ {\prime}} \left[ c o v \left(\mu_ {g p} \left(\left(x, Z\right), y\right), \mu_ {g p} \left(\left(x ^ {\prime}, Z ^ {\prime}\right), y ^ {\prime}\right)\right) \right] \tag {49}
$$

and the rest is just algebra,

$$
= \Phi_ {\Omega_ {x}} (x) ^ {\top} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) r _ {y, y ^ {\prime}} - \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) r _ {y y} R _ {y y} ^ {- 1} r _ {y y ^ {\prime}} \tag {50}
$$

# B.5 BayesIME derivation

Now we have derived the Causal BAYESCME, it is time to compute $\langle f , \mu _ { g p } ^ { d o } ( x , \cdot ) \rangle$ where $f \in \mathcal { H } _ { k _ { y } }$ This requires us to be able to compute $\langle f , r _ { y } ( \cdot , y ) \rangle$ which corresponds to the following:

$$
\begin{array}{l} \langle f, r _ {y} (\cdot , y) \rangle_ {\mathcal {H} _ {k _ {y}}} = \left\langle f, \int k _ {y} (\cdot , u) k _ {y} (u, y) \nu (d u) \right\rangle (51) \\ = \int f (u) k _ {y} (u, y) \nu (d u) (52) \\ \end{array}
$$

when $f$ is a KRR learnt from $\mathcal { D } _ { 2 }$ , i.e $f ( y ) = k _ { y \tilde { \mathbf { y } } } ( K _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } + \lambda _ { f } I ) ^ { - 1 } \mathbf { t }$ , we have

$$
= \mathbf {t} ^ {\top} \left(K _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} \int k _ {\tilde {\mathbf {y}} u} k _ {y} (u, y) \nu (d u) \tag {53}
$$

$$
= \mathbf {t} ^ {\top} \left(K _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} r _ {\tilde {\mathbf {y}} y} \tag {54}
$$

Now we are ready to derive BAYESIME.

Proposition 4. Given dataset $D _ { 1 } = \{ ( x _ { i } , y _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N }$ and ${ \cal D } _ { 2 } = \{ ( \tilde { y } _ { j } , t _ { j } ) \} _ { j = 1 } ^ { M }$ , if $f$ is a KRR learnt from $\mathcal { D } _ { 2 }$ and $\mu _ { Y \mid d o ( X ) }$ modelled as a V-GP using $\mathcal { D } _ { 1 }$ , then $g = \langle f , \mu _ { Y | d o ( X ) } \rangle \sim \mathcal { G P } ( m _ { 2 } , \kappa _ { 2 } )$ where,

$$
m _ {2} (x) = \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} R _ {\mathbf {y} \tilde {\mathbf {y}}} A \tag {55}
$$

$$
\kappa_ {2} (x, x ^ {\prime}) = B \Phi_ {\Omega_ {x}} (x) ^ {\top} \Phi_ {\Omega_ {x}} (x) - C \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} \left(x ^ {\prime}\right) \tag {56}
$$

where $A = ( K _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } + \lambda _ { f } I ) ^ { - 1 } \mathbf { t }$ , $B = A ^ { \top } R _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } A$ and $C = A ^ { \top } R _ { \tilde { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } R _ { \mathbf { y } \tilde { \mathbf { y } } } A$

Proof of Proposition 4. Using the $\mu _ { g p } ^ { d o }$ notation from Proposition $C . 1$ , we can write the inner product as $\langle \mu _ { g p } ^ { d o } ( x , \cdot ) , f \rangle$ , where the mean is,

$$
m _ {2} (x) = \mathbb {E} \left[ \mu_ {g p} ^ {d o} (x, \cdot) \right] ^ {\top} f \tag {57}
$$

$$
= \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} R (\mathbf {y}, \cdot) ^ {\top} f \tag {58}
$$

$$
= \Phi_ {\Omega_ {x}} (x) ^ {\top} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} K _ {\mathbf {y y}} R _ {\mathbf {y y}} ^ {- 1} R _ {\mathbf {y} \tilde {\mathbf {y}}} \left(K _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} \mathbf {t} \tag {59}
$$

where we used the fact $f$ is a KRR learnt from $\mathcal { D } _ { 2 }$ . The covariance can then be computed by realising $c o v ( f ^ { \top } \mu _ { g p } ^ { d o } ( x , \cdot ) , f ^ { \top } \mu _ { g p } ^ { d o } ( x ^ { \prime } , \cdot ) ) = f ^ { \top } c o v ( \mu _ { g p } ^ { d o } ( x , \cdot ) , \mu _ { g p } ^ { d o } ( x ^ { \prime } , \cdot ) ) f .$

# B.6 BayesIMP Derivations

BAYESIMP can be understood as a model characterising the RKHS inner product of Gaussian Processes. In the following, we will first introduce some general theory of inner product of GPs, and introduce a finite dimensional scheme later on. Finally, we will show how BAYESIMP can be derived right away from this general framework.

Before that, we will showcase the following identity for computing variance of inner products of independent multivariate Gaussians,

Proposition C.2. Let $\mu _ { X } : = \mathbb { E } [ X ]$ and $\Sigma _ { X } : = V a r ( X )$ be the mean and variance of a multivariate Gaussian rv, similarly $\mu _ { Y } , \Sigma _ { Y }$ for Gaussian $r \nu Y$ . If $X$ and $Y$ are independent, then the variance of their inner product is given by the following expression,

$$
V a r \left(X ^ {\top} Y\right) = \mu_ {X} ^ {\top} \Sigma_ {Y} \mu_ {X} + \mu_ {Y} ^ {\top} \Sigma_ {X} \mu_ {Y} + t r \left(\Sigma_ {Y} \Sigma_ {X}\right) \tag {60}
$$

Moreover, the covariance between $X ^ { \top } Y _ { 1 }$ , $X ^ { \top } Y _ { 2 }$ follows a similar form,

$$
c o v \left(X ^ {\top} Y _ {1}, X ^ {\top} Y _ {2}\right) = \mu_ {X} ^ {\top} \Sigma_ {Y _ {1} Y _ {2}} \mu_ {X} + \mu_ {Y _ {1}} ^ {\top} \Sigma_ {X} \mu_ {Y _ {2}} + \operatorname {t r} \left(\Sigma_ {X} \Sigma_ {Y _ {1} Y _ {2}}\right) \tag {61}
$$

Proof.

$$
\begin{array}{l} \operatorname {V a r} \left[ X ^ {\top} Y \right] = \mathbb {E} \left[ \left(X ^ {\top} Y\right) ^ {2} \right] - \mathbb {E} \left[ X ^ {\top} Y \right] ^ {2} \\ = \mathbb {E} \left[ X ^ {\top} Y Y ^ {\top} X \right] - \left(\mathbb {E} [ X ] ^ {\top} \mathbb {E} [ Y ]\right) ^ {2} \\ = \mathbb {E} \left[ \operatorname {t r} \left(X X ^ {\top} Y Y ^ {\top}\right) \right] - \left(\mu_ {X} ^ {\top} \mu_ {Y}\right) ^ {2} \\ = \operatorname {t r} \left(\mathbb {E} \left[ X X ^ {\top} \right] \mathbb {E} \left[ Y Y ^ {\top} \right]\right) - \left(\mu_ {X} ^ {\top} \mu_ {Y}\right) ^ {2} \\ = \operatorname {t r} \left(\left(\mu_ {X} \mu_ {X} ^ {\top} + \Sigma_ {X}\right) \left(\mu_ {Y} \mu_ {Y} ^ {\top} + \Sigma_ {Y}\right)\right) - \left(\mu_ {X} ^ {\top} \mu_ {Y}\right) ^ {2} \\ = \operatorname {t r} \left(\mu_ {X} \mu_ {X} ^ {\top} \mu_ {Y} \mu_ {Y} ^ {\top}\right) + \operatorname {t r} \left(\mu_ {X} \mu_ {X} ^ {\top} \Sigma_ {Y}\right) + \operatorname {t r} \left(\Sigma_ {X} \mu_ {Y} \mu_ {Y} ^ {\top}\right) + \operatorname {t r} \left(\Sigma_ {X} \Sigma_ {Y}\right) - \left(\mu_ {X} ^ {\top} \mu_ {Y}\right) ^ {2} \\ = \left(\mu_ {X} ^ {\top} \mu_ {Y}\right) ^ {2} + \operatorname {t r} \left(\mu_ {X} ^ {\top} \Sigma_ {Y} \mu_ {X}\right) + \operatorname {t r} \left(\mu_ {Y} ^ {\top} \Sigma_ {X} \mu_ {Y}\right) + \operatorname {t r} \left(\Sigma_ {X} \Sigma_ {Y}\right) - \left(\mu_ {X} ^ {\top} \mu_ {Y}\right) ^ {2} \\ = \mu_ {X} ^ {\top} \Sigma_ {Y} \mu_ {X} + \mu_ {Y} ^ {\top} \Sigma_ {X} \mu_ {Y} + \operatorname {t r} \left(\Sigma_ {X} \Sigma_ {Y}\right) \tag {62} \\ \end{array}
$$

Generalising to the case for covariance is straight forward.

# RKHS inner product of Gaussian Processes

Let $f _ { 1 } \sim G P ( m _ { 1 } , \kappa _ { 1 } )$ and $f _ { 2 } \sim G P ( m _ { 2 } , \kappa _ { 2 } )$ . We assume that $f$ and $g$ are both supported within the RKHS $\mathcal { H } _ { k }$ . Can we characterise the distribution of $\langle f _ { 1 } , f _ { 2 } \rangle _ { \mathcal { H } _ { k } }$ ?

This situation would arise if $f _ { 1 }$ and $f _ { 2 }$ arise as GP posteriors in a regression model corresponding to the priors $f _ { 1 } \sim G P ( 0 , r _ { 1 } )$ , $f _ { 2 } \sim G P ( 0 , r _ { 2 } )$ where $r _ { 1 } , r _ { 2 }$ satisfy the nuclear dominance property. In particular, we could choose

$$
r _ {1} (u, v) = \int k (u, z) k (z, v) \nu_ {1} (d z),
$$

$$
r _ {2} (u, v) = \int k (u, z) k (z, v) \nu_ {2} (d z).
$$

Posterior means in that case can be expanded as

$$
m _ {1} = \sum \alpha_ {i} r _ {1} (\cdot , x _ {i}), \qquad m _ {2} = \sum \beta_ {j} r _ {2} (\cdot , y _ {j}).
$$

We assume that $f _ { 1 }$ and $f _ { 2 }$ are independent, i.e. they correspond to posteriors computed on independent data. Then

$$
\begin{array}{l} \mathbb {E} \left\langle f _ {1}, f _ {2} \right\rangle_ {\mathcal {H} _ {k}} = \left\langle m _ {1}, m _ {2} \right\rangle_ {\mathcal {H} _ {k}} \\ = \left\langle \sum \alpha_ {i} r _ {1} (\cdot , x _ {i}), \sum \beta_ {j} r _ {2} (\cdot , y _ {j}) \right\rangle_ {\mathcal {H} _ {k}} \\ = \alpha^ {\top} Q \beta , \\ \end{array}
$$

where

$$
\begin{array}{l} Q _ {i j} = q \left(x _ {i}, y _ {j}\right) := \langle r _ {1} (\cdot , x _ {i}), r _ {2} (\cdot , y _ {j}) \rangle_ {\mathcal {H} _ {k}} \\ = \left\langle \int k (\cdot , z) k (z, x _ {i}) \nu_ {1} (d z), \int k (\cdot , z ^ {\prime}) k (z ^ {\prime}, y _ {j}) \nu_ {2} (d z ^ {\prime}) \right\rangle_ {\mathcal {H} _ {k}} \\ = \int \int \langle k (\cdot , z), k (\cdot , z ^ {\prime}) \rangle_ {\mathcal {H} _ {k}} k (z, x _ {i}) k (z ^ {\prime}, y _ {j}) \nu_ {1} (d z) \nu_ {2} (d z ^ {\prime}) \\ = \int \int k (z, z ^ {\prime}) k (z, x _ {i}) k (z ^ {\prime}, y _ {j}) \nu_ {1} (d z) \nu_ {2} (d z ^ {\prime}). \\ \end{array}
$$

The variance would be given, in analogy to the finite dimensional case, by

$$
\operatorname {v a r} \left\langle f _ {1}, f _ {2} \right\rangle_ {\mathcal {H} _ {k}} = \left\langle m _ {1}, \Sigma_ {2} m _ {1} \right\rangle_ {\mathcal {H} _ {k}} + \left\langle m _ {2}, \Sigma_ {1} m _ {2} \right\rangle_ {\mathcal {H} _ {k}} + \operatorname {t r} \left(\Sigma_ {1} \Sigma_ {2}\right),
$$

with $\begin{array} { r } { \Sigma _ { 1 } f = \int \kappa _ { 1 } \left( \cdot , u \right) f ( u ) d u } \end{array}$ and similarly for $\Sigma _ { 2 }$ . Thus

$$
\begin{array}{l} \langle m _ {1}, \Sigma_ {2} m _ {1} \rangle_ {\mathcal {H} _ {k}} = \left\langle \sum \alpha_ {i} r _ {1} (\cdot , x _ {i}), \sum \alpha_ {j} \int \kappa_ {2} (\cdot , u) r _ {1} (u, x _ {j}) d u \right\rangle_ {\mathcal {H} _ {k}} \\ = \sum \sum \alpha_ {i} \alpha_ {j} \int \left\langle r _ {1} (\cdot , x _ {i}), \kappa_ {2} (\cdot , u) \right\rangle_ {\mathcal {H} _ {k}} r _ {1} (u, x _ {j}) d u. \\ \end{array}
$$

Now, given that kernel $\kappa _ { 2 }$ depends on $r _ { 2 }$ in a simple way, it should be possible to write down the full expression similarly as for $Q _ { i j }$ above. In particular

$$
\kappa_ {2} (\cdot , u) = r _ {2} (\cdot , u) - r _ {2} (\cdot , \mathbf {y}) \left(R _ {2, \mathbf {y y}} + \sigma_ {2} ^ {2} I\right) ^ {- 1} r _ {2} (\mathbf {y}, u).
$$

Hence

$$
\left\langle r _ {1} \left(\cdot , x _ {i}\right), \kappa_ {2} \left(\cdot , u\right) \right\rangle_ {\mathcal {H} _ {k}} = q \left(x _ {i}, u\right) - q \left(x _ {i}, \mathbf {y}\right) \left(R _ {2, \mathbf {y y}} + \sigma_ {2} ^ {2} I\right) ^ {- 1} r _ {2} \left(\mathbf {y}, u\right).
$$

However, this further requires approximating integrals of the type

$$
\int q (x _ {i}, u) r _ {1} (u, x _ {j}) d u = \int \int \int \int k (z, z ^ {\prime}) k (z, x _ {i}) k (z ^ {\prime}, u) k (u, z ^ {\prime \prime}) k (z ^ {\prime \prime}, x _ {j}) \nu_ {1} (d z) \nu_ {2} (d z ^ {\prime}) \nu_ {1} (d z ^ {\prime \prime}) d u,
$$

etc. Thus, while possible in principle, this approach to compute the variance is cumbersome.

# A finite dimensional approximation

To approximate the variance, hence, it is simpler to consider finite-dimensional approximations to $f _ { 1 }$ and $f _ { 2 }$ . Namely, collate $\{ x _ { i } \}$ and $\{ y _ { j } \}$ into a single set of points $\xi$ (note that we could here take an arbitrary set of points), and consider finite-dimensional GPs given by

$$
\tilde {f} _ {1} = \sum a _ {j} k (\cdot , \xi_ {j}), \quad \tilde {f} _ {2} = \sum b _ {j} k (\cdot , \xi_ {j}),
$$

where we selects distribution of $a$ and $b$ such that evaluations of $\tilde { f } _ { 1 }$ and $\tilde { f } _ { 2 }$ on $\xi$ , $K _ { \xi \xi } a$ and $K _ { \xi \xi } b$ respectively, have the same distributions as evaluations of $f _ { 1 }$ and $f _ { 2 }$ on $\xi$ . In particular, we take

$$
a \sim \mathcal {N} \left(K _ {\xi \xi} ^ {- 1} m _ {1} (\xi), K _ {\xi \xi} ^ {- 1} \mathcal {K} _ {1, \xi \xi} K _ {\xi \xi} ^ {- 1}\right), \quad b \sim \mathcal {N} \left(K _ {\xi \xi} ^ {- 1} m _ {2} (\xi), K _ {\xi \xi} ^ {- 1} \mathcal {K} _ {2, \xi \xi} K _ {\xi \xi} ^ {- 1}\right),
$$

where we denoted by $m _ { 1 } \left( \xi \right)$ a vector such that $[ m _ { 1 } \left( \xi \right) ] _ { i } = m _ { 1 } \left( \xi _ { i } \right)$ and by $\mathcal { K } _ { 1 , \xi \xi }$ a matrix such that $[ \mathcal { K } _ { 1 , \xi \xi } ] _ { i j } = \kappa _ { 1 } ( \xi _ { i } , \bar { \xi _ { j } } )$ .

Then, clearly

$$
\begin{array}{l} \left\langle \tilde {f} _ {1}, \tilde {f} _ {2} \right\rangle_ {\mathcal {H} _ {k}} = a ^ {\top} K _ {\xi \xi} b \\ = \left(K _ {\xi \xi} ^ {1 / 2} a\right) ^ {\top} \left(K _ {\xi \xi} ^ {1 / 2} b\right), \\ \end{array}
$$

and now we are left with the problem of computing the mean and the variance of inner product between two independent Gaussian vectors, as given in Proposition $C . 2$ . We have

$$
\begin{array}{l} \mathbb {E} \left\langle \tilde {f} _ {1}, \tilde {f} _ {2} \right\rangle_ {\mathcal {H} _ {k}} = \left(K _ {\xi \xi} ^ {1 / 2} K _ {\xi \xi} ^ {- 1} m _ {1} (\xi)\right) ^ {\top} \left(K _ {\xi \xi} ^ {1 / 2} K _ {\xi \xi} ^ {- 1} m _ {2} (\xi)\right) \\ = m _ {1} (\xi) ^ {\top} K _ {\xi \xi} ^ {- 1} K _ {\xi \xi} K _ {\xi \xi} ^ {- 1} m _ {2} (\xi) \\ = m _ {1} (\xi) ^ {\top} K _ {\xi \xi} ^ {- 1} m _ {2} (\xi), \\ \end{array}
$$

and

$$
\begin{array}{l} \operatorname {v a r} \left\langle \tilde {f} _ {1}, \tilde {f} _ {2} \right\rangle_ {\mathcal {H} _ {k}} = \left(K _ {\xi \xi} ^ {1 / 2} K _ {\xi \xi} ^ {- 1} m _ {1} (\xi)\right) ^ {\top} K _ {\xi \xi} ^ {- 1 / 2} \mathcal {K} _ {2, \xi \xi} K _ {\xi \xi} ^ {- 1 / 2} \left(K _ {\xi \xi} ^ {1 / 2} K _ {\xi \xi} ^ {- 1} m _ {1} (\xi)\right) \\ + \left(K _ {\xi \xi} ^ {1 / 2} K _ {\xi \xi} ^ {- 1} m _ {2} (\xi)\right) ^ {\top} K _ {\xi \xi} ^ {- 1 / 2} \mathcal {K} _ {1, \xi \xi} K _ {\xi \xi} ^ {- 1 / 2} \left(K _ {\xi \xi} ^ {1 / 2} K _ {\xi \xi} ^ {- 1} m _ {2} (\xi)\right) \\ + \quad \operatorname {t r} \left(K _ {\xi \xi} ^ {- 1 / 2} \mathcal {K} _ {1, \xi \xi} K _ {\xi \xi} ^ {- 1 / 2} K _ {\xi \xi} ^ {- 1 / 2} \mathcal {K} _ {2, \xi \xi} K _ {\xi \xi} ^ {- 1 / 2}\right) \\ = m _ {1} (\xi) ^ {\top} K _ {\xi \xi} ^ {- 1} \mathcal {K} _ {2, \xi \xi} K _ {\xi \xi} ^ {- 1} m _ {1} (\xi) \\ + \quad m _ {2} (\xi) ^ {\top} K _ {\xi \xi} ^ {- 1} \mathcal {K} _ {1, \xi \xi} K _ {\xi \xi} ^ {- 1} m _ {2} (\xi) \\ + \quad \operatorname {t r} \left(\mathcal {K} _ {1, \xi \xi} K _ {\xi \xi} ^ {- 1} \mathcal {K} _ {2, \xi \xi} K _ {\xi \xi} ^ {- 1}\right). \\ \end{array}
$$

# Coming back to BayesIMP

Now coming back to the derivation of BayesIMP. We will first provide two finite approximation of $f$ and $\mu _ { g p } ^ { d o } ( x , \bar { \cdot } )$ in the following two propositions. Recall these finite approximations are set up such that they match the distributions of evaluations of $f$ and $\mu _ { g p } ^ { d o }$ at $\hat { \mathbf { y } } = [ \mathbf { y } ^ { \top } \tilde { \mathbf { y } } ^ { \top } ] ^ { \top }$ . The latter thus act as landmark points for the finite dimensional approximations.

Proposition C.3 (Finite dimensional approximation of $f$ ). Let $\hat { \mathbf { y } } = [ \mathbf { y } ^ { \top } \tilde { \mathbf { y } } ^ { \top } ] ^ { \top }$ be the concatenation of y and ˜y. We can approximate $f$ with ,

$$
\tilde {f} | \mathbf {t} \sim N \left(m _ {\tilde {f}}, \Sigma_ {\tilde {f}}\right) \tag {63}
$$

where,

$$
m _ {\tilde {f}} = \Phi_ {\hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} R _ {\hat {\mathbf {y}} \tilde {\mathbf {y}}} \left(R _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} \mathbf {t} \tag {64}
$$

$$
\Sigma_ {\tilde {f}} = \Phi_ {\hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \bar {R} _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \Phi_ {\hat {\mathbf {y}}} ^ {\top} \tag {65}
$$

and $\bar { R } _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } = R _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } - R _ { \hat { \mathbf { y } } \tilde { \mathbf { y } } } ( R _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } + \lambda _ { f } I ) ^ { - 1 } R _ { \tilde { \mathbf { y } } \hat { \mathbf { y } } } .$ .

Similarly for $\mu _ { g p } ^ { d o } ( x , \cdot )$ , we have the following

Proposition C.4 (Finite dimensional approximation of $\mu _ { g p } ^ { d o } ( x , \cdot ) )$ . Let $\hat { \mathbf { y } } = [ \mathbf { y } ^ { \top } \tilde { \mathbf { y } } ^ { \top } ] ^ { \top }$ be the concatenation of y and ˜y. We can approximate $\mu _ { g p } ^ { d o } ( x , \cdot ) \ w i t h$ ,

$$
\left. \tilde {\mu} _ {g p} ^ {d o} (x, \cdot) \right| \operatorname {v e c} \left(K _ {\mathbf {y y}}\right) \sim N \left(m _ {\tilde {\mu}}, \Sigma_ {\tilde {\mu}}\right) \tag {66}
$$

where,

$$
m _ {\tilde {\mu}} = \Phi_ {\hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} R _ {\hat {\mathbf {y}} \mathbf {y}} R _ {\mathbf {y} \mathbf {y}} ^ {- 1} K _ {\mathbf {y} \mathbf {y}} \left(K _ {\Omega_ {x}} + \lambda I\right) ^ {- 1} \Phi_ {\Omega_ {x}} (x) \tag {67}
$$

$$
\Sigma_ {\tilde {\mu}} = \Phi_ {\hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {\mu} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \Phi_ {\hat {\mathbf {y}}} ^ {\top} \tag {68}
$$

where $K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { \mu } = \Phi _ { \Omega _ { x } } ( x ) ^ { \top } \Phi _ { \Omega _ { x } } ( x ) R _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } - \left( \Phi _ { \Omega _ { x } } ( x ) ^ { \top } ( K _ { \Omega _ { x } } + \lambda I ) ^ { - 1 } \Phi _ { \Omega _ { x } } ( x ) \right) R _ { \hat { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } R _ { \mathbf { y } \hat { \mathbf { y } } }$

Now we have everything we need to derive the main algorithm in our paper, the BAYESIMP. Note that we did not introduce the $\mu _ { g p } ^ { d o }$ notation in the main text to avoid confusion as we did not have space to properly define $\mu _ { g p } ^ { d o }$ .

Proposition 5 (BAYESIMP). Let $f$ and $\mu _ { Y \mid d o ( X ) }$ be GPs learnt as above. Denote $\tilde { f }$ and ${ \tilde { \mu } } _ { Y \mid d o ( X ) }$ as the finite dimensional approximation of $f$ and $\mu _ { Y \mid d o ( X ) }$ respectively. Then $\tilde { g } = \langle \tilde { f } , \tilde { \mu } _ { Y | d o ( X ) } \rangle$ has the following mean and covariance:

$$
m _ {3} (x) = E _ {x} K _ {\mathbf {y} \hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} R _ {\hat {\mathbf {y}} \tilde {\mathbf {y}}} \left(R _ {\tilde {\mathbf {y}} \tilde {\mathbf {y}}} + \lambda_ {f} I\right) ^ {- 1} \mathbf {t} \tag {69}
$$

$$
\kappa_ {3} (x, x ^ {\prime}) = \underbrace {E _ {x} \Theta_ {1} ^ {\top} \tilde {R} _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} \Theta_ {1} E _ {x ^ {\prime}} ^ {\top}} _ {\text {U n c e r t a i n t y f r o m} \mathcal {D} _ {1}} + \underbrace {\Theta_ {2} ^ {(a)} F _ {x x ^ {\prime}} - \Theta_ {2} ^ {(b)} G _ {x x ^ {\prime}}} _ {\text {U n c e r t a i n t y f r o m} \mathcal {D} _ {2}} + \underbrace {\Theta_ {3} ^ {(a)} F _ {x x ^ {\prime}} - \Theta_ {3} ^ {(b)} G _ {x x ^ {\prime}}} _ {\text {U n c e r t a i n t y f r o m I t e r a c t i o n}} \tag {70}
$$

where $E _ { x } \ = \ \Phi _ { \Omega _ { x } } ( x ) ^ { \top } ( K _ { \Omega _ { x } } + \lambda I ) ^ { - 1 } , F _ { x x ^ { \prime } } \ = \ \Phi _ { \Omega _ { x } } ( x ) ^ { \top } \Phi _ { \Omega _ { x } } ( x ^ { \prime } ) , G _ { x x ^ { \prime } } \ = \ \Phi _ { \Omega _ { x } } ( x ) ^ { \top } ( K _ { \Omega _ { x } } + K _ { \Omega _ { x } } ) ,$ $\lambda I ) ^ { - 1 } \Phi _ { \Omega _ { x } } ( x ^ { \prime } )$ , and $\Theta _ { 1 } = K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } R _ { \hat { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } K _ { \mathbf { y } \mathbf { y } }$ , $\Theta _ { 2 } ^ { ( a ) } = \Theta _ { 4 } ^ { \top } R _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } \Theta _ { 4 } , \Theta _ { 2 } ^ { ( b ) } = \Theta _ { 4 } ^ { \top } R _ { \hat { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } R _ { \mathbf { y } \hat { \mathbf { y } } } \Theta _ { 4 }$ and $\begin{array} { r c l r } { \Theta _ { 3 } ^ { ( a ) } } & { = } & { t r ( K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } R _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } \bar { R } _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ) , \Theta _ { 3 } ^ { ( b ) } } & { = } & { t r ( R _ { \hat { \mathbf { y } } \mathbf { y } } R _ { \mathbf { y } \mathbf { y } } ^ { - 1 } R _ { \mathbf { y } \hat { \mathbf { y } } } K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } \bar { R } _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } ) , } \end{array}$ and $\begin{array} { r l } { \Theta _ { 4 } } & { { } = } \end{array}$ $K _ { \hat { \mathbf { y } } \hat { \mathbf { y } } } ^ { - 1 } R _ { \hat { \mathbf { y } } \tilde { \mathbf { y } } } ( K _ { \tilde { \mathbf { y } } \tilde { \mathbf { y } } } + \lambda _ { f } ) ^ { - 1 } \mathbf { t }$ . $\bar { R } _ { \hat { \mathbf { y } } \hat { \mathbf { y } } }$ is the posterior covariance of $f$ evaluated at $\hat { \mathbf { y } }$

Proof of Proposition 5. Since $\tilde { g } = \langle \tilde { f } , \tilde { \mu } _ { g p } ^ { d o } \rangle$ is an inner product between two finite dimensional GPs, we know the variance (as given by Proposition $C . 2 \mathrm { _ }$ ) is characterised by,

$$
\operatorname {v a r} (g) = m _ {\tilde {\mu}} ^ {\top} \Sigma_ {\tilde {f}} m _ {\tilde {\mu}} + m _ {\tilde {f}} ^ {\top} \Sigma_ {\tilde {\mu}} m _ {\tilde {f}} + \operatorname {t r} \left(\Sigma_ {\tilde {f}} \Sigma_ {\tilde {\mu}}\right) \tag {71}
$$

Expanding out each terms we get Proposition 5:

$$
m _ {\tilde {\mu}} ^ {\top} \Sigma_ {\tilde {f}} m _ {\tilde {\mu}} = E _ {x} \Theta_ {1} ^ {\top} \tilde {R} _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} \Theta_ {1} E _ {x ^ {\prime}} ^ {\top} \tag {72}
$$

$$
m _ {\tilde {f}} ^ {\top} \Sigma_ {\tilde {\mu}} m _ {\tilde {f}} = \Theta_ {2} ^ {(a)} F _ {x x ^ {\prime}} - \Theta_ {2} ^ {(b)} G _ {x x ^ {\prime}} \tag {73}
$$

while the first two terms resembles the uncertainty obtained from IMP and BAYESIME, the trace term is new and we will expand it out here,

$$
\begin{array}{l} \operatorname {t r} \left(\Sigma_ {\hat {f}} \Sigma_ {\hat {\mu}}\right) = \operatorname {t r} \left(\Phi_ {\hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {\mu} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \Phi_ {\hat {\mathbf {y}}} ^ {\top} \Phi_ {\hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \bar {R} _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \Phi_ {\hat {\mathbf {y}}} ^ {\top}\right) (75) \\ = \operatorname {t r} \left(K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {\mu} K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \bar {R} _ {\hat {\mathbf {y}} \hat {\mathbf {y}}}\right) (76) \\ = \operatorname {t r} \left(K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \left(F _ {x x ^ {\prime}} R _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} - G _ {x x ^ {\prime}} R _ {\hat {\mathbf {y}} \mathbf {y}} R _ {\mathbf {y} \mathbf {y}} ^ {- 1} R _ {\mathbf {y} \hat {\mathbf {y}}}\right) K _ {\hat {\mathbf {y}} \hat {\mathbf {y}}} ^ {- 1} \bar {R} _ {\hat {\mathbf {y}} \hat {\mathbf {y}}}\right) (77) \\ = \Theta_ {3} ^ {(a)} F _ {x x ^ {\prime}} - \Theta_ {3} ^ {(b)} G _ {x x ^ {\prime}} (78) \\ \end{array}
$$

# C Details on Experimental setup

# C.1 Details on Ablation Study

# C.1.1 Data Generating Process

We use the following causal graphs, $X  Y$ and $Y  T$ , to demonstrate a simple scenario for our data fusion setting. As linking functions, we used for $\mathcal { D } _ { 1 }$ , $Y = x c o s ( \pi x ) + \epsilon _ { 1 }$ and for $\mathcal { D } _ { 2 }$ , $T = 0 . 5 * y * c o s ( y ) \bar { + } \epsilon _ { 2 }$ . where $\mathsf { \bar { \epsilon } } _ { i } \sim \mathcal { N } ( 0 , \sigma _ { i } )$ . Here below we plotted the data for illustration purposes.

![](images/3e3ea7fb0d5ada01dfc459f838e664ad0dca9a519628fc2e2862f72030e1a250.jpg)

![](images/f19b6086a177ce2d8799c4c0eeb2aa8b30e49446e112de748303956714bbbb91.jpg)  
Figure 9: (Left) Illustration of $\mathcal { D } _ { 1 }$ (Right) Illustration of $\mathcal { D } _ { 2 }$

# C.1.2 Explanation on the extrapolation effect

In the main text we referred to the case where IMP is better than BAYESIME as extrapolation effect. We note from the figure above that in $\mathcal { D } _ { 1 }$ we have $x$ around $- 4$ being mapped onto $y$ values around $- 3$ . Note however, that in $\mathcal { D } _ { 2 }$ , we do not observe any values $\tilde { Y }$ below $- 2$ . Hence, because IMP uses a GP model for $\mathcal { D } _ { 2 }$ we are able to account for this mismatch in support and hence attribute more uncertainty to this region, i.e. we see the spike in uncertainty in Fig.5 for IMP.

# C.1.3 Calibration Plots

To investigate the accuracy of the uncertainty quantification in the proposed methods, we perform a (frequentist) calibration analysis of the credible intervals stemming from each method. Fig. 10 gives the calibration plots of the Sampling methods (sampling-based method of [15]) as well as the three proposed methods. On the x-axis is the portion of the posterior mass, corresponding to the width of the credible interval. We will interpret that as a nominal coverage probability of the true function values. On the y-axis is the true coverage probability estimated using the percentage of the times true function values do lie within the corresponding credible intervals. A perfectly calibrated method should have nominal coverage probability equal to the true coverage probability, i.e. being closer to the diagonal line is better.

# C.2 Details on Synthetic Data experiments

# C.2.1 Data Generating Process for simple synthetic dataset

For the first simple synthetic dataset (See Fig.6 (Top)) we used the following data generating graph is defined as.

• X −→ U : U = 2 ∗ X +    
$\bullet \ Z  X : X = 3 * \cos ( Z ) + \epsilon$   
• {Z, U } −→ Y : Y = U + exp(−Z) + 

![](images/6e69097159e17eabd8bd9e9d8383a52912885bd29c5a006d7e87d4dd913f11f6.jpg)  
Figure 10: Calibration plots of Sampling method as well as our 3 proposed methods. We clearly see that BAYESIMP is the best calibrated method amongst all other methods.

$$
\bullet Y \rightarrow T: T = \cos (Y) - \exp (- y / 2 0) + \epsilon
$$

where $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and $Z \sim \mathcal { U } [ - 4 , 4 ]$ , where for $\mathcal { D } _ { 2 }$ we have that $\tilde { Y } \sim \mathcal { U } [ - 1 0 , 1 0 ]$ . In addition, with probability $\pi = 1 / 2$ we shift $U$ by $+ 1$ horizontally and $- 3$ vertically to thus create the multimodality in the data. In order to generate from the interventional distribution, we simply remove the edge from $Z \to X$ and fix the value of $x$ .

# C.2.2 Data Generating Process for harder synthetic dataset from [15]

For the first simple synthetic dataset (Fig.6(Bottom)) we used the same data generating format as in [15].

• U1 = 1   
• U2 = 2   
• F = 3

where the noise is fixed to be $\mathcal { N } ( 0 , 1 )$ and where we switch with $\pi = 1 / 2$ from mode $Y _ { 1 }$ and $Y _ { 2 }$ , where $\tilde { Y } \sim \mathcal { U } [ - 2 , 9 ]$ for $\mathcal { D } _ { 2 }$ .

# C.3 Details on Healthcare Data experiments

# C.3.1 Data Generating Process

For the healthcare dataset, $\mathcal { D } _ { 1 }$ , (Fig.1) we used the same data generating format as in [15] with the difference that we make statin continuous and increased the age range.

$$
\bullet \operatorname {a g e} = \mathcal {U} [ 1 5, 7 5 ]
$$

• $b m i = \mathcal { N } ( 2 7 - 0 . 0 1 * a g e , 0 . 7 )$   
$\cdot \ a s p i r i n = \sigma ( - 8 . 0 + 0 . 1 * a g e + 0 . 0 3 * b m i )$   
$\cdot \ s t a t i n = - 1 3 + 0 . 1 * a g e + 0 . 2 * b m i$   
  
)

As for the second dataset, $\mathcal { D } _ { 2 }$ we firstly fit a GP on the data collected from [7]. Once we have the posterior GP, we can then use it as a generator for the $\mathcal { D } _ { 2 }$ as it takes as input $P S A$ . This generator hence acts as a link between $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ . This way we are able to create a simulator that allows us to obtain samples from E[Cancer volume $\left| d o ( S t a t i n ) \right]$ for our causal BO setup.

# C.4 Bayesian Optimisation experiments with IMP and BAYESIME

![](images/89f51ee0bf5bf728fa36b59feae42ca738193d399b613efaaf8e73163a476130.jpg)

![](images/93bcb02812f4b680dec5e14a69ff33e709baf106a3ce768e3398acc5bc4c94a8.jpg)

![](images/a1202c1685e2f39acf2ab0797598ba71858fb111980ec9b8d4ab2de1f016fb83.jpg)  
Figure 11: (Left) Simple graph using backdoor adjustment (Middle) Simple graph using front-door adjustment (Right) Harder graph using front-door adjustment. BAYESIMP strikes the right balance between IMP and BAYESIME and all three perform better than CBO and the GP baseline.

The main text compares BAYESIMP to CBO and the baseline GP with no learnt prior in the Bayesian Optimisation experiments. Here, we include IMP and BAYESIME (i.e. simplified versions of BAYESIMP that account for only one source of uncertainty each) in those comparisons. We see from Fig.11 that BAYESIMP is comparable to IMP and BAYESIME in most cases. While BAYESIMP is not the best performing method in every scenario, it does hit a good middle ground between the first two proposed methods. For Fig.11 (Left, Middle) we used $N = 1 0 0$ and $M = 5 0$ . In the left figure, BAYESIME and BAYESIMP are very similar, whereas IMP is considerably worst. In the middle figure, all methods seems to perform well without much difference. In the right figure, we have $N = 5 0 0$ and $M = 5 0$ and this is a case where IMP is best, while BAYESIME appears to get stuck in a local optimum (recall that BAYESIME does not take into account uncertainty in $\mathcal { D } _ { 2 }$ where there is little data). We note that all three methods converge faster than the current SOTA CBO.