# Invariant Ancestry Search

Phillip B. Mogensen∗1, Nikolaj Thams1, and Jonas Peters1

$^ { 1 }$ Department of Mathematical Sciences, University of Copenhagen, Denmark

# Abstract

Recently, methods have been proposed that exploit the invariance of prediction models with respect to changing environments to infer subsets of the causal parents of a response variable. If the environments influence only few of the underlying mechanisms, the subset identified by invariant causal prediction (ICP), for example, may be small, or even empty. We introduce the concept of minimal invariance and propose invariant ancestry search (IAS). In its population version, IAS outputs a set which contains only ancestors of the response and is a superset of the output of ICP. When applied to data, corresponding guarantees hold asymptotically if the underlying test for invariance has asymptotic level and power. We develop scalable algorithms and perform experiments on simulated and real data.

# 1 Introduction

Causal reasoning addresses the challenge of understanding why systems behave the way they do and what happens if we actively intervene. Such mechanistic understanding is inherent to human cognition, and developing statistical methodology that learns and utilizes causal relations is a key step in improving both narrow and broad AI (Jordan, 2019; Pearl, 2018). Several approaches exist for learning causal structures from observational data. Approaches such as the PC-algorithm (Spirtes et al., 2000) or greedy equivalence search (Chickering, 2002) learn (Markov equivalent) graphical representations of the causal structure Lauritzen (1996). Other approaches learn the graphical structure under additional assumptions, such as non-Gaussianity Shimizu et al. (2006) or non-linearity Hoyer et al. (2009); Peters et al. (2014). Zheng et al. (2018) convert the problem into a continuous optimization problem, at the expense of identifiability guarantees.

Invariant causal prediction (ICP) (Peters et al., 2016; Heinze-Deml et al., 2018; Pfister et al., 2019; Gamella & Heinze-Deml, 2020; Martinet et al., 2021) assumes that data are sampled from heterogeneous environments (which can be discrete, categorical or continuous), and identifies direct causes of a target $Y$ , also known as causal parents of $Y$ . Learning ancestors (or parents) of a response $Y$ yields understanding of anticipated changes when intervening in the system. It is a less ambitious task than learning the complete graph but may allow for methods that come with weaker assumptions and stronger guarantees.

More concretely, for predictors $X _ { 1 } , \ldots , X _ { d }$ , ICP searches for subsets $S \subseteq \{ 1 , \ldots , d \}$ that are invariant; a set $X _ { S }$ of predictors is called invariant if it renders $Y$ independent of the environment, conditional on $X _ { S }$ . ICP then outputs the intersection of all invariant predictor sets $S _ { \mathrm { I C P } } : = \cap _ { S \mathrm { i n v a r i a n t } } S$ . Peters et al. (2016) show that if invariance is tested empirically from data at level $\alpha$ , the resulting intersection $\hat { S } _ { \mathrm { I C P } }$ is a subset of direct causes of $Y$ with probability at least $1 - \alpha$ . 1

In many cases, however, the set learned by ICP forms a strict subset of all direct causes or may even be empty. This is because disjoint sets of predictors can be invariant, yielding an empty intersection, which may happen both for finite samples as well as in the population setting. In this work, we introduce and characterize minimally invariant sets of predictors, that is, invariant sets $S$ for which no proper subset is invariant. We propose to consider the union $S _ { \mathrm { { I A S } } }$ of all minimally invariant sets, where IAS stands for invariant ancestry search. We prove that $S _ { \mathrm { { I A S } } }$ is a subset of causal ancestors of $Y$ , invariant, non-empty and contains $S _ { \mathrm { I C P } }$ . Learning causal ancestors of a response may be desirable for several reasons: e.g., they are the variables that may have an influence on the response variable when intervened on. In addition, because IAS yields an invariant set, it can be used to construct predictions that are stable across environments (e.g., Rojas-Carulla et al., 2018; Christiansen et al., 2022).

In practice, we estimate minimally invariant sets using a test for invariance. If such a test has asymptotic power against some of the non-invariant sets (specified in Section 5.2), we show that, asymptotically, the probability of $\hat { S } _ { \mathrm { I A S } }$ being a subset of the ancestors is at least $1 - \alpha$ . This puts stronger assumptions on the invariance test than ICP (which does not require any power) in return for discovering a larger set of causal ancestors. We prove that our approach retains the ancestral guarantee if we test minimal invariance only among subsets up to a certain size. This yields a computational speed-up compared to testing minimal invariance in all subsets, but comes at the cost of potentially finding fewer causal ancestors.

The remainder of this work is organized as follows. In Section 2 we review relevant background material, and we introduce the concept of minimal invariance in Section 3. Section 4 contains an oracle algorithm for finding minimally invariant sets (and a closed-form expression of $S _ { \mathrm { I C P } }$ ) and Section 5 presents theoretical guarantees when testing minimal invariance from data. In Section 6 we evaluate our method in several simulation studies as well as a real-world data set on gene perturbations. Code is provided at https://github. com/PhillipMogensen/InvariantAncestrySearch.

# 2 Preliminaries

# 2.1 Structural Causal Models and Graphs

We consider a setting where data are sampled from a structural causal model (SCM) Pearl (2009); Bongers et al. (2021)

$$
Z _ {j} := f _ {j} (\mathrm {P A} _ {j}, \epsilon_ {j}),
$$

for some functions $f _ { j }$ , parent sets $\mathrm { P A } _ { j }$ and noise distributions $\epsilon _ { j }$ . Following Peters et al. (2016); Heinze-Deml et al. (2018), we consider an SCM over variables $Z : = ( E , X , Y )$ where $E$ is an exogenous environment variable (i.e., $\mathrm { P A } _ { E } = \varnothing$ ), $Y$ is a response variable and $X = ( X _ { 1 } , \ldots , X _ { d } ) $ is a collection of predictors of $Y$ . We denote by $\mathcal { P }$ the family of all possible distributions induced by an SCM over $( E , X , Y )$ of the above form.

For a collection of nodes $j \in [ d ] : = \{ 1 , \ldots , d \}$ and their parent sets $\mathrm { P A } _ { j }$ , we define a directed graph $\mathcal { G }$ with nodes $[ d ]$ and edges $j ^ { \prime }  j$ for all $j ^ { \prime } \in \mathrm { P A } _ { j }$ . We denote by $\mathrm { C H } _ { j }$ , $\mathrm { A N } _ { j }$ $\jmath$ and DE $\jmath$ the children, ancestors and descendants of a variable $j$ , respectively, neither containing $j$ . A graph $\mathcal { G }$ is called a directed acyclic graph (DAG) if it does not contain any directed cycles. See Pearl (2009) for more details and the definition of $d$ -separation.

Throughout the remainder of this work, we make the following assumptions about causal sufficiency and exogeneity of $E$ (Section 7 describes how these assumptions can be relaxed).

Assumption 2.1. Data are sampled from an SCM over nodes $( E , X , Y )$ , such that the corresponding graph is a DAG, the distribution is faithful with respect to this DAG, and the environments are exogenous, i.e., $\mathrm { P A } _ { E } = \varnothing$ .

# 2.2 Invariant Causal Prediction

Invariant causal prediction (ICP), introduced by Peters et al. (2016), exploits the existence of heterogeneity in the data, here encoded by an environment variable $E$ , to learn a subset of causal parents of a response variable $Y$ . A subset of predictors $S \subseteq [ d ]$ is invariant if $Y \perp \perp E \mid S$ , and we define $\mathcal { T } : = \{ S \subseteq [ d ] \mid S \mathrm { { \ i n v a r i a n t } } \}$ to be the set of all invariant sets. We denote the corresponding hypothesis that $S$ is invariant by

$$
H _ {0, S} ^ {\mathcal {I}}: \quad S \in \mathcal {I}.
$$

Formally, $H _ { 0 , S } ^ { \mathcal { L } }$ corresponds to a subset of distributions in $\mathcal { P }$ , and we denote by $H _ { A , S } ^ { \mathcal { L } } : =$ $\mathcal { P } \setminus H _ { 0 , S } ^ { \mathcal { L } }$ the alternative hypothesis to $H _ { 0 , S } ^ { \mathcal { L } }$ . Peters et al. (2016) define the oracle output

$$
S _ {\mathrm {I C P}} := \bigcap_ {S: H _ {0, S} ^ {\mathcal {T}} \text {t r u e}} S \tag {1}
$$

(with $S _ { \mathrm { I C P } } = \emptyset$ if no sets are invariant) and prove $S _ { \mathrm { I C P } } \subseteq \mathrm { P A } _ { Y }$ . If provided with a test for the hypotheses $H _ { 0 , S } ^ { \mathcal { L } }$ , we can test all sets $S \subseteq [ d ]$ for invariance and take the intersection over all accepted sets: $\begin{array} { r } { \hat { S } _ { \mathrm { I C P } } : = \bigcap _ { S : H _ { 0 , S } ^ { \mathcal { T } } } } \end{array}$ not rejected $S$ ; If the invariance test has level $\alpha$ , $\hat { S } _ { \mathrm { I C P } } \subseteq \mathrm { P A } _ { Y }$ with probability at least $1 - \alpha$ .

However, even for the oracle output in Equation (1), there are many graphs for which $S _ { \mathrm { I C P } }$ is a strict subset of $\mathrm { P A } _ { Y }$ . For example, in Figure 1 (left), since both $\{ 1 , 2 \}$ and $\{ 3 \}$ are invariant, $S _ { \mathrm { I C P } } \subseteq \{ 1 , 2 \} \cap \{ 3 \} = \emptyset$ . This does not violate $S _ { \mathrm { I C P } } \subseteq \mathrm { P A } _ { Y }$ , but is non-informative. Similarly, in Figure 1 (right), $S _ { \mathrm { I C P } } = \{ 1 \}$ , as all invariant sets contain $\{ 1 \}$ . Here, $S _ { \mathrm { I C P } }$ contains some information, but is not able to recover the full parental set. In neither of these two cases, $S _ { \mathrm { I C P } }$ is an invariant set. If the environments are such that each parent of $Y$ is either affected by the environment directly or is a parent of an affected node, then $S _ { \mathrm { I C P } } = \mathrm { P A } _ { Y }$ (Peters et al., 2016, proof of Theorem 3). The shortcomings of ICP thus relate to settings where the environments act on too few variables or on uninformative ones.

$$
E \stackrel {{\smile}} {{\rightarrow}}\begin{array}{c}X _ {1}\\\smile\\X _ {2}\end{array}\stackrel {{Y}} {{\rightarrow}}\begin{array}{c}Y\\\uparrow\\\smile\\X _ {3}\end{array}\stackrel {{\downarrow}} {{\rightarrow}} X _ {4} \qquad E \stackrel {{\smile}} {{\rightarrow}}\begin{array}{c}X _ {1} \to Y\\\uparrow\\X _ {2}\end{array}
$$

Figure 1: Two structures where $S _ { \mathrm { I C P } } \subsetneq \mathrm { P A } _ { Y }$ . (left ) $S _ { \mathrm { I C P } } = \emptyset$ . (right ) $S _ { \mathrm { { I C P } } } = \{ 1 \}$ . In both, our method outputs $S _ { \mathrm { { I A S } } } = \{ 1 , 2 , 3 \}$ .

For large $d$ , it has been suggested to apply ICP to the variables in the Markov boundary Pearl (2014), $\mathrm { M B } _ { Y } = \mathrm { P A } _ { Y } \cup \mathrm { C H } _ { Y } \cup \mathrm { P A } ( \mathrm { C H } _ { Y } )$ (we denote the oracle output by $S _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ ). As $\mathrm { P A } _ { Y } \subseteq \mathrm { M B } _ { Y }$ $Y$ , it still holds that $S _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ is a subset of the causal parents of the response.2 However, the procedure must still be applied to $2 ^ { | \mathrm { M B } _ { Y } | }$ sets, which is only feasible if the Markov boundary is sufficiently small. In practice, the Markov boundary can, for example, be estimated using Lasso regression or gradient boosting techniques (Tibshirani, 1996; Meinshausen & B¨uhlmann, 2006; Friedman, 2001).

# 3 Minimal Invariance and Ancestry

We now introduce the concept of minimally invariant sets, which are invariant sets that do not have any invariant subsets. We propose to consider $S _ { \mathrm { { I A S } } }$ , the oracle outcome of invariant ancestry search, defined as the union of all minimally invariant sets. We will see that $S _ { \mathrm { { I A S } } }$ is an invariant set, it consists only of ancestors of $Y$ , and it contains $S _ { \mathrm { I C P } }$ as a subset.

Definition 3.1. Let $S \subseteq [ d ]$ . We say that $S$ is minimally invariant if and only if

$$
S \in \mathcal {I} \text {a n d} \forall S ^ {\prime} \subsetneq S: S ^ {\prime} \notin \mathcal {I};
$$

that is, $S$ is invariant and no subset of $S$ is invariant. We define $\mathcal { M } \mathbb { Z } : = \{ S \mid S$ minimally invariant}.

The concept of minimal invariance is closely related to the concept of minimal $d$ -separators (Tian et al., 1998). This connection allows us to state several properties of minimal invariance. For example, an invariant set is minimally invariant if and only if it is non-invariant as soon as one of its elements is removed.

Proposition 3.2. Let $S \subseteq [ d ]$ . Then $S \in \mathcal { M } \mathcal { I }$ if and only if $S \in \mathcal { L }$ and for all $j \in S$ , it holds that $S \setminus \{ j \} \not \in \mathcal { I }$ .

The proof follows directly from (Tian et al., 1998, Corollary 2). We can therefore decide whether a given invariant set $S$ is minimally invariant using $\mathcal { O } ( | S | )$ checks for invariance, rather than $\mathcal { O } ( 2 ^ { | S | } )$ (as suggested by Definition 3.1). We use this insight in Section 5.1, when we construct a statistical test for whether or not a set is minimally invariant.

To formally define the oracle outcome of IAS, we denote the hypothesis that a set $S$ is minimally invariant by

$$
H _ {0, S} ^ {\mathcal {M I}}: \quad S \in \mathcal {M I}
$$

(and the alternative hypothesis, $S \notin \mathcal { M } \mathcal { T }$ , by $H _ { A , S } ^ { \mathcal { M } \mathbb { Z } }$ ) and define the quantity of interest

$$
S _ {\text {I A S}} := \bigcup_ {S: H _ {0, S} ^ {\mathcal {M I}} \text {t r u e}} S \tag {2}
$$

with the convention that a union over the empty set is the empty set.

The following proposition states that $S _ { \mathrm { { I A S } } }$ is a subset of the ancestors of the response $Y$ . Similarly to PA $Y$ , variables in AN $Y$ are causes of $Y$ in that for each ancestor there is a directed causal path to $Y$ . Thus, generically, when intervened, these variables have a causal effect on the response.

# Proposition 3.3. It holds that $S _ { \mathrm { I A S } } \subseteq \mathrm { A N } _ { Y }$ $Y$ .

The proof follows directly from (Tian et al., 1998, Theorem 2); see also (Acid & De Campos, 2013, Proposition 2). The setup in these papers is more general than what we consider here; we therefore provide direct proofs for Propositions 3.2 and 3.3 in Appendix A, which may provide further intuition for the results.

Finally, we show that the oracle output of IAS contains that of ICP and, contrary to ICP, it is always an invariant set.

Proposition 3.4. Assume that $E \notin \mathrm { P A } _ { Y }$ . It holds that

(i) $S _ { \mathrm { I A S } } \in \mathcal { I }$ and   
(ii) $S _ { \mathrm { I C P } } \subseteq S _ { \mathrm { I A S } }$ , with equality if and only if $S _ { \mathrm { I C P } } \in \mathcal { I }$ .

# 4 Oracle Algorithms

When provided with an oracle that tells us whether a set is invariant or not, how can we efficiently compute $S _ { \mathrm { { I C P } } }$ and $S _ { \mathrm { { I A S } } }$ ? Here, we assume that the oracle is given by a DAG, see Assumption 2.1. A direct application of Equations (1) and (2) would require checking a number of sets that grows exponentially in the number of nodes. For $S _ { \mathrm { I C P } }$ , we have the following characterization.3

# Proposition 4.1. If $E \notin \mathrm { P A } _ { Y }$ , then $S _ { \mathrm { I C P } } = \mathrm { P A } _ { Y } \cap \left( \mathrm { C H } _ { E } \cup \mathrm { P A } ( \mathrm { A N } _ { Y } \cap \mathrm { C H } _ { E } ) \right) .$

This allows us to efficiently read off $S _ { \mathrm { { I C P } } }$ from the DAG, (e.g., it can naively be done in $\mathcal { O } ( ( d + 2 ) ^ { 2 . 3 7 3 } \log ( d + 2 ) )$ time, where the exponent 2.373 comes from matrix multiplication). For $S _ { \mathrm { { I A S } } }$ , to the best of our knowledge, there is no closed form expression that has a similarly simple structure.

Instead, for IAS, we exploit the recent development of efficient algorithms for computing all minimal $d$ -separators (for two given sets of nodes) in a given DAG (see, e.g., Tian et al., 1998; van der Zander et al., 2019). A set $S$ is called a minimal $d$ -separator of $E$ and $Y$ if it $d$ -separates $E$ and $Y$ given $S$ and no strict subset of $S$ satisfies this property. These algorithms are often motivated by determining minimal adjustment sets (e.g., Pearl, 2009) that can be used to compute the total causal effect between two nodes, for example. If the underlying distribution is Markov and faithful with respect to the DAG, then a set $S$ is

minimally invariant if and only if it is a minimal $d$ -separator for $E$ and $Y$ . We can therefore use the same algorithms to find minimally invariant sets; van der Zander et al. (2019) provide an algorithm (based on work by Takata (2010)) for finding minimal $d$ -separators with polynomial delay time. Applied to our case, this means that while there may be exponentially many minimally invariant sets,4 when listing all such sets it takes at most polynomial time until the next set or the message that there are nor further sets is output. In practice, on random graphs, we found this to work well (see Section 6.1). But since $S _ { \mathrm { { I A S } } }$ is the union of all minimally invariant sets, even faster algorithms may be available; to the best of our knowledge, it is an open question whether finding $S _ { \mathrm { { I A S } } }$ is an NP-hard problem (see Appendix B for details).

We provide a function for listing all minimally invariant sets in our python code; it uses an implementation of the above mentioned algorithm, provided in the R (R Core Team, 2021) package dagitty (Textor et al., 2016). In Section 6.1, we study the properties of the oracle set $S _ { \mathrm { { I A S } } }$ . When applied to 500 randomly sampled, dense graphs with $d = 1 5$ predictor nodes and five interventions, the dagitty implementation had a median speedup of a factor of roughly 17, compared to a brute-force search (over the ancestors of $Y$ ). The highest speedup achieved was by a factor of more than 1,900.

The above mentioned literature can be used only for oracle algorithms, where the graph is given. In the following sections, we discuss how to test the hypothesis of minimal invariance from data.

# 5 Invariant Ancestry Search

# 5.1 Testing a Single Set for Minimal Invariance

Usually, we neither observe a full SCM nor its graphical structure. Instead, we observe data from an SCM, which we want to use to decide whether a set is in $\mathcal { M T }$ , such that we make the correct decision with high probability. We now show that a set $S$ can be tested for minimal invariance with asymptotic level and power if given a test for invariance that has asymptotic level and power.

Assume that $\mathcal { D } _ { n } = ( X _ { i } , E _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ are observations (which may or may not be independent) tra of $( X , E , Y )$ $( S , { \mathcal { D } } _ { n } , \alpha )$ and let $\phi _ { n } ^ { \mathcal { M } \mathcal { T } }$ $\phi _ { n } ^ { \mathcal { M } \bot } : \mathrm { p o w e r s e t } ( [ d ] ) \times \mathcal { D } _ { n } \times ( 0 , 1 )  \{ 0 , 1 \}$ $\phi _ { n } ^ { \mathcal { M T } } ( S , \mathcal { D } _ { n } , \alpha )$ be a decision rule that her the hypothesis $H _ { 0 , S } ^ { \mathcal { M } \mathbb { Z } }$ should be rejected ( $\phi _ { n } ^ { \mathcal { M } \mathcal { L } } = 1$ ) at significance threshold $\alpha$ , or not ( $\phi _ { n } ^ { \mathcal { M } \mathcal { L } } = 0$ ). To ease notation, we suppress the dependence on $\mathcal { D } _ { n }$ and $\alpha$ when the statements are unambiguous.

A test $\psi _ { n }$ for the hypothesis $H _ { 0 }$ has pointwise asymptotic level if

$$
\forall \alpha \in (0, 1): \sup  _ {\mathbb {P} \in H _ {0}} \lim  _ {n \rightarrow \infty} \mathbb {P} \left(\psi_ {n} = 1\right) \leq \alpha \tag {3}
$$

and pointwise asymptotic power if

$$
\forall \alpha \in (0, 1): \inf  _ {\mathbb {P} \in H _ {A}} \lim  _ {n \rightarrow \infty} \mathbb {P} (\psi_ {n} = 1) = 1. \tag {4}
$$

If the limit and the supremum (resp. infimum) in Equation (3) (resp. Equation (4)) can be interchanged, we say that $\psi _ { n }$ has uniform asymptotic level (resp. power).

Tests for invariance have been examined in the literature. Peters et al. (2016) propose two simple methods for testing for invariance in linear Gaussian SCMs when the environments are discrete, although the methods proposed extend directly to other regression scenarios. Pfister et al. (2019) propose resampling-based tests for sequential data from linear Gaussian SCMs. Furthermore, any valid test for conditional independence between $Y$ and $E$ given a set of predictors $S$ can be used to test for invariance. Although for continuous $X$ , there exists no general conditional independence test that has both level and non-trivial power (Shah & Peters, 2020), it is possible to impose restrictions on the data-generating process that ensure the existence of non-trivial tests (e.g., Fukumizu et al., 2008; Zhang et al., 2011; Berrett et al., 2020; Shah & Peters, 2020; Thams et al., 2021). Heinze-Deml et al. (2018) provide an overview and a comparison of several conditional independence tests in the context of invariance.

To test whether a set $S \subseteq [ d ]$ is minimally invariant, we define the decision rule

$$
\phi_ {n} ^ {\mathcal {M I}} (S) := \left\{ \begin{array}{l l} 1 & \text {i f} \phi_ {n} (S) = 1 \text {o r} \min  _ {j \in S} \phi_ {n} (S \setminus \{j \}) = 0, \\ 0 & \text {o t h e r w i s e}, \end{array} \right. \tag {5}
$$

where $\phi _ { n } ^ { \mathcal { M } \mathcal { L } } ( \emptyset ) : = \phi _ { n } ( \emptyset )$ . Here, $\phi _ { n }$ is a test for the hypothesis $H _ { 0 , S } ^ { \mathcal { L } }$ , e.g., one of the tests mentioned above. This decision rule rejects $H _ { 0 , S } ^ { \mathcal { M Z } }$ either if $H _ { 0 , S } ^ { \mathcal { L } }$ is rejected by $\phi _ { n }$ or if there exists $j \in S$ such that $H _ { 0 , S \backslash \{ j \} } ^ { \mathcal { L } }$ is not rejected. If $\phi _ { n }$ has pointwise (resp. uniform) asymptotic level and power, then $\phi _ { n } ^ { \mathcal { M } \mathcal { L } }$ has pointwise (resp. uniform) asymptotic level and pointwise (resp. uniform) asymptotic power of at least $1 - \alpha$ .

Theorem 5.1. Let $\phi _ { n } ^ { \mathcal { M L } }$ be defined as in Equation (5) and let $S \subseteq [ d ]$ . Assume that the decision rule $\phi _ { n }$ has pointwise asymptotic level and power for $S$ and for all $S \setminus \{ j \} , j \in S$ . Then, $\phi _ { n } ^ { \mathcal { M } \mathcal { L } }$ has pointwise asymptotic level and pointwise asymptotic power of at least $1 - \alpha$ i.e.,

$$
\inf_{\mathbb{P}\in H^{ \mathcal{M}\mathcal{I}}_{A,S}}\lim_{n\to \infty}\mathbb{P}(\phi_{n}^{\mathcal{M}\mathcal{I}}(S) = 1)\geq 1 - \alpha .
$$

If $\phi _ { n }$ has uniform asymptotic level and power, then $\phi _ { n } ^ { \mathcal { M } \mathcal { L } }$ has uniform asymptotic level and uniform asymptotic power of at least $1 - \alpha$ .

Due to Proposition 3.3, a test for $H _ { 0 , S } ^ { \mathcal { M Z } }$ is implicitly a test for $S \subseteq \mathrm { A N } _ { Y }$ , a d can thus $S$ $Y$ rejecting $H _ { 0 , S } ^ { \mathcal { M Z } }$ is not evidence for $S \nsubseteq \mathrm { A N }$ ; it is evidence for $S \notin \mathcal { M } \mathcal { T }$ .

# 5.2 Learning $S _ { \mathrm { { I A S } } }$ from Data

We now consider the task of estimating the set $S _ { \mathrm { { I A S } } }$ from data. If we are given a test for invariance that has asymptotic level and power and if we correct for multiple testing appropriately, we can estimate $S _ { \mathrm { { I A S } } }$ by $\hat { S } _ { \mathrm { I A S } }$ , which, asymptotically, is a subset of AN $Y$ with large probability.

Theorem 5.2. Assume that the decision rule $\phi _ { n }$ has pointwise asymptotic level for all minimally invariant sets and pointwise asymptotic power for all $S \subseteq [ d ]$ such that $S$ is not a superset of a minimally invariant set. Define $C : = 2 ^ { d }$ and let $\widehat { \mathcal { T } } : = \bigl \{ S \subseteq [ d ] \ | \ \phi _ { n } ( S , \alpha C ^ { - 1 } ) = 0 \bigr \}$ be the set of all sets for which the hypothesis of invariance is not rejected and define $\widehat { \mathcal { M } } \mathcal { Z } : = \left\{ S \in \widehat { \mathcal { I } } \mid \forall S ^ { \prime } \subseteq S : S ^ { \prime } \notin \widehat { \mathcal { I } } \right\}$ and $\textstyle { \hat { S } } _ { \mathrm { I A S } } : = \bigcup _ { S \in { \widehat { \mathcal { M } } } { \mathcal { T } } } S$ . It then holds that

$$
\begin{array}{l} \lim  _ {n \rightarrow \infty} \mathbb {P} (\hat {S} _ {\mathrm {I A S}} \subseteq \mathrm {A N} _ {Y}) \geq \lim  _ {n \rightarrow \infty} \mathbb {P} (\hat {S} _ {\mathrm {I A S}} = S _ {\mathrm {I A S}}) \\ \geq 1 - \alpha . \\ \end{array}
$$

A generic algorithm for implementing $\hat { S } _ { \mathrm { I A S } }$ is given in Appendix D.

Remark 5.3. Consider a decision rule $\phi _ { n }$ that just (correctly) rejects the empty set (e.g., because the $p$ -value is just below the threshold $\alpha$ ), indicating that the effect of the environments is weak. It is likely that there are other sets $S ^ { \prime } \not \in \mathcal { I }$ , which the test may not have sufficient power against and are (falsely) accepted as invariant. If one of such sets contains non-ancestors of $Y$ , this yields a violation of $\hat { S } _ { \mathrm { I A S } } \subseteq \mathrm { A N } _ { Y }$ . To guard against this, testing $S = \emptyset$ can be done at a lower significance level, $\alpha _ { 0 } < \alpha$ . This modified IAS approach is conservative and may return ${ \hat { S } } _ { \mathrm { I A S } } = \emptyset$ if the environments do not have a strong impact on $Y$ , but it retains the guarantee $\begin{array} { r } { \operatorname* { l i m } _ { n  \infty } \mathbb { P } ( \hat { S } _ { \mathrm { I A S } } \subseteq \mathrm { A N } _ { Y } ) \geq 1 - \alpha } \end{array}$ of Theorem 5.2.

The multiple testing correction performed in Theorem 5.2 is strictly conservative because we only need to correct for the number of minimally invariant sets, and there does not exist $2 ^ { d }$ minimally invariant sets. Indeed, the statement of Theorem 5.2 remains valid for $C = C ^ { \prime }$ if the underlying DAG has at most $C ^ { \prime }$ minimally invariant sets. We hypothesize that a DAG can contain at most $3 ^ { | d / 3 | }$ minimally invariant sets and therefore propose using $C = 3 ^ { | d / 3 | }$ in practice. If this hypothesis is true, Theorem 5.2 remains valid (for any DAG), using $C = 3 ^ { | d / 3 | }$ (see Appendix C for a more detailed discussion).

Alternatively, as shown in the following section, we can restrict the search for minimally invariant sets to a predetermined size. This requires milder correction factors and comes with computational benefits.

# 5.3 Invariant Ancestry Search in Large Systems

We now develop a variation of Theorem 5.2, which allows us to search for ancestors of $Y$ in large graphs, at the cost of only identifying minimally invariant sets up to some a priori determined size.

Similarly to ICP (see Section 2.2), one could restrict IAS to the variables in MB $Y$ but the output may be smaller than $S _ { \mathrm { { I A S } } }$ ; in particular, there are only non-parental ancestors in MB $Y$ if these are parents to both a parent a child of $Y$ (For instance, in the graph $E \to X _ { 1 } \to \dots \to X _ { d } \to Y$ , $S _ { \mathrm { I A S } } = \{ 1 , \ldots , d \}$ but restricting IAS to MB $Y$ would yield the set $\{ d \}$ .) Thus, we do not expect such an approach to be particularly fruitful in learning ancestors.

Here, we propose an alternative approach and define

$$
S _ {\text {I A S}} ^ {m} := \bigcup_ {S: S \in \mathcal {M I} \text {a n d} | S | \leq m} S \tag {6}
$$

as the union of minimally invariant sets that are no larger than $m \leq d$ . For computing $S _ { \mathrm { I A S } } ^ { m }$ one only needs to check invariance of the $\textstyle \sum _ { i = 0 } ^ { m } { \binom { d } { i } }$ sets that are no larger than $m$ . $S _ { \mathrm { I A S } } ^ { m }$ itself, however, can be larger than The following proposition characte $m$ : in the graph aes properties of Equation (6), . $S _ { \mathrm { I A S } } ^ { 1 } = \{ 1 , \ldots , d \}$ . $S _ { \mathrm { I A S } } ^ { m }$

Proposition 5.4. Let $m < d$ and let $m _ { \mathrm { m i n } }$ and $m _ { \mathrm { m a x } }$ be the size of a smallest and a largest minimally invariant set, respectively. The following statements are true:

(i) $S _ { \mathrm { I A S } } ^ { m } \subseteq \mathrm { A N } _ { Y }$   
(ii) If $m \geq m _ { \mathrm { m a x } }$ , then $S _ { \mathrm { I A S } } ^ { m } = S _ { \mathrm { I A S } }$   
(iii) If $m \geq m _ { \operatorname* { m i n } }$ and $E \notin \mathrm { P A } _ { Y }$ , then $S _ { \mathrm { I A S } } ^ { m } \in \mathcal { I }$   
(iv) If $m \geq m _ { \operatorname* { m i n } }$ and $E \notin \mathrm { P A } _ { Y }$ , then $S _ { \mathrm { I C P } } \subseteq S _ { \mathrm { I A S } } ^ { m }$ with equality if and only if $S _ { \mathrm { I C P } } \in \mathcal { I }$

If $m < m _ { \mathrm { m i n } }$ and $S _ { \mathrm { I C P } } \neq \emptyset$ , then $S _ { \mathrm { I C P } } \subseteq S _ { \mathrm { I A S } } ^ { m }$ does not hold. However, we show in Section 6.1 using simulations that $S _ { \mathrm { I A S } } ^ { m }$ is larger than $S _ { \mathrm { I C P } }$ in many sparse graphs, even for $m = 1$ , when few nodes are intervened on.

In addition to the computational speedup offered by considering $S _ { \mathrm { I A S } } ^ { m }$ instead of $S _ { \mathrm { { I A S } } }$ the set $S _ { \mathrm { { I A S } } }$ can be estimated from data using a smaller correction factor than the one employed in Theorem 5.2. This has the benefit that in practice, smaller sample sizes may be needed to detect non-invariance.

Theorem 5.5. Let $m \leq d$ and define $\begin{array} { r } { C ( m ) : = \sum _ { i = 0 } ^ { m } { \binom { d } { i } } } \end{array}$ . Assume that the decision rule $\phi _ { n }$ has pointwise asymptotic level for all minimally invariant sets of size at most m and pointwise power for all sets of size at most m that are not supersets of a minimally invariant set. Let $\widehat { \mathcal { T } } ^ { m } : = \{ S \subseteq [ d ] \vert \phi _ { n } ( S , \alpha C ( m ) ^ { - 1 } ) = 0 \}$ and $| S | \le m \}$ , be the set of all sets of size at most m for which the hypothesis of invariance is not rejected and define $\widehat { { M I } } ^ { m } : =$ $\left\{ S \in { \widehat { \mathbb { Z } } } ^ { m } \mid \forall S ^ { \prime } \subsetneq S : S ^ { \prime } \not \in { \widehat { \mathbb { Z } } } ^ { m } \right\}$ and $\begin{array} { r } { \hat { S } _ { \mathrm { I A S } } ^ { m } : = \bigcup _ { S \in \widehat { M } ^ { T } } S } \end{array}$ . It then holds that

$$
\begin{array}{l} \lim _ {n \to \infty} \mathbb {P} (\hat {S} _ {\mathrm {I A S}} ^ {m} \subseteq \mathrm {A N} _ {Y}) \geq \lim _ {n \to \infty} \mathbb {P} (\hat {S} _ {\mathrm {I A S}} ^ {m} = S _ {\mathrm {I A S}} ^ {m}) \\ \geq 1 - \alpha . \\ \end{array}
$$

The method proposed in Theorem 5.5 outputs a non-empty set if there exists a non-empty set of size at most $m$ , for which the hypothesis of invariance cannot be rejected. In a sparse graph, it is likely that many small sets are minimally invariant, whereas if the graph is dense, it may be that all invariant sets are larger than $m$ , such that $S _ { \mathrm { I A S } } ^ { m } = \emptyset$ . In dense graphs however, many other approaches may fail too; for example, it is also likely that the size of the Markov boundary is so large that applying ICP on MB $Y$ is not feasible.

# 6 Experiments

We apply the methods developed in this paper in a population-case experiment using oracle knowledge (Section 6.1), a synthetic experiment using finite sample tests (Section 6.2), and a real-world data set from a gene perturbation experiment (Section 6.3). In Sections 6.1 and 6.2 we consider a setting with two environments: an observational environment ( $E = 0$ ) and an intervention environment ( $E = 1$ ), and examine how the strength and number of interventions affect the performance of IAS.

# 6.1 Oracle IAS in Random Graphs

For the oracle setting, we know that SIAS ⊆ AN $Y$ (Proposition 3.3) and $S _ { \mathrm { I C P } } \subseteq S _ { \mathrm { I A S } }$ (Proposition 3.4). We first verify that the inclusion $S _ { \mathrm { I C P } } \subseteq S _ { \mathrm { I A S } }$ is often strict in lowdimensional settings when there are few interventions. Second, we show that the set $S _ { \mathrm { I A S } } ^ { m }$ is often strictly larger than the set $S _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ in large, sparse graphs with few interventions.

In principle, for a given number of covariates, one can enumerate all DAGs and, for each DAG, compare $S _ { \mathrm { I C P } }$ and $S _ { \mathrm { { I A S } } }$ . However, because the space of DAGs grows superexponentially in the number of nodes (Chickering, 2002), this is infeasible. Instead, we sample graphs from the space of all DAGs that satisfy Assumption 2.1 and $Y \in \mathrm { D E } _ { E }$ $E ^ { \prime }$ (see Appendix E.1 for details).

In the low-dimensional setting ( $d \leq 2 0$ ), we compute $S _ { \mathrm { I C P } }$ and $S _ { \mathrm { { I A S } } }$ , whereas in the larger graphs ( $d \geq 1 0 0$ ), we compute $S _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ and the reduced set $S _ { \mathrm { I A S } } ^ { m }$ for $m \in \{ 1 , 2 \}$ when $d = 1 0 0$ and for $m = 1$ when $d = 1$ ,000. Because there is no guarantee that IAS outputs a superset of ICP when searching only up to sets of some size lower than $d$ , we compare the size of the sets output by either method. For the low-dimensional setting, we consider both sparse and dense graphs, but for larger dimensions, we only consider sparse graphs. In the sparse setting, the DAGs are constructed such that there is an expected number of $d + 1$ edges between the $d + 1$ nodes $X$ and $Y$ ; in the dense setting, the expected number of edges equals $0 . 7 5 \cdot d ( d + 1 ) / 2$ .

The results of the simulations are displayed in Figures 2 and 3. In the low-dimensional setting, $S _ { \mathrm { { I A S } } }$ is a strict superset of $S _ { \mathrm { I C P } }$ for many graphs. This effect is the more pronounced, the larger the $d$ and the fewer nodes are intervened on, see Figure 2. In fact, when there are interventions on all predictors, we know that $S _ { \mathrm { I A S } } = S _ { \mathrm { I C P } } = \mathrm { P A } _ { Y }$ (Peters et al., 2016, Theorem 2), and thus the probability that $S _ { \mathrm { I C P } } \subsetneq S _ { \mathrm { I A S } }$ is exactly zero. For the larger graphs, we find that the set $S _ { \mathrm { I A S } } ^ { m }$ is, on average, larger than $S _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ , in particular when $d = 1$ ,000 or when $m = 2$ , see Figure 3. In the setting with $d = 1 0 0$ and $m = 1$ , the two sets are roughly the same size, when $1 0 \%$ of the predictors are intervened on. The set $S _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ becomes larger than $S _ { \mathrm { I A S } } ^ { 1 }$ after roughly $1 5 \%$ of the predictors nodes are intervened on (not shown). For both $d = 1 0 0$ and d = 1,000, the average size of the Markov boundary of $Y$ was found to be approximately 3.5.

# 6.2 Simulated Linear Gaussian SCMs

In this experiment, we show through simulation that IAS finds more ancestors than ICP in a finite sample setting when applied to linear Gaussian SCMs. To compare the outputs of IAS and ICP, we use the Jaccard similarity between $\hat { S } _ { \mathrm { I A S } }$ ( $\hat { S } _ { \mathrm { I A S } } ^ { 1 }$ when $d$ is large) and AN $Y$ and between $\hat { S } _ { \mathrm { I C P } }$ ( $\hat { S } _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ when $d$ is large5) and AN $Y$ . 6

We sample data from sparse linear Gaussian models with i.i.d. noise terms in two scenarios, $d = 6$ and $d = 1 0 0$ . In both cases, coefficients for the linear assignments are drawn randomly. We consider two environments; one observational and one interventional; in the interventional environment, we apply do-interventions of strength one to children of $E$ , i.e.,

![](images/3da73bb12f8f9bd21f3fa75eee1d31ef8aa99c153e1a1334b877dddb980cc0fb.jpg)

![](images/9b588e9588bf2ec0f2d265dc107bb3a0490a6543943d363ef80b92dcc9855b61.jpg)  
Proportion of predictors intervened on (%)   
Figure 2: Low-dimensional oracle experiment, see Section 6.1. In all cases, as predicted by theory, $S _ { \mathrm { I C P } }$ is contained in $S _ { \mathrm { { I A S } } }$ . For many graphs, $S _ { \mathrm { { I A S } } }$ is strictly larger than $S _ { \mathrm { I C P } }$ . On average, this effect is more expressed when there are fewer intervened nodes. $\mathbb { P } _ { n }$ refers to the distribution used to sample graphs and every point in the figure is based on 50,000 independently sampled graphs; $d$ denotes the number of covariates $X$ . Empirical confidence bands are plotted around each line, but are very narrow.

we fix the value of a child of $E$ to be one. We standardize the data along the causal order, to prevent variance accumulation along the causal order (Reisach et al., 2021). Throughout the section, we consider a significance level of $\alpha = 5 \%$ . For a detailed description of the simulations, see Appendix E.2.

To test for invariance, we employ the test used in Peters et al. (2016): We calculate a $p$ -value for the hypothesis of invariance of $S$ by first linearly regressing $Y$ onto $X _ { S }$ (ignoring $E$ ), and second testing whether the mean and variance of the prediction residuals is equal across environments. For details, see Peters et al. (2016, Section 3.2.1). Schultheiss et al. (2021) also consider the task of estimating ancestors but since their method is uninformative for Gaussian data and does not consider environments, it is not directly applicable here.

In Theorem 5.2, we assume asymptotic power of our invariance test. When $d = 6$ , we test hypotheses with a correction factor $C = 3 ^ { \left\lceil 6 / 3 \right\rceil } = 9$ , as suggested in Appendix C, in an attempt to reduce false positive findings. In Appendix E.3, we repeat the experiment of this section with $C = 2 ^ { 6 }$ and find almost identical results. We hypothesize, that the effects of a reduced $C$ is more pronounced at larger $d$ . When $d = 1 0 0$ , we test hypotheses with the correction factor $C ( 1 )$ of Theorem 5.5. In both cases, we test the hypothesis of invariance of the empty set at level $\alpha _ { 0 } = 1 0 ^ { - 6 }$ (cf. Remark 5.3). In Appendix E.4, we investigate the effects on the quantities $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } \subseteq \mathbf { A N } _ { Y } )$ and $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } ^ { 1 } \subseteq \mathrm { A N } _ { Y } )$ when varying $\alpha _ { 0 }$ , confirming that choosing $\alpha _ { 0 }$ too high can lead to a reduced probability of $\hat { S } _ { \mathrm { I A S } }$ being a subset of ancestors.

In Figure 4 the results of the simulations are displayed. In SCMs where the oracle versions $S _ { \mathrm { { I A S } } }$ and $S _ { \mathrm { I C P } }$ are not equal, $\hat { S } _ { \mathrm { I A S } }$ achieved, on average, a higher Jaccard similarity to AN than $\hat { S } _ { \mathrm { I C P } }$ . This effect is less pronounced when $d = 1 0 0$ . We believe that the $Y$ difference in Jaccard similarities is more pronounced when using larger values of $m$ . When $S _ { \mathrm { I A S } } = S _ { \mathrm { I C P } }$ , the two procedures achieve roughly the same Jaccard similarities to AN $Y$ , as

![](images/848fb62c037b9a495bed4337febcac30a12ac9b84d620c2f080b38cc2ef00d31.jpg)

![](images/e0e5ebafebb4c434784cdfdda34485a1bd55bc987bc35fe2d12dffa8e846fc60.jpg)  
Proportion of predictors intervened on $( \% )$   
Figure 3: High-dimensional oracle experiment with sparse graphs, see Section 6.1. The average size of the set $S _ { \mathrm { I A S } } ^ { m }$ is larger than the average size of the set $S _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ , both when using IAS to search for sets up to sizes $m = 1$ and $m = 2$ . Except for the choice of $d$ , the setup is the same as in Figure 2.

expected. When the number of observations is one hundred, IAS generally fails to find any ancestors and outputs the empty set (see Figure 7), indicating that the we do not have power to reject the empty set when there are few observations. This is partly by design; we test the empty set for invariance at reduced level $\alpha _ { 0 }$ in order to protect against making false positive findings when the environment has a weak effect on $Y$ . However, even without testing the empty set at a reduced level, IAS has to correct for making multiple comparisons, contrary to ICP, thus lowering the marginal significance level each set is tested at. When computing the jaccard similarities with either $\alpha _ { 0 } = \alpha$ or $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ , the results were similar (not shown). We repeated the experiments with $d = 6$ with a weaker influence of the environment (do-interventions of strength 0.5 instead of 1) and found comparable results, with slightly less power in that the empty set is found more often, see Appendix E.5.

We compare our method with a variant, called IASest. graph, where we first estimate (e.g., using methods proposed by Mooij et al. 2020 or Squires et al. 2020) a member graph of the Markov equivalence class (‘I-MEC’) and apply the oracle algorithm from Section 4 (by reading of d-separations in that graph) to estimate $\mathcal { M T }$ . In general, however, such an approach comes with additional assumptions; furthermore, even in the linear setup considered here, its empirical performance for large graphs is worse than the proposed method IAS, see Appendix E.7.

# 6.3 IAS in High Dimensional Genetic Data

We evaluate our approach in a data set on gene expression in yeast Kemmeren et al. (2014). The data contain full-genome mRNA expressions of $d = 6$ ,170 genes and consists of $n _ { \mathrm { o b s } } = 1 6 0$ unperturbed observations ( $E = 0$ ) and $n _ { \mathrm { i n t } } = 1$ ,479 intervened-upon observations ( $E = 1$ ); each of the latter observations correspond to the deletion of a single (known) gene. For each response gene $\mathsf { g e n e } _ { Y } \in [ d ]$ , we apply the procedure from Section 5.3 with $m = 1$ to search for ancestors.

We first test for invariance of the empty set, i.e., whether the distribution of geneY differs

![](images/3ecfd6146583a7f71c1cd865009db588a9520795756f136e128c72b7926524e5.jpg)

![](images/dd2bf6aa9ac573141e98203b2c27329f1786cc6cc3232b29ad231d1cdb3e2961.jpg)

![](images/7f0f0fcbb6bfd8ee4f68de47d89e187e077219b13007494e124d1d62275d3d18.jpg)

![](images/64b302d90fe14f048f7b150c074c5ffd7376240fac04adb7404dbfbb8d08da06.jpg)  
Figure 4: Comparison between the finite sample output of IAS and ICP and ANY on either simulated data, see Section 6.2. The plots show the Jaccard similarities between AN $\hat { S } _ { \mathrm { I A S } }$ ( $\hat { S } _ { \mathrm { I A S } } ^ { 1 }$ when $d = 1 0 0$ ) in red or $\hat { S } _ { \mathrm { I C P } }$ ( $\hat { S } _ { \mathrm { I C P } } ^ { \mathrm { M B } }$ when $d = 1 0 0$ ) in blue and AN $Y$ and $Y$ When $S _ { \mathrm { I C P } } \neq S _ { \mathrm { I A S } }$ (left column), $ { \hat { S } } _ { \mathrm { I A S } }$ is more similar to AN $Y$ than $\hat { S } _ { \mathrm { I C P } }$ . The procedures are roughly equally similar to AN $Y$ when $S _ { \mathrm { I C P } } = S _ { \mathrm { I A S } }$ (right column). Graphs represented in each boxplot: 42 (top left), 58 (top right), 40 (bottom left) and 60 (bottom right).

between the observational and interventional environment. We test this at a conservative level $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ in order to protect against a high false positive rate (see Remark 5.3). For 3,631 out of 6,170 response genes, the empty set is invariant, and we disregard them as response genes.

For each response gene, for which the empty set is not invariant, we apply our procedure. More specifically, when testing whether $\mathtt { g e n e } _ { X }$ is an ancestor of $\mathtt { g e n e } _ { Y }$ , we exclude any observation in which either $\mathtt { g e n e } _ { X }$ or ${ \tt g e n e } _ { Y }$ was intervened on. We then test whether the empty set is still rejected, at level $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ , and whether $\mathtt { g e n e } _ { X }$ is invariant at level $\alpha = 0 . 2 5$ . Since a set $\{ { \mathsf { g e n e } } _ { X } \}$ is deemed minimally invariant if the $p$ -value exceeds $\alpha$ , setting $\alpha$ large is conservative for the task of finding ancestors. Indeed, when estimating $\hat { S } _ { \mathrm { I A S } } ^ { m }$ , one can test the sets of size ly rejecting a minimally invari $m$ at a highert set of size evel  doe $\alpha _ { 1 } > \alpha$ . This is conservareak the inclusion $m$ $\hat { S } _ { \mathrm { I A S } } ^ { m } \subseteq \mathrm { A N } _ { Y }$ However, if one has little power against the non-invariant sets of size $m$ , testing at level $\alpha _ { 1 }$ can protect against false positives.7

![](images/466bd3cc03c2323440cf3771f9b4ed4d2a98a5a6053dc08c7e94b43d0b364f91.jpg)  
Figure 5: True positive rates and number of gene pairs found in the experiment in Section 6.3. On the $x$ -axis, we change $\alpha _ { 0 }$ , the threshold for invariance of the empty set. When $\alpha _ { 0 }$ is small, we only search for pairs if the environment has a very significant effect on $Y$ . For smaller $\alpha _ { 0 }$ , fewer pairs are found to be invariant (blue line), but those found, are more likely to be true positives (red line). This supports the claim that the lower $\alpha _ { 0 }$ is, the more conservative our approach is.

We use the held-out data point, where $\mathtt { g e n e } _ { X }$ is intervened on, to determine as ground truth, whether $\mathtt { g e n e } _ { X }$ is indeed an ancestor of $\mathtt { g e n e } _ { Y }$ . We define $\mathtt { g e n e } _ { X }$ as a true ancestor of $\mathtt { g e n e } _ { Y }$ if the value of $\mathtt { g e n e } _ { Y }$ when ${ \tt g e n e } _ { X }$ is intervened on, lies in the $q _ { T P } = 1 \%$ tails of the observational distribution of $\mathtt { g e n e } _ { Y }$ .

We find 23 invariant pairs $( \mathtt { g e n e } _ { X } , \mathtt { g e n e } _ { Y } )$ ; of these, 7 are true positives. In comparison, Peters et al. (2016) applies ICP to the same data, and with the same definition of true positives. They predict 8 pairs, of which 6 are true positives. This difference is in coherence with the motivation put forward in Section 5.2: Our approach predicts many more ancestral pairs (8 for ICP compared to 23 for IAS). Since ICP does not depend on power of the test, they have a lower false positive rate ( $2 5 \%$ for ICP compared to $6 9 . 6 \%$ for IAS).

In Figure 5, we explore how changing $\alpha _ { 0 }$ and $q _ { T P }$ impacts the true positive rate. Reducing $\alpha _ { 0 }$ increases the true positive rate, but lowers the number of gene pairs found (see Figure 5). This is because a lower $\alpha _ { 0 }$ makes it more difficult to detect non-invariance of the empty set, making the procedure more conservative (with respect to finding ancestors); see Remark 5.3. For example, when $\alpha _ { 0 } \leq 1 0 ^ { - 1 5 }$ , the true positive rate is above 0.8; however, 5 or fewer pairs are found. When searching for ancestors, the effect of intervening may be reduced by noise from intermediary variables, so $q _ { T B } = 1 \%$ might be too strict; in Appendix E.6, we analyze the impact of increasing $q _ { T B }$ .

# 7 Extensions

# 7.1 Latent variables

In Assumption 2.1, we assume that all variables $X$ are observed and that there are no hidden variables $H$ . Let us write $X = X _ { O } { \dot { \cup } } X _ { H }$ , where only $X _ { O }$ is observed and define ${ \mathcal { T } } : = \{ S \subseteq X _ { O } \mid S$ invariant}. We can then define

$$
S_{\text{IAS},O}:= \bigcup_{\substack{S\subseteq X_{O}:H_{0,S}^{\mathcal{M}\mathcal{I}}\text{true}}}S
$$

(again with the convention that a union over the empty set is the empty set), and have the following modification of Proposition 3.3.

# Proposition 7.1. It holds that $S _ { \mathrm { I A S } , O } \subseteq \mathrm { A N } _ { Y }$ .

All results in this paper remain correct in the presence of hidden variables, except for Proposition 3.4 and Proposition 5.4 (iii-iv).8 Thus, the union of the observed minimally invariant sets, SIAS,O is a subset of AN $Y$ and can be learned from data in the same way as if no latent variables were present.

# 7.2 Non-exogenous environments

Throughout this paper, we have assumed that the environment variable is exogenous (Assumption 2.1). However, all of the results stated in this paper, except for Proposition 4.1, also hold under the alternative assumption that $E$ is an ancestor of $Y$ , but not necessarily exogenous. From the remaining results, only the proof of Proposition 3.2 uses exogeneity of $E$ , but here the result follows from Tian et al. (1998). In all other proofs, we account for both options. This extension also remains valid in the presence of hidden variables, using the same arguments as in Section 7.1.

# 8 Conclusion and Future Work

Invariant Ancestry Search (IAS) provides a framework for searching for causal ancestors of a response variable $Y$ through finding minimally invariant sets of predictors by exploiting the existence of exogenous heterogeneity. The set $S _ { \mathrm { { I A S } } }$ is a subset of the ancestors of $Y$ , a superset of $S _ { \mathrm { I C P } }$ and, contrary to $S _ { \mathrm { I C P } }$ , invariant itself. Furthermore, the hierarchical structure of minimally invariant sets allows IAS to search for causal ancestors only among subsets up to a predetermined size. This avoids exponential runtime and allows us to apply the algorithm to large systems. We have shown that, asymptotically, $S _ { \mathrm { { I A S } } }$ can be identified from data with high probability if we are provided with a test for invariance that has asymptotic level and power. We have validated our procedure both on simulated and real data. Our proposed framework would benefit from further research in the maximal number

of minimally invariant sets among graphs of a fixed size, as this would provide larger finite sample power for identifying ancestors. Further it is of interest to establish finite sample guarantees or convergence rates for IAS, possibly by imposing additional assumptions on the class of SCMs. Finally, even though current implementations are fast, it is an open theoretical question whether computing $S _ { \mathrm { { I A S } } }$ in the oracle setting of Section 4 is NP-hard, see Appendix B.

# Acknowledgements

NT and JP were supported by a research grant (18968) from VILLUM FONDEN.

# References

Acid, S. and De Campos, L. M. An algorithm for finding minimum d-separating sets in belief networks. In Proceedings of the 29th Annual Conference on Uncertainty in Artificial Intelligence (UAI), 2013.   
Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-Paz, D. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.   
Berrett, T. B., Wang, Y., Barber, R. F., and Samworth, R. J. The conditional permutation test for independence while controlling for confounders. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(1):175–197, 2020.   
Bongers, S., Forr´e, P., Peters, J., and Mooij, J. M. Foundations of structural causal models with cycles and latent variables. Annals of Statistics, 49(5):2885–2915, 2021.   
Chickering, D. M. Optimal structure identification with greedy search. Journal of Machine Learning Research, 3:507–554, 2002.   
Christiansen, R., Pfister, N., Jakobsen, M. E., Gnecco, N., and Peters, J. A causal framework for distribution generalization. IEEE Transactions on Pattern Analysis and Machine Intelligence (accepted), 2022.   
Friedman, J. H. Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5):1189–1232, 2001.   
Fukumizu, K., Gretton, A., Sun, X., and Sch¨olkopf, B. Kernel measures of conditional dependence. In Advances in Neural Information Processing Systems (NeurIPS), volume 20, 2008.   
Gamella, J. L. and Heinze-Deml, C. Active invariant causal prediction: Experiment selection through stability. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, 2020.   
Gaspers, S. and Mackenzie, S. On the number of minimal separators in graphs. In International Workshop on Graph-Theoretic Concepts in Computer Science, pp. 116–121. Springer, 2015.   
Heinze-Deml, C., Peters, J., and Meinshausen, N. Invariant causal prediction for nonlinear models. Journal of Causal Inference, 6(2), 2018.   
Hoyer, P., Janzing, D., Mooij, J. M., Peters, J., and Sch¨olkopf, B. Nonlinear causal discovery with additive noise models. In Advances in Neural Information Processing Systems (NeurIPS), volume 21, 2009.   
Jordan, M. I. Artificial intelligence — the revolution hasn’t happened yet. Harvard Data Science Review, 1(1), 2019.

Kemmeren, P., Sameith, K., Van De Pasch, L. A., Benschop, J. J., Lenstra, T. L., Margaritis, T., O’Duibhir, E., Apweiler, E., van Wageningen, S., Ko, C. W., et al. Large-scale genetic perturbations reveal regulatory networks and an abundance of gene-specific repressors. Cell, 157(3):740–752, 2014.   
Lauritzen, S. L. Graphical models. Clarendon Press, 1996.   
Magliacane, S., van Ommen, T., Claassen, T., Bongers, S., Versteeg, P., and Mooij, J. M. Domain adaptation by using causal inference to predict invariant conditional distributions. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 31, pp. 10846–10856. Curran Associates, Inc., 2018.   
Martinet, G., Strzalkowski, A., and Engelhardt, B. E. Variance minimization in the Wasserstein space for invariant causal prediction. arXiv preprint arXiv:2110.07064, 2021.   
Meinshausen, N. and B¨uhlmann, P. High-dimensional graphs and variable selection with the lasso. Annals of Statistics, 34(3):1436–1462, 2006.   
Mooij, J. M., Magliacane, S., and Claassen, T. Joint causal inference from multiple contexts. Journal of Machine Learning Research, 21(99):1–108, 2020. URL http://jmlr.org/ papers/v21/17-123.html.   
Pearl, J. Causality. Cambridge university press, 2009.   
Pearl, J. Probabilistic reasoning in intelligent systems: networks of plausible inference. The Morgan Kaufmann series in representation and learning. Morgan Kaufmann, 2014.   
Pearl, J. Theoretical impediments to machine learning with seven sparks from the causal revolution. arXiv preprint arXiv:1801.04016, 2018.   
Peters, J., Mooij, J. M., Janzing, D., and Sch¨olkopf, B. Causal discovery with continuous additive noise models. Journal of Machine Learning Research, 15:2009–2053, 2014.   
Peters, J., B¨uhlmann, P., and Meinshausen, N. Causal inference by using invariant prediction: identification and confidence intervals. Journal of the Royal Statistical Society. Series B (Statistical Methodology), pp. 947–1012, 2016.   
Pfister, N., B¨uhlmann, P., and Peters, J. Invariant causal prediction for sequential data. Journal of the American Statistical Association, 114(527):1264–1276, 2019.   
R Core Team. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria, 2021. URL https://www.R-project.org/.   
Reisach, A., Seiler, C., and Weichwald, S. Beware of the simulated DAG! Causal discovery benchmarks may be easy to game. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, 2021.   
Rojas-Carulla, M., Sch¨olkopf, B., Turner, R., and Peters, J. Invariant models for causal transfer learning. The Journal of Machine Learning Research, 19(1):1309–1342, 2018.

Schultheiss, C., B¨uhlmann, P., and Yuan, M. Higher-order least squares: assessing partial goodness of fit of linear regression. arXiv preprint arXiv:2109.14544, 2021.   
Shah, R. D. and Peters, J. The hardness of conditional independence testing and the generalised covariance measure. Annals of Statistics, 48(3):1514–1538, 2020.   
Shimizu, S., Hoyer, P. O., Hyv¨arinen, A., Kerminen, A., and Jordan, M. A linear non-Gaussian acyclic model for causal discovery. Journal of Machine Learning Research, 7 (10), 2006.   
Spirtes, P., Glymour, C. N., Scheines, R., and Heckerman, D. Causation, prediction, and search. MIT press, 2000.   
Squires, C., Wang, Y., and Uhler, C. Permutation-based causal structure learning with unknown intervention targets. In Conference on Uncertainty in Artificial Intelligence, pp. 1039–1048. PMLR, 2020.   
Takata, K. Space-optimal, backtracking algorithms to list the minimal vertex separators of a graph. Discrete Applied Mathematics, 158:1660–1667, 2010.   
Textor, J., van der Zander, B., Gilthorpe, M. S., Li´skiewicz, M., and Ellison, G. T. Robust causal inference using directed acyclic graphs: the R package ‘dagitty’. International Journal of Epidemiology, 45(6):1887–1894, 2016.   
Thams, N., Saengkyongam, S., Pfister, N., and Peters, J. Statistical testing under distributional shifts. arXiv preprint arXiv:2105.10821, 2021.   
Tian, J., Paz, A., and Pearl, J. Finding minimal d-separators. Technical report, University of California, Los Angeles, 1998.   
Tibshirani, R. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society, Series B, 58:267–288, 1996.   
van der Zander, B., Li´skiewicz, M., and Textor, J. Separators and adjustment sets in causal graphs: Complete criteria and an algorithmic framework. Artificial Intelligence, 270:1–40, 2019.   
Zhang, K., Peters, J., Janzing, D., and Sch¨olkopf, B. Kernel-based conditional independence test and application in causal discovery. In Proceedings of the 27th Annual Conference on Uncertainty in Artificial Intelligence (UAI), pp. 804–813, 2011.   
Zheng, X., Aragam, B., Ravikumar, P. K., and Xing, E. P. DAGs with NO TEARS: Continuous optimization for structure learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, 2018.

# A Proofs

# A.1 A direct Proof of Proposition 3.2

Proof. Assume that $E$ is exogenous. If $E \in \mathrm { P A } _ { Y }$ , then there are no minimally invariant sets, and the statement holds trivially. If $E \notin \mathrm { P A } _ { Y }$ , then assume for contradiction, that an invariant set $S _ { 0 } \subsetneq S$ exists. By assumption, $| S \setminus S _ { 0 } | > 1$ , because otherwise $S _ { 0 }$ would be non-invariant.

We can choose $S _ { 1 } \subseteq S$ and $k _ { 0 } , k _ { 1 } , \ldots , k _ { l } \in S$ with $l \geq 1$ such that for all $i = 1 , \ldots , l :$ $k _ { i } \notin \mathrm { D E } _ { k _ { 0 } }$ and

$$
S _ {0} \cup S _ {1} \cup \{k _ {0}, \dots , k _ {l} \} = S \quad \in \mathcal {I}
$$

$$
\text {f o r} 0 \leq i <   l: S _ {0} \cup S _ {1} \cup \{k _ {0}, \ldots , k _ {i} \} \quad \notin \mathcal {I}
$$

$$
S _ {0} \cup S _ {1} \quad \in \mathcal {I}.
$$

This can be done by iteratively removing elements from $S \setminus S _ { 0 }$ , removing first the earliest elements in the causal order. The first invariant set reached in this process is then $S _ { 0 } \cup S _ { 1 }$ .

Since $S _ { 0 } \cup S _ { 1 } \cup \{ k _ { 0 } \}$ is non-invariant, there exists a path $\pi$ between $E$ and $Y$ that is open given $S _ { 0 } \cup S _ { 1 } \cup \{ k _ { 0 } \}$ but blocked given $S _ { 0 } \cup S _ { 1 }$ . Since removing $k _ { 0 }$ blocks $\pi$ , $k _ { 0 }$ must be a collider or a descendant of a collider $c$ on $\pi$ :

![](images/4f6dfa10729d4739120da5b37d3067940e41cbc4c1b45e19eb93c153113ad676.jpg)

Here, $-$ represents an edge that either points left or right. Since $\pi$ is open given $S _ { 0 } \cup S _ { 1 }$ , the two sub-paths $\pi _ { E }$ and $\pi _ { Y }$ are open given $S _ { 0 } \cup S _ { 1 }$ .

Additionally, since $S _ { 0 } \cup S _ { 1 } \cup \{ k _ { 1 } , \ldots , k _ { l } \} = S \setminus \{ k _ { 0 } \}$ is non-invariant, there exists a path $\tau$ between $E$ and $Y$ that is unblocked given $S _ { 0 } \cup S _ { 1 } \cup \{ k _ { 1 } , . . . , k _ { l } \}$ and blocked given $S _ { 0 } \cup S _ { 1 } \cup \{ k _ { 1 } , . . . , k _ { l } \} \cup \{ k _ { 0 } \}$ . It follows that $k _ { 0 }$ lies on $\tau$ (otherwise $\tau$ cannot be blocked by adding $k _ { 0 }$ ) and $k _ { 0 }$ has at least one outgoing edge. Assume, without loss of generality that there is an outgoing edge towards $Y$ . Since $\tau$ is open given $S _ { 0 } \cup S _ { 1 } \cup \{ k _ { 1 } , . . . , k _ { l } \}$ , so is ${ \cal T } _ { Y }$ .

![](images/f9d25911a5be41eb303ebf2b915f46cfb406f9401dfa75c189694874df489deb.jpg)

the path If there are no colliders on $E : \{ : \dots \} \ : c  \cdot \cdot \cdot  k _ { 0 }  ^ { \overset { \pi _ { E } } { \dots } } \cdot \cdot \cdot$ $\tau _ { Y }$ , then is a $\tau _ { Y }$ is also open given open given $S _ { 0 } \cup S _ { 1 }$ $S _ { 0 } \cup S _ { 1 }$ ∪ , contradicting invariance of . But then the path $S _ { 0 } \cup S _ { 1 }$ .

If there are colliders on $\tau _ { Y }$ , let $m$ be the collider closest to $k _ { 0 }$ , meaning that $m \in \mathrm { D E } _ { k _ { 0 } }$ Since $\tau _ { Y }$ is open given $S _ { 0 } { \cup } S _ { 1 } { \cup } \{ k _ { 1 } , . . . , k _ { l } \}$ , it means that either $m$ or a descendant of $m$ is in

![](images/b3e0387eab14133302d9814e03c2c730825fdf73947b496deeac773a1d5bda35.jpg)

$S _ { 0 } \cup S _ { 1 } \cup \{ k _ { 1 } , . . . , k _ { l } \}$ . Since $\{ k _ { 1 } , \dots , k _ { l } \} \cap \mathrm { D E } _ { k _ { 0 } } = \emptyset$ , there exist $v \in ( S _ { 0 } \cup S _ { 1 } ) \cap ( \{ m \} \cup \mathrm { D E } _ { m } )$ . But then $v \in { \mathrm { D E } } _ { k _ { 0 } } \cap ( S _ { 0 } \cup S _ { 1 } )$ , meaning that $\pi$ is open given $S _ { 0 } \cup S _ { 1 }$ , contradicting invariance of $S _ { 0 } \cup S _ { 1 }$ .

We could assume that $T Y$ had an outgoing edge from $k _ { 0 }$ without loss of generality, because if there was instead an outgoing edge from $k _ { 0 }$ on $\tau _ { E }$ , the above argument would work with $\pi _ { Y }$ and $\tau _ { E }$ instead. This concludes the proof.

# A.2 A direct proof of Proposition 3.3

Proof. If $E$ is a parent of $Y$ , we have $\mathcal { M } \mathcal { T } = \emptyset$ and the statement follows trivially. Thus, assume that $E$ is not a parent of $Y$ . We will show that if $S \in \mathcal { Z }$ is not a subset of AN $Y$ , then $S ^ { * } : = S \cap \mathrm { A N } _ { Y } \in \mathcal { I }$ , meaning that $S \notin \mathcal { M } \mathcal { T }$ .

Assume for contradiction that there is a path $p$ between $E$ and $Y$ that is open given $S ^ { * }$ . Since $S \in \mathcal { Z }$ , $p$ is blocked given $S$ . Then there exists a non-collider $Z$ on $p$ that is in $S \setminus \mathrm { A N } _ { Y }$ . We now argue that all nodes on $p$ are ancestors of $Y$ , yielding a contradiction.

First, assume that there are no colliders on $p$ . If $E$ is exogenous, then $p$ is directed from $E$ to $Y$ . (If $E$ is an ancestor of $Y$ , any node on is either an ancestor of $Y$ or $E$ , and thus $p$ $Y$ .) Second, assume that there are colliders on $p$ . Since $p$ is open given the smaller set $S ^ { * } \subsetneq S$ , all colliders on $p$ are in $S ^ { * }$ or have a descendant in $S ^ { * }$ ; therefore all colliders are ancestors of $Y$ . If $E$ is exogenous, any node on $p$ is either an ancestor of $Y$ or of a collider on $p$ . (If $E$ is an ancestor of $Y$ , any node on $p$ is either an ancestor of $Y$ , of a collider on $p$ or of $E$ , and thus also $Y$ .) This completes the proof of Proposition 3.3.

# A.3 Proof of Proposition 3.4

Proof. First, we show that $S _ { \mathrm { I A S } } \in \mathcal { I }$ . If $S _ { \mathrm { { I A S } } }$ is the union of a single minimally invariant set, it trivially holds that $S _ { \mathrm { I A S } } \in \mathcal { I }$ . Now assume that $S _ { \mathrm { { I A S } } }$ is the union of at least two minimally invariant sets, $S _ { \mathrm { I A S } } = S _ { 1 } \cup \ldots \cup S _ { n }$ , $n \geq 2$ , and assume for a contradiction that there exists a path $\pi$ between $E$ and $Y$ that is unblocked given $S _ { \mathrm { { I A S } } }$ .

Since $\pi$ is blocked by a strict subset of $S _ { \mathrm { { I A S } } }$ , it follows that $\pi$ has at least one collider; further every collider of $\pi$ is either in $S _ { \mathrm { { I A S } } }$ or has a descendant in $S _ { \mathrm { { I A S } } }$ , and hence every collider of $\pi$ is an ancestor of $Y$ , by Proposition 3.3. If $E$ is exogenous, $\pi$ has the following shape

![](images/fea5cf52acd4366897418272a274056e118e74eca134b62964e3d179a1f0d8a2.jpg)

(If $E$ is not exogenous but $E \in \operatorname { A N } _ { Y }$ $Y$ , then $\pi$ takes either the form displayed above or the

shape displayed below. However, no matter which of the shapes $\pi$ takes, the proof proceeds

$$
\begin{array}{c}\overbrace {E \leftarrow \cdots \rightarrow c _ {1} \leftarrow \cdots \rightarrow c _ {2} \leftarrow \cdots \longrightarrow c _ {k} \leftarrow \cdots \rightarrow Y}.\end{array}
$$

the same.) The paths $\pi _ { 1 } , \ldots , \pi _ { k + 1 }$ , $k \geq 1$ , do not have any colliders and are unblocked given $S _ { \mathrm { { I A S } } }$ . In particular, $\pi _ { 1 } , \ldots , \pi _ { k + 1 }$ are unblocked given $S _ { 1 }$ .

The path $\pi _ { k + 1 }$ must have a final edge pointing to $Y$ , because otherwise it would be a directed path from $Y$ to $c _ { k }$ , which contradicts acyclicity since $c _ { k }$ is an ancestor of $Y$ .

As $c _ { 1 }$ is an ancestor of $Y$ , there exists a directed path, say $\rho _ { 1 }$ , from $c _ { 1 }$ to $Y$ . Since $\pi _ { 1 }$ is open given $S _ { 1 }$ and since $S _ { 1 }$ is invariant, it follows that $\rho _ { 1 }$ must be blocked by $S _ { 1 }$ (otherwise the path $E \stackrel { \pi _ { 1 } } {  } c _ { 1 } \stackrel { \rho _ { 1 } } {  } Y$ would be open). For this reason, $S _ { 1 }$ contains a descendant of the collider $c _ { 1 }$ .

Similarly, if $\rho _ { 2 }$ is a directed path from $c _ { 2 }$ to $Y$ , then $S _ { 1 }$ blocks $\rho _ { 2 }$ , because otherwise the path $E \ { \stackrel { \pi _ { 1 } } { \to } } \ c _ { 1 } \gets \ { \stackrel { \pi _ { 2 } } { \dots } } \to c _ { 2 } \ { \stackrel { \rho _ { 2 } } { \to } } \ Y$ would be open. Again, for this reason, $S _ { 1 }$ contains a descendant of $c _ { 2 }$ .

Iterating this argument, it follows that $S _ { 1 }$ contains a descendant of every collider on $\pi$ and since $\pi _ { 1 } , \ldots , \pi _ { k + 1 }$ are unblocked by $S _ { 1 }$ , $\pi$ is open given $S _ { 1 }$ . This contradicts invariance of $S _ { 1 }$ and proves that $S _ { \mathrm { I A S } } \in \mathcal { I }$ .

We now show that $S _ { \mathrm { I C P } } \subseteq S _ { \mathrm { I A S } }$ with equality if and only if $S _ { \mathrm { I C P } } \in \mathcal { I }$ . First, $S _ { \mathrm { I C P } } \subseteq S _ { \mathrm { I A S } }$ because $S _ { \mathrm { { I A S } } }$ is a union of the minimally invariant sets, and $S _ { \mathrm { I C P } }$ is the intersection over all invariant sets. We now show the equivalence statement.

Assume first that $S _ { \mathrm { I C P } } \in \mathcal { I }$ . As $S _ { \mathrm { I C P } }$ is the intersection of all invariant sets, $S _ { \mathrm { I C P } } \in \mathcal { I }$ implies that there exists exactly one invariant set, that is contained in all other invariant sets. By definition, this means that there is only one minimally invariant set, and that this set is exactly $S _ { \mathrm { I C P } }$ . Thus, $S _ { \mathrm { I A S } } = S _ { \mathrm { I C P } }$ .

Conversely assume that $S _ { \mathrm { I C P } } \notin \mathbb { Z }$ . By construction, $S _ { \mathrm { I C P } }$ is contained in any invariant set, in particular in the minimally invariant sets. However, since $S _ { \mathrm { I C P } }$ is not invariant itself, this containment is strict, and it follows that $S _ { \mathrm { I C P } } \subsetneq S _ { \mathrm { I A S } }$ .

![](images/b5ed8fcb5167190faf54b3e4985ba1282da671a12d432967870fd56ac60ab8cb.jpg)

# A.4 Proof of Proposition 4.1

Proof. First we show $\mathrm { P A } _ { Y } \cap \left( \mathrm { C H } _ { E } \cup \mathrm { P A } ( \mathrm { A N } _ { Y } \cap \mathrm { C H } _ { E } ) \right) \subseteq S _ { \mathrm { I C P } }$ . If $j \in \mathrm { P A } _ { Y } \cap \mathrm { C H } _ { E }$ , any invariant set contains $j$ , because otherwise the path $E  j  Y$ is open. Similarly, if $j \in \mathrm { P A } _ { Y } \cap \mathrm { P A } ( \mathrm { A N } _ { Y } \cap \mathrm { C H } _ { E } )$ , any invariant set contains $j$ (there exists a node $j ^ { \prime }$ such that $E  j ^ { \prime }  \cdots  Y$ and $E  j ^ { \prime }  j  Y$ , and any invariant set $S$ must contain $j ^ { \prime }$ or one of its descendants; thus, it must also contain $j$ to ensure that the path $E \to j ^ { \prime }  j \to Y$ is blocked by $S$ .) It follows that for all invariant $S$ ,

$$
\mathrm {P A} _ {Y} \cap \left(\mathrm {C H} _ {E} \cup \mathrm {P A} \left(\mathrm {A N} _ {Y} \cap \mathrm {C H} _ {E}\right)\right) \subseteq S,
$$

such that

$$
\mathrm {P A} _ {Y} \cap \left(\mathrm {C H} _ {E} \cup \mathrm {P A} \left(\mathrm {A N} _ {Y} \cap \mathrm {C H} _ {E}\right)\right) \subseteq \bigcap_ {S \text {i n v a r i a n t}} S.
$$

To show $S _ { \mathrm { I C P } } \subseteq \mathrm { P A } _ { Y } \cap ( \mathrm { C H } _ { E } \cup \mathrm { P A } ( \mathrm { A N } _ { Y } \cap \mathrm { C H } _ { E } ) )$ , take any $j \not \in \mathrm { P A } _ { Y } \cap ( \mathrm { C H } _ { E } \cup \mathrm { P A } ( \mathrm { A N } _ { Y } \cap \mathrm { C H } _ { E } ) )$ . We argue, that an invariant set $S$ not containing $j$ exists, such that $j \not \in S _ { \mathrm { I C P } } = \lceil \rceil _ { S \mathrm { i n v a r i a n t } } S$ . If $j \not \in { \mathrm { P A } } _ { Y }$ , let $S = { \mathrm { P A } } _ { Y }$ , which is invariant. If $j \in \mathrm { P A } _ { Y }$ , define

$$
\bar {S} = \left(\mathrm {P A} _ {Y} \backslash \{j \}\right) \cup \mathrm {P A} _ {j} \cup \left(\mathrm {C H} _ {j} \cap \mathrm {A N} _ {Y}\right) \cup \mathrm {P A} \left(\mathrm {C H} _ {j} \cap \mathrm {A N} _ {Y}\right).
$$

Because $j \notin \mathrm { C H } _ { E }$ and $j \not \in \mathrm { P A } ( \mathrm { A N } _ { Y } \cap \mathrm { C H } _ { E } )$ , we have $E \not \in S$ . Also observe that $S \subseteq \mathrm { A N } _ { Y }$ . We show that any path between $E$ and $Y$ is blocked by $S$ , by considering all possible paths:

$\cdots { \bf j } ^ { \prime } \to { \bf Y }$ for $\mathbf { j } ^ { \prime } \neq \mathbf { j }$ : Blocked because $j ^ { \prime } \in \operatorname { P A } _ { Y } \setminus \{ j \}$ .

$\mathbf { \partial } \cdot \cdot \cdot \mathbf { v }  \mathbf { j }  \mathbf { Y }$ : Blocked because $v \in \mathrm { P A } _ { j } \subseteq S$ and $E \notin \mathrm { P A } _ { j }$ .

$\mathbf { \partial } \cdot \cdot \cdot \mathbf { v } \to \mathbf { c } \gets \mathbf { j } \to \mathbf { Y }$ and $\mathbf { c } \in \mathrm { A N } _ { \mathbf { Y } }$ : Blocked because $v \in \mathrm { P A } _ { j } ( \mathrm { C H } _ { j } \cap \mathrm { A N } _ { Y } )$ .

$\mathbf { \partial } \cdot \cdot \cdot \mathbf { v } \to \mathbf { c } \gets \mathbf { j } \to \mathbf { Y }$ and $\mathbf { c } \not \in \mathrm { A N } _ { \mathbf { Y } }$ : Blocked because $S \subseteq \mathrm { A N } _ { Y }$ , and since $c \notin \mathrm { A N } _ { Y }$ , $S \cap$ $\mathrm { D E } _ { c } = \emptyset$ and the path is blocked given $S$ because of the collider $c$ .

$\mathbf { \partial } \cdot \mathbf { \partial } \cdot \mathbf { \partial } \to \mathbf { c } \gets \dots \gets \mathbf { v } \gets \mathbf { j } \to \mathbf { Y }$ and c ∈ ANY: Blocked because $v \in \mathrm { A N } _ { c }$ $c$ and $c \in \mathrm { A N } _ { Y }$ , so $v \in { \mathrm { C H } } _ { j } \cap { \mathrm { A N } } _ { Y } \subseteq { \bar { S } }$ .

$\mathbf { \dots } \to \mathbf { c }  \dots  \mathbf { v }  \mathbf { j }  \mathbf { Y }$ and $\mathbf { c } \not \in \mathrm { A N } _ { \mathbf { Y } }$ : Same reason as for the case $\mathbf { \dot { \mathbf { \ell } } } \cdot \mathbf { \mu } \cdot \mathbf { v } \to \mathbf { c } \gets \mathbf { j } \to \mathbf { Y }$ and $\mathbf { c } \not \in \mathrm { A N } _ { \mathbf { Y } } { \mathrm { : } }$ ’.

$\dotsb \to \mathbf { c }  \dotsb  \mathbf { Y }$ Since $S \subseteq \mathrm { A N } _ { Y }$ , we must have $S \cap \mathrm { D E } _ { c } = \emptyset$ (otherwise this would create a directed cycle from $Y  \cdots  Y$ ). Hence the path is blocked given $S$ because of the collider $c$ .

Since there are no open paths from $E$ to $Y$ given $S$ , $S$ is invariant, and $S _ { \mathrm { I C P } } \subseteq S$ . Since $j \not \in S$ , it follows that $j \notin S _ { \mathrm { I C P } }$ . This concludes the proof.

# A.5 Proof of Theorem 5.1

Proof. Consider first the case where all marginal tests have pointwise asymptotic power and pointwise asymptotic level.

Pointwise asymptotic level: Let $\mathbb { P } _ { 0 } \in H _ { 0 , S } ^ { \mathcal { M } \mathcal { L } }$ . By the assumption of pointwise asymptotic level, there exists a non-negative sequence $( \epsilon _ { n } ) _ { n \in \mathbb { N } }$ such that $\begin{array} { r } { \operatorname* { l i m } _ { n \to \infty } \epsilon _ { n } = 0 } \end{array}$ and $\mathbb { P } _ { 0 } ( \phi _ { n } ( S ) = 1 ) \le \alpha + \epsilon _ { n }$ . Then

$$
\begin{array}{l} \mathbb {P} _ {0} (\phi_ {n} ^ {\mathcal {M I}} (S) = 1) = \mathbb {P} _ {0} \left((\phi_ {n} (S) = 1) \cup \bigcup_ {j \in S} (\phi_ {n} (S \setminus \{j \}) = 0)\right) \\ \leq \mathbb {P} _ {0} \left(\phi_ {n} (S) = 1\right) + \sum_ {j \in S} \mathbb {P} _ {0} \left(\phi_ {n} (S \setminus \{j \}) = 0\right) \\ \leq \alpha + \epsilon_ {n} + \sum_ {j \in S} \mathbb {P} _ {0} \left(\phi_ {n} (S \setminus \{j \}) = 0\right) \\ \rightarrow \alpha + 0 \quad \text {a s} n \rightarrow \infty \\ = \alpha . \\ \end{array}
$$

The convergence step follows from

$$
H _ {0, S} ^ {\mathcal {M I}} = H _ {0, S} ^ {\mathcal {I}} \cap \bigcap_ {j \in S} H _ {A, S \backslash \{j \}} ^ {\mathcal {I}}
$$

and from the assumption of pointwise asymptotic level and power. As $\mathbb { P } _ { 0 } \in H _ { 0 , S } ^ { \mathcal { M } \mathcal { L } }$ was arbitrary, this shows that $\phi _ { n } ^ { \mathcal { M } \mathcal { L } }$ has pointwise asymptotic level.

Pointwise asymptotic power: To show that the decision rule has pointwise asymptotic power, consider any $\mathbb { P } _ { A } \in H _ { A , S } ^ { \mathcal { M } \mathcal { L } }$ . We have that

$$
H _ {A, S} ^ {\mathcal {M I}} = H _ {A, S} ^ {\mathcal {I}} \cup \left(H _ {0, S} ^ {\mathcal {I}} \cap \bigcup_ {j \in S} H _ {0, S \backslash \{j \}} ^ {\mathcal {I}}\right). \tag {7}
$$

As the two sets $H _ { A , S } ^ { \mathcal { L } }$ and

$$
H _ {0, S} ^ {\mathcal {I}} \cap \bigcup_ {j \in S} H _ {0, S \backslash \{j \}} ^ {\mathcal {I}}
$$

are disjoint, we can consider them one at a time. Consider first the case $\mathbb { P } _ { A } \in H _ { A , S } ^ { \mathcal { L } }$ . This means that $S$ is not invariant and thus

$$
\begin{array}{l} \mathbb {P} _ {A} (\phi_ {n} ^ {\mathcal {M I}} (S) = 1) = \mathbb {P} _ {A} \left((\phi_ {n} (S) = 1) \cup \bigcup_ {j \in S} (\phi_ {n} (S \setminus \{j \}, \alpha) = 0)\right) \\ \geq \mathbb {P} _ {A} (\phi_ {n} (S) = 1) \\ \rightarrow 1 \quad \text {a s} n \rightarrow \infty \\ \end{array}
$$

by the assumption of pointwise asymptotic power.

Next, assume that there exists $j ^ { \prime } \in S$ such that $\mathbb { P } _ { A } \in ( H _ { 0 , S } ^ { \mathcal { I } } \cap H _ { 0 , S \backslash \{ j ^ { \prime } \} } ^ { \mathcal { I } } )$ . Then,

$$
\begin{array}{l} \mathbb {P} _ {A} (\phi_ {n} ^ {\mathcal {M I}} (S) = 1) = \mathbb {P} _ {0} \left((\phi_ {n} (S) = 1) \cup \bigcup_ {j \in S} (\phi_ {n} (S \setminus \{j \}) = 0)\right) \\ \geq \mathbb {P} _ {A} \left(\phi_ {n} \left(S \setminus \{j ^ {\prime} \}\right) = 0\right) \\ \geq 1 - \alpha - \epsilon_ {n} \\ \rightarrow 1 - \alpha \qquad \text {a s} n \rightarrow \infty . \\ \end{array}
$$

Thus, for arbitrary $\mathbb { P } _ { A } \in H _ { A , S } ^ { \mathcal { M } \mathcal { L } }$ we have shown that $\mathbb { P } _ { A } \left( \phi _ { n } ^ { \mathcal { M } 2 } ( S ) = 1 \right) \ge 1 - \alpha$ in the limit. This shows that $\phi _ { n } ^ { \mathcal { M } \mathcal { L } }$ has pointwise asymptotic power of at least $1 - \alpha$ . This concludes the argument for pointwise asymptotic power.

Next, consider the case that the marginal tests have uniform asymptotic power and uniform asymptotic level. The calculations for showing that $\phi _ { n } ^ { \mathcal { M } \mathcal { L } }$ has uniform asymptotic level and uniform asymptotic power of at least $1 - \alpha$ are almost identical to the pointwise calculations.

Uniform asymptotic level: By the assumption of uniform asymptotic level, there exists a non-negative sequence n such that limn→∞ n = 0 and supP∈HI0,S P $\epsilon _ { n }$ $\scriptstyle \operatorname* { l i m } _ { n \to \infty } \epsilon _ { n } = 0$ $\begin{array} { r } { \operatorname* { s u p } _ { \mathbb { P } \in H _ { 0 , S } ^ { \mathcal { T } } } \mathbb { P } ( \phi _ { n } ( S ) = 1 ) \le \alpha + \epsilon _ { n } } \end{array}$ . Then,

$$
\begin{array}{l} \sup  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {M I}}} \mathbb {P} (\phi_ {n} ^ {\mathcal {M I}} (S) = 1) = \sup  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {M I}}} \mathbb {P} \left((\phi_ {n} (S) = 1) \cup \bigcup_ {j \in S} (\phi_ {n} (S \setminus \{j \}) = 0)\right) \\ \leq \sup  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {M I}}} \left(\mathbb {P} \left(\phi_ {n} (S) = 1\right) + \sum_ {j \in S} \mathbb {P} \left(\phi_ {n} (S \setminus \{j \}) = 0\right)\right) \\ \leq \sup  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {M I}}} \mathbb {P} \left(\phi_ {n} (S) = 1\right) + \sum_ {j \in S} \sup  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {M I}}} \mathbb {P} \left(\phi_ {n} (S \setminus \{j \}) = 0\right) \\ \leq \alpha + \epsilon_ {n} + \sum_ {j \in S} \left(1 - \inf  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {M I}}} \mathbb {P} (\phi_ {n} (S \setminus \{j \}) = 1)\right) \\ \rightarrow \alpha + 0 + \sum_ {j \in S} (1 - 1) \quad \text {a s} n \rightarrow \infty \\ = \alpha . \\ \end{array}
$$

Uniform asymptotic power: From (7), it follows that

$$
\inf _ {\mathbb {P} \in H _ {A, S} ^ {\mathcal {M I}}} \mathbb {P} (\phi_ {n} ^ {\mathcal {M I}} (S) = 1) = \min \left\{\inf _ {\mathbb {P} \in H _ {A, S} ^ {\mathcal {I}}} \mathbb {P} (\phi_ {n} ^ {\mathcal {M I}} (S) = 1), \inf _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {I}} \cap \bigcup_ {j \in S} H _ {0, S \setminus \{j \}} ^ {\mathcal {I}}} \mathbb {P} (\phi_ {n} ^ {\mathcal {M I}} (S) = 1) \right\}.
$$

We consider the two inner terms in the above separately. First,

$$
\begin{array}{l} \inf  _ {\mathbb {P} \in H _ {A, S} ^ {\mathcal {I}}} \mathbb {P} \left(\phi_ {n} ^ {\mathcal {M I}} (S) = 1\right) = \inf  _ {\mathbb {P} \in H _ {A, S} ^ {\mathcal {I}}} \mathbb {P} \left(\left(\phi_ {n} (S) = 1\right) \cup \bigcup_ {j \in S} \left(\phi_ {n} (S \setminus \{j \}) = 0\right)\right) \\ \geq \inf_ {\mathbb {P} \in H _ {A, S} ^ {\mathcal {T}}} \mathbb {P} (\phi_ {n} (S) = 1) \\ \rightarrow 1 \quad \text {a s} n \rightarrow \infty . \\ \end{array}
$$

Next,

$$
\begin{array}{l} \inf_{\mathbb{P}\in H^{I}_{0,S}\cap \bigcup_{j\in S}H^{I}_{0,S\setminus \{j\}}}\mathbb{P}(\phi_{n}^{\mathcal{M}\mathcal{I}}(S) = 1) = \inf_{\mathbb{P}\in H^{I}_{0,S}\cap \bigcup_{j\in S}H^{I}_{0,S\setminus \{j\}}}\mathbb{P}\left((\phi_{n}(S) = 1)\cup \bigcup_{j\in S}(\phi_{n}(S\setminus \{j\}) = 0)\right) \\ = \min  _ {j \in S} \left\{\inf  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {I}} \cap H _ {0, S \setminus \{j \}} ^ {\mathcal {I}}} \mathbb {P} \left(\left(\phi_ {n} (S) = 1\right) \cup \bigcup_ {j \in S} \left(\phi_ {n} (S \setminus \{j \}) = 0\right)\right) \right\} \\ \geq \min  _ {j \in S} \left\{\inf  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {I}} \cap H _ {0, S \setminus \{j \}} ^ {\mathcal {I}}} \mathbb {P} (\phi_ {n} (S \setminus \{j \}) = 0) \right\} \\ = \min  _ {j \in S} \left\{1 - \sup  _ {\mathbb {P} \in H _ {0, S} ^ {\mathcal {I}} \cap H _ {0, S \backslash \{j \}} ^ {\mathcal {I}}} \mathbb {P} \left(\phi_ {n} (S \setminus \{j \}) = 1\right) \right\} \\ \geq 1 - \alpha - \epsilon_ {n} \\ \rightarrow 1 - \alpha \quad \text {a s} n \rightarrow \infty . \\ \end{array}
$$

This shows that $\phi _ { n } ^ { \mathcal { M L } }$ has uniform asymptotic power of at least $1 - \alpha$ , which completes the proof. □

# A.6 Proof of Theorem 5.2

Proof. We have that

$$
\lim  _ {n \rightarrow \infty} \mathbb {P} (\hat {S} _ {\mathrm {I A S}} \subseteq \mathrm {A N} _ {Y}) \geq \lim  _ {n \rightarrow \infty} \mathbb {P} (\hat {S} _ {\mathrm {I A S}} = S _ {\mathrm {I A S}})
$$

as $S _ { \mathrm { I A S } } \subseteq \mathrm { A N } _ { Y }$ by Proposition 3.4. Furthermore, we have

$$
\mathbb {P} \big (\hat {S} _ {\mathrm {I A S}} = S _ {\mathrm {I A S}} \big) \geq \mathbb {P} \big (\widehat {\mathcal {M I}} = \mathcal {M I} \big).
$$

Let $A : = \{ S \mid S \not \in { \mathcal { T } } \} \setminus \{ S \mid \exists S ^ { \prime } \subset S$ s.t. $S ^ { \prime } \in \mathcal { M } \mathbb { Z } \}$ be those non-invariant sets that do not contain a minimally invariant set and observe that

$$
\left(\widehat {\mathcal {M I}} = \mathcal {M I}\right) \supseteq \bigcap_ {S \in \mathcal {M I}} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 0\right) \cap \bigcap_ {S \in A} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 1\right). \tag {8}
$$

To see why this is true, note that to correctly recover $\mathcal { M T }$ , we need to 1) accept the hypothesis of minimal invariance for all minimally invariant sets and 2) reject the hypothesis of invariance for all non-invariant sets that are not supersets of a minimally invariant set (any superset of a set for which the hypothesis of minimal invariance is not rejected is removed in

the computation of $\widehat { \mathcal { M } \mathcal { I } }$ ). Then,

$$
\begin{array}{l} \mathbb {P} (\widehat {\mathcal {M I}} = \mathcal {M I}) \geq \mathbb {P} \left(\bigcap_ {S \in \mathcal {M I}} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 0\right) \cap \bigcap_ {S \in A} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 1\right)\right) \\ \geq 1 - \mathbb {P} \left(\bigcup_ {S \in \mathcal {M I}} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 1\right)\right) - \sum_ {S \in A} \mathbb {P} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 0\right) \\ \geq 1 - \sum_ {S \in \mathcal {M I}} \mathbb {P} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 1\right) - \sum_ {S \in A} \mathbb {P} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 0\right) \\ \geq 1 - \sum_ {S \in \mathcal {M I}} \left(\alpha C ^ {- 1} + \epsilon_ {n, S}\right) - \sum_ {S \in A} \mathbb {P} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 0\right) \\ \geq 1 - | \mathcal {M I} | \alpha C ^ {- 1} + \sum_ {S \in \mathcal {M I}} \epsilon_ {n, S} - \sum_ {S \in A} \mathbb {P} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 0\right) \\ \geq 1 - \alpha + \sum_ {S \in \mathcal {M I}} \epsilon_ {n, S} - \sum_ {S \in A} \mathbb {P} \left(\phi_ {n} (S, \alpha C ^ {- 1}) = 0\right) \\ \rightarrow 1 - \alpha \quad \text {a s} n \rightarrow \infty , \\ \end{array}
$$

where $( \epsilon _ { n , S } ) _ { n \in \mathbb { N } , S \in \mathcal { M T } }$ are non-negative sequences that converge to zero and the last step follows from the assumption of asymptotic power. The sequences $( \epsilon _ { n , S } ) _ { n \in \mathbb { N } , S \in \mathcal { M T } }$ exist by the assumption of asymptotic level. □

# A.7 Proof of Proposition 5.4

Proof. We prove the statements one by one.

(i) Since $S _ { \mathrm { I A S } } ^ { m }$ is the union over some of the minimally invariant sets, $S _ { \mathrm { I A S } } ^ { m } \subseteq S _ { \mathrm { I A S } }$ . Then the statement follows from Proposition 3.3.   
(ii) If $m \geq m _ { \mathrm { m a x } }$ , all $S \in \mathcal { M } \mathcal { T }$ satisfy the requirement $| S | \le m$ .   
(iii) If $m \geq m _ { \operatorname* { m i n } }$ , then $S _ { \mathrm { I A S } } ^ { m }$ contains at least one minimally invariant set. The statement then follows from the first part of the proof of Proposition 3.4 given in Appendix A.3.   
(iv) $S _ { \mathrm { I A S } } ^ { m }$ contains at least one minimally invariant set and, by (iii), it is itself invariant. Thus, if $S _ { \mathrm { I C P } } \notin \mathcal { I }$ , then $S _ { \mathrm { I C P } } \subsetneq S _ { \mathrm { I A S } } ^ { m }$ . If $S _ { \mathrm { I C P } } \in \mathcal { I }$ , then there exists only one minimally invariant set, which is $S _ { \mathrm { I C P } }$ IAS  (see proof of Proposition 3.4), and we have $S _ { \mathrm { I C P } } = S _ { \mathrm { I A S } } ^ { m }$ . This concludes the proof. □

# A.8 Proof of Theorem 5.5

Proof. The proof is identical to the proof of Theorem 5.2, when changing the correction factor $2 ^ { - d }$ to $C ( m ) ^ { - 1 }$ , adding superscript $m$ ’s to the quantities $\widehat { \mathcal { M } \mathcal { I } }$ , $\hat { S } _ { \mathrm { I A S } }$ and $S _ { \mathrm { { I A S } } }$ , and adding the condition $| S | \le m$ to all unions, intersections and sums. □

# A.9 Proof of Proposition 7.1

By Proposition 3.3, we have SIAS ⊆ ANY , and since $S _ { \mathrm { I A S } , O } ~ \subseteq ~ S _ { \mathrm { I A S } }$ , the claim follows immediately.

# B Oracle Algorithms for Learning SIAS

In this section, we review some of the existing literature on minimal $d$ -separators, which can be exploited to give an algorithmic approach for finding $S _ { \mathrm { { I A S } } }$ from a DAG. We first introduce the concept of $M$ -minimal separation with respect to a constraining set $I$ .

Definition B.1 (van der Zander et al. (2019), Section 2.2). Let $I \subseteq [ d ]$ , $K \subseteq [ d ]$ , and $S \subseteq [ d ]$ . We say that $S$ is a $K$ -minimal separator of $E$ and $Y$ with respect to a constraining set $I$ if all of the following are true:

(i) $I \subseteq S$ .   
(ii) $S \in \mathcal { Z }$ .   
(iii) There does not exists $S ^ { \prime } \in \mathcal { T }$ such that $K \subseteq S ^ { \prime } \subseteq S$ .

We denote by $M _ { K , I }$ the set of all $K$ -minimal separating sets with respect to constraining set $I$ .

(In this work, $S \in \mathcal { Z }$ means $E \perp \perp \boldsymbol { Y } \mid \boldsymbol { S }$ , but it can stand for other separation statements, too.) The definition of a $K$ -minimal separator coincides with the definition of a minimally invariant set if both $K$ and the constraining set $I$ are equal to the empty set. An $\varnothing$ -minimal separator with respect to constraining set $I$ is called a strongly-minimal separator with respect to constraining set $I$ .

We can now represent (2) using this notation. $M _ { \varnothing , \varnothing }$ contains the minimally invariant sets and thus

$$
S _ {\text {I A S}} := \bigcup_ {S \in M _ {\emptyset , \emptyset}} S.
$$

Listing the set $M _ { I , I }$ of all $I$ -minimal separators with respect to the constraining set $I$ (for any $I$ ) can be done in polynomial delay time $\mathcal { O } ( d ^ { 3 } )$ (van der Zander et al., 2019; Takata, 2010), where delay here means that finding the next element of $M _ { I , I }$ (or announcing that there is no further element) has cubic complexity. This is the algorithm we exploit, as described in the main part of the paper.

Furthermore, we have

$$
i \in S _ {\mathrm {I A S}} \quad \Leftrightarrow \quad M _ {\emptyset , \{i \}} \neq \emptyset .
$$

This is because $i \in S _ { \mathrm { I A S } }$ if and only if there is a minimally invariant set that contains $i$ , which is the case if and only if there exist a strongly minimal separating set with respect to constraining set $\{ i \}$ . Thus, we can construct $S _ { \mathrm { { I A S } } }$ by checking, for each $i$ , whether there is an element in $M _ { \emptyset , \{ i \} }$ . Finding a strongly-minimal separator with respect to constraining set $I$ , i.e., finding an element in $M _ { \varnothing , I }$ , is NP-hard if the set $I$ is allowed to grow (van der Zander et al., 2019). To the best of our knowledge, however, it is unknown whether finding an element in $M _ { \emptyset , \{ i \} }$ , for a singleton $\{ i \}$ is NP-hard.

# C The Maximum Number of Minimally Invariant Sets

If one does not have a priori knowledge about the graph of the system being analyzed, one can still apply Theorem 5.2 with a correction factor $2 ^ { d }$ , as this ensures (with high probability) that no minimally invariant sets are falsely rejected. However, we know that the correction factor is strictly conservative, as there cannot exist $2 ^ { d }$ minimally invariant sets in a graph. Thus, correcting for $2 ^ { d }$ tests, controls the familywise error rate (FWER) among minimally invariant sets, but increases the risk of falsely accepting a non-invariant set relatively more than what is necessary to control the FWER. Here, we discuss the maximum number of minimally invariant sets that can exist in a graph with $d$ predictor nodes and how a priori knowledge about the sparsity of the graph and the number of interventions can be leveraged to estimate a less strict correction that still controls the FWER.

As minimally invariant sets only contain ancestors of $Y$ (see Proposition 3.3), we only need to consider graphs where $Y$ comes last in a causal ordering. Since $d$ -separation is equivalent to undirected separation in the moralized ancestral graph (Lauritzen, 1996), finding the largest number of minimally invariant sets is equivalent to finding the maximum number of minimal separators in an undirected graph with $d + 2$ nodes. It is an open question how many minimal separators exists in a graph with $d + 2$ nodes, but it is known that a lower bound for the maximum number of minimal separators is in $\Omega ( 3 ^ { d / 3 } )$ (Gaspers & Mackenzie, 2015). We therefore propose using a correction factor of $C = 3 ^ { \lceil d / 3 \rceil }$ when estimating the set $\hat { S } _ { \mathrm { I A S } }$ from Theorem 5.2 if one does not have a priori knowledge of the number of minimally invariant sets in the DAG of the SCM being analyzed. This is a heuristic choice and is not conservative for all graphs.

Theorem 5.2 assumes asymptotic power of the invariance test, but as we can only have a finite amount of data, we will usually not have full power against all non-invariant sets that are not supersets of a minimally invariant set. Therefore, choosing a correction factor that is potentially too low represents a trade-off between error types: if we correct too little, we stand the risk of falsely rejecting a minimally invariant set but not rejecting a superset of it, whereas when correcting too harshly, there is a risk of failing to reject non-invariant sets due to a lack of power.

If one has a priori knowledge of the sparsity or the number of interventions, these can be leveraged to estimate the maximum number of minimally invariant sets using simulation, by the following procedure:

1. For $b = 1 , \ldots , B$ :

(a) Sample a DAG with $d$ predictor nodes, Ninterventions $\sim \mathbb { P } _ { N }$ interventions and $p \sim \mathbb { P } _ { p }$ probability of an edge being present in the graph over $( X , Y )$ , such that $Y$ is last in a causal ordering. The measures $\mathbb { P } _ { N }$ and $\mathbb { P } _ { p }$ are distributions representing a priori knowledge. For instance, in a controlled experiment, the researcher may have chosen the number $N _ { 0 }$ of interventions. Then, $\mathbb { P } _ { N }$ is a degenerate distribution with $\mathbb { P } _ { N } ( N _ { 0 } ) = 1$ .   
(b) Compute the set of all minimally invariant sets, e.g., using the adjustmentSets algorithm from dagitty (Textor et al., 2016).   
(c) Return the number of minimally invariant sets.

2. Return the largest number of minimally sets found in the $B$ repetitions above.

Instead of performing $B$ steps, one can continually update the largest number of minimally invariant sets found so far and end the procedure if the maximum has not updated in a predetermined number of steps, for example.

# D A Finite Sample Algorithm for Computing SˆIAS

In this section, we provide an algorithm for computing the sets $\hat { S } _ { \mathrm { I A S } }$ and $\hat { S } _ { \mathrm { I A S } } ^ { m }$ presented in Theorems 5.2 and 5.5. The algorithm finds minimally invariant sets by searching for invariant sets among sets of increasing size, starting from the empty set. This is done, because the first (correctly) accepted invariant is a minimally invariant set. Furthermore, any set that is a superset of an accepted invariant set, does not need to be tested (as this set cannot be minimal). Tests for invariance can be computationally expensive if one has large amounts of data. Therefore, skipping unnecessary tests offers a significant speedup. In the extreme case, where all singletons are found to be invariant, the algorithm completes in $d + 1$ steps, compared to $\textstyle \sum _ { i = 0 } ^ { m } { \binom { d } { i } }$ steps ( $2 ^ { d }$ if $m = d$ ). This is implemented in lines 8-10 of Algorithm 1.

Algorithm 1 An algorithm for computing SˆIAS from data   
input A decision rule $\phi_{n}$ for invariance, significance thresholds $\alpha_0,\alpha$ , max size of sets to test $m$ (potentially $m = d$ ) and data   
output The set $\hat{S}_{\mathrm{IAS}}$ 1: Initialize $\widehat{\mathcal{M}}\widehat{\mathcal{I}}$ as an empty list.   
2: $PS\gets \{S\subseteq [d]\mid |S|\leq m\}$ 3: if $\phi_n(\emptyset ,\alpha_0) = 0$ then   
4: End the procedure and return $\hat{S}_{\mathrm{IAS}} = \emptyset$ 5: end if   
6: Sort $PS$ in increasing order according the set sizes   
7: for $S\in PS$ do   
8: if $S\supseteq S^{\prime}$ for any $S^{\prime}\in \widehat{\mathcal{M}}\widehat{\mathcal{I}}$ then   
9: Skip the test of $S$ and go to next iteration of the loop   
10: else   
11: Add $S$ to $\widehat{\mathcal{M}}\widehat{\mathcal{I}}$ if $\phi_n(S,\alpha) = 0$ , else continue   
12: end if   
13: if The union of $\widehat{\mathcal{M}}\widehat{\mathcal{I}}$ contains all nodes then   
14: Break the loop   
15: end if   
16: end for   
17: Return $\hat{S}_{\mathrm{IAS}}$ as the union of all sets in $\widehat{\mathcal{M}}\widehat{\mathcal{I}}$

# E Additional Experiment Details

# E.1 Simulation Details for Section 6.1

We sample graphs that satisfy Assumption 2.1 with the additional requirement that $Y \in \mathrm { D E } _ { Y }$ by the following procedure:

1. Sample a DAG $\mathcal { G }$ for the graph of $( X , Y )$ with $d + 1$ nodes, for $d \in \{ 4 , 6 , \dots , 2 0 \} \cup$ $\{ 1 0 0 , 1 , 0 0 0 \}$ , and choose $Y$ to be a node (chosen uniformly at random) that is not a root node.   
2. Add a root node $E$ to $\mathcal { G }$ with $N$ interventions children that are not $Y$ . When $d \leq 2 0$ , Ninterventions $\in \{ 1 , \ldots , d \}$ and when $d \geq 1 0 0$ , $N _ { \mathrm { i n t e r v e n t i o n s } } \in \{ 1 , \dots , 0 . 1 \times d \}$ (i.e., we consider interventions on up to ten percent of the predictor nodes).   
3. Repeat the first two steps if $Y \notin \mathrm { D E } _ { E }$ $E ^ { \prime }$ .

# E.2 Simulation Details for Section 6.2

We simulate data for the experiment in Section 6.2 (and the additional plots in Appendix E.4) by the following procedure:

1. Sample data from a single graph by the following procedure:

(a) Sample a random graph $\mathcal { G }$ of size $d + 1$ and sample $Y$ (chosen uniformly at random) as any node that is not a root node in this graph.   
(b) Sample coefficients, $\beta _ { i \to j }$ , for all edges ( $i  j$ ) in $\mathcal { G }$ from $U ( ( - 2 , 0 . 5 ) \cup ( 0 . 5 , 2 ) )$ ) independently.   
(c) Add a node $E$ with no incoming edges and Ninterventions children, none of which are $Y$ . When $d = 6$ , we set $N _ { \mathrm { i n t e r v e n t i o n s } } = 1$ and when $d = 1 0 0$ , we sample Ninterventions uniformly from $\{ 1 , \ldots , 1 0 \}$ .   
(d) If $Y$ is not a descendant of $E$ , repeat steps (a), (b) and (c) until a graph where $Y \in \mathrm { D E } _ { E }$ is obtained.   
(e) For $n \in \{ 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$ :

i. Draw 50 datasets of size $n$ from an SCM with graph $\mathcal { G }$ and coefficients $\beta _ { i \to j }$ and with i.i.d. $N ( 0 , 1 )$ noise innovations. The environment variable, $E$ , is sampled independently from a Bernoulli distribution with probability parameter $p =$ 0.5, corresponding to (roughly) half the data being observational and half the data interventional. The data are generated by looping through a causal ordering of $( X , Y )$ , starting at the bottom, and standardizing a node by its own empirical standard deviation before generating children of that node; that is, a node $X _ { j }$ is first generated from $\mathrm { P A } _ { j }$ and then standardized before generating any node in $\mathrm { C H } _ { j }$ . If $X _ { j }$ is intervened on, we standardize it prior to the intervention.

ii. For each sampled dataset, apply IAS and ICP. Record the Jaccard similarities between IAS and AN $Y$ and between ICP and AN $Y$ , and record whether or not is was a subset of AN $Y$ and whether it was empty.

iii. Estimate the quantity plotted (average Jaccard similarity in Figure 4 or probability of $\hat { S } _ { \mathrm { I A S } } \subseteq \mathrm { A N } _ { Y }$ or ${ \hat { S } } _ { \mathrm { I A S } } = \emptyset$ in Figure 7) from the 50 simulated datasets.

(f) Return the estimated quantities from the previous step.

2. Repeat the above 100 times and save the results in a data-frame.

# E.3 Analysis of the Choice of $C$ in Section 6.2

We have repeated the simulation with $d = 6$ from Section 6.2 but with a correction factor of $C = 2 ^ { 6 }$ , as suggested by Theorem 5.2 instead of the heuristic correction factor of $C = 9$ suggested in Appendix C. Figure 6 shows the results. We see that the results are almost identical to those presented in Figure 4. Thus, in the scenario considered here, there is no change in the performance of $ { \hat { S } } _ { \mathrm { I A S } }$ (as measured by Jaccard similarity) between using a correction factor of $C = 2 ^ { 6 }$ and a correction factor of $C = 3 ^ { \lceil 6 / 3 \rceil } = 9$ . In larger graphs, it is likely that there is a more pronounced difference. E.g., at $d = 1 0$ , the strictly conservative correction factor suggested by Theorem 5.2 is $2 ^ { 1 0 } = 1 0 2 4$ , whereas the correction factor suggested in Appendix C is only $3 ^ { \lceil 1 0 / 3 \rceil } = 3 ^ { 4 } = 8 1$ , and at $d = 2 0$ the two are 220 = 1,048,576 and $3 ^ { \lceil 2 0 / 3 \rceil } = 3 ^ { 7 } = 2 1 8 7$ .

![](images/651cc4fa7e5a1f07c249668f1d9c69ba7ef10722fcb37225cd45fb550083edc6.jpg)  
Figure 6: The same figure as in Figure 4, but with a correction factor of $C = 2 ^ { 6 } = 6 4$ instead of $C = 3 ^ { \lceil 6 / 3 \rceil } = 9$ . Only $d = 6$ shown here, as the correction factor for $d = 1 0 0$ is unchanged. Here, the guarantees of Theorem 5.2 are not violated by a potentially too small correction factor, and the results are near identical to those given in Figure 4 using a milder correction factor.

# E.4 Analysis of the Choice of $\alpha _ { 0 }$ in Section 6.2

Here, we investigate the quantities $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } \subseteq \mathrm { A N } _ { Y } )$ , $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } ^ { 1 } \subseteq \mathrm { A N } _ { Y } )$ ), $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } = \varnothing )$ ) and $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } ^ { 1 } = \varnothing$ ) using the same simulation setup as described in Section 6.2. Furthermore, we also ran the simulations for values $\alpha _ { 0 } = \alpha$ (testing all hypotheses at the same level), $\alpha _ { 0 } = 1 0 ^ { - 6 }$ (conservative, see Remark 5.3) as in Section 6.2 and $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ (very conservative). The results for $\alpha = 1 0 ^ { - 6 }$ (shown in Figure 7) were recorded in the same simulations that produced the output for Figure 4. For $\alpha _ { 0 } \in \{ \alpha , 1 0 ^ { - 1 2 } \}$ (shown in Figure 8 and Figure 9, respectively) we only simulated up to 10,000 observations, to keep computation time low.

Generally, we find that the probability of IAS being a subset of the ancestors seems to generally hold well and even more so with large sample sizes. (see Figures 7 to 9), in line with Theorem 5.2. When given 100,000 observations, the probability of IAS being a subset of ancestors is roughly equal to one for almost all SCMs, although there are a few SCMs, where IAS is never a subset of the ancestors (see Figure 7). For $\alpha _ { 0 } = 1 0 ^ { - 6 }$ , the median probability of IAS containing only ancestors is one in all cases, except for $d = 1 0 0$ with 1,000 observations – here, the median probability is 87%.

In general, varying $\alpha _ { 0 }$ has the effect hypothesized in Remark 5.3: lowering $\alpha _ { 0 }$ increases the probability that IAS contains only ancestors, but at the cost of increasing the probability that it is empty (see Figures 7 to 9). For instance, the median probability of IAS being a subset of ancestors when $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ is one for all sample sizes, but the output is always empty when there are 100 observations and empty roughly half the time even at 1,000 observations when $d = 1 0 0$ (see Figure 9). In contrast, not testing the empty set at a reduced level, means that the output of IAS is rarely empty, but the probability of IAS containing only ancestors decreases. Still, even with $\alpha _ { 0 } = \alpha$ , the median probability of IAS containing only ancestors was never lower than 80% (see Figure 8). Thus, choosing $\alpha _ { 0 }$ means choosing a trade-off between finding more ancestor-candidates, versus more of them being false positives.

![](images/daccdc0d37bf8139732d8e5022de378ddc45303f2e4a031e6fb634d70edbb696.jpg)  
Figure 7: The empirical probabilities of recovering a subset of AN $Y$ (top row) and recovering an empty set (bottom row), when testing the empty set for invariance at level $\alpha _ { 0 } = 1 0 ^ { - 6 }$ Generally, our methods seem to hold level well, especially when sample sizes are large. When the sample size is small, the output is often the empty set. When $d = 6$ , we estimate $\hat { S } _ { \mathrm { I A S } }$ (left column) and when $d = 1 0 0$ , we estimate $\hat { S } _ { \mathrm { I A S } } ^ { 1 }$ (right column). The results here are from the simulations that also produced Figure 4. Medians are displayed as orange lines through each boxplot. Each point represents the probability that the output set is ancestral (resp. empty) for a randomly selected SCM, as estimated by repeatedly sampling data from the same SCM for every $n \in \{ 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$ . Observations from the same SCM are connected by a line. Each figure contains data from 100 randomly drawn SCMs. Points have been perturbed slightly along the $x$ -axis to improve readability.

![](images/ea2eb03e5908efde08b0d570cb3643862e50c917a378957c5a384df609906795.jpg)

![](images/508b19886175a80d04793553d77e3081306867c8ca2f3007aec38ebccb5d52e1.jpg)

![](images/f41e4f42587df00d9b39baa3de5283ec073fc97381b4ef218e825c4e7ea44002.jpg)  
Number of observations

![](images/91825074ab3830504af0b9b75f3e27d29237515acf802648bd11db634e838bf4.jpg)  
Number of observations   
Figure 8: The same figure as Figure 7, but with $\alpha _ { 0 } = \alpha = 0 . 0 5$ and $n \in \{ 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } \}$ . Testing the empty set at the non-conservative level $\alpha _ { 0 } = \alpha$ means that the empty set is output less often for small sample sizes, but decreases the probability that the output is a subset of ancestors. Thus, we find more ancestor-candidates, but make more mistakes when $\alpha _ { 0 } = \alpha$ . However, the median probability of the output being a subset of ancestors is at least 80% in all configurations.

# E.5 Analysis of the strength of inverventions in Section 6.2

Here, we repeat the $d = 6$ simulations from Section 6.2 with a reduced strength of the environment to investigate the performance of IAS under weaker interventions. We sample from the same SCMs as sampled in Section 6.2, but reduce the strength of the interventions to be 0.5 instead of 1. That is, the observational distributions are the same as in Section 6.2, but interventions to a node $X _ { j }$ are here half as strong as in Section 6.2.

The Jaccard similarity between $\hat { S } _ { \mathrm { I A S } }$ and AN $Y$ is generally lower than what we found in Figure 4 (see Figure 10). This is likely due to having lower power to detect non-invariance, which has two implications. First, lower power means that we may fail to reject the empty set, meaning that we output nothing. Then, the Jaccard similarity between $\hat { S } _ { \mathrm { I A S } }$ and AN $Y$ is zero. Second, it may be that we correctly reject the empty set, but fail to reject another non-invariant set which is not an ancestor of $Y$ which is then potentially included in the output. Then, the $\hat { S } _ { \mathrm { I A S } }$ and AN $Y$ is lower, because we increase the number of false findings.

We find that the probability that $\hat { S } _ { \mathrm { I A S } }$ is a subset of ancestors is generally unchanged for the lower intervention strength, but the probability of $\hat { S } _ { \mathrm { I A S } }$ generally increases for small sample sizes (see Table 1). This indicates that IAS does not make more mistakes under the weaker interventions, but it is more often uninformative. We see also that in both settings, $\hat { S } _ { \mathrm { I A S } }$ is empty more often than $\hat { S } _ { \mathrm { I C P } }$ for low sample sizes, but less often for larger samples (see Table 1). This is likely because IAS tests the empty set at a much lower level than ICP does ( $1 0 ^ { - 6 }$ compared to 0.05). Thus, IAS requires more power to find anything, but once it has sufficient power, it finds more than ICP (see also Figure 10). The median probability of ICP returning a subset of the ancestors was always at least 95% (not shown).

![](images/8a6829a4d4f288ff183ca6d9d762809452b9d2efe20c7d7a39e67607e2bf2b52.jpg)

![](images/20710c21bb6f40bedce0070cb14ea58c4647769d06785220bb962431f21a8191.jpg)

![](images/bf66fed51ea03d50dd7f3b95066d1bfdf9bdfeec75250d64abe50cde9a06811f.jpg)  
Number of observations

![](images/b2c1250a3971b959e0846ad243da15206ff38f7699655fdc87dc143b24b5fa7b.jpg)  
Number of observations   
Figure 9: The same figure as Figure 7, but with $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ and $n \in \{ 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } \}$ . Testing the empty set at at very conservative level $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ means that the empty set is output more often (for one hundred observations, we only find the empty set), but increases the probability that the output is a subset of ancestors. Thus, testing at a very conservative level $\alpha _ { 0 } = 1 0 ^ { - 1 2 }$ means that we do not make many mistakes, but the output is often non-informative.

![](images/f52c9066ebc23155fcac4000159d13120f4f74f9deb6ec25579c0e735eee93b3.jpg)  
Number of observations

![](images/22b8c0fc108c729189eaf74406c3f65fe7e3cb3d151aef135b9e027f30b67bc8.jpg)  
Number of observations   
Figure 10: The same figure as the one presented in Figure 4, but with weaker environments (do-interventions of strength 0.5 compared to 1 in Figure 4). Generally, IAS performs the same for weaker interventions as for strong interventions, when there are more than 10,000 observations. Graphs represented in each boxplot: 42 (left), 58 (right).

Table 1: Summary of the quantities $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } \subseteq \mathrm { A N } _ { Y } )$ , $\mathbb { P } ( \hat { S } _ { \mathrm { I A S } } = \varnothing )$ and $\mathbb { P } ( \hat { S } _ { \mathrm { I C P } } = \varnothing )$ ) for weak and strong do-interventions (strength 0.5 and 1, respectively) when $d = 6$ . Numbers not in parentheses are means, numbers in parentheses are medians. The level is generally unchanged when the environments have a weaker effect, but the power is lower, in the sense that the empty set is output more often.   

<table><tr><td></td><td></td><td>P(SIAS ⊆ ANY)</td><td>P(SIAS = ∅)</td><td>P(SICP = ∅)</td></tr><tr><td rowspan="4">Strong interventions</td><td>n = 100</td><td>96.6% (100%)</td><td>89.6% (98%)</td><td>52.3% (52%)</td></tr><tr><td>n = 1,000</td><td>75.7% (100%)</td><td>10.0% (0%)</td><td>30.4% (14%)</td></tr><tr><td>n = 10,000</td><td>83.7% (100%)</td><td>1.0% (0%)</td><td>24.9% (10%)</td></tr><tr><td>n = 100,000</td><td>93.8% (100%)</td><td>0.2% (0%)</td><td>22.9% (10%)</td></tr><tr><td rowspan="4">Weak interventions</td><td>n = 100</td><td>99.3% (100%)</td><td>98.7% (100%)</td><td>72.0% (84%)</td></tr><tr><td>n = 1,000</td><td>81.1% (100%)</td><td>40.2% (26%)</td><td>36.9% (24%)</td></tr><tr><td>n = 10,000</td><td>80.8% (100%)</td><td>1.7% (0%)</td><td>27.5% (15%)</td></tr><tr><td>n = 100,000</td><td>92.6% (100%)</td><td>1.1% (0%)</td><td>24.8% (14%)</td></tr></table>

# E.6 Analysis of the Choice of $q _ { T B }$ in Section 6.3

In this section, we analyze the effect of changing the cut-off $q _ { T B }$ that determines when a gene pair is considered a true positive in Section 6.3. For the results in the main paper, we use $q _ { T B } = 1 \%$ , meaning that the pair $( \mathtt { g e n e } _ { X } , \mathtt { g e n e } _ { Y } )$ is considered a true positive if the value of ${ \tt g e n e } _ { Y }$ when intervening on ${ \tt g e n e } _ { X }$ is outside of the 0.01- and 0.99-quantiles of $\mathtt { g e n e } _ { Y }$ in the observational distribution. In Figure 11, we plot the true positive rates for several other choices of $q _ { T B }$ . We compare to the true positive rate of random guessing, which also increases if the criterion becomes easier to satisfy. We observe that the choice of $q _ { T B }$ does not substantially change the excess true positive rate of our method compared to random guessing. This indicates that while the true positives in this experiments are inferred from data, the conclusions drawn in Figure 5 are robust with respect to some modelling choices of $q _ { T B }$ .

![](images/7610055dfea159b94241deb1377c7325daa8cc10766ab70b78e7850e165531f3.jpg)  
Figure 11: True positive rates (TPRs) for the gene experiment in Section 6.3. qT B specifies the quantile in the observed distribution that an intervention effect has to exceed to be considered a true positive. While the TPR increases for our method when $q _ { T B }$ is increased, the TPR of random guessing increases comparably. This validates that changing the definition of true positives in this experiment by choosing a different $q _ { T B }$ does not change the conclusion of the experiment substantially.

# E.7 Learning causal ancestors by estimating the I-MEC

In this section, we repeat the experiments performed in Section 6.2, this time including a procedure (here denoted IASest. graph), where we perform the following steps.

1. Estimate a member graph of the I-MEC and the location of the intervention sites using Unknown-Target Interventional Greedy Sparsest Permutation (UT-IGSP) (Squires et al., 2020) using the implemention from the Python package CausalDAG.9   
2. Apply the oracle algorithm described in Section 4 to the estimated graph to obtain an estimate of $\mathcal { M T }$ .

# 3. Output the union of all sets in the estimate of $\mathcal { M T }$ .

The results for the low-dimensional experiment are displayed in Figure 12 and the results for the high-dimensional experiment are displayed in Table 2. Here, we see that IASest. graph generally performs well (as measured by Jaccard similarity) in the low-dimensional setting ( $d = 6$ ), and even better than IAS for sample sizes $N \leq 1 0 ^ { 3 }$ , but is slightly outperformed by IAS for larger sample sizes. However, in the high-dimensional setting ( $d = 1 0 0$ ), we observe that IASest. graph fails to hold level and identifies only very few ancestors (see Table 2). We hypothesize that the poor performance of IASest. graph in the high-dimensional setting is due to IASest. graph attempting to solve a more difficult task than IAS. IASest. graph first estimates a full graph (here using UT-IGSP), even though only a subgraph of the full graph is of relevance in this scenario. In addition, UT-IGSP aims to estimate the site of the unknown interventions. In contrast, IAS only needs to identify nodes that are capable of blocking all paths between two variables, and does not need to know the site of the interventions.

![](images/b2e48c25b1a14e445097942c881166e1f56776a4044378108a4509629531ce4b.jpg)  
Figure 12: Comparison between the finite sample output of IAS and the procedure described in Appendix E.7, in the low-dimensional case. Generally, these procedures have similar performance, although IAS performs worse for small sample sizes but slightly better for high sample sizes.

Table 2: Identifying ancestors by first estimating the I-MEC of the underlying DAG and then applying the oracle algorithm of Section 4 fails to hold level and identifies fewer ancestors than applying IAS, when in a high-dimensional setting.   

<table><tr><td></td><td colspan="2">d=100,N=103</td><td colspan="2">d=100,N=104</td><td colspan="2">d=100,N=105</td></tr><tr><td></td><td>IAS</td><td>IASest.graph</td><td>IAS</td><td>IASest.graph</td><td>IAS</td><td>IASest.graph</td></tr><tr><td>P(S.⊆ANY)</td><td>84.64%</td><td>15.30%</td><td>94.04%</td><td>14.92%</td><td>94.72%</td><td>14.74%</td></tr><tr><td>P(S.=∅)</td><td>51.96%</td><td>12.32%</td><td>12.72%</td><td>11.84%</td><td>6.98%</td><td>11.42%</td></tr><tr><td>J(S.,ANY)</td><td>0.19</td><td>0.10</td><td>0.33</td><td>0.10</td><td>0.35</td><td>0.11</td></tr></table>