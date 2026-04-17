# Active clustering for labeling training data

Quentin Lutz∗ Nokia Bell Labs

Élie de Panafieu Nokia Bell Labs

Alex Scott University of Oxford

Maya Stein University of Chile

October 28, 2021

# Abstract

Gathering training data is a key step of any supervised learning task, and it is both critical and expensive. Critical, because the quantity and quality of the training data has a high impact on the performance of the learned function. Expensive, because most practical cases rely on humans-in-the-loop to label the data. The process of determining the correct labels is much more expensive than comparing two items to see whether they belong to the same class. Thus motivated, we propose a setting for training data gathering where the human experts perform the comparatively cheap task of answering pairwise queries, and the computer groups the items into classes (which can be labeled cheaply at the very end of the process). Given the items, we consider two random models for the classes: one where the set partition they form is drawn uniformly, the other one where each item chooses its class independently following a fixed distribution. In the first model, we characterize the algorithms that minimize the average number of queries required to cluster the items and analyze their complexity. In the second model, we analyze a specific algorithm family, propose as a conjecture that they reach the minimum average number of queries and compare their performance to a random approach. We also propose solutions to handle errors or inconsistencies in the experts’ answers.

# 1 Introduction

There is an increasing demand for software implementing supervised learning for classification. Training data input for such software consists of items belonging to distinct classes. The output is a classifier: a function that predicts, for any new item, the class it most likely belongs to. Its quality depends critically on the available learning data, in terms of both quantity and quality [21]. But labeling large quantities of data is costly. This task cannot be fully automated, as doing so would assume access to an already trained classifier. Thus, human intervention, although expensive, is required. In this article, we focus on helping the human experts build the learning data efficiently.

One natural way for the human experts to proceed is to learn (or discover) the classes and write down their characteristics. Then, items are considered one by one, assigning an existing class to each of them, or creating a new one if necessary. This approach requires the experts to learn the various classes, which, depending on the use-case, can be difficult. A different approach,

proposed to us by Nokia engineer Maria Laura Maag, is to discover the partition by querying the experts on two items at a time asking whether these belong to the same class or not. This approach avoids the part of the process where classes are learned, and can therefore be cheaper. It is the setting we consider here, and we call the corresponding algorithm an active clustering algorithm, or for short, $A C$ algorithm.

More precisely, we assume there is a set of size $n$ , with a partition unknown to us. An $A C$ algorithm will, in each step, choose a pair of elements and asks the oracle whether they belong to the same partition class or not. The choices of the queries of the algorithm are allowed to depend on earlier answers. The algorithm will use transitivity inside the partition classes: if each of the pairs $x$ , $y$ and $y$ , $z$ is known to lie in the same class (for instance because of positive answers from the oracle), then the algorithm will ‘know’ that $x$ and $z$ are also in the same class, and it will not submit a query for this pair. The algorithm terminates once the partition is recovered, i.e. when all elements from the same partition class have been shown to belong to the same class, and when for each pair of partition classes, there has been at least one query between their elements.

We investigate AC algorithms under two different random models for the unknown set partition. In Section 2.1, the set partition is sampled uniformly at random, while in Section 2.2, the number of blocks is fixed and each item chooses its class independently following the same distribution. Section 2.3 analyzes the cases where the experts’ answers contain errors or inconsistencies. Our proofs, sketched in Section 3, rely on a broad variety of mathematical tools: probability theory, graph theory and analytic combinatorics. We conclude in Section 4, with some interesting open problems and research topics.

Related works. Note that our model has similarities with sorting algorithms. Instead of an array of elements equipped with an order, we consider a set of items equipped with a set partition structure. In the sorting algorithm, the oracle determines the order of a pair of elements, while in our setting, the oracle tells whether two items belong to the same class. Both in sorting and AC algorithms, the goal is the design of algorithms using few queries.

The cost of annotating raw data to turn it into training data motivated the exploration of several variants of supervised learning. Transfer learning reduces the quantity of labeled data required by using the knowledge gained while solving a different but related problem. Semi-supervised learning reduces it by learning also from the inherent clustering present in unlabeled data. In active learning, the learner chooses the data needing labeling and the goal is to maximize the learning potential for a given small number of queries. However, users who want to use supervised learning software for classification as a black-box can mitigate the annotating cost only by modifying the labeling process.

Recent papers [9, 18] acknowledge that humans prefer pairwise queries over pointwise queries as they are better suited for comparisons. Pairwise queries have been considered in semisupervised clustering [5, 24, 16] where they are called must-link and cannot-link constraints. The pairs of vertices linked by those constraints are random in general, but chosen adaptively in active semi-supervised clustering [4]. In both cases, the existence of a similarity measure between items is assumed, and a general theoretical study of this setting is provided by [19]. This is not the case in [11, 17], where a hierarchical clustering is built by successively choosing pairs of items and measuring their similarity. There, the trade-off between the number of measurements and the accuracy of the hierarchical clustering is investigated. The difference with the current paper is that the similarity measure takes real values, while we consider Boolean queries, and that their output is a hierarchical clustering, while our clustering (a set partition) is flat.

The problem we consider is motivated by its application to annotation for supervised machine learning. It also belongs to the field of combinatorial search [1, 7]. Related papers [2, 15] consider the problem of reconstructing a set partition using queries on sets of elements, where the answer

to such a query is whether there is an edge in the queried set, or the number of distinct blocks of the partition present in the queried set, respectively. Our setting is a more constrained case corresponding to queries of size two. Another similar setting is that of entity resolution [13] where recent developments also assume perfect oracles and transitive properties using pairwise queries to reconstruct a set partition [23, 25]. In this case, clusters correspond to duplicates of the same entity. Most solutions have a real-valued similarity measure between elements but rely on human verification to improve the results. The entity resolution literature also considers the noisy case for a fixed number of clusters [8, 9, 18, 10].

# 2 Our results

# 2.1 Uniform distribution on partitions

We start by considering the setting where the partition of the $n$ -set is chosen uniformly at random among all possible partitions. The average complexity of an AC algorithm is the average number of queries used to recover the random partition. We will define a class of AC algorithms, called chordal algorithms, and prove in Theorem 2 that an algorithm has minimal average complexity if and only if it is chordal. Theorem 3 shows that all chordal algorithms have the same distribution on their number of queries. Finally, this distribution is characterized both exactly and asymptotically in Theorem 4.

It will be useful to use certain graphs marking the ‘progress’ of the AC algorithm on our $n$ -set. Given a partition $P$ of an $n$ -set and an AC algorithm, we can associate a graph $G _ { t }$ to each step $t$ of the algorithm. Namely, $G _ { 0 }$ has $n$ vertices, labeled with the elements of our set, and at time $t \geq 1$ , the graph $G _ { t }$ is obtained by adding an edge for each negatively-answered query, and successively merging pairs of vertices that received a positive answer. (A vertex $u$ obtained by joining vertices $v , w$ is adjacent to all neighbors of each of $v , w$ , and we label $u$ with the union of its earlier labels.) We call $G _ { t }$ the aggregated graph at time $t$ . Note that after the last step of the algorithm, the aggregated graph is complete and each of its vertices corresponds to one of the blocks of $P$ . Also note that any fixed $G _ { t }$ (with labels on vertices also fixed) may appear as the aggregated graph for more than one partition (possibly at a different time $t ^ { \prime }$ ). We call the set of all these partitions the realizations of $G _ { t }$ . Those notions are illustrated in Figure 1.

We now need a quick graph-theoretic definition. A cycle $C$ in a graph is induced if all edges induced by $V ( C )$ belong to the cycle. A graph is chordal if its induced cycles all have length three. Chordal graphs appear in many applications. We say that an AC algorithm is chordal if one of the following equivalent conditions is satisfied (see the Appendix for a proof of the equivalence of the conditions):

(i) for all input partitions, each aggregated graph is chordal,   
(ii) for all input partitions, no aggregated graph has an induced $C _ { 4 }$ ,   
(iii) for all input partitions, and for every query $u$ , $\boldsymbol { v }$ made on some aggregated graph $G _ { t }$ , the intersection of the neighborhoods of $u$ and $v$ separates $u$ and $\boldsymbol { v }$ in $G _ { t }$

where a set $S$ (possibly empty) is said to separate vertices $u , v \in V ( G ) \setminus S$ if $u$ and $v$ lie in distinct components of $G - S$ . The queries that keep a graph chordal are exactly those satisfying Condition (iii). Thus, it is used in chordal algorithms to identify the candidate queries. On the other hand, a query satisfying Condition (ii) might turn a chordal graph non-chordal by creating an induced cycle of length more than 4 (in which case, Condition (ii) will fail on a later query). Two examples of chordal algorithms are presented below.

![](images/fdd90ab827015c204294dc70fd6074199be1437ae4c040f26d957ccc76478121.jpg)

![](images/f4355803ad5a5899b97e32559bbe6dfafcbe83cd2ab0544f865d3363e41a09ad.jpg)  
Figure 1: Aggregated graphs obtained after various queries and organized into query trees. The positive answers are displayed on top, the negative ones on the bottom. Left, a non-chordal query, leading to an average complexity of 13/5. Right, chordal queries from the same initial situation, leading to an optimal average complexity of 12/5.

Definition 1. The clique algorithm is an AC algorithm that gradually grows the partition, starting with the empty partition. Each new item is compared successively to an item of each block of the partition, until it is either added to a block (positive answer to a query) or all blocks have been visited, in which case a new block is created for this item. The universal algorithm finds the block containing the item of largest label by comparing it to all remaining items. The algorithm is then applied to partition the other items, and the previous block is added to the result.

Theorem 2. On partitions of size n chosen uniformly at random, an AC algorithm has minimal average complexity if and only if it is chordal.

The complexity distribution of an AC algorithm of a set $S$ with $n \geq 1$ elements is a tuple $( a _ { 0 } , a _ { 1 } , a _ { 2 } , . . . )$ where $a _ { i }$ is the number of partitions of $S$ for which the algorithm has complexity $i$ . Clearly, $a _ { i } = 0$ for all $i < n - 1$ , and $\boldsymbol { a } _ { \binom { n } { 2 } } = 1$ , for any AC algorithm. Our next result shows that constraining the average complexity to be minimal fixes the complexity distribution.

Theorem 3. On partitions of size n chosen uniformly at random, all chordal AC algorithms have the same complexity distribution.

Note that either of Theorems 2 and 3 implies the weaker statement that all chordal AC algorithms of an $n$ -set have the same average complexity, but with very different proofs. Sketches of the proofs of Theorems 2 and 3 can be found in sections 3.1 and 3.2.

In practice, several human experts work in parallel to annotate the training data. The chordal algorithms can easily be parallelized: when an expert becomes available, give him a query that would keep the aggregated graph chordal if all pending queries received negative answers. The condition ensures the chordality of the parallelized algorithm.

Our third theorem for this setting describes the distribution of the number of queries used by chordal algorithms on partitions chosen uniformly at random. Two formulas for the probability generating function are proposed. The first one is well suited for computer algebra manipulations, while the second one, more elegant, is better suited for asymptotic analysis. It is a $q$ -analog of

Dobiński’s formula for Bell numbers

$$
B _ {n} = \frac {1}{e} \sum_ {m \geq 0} \frac {m ^ {n}}{n !}
$$

(see e.g. [14, p.762]), counting the number of partitions of size $n$ . In order to state the theorem, we introduce the $q$ -analogs of an integer, the factorial, the Pochhammer symbol and the exponential

$$
[ n ] _ {q} = \sum_ {k = 0} ^ {n - 1} q ^ {k}, \quad [ n ] _ {q}! = \prod_ {k = 1} ^ {n} [ k ] _ {q}, \quad (a; q) _ {n} = \prod_ {k = 0} ^ {n - 1} (1 - a q ^ {k}), \quad e _ {q} (z) = \sum_ {n \geq 0} \frac {z ^ {n}}{[ n ] _ {q} !}.
$$

Observe that the $q$ -analog reduce to their classic counterparts for $q = 1$ . The Lambert function $W ( x )$ is defined for any positive $x$ as the unique solution of the equation $w e ^ { w } = x$ . As $x$ tends to infinity, we have $W ( x ) = \log ( x ) - \log ( \log ( x ) ) + o ( 1 )$ .

Theorem 4. Let $X _ { n }$ denote the complexity of a chordal algorithm on a partition of size n chosen uniformly at random. The distribution of $X _ { n }$ has probability generating function in the variable $q$ equal to the two following expressions

$$
\frac {1}{B _ {n}} \left(\frac {q}{1 - q}\right) ^ {n} \sum_ {k = 0} ^ {n} {\binom {n} {k}} (- 1) ^ {k} \left(\frac {1 - q}{q}; q\right) _ {k} \qquad a n d \qquad \frac {1}{B _ {n}} \frac {1}{e _ {q} (1 / q)} \sum_ {m \geq 0} \frac {[ m ] _ {q} ^ {n}}{[ m ] _ {q} !} q ^ {n - m}.
$$

The normalized variable $( X _ { n } - E _ { n } ) / \sigma _ { n }$ converges in distribution to a standard Gaussian law, where

$$
E _ {n} = \frac {1}{4} (2 W (n) - 1) e ^ {2 W (n)} \qquad a n d \qquad \sigma_ {n} = \frac {1}{3} \sqrt {\frac {3 W (n) ^ {2} - 4 W (n) + 2}{W (n) + 1}} e ^ {3 W (n)}.
$$

As a corollary, the average complexity of chordal algorithms is asymptotically ${ \binom { n } { 2 } } / \log ( n )$ . Because of this almost quadratic optimal complexity, using pairwise queries can only be better than direct classification for relatively small data sets. We arrive at a different conclusion for the model presented in the next section, where AC algorithm have linear complexity.

# 2.2 Random partitions with fixed number of blocks

In many real-world applications, we have information on the number of classes and their respective sizes in the data requiring annotation. This motivates the introduction of the following alternative random model for the set partition investigated by the AC algorithms. Now, the set of partitions of the $n$ -set $S$ is distributed in a not necessarily uniform way, but each element of $S$ has a probability of $p _ { i }$ to belong to the partition class $C _ { i }$ (and the probabilities $p _ { i }$ sum up to 1). Our main focus here will be on the most applicable and accessible variant of the above-described model, where the number of partition class is a fixed number $k$ , and $k$ is small compared to $n$ . Without loss of generality, we can assume that the probabilities $p _ { 1 } , p _ { 2 } , \ldots , p _ { k }$ in this model satisfy $p _ { 1 } \geq p _ { 2 } \geq \cdots \geq p _ { k }$ .

Intuitively, an algorithm with low average complexity should avoid making many comparisons between different classes (one such comparison is always necessary between any pair of classes, but more comparisons are not helpful). For this reason, it seems plausible that an AC algorithm with optimal complexity should compare a new element first with the largest class identified up to this moment, as both the new element and the largest class are most likely to coincide with $C _ { 1 }$ , the ‘most likely class’. This is precisely how the clique AC algorithm from Definition 1 operates, where we add the additional refinement that each new item is compared to the blocks discovered so far in decreasing order of their sizes. With this refinement, we conjecture the following.

Conjecture 5. For an $n$ -set with a random partition with probabilities $p _ { 1 } , p _ { 2 } , \ldots , p _ { k }$ , the clique algorithm has minimal average complexity among all AC algorithms.

In support of Conjecture 5, we present the following results. Firstly, we exhibit the limit behaviour of the expected number of queries of the clique algorithm. The simple proof of this theorem can be found in the appendix. It allows one to decide for practical cases whether direct classification by human experts or pairwise queries will be more efficient to annotate the training data.

Theorem 6. Let $p _ { 1 } \geq \cdots \geq p _ { k }$ be fixed with $\textstyle \sum _ { i = 1 } ^ { k } p _ { i } = 1$ . Let $X$ denote the expected number of queries made by the clique algorithm with these parameters for an $n$ -set. Then $\mathbb { E } [ X ] \sim \textstyle \sum _ { i = 1 } ^ { k } i p _ { i } n$ .

We now compare the complexity of the clique algorithm with the complexity of an algorithm that chooses its queries randomly. More precisely, we define the random algorithm to be the one that at each step $t$ compares a pair of elements chosen uniformly at random among all pairs whose relation is not known at this time (that is, they are neither known to belong to the same class, nor known to belong to different classes). The reason for analyzing the random algorithm is that one may think of this algorithm as the one that models the situation where no strategy at all is used by the human experts, which might make this procedure cheap to implement. It turns out, however, that there is a cost in form of larger average complexity associated to the random algorithm, if compared to our proposed candidate for the optimal algorithm, the clique algorithm.

Theorem 7. Let $p _ { 1 } \geq \cdots \geq p _ { k }$ be fixed with $\textstyle \sum _ { i = 1 } ^ { k } p _ { i } = 1$ . Let $X$ be the number of queries of a random algorithm with these parameters for an $n$ -set. Then $\begin{array} { r } { \mathbb { E } [ X ] \sim n - k + \sum _ { i < j } f ( p _ { i } , p _ { j } ) n _ { \bar { } } } \end{array}$ , where

$$
f (\alpha , \beta) = \left\{ \begin{array}{l l} 2 \alpha & \alpha = \beta \\ \frac {2 \alpha \beta \ln (\alpha / \beta)}{\alpha - \beta} & \alpha \neq \beta . \end{array} \right.
$$

Note that if all the $p _ { i }$ are equal to $1 / k$ , then by Theorems 6 and 7, the expected number of queries produced by the random algorithm is asymptotically $\begin{array} { r } { n - k + ( 2 n / k ) { \binom { k } { 2 } } = k ( n - 1 ) } \end{array}$ , while for the clique algorithm this number is $( k + 1 ) n / 2$ .

# 2.3 Noisy queries

We now discuss the situation when answers to the query can be inconsistent, due to errors from the human experts or ambiguity in the data. Different perspectives can be found in [10, 18]. Let $G$ denote the graph where vertices correspond to items from the data set, edges to queries, so each edge is either positive or negative depending on the answer. Because of potential errors, the aggregated graph is not sufficient for the analysis anymore, but it can be recovered by contracting to a vertex each positive component of $G$ .

Correcting errors. An inconsistency is detected if and only if $G$ contains a contradictory cycle: a cycle where all edges except one are positive. At each step, we consider the shortest contradictory cycle and ask a query that cuts it in two, following a divide-and-conquer strategy. After a logarithmic number of queries, if no additional error occurred, a false answer or ambiguous data is identified and the answer is corrected. At any point, if the number of contradictions detected grows out of proportion (edit war between human experts), a classic clustering algorithm can be applied as a last resort to settle the differences of opinions. We now focus on the problem of detecting inconsistencies.

Bounded number of errors. To ensure the detection of at most $k$ errors, each positive component of graph $G$ must be $( k + 1 )$ -edge-connected and any two positive components must be linked by at least $k + 1$ negative edges. The minimal number of queries for $n$ items and $b$ blocks, assuming no block has size 1, is then $( k + 1 ) \bigl ( \binom { b } { 2 } + n / 2 \bigr )$ , because each vertex of $G$ has at least $k + 1$ positive neighbors.

Small probability of error. To avoid this high cost, the error detection criteria can be relaxed. We now consider that each answer has a small probability $p$ of error and add a few queries at the end of the AC algorithm, while keeping the probability of undetected errors low. More precisely, let $c _ { 0 } + c _ { 1 } p + c _ { 2 } p ^ { 2 } + \cdot \cdot \cdot$ denote the Taylor expansion of this probability at $p = 0$ . Since $p$ is assumed to be small, our aim is to minimize the vector $( c _ { 0 } , c _ { 1 } , \ldots )$ for the lexicographic order. During the AC algorithm, when a query between two classes is needed, we choose the items in those classes to maintain the following structure. Any positive component should be a tree with vertices of degree at most 3. We call 2-path a path of vertices of degree 2 linking two vertices of degree 3. The 2-paths should all have length close to a parameter $r$ of the algorithm. At the end of the algorithm, we introduce additional queries between pairs of leaves of the same tree, turning each positive component of size at least 2 into either a 3-edge-connected graph or a cycle of length close to $r$ . Queries are also added to ensure that any two positive components are linked by at least 3 negative edges. Assuming the partition contains $b$ blocks and the average length of the 2-paths is $r$ , the number of additional queries compared to the noiseless setting is approximately ${ \frac { n } { 3 r + 2 } } + b$ inside the blocks, plus the potential queries between blocks (at most $2 { \binom { b } { 2 } }$ ). This setting ensures $c _ { 0 } = c _ { 1 } = 0$ and minimizes $c _ { 2 }$ (see [6]). If the length of all 2-paths and the sizes of positive components that are cycles are bounded by $r ^ { \prime }$ , then $\begin{array} { r } { c _ { 2 } = \binom { r ^ { \prime } + 1 } { 2 } \frac { 3 n } { 3 r + 2 } } \end{array}$ . Thus, the choice of the parameter $r$ is a trade-off between the number of additional queries and the robustness to noise of the algorithm.

# 3 Proofs

# 3.1 Proof of Theorem 2

We will prove Theorem 2 by analyzing the types of queries made by an AC algorithm. For this, we classify the queries made by an AC algorithm into three types:

• Core queries are those that receive a positive answer.   
• A query at time $t$ is excessive if it compares vertices $x$ and $y$ that are joined by an induced path in $G _ { t }$ on an even number of vertices that alternates between two partition classes.   
• A query at time $t$ is productive if it is neither core nor excessive.

Note that each query is of exactly one type, as excessive queries have to receive negative answers. Also note that as the core queries are the only ones that shrink $G _ { t }$ , the number of core queries does not depend on the algorithm.

Lemma 8. For any $A C$ algorithm and any partition of a set of size n containing $k$ classes, the number of core queries is exactly $n - k$ .

Next, we show that for a random partition, also the number of productive queries is, in expectation, the same for all AC algorithms.

Lemma 9. Let $\mathcal { P } _ { k }$ be the set of partitions of $[ n ]$ into exactly k sets, and choose $P \in \mathcal { P } _ { k }$ uniformly at random. Then all algorithms have the same expected number of productive queries.

Because of space constraints, we only present a sketch of the proof of this lemma here. The full proof of Lemma 9 is in the appendix.

Sketch of the proof of Lemma 9. Consider a partition of $[ n ]$ into $k$ sets $C _ { 1 } , C _ { 2 } , \ldots , C _ { k }$ , and let $q _ { i j }$ be the number of productive queries comparing a vertex from $C _ { i }$ with a vertex from $C _ { j }$ . Then $\mathbb { E } [ q _ { i j } ] = \mathbb { E } [ q _ { 1 2 } ]$ for all $i , j$ , and by linearity of expectation the expected number of productive queries is ${ \binom { k } { 2 } } \mathbb { E } [ q _ { 1 2 } ]$ . Thus, we are done if we can prove that $\mathbb { E } [ q _ { 1 2 } ]$ is independent from the choice of the algorithm. It suffices to show that ${ \mathbb E } [ q _ { 1 2 } | C _ { 1 } \cup C _ { 2 } = S ]$ is independent from the choice of algorithm because

$$
\mathbb {E} [ q _ {1 2} ] = \sum_ {S \subset [ n ], | S | \geq 2} \mathbb {E} [ q _ {1 2} | C _ {1} \cup C _ {2} = S ] \mathbb {P} [ C _ {1} \cup C _ {2} = S ].
$$

In other words, we wish to understand the expected number of productive queries of an AC algorithm working on a partition with exactly two classes of a set of size $| S |$ , chosen uniformly at random (there are $2 ^ { | S | } - 2$ such partitions). Observe that this is very similar to calculating the expected number of productive queries for a uniformly-chosen partition with at most two classes (there are $2 ^ { | S | }$ such partitions). In fact, it it not hard to see that these numbers only differ by a factor of 2|S| $\frac { 2 ^ { | S | } } { 2 ^ { | S | - 2 } }$ . So we can restrict our attention to the expectation of the latter number, which is easier to calculate, as the model is equivalent to assigning independently to each element of $S$ a value in $\{ 0 , 1 \}$ uniformly at random.

More precisely, we will now argue that the expected number of productive queries in this scenario is $\scriptstyle { \frac { n - 1 } { 2 } }$ , for each partition algorithm. For this, note that each component of any aggregated graph $G _ { t }$ is either a single vertex or a nonempty bipartite graph. Moreover, each component has two possible colorings, and it is not hard to show by induction that these colorings are equally likely. So, whenever the algorithm makes a query for two vertices from distinct components, answers ‘yes’ and ‘no’ are equally likely. As the algorithm makes $n - 1$ such queries (since each such query reduces the number of components by 1), it follows from linearity of expectation, that the expected number of productive queries is $( n - 1 ) / 2$ . □

Finally, we analyze the excessive queries of an AC algorithm.

Lemma 10. An AC algorithm makes no excessive queries if and only if it is chordal.

For the proof, we need the following easy lemma whose proof can be found in the appendix.

Lemma 11. For each non-chordal AC algorithm, there is an input partition and a time t such that $G _ { t }$ has an induced $C _ { 4 }$ one of whose edges comes from a negative query in step $t - 1$ .

Proof of Lemma 10. By definition, a chordal AC algorithm has no aggregated graphs with induced cycles of length at least 4. So, since an excessive query, if answered negatively, creates an induced cycle of even length at least 4, chordal algorithms make no excessive queries.

For the other direction, let us show that any non-chordal AC algorithm makes at least one excessive query for some partition. By Fact 11, a non-chordal AC algorithm has, for some partition of the ground set, an aggregated graph $G _ { t }$ with an induced $C _ { 4 }$ , on vertices $v _ { 1 } , v _ { 2 } , v _ { 3 } , v _ { 4 }$ , in this order, such that the last query concerned the pair $v _ { 1 } , v _ { 4 }$ . There is a realization of $G _ { t }$ where $v _ { 1 }$ and $v _ { 3 }$ are in one partition class, and $v _ { 2 }$ and $v _ { 4 }$ are in another partition class. For this partition, the query $v _ { 1 } , v _ { 4 }$ at time $t - 1$ is an excessive query. □

We are ready for the proof of Theorem 2.

Proof of Theorem 2. By Lemmas 8 and 9, all AC algorithms have the same expected number of core queries and productive queries. So the optimal algorithms are the ones with the minimum expected number of excessive queries. By Lemma 10, these are the chordal algorithms. □

# 3.2 Proof of Theorem 3

The full proof is in the appendix. We prove a more general result which allows for the algorithm to start with any aggregated graph instead of the empty graph. We prove this using induction on the number of non-edges of $G = ( V , E )$ , the base case being trivial. We fix some useful notation now: For $x , y \in V$ , set $G ( x y ) = ( V , E \cup \{ x y \} )$ , and let $G _ { x y }$ be obtained from $G$ by identifying $x$ and $y$ .

For the induction step consider a graph $G$ with $k + 1$ missing edges, and let $A _ { 0 }$ , $A _ { 1 }$ be two distinct chordal AC algorithms for $G$ . By induction, we can assume that $A _ { 0 }$ and $A _ { 1 }$ differ in their first queries. Say the first query of $A _ { i }$ is $u _ { i }$ , $v _ { i }$ , for $i = 0 , 1$ . Then $G ( u _ { i } v _ { i } )$ is chordal for $i = 0 , 1$ . Note that we can assume that $u _ { 0 } \neq u _ { 1 }$ . We distinguish two cases.

Case 1. $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is chordal and moreover, if $v _ { 0 } = v _ { 1 }$ then $u _ { 0 } u _ { 1 } \notin E ( G )$ . Then, for $i = 0 , 1$ , the edge $u _ { i } v _ { i }$ can be chosen as the first edge of a chordal AC algorithm for $G ( u _ { 1 - i } v _ { 1 - i } )$ or for $G _ { u _ { 1 - i } v _ { 1 - i } }$ . As the induction hypothesis applies to $G ( u _ { 1 - i } v _ { 1 - i } )$ and to $G _ { u _ { 1 - i } v _ { 1 - i } }$ , we can assume that $u _ { i } v _ { i }$ is the second edge in $A _ { 1 - i }$ . Observe that for each $i = 0 , 1$ after the second query of $A _ { i }$ , we arrive at one of the four graphs $( G _ { u _ { 0 } v _ { 0 } } ) _ { u _ { 1 } v _ { 1 } }$ , $G ( u _ { 0 } v _ { 0 } ) _ { u _ { 1 } v _ { 1 } }$ , $G ( u _ { 1 } v _ { 1 } ) _ { u _ { 0 } v _ { 0 } }$ , $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ Thus the complexity distribution of $A _ { 0 }$ and $A _ { 1 }$ is identical (as it can be computed from the complexity distribution for the algorithms starting at these four graphs).

Case 2. $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is chordal, $v _ { 0 } = v _ { 1 }$ and $u _ { 0 } u _ { 1 } \in E ( G )$ , or $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is not chordal. This case is harder to analyze, but one can show that either there is an edge $u v \in E ( G )$ such that $G ( u v )$ , $G ( u v ) ( u _ { 0 } v _ { 0 } )$ and $G ( u v ) ( u _ { 1 } v _ { 1 } )$ are chordal, or $G$ has a very specific shape. In the former case, we proceed as in the previous paragraph using the edge $_ { u v }$ as a proxy. In the latter case our argument relies on our knowledge on the structure of $G$ .

# 3.3 Proof of Theorem 4

According to Theorem 3, all chordal algorithms share the same complexity distribution, so we investigate a specific chordal algorithm, the universal $A C$ algorithm (see Definition 1) without loss of generality. This algorithm first computes the block $B$ containing the largest item by comparing it to all other items, then calculates the partition $Q$ for the remaining items, and finally inserts the block $B$ in $Q$ . Let query(p) denote the number of queries used by the universal AC algorithm to recover partition $p$ . Let us introduce the generating function

$$
P (z, q) = \sum_ {\mathrm {p a r t i t i o n} p} q ^ {\mathrm {q u e r y} (p)} \frac {z ^ {| p |}}{| p | !}.
$$

The symbolic method presented in [14] translates the description of the algorithm into a the differential equation characterizing $P ( z , q )$

$$
\partial_ {z} P (z, q) = P (q z, q) e ^ {q z}
$$

with initial condition $P ( 0 , q ) = 1$ . Observing that the function $\scriptstyle e ^ { { \frac { q } { 1 - q } } z }$ is solution of a similar differential equation, we consider solutions of the form $P ( z , q ) = A ( z , q ) e ^ { \frac { q } { 1 - q } z }$ . The differential equation on $P ( z , q )$ translates into a differential equation on $A ( z , q )$

$$
\partial_ {z} A (z, q) + \frac {q}{1 - q} A (z, q) = A (q z, q)
$$

with initial condition $A ( 0 , q ) = 1$ . Decomposing $A ( z , q )$ as a series in $z$ , we find the solution

$$
A (z, q) = \sum_ {k \geq 0} \left(\frac {1 - q}{q}; q\right) _ {k} \frac {\left(- \frac {q}{1 - q} z\right) ^ {k}}{k !}.
$$

By definition, the probability generating function PGF $_ n ( q )$ of the complexity distribution when the set of items has size $n$ is linked to our generating function by the relation

$$
\mathrm {P G F} _ {n} (q) = \frac {n !}{B _ {n}} [ z ^ {n} ] P (z, q),
$$

where the Bell number $B _ { n }$ counts the number of partitions of size $n$ . The first exact expression from the theorem is obtained directly by coefficient extraction. The second one requires using the following classic $q$ -identities

$$
[ n ] _ {q}! = \frac {(q ; q) _ {n}}{(1 - q) ^ {n}}, \quad \frac {1}{(x ; q) _ {\infty}} = \sum_ {n \geq 0} \frac {x ^ {n}}{(q ; q) _ {n}}, \quad e _ {q} (x) = ((1 - q) x; q) _ {\infty} ^ {- 1}.
$$

To obtain the Gaussian limit law, we prove that the Laplace transform of the normalized random variable $X _ { n } ^ { \star } = ( X _ { n } - E _ { n } ) / \sigma _ { n }$

$$
\mathbb {E} (e ^ {s X _ {n} ^ {\star}}) = \mathrm {P G F} _ {n} (e ^ {s / \sigma_ {n}}) e ^ {- s E _ {n} / \sigma_ {n}}
$$

converges to the Laplace transform of the standard Gaussian $e ^ { s ^ { 2 } / 2 }$ pointwise for $s$ in a neighborhood of 0. To do so, we apply to the second expression the Laplace method for sums [14, p. 761], using in the process a $q$ -analog of Stirling’s approximation [20].

# 3.4 Proof of Theorem 7

Since Lemma 8 is still valid in this setting, it suffices to prove the following, with $f$ as in Theorem 7.

Theorem 12. Let $p _ { 1 } \geq \cdots \geq p _ { k }$ be fixed with $\textstyle \sum _ { i = 1 } ^ { k } p _ { i } = 1$ . Let $X$ denote the number of edges between classes using a random algorithm. Then $\begin{array} { r } { \mathbb { E } [ X ] \sim \sum _ { i < j } f ( p _ { i } , p _ { j } ) n } \end{array}$ .

Proof. We only give a sketch here, see the appendix for further detail. Instead of analyzing the random algorithm, we analyze the following process. Begin with all vertices marked active. At each time step, pick (with replacement) a random pair $\{ u , v \}$ and:

• If $u , v$ are active and from distinct classes $i , j$ then say we have generated an $_ { i j }$ -crossedge.   
• If $u , v$ , are both active and in class $i$ then mark exactly one of $u$ and $v$ inactive.   
• If one of $u , v$ is inactive, then do nothing.

Note that as the process runs, the number of active vertices is monotonic decreasing, and we are increasingly likely to choose pairs where one vertex is inactive. These contribute to the new process, but do not generate new comparisons between pairs. So we are looking at a (randomly) slowed down version of the random algorithm; but this makes the analysis much simpler.

Let $x _ { i } ( t )$ denote the number of active class $_ i$ vertices after $t$ time steps, and $x _ { i j } ( t )$ denote the number of $_ { i j }$ -crossedges that are generated in the first $t$ time steps. Then at step $t + 1$ , the probability that we pick two active vertices in class $_ i$ is

$$
\left( \begin{array}{c} x _ {i} (t) \\ 2 \end{array} \right) / \left( \begin{array}{c} n \\ 2 \end{array} \right) \sim \frac {x _ {i} (t) ^ {2}}{n ^ {2}}.
$$

Writing $p = p _ { i }$ , we estimate $x _ { i } ( t )$ via a function $x = x ( t )$ satisfying the differential equation $\begin{array} { r } { \partial _ { t } x ( t ) = - \frac { x ( t ) ^ { 2 } } { n ^ { 2 } } } \end{array}$ x(t)2n2 , with x(0) = pn This has solution $x ( 0 ) = p n$

$$
x (t) = \frac {n ^ {2}}{t + n / p} = \frac {p n}{1 + p t / n}.
$$

One can show that with high probability, $x _ { i } ( t )$ tracks $x ( t )$ quite closely.

Now we estimate the number of $_ { i j }$ -crossedges. Using our estimates for $x _ { i } ( t )$ and $x _ { j } ( t )$ , we see that the probability of an $i j$ crossedge at step $t + 1$ is

$$
x _ {i} (t) x _ {j} (t) / \binom {n} {2} \sim \frac {2 x _ {i} (t) x _ {j} (t)}{n ^ {2}} \approx \frac {2 n ^ {2}}{(t + n / p _ {i}) (t + n / p _ {j})}.
$$

Let $p = p _ { i }$ and $q = p _ { j }$ . Similarly as before, we can model the growth of $x _ { i j } ( t )$ by a function $c = c ( t )$ satisfying the differential equation

$$
\partial_ {t} c (t) = \frac {2 n ^ {2}}{(t + n / p) (t + n / q)}
$$

with $c ( 0 ) = 0$ . Calculations show that if $p = q$ , then

$$
c (t) = 2 p n \left(1 - \frac {1}{2 + 2 p t / n}\right) \sim 2 p n
$$

and if $p \neq q$ , then

$$
c (t) \sim \frac {2 n p q \ln (p / q)}{p - q}
$$

as $t / n  \infty$ . In order to prove the theorem, we now run the process for time $K n$ , where $K$ is a large constant. We note that at this point, we have (with high probability) at most $c n$ remaining active vertices for some small constant $c$ . But now we revert to the original process: noting that for any partition into $k$ classes, a fraction of at least (about) $1 / k$ of the pairs lie inside some class, we see that at each step the process reduces the number of vertices with constant probability. The remaining expected running time is therefore $O ( k c n )$ . □

We note that it is also possible to prove a central limit theorem for the running time of the clique algorithm (roughly: after $n ^ { 1 / 2 }$ comparisons, we are very likely to have seen representatives from every set in the partition, and more from the $i$ th class than the $j$ th class whenever $p _ { i } > p _ { j }$ ; we estimate the contribution from the remaining steps by an sum of independent random variables). We do not pursue the details here.

# 4 Conclusion

In this article, motivated by the building of training data for supervised learning, we studied active clustering using pairwise comparisons and without any other additional information (such as a similarity measure between items).

Two random models for the secret set partition where considered: the uniform model and the bounded number of blocks model. They correspond to different practical annotation situations. The uniform model reflects the case where nothing is known about the set partition. In that case, many clusters will typically be discarded at the end of the clustering process, as they are too small for the supervised learning algorithm to learn anything from them. Thus, this is a worst case scenario. When some information is available on the set partition, such as its number of blocks or their relative sizes, the bounded number of blocks model becomes applicable.

Comparison between direct labeling and pairwise comparisons. As a practical application of this work, we provided tools to decide whether direct labeling or pairwise comparisons will be more efficient for a given annotation task. One should first decide whether the uniform model or the bounded number of blocks model best represents the data. In both cases, our theorems provide estimates for the number of pairwise queries required. Then the time taken by an expert to answer a direct labeling query or a pairwise comparison should be measured. Finally, combining those estimates, the time required to annotate the data using direct labeling or pairwise comparison can be compared. We also provided tools to detect and correct errors in the experts’ answers.

Similarity measures. Generally, a similarity measure on the data is available and improves the quality of the queries we can propose to the human experts. This similarity measure can be a heuristic that depends only on the format of the data. For example, if we are classifying technical documents, a text distance can be used. The similarity could also be trained on a small set of data already labeled. This setting has been analyzed by [19]. Our motivation for annotating data is to train a supervised learning algorithm. However, one could use active learning to merge and replace the annotation and learning steps. The problem is then to find the queries that will improve the learned classifier the most.

In this article, we focused on the case where such similarity measures are not available. However, we are confident that the mathematical tools developed here will be useful to analyze more elaborate settings as well. In particular, the aggregated graph contains exactly the information on the transitive closure of the answers to the pairwise queries, so its structure should prove relevant whenever pairwise comparisons are considered.

Open problems. We leave as open problems the proof that the clique algorithm reaches the minimal average complexity in the bounded number of blocks model, and the complexity analysis of the random algorithm in the uniform model.

Acknowledgments and Disclosure of Funding. We thank Maria Laura Maag (Nokia) for introducing us to the problem of clustering using pairwise comparisons, and Louis Cohen, who could sadly not continue working with us. The authors of this paper met through and were supported by the RandNET project (Rise project H2020-EU.1.3.3). Alex Scott was supported by EPSRC grant EP/V007327/1. Quentin Lutz and Élie de Panafieu produced part of this work at Lincs (www.lincs.fr). Quentin Lutz is supported by Nokia Bell Labs (CIFRE convention 2018/1648). Maya Stein was supported by ANID Regular Grant 1180830, and by ANID PIA ACE210010.

# References

[1] Martin Aigner. Combinatorial search. John Wiley & Sons, Inc., 1988.   
[2] Noga Alon and Vera Asodi. “Learning a Hidden Subgraph”. In: SIAM J. Discret. Math. 18 (4 2005), pp. 697–712.   
[3] Noga Alon and Joel H. Spencer. The probabilistic method. Fourth. Wiley Series in Discrete Mathematics and Optimization. John Wiley & Sons, Inc., Hoboken, NJ, 2016, pp. xiv+375. isbn: 978-1-119-06195-3.

[4] Sugato Basu, Arindam Banerjee, and Raymond J Mooney. “Active semi-supervision for pairwise constrained clustering”. In: Proceedings of the 2004 SIAM international conference on data mining. SIAM. 2004, pp. 333–344.   
[5] Sugato Basu, Mikhail Bilenko, and Raymond J Mooney. “A probabilistic framework for semi-supervised clustering”. In: Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining. 2004, pp. 59–68.   
[6] Douglas Bauer et al. “Combinatorial optimization problems in the analysis and design of probabilistic networks”. In: Networks 15.2 (1985), pp. 257–271.   
[7] Mathilde Bouvel, Vladimir Grebinski, and Gregory Kucherov. “Combinatorial search on graphs motivated by bioinformatics applications: A brief survey”. In: International Workshop on Graph-Theoretic Concepts in Computer Science. Springer. 2005, pp. 16–27.   
[8] Nicolo Cesa-Bianchi et al. A correlation clustering approach to link classification in signed networks. JMLR Workshop and Conference Proceedings, extended version on arXiv:1301.4769, 2012.   
[9] I Eli Chien, Huozhi Zhou, and Pan Li. “HS2: Active learning over hypergraphs with pointwise and pairwise queries”. In: The 22nd International Conference on Artificial Intelligence and Statistics. PMLR. 2019, pp. 2466–2475.   
[10] Susan Davidson et al. “Top-k and clustering with noisy comparisons”. In: ACM Transactions on Database Systems (TODS) 39.4 (2014), pp. 1–39.   
[11] Brian Eriksson et al. “Active clustering: Robust and efficient hierarchical clustering using adaptively selected similarities”. In: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings. 2011, pp. 260–268.   
[12] Thomas Ernst. A comprehensive treatment of $q$ -calculus. Springer Science & Business Media, 2012.   
[13] Ivan P Fellegi and Alan B Sunter. “A theory for record linkage”. In: Journal of the American Statistical Association 64.328 (1969), pp. 1183–1210.   
[14] Philippe Flajolet and Robert Sedgewick. Analytic combinatorics. Cambridge University Press, 2009.   
[15] Vladimir Grebinski and Gregory Kucherov. “Reconstructing set partitions”. In: Tenth Annual ACM-SIAM Symposium on Discrete Algorithms-SODA’99. ACM/SIAM. 1999, p. 2.   
[16] Daniel Gribel, Michel Gendreau, and Thibaut Vidal. “Semi-Supervised Clustering with Inaccurate Pairwise Annotations”. In: arXiv preprint arXiv:2104.02146 (2021).   
[17] Akshay Krishnamurthy et al. “Efficient active algorithms for hierarchical clustering”. In: arXiv preprint arXiv:1206.4672 (2012).   
[18] Arya Mazumdar and Barna Saha. Clustering with Noisy Queries. 2017. arXiv: 1706.07510 [stat.ML].   
[19] Arya Mazumdar and Barna Saha. “Query complexity of clustering with side information”. In: Advances in Neural Information Processing Systems. 2017, pp. 4685–4696.   
[20] Daniel S Moak. “The q-analogue of Stirling’s formula”. In: The Rocky Mountain Journal of Mathematics (1984), pp. 403–413.   
[21] Curtis G Northcutt, Anish Athalye, and Jonas Mueller. “Pervasive label errors in test sets destabilize machine learning benchmarks”. In: arXiv preprint arXiv:2103.14749 (2021).

[22] The Sage Developers. SageMath, the Sage Mathematics Software System (Version 9.0). https://www.sagemath.org. 2020.   
[23] Norases Vesdapunt, Kedar Bellare, and Nilesh Dalvi. “Crowdsourcing algorithms for entity resolution”. In: Proceedings of the VLDB Endowment 7.12 (2014), pp. 1071–1082.   
[24] Kiri Wagstaff et al. “Constrained k-means clustering with background knowledge”. In: Icml. Vol. 1. 2001, pp. 577–584.   
[25] Jiannan Wang et al. “Leveraging Transitive Relations for Crowdsourced Joins”. In: SIGMOD ’13: Proceedings of the 2013 ACM SIGMOD International Conference on Management of Data. arXiv Preprint 1408.6916. 2013, pp. 229–240.

# A Appendix

# A.1 Equivalence in the definition of chordal algorithms

In this short section we show the equivalence of the conditions in the definition of chordal AC algorithms from section 2.1, and some related easy lemmas. We start by proving a lemma from section 3.1.

Proof of Lemma 11. Since the algorithm is not chordal, for some input partition one of the aggregated graphs $G _ { t }$ of the AC algorithm has an induced cycle $C$ of length at least 4. Consider a realization of $G _ { t }$ where the vertices of $C$ are all in distinct partition classes. Then there is a time $t ^ { \prime }$ when four of the vertices of $C$ form an induced $C _ { 4 }$ in $G _ { t ^ { \prime } }$ . □

The next lemma is used both in the proof of the equivalence of the conditions in the definition of chordal AC algorithms, and in section A.3. Recall that for a graph $G = ( V , E )$ we write $G ( u v ) = ( V , E \cup \{ u v \} )$ .

Lemma 13. Let $G$ be a chordal graph, let $u , v \in V ( G )$ be distinct and nonadjacent, and assume $G ( u , v )$ is chordal. Then $N ( u ) \cap N ( v )$ is complete and separates u and v (in $G$ ).

Proof. Note that $N ( u ) \cap N ( v )$ is complete, as otherwise there are two non-adjacent vertices $x , y \in N ( u ) \cap N ( v )$ , and $( u , x , v , y )$ is an induced cycle in $G$ , a contradiction. It remains to show that $N ( u ) \cap N ( v )$ separates $u$ from $v$ . Assume this is not the case, then there is an induced path $P = ( u , p _ { 1 } , . . . , p _ { k } , v )$ that avoids $N ( u ) \cap N ( v )$ . In particular, $k \geq 2$ . Adding the edge uv to $P$ we obtain an induced cycle of length at least 4, a contradiction to $G ( u , v )$ being chordal. □

Now we prove the equivalence of the three conditions.

Lemma 14. The following conditions are equivalent for any AC algorithm.

(i) for all input partitions, each aggregated graph is chordal,   
(ii) for all input partitions, no aggregated graph has an induced $C _ { 4 }$ ,   
(iii) for all input partitions, for each query $u$ , $v$ , the intersection of the neighborhoods of u and v separates u and $v$ .

Proof. By Lemma 11, we know that (ii) implies (i), and by Lemma 13 we know that (i) implies (iii). So we only need to show that (iii) implies (ii). For this, assume there is an input partition, and a time $t$ , such that the aggregated graph $G _ { t }$ at time $t$ has an induced cycle $( v _ { 1 } , v _ { 2 } , v _ { 3 } , v _ { 4 } )$ . We can assume that $t$ is the first such time. Then either $G _ { t }$ arose by adding an edge of this cycle, say the edge $v _ { 1 } v _ { 4 }$ , or by identifying two vertices $x$ and $y$ to a new vertex from the cycle, say $v _ { 1 }$ . In

the first case, the query $v _ { 1 }$ , $v _ { 4 }$ in step $t - 1$ did not meet the requirement in (iii), a contradiction. In the second case, $x$ and $y$ are joined by the induced path $( x , v _ { 2 } , v _ { 3 } , v _ { 4 } , y )$ or $( y , v _ { 2 } , v _ { 3 } , v _ { 4 } , x )$ , so the query $x$ , $y$ in step $t - 1$ did not meet the requirement in (iii), a contradiction.

# A.2 Full proof of Lemma 9

Let $\mathcal { P } _ { 2 } ( n )$ denote the family of partitions of size $n$ where each element receives a random value in $\{ 0 , 1 \}$ uniformly and independently, so that two elements belong to the same block if and only if they share the same value. Thus, all partitions in $\mathcal { P } _ { 2 } ( n )$ have one or two blocks.

Lemma 15. The expected number of productive queries made by any AC algorithm on a random partition from $\mathcal { P } _ { 2 } ( n )$ is exactly n−1 $\scriptstyle { \frac { n - 1 } { 2 } }$

Proof. Let us consider the (random) sequence of graphs $G _ { 1 } , \ldots , G _ { n }$ produced by running the algorithm on a random element of $\mathcal { P } _ { 2 } ( n )$ . As core queries result in contractions, it follows that (for every $t$ ) every component of $G _ { t }$ is either a single vertex or a nonempty bipartite graph. Note that for each component there are two possible colorings; thus the number of possible colorings of $G _ { i }$ is $2 ^ { \kappa ( G _ { i } ) }$ , where $\kappa$ denotes the number of components.

We prove by induction that:

• For each $i$ , if the $_ i$ th query joins two components of $G _ { i - 1 }$ then it is productive with probability half.   
• The $2 ^ { \kappa ( G _ { i } ) }$ colorings of $G _ { i }$ are equally likely.

The second property holds for $G _ { 0 }$ , so it is enough to prove the inductive step. Consider the query made at stage $t$ , say joining vertices $x$ and $y$ . If $x$ and $y$ lie in the same component of $G _ { t - 1 }$ then the query is excessive, the components do not change, and the two bullets hold. Thus we may restrict our attention to queries such that $x$ and $y$ lie in distinct components, say $H _ { x }$ and $H _ { y }$ . There are four possible colorings of $H _ { x } \cup H _ { y }$ . The four colorings are equally likely and are independent from the coloring of the rest of the graph. It is now easily checked that $x$ and $y$ have the same color with probability $1 / 2$ , and so the probability that the query is productive is $1 / 2$ . Furthermore, adding the edge xy joins $H _ { x }$ and $H _ { y }$ into a single component, and the two possible colorings of this component are equally likely (and remain independent from the coloring of the rest of $G$ ). Thus the two bullets hold, and the induction is complete.

There are in total $n - 1$ steps at which the algorithm makes a query joining distinct components (as each such query reduces the number of components by 1). So, by linearity of expectation, the expected number of productive queries is $( n - 1 ) / 2$ . □

Lemma 16. The expected number of productive queries of an $A C$ algorithm working on a partition of a set $S$ containing exactly two blocks chosen uniformly at random is exactly

$$
\frac {2 ^ {| S |}}{2 ^ {| S |} - 2} \frac {| S | - 1}{2}.
$$

Proof. Consider an AC algorithm, and let $\alpha ( | S | )$ denote the expected number of productive queries. Now run the algorithm on a partition from $\mathcal { P } _ { 2 } ( | S | )$ . If the algorithm is fed one of the two constant colorings then it makes exactly $| S | - 1$ queries, all of which are core (and therefore not productive).

The probability that a partition $P \in \mathcal { P } _ { 2 } ( n )$ is constant is $2 / 2 ^ { | S | }$ ; and if we condition on $P$ being nonconstant then it is uniformly distributed among the set of partitions with exactly two blocks. By Lemma 15, we conclude that the expected number of productive queries satisfies

$$
\frac {| S | - 1}{2} = \alpha (| S |) \mathbb {P} [ P \mathrm {n o n c o n s t a n t} ] + 0 \mathbb {P} [ P \mathrm {c o n s t a n t} ] = \frac {2 ^ {| S |} - 2}{2 ^ {| S |}} \alpha (| S |).
$$

The result follows by solving for $\alpha ( | S | )$ .

![](images/7fd92f70db852aa6ff7367dfa7b7be5ac1ed3259133dd64e08ee57236b42ed07.jpg)

We are now ready for the full proof of Lemma 9.

Proof of Lemma $g$ . Fix $k$ and consider a partition of $[ n ]$ into exactly $k$ sets $C _ { 1 } , C _ { 2 } , \ldots , C _ { k }$ . Let $\alpha _ { i j }$ denote the expected number of productive queries that compare a vertex from $C _ { i }$ with a vertex from $C _ { j }$ . Then all $\alpha _ { i j }$ are equal, and by linearity of expectation the expected number of productive queries is $\binom { k } { 2 } \alpha _ { 1 2 }$ . Thus it is enough to prove that $\alpha _ { \mathrm { 1 2 } }$ does not depend on the choice of algorithm.

Let $q _ { 1 2 }$ be the number of productive queries comparing a vertex from $C _ { 1 }$ with a vertex from $C _ { 2 }$ . (so $\alpha _ { 1 2 } = \mathbb { E } q _ { 1 2 }$ ) Then, considering the set $S = C _ { 1 } \cup C _ { 2 }$ , and applying Lemma 16, we obtain

$$
\begin{array}{l} \alpha_ {1 2} = \sum_ {S \subset [ n ], | S | \geq 2} \mathbb {E} \left[ q _ {1 2} \mid C _ {1} \cup C _ {2} = S \right] \mathbb {P} \left[ C _ {1} \cup C _ {2} = S \right] \\ = \sum_ {S \subset [ n ], | S | \geq 2} \frac {| S | - 1}{2} \frac {2 ^ {| S |}}{2 ^ {| S |} - 2} \mathbb {P} \left[ C _ {1} \cup C _ {2} = S \right]. \tag {1} \\ \end{array}
$$

The last line is independent from the choice of algorithm, which concludes the proof.

![](images/5e8aab747004d5ade5988e7c56b52bf7836a0f48ee5ba4440953267170938360.jpg)

# A.3 Proof of Theorem 3

This section is devoted to the proof of Theorem 3. For this we will need several auxiliary lemmas. We also recall a useful notation introduced earlier: For a graph $G = ( V , E )$ let $G ( u v ) = ( V , E \cup \{ u v \} )$ , and let $G _ { u v }$ be the graph obtained from $G$ by identifying $u$ and $v$ .

The first of our lemmas shows that every chordal graph can ‘grow’ an edge while staying chordal.

Lemma 17. Let $H$ be a chordal graph that is not complete. Then $H$ has a non-edge e such that $H ( e )$ is chordal.

Proof. Let $u$ be a non-universal vertex of $H$ . Among all non-neighbors of $u$ , choose $p _ { 1 }$ such that $| N ( u ) \cap N ( p _ { 1 } ) |$ is maximized. We claim that $H ( u p _ { 1 } )$ is chordal.

Indeed, otherwise there is an induced cycle $C = ( u , p _ { 1 } , \dots , p _ { k } , u )$ , with $k \geq 3$ . As $p _ { k } \in N ( u ) \cap$ $N ( p _ { k - 1 } ) \setminus N ( p _ { 1 } )$ , our choice of $p _ { 1 }$ guarantees that there is a vertex $w \in N ( u ) \cap N ( p _ { 1 } ) \setminus N ( p _ { k - 1 } )$ . Let $j \le k - 2$ be the largest index in $[ k - 2 ]$ such that $w p _ { j } \in E ( H )$ . Then, depending on whether the edge $w p _ { k }$ is present, either $( w , p _ { j } , \ldots , p _ { k } , u , w )$ or $( w , p _ { j } , \dots , p _ { k } , w )$ is an induced cycle of length at least 4 in $H$ , a contradiction. □

Our next two lemmas are more technical. They give a structural characterization of those aggregated graphs where consecutive queries cannot easily be interchanged. In order to make their statement easier, let us say that a graph $G$ has a complete separation $( A , K , B )$ if $V ( G )$ is the disjoint union of $A , B , K$ so that $A \neq \emptyset \neq B$ , each of $A \cup K$ and $B \cup K$ is complete, and there are no edges from $A$ to $B$ . (Observe that we allow $K$ to be empty.)

Lemma 18. Let $G$ be a chordal graph that does not have a complete separation. For $i = 0 , 1$ let $u _ { i } v _ { i }$ be a non-edge of $G$ such that $G ( u _ { i } v _ { i } )$ is chordal, and $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is non-chordal. Then there is a non-edge uv of $G$ such that $G ( u v )$ is chordal, and $G ( u _ { i } v _ { i } ) ( u v )$ is chordal for $i = 0 , 1$ .

Lemma 19. Let $G$ be a chordal graph that does not have a complete separation. Let $u _ { 0 } , u _ { 1 } , v \in$ $V ( G )$ such that for $i = 0 , 1$ , we have $u _ { i } v \notin E ( G )$ and $G ( u _ { i } v )$ is chordal, and moreover, $u _ { 0 } u _ { 1 } \in$ $E ( G )$ . Then there is a non-edge uw of $G$ such that $G ( u w )$ is chordal, and $G ( u _ { i } v _ { i } ) ( u w )$ is chordal for $i = 0 , 1$ .

The proofs of these two lemmas rely on purely structural arguments and will be postponed to the end of the section.

We are now ready to prove Theorem 3. Actually, we will prove a more general result which allows for the algorithm to start with any aggregated graph instead of starting with the empty graph. More precisely, if $G$ is an aggregated graph at time $t$ for some AC algorithm for an $n$ -set $S$ , then we call the restriction of the algorithm to all queries after time $t$ that eventually lead to a realization of $G$ a $A C$ algorithm starting at $G$ . We define the complexity distribution of this algorithm analogously to our earlier definition. In particular, if $G$ is complete, then the complexity distribution is $( a _ { 0 } , a _ { 1 } , a _ { 2 } , \ldots , a _ { \binom { n } { 2 } } ) = ( 1 , 0 , 0 , \ldots , 0 )$ .

Theorem 20. For any $G$ , all chordal $A C$ algorithms starting at $G$ have the same complexity distribution.

Proof. We proceed by induction on the number of non-edges of $G$ . Proving the base case is trivial as, starting from a complete graph, the only AC algorithm has distribution $( a _ { 0 } , a _ { 1 } , a _ { 2 } , \ldots , a _ { \binom { n } { 2 } } ) =$ $( 1 , 0 , 0 , \ldots , 0 )$ .

For the induction step assume that for any chordal graph with $k$ or less missing edges, all chordal AC algorithms have the same complexity distribution, and consider a graph $G$ with $k + 1$ missing edges. Let $A _ { 0 }$ , $A _ { 1 }$ be two distinct chordal AC algorithms for $G$ . If their first queries are the same, say they query the edge $e$ , then by induction we know that for both $G _ { e }$ and $G ( e )$ , the two algorithms have the same distribution if we let them start there. As the distribution for an algorithm starting at $G$ is uniquely obtained from the complexity distributions of the same algorithm starting at $G _ { e }$ and at $G ( e )$ , we see that $A _ { 0 }$ and $A _ { 1 }$ have the same complexity distribution.

So we can assume that $A _ { 0 }$ and $A _ { 1 }$ differ in their first queries. Say the first query of $A _ { i }$ is $u _ { i }$ , $v _ { i }$ , for $i = 0 , 1$ . Then $G ( u _ { i } v _ { i } )$ is chordal for $i = 0 , 1$ . Note that we can assume that $u _ { 0 } \neq u _ { 1 }$ . We will distinguish two cases.

First, let us assume that $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is chordal and moreover, if $v _ { 0 } = v _ { 1 }$ then $u _ { 0 } u _ { 1 } \notin E ( G )$ . Then, for $i = 0 , 1$ , the edge $u _ { i } v _ { i }$ can be chosen as the first edge of a chordal AC algorithm for $G ( u _ { 1 - i } v _ { 1 - i } )$ or for $G _ { u _ { 1 - i } v _ { 1 - i } }$ . As the induction hypothesis applies to $G ( u _ { 1 - i } v _ { 1 - i } )$ and to $G _ { u _ { 1 - i } v _ { 1 - i } }$ , we can assume that $u _ { i } v _ { i }$ is the second edge in $A _ { 1 - i }$ . Observe that for each $i = 0 , 1$ after the second query of $A _ { i }$ , we arrive at one of the four graphs $\left( G _ { u _ { 0 } v _ { 0 } } \right) _ { u _ { 1 } v _ { 1 } }$ , $G ( u _ { 0 } v _ { 0 } ) _ { u _ { 1 } v _ { 1 } }$ , $G ( u _ { 1 } v _ { 1 } ) _ { u _ { 0 } v _ { 0 } }$ , $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ . Thus the complexity distribution of $A _ { 0 }$ and $A _ { 1 }$ is identical (as is can be computed from the complexity distribution for the algorithms starting at these four graphs).

Now, let us assume that either $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is chordal, $v _ { 0 } ~ = ~ v _ { 1 }$ and $u _ { 0 } u _ { 1 } \in E ( G )$ , or $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is not chordal. Then, by Lemmas 18 and 19, we know that either there is an edge $u v \in E ( G )$ such that $G ( u v )$ , $G ( u v ) ( u _ { 0 } v _ { 0 } )$ and $G ( u v ) ( u _ { 1 } v _ { 1 } )$ are chordal, or $V ( G )$ can be partitioned into three sets, $A$ , $B$ and $K$ , such that $A \cup K$ and $B \cup K$ are complete and $K$ separates $A$ from $B$ . If the former is the case, we can proceed as in the previous paragraph to see that every chordal AC algorithm starting with $u _ { 0 } v _ { 0 }$ has the same complexity distribution as any of the chordal AC algorithms starting with $_ { u v }$ (note that such algorithms exist by Lemma 17), which, in turn, has the same complexity distribution as any of the chordal AC algorithms starting with $u _ { 1 } v _ { 1 }$ , leading to the desired conclusion.

So assume there are sets $A$ , $B$ and $K$ as above. By symmetry, we can assume that $u _ { 0 } , u _ { 1 } \in A$ and $v _ { 0 } , v _ { 1 } \in B$ . Consider the automorphism $\sigma$ of $G$ that maps $u _ { 0 }$ to $u _ { 1 }$ and $v _ { 0 }$ to $v _ { 1 }$ while keeping all other vertices fixed. We can now view $A _ { 0 }$ as an algorithm in $\sigma ( G )$ that starts with the edge $u _ { 1 } v _ { 1 }$ . By the induction hypothesis, we conclude that $A _ { 1 }$ (for $G$ ) has the same distribution as $A _ { 0 }$ (for $\sigma ( G )$ , and thus also for $G$ ). □

It remains to prove Lemmas 18 and 19. We start by giving a characterization of graphs that remain chordal when we add either one of two edges, but not if we add both.

Lemma 21. Let $G$ be a chordal graph and let $u _ { 0 } , u _ { 1 } , v _ { 0 } , v _ { 1 } \in V ( G )$ such that for $i = 0 , 1$ , vertices $u _ { 0 } , u _ { 1 } , v _ { i }$ are all distinct, $u _ { i } v _ { i } \notin E ( G )$ , and $G ( u _ { i } v _ { i } )$ is chordal. If $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is not chordal, then $K : = N ( u _ { 0 } ) \cap N ( u _ { 1 } ) \cap N ( v _ { 0 } ) \cap N ( v _ { 1 } )$ is complete, and $G - K$ has two distinct components $A$ and $B$ , such that either $u _ { 0 } , u _ { 1 } \in A$ and $v _ { 0 } , v _ { 1 } \in B$ , or $u _ { 0 } , v _ { 1 } \in A$ and $u _ { 1 } , v _ { 0 } \in B$ .

Proof. As $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ is not chordal, we know that $G ( u _ { 0 } v _ { 0 } ) ( u _ { 1 } v _ { 1 } )$ has an induced cycle $C =$ $( u _ { 1 } , v _ { 1 } , . . . , v _ { \ell - 1 } , v _ { \ell } , . . . , v _ { k } , u _ { 1 } )$ , with $k \geq \ell \geq 2$ , where either $v _ { \ell - 1 } = v _ { 0 }$ and $v _ { \ell } = u _ { 0 }$ , or ${ v \ell { - } 1 } = { u } _ { 0 }$ and $v _ { \ell } = v _ { 0 }$ . According to Lemma 13, since $G ( u _ { i } v _ { i } )$ is chordal, the set $K _ { i } : = N ( u _ { i } ) \cap N ( v _ { i } )$ is complete and separates $u _ { i }$ and $v _ { i }$ in $G$ , for each $i \in \{ 0 , 1 \}$ . Since $C$ has at least four vertices, and $K _ { i }$ has neighbors $u _ { i } , v _ { i }$ , we know that $V ( C ) \cap K _ { i } = \emptyset$ , for $i = 0 , 1$ .

Assume there is a vertex $x \in K _ { 0 } \setminus K _ { 1 }$ . Then $( v _ { 1 } , . . . , v _ { \ell - 1 } , x , v _ { \ell } , . . . , v _ { k } , u _ { 1 } )$ is a path in $G - K _ { 1 }$ , in contradiction to the fact that $K _ { 1 }$ separates $u _ { 1 }$ from $v _ { 1 }$ . So $K _ { 0 } \subseteq K _ { 1 }$ , and with the help of a symmetric argument we see that $K _ { 0 } = K _ { 1 }$ . In order to finish the proof it suffices to note that the paths $( v _ { 1 } , . . . , v _ { \ell - 1 } )$ and $( v _ { \ell } , . . . , v _ { k } )$ ensure that there are components $A$ and $B$ as desired. $\sqcup$

We now see that a graph that is obtained by gluing two graphs along a complete subgraph is chordal if and only if the two smaller graphs are.

Lemma 22. Let $G$ be a graph, let $A , B , K$ be a partition of $V ( G )$ such that $K$ is complete and there are no edges between $A$ and $B$ . Then $G$ is chordal if and only if $G [ A \cup K ]$ and $G [ K \cup B ]$ are both chordal.

Proof. As induced subgraphs of chordal graphs are chordal, we only need to show that if both $G [ A \cup K ]$ and $G [ K \cup B ]$ are both chordal, then also $G$ is. For this, it suffices to observe that any cycle of $G$ that contains vertices from both $A$ and $B$ has to pass twice through $K$ . □

We are now ready to prove Lemmas 18 and 19.

Proof of Lemma 18. Use Lemma 21 to see that the intersection $K$ of the neighborhoods of $u _ { 0 }$ , $v _ { 0 }$ , $u _ { 1 }$ , $v _ { 1 }$ is either a clique or empty, and $G - K$ has two connected components $A$ , $B$ such that $u _ { 0 } , u _ { 1 } \in A$ and $v _ { 0 } , v _ { 1 } \in B$ (after possibly changing the roles of $u _ { 0 }$ and $v _ { 0 }$ ).

As $G$ has no complete separation, and as there are no edges from $A$ to $B$ , one of $A \cup K$ , $B \cup K$ has to have a non-edge; because of symmetry we can assume this is $A \cup K$ . According to Lemma 22, the subgraph of $G$ induced by $A \cup K$ is chordal. Then, according to Lemma 17, there is also a non-edge $u v$ with $u , v \in A \cup K$ having the additional property that $G ( u v )$ is chordal. As $K$ is complete, we can assume that $u \in A$ .

If there is no non-edge $u v$ as desired, we have that $G ( u _ { i } v _ { i } ) ( u v )$ is non-chordal for some $i \in \{ 0 , 1 \}$ ; by symmetry, let us assume $G ( u _ { 1 } v _ { 1 } ) ( u v )$ is non-chordal. So, we may apply Lemma 21 to see that the intersection $K ^ { \prime }$ of the neighborhoods of $u$ , $v$ , $u _ { 1 }$ , $v _ { 1 }$ is either a clique or empty, and $G - K ^ { \prime }$ has two connected components $A ^ { \prime }$ , $B ^ { \prime }$ such that $u , u _ { 1 } \in A ^ { \prime }$ and $v , v _ { 1 } \in B ^ { \prime }$ (after possibly changing the roles of $u$ and $v$ ). Note that $K ^ { \prime } \subseteq N ( u _ { 1 } ) \cap N ( v _ { 1 } ) \subseteq K$ . Furthermore, $K \subseteq K ^ { \prime }$ , since $K ^ { \prime }$ separates $u _ { 1 }$ from $v _ { 1 }$ and $K \subseteq N ( u _ { 1 } ) \cap N ( v _ { 1 } )$ . So $K = K ^ { \prime }$ .

In particular, $v \not \in K$ , that is, $v \in A$ . So, as $v _ { 1 } \in B$ , we know that $v , v _ { 1 }$ lie in distinct components of $G - K$ . However, we also have that $v , v _ { 1 }$ belong to the same component (namely, $A$ ) of $G - K ^ { \prime } = G - K$ , a contradiction. So the desired non-edge uv exists. □

Proof of Lemma 19. We start by proving that $N ( u _ { 0 } ) \cap N ( v ) = N ( u _ { 1 } ) \cap N ( v )$ . For this assume there is an $i \in \{ 0 , 1 \}$ and a vertex $x \in ( N ( u _ { 1 - i } ) \cap N ( v ) ) \setminus N ( u _ { i } )$ . Then $( u _ { 1 - i } , x , v , u _ { i } , u _ { 1 - i } )$ is an induced cycle of length 4 in $G ( u _ { i } v )$ , a contradiction since this graph is chordal. This proves the equality, and we set $K : = N ( u _ { 0 } ) \cap N ( v ) = N ( u _ { 1 } ) \cap N ( v )$ .

Because of Lemma 13, $K$ is complete and separates $u _ { 0 } , u _ { 1 }$ from $\boldsymbol { v }$ . Since $G$ does not have a complete separation, at least one of $G [ A \cup K ]$ , $G [ B \cup K ]$ is not complete, but by Lemma 22 both are chordal. So by Lemma 17 and, again, Lemma 22, there is a non-edge $u w$ such that $G ( u w )$ is chordal. Now, if $u w$ is not as desired, say because $G ( u _ { 0 } v ) ( u w )$ is non-chordal, then there is an induced cycle $C$ of length at least 4 going through both $u w$ and $u _ { 0 } v$ . However, $C$ has to meet $K$ , which implies $C$ is a triangle, a contradiction. □

# A.4 Proof of Theorem 4

Symbolic method. To any sequence of numbers can be associated a generating function. For example, consider the sequence $( B _ { n } ) _ { n \geq 0 }$ , where the Bell number $B _ { n }$ denotes the number of partitions of size $n$ . The exponential generating function of partitions is then defined as

$$
P (z) = \sum_ {n \geq 0} B _ {n} \frac {z ^ {n}}{n !}.
$$

The sum can be expressed at the partition level as well

$$
P(z) = \sum_ {p \in \mathrm {P a r t i t i o n s} (n)} \frac {z ^ {| p |}}{| p | !}.
$$

The symbolic method, presented in [14], translates combinatorial descriptions into generating function equations. For example, since a partition is a set of nonempty sets, the exponential generating function of partitions is equal to

$$
P (z) = e ^ {\exp (z) - 1}.
$$

The reader unfamiliar with the symbolic method can verify this result by working on recurrences at the coefficient level. In this example, choosing a partition of size $n$ is equivalent with choosing its number of blocks $k$ , the size $n _ { j } \geq 1$ of the $j$ th block for each $1 \leq j \leq k$ , and finally the content of those blocks, so

$$
B_{n} = \sum_{k = 0}^{n}\sum_{\substack{n_{1} + \dots +n_{k} = n\\ \forall j, n_{j}\geq 1}}\frac{1}{k!}\binom {n}{n_{1},\ldots ,n_{k}}.
$$

Multiplying by $z ^ { n } / n !$ , summing over $n$ and reorganizing the terms, we indeed recover

$$
\sum_ {n \geq 0} B _ {n} \frac {z ^ {n}}{n !} = e ^ {\exp (z) - 1}.
$$

In the rest of the combinatorial proofs, the symbolic method will be preferred and we will let the motivated reader translate those proofs at the recurrence level.

$q$ -analogs. Several families of integer identities have been generalized by introducing $q$ -analogs. An introduction can be found in [12]. The $q$ -analog of integer $n$ is defined as

$$
[ n ] _ {q} = 1 + q + \dots + q ^ {n - 1} = \frac {1 - q ^ {n}}{1 - q}.
$$

The $q$ -factorial of the integer $n$ is

$$
[ n ] _ {q}! = \prod_ {j = 1} ^ {n} [ j ] _ {q}.
$$

def universal_acelement_set): if is_emptyelement_set): return EMPTY_PARTITION $\mathbf{u} =$ element_set.pop() block $=$ {u} for v in element_set: if query(u,v): block.add(v) element_set.remove(v) partition $=$ universal_acelement_set) partition.add_block(block) return partition

Figure 2: The universal AC algorithm considers an element, compare it to the other elements to find its block, then partition the remaining elements. It is named after the graph theory convention to call “universal ” a vertex linked to all vertices of a graph.

The $q$ -exponential is defined as

$$
e _ {q} (z) = \sum_ {n \geq 0} \frac {z ^ {n}}{[ n ] _ {q} !}.
$$

The $q$ -Pochhammer symbol is defined as

$$
(a; q) _ {n} = \prod_ {k = 0} ^ {n - 1} (1 - a q ^ {k})
$$

Observe that the $q$ -analog reduce to their classic counterparts for $q = 1$

Lemma 23. The $q$ -analogs satisfy the following classic identities

$$
[ n ] _ {q}! = \frac {(q ; q) _ {n}}{(1 - q) ^ {n}}, \quad \frac {1}{(x ; q) _ {\infty}} = \sum_ {n \geq 0} \frac {x ^ {n}}{(q ; q) _ {n}}, \quad e _ {q} (x) = ((1 - q) x; q) _ {\infty} ^ {- 1}.
$$

Characterizing the complexity generating function. According to Theorem 3, all chordal AC algorithms share the same distribution on the number of queries for random partitions of size $n$ chosen uniformly. Thus, we study one particular chordal algorithm: the universal $A C$ algorithm, presented in Figure 2. Let Partitions(n) denote the set of all partitions of size $n$ $| p |$ the size of the partition $p$ , and queries(p) the number of queries used by the universal AC algorithm to reconstruct the partition $p$ .

Theorem 24. The generating function

$$
P (z, q) = \sum_ {p \in \text {P a r t i t i o n s} (n)} q ^ {\text {q u e r i e s} (p)} \frac {z ^ {| p |}}{| p | !}
$$

is characterized by the differential equation

$$
\partial_ {z} P (z, q) = P (q z, q) e ^ {q z}
$$

and the initial condition $P ( 0 , q ) = 1$ .

Proof. Consider a partition $p$ of size $n$ . Let $b$ denote the set of all elements of the block of $p$ containing $n$ , except $n$ . Let $r$ denote the partition $p$ without the block containing $n$ . Then $p$ can be recovered from the pair $( r , b )$ as follows. The size $n$ of $p$ is $\left| \boldsymbol { r } \right| + \left| \boldsymbol { b } \right| + 1$ , so the block containing $n$ in $p$ was $b \cup \{ n \}$ and adding this block to $r$ recovers $p$ . We have just proven that this construction is a bijection between the partitions and the relabeled pairs containing a partition and a set. This bijection translates into the following identity on the generating function $P ( z )$ of partitions

$$
\partial_ {z} P (z) = P (z) e ^ {z}.
$$

This is no surprise, as we already know the expression of this generating function

$$
P (z) = e ^ {\exp (z) - 1}
$$

from the paragraph on the symbolic method, and it indeed satisfies this differential equation. However, the same approach is useful to study the generating function of the number of queries used by the universal AC algorithm.

Consider a partition $p$ and its decomposition as a pair $( r , b )$ described above. The universal AC algorithm starts by comparing one element, which is assumed to have the largest label without loss of generality, to the other elements. This requires $| r | + | b |$ queries. Then the algorithm is called recursively on the partition $r$ . Thus, the generating function of partitions with an additional parameter $q$ marking the number of queries used by the universal AC algorithm is characterized by the differential equation

$$
\partial_ {z} P (z, q) = P (q z, q) e ^ {q z}.
$$

If the initial partition is empty, then there are no queries to ask, which implies the initial condition $P ( 0 , q ) = 1$ . □

Exact expressions. The following theorem provides a solution for the differential equation from last Theorem.

Theorem 25. Let Poch $( z , a , q )$ denote the exponential generating function associated to the $q$ -Pochhammer symbol

$$
\operatorname {P o c h} (z, a, q) = \sum_ {k \geq 0} (a; q) _ {k} \frac {z ^ {k}}{k !},
$$

then the generating function of the universal AC algorithm complexity is

$$
P (z, q) = \mathrm {P o c h} \left(- \frac {q}{1 - q} z, \frac {1 - q}{q}, q\right) e ^ {\frac {q}{1 - q} z}
$$

Proof. The function $f ( z , q ) = e ^ { \frac { q } { 1 - q } z }$ satisfies a similar differential equation

$$
\partial_ {z} f (z, q) = \frac {q}{1 - q} f (q z, q) e ^ {q z}.
$$

Thus, we investigate solutions of the differential equation from Theorem 24 of the form $P ( z , q ) =$ $A ( z , q ) e ^ { \frac { q } { 1 - q } z }$ . The differential equation on $P ( z , q )$ implies the following differential equation for $A ( z , q )$

$$
\partial_ {z} A (z, q) + \frac {q}{1 - q} A (z, q) = A (q z, q).
$$

with initial condition $A ( 0 , q ) = 1$ . Decomposing $A ( z , q )$ as a series in $z$

$$
A (z, q) = \sum_ {k \geq 0} a _ {k} (q) \frac {z ^ {k}}{k !},
$$

we obtain a recurrence on the $a _ { k } ( q )$

$$
a _ {k + 1} (q) = - \frac {q}{1 - q} a _ {k} (q) + q ^ {k} a _ {k} (q) = - \frac {q}{1 - q} \left(1 - (1 - q) q ^ {k - 1}\right) a _ {k} (q),
$$

with $a _ { 0 } ( q ) = 1$ . We deduce

$$
a _ {k} (q) = \left(- \frac {q}{1 - q}\right) ^ {k} \prod_ {j = 0} ^ {k - 1} \left(1 - \frac {1 - q}{q} q ^ {j}\right) = \left(- \frac {q}{1 - q}\right) ^ {k} \left(\frac {1 - q}{q}; q\right) _ {k}.
$$

To conclude, we observe that $\textstyle \operatorname { P o c h } \left( - { \frac { q } { 1 - q } } z , { \frac { 1 - q } { q } } , q \right) e ^ { \frac { q } { 1 - q } z }$ is indeed solution of the differential equation characterizing $P ( z , q )$ .

Extracting the coefficient $n ! [ z ^ { n } ]$ from the solution, we obtain a first exact expression for $P _ { n } ( q )$ in the following Theorem, and another one will be provided in Theorem 27. This first expression is well suited for exact computations using a computer algebra system (we used [22] to verify our calculations). In particular, the $k$ th factorial moment of the random variable $X _ { n }$ counting the number of queries used by the universal AC algorithm on partitions of size $n$ chosen uniformly at random is

$$
\mathbb {E} (X _ {n} (X _ {n} - 1) \dots (X _ {n} - k + 1)) = \frac {1}{B _ {n}} \partial_ {q = 1} ^ {k} P _ {n} (q).
$$

Theorem 26. The generating function of the number of queries used by the universal AC algorithm on partitions of size n is

$$
P _ {n} (q) = \left(\frac {q}{1 - q}\right) ^ {n} \sum_ {k \geq 0} \binom {n} {k} (- 1) ^ {k} \left(\frac {1 - q}{q}; q\right) _ {k}.
$$

We provide a second expression for the complexity generating function, more elegant and better suited for asymptotics analysis. It is a $q$ -analog of the following classic formula for the Bell numbers, which counts the number of partitions of size $n$

$$
B _ {n} = \frac {1}{e} \sum_ {m \geq 0} \frac {m ^ {n}}{m !}.
$$

This formula is obtained from the generating function of partitions $P ( z ) = e ^ { \exp ( z ) - 1 }$

$$
B _ {n} = n! [ z ^ {n} ] e ^ {\exp (z) - 1} = \frac {n !}{e} [ z ^ {n} ] e ^ {\exp (z)} = \frac {n !}{e} [ z ^ {n} ] \sum_ {m \geq 0} \frac {e ^ {m z}}{m !} = \frac {1}{e} \sum_ {m \geq 0} \frac {m ^ {n}}{m !}.
$$

Theorem 27. The generating function of the number of queries used by the universal AC algorithm on partitions of size n is

$$
P _ {n} (q) = \frac {1}{e _ {q} (1 / q)} \sum_ {m \geq 0} \frac {[ m ] _ {q} ^ {n}}{[ m ] _ {q} !} q ^ {n - m}
$$

The sum converges for $q > 1 / 2$ .

Proof. The $q$ -Pochhammer generating function is rewritten as

$$
\begin{array}{l} \operatorname {P o c h} (z, a, q) = \sum_ {k \geq 0} (a; q) _ {k} \frac {z ^ {k}}{k !} \\ = (a; q) _ {\infty} \sum_ {k \geq 0} \frac {1}{(a q ^ {k} ; q) _ {\infty}} \frac {z ^ {k}}{k !} \\ = (a; q) _ {\infty} \sum_ {k \geq 0} \sum_ {m \geq 0} [ x ^ {m} ] \frac {1}{(a x ; q) _ {\infty}} q ^ {m k} \frac {z ^ {k}}{k !} \\ = (a; q) _ {\infty} \sum_ {m \geq 0} a ^ {m} [ x ^ {m} ] \frac {1}{(x ; q) _ {\infty}} e ^ {q ^ {m} z} \\ \end{array}
$$

Applying Lemma 23 we conclude

$$
\operatorname {P o c h} (z, a, q) = (a; q) _ {\infty} \sum_ {m \geq 0} \frac {a ^ {m}}{(q ; q) _ {m}} e ^ {q ^ {m} z}
$$

Injecting this in the expression of $P _ { n } ( q )$ , we obtain

$$
\begin{array}{l} P _ {n} (q) = n! [ z ^ {n} ] \operatorname {P o c h} \left(- \frac {q}{1 - q} z, \frac {1 - q}{q}, q\right) e ^ {\frac {q}{1 - q} z} \\ = \left(\frac {q}{1 - q}\right) ^ {n} \left(\frac {1 - q}{q}; q\right) _ {\infty} n! [ z ^ {n} ] \sum_ {m \geq 0} \frac {\left(\frac {1 - q}{q}\right) ^ {m}}{(q ; q) _ {m}} e ^ {- q ^ {m} z} e ^ {z} \\ = \left(\frac {q}{1 - q}\right) ^ {n} \left(\frac {1 - q}{q}; q\right) \sum_ {\infty} \frac {\left(\frac {1 - q}{q}\right) ^ {m}}{(q ; q) _ {m}} (1 - q ^ {m}) ^ {n} \\ = q ^ {n} \left(\frac {1 - q}{q}; q\right) \sum_ {\infty} \frac {\left(\frac {1 - q}{q}\right) ^ {m}}{(q ; q) _ {m}} [ m ] _ {q} ^ {n}. \\ \end{array}
$$

To conclude, Lemma 23 is applied

$$
P _ {n} (q) = \frac {1}{e _ {q} (1 / q)} \sum_ {m \ge 0} \frac {[ m ] _ {q} ^ {n}}{[ m ] _ {q} !} q ^ {n - m}.
$$

We apply d’Alembert’s criteria to find the values of $q$ for which this formal sum converges:

$$
\lim _ {m \to \infty} \frac {\frac {[ m + 1 ] _ {q} ^ {n}}{[ m + 1 ] _ {q} !} q ^ {n - m - 1}}{\frac {[ m ] _ {q} ^ {n}}{[ m ] _ {q} !} q ^ {n - m}} = \frac {1 - q}{q}
$$

is smaller than 1 when $q > 1 / 2$ .

Limit law.

Lemma 28. As n and m tend to infinity, $q = e ^ { s }$ , ms and ns tend to 0, we have

$$
\begin{array}{l} [ m ] _ {q} ^ {n} = m ^ {n} \exp \left(\frac {1}{2} n m s + \frac {1}{1 2} n m ^ {2} \frac {s ^ {2}}{2}\right) (1 + \mathcal {O} (n m ^ {3} s ^ {3} + n s)) \\ [ m ] _ {q}! = m ^ {m} e ^ {- m} \sqrt {2 \pi [ m ] _ {q}} \exp \left(\frac {1}{4} m ^ {2} s + \frac {1}{3 6} m ^ {3} \frac {s ^ {2}}{2}\right) \left(1 + \mathcal {O} (m ^ {4} s ^ {3} + m s) + o (1)\right). \\ \end{array}
$$

Proof. Let $S ( x )$ denote the function $( e ^ { x } - 1 - x ) / x$ , then

$$
[ m ] _ {q} ^ {n} = \left(\frac {1 - e ^ {s m}}{1 - e ^ {s}}\right) ^ {n} = m ^ {n} \left(\frac {1 + S (s m)}{1 + S (s)}\right) ^ {n} = m ^ {n} e ^ {n \log (1 + S (s m)) - n \log (1 + S (s))}.
$$

We use the development

$$
\log (1 + S (x)) = \log \left(1 + \frac {x}{2} + \frac {x ^ {2}}{6} + \mathcal {O} (x ^ {3})\right) = \frac {x}{2} + \frac {x ^ {2}}{2 4} + \mathcal {O} (x ^ {3})
$$

to obtain

$$
[ m ] _ {q} ^ {n} = m ^ {n} \exp \left(\frac {1}{2} n m s + \frac {1}{1 2} n m ^ {2} \frac {s ^ {2}}{2} + \mathcal {O} (n m ^ {3} s ^ {3} + n s)\right)
$$

According to Moak [20], we have the following $q$ -analog of Stirling formula when $x \to \infty$ while $x \log ( q ) \to 0$

$$
\log (\Gamma_ {q} (x)) = (x - 1 / 2) \log ([ x ] _ {q}) + \frac {\operatorname {L i} _ {2} (1 - q ^ {x})}{\log (q)} + \frac {1}{2} \log (2 \pi) + o (1)
$$

where $\operatorname { L i } _ { 2 } ( z )$ denotes the Dilogarithm function

$$
\operatorname {L i} _ {2} (z) = \sum_ {k \geq 1} \frac {z ^ {k}}{k ^ {2}}
$$

We deduce

$$
\begin{array}{l} [ m ] _ {q}! = [ m ] _ {q} \Gamma_ {q} (m) = [ m ] _ {q} \exp \left((m - 1 / 2) \log ([ m ] _ {q}) + \frac {\operatorname {L i} _ {2} (1 - q ^ {m})}{\log (q)} + \frac {1}{2} \log (2 \pi) + o (1)\right) \\ = [ m ] _ {q} ^ {m} \exp \left(\frac {\operatorname {L i} _ {2} (1 - q ^ {m})}{\log (q)}\right) \sqrt {2 \pi [ m ] _ {q}} (1 + o (1)) \\ \end{array}
$$

The first part of the lemma provides

$$
[ m ] _ {q} ^ {m} = m ^ {m} \exp \left(\frac {1}{2} m ^ {2} s + \frac {1}{1 2} m ^ {3} \frac {s ^ {2}}{2}\right) \left(1 + \mathcal {O} \left(m ^ {4} s ^ {3} + m s\right)\right).
$$

The Dilogarithm is expanded as

$$
\frac {\operatorname {L i} _ {2} (1 - q ^ {m})}{\log (q)} = \frac {1}{s} \sum_ {k \geq 1} \frac {1}{k ^ {2}} \left(- m s \left(1 + S (m s)\right)\right) ^ {k} = - m \left(1 + \frac {1}{4} m s + \frac {1}{1 8} m ^ {2} \frac {s ^ {2}}{2} + \mathcal {O} (m s) ^ {3}\right).
$$

Injecting those past two expansions in the previous one concludes the proof.

Theorem 29. The asymptotic mean $E _ { n }$ and standard deviation $\sigma _ { n }$ of the number $X _ { n }$ of queries used by the universal AC algorithm on a partition of size n chosen uniformly at random are

$$
E _ {n} = \frac {1}{4} (2 \zeta - 1) e ^ {2 \zeta} a n d \sigma_ {n} = \frac {1}{3} \sqrt {\frac {3 \zeta^ {2} - 4 \zeta + 2}{\zeta + 1} e ^ {3 \zeta}},
$$

where $\zeta$ is the unique positive solution of

$$
\zeta e ^ {\zeta} = n.
$$

The normalized random variable

$$
X _ {n} ^ {\star} = \frac {X _ {n} - E _ {n}}{\sigma_ {n}}
$$

follows in the limit a normalized Gaussian law.

Those results are tested numerically in Figures 3 and 4.

Proof. To prove the limit law, we show that the Laplace transform $\mathbb { E } ( e ^ { s X _ { n } ^ { \star } } )$ converges pointwise to the Laplace transform of the normalized Gaussian $e ^ { s ^ { 2 } / 2 }$ . We have

$$
\mathbb {E} (e ^ {s X _ {n} ^ {\star}}) = e ^ {- s E _ {n} / \sigma_ {n}} \mathbb {E} (e ^ {s X _ {n} / \sigma_ {n}}) = \frac {e ^ {- s E _ {n} / \sigma_ {n}}}{B _ {n}} P _ {n} (e ^ {s / \sigma_ {n}})
$$

For any fixed real value $s$ , we compute the asymptotics of $P _ { n } ( e ^ { s / \sigma _ { n } } )$ . Let $q : = e ^ { s / \sigma _ { n } }$ , so $q$ tends to 1, then

$$
P _ {n} (e ^ {s / \sigma_ {n}}) = \frac {1}{e _ {q} (e ^ {- s / \sigma_ {n}})} \sum_ {m \geq 0} \frac {[ m ] _ {q} ^ {n}}{[ m ] _ {q} !} e ^ {(n - m) s / \sigma_ {n}}.
$$

Motivated by the asymptotics from Lemma 28, we rewrite this expression as

$$
P _ {n} \left(e ^ {s / \sigma_ {n}}\right) = \frac {1}{e _ {q} \left(e ^ {- s / \sigma_ {n}}\right)} \sum_ {m \geq 0} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \tag {2}
$$

where

$$
A _ {n, s} (m) = \frac {[ m ] _ {q} ^ {n}}{m ^ {n} \exp \left(\frac {1}{2} n m \frac {s}{\sigma_ {n}} + \frac {1}{1 2} n m ^ {2} \frac {(s / \sigma_ {n}) ^ {2}}{2}\right)} \frac {m ^ {m} e ^ {- m} \exp \left(\frac {1}{4} m ^ {2} \frac {s}{\sigma_ {n}} + \frac {1}{3 6} m ^ {3} \frac {(s / \sigma_ {n}) ^ {2}}{2}\right)}{[ m ] _ {q} !} e ^ {(n - m) s / \sigma_ {n}},
$$

$$
\phi_ {n, s} (m) = - n \log (m) + m \log (m) - m - \frac {1}{4} (2 n - m) m \frac {s}{\sigma_ {n}} - \frac {1}{3 6} (3 n - m) m ^ {2} \frac {(s / \sigma_ {n}) ^ {2}}{2}.
$$

The dominant contribution to the sum comes from integers $m$ close to the minimum of $\phi _ { n , s } ( m )$ , so we study this function. The successive derivatives of $\phi _ { n , s } ( m )$ are

$$
\phi_ {n, s} ^ {\prime} (m) = - \frac {n}{m} + \log (m) - \frac {1}{2} (n - m) \frac {s}{\sigma_ {n}} - \frac {1}{1 2} (2 n - m) m \frac {(s / \sigma_ {n}) ^ {2}}{2},
$$

$$
\phi_ {n, s} ^ {\prime \prime} (m) = \frac {n}{m ^ {2}} + \frac {1}{m} + \frac {s}{2 \sigma_ {n}} - \frac {1}{6} (n - m) \frac {(s / \sigma_ {n}) ^ {2}}{2}
$$

$$
\phi_ {n, s} ^ {\prime \prime \prime} (m) = - \frac {2 n}{m ^ {3}} - \frac {1}{m ^ {2}} + \frac {(s / \sigma_ {n}) ^ {2}}{1 2}
$$

When $n$ is large enough, the second derivative of $\phi _ { n , s } ( m )$ is strictly positive for all $m > 0$ , so the function is convexe. It reaches its unique minimum at a value denoted by $m ( s )$ and characterized by $\phi _ { n , s } ^ { \prime } ( m ( s ) ) = 0$ . Injecting the Taylor expansion

$$
m (s) = m _ {0} + m _ {1} \frac {s}{\sigma_ {n}} + m _ {2} \frac {(s / \sigma_ {n}) ^ {2}}{2} + \dots
$$

in this equation, rewriting $n$ as $\zeta e ^ { \zeta }$ and extracting the coefficients of the powers of $s$ , we obtain

$$
m _ {0} = e ^ {\zeta}, \qquad m _ {1} = \frac {1}{2} \frac {\zeta - 1}{\zeta + 1} e ^ {2 \zeta}, \qquad m _ {2} = \frac {1}{3} \frac {2 \zeta^ {3} - 3 \zeta^ {2} + 2}{(\zeta + 1) ^ {3}} e ^ {3 \zeta}
$$

The dominant contribution to the sum defining $P _ { n } ( e ^ { s / \sigma _ { n } } )$ comes from values $m$ close to $m ( s )$ . The central part $C _ { n }$ is defined as the integers $m$ such that $| m - m ( s ) | < c _ { n }$ . A heuristic proposed by [14] is to find $c _ { n }$ such that

$$
| \phi_ {n, s} ^ {\prime \prime} (m (s)) | c _ {n} ^ {2} \to + \infty \quad \mathrm {a n d} \quad | \phi_ {n, s} ^ {\prime \prime \prime} (m (s)) | c _ {n} ^ {3} \to 0.
$$

As $n$ , and thus $\zeta$ , tend to infinity, we have

$$
m (s) \sim e ^ {\zeta} \sim \frac {n}{\log (n)}, \qquad | \phi_ {n, s} ^ {\prime \prime} (m (s)) | \sim \zeta e ^ {- \zeta} \sim \frac {\log (n) ^ {2}}{n}, \qquad | \phi_ {n, s} ^ {\prime \prime \prime} (m (s)) | \sim \zeta e ^ {- 2 \zeta} \sim \frac {\log (n) ^ {3}}{n ^ {2}},
$$

so we choose $c _ { n } = e ^ { 3 \zeta / 5 }$ . Uniformly for $m$ in $C _ { n }$ , we have

$$
m (s) = e ^ {\zeta} + \frac {1}{2} \frac {\zeta - 1}{\zeta + 1} e ^ {2 \zeta} \frac {s}{\sigma_ {n}} + \frac {1}{3} \frac {2 \zeta^ {3} - 3 \zeta^ {2} + 2}{(\zeta + 1) ^ {3}} e ^ {3 \zeta} \frac {(s / \sigma_ {n}) ^ {2}}{2} + \mathcal {O} (e ^ {- \zeta / 2}),
$$

$$
\phi_ {n, s} (m) = - (\zeta^ {2} - \zeta + 1) e ^ {\zeta} - E _ {n} \frac {s}{\sigma_ {n}} - \frac {s ^ {2}}{2} + (\zeta + 1) e ^ {- \zeta} \frac {(m - m (s)) ^ {2}}{2} + \mathcal {O} (e ^ {- \zeta / 4}),
$$

$$
A _ {n, s} (m) = \frac {1}{\sqrt {2 \pi e ^ {\zeta}}} \left(1 + o (1)\right).
$$

In fact, computing the Taylor expansion of $\phi _ { n , s } ( m )$ at $m ( s )$ came first. We chose the values of $E _ { n }$ and $\sigma _ { n }$ so that the coefficients in $s$ and $s ^ { 2 }$ are the ones presented in the above equation. The error term is then obtained using the Lagrange form of the remainder in Taylor’s Theorem. We deduce the following asymptotics for the central part of the sum

$$
\sum_ {m \in C _ {n}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \sim \frac {1}{\sqrt {2 \pi e ^ {\zeta}}} e ^ {(\zeta^ {2} - \zeta + 1) e ^ {\zeta} + E _ {n} \frac {s}{\sigma_ {n}} + \frac {s ^ {2}}{2}} \sum_ {m \in C _ {n}} e ^ {- (\zeta + 1) e ^ {- \zeta} \frac {(m - m (s)) ^ {2}}{2}}.
$$

Applying the Euler-Maclaurin formula to turn the sum into an integral, we obtain

$$
\sum_ {m \in C _ {n}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \sim \frac {1}{\sqrt {2 \pi e ^ {\zeta}}} e ^ {(\zeta^ {2} - \zeta + 1) e ^ {\zeta} + E _ {n} s / \sigma_ {n} + s ^ {2} / 2} \int_ {- c _ {n}} ^ {c _ {n}} e ^ {- (\zeta + 1) e ^ {- \zeta} \frac {x ^ {2}}{2}} d x.
$$

After the variable change $y = \sqrt { ( \zeta + 1 ) e ^ { - \zeta } } x$ , observing that $\sqrt { ( \zeta + 1 ) e ^ { - \zeta } } c _ { n }$ tends to infinity, the integral is approximated as a Gaussian integral and we conclude

$$
\sum_ {m \in C _ {n}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \sim \frac {1}{\sqrt {\zeta + 1}} e ^ {(\zeta^ {2} - \zeta + 1) e ^ {\zeta} + E _ {n} s / \sigma_ {n} + s ^ {2} / 2}.
$$

When we compare the asymptotics of the central part to the asymptotics of the Bell numbers (see, e.g., [14])

$$
B _ {n} \sim \frac {e ^ {(\zeta^ {2} - \zeta + 1) e ^ {\zeta} - 1}}{\sqrt {\zeta + 1}},
$$

we see from Equation (2) that, as expected,

$$
\frac {e ^ {- s E _ {n} / \sigma_ {n}}}{B _ {n}} \frac {1}{e _ {q} (e ^ {- s / \sigma_ {n}})} \sum_ {m \in C _ {n}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \sim e ^ {s ^ {2} / 2}.
$$

Let us now prove that the part of the sum corresponding to $m \leq m ( s ) - c _ { n }$ is negligible compared to the central part. According to Lemma 28, we have

$$
A _ {n, s} (m) = \left(1 + \mathcal {O} (n m ^ {3} / \sigma_ {n} ^ {3})\right) \left(1 + \mathcal {O} (m ^ {4} / \sigma_ {n} ^ {3})\right) \frac {1}{\sqrt {2 \pi [ m ] _ {q}}} = \mathcal {O} (m ^ {8}).
$$

Since $\phi _ { n , s } ( m )$ is convexe (for large enough $n$ ), we have $\phi _ { n , s } ( m ) \geq \phi _ { n , s } ( m ( s ) - c _ { n } )$ for all $m < m ( s ) - c _ { n }$ . Since

$$
\phi_ {n, s} (m (s) - c _ {n}) = - (\zeta^ {2} - \zeta + 1) e ^ {\zeta} - E _ {n} \frac {s}{\sigma_ {n}} - \frac {s ^ {2}}{2} + (\zeta + 1) e ^ {- \zeta} \frac {c _ {n} ^ {2}}{2} + \mathcal {O} (e ^ {- \zeta / 4})
$$

and $e ^ { - \zeta } c _ { n } ^ { 2 }$ tends to infinity as $e ^ { \zeta / 5 }$ , we obtain for all $m < m ( s ) - c _ { n }$

$$
\phi_ {n, s} (m) \geq - (\zeta^ {2} - \zeta + 1) e ^ {\zeta} - E _ {n} \frac {s}{\sigma_ {n}} - \frac {s ^ {2}}{2} + \Theta (e ^ {\zeta / 5}).
$$

We conclude

$$
\begin{array}{l} \sum_ {m <   m (s) - c _ {n}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \leq \sum_ {m <   m (s) - c _ {n}} \mathcal {O} \left(m ^ {8}\right) e ^ {\left(\zeta^ {2} - \zeta + 1\right) e ^ {\zeta} + E _ {n} s / \sigma_ {n} + s ^ {2} / 2 - \Theta \left(\exp \left(\zeta / 5\right)\right)} \\ \leq e ^ {(\zeta^ {2} - \zeta + 1) e ^ {\zeta} + E _ {n} s / \sigma_ {n} + s ^ {2} / 2} m (s) ^ {9} e ^ {- \Theta (\exp (\zeta / 5))}. \\ \end{array}
$$

Since $m ( s ) \sim e ^ { \zeta }$ , this result is exponentially small, with respect to $n$ , compared to the central part. Let us now prove that the part of the sum beyond the central part is negligible as well. There is a constant $C$ such that for $n$ large enough and any $m \geq C e ^ { 3 \zeta / 2 }$ , we have

$$
- \frac {1}{4} (2 n - m) m \frac {s}{\sigma_ {n}} - \frac {1}{3 6} (3 n - m) m ^ {2} \frac {(s / \sigma_ {n}) ^ {2}}{2} \geq 0.
$$

In that case, we obtain the simple bound

$$
\phi_ {n, s} (m) \geq - n \log (m) + m \log (m) - m \geq m - n \log (m).
$$

Injecting this bound and $A _ { n , s } ( m ) = \mathcal { O } ( m ^ { \mathrm { s } } )$ in the sum, we obtain

$$
\sum_ {m \geq C e ^ {3 \zeta / 2}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \leq \mathcal {O} (1) \sum_ {m \geq C e ^ {3 \zeta / 2}} m ^ {n + 8} e ^ {- m}.
$$

The sum is bounded by an integral and $n + 8$ integration by part are applied

$$
\begin{array}{l} \sum_ {m \geq C e ^ {3 \zeta / 2}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \leq \mathcal {O} (1) (n + 8)! (C e ^ {3 \zeta / 2}) ^ {n + 8} e ^ {- C \exp (3 \zeta / 2)} \\ \leq \mathcal {O} (1) n ^ {n} \left(C e ^ {3 \zeta / 2}\right) ^ {n + 8} e ^ {- C \exp (3 \zeta / 2)} \\ \end{array}
$$

Since $n = \zeta e ^ { \zeta }$ , we have $n ^ { n } = e ^ { \mathcal { O } ( \zeta ^ { 2 } ) \exp ( \zeta ) }$ , so

$$
\sum_ {m \geq C e ^ {3 \zeta / 2}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \leq \mathcal {O} (1) e ^ {- \Theta (\exp (3 \zeta / 2))}
$$

which is negligible compared to the central part of the sum. The last part we consider is $m ( s ) + c _ { n } \leq m \leq C e ^ { 3 \zeta / 2 }$ . Since $\phi _ { n , s } ( m )$ is decreasing there and $A _ { n , s } ( m ) = \mathcal { O } ( m ^ { \mathrm { s } } )$ , we have

$$
\sum_ {m = m (s) + c _ {n}} ^ {C e ^ {3 \zeta / 2}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} = \mathcal {O} (e ^ {2 7 \zeta / 2}) e ^ {- \phi_ {n, s} (m (s) + c _ {n})}
$$

As for the case $m = m ( s ) - c _ { n }$ , we find

$$
\phi_ {n, s} (m (s) + c _ {n}) = - (\zeta^ {2} - \zeta + 1) e ^ {\zeta} - E _ {n} \frac {s}{\sigma_ {n}} - \frac {s ^ {2}}{2} + \Theta (e ^ {\zeta / 5})
$$

and conclude

$$
\sum_ {m = m (s) + c _ {n}} ^ {C e ^ {3 \zeta / 2}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} = e ^ {(\zeta^ {2} - \zeta + 1) e ^ {\zeta} + E _ {n} \frac {s}{\sigma n} + \frac {s ^ {2}}{2}} \mathcal {O} (e ^ {2 7 \zeta / 2}) e ^ {- \Theta (e ^ {\zeta / 5})},
$$

![](images/d65872b7c41684396e837f409d40a5f0c43ef290a5cb3696fa45994ea37e5188.jpg)  
Figure 3: In green, the probability density function of the normal distribution. In blue, purple and red, the empirical density functions for the number of queries, normalized by their mean and standard deviation, for $n$ in $\{ 1 0 0 , 3 0 0 , 6 0 0 \}$ . We observe a slow convergence to the Gaussian limit law.

which is negligible compared to the central part. In conclusion, we have

$$
\mathbb {E} \left(e ^ {s X _ {n} ^ {\star}}\right) = \frac {e ^ {- s E _ {n} / \sigma_ {n}}}{B _ {n}} P _ {n} \left(e ^ {s / \sigma_ {n}}\right) \sim \sum_ {m \in C _ {n}} A _ {n, s} (m) e ^ {- \phi_ {n, s} (m)} \sim e ^ {s ^ {2} / 2}.
$$

Since the Laplace transform of $X _ { n } ^ { \star }$ converges pointwise to the Laplace transform of the normalized Gaussian law, $X _ { n } ^ { \star }$ converges in distribution to this Gaussian.

# A.5 Proof of Theorem 6

Suppose first that $p _ { 1 } > \cdots > p _ { k }$ . At time √ $t$ , we have identified classes √ $C _ { 1 } ^ { t } , \ldots , C _ { k } ^ { t }$ , say with sizes $c + i ^ { t } = | C _ { i } ^ { t } |$ . It takes time at most $k \sqrt { n }$ to process the first $\sqrt { n }$ vertices, and at that point and all subsequent times we have (with exponentially small failure probability) that $c _ { i } ^ { t } \sim p _ { i } \sqrt { n }$ for each $_ i$ , where $C _ { i } ^ { t }$ consists of elements of type $i$ . In particular $| C _ { 1 } ^ { t } | > | C _ { 2 } ^ { t } | \ge \cdots > | C _ { k } ^ { t } |$ . The expected number of comparisons used for each subsequent element is $\sum i = 1 ^ { k } p _ { i }$ , as the probability that the element is of type $i$ is $p _ { i }$ , and in this case we need $_ i$ comparisons to place it in $C _ { i } ^ { t }$ .

The case where some of the $p _ { i }$ are the same is similar, except that if $p _ { i } = p _ { j }$ then $c _ { i } ^ { t }$ and $c _ { j } ^ { t }$ may switch order over time. However this is fine: we need only have the property that if $p _ { i } > p _ { j }$ then $c _ { i } ^ { t } > c _ { j } ^ { t }$ , and the same argument then works, possibly permuting colors for classes $i , j$ with $p _ { i } = p _ { j }$ .

![](images/168920bc28f93f5490b26139b64294372ff0e6068b5509ccd94e24aff54fb458.jpg)  
Figure 4: A plot testing the asymptotic mean and standard deviations stated by Theorem 4. Let $\alpha _ { n }$ and $\beta _ { n }$ denote the sequences defined by $\mathbb { E } ( X _ { n } ) ~ = ~ \frac { 1 } { 4 } ( 2 W ( n ) - 1 + \alpha _ { n } ) e ^ { 2 W ( n ) }$ and σ(Xn) = 13 q $\begin{array} { r } { \sigma ( X _ { n } ) = \frac { 1 } { 3 } \sqrt { \frac { 3 W ( n ) ^ { 2 } - 4 W ( n ) + 2 - \beta _ { n } } { W ( n ) + 1 } } e ^ { 3 W ( n ) } } \end{array}$ 3W (n)2−4W (n)+2−βn e3W (n). Plot of αn in blue, βn in red, for n from 3 to 2000. We $\alpha _ { n }$ $\beta _ { n }$ $n$ observe a slow convergence to 0.

# A.6 Proof of Theorem 12

The idea of the proof is as follows: we estimate the behaviour of the process until large linear time, and then note that it finishes in negligible time. Let us note first that in any partition of a set of size $t$ , where $t$ is large, a fraction of at least (about) $1 / k$ of the pairs lie inside classes. It follows that at each time step, we reduce the number of vertices by 1 with constant probability, and so the the algorithm finishes in expected time $O ( k t )$ . When handling an instance of size $n$ , it is therefore enough to run the algorithm until $o ( n )$ vertices remain, and then note that expected time to complete is still $o ( n )$ .

We consider an instance of size $n$ , and analyze the following process. Begin with all vertices marked active. At each time step, pick (with replacement) a random pair $\{ u , v \}$ and:

• If $u , v$ are active and from distinct classes $i , j$ then say we have generated an $_ { i j }$ -crossedge.   
• If $u , v$ , are both active and in class $i$ then mark exactly one of $u$ and $v$ inactive.   
• If one of $u , v$ is inactive, then do nothing.

Note that as the process runs, the number of active vertices is monotonic decreasing, and we are increasingly likely to choose pairs where one vertex is inactive. These contribute to the new process, but do not generate new comparisons between pairs. So we are looking at a (randomly) slowed down version of the random algorithm; but this makes the analysis much simpler!

Let $x _ { i } ( t )$ denote the number of active class $_ i$ vertices after $t$ time steps, and $x _ { i j } ( t )$ denote the number of $_ { i j }$ -crossedges that are generated in the first $t$ time steps. Then at step $t + 1$ , the probability that we pick two active vertices in class $_ i$ is

$$
\left( \begin{array}{c} x _ {i} (t) \\ 2 \end{array} \right) / \left( \begin{array}{c} n \\ 2 \end{array} \right) \sim \frac {x _ {i} (t) ^ {2}}{n ^ {2}}.
$$

Writing $p = p _ { i }$ , we estimate $x _ { i } ( t )$ via a function $x = x ( t )$ satisfying the differential equation

$$
x (0) = p n
$$

$$
x ^ {\prime} (t) = - \frac {x (t) ^ {2}}{n ^ {2}}.
$$

This has solution

$$
x (t) = \frac {n ^ {2}}{t + n / p} = \frac {p n}{1 + p t / n}.
$$

Note that at time $\lambda n$ , as $\lambda$ gets large, we get

$$
x (\lambda n) \sim \frac {p n}{1 + \lambda p}.
$$

The actual value of $x _ { i } ( t )$ closely tracks the differential equation with very high probability. This follows straightforwardly from a standard application of the Rödl nibble or differential equation method (see for example [3]): we throw in the edges in batches of size $\epsilon n$ , and note that a Chernoff bound implies that we stick close to the the differential equation as we are in total making $O ( n )$ comparisons.

Now let’s estimate the number of $_ { i j }$ -crossedges. Using our estimates for $x _ { i } ( t )$ and $x _ { j } ( t )$ , we see that the probability of an $i j$ crossedge at step $t + 1$ is

$$
x _ {i} (t) x _ {j} (t) / \binom {n} {2} \sim \frac {2 x _ {i} (t) x _ {j} (t)}{n ^ {2}} \approx \frac {2 n ^ {2}}{(t + n / p _ {i}) (t + n / p _ {j})}.
$$

Let $p = p _ { i }$ and $q = p _ { j }$ . We can model the growth of $x _ { i j } ( t )$ by a function $c = c ( t )$ satisfying the differential equation

$$
c (0) = 0
$$

$$
c ^ {\prime} (t) = \frac {2 n ^ {2}}{(t + n / p) (t + n / q)}.
$$

If $p = q$ , we have

$$
c ^ {\prime} (t) = \frac {2 n ^ {2}}{(t + n / p) ^ {2}}
$$

and

$$
c (t) = \int_ {0} ^ {t} c ^ {\prime} (s) d s = \int_ {0} ^ {t} \frac {2 n ^ {2}}{(s + n / p) ^ {2}} d s = \left[ \frac {- 2 n ^ {2}}{s + n / p} \right] _ {0} ^ {t},
$$

giving

$$
c (t) = 2 p n \left(1 - \frac {1}{2 + 2 p t / n}\right) \sim 2 p n
$$

as $t / n  \infty$

If $p \neq q$ , we have

$$
\begin{array}{l} c (t) = \int_ {0} ^ {t} \frac {2 n ^ {2}}{(s + n / p) (s + n / q)} d s \\ = \frac {2 n p q}{p - q} \int_ {0} ^ {t} \frac {1}{s + n / p} - \frac {1}{s + n / q} d s \\ = \frac {2 n p q}{p - q} \ln \left(\frac {1 + p t / n}{1 + q t / n}\right) \\ \sim \frac {2 n p q \ln (p / q)}{p - q} \\ \end{array}
$$

as $t / n  \infty$ . Combining this with the remarks at the beginning of the proof gives the result.

# A.7 Proofs of Section 2.3

Consider an AC algorithm applied to a set of size $n$ . Let $Q$ denote the set of queries with their answers when the algorithm terminates. We associate to $Q$ a graph $G$ where each vertex corresponds to an element of the starting set, each edge is either positive or negative and corresponds to the answers from $Q$ . As stated in Paragraph Correcting errors, a contradiction is detected in $Q$ if and only if $G$ contains a contradictory cycle (a cycle where all edges except one are positive). In that case, additional queries are submitted until the responsible answers are identified and corrected.

If $Q$ contains no more contradiction, it characterizes a set partition $p$ (otherwise, the AC algorithm has not terminated) of size $n$ with a number of blocks denoted by $b$ . However, this partition might not be the correct solution if $Q$ contains undetected errors. Assume each answer is wrong with a small probability $p$ and let P(undetected error $|  { Q }$ ) denote the probability that $Q$ contains errors but no contradictory cycle. Its Taylor coefficients at $p = 0$ are denoted by $( c _ { i } ) _ { i \geq 0 }$

$$
\mathbb {P} (\text {u n d e t e c t e d e r r o r} \mid Q) = c _ {0} + c _ {1} p + c _ {2} p ^ {2} + \dots
$$

We would like $Q$ to contain few additional queries compared to the noiseless case, while ensuring that P(undetected error $|  { Q }$ ) is small. Since $p$ is assumed to be small, the second conditions corresponds to minimizing the vector $( c _ { 0 } , c _ { 1 } , \ldots )$ for the lexicographic order. Indeed, if we consider a larger vector $( c _ { 0 } ^ { \prime } , c _ { 1 } ^ { \prime } , \ldots )$ , then there is a positive $\epsilon$ such that for any $0 < p < \epsilon$ , we have

$$
c _ {0} + c _ {1} p + c _ {2} p ^ {2} + \dots <   c _ {0} ^ {\prime} + c _ {1} ^ {\prime} p + c _ {2} p ^ {2} + \dots
$$

Let $d _ { k }$ denote the number of sets of $k$ edges from $G$ which could be switched (positive edges become negative ones, negative edges become positive ones) without creating a contradictory cycle. This corresponds to the number of ways to change $k$ answers in $Q$ and obtain a result without contradiction. Let $m$ denote the total number of edges in $G$ . Then the probability that there is an undetected error is

$$
\sum_ {k \geq 0} d _ {k} p ^ {k} (1 - p) ^ {m - k}.
$$

Since this probability is $c _ { 0 } + c _ { 1 } p + c _ { 2 } p ^ { 2 } + \cdot \cdot \cdot$ , we deduce for all k ≥ 0

$$
c _ {k} = \sum_ {j = 0} ^ {k} d _ {j} [ x ^ {k} ] x ^ {j} (1 - x) ^ {m - j}.
$$

The triangular shape of this system of equations implies that the vector $( c _ { 0 } , c _ { 1 } , \ldots )$ is minimal for the lexicographic order if and only if the vector $( d _ { 0 } , d _ { 1 } , \ldots )$ is minimal for the lexicographic order.

Since $G$ contains no contradictory cycle, we have $d _ { 0 } = 0$ . If all positive components of $G$ not reduced to one vertex are 2-edge-connected (i.e. they have no vertex of degree 1), then switching a positive answer must create a contradictory cycle. If every pair of positive components from $G$ is linked by at least 2 negative edges, then switching a negative answer must create a contradictory cycle. Thus, if $G$ satisfies those two conditions, we have $d _ { 1 } = 0$ .

In order to make $d _ { 2 }$ vanish, we would need every positive component of $G$ to be 3-edgeconnected, so to have minimal degree at least 3 (unless the component is reduced to one vertex). As we saw in Paragraph Bounded number of errors from Section 2.3, this would substantially increase the number of queries. Thus, we choose a different approach, fixing a parameter $r$ that influences the number of queries added compared to the noiseless case, then constructing $Q$ so that $d _ { 2 }$ (and hence $c _ { 2 }$ ) is minimized. We will first describe the structure of $G$ (and hence $Q$ ), then provide an algorithm reaching this structure.

First, we ensure that each pair of positive components of $G$ are linked by at least 3 negative answers. This adds at most $2 { \binom { b } { 2 } }$ queries, but should in practice be small, as in our random models, in the noiseless case, all pairs of positive components are typically already linked by more than 3 negative edges. This ensures that there are no pairs of negative answers from $Q$ that can be switched to create a contradictory cycle. Thus, $d _ { 2 }$ counts the number of pairs of positive edges that can be switched without creating a contradictory cycle. Let $n$ denote the number of vertices of $G$ and its number of positive edges. If the partition characterized by $G$ contains $b$ $m$ $p$ blocks, then the minimal possible value of $m$ is $n - b$ . It corresponds to all positive components being trees and is reached in the noiseless case. Since we want $m$ to not grow too far from this lower bound, we assume $m < 3 n / 2$ . According to [6, Section 4], the structure of $G$ minimizing $d _ { 2 }$ is a graph where all vertices have degree 2 or 3, and there is an integer $s$ such that the maximal paths or cycles containing only vertices of degree 2 all have length $s$ or $s - 1$ . In practice, during the query of human experts, this constraint might be difficult to satisfy exactly. So let us denote by $r$ the average length of those 2-paths, and by $r ^ { \prime }$ an upper-bound. The number of positive edges is then

$$
m = n \left(1 + \frac {1}{3 r + 2}\right),
$$

so the number of edges added compared to the noiseless case is

$$
m - (n - b) = \frac {n}{3 r + 2} + b.
$$

The number of 2-paths is

$$
\frac {3 n}{3 r + 2},
$$

so a bound on $d _ { 2 }$ is

$$
\left( \begin{array}{c} r ^ {\prime} + 1 \\ 2 \end{array} \right) \frac {3 n}{3 r + 2}.
$$

We now describe an algorithm reaching this desirable structure. It inputs a positive parameter $r$ . The largest $r$ is, the smaller the number of additional edges is compared to the noiseless case, but also the higher the probability of undetected error becomes. We start with an AC algorithm designed for the noiseless case (the clique algorithm for example) and maintain the graph $G$ described above. In the first phase, the positive components of $G$ are trees. Whenever the noiseless algorithm proposes a query between two positive components of $G$ , we choose one vertex on each of those components such that a positive answer would create a positive component that is a tree where all vertices have degree at most 3, and all 2-paths have length close to $r$ . The second phase starts when the noiseless algorithm has terminated. Additional queries are added

• between positive components so that each pair is linked by at least 3 negative answers,   
• between the leaves of each tree corresponding to a positive component of $G$ . No vertex should be left with positive degree 1, the 2-paths should have length close to $r$ , and each positive component should be 2-edge-connected.

If at any point a contradictory cycle is detected, queries cutting it in two are submitted until the conflict is resolved.