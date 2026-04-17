# On Provable Benefits of Depth in Training Graph Convolutional Networks

Weilin Cong

Penn State wxc272@psu.edu

Morteza Ramezani

Penn State morteza@cse.psu.edu

Mehrdad Mahdavi

Penn Statemzm616@psu.edu

# Abstract

Graph Convolutional Networks (GCNs) are known to suffer from performance degradation as the number of layers increases, which is usually attributed to oversmoothing. Despite the apparent consensus, we observe that there exists a discrepancy between the theoretical understanding of over-smoothing and the practical capabilities of GCNs. Specifically, we argue that over-smoothing does not necessarily happen in practice, a deeper model is provably expressive, can converge to global optimum with linear convergence rate, and achieve very high training accuracy as long as properly trained. Despite being capable of achieving high training accuracy, empirical results show that the deeper models generalize poorly on the testing stage and existing theoretical understanding of such behavior remains elusive. To achieve better understanding, we carefully analyze the generalization capability of GCNs, and show that the training strategies to achieve high training accuracy significantly deteriorate the generalization capability of GCNs. Motivated by these findings, we propose a decoupled structure for GCNs that detaches weight matrices from feature propagation to preserve the expressive power and ensure good generalization performance. We conduct empirical evaluations on various synthetic and real-world datasets to validate the correctness of our theory.

# 1 Introduction

In recent years, Graph Convolutional Networks (GCNs) have achieved state-of-the-art performance in dealing with graph-structured applications, including social networks [27, 20, 50, 11, 43], traffic prediction [9, 44, 33, 30], knowledge graphs [51, 52, 42], drug reaction [13, 15] and recommendation system [2, 59]. Despite the success of GCNs, applying a shallow GCN model that only uses the information of a very limited neighborhood on a large sparse graph has shown to be not effective [22, 6, 19, 8, 45]. As a result, a deeper GCN model would be desirable to reach and aggregate information from farther neighbors. The inefficiency of shallow GCNs is exacerbated even further when the labeled nodes compared to graph size is negligible, as a shallow GCN cannot sufficiently propagate the label information to the entire graph with only a few available labels [34].

Although a deeper GCN is preferred to perceive more graph structure information, unlike traditional deep neural networks, it has been pointed out that deeper GCNs potentially suffer from oversmoothing [34, 40, 25, 4, 57], vanishing/exploding gradients [32], over-squashing [1], and training difficulties [61, 37], which significantly affect the performance of GCNs as the depth increases. Among these, the most widely accepted reason is “over-smoothing”, which is referred to as a phenomenon due to applying multiple graph convolutions such that all node embeddings converge to a single subspace (or vector) and leads to indistinguishable node representations.

The conventional wisdom is that adding to the number of layers causes over-smoothing, which impairs the expressiveness power of GCNs and consequently leads to a poor training accuracy. However, we observe that there exists a discrepancy between theoretical understanding of their inherent capabilities

![](images/ae31f9dc50d26c68db8913533a145f82c863b78dd76281522286886dee997487.jpg)

![](images/ba5ac88bfa0c135925fd3c973e2c80a87dd29084ac7b187ca5397c5421eef495.jpg)  
Figure 1: Comparison of F1-score for GCN with different depth on Cora dataset, where deeper models can achieve high training accuracy, but complicate the training by requiring more iterations to converge and suffer from poor generalization.

and practical performances. According to the definition of over-smoothing that the node representation becomes indistinguishable as GCNs goes deeper, the classifier has difficulty assigning the correct label for each node if over-smoothing happens. As a result, the training accuracy is expected to be decreasing as the number of layers increases. However, as shown in Figure 1, GCNs are capable of achieving high training accuracy regardless of the number of layers. But, as it can be observed, deeper GCNs require more training iterations to reach a high training accuracy, and its generalization performance on evaluation set decreases as the number of layers increases. This observation suggests that the performance degradation is likely due to inappropriate training rather than the low expressive power caused by over-smoothing. Otherwise, a low expressiveness model cannot achieve almost perfect training accuracy simply by proper training tricks alone.1 Indeed, recent years significant advances have been witnessed on tweaking the model architecture to overcome the training difficulties in deeper GCN models and achieve good generalization performance [37, 6, 32, 61].

Contributions. Motivated by aforementioned observation, i.e., still achieving high training accuracy when trained properly but poor generalization performance, we aim at answering two fundamental questions in this paper:

# Q1: Does increasing depth really impair the expressiveness power of GCNs?

In Section 4, we argue that there exists a discrepancy between over-smoothing based theoretical results and the practical capabilities of deep GCN models, demonstrating that over-smoothing is not the key factor that leads to the performance degradation in deeper GCNs. In particular, we mathematically show that over-smoothing [40, 25, 34, 4] is mainly an artifact of theoretical analysis and simplifications made in analysis. Indeed, by characterizing the representational capacity of GCNs via Weisfeiler-Lehman (WL) graph isomorphism test [38, 54], we show that deeper GCN model is at least as expressive as the shallow GCN model, the deeper GCN models can distinguish nodes with a different neighborhood that the shallow GCN cannot distinguish, as long as the GCNs are properly trained. Besides, we theoretically show that more training iterations is sufficient (but not necessary due to the assumptions made in our theoretical analysis) for a deeper model to achieve the same training error as the shallow ones, which further suggests the poor training error in deep GCN training is most likely due to inappropriate training.

# Q2: If expressive, why then deep GCNs generalize poorly?

In Section 5, in order to understand the performance degradation phenomenon in deep GCNs during the evaluation phase, we give a novel generalization analysis on GCNs and its variants (e.g., ResGCN, APPNP, and GCNII) under the semi-supervised setting for the node classification task. We show that the generalization gap of GCNs is governed by the number of training iterations, largest node degree, the largest singular value of weight matrices, and the number of layers. In particular, our result suggests that a deeper GCN model requires more iterations of training and optimization tricks to converge (e.g., adding skip-connections), which leads to a poor generalization. More interestingly, our generalization analysis shows that most of the so-called methods to solve oversmoothing [46, 60, 6, 28] can greatly improve the generalization ability of the model, therefore results in a deeper model.

The aforementioned findings naturally lead to the algorithmic contribution of this paper. In Section 6, we present a novel framework, Decoupled GCN (DGCN), that is capable of training deeper GCNs and can significantly improve the generalization performance. The main idea is to isolate the expressive power from generalization ability by decoupling the weight parameters from feature propagation. In Section 7, we conduct experiments on the synthetic and real-world datasets to validate the correctness of the theoretical analysis and the advantages of DGCN over baseline methods.

# 2 Related works

Expressivity of GCNs. Existing results on expressive power of GCNs are mixed, [38, 36, 7, 5] argue that deeper model has higher expressive power but [40, 25, 24] have completely opposite result. On the one hand, [38] shows a deeper GCN is as least as powerful as the shallow one in terms of distinguishing non-isomorphic graphs. [36] shows that deep and wide GCNs is Turing universal, however, GCNs lose power when their depth and width are restricted. [7] and [5] measure the expressive power of GCNs via its subgraph counting capability and attribute walks, and both show the expressive power of GCNs grows exponentially with the GCN depth. On the other hand, [24] studies the infinity wide GCNs and shows that the covariance matrix of its outputs converges to a constant matrix at an exponential rate. [40, 25] characterize the expressive power using the distance between node embeddings to a node feature agnostic subspace and show the distance is decreasing as the number of layers increases. Details are deferred to Section 4 and Appendix B. Such contradictory results motivate us to rethink the role of over-smoothing on expressiveness.

Generalization analysis of GCNs. In recent years, many papers are working on the generalization of GCNs using uniform stability [49, 62], Neural Tangent Kernel [14], VC-dimension [48], Rademacher complexity [17, 41], algorithm alignment [55, 56] and PAC-Bayesian [35]. Existing works only focus on a specific GCN structure, which cannot be used to understand the impact of GCN structures on its generalization ability. The most closely related to ours is [49], where they analyze the stability of the single-layer GCN model, and show that the stability of GCN depends on the largest absolute eigenvalue of its Laplacian matrix. However, their result is under the inductive learning setting and extending the results to the multi-layer GCNs with different structures is non-trivial.

Literature with similar observations. Most recently, several works have similar observations on the over-smoothing issue to ours. [61] argues that the main factors to performance degradation are vanishing gradient, training instability, and over-fitting, rather than over-smoothing, and proposes a node embedding normalization heuristic to alleviate the aforementioned issues. [37] argues that the performance degradation is mainly due to the training difficulty, and proposes a different graph Laplacian formulation, weight parameter initialization, and skip-connections to improve the training difficulty. [58] argues that deep GCNs can learn to overcome the over-smoothing issue during training, and the key factor of performance degradation is over-fitting, and proposes a node embedding normalization method to help deep GCNs overcome the over-smoothing issue. [29] improves the generalization ability of GCNs by using adversarial training and results in a consistent improvement than all GCNs baseline models without adversarial training. All aforementioned literature only gives heuristic explanations based on the empirical results, and do not provide theoretical arguments.

# 3 Preliminaries

Notations and setup. We consider the semi-supervised node classification problem, where a selfconnected graph $\mathcal { G } = ( \nu , \mathcal { E } )$ with $N = | \nu |$ nodes is given in which each node $i \in \nu$ is associated with a feature vector $\mathbf { x } _ { i } \in \mathbb { R } ^ { d _ { 0 } }$ , and only a subset of nodes $\mathcal { V } _ { \mathrm { t r a i n } } \subset \mathcal { V }$ are labeled, i.e., $y _ { i } \in \{ 1 , \ldots , | { \mathcal { C } } | \}$ for each $i \in \mathcal { V } _ { \mathrm { t r a i n } }$ and $\mathcal { C }$ is the set of all candidate classes. Let $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ denote the adjacency matrix and $\mathbf { D } \in \mathbb { R } ^ { N \times N }$ denote the corresponding degree matrix with $D _ { i , i } = \deg ( i )$ and $D _ { i , j } = 0$ if $i \neq j$ . Then, the propagation matrix (using the Laplacian matrix defined in [27]) is computed as $\mathbf { P } = \mathbf { D } ^ { - 1 / 2 } \mathbf { A } \mathbf { D } ^ { - 1 / 2 }$ . Our goal is to learn a GCN model using the node features for all nodes $\{ { \bf x } _ { i } \} _ { i \in \mathcal { V } }$ and node labels for the training set nodes $\{ y _ { i } \} _ { i \in \mathcal { V } _ { \operatorname { t r a i n } } }$ , and expect it generalizes well on the unlabeled node set $\mathcal { V } _ { \mathrm { t e s t } } = \mathcal { V } \setminus \mathcal { V } _ { \mathrm { t r a i n } }$ .

GCN architectures. In this paper, we consider the following architectures for training GCNs:

![](images/67906d1492ff84d771c91e17a1dbd9754f4f19640877645f30bcff3cec14a59b.jpg)

![](images/7c13163b70873f5e56c00ac78ee44c80abdd890b861103cc0e1a47b86acd8caa.jpg)  
Figure 2: Comparison of intra- and inter-class normalized node embeddings $\mathbf { H } ^ { ( \ell ) } / \| \mathbf { H } ^ { ( \ell ) } \| _ { \mathrm { F } }$ pairwise distance on Cora dataset. See Appendix A for more evidences.

• Vanilla GCN [27] computes node embeddings by $\mathbf { H } ^ { ( \ell ) } = \sigma ( \mathbf { P H } ^ { ( \ell - 1 ) } \mathbf { W } ^ { ( \ell ) } )$ , where $\mathbf { H } ^ { ( \ell ) } =$ $\{ \mathbf { h } _ { i } ^ { ( \ell ) } \} _ { i = 1 } ^ { N }$ th layer node embedding matrix, $\mathbf { h } _ { i } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell } }$ is the embedding of ith node, $\mathbf { W } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell } \times d _ { \ell - 1 } }$ is the `th layer weight matrix, and $\sigma ( \cdot )$ is the ReLU activation.   
• ResGCN [32] solves the vanishing gradient issue by adding skip-connections between adjacency layers. More specifically, ResGCN computes node embeddings by $\mathbf { H } ^ { ( \ell ) } = \sigma ( \mathbf { P H } ^ { ( \ell - 1 ) } \mathbf { \dot { W } } ^ { ( \ell ) } ) \dot { + }$ $\mathbf { H } ^ { ( \ell - 1 ) }$ No fully connected layer after node feature, where node embeddings of the previous layer is added to the output of the current layer to facilitate the training of deeper GCN models.   
• APPNP [28] adds skip-connections from the input layer to each hidden layer to preserve the feature information. APPNP computes node embeddings by $\mathbf { H } ^ { ( \ell ) } = \alpha _ { \ell } \mathbf { P } \mathbf { H } ^ { ( \ell - 1 ) } + ( \mathbf { \bar { l } } - \alpha _ { \ell } ) \mathbf { H } ^ { ( 0 ) } \mathbf { W }$ , where $\alpha _ { \ell } \in [ 0 , 1 ]$ balances the amount of information preserved at each layer. By decoupling feature transformation and propagation, APPNP can aggregate information from multi-hop neighbors without significantly increasing the computation complexity.   
• GCNII [6] improves the capacity of APPNP by adding non-linearty and weight matrix at each individual layer. GCNII computes node embeddings by $\mathbf { H } ^ { ( \ell ) } = \bar { \sigma } \big ( ( \alpha _ { \ell } \mathbf { P } \bar { \mathbf { H } ^ { ( \ell - 1 ) } } + ( 1 -$ $\alpha _ { \ell } ) \mathbf { H } ^ { ( 0 ) } ) \bar { \mathbf { W } } ^ { ( \ell ) } )$ , where $\bar { \mathbf { W } } ^ { ( \ell ) } = \beta _ { \ell } \mathbf { W } ^ { ( \ell ) } + ( 1 - \beta _ { \ell } ) \mathbf { I }$ , constant $\alpha _ { \ell }$ same as APPNP, and constant $\beta _ { \ell } \in [ 0 , 1 ]$ restricts the power of `th layer parameters.

# 4 On true expressiveness and optimization landscape of deep GCNs

Empirical validation of over-smoothing. In this section, we aim at answering the following fundamental question: “Does over-smoothing really cause the performance degradation in deeper GCNs?” As first defined in [34], over-smoothing is referred to as a phenomenon where all node embeddings converge to a single vector after applying multiple graph convolution operations to the node features. However, [34] only considers the graph convolution operation without non-linearity and the per-layer weight matrices. To verify whether over-smoothing exists in normal GCNs, we measure the pairwise distance between the normalized node embeddings with varying the model depth.2 As shown in Figure 2, without the weight matrices and non-linear activation functions, the pairwise distance between node embeddings indeed decreases as the number of layers increases. However, by considering the weight matrices and non-linearity, the pairwise distances are actually increasing after a certain depth which contradicts the definition of over-smoothing that node embeddings become indistinguishable when the model becomes deeper. That is, graph convolution makes adjacent node embeddings get closer, then non-linearity and weight matrices help node embeddings preserve distinguishing-ability after convolution.

Most recently, [40, 25] generalize the idea of over-smoothing by taking both the non-linearity and weight matrices into considerations. More specifically, the expressive power of the `th layer node embeddings $\mathbf { H } ^ { ( \ell ) }$ is measured using $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ , which is defined as the distance of node embeddings to a subspace $\mathcal { M }$ that only has node degree information. Let denote $\lambda _ { L }$ as the second largest eigenvalue of graph Laplacian and $\lambda _ { W }$ as the largest singular value of weight matrices. [40, 25] show that the expressive power $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ is bounded by $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } ) \leq ( \lambda _ { W } \lambda _ { L } ) ^ { \ell } \cdot d _ { \mathcal { M } } ( \mathbf { X } )$ , i.e., the expressive power of node embeddings will be exponentially decreasing or increasing as the number

![](images/d4c3e5f0c55e525d8432261ce0ea703d94e1a9ca933072974bfafe0a6b4411da.jpg)

![](images/ce61c9d64cddeef4adeff6fc9af3ac24fb1dd00e8aa47cd065d7a6137230e50d.jpg)  
Figure 3: Compare $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ and $\lambda _ { W } ( \mathbf { W } ^ { ( \ell ) } )$ on both trained- (using 500 gradient update) and untrained-GCN models on Cora dataset.

of layers increases, depending on whether $\lambda _ { W } \lambda _ { L } < 1$ or $\lambda _ { W } \lambda _ { L } > 1$ . They conclude that deeper GCN exponentially loss expressive power by assuming $\lambda _ { W } \lambda _ { L } < 1$ . However, we argue that this assumption does not always hold. To see this, let suppose $\mathbf { W } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell - 1 } \times d _ { \ell } }$ is initialized by uniform distribution $\mathcal { N } ( 0 , \sqrt { 1 / d _ { \ell - 1 } } )$ . By the Gordon’s theorem for Gaussian matrices [10], we know its expected largest singular value is bounded by $\mathbb { E } [ \lambda _ { W } ] \le 1 + \sqrt { d _ { \ell } / d _ { \ell - 1 } }$ , which is strictly greater than 1 and $\lambda _ { W }$ usually increases during training. The above discussion also holds for other commonly used initialization methods [18, 21]. Furthermore, since most real world graphs are sparse with $\lambda _ { L }$ close to 1, e.g., Cora has $\lambda _ { L } = 0 . 9 9 6 4$ , Citeseer has $\lambda _ { L } = 0 . 9 9 8 7$ , and Pubmed has $\lambda _ { L } = 0 . 9 9 0 5$ , making assumption on $\lambda _ { W } \lambda _ { L } < 1$ is not realistic. As shown in Figure 3, when increasing the number of layers, we observe that the distance $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ is decreasing on untrained-GCN models, however, the distance $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ is increasing on trained-GCN models, which contradicts the conclusion in [40]. Due to the space limit, we defer the more empirical evidence to Appendix B. Through these findings, we cast doubt on the power of over-smoothing based analysis to provide a complete picture of why deep GCNs perform badly.

Definition 1. Let $\mathcal { T } _ { i } ^ { L }$ denote the $L$ -layer computation tree of node $i$ , which represents the structured $L$ -hop neighbors of node i, where the children of any node $j$ in the tree are the nodes in $\mathcal { N } ( i )$ .

Exponentially growing expressieness without strong assumptions. Indeed, we argue that deeper GCNs have stronger expressive power than the shallow GCNs. To prove this, we employ the connection between WL test3 [31] and GCNs. Recently, [38] shows that GCNs have the same expressiveness as the WL-test for graph isomorphism if they are appropriately trained, i.e., a properly trained $L$ -layer GCN computes different node representations for two nodes if their $L$ -layer computation tree (Definition 1) have different structure or different features on the corresponding nodes. Since $L$ -GCN can encode any different computation tree into different representations, it is natural to characterize the expressiveness of $L$ -GCN by the number of computation graph it can encode.

Theorem 1. Suppose $\mathcal { T } ^ { L }$ is a computation tree with binary node features and node degree at least d. Then, by assuming the computation tree of two nodes are disjoint, the richness (i.e., the number of computation graphs a model can encode) of the output of L-GCN defined on $\mathcal { T } ^ { L }$ is at least $| \bar { L } \bar { - } G C \bar { N } ( \bar { \cal T } ^ { L } ) | \geq 2 ( \bar { d } - 1 ) ^ { L - 1 }$ .

The proof is deferred to Appendix C. The above theorem implies that the richness of $L$ -GCN grows at least exponentially with respect to the number of layers.

Comparison of expressiveness metrics. Although distance-based expressiveness metric [40, 25] is strong than WL-based metric in the sense that node embeddings can be distinct but close to each other, the distance-based metric requires explicit assumptions on the GCN structures, weight matrices, and graph structures comparing to the WL-based metric, which has been shown that are not likely hold. On the other hand, WL-based metric has been widely used in characterizing the expressive power of GCNs in graph-level task [38, 36, 7, 5]. More details are deferred to related works (Section 2).

Although expressive, it is still unclear why the deeper GCN requires more training iterations to achieve small training error and reach the properly trained status. To understand this, we show in Theorem 2 that under assumptions on the width of the final layer, the deeper GCN can converge to its global optimal with linear convergence rate. More specifically, the theorem claims that if the dimension of the last layer of GCN $d _ { L }$ is larger than the number of data $N ^ { 4 }$ , then we can guarantee the loss ${ \mathcal { L } } ( \theta _ { T } ) \leq \varepsilon$ after $T = \mathcal { O } ( 1 / \varepsilon )$ iterations of the gradient updates. Besides, more training iterations is sufficient (but not necessary due to the assumptions made in our theoretical analysis) for a deeper model to achieve the same training error as the shallow ones.

Theorem 2. Let $\pmb { \theta } _ { t } = \{ \mathbf { W } _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell - 1 } \times d _ { \ell } } \} _ { \ell = 1 } ^ { L + 1 }$ be the model parameter at the t-th iteration and using square loss $\begin{array} { r } { \mathcal { L } ( \pmb { \theta } ) = \frac { 1 } { 2 } \| \mathbf { H } ^ { ( L ) } \mathbf { W } ^ { ( L + 1 ) } - \mathbf { Y } \| _ { \mathrm { F } } ^ { 2 } } \end{array}$ , $\mathbf { H } ^ { ( \ell ) } = \sigma ( \mathbf { P H } ^ { ( \ell - 1 ) } \mathbf { W } ^ { ( \ell ) } )$ as objective function. Then, under the condition that $d _ { L } \geq N$ we can obtain $\mathcal { L } ( \pmb { \theta } _ { T } ) \leq \epsilon i f T \geq C ( L ) \log ( \mathcal { L } ( \pmb { \theta } _ { 0 } ) / \epsilon )$ , where  is the desired error and $C ( L )$ is a function of GCN depth $L$ that grows as GCN becomes deeper.

A formal statement of Theorem 2 and its proof are deferred to Appendix D. Besides, gradient stability also provides an alternative way of empirically understanding why deeper GCN requires more iterations: deeper neural networks are prone to exploding/vanishing gradient, which results in a very noisy gradient and requires small learning rate to stabilize the training. This issue can be significantly alleviated by adding skip-connections (Appendix E.5). When training with adaptive learning rate mechanisms, such as Adam $[ 2 6 ] ^ { 5 }$ , noisy gradient will result in a much smaller update on current model compared to a stabilized gradient, therefore more training iterations are required.

# 5 A different view from generalization

In the previous section, we provided evidence that a well-trained deep GCN is at least as powerful as a shallow one. However, it is still unclear why a deeper GCN has worse performance than a shallow GCN during the evaluation phase. To answer this question, we provide a different view by analyzing the impact of GCN structures on the generalization.

Transductive uniform stability. In the following, we study the generalization ability of GCNs via transductive uniform stability [16], where the generalization gap is defined as the difference between the training and testing errors for the random partition of a full dataset into training and testing sets. Transductive uniform stability is defined under the notation that the output of a classifier does not change much if the input is perturbed a bit, which is an extension of uniform stability [3] from the inductive to the transductive setting. The previous analysis on the uniform stability of GCNs [49] only shows the result of GCN with one graph convolutional layer under inductive learning setting, which cannot explain the effect of depth, model structure, and training data size on the generalization, and its extension to multi-layer GCNs and other GCN structures are non-trivial.

Problem setup. Let $m = | \mathcal { V } _ { \mathrm { t r a i n } } |$ and $u = | \nu _ { \mathrm { t e s t } } |$ denote the training and test dataset sizes, respectively. Under the transductive learning setting, we start with a fixed set of points $X _ { m + u } = \{ x _ { 1 } , \ldots , x _ { m + u } \}$ For notational convenience, we assume $X _ { m }$ are the first $m$ data points and $X _ { u }$ are the last $u$ data points of $X _ { m + u }$ . We randomly select a subset $X _ { m } ~ \subset ~ X _ { m + u }$ uniformly at random and reveal the labels $Y _ { m }$ for the selected subset for training, but the labels for the remaining $u$ data points $Y _ { u } = Y _ { m + u } \setminus Y _ { m }$ are not available during the training phase. Let $S _ { m } = ( ( x _ { 1 } , y _ { 1 } ) , \bar { \bf \Phi } \cdot \cdot \cdot , ( x _ { m } , \bar { y } _ { m } ) )$ denotes the labeled set and $X _ { u } = ( x _ { m + 1 } , \dots , x _ { m + u } )$ denotes the unlabeled set. Our goal is to learn a model to label the remaining unlabeled set as accurately as possible.

For the analysis purpose, we assume a binary classifier is applied to the final layer node representation $f ( \mathbf { h } _ { i } ^ { ( L ) } ) = \tilde { \sigma } ( \mathbf { v } ^ { \top } \mathbf { h } _ { i } ^ { ( L ) } )$ with $\tilde { \sigma } ( \cdot )$ denotes the sigmoid function. We predict $\hat { y } _ { i } = 1$ if $f ( \mathbf { h } _ { i } ^ { ( L ) } ) > 1 / 2$ and $\hat { y } _ { i } = 0$ otherwise, with ground truth label $y _ { i } \in \{ 0 , 1 \}$ . Let denote the perturbed dataset as $S _ { m } ^ { i j } \triangleq ( S _ { m } \setminus \{ ( x _ { i } , y _ { i } ) \} ) \cup \{ ( x _ { j } , y _ { j } ) \}$ and $X _ { u } ^ { i j } \triangleq ( X _ { u } \setminus \{ x _ { j } \} ) \cup \{ x _ { i } \}$ , which is obtained by replacing the ith example in training set $S _ { m }$ with the $j$ th example from the testing set $X _ { u }$ . Let $\pmb { \theta }$ and $\pmb { \theta } ^ { i j }$ denote the weight parameters trained on the original dataset $( S _ { m } , X _ { u } )$ and the perturbed dataset $( S _ { m } ^ { i j } , X _ { u } ^ { i j } )$

respectively. Then, we say transductive learner $f$ is $\epsilon$ -uniformly stable if the outputs change less than $\epsilon$ when we exchange two examples from the training set and testing set, i.e., for any $S _ { m } \subset S _ { m + u }$ and any $i , j \in [ m + u ]$ it holds that $\begin{array} { r } { \operatorname* { s u p } _ { i \in [ m + u ] } | f ( \mathbf { h } _ { i } ^ { ( L ) } ) - \tilde { f } ( \tilde { \mathbf { h } } _ { i } ^ { ( L ) } ) ) | \leq \epsilon } \end{array}$ , where $f ( \mathbf { h } _ { i } ^ { ( L ) } )$ and $\tilde { f } ( \tilde { \mathbf { h } } _ { i } ^ { ( L ) } )$ denote the prediction of node $i$ using parameters $\pmb { \theta }$ and $\pmb { \theta } ^ { i j }$ respectively.

To define testing error and training error in our setting, let us introduce the difference in probability between the correct and the incorrect label as $p ( z , y ) \triangleq y ( 2 z - 1 ) + ( 1 - y ) ( 1 - 2 z )$ with $p ( z , y ) \leq 0$ if there exists a classification error. Let denote the $\gamma$ -margin loss as $\Phi _ { \gamma } ( x ) = \operatorname* { m i n } ( 1 , \operatorname* { m a x } ( 0 , 1 - x / \gamma ) )$ . Then, the testing error is defined as $\begin{array} { r } { \mathcal { R } _ { u } ( f ) = \frac { 1 } { u } \sum _ { i = m + 1 } ^ { m + u } \mathbf { 1 } \{ p ( f ( \mathbf { h } _ { i } ^ { ( L ) } ) , y _ { i } ) \leq 0 \} } \end{array}$ and the training loss is defined as $\begin{array} { r } { \mathcal { R } _ { m } ^ { \gamma } ( f ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \Phi _ { \gamma } ( - p ( f ( \mathbf { h } _ { i } ^ { ( L ) } ) , y _ { i } ) ) } \end{array}$ . 6

Theorem 3 (Transductive uniform stability bound [16]). Let $f$ be a -uniformly stable transductive learner and $\gamma , \delta > 0$ , and define $Q = m u / ( m + u )$ . Then, with probability at least $1 - \delta$ over all training and testing partitions, we have $\begin{array} { r } { \mathcal { R } _ { u } ( f ) \leq \mathcal { R } _ { m } ^ { \gamma } ( f ) + \frac { 2 } { \gamma } \mathcal { O } \left( \epsilon \sqrt { Q \ln ( \delta ^ { - 1 } ) } \right) + \mathcal { O } \left( \frac { \ln ( \delta ^ { - 1 } ) } { \sqrt { Q } } \right) } \end{array}$ .

Recall that as we discussed in Section 4, deeper GCN is provable more expressive and can achieve very low training error $\mathcal { R } _ { m } ^ { \gamma } ( f )$ if properly training. Then, if the dataset size is sufficiently large, the testing error $\mathcal { R } _ { u } ( f )$ will be dominated by the generalization gap, which is mainly controlled by uniformly stable constant . In the following, we explore the impact of GCN structures on . Our key idea is to decompose the $\epsilon$ into three terms: the Lipschitz continuous constant $\rho _ { f }$ , upper bound on gradient $G _ { f }$ , and the smoothness constant $L _ { f }$ of GCNs. Please refer to Lemma 1 for details.

Lemma 1. Suppose function $f ( \mathbf { h } ^ { ( L ) } )$ is $\rho _ { f }$ -Lipschitz continuous, $L _ { f }$ -smooth, and the gradient of loss w.r.t. the parameter is bounded by $G _ { f }$ . After $T$ steps of full-batch gradient descent, we have $\begin{array} { r } { \epsilon = \frac { 2 \eta \rho _ { f } G _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + \eta L _ { f } ) ^ { t - 1 } } \end{array}$ PTt=1(1 + ηLf )t−1.

The proof is deferred to Appendix K. By using Lemma 1, we can derive constant $\epsilon$ of different model structures by comparing the Lipschitz continuity, smoothness, and gradient scale.

Before proceeding to our result, we make the following standard assumption on the node feature vectors and weight matrices, which are previously used in generalization analysis of GCNs [17, 35].

Assumption 1. We assume the norm of node feature vectors, weight parameters are bounded, i.e., $\| \mathbf { x } _ { i } \| _ { 2 } \leq B _ { x }$ , $\| \mathbf { W } ^ { ( \ell ) } \| _ { 2 } \le B _ { w }$ , and $\| \mathbf { v } \| _ { 2 } \leq 1$ .

In Theorem 4, we show that the generalization bounds of GCN and its variants are dominated by the following terms: maximum node degree $d$ , model depth $L$ , training/validation set size $( m , u )$ , training iterations $T$ , and spectral norm of the weight matrices $B _ { w }$ . The larger the aforementioned variables are, the larger the generalization gap is. We defer the formal statements and proofs to Appendices F, G, H, and I.

Theorem 4 (Inforwhere the result of ay model is -uniformly stable with are summarized in Table 1, and oth $\begin{array} { r } { \epsilon = \frac { 2 \eta \rho _ { f } G _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + \eta L _ { f } ) ^ { t - 1 } } \end{array}$ $\rho _ { f } , G _ { f } , L _ { f }$

$$
\begin{array}{l} B _ {d} ^ {\alpha} = (1 - \alpha) \sum_ {\ell = 1} ^ {L} (\alpha \sqrt {d}) ^ {\ell - 1} + (\alpha \sqrt {d}) ^ {L}, B _ {w} ^ {\beta} = \beta B _ {w} + (1 - \beta), \\ B _ {\ell , d} ^ {\alpha , \beta} = \max  \left\{\beta \left((1 - \alpha) L + \alpha \sqrt {d}\right), (1 - \alpha) L B _ {w} ^ {\beta} + 1 \right\}. \tag {1} \\ \end{array}
$$

In the following, we provide intuitions and discussions on the generalization bound of each algorithm:

• Deep GCN requires iterations $T$ to achieve small training error. Since the generalization bound increases with $T$ , more iterations significantly hurt its generalization power. Note that our results considers both $B _ { w } \le 1$ and $B _ { w } > 1$ , where increasing model depth will not hurt the generalization if $B _ { w } \le 1$ , and the generalization gap becomes sensitive to the model depth if $B _ { w } > 1$ . Notice that $B _ { w } > 1$ is more likely to happen during training as we discussed in Section 4.   
• ResGCN resolves the training difficulties by adding skip-connections between hidden layers. Although it requires less training iterations $T$ , adding skip-connections enlarges the dependency

Table 1: Comparison of uniform stability constant $\epsilon$ of GCN variants, where $\mathcal { O } ( \cdot )$ is used to hide constants that shared between all bounds.   

<table><tr><td></td><td>ρf and Gf</td><td>Lf</td><td>C1 and C2</td></tr><tr><td>εGCN</td><td>O(C1LC2)</td><td>O(C1LC2((L+2)C1LC2+2))</td><td>C1 = max{1,√dBw}, C2 = √d(1+Bx)</td></tr><tr><td>εResGCN</td><td>O(C1LC2)</td><td>O(C1LC2((L+2)C1LC2+2))</td><td>C1 = 1 + √dBw, C2 = √d(1+Bx)</td></tr><tr><td>εAPPNP</td><td>O(C1)</td><td>O(C1(C1C2)+1)</td><td>C1 = BdaBx, C2 = max{1,Bw}</td></tr><tr><td>εGCNII</td><td>O(βC1LC2)</td><td>O(αβC1LC2((αβL+2)C1LC2+2β))</td><td>C1 = max{1,α√dBw}, C2 = √d + Bα,βbBx</td></tr><tr><td>εDGCN</td><td>O(C1)</td><td>O(C1(C1C2)+1)</td><td>C1 = (√d)LBx, C2 = max{1,Bw}</td></tr></table>

on the number of layers $L$ and the spectral norm of weight matrices $B _ { w }$ , therefore results in a larger generalization gap and a poor generalization performance..

• APPNP alleviates the aforementioned dependency by decoupling the weight parameters and feature propagation. As a result, its generalization gap does not significantly change as $L$ and $B _ { w }$ increase. The optimal $\alpha$ that minimizes the generalization gap can be obtained by finding the $\alpha$ that minimize the term $B _ { d } ^ { \alpha }$ . Although APPNP can significantly reduce the generalization gap, because a single weight matrix is shared between all layers, its expressive power is not enough for large-scale challenging graph datasets [23].   
• To gain expressiveness, GCNII proposes to add the weight matrices back and add another hyper-parameter that explicitly controls the dependency on $B _ { w }$ . Although GCNII achieves the state-of-the-art performances on several graph datasets, the selection of hyper-parameters is non-trivial compared to APPNP because $\alpha , \beta$ are coupled with $L , B _ { w }$ , and $d$ . In practice, [6] builds a very deep GCNII by choosing $\beta$ dynamically decreases as the number of layers and different $\alpha$ values for different datasets.   
• By property chosen hyper-parameters, we have the following order on the generalization gap given the same training iteration T $: { \mathrm { A P P N P } } \leq { \mathrm { G C N I I } } \leq { \mathrm { G C N } } \leq { \mathrm { R e s G C N } }$ , which exactly match our empirical evaluation on the generalization gap in Section 7 and Appendix E.

Remark 1. It is worthy to provide an alternative view of DropEdge [46] and PairNorm [60] algorithms from a generalization perspective. To improve the generalization power of standard GCNs, DropEdge randomly drops edges in the training phase, which leads to a smaller maximum node degree $d _ { s } < d$ . PairNorm applies normalization on intermediate node embeddings to ensure that the total pairwise feature distances remain constant across layers, which leads to less dependency on $d$ and $B _ { w }$ . However, since deep GCN requires significantly more iterations to achieve low training error than shallow one, the performance of applying DropEdge and PairNorm on GCNs is still degrading as the number of layers increases. Most importantly, our empirical results in Appendix E.3 and E.4 suggest that applying Dropout and PairNorm is hurting the training accuracy (i.e., not alleviating over-smoothing) but reducing the generalization gap.

# 6 Decoupled GCN

We propose to decouple the expressive power from generalization ability with Decoupled GCN (DGCN). The DGCN model can be mathematically formulated as $\begin{array} { r } { \mathbf { Z } \ = \ \sum _ { \ell = 1 } ^ { L } \alpha _ { \ell } f ^ { ( \ell ) } ( \mathbf { X } ) } \end{array}$ and $f ^ { ( \ell ) } ( \mathbf { X } ) = \mathbf { P } ^ { \ell } \mathbf { X } \big ( \beta _ { \ell } \mathbf { W } ^ { ( \ell ) } + ( 1 - \beta _ { \ell } ) \mathbf { I } \big )$ , where $\mathbf { W } ^ { ( \ell ) }$ , $\alpha _ { \ell }$ and $\beta _ { \ell }$ are the learnable weights for `th layer function $f ^ { ( \ell ) } ( { \mathbf { X } } )$ . The design of DGCN has the following key ingredients:

• (Decoupling) The generalization gap in GCN grows exponentially with the number of layers. To overcome this issue, we propose to decouple the weight matrices from propagation by assigning weight $\mathbf { W } ^ { ( \ell ) }$ to each individual layerwise function $f ^ { ( \ell ) } ( { \mathbf { X } } )$ . DGCN can be thought of as an ensemble of multiple SGCs [53] with depth from 1 to $L$ . By doing so, the generalization gap has less dependency on the number of weight matrices, and deep models with large receptive fields can incorporate information of the global graph structure. Please refer to Theorem 5 for the details.   
• (Learnable $\alpha _ { \ell }$ ) After decoupling the weight matrices from feature propagation, the layerwise function $f ^ { ( \ell ) } ( { \mathbf { X } } )$ with more propagation steps can suffer from less expressive power. Therefore, we propose to assign a learnable weight $\alpha _ { \ell }$ for each step of feature propagation. Intuitively, DGCN assigns smaller weight $\alpha _ { \ell }$ to each layerwise function $f ^ { ( \ell ) } ( { \mathbf { X } } )$ with more propagation

![](images/1a616d95d74dc17b50407e316dba2169bbb99e15b0f7bdfd9f2bb18d9fd25025.jpg)

![](images/ed014dd80b737f22bacceb078b05da74213e285a76a7dfb8163fbf9db51e93d1.jpg)

![](images/05798755db9ee5c35536857c1ecadf302f167918be8ca65e5822c222a04a9602.jpg)

![](images/b422718bc9585727353d646a0654a0922fbcb68354e4bf89d0ceb80dc273cb90.jpg)

![](images/ba156256c916c3dc64c9616d34c7d134edb6000ebf33bb5ff7ef1ad2faf216e4.jpg)

![](images/c79d9bfe7b833fde1b6d2a09ced42605692ca6e60c534b67be4ebf10d6745920.jpg)  
Figure 4: Comparison of generalization error on synthetic dataset. The curve early stopped at the largest training accuracy iteration.

steps at the beginning of training. Throughout the training, DGCN gradually adjusts the weight to leverage more useful large receptive field information.

• (Learnable $\beta _ { \ell }$ ) A learnable weight $\beta _ { \ell } \in [ 0 , 1 ]$ is assigned to each weight matrix to balance the expressiveness with model complexity, which guarantees a better generalization ability.

Theorem 5. Let suppose DGCN-uniformly stable wit $\alpha _ { \ell }$ $\beta _ { \ell }$ during training. We say DGCN iswhere $\begin{array} { r } { \epsilon _ { D G C N } = \frac { 2 \eta \rho _ { f } \bar { G } _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + \eta L _ { f } ) ^ { t - 1 } } \end{array}$

$$
\rho_ {f} = G _ {f} = \mathcal {O} \left((\sqrt {d}) ^ {L} B _ {x}\right), L _ {f} = \mathcal {O} \left((\sqrt {d}) ^ {L} B _ {x} \left((\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \} + 1)\right)\right). \tag {2}
$$

The details are deferred to Appendix J, and comparison of bound to other GCN variants are summarized in Table 1. Depending on the automatic selection of $\alpha _ { \ell } , \beta _ { \ell }$ , the generalization bound of DGCN is between APPNP and GCN. In the following, we make connection to many GCN structures:

Connections to APPNP: APPNP can be thought of as a variant of DGCN. More specifically, the layerwise weight in APPNP is computed as $\alpha _ { \ell } = \alpha ( 1 - \alpha ) ^ { \ell }$ for $\ell < L$ and $\alpha _ { \ell } \stackrel { \cdot } { = } ( 1 - \alpha ) ^ { \cdot }$ for $\ell = L$ given some constant $\alpha \in ( 0 , 1 )$ , and the weight matrix is shared between all layers. Although DGCN has $L$ weight matrices, its generalization is independent of the number of weight matrices, and thus enjoys a low generalization error with high expressiveness.   
• Connections to GCNII: GCNII can be regarded as a variant of DGCN. Compared to GCNII, the decoupled propagation of DGCN significantly reduces the dependency of generalization error to the weight matrices. Besides, the learnable weights $\alpha _ { \ell }$ and $\beta _ { \ell }$ allow DGCN to automatically adapt to challenging large-scale datasets without time-consuming hyper-parameter selection.   
• Connections to ResGCN: By expanding the forward computation of ResGCN, we know trainwith nsemble of GCNs from 1 to . In other word, ResNet can b $L$ layer, i.e.,egarded asL $\begin{array} { r } { \mathbf { H } ^ { ( L ) } = \sum _ { \ell = 1 } ^ { L } \alpha _ { \ell } \sigma \big ( \mathbf { P H } ^ { ( \ell - 1 ) } \mathbf { W } ^ { ( \ell ) } \big ) } \end{array}$ $\alpha _ { \ell } = 1$ the “summation of the model complexity” of $L$ -layer. However, DGCN is using $\begin{array} { r } { \sum _ { \ell = 1 } ^ { L ^ { - } } \alpha _ { \ell } = 1 } \end{array}$ , which can be thought of as a “weighted average of model complexity”. Therefore, ResGCN is a special case of DGCN with equal weights $\alpha _ { \ell }$ on each layerwise function. With just a simple change on the ResNet structure, our model DGCN is both easy to train and good to generalize.

# 7 Experiments

Synthetic dataset. We empirically compare the generalization error of different GCN structures on the synthetic dataset. In particular, we create the synthetic dataset by contextual stochastic block model (CSBM) [12] with two equal-size classes. CSBM is a graph generation algorithm that adds Gaussian random vectors as node features on top of classical SBM. CSBM allows for smooth control over the information ratio between node features and graph topology by a pair of hyper-parameter $( \mu , \lambda )$ , where $\mu$ controls the diversity of the Gaussian distribution and $\lambda$ controls the number of edges between intra- and inter-class nodes. We generate random graphs with 1000 nodes, average node

degree as 5, and each node has a Gaussian random vector of dimension 1000 as node features. We chose $7 5 \%$ nodes as training set, $1 5 \%$ of nodes as validation set for hyper-parameter tuning, and the remaining nodes as testing set. We conduct an experiment 20 times by randomly selecting $( \mu , \lambda )$ such that both node feature and graph topology are equally informative.

As shown in Figure 4, we have the following order on the generalization gap given the same training iteration $T$ : $\mathrm { A P P N P } \leq \mathrm { G C N I I } \leq \mathrm { G C N } \leq \mathrm { R e s G C N }$ , which exactly match the theoretical result in Theorem 4. More specifically, ResGCN has the largest generalization gap due to the skip-connections, APPNP has the smallest generalization gap by removing the weight matrices in each individual layer. GCNII achieves a good balance between GCN and APPNP by balancing the expressive and generalization power. Finally, DGCN enjoys a small generalization error by using the decoupled GCN structure.

Open graph benchmark dataset. As pointed out by Hu et al. [23], the traditional commonly-used graph datasets are unable to provide a reliable evaluation due to various factors including dataset size, leakage of node features and no consensus on data splitting. To truly evaluate the expressive and the generalization power of existing methods, we evaluate on the open graph benchmark (OGB) dataset. Experiment setups are based on the default setting for GCN implementation on the leaderboard. We choose the hidden dimension as 128, learning rate as 0.01, dropout ratio as 0.5 for Arxiv dataset, and no dropout for Products and Protein datasets. We train 300/1000/500 epochs for Products, Proteins, and Arxiv dataset respectively. Due to limited GPU memory, the number of layers is selected as the one with the best performance between 2 to 16 layers for Arxiv dataset, 2 to 8 layers for Protein dataset, and 2 to 4 for Products dataset. We choose $\alpha _ { \ell }$ from $\{ 0 . 9 , 0 . 8 , 0 . 5 \}$ for APPNP and GCNII, and use $\beta _ { \ell } = 0 . 5 / \ell$ for GCNII, and select the setup with the best validation result for comparison.

As shown in Table 2, DGCN achieves a compatible performance to $\mathrm { G C N I I } ^ { 7 }$ without the need of manually tuning the hyper-parameters for all settings, and it significantly outperform APPNP and ResGCN. Due to the space limit, the detailed setups and more results can be found in Appendix E. Notice that generalization bounds are more valuable when comparing two models with same training accuracy (therefore we first show in Section 4 that deeper model can also achieve low training error before our discussion on generalization in Section 5).

In Table 2, because ResGCN has no restriction on the weight matrices, it can achieve lower training error and its test performance is mainly restricted by its generalization error. However, because GCNII and APPNP have restrictions on the weight matrices, their performance is mainly restricted by their training error. A model with small generalization error and no restriction on the weight (e.g., DGCN)

Table 2: Comparison of F1-score on OGB dataset.   

<table><tr><td>%</td><td>Products</td><td>Proteins</td><td>Arvix</td></tr><tr><td>GCN</td><td>75.39 ± 0.21</td><td>71.66 ± 0.48</td><td>71.56 ± 0.19</td></tr><tr><td>ResGCN</td><td>75.53 ± 0.12</td><td>74.50 ± 0.41</td><td>72.56 ± 0.31</td></tr><tr><td>APPNP</td><td>66.35 ± 0.10</td><td>71.78 ± 0.29</td><td>68.02 ± 0.55</td></tr><tr><td>GCNII</td><td>71.93 ± 0.35†</td><td>75.60 ± 0.47</td><td>72.57 ± 0.23‡</td></tr><tr><td>DGCN</td><td>76.09 ± 0.29</td><td>75.45 ± 0.24</td><td>72.63 ± 0.12</td></tr></table>

is preferred as it has higher potential to reach a better test accuracy by reducing its training error.

# 8 Conclusion

In this work, we show that there exists a discrepancy between over-smoothing based theoretical results and the practical behavior of deep GCNs. Our theoretical result shows that a deeper GCN can be as expressive as a shallow GCN, if it is properly trained. To truly understand the performance decay issue of deep GCNs, we provide the first transductive uniform stability-based generalization analysis of GCNs and other GCN structures. To improve the optimization issue and benefit from depth, we propose DGCN that enjoys a provable high expressive power and generalization power. We conduct empirical evaluations on various synthetic and real-world datasets to validate the correctness of our theory and advantages over the baselines.

# Acknowledgements

This work was supported in part by NSF grant 2008398.

# References

[1] Uri Alon and Eran Yahav. On the bottleneck of graph neural networks and its practical implications. In International Conference on Learning Representations, 2020.   
[2] Rianne van den Berg, Thomas N Kipf, and Max Welling. Graph convolutional matrix completion. arXiv preprint arXiv:1706.02263, 2017.   
[3] Olivier Bousquet and André Elisseeff. Stability and generalization. Journal of machine learning research, 2(Mar):499–526, 2002.   
[4] Chen Cai and Yusu Wang. A note on over-smoothing for graph neural networks. arXiv preprint arXiv:2006.13318, 2020.   
[5] Lei Chen, Zhengdao Chen, and Joan Bruna. On graph neural networks versus graph-augmented mlps. In International Conference on Learning Representations, 2020.   
[6] Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 1725–1735. PMLR, 2020.   
[7] Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count substructures? In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
[8] Weilin Cong, Rana Forsati, Mahmut T. Kandemir, and Mehrdad Mahdavi. Minimal variance sampling with provable guarantees for fast training of graph neural networks. In KDD ’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020, pages 1393–1403. ACM, 2020.   
[9] Zhiyong Cui, Kristian Henrickson, Ruimin Ke, and Yinhai Wang. Traffic graph convolutional recurrent neural network: A deep learning framework for network-scale traffic learning and forecasting. IEEE Transactions on Intelligent Transportation Systems, 2019.   
[10] Kenneth R Davidson and Stanislaw J Szarek. Local operator theory, random matrices and banach spaces. Handbook of the geometry of Banach spaces, 1(317-366):131, 2001.   
[11] Songgaojun Deng, Huzefa Rangwala, and Yue Ning. Learning dynamic context graphs for predicting social events. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 1007–1016. ACM, 2019.   
[12] Yash Deshpande, Subhabrata Sen, Andrea Montanari, and Elchanan Mossel. Contextual stochastic block models. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 8590–8602, 2018.   
[13] Kien Do, Truyen Tran, and Svetha Venkatesh. Graph transformation policy network for chemical reaction prediction. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 750–760. ACM, 2019.   
[14] Simon S. Du, Kangcheng Hou, Ruslan Salakhutdinov, Barnabás Póczos, Ruosong Wang, and Keyulu Xu. Graph neural tangent kernel: Fusing graph neural networks with graph kernels. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 5724–5734, 2019.   
[15] David Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gómez-Bombarelli, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P. Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pages 2224–2232, 2015.   
[16] Ran El-Yaniv and Dmitry Pechyony. Stable transductive learning. In Learning Theory, 19th Annual Conference on Learning Theory, COLT 2006, Pittsburgh, PA, USA, June 22-25, 2006, Proceedings, volume 4005 of Lecture Notes in Computer Science, pages 35–49. Springer, 2006.

[17] Vikas K. Garg, Stefanie Jegelka, and Tommi S. Jaakkola. Generalization and representational limits of graph neural networks. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 3419–3430. PMLR, 2020.   
[18] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 249–256. JMLR Workshop and Conference Proceedings, 2010.   
[19] Shunwang Gong, Mehdi Bahri, Michael M. Bronstein, and Stefanos Zafeiriou. Geometrically principled connections in graph neural networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 11412–11421. IEEE, 2020.   
[20] William L. Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 1024–1034, 2017.   
[21] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pages 1026–1034. IEEE Computer Society, 2015.   
[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 770–778. IEEE Computer Society, 2016.   
[23] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
[24] Wei Huang, Yayong Li, Weitao Du, Richard Yi Da Xu, Jie Yin, and Ling Chen. Wide graph neural networks: Aggregation provably leads to exponentially trainability loss. arXiv preprint arXiv:2103.03113, 2021.   
[25] Wenbing Huang, Yu Rong, Tingyang Xu, Fuchun Sun, and Junzhou Huang. Tackling oversmoothing for general graph convolutional networks. arXiv e-prints, pages arXiv–2008, 2020.   
[26] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.   
[27] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.   
[28] Johannes Klicpera, Aleksandar Bojchevski, and Stephan Günnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.   
[29] Kezhi Kong, Guohao Li, Mucong Ding, Zuxuan Wu, Chen Zhu, Bernard Ghanem, Gavin Taylor, and Tom Goldstein. Flag: Adversarial data augmentation for graph neural networks. arXiv preprint arXiv:2010.09891, 2020.   
[30] Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 1269–1278. ACM, 2019.   
[31] AA Leman and B Weisfeiler. A reduction of a graph to a canonical form and an algebra arising during this reduction. Nauchno-Technicheskaya Informatsiya, 2(9):12–16, 1968.   
[32] Guohao Li, Matthias Müller, Ali K. Thabet, and Bernard Ghanem. Deepgcns: Can gcns go as deep as cnns? In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages 9266–9275. IEEE, 2019.

[33] Jia Li, Zhichao Han, Hong Cheng, Jiao Su, Pengyun Wang, Jianfeng Zhang, and Lujia Pan. Predicting path failure in time-evolving graphs. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 1279–1289. ACM, 2019.   
[34] Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 3538–3545. AAAI Press, 2018.   
[35] Renjie Liao, Raquel Urtasun, and Richard Zemel. A pac-bayesian approach to generalization bounds for graph neural networks. arXiv preprint arXiv:2012.07690, 2020.   
[36] Andreas Loukas. What graph neural networks cannot learn: depth vs width. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.   
[37] Sitao Luan, Mingde Zhao, Xiao-Wen Chang, and Doina Precup. Training matters: Unlocking potentials of deeper graph convolutional neural networks. arXiv preprint arXiv:2008.08838, 2020.   
[38] Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 4602–4609. AAAI Press, 2019.   
[39] Quynh Nguyen. On the proof of global convergence of gradient descent for deep relu networks with linear widths. arXiv preprint arXiv:2101.09612, 2021.   
[40] Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.   
[41] Kenta Oono and Taiji Suzuki. Optimization and generalization analysis of transduction through gradient boosting and application to multi-scale graph neural networks. Advances in Neural Information Processing Systems, 33, 2020.   
[42] Namyong Park, Andrey Kan, Xin Luna Dong, Tong Zhao, and Christos Faloutsos. Estimating node importance in knowledge graphs using graph neural networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 596–606. ACM, 2019.   
[43] Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang. Deepinf: Modeling influence locality in large social networks. In KDD, 2018.   
[44] Afshin Rahimi, Trevor Cohn, and Timothy Baldwin. Semi-supervised user geolocation via graph convolutional networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2009–2019. Association for Computational Linguistics, 2018.   
[45] Morteza Ramezani, Weilin Cong, Mehrdad Mahdavi, Anand Sivasubramaniam, and Mahmut Kandemir. Gcn meets gpu: Decoupling “when to sample” from “how to sample”. Advances in Neural Information Processing Systems, 33, 2020.   
[46] Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph convolutional networks on node classification. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.   
[47] Ruslan Salakhutdinov. Deep learning. In The 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’14, New York, NY, USA - August 24 - 27, 2014, page 1973. ACM, 2014.   
[48] Franco Scarselli, Ah Chung Tsoi, and Markus Hagenbuchner. The vapnik–chervonenkis dimension of graph and recursive neural networks. Neural Networks, 108:248–259, 2018.

[49] Saurabh Verma and Zhi-Li Zhang. Stability and generalization of graph convolutional neural networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 1539– 1548. ACM, 2019.   
[50] Hao Wang, Tong Xu, Qi Liu, Defu Lian, Enhong Chen, Dongfang Du, Han Wu, and Wen Su. MCNE: an end-to-end framework for learning multiple conditional network representations of social network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 1064–1072. ACM, 2019.   
[51] Hongwei Wang, Fuzheng Zhang, Mengdi Zhang, Jure Leskovec, Miao Zhao, Wenjie Li, and Zhongyuan Wang. Knowledge-aware graph neural networks with label smoothness regularization for recommender systems. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 968–977. ACM, 2019.   
[52] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. KGAT: knowledge graph attention network for recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pages 950–958. ACM, 2019.   
[53] Felix Wu, Amauri H. Souza Jr., Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Q. Weinberger. Simplifying graph convolutional networks. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 6861–6871. PMLR, 2019.   
[54] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.   
[55] Keyulu Xu, Jingling Li, Mozhi Zhang, Simon S. Du, Ken-ichi Kawarabayashi, and Stefanie Jegelka. What can neural networks reason about? In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.   
[56] Keyulu Xu, Mozhi Zhang, Jingling Li, Simon Shaolei Du, Ken-Ichi Kawarabayashi, and Stefanie Jegelka. How neural networks extrapolate: From feedforward to graph neural networks. In International Conference on Learning Representations, 2021.   
[57] Yujun Yan, Milad Hashemi, Kevin Swersky, Yaoqing Yang, and Danai Koutra. Two sides of the same coin: Heterophily and oversmoothing in graph convolutional neural networks. arXiv preprint arXiv:2102.06462, 2021.   
[58] Chaoqi Yang, Ruijie Wang, Shuochao Yao, Shengzhong Liu, and Tarek Abdelzaher. Revisiting" over-smoothing" in deep gcns. arXiv preprint arXiv:2003.13663, 2020.   
[59] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2018, London, UK, August 19-23, 2018, pages 974–983. ACM, 2018.   
[60] Lingxiao Zhao and Leman Akoglu. Pairnorm: Tackling oversmoothing in gnns. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.   
[61] Kuangqi Zhou, Yanfei Dong, Wee Sun Lee, Bryan Hooi, Huan Xu, and Jiashi Feng. Effective training strategies for deep graph neural networks. arXiv preprint arXiv:2006.07107, 2020.   
[62] Xianchen Zhou and Hongxia Wang. The generalization error of graph convolutional networks may enlarge with more layers. Neurocomputing, 424:97–106, 2021.

# Supplementary Material On Provable Benefits of Depth in Training Graph Convolutional Networks

Organization. In Section A, we provide additional empirical evaluations on the change of pairwise distances for node embeddings as the number of layers increases. In Section B, we summarize the existing theoretical results on over-smoothing and provide empirical validation on whether oversmoothing happens in practice. In Sections C and D, we provide the proof of Theorem 1 (expressive power) and Theorem 2 (convergence to global optimal and characterization of number of iterations), respectively. In Section E, we provide more empirical results on the effectiveness of the proposed algorithm. In Sections F, G, H, I, and J we provide the generalization analysis of GCN, ResGCN, APPNP, GCNII, and DGCN, respectively. Code can be found at the following repository:

https://github.com/CongWeilin/DGCN.

# A Empirical results on the pairwise distance of node embeddings

In this section, we provide additional empirical evaluations on the change of pairwise distance for node embeddings. First we introduce a concrete definition of the intra- and inter-class pairwise distances, then provide the experimental setups, and finally illustrate the results.

Definition of pairwise distance. To verify whether over-smoothing exists in GCNs, we define the intra-class pairwise distance of $\mathbf { H } ^ { ( \ell ) }$ as the average pairwise Euclidean distance of two-node embeddings if they have the same ground truth label. Similarly, we define the inter-class pairwise distance of $\mathbf { H } ^ { ( \ell ) }$ as the average pairwise distance of two node embeddings if they have different ground truth labels.

$$
\operatorname {i n t r a} (\mathbf {H} ^ {(\ell)}) \triangleq \frac {\sum_ {i , j = 1} ^ {N} \mathbf {1} \left\{y _ {i} = y _ {j} \right\} \cdot \left\| \boldsymbol {h} _ {i} ^ {(\ell)} - \boldsymbol {h} _ {j} ^ {(\ell)} \right\| _ {2} ^ {2}}{\left\| \mathbf {H} ^ {(\ell)} \right\| _ {F} \cdot \sum_ {i , j = 1} ^ {N} \mathbf {1} \left\{y _ {i} = y _ {j} \right\}},
$$

$$
\mathrm {i n t e r} (\mathbf {H} ^ {(\ell)}) \triangleq \frac {\sum_ {i , j = 1} ^ {N} \mathbf {1} \{y _ {i} \neq y _ {j} \} \cdot \| \pmb {h} _ {i} ^ {(\ell)} - \pmb {h} _ {j} ^ {(\ell)} \| _ {2} ^ {2}}{\| \mathbf {H} ^ {(\ell)} \| _ {F} \cdot \sum_ {i , j = 1} ^ {N} \mathbf {1} \{y _ {i} \neq y _ {j} \}}.
$$

Both intra- and inter-class distances are normalized using the Frobenius norm to eliminate the difference caused by the scale of the node embedding matrix. By the definition of over-smoothing in [34], the distance between intra- and inter-class pairwise distance should be decreasing sharply as the number of GCN layers increases.

Setup. In Figure 5 and Figure 6, we plot the training error in a 10-layer GCN [27], SGC [53], and APPNP [28] models until convergence, and choose the model with the best validation score for pairwise distance computation. The pairwise distance at the `th layer is computed by the node embedding matrix $\mathbf { H } ^ { ( \ell ) }$ generated by the selected model. We repeat each experiment 10 times and plot the mean and standard deviation of the results.

Results. As shown in Figure 5 and Figure 6, when ignoring the weight matrices, i.e., sub-figures in box (b), the intra- and inter-class pairwise distance are decreasing but the gap between intra- and inter-class pairwise distance is increasing as the model goes deeper. That is, the difference between intra- and inter-class nodes’ embeddings is increasing and becoming more discriminative. On the other hand, when considering the weight matrices, i.e., sub-figures in box (a), the difference between intra- and inter-class node embeddings is large and increasing as the model goes deeper. In other words, weight matrices learn to make node embeddings discriminative and generate expressive node embeddings during training.

![](images/3319179f42a7fc64b82ffba20cfcafc77a68cdb0e2b12772e9cc883d29d4e5ac.jpg)

![](images/ccc89e6984fe8716028177947256e0d40ef0f5a34c68c9556a41dd35961ca64b.jpg)

![](images/d67f40bb976abacc562bee4f79122c7cefd74ded7c0ac488d6c6475212272bbe.jpg)

![](images/e6c915c33cae1bea706f1cea5593184d37d19997e6d1ac49b8c356312e5c533c.jpg)  
(b

![](images/231653edb8734d66c143504a6fa64c4fed9a6b065765644beb779643c4cd963e.jpg)  
Figure 5: Comparison of the pairwise distance for intra- and inter-class node embeddings on Cora dataset for different models by increasing the number of layers in the proposed architecture. There exist a fully connected layer after node feature

![](images/3a54c54346617c71a0aee13084b82418761348d363f3a9c32c32b5813e760b4b.jpg)  
  
?(ℓ) = ??(ℓ$%) , ?(&) = ?'?

![](images/c88a3ac0c0e8c40dbfa8005f6a01cfde42b2f520c2afb783f06a301dbab71961.jpg)  
?(ℓ) = ???(ℓ$%) + (1 − ?)?(&) , ?(&) = ?'?

![](images/72f367747e4779e30da4e88d19291654f09d52501e9c8b70948fea6962b76286.jpg)  
?(ℓ) = ?(?? ℓ$% ?(ℓ) ), ?(&) = ?

![](images/d0637923905a85a4269ec36900e488f850070469573dfd5d4ed8d7c6ae09197e.jpg)  
  
?(ℓ) = ??(ℓ$%) , ?(&) = ?

![](images/208d665c7f5387e074c92d3a31422f7c06894c5fd27cccbf04b169c2d0da2a4c.jpg)  
?(ℓ) = ???(ℓ$%) + (1 − ?)?(&) , ?(&) = ?   
Figure 6: Comparison of the pairwise distance for intra- and inter-class node embeddings on Citeseer dataset for different models by increasing the number of layers in the proposed architecture.

# B A summary of theoretical results on over-smoothing

In this section, we survey the existing theories on understanding over-smoothing in GCNs from [25, 40, 4] and illustrate why the underlying assumptions may not hold in practice. Finally, we discuss conditions where over-smoothing and over-fitting might happen simultaneously.

Before processing, let first recall the notation we defined in Section 3. Recall that $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ denotes the adjacency matrix of the self-connected graph, i.e., $A _ { i , j } = 1$ if edge $( i , j ) \in \mathcal { E }$ and $A _ { i , j } = 0$ otherwise, $\mathbf { D } \in \mathbb { R } ^ { N \times N }$ denotes the corresponding degree matrix, i.e., $D _ { i , i } = \deg ( i )$ and $D _ { i , j } = 0$ if $i \neq j$ , and the symmetric Laplacian matrix is computed as $\mathbf { L } = \mathbf { D } ^ { - 1 / 2 } \mathbf { A } \mathbf { D } ^ { - 1 / 2 }$ .

# B.1 Expressive power based analysis [41, 25]

Proposition 1 (Proposition 1 in [40], Theorem 1 of [25]). Let $\lambda _ { 1 } \geq . . . \geq \lambda _ { N }$ denote the eigenvalues of the Laplacian matrix L in descending order, and let $\mathbf { e } _ { i }$ denote the eigenvector associated with the eigenvalue $\lambda _ { i }$ . Suppose graph has $M \leq N$ connected components, then we have $\lambda _ { 1 } = . . . = \lambda _ { M } = 1$

and $1 > \lambda _ { M + 1 } > . . . > \lambda _ { N }$ . Let ${ \bf E } = \{ { \bf e } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { M \times N }$ denote the stack of eigenvectors that associated to eigenvalues $\lambda _ { 1 } , \dots , \lambda _ { M }$ . Then for all $i \in [ M ]$ , the eigenvector $\mathbf { e } _ { i } \in \mathbb { R } ^ { N }$ is defined as

$$
\mathbf {e} _ {i} = \left\{e _ {i} (j) \right\} _ {j = 1} ^ {N}, e _ {i} (j) \propto \left\{ \begin{array}{l l} \sqrt {\deg (j)} & \text {i f j t h n o d e i s i n t h e i t h c o n n e c t e d c o m p o n e n t} \\ 0 & \text {o t h e r w i s e} \end{array} . \right. \tag {3}
$$

Proof. The proof can be found in Section B of [40].

![](images/6719cda3a0df7cd3da5c21d006463077fa4e7552ae6581431ef05608e7747989.jpg)

[40] proposes to measure the expressive power of node embeddings using its distance to a subspace $\mathcal { M }$ that only has node degree information, i.e., regardless of the node feature information.

Definition 2 (Subspace and distance to subspace). We define subspace M as

$$
\mathcal {M} := \left\{\mathbf {E} ^ {\top} \mathbf {R} \in \mathbb {R} ^ {N \times d} \mid \mathbf {R} \in \mathbb {R} ^ {M \times d} \right\}, \tag {4}
$$

where ${ \bf E } = \{ { \bf e } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { M \times N }$ is the orthogonal basis of Laplacian matrix L, and R is any random matrix. The distance of a node embedding matrix $\mathbf { H } ^ { ( \ell ) } \in \mathbb { R } ^ { N \times d }$ to the subspace can be computed as $\begin{array} { r } { d _ { \mathcal { M } } \big ( \mathbf { H } ^ { ( \ell ) } \big ) = \operatorname* { i n f } _ { \mathbf { Y } \in \mathcal { M } } \| \mathbf { H } ^ { ( \ell ) } - \mathbf { Y } \| _ { \mathrm { F } } } \end{array}$ , where $\| \cdot \| _ { \mathrm { F } }$ denotes the Frobenius norm of the matrix.

Subspace $\mathcal { M }$ is a $d$ -dimensional subspace defined by orthogonal vectors $\{ { \mathbf { e } } _ { i } \} _ { i = 1 } ^ { M }$ that only captures node degree information. Intuitively, a smaller distance $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ means that $\mathbf { H } ^ { ( \ell ) }$ is losing its feature information but only carries the degree information.

In the following, we summarize the theoretical results of over-smoothing in [25] and extend their results to SGC and GCNII, which will be used in our discussion below.

Theorem 6 (Theorem 2 of [25]). Let $\lambda = \operatorname* { m a x } _ { i \in [ M + 1 , N ] } \left| \lambda _ { i } \right|$ denote the largest absolute eigenvalue of Laplacian matrix L which is bounded by 1, let s denote the largest singular-value of weight parameter $\mathbf { W } ^ { ( \ell ) }$ , $\ell \in [ L ]$ , then we have

$$
d _ {\mathcal {M}} \left(\mathbf {H} ^ {(\ell)}\right) - \varepsilon \leq \gamma \left(d _ {\mathcal {M}} \left(\mathbf {H} ^ {(\ell - 1)}\right) - \varepsilon\right), \tag {5}
$$

where $\gamma$ and $\varepsilon$ are functions of λ and s that depend on the model structure.

GCN. For vanilla GCN model with the graph convolutional operation defined as

$$
\mathbf {H} ^ {(\ell)} = \sigma \left(\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}\right), \mathbf {H} ^ {(0)} = \mathbf {X}, \tag {6}
$$

we have $\gamma _ { \mathsf { G C N } } = \lambda s$ and $\epsilon _ { \mathsf { G C N } } = 0$ .

Under the assumption that $\gamma _ { \mathsf { G C N } } \leq 1$ , i.e., the graph is densely connected and the largest singular value of weight matrices is small, we have $d _ { \mathcal { M } } ( \bar { \mathbf { H } } ^ { ( \ell + 1 ) } ) \leq \gamma _ { \mathsf { G C N } } \cdot d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ . In other words, the vanilla GCN model is losing expressive power as the number of layers increases.

However, if we suppose there exist a trainable bias parameter ${ \bf B } ^ { ( \ell ) } = \{ { \bf b } ^ { ( \ell ) } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times d _ { \ell } }$ in each graph convolutional layer

$$
\mathbf {H} ^ {(\ell)} = \sigma \left(\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} + \mathbf {B} ^ {(\ell)}\right), \mathbf {H} ^ {(0)} = \mathbf {X}, \tag {7}
$$

we have $\gamma _ { \sf G C N } = \lambda s$ and $\epsilon _ { \mathsf { G C N } } = d _ { \mathcal { M } } ( \mathbf { B } ^ { ( \ell ) } )$ . When $d _ { \mathcal { M } } ( \mathbf { B } ^ { ( \ell ) } )$ is considerably large, there is no guarantee that GCN-bias suffers from losing expressive power issue.

ResGCN. For GCN with residual connection, the graph convolution operation is defined as

$$
\mathbf {H} ^ {(\ell)} = \sigma \left(\mathbf {L H} ^ {(\ell)} \mathbf {W} ^ {(\ell)}\right) + \mathbf {H} ^ {(\ell - 1)}, \mathbf {H} ^ {(0)} = \mathbf {X W} ^ {(0)}, \tag {8}
$$

where we have $\gamma _ { \mathsf { R e s G C N } } = 1 + \lambda s$ and $\epsilon _ { \mathsf { R e s G C N } } = 0$ . Since $\gamma _ { \mathsf { R e s G C N } } \geq 1$ , there is no guarantee that ResGCN suffers from losing expressive power.

APPNP. For APPNP with graph convolution operation defined as

$$
\mathbf {H} ^ {(\ell)} = \alpha \mathbf {L} \mathbf {H} ^ {(\ell)} + (1 - \alpha) \mathbf {H} ^ {(0)}, \mathbf {H} ^ {(0)} = \mathbf {X} \mathbf {W} ^ {(0)}, \tag {9}
$$

we have $\gamma _ { \mathsf { A P P N P } } = \alpha \lambda$ and $\begin{array} { r } { \epsilon _ { \mathsf { A P P N P } } = \frac { ( 1 - \alpha ) d _ { \mathsf { M } } ( \mathbf { H } ^ { ( 0 ) } ) } { 1 - \gamma _ { \mathsf { A P P N P } } } } \end{array}$ . Although γAPPNP $< 1$ , because APPNP can be large, there is no guarantee that APPNP suffers from losing expressive power.

GCNII. For GCNII with graph convolution operation defined as

$$
\mathbf {H} ^ {(\ell)} = \sigma \Big (\big (\alpha \mathbf {L} \mathbf {H} ^ {(\ell)} + (1 - \alpha) \mathbf {H} ^ {(0)} \big) \big (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I} _ {N} \big) \Big), \quad \mathbf {H} ^ {(0)} = \mathbf {X} \mathbf {W} ^ {(0)}, \tag {10}
$$

we have $\gamma _ { \sf G C N I I } = \Big ( 1 - ( 1 - \beta ) ( 1 - s ) \Big ) \alpha \lambda$ , and $\begin{array} { r } { \epsilon _ { \mathsf { G C N I I } } = \frac { ( 1 - \alpha ) d _ { \mathsf { M } } ( \mathbf { H } ^ { ( 0 ) } ) } { 1 - \gamma _ { \mathsf { G C N I I } } } } \end{array}$ . Although we have $\gamma _ { \mathsf { G C N I I } } < 1$ , because $\epsilon _ { \mathsf { G C N I I } }$ can be considerably large, there is no guarantee that GCNII suffers from losing expressive power.

SGC. The result can be also extended to the linear model SGC [53], where the graph convolution is defined as

$$
\mathbf {H} ^ {(\ell)} = \mathbf {L} \mathbf {H} ^ {(\ell - 1)}, \mathbf {H} ^ {(0)} = \mathbf {X} \mathbf {W} ^ {(0)}, \tag {11}
$$

we have $\gamma _ { \mathsf { S G C } } = \lambda$ , and $\epsilon _ { S G C } = 0$ , which guarantees losing expressive power as the number of layers increases.

Discussion on the result of Theorem 6. In summary, the linear model SGC always suffers from losing expressive power without the assumption on the trainable parameters. Under the assumption that the multiplication of the largest singular-value of trainable parameter and the largest absolute eigenvalue of Laplacian matrix smaller than 1, i.e., $\lambda s \leq 1$ , we can only guarantee that GCN suffers from losing expressive power issue, but cannot have the same guarantee on GCN-bias, ResGCN, APPNP, and GCNII. However, as we show in Figure 3, the assumption is not going to hold in practice, and the distances are not decreasing for most cases.

Remark 2. [40] conducts experiments on Erd˝os-Rényi graph to show that when the graph is sufficiently dense and large, vanilla GCN suffers from expressive lower loss. Erd˝os-Rényi graph is constructed by connecting nodes randomly. Each edge is included in the graph with probability $p$ independent from every other edge. To guarantee a small $\lambda$ , a dense graph with larger p is required. For example, in the Section 6.2 of [40], they choose $p = 0 . 5$ (one node is connected to $5 0 \%$ of other nodes) such that $\lambda = 0 . 0 6 3$ and choose $p = 0 . 1$ (one node is connected to $1 0 \%$ of other nodes) such that $\lambda = 0 . 1 9 5$ . However, real world datasets are sparse and have a λ that is closer to 1. For example, Cora has $\lambda = 0 . 9 9 6 4$ , Citeseer has $\lambda = 0 . 9 9 8 7$ , and Pubmed has $\lambda = 0 . 9 9 0 5$ .

In Figure 7 and Figure 8, we compare the testing set F1-score, expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ before and after training. $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ is computed on the final output of a `-layer GCN model. The “after training results” are computed on the model with the best testing score. We repeat each experiment 10 times and report the average values. As shown in Figure 7 and Figure 8, the expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of the untrained vanilla GCN is decreasing as the number of layers increases. However, the expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of the trained vanilla GCN increases with more layers. Besides, we observe that there is no obvious connection between the testing F1-score to $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ for other GCN variant. For example, the testing F1-score of APPNP is increasing while its $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ is not changing much.

In Figure 9 and Figure 10, we compare the testing set F1-score, expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ before and after training. $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ is computed on the `th layer intermediate node embeddings of a 10-layer GCN model. The “after training results” are computed on the model with the best testing score. We repeat experiment 10 times and report the average values. As shown in Figure 9 and Figure 10, the expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of the untrained vanilla GCN is decreasing as the number of layers increases. However, the expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of the trained vanilla GCN increases as the number of layers increases.

# B.2 Dirichlet energy based analysis [4]

Definition 3 (Dirichlet energy). The Dirichlet energy of node embedding matrix $\mathbf { H } ^ { ( \ell ) }$ is defined as

$$
D E \left(\mathbf {H} ^ {(\ell)}\right) = \frac {1}{2} \sum_ {i, j = 1} ^ {N} L _ {i, j} \left\| \frac {\mathbf {h} _ {i} ^ {(\ell)}}{\sqrt {\deg (i)}} - \frac {\mathbf {h} _ {j} ^ {(\ell)}}{\sqrt {\deg (j)}} \right\| _ {2} ^ {2}. \tag {12}
$$

Intuitively, the Dirichlet energy measures the smoothness of node embeddings. Based on the definition of Dirichlet energy, they obtain a similar result as shown in [41] that

$$
\mathrm {D E} \left(\mathbf {H} ^ {(\ell + 1)}\right) \leq \lambda s \cdot \mathrm {D E} \left(\mathbf {H} ^ {(\ell)}\right), \tag {13}
$$

![](images/7c0557fa1250bc1dc722a3fbe0fa726555165b2ea49f7d591095a764b4288206.jpg)

![](images/cf9391c9f55aa06b3293f78a3f37538c4bfacf7d11035f8493d10e64b1081d40.jpg)

![](images/6e001bfc47b6f2be5df919e30a3774cfda23e00286e7de3940ae5ac26cf9139c.jpg)

![](images/97193754413f48041a45021f2c9be207c948ad5581ff8f09feac155645cf114c.jpg)

![](images/a054ea16c15608d0f32f06bc6da3bf614d3b0573418c1c2caf084dd24d7fe062.jpg)

![](images/fcc209f1e126a25f948d84d563a6d5ea45d1ebbfe26f3e76168a553c71c6f420.jpg)

![](images/31130af2e7a96e12d2a07048a222c13470d6bafd06487fd1a7a4dc3a6fdf8ec8.jpg)

![](images/6bde959b26dda95b65e78f804456687ae821f56814eef85542ea7036d336dc76.jpg)

![](images/ca2314a7de987f66a0f7ebbd872c2d20c172ade08839066ad6acc63f729eefd7.jpg)

![](images/854ad811b75af3133e33ec43160ea405c07087dad70181db34fef99b3d09e085.jpg)

![](images/68f66e0acd330c35a775050ddb33dadc789b84f8ccb4f77e8d27ab9e8da19f66.jpg)

![](images/55ed819d3906df0d67faebf3781a5ac3d460772ed061edf010b33a7802619945.jpg)

![](images/3e676c073b4dec75ff5c07a2b8715e5dab35073842a9066ead1fdc1622951445.jpg)

![](images/eeb41bbfcee088bf7b12771b91afade285b60ac24aea242986dd44f43382c48e.jpg)

![](images/bb224602374622c2a244f89841479d2e253050a0f1730b36eb1cb7b3ccf0291c.jpg)

![](images/f2ff5d858216bca2ef5ede346a55cb50fe8adc80a715f646839dfe145246c2f7.jpg)

![](images/c305626c96834196ba69595905aea0fbc48fdfda4e098963110e491febb68cd4.jpg)

![](images/ff0bf36b3071214077450cb78b438a13b5fe630b1ab3fc198cb5a33d480b673f.jpg)  
Figure 7: Comparison of F1-score, expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of GCN, GCN-bias, SGC, ResGCN, APPNP, and GCNII before and after training on Cora dataset. The average “largest sigular CORAvalue” of weight matrices in graph convolutional layers are reported, where “nan” stands for no weight matrices inside graph convolution layers. We only consider GCN model with depth from 2-layers to 10-layers. (Better viewed in PDF version)

![](images/35109e3ca428776a65bea9712c97e490dc2a3c175f0ccd3ca1969924a737ccd8.jpg)

![](images/5c06f0e352ca1397d2c675449e431fead3f47b44ceaef00866dea8a40a540b46.jpg)

![](images/0e3f972947706e3bb03f14627e4cb20ad057ba020c8884d7034c43dabd032a6a.jpg)

![](images/514fc3148ce11d9a59b3c06434c0fad11d496a9509e733468e4ffe1e271c5e55.jpg)

![](images/d4f37f4dccea2e48928be87827aa3acb1ad5641d4343a63e00dded90220d8036.jpg)

![](images/5c16973114ee23809399c0c3906fcea2880c028a919d51141cce080e5d032625.jpg)

![](images/08b14f7ab11b388fdd9900dd1fe45c44601602905719b3aa97ba488abc792dea.jpg)

![](images/939315b4578d6540dbeddfb4048efce11e21e5d63237b1c0c7d286683ab0adf4.jpg)

![](images/950977bf950b9d8a1e4229c3fa7c85b57d1251a38fc07be6b33db3ab100af97f.jpg)

![](images/7e3a2e6642935287ebb1b5af35122f24ffc6401420a4e355e17936056648fa14.jpg)

![](images/df1b62c48b7ba84dad0f8ee499775756aa0148348d1672c0e1d9bab3da9c4384.jpg)

![](images/5ed4fbb9f02c7841d444eb6454d81a5cd07de876fc6889827cd6520c0c27393c.jpg)

![](images/36259a3ed0246be031ccf27d978d9f63b138ac278474c5c3d8d7787425ac9d2e.jpg)

![](images/a1f972bc5c35a82c64ac8cb8a9092612c581cbd7975b786cab199b0cb4b1d506.jpg)

![](images/edc48c84815f8a02dd0453d14802420f9af89588e77e254aa573aaad578f8a32.jpg)

![](images/a7d9d8b1f0c3725e54e75c865603f6b45c00f5d47edb3c4b3852720765db1380.jpg)

![](images/c090baca5c7ce550bf5f05372a87bf787ecb9b894b1a39d9e261b8eb9cc0206a.jpg)

![](images/4298b0aaae63a60083d3ba537c8cf74c86f8333df543f328c47fb346d0dd4e09.jpg)  
Figure 8: Comparison of F1-score, expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of GCN, GCN-bias, SGC, ResGCN, APPNP, and GCNII before and after training on Citeseer dataset. The average “largest CITESEERsingular value” of weight matrices in graph convolutional layers are reported, where “nan” stands for no weight matrices inside graph convolution layers. We only consider GCN model with depth from 2-layers to 10-layers. (Better viewed in PDF version)

where $\lambda$ is largest absolute eigenvalue of Laplacian matrix L that is less than 1, and $s$ is the largest singular-value of trainable parameter $\dot { \mathbf { W } } ^ { ( \ell ) }$ .

Remark 3. The Dirichlet energy used in [4] is closely related to the pairwise distance as shown in Figure 2. In Figure 2, we measure the “smoothness” or “expressive power” of node embeddings using a normalized Dirichlet energy function, i.e., $\frac { D E ( \mathbf { H } ^ { ( \ell ) } ) } { \| \mathbf { H } ^ { ( \ell ) } \| _ { F } ^ { 2 } }$ for inner- and cross-class nodes respectively, which refers to the real smoothness metric of graph signal as pointed out in the footnote 1 of [4].

Intuitively, under the assumption that the largest singular-value of weight matrices less than 1, as the number of layers increase, i.e., $\ell \to \infty$ , the Dirichlet energy of node embeddings $\mathtt { D E } ( \mathbf { H } ^ { ( \ell ) } )$ is decreasing. However, recall that this assumption is not likely to hold in practice because the real-world graphs are usually sparse and the largest singular value of weight matrices is usually larger than 1.

Besides, generalizing the result to other GCN variants (e.g., ResGCN, GCNII) is non-trivial. For example in ResGCN, we have

$$
\begin{array}{l} \mathrm {D E} (\mathbf {H} ^ {(\ell)}) = \mathrm {D E} (\sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) + \mathbf {H} ^ {(\ell - 1)}) \\ \leq 2 \mathrm {D E} (\sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)})) + 2 \mathrm {D E} (\mathbf {H} ^ {(\ell - 1)}) \tag {14} \\ \leq 2 (\lambda s + 1) \mathrm {D E} \left(\mathbf {H} ^ {(\ell - 1)}\right). \\ \end{array}
$$

![](images/057852eb8c777dc0b2a674e69694138e9a5102b2fa22b44b790d27cb9778a8dc.jpg)

![](images/29fa6f663e065bf3cba431f00a81f5eb626937e722be6047284d24e54c9d9fad.jpg)  
Figure 9: Comparison of F1-score, expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of 10-layer GCN, GCN-bias, SGC, ResGCN, APPNP, and GCNII before and after training on Cora dataset. The average “largest singular value” of weight matrices in graph convolutional layers are reported, where “nan” stands forCiteseer_one_model no weight matrices inside graph convolution layers. (Better viewed in PDF version)   
Figure 10: Comparison of F1-score, expressive power metric $d _ { \mathcal { M } } ( \mathbf { H } ^ { ( \ell ) } )$ of 10-layer GCN, GCN-bias, SGC, ResGCN, APPNP, and GCNII before and after training on Citeseer dataset. The average “largest singular value” of weight matrices in graph convolutional layers are reported, where “nan” stands for no weight matrices inside graph convolution layers. (Better viewed in PDF version)

The coefficient 2 on the right hand side of the inequality makes the result less meaningful.

# B.3 A condition to have over-smoothing and over-fitting happen simultaneously

As it has alluded to before, in this paper, we argue that the performance degradation issue is mainly due to over-fitting, and over-smoothing is not likely to happen in practice. One might doubt whether over-smoothing can happen simultaneously with over-fitting. In the following, we show that oversmoothing and over-fitting might happen at the same time if a model can distinguish all nodes by only using graph structure information (e.g., node degree) and ignoring node feature information. However, such a condition is not likely in the practice.

For simplicity, let us assume that the graph contains only a single connected component. Let first consider the over-smoothing notation as defined in [40]. We can write the node embedding of node $i$ at the `th layer as $\mathbf { h } _ { i } ^ { ( \bar { \ell } ) } = \mathbf { e } \mathbf { R } _ { i , \ell } + \varepsilon _ { i , \ell } \in \mathbb { R } ^ { d }$ , where e is the eigen vector associated with the largest eigenvalue, $\mathbf { R } _ { i , \ell } \in \mathbb { R } ^ { N \times d }$ is the random projection matrix, and $\varepsilon _ { i , \ell } \in \mathbb { R } ^ { d }$ is a $d$ -dimensional random vector such that $d _ { \mathcal { M } } ( \mathbf { h } _ { i } ^ { ( \ell ) } ) = \lVert \boldsymbol { \varepsilon } _ { i , \ell } \rVert _ { 2 }$ corresponds to its distance to subspace $\mathcal { M }$ . According to the notation in [40], we have $\| \pmb { \varepsilon } _ { i , \ell } \| _ { 2 } \to 0$ as $\ell \to + \infty$ for any $i \in \mathcal V$ . The minimum training loss can be achieved by choosing the final layer weight vector $\mathbf { w } \in \mathbb { R } ^ { d }$ such that $\begin{array} { r } { \sum _ { i \in \mathcal { V } } \mathrm { L o s s } \big ( \mathbf { \bar { w } } ^ { \top } ( \mathbf { e R } _ { i , \ell } + \varepsilon _ { i , \ell } ) , y _ { i } \big ) } \end{array}$ is small. A small training error is achieved when $\mathbf { e R } _ { i , \ell }$ is discriminative. In other words, a model is under both over-smoothing and over-fitting conditions only if it can make predictions based on the graph structure, without leveraging node feature information. On the other hand, considering the over-smoothing notation as defined in [4], we know that oversmoothing happens if $\frac { \mathbf { h } _ { i } ^ { ( \ell ) } } { \sqrt { \operatorname* { d e g } ( i ) } } \approx \frac { \mathbf { h } _ { j } ^ { ( \ell ) } } { \sqrt { \operatorname* { d e g } ( j ) } }$ h(`) for any $i , j \in \mathcal { V }$ , i.e., a classifier can distinguish any nodes based on its node degree (only based on graph information without leveraging its feature information).

# C Proof of Theorem $^ 1$

Given a $L$ -layer computation tree $\mathcal { T } ^ { L }$ , let $v _ { i } ^ { \ell }$ denote the ith node in the `th layer of a $L$ -layer computation tree $\mathcal { T } ^ { L }$ . Let suppose each node has at least $d$ neighbors and each node has binary feature.

Let $P ( v _ { i } ^ { \ell } )$ denote the parent of node $v _ { i } ^ { \ell }$ and $C ( v _ { i } ^ { \ell } )$ denote the set of children of node $v _ { i } ^ { \ell }$ with $| C ( v _ { i } ^ { \ell } ) | \geq d$ . By the definition of computation tree, we know that $P ( v _ { i } ^ { \ell } ) \in C ( v _ { i } ^ { \ell } )$ for any $\ell > 1$ . As a result, we know that $C ( v _ { i } ^ { \ell } )$ has at least $( d - 1 )$ different choices since one of the node in $C ( v _ { i } ^ { \ell } )$ must be the same as $P ( v _ { i } ^ { \ell } )$ .

Therefore, we know that a $L$ -layer computation tree $\mathcal { T } ^ { L }$ has at least $2 ( d - 1 ) ^ { L - 1 }$ different choices, where the constant 2 is because each root node has at least two different choices.

# D Proof of Theorem 2

In this section, we study the convergence of $L$ -layer deep GCN model, and show that a deeper model requires more iterations $T$ to achieve $\epsilon$ training error. Before preceding, let first introduce the notations used for the convergence analysis of deep GCN. Let $\mathcal G ( \nu , \mathcal { E } )$ as the graph of $N = | \nu |$ nodes and each node in the graph is associated with a node feature and target label pair $\left( \mathbf { x } _ { i } , \mathbf { y } _ { i } \right)$ where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d _ { 0 } }$ and $\mathbf { y } _ { i } \in \mathbb { R } ^ { d _ { L + 1 } }$ . Let ${ \bf X } = \{ { \bf x } _ { i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times d _ { 0 } }$ and $\mathbf { Y } = \{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times d _ { L + 1 } }$ be the stack of node vector and target vector of the training data set of size $N$ . Given a set of $L$ matrices $\mathbf { \theta } \mathbf { \Lambda } \theta = \{ \mathbf { W } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ , we consider the following training objective with squared loss function to learn the parameters of the model:

$$
\underset {\theta} {\text {m i n i m i z e}} \mathcal {L} (\boldsymbol {\theta}) = \frac {1}{2} \| \hat {\mathbf {Y}} (\boldsymbol {\theta}) - \mathbf {Y} \| _ {\mathrm {F}} ^ {2}, \quad \hat {\mathbf {Y}} (\boldsymbol {\theta}) = \mathbf {H} ^ {(L)} \mathbf {W} ^ {(L + 1)}, \quad \mathbf {H} ^ {(\ell)} = \sigma \left(\mathbf {L} \mathbf {H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}\right), \tag {15}
$$

where $\sigma ( \cdot )$ is ReLU activation.

# D.1 Useful lemmas

The following lemma analyzes the Lipschitz continuity of deep GCNs, which characterizes the change of inner layer representations by slight change of the weight parameters.

Lemma 2. Let $\pmb { \theta } _ { 1 } = \{ \mathbf { W } _ { 1 } ^ { ( \ell ) } \} _ { \ell \in [ L + 1 ] }$ and $\pmb { \theta } _ { 2 } = \{ \mathbf { W } _ { 2 } ^ { ( \ell ) } \} _ { \ell \in [ L + 1 ] }$ be two set of parameters. Let $\lambda _ { \ell } \geq \operatorname* { m a x } \{ \| \mathbf { W } _ { 1 } ^ { ( \ell ) } \| _ { 2 } , \| \mathbf { W } _ { 2 } ^ { ( \ell ) } \| _ { 2 } \}$ , $s \geq \| \mathbf { L } \| _ { 2 }$ , and $\begin{array} { r } { \lambda _ { i  j } = \prod _ { \ell = i } ^ { j } \lambda _ { \ell } , } \end{array}$ , we have

$$
\left\| \mathbf {H} _ {1} ^ {(\ell)} - \mathbf {H} _ {2} ^ {(\ell)} \right\| _ {\mathrm {F}} \leq s ^ {\ell} \lambda_ {1 \rightarrow \ell} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {j = 1} ^ {\ell} \lambda_ {j} ^ {- 1} \| \mathbf {W} _ {1} ^ {(j)} - \mathbf {W} _ {2} ^ {(j)} \| _ {2}. \tag {16}
$$

Proof of Lemma 2. Our proof relies on the following inequality

$$
\forall \mathbf {A} \in \mathbb {R} ^ {a \times b}, \mathbf {B} \in \mathbb {R} ^ {b \times c}, \| \mathbf {A B} \| _ {\mathrm {F}} \leq \| \mathbf {A} \| _ {2} \| \mathbf {B} \| _ {\mathrm {F}} \text {a n d} \| \mathbf {A B} \| _ {\mathrm {F}} \leq \| \mathbf {A} \| _ {\mathrm {F}} \| \mathbf {B} \| _ {2}. \tag {17}
$$

By definition of $\mathbf { H } ^ { ( \ell ) }$ , we have for $\ell \in [ L ]$

$$
\begin{array}{l} \| \mathbf {H} ^ {(\ell)} \| _ {\mathrm {F}} = \| \sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) \| _ {\mathrm {F}} \\ \leq \left\| \mathbf {L} \mathbf {H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} \right\| _ {\mathrm {F}} \\ \leq \left\| \mathbf {L} \right\| _ {2} \left\| \mathbf {H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} \right\| _ {\mathrm {F}} \\ \leq \| \mathbf {L} \| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} \| \mathbf {H} ^ {(\ell - 1)} \| _ {\mathrm {F}} \tag {18} \\ \leq \| \mathbf {X} \| _ {\mathrm {F}} \prod_ {j = 1} ^ {\ell} \left(\| \mathbf {L} \| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2}\right). \\ \end{array}
$$

Let $\pmb { \theta } _ { 1 } = \{ \mathbf { W } _ { 1 } ^ { ( \ell ) } \} _ { \ell \in [ L + 1 ] }$ and $\pmb { \theta } _ { 2 } = \{ \mathbf { W } _ { 2 } ^ { ( \ell ) } \} _ { \ell \in [ L + 1 ] }$ be two set of parameters, we have

$$
\begin{array}{l} \| \mathbf {H} _ {1} ^ {(\ell)} - \mathbf {H} _ {2} ^ {(\ell)} \| _ {\mathrm {F}} = \| \sigma (\mathbf {L H} _ {1} ^ {(\ell - 1)} \mathbf {W} _ {1} ^ {(\ell)}) - \sigma (\mathbf {L H} _ {2} ^ {(\ell - 1)} \mathbf {W} _ {2} ^ {(\ell)}) \| _ {\mathrm {F}} \\ \underset {(a)} {\leq} \left\| \mathbf {L H} _ {1} ^ {(\ell - 1)} \mathbf {W} _ {1} ^ {(\ell)} - \mathbf {L H} _ {2} ^ {(\ell - 1)} \mathbf {W} _ {2} ^ {(\ell)} \right\| _ {\mathrm {F}} \\ \leq \| \mathbf {L} \| _ {2} \| \mathbf {H} _ {1} ^ {(\ell - 1)} \mathbf {W} _ {1} ^ {(\ell)} - \mathbf {H} _ {2} ^ {(\ell - 1)} \mathbf {W} _ {2} ^ {(\ell)} \| _ {\mathrm {F}} \\ = \| \mathbf {L} \| _ {2} \left(\| \mathbf {H} _ {1} ^ {(\ell - 1)} \left(\mathbf {W} _ {1} ^ {(\ell)} - \mathbf {W} _ {2} ^ {(\ell)}\right) + \left(\mathbf {H} _ {1} ^ {(\ell - 1)} - \mathbf {H} _ {2} ^ {(\ell - 1)}\right) \mathbf {W} _ {2} ^ {(\ell)} \| _ {\mathrm {F}}\right) \tag {19} \\ \leq \| \mathbf {L} \| _ {2} \left(\| \mathbf {H} _ {1} ^ {(\ell - 1)} \| _ {\mathrm {F}} \| \mathbf {W} _ {1} ^ {(\ell)} - \mathbf {W} _ {2} ^ {(\ell)} \| _ {2} + \| \mathbf {H} _ {1} ^ {(\ell - 1)} - \mathbf {H} _ {2} ^ {(\ell - 1)} \| _ {\mathrm {F}} \| \mathbf {W} _ {2} ^ {(\ell)} \| _ {2}\right) \\ \leq_ {(b)} \| \mathbf {L} \| _ {2} \| \mathbf {X} \| _ {\mathrm {F}} \left[ \prod_ {j = 1} ^ {\ell - 1} \left(\| \mathbf {L} \| _ {2} \| \mathbf {W} _ {1} ^ {(\ell)} \| _ {2}\right) \right] \| \mathbf {W} _ {1} ^ {(\ell)} - \mathbf {W} _ {2} ^ {(\ell)} \| _ {2} \\ + \| \mathbf {L} \| _ {2} \| \mathbf {W} _ {2} ^ {(\ell)} \| _ {2} \| \mathbf {H} _ {1} ^ {(\ell - 1)} - \mathbf {H} _ {2} ^ {(\ell - 1)} \| _ {\mathrm {F}}, \\ \end{array}
$$

where $( a )$ is due to $\sigma ( \cdot )$ is 1-Lipschitz continuous, $( b )$ is due to Eq. 18.

Let $\lambda _ { \ell } \geq \operatorname* { m a x } \{ \| \mathbf { W } _ { 1 } ^ { ( \ell ) } \| _ { 2 } , \| \mathbf { W } _ { 2 } ^ { ( \ell ) } \| _ { 2 } \}$ , $s \geq \| \mathbf { L } \| _ { 2 }$ , and $\begin{array} { r } { \lambda _ { i  j } = \prod _ { \ell = i } ^ { j } \lambda _ { \ell } , } \end{array}$ , we have

$$
\begin{array}{l} \| \mathbf {H} _ {1} ^ {(\ell)} - \mathbf {H} _ {2} ^ {(\ell)} \| _ {\mathrm {F}} \leq s ^ {\ell} \lambda_ {1 \rightarrow (\ell - 1)} \| \mathbf {X} \| _ {\mathrm {F}} \| \mathbf {W} _ {1} ^ {(\ell)} - \mathbf {W} _ {2} ^ {(\ell)} \| _ {2} + s \lambda_ {\ell} \| \mathbf {H} _ {1} ^ {(\ell - 1)} - \mathbf {H} _ {2} ^ {(\ell - 1)} \| _ {\mathrm {F}} \\ \leq s ^ {\ell} \lambda_ {1 \rightarrow \ell} \| \mathbf {X} \| _ {\mathrm {F}} \left(\sum_ {j = 1} ^ {\ell} \lambda_ {j} ^ {- 1} \| \mathbf {W} _ {1} ^ {(j)} - \mathbf {W} _ {2} ^ {(j)} \| _ {2}\right). \tag {20} \\ \end{array}
$$

![](images/b6acd455282901151e61b8e14ab322c5e72032b377aeb64fb8d4aa09d31a51c8.jpg)

The following lemma derives the upper bound of the gradient with respect to the weight parameters, which plays an important role in the convergence analysis of deep GCN.

Lemma 3. Let $\pmb { \theta } = \{ \mathbf { W } ^ { ( \ell ) } \} _ { \ell \in [ L + 1 ] }$ be the set of parameters, $\lambda _ { \ell } \geq \operatorname* { m a x } \{ \| \mathbf { W } _ { 1 } ^ { ( \ell ) } \| _ { 2 } , \| \mathbf { W } _ { 2 } ^ { ( \ell ) } \| _ { 2 } \} ,$ , $s \geq \| \mathbf { L } \| _ { 2 }$ , and $\begin{array} { r } { \lambda _ { i  j } = \prod _ { \ell = i } ^ { j } \lambda _ { \ell } } \end{array}$ , we have

$$
\left\| \frac {\partial \mathcal {L} (\boldsymbol {\theta})}{\partial \mathbf {W} ^ {(\ell)}} \right\| _ {\mathrm {F}} \leq \frac {s ^ {L} \lambda_ {1 \rightarrow (L + 1)}}{\lambda_ {\ell}} \| \mathbf {X} \| _ {\mathrm {F}} \| \widehat {\mathbf {Y}} - \mathbf {Y} \| _ {\mathrm {F}}. \tag {21}
$$

Proof of Lemma 3. By the definition of ∂L(θ)∂W(`) , we have $\frac { \partial \mathcal { L } ( \pmb { \theta } ) } { \partial \mathbf { W } ^ { ( \ell ) } }$

$$
\begin{array}{l} \left\| \frac {\partial \mathcal {L} (\boldsymbol {\theta})}{\partial \mathbf {W} ^ {(\ell)}} \right\| _ {\mathrm {F}} = \left\| \left(\frac {\partial \widehat {\mathbf {Y}}}{\partial \mathbf {W} ^ {(\ell)}}\right) ^ {\top} (\widehat {\mathbf {Y}} - \mathbf {Y}) \right\| _ {\mathrm {F}} \\ = \left\| \frac {\partial \left[ \mathbf {I} _ {N} \otimes \left(\mathbf {W} ^ {(L + 1)}\right) ^ {\top} \right] \operatorname {v e c} \left(\mathbf {H} ^ {(L)}\right)}{\partial \operatorname {v e c} \left(\mathbf {W} ^ {(\ell)}\right)} \operatorname {v e c} \left(\widehat {\mathbf {Y}} - \mathbf {Y}\right) \right\| _ {2} \\ = \left\| \left[ \mathbf {I} _ {N} \otimes \left(\mathbf {W} ^ {(L + 1)}\right) ^ {\top} \right] \frac {\partial \operatorname {v e c} \left(\mathbf {H} ^ {(L)}\right)}{\partial \operatorname {v e c} \left(\mathbf {H} ^ {(\ell)}\right)} \frac {\partial \operatorname {v e c} \left(\mathbf {H} ^ {(\ell)}\right)}{\partial \operatorname {v e c} \left(\mathbf {W} ^ {(\ell)}\right)} \operatorname {v e c} (\widehat {\mathbf {Y}} - \mathbf {Y}) \right\| _ {2} \\ \leq_ {(a)} \left\| \left[ \mathbf {I} _ {N} \otimes (\mathbf {W} ^ {(L + 1)}) ^ {\top} \right] \frac {\partial \operatorname {v e c} \left(\mathbf {L H} ^ {(L - 1)} \mathbf {W} ^ {(L)}\right)}{\partial \operatorname {v e c} (\mathbf {H} ^ {(\ell)})} \frac {\partial \operatorname {v e c} \left(\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}\right)}{\partial \operatorname {v e c} (\mathbf {W} ^ {(\ell)})} \operatorname {v e c} (\widehat {\mathbf {Y}} - \mathbf {Y}) \right\| _ {2} \\ = \left\| \left[ \mathbf {I} _ {N} \otimes \left(\mathbf {W} ^ {(L + 1)}\right) ^ {\top} \right] \frac {\partial \operatorname {v e c} \left(\mathbf {L} \otimes \left(\mathbf {W} ^ {(L)}\right) ^ {\top}\right) \operatorname {v e c} \left(\mathbf {H} ^ {(L - 1)}\right)}{\partial \operatorname {v e c} \left(\mathbf {H} ^ {(\ell)}\right)} \frac {\partial \left[ \mathbf {L} \mathbf {H} ^ {(\ell - 1)} \otimes \mathbf {I} _ {d _ {\ell}} \right] \operatorname {v e c} \left(\mathbf {W} ^ {(\ell)}\right)}{\partial \operatorname {v e c} \left(\mathbf {W} ^ {(\ell)}\right)} \operatorname {v e c} (\widehat {\mathbf {Y}} - \mathbf {Y}) \right\| _ {2} \\ = \left\| \left[ \mathbf {I} _ {N} \otimes \left(\mathbf {W} ^ {(L + 1)}\right) ^ {\top} \right] \operatorname {v e c} \left(\mathbf {L} ^ {L - \ell} \otimes \left(\mathbf {W} ^ {(\ell + 1)} \dots \mathbf {W} ^ {(L)}\right) ^ {\top}\right) \left[ \mathbf {L H} ^ {(\ell - 1)} \otimes \mathbf {I} _ {d _ {\ell}} \right] \operatorname {v e c} \left(\widehat {\mathbf {Y}} - \mathbf {Y}\right) \right\| _ {2} \\ = \left\| \underbrace {\left(\mathbf {L} ^ {L - \ell + 1} \mathbf {H} ^ {(\ell - 1)}\right) \otimes \left(\mathbf {W} ^ {(\ell + 1)} \dots \mathbf {W} ^ {(L + 1)}\right) ^ {\top}} _ {\mathbb {R} ^ {N d _ {L + 1} \times d _ {\ell - 1} d _ {\ell}}} \operatorname {v e c} (\widehat {\mathbf {Y}} - \mathbf {Y}) \right\| _ {\mathrm {F}} \\ \end{array}
$$

$$
\begin{array}{l} = \left\| \left(\mathbf {W} ^ {(\ell + 1)} \dots \mathbf {W} ^ {(L + 1)}\right) \left(\widehat {\mathbf {Y}} - \mathbf {Y}\right) ^ {\top} \mathbf {L} ^ {L - \ell + 1} \mathbf {H} ^ {(\ell - 1)} \right\| _ {2} \\ \leq s ^ {L - \ell + 1} \lambda_ {(\ell + 1) \rightarrow (L + 1)} \| \widehat {\mathbf {Y}} - \mathbf {Y} \| _ {\mathrm {F}} \| \mathbf {H} ^ {(\ell - 1)} \| _ {\mathrm {F}}, \\ \end{array}
$$

where $( a )$ is due to $\| \sigma ( x ) \| _ { 2 } \leq \| x \| _ { 2 }$ .

Using similar proof strategy, we can upper bound $\| \mathbf { H } ^ { ( \ell - 1 ) } \| _ { \mathrm { F } }$ by

$$
\begin{array}{l} \| \mathbf {H} ^ {(\ell - 1)} \| _ {\mathrm {F}} = \| \sigma (\mathbf {L H} ^ {(\ell - 2)} \mathbf {W} ^ {(\ell - 1)}) \| _ {\mathrm {F}} \\ \leq \left\| \mathbf {L} \mathbf {H} ^ {(\ell - 2)} \mathbf {W} ^ {(\ell - 1)} \right\| _ {\mathrm {F}} \tag {22} \\ \leq s ^ {\ell - 1} \lambda_ {1 \rightarrow (\ell - 1)} \| \mathbf {X} \| _ {\mathrm {F}}. \\ \end{array}
$$

By combining the above two equations, we have

$$
\left\| \frac {\partial \mathcal {L} (\boldsymbol {\theta})}{\partial \mathbf {W} ^ {(\ell)}} \right\| _ {\mathrm {F}} \leq \frac {s ^ {L} \lambda_ {1 \rightarrow (L + 1)}}{\lambda_ {\ell}} \| \mathbf {X} \| _ {\mathrm {F}} \| \widehat {\mathbf {Y}} - \mathbf {Y} \| _ {\mathrm {F}}. \tag {23}
$$

# D.2 Main result

In the following, we provide the convergence analysis on deep GCNs. In particular, Theorem 7 shows that under assumption on the weight initialization (Eq. 24) and the width of the last layer’s input dimension $\langle d _ { L } \geq N )$ , the deep GCN enjoys linear convergence rate. Recall our discussion in Section 4 that $\lambda _ { \ell } \geq 1$ (by Gordon’s Theorem for Gaussian matrices) and $\| \mathbf { L } \| _ { 2 } = 1$ for ${ \bf L } = { \bf D } ^ { - 1 / 2 } { \bf A } { \bf D } ^ { - 1 / 2 }$ , we know that $s \lambda _ { \ell } \geq 1$ , thus deeper model requires a smaller learning rate $\eta$ according to Eq. 25. Since the number of training iteration $T$ is reverse proportional to learning rate $\eta$ , which explains why a deeper model requires more training iterations than the shallow one.

Theorem 7. Consider a deep GCN with ReLU activation (defined in Eq. 15) where the width of the last hidden layer satisfies $d _ { L } \geq N$ . Let $\{ C _ { \ell } \} _ { \ell \in [ L + 1 ] }$ be any sequence of positive numbers and define $\alpha _ { 0 } = \lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { 0 } ^ { ( L ) } )$ , $\begin{array} { r } { \lambda _ { \ell } = \| \mathbf { W } _ { 0 } ^ { ( \ell ) } \| _ { 2 } + C _ { \ell } , ~ \lambda _ { i \to j } = \prod _ { \ell = i } ^ { j } \lambda _ { \ell } . } \end{array}$

Assume the following conditions are satisfied at the initialization:

$$
\alpha_ {0} ^ {2} \geq 1 6 s ^ {2 L} \| \mathbf {X} \| _ {\mathrm {F}} \max _ {\ell \in [ L + 1 ]} \frac {\lambda_ {1 \rightarrow (L + 1)}}{\lambda_ {\ell} C _ {\ell}} \sqrt {2 \mathcal {L} (\pmb {\theta} _ {0})},
$$

$$
\alpha_ {0} ^ {3} \geq 3 2 s ^ {2 L} \| \mathbf {X} \| _ {\mathrm {F}} ^ {2} \lambda_ {L + 1} \sum_ {\ell = 1} ^ {L} \frac {\lambda_ {1 \rightarrow L} ^ {2}}{\lambda_ {\ell} ^ {2}} \sqrt {2 \mathcal {L} \left(\boldsymbol {\theta} _ {0}\right)}, \tag {24}
$$

$$
\alpha_ {0} ^ {2} \geq 1 6 s ^ {2 L} \| \mathbf {X} \| _ {\mathrm {F}} ^ {2} \lambda_ {L + 1} ^ {2} \sum_ {\ell = 1} ^ {L} \frac {\lambda_ {1 \rightarrow L} ^ {2}}{\lambda_ {\ell} ^ {2}}.
$$

Let the learning rate satisfies

$$
\eta \leq \min  \left(\frac {8}{\alpha_ {0} ^ {2}}, \frac {\left(\sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 2}\right)}{s ^ {2 L} \lambda_ {1 \rightarrow (L + 1)} ^ {2} \left\| \mathbf {X} \right\| _ {\mathrm {F}} ^ {2} \left(\sum_ {\ell = 1} ^ {L + 1} \lambda_ {\ell} ^ {- 2}\right) ^ {2}}\right). \tag {25}
$$

Then deep GCN can achieve $\mathcal { L } ( \pmb { \theta } _ { T } ) \leq \epsilon$ for any $T$ satisfied

$$
T \geq \frac {8}{\eta \alpha_ {0} ^ {2}} \log \left(\frac {\mathcal {L} (\boldsymbol {\theta} _ {0})}{\epsilon}\right). \tag {26}
$$

Proof of Theorem 7. The proof follows the proof of Theorem 2.2 in [39] by extending result from MLP to GCN. The key idea of the proof is to show the followings hold by induction:

$$
\left\{ \begin{array}{l l} \| \mathbf {W} _ {r} ^ {(\ell)} \| _ {2} \leq \lambda_ {\ell} & \text {f o r a l l} \ell \in [ L ], r \in [ 0, t ], \\ \lambda_ {\min } \left(\mathbf {H} _ {r} ^ {(L)}\right) \geq \frac {\alpha_ {0}}{2} & \text {f o r a l l} r \in [ 0, t ], \\ \mathcal {L} (\boldsymbol {\theta} _ {r}) \leq (1 - \eta \alpha_ {0} ^ {2} / 8) \mathcal {L} (\boldsymbol {\theta} _ {0}) & \text {f o r a l l} r \in [ 0, t ]. \end{array} \right. \tag {27}
$$

Clearly, Eq. 27 holds for $t = 0$ . Let assume Eq. 27 holds up to iteration $t$ , and let show it also holds for $( t + 1 ) \mathrm { t h }$ iteration.

Let first prove the first inequality holds in Eq. 27. By the triangle inequality, we can decompose the distance between $\mathbf { W } _ { t + 1 } ^ { ( \ell ) }$ to $\mathbf { \bar { W } } _ { 0 } ^ { ( \ell ) }$ by

$$
\begin{array}{l} \| \mathbf {W} _ {t + 1} ^ {(\ell)} - \mathbf {W} _ {0} ^ {(\ell)} \| _ {\mathrm {F}} \leq \sum_ {k = 0} ^ {t} \| \mathbf {W} _ {k + 1} ^ {(\ell)} - \mathbf {W} _ {k} ^ {(\ell)} \| _ {\mathrm {F}} \\ = \eta \sum_ {k = 0} ^ {t} \left\| \frac {\partial \mathcal {L} (\boldsymbol {\theta})}{\partial \mathbf {W} _ {k} ^ {(\ell)}} \right\| _ {\mathrm {F}} \\ \leq \eta s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \lambda_ {\ell} ^ {- 1} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {k = 0} ^ {t} \| \widehat {\mathbf {Y}} _ {k} - \mathbf {Y} \| _ {\mathrm {F}} \tag {28} \\ \leq \eta s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \lambda_ {\ell} ^ {- 1} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {k = 0} ^ {t} \sqrt {2 \mathcal {L} (\boldsymbol {\theta} _ {k})} \\ \leq_ {(a)} \eta s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \lambda_ {\ell} ^ {- 1} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {k = 0} ^ {t} \left(1 - \eta \frac {\alpha_ {0} ^ {2}}{8}\right) ^ {k / 2} \sqrt {2 \mathcal {L} (\pmb {\theta} _ {0})}, \\ \end{array}
$$

where inequality $( a )$ follows the induction condition.

Let $u = \sqrt { 1 - \eta \alpha _ { 0 } ^ { 2 } / 8 }$ we have $\eta = 8 ( 1 - u ^ { 2 } ) / \alpha _ { 0 } ^ { 2 }$ and

$$
\sum_ {k = 0} ^ {t} \left(1 - \eta \frac {\alpha_ {0} ^ {2}}{8}\right) ^ {k / 2} = \sum_ {k = 0} ^ {t} u ^ {k} = \frac {1 - u ^ {t + 1}}{1 - u}. \tag {29}
$$

Then, plugging the result into Eq. 28, we have

$$
\begin{array}{l} \left\| \mathbf {W} _ {t + 1} ^ {(\ell)} - \mathbf {W} _ {0} ^ {(\ell)} \right\| _ {\mathrm {F}} \leq s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \lambda_ {\ell} ^ {- 1} \| \mathbf {X} \| _ {\mathrm {F}} \frac {8 \left(1 - u ^ {2}\right)}{\alpha_ {0} ^ {2}} \frac {1 - u ^ {t + 1}}{1 - u} \sqrt {2 \mathcal {L} \left(\boldsymbol {\theta} _ {0}\right)}, \tag {30} \\ \underset {(a)} {\leq} \frac {1 6}{\alpha_ {0} ^ {2}} s ^ {L} \lambda_ {1 \to (L + 1)} \lambda_ {\ell} ^ {- 1} \| \mathbf {X} \| _ {\mathrm {F}} \sqrt {2 \mathcal {L} (\boldsymbol {\theta} _ {0})}, \\ \end{array}
$$

where $( a )$ is due to $u \in ( 0 , 1 )$ . Let $C _ { \ell }$ as a positive constant that

$$
C _ {\ell} \geq \frac {1 6}{\alpha_ {0} ^ {2}} s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \lambda_ {\ell} ^ {- 1} \| \mathbf {X} \| _ {\mathrm {F}} \sqrt {2 \mathcal {L} \left(\boldsymbol {\theta} _ {0}\right)}. \tag {31}
$$

By Wely’s inequality, we have

$$
\left\| \mathbf {W} _ {t + 1} ^ {(\ell)} \right\| _ {2} \leq \left\| \mathbf {W} _ {0} ^ {(\ell)} \right\| _ {2} + C _ {\ell} = \lambda_ {\ell}. \tag {32}
$$

To prove the second inequality holds in Eq. 27, we first upper bound $\| \mathbf { H } _ { t + 1 } ^ { ( L ) } - \mathbf { H } _ { 0 } ^ { ( L ) } \| _ { \mathrm { F } }$

$$
\begin{array}{l} \| \mathbf {H} _ {t + 1} ^ {(L)} - \mathbf {H} _ {0} ^ {(L)} \| _ {\mathrm {F}} \leq s ^ {L} \lambda_ {1 \to L} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 1} \| \mathbf {W} _ {t + 1} ^ {(\ell)} - \mathbf {W} _ {0} ^ {(\ell)} \| _ {2} \\ \leq_ {(b)} s ^ {L} \lambda_ {1 \rightarrow L} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 1} \left(\frac {1 6}{\alpha_ {0} ^ {2}} s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \lambda_ {\ell} ^ {- 1} \| \mathbf {X} \| _ {\mathrm {F}} \sqrt {2 \mathcal {L} \left(\boldsymbol {\theta} _ {0}\right)}\right) \tag {33} \\ = \frac {1 6}{\alpha_ {0} ^ {2}} s ^ {2 L} \lambda_ {1 \rightarrow L} \lambda_ {1 \rightarrow (L + 1)} \| \mathbf {X} \| _ {\mathrm {F}} ^ {2} \sqrt {2 \mathcal {L} (\boldsymbol {\theta} _ {0})} \left(\sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 2}\right) \\ \stackrel {<  } {_ {(c)}} \frac {\alpha_ {0}}{2}, \\ \end{array}
$$

where $( a )$ is due to Lemma 2, (b) is due to Eq. 30, and (c) is due to the second condition in Eq. 24.

By Weyl’s inequality, this implies $\begin{array} { r } { | \lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { t + 1 } ^ { ( L ) } ) - \lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { 0 } ^ { ( L ) } ) | = | \lambda _ { \operatorname* { m i n } } ( \mathbf { H } _ { t + 1 } ^ { ( L ) } ) - \alpha _ { 0 } | \leq \frac { \alpha _ { 0 } } { 2 } } \end{array}$ and

$$
\lambda_ {\min } \left(\mathbf {H} _ {t + 1} ^ {(L)}\right) \geq \frac {\alpha_ {0}}{2}. \tag {34}
$$

Let G = H(L)t W(L+1)t+1 , t $\mathbf { G } = \mathbf { H } _ { t } ^ { ( L ) } \mathbf { W } _ { t + 1 } ^ { ( L + 1 ) }$ hen we have

$$
\begin{array}{l} 2 \mathcal {L} (\boldsymbol {\theta} _ {t + 1}) = \| \widehat {\mathbf {Y}} _ {t + 1} - \mathbf {Y} \| _ {\mathrm {F}} ^ {2} \\ = \left\| \widehat {\mathbf {Y}} _ {t + 1} - \widehat {\mathbf {Y}} _ {t} + \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \right\| _ {\mathrm {F}} ^ {2} \\ = \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} ^ {2} + \| \widehat {\mathbf {Y}} _ {t + 1} - \widehat {\mathbf {Y}} _ {t} \| _ {\mathrm {F}} ^ {2} + 2 \langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \widehat {\mathbf {Y}} _ {t + 1} - \widehat {\mathbf {Y}} _ {t} \rangle_ {\mathrm {F}} \\ = 2 \mathcal {L} (\boldsymbol {\theta} _ {t}) + \| \widehat {\mathbf {Y}} _ {t + 1} - \widehat {\mathbf {Y}} _ {t} \| _ {\mathrm {F}} ^ {2} + 2 \langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \widehat {\mathbf {Y}} _ {t + 1} - \mathbf {G} \rangle_ {\mathrm {F}} + 2 \langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \mathbf {G} - \widehat {\mathbf {Y}} _ {t} \rangle_ {\mathrm {F}}. \tag {35} \\ \end{array}
$$

We can upper bound $\| \widehat { \mathbf Y } _ { t + 1 } - \widehat { \mathbf Y } _ { t } \| _ { \mathrm F } ^ { 2 }$ in Eq. 35 by

$$
\begin{array}{l} \left\| \widehat {\mathbf {Y}} _ {t + 1} - \widehat {\mathbf {Y}} _ {t} \right\| _ {\mathrm {F}} = \left\| \mathbf {H} _ {t + 1} ^ {(L)} \mathbf {W} _ {t + 1} ^ {(L + 1)} - \mathbf {H} _ {t} ^ {(L)} \mathbf {W} _ {t} ^ {(L + 1)} \right\| _ {\mathrm {F}} \\ = \| \mathbf {H} _ {t + 1} ^ {(L)} \left(\mathbf {W} _ {t + 1} ^ {(L + 1)} - \mathbf {W} _ {t} ^ {(L + 1)}\right) + \left(\mathbf {H} _ {t + 1} ^ {(L)} - \mathbf {H} _ {t} ^ {(L)}\right) \mathbf {W} _ {t} ^ {(L + 1)} \| _ {\mathrm {F}} \\ \leq \| \mathbf {H} _ {t + 1} ^ {(L)} \| _ {\mathrm {F}} \| \mathbf {W} _ {t + 1} ^ {(L + 1)} - \mathbf {W} _ {t} ^ {(L + 1)} \| _ {2} + \| \mathbf {H} _ {t + 1} ^ {(L)} - \mathbf {H} _ {t} ^ {(L)} \| _ {\mathrm {F}} \| \mathbf {W} _ {t} ^ {(L + 1)} \| _ {2} \\ \leq_ {(a)} s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 1} \| \mathbf {W} _ {t + 1} ^ {(\ell)} - \mathbf {W} _ {t} ^ {(\ell)} \| _ {2} + s ^ {L} \lambda_ {1 \rightarrow L} \| \mathbf {X} \| _ {\mathrm {F}} \| \mathbf {W} _ {t + 1} ^ {(L + 1)} - \mathbf {W} _ {t} ^ {(L + 1)} \| _ {2} \\ = s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {\ell = 1} ^ {L + 1} \lambda_ {\ell} ^ {- 1} \| \mathbf {W} _ {t + 1} ^ {(\ell)} - \mathbf {W} _ {t} ^ {(\ell)} \| _ {2} \\ \leq_ {(b)} \eta s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {\ell = 1} ^ {L + 1} \lambda_ {\ell} ^ {- 1} \left(\frac {s ^ {L} \lambda_ {1 \rightarrow (L + 1)}}{\lambda_ {\ell}} \| \mathbf {X} \| _ {\mathrm {F}} \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}}\right) \\ = \eta s ^ {2 L} \lambda_ {1 \rightarrow (L + 1)} ^ {2} \| \mathbf {X} \| _ {\mathrm {F}} ^ {2} \left(\sum_ {\ell = 1} ^ {L + 1} \lambda_ {\ell} ^ {- 2}\right) \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}}, \tag {36} \\ \end{array}
$$

where $( a )$ is due to Eq. 18 and Lemma 2, and $( b )$ follows from Lemma 3.

We can upper bound $\langle \widehat { \mathbf { Y } } _ { t } - \mathbf { Y } , \widehat { \mathbf { Y } } _ { t + 1 } - \mathbf { G } \rangle _ { \mathrm { F } }$ in Eq. 35 by

$$
\begin{array}{l} \langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \widehat {\mathbf {Y}} _ {t + 1} - \mathbf {G} \rangle_ {\mathrm {F}} \\ \leq \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} \| \widehat {\mathbf {Y}} _ {t + 1} - \mathbf {G} \| _ {\mathrm {F}} \\ = \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} \| \left(\mathbf {H} _ {t + 1} ^ {(L)} - \mathbf {H} _ {t} ^ {(L)}\right) \mathbf {W} _ {t + 1} ^ {(L + 1)} \| _ {\mathrm {F}} \\ \leq_ {(a)} \lambda_ {L + 1} \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} \left(s ^ {L} \lambda_ {1 \rightarrow L} \| \mathbf {X} \| _ {\mathrm {F}} \sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 1} \| \mathbf {W} _ {t + 1} ^ {(\ell)} - \mathbf {W} _ {t} ^ {(\ell)} \| _ {2}\right) \\ = s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} \| \mathbf {X} \| _ {\mathrm {F}} \left(\sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 1} \| \mathbf {W} _ {t + 1} ^ {(\ell)} - \mathbf {W} _ {t} ^ {(\ell)} \| _ {2}\right) \tag {37} \\ \leq_ {(b)} \eta s ^ {L} \lambda_ {1 \rightarrow (L + 1)} \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} \| \mathbf {X} \| _ {\mathrm {F}} \left(\sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 1} \frac {s ^ {L} \lambda_ {1 \rightarrow (L + 1)}}{\lambda_ {\ell}} \| \mathbf {X} \| _ {\mathrm {F}} \| \widehat {\mathbf {Y}} - \mathbf {Y} \| _ {\mathrm {F}}\right) \\ = \eta s ^ {2 L} \lambda_ {1 \rightarrow (L + 1)} ^ {2} \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} ^ {2} \| \mathbf {X} \| _ {\mathrm {F}} ^ {2} \left(\sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {- 2}\right), \\ \end{array}
$$

where $( a )$ is due to Lemma 2, and (b) follows from Lemma 3.

We can upper bound $\langle \widehat { \mathbf Y } _ { t } - \mathbf Y , \mathbf G - \widehat { \mathbf Y } _ { t } \rangle _ { \mathrm F }$ by

$$
\begin{array}{l} \langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \mathbf {G} - \widehat {\mathbf {Y}} _ {t} \rangle_ {\mathrm {F}} = \langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \mathbf {H} _ {t} ^ {(L)} (\mathbf {W} _ {t + 1} ^ {(L + 1)} - \mathbf {W} _ {t} ^ {(L + 1)}) \rangle_ {\mathrm {F}} \\ = \eta \left\langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \mathbf {H} _ {t} ^ {(L)} \frac {\partial \mathcal {L} (\boldsymbol {\theta})}{\partial \mathbf {W} _ {t} ^ {(L + 1)}} \right\rangle_ {\mathrm {F}} \\ = - \eta \left\langle \widehat {\mathbf {Y}} _ {t} - \mathbf {Y}, \mathbf {H} _ {t} ^ {(L)} \left((\mathbf {H} _ {t} ^ {(L)}) ^ {\top} (\widehat {\mathbf {Y}} _ {t} - \mathbf {Y})\right) \right\rangle_ {\mathrm {F}} \\ \underset {(a)} {=} - \eta \operatorname {t r} \left((\widehat {\mathbf {Y}} _ {t} - \mathbf {Y}) ^ {\top} \mathbf {H} _ {t} ^ {(L)} (\mathbf {H} _ {t} ^ {(L)}) ^ {\top} (\widehat {\mathbf {Y}} _ {t} - \mathbf {Y})\right) \\ = _ {(b)} - \eta \operatorname {t r} \left(\mathbf {H} _ {t} ^ {(L)} \left(\mathbf {H} _ {t} ^ {(L)}\right) ^ {\top} \left(\widehat {\mathbf {Y}} _ {t} - \mathbf {Y}\right) \left(\widehat {\mathbf {Y}} _ {t} - \mathbf {Y}\right) ^ {\top}\right) \tag {38} \\ \stackrel {\leq} {_ {(c)}} - \eta \lambda_ {\min } \left(\mathbf {H} _ {t} ^ {(L)} (\mathbf {H} _ {t} ^ {(L)}) ^ {\top}\right) \operatorname {t r} \left((\widehat {\mathbf {Y}} _ {t} - \mathbf {Y}) (\widehat {\mathbf {Y}} _ {t} - \mathbf {Y}) ^ {\top}\right) \\ \underset {(d)} {=} - \eta \lambda_ {\min} \left(\mathbf {H} _ {t} ^ {(L)} \left(\mathbf {H} _ {t} ^ {(L)}\right) ^ {\top}\right) \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} ^ {2} \\ \underset {(e)} {=} - \eta \lambda_ {\min} ^ {2} \left(\mathbf {H} _ {t} ^ {(L)}\right) \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} ^ {2}, \\ \underset {(f)} {\leq} - \eta \frac {\alpha_ {0} ^ {2}}{4} \| \widehat {\mathbf {Y}} _ {t} - \mathbf {Y} \| _ {\mathrm {F}} ^ {2}, \\ \end{array}
$$

where $( a )$ is due to $\langle \mathbf { A } , \mathbf { B } \rangle = \mathrm { t r } ( \mathbf { A } ^ { \top } \mathbf { B } )$ , $( b )$ is because of the cyclic property of the trace, $( c )$ is due to $\operatorname { t r } ( \mathbf { A } \mathbf { B } ) \geq \lambda _ { \operatorname* { m i n } } ( \mathbf { A } ) \mathrm { t r } ( \mathbf { B } )$ for any symmetric matrices $\mathbf { A } , \mathbf { B } \in \mathbb { S }$ and $\mathbf { B }$ is positive semi-definite, $( d )$ is due to $\mathrm { t r } ( \mathbf { A A } ^ { \top } ) = \| \mathbf { A } \| _ { \mathrm { F } } ^ { 2 }$ , (e) requires the $d _ { L } \geq N$ assumption8, and $( f )$ is due to Eq. 34.

Let $A = s ^ { 2 L } \lambda _ { 1  ( L + 1 ) } ^ { 2 } \Vert \mathbf { X } \Vert _ { \mathrm { F } } ^ { 2 } ( \sum _ { \ell = 1 } ^ { L + 1 } \lambda _ { \ell } ^ { - 2 } )$ and $B = s ^ { 2 L } \lambda _ { 1  ( L + 1 ) } ^ { 2 } \Vert \mathbf { X } \Vert _ { \mathrm { F } } ^ { 2 } ( \sum _ { \ell = 1 } ^ { L } \lambda _ { \ell } ^ { - 2 } )$ , we have

$$
\begin{array}{l} \mathcal {L} \left(\boldsymbol {\theta} _ {t + 1}\right) \leq \left(1 - \eta \frac {\alpha_ {0} ^ {2}}{4} + \eta^ {2} A ^ {2} + \eta B\right) \mathcal {L} \left(\boldsymbol {\theta} _ {t}\right) \\ \leq (a) \left(1 - \eta \left(\frac {\alpha_ {0} ^ {2}}{4} - 2 B\right)\right) \mathcal {L} (\boldsymbol {\theta} _ {t}) \tag {39} \\ \underset {(b)} {\leq} \left(1 - \eta \frac {\alpha_ {0} ^ {2}}{8}\right) \mathcal {L} (\boldsymbol {\theta} _ {t}), \\ \end{array}
$$

where $( a )$ requires choosing $\eta$ such that $\eta \le B / A ^ { 2 }$ , and $( b )$ holds due to the first condition in Eq. 24. Therefore, we have

$$
\mathcal {L} \left(\boldsymbol {\theta} _ {T}\right) \leq \left(1 - \eta \frac {\alpha_ {0} ^ {2}}{8}\right) ^ {T} \mathcal {L} \left(\boldsymbol {\theta} _ {0}\right) \leq \exp \left(- \eta T \frac {\alpha_ {0} ^ {2}}{8}\right) \mathcal {L} \left(\boldsymbol {\theta} _ {0}\right) = \epsilon . \tag {40}
$$

By taking log on both side, we have

$$
T \geq \frac {8}{\eta \alpha_ {0} ^ {2}} \log \left(\frac {\mathcal {L} \left(\boldsymbol {\theta} _ {0}\right)}{\epsilon}\right). \tag {41}
$$

![](images/3589f20902f6c05e012e6bfe7ce757bedb9aedd1f501f379c77576ce34206f42.jpg)

# E Additional empirical results

In this section, we report additional empirical results to illustrate the correctness of our theoretical analysis and the benefits of decoupled GCN structure.

# E.1 Comparison of generalization error on the real-world datasets.

We empirically compare the generalization error of different GCN structures on the real-world datasets, where the generalization error is measured by the difference between validation loss and training loss.

Setups. We select hidden dimension as 64, $\alpha _ { \ell } = 0 . 9$ for APPNP and GCNII, $\begin{array} { r } { \beta _ { \ell } \approx \frac { 0 . 5 } { \ell } } \end{array}$ for GCNII, and without any weight decay ( $\ell _ { 2 }$ regularization on weight matrices) or dropout operators. Note that although in our theoretical analysis we suppose $\beta _ { \ell }$ is a constant that does not change with respect to the layers, in this experiment we follow the configuration in [6] by selecting $\begin{array} { r } { \beta _ { \ell } = \log ( \frac { 0 . 5 } { \ell } + 1 ) } \end{array}$ , which guarantees a small generalization error on most small scale datasets. The same setup is also used for the results on synthetic dataset in Figure 4.

Results. As shown in Figure 11 and Figure 12, ResGCN has the largest generalization gap due to the skip-connections, APPNP has the smallest generalization error by removing the weight matrices in each individual layers. On the other hand, GCNII achieves a good balance between GCN and APPNP by balancing the expressive and generalization power. Finally, DGCN enjoys a small generalization error by using the decoupled GCN structure.

![](images/db1afefeea76211ce4dadd9160f6b9656ea1bd0faf16d1feef8ae1cf57c67ad5.jpg)

![](images/58b63aeed41c0c7b5fc0c09179ebea859beba9b60a4f84624f4596e1734e1f41.jpg)

![](images/d9a7067170a7a7e958c7332600bc20185c6a28bb6d77ef81823ba733831005e8.jpg)

![](images/41dc0d80c4704d0650c3cf6ae982129b670ee07a061158355d8a5a416e540977.jpg)

![](images/1735e05b5c889e028489354341abcb99fcb08395731e837f151a5c09cb56dc35.jpg)

![](images/521fbec5a4e01a0683e14d251beb874ac00aa265c33f139651bce926c0d7fae3.jpg)  
Figure 11: Comparison of generalization error and training F1-score on the Cora dataset. The curve corastops early at the largest training accuracy iteration.

# E.2 Effect of hyper-parameters in GCNII.

In the following, we first compare the effect of $\alpha _ { \ell }$ on the generalization error of GCNII. We select the hidden dimension as 64, and no dropout or weight decay is used for the training. As shown in Figure 13 and Figure 14, increasing $\alpha _ { \ell }$ leads to a smaller generalization error but a slower convergence speed, i.e., GCNII trades off the expressiveness for generalization power by increasing $\alpha _ { \ell }$ from 0 to 1. In practice, $\alpha _ { \ell } = 0 . 9$ is utilized in GCNII [6] for empirical evaluations. 9

Then, we compare the effect of $\beta _ { \ell }$ on the generalization error of GCNII. Similar to the previous experiments, we choose hidden dimension as 64, without applying dropout or weight decay during training. As shown in Figure 15 and Figure 16, decreasing $\beta _ { \ell }$ leads to a smaller generalization error. In practice, $\begin{array} { r } { \beta _ { \ell } = \log ( \frac { 0 . 5 } { \ell } + 1 ) } \end{array}$ is utilized in GCNII [6] for empirical evaluations, which guarantees a small generalization error.

![](images/406db5c532c9d60cdacc2c28bab155662a7d3932101983ece08c9702c3fe212e.jpg)

![](images/db461b7f1ce764d585cb0a26ee854464796e6175b97abfb264fb36931193c75e.jpg)

![](images/496f85f1e136df56cfb0fb6a192baea3350da65a1525053950e07cab4aecf80c.jpg)

![](images/b38dbb4ad8b0846004600d5863b2282b38466c018dac9f42832fbcd1c64c5bdd.jpg)

![](images/acc256d302949cdb998b89673e557286b62f0fcd593f79da16dc1ccb5aa1eee2.jpg)

![](images/6763bb29ca6936954f135dc93095c278d127cedc51dd563696fc36b9054e925a.jpg)  
Figure 12: Comparison of generalization error and training F1-score on the Citeseer dataset. The Citeseercurve stops early at the largest training accuracy iteration.

![](images/03741f6fa04fef66f981ad71c67fa362dda0a037e4bf0e4dd03b09de567171e9.jpg)

![](images/4fd343f1174c741a3ec8559c1ba384d69269055d80dd4bb63443dbb74ec54142.jpg)

![](images/d03076a596493968d40700bf239440b634d332359c67436ea16ff1c19b166f0d.jpg)

![](images/14ad7e5db359c572a6bf77dd8b694157615aef50df39d5c862bb37bef5918d08.jpg)

![](images/49cd4b3b4e5b9d99ee297a08d403027a56dd967efae16c1acc18c6f3804e90f1.jpg)

![](images/5a82cc39bd22c26df1a109444fe86462b4a5b37e70c7fa85ab1f2982a45864b6.jpg)  
Figure 13: Comparison of $\alpha _ { \ell }$ on the generalization error on Cora dataset. The curve stops early at the largest training accuracy iteration.

# E.3 Effect of DropEdge on generalization

In the following, we explore the effect of node embedding augmentation technique DropEdge [46] on the generalization of GCNs. Recall that DropEdge proposes to randomly drop a certain rate of edges in the input graph at each iteration and compute node embedding based on the sparser graph. The forward propagation rule of this technique can be formulated as $\bar { \mathbf { H } } ^ { ( \ell - 1 ) } = \sigma ( \widetilde { \mathbf { L } } \bar { \mathbf { H } } ^ { ( \ell - 1 ) } \bar { \mathbf { W } } ^ { ( \ell ) } ) )$ where $\widetilde { \mathbf { L } }$ is constructed by the adjacency matrix of the sparser graph with $\mathrm { s u p p } ( \widetilde { \mathbf { L } } ) \ll \mathrm { s u p p } ( \mathbf { L } )$ .

Again, we choose hidden dimension as 64, without applying dropout or weight decay during training. As shown in Figure 17 and Figure 18, DropEdge reduces the generalization error by restricting the number of nodes used during training. Besides, we observe that both training accuracy and generalization error decrease when the fraction of remaining edges in the graph decreases, which implies that edge dropping is impacting the generalization of GCN rather than the discriminativeness of node representations.

# E.4 Effect of PairNorm on generalization

In the following, we explore the effect of node embedding augmentation technique PairNorm [60] on the generalization of GCNs. PairNorm proposes to normalized node embeddings by $\mathbf { H } ^ { ( \ell ) } =$

![](images/8922b9582f1f74d4b6d10d10d7734ab8033293e8032ceae05626b66b69a128c3.jpg)

![](images/fd647a29558aa72e7215a35d9bad861a4fa6840b05797404f6a4a3fce26af06b.jpg)

![](images/6a2e28b2341fa4ff159a7b59a6020ec91367a0535d56c9119937e3b270c31129.jpg)

![](images/0456aa5ff127edcf2f266c21938e039fd14b4644097f7056ec85c951891d9034.jpg)

![](images/9bcabeb4ad5218a1be8225f7e8c611b0ab989c15543bc642a480f0d54207d853.jpg)

![](images/0a8220394ff67e3756b6b38355f18a3c32bdab2165fc994ebd26416a16c4fbaa.jpg)  
Figure 14: Comparison of $\alpha _ { \ell }$ on the generalization error on Citeseer dataset. The curve stops early at the largest training accuracy iteration.

![](images/81320ae96ed6a149bd107099a5b746a5c8ed6d2051e618962b4e95a32374a050.jpg)

![](images/3588b22d2bbfaeba95f0c9d0627558fdf734ad43faba6347d4b96d5124b2add3.jpg)

![](images/77efa1217599c806cdd719617f12c889213fc72ce46e5ee6c2a31fd7362e1aa3.jpg)

![](images/9615369deae3ed65e0a38210cf23d715b3fe2da7f782b80577b6c2b72a9cbe5d.jpg)

![](images/439454c8d04bd160bb6bc769cc810c4d8b0cca7131947c997ea99a6e44f72a8a.jpg)

![](images/6c8a51602bcdf0658b1c0d586591762f92c34019b2aca8f616ed6fc08b352750.jpg)  
Figure 15: Comparison of $\beta _ { \ell }$ on the generalization error on Cora dataset. The curve stops early at the largest training accuracy iteration.

$\begin{array} { r } { \mathrm { P N } ( \mathbf { H } ^ { ( \ell ) } ) ~ = ~ \gamma \frac { \mathbf { H } ^ { ( \ell ) } - \pmb { \mu } ( \mathbf { H } ^ { ( \ell ) } ) } { \sigma ( \mathbf { H } ^ { ( \ell ) } ) } } \end{array}$ where the average node embeddinof node embeddings is computed as $\mu ( \mathbf { H } ^ { ( \ell ) } ) \ =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf h } _ { i } ^ { ( \ell ) } } \end{array}$ $\begin{array} { r } { \sigma ^ { 2 } ( { \bf H } ^ { ( \ell ) } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | | { \bf h } _ { i } ^ { ( \ell ) } - \frac { \ d H _ { i } } { \ d N } | | ^ { 2 } . } \end{array}$ $\mu ( \mathbf { H } ^ { ( \ell ) } ) \lVert _ { 2 } ^ { 2 }$ , and $\gamma \geq 0$ controls the scale of node embeddings.

We choose hidden dimension as 64, and no dropout or weight decay is used during training. As shown in Figure 19 and Figure 20, a larger scale ratio $\gamma$ can improve the discriminativeness of node embeddings, but will hurt the generalization error. A smaller scale ratio leads to a small generalization error, but it makes the node embeddings harder to discriminate, therefore over-smoothing happens (e.g., using $\gamma = 0 . 1$ can be think of as creating over-smoothing effect on node embeddings).

# E.5 Illustrating the gradient instability issue during GCN training

Gradient instability refers to a phenomenon that the gradient changes significantly at every iteration. The main reason for gradient instability is because the scale of weight matrices is large, which causes the calculated gradients become large. Notice that the gradient instability issue is more significant on vanilla GCN than other sequential GCN structures such as ResGCN and GCNII, which is one of the key factors that impacts the training phase of vanilla GCNs.

![](images/85ff6b741281d0d965119c346586b5aacdf13a039e7341ab7d09a78cc04aaa53.jpg)

![](images/1a7b57d30dca6586092ccdf2c0cc1576d7963c630f8ddd25d52264157f6ec382.jpg)

![](images/9a0e4cc9b09353203b281cc762c73692b2eda7bfa391de52eb3dd8fb229a87f5.jpg)

![](images/6be1ed498e7a29ba4d3dbd232c0d2334c8feda13a44a7bb3c3e082761732c0a5.jpg)

![](images/32d50e4cd360151b9c56b547a1f46e73ed8bb97012e4c5873890fbd5d4922e59.jpg)

![](images/12944c78c34b31fa09765ec283e95413fff23686f03d9ea922c56e16d57764ef.jpg)

![](images/517c618ab5235d7f88f987759e883e7a083208e900766caa96fe13b333d66aae.jpg)  
Figure 16: Comparison of $\beta _ { \ell }$ on the generalization error on Citeseer dataset. The curve stops early at the largest training accuracy iteration.

![](images/2471bec5c03fd79a4acff842098f8cd53af34cc1087211fb19f104b5fd627e7c.jpg)

![](images/20c603a7054f90048f3e0e4d46d6ddbf6919f03cdaeb65ac02f731893cf77625.jpg)

![](images/2449730523e422f6b8eb7d77f622361e7711205e27ca4a169cbd4e2315332b2d.jpg)

![](images/848c6461bd47240ec3afaed6c9f5e58a5b5438d8cd37faa450bfd836f1b9d890.jpg)

![](images/9234e3e75299374e3a0dae8aa8dcce3a9b8cad3a44f10618e146a3e89374b401.jpg)  
Figure 17: Comparison of generalization error of DropEdge on Cora dataset. The curve stops early at the largest training accuracy iteration.

To see this, let $\mathbf { W } ^ { ( \ell ) }  \mathbf { W } ^ { ( \ell ) } - \eta \mathbf { G } ^ { ( \ell ) }$ denote the gradient descent update of the `th layer weight matrix, where $\mathbf { G } ^ { ( \ell ) }$ is the gradient with respect to the `th layer weight matrix $\mathbf { W } ^ { ( \ell ) }$ . The upper bounds of the gradient for GCN, ResGCN, and GCNII are computed as

$$
\left\| \mathbf {G} ^ {(\ell)} \right\| _ {2} = \mathcal {O} \left(\left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L}\right), \quad \text {G C N (E q . 7 1)}
$$

$$
\left\| \mathbf {G} ^ {(\ell)} \right\| _ {2} = \mathcal {O} \left(\left(1 + \sqrt {d} B _ {w}\right) ^ {L}\right), \quad \text {R e s G C N (E q . 1 0 5)} \tag {42}
$$

$$
\| \mathbf {G} ^ {(\ell)} \| _ {2} = \mathcal {O} \left(\beta \left(\max  \{1, \alpha \sqrt {d} B _ {w} \}\right) ^ {L}\right), \quad \text {G C N I I (E q . 1 6 0)}
$$

where $d$ is the largest number of node degree, $L$ is the number of layers, and $\| \mathbf { W } ^ { ( \ell ) } \| _ { 2 } \le B _ { w }$ is the largest singular value of weight matrix. Details please refer to the derivative of Eq. 71, Eq. 105 , and Eq. 160.

From Eq. 42, we know that the largest singular value of weight matrices is the key factor that affects the scale of the gradient. However, upper bound can be vacuous if we simply ignore the impact of network structure on $B _ { w }$ .

From Figure 21 and Figure 22, we can observe that the residual connection has implicit regularization on the weight matrices, which makes the weight matrices in ResGCN has a smaller largest singular values than GCN. As a result, the ResGCN does not suffer from gradient instability even its gradient

![](images/99eb1c85b73de6257052c0af1aafe4a5b469e5e8a5188d152dafe7b46add7c3b.jpg)

![](images/abd718ca36681da45d229dbf1839bc8d03dc746abafbe3441a4386b34f215b59.jpg)

![](images/e262e43367f33e417eeab20ff70db96e916a14e2af856aad17be82bd5a6951a0.jpg)

![](images/4f1c2cd20b48aac5b17a6d899659b6ba096badff7ea9429365e1abced6327c9e.jpg)

![](images/f8bb6c3edc6c68643686a3c7fe03b64bacb2de34f9ef4bc8c7d62cc20cb7d8eb.jpg)

![](images/a9cd136cdccf3902fb85aa270e51e2513731525477dd13cc6ab90f6af3d8e6c6.jpg)  
Figure 18: Comparison of generalization error of DropEdge on Citeseer dataset. The curve stops early at the largest training accuracy iteration.

![](images/71cbdf5b4d745358c710a863a44828367c2c7a9748527e774f572649daee5f53.jpg)

![](images/832d5335f856223d65cba4067c5c0ff39aee7e0a2034abd80859b2d6cdf0e152.jpg)

![](images/5c0813929fa62286444375a78027b69e21cdc55f10882f2d5e46c1cfe911c46c.jpg)

![](images/07b483fa58a8f978cdfe51e9bc6453a3c6b7db0edc0861c3321a38ff08d78a0d.jpg)

![](images/b6c3ee409768900eae6a5f104637b4af0cabfb4d9f5cb5799fa589455dcd61d5.jpg)

![](images/e2847c0d796054a42987fbe2a8975b2b4d5b66a3362a090301a2dae956953b9d.jpg)  
Figure 19: Comparison of generalization error of PairNorm on Cora dataset. The curve stops early at the largest training accuracy iteration.

norm upper bound in Eq. 42 is larger than GCN. Furthermore, although the largest singular value of the weight matrices for GCNII is larger than GCN, by selecting a small enough $\beta _ { \ell }$ , GCNII can be less impacted by gradient instability than vanilla GCN.

# E.6 Illustrating how more training leads to high training F1-score

As a compliment to Figure 1, we provide training and validation F1-score of the baseline models. During training, we chose hidden dimension as 64, Adam optimizer with learning rate 0.001, without any dropout or weight decay. Please note that removing dropout and weight decay is necessary because both operations are designed to prevent neural networks from overfitting, and will hurt the best training accuracy that a model can achieve. As shown in Figure 23 and Figure 24, all methods can achieve high training F1-score regardless the number of layers, which indicates node embeddings are distinguishable.

# E.7 Effect of number of layers on real-word datasets

In the following, we demonstrate the effect of the number of layers and hyper-parameters on the performance of the model on OGB Arxiv [23]. We follow the default hyper-parameter setup of GCN

![](images/6e9dc5bbb42415765150c4ba9701722a5d17f16aa9213867c64b8738b419ebf6.jpg)

![](images/706ffad62068fdaf96c40dd66eddd14259eca79e9215884e015fd37f8f40f572.jpg)

![](images/a71004d2e353cca47eaf715d809fdc45134738fa0cca79b2e3ba0b65f0e52426.jpg)

![](images/576e96a649df7aca033795fc053c763bdc3a00b9daa82a4200a11192696a80f1.jpg)

![](images/543debc0d45397a7496733bbf4b228ff52353433f6881ea19cc10d0495b81c2f.jpg)

![](images/5e4c435066b361fae878ef73299ad6a1fc80e708b916d7f60df59cf7c0f076fc.jpg)

![](images/c9748972fa18f49dd2657d1d53568bca68d5a3539a91b61f41d2b7cf68f782bb.jpg)  
Figure 20: Comparison of generalization error of PairNorm on Citeseer dataset. The curve stops early at the largest training accuracy iteration.

![](images/2b354ecbc78de29a0457b6f6e710990eb4c3a443b46eaea85105c4e3b57abcde.jpg)

![](images/e4bfc583532afb796f02f6b0bf63d6c3b8d8d1a3c39465294f93d032d341ed8b.jpg)

![](images/bd13bdcc7f2c676eb25c109547969cc93f373507a65531cb0e785351882f3ded.jpg)

![](images/4144425e917993e2fbcdb508003977e4534c7447342fb13d69ada909aa4867d0.jpg)

![](images/af0c1b0c64034322c5a93a8ce10b61610c77993f4cee8d0e5951ef5b648ed379.jpg)  
Figure 21: Comparison of gradient norm on Cora dataset. The curve stops early at the largest training accuracy iteration.

on the leaderboard,10 i.e., we choose hidden dimension as 128, dropout ratio as 0.5, Adam optimizer with learning rate as 0.01, and applying batch normalization after each graph convolutional layer. As shown in Table 3, the number of layers and the choice of hyper-parameters can largely impact the performance of the models. Since DGCN can automatically adjust the $\alpha _ { \ell }$ and $\beta _ { \ell }$ to better adapt to the change of model depth, it achieves a comparable and more stable performance than most baseline models.

# F Generalization bound for GCN

In this section, we provide detailed proof on the generalization bound of GCN. Recall that the update rule of GCN is defined as

$$
\mathbf {H} ^ {(\ell)} = \sigma \left(\mathbf {L} \mathbf {H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}\right), \tag {43}
$$

where $\sigma ( \cdot )$ is the ReLU activation function. Note that although ReLU function $\sigma ( x )$ is not differentiable when $x = 0$ , for analysis purpose we suppose the $\sigma ^ { \prime } ( 0 ) = 0$ .11

![](images/cb141fed2c8da51ef143b58579e3b7bc995a9e42f2dc89274335d1ea1cdeb815.jpg)

![](images/8b803046da9ac0bca5602539de4e21bb70e813b50ac28fe2cf7c632977a481e7.jpg)

![](images/413b0d5ad7d883ff3b1e2c6058bdddf841d13a8a762bb12e8ed1f38112a0edaa.jpg)

![](images/945d62e461429f29eaed9f84aeb46f14f5c2c500a45d365b8db3a61c1167b32e.jpg)

![](images/899950a5bb34b0bc198daec05000482ff39d0a5ebe9bc24dc26d34393e053737.jpg)

![](images/b7ac988579275140d8a3c08dd2e76129cd889ef7f03c6f5695becdcefaffd441.jpg)

![](images/ca72b793ceb2674e6c33652e2abb6fc0903e545e9ed491cc47b8ba61e406c3fc.jpg)  
Figure 22: Comparison of gradient norm on Citeseer dataset. The curve stops early at the largest training accuracy iteration.

![](images/929a7406919dc092934fa97446616ec744895f5393143262af13a2c975e4c65a.jpg)

![](images/f3674737f391e10f7f8dab91cd164b19e9a2cb0de839f8e1abddc558a68d4fef.jpg)

![](images/4f2545a91e227eb3dd1a130fddfc8874473000c29d48af8efcebff73a3c69588.jpg)  
Figure 23: Comparison of training F1-score and number of iterations on Cora dataset.

The training of GCN is an empirical risk minimization with respect to a set of parameters $\theta =$ $\{ \mathbf { W } ^ { ( 1 ) } , \ldots , \mathbf { W } ^ { ( L ) } , \mathbf { v } \}$ , i.e.,

$$
\mathcal {L} (\boldsymbol {\theta}) = \frac {1}{m} \sum_ {i = 1} ^ {m} \Phi_ {\gamma} (- p \left(f \left(\mathbf {h} _ {i} ^ {(L)}\right), y _ {i}\right)), f \left(\mathbf {h} _ {i} ^ {(L)}\right) = \tilde {\sigma} \left(\mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}\right), \tag {44}
$$

where ${ \bf h } _ { i } ^ { ( L ) }$ is the node representation of the ith node at the final layer, $f ( \mathbf { h } _ { i } ^ { ( L ) } )$ is the predicted label for the ith node, $\begin{array} { r } { \tilde { \sigma } ( x ) = \frac { 1 } { \exp \left( - x \right) + 1 } } \end{array}$ is the sigmoid function, and loss function $\Phi _ { \gamma } ( - p ( z , y ) )$ is $\frac { 2 } { \gamma }$ -Lipschitz continuous with respect to its first input $z$ with $p ( z , y )$ as defined in Section 5. For simplification, we will use $\mathrm { L o s s } ( z , y )$ which represents $\Phi _ { \gamma } ( - p ( z , y ) )$ in the proof.

To establish the generalization of GCN as stated in Theorem 3, we utilize the following result on transductive uniform stability from [16].

Theorem 8 (Transductive uniform stability bound [16]). Let $f$ be a -uniformly stable transductive learner and $\gamma , \delta > 0$ , and define $Q = m u / ( m + u )$ . Then, with probability at least $1 - \delta$ over all

![](images/70554dd6026458fa97c70f4d529b8856865e00b13e9e7f88b1a8eb5856fdf333.jpg)

![](images/805e560e15989a65086588ec038c64a29b2cc128c0ab8bb79377c4b88fc16402.jpg)

![](images/0569dc0db06c5d8885263abe97e03ba0fde301b7c60189694b95b08a3c089ff1.jpg)

![](images/8c278c090d4999506a84cfaa24e91218411f0d18af19a7b937daa2a163e5b2fe.jpg)  
Figure 24: Comparison of training F1-score and number of iterations on Citeseer dataset.

Table 3: Comparison of F1-score on OGB-Arxiv dataset for different number of layers   

<table><tr><td>Model</td><td>α</td><td>2 Layers</td><td>4 Layers</td><td>8 Layers</td><td>12 Layers</td><td>16 Layers</td></tr><tr><td>GCN</td><td>-</td><td>71.02% ± 0.14</td><td>71.56% ± 0.19</td><td>71.28% ± 0.33</td><td>70.28% ± 0.23</td><td>69.37% ± 0.46</td></tr><tr><td>ResGCN</td><td>-</td><td>70.66% ± 0.48</td><td>72.41% ± 0.31</td><td>72.56% ± 0.31</td><td>72.46% ± 0.23</td><td>72.11% ± 0.28</td></tr><tr><td>GCNII</td><td>0.9</td><td>71.35% ± 0.21</td><td>72.57% ± 0.23</td><td>72.06% ± 0.42</td><td>71.31% ± 0.62</td><td>69.99% ± 0.80</td></tr><tr><td>GCNII</td><td>0.8</td><td>71.14% ± 0.27</td><td>72.32% ± 0.19</td><td>71.90% ± 0.41</td><td>71.21% ± 0.23</td><td>70.56% ± 0.72</td></tr><tr><td>GCNII</td><td>0.5</td><td>70.54% ± 0.30</td><td>72.09% ± 0.25</td><td>71.92% ± 0.32</td><td>71.24% ± 0.47</td><td>71.02% ± 0.58</td></tr><tr><td>APPNP</td><td>0.9</td><td>67.38% ± 0.34</td><td>68.02% ± 0.55</td><td>66.62% ± 0.48</td><td>67.43% ± 0.50</td><td>67.42% ± 1.00</td></tr><tr><td>APPNP</td><td>0.8</td><td>66.71% ± 0.32</td><td>68.25% ± 0.43</td><td>66.40% ± 0.89</td><td>66.51% ± 2.09</td><td>66.56% ± 0.74</td></tr><tr><td>DGCN</td><td>-</td><td>71.21% ± 0.25</td><td>72.29% ± 0.18</td><td>72.39% ± 0.21</td><td>72.63% ± 0.12</td><td>72.41% ± 0.07</td></tr></table>

training and testing partitions, we have

$$
\mathcal {R} _ {u} (f) \leq \mathcal {R} _ {m} ^ {\gamma} (f) + \frac {2}{\gamma} \mathcal {O} \Big (\epsilon \sqrt {Q \ln (\delta^ {- 1})} \Big) + \mathcal {O} \Big (\frac {\ln (\delta^ {- 1})}{\sqrt {Q}} \Big).
$$

Then, in Lemma 4, we derive the uniform stability constant for GCN, i.e., $\epsilon _ { \mathrm { G C N } }$

Lemma 4. The uniform stability constant for GCN is computed as where $\begin{array} { r } { \epsilon _ { G C I I } = \frac { 2 \eta \rho _ { f } G _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + \eta L _ { F } ) ^ { t - 1 } } \end{array}$ PTt=1(1+ηLF )t−1

$$
\rho_ {f} = C _ {1} ^ {L} C _ {2}, G _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2}, L _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} ((L + 2) C _ {1} ^ {L} C _ {2} + 2), \tag {45}
$$

$$
C _ {1} = \max  \{1, \sqrt {d} B _ {w} \}, C _ {2} = \sqrt {d} (1 + B _ {x}).
$$

By plugging the result in Lemma 4 back to Theorem 8, we establish the generalization bound for GCN.

The key idea of the proof is to decompose the change of the GCN output into two terms (in Lemma 5) which depend on

• (Lemma 6) The maximum change of node embeddings, i.e., $\Delta h _ { \mathrm { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { H } ^ { ( \ell ) } -$ $\tilde { \mathbf { H } } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 }$ ,   
• (Lemma 7) The maximum node embeddings, i.e., $h _ { \operatorname* { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { H } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 }$

Lemma 5. Let $f ( \mathbf { h } _ { i } ^ { ( L ) } ) = \tilde { \sigma } ( \mathbf { v } ^ { \top } \mathbf { h } _ { i } ^ { ( L ) } ) , \tilde { f } ( \tilde { \mathbf { h } } _ { i } ^ { ( L ) } ) = \tilde { \sigma } ( \tilde { \mathbf { v } } ^ { \top } \tilde { \mathbf { h } } _ { i } ^ { ( L ) } )$ denote the prediction of node i using parameters $\mathbf { \boldsymbol { \theta } } = \{ \mathbf { W } ^ { ( 1 ) } , \ldots , \mathbf { W } ^ { ( L ) } , \mathbf { v } \} , \tilde { \boldsymbol { \theta } } = \{ \tilde { \mathbf { W } } ^ { ( 1 ) } , \ldots , \tilde { \mathbf { W } } ^ { ( L ) } , \tilde { \mathbf { v } } \}$ } (i.e., the two set of parameters

trained on the original and the perturbed dataset) respectively. Then we have

$$
\max _ {i} | f (\mathbf {h} _ {i} ^ {(L)}) - \tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)}) | \leq \Delta h _ {\max} ^ {(L)} + h _ {\max} ^ {(L)} \| \Delta \mathbf {v} \| _ {2},
$$

$$
\left. \right. \max  _ {i} \left\| \frac {\partial f \left(\mathbf {h} _ {i} ^ {(L)}\right)}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial \tilde {f} \left(\tilde {\mathbf {h}} _ {i} ^ {(L)}\right)}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \leq \Delta h _ {\max } ^ {(L)} + \left(h _ {\max } ^ {(L)} + 1\right) \| \Delta \mathbf {v} \| _ {2}, \tag {46}
$$

where $\Delta \mathbf { v } = \mathbf { v } - \tilde { \mathbf { v } }$

Lemma 6 (Upper bound of h(`)max for GCN). Let suppose Assumption 1 hold. Then, the maximum $h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ node embeddings for any node at the `th layer is bounded by

$$
h _ {\max } ^ {(\ell)} \leq B _ {x} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {\ell}. \tag {47}
$$

Lemma 7 (Upper bound of $\Delta h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ for GCN). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta h _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {x} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {48}
$$

where $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) }$

Besides, in Lemma 8, we derive the upper bound on the maximum change of node embeddings before the activation function, i.e., $\begin{array} { r } { \Delta z _ { \mathrm { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { Z } ^ { ( \ell ) } - \tilde { \mathbf { Z } } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 } } \end{array}$ . $\Delta \bar { z } _ { \mathrm { m a x } } ^ { ( \ell ) }$ will be used in the computation of the gradient related upper bounds.

Lemma 8 (Upper bound of $\Delta z _ { \mathrm { m a x } } ^ { ( \ell ) }$ for GCN). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings before the activation function on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta z _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {x} \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {49}
$$

where $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) }$

Then, in Lemma 9, we decompose the change of the model parameters into two terms which depend on

• The maximum change of gradient passing from the $( \ell + 1 )$ th layer to the `th layer $\Delta d _ { \mathrm { m a x } } ^ { ( \ell ) } =$ $\begin{array} { r } { \operatorname* { m a x } _ { i } \left\| \left[ \frac { \partial \sigma ( \mathbf { L } \mathbf { H } ^ { ( \ell - 1 ) } \mathbf { \bar { W } } ^ { ( \ell ) } ) } { \partial \mathbf { H } ^ { ( \ell - 1 ) } } - \frac { \partial \sigma ( \mathbf { L } \hat { \tilde { \mathbf { H } } } ^ { ( \ell - 1 ) } \mathbf { \bar { W } } ^ { ( \ell ) } ) } { \partial \hat { \mathbf { H } } ^ { ( \ell - 1 ) } } \right] _ { i , : } \right\| _ { 2 } , } \end{array}$ i, :  2 ,   
• The maximum gradient passing from the $( \ell + 1 ) \mathrm { t h }$ layer to the `th layer $d _ { \operatorname* { m a x } } ^ { ( \ell ) } ~ =$ $\begin{array} { r } { \operatorname* { m a x } _ { i } \Big \| \left[ \frac { \partial \sigma ( \mathbf { L H } ^ { ( \ell - 1 ) } \mathbf { W } ^ { ( \ell ) } ) } { \partial \mathbf { H } ^ { ( \ell - 1 ) } } \right] _ { i , : } ^ { - } \Big \| _ { 2 } . } \end{array}$

Lemma 9 (Upper bound of $d _ { \operatorname* { m a x } } ^ { ( \ell ) } , \Delta d _ { \operatorname* { m a x } } ^ { ( \ell ) }$ for GCN). Let suppose Assumption 1 hold. Then, the upper bound on the maximum gradient passing from layer $\ell + 1$ to layer ` for any node is

$$
d _ {\max } ^ {(\ell)} \leq \frac {2}{\gamma} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell + 1}, \tag {50}
$$

and the upper bound on the maximum change between the gradient passing from layer $\ell + 1$ to layer ` on two different set of weight parameters for any node is

$$
\Delta d _ {\max } ^ {(\ell)} \leq \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell} \frac {2}{\gamma} \left((L + 1) \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {51}
$$

where $\begin{array} { r } { \| \Delta \pmb { \theta } \| _ { 2 } = \| \mathbf { v } - \tilde { \mathbf { v } } \| _ { 2 } + \sum _ { \ell = 1 } ^ { L } \| \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } \| _ { 2 } } \end{array}$ denotes the change of two set of parameters.

Finally, based on the previous result, in Lemma 10, we decompose the change of the model parameters into two terms which depend on

• The change of gradient with respect to the `th layer weight parameters $\| \Delta \mathbf { G } ^ { ( \ell ) } \| _ { 2 }$

• The gradient with respect to the `th layer weight parameters $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$ ,

where $\mathbf { G } ^ { ( L + 1 ) }$ denotes the gradient with respect to the weight $\mathbf { v }$ of the binary classifier and $\mathbf { G } ^ { ( \ell ) }$ denotes the gradient with respect to the weight $\mathbf { W } ^ { ( \ell ) }$ of the $\ell$ th graph convolutional layer. Notice that $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ reflects the smoothness of GCN model and $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$ corresponds to the upper bound of gradient.

Lemma 10 (Upper bound of √ $\| \mathbf G ^ { ( \ell ) } \| _ { 2 } , \| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ for GCN). Let suppose Assumption 1 hold and let $C _ { 1 } = \operatorname* { m a x } \lbrace 1 , \sqrt { d } B _ { w } \rbrace$ and $C _ { 2 } = \sqrt { d } ( B _ { x } + 1 )$ . Then, the gradient and the maximum change between gradients computed on two different set of weight parameters are bounded by

$$
\sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2},
$$

$$
\sum_ {\ell = 1} ^ {L + 1} \| \Delta \mathbf {G} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} \left((L + 2) C _ {1} ^ {L} C _ {2} + 2\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {52}
$$

where $\begin{array} { r } { \| \Delta \pmb { \theta } \| _ { 2 } = \| \mathbf { v } - \tilde { \mathbf { v } } \| _ { 2 } + \sum _ { \ell = 1 } ^ { L } \| \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } \| _ { 2 } } \end{array}$ denotes the change of two set of parameters.

Equipped with above intermediate results, we now proceed to prove Lemma 4.

Proof of Lemma 4. Recall that our goal is to explore the impact of different GCN structures on the uniform stability constant $\epsilon _ { \mathrm { G C N } }$ , which is a function of $\rho _ { f } , G _ { f }$ , and $L _ { f }$ . Let $C _ { 1 } = \operatorname* { m a x } \lbrace 1 , \sqrt { d } B _ { w } \rbrace$ and $C _ { 2 } = \sqrt { d } ( B _ { x } + 1 )$ . Firstly, by plugging Lemma 7 and Lemma 6 into Lemma 5, we have

$$
\begin{array}{l} \max _ {i} | f (\mathbf {h} _ {i} ^ {(L)}) - \tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)}) | \leq \Delta h _ {\max} ^ {(L)} + h _ {\max} ^ {(L)} \| \Delta \mathbf {v} \| _ {2} \\ \leq \sqrt {d} B _ {x} \cdot \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \tag {53} \\ \leq C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}. \\ \end{array}
$$

Therefore, we know that the function $f$ is $\rho _ { f }$ -Lipschitz continuous, with $\rho _ { f } = C _ { 1 } ^ { L } C _ { 2 }$ . Then, by Lemma 10, we know that the function $f$ is $L _ { f }$ -smoothness, and the gradient of each weight matrix is bounded by $G _ { f }$ , with

$$
G _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2}, L _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} ((L + 2) C _ {1} ^ {L} C _ {2} + 2). \tag {54}
$$

By plugging $\epsilon _ { \mathrm { G C N } }$ into Theorem 3, we obtain the generalization bound of GCN.

# F.1 Proof of Lemma 5

By the definition of $f ( \mathbf { h } _ { i } ^ { ( L ) } )$ and $\tilde { f } ( \tilde { \mathbf { h } } _ { i } ^ { ( L ) } )$ , we have

$$
\begin{array}{l} \max _ {i} | f (\mathbf {h} _ {i} ^ {(L)}) - \tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)}) | = \max _ {i} | \tilde {\sigma} (\mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}) - \tilde {\sigma} (\tilde {\mathbf {v}} ^ {\top} \tilde {\mathbf {h}} _ {i} ^ {(L)}) | \\ \leq \max  _ {i} \| \mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)} - \tilde {\mathbf {v}} ^ {\top} \tilde {\mathbf {h}} _ {i} ^ {(L)} \| _ {2} \tag {55} \\ \leq \max _ {i} \| \mathbf {v} ^ {\top} (\mathbf {h} _ {i} ^ {(L)} - \tilde {\mathbf {h}} _ {i} ^ {(L)}) \| _ {2} + \max _ {i} \| \tilde {\mathbf {h}} _ {i} ^ {(L)} (\mathbf {v} - \tilde {\mathbf {v}}) \| _ {2} \\ \leq \Delta h _ {\max } ^ {(L)} + h _ {\max } ^ {(L)} \Delta \mathbf {v}, \\ \end{array}
$$

where $( a )$ is due to the fact that sigmoid function is 1-Lipschitz continuous.

$$
\begin{array}{l} \max _ {i} \left\| \frac {\partial f (\mathbf {h} _ {i} ^ {(L)})}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial \tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)})}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} = \max _ {i} \| \tilde {\sigma} ^ {\prime} (\mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}) \mathbf {v} ^ {\top} - \tilde {\sigma} ^ {\prime} (\tilde {\mathbf {v}} ^ {\top} \tilde {\mathbf {h}} _ {i} ^ {(L)}) \tilde {\mathbf {v}} ^ {\top} \| _ {2} \\ \leq \max _ {i} \| \tilde {\sigma} ^ {\prime} (\mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}) \mathbf {v} ^ {\top} - \tilde {\sigma} ^ {\prime} (\mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}) \tilde {\mathbf {v}} ^ {\top} \| _ {2} \\ + \max  _ {i} \| \tilde {\sigma} ^ {\prime} \left(\mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}\right) \tilde {\mathbf {v}} ^ {\top} - \tilde {\sigma} ^ {\prime} \left(\tilde {\mathbf {v}} ^ {\top} \tilde {\mathbf {h}} _ {i} ^ {(L)}\right) \tilde {\mathbf {v}} ^ {\top} \| _ {2} \tag {56} \\ \underset {(a)} {\leq} \Delta \mathbf {v} + \left(\Delta h _ {\max } ^ {(L)} + h _ {\max } ^ {(L)} \Delta \mathbf {v}\right) \\ = \Delta h _ {\max } ^ {(L)} + \left(h _ {\max } ^ {(L)} + 1\right) \Delta \mathbf {v}, \\ \end{array}
$$

where $( a )$ is due to the fact that sigmoid function and its gradient are 1-Lipschitz continuous.

# F.2 Proof of Lemma 6

$h _ { \operatorname* { m a x } } ^ { ( \ell ) }$

$$
\begin{array}{l} h _ {\max } ^ {(\ell)} = \max  _ {i} \| [ \sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) ] _ {i,:} \| _ {2} \\ \stackrel {<  } {\underset {(a)} {\max}} _ {i} \left\| \left[ \mathbf {L} \mathbf {H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} \right] _ {i,:} \right\| _ {2} \\ \leq \max _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} ] _ {i,:} \| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} \\ = \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \mathbf {h} _ {j} ^ {(\ell - 1)} \right\| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} \tag {57} \\ \leq \max  _ {i} \| \sum_ {j = 1} ^ {N} L _ {i, j} \| _ {2} \cdot \max  _ {j} \| \mathbf {h} _ {j} ^ {(\ell - 1)} \| _ {2} \cdot \| \mathbf {W} ^ {(\ell)} \| _ {2} \\ \stackrel {<  } {\underset {(b)} {\leq}} \sqrt {d} \| \mathbf {W} ^ {(\ell)} \| _ {2} \cdot h _ {\max } ^ {(\ell - 1)} \\ \leq \sqrt {d} B _ {w} \cdot h _ {\max } ^ {(\ell - 1)} \leq \left(\max  \{1, \sqrt {d} B _ {w}\}\right) ^ {\ell} B _ {x}, \\ \end{array}
$$

where $( a )$ is due to $\| { \boldsymbol { \sigma } } ( \mathbf { x } ) \| _ { 2 } \leq \| \mathbf { x } \| _ { 2 }$ and $( b )$ is due to Lemma 31.

# F.3 Proof of Lemma 7

By the definition of $\Delta h _ { \operatorname* { m a x } } ^ { ( \ell ) }$

$$
\begin{array}{l} \Delta h _ {\max} ^ {(\ell)} = \max _ {i} \| \mathbf {h} _ {i} ^ {(\ell)} - \tilde {\mathbf {h}} _ {i} ^ {(\ell)} \| _ {2} \\ = \max  _ {i} \| [ \sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) - \sigma (\mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)}) ] _ {i,:} \| _ {2} \\ = \max  _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \max  _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} (\mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)}) + \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) \tilde {\mathbf {W}} ^ {(\ell)} ] _ {i,:} \| _ {2} \tag {58} \\ \leq \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \mathbf {h} _ {j} ^ {(\ell - 1)} \right\| _ {2} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \Delta \mathbf {h} _ {j} ^ {(\ell - 1)} \right\| _ {2} \| \tilde {\mathbf {W}} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} \Delta h _ {\max } ^ {(\ell - 1)} \| \tilde {\mathbf {W}} ^ {(\ell)} \| _ {2} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} B _ {w} \Delta h _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}. \\ \end{array}
$$

By induction, we have

$$
\begin{array}{l} \Delta h _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {w} \Delta h _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq (\sqrt {d} B _ {w}) ^ {2} \Delta h _ {\max } ^ {(\ell - 2)} + \sqrt {d} \left(h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + (\sqrt {d} B _ {w}) h _ {\max } ^ {(\ell - 2)} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2}\right) \\ \leq (\sqrt {d} B _ {w}) ^ {\ell} \Delta h _ {\max } ^ {(0)} + \sqrt {d} \left(h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + (\sqrt {d} B _ {w}) h _ {\max } ^ {(\ell - 2)} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \dots + (\sqrt {d} B _ {w}) ^ {\ell - 1} h _ {\max } ^ {(0)} \| \Delta \mathbf {W} ^ {(1)} \| _ {2}\right) \\ = \sqrt {d} \left(h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + (\sqrt {d} B _ {w}) h _ {\max } ^ {(\ell - 2)} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \dots + (\sqrt {d} B _ {w}) ^ {\ell - 1} h _ {\max } ^ {(0)} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2}\right) \\ \end{array}
$$

where $( a )$ is due to $\Delta h _ { \mathrm { m a x } } ^ { ( 0 ) } = 0$ . Plugging in the upper bound of $h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ in Lemma 6, yields

$$
\begin{array}{l} \Delta h _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(1)} \| _ {2}\right) \tag {60} \\ \leq \sqrt {d} B _ {x} \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(1)} \| _ {2}\right). \\ \end{array}
$$

# F.4 Proof of Lemma 8

By the definition of $\mathbf { Z } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} = \mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)} \\ = \mathbf {L} \left(\mathbf {H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \tilde {\mathbf {H}} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} + \tilde {\mathbf {H}} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)}\right) \tag {61} \\ = \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) \mathbf {W} ^ {(\ell)} + \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} (\mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)}). \\ \end{array}
$$

By taking norm on the both side of the equation, we have

$$
\begin{array}{l} \Delta z _ {\max } ^ {(\ell)} = \max  _ {i} \| [ \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \sqrt {d} B _ {w} \cdot \max  _ {i} \| \left[ \mathbf {Z} ^ {(\ell - 1)} - \tilde {\mathbf {Z}} ^ {(\ell - 1)} \right] _ {i,:} \| _ {2} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \tag {62} \\ = \sqrt {d} B _ {w} \cdot z _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}. \\ \end{array}
$$

By induction, we have

$$
\begin{array}{l} \Delta z _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {w} \cdot \Delta z _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} B _ {w} \cdot \Delta z _ {\max } ^ {(\ell - 1)} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} B _ {w} \left(\sqrt {d} B _ {w} \cdot \Delta z _ {\max } ^ {(\ell - 2)} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 2} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2}\right) + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ = (\sqrt {d} B _ {w}) ^ {2} \cdot \Delta z _ {\max } ^ {(\ell - 2)} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right) \\ \leq \dots \\ \leq \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {63} \\ \end{array}
$$

where (a) is due to $h _ { \operatorname* { m a x } } ^ { ( \ell - 1 ) } \leq ( \sqrt { d } B _ { w } ) ^ { \ell - 1 } B _ { x }$

# F.5 Proof of Lemma 9

For notation simplicity, let $\mathbf { D } ^ { ( \ell ) }$ denote the gradient passing from `th to $( \ell - 1 )$ th layer. By the definition of $d _ { \operatorname* { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} d _ {\mathrm {m a x}} ^ {(\ell)} = \max _ {i} \| \mathbf {d} _ {i} ^ {(\ell)} \| _ {2} \\ = \max  _ {i} \| [ \mathbf {L} ^ {\top} \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \odot \mathbf {D} ^ {(\ell + 1)} \mathbf {W} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \stackrel {<  } {\underset {(a)} {\max}} _ {i} \| [ \mathbf {L} ^ {\top} \mathbf {D} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} ^ {\top} \mathbf {d} _ {j} ^ {(\ell + 1)} \right\| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} \tag {64} \\ \leq \sqrt {d} \| \mathbf {W} ^ {(\ell)} \| _ {2} \cdot d _ {\max } ^ {(\ell + 1)} \\ \leq \sqrt {d} B _ {w} \cdot d _ {\max } ^ {(\ell + 1)} \\ \leq \frac {2}{(b)} \left(\sqrt {d} B _ {w}\right) ^ {L - \ell + 1} \leq \frac {2}{\gamma} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell + 1} \\ \end{array}
$$

where $( a )$ is due to each element in $\sigma ^ { \prime } ( { \bf Z } ^ { ( \ell ) } )$ is either 0 or 1 depending on $\mathbf { Z } ^ { ( \ell ) }$ , and $( b )$ is due to k ∂Loss(f (h(L)i ),yi)∂h(L) k2 ≤ 2/γ. By taking norm on the both sides, we have $\| \frac { \partial \mathrm { L o s s } ( f ( \mathbf { h } _ { i } ^ { ( L ) } ) , y _ { i } ) } { \partial \mathbf { h } _ { i } ^ { ( L ) } } \| _ { 2 } \leq 2 / \gamma$

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} = \max  _ {i} \| [ \mathbf {D} ^ {(\ell)} - \tilde {\mathbf {D}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ = \max  _ {i} \| [ \mathbf {L} ^ {\top} (\mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})) [ \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \\ \leq \max  _ {i} \| [ \mathbf {L} ^ {\top} (\mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} ] _ {i,}: \| _ {2} \\ + \max  _ {i} \| [ \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \\ + \max  _ {i} \| [ \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})) [ \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \\ \end{array}
$$

$$
\begin{array}{l} \leq \max  _ {i} \| [ \mathbf {L} ^ {\top} ((\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \\ + \max  _ {i} \| [ \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} [ \mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \\ + \max  _ {i} \| [ \mathbf {L} \tilde {\mathbf {D}} ^ {(\ell + 1)} (\mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)}) \mathbf {W} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \sqrt {d} B _ {w} \Delta d _ {\max } ^ {(\ell + 1)} + \sqrt {d} d _ {\max } ^ {(\ell + 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {w} d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)} \\ \leq \sqrt {d} B _ {w} \Delta d _ {\max } ^ {(\ell + 1)} + \underbrace {d _ {\max } ^ {(\ell + 1)} \left(\sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {w} \Delta z _ {\max } ^ {(\ell)}\right)} _ {(A)}, \tag {65} \\ \end{array}
$$

where inequality $( a )$ is due to the gradient of ReLU activation function is either 1 or 0.

Knowing that $\begin{array} { r } { d _ { \mathrm { m a x } } ^ { ( \ell + 1 ) } \leq \frac { 2 } { \gamma } ( \sqrt { d } B _ { w } ) ^ { L - \ell } } \end{array}$ and using the upper bound of $z _ { \mathrm { m a x } } ^ { ( \ell ) }$ as derived in Lemma 8, we can upper bound $( A )$ as

$$
\begin{array}{l} d _ {\max } ^ {(\ell + 1)} \left(\sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {w} \Delta z _ {\max } ^ {(\ell)}\right) \\ \leq \frac {2}{\gamma} (\sqrt {d} B _ {w}) ^ {L - \ell} \left(\sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right)\right) \\ \leq \frac {2}{\gamma} (\sqrt {d} B _ {w}) ^ {L - \ell} \left(\sqrt {d} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell}\right) \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right) \tag {66} \\ \leq \frac {2}{\gamma} (\sqrt {d} B _ {w}) ^ {L - \ell} \sqrt {d} (1 + B _ {x}) (\max  \{1, \sqrt {d} B _ {w} \}) ^ {\ell} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \frac {2}{\gamma} \sqrt {d} (1 + B _ {x}) (\max  \{1, \sqrt {d} B _ {w} \}) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2}. \\ \end{array}
$$

Plugging it back, we have

$$
\begin{array}{l} \Delta d _ {\max} ^ {(\ell)} \leq \sqrt {d} B _ {w} \Delta d _ {\max} ^ {(\ell + 1)} + \frac {2}{\gamma} \sqrt {d} (B _ {x} + 1) (\max \{1, \sqrt {d} B _ {w} \}) ^ {L} \| \Delta \pmb {\theta} \| _ {2} \\ \leq \sqrt {d} B _ {w} \Big (\sqrt {d} B _ {w} \Delta d _ {\max } ^ {(\ell + 2)} + \frac {2}{\gamma} \sqrt {d} (B _ {x} + 1) \big (\max  \{1, \sqrt {d} B _ {w} \} \big) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \Big) \\ + \frac {2}{\gamma} \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \left(1 + \sqrt {d} B _ {w} + \dots + (\max  \{1, \sqrt {d} B _ {w} \}) ^ {L - \ell - 1}\right) \cdot \frac {2}{\gamma} \sqrt {d} (B _ {x} + 1) (\max  \{1, \sqrt {d} B _ {w} \}) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \\ + \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell + 1} \Delta d _ {\max } ^ {L + 1} \\ \leq \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell} \cdot \frac {2}{\gamma} L \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} + \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell} \Delta d _ {\max } ^ {L + 1} \\ = \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell} \left(\frac {2}{\gamma} L \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} + \Delta d _ {\max } ^ {L + 1}\right). \tag {67} \\ \end{array}
$$

$\Delta d _ { \mathrm { m a x } } ^ { ( L + 1 ) }$ $\Delta d _ { \mathrm { m a x } } ^ { ( L + 1 ) }$

$$
\begin{array}{l} \Delta d _ {\mathrm {m a x}} ^ {(L + 1)} = \max _ {i} \| \mathbf {d} _ {i} ^ {(L + 1)} - \tilde {\mathbf {d}} _ {i} ^ {(L + 1)} \| _ {2} \\ = \max  _ {i} \left\| \frac {\partial \operatorname {L o s s} (f (\mathbf {h} _ {i} ^ {(L)} , y _ {i})}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial \operatorname {L o s s} (\tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)} , y _ {i})}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \\ \leq_ {(a)} \frac {2}{\gamma} \max  _ {i} \left\| \frac {\partial f \left(\mathbf {h} _ {i} ^ {(L)}\right)}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial f \left(\tilde {\mathbf {h}} _ {i} ^ {(L)}\right)}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \\ \leq \frac{2}{b}\frac{2}{\gamma}\Bigl(\sqrt{d} B_{x}(\max \{1,\sqrt{d} B_{w}\})^{L - 1}(\| \Delta \mathbf{W}^{(1)}\|_{2} + \ldots +\| \Delta \mathbf{W}^{(L)}\|_{2}) + \bigl(B_{x}(\max \{1,\sqrt{d} B_{w}\})^{L} + 1\bigr)\Delta \mathbf{v}\Bigr) \\ \leq \frac {2}{\gamma} \left(\sqrt {d} B _ {x} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} + 1\right) \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(L)} \| _ {2} + \| \Delta \mathbf {v} \| _ {2}\right) \\ = \frac {2}{\gamma} \left(\sqrt {d} B _ {x} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {68} \\ \end{array}
$$

where $( a )$ is due to fact that $\nabla \mathrm { L o s s } ( z , y )$ is $2 / \gamma$ -Lipschitz continuous with respect to $z$ , $( b )$ follows from Lemma 6 and Lemma 7.

Therefore, we have

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} \leq \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell} \left(\frac {2}{\gamma} L \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} + \Delta d _ {\max } ^ {L + 1}\right) \\ \leq \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L - \ell} \frac {2}{\gamma} \left(L \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} + \sqrt {d} B _ {x} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {L - \ell} \frac {2}{\gamma} \Big ((L + 1) \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {L} + 1 \Big) \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {69} \\ \end{array}
$$

# F.6 Proof of Lemma 10

By the definition of $\mathbf { G } ^ { ( \ell ) } , \ell \in [ L ]$ , we have

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} \| _ {2} = \frac {1}{m} \| [ \mathbf {L H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \| _ {2} \\ = \frac {1}{m} \| [ \mathbf {L H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell)} \| _ {2} \tag {70} \\ \leq \sqrt {d} h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell)}. \\ \end{array}
$$

By plugging in the result from Lemma 6 and Lemma 9, we have

$$
\left\| \mathbf {G} ^ {(\ell)} \right\| _ {2} \leq \frac {2}{\gamma} \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L}. \tag {71}
$$

Besides, recall that $\mathbf { G } ^ { ( L + 1 ) }$ denotes the gradient with respect to the weight of binary classifier. Therefore, we have

$$
\left\| \mathbf {G} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{\gamma} h _ {\max } ^ {(L)} \leq \frac {2}{\gamma} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} B _ {x}, \tag {72}
$$

which is smaller than the right hand side of the Eq. 71. Therefore, we have

$$
\sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} \| _ {2} \leq (L + 1) \frac {2}{\gamma} \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L}. \tag {73}
$$

Similarly, we can upper bound the difference between gradients computed on two different set of weight parameters for the `th layer as

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} = \frac {1}{m} \| [ \mathbf {L} \mathbf {H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)}) \| _ {2} \\ \leq \frac {1}{m} \| [ \mathbf {L H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \| _ {2} \\ + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \| _ {2} \\ + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)}) \| _ {2} \\ \leq \frac {1}{m} \| [ \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \| _ {2} + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} (\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)}) \| _ {2} \\ + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \left(\sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})\right) \| _ {2} \\ \stackrel {\leq} {\underset {(b)} {\max}} \left\| \left[ \left[ \mathbf {L} \left(\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}\right) \right] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \right] _ {i,:} \right\| _ {2} \\ + \max  _ {i} \| \left[ \left[ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \right] ^ {\top} \left(\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)}\right) \right] _ {i,:} \| _ {2} \\ + \max _ {i} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \| _ {2} \| \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} \left(\underbrace {\Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)}} _ {(A)} + \underbrace {h _ {\max } ^ {(\ell - 1)} \Delta d _ {\max } ^ {(\ell + 1)}} _ {(B)} + \underbrace {h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)}} _ {(C)}\right), \tag {74} \\ \end{array}
$$

where inequalities $( a )$ and $( b )$ are due to the fact that the gradient of ReLU is either 0 or 1.

We now proceed to upper bound the three terms on the right hand side. By plugging the result from√ √ Lemma 6, Lemma 7, Lemma 9, and letting $C _ { 1 } = \operatorname* { m a x } \lbrace 1 , \sqrt { d } B _ { w } \rbrace$ and $C _ { 2 } = \sqrt { d } ( B _ { x } + 1 )$ , we can upper bound the term $( A )$ as

$$
\begin{array}{l} \Delta h _ {\max} ^ {(\ell - 1)} d _ {\max} ^ {(\ell + 1)} \leq \sqrt {d} B _ {x} (\max \{1, \sqrt {d} B _ {w} \}) ^ {\ell - 2} \cdot \frac {2}{\gamma} (\max \{1, \sqrt {d} B _ {w} \}) ^ {L - \ell} \\ \leq \frac {2}{\gamma} \sqrt {d} B _ {x} \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \tag {75} \\ \leq \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}, \\ \end{array}
$$

upper bound $( B )$ as

$$
\begin{array}{l} h _ {\max } ^ {(\ell - 1)} \Delta d _ {\max } ^ {(\ell + 1)} \leq B _ {x} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} \frac {2}{\gamma} \left((L + 1) \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \left((L + 1) C _ {1} ^ {L} C _ {2} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {76} \\ \end{array}
$$

and upper bound $( C )$ as

$$
\begin{array}{l} h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)} \leq B _ {x} (\max  \{1, \sqrt {d} B _ {w} \}) ^ {\ell - 1} \cdot \frac {2}{\gamma} (\max  \{1, \sqrt {d} B _ {w} \}) ^ {L - \ell} \cdot \sqrt {d} B _ {x} (\max  \{1, \sqrt {d} B _ {w} \}) ^ {\ell - 1} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \frac {2}{\gamma} \sqrt {d} B _ {x} ^ {2} \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {2 L} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \frac {2}{\gamma} \left(C _ {1} ^ {L} C _ {2}\right) ^ {2} \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {77} \\ \end{array}
$$

By combining the results above, we can upper bound $\Delta \mathbf { G } ^ { ( \ell ) }$ as

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} \leq \sqrt {d} \left(\Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} + h _ {\max } ^ {(\ell - 1)} \Delta d _ {\max } ^ {(\ell + 1)} + h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \cdot \Delta z _ {\max } ^ {(\ell)}\right) \\ \leq \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \left((L + 2) C _ {1} ^ {L} C _ {2} + 2\right) \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {78} \\ \end{array}
$$

Similarly, we can upper bound the difference between gradient for the weight of binary classifier as

$$
\begin{array}{l} \left\| \mathbf {G} ^ {(L + 1)} - \tilde {\mathbf {G}} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{\gamma} \Delta h _ {\max } ^ {(L)} \tag {79} \\ \leq \frac {2}{\gamma} \sqrt {d} B _ {x} \left(\left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {L - 1} \left(\| \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \mathbf {W} ^ {(L)} \| _ {2}\right), \\ \end{array}
$$

which is smaller than the right hand side of the previous equation. Therefore, we have

$$
\sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} \leq (L + 1) \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \left((L + 2) C _ {1} ^ {L} C _ {2} + 2\right) \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {80}
$$

# G Generalization bound for ResGCN

In the following, we provide detailed proof on the generalization bound of ResGCN. Recall that the update rule of ResGCN is defined as

$$
\mathbf {H} ^ {(\ell)} = \sigma \left(\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}\right) + \mathbf {H} ^ {(\ell - 1)}, \tag {81}
$$

where $\sigma ( \cdot )$ is ReLU activation function. Please notice that although ReLU function $\sigma ( x )$ is not differentiable when $x = 0$ , for analysis purpose, we suppose the $\sigma ^ { \prime } ( 0 ) = 0$ .

The training of ResGCN is an empirical risk minimization with respect to a set of parameters $\pmb { \theta } = \{ \mathbf { W } ^ { ( 1 ) } , \dots , \mathbf { W } ^ { ( L ) } , \mathbf { v } \}$ , i.e.,

$$
\mathcal {L} (\boldsymbol {\theta}) = \frac {1}{m} \sum_ {i = 1} ^ {m} \Phi_ {\gamma} (- p (f \left(\mathbf {h} _ {i} ^ {(L)}\right), y _ {i})), f \left(\mathbf {h} _ {i} ^ {(L)}\right) = \tilde {\sigma} \left(\mathbf {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}\right), \tag {82}
$$

where ${ \bf h } _ { i } ^ { ( L ) }$ is the node representation of the ith node at the final layer, $f ( \mathbf { h } _ { i } ^ { ( L ) } )$ is the prediction of the ith node, $\begin{array} { r } { \tilde { \sigma } ( x ) = \frac { 1 } { \exp \left( - x \right) + 1 } } \end{array}$ is the sigmoid function, and loss function $\Phi _ { \gamma } ( - p ( z , y ) )$ is $\textstyle { \frac { 2 } { \gamma } }$ - Lipschitz continuous with respect to its first input $z$ . For simplification, we will use $\mathrm { L o s s } ( z , y )$ denote $\bar { \Phi _ { \gamma } } ( - p ( z , y ) )$ in the proof.

To establish the generalization of ResGCN as stated in Theorem 3, we utilize the result on transductive uniform stability from [16] (Theorem 8 in Appendix F). Then, in Lemma 11, we derive the uniform stability constant for ResGCN, i.e., ResGCN.

Lemma 11. The uniform stability constant for ResGCN is computed as $\begin{array} { r } { \epsilon _ { R e s G C N } = \frac { 2 \eta \rho _ { f } G _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + } \end{array}$ m PTt=1(1 + $\eta L _ { F } ) ^ { t - 1 }$ where

$$
\rho_ {f} = C _ {1} ^ {L} C _ {2}, G _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2}, L _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} ((L + 2) C _ {1} ^ {L} C _ {2} + 2), \tag {83}
$$

$$
C _ {1} = 1 + \sqrt {d} B _ {w}, C _ {2} = \sqrt {d} (B _ {x} + 1).
$$

By plugging the result in Lemma 11 back to Theorem 8, we establish the generalization bound for ResGCN.

Similar to the proof on the generalization bound of GCN in Section F, the key idea of the proof is to decompose the change if the ResGCN output into two terms (in Lemma 5) which depend on

• (Lemma 12) The maximum change of node representation, i.e., $\Delta h _ { \mathrm { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \parallel [ \mathbf { H } ^ { ( \ell ) } -$ $\tilde { \mathbf { H } } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 }$ ,   
• (Lemma 13) The maximum node representation, i.e., $h _ { \operatorname* { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { H } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 }$

Lemma 12 (Upper bound of maximum node embeddings fo $h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ for ResGCN). Let suppose Assumptoon node at the `th layer is bounded by $\cdot$ hold. Then, the

$$
h _ {\max } ^ {(\ell)} \leq B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {\ell}. \tag {84}
$$

Lemma 13 (Upper bound of $\Delta h _ { \mathrm { m a x } } ^ { ( \ell ) }$ for ResGCN). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta h _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {85}
$$

where $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) }$

before the activation function, i.e., Besides, in Lemma 14, we derive the upper bound on the maximum change of node embeddings $\begin{array} { r } { \Delta z _ { \mathrm { m a x } } ^ { ( \bar { \ell } ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { Z } ^ { ( \ell ) } - \tilde { \mathbf { Z } } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 } } \end{array}$ , which will be used for the proof of gradient related upper bounds.

Lemma 14 (Upper bound of $\Delta z _ { \mathrm { m a x } } ^ { ( \ell ) }$ for ResGCN). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings before the activation function on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta z _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {86}
$$

where $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) }$

Then, in Lemma 15, we decompose the change of the model parameters into two terms which depend on

• The maximum change of gradient passing from the $( \ell + 1 )$ th layer to the `th layer

$$
\Delta d _ {\max} ^ {(\ell)} = \max _ {i} \left\| \left[ \frac {\partial (\sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) + \mathbf {H} ^ {(\ell - 1)})}{\partial \mathbf {H} ^ {(\ell - 1)}} - \frac {\partial (\sigma (\mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)}) + \tilde {\mathbf {H}} ^ {(\ell - 1)})}{\partial \tilde {\mathbf {H}} ^ {(\ell - 1)}} \right] _ {i,:} \right\| _ {2}.
$$

• The maximum gradient passing from the $( \ell + 1 )$ th layer to the `th layer

$$
d _ {\max } ^ {(\ell)} = \max  _ {i} \left\| \left[ \frac {\partial (\sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) + \mathbf {H} ^ {(\ell - 1)})}{\partial \mathbf {H} ^ {(\ell - 1)}} \right] _ {i,:} \right\| _ {2}.
$$

Lemma 15 (Upper bound of $d _ { \operatorname* { m a x } } ^ { ( \ell ) } , \Delta d _ { \operatorname* { m a x } } ^ { ( \ell ) }$ for ResGCN). Let suppose Assumption 1 hold. Then, the maximum gradient passing from layer $\ell + 1$ to layer ` for any node is bounded by

$$
d _ {\max } ^ {(\ell)} \leq \frac {2}{\gamma} (1 + \sqrt {d} B _ {w}) ^ {L - \ell + 1}, \tag {87}
$$

and the maximum change between the gradient passing from layer $\ell + 1$ to layer ` on two different set of weight parameters for any node is bounded by

$$
\Delta d _ {\max } ^ {(\ell)} \leq \frac {2}{\gamma} (1 + \sqrt {d} B _ {w}) ^ {L - \ell} \left((L + 1) \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {88}
$$

where $\begin{array} { r } { \| \Delta \pmb { \theta } \| _ { 2 } = \| \mathbf { v } - \tilde { \mathbf { v } } \| _ { 2 } + \sum _ { \ell = 1 } ^ { L } \| \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } \| _ { 2 } } \end{array}$ denotes the change of two set of parameters.

Finally, based on the previous result, in Lemma 16, we decompose the change of the model parameters into two terms which depend on

• The change of gradient with respect to the `th layer weight matrix $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ ,   
• The gradient with respect to the `th layer weight matrix $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$

where $\mathbf { G } ^ { ( L + 1 ) }$ denotes the gradient with respect to the weight $\mathbf { v }$ of the binary classifier and $\mathbf { G } ^ { ( \ell ) }$ denotes the gradient with respect to the weight $\mathbf { W } ^ { ( \ell ) }$ of the `th layer graph convolutional layer. Notice that $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ reflect the smoothness of ResGCN model and $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$ correspond the upper bound of gradient.

Lemma 16 (Upper bound of √ $\| \mathbf G ^ { ( \ell ) } \| _ { 2 } , \| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ for ResGCN). Let suppose Assumption 1 hold and let $C _ { 1 } = 1 + \sqrt { d } B _ { w }$ and $C _ { 2 } = \sqrt { d } ( B _ { x } + 1 )$ . Then, the gradient and the maximum change between gradients on two different set of weight parameters are bounded by

$$
\begin{array}{l} \sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}, \\ \sum_ {\ell = 1} ^ {L + 1} \left\| \Delta \mathbf {G} ^ {(\ell)} \right\| _ {2}, \leq \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} \left((L + 2) C _ {1} ^ {L} C _ {2} + 2\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {89} \\ \end{array}
$$

where $\begin{array} { r } { \| \Delta \pmb { \theta } \| _ { 2 } = \| \mathbf { v } - \tilde { \mathbf { v } } \| _ { 2 } + \sum _ { \ell = 1 } ^ { L } \| \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } \| _ { 2 } } \end{array}$ denotes the change of two set of parameters.

Equipped with above intermediate results, we now proceed to prove Lemma 11.

Proof. Recall that our goal is to explore the impact of different GCN structures on the uniform stability constant $\epsilon _ { \mathrm { R e s G C N } }$ , which is a function of $\rho _ { f }$ , $G _ { f }$ , and $L _ { f }$ . Let $C _ { 1 } = 1 + \sqrt { d } B _ { w }$ , $C _ { 2 } = \sqrt { d } ( B _ { x } + 1 )$ . Firstly, by plugging Lemma 13 and Lemma 12 into Lemma 5, we have that

$$
\begin{array}{l} \max _ {i} | f (\mathbf {h} _ {i} ^ {(L)}) - \tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)}) | \leq \Delta h _ {\max} ^ {(L)} + h _ {\max} ^ {(L)} \| \Delta \mathbf {v} \| _ {2} \\ \leq \sqrt {d} B _ {x} \cdot \left(\max  \{1, \sqrt {d} B _ {w} \}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \tag {90} \\ \leq C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}. \\ \end{array}
$$

Therefore, we know that the function $f$ is $\rho _ { f }$ -Lipschitz continuous, with $\rho _ { f } = C _ { 1 } ^ { L } C _ { 2 }$ . Then, by Lemma 16, we know that the function $f$ is $L _ { f }$ -smoothness, and the gradient of each weight matrices is bounded by $G _ { f }$ , with

$$
G _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2}, L _ {f} = \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2} ((L + 2) C _ {1} ^ {L} C _ {2} + 2). \tag {91}
$$

By plugging $\epsilon _ { \mathrm { R e s G C N } }$ into Theorem 3, we obtain the generalization bound of ResGCN.

# G.1 Proof of Lemma 12

$h _ { \operatorname* { m a x } } ^ { ( \ell ) }$

$$
\begin{array}{l} h _ {\mathrm {m a x}} ^ {(\ell)} = \max _ {i} \| [ \sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) + \mathbf {H} ^ {(\ell - 1)} ] _ {i,:} \| _ {2} \\ \leq \max _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} ] _ {i,:} \| _ {2} + \max _ {i} \| \mathbf {h} _ {i} ^ {(\ell - 1)} \| _ {2} \\ \leq \max _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} ] _ {i,:} \| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} + h _ {\max} ^ {(\ell - 1)} \\ = \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \mathbf {h} _ {j} ^ {(\ell - 1)} \right\| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} + h _ {\max } ^ {(\ell - 1)} \tag {92} \\ \leq \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \right\| _ {2} \cdot \max  _ {j} \| \mathbf {h} _ {j} ^ {(\ell - 1)} \| _ {2} \cdot \| \mathbf {W} ^ {(\ell)} \| _ {2} + h _ {\max } ^ {(\ell - 1)} \\ \underset {(a)} {\leq} \left(1 + \sqrt {d} \| \mathbf {W} ^ {(\ell)} \| _ {2}\right) \cdot h _ {\max } ^ {(\ell - 1)} \\ \leq \left(1 + \sqrt {d} B _ {w}\right) \cdot h _ {\max } ^ {(\ell - 1)} \leq B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {\ell} \\ \end{array}
$$

where inequality $( a )$ follows from Lemma 31.

# G.2 Proof of Lemma 13

By the definition of $\Delta h _ { \mathrm { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} \Delta h _ {\mathrm {m a x}} ^ {(\ell)} = \max _ {i} \| \mathbf {h} _ {i} ^ {(\ell)} - \tilde {\mathbf {h}} _ {i} ^ {(\ell)} \| _ {2} \\ = \max  _ {i} \| [ \sigma (\mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)}) - \sigma (\mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)}) + \mathbf {H} ^ {(\ell)} - \tilde {\mathbf {H}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \max _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)} ] _ {i,:} \| _ {2} + \max _ {i} \| [ \mathbf {H} ^ {(\ell)} - \tilde {\mathbf {H}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \max _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} (\mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)}) - \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) \tilde {\mathbf {W}} ^ {(\ell)} ] _ {i,:} \| _ {2} + \Delta h _ {\max } ^ {(\ell - 1)} \\ \leq \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \mathbf {h} _ {j} ^ {(\ell - 1)} \right\| _ {2} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \Delta \mathbf {h} _ {j} ^ {(\ell - 1)} \right\| _ {2} \| \tilde {\mathbf {W}} ^ {(\ell)} \| _ {2} + \Delta h _ {\max } ^ {(\ell - 1)} \\ \leq \left(1 + \sqrt {d} B _ {w}\right) \cdot \Delta h _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}, \tag {93} \\ \end{array}
$$

where inequality $( a )$ follows from Lemma 31.

By induction, we have

$$
\begin{array}{l} \Delta h _ {\max } ^ {(\ell)} \leq (1 + \sqrt {d} B _ {w}) \cdot \Delta h _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq \left(1 + \sqrt {d} B _ {w}\right) ^ {2} \cdot \Delta h _ {\max } ^ {(\ell - 2)} \\ + \sqrt {d} \left(h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + (1 + \sqrt {d} B _ {w}) h _ {\max } ^ {(\ell - 2)} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2}\right) \\ \end{array}
$$

$$
\begin{array}{l} \leq \left(1 + \sqrt {d} B _ {w}\right) ^ {\ell} \cdot \Delta h _ {\max } ^ {(0)} \\ + \sqrt {d} \left(h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + (1 + \sqrt {d} B _ {w}) h _ {\max } ^ {(\ell - 2)} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \dots + (1 + \sqrt {d} B _ {w}) ^ {\ell - 1} h _ {\max } ^ {(0)} \| \Delta \mathbf {W} ^ {(1)} \| _ {2}\right) \\ = \sqrt {d} \left(h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + (1 + \sqrt {d} B _ {w}) h _ {\max } ^ {(\ell - 2)} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \dots + (1 + \sqrt {d} B _ {w}) ^ {\ell - 1} h _ {\max } ^ {(0)} \| \Delta \mathbf {W} ^ {(1)} \| _ {2}\right), \tag {94} \\ \end{array}
$$

$\Delta h _ { \mathrm { m a x } } ^ { ( 0 ) } = 0$

Plugging in the upper bound of $h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ in Lemma 13, we have

$$
\Delta h _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {\ell - 1} \left(\| \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \mathbf {W} ^ {(\ell)} \| _ {2}\right). \tag {95}
$$

# G.3 Proof of Lemma 14

By the definition of $\mathbf { Z } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} = \mathbf {L H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)} \\ = \mathbf {L} \left(\mathbf {H} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \tilde {\mathbf {H}} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} + \tilde {\mathbf {H}} ^ {(\ell - 1)} \mathbf {W} ^ {(\ell)} - \tilde {\mathbf {H}} ^ {(\ell - 1)} \tilde {\mathbf {W}} ^ {(\ell)}\right) \tag {96} \\ = \mathbf {L} \left(\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}\right) \mathbf {W} ^ {(\ell)} + \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} \left(\mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)}\right). \\ \end{array}
$$

Then, by taking the norm of both sides, we have

$$
\begin{array}{l} \Delta z _ {\mathrm {m a x}} ^ {(\ell)} = \max _ {i} \| [ \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \sqrt {d} B _ {w} \cdot \max  _ {i} \| \left[ \mathbf {Z} ^ {(\ell - 1)} - \tilde {\mathbf {Z}} ^ {(\ell - 1)} \right] _ {i,:} \| _ {2} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \tag {97} \\ = \sqrt {d} B _ {w} \cdot z _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}. \\ \end{array}
$$

By induction, we have

$$
\begin{array}{l} \Delta z _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {w} \cdot \Delta z _ {\max } ^ {(\ell - 1)} + \sqrt {d} h _ {\max } ^ {(\ell - 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} B _ {w} \cdot \Delta z _ {\max } ^ {(\ell - 1)} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} B _ {w} \left(\sqrt {d} B _ {w} \cdot \Delta z _ {\max } ^ {(\ell - 2)} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 2} \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2}\right) + \sqrt {d} B _ {x} (1 + \sqrt {d} B _ {w}) ^ {\ell - 1} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ = (\sqrt {d} B _ {w}) ^ {2} \cdot \Delta z _ {\max } ^ {(\ell - 2)} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right) \\ \leq \dots \\ \leq \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {98} \\ \end{array}
$$

where (a) is due to $h _ { \operatorname* { m a x } } ^ { ( \ell - 1 ) } \leq ( \sqrt { d } B _ { w } ) ^ { \ell - 1 } B _ { x }$

# G.4 Proof of Lemma 15

For notation simplicity, let $\mathbf { D } ^ { ( \ell ) }$ denote the gradient passing from the `th to the $( \ell - 1 )$ th layer. By the definition of $d _ { \operatorname* { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} d _ {\mathrm {m a x}} ^ {(\ell)} = \max _ {i} \| \mathbf {d} _ {i} ^ {(\ell)} \| _ {2} \\ = \max  _ {i} [ \| \mathbf {L} ^ {\top} \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \odot \mathbf {D} ^ {(\ell + 1)} \mathbf {W} ^ {(\ell)} + \mathbf {D} ^ {(\ell + 1)} \| ] _ {i,:} \\ \leq \max _ {i} [ \| \mathbf {L} ^ {\top} \mathbf {D} ^ {(\ell + 1)} \| _ {2} ] _ {i,:} \| \mathbf {W} ^ {(\ell)} \| _ {2} + d _ {\max } ^ {(\ell + 1)} \\ \leq \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} ^ {\top} \mathbf {d} _ {j} ^ {(\ell + 1)} \right\| _ {2} \| \mathbf {W} ^ {(\ell)} \| _ {2} + d _ {\max } ^ {(\ell + 1)} \tag {99} \\ \leq (1 + \sqrt {d} \| \mathbf {W} ^ {(\ell)} \| _ {2}) d _ {\max } ^ {(\ell + 1)} \\ \underset {(a)} {\leq} (1 + \sqrt {d} B _ {w}) \cdot d _ {\max } ^ {(\ell + 1)} \leq \frac {2}{\gamma} (1 + \sqrt {d} B _ {w}) ^ {L - \ell + 1}, \\ \end{array}
$$

where inequality $( a )$ follows from Lemma 31.

By the definition of $\Delta d _ { \mathrm { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} = \max  _ {i} \| [ \mathbf {D} ^ {(\ell)} - \tilde {\mathbf {D}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ = \max _ {i} \| [ \mathbf {L} ^ {\top} (\mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} + \mathbf {D} ^ {(\ell + 1)} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})) [ \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} - \tilde {\mathbf {D}} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \\ \leq \max  _ {i} \| [ \mathbf {L} ^ {\top} (\mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \\ + \max  _ {i} \| [ \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \\ + \max  _ {i} \| [ \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)})) [ \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} - \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})) [ \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} + \max  _ {i} \| [ \mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \\ \leq \max _ {i} \| [ \mathbf {L} ^ {\top} ((\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)}) [ \mathbf {W} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} + \max _ {i} \| [ \mathbf {L} ^ {\top} (\tilde {\mathbf {D}} ^ {(\ell + 1)} [ \mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \\ + \max  _ {i} \| [ \mathbf {L} ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} [ \tilde {\mathbf {W}} ^ {(\ell)} ] ^ {\top} ] _ {i,:} \| _ {2} \max  _ {i} \| [ \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} ] _ {i,:} \| + \max  _ {i} \| [ \mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \\ \leq (1 + \sqrt {d} B _ {w}) \Delta d _ {\max} ^ {(\ell + 1)} + \sqrt {d} d _ {\max} ^ {(\ell + 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {w} d _ {\max} ^ {(\ell + 1)} \Delta z _ {\max} ^ {(\ell)} \\ = (1 + \sqrt {d} B _ {w}) \Delta d _ {\max } ^ {(\ell + 1)} + \underbrace {d _ {\max } ^ {(\ell + 1)} \left(\sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {w} \Delta z _ {\max } ^ {(\ell)}\right)} _ {(A)}, \tag {100} \\ \end{array}
$$

where inequality $( a )$ is due to the fact that the gradient of ReLU is either 1 or 0.

Knowing that $d _ { \operatorname* { m a x } } ^ { ( \ell + 1 ) } \ \leq \ { \frac { 2 } { \gamma } } ( 1 + \sqrt { d } B _ { w } ) ^ { L - \ell }$ and $z _ { \mathrm { m a x } } ^ { ( \ell ) } \leq \sqrt { d } B _ { x } ( \sqrt { d } B _ { w } ) ^ { \ell - 1 } \Big ( \| \mathbf { W } ^ { ( 1 ) } \| _ { 2 } + \ldots +$ $\| \mathbf { W } ^ { ( \ell ) } \| _ { 2 } \Big )$ , we can upper bound $( A )$ by

$$
\begin{array}{l} d _ {\max } ^ {(\ell + 1)} \left(\sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {w} \Delta z _ {\max } ^ {(\ell)}\right) \\ \leq \frac {2}{\gamma} \left(1 + \sqrt {d} B _ {w}\right) ^ {L - \ell} \left(\sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell} \left(\| \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \mathbf {W} ^ {(\ell)} \| _ {2}\right)\right) \tag {101} \\ \leq \frac {2}{\gamma} (1 + \sqrt {d} B _ {w}) ^ {L - \ell} (\sqrt {d} + \sqrt {d} B _ {x} (\sqrt {d} B _ {w}) ^ {\ell}) (\| \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \mathbf {W} ^ {(\ell)} \| _ {2}) \\ \leq \frac {2}{\gamma} \sqrt {d} \left(1 + B _ {x}\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2}. \\ \end{array}
$$

By plugging it back, we have

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} \leq \sqrt {d} B _ {w} \Delta d _ {\max } ^ {(\ell + 1)} + \frac {2}{\gamma} \sqrt {d} (B _ {x} + 1) (1 + \sqrt {d} B _ {w}) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \sqrt {d} B _ {w} \left(\sqrt {d} B _ {w} \Delta d _ {\max } ^ {(\ell + 2)} + \frac {2}{\gamma} \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2}\right) \\ + \frac {2}{\gamma} \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \left(1 + \sqrt {d} B _ {w} + \dots + (1 + \sqrt {d} B _ {w}) ^ {L - \ell - 1}\right) \cdot \frac {2}{\gamma} \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \\ + \left(1 + \sqrt {d} B _ {w}\right) ^ {L - \ell + 1} \Delta d _ {\max } ^ {L + 1} \\ \leq \left(1 + \sqrt {d} B _ {w}\right) ^ {L - \ell} \cdot \frac {2}{\gamma} L \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} + \left(1 + \sqrt {d} B _ {w}\right) ^ {L - \ell} \Delta d _ {\max } ^ {L + 1} \\ = \left(1 + \sqrt {d} B _ {w}\right) ^ {L - \ell} \left(\frac {2}{\gamma} L \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} + \Delta d _ {\max } ^ {L + 1}\right). \tag {102} \\ \end{array}
$$

Let first explicit upper bound $\Delta d _ { \mathrm { m a x } } ^ { ( L + 1 ) }$ . By definition, we can write $\Delta d _ { \mathrm { m a x } } ^ { ( L + 1 ) }$

$$
\begin{array}{l} \Delta d _ {\mathrm {m a x}} ^ {(L + 1)} = \max _ {i} \| \mathbf {d} _ {i} ^ {(L + 1)} - \tilde {\mathbf {d}} _ {i} ^ {(L + 1)} \| _ {2} \\ = \max  _ {i} \left\| \frac {\partial \operatorname {L o s s} (f (\mathbf {h} _ {i} ^ {(L)} , y _ {i})}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial \operatorname {L o s s} (\tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)} , y _ {i})}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \\ \leq \frac {2}{(a)} \max  _ {i} \left\| \frac {\partial f \left(\mathbf {h} _ {i} ^ {(L)}\right)}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial f \left(\tilde {\mathbf {h}} _ {i} ^ {(L)}\right)}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \\ \leq \frac {2}{\gamma} \left(\sqrt {d} B _ {x} (1 + \sqrt {d} B _ {w}) ^ {L - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(L)} \| _ {2}\right) + \left(B _ {x} (1 + \sqrt {d} B _ {w}) ^ {L} + 1\right) \Delta \mathbf {v}\right) \\ \leq \frac {2}{\gamma} \left(\sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + 1\right) \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(L)} \| _ {2} + \| \Delta \mathbf {v} \| _ {2}\right) \\ = \frac {2}{\gamma} \left(\sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {103} \\ \end{array}
$$

where inequality $( a )$ is due to the fact that $\nabla \mathrm { L o s s } ( z , y )$ is $\frac { 2 } { \gamma }$ -Lipschitz continuous with respect to $z$ , and inequality $( b )$ follows from Lemma 12 and Lemma 13. Therefore, we have

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} \leq (1 + \sqrt {d} B _ {w}) ^ {L - \ell} \left(\frac {2}{\gamma} L \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} + \Delta d _ {\max } ^ {L + 1}\right) \\ \leq \left(1 + \sqrt {d} B _ {w}\right) ^ {L - \ell} \frac {2}{\gamma} \left(L \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + \sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \left(1 + \sqrt {d} B _ {w}\right) ^ {L - \ell} \frac {2}{\gamma} \left(\left(L + 1\right) \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {104} \\ \end{array}
$$

# G.5 Proof of Lemma 16

First By the definition of $\mathbf { G } ^ { ( \ell ) } , \ell \in [ L ]$ , we have

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} \| _ {2} = \frac {1}{m} \| [ \mathbf {L H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \| _ {2} \\ \leq \frac {1}{m} \| [ \mathbf {L H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell)} \| _ {2} \tag {105} \\ \leq \sqrt {d} h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell)} \\ \leq \frac {2}{(a)} \frac {\gamma}{\gamma} \sqrt {d} (B _ {x} + 1) (1 + \sqrt {d} B _ {w}) ^ {L}, \\ \end{array}
$$

where inequality $( a )$ follows from Lemma 12 and Lemma 15, and the fact that loss function is $\frac { 2 } { \gamma }$ -Lipschitz continuous.

Similarly, by the definition of $\mathbf { G } ^ { ( L + 1 ) }$ , we have

$$
\left\| \mathbf {G} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{(a)} h _ {\max } ^ {(L)} \leq \frac {2}{(b)} (B _ {x} + 1) (1 + \sqrt {d} B _ {w}) ^ {L}, \tag {106}
$$

which is smaller then the right hind side of Eq. 105, and inequality $( a )$ is due to the fact that loss function is $\frac { 2 } { \gamma }$ -Lipschitz continuous and inequality (b) follows from Lemma 12.

By combining the above two inequalities together, we have

$$
\sum_ {\ell = 1} ^ {L + 1} \left\| \mathbf {G} ^ {(\ell)} \right\| _ {2} \leq \frac {2}{\gamma} (L + 1) \sqrt {d} \left(B _ {x} + 1\right) \left(1 + \sqrt {d} B _ {w}\right) ^ {L}. \tag {107}
$$

Furthermore, we can upper bound the difference between gradients computed on two different set of weight parameters for the `th layer as

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} = \frac {1}{m} \| [ \mathbf {L H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)}) \| _ {2} \\ \leq \frac {1}{m} \| [ \mathbf {L H} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \| _ {2} \\ + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \| _ {2} \\ + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)}) \| _ {2} \\ \leq \frac {1}{m} \| [ \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \| _ {2} + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} (\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)}) \| _ {2} \\ + \frac {1}{m} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \left(\sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})\right) \| _ {2} \\ \stackrel {\leq} {\underset {(b)} {\max}} \underset {i} {\max} \| [ [ \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \\ + \max _ {i} \| [ [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} (\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)}) ] _ {i,:} \| _ {2} \\ + \max _ {i} \| [ \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \| _ {2} \| \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} \| _ {2} \\ \leq \sqrt {d} \left(\underbrace {\Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)}} _ {(A)} + \underbrace {h _ {\max } ^ {(\ell - 1)} \Delta d _ {\max } ^ {(\ell + 1)}} _ {(B)} + \underbrace {h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)}} _ {(C)}\right), \tag {108} \\ \end{array}
$$

where inequality $( a )$ and $( b )$ is due to the fact that the gradient of ReLU activation function is element-wise either 0 or 1.

By plugging the result from Lemma 12, Lemma 13, Lemma 15, and letting √ $C _ { 1 } = 1 + \sqrt { d } B _ { w }$ and $C _ { 2 } = \sqrt { d } ( B _ { x } + 1 )$ we can upper bound $( A )$ as

$$
\begin{array}{l} \Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \leq \sqrt {d} B _ {x} (1 + \sqrt {d} B _ {w}) ^ {\ell - 2} \cdot \frac {2}{\gamma} (1 + \sqrt {d} B _ {w}) ^ {L - \ell} \\ \leq \frac {2}{\gamma} \sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L} \| \Delta \boldsymbol {\theta} \| _ {2} \tag {109} \\ \leq \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}, \\ \end{array}
$$

upper bound $( B )$ as

$$
\begin{array}{l} h _ {\max } ^ {(\ell - 1)} \Delta d _ {\max } ^ {(\ell + 1)} \leq B _ {x} \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {L} \frac {2}{\gamma} \left((L + 1) \sqrt {d} \left(B _ {x} + 1\right) \left(\max  \left\{1, \sqrt {d} B _ {w} \right\}\right) ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \left((L + 1) C _ {1} ^ {L} C _ {2} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {110} \\ \end{array}
$$

and upper bound $( C )$ as

$$
\begin{array}{l} h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)} \leq B _ {x} (1 + \sqrt {d} B _ {w}) ^ {\ell - 1} \cdot \frac {2}{\gamma} (1 + \sqrt {d} B _ {w}) ^ {L - \ell} \cdot \sqrt {d} B _ {x} (1 + \sqrt {d} B _ {w}) ^ {\ell - 1} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \frac {2}{\gamma} \sqrt {d} B _ {x} ^ {2} \left(1 + \sqrt {d} B _ {w}\right) ^ {2 L} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \frac {2}{\gamma} \left(C _ {1} ^ {L} C _ {2}\right) ^ {2} \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {111} \\ \end{array}
$$

By combining the results above, we can upper bound $\Delta \mathbf { G } ^ { ( \ell ) }$ as

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} \leq \sqrt {d} \Big (\Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} + h _ {\max } ^ {(\ell - 1)} \Delta d _ {\max } ^ {(\ell + 1)} + h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \cdot \Delta z _ {\max } ^ {(\ell)} \Big) \\ \leq \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \left((L + 2) C _ {1} ^ {L} C _ {2} + 2\right) \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {112} \\ \end{array}
$$

By plugging the result from Lemma 12, Lemma 13, Lemma 15, we can upper bound $\Delta \mathbf { G } ^ { ( \ell ) }$ as

$$
\begin{array}{l} \left\| \Delta \mathbf {G} ^ {(\ell)} \right\| _ {2} \leq \sqrt {d} \left(\Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} + h _ {\max } ^ {(\ell - 1)} \Delta d _ {\max } ^ {(\ell + 1)}\right) \\ \leq \frac {2}{\gamma} \left(\sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + \left(1 + \sqrt {d} B _ {w}\right) + \sqrt {d}\right) \| \Delta \boldsymbol {\theta} \| _ {2} \cdot \sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L}. \tag {113} \\ \end{array}
$$

Similarly, we can upper bound the difference between gradient for the weight parameters of the binary classifier as

$$
\begin{array}{l} \left\| \mathbf {G} ^ {(L + 1)} - \tilde {\mathbf {G}} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{(a)} \frac {\gamma}{\gamma} \Delta h _ {\max } ^ {(L)} \tag {114} \\ \leq \frac {2}{\gamma} \sqrt {d} B _ {x} (1 + \sqrt {d} B _ {w}) ^ {L - 1} (\| \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \mathbf {W} ^ {(L)} \| _ {2}), \\ \end{array}
$$

where $( a )$ is due to the fact that loss function is $\frac { 2 } { \gamma }$ -Lipschitz continuous.

Therefore, by combining the above inequalites, we have

$$
\sum_ {\ell = 1} ^ {L} \left\| \Delta \mathbf {G} ^ {(\ell)} \right\| _ {2} \leq \frac {2}{\gamma} (L + 1) \left(\sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L} + \left(1 + \sqrt {d} B _ {w}\right) + \sqrt {d}\right) \left\| \Delta \boldsymbol {\theta} \right\| _ {2} \cdot \sqrt {d} B _ {x} \left(1 + \sqrt {d} B _ {w}\right) ^ {L}. \tag {115}
$$

# H Generalization bound for APPNP

In the following, we provide detailed proof on the generalization bound of APPNP. Recall that the update rule of APPNP is defined as

$$
\mathbf {H} ^ {(\ell)} = \alpha \mathbf {L H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {H} ^ {(0)}, \mathbf {H} ^ {(0)} = \mathbf {W X}. \tag {116}
$$

The training of APPNP is an empirical risk minimization with respect to weight parameters $\theta =$ $\{ \mathbf { W } ^ { ( 1 ) } , \ldots , \mathbf { W } ^ { ( L ) } , \pmb { v } \}$ , i.e.,

$$
\mathcal {L} (\boldsymbol {\theta}) = \frac {1}{m} \sum_ {i = 1} ^ {m} \Phi_ {\gamma} (- p (f \left(\mathbf {h} _ {i} ^ {(L)}\right), y _ {i})), f \left(\mathbf {h} _ {i} ^ {(L)}\right) = \tilde {\sigma} \left(\boldsymbol {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}\right), \tag {117}
$$

where ${ \bf h } _ { i } ^ { ( L ) }$ is the node representation of the ith node at the final layer, $f ( \mathbf { h } _ { i } ^ { ( L ) } )$ is the predicted label for the ith node, $\begin{array} { r } { \tilde { \sigma } ( x ) = \frac { 1 } { \exp \left( - x \right) + 1 } } \end{array}$ is the sigmoid function, and loss function $\Phi _ { \gamma } ( - p ( z , y ) )$ is $\frac { 2 } { \gamma }$ -Lipschitz continuous with respect to its first input $z$ . For simplification, we will use Loss $( z , y )$ denote $\Phi _ { \gamma } ( - p ( z , y ) )$ in the proof.

To establish the generalization of APPNP as stated in Theorem 3, we utilize the result on transductive uniform stability from [16] (Theorem 8 in Appendix F). Then, in Lemma 17, we derive the uniform stability constant for APPNP, i.e., APPNP.

Lemma 17. The uniform stability constant for APPNP is computed as $\begin{array} { r } { \epsilon _ { A P P N P } = \frac { 2 \eta \rho _ { f } G _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + } \end{array}$ 2ηρf Gf P m Tt=1(1 + $\eta L _ { F } ) ^ { t - 1 }$ where

$$
\rho_ {f} = C _ {1} B _ {w}, G _ {f} = \frac {4}{\gamma} C _ {1}, L _ {f} = \frac {4}{\gamma} C _ {1} \left(C _ {1} C _ {2} + 1\right), \tag {118}
$$

$$
C _ {1} = B _ {d} ^ {\alpha} B _ {x}, C _ {2} = \max  \{1, B _ {w} \}, B _ {d} ^ {\alpha} = (1 - \alpha) \sum_ {\ell = 1} ^ {L} (\alpha \sqrt {d}) ^ {\ell - 1} + (\alpha \sqrt {d}) ^ {L}.
$$

By plugging the result in Lemma 17 back to Theorem 8, we establish the generalization bound for APPNP.

The key idea of the proof is to decompose the change of the APPNP output into two terms (in Lemma 5) which depend on

• (Lemma 18) The maximum change of node representation $\Delta h _ { \operatorname* { m a x } } ^ { ( L ) } = \operatorname* { m a x } _ { i } \Vert [ \mathbf { H } ^ { ( L ) } - \mathbf { \Omega }$ $\tilde { \mathbf { H } } ^ { ( L ) } ] _ { i , : } \| _ { 2 }$ ,   
• (Lemma 19) The maximum node representation $h _ { \operatorname* { m a x } } ^ { ( L ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { H } ^ { ( L ) } ] _ { i , : } \| _ { 2 }$

Lemma 18 (Upper bound of $h _ { \mathrm { m a x } } ^ { ( L ) }$ for APPNP). Let suppose Assumption 1 hold. Then, the maximum node embeddings for any node at the `th layer is bounded by

$$
h _ {\max } ^ {(L)} \leq B _ {d} ^ {\alpha} B _ {x} B _ {w}, \tag {119}
$$

where $\begin{array} { r } { B _ { d } ^ { \alpha } = ( 1 - \alpha ) \sum _ { \ell = 1 } ^ { L } ( \alpha \sqrt { d } ) ^ { \ell - 1 } + ( \alpha \sqrt { d } ) ^ { L } . } \end{array}$

Lemma 19 (Upper bound of $\Delta h _ { \mathrm { m a x } } ^ { ( L ) }$ for APPNP). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta h _ {\max } ^ {(L)} \leq B _ {d} ^ {\alpha} B _ {x} \| \Delta \mathbf {W} \| _ {2}, \tag {120}
$$

where $\begin{array} { r } { \mathbf { \nabla } \cdot B _ { d } ^ { \alpha } = ( 1 - \alpha ) \sum _ { \ell = 1 } ^ { L } ( \alpha \sqrt { d } ) ^ { \ell - 1 } + ( \alpha \sqrt { d } ) ^ { L } \mathbf { \nabla } a n d \Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } . } \end{array}$ $\begin{array} { r } { B _ { d } ^ { \alpha } = ( 1 - \alpha ) \sum _ { \ell = 1 } ^ { L } ( \alpha \sqrt { d } ) ^ { \ell - 1 } + ( \alpha \sqrt { d } ) ^ { L } } \end{array}$

Then, in Lemma 20, we decompose the change of the model parameters into two terms which depend on

• The change of gradient with respect to the `th layer weight matrix $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$   
• The gradient with respect to the `th layer weight matrix $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$ ,

where $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ reflects the smoothness of APPNP model and $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$ corresponds the upper bound of gradient.

Lemma 20 (Upper bound of $\| \mathbf G ^ { ( \ell ) } \| _ { 2 } , \| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ for APPNP). Let suppose Assumption $\cdot$ hold. Then, the gradient and the maximum change between gradients on two different set of weight parameters are bounded by

$$
\begin{array}{l} \sum_ {\ell = 1} ^ {2} \left\| \mathbf {G} ^ {(\ell)} \right\| _ {2} \leq \frac {4}{\gamma} B _ {d} ^ {\alpha} B _ {x}, \\ \sum_ {\ell = 1} ^ {2} \| \Delta \mathbf {G} ^ {(1)} \| _ {2} \leq \frac {4}{\gamma} B _ {d} ^ {\alpha} B _ {x} \left(B _ {d} ^ {\alpha} B _ {x} \max  \{1, B _ {w} \} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \\ \end{array}
$$

where $\| \Delta \pmb { \theta } \| _ { 2 } = \| \pmb { v } - \tilde { \pmb { v } } \| _ { 2 } + \| \mathbf { W } - \tilde { \mathbf { W } } \| _ { 2 }$ denotes the change of two set of parameters.

Equipped with above intermediate results, we now proceed to prove Lemma 17.

Proof. Recall that our goal is to explore the impact of different GCN structures on the uniform stability constant APPNP, which is a function of $\rho _ { f }$ , $G _ { f }$ , and $L _ { f }$ . By Lemma 19, we know that the function $f$ is $\rho _ { f }$ -Lipschitz continuous, with $\rho _ { f } = B _ { d } ^ { \alpha } B _ { x }$ .

By Lemma 20, we know that the function $f$ is $L _ { f }$ -smoothness, and the gradient of each weight matrices is bounded by $G _ { f }$ , with

$$
G _ {f} \leq \frac {4}{\gamma} B _ {d} ^ {\alpha} B _ {x}, L _ {f} \leq \frac {4}{\gamma} B _ {d} ^ {\alpha} B _ {x} \Big (B _ {d} ^ {\alpha} B _ {x} \max  \{1, B _ {w} \} + 1 \Big).
$$

By plugging APPNP into Theorem 3, we obtain the generalization bound of APPNP.

# H.1 Proof of Lemma 18

By the definition of $h _ { \mathrm { m a x } } ^ { ( L ) }$

$$
\begin{array}{l} h _ {\max } ^ {(L)} = \max  _ {i} \| [ \alpha \mathbf {L} \mathbf {H} ^ {(L - 1)} + (1 - \alpha) \mathbf {H} ^ {(0)} ] _ {i,:} \| _ {2} \\ \leq \alpha \max  _ {i} \| [ \mathbf {L H} ^ {(L - 1)} ] _ {i,:} \| _ {2} + (1 - \alpha) \max  _ {i} \| [ \mathbf {H} ^ {(0)} ] _ {i,:} \| _ {2} \\ = \alpha \max  _ {i} \left\| \sum_ {j = 1} ^ {N} L _ {i, j} \mathbf {h} _ {j} ^ {(L - 1)} \right\| _ {2} + (1 - \alpha) B _ {w} B _ {x} \tag {122} \\ \leq \alpha \sqrt {d} h _ {\max} ^ {(L - 1)} + (1 - \alpha) B _ {w} B _ {x} \\ \leq_ {(b)} \left(\left(1 - \alpha\right) \sum_ {\ell = 1} ^ {L} \left(\alpha \sqrt {d}\right) ^ {\ell - 1} + \left(\alpha \sqrt {d}\right) ^ {L}\right) B _ {x} B _ {w}, \\ \end{array}
$$

where inequality $( a )$ is follows from Lemma 31 and inequality $( b )$ can be obtained by recursively applying $( a )$ .

Let denote $\begin{array} { r } { B _ { d } ^ { \alpha } = ( 1 - \alpha ) \sum _ { \ell = 1 } ^ { L } ( \alpha \sqrt { d } ) ^ { \ell - 1 } + ( \alpha \sqrt { d } ) ^ { L } } \end{array}$ , then we have

$$
h _ {\max } ^ {(L)} \leq B _ {d} ^ {\alpha} B _ {x} B _ {w}. \tag {123}
$$

# H.2 Proof of Lemma 19

By the definition of $\Delta h _ { \mathrm { m a x } } ^ { ( L ) }$ , we have

$$
\begin{array}{l} \Delta h _ {\max } ^ {(L)} = \max  _ {i} \| \mathbf {h} _ {i} ^ {(L)} - \tilde {\mathbf {h}} _ {i} ^ {(L)} \| _ {2} \\ = \max _ {i} \| [ (\alpha \mathbf {L} \mathbf {H} ^ {(L - 1)} + (1 - \alpha) \mathbf {H} ^ {(0)}) - (\alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(L - 1)} + (1 - \alpha) \tilde {\mathbf {H}} ^ {(0)}) ] _ {i,:} \| _ {2} \\ \leq \alpha \max  _ {i} \| [ \mathbf {L} \left(\mathbf {H} ^ {(L - 1)} - \tilde {\mathbf {H}} ^ {(L - 1)}\right) ] _ {i,:} \| _ {2} + (1 - \alpha) \max  _ {i} \| [ \mathbf {H} ^ {(0)} - \tilde {\mathbf {H}} ^ {(0)} ] _ {i,:} \| _ {2} \tag {124} \\ \leq_ {(a)} \alpha \sqrt {d} \Delta h _ {\max } ^ {(L - 1)} + (1 - \alpha) B _ {x} \| \Delta \mathbf {W} \| _ {2} \\ \leq_ {(b)} \left(\left(1 - \alpha\right) \sum_ {L = 1} ^ {L} \left(\alpha \sqrt {d}\right) ^ {\ell - 1} + \left(\alpha \sqrt {d}\right) ^ {\ell}\right) B _ {x} \| \Delta \mathbf {W} \| _ {2}, \\ \end{array}
$$

where inequality $( a )$ is follows from Lemma 31 and inequality $( b )$ can be obtained by recursively applying $( a )$ .

Let denote $\begin{array} { r } { B _ { d } ^ { \alpha } = ( 1 - \alpha ) \sum _ { \ell = 1 } ^ { L } ( \alpha \sqrt { d } ) ^ { \ell - 1 } + ( \alpha \sqrt { d } ) ^ { L } } \end{array}$ , then we have

$$
\Delta h _ {\max } ^ {(L)} \leq B _ {d} ^ {\alpha} B _ {x} \| \Delta \mathbf {W} \| _ {2}. \tag {125}
$$

# H.3 Proof of Lemma 20

By definition, we know that $\mathbf { H } ^ { ( L ) }$ is computed as

$$
\mathbf {H} ^ {(L)} = \left((\alpha \mathbf {L}) ^ {L} + (1 - \alpha) \left(1 + (\alpha \mathbf {L}) + \dots + (\alpha \mathbf {L}) ^ {L - 1}\right)\right) \mathbf {X} \mathbf {W}. \tag {126}
$$

Therefore, the gradient with respect to weight W is computed as

$$
\begin{array}{l} \| \mathbf {G} ^ {(1)} \| _ {2} \underset {(a)} {\leq} \frac {2}{\gamma} \max  _ {i} \left\| \frac {\partial f (\mathbf {h} _ {i})}{\partial \mathbf {W}} \right\| _ {2} \\ = \frac {2}{\gamma} \max  _ {i} \left\| \left[ \left(\left(\alpha \mathbf {L}\right) ^ {L} + (1 - \alpha) \left(1 + (\alpha \mathbf {L}) + \dots + (\alpha \mathbf {L}) ^ {L - 1}\right)\right) \mathbf {X} \right] _ {i,:} \right\| _ {2} \tag {127} \\ \leq \frac {2}{\gamma} \left((\alpha \sqrt {d}) ^ {L} + (1 - \alpha) \left(1 + (\alpha \sqrt {d}) + \dots + (\alpha \sqrt {d}) ^ {L - 1}\right)\right) B _ {x} \\ \leq \frac {2}{\gamma} B _ {d} ^ {\alpha} B _ {x}, \\ \end{array}
$$

where inequality $( a )$ is due to the fact that the loss function is $\frac { 2 } { \gamma }$ -Lipschitz continuous, and $B _ { d } ^ { \alpha } =$ $\begin{array} { r } { ( 1 - \alpha ) \sum _ { \ell = 1 } ^ { L } ( \alpha \sqrt { d } ) ^ { \ell - 1 } + ( \alpha \sqrt { d } ) ^ { L } } \end{array}$ . Besides, the gradient with respect to weight $\textbf {  { v } }$ is computed as

$$
\begin{array}{l} \| \mathbf {G} ^ {(2)} \| _ {2} = \left\| \frac {\partial \Phi_ {\gamma} (- p (f (\mathbf {h} _ {i} ^ {(L)} , y _ {i}))}{\partial \boldsymbol {v}} \right\| _ {2} \\ \leq \frac {2}{\gamma} h _ {\max } ^ {(L)} \tag {128} \\ \leq \frac {2}{\gamma} B _ {d} ^ {\alpha} B _ {x} B _ {w}. \\ \end{array}
$$

Combine the result above, we have $\begin{array} { r } { \sum _ { \ell = 1 } ^ { 2 } \| \mathbf G ^ { ( \ell ) } \| _ { 2 } \le \frac { 4 } { \gamma } B _ { d } ^ { \alpha } B _ { x } B _ { w } } \end{array}$ . Then, the difference between two gradient computed on two different set of weight parameters is computed as

$$
\begin{array}{l} \| \mathbf {G} ^ {(1)} - \tilde {\mathbf {G}} ^ {(1)} \| _ {2} \leq \frac {2}{\gamma} \max  _ {i} \left\| \frac {\partial f (\mathbf {h} _ {i} ^ {(L)})}{\partial \mathbf {h} _ {i} ^ {(L)}} \frac {\partial \mathbf {h} _ {i} ^ {(L)}}{\partial \mathbf {W}} - \frac {\partial f (\tilde {\mathbf {h}} _ {i} ^ {(L)})}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \frac {\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}}{\partial \tilde {\mathbf {W}}} \right\| _ {2} \\ = \frac {2}{\gamma} B _ {x} B _ {d} ^ {\alpha} \left\| \frac {\partial f \left(\mathbf {h} _ {i} ^ {(L)}\right)}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial f \left(\tilde {\mathbf {h}} _ {i} ^ {(L)}\right)}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \\ \leq \frac {2}{\gamma} B _ {x} B _ {d} ^ {\alpha} \left(\Delta h _ {\max } ^ {(L)} + \left(h _ {\max } ^ {(L)} + 1\right) \| \Delta \boldsymbol {v} \| _ {2}\right) \tag {129} \\ \leq \frac {2}{(b)} \frac {2}{\gamma} B _ {x} B _ {d} ^ {\alpha} \left(B _ {x} B _ {d} ^ {\alpha} \| \Delta \mathbf {W} \| _ {2} + \left(B _ {x} B _ {d} ^ {\alpha} B _ {w} + 1\right) \| \Delta \boldsymbol {v} \| _ {2}\right) \\ \leq \frac {2}{\gamma} B _ {x} B _ {d} ^ {\alpha} \left(B _ {x} B _ {d} ^ {\alpha} \max  \{1, B _ {w} \} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \\ \end{array}
$$

where inequality $( a )$ follows from Lemma 5 and inequality $( b )$ follows from Lemma 19 and Lemma 18.

Similarly, the difference of gradient w.r.t. weight $\textbf {  { v } }$ is computed as

$$
\begin{array}{l} \left\| \mathbf {G} ^ {(2)} - \tilde {\mathbf {G}} ^ {(2)} \right\| _ {2} = \left\| \frac {\partial \Phi_ {\gamma} (- p (f (\mathbf {h} _ {i} ^ {(L)} , y _ {i}))}{\partial \boldsymbol {v}} - \frac {\partial \Phi_ {\gamma} (- p (\tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)} , y _ {i}))}{\partial \tilde {\boldsymbol {v}}} \right\| _ {2} \\ \leq \frac {2}{\gamma} \Delta h _ {\max } ^ {(L)} \tag {130} \\ \leq \frac {2}{\gamma} B _ {d} ^ {\alpha} B _ {x} \| \Delta \mathbf {W} \| _ {2}. \\ \end{array}
$$

Combining result above yields $\begin{array} { r } { \sum _ { \ell = 1 } ^ { 2 } \| \mathbf { G } ^ { ( \ell ) } - \tilde { \mathbf { G } } ^ { ( \ell ) } \| _ { 2 } \leq \frac { 4 } { \gamma } B _ { x } B _ { d } ^ { \alpha } \Big ( B _ { x } B _ { d } ^ { \alpha } \operatorname* { m a x } \{ 1 , B _ { w } \} + 1 \Big ) \| \Delta \pmb { \theta } \| _ { 2 } } \end{array}$ as desired.

# I Generalization bound for GCNII

In the following, we provide generalization bound of the GCNII. Recall that the update rule of GCNII is defined as

$$
\mathbf {H} ^ {(\ell)} = \sigma \left(\left(\alpha \mathbf {L H} ^ {(\ell)} + (1 - \alpha) \mathbf {X}\right) \left(\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I} _ {N}\right)\right), \tag {131}
$$

where $\sigma ( \cdot )$ is ReLU activation function. Although ReLU function $\sigma ( x )$ is not differentiable when $x = 0$ , for analysis purpose, we suppose the $\sigma ^ { \prime } ( 0 ) = 0$ which is also used in optimization.

The training of GCNII is an empirical risk minimization with respect to a set of parameters $\theta =$ $\{ \mathbf { W } ^ { ( 1 ) } , \ldots , \mathbf { W } ^ { ( L ) } , \pmb { v } \}$ , i.e.,

$$
\mathcal {L} (\boldsymbol {\theta}) = \frac {1}{m} \sum_ {i = 1} ^ {m} \Phi_ {\gamma} (- p \left(f \left(\mathbf {h} _ {i} ^ {(L)}\right), y _ {i}\right)), f \left(\mathbf {h} _ {i} ^ {(L)}\right) = \tilde {\sigma} \left(\boldsymbol {v} ^ {\top} \mathbf {h} _ {i} ^ {(L)}\right), \tag {132}
$$

where $\mathbf { h } _ { i } ^ { ( L ) }$ is the node representation of the ith node at the final layer, $f ( \mathbf { h } _ { i } ^ { ( L ) } )$ is the predicted label for the ith node, σ˜(x) = $\begin{array} { r } { \tilde { \sigma } ( x ) = \frac { 1 } { \exp \left( - x \right) + 1 } } \end{array}$ is the sigmoid function, and loss function $\Phi _ { \gamma } ( - p ( z , y ) )$ is $\frac { 2 } { \gamma }$ -Lipschitz continuous with respect to its first input $z$ . For simplification, let Loss $( z , y )$ denote $\Phi _ { \gamma } \dot { ( } - p ( z , y ) )$ in the proof.

To establish the generalization of GCNII as stated in Theorem 3, we utilize the result on transductive uniform stability from [16] (Theorem 8 in Appendix F). Then, in Lemma 21, we derive the uniform stability constant for GCNII, i.e., $\epsilon _ { \mathrm { G C N I I } }$ .

Lemma 21. The uniform stability constant for GCNII is computed as $\eta L _ { F } ) ^ { t - 1 }$ where $\begin{array} { r } { \epsilon _ { G C N I I } = \frac { 2 \eta \rho _ { f } G _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + } \end{array}$ m PTt=1(1 +

$$
\rho_ {f} = \beta C _ {1} ^ {L} C _ {2}, G _ {f} = \beta \frac {2}{\gamma} (L + 1) C _ {1} ^ {L} C _ {2},
$$

$$
L _ {f} = \alpha \beta \frac {2}{\gamma} (L + 1) \sqrt {d} C _ {1} ^ {L} C _ {2} \left(\left(\alpha \beta L + \beta B _ {w} + 2\right) C _ {1} ^ {L} C _ {2} + 2 \beta\right), \tag {133}
$$

$$
C _ {1} = \max  \{1, \alpha \sqrt {d} B _ {w} ^ {\beta} \}, C _ {2} = \sqrt {d} + B _ {\ell , d} ^ {\alpha , \beta} B _ {x},
$$

$$
B _ {w} ^ {\beta} = \beta B _ {w} + (1 - \beta), B _ {\ell , d} ^ {\alpha , \beta} = \max  \left\{\beta \big ((1 - \alpha) L + \alpha \sqrt {d} \big), (1 - \alpha) L B _ {w} ^ {\beta} + 1 \right\}.
$$

By plugging the result in Lemma 21 back to Theorem 8, we establish the generalization bound for GCNII.

The key idea of the proof is to decompose the change of the GCNII output into two terms (in Lemma 5) which depend on

• (Lemma 22) The maximum change of node representation $\Delta h _ { \mathrm { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { H } ^ { ( \ell ) } - \mathbf { \ell }$ $\tilde { \mathbf { H } } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 }$ ,   
• (Lemma 23) The maximum node representation $h _ { \operatorname* { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { H } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 }$

Lemma 22 (Upper bound of $h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ for GCNII). Let suppose Assumption 1 hold. Then, the maximum node embeddings for any node at the `th layer is bounded by

$$
h _ {\max } ^ {(\ell)} \leq \left((1 - \alpha) \ell B _ {w} ^ {\beta} + 1\right) B _ {x} \cdot \left(\max  \{1, \alpha \sqrt {d} B _ {w} ^ {\beta} \}\right) ^ {\ell}, \tag {134}
$$

where $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ and $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) }$ .

Lemma 23 (Upper bound of $\Delta h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ for GCNII). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta h _ {\max } ^ {(\ell)} \leq \beta B _ {x} \left((1 - \alpha) \ell + \alpha \sqrt {d}\right) \left(\max  \left\{1, \alpha \sqrt {d} B _ {w} ^ {\beta} \right\}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {135}
$$

where $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ and $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } .$

Besides, in Lemma 24, we derive the upper bound on the maximum change of node embeddings before the activation function, i.e., $\begin{array} { r } { \Delta z _ { \mathrm { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { Z } ^ { ( \ell ) } - \tilde { \mathbf { Z } } ^ { ( \ell ) } ] _ { i , : } \| _ { 2 } } \end{array}$ , which will be used to derive the gradient related upper bounds.

Lemma 24 (Upper bound of $\Delta z _ { \mathrm { m a x } } ^ { ( \ell ) }$ for GCNII). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings before the activation function on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta z _ {\max } ^ {(\ell)} \leq \beta B _ {x} \left((1 - \alpha) \ell + \alpha \sqrt {d}\right) \left(\max  \left\{1, \alpha \sqrt {d} B _ {w} ^ {\beta} \right\}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {136}
$$

where $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ and $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } ,$ .

Then, in Lemma 25, we decompose the change of the model parameters into two terms which depend on

• The maximum change of gradient passing from $( \ell + 1 )$ th layer to the `th layer $\Delta d _ { \mathrm { m a x } } ^ { ( \ell ) } =$ $\begin{array} { r } { \operatorname* { m a x } _ { i } \Big \| \Big [ \frac { \partial \mathbf { H } ^ { ( \ell ) } } { \partial \mathbf { H } ^ { ( \ell - 1 ) } } - \frac { \bar { \partial \mathbf { H } } ^ { ( \ell ) } } { \partial \tilde { \mathbf { H } } ^ { ( \ell - 1 ) } } \Big ] _ { i , : } \Big \| _ { 2 } , } \end{array}$ ∂H(`) ∂H˜ (`)   
• The maximum gradient passing from $( \ell + 1 ) { \mathrm { t h } }$ layer to the `th layer $d _ { \operatorname* { m a x } } ^ { ( \ell ) } ~ =$ $\begin{array} { r } { \operatorname* { m a x } _ { i } \left\| \left[ \frac { \partial \mathbf { H } ^ { ( \ell ) } } { \partial \mathbf { H } ^ { ( \ell - 1 ) } } \right] _ { i , : } ^ { - } \right\| _ { 2 } . } \end{array}$

Lemma 25 (Upper bound of √ $\mathbf { d } _ { \mathrm { m a x } } ^ { ( \ell ) } , \Delta \mathbf { d } _ { \mathrm { m a x } } ^ { ( \ell ) }$ for GCNII). Let suppose Assumption 1 hold and let denote $C _ { 1 } = \mathrm { m a x } \{ 1 , \alpha \sqrt { d } B _ { w } ^ { \beta } \}$ , $C _ { 2 } = \sqrt { d } + B _ { \ell , d } ^ { \alpha , \beta } B _ { x }$ , $B _ { w } ^ { \beta } \ : = \ : \beta B _ { w } \ : + \ : ( 1 \ : - \ : \beta )$ , and $B _ { \ell , d } ^ { \alpha , \beta } \ =$ max $\left\{ \beta \big ( ( 1 - \alpha ) L + \alpha \sqrt { d } \big ) , ( 1 - \alpha ) L B _ { w } ^ { \beta } + 1 \right\}$ . Then, the maximum gradient passing from layer $\ell + 1$ to layer ` for any node is bounded by

$$
d _ {\max } ^ {(\ell)} \leq \frac {2}{\gamma} C _ {1} ^ {L - \ell + 1} \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {137}
$$

and the maximum change between the gradient passing from layer $\ell + 1$ to layer ` on two different set of weight parameters for any node is bounded by

$$
\Delta d _ {\max } ^ {(\ell)} \leq \frac {2}{\gamma} C _ {1} ^ {L - \ell} \left((\alpha \beta L + \beta B _ {w} + 1) C _ {1} ^ {L} C _ {2} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {138}
$$

where $\begin{array} { r } { \| \Delta \pmb { \theta } \| _ { 2 } = \| \pmb { v } - \tilde { \pmb { v } } \| _ { 2 } + \sum _ { \ell = 1 } ^ { L } \| \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } \| _ { 2 } } \end{array}$ denotes the change of two set of parameters.

Finally, in Lemma 26, we decompose the change of the model parameters into two terms which depend on

• The change of gradient with respect to the `th layer weight matrix $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ ,   
• The gradient with respect to the `th layer weight matrix $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$

where $\mathbf { G } ^ { ( L + 1 ) }$ denotes the gradient with respect to the weight $\textbf {  { v } }$ of the binary classifier and $\mathbf { G } ^ { ( \ell ) }$ denotes the gradient with respect to the weight matrices $\mathbf { W } ^ { ( \ell ) }$ of the graph convolutional layer. Notice that $\lVert \Delta \mathbf { G } ^ { ( \ell ) } \rVert _ { 2 }$ reflect the smoothness of GCN model and $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$ correspond the upper bound of gradient.

Lemma 26 (Upper bound of $\mathbf { G } ^ { ( \ell ) } , \Delta \mathbf { G } ^ { ( \ell ) }$ for GCNII). Let suppose Assumption 1 hold, and let $C _ { 1 } = \operatorname* { m a x } \lbrace 1 , \alpha \sqrt { d } B _ { w } ^ { \beta } \rbrace$ , $C _ { 2 } = \sqrt { d } + B _ { \ell , d } ^ { \alpha , \beta } B _ { x }$ , $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ , and $B _ { \ell , d } ^ { \alpha , \beta } = \operatorname* { m a x } \big \{ \beta \big ( ( 1 -$ $\alpha ) L + \alpha \sqrt { d } ) , ( 1 - \alpha ) L B _ { w } ^ { \beta } + 1 \}$ . Then, the gradient and the maximum change between gradients on two different set of weight parameters are bounded by

$$
\sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) \beta C _ {1} ^ {L} C _ {2}, \tag {139}
$$

$$
\sum_ {\ell = 1} ^ {L + 1} \| \Delta \mathbf {G} ^ {(\ell)} \| _ {2} \leq \alpha \beta \frac {2}{\gamma} (L + 1) \sqrt {d} C _ {1} ^ {L} C _ {2} \Big ((\alpha \beta L + \beta B _ {w} + 2) C _ {1} ^ {L} C _ {2} + 2 \beta \Big) \| \Delta \boldsymbol {\theta} \| _ {2},
$$

where $\begin{array} { r } { \| \Delta \pmb { \theta } \| _ { 2 } = \| \pmb { v } - \tilde { \pmb { v } } \| _ { 2 } + \sum _ { \ell = 1 } ^ { L } \| \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } \| _ { 2 } } \end{array}$ denotes the change of two set of parameters.

Equipped with above intermediate results, we now proceed to prove Lemma 21.

Proof. Recall that our goal is to explore the impact of different GCN structures on the uniform stability constant √ $\epsilon _ { \mathrm { G C N I I } }$ , which is a function of ρf , $G _ { f }$ , and $L _ { f }$ . Let denote $C _ { 1 } = \operatorname* { m a x } \lbrace 1 , \alpha \sqrt { d } B _ { w } ^ { \beta } \rbrace$ , $C _ { 2 } = \sqrt { d } + B _ { \ell , d } ^ { \alpha , \beta } B _ { x } , B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ , and $B _ { \ell , d } ^ { \alpha , \beta } = \operatorname* { m a x } \left\{ \beta \left( ( 1 - \alpha ) L + \alpha \sqrt { d } \right) , ( 1 - \alpha ) L B _ { w } ^ { \beta } + 1 \right\}$ By Lemma 23, we know that the function $f$ is $\rho _ { f }$ -Lipschitz continuous, with

$$
\rho_ {f} = \beta C _ {1} ^ {L} C _ {2}. \tag {140}
$$

By Lemma 26, we know that the function $f$ is $L _ { f }$ -smoothness, and the gradient of each weight matrices is bounded by $G _ { f }$ , with

$$
G _ {f} = \frac {2}{\gamma} (L + 1) \beta C _ {1} ^ {L} C _ {2}, \tag {141}
$$

$$
L _ {f} = \alpha \beta \frac {2}{\gamma} (L + 1) \sqrt {d} C _ {1} ^ {L} C _ {2} \Big ((\alpha \beta L + \beta B _ {w} + 2) C _ {1} ^ {L} C _ {2} + 2 \beta \Big).
$$

where Bα,β`,d = $B _ { \ell , d } ^ { \alpha , \beta } = \operatorname* { m a x } \Big \{ \beta \big ( ( 1 - \alpha ) L + \alpha \sqrt { d } \big ) , ( 1 - \alpha ) L B _ { w } ^ { \beta } + 1 \Big \} .$ )

By plugging $\epsilon _ { \mathrm { G C N I I } }$ into Theorem 3, we obtain the generalization bound of GCNII.

![](images/0a8bf247a6162c2289a8282daf6052fc9ca80f7f5cf84f0977613d7e143ecb9d.jpg)

# I.1 Proof of Lemma 22

By the definition of $h _ { \operatorname* { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} h _ {\max } ^ {(\ell)} = \max  _ {i} \| [ \sigma (\left(\alpha \mathbf {L H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}\right) (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I})) ] _ {i, :} \| _ {2} \\ \leq \max  _ {i} \| [ (\alpha \mathbf {L H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}) (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}) ] _ {i,:} \| _ {2} \\ \leq \max _ {i} \| [ (\alpha \mathbf {L H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}) ] _ {i,:} \| _ {2} \| \beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I} \| _ {2} \\ \leq \left(\beta B _ {w} + (1 - \beta)\right) \cdot \max  _ {i} \| \left[ \left(\alpha \mathbf {L} \mathbf {H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}\right) \right] _ {i,:} \| _ {2} \tag {142} \\ \leq \left(\beta B _ {w} + (1 - \beta)\right) \cdot \left(\alpha \max _ {i} \| [ \mathbf {L H} ^ {(\ell - 1)} ] _ {i,:} \| _ {2} + (1 - \alpha) \max _ {i} \| [ \mathbf {X} ] _ {i,:} \| _ {2}\right) \\ \leq \left(\beta B _ {w} + (1 - \beta)\right) \cdot \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}\right) \\ = \alpha \sqrt {d} (\beta B _ {w} + (1 - \beta)) h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x} (\beta B _ {w} + (1 - \beta)), \\ \end{array}
$$

where inequality $( a )$ is due to $\| \sigma ( x ) \| _ { 2 } \leq \| x \| _ { 2 }$ .

Let $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ . By induction, we have

$$
h _ {\max } ^ {(\ell)} \leq \left((1 - \alpha) B _ {x} B _ {w} ^ {\beta}\right) \sum_ {\ell^ {\prime} = 1} ^ {\ell} \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell^ {\prime} - 1} + \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell} B _ {x}. \tag {143}
$$

If $\alpha \sqrt { d } B _ { w } ^ { \beta } \leq 1$ we have,

$$
h _ {\max } ^ {(\ell)} \leq (1 - \alpha) B _ {x} B _ {w} ^ {\beta} \ell + B _ {x} = \left((1 - \alpha) \ell B _ {w} ^ {\beta} + 1\right) B _ {x}, \tag {144}
$$

and if $\alpha \sqrt { d } B _ { w } ^ { \beta } > 1$ we have

$$
\begin{array}{l} h _ {\max } ^ {(\ell)} \leq \left((1 - \alpha) B _ {x} B _ {w} ^ {\beta}\right) \ell \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell - 1} + \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell} B _ {x} \tag {145} \\ \leq \left((1 - \alpha) \ell B _ {w} ^ {\beta} + 1\right) B _ {x} \cdot \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell}. \\ \end{array}
$$

By combining results above, we have

$$
h _ {\max } ^ {(\ell)} \leq \left((1 - \alpha) \ell B _ {w} ^ {\beta} + 1\right) B _ {x} \cdot \left(\max  \{1, \alpha \sqrt {d} B _ {w} ^ {\beta} \}\right) ^ {\ell}. \tag {146}
$$

# I.2 Proof of Lemma 23

By the definition of $\Delta h _ { \mathrm { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} \Delta h _ {\mathrm {m a x}} ^ {(\ell)} = \max _ {i} \| \mathbf {h} _ {i} ^ {(\ell)} - \tilde {\mathbf {h}} _ {i} ^ {(\ell)} \| _ {2} \\ = \max  _ {i} \left\| \left[ \sigma \left(\left(\alpha \mathbf {L} \mathbf {H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}\right) \left(\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}\right)\right) \right. \right. \\ \left. - \sigma \Big (\big (\alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} \big) \big (\beta \tilde {\mathbf {W}} ^ {(\ell)} + (1 - \beta) \mathbf {I} \big) \Big) \right] _ {i,:} \| _ {2} \\ \leq \max  _ {i} \left\| \left[ (\alpha \mathbf {L} \mathbf {H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}) (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}) \right. \right. \\ \left. - \left(\alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}\right) \left(\beta \tilde {\mathbf {W}} ^ {(\ell)} + (1 - \beta) \mathbf {I}\right) \right] _ {i,:} \| _ {2} \\ \leq \max  _ {i} \left\| \left[ \left(\alpha \mathbf {L H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X}\right) \left(\beta \left(\mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)}\right)\right) \right] _ {i,:} \right\| _ {2} \\ + \max  _ {i} \left\| \left[ \left(\alpha \mathbf {L} \left(\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}\right)\right) \left(\beta \tilde {\mathbf {W}} ^ {(\ell)} + (1 - \beta) \mathbf {I}\right) \right] _ {i,:} \right\| _ {2} \\ \leq \max _ {i} \| [ \alpha \mathbf {L H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] _ {i,:} \| _ {2} \cdot \beta \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \alpha (\beta B _ {w} + (1 - \beta)) \max _ {i} \| [ \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) ] _ {i,:} \| _ {2} \\ \leq \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}\right) \cdot \beta \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \alpha \left(\beta B _ {w} + (1 - \beta)\right) \sqrt {d} \Delta h _ {\max } ^ {(\ell - 1)}, \tag {147} \\ \end{array}
$$

where $( a )$ is due to the fact that $\| \sigma ( x ) \| _ { 2 } \leq \| x \| _ { 2 }$ .

Let $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ . Then, by induction we have

$$
\begin{array}{l} \Delta h _ {\max } ^ {(\ell)} \leq \alpha \sqrt {d} B _ {w} ^ {\beta} \cdot \Delta h _ {\max } ^ {(\ell - 1)} + \beta \Big (\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x} \Big) \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq (\alpha \sqrt {d} B _ {w} ^ {\beta}) ^ {2} \cdot \Delta h _ {\max} ^ {(\ell - 2)} + \beta \Big (\alpha \sqrt {d} h _ {\max} ^ {(\ell - 1)} + (1 - \alpha) B _ {x} \Big) \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ + \beta (\alpha \sqrt {d} B _ {w} ^ {\beta}) \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 2)} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} \\ \dots \tag {148} \\ \leq \beta \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ + \beta (\alpha \sqrt {d} B _ {w} ^ {\beta}) \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 2)} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \dots \\ + \beta (\alpha \sqrt {d} B _ {w} ^ {\beta}) ^ {\ell - 1} \left(\alpha \sqrt {d} h _ {\max } ^ {(0)} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(1)} \| _ {2}. \\ \end{array}
$$

If $\alpha \sqrt { d } B _ { w } ^ { \beta } \leq 1$ , using Lemma 23 we have

$$
\begin{array}{l} \Delta h _ {\max } ^ {(\ell)} \leq \beta \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \beta \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 2)} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \dots \\ + \beta \left(\alpha \sqrt {d} h _ {\max } ^ {(0)} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(1)} \| _ {2} \\ \leq \beta B _ {x} \left((1 - \alpha) \ell + \alpha \sqrt {d}\right) \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right), \tag {149} \\ \end{array}
$$

and if $\alpha \sqrt { d } B _ { w } ^ { \beta } > 1$ , using Lemma 23 we have

$$
\begin{array}{l} \Delta h _ {\max } ^ {(\ell)} \leq \beta \left(\alpha \sqrt {d} \left((1 - \alpha) (\ell - 1) B _ {w} ^ {\beta} + 1\right) B _ {x} \cdot \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell - 1} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ + \beta (\alpha \sqrt {d} B _ {w} ^ {\beta}) \left(\alpha \sqrt {d} \left((1 - \alpha) (\ell - 2) B _ {w} ^ {\beta} + 1\right) B _ {x} \cdot \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell - 2} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(\ell - 1)} \| _ {2} + \dots \\ + \beta (\alpha \sqrt {d} B _ {w} ^ {\beta}) ^ {\ell - 1} \left(\alpha \sqrt {d} B _ {x} + (1 - \alpha) B _ {x}\right) \| \Delta \mathbf {W} ^ {(1)} \| _ {2} \\ \leq \beta B _ {x} \left((1 - \alpha) \ell + \alpha \sqrt {d}\right) \left(\alpha \sqrt {d} B _ {w} ^ {\beta}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right). \tag {150} \\ \end{array}
$$

By combining the result above, we have

$$
\Delta h _ {\max } ^ {(\ell)} \leq \beta B _ {x} \left((1 - \alpha) \ell + \alpha \sqrt {d}\right) \left(\max  \left\{1, \alpha \sqrt {d} B _ {w} ^ {\beta} \right\}\right) ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right). \tag {151}
$$

# I.3 Proof of Lemma 24

The proof is similar to the proof of Lemma 23.

# I.4 Proof of Lemma 25

Let $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ . By the definition of $d _ { \operatorname* { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} d _ {\mathrm {m a x}} ^ {(\ell)} = \max _ {i} \| [ \mathbf {D} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ = \alpha \max  _ {i} \| [ \mathbf {L} ^ {\top} \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \odot \mathbf {D} ^ {(\ell + 1)} (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}) ] _ {i,:} \| _ {2} \\ \leq \alpha \max  _ {i} \| [ \mathbf {L} ^ {\top} \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \odot \mathbf {D} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \cdot B _ {w} ^ {\beta} \tag {152} \\ \underset {(a)} {\leq} \alpha \sqrt {d} B _ {w} ^ {\beta} \cdot d _ {\max } ^ {(\ell + 1)} \\ \leq \frac {2}{\gamma} \left(\max  \left\{1, \alpha \sqrt {d} B _ {w} ^ {\beta} \right\}\right) ^ {L - \ell + 1}, \\ \end{array}
$$

where inequality $( a )$ is due to the gradient of ReLU activation is element-wise either 0 and 1.

By the definition of $\Delta d _ { \mathrm { m a x } } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} = \max  _ {i} \| [ \mathbf {D} ^ {(\ell)} - \tilde {\mathbf {D}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \max  _ {i} \alpha \| \left[ \mathbf {L} ^ {\top} \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \odot \mathbf {D} ^ {(\ell + 1)} \big (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I} \big) - \mathbf {L} ^ {\top} \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)}) \odot \tilde {\mathbf {D}} ^ {(\ell + 1)} \big (\beta \tilde {\mathbf {W}} ^ {(\ell)} - (1 - \beta) \mathbf {I} \big) \right] _ {i,:} \| _ {2} \\ \stackrel {<  } {\underset {(a)} {\max}} \alpha \| [ \mathbf {L} ^ {\top} \mathbf {D} ^ {(\ell + 1)} (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}) - \mathbf {L} ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}) ] _ {i,:} \| _ {2} \\ + \max _ {i} \alpha \| [ \mathbf {L} ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}) ] _ {i,:} - \mathbf {L} ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} (\beta \tilde {\mathbf {W}} ^ {(\ell)} - (1 - \beta) \mathbf {I}) ] _ {i,:} \| _ {2} \\ + \alpha \max  _ {i} \| [ \mathbf {L} ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} (\beta \tilde {\mathbf {W}} ^ {(\ell)} - (1 - \beta) \mathbf {I}) ] _ {i,:} \| _ {2} \max  _ {i} \| [ \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \max  _ {i} \alpha \| [ \mathbf {L} ^ {\top} (\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)}) (\beta \mathbf {W} ^ {(\ell)} + (1 - \beta) \mathbf {I}) ] _ {i,:} \| _ {2} \\ + \max  _ {i} \alpha \| [ \mathbf {L} ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} (\beta (\mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)})) ] _ {i,:} \| _ {2} \\ + \alpha \max  _ {i} \| [ \mathbf {L} ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} (\beta \tilde {\mathbf {W}} ^ {(\ell)} - (1 - \beta) \mathbf {I}) ] _ {i,:} \| _ {2} \max  _ {i} \| [ \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} ] _ {i,:} \| _ {2} \\ \leq \alpha \sqrt {d} B _ {w} ^ {\beta} \cdot \Delta d _ {\max } ^ {(\ell + 1)} + \alpha \beta \sqrt {d} d _ {\max } ^ {(\ell + 1)} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \alpha \sqrt {d} B _ {w} ^ {\beta} d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)} \\ = \alpha \sqrt {d} B _ {w} ^ {\beta} \cdot \Delta d _ {\max } ^ {(\ell + 1)} + \underbrace {\alpha d _ {\max } ^ {(\ell + 1)} \left(\beta \sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + \sqrt {d} B _ {w} ^ {\beta} \Delta z _ {\max } ^ {(\ell)}\right)} _ {(A)}, \tag {153} \\ \end{array}
$$

where inequality $( a )$ and $( b )$ is due to the gradient of ReLU activation is element-wise either 0 and 1.

Let $C _ { 1 } = \operatorname* { m a x } \lbrace 1 , \alpha \sqrt { d } B _ { w } ^ { \beta } \rbrace$ and $B _ { \ell , d } ^ { \alpha , \beta } = \operatorname* { m a x } \left. \beta \big ( ( 1 - \alpha ) L + \alpha \sqrt { d } \big ) , ( 1 - \alpha ) L B _ { w } ^ { \beta } + 1 \right.$ . Recall that $\begin{array} { r } { d _ { \mathrm { m a x } } ^ { ( \ell + 1 ) } \le \frac { 2 } { \gamma } ( \operatorname* { m a x } \{ 1 , \alpha \sqrt { d } B _ { w } ^ { \beta } \} ) ^ { L - \ell } } \end{array}$ and the upper bound of $\Delta z _ { \mathrm { m a x } } ^ { ( \ell ) }$ in Lemma 24, we have

$$
d _ {\max } ^ {(\ell + 1)} \leq \frac {2}{\gamma} C _ {1} ^ {L - \ell}, \Delta z _ {\max } ^ {(\ell)} \leq \beta B _ {x} B _ {\ell , d} ^ {\alpha , \beta} C _ {1} ^ {\ell - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right). \tag {154}
$$

Therefore, we have upper bound $( A )$ by

$$
\begin{array}{l} \alpha \sqrt {d} d _ {\max } ^ {(\ell + 1)} \left(\beta \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + B _ {w} ^ {\beta} \Delta z _ {\max } ^ {(\ell)}\right) \\ \leq \alpha \beta \cdot \frac {2}{\gamma} C _ {1} ^ {L - \ell} \cdot \left(\sqrt {d} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} + B _ {x} B _ {\ell , d} ^ {\alpha , \beta} C _ {1} ^ {\ell} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2}\right)\right) \tag {155} \\ \leq \alpha \beta \frac {2}{\gamma} C _ {1} ^ {L} (\sqrt {d} + B _ {\ell , d} ^ {\alpha , \beta} B _ {x}) \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \alpha \beta \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}, \\ \end{array}
$$

where $C _ { 2 } = \sqrt { d } + B _ { \ell , d } ^ { \alpha , \beta } B _ { x }$ and $B _ { \ell , d } ^ { \alpha , \beta } = \operatorname* { m a x } \left\{ \beta \big ( ( 1 - \alpha ) L + \alpha \sqrt { d } \big ) , ( 1 - \alpha ) L B _ { w } ^ { \beta } + 1 \right\}$ . By plugging it back and by induction, we have

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} \leq C _ {1} \Delta d _ {\max } ^ {(\ell + 1)} + \alpha \beta \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq C _ {1} ^ {2} \Delta d _ {\max } ^ {(\ell + 2)} + \left(1 + C _ {1}\right) \cdot \alpha \beta \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2} \tag {156} \\ \leq C _ {1} ^ {L - \ell + 1} \Delta d _ {\max } ^ {(L + 1)} + (1 + C _ {1} + \dots + C _ {1} ^ {L - \ell + 1}) \cdot \alpha \beta \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq C _ {1} ^ {L - \ell} \left(\Delta d _ {\max } ^ {(L + 1)} + \alpha \beta \frac {2}{\gamma} L C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}\right). \\ \end{array}
$$

Let first explicit upper bound $\Delta d _ { \mathrm { m a x } } ^ { ( L + 1 ) }$ . By definition, $\Delta d _ { \mathrm { m a x } } ^ { ( L + 1 ) }$

$$
\begin{array}{l} \Delta d _ {\max } ^ {(L + 1)} = \max  _ {i} \| \mathbf {d} _ {i} ^ {(L + 1)} - \tilde {\mathbf {d}} _ {i} ^ {(L + 1)} \| _ {2} \\ = \max  _ {i} \left\| \frac {\partial \operatorname {L o s s} (f \left(\mathbf {h} _ {i} ^ {(L)}\right) , y _ {i})}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial \operatorname {L o s s} (\tilde {f} (\tilde {\mathbf {h}} _ {i} ^ {(L)} , y _ {i})}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \\ \leq \frac {2}{\gamma} \max  _ {i} \left\| \frac {\partial f \left(\mathbf {h} _ {i} ^ {(L)}\right)}{\partial \mathbf {h} _ {i} ^ {(L)}} - \frac {\partial f \left(\tilde {\mathbf {h}} _ {i} ^ {(L)}\right)}{\partial \tilde {\mathbf {h}} _ {i} ^ {(L)}} \right\| _ {2} \\ \leq \frac {2}{\gamma} \left(\Delta h _ {\max } ^ {(L)} + \left(h _ {\max } ^ {(L)} + 1\right) \Delta \boldsymbol {v}\right) \\ = \frac {2}{\gamma} \left[ B _ {x} B _ {\ell , d} ^ {\alpha , \beta} C _ {1} ^ {L - 1} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(L)} \| _ {2}\right) + \left(B _ {x} B _ {\ell , d} ^ {\alpha , \beta} C _ {1} ^ {L} + 1\right) \| \Delta \boldsymbol {v} \| _ {2} \right] \\ \leq \frac {2}{\gamma} \left(B _ {\ell} ^ {\alpha} B _ {x} C _ {1} ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} \\ = \frac {2}{\gamma} \left(B _ {\ell} ^ {\alpha} B _ {x} C _ {1} ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {157} \\ \end{array}
$$

where $C _ { 1 } = \operatorname* { m a x } \lbrace 1 , \alpha \sqrt { d } B _ { w } ^ { \beta } \rbrace$ , $B _ { w } ^ { \beta } = \beta B _ { w } + ( 1 - \beta )$ and $B _ { \ell , d } ^ { \alpha , \beta } = \operatorname* { m a x } \Big \{ \beta \big ( ( 1 - \alpha ) L + \alpha \sqrt { d } \big ) , ( 1 -$ $\alpha ) L B _ { w } ^ { \beta } + 1 \biggr \}$ . By plugging it back, we have

$$
\begin{array}{l} \Delta d _ {\max } ^ {(\ell)} \leq C _ {1} ^ {L - \ell} \left(\frac {2}{\gamma} \left(B _ {\ell , d} ^ {\alpha , \beta} B _ {x} C _ {1} ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} + \alpha \beta \frac {2}{\gamma} L C _ {1} ^ {L} (\sqrt {d} + B _ {\ell , d} ^ {\alpha , \beta} B _ {x}) \| \Delta \boldsymbol {\theta} \| _ {2}\right) \\ \leq C _ {1} ^ {L - \ell} \frac {2}{\gamma} \left(B _ {\ell , d} ^ {\alpha , \beta} B _ {x} C _ {1} ^ {L} + 1 + \alpha \beta L C _ {1} ^ {L} \left(\sqrt {d} + B _ {\ell , d} ^ {\alpha , \beta} B _ {x}\right)\right) \| \Delta \boldsymbol {\theta} \| _ {2} \tag {158} \\ \leq C _ {1} ^ {L - \ell} \frac {2}{\gamma} \left((\alpha \beta L + 1) (\sqrt {d} + B _ {\ell , d} ^ {\alpha , \beta} B _ {x}) C _ {1} ^ {L} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} \\ = C _ {1} ^ {L - \ell} \frac {2}{\gamma} \left(\left(\alpha \beta L + 1\right) C _ {1} ^ {L} C _ {2} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}. \\ \end{array}
$$

# I.5 Proof of Lemma 26

By the definition of $\mathbf { G } ^ { ( \ell ) }$ , we have

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} \| _ {2} = \frac {1}{m} \left\| \boldsymbol {\beta} \cdot [ \alpha \mathbf {L} \mathbf {H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \mathbf {D} ^ {(\ell)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) \right\| _ {2} \\ \leq \frac {1}{m} \beta \left\| \left[ \alpha \mathbf {L} \mathbf {H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} \right] ^ {\top} \mathbf {D} ^ {(\ell)} \right\| _ {2} \tag {159} \\ \leq \beta \Big (\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x} \Big) d _ {\max } ^ {(\ell)}. \\ \end{array}
$$

Plugging in the results from Lemma 22 and Lemma 25, we have

$$
\begin{array}{l} \left\| \mathbf {G} ^ {(\ell)} \right\| _ {2} \leq \frac {2}{\gamma} \beta B _ {x} \left(\alpha \sqrt {d} \left((1 - \alpha) (\ell - 1) B _ {w} ^ {\beta} + 1\right) \cdot C _ {1} ^ {\ell - 1} + (1 - \alpha)\right) C _ {1} ^ {L - \ell + 1} \tag {160} \\ \leq \frac {2}{\gamma} \beta B _ {x} \big ((1 - \alpha) \ell + \alpha \sqrt {d} \big) C _ {1} ^ {L} \leq \frac {2}{\gamma} \beta C _ {1} ^ {L} C _ {2}, \\ \end{array}
$$

where $C _ { 2 } = \sqrt { d } + B _ { \ell , d } ^ { \alpha , \beta } B _ { x }$ . Similarly, by the definition of $\mathbf { G } ^ { ( L + 1 ) }$ , we have

$$
\left\| \mathbf {G} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{\gamma} h _ {\max } ^ {(L)} \leq \frac {2}{\gamma} C _ {1} ^ {L} C _ {2}. \tag {161}
$$

Therefore, we have

$$
\sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) \beta C _ {1} ^ {L} C _ {2}. \tag {162}
$$

Furthermore, we can upper bound the difference between gradient for $\ell \in [ L ]$ as

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} = \frac {1}{m} \left\| \beta [ \alpha \mathbf {L} \mathbf {H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - \beta [ \alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)}) \right\| _ {2} \\ \leq \frac {1}{(a)} m \beta \| [ \alpha \mathbf {L} \mathbf {H} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} - [ \alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} \| _ {2} \\ + \frac {1}{m} \beta \| [ \alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} - [ \alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \| _ {2} \\ + \frac {1}{m} \beta \| [ \alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} \odot \left(\sigma^ {\prime} (\mathbf {Z} ^ {(\ell)}) - \sigma^ {\prime} (\tilde {\mathbf {Z}} ^ {(\ell)})\right) \| _ {2} \\ \leq_ {(b)} \beta \Big (\alpha \max  _ {i} \| [ [ \mathbf {L} (\mathbf {H} ^ {(\ell - 1)} - \tilde {\mathbf {H}} ^ {(\ell - 1)}) ] ^ {\top} \mathbf {D} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \\ + \max  _ {i} \| [ [ \alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} (\mathbf {D} ^ {(\ell + 1)} - \tilde {\mathbf {D}} ^ {(\ell + 1)}) ] _ {i,:} \| _ {2} + \\ + \max  _ {i} \| [ [ \alpha \mathbf {L} \tilde {\mathbf {H}} ^ {(\ell - 1)} + (1 - \alpha) \mathbf {X} ] ^ {\top} \tilde {\mathbf {D}} ^ {(\ell + 1)} ] _ {i,:} \| _ {2} \max  _ {i} \| [ \mathbf {Z} ^ {(\ell)} - \tilde {\mathbf {Z}} ^ {(\ell)} ] _ {i,:} \| _ {2}) \\ \leq \beta \Big (\underbrace {\alpha \sqrt {d} \Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)}} _ {(A)} + \underbrace {(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}) \Delta d _ {\max } ^ {(\ell + 1)}} _ {(B)} \\ + \underbrace {\left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}\right) d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)}} _ {(C)}, \tag {163} \\ \end{array}
$$

where inequality $( a )$ and $( b )$ is due to the gradient of ReLU activation is element-wise either 0 or 1.

We can bound $( A )$ as

$$
\alpha \sqrt {d} \Delta h _ {\max } ^ {(\ell - 1)} d _ {\max } ^ {(\ell + 1)} \leq \alpha \beta \sqrt {d} \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {164}
$$

To upper bound $( B )$ and $( C )$ , let first consider the upper bound of the following term

$$
\begin{array}{l} \alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x} \leq \alpha \sqrt {d} \left((1 - \alpha) (\ell - 1) B _ {w} ^ {\beta} + 1\right) B _ {x} C _ {1} ^ {\ell - 1} + (1 - \alpha) B _ {x} \\ \leq \alpha \sqrt {d} \left((1 - \alpha) \ell B _ {w} ^ {\beta} + 1\right) B _ {x} C _ {1} ^ {\ell - 1} \tag {165} \\ \leq \alpha \sqrt {d} C _ {1} ^ {\ell - 1} C _ {2}. \\ \end{array}
$$

Therefore, we can bound $( B )$ as

$$
\begin{array}{l} \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}\right) \Delta d _ {\max } ^ {(\ell + 1)} \leq \alpha \sqrt {d} C _ {1} ^ {\ell - 1} C _ {2} \cdot C _ {1} ^ {L - \ell} \frac {2}{\gamma} \left((\alpha \beta L + 1) C _ {1} ^ {L} C _ {2} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2} \\ \leq \alpha \sqrt {d} \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \left(\left(\alpha \beta L + 1\right) C _ {1} ^ {L} C _ {2} + 1\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {166} \\ \end{array}
$$

and we can bound $( C )$ by

$$
\begin{array}{l} \left(\alpha \sqrt {d} h _ {\max } ^ {(\ell - 1)} + (1 - \alpha) B _ {x}\right) d _ {\max } ^ {(\ell + 1)} \Delta z _ {\max } ^ {(\ell)} \leq \alpha \sqrt {d} C _ {1} ^ {\ell - 1} C _ {2} \cdot \frac {2}{\gamma} C _ {1} ^ {L - \ell} \cdot \beta C _ {1} ^ {\ell - 1} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2} \tag {167} \\ \leq \alpha \beta \sqrt {d} \frac {2}{\gamma} \left(C _ {1} ^ {L} C _ {2}\right) ^ {2} \| \Delta \boldsymbol {\theta} \| _ {2}. \\ \end{array}
$$

By combining result above, we have

$$
\left\| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \right\| _ {2} \leq \alpha \beta \frac {2}{\gamma} \sqrt {d} C _ {1} ^ {L} C _ {2} \left(2 \beta + (\alpha \beta L + 1) C _ {1} ^ {L} C _ {2}\right). \tag {168}
$$

Similarly, we can upper bound the difference between gradient for the weight of the binary classifier as

$$
\left\| \mathbf {G} ^ {(L + 1)} - \tilde {\mathbf {G}} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{\gamma} \Delta h _ {\max } ^ {(L)} \leq \beta \frac {2}{\gamma} C _ {1} ^ {L} C _ {2} \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {169}
$$

which is upper bound by the right hand side of the previous equation. Therefore, we have

$$
\sum_ {\ell = 1} ^ {L + 1} \left\| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \right\| _ {2} \leq \alpha \beta \frac {2}{\gamma} (L + 1) \sqrt {d} C _ {1} ^ {L} C _ {2} \left(2 \beta + (\alpha \beta L + 1) C _ {1} ^ {L} C _ {2}\right). \tag {170}
$$

# J Generalization bound for DGCN

In the following, we provide proof of the generalization bound of DGCN in Theorem 5. Recall that the update rule of DGCN is defined as

$$
\mathbf {Z} = \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \mathbf {H} ^ {(\ell)}, \mathbf {H} ^ {(\ell)} = \mathbf {L} ^ {\ell} \mathbf {X} \left(\beta_ {\ell} \mathbf {W} ^ {(\ell)} + (1 - \beta_ {\ell}) \mathbf {I}\right). \tag {171}
$$

The training of DGCN is an empirical risk minimization with respect to a set of parameters $\theta =$ $\{ \mathbf { W } ^ { ( 1 ) } , \ldots , \mathbf { W } ^ { ( L ) } , \mathbf { v } \}$ , i.e.,

$$
\mathcal {L} (\boldsymbol {\theta}) = \frac {1}{m} \sum_ {i = 1} ^ {m} \Phi_ {\gamma} (- p (f (\mathbf {z} _ {i}), y _ {i})), f (\mathbf {z} _ {i}) = \tilde {\sigma} (\mathbf {v} ^ {\top} \mathbf {z} _ {i}), \tag {172}
$$

where $\mathbf { h } _ { i } ^ { ( L ) }$ is the node representation of the ith node at the final layer, $f ( \mathbf { h } _ { i } ^ { ( L ) } )$ is the predicted label for the ith node, $\begin{array} { r } { \tilde { \sigma } ( x ) = \frac { 1 } { \exp \left( - x \right) + 1 } } \end{array}$ 1exp(−x)+1 is the sigmoid function, and loss function Φγ(−p(z, y)) $\Phi _ { \gamma } ( - p ( z , y ) )$ is $\textstyle { \frac { 2 } { \gamma } }$ -Lipschitz continuous with respect to its first input $z$ . For simplification, let Loss $( z , y )$ denote $\Phi _ { \gamma } \dot { ( } - p ( z , y ) )$ in the proof.

For analysis purpose, we suppose $\alpha _ { \ell }$ and $\beta _ { \ell }$ are hyper-parameters that are pre-selected before training and fixed during the training phase. However, in practice, these two parameters are tuned during the training phase.

To establish the generalization of DGCN as stated in Theorem 3, we utilize the result on transductive uniform stability from [16] (Theorem 8 in Appendix F). Then, in Lemma 27, we derive the uniform stability constant for DGCN, i.e., DGCN.

Lemma 27. The uniform stability constant for DGCN is computed as $\begin{array} { r } { \epsilon _ { D G C N } = \frac { 2 \eta \rho _ { f } G _ { f } } { m } \sum _ { t = 1 } ^ { T } ( 1 + } \end{array}$ 2ηρf Gf P m $\eta L _ { F } ) ^ { t - 1 }$ where

$$
\begin{array}{l} \rho_ {f} = (\sqrt {d}) ^ {L} B _ {x}, G _ {f} = \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x}, \tag {173} \\ L _ {f} = \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x} \Big ((\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \} + 1 \Big). \\ \end{array}
$$

By plugging the result in Lemma 27 back to Theorem 8, we establish the generalization bound for DGCN.

The key idea of the proof is to decompose the change of the network output into two terms (in Lemma 5) which depend on

• (Lemma 28) The maximum change of node representation $\Delta z _ { \operatorname* { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { Z } - \tilde { \mathbf { Z } } ] _ { i , : } \| _ { 2 }$   
$z _ { \mathrm { m a x } } ^ { ( \ell ) } = \operatorname* { m a x } _ { i } \| [ \mathbf { Z } ] _ { i , : } \| _ { 2 }$

Lemma 28 (Upper bound of $z _ { \mathrm { m a x } }$ for DGCN). Let suppose Assumption 1 hold. Then, the maximum node embeddings for any node at the `th layer is bounded by

$$
z _ {\max } \leq (\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \}. \tag {174}
$$

Lemma 29 (Upper bound of $\Delta z _ { \mathrm { m a x } }$ for DGCN). Let suppose Assumption 1 hold. Then, the maximum change between the node embeddings on two different set of weight parameters for any node at the `th layer is bounded by

$$
\Delta z _ {\max } \leq (\sqrt {d}) ^ {L} B _ {x} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(L)} \| _ {2}\right), \tag {175}
$$

where $\Delta \mathbf { W } ^ { ( \ell ) } = \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) }$ .

Then, in Lemma 30, we decompose the change of the model parameters into two terms which depend on

• The change of gradient with respect to the `th layer weight matrix $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ ,   
• The gradient with respect to the `th layer weight matrix $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$

where $\| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ reflect the smoothness of APPNP model and $\| \mathbf G ^ { ( \ell ) } \| _ { 2 }$ correspond the upper bound of gradient.

Lemma 30 (Upper bound of $\| \mathbf G ^ { ( \ell ) } \| _ { 2 } , \| \Delta \mathbf G ^ { ( \ell ) } \| _ { 2 }$ for DGCN). Let suppose Assumption 1 hold. Then, the gradient and the maximum change between gradients on two different set of weight parameters are bounded by

$$
\begin{array}{l} \sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x}, \\ \sum_ {\ell = 1} ^ {L + 1} \| \Delta \mathbf {G} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x} \left((\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \} + 1)\right) \| \Delta \boldsymbol {\theta} \| _ {2}, \tag {176} \\ \end{array}
$$

where $\begin{array} { r } { \| \Delta \pmb { \theta } \| _ { 2 } = \| \mathbf { v } - \tilde { \mathbf { v } } \| _ { 2 } + \sum _ { \ell = 1 } ^ { L } \| \mathbf { W } ^ { ( \ell ) } - \tilde { \mathbf { W } } ^ { ( \ell ) } \| _ { 2 } } \end{array}$ denotes the change of two set of parameters.

Equipped with above intermediate results, we now proceed to prove Lemma 27.

Proof. Recall that our goal is to explore the impact of different GCN structures on the uniform stability constant $\epsilon _ { \mathrm { D G C N } }$ , which is a function of $\rho _ { f }$ , $G _ { f }$ , and $L _ { f }$ . By Lemma 29, we know that the function $f$ is $\rho _ { f }$ -Lipschitz continuous, with $\rho _ { f } = ( \sqrt { d } ) ^ { L } B _ { x }$ . By Lemma 30, we know the function $f$ is $L _ { f }$ -smoothness, and the gradient of each weight matrices is bounded by $G _ { f }$ , with

$$
G _ {f} \leq \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x}, L _ {f} \leq \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x} \left((\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \} + 1)\right). \tag {177}
$$

By plugging $\epsilon _ { \mathrm { D G C N } }$ into Theorem 3, we obtain the generalization bound of DGCN.

# J.1 Proof of Lemma 28

By the definition of $z _ { \mathrm { m a x } }$ , we have

$$
\begin{array}{l} z _ {\max } = \max  _ {i} \left\| \left[ \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \mathbf {L} ^ {\ell} \mathbf {X} \left(\beta_ {\ell} \mathbf {W} ^ {(\ell)} + (1 - \beta_ {\ell}) \mathbf {I}\right) \right] _ {i,:} \right\| _ {2} \\ \leq \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \max  _ {i} \| [ \mathbf {L} ^ {\ell} \mathbf {X} (\beta_ {\ell} \mathbf {W} ^ {(\ell)} + (1 - \beta_ {\ell}) \mathbf {I}) ] _ {i, :} \| _ {2} \tag {178} \\ \leq_ {(a)} \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} (\sqrt {d}) ^ {\ell} B _ {x} \left(\beta_ {\ell} B _ {w} + (1 - \beta_ {\ell})\right) \\ \underset {(b)} {\leq} (\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \}, \\ \end{array}
$$

wheand ty . $( a )$ follows from Lemma 31 and inequality $( b )$ is due to the fact that $\begin{array} { r } { \sum _ { \ell = 1 } ^ { L } \alpha _ { \ell } = 1 } \end{array}$ $\alpha _ { \ell } \in [ 0 , 1 ]$

# J.2 Proof of Lemma 29

By the definition of $\Delta z _ { \mathrm { m a x } }$ , we have

$$
\begin{array}{l} \Delta z _ {\max } = \max  _ {i} \| [ \mathbf {Z} - \tilde {\mathbf {Z}} ] _ {i,:} \| _ {2} \\ = \max  _ {i} \left\| \left[ \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \mathbf {L} ^ {\ell} \mathbf {X} \left(\beta_ {\ell} \mathbf {W} ^ {(\ell)} + (1 - \beta_ {\ell}) \mathbf {I}\right) - \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \mathbf {L} ^ {\ell} \mathbf {X} \left(\beta_ {\ell} \tilde {\mathbf {W}} ^ {(\ell)} + (1 - \beta_ {\ell}) \mathbf {I}\right) \right] _ {i,:} \right\| _ {2} \\ \leq \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \max  _ {i} \left\| \left[ \mathbf {L} ^ {\ell} \mathbf {X} \left(\beta_ {\ell} \mathbf {W} ^ {(\ell)} + (1 - \beta_ {\ell}) \mathbf {I}\right) - \mathbf {L} ^ {\ell} \mathbf {X} \left(\beta_ {\ell} \tilde {\mathbf {W}} ^ {(\ell)} + (1 - \beta_ {\ell}) \mathbf {I}\right) \right] _ {i,:} \right\| _ {2} \\ = \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \beta_ {\ell} \max  _ {i} \left\| \left[ \mathbf {L} ^ {\ell} \mathbf {X} \left(\mathbf {W} ^ {(\ell)} - \tilde {\mathbf {W}} ^ {(\ell)}\right) \right] _ {i,:} \right\| _ {2} \\ \leq_ {(a)} \sum_ {\ell = 1} ^ {L} \alpha_ {\ell} \beta_ {\ell} (\sqrt {d}) ^ {\ell} B _ {x} \| \Delta \mathbf {W} ^ {(\ell)} \| _ {2} \\ \leq (\sqrt {d}) ^ {L} B _ {x} \left(\| \Delta \mathbf {W} ^ {(1)} \| _ {2} + \dots + \| \Delta \mathbf {W} ^ {(L)} \| _ {2}\right), \tag {179} \\ \end{array}
$$

wheand ty . $( a )$ follows from Lemma 31 and inequality (b) is due to the fact that $\begin{array} { r } { \sum _ { \ell = 1 } ^ { L } \alpha _ { \ell } = 1 } \end{array}$ $\alpha _ { \ell } \in [ 0 , 1 ]$

# J.3 Proof of Lemma 30

Recall that $\mathbf { G } ^ { ( \ell ) }$ is defined as the gradient of loss with respect to $\mathbf { W } ^ { ( \ell ) }$ , which can be formulated as

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} \| _ {2} = \left\| \frac {1}{m} \sum_ {i = 1} ^ {m} \frac {\partial \operatorname {L o s s} (f (\mathbf {z} _ {i}) , y _ {i})}{\partial \mathbf {z} _ {i}} \frac {\partial \mathbf {z} _ {i}}{\partial \mathbf {h} _ {i} ^ {(\ell)}} \frac {\partial \mathbf {h} _ {i} ^ {(\ell)}}{\partial \mathbf {W} ^ {(\ell)}} \right\| _ {2} \\ \leq \max  _ {i} \left\| \frac {\partial \operatorname {L o s s} (f (\mathbf {z} _ {i}) , y _ {i})}{\partial \mathbf {z} _ {i}} \frac {\partial \mathbf {z} _ {i}}{\partial \mathbf {h} _ {i} ^ {(\ell)}} \frac {\partial \mathbf {h} _ {i} ^ {(\ell)}}{\partial \mathbf {W} ^ {(\ell)}} \right\| _ {2} \\ \leq \alpha_ {\ell} \beta_ {\ell} (\sqrt {d}) ^ {\ell} B _ {x} \max  _ {i} \left\| \frac {\partial \operatorname {L o s s} (f \left(\mathbf {z} _ {i}\right) , y _ {i})}{\partial \mathbf {z} _ {i}} \right\| _ {2} \tag {180} \\ \stackrel {<  } {\underset {(a)} {\leq}} \frac {2}{\gamma} \alpha_ {\ell} \beta_ {\ell} (\sqrt {d}) ^ {\ell} B _ {x} \\ \leq \frac {2}{\gamma} (\sqrt {d}) ^ {L} B _ {x}, \\ \end{array}
$$

where $( a )$ is due to the fact that loss function is $\frac { 2 } { \gamma }$ -Lipschitz conitnous.

Besides, recall that $\mathbf { G } ^ { ( L + 1 ) }$ denotes the gradient with respect to the weight of binary classifier. Therefore, we have

$$
\left\| \mathbf {G} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{\gamma} h _ {\max } ^ {(L)} \leq \frac {2}{\gamma} (\sqrt {d}) ^ {L} B _ {x}, \tag {181}
$$

where inequality $( a )$ follows from Lemma 29.

Therefore, combining the results above, we have

$$
\sum_ {\ell = 1} ^ {L + 1} \left\| \mathbf {G} ^ {(\ell)} \right\| _ {2} \leq \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x}. \tag {182}
$$

Furthermore, by the definition of $\lVert \mathbf { G } ^ { ( \ell ) } - \tilde { \mathbf { G } } ^ { ( \ell ) } \rVert _ { 2 }$ , we have

$$
\begin{array}{l} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} \max  _ {i} \| \frac {\partial f (\mathbf {z} _ {i})}{\partial \mathbf {z} _ {i}} \frac {\partial \mathbf {z} _ {i}}{\partial \mathbf {W}} - \frac {\partial f (\tilde {\mathbf {z}} _ {i})}{\partial \tilde {\mathbf {z}} _ {i}} \frac {\partial \tilde {\mathbf {z}} _ {i}}{\partial \tilde {\mathbf {W}}} \| _ {2} \\ = \frac {2}{\gamma} \alpha_ {\ell} \beta_ {\ell} (\sqrt {d}) ^ {\ell} B _ {x} \left\| \frac {\partial f \left(\mathbf {z} _ {i}\right)}{\partial \mathbf {z} _ {i}} - \frac {\partial f \left(\tilde {\mathbf {z}} _ {i}\right)}{\partial \tilde {\mathbf {z}} _ {i}} \right\| _ {2} \tag {183} \\ \leq \frac {2}{\gamma} \alpha_ {\ell} \beta_ {\ell} B _ {x} (\sqrt {d}) ^ {\ell} \left(\Delta z _ {\max } + (z _ {\max } + 1) \| \Delta \mathbf {v} \| _ {2}\right) \\ \leq \frac {2}{\gamma} (\sqrt {d}) ^ {L} B _ {x} \left((\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \} + 1)\right) \| \Delta \boldsymbol {\theta} \| _ {2}. \\ \end{array}
$$

Similarly, we can upper bound the difference between gradient for the weight of the binary classifier as

$$
\begin{array}{l} \left\| \mathbf {G} ^ {(L + 1)} - \tilde {\mathbf {G}} ^ {(L + 1)} \right\| _ {2} \leq \frac {2}{\gamma} \Delta h _ {\max } ^ {(L)} \tag {184} \\ \leq \frac {2}{\gamma} (\sqrt {d}) ^ {L} B _ {x} B _ {w}, \\ \end{array}
$$

which is bounded by the right hand side of the last equation. Therefore, we have

$$
\sum_ {\ell = 1} ^ {L + 1} \| \mathbf {G} ^ {(\ell)} - \tilde {\mathbf {G}} ^ {(\ell)} \| _ {2} \leq \frac {2}{\gamma} (L + 1) (\sqrt {d}) ^ {L} B _ {x} \left((\sqrt {d}) ^ {L} B _ {x} \max  \{1, B _ {w} \} + 1)\right) \| \Delta \boldsymbol {\theta} \| _ {2}. \tag {185}
$$

# K Omitted proofs of useful lemmas

# K.1 Proof of Lemma 1

Let $f _ { k } ( \pmb \theta )$ denote applying function $f$ with parameter $\pmb { \theta }$ on the $k$ th data point and $\theta _ { t }$ as the weight parameter at the tth iteration. Because function $f _ { k } ( \pmb \theta )$ is $\rho _ { f }$ -Lipschitz with respect to parameter $\pmb \theta$ , we

can bound the difference between model output by the difference between parameters $\pmb { \theta } _ { T } ^ { i j } , \pmb { \theta } _ { T }$ , i.e.,

$$
\epsilon = \max  _ {k} | f _ {k} \left(\boldsymbol {\theta} _ {T} ^ {i j}\right) - f _ {k} \left(\boldsymbol {\theta} _ {T}\right) | \leq \rho_ {f} \| \boldsymbol {\theta} _ {T} ^ {i j} - \boldsymbol {\theta} _ {T} \| _ {2}. \tag {186}
$$

Let $\begin{array} { r } { \nabla f ( \pmb { \theta } ) = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \nabla f _ { k } ( \pmb { \theta } ) } \end{array}$ and $\begin{array} { r } { \nabla f ( \pmb { \theta } ^ { i j } ) = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \nabla f _ { k } ( \pmb { \theta } ^ { i j } ) } \end{array}$ denote the full-batch gradient computed on the original and perturbed dataset respectively.

Recall that both models have the same initialization $\theta _ { 0 } = \theta _ { 0 } ^ { i j }$ . At each iteration the full-batch gradient are computed on $( m - 1 )$ data points that identical in two dataset, and only 1 data point that are different in two dataset. Therefore, we can bound the change of model parameters $\theta _ { t } ^ { i j } , \theta _ { t }$ after one gradient update step as

$$
\begin{array}{l} \left\| \boldsymbol {\theta} _ {t + 1} ^ {i j} - \boldsymbol {\theta} _ {t + 1} \right\| _ {2} \leq \left(1 + \left(1 - \frac {1}{m}\right) \eta L _ {f}\right) \left\| \boldsymbol {\theta} _ {t} ^ {i j} - \boldsymbol {\theta} _ {t} \right\| _ {2} + \frac {2 \eta G _ {f}}{m} \tag {187} \\ \leq \left(1 + \eta L _ {f}\right) \| \boldsymbol {\theta} _ {t} ^ {i j} - \boldsymbol {\theta} _ {t} \| _ {2} + \frac {2 \eta G _ {f}}{m}. \\ \end{array}
$$

Then after $T$ iterations, we can bound the different between two parameters as

$$
\left\| \boldsymbol {\theta} _ {T} ^ {i j} - \boldsymbol {\theta} _ {T} \right\| _ {2} \leq \frac {2 \eta G _ {f}}{m} \sum_ {t = 1} ^ {T} (1 + \eta L _ {F}) ^ {t - 1}. \tag {188}
$$

By plugging the result back to Eq. 186, we have

$$
\epsilon = \frac {2 \eta \rho_ {f} G _ {f}}{m} \sum_ {t = 1} ^ {T} (1 + \eta L _ {F}) ^ {t - 1}. \tag {189}
$$

# K.2 Upper bound on the $\ell _ { 1 }$ -norm of Laplacian matrix

Lemma 31 (Lemma A.3 in [35]). Let A denote the adjacency matrix of a self-connected undirected graph $\mathcal G ( \nu , \mathcal { E } )$ , and $\mathbf { D }$ denote its corresponding degree matrix, and d denote the maximum number of node degree. We define the graph Laplacian matrix as $\mathbf { L } = \mathbf { D } ^ { - 1 / 2 } \mathbf { A } \mathbf { D } ^ { - 1 / 2 }$ . Then we have

$$
\max  _ {i} \| [ \mathbf {L} ] _ {i,:} \| _ {2} \leq \sqrt {d}. \tag {190}
$$

Proof. By the definition of Laplacian matrix, we know that the ith row and $j$ th column of $\mathbf { L }$ is defined as Li,j = $\begin{array} { r } { \dot { L _ { i , j } } = \frac { 1 } { \sqrt { \deg ( i ) } \sqrt { \deg ( j ) } } } \end{array}$ . Therefore, we have

$$
\begin{array}{l} \max _ {i} \sum_ {j = 1} ^ {N} L _ {i, j} = \max _ {i} \sum_ {j \in \mathcal {N} (i)} \frac {1}{\sqrt {\deg (i)} \sqrt {\deg (j)}} \\ \leq \max  _ {i} \sum_ {j \in \mathcal {N} (i)} \frac {1}{\sqrt {\deg (i)}} \tag {191} \\ = \max _ {i} \sqrt {\deg (i)} \leq \sqrt {d}, \\ \end{array}
$$

where $( a )$ is due to $\begin{array} { r } { \frac { 1 } { \sqrt { \deg ( i ) } } \leq 1 } \end{array}$ for any node $i$ .

![](images/5cdd1a84d9e6dccc2d071081a8d6a899bc9a281bddee117404d6bc3fc1fbf7d0.jpg)