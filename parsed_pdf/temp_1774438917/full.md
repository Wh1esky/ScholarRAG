# ONLINE FACILITY LOCATION WITH PREDICTIONS

Shaofeng H.-C. Jiang

Peking University

Email: shaofeng.jiang@pku.edu.cn

Erzhi Liu

Shanghai Jiao Tong University

Email: lezdzh@sjtu.edu.cn

You Lyu

Shanghai Jiao Tong University

Email: vergil@sjtu.edu.cn

Zhihao Gavin Tang

Shanghai University of Finance and Economics

Email: tang.zhihao@mail.shufe.edu.cn

Yubo Zhang

Peking University

Email: zhangyubo18@pku.edu.cn

# ABSTRACT

We provide nearly optimal algorithms for online facility location (OFL) with predictions. In OFL, $n$ demand points arrive in order and the algorithm must irrevocably assign each demand point to an open facility upon its arrival. The objective is to minimize the total connection costs from demand points to assigned facilities plus the facility opening cost. We further assume the algorithm is additionally given for each demand point $x _ { i }$ a natural prediction $f _ { x _ { i } } ^ { \mathrm { p r e d } }$ , which is supposed to be the facility $f _ { x _ { i } } ^ { \mathrm { o p t } }$ that serves $x _ { i }$ in the offline optimal solution.

Our main result is an $\begin{array} { r } { O ( \operatorname* { m i n } \{ \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } , \log n \} ) } \end{array}$ -competitive algorithm where $\eta _ { \infty }$ is the maximum prediction error (i.e., the distance between $f _ { x _ { i } } ^ { \mathrm { p r e d } }$ xi and $f _ { x _ { i } } ^ { \mathrm { o p t } } .$ ). Our algorithm overcomes the fundamental $\Omega { \big ( } { \frac { \log n } { \log \log n } } { \big ) }$ lower bound of OFL (without predictions) when $\eta _ { \infty }$ is small, and it still maintains $O ( \log n )$ ratio even when $\eta _ { \infty }$ is unbounded. Furthermore, our theoretical analysis is supported by empirical evaluations for the tradeoffs between $\eta _ { \infty }$ and the competitive ratio on various real datasets of different types.

# 1 INTRODUCTION

We study the online facility location (OFL) problem with predictions. In OFL, given a metric space $( \mathcal { X } , d )$ , a ground set $\mathcal { D } \subseteq \mathcal { X }$ of demand points, a ground set ${ \mathcal { F } } \subseteq { \mathcal { X } }$ of facilities, and an opening cost function $w : \mathcal { F } \to \mathbb { R } _ { + }$ , the input is a sequence of demand points $X : = ( x _ { 1 } , \ldots , x _ { n } ) \subseteq { \mathcal { D } }$ , and the online algorithm must irrevocably assign $x _ { i }$ to some open facility, denoted as $f _ { i }$ , upon its arrival, and an open facility cannot be closed. The goal is to minimize the following objective

$$
\sum_ {f \in F} w (f) + \sum_ {x _ {i} \in X} d \left(x _ {i}, f _ {i}\right),
$$

where $F$ is the set of open facilities.

Facility location is a fundamental problem in both computer science and operations research, and it has been applied to various domains such as supply chain management (Melo et al., 2009), distribution system design (Klose & Drexl, 2005), healthcare (Ahmadi-Javid et al., 2017), and more applications can be found in surveys (Drezner & Hamacher, 2002; Laporte et al., 2019). Its online variant, OFL, was also extensively studied and well understood. The state of the art is an $O { \big ( } { \frac { \log n } { \log \log n } } { \big ) }$ -competitive deterministic algorithm proposed by Fotakis (2008), and the same work shows this ratio is tight.

We explore whether or not the presence of certain natural predictions could help to bypass the O( log nlog log n ) $O { \big ( } { \frac { \log n } { \log \log n } } { \big ) }$ barrier. Specifically, we consider the setting where each demand point $x _ { i }$ receives a

predicted facility $f _ { x _ { i } } ^ { \mathrm { p r e d } }$ that is supposed to be $x _ { i }$ ’s assigned facility in the (offline) optimal solution. This prediction is very natural, and it could often be easily obtained in practice. For instance, if the dataset is generated from a latent distribution, then predictions may be obtained by analyzing past data. Moreover, predictions could also be obtained from external sources, such as expert advice, and additional features that define correlations among demand points and/or facilities. Previously, predictions of a similar flavor were also considered for other online problems, such as online caching (Rohatgi, 2020; Lykouris & Vassilvitskii, 2021), ski rental (Purohit et al., 2018; Gollapudi & Panigrahi, 2019) and online revenue maximization (Medina & Vassilvitskii, 2017).

# 1.1 OUR RESULTS

Our main result is a near-optimal online algorithm that offers a smooth tradeoff between the prediction error and the competitive ratio. In particular, our algorithm is $O ( 1 )$ -competitive when the predictions are perfectly accurate, i.e., f predxi is the nearest neighbor of xi in the optimal solution for $f _ { x _ { i } } ^ { \mathrm { p r e d } }$ $x _ { i }$ every $1 \leq i \leq n$ . On the other hand, even when the predictions are completely wrong, our algorithm still remains $O ( \log n )$ -competitive. As in the literature of online algorithms with predictions, our algorithm does not rely on the knowledge of the prediction error $\eta _ { \infty }$ .

Theorem 1.1. There exists an $\begin{array} { r } { O ( \operatorname* { m i n } \{ \log n , \operatorname* { m a x } \{ 1 , \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } \} \} ) } \end{array}$ )-competitive algorithm for the OFL with predictions, where $\begin{array} { r } { \eta _ { \infty } : = \operatorname* { m a x } _ { 1 \leq i \leq n } d ( f _ { x _ { i } } ^ { \mathrm { p r e d } } , f _ { x _ { i } } ^ { \mathrm { o p t } } ) } \end{array}$ , f xi measures the maximum prediction error, and $f _ { x _ { i } } ^ { \mathrm { o p t } }$ is the nearest neighbor of $x _ { i }$ from the offline optimal solution OPT1.

Indeed, we can also interpret our result under the robustness-consistency framework which is widely considered in the literature of online algorithms with predictions (cf. Lykouris & Vassilvitskii (2021)), where the robustness and consistency stand for the competitive ratios of the algorithm in the cases of arbitrarily bad predictions and perfect predictions, respectively. Under this language, our algorithm is $O ( 1 )$ -consistent and $O ( \log n )$ -robust. We also remark that our robustness bound nearly matches the lower bound for the OFL without predictions.

whether or not it makes sense to con which is the total error of predictions. er a related error measure,r lower bound result (Theore $\eta _ { 1 } : =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { n } d \bar { ( } f _ { x _ { i } } ^ { \mathrm { p r e d } } , f _ { x _ { i } } ^ { \mathrm { o p t } } ) } \end{array}$ , f x $\eta _ { 1 }$ $\eta _ { \infty }$ (see Section 4 for a more detailed explanation). The lower bound also asserts that our algorithm is nearly tight in the dependence of $\eta _ { \infty }$ .

Theorem 1.2. Consider OFL with predictions with a uniform opening cost of 1. For every $\eta _ { \infty } \in$ (0, 1], there exists a class of inputs, such that no (randomized) online algorithm is $O \Big ( \frac { \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } } { \log \log n } \Big )$ competitive, even when $\eta _ { 1 } = O ( 1 )$ .

As suggested by an anonymous reviewer, it might be possible to make the error measure $\eta _ { \infty }$ more “robust” by taking “outliers” into account. A more detailed discussion on this can be found in Section A.

Empirical evaluation. We simulate our algorithm on both Euclidean and graph (shortest-path) datasets, and we measure the tradeoff between $\eta _ { \infty }$ and the empirical competitive ratio, by generating random predictions whose $\eta _ { \infty }$ is controlled to be around some target value. We compare with two baselines, the $O ( \log n )$ -competitive algorithm by Meyerson (2001) which do not use the prediction at all, and a naive algorithm that always trusts the prediction. Our algorithm significantly outperforms Meyerson’s algorithm when $\eta _ { \infty }  0$ , and is still comparable to Meyerson’s algorithm when $\eta _ { \infty }$ is large where the follow-prediction baseline suffers a huge error.

We observe that our lead is even more significant on datasets with non-uniform opening cost, and this suggests that the prediction could be very useful in the non-uniform setting. This phenomenon seems to be counter-intuitive since the error guarantee $\eta _ { \infty }$ only concerns the connection costs without taking the opening cost into consideration (i.e., it could be that a small $\eta _ { \infty }$ is achieved by predictions of huge opening cost), which seems to mean the prediction may be less useful. Therefore, this actually demonstrates the superior capability of our algorithm in using the limited information from the predictions even in the non-uniform opening cost setting.

Finally, we test our algorithm along with a very simple predictor that does not use any sophisticated ML techniques or specific features of the dataset, and it turns out that such a simple predictor already achieves a reasonable performance. This suggests that more carefully engineered predictors in practice is likely to perform even better, and this also justifies our assumption of the existence of a good predictor.

Comparison to independent work. A recent independent work by Fotakis et al. (2021a) considers OFL with predictions in a similar setting. We highlight the key differences as follows.

(i) We allow arbitrary opening cost $w ( \cdot )$ , while Fotakis et al. (2021a) only solves the special case of uniform facility location where all opening costs are equal. This is a significant difference, since as we mentioned (and will discuss in Section 1.2)), the non-uniform opening cost setting introduces outstanding technical challenges that require novel algorithmic ideas.   
(ii) For the uniform facility location settinwhile the ratio in Fotakis et al. (2021a) is e ratio i, where $\begin{array} { r } { { \cal O } ( \log \frac { n \eta _ { \infty } } { \mathrm { { O P T } } } ) = { \cal O } ( \log \frac { n \eta _ { \infty } } { w } ) } \end{array}$ log(err0)log(err−1 log(err0))  err0 = nη∞/w, err1 = η1/w $\frac { \log ( \mathrm { e r r } _ { 0 } ) } { \log ( \mathrm { e r r } _ { 1 } ^ { - 1 } \log ( \mathrm { e r r } _ { 0 } ) ) }$ $\mathrm { e r r } _ { 0 } = n \eta _ { \infty } / w$ $\mathrm { e r r } _ { 1 } = \eta _ { 1 } / w$ and $w$ is the uniform opening cost. Our ratio is comparable to theirs when $\eta _ { 1 }$ is relatively small. However, theirs seems to be better when $\eta _ { 1 }$ is large, and in particular, it was claimed in Fotakis et al. (2021a) that the ratio becomes constant when $\mathrm { e r r } _ { 1 } \approx \mathrm { e r r } _ { 0 }$ . Unfortunately, we think this claim contradicts with our lower bound (Theorem 1.2) which essentially asserts that $\eta _ { 1 } = { \cal { O } } ( \eta _ { \infty } )$ is not helpful for the ratio. Moreover, when $\eta _ { 1 }$ is large, we find their ratio is actually not well-defined since the denominator $\log ( \mathrm { { e r r } _ { 1 } ^ { - 1 } \log ( \vec { \mathrm { { e r r } _ { 0 } } } ) ) }$ is negative. Therefore, we believe there might be some unstated technical assumptions, or there are technical issues remaining to be fixed, when $\eta _ { 1 }$ is large.

Another independent work by Panconesi et al. (2021) considers a different model of predictions which is not directly comparable to ours. Before the arrival of the first demand point, $k$ sets $S _ { 1 } , S _ { 2 } , \ldots , S _ { k } \subseteq { \mathcal { F } }$ are provided as multiple machine-learned advice, each suggesting a list of facilities to be opened. Their algorithm only uses facilities in $\textstyle S = \bigcup _ { i } S _ { i }$ and achieves a cost of $O ( \log ( | S | ) \operatorname { O P T } ( S ) )$ where $\mathrm { O P T } ( S )$ is the optimal using only facilities in $s$ , and the algorithm also has a worst-case ratio of $O { \big ( } { \frac { \log n } { \log \log n } } { \big ) }$ in case the predictions are inaccurate. Similar to the abovementioned Fotakis et al. (2021a) result, this result only works for uniform opening cost while ours works for arbitrary opening costs.

# 1.2 TECHNICAL OVERVIEW

Since we aim for a worst-case (i.e., when the predictions are inaccurate) $O ( \log n )$ ratio, our algorithm is based on an $O ( \log n )$ -competitive algorithm (that does not use predictions) by Meyerson (2001), and we design a new procedure to make use of predictions. In particular, upon the arrival of each demand point, our algorithm first runs the steps of Meyerson’s algorithm , which we call the Meyerson step, and then runs a new procedure (proposed in this paper) to open additional facilities that are “near” the prediction, which we call the Prediction step.

We make use of the following crucial property of Meyerson’s algorithm. Suppose before the first demand point arrives, the algorithm is already provided with an initial facility set $\bar { F }$ , such that for every facility $f ^ { * }$ opened in OPT, $\bar { F }$ contains a facility $f ^ { \prime }$ that is of distance $\eta _ { \infty }$ to $f ^ { * }$ , then Meyerson’s algorithm is $O ( \log { \frac { n \eta _ { \infty } } { \mathrm { O P T } } } )$ -competitive. To this end, our Prediction step aims to open additional facilities so that for each facility in OPT, the algorithm would soon open a close enough facility.

Uniform opening cost. There is a simple strategy that achieves this goal for the case of uniform opening cost. Whenever the Meyerson step decides to open a facility, we further open a facility at the predicted location. By doing so, we would pay twice as the Meyerson algorithm does in order to open the extra facility. As a reward, when the prediction error $\eta _ { \infty }$ is small, the extra facility would be good and close enough to the optimal facility.

Non-uniform opening cost. Unfortunately, this strategy does not extend to the non-uniform case. Specifically, opening a facility exactly at the predicted location could be prohibitively expensive as it may have huge opening cost. Instead, one needs to open a facility that is “close” to the prediction, but attention must be paid to the tradeoff between the opening cost and the proximity to the prediction.

The design of the Prediction step for the non-uniform case is the most technical and challenging of our paper. We start with opening an initial facility that is far from the prediction, and then we open a series of facilities within balls (centered at the prediction) of geometrically decreasing radius, while we also allow the opening cost of the newly open facility to be doubled each time. We stop opening facilities if the total opening cost exceeds the cost incurred by the preceding Meyerson step, in order to have the opening cost bounded.

We show that our procedure always opens a “correct” facility $\hat { f }$ that is $\Theta ( \eta _ { \infty } )$ apart to the corresponding facility in OPT, and that the total cost until $\hat { f }$ is opened is merely $O ( \mathrm { O P T } )$ . This is guaranteed by the gradually decreasing ball radius when we build additional facilities (so that the “correct” facility will not be skipped), and that the total opening cost could be bounded by that of the last opened facility (because of the doubling opening cost). Once we open this $\hat { f }$ , subsequent runs of Meyerson steps would be $O ( \log { \frac { n \eta _ { \infty } } { \mathrm { O P T } } } )$ -competitive.

# 1.3 RELATED WORK

OFL is introduced by Meyerson (2001), where a randomized algorithm that achieves competitive ratios of $O ( 1 )$ and $O ( \log n )$ are obtained in the setting of random arrival order and adversarial arrival order, respectively. Later, Fotakis (2008) gave a lower bound of $\Omega { \big ( } { \frac { \log n } { \log \log n } } { \big ) }$ and proposed a deterministic algorithm matching the lower bound. In the same paper, Fotakis also claimed that an improved analysis of Meyerson’s algorithm actually yields an $O { \big ( } { \frac { \log n } { \log \log n } } { \big ) }$ competitive ratio in the setting of adversarial order. Apart from this classical OFL, other variants of it are studied as well. Diveki & Imreh ´ (2011) and Bamas et al. (2020b) considered the dynamic variant, in which an open facility can be reassigned. Another variant, OFL with evolving metrics, was studied by Eisenstat et al. (2014) and Fotakis et al. (2021b).

There is a recent trend of studying online algorithms with predictions and many classical online problems have been revisited in this framework. Lykouris & Vassilvitskii (2021) considered online caching problem with predictions and gave a formal definition of consistency and robustness. For the ski-rental problem, Purohit et al. (2018) gave an algorithm with a hyper parameter to maintain the balance of consistency and robustness. They also considered non-clairvoyant scheduling on a single machine. Gollapudi & Panigrahi (2019) designed an algorithm using multiple predictors to achieve low prediction error. Recently, Wei & Zhang (2020) showed a tight robustness-consistency tradeoff for the ski-rental problem. For the caching problem, Lykouris & Vassilvitskii (2021) adapted Marker algorithm to use predictions. Following this work, Rohatgi (2020) provided an improved algorithm that performs better when the predictor is misleading. Bamas et al. (2020b) extended the online primal-dual framework to incorporate predictions and applied their framework to solve the online covering problem. Medina & Vassilvitskii (2017) considered the online revenue maximization problem and Bamas et al. (2020a) studied the online speed scaling problem. Both Antoniadis et al. (2020) and Dutting et al. ¨ (2021) considered the secretary problems with predictions. Jiang et al. (2021) studied online matching problems with predictions in the constrained adversary framework. Azar et al. (2021) and Lattanzi et al. (2020) considered flow time scheduling and online load balancing, respectively, in settings with error-prone predictions.

# 2 PRELIMINARIES

Recall that we assume a underlying metric space $( \mathcal { X } , d )$ . For $S \subseteq { \mathcal { X } } , x \in { \mathcal { X } }$ , define $d ( x , S ) : =$ $\mathrm { m i n } _ { y \in S } d ( x , y )$ . For an integer $t$ , let $[ t ] : = \{ 1 , 2 , \ldots , t \}$ . We normalize the opening cost function $w$ so that the minimum opening cost equals 1, and we round down the opening cost of each facility to the closest power of 2. This only increases the ratio by a factor of 2 which we can afford (since we make no attempt to optimize the constant hidden in big $O$ ). Hence, we assume the domain of $w$ is $\{ 2 ^ { i - 1 } \mid i \in [ L ] \}$ for some $L$ . Further, we use $G _ { k }$ to denote the set of facilities whose opening cost is at most $2 ^ { k - 1 }$ , i.e. $G _ { k } : = \{ f \in \mathcal { F } | w ( f ) \le 2 ^ { k - 1 } \} , \forall 1 \le k \le L$ . Observe that $G _ { 0 } = \emptyset$ .

Algorithm 1 Prediction-augmented Meyerson   
1: initialize $F\gets \emptyset, F_{\mathrm{P}}\gets \emptyset$ $\triangleright F, F_{\mathrm{P}}$ are the set of all open facilities and those opened by PRED   
2: for arriving demand point $x\in X$ and its associated $f_{x}^{\mathrm{pred}}$ do   
3: costM(x) $\leftarrow$ MEY (x)   
4: PRED $(f_x^{\mathrm{pred}},\mathrm{cost}_M(x))$ 5: procedure MEY(x)   
6: for $k\in [L]$ do   
7: $f_{k}\gets \arg \min_{f\in F\cup G_{k}}d(x,f)$ 8: $\delta_k\gets d(x,f_k)$ 9: $p_k\gets (\delta_{k - 1} - \delta_k) / 2^k$ , where $\delta_0 = d(x,F)$ 10: let $s_k\gets \sum_{i = k}^{L}p_i$ for $k\in [L]$ 11: sample $r\sim \mathrm{Uni}[0,1]$ , and let $i\in [L]$ be the index such that $r\in [s_{i + 1},s_i)$ 12: if such $i$ exists then   
13: $F\gets F\cup f_i$ ▷open facility at fi   
14: connect $x$ to the nearest neighbor in $F$ 15: return $d(x,F) + w(f_i)$ ▷if i does not exist, simply return $d(x,F)$ 16: procedure PRED $(f_x^{\mathrm{pred}},q)$ 17: repeat   
18: $r\gets \frac{1}{2} d(f_x^{\mathrm{pred}},F_{\mathrm{P}})$ 19: $f_{\mathrm{open}}\gets \arg \min_{f:d(d(f_x^{\mathrm{pred}},f)\leq r}w(f)\triangleright$ break ties by picking the closest facility to $f_{x}^{\mathrm{pred}}$ 20: if $q\geq w(f_{\mathrm{open}})$ then   
21: $F\gets F\cup f_{\mathrm{open}},F_{\mathrm{P}}\gets F_{\mathrm{P}}\cup f_{\mathrm{open}}$ ▷open facility at fopen   
22: $q\gets q - w(f_{\mathrm{open}})$ 23: until $q <   w(f_{\mathrm{open}})$ 24: open facility at fopen with probability $\frac{q}{w(f_{\mathrm{open}})}$ , and update $F,F_{\mathrm{P}}$

# 3 OUR ALGORITHM: PREDICTION-AUGMENTED MEYERSON

We prove Theorem 1.1 in this section. Our algorithm, stated in Algorithm 1, consists of two steps, the Meyerson step and the Prediction step. Upon the arrival of each demand point $x$ , we first run the Meyerson algorithm Meyerson (2001), and let $\mathrm { c o s t } _ { \mathrm { M } } ( x )$ be the cost from the Meyerson step. Roughly, for a demand point $x$ , the Meyerson algorithm first finds for each $k \geq 1$ , a facility $f _ { k }$ which is the nearest to $x$ , among facilities of opening cost at most $2 ^ { k - 1 }$ and those have been opened, and let $\delta _ { k } \ = \ d ( x , f _ { k } )$ . Then, open a random facility from $\{ f _ { k } \} _ { k }$ , such that $f _ { k }$ is picked with probability $( \delta _ { k - 1 } - \delta _ { k } ) / 2 ^ { k }$ . For technical reasons, we do not normalize $\sum _ { k } p _ { k }$ , and we simply truncate if $\textstyle \sum _ { k } p _ { k } > 1$ . Next, we pass the cost incurred in the Meyerson step (which is random) as the budget $q$ to the Prediction step, and the Prediction step would use this budget to open a series of facilities through the repeat-until loop. Specifically, for each iteration in this loop, a facility is to be opened around distance $r$ to the prediction, where $r$ is geometrically decreasing, and the exact facility is picked to be the one with the minimum opening cost. The ratio of this algorithm is stated as follows.

Theorem 3.1. Prediction-augmented Meyerson is $\begin{array} { r } { O ( \operatorname* { m i n } \{ \log n , \operatorname* { m a x } \{ 1 , \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } \} \} ) } \end{array}$ )-competitive.

Calibrating predictions. Recall that $\begin{array} { r } { \eta _ { \infty } = \operatorname* { m a x } _ { i \in [ n ] } d ( f _ { x _ { i } } ^ { \mathrm { p r e d } } , f _ { x _ { i } } ^ { \mathrm { o p t } } ) } \end{array}$ can be unbounded. We show how to “calibrate” the bad predictions so that $\eta _ { \infty } = { \cal O } ( \mathrm { O P T } )$ . Specifically, when a demand point $x$ arrives, we compute $\begin{array} { r } { f _ { x } ^ { \prime } : = \arg \operatorname* { m i n } _ { f \in \mathcal { F } } \{ d ( x , f ) + w ( f ) \} } \end{array}$ , and we calibrate $f _ { x } ^ { \mathrm { p r e d } }$ by letting $f _ { x } ^ { \mathrm { p r e d } } = f _ { x } ^ { \prime }$ if $d ( x , f _ { x } ^ { \mathrm { p r e d } } ) \geq 2 d ( x , f _ { x } ^ { \prime } ) + w ( f _ { x } ^ { \prime } )$ .

We show the calibrated predictions satisfy $\eta _ { \infty } = { \cal O } ( \mathrm { O P T } )$ . Indeed, for every demand point $x$ , $d ( f _ { x } ^ { \prime } , f _ { x } ^ { \mathrm { o p t } } ) \leq d ( x , f _ { x } ^ { \prime } ) + d ( x , f _ { x } ^ { \mathrm { o p t } } ) \leq 2 d ( x , f _ { x } ^ { \prime } ) + w ( f _ { x } ^ { \prime } ) \leq O ( \mathrm { O P T } )$ , where the second to last cost of first opening inequality follows from the optimality (i.e., the connection cost $f _ { x } ^ { \prime }$ then connecting $x$ to $f _ { x } ^ { \prime } .$ ). Note that, if nη $\eta _ { \infty } = { \cal O } ( \mathrm { O P T } )$ $d ( x , f _ { x } ^ { \mathrm { o p t } } )$ has to be smaller than the then $\begin{array} { r } { \frac { n \eta _ { \infty } } { \mathrm { O P T } } = O ( n ) } \end{array}$ , hence it suffices to prove a single bound $\begin{array} { r } { O ( \operatorname* { m a x } \{ 1 , \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } \} ) } \end{array}$ for Theorem 3.1 (i.e., ignoring the outer min).

Algorithm analysis. Let $F _ { \mathrm { o p t } }$ be the set of open facilities in the optimal solution. We examine each $f ^ { * } \in F _ { \mathrm { o p t } }$ and its corresponding demand points separately, i.e. those demand points connecting to $f ^ { * }$ in the optimal solution. We denote this set of demand points by $X ( f ^ { * } )$ .

Definition 3.2 (Open facilities). For every demand point $x$ , let $F ( x )$ be the set of open facilities right after the arrival of request $x$ , $\bar { F } ( x )$ be the set of open facilities right before the arrival of $x$ , and ${ \hat { F } } ( x ) = F ( x ) \setminus { \bar { F } } ( x )$ be the set of the newly-open facilities on the arrival of $x$ . Moreover, let $F _ { \mathrm { M } } ( x ) , F _ { \mathrm { P } } ( x )$ be a partition of $F ( x )$ , corresponding to the facilities opened by MEY and PRED, respectively. Let $\bar { F } _ { \mathrm { M } } ( x ) , \hat { F } _ { \mathrm { M } } ( x ) , \bar { F } _ { \mathrm { P } } ( x ) , \hat { F } _ { \mathrm { P } } ( x )$ be defined similarly.

Definition 3.3 (Costs). Let $\mathrm { c o s t } _ { \mathrm { M } } ( x ) = w ( \hat { F } _ { \mathrm { M } } ( x ) ) + d \left( x , \left( F _ { \mathrm { M } } ( x ) \cup \bar { F } _ { \mathrm { P } } ( x ) \right) \right)$ be the the total cost from MEY step. Let $\mathrm { c o s t } _ { \mathrm { P } } ( x ) = w ( \hat { F } _ { \mathrm { P } } ( x ) )$ be the cost from PRED step. Let $\mathrm { c o s t } ( x ) =$ $\mathrm { c o s t } _ { \mathrm { M } } ( x ) + \mathrm { c o s t } _ { \mathrm { P } } ( x )$ be the total cost of $x$ .

Recall that after the MEY step, we assign a total budget of $\mathrm { c o s t } _ { \mathrm { M } } ( x )$ to the PRED step. That is, the expected cost from the PRED step is upper bounded by the cost from the MEY step. We formalize this intuition in the following lemma.

Lemma 3.4. For each demand point x ∈ $X , \mathbb { E } [ \mathrm { c o s t } ( x ) ] = 2 \mathbb { E } [ \mathrm { c o s t } _ { \mathrm { M } } ( x ) ] .$

Proof. Consider the expected cost from the PRED step. Given arbitrary $F _ { p } , q$ , there is only one random event from the algorithm and it is straightforward to see that the expected cost equals $q$ . Notice that our algorithm assigns a total budget of $q = \cosh ( x )$ . Thus, $\mathbb { E } [ \mathrm { c o s t } ( \boldsymbol { x } ) ] = \mathbb { E } [ \mathrm { c o s t } \mathbf { \bar { M } } ( \boldsymbol { x } ) \bar { + }$ $\mathrm { c o s t } _ { \mathrm { P } } ( x ) ] = 2 \mathbb { E } [ \mathrm { \bar { c } o s t } _ { \mathrm { M } } ( x ) ]$ .

Therefore, we are left to analyze the total expected cost from the MEY step. The following lemma is the most crucial to our analysis. Before continuing to the proof of the lemma, we explain how it concludes the proof of our main theorem.

Lemma 3.5. For every facility $f ^ { \ast } \in F _ { \mathrm { o p t } } ,$ , we have

$$
\sum_ {x \in X (f ^ {*})} \mathbb {E} [ \operatorname {c o s t} _ {\mathrm {M}} (x) ] \leq O \left(\max  \left\{1, \log \frac {n \eta_ {\infty}}{\mathrm {O P T}} \right\}\right) \cdot \left(w (f ^ {*}) + \sum_ {x \in X (f ^ {*})} d (x, f ^ {*})\right) + O \left(\frac {| X (f ^ {*}) |}{n} \cdot \mathrm {O P T}\right).
$$

Proof of Theorem 3.1. Summing up the equation of Lemma 3.5 for every $f ^ { \ast } \in F _ { \mathrm { o p t } }$ , we have

$$
\begin{array}{l} \mathbb {E} [ \operatorname {A L G} ] = \sum_ {x \in X} \mathbb {E} [ \operatorname {c o s t} (x) ] = 2 \sum_ {x \in X} \mathbb {E} [ \operatorname {c o s t} _ {\mathrm {M}} (x) ] = 2 \sum_ {f ^ {*} \in F _ {\mathrm {o p t}}} \sum_ {x \in X (f ^ {*})} \mathbb {E} [ \operatorname {c o s t} _ {\mathrm {M}} (x) ] \\ \leq \sum_ {f ^ {*} \in F _ {\mathrm {o p t}}} \left(O \left(\max  \left\{1, \log \frac {n \eta_ {\infty}}{\mathrm {O P T}} \right\}\right) \cdot \left(w (f ^ {*}) + \sum_ {x \in X (f ^ {*})} d (x, f ^ {*})\right) + O \left(\frac {| X (f ^ {*}) |}{n} \cdot \mathrm {O P T}\right)\right) \\ = O \left(\max  \left\{1, \log \frac {n \eta_ {\infty}}{\mathrm {O P T}} \right\}\right) \cdot \mathrm {O P T}, \\ \end{array}
$$

where the second equality is by Lemma 3.4 and the inequality is by Lemma 3.5, and this also implies that the ratio is $O ( 1 )$ when $\eta _ { \infty } = 0$ .

# 3.1 PROOF OF LEMMA 3.5

Fix an arbitrary $f ^ { \ast } \in F _ { \mathrm { o p t } }$ . Let $\ell ^ { * }$ be the integer such that $w ( f ^ { * } ) = 2 ^ { \ell ^ { * } - 1 }$ , or equivalently $f ^ { * } \in G _ { \ell ^ { * } }$ . Let $X ( f ^ { * } ) = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { m } \}$ be listed according to their arrival order. We remark that there can be other demand points arriving between $x _ { i }$ and $x _ { i + 1 }$ , but they must be connected to other facilities in the optimal solution. Let $\ell _ { i }$ be the index $\ell$ such that $d ( f ^ { * } , \breve { F } ( x _ { i } ) ) \in [ 2 ^ { \ell - 1 } , 2 ^ { \ell } )$ . $\ell _ { i }$ can be negative and if $\bar { d } ( f ^ { * } , \bar { F } ( x _ { i } ) ) = 0$ , let $\ell _ { i } = - \infty$ . Observe that $d ( f ^ { * } , { \bar { F } } ( x _ { i } ) )$ and $\ell _ { i }$ are non-increasing in $i$ . Let $\tau$ be the largest integer $i$ with $d ( f ^ { * } , \bar { F } ( x _ { i } ) ) > \operatorname* { m i n } \{ 7 \eta _ { \infty } , 4 w ( f ^ { * } ) \}$ . Note that $\tau$ is a random variable. We refer to the demand sequence before $x _ { \tau }$ as the long-distance stage and the sequence after $x _ { \tau }$ as the short-distance stage. In this section, we focus on the case when $\bar { 7 } \eta _ { \infty } \leq 4 w ( f ^ { \ast } \bar { ) }$ . The proof for the other case is very similar and can be found in Section 3.2.

Long-distance stage. We start with bounding the total cost in the long-distance stage. Intuitively, our procedure PRED would quickly build a facility that is close to $f ^ { * }$ within a distance of ${ \cal O } ( \eta _ { \infty } )$ , since it is guaranteed that for each demand point $x _ { i }$ , the distance between $f ^ { * }$ and its corresponding prediction f predx $\bar { f } _ { x _ { i } } ^ { \mathrm { p r e d } }$ is at most $\eta _ { \infty }$ . Formally, we prove the following statements.

Lemma 3.6. $\begin{array} { r } { \sharp \left[ \sum _ { i < \tau } \mathrm { c o s t _ { M } } ( x _ { i } ) \right] \le 2 w ( f ^ { * } ) . } \end{array}$

Proof. Let $G = \{ g _ { 1 } , g _ { 2 } , \dots , g _ { t } \}$ be the facilities opened by the PRED steps on the arrivals of all $\{ x _ { i } \} _ { i < \tau }$ , and are listed according to their opening order. Observe that the budget that is assigned to the PRED step is $\mathrm { c o s t } _ { \mathrm { M } } ( x _ { i } )$ , and the expected opening cost of each PRED step equals its budget. By the definition of $G$ , we have E $\begin{array} { r } { \left[ \sum _ { i < \tau } \mathrm { c o s t . } \operatorname { d } ( x _ { i } ) \right] = \mathbb { E } \left[ \sum _ { k \in [ t ] } w ( g _ { k } ) \right] . } \end{array}$ .

Next, we prove that the opening costs of $g _ { k }$ ’s are always strictly increasing. For each $k \in$ $[ t - 1 ]$ , suppose $g _ { k }$ is opened on the arrival of demand $x$ and $g _ { k + 1 }$ is opened on the arrival of demand $y$ . Let $F$ denote the set of all open facilities right before $g _ { k + 1 }$ is opened. Let $f ~ : = ~ \arg \operatorname* { m i n } _ { f : d ( f _ { y } ^ { \mathrm { p r e d } } , f ) \leq \frac { 1 } { 2 } d ( f _ { y } ^ { \mathrm { p r e d } } , g _ { k } ) } w ( f )$ , we must have $w ( g _ { k + 1 } ) ~ \geq ~ w ( f )$ , since $\begin{array} { r l } { g _ { k + 1 } } & { { } = } \end{array}$ ar $\begin{array} { r } { \cdot \mathrm { g } \operatorname* { m i n } _ { f : d ( f _ { y } ^ { \mathrm { p r e d } } , f ) \leq \frac { 1 } { 2 } d ( f _ { y } ^ { \mathrm { p r e d } } , F ) } w ( f ) } \end{array}$ and that $d ( f _ { y } ^ { \mathrm { p r e d } } , F ) \leq d ( f _ { y } ^ { \mathrm { p r e d } } , g _ { k } )$ . Moreover,

$$
\begin{array}{l} d (f, f _ {x} ^ {\text {p r e d}}) \leq d (f, f _ {y} ^ {\text {p r e d}}) + d (f _ {x} ^ {\text {p r e d}}, f _ {y} ^ {\text {p r e d}}) \leq \frac {1}{2} d (f _ {y} ^ {\text {p r e d}}, g _ {k}) + 2 \eta_ {\infty} \\ \leq \frac {1}{2} \left(d (f _ {x} ^ {\mathrm {p r e d}}, g _ {k}) + d (f _ {x} ^ {\mathrm {p r e d}}, f _ {y} ^ {\mathrm {p r e d}})\right) + 2 \eta_ {\infty} \leq \frac {1}{2} d (f _ {x} ^ {\mathrm {p r e d}}, g _ {k}) + 3 \eta_ {\infty} <   d (f _ {x} ^ {\mathrm {p r e d}}, g _ {k}). \\ \end{array}
$$

Here, the second and fourth inequalities use the fact that $d ( f _ { x } ^ { \mathrm { p r e d } } , f _ { y } ^ { \mathrm { p r e d } } ) ~ \leq ~ d ( f _ { x } ^ { \mathrm { p r e d } } , f ^ { * } ) ~ +$ $d ( f ^ { * } , f _ { y } ^ { \mathrm { p r e d } } ) ~ \leq ~ 2 \eta _ { \infty }$ since both $x , y ~ \in ~ X ( f ^ { * } )$ . The last inequality follows from the fact that $d ( f _ { x } ^ { \mathrm { p r e d } } , g _ { k } ) \geq d ( g _ { k } , f ^ { * } ) - d ( f _ { x } ^ { \mathrm { p r e d } } , f ^ { * } ) > 7 \eta _ { \infty } - \eta _ { \infty } = 6 \eta _ { \infty }$ by the definition of $\tau$ .

Consider the moment when we open $g _ { k }$ , the fact that we do not open facility $f$ implies that $w ( g _ { k } ) <$ $w ( f )$ . Therefore, $w ( g _ { k } ) < w ( f ) \leq w ( g _ { k + 1 } )$ . Recall that the opening cost of each facility is a power of 2. We have that $\begin{array} { r } { \sum _ { k \in [ t ] } w ( g _ { k } ) \leq 2 w ( g _ { t } ) } \end{array}$ .

Finally, we prove that the opening cost $w ( g _ { t } )$ is at most $w ( f ^ { * } )$ . Consider the moment when $g _ { t }$ is open, let $x$ be the corresponding demand point. Notice that

$$
d (f ^ {*}, f _ {x} ^ {\mathrm {p r e d}}) \leq \eta_ {\infty} <   \frac {1}{2} d (f ^ {*}, \bar {F} (x)) \leq \frac {1}{2} d (f ^ {*}, \bar {F} _ {\mathrm {P}} (x)),
$$

by the definition of $\tau$ . However, we open $g _ { t }$ instead of $f ^ { * }$ on Line 20 of our algorithm. It must be the case that $w ( f ^ { * } ) \geq w ( g _ { t } )$ .

To conclude the proof, we have $\begin{array} { r } { \sum _ { k \in [ t ] } w ( g _ { k } ) \leq 2 w ( g _ { t } ) \leq 2 w ( f ^ { * } ) . } \end{array}$

Lemma 3.7. $\begin{array} { r } { \mathbb { E } [ \mathrm { c o s t _ { M } } ( x _ { \tau } ) ] \leq 2 w ( f ^ { * } ) + 2 \sum _ { i = 1 } ^ { m } d ( x _ { i } , f ^ { * } ) . } \end{array}$ .

Proof. We bound the opening cost and connection cost separately.

Opening cost. By the algorithm, we know for every $1 \leq i \leq m$ ,

$$
\mathbb {E} \left[ w (\hat {F} _ {\mathrm {M}} (x _ {i})) \mathbf {1} (w (\hat {F} _ {\mathrm {M}} (x _ {i})) > w (f ^ {*})) \right] \leq \sum_ {i > l ^ {*}} \frac {\delta_ {i - 1} - \delta_ {i}}{2 ^ {i}} \cdot 2 ^ {i} = \delta_ {l ^ {*}} \leq d (x _ {i}, f ^ {*}).
$$

Hence, using the fact that at most one facility is opened by Meyerson step for each demand,

$$
\begin{array}{l} \mathbb {E} [ w (\hat {F} _ {\mathrm {M}} (x _ {\tau})) ] \leq \mathbb {E} \left[ \max  _ {f \in F _ {\mathrm {M}} (x _ {m})} w (f) \right] \leq w (f ^ {*}) + \mathbb {E} \left[ \sum_ {f \in F _ {\mathrm {M}} (x _ {m}), w (f) > w (f ^ {*})} w (f) \right] \\ \leq w (f ^ {*}) + \sum_ {i = 1} ^ {m} d (x _ {i}, f ^ {*}). \\ \end{array}
$$

This finishes the analysis for the opening cost.

Connection cost. We claim that $d ( x _ { \tau } , F ( x _ { \tau } ) ) \leq w ( f ^ { * } ) + d ( x _ { \tau } , f ^ { * } )$ (with probability 1).

Let $\delta _ { 0 } , \delta _ { 1 } , \dots , \delta _ { L }$ be defined as in our algorithm. We assume $\delta _ { 0 } = d ( x _ { \tau } , \bar { F } ( x _ { \tau } ) ) > w ( f ^ { * } ) +$ $d ( x _ { \tau } , f ^ { \ast } )$ , as otherwise the claim holds immediately. Let $k : = \operatorname* { m a x } \{ k : \delta _ { k } > w ( f ^ { * } ) + d ( x _ { \tau } , f ^ { * } ) \}$ . Then

$$
\begin{array}{l} \Pr \left[ d \left(x _ {\tau}, F \left(x _ {\tau}\right)\right) \leq w \left(f ^ {*}\right) + d \left(x _ {\tau}, f ^ {*}\right) \right] \\ = \min  \left(\sum_ {i = k + 1} ^ {L} \frac {\delta_ {i - 1} - \delta_ {i}}{2 ^ {i}}, 1\right) \geq \min  \left(\sum_ {i = k + 1} ^ {\ell^ {*}} \frac {\delta_ {i - 1} - \delta_ {i}}{2 ^ {\ell^ {*}}}, 1\right) = \min  \left(\frac {\delta_ {k} - \delta_ {\ell^ {*}}}{2 ^ {\ell^ {*}}}, 1\right) = 1. \\ \end{array}
$$

Therefore, $\begin{array} { r } { \mathbb { E } [ \mathrm { c o s t } _ { \mathrm { M } } ( x _ { \tau } ) ] = \mathbb { E } [ w ( \hat { F } _ { \mathrm { M } } ( x _ { \tau } ) ) ] + \mathbb { E } [ d ( f ^ { * } , F ( x _ { \tau } ) ) ] \le 2 w ( f ^ { * } ) + 2 \sum _ { i = 1 } ^ { m } d ( x _ { i } , f ^ { * } ) , } \end{array}$ which concludes the proof.

Short-distance stage. Next, we bound the costs for the subsequence of demand points whose arrival is after the event that some facility near $f ^ { * }$ is open. Formally, these demand points are $x _ { i }$ ’s with $i > \tau$ . The analysis of this part is conceptually similar to a part of the analysis of Meyerson’s algorithm. Howeveto get an improved $O ( \log { \frac { n \eta _ { \infty } } { \mathrm { O P T } } } )$ re refined in that ratio instead of $O ( \log n )$ the already opened facility that is near . Moreover, Meyerson did not provide $f ^ { * }$ full detail of this analysis, and our complete self-contained proof fills in this gap.

By the definition of $\tau$ , we have $d ( f ^ { * } , { \bar { F } } ( x _ { i } ) ) \leq 7 \eta _ { \infty }$ for such $i > \tau$ . For integer $\ell$ , let $I _ { \ell } = \{ i >$ $\bar { \tau } \ | \ \ell _ { i } = \ell \}$ . Then for $t$ such that $2 ^ { t - 1 } > \mathrm { m i n } \{ 7 \eta _ { \infty } , 4 w ( f ^ { * } ) \}$ , we have $I _ { t } ~ = ~ \emptyset$ . Let $\boldsymbol { \underline { { \ell } } }$ be the integer that We partition $\frac { \mathrm { O P T } } { n } \in [ 2 ^ { \underline { { \ell } } - 1 } , 2 ^ { \underline { { \ell } } } )$ , and lets with he integer suinto groups $\operatorname* { m i n } \{ 7 \eta _ { \infty } , 4 w ( f ^ { * } ) \} \in [ 2 ^ { \overline { { \ell } } - 1 } , 2 ^ { \overline { { \ell } } } )$ .e $i > \tau$ $I _ { \leq \underline { { \ell } } - 1 } , I _ { \underline { { \ell } } } , \dots , I _ { \overline { { \ell } } }$ $\ell _ { i }$ $I { \ k _ { \le \underline { { \ell } } - 1 } } = \bigcup _ { \ell \le \underline { { \ell } } - 1 } I _ { \ell }$ .

Lemma 3.8. $\begin{array} { r } { \mathbb { E } \left[ \sum _ { i \in I _ { \leq \underline { { \epsilon } } - 1 } } \mathrm { c o s t _ { M } } ( x _ { i } ) \right] \leq \frac { 2 | X ( f ^ { * } ) | } { n } \cdot \mathrm { O P T } . } \end{array}$

Proof. For $i \in I _ { \leq \underline { { \ell } } - 1 }$ , we have that its connection cost $d ( x _ { i } , F ( x _ { i } ) ) \leq d ( x _ { i } , \bar { F } ( x _ { i } ) ) \leq 2 ^ { \underline { { \ell } } - 1 } \leq$ $\textstyle { \frac { \operatorname { O P T } } { n } }$ . Let $\delta _ { 0 } , \delta _ { 1 } , \dots , \delta _ { L }$ and $f _ { 1 } , \ldots , f _ { L }$ be defined as in our algorithm upon $x _ { i }$ ’s arrival. Then, its expected opening cost in the MEY step equals

$$
\begin{array}{l} \mathbb {E} \left[ w \left(\hat {F} _ {\mathrm {M}} \left(x _ {i}\right)\right) \right] = \sum_ {k \in [ L ]} \Pr \left[ \hat {F} _ {\mathrm {M}} \left(x _ {i}\right) = \left\{f _ {k} \right\} \right] \cdot w \left(f _ {k}\right) \leq \sum_ {k \in [ L ]} \min  \left(p _ {k}, 1\right) \cdot 2 ^ {k - 1} \\ \leq \sum_ {k \in [ L ]} \min  \left(\frac {\delta_ {k - 1} - \delta_ {k}}{2 ^ {k}}, 1\right) \cdot 2 ^ {k - 1} \leq \sum_ {k \in [ L ]} \frac {\delta_ {k - 1} - \delta_ {k}}{2} \leq \frac {\delta_ {0}}{2} \leq \frac {\mathrm {O P T}}{2 n}. \\ \end{array}
$$

To sum up,

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {i \in I _ {\leq \underline {{\ell}} - 1}} \operatorname {c o s t} _ {\mathrm {M}} (x _ {i}) \right] \leq \mathbb {E} \left[ \sum_ {i \in I _ {\leq \underline {{\ell}} - 1}} \left(d (x _ {i}, F (x _ {i})) + w (\hat {F} _ {\mathrm {M}} (x _ {i}))\right) \right] \leq \sum_ {i \in I _ {\leq \underline {{\ell}} - 1}} \frac {3}{2 n} \text {O P T} \\ <   \frac {2 | X (f ^ {*}) |}{n} \cdot \text {O P T}. \\ \end{array}
$$

![](images/7924f35474c72d074f996799efd98b93e3aadc47498181804db8c8450772d080.jpg)

Lemma 3.9. For every $\begin{array} { r } { \ell \in [ \underline { { \ell } } , \overline { { \ell } } ] , \mathbb { E } \left[ \sum _ { i \in I _ { \ell } } \mathrm { c o s t _ { M } } ( x _ { i } ) \right] \le 1 8 \sum _ { i \in I _ { \ell } } d ( x _ { i } , f ^ { * } ) + 3 2 w ( f ^ { * } ) . } \end{array}$

Proof. Recall that we relabeled and restricted the indices of points in $X$ to $X ( f ^ { * } ) = \{ x _ { 1 } , \ldots , x _ { m } \}$ . However, in this proof, we also need to talk about the other points in the original data set (that do not belong to $X ( f ^ { * } ) )$ . Hence, to avoid confusions, we rewrite $X = \{ y _ { 1 } , \dots , { \bar { y } } _ { n } \}$ (so $y _ { 1 } , \ldots , y _ { n }$ are the demand points in order), and we define $\sigma : [ m ]  [ n ]$ that maps elements $x _ { j } \in X ( f ^ { * } )$ to its identity $y _ { i } \in X$ , i.e., $\sigma ( j ) = i$ .

For $i \in [ n ]$ , let

$$
z _ {i} ^ {\mathrm {M}} := \left\{ \begin{array}{l l} \hat {F} _ {\mathrm {M}} (y _ {i}) & \hat {F} _ {\mathrm {M}} (y _ {i}) \neq \emptyset \\ \text {n o n e} & \text {o t h e r w i s e} \end{array} \right.,
$$

and define $z _ { i } ^ { \mathrm { P } }$ similarly. Since the MEY step opens at most one facility, $z _ { i } ^ { \mathrm { M } }$ is either a singleton or “none”. Finally, define $z _ { i } : = z _ { i } ^ { \mathrm { M } } \cup z _ { i } ^ { \mathrm { P } }$ .

Define $\tau _ { \ell } = \operatorname* { m i n } \{ i \in [ n ] \ | \ z _ { i } ^ { \mathrm { M } } \neq $ none and $d ( f ^ { * } , z _ { i } ^ { \mathrm { M } } ) < 2 ^ { \ell - 1 } \}$ , we have

$$
\mathbb {E} \left[ \sum_ {i \in I _ {\ell}} \operatorname {c o s t} _ {\mathrm {M}} \left(x _ {i}\right) - 1 8 d \left(x _ {i}, f ^ {*}\right) \right] = \mathbb {E} \left[ \sum_ {i \in [ \tau_ {\ell} ]} \left(\operatorname {c o s t} _ {\mathrm {M}} \left(y _ {i}\right) - 1 8 d \left(y _ {i}, f ^ {*}\right)\right) \mathbf {1} (i \in \sigma \left(I _ {\ell}\right)) \right]. \tag {1}
$$

Next, we prove a stronger version by induction, and the induction hypothesis goes as follows. For every $j \in [ n ]$ ,

$$
\mathbb {E} \left[ \sum_ {i = j} ^ {\tau_ {\ell}} \left(\operatorname {c o s t} _ {\mathrm {M}} \left(y _ {i}\right) - 1 8 d \left(y _ {i}, f ^ {*}\right)\right) \mathbf {1} \left(i \in \sigma \left(I _ {\ell}\right)\right) \mid \mathcal {E} _ {j - 1} \right] \leq 3 2 w \left(f ^ {*}\right),
$$

where $\mathcal { E } _ { j } = ( z _ { 1 } , \ldots , z _ { j } )$ . Clearly, applying this with $j = 1$ implies (1), hence it suffices to prove this hypothesis.

Base case. The base case is $j ~ = ~ n$ , and we would do a backward induction on $j$ . Since we condition on $\mathcal { E } _ { j - 1 }$ , the variables $\delta _ { 0 } , \ldots , \delta _ { L }$ in the MEY step of $y _ { j }$ , as well as wether or not $j \in \sigma ( I _ { \ell } )$ , are fixed. Hence,

$$
\mathbb {E} [ \mathrm {c o s t} _ {\mathrm {M}} (y _ {j}) \mid \mathcal {E} _ {j - 1} ] \leq d (y _ {j}, \bar {F} (y _ {j})) + \sum_ {i = 1} ^ {L} \frac {\delta_ {i - 1} - \delta_ {i}}{2 ^ {i}} \cdot 2 ^ {i} \leq 2 d (y _ {j}, \bar {F} (y _ {j})).
$$

This implies

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {i = j} ^ {\tau_ {\ell}} \left(\operatorname {c o s t} _ {\mathrm {M}} \left(y _ {i}\right) - 1 8 d \left(y _ {i}, f ^ {*}\right)\right) \mathbf {1} \left(i \in \sigma \left(I _ {\ell}\right)\right) \mid \mathcal {E} _ {j - 1} \right] \\ \leq \mathbb {E} \left[ \left(\operatorname {c o s t} _ {\mathrm {M}} \left(y _ {n}\right) - 1 8 d \left(y _ {n}, f ^ {*}\right)\right) \mathbf {1} \left(n \in \sigma \left(I _ {\ell}\right)\right) \mid \mathcal {E} _ {n - 1} \right] \\ \leq 2 d \left(y _ {n}, \bar {F} \left(y _ {n}\right)\right) - 1 8 d \left(y _ {n}, f ^ {*}\right) | _ {n \in \sigma \left(I _ {\ell}\right)} \\ \leq 2 d \left(y _ {n}, \bar {F} \left(y _ {n}\right)\right) - 2 d \left(y _ {n}, f ^ {*}\right) | _ {n \in \sigma \left(I _ {\ell}\right)} \\ \leq 2 d \left(\bar {F} \left(y _ {n}\right), f ^ {*}\right) | _ {n \in \sigma \left(I _ {\ell}\right)} \\ \leq 2 ^ {\ell + 1} \leq 2 ^ {\ell^ {*} + 1} \leq O (w (f ^ {*})). \\ \end{array}
$$

Inductive step. Next, assume the hypothesis holds for $j + 1 , j + 2 , \dots , n$ , and we would prove the hypothesis for $j$ . We proceed with the following case analysis.

Case 1: $j \notin \sigma ( I _ { \ell } )$ . We do a conditional expectation argument, and we have

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {i = j} ^ {\tau_ {\ell}} (\mathrm {c o s t} _ {\mathrm {M}} (y _ {i}) - 1 8 d (y _ {i}, f ^ {*})) \mathbf {1} (i \in \sigma (I _ {\ell})) \mid \mathcal {E} _ {j - 1} \right] \\ = \sum_ {z _ {j} ^ {\mathrm {M}}, z _ {j} ^ {\mathrm {P}}} \Pr \left[ z _ {j} ^ {\mathrm {M}}, z _ {j} ^ {\mathrm {P}} \mid \mathcal {E} _ {j - 1} \right] \mathbb {E} \left[ \sum_ {i = j} ^ {\tau_ {\ell}} \left(\operatorname {c o s t} _ {\mathrm {M}} \left(y _ {i}\right) - 1 8 d \left(y _ {i}, f ^ {*}\right)\right) \mathbf {1} (i \in \sigma \left(I _ {\ell}\right)) \mid \mathcal {E} _ {j - 1}, z _ {j} ^ {\mathrm {M}}, z _ {j} ^ {\mathrm {P}} \right] \\ \leq 3 2 w \left(f ^ {*}\right). \\ \end{array}
$$

where the second step follows by induction hypothesis.

Case 2: $j \in I _ { \ell }$ and $8 d ( y _ { j } , f ^ { * } ) > d ( f ^ { * } , \bar { F } ( y _ { j } ) )$ . We have

$$
d \left(y _ {j}, F \left(y _ {j}\right)\right) \leq d \left(y _ {j}, f ^ {*}\right) + d \left(f ^ {*}, F \left(y _ {j}\right)\right) \leq 9 d \left(y _ {j}, f ^ {*}\right).
$$

Hence $\mathbb { E } [ \mathrm { c o s t _ { M } } ( y _ { j } ) - 1 8 d ( y _ { j } , f ^ { * } ) ] \le 0$ , and the hypothesis follows from a similar argument as in Case 1.

Case 3: $j \in I _ { \ell }$ , and $8 d ( y _ { j } , f ^ { * } ) \leq d ( f ^ { * } , \bar { F } ( y _ { j } ) )$ . We have

$$
\delta_ {\ell^ {*}} \leq d \left(y _ {j}, f ^ {*}\right) \leq \frac {1}{8} d \left(f ^ {*}, \bar {F} \left(y _ {j}\right)\right) \leq 2 ^ {\ell - 3}.
$$

Let $D = \{ f : d ( f ^ { * } , f ) \geq 2 ^ { \ell - 1 } \} \cup \{ \mathrm { n o n e } \}$ . Let $k = \operatorname* { m i n } \{ k ^ { \prime } \mid \delta _ { k ^ { \prime } } \leq 2 ^ { \ell - 2 } \}$ . Then $\forall i \geq k$ , $d ( f _ { i } , f ^ { * } ) \leq d ( y _ { j } , f _ { i } ) + d ( y _ { j } , f ^ { * } ) < 2 ^ { \ell - 1 } .$ .

$$
\begin{array}{l} = \min  \left(\sum_ {i \geq k} p _ {i}, 1\right) = \min  \left(\sum_ {i \geq k} \frac {\delta_ {i - 1} - \delta_ {i}}{2 ^ {i}}, 1\right) \\ \geq \min  \left(\frac {\delta_ {k - 1} - \delta_ {\ell^ {*}}}{2 ^ {\ell^ {*}}}, 1\right) \geq \min  \left(\frac {2 ^ {\ell - 2} - 2 ^ {\ell - 3}}{2 ^ {\ell^ {*}}}, 1\right) \geq \min  \left(2 ^ {\ell - 3 - \ell^ {*}}, 1\right) \\ \geq \frac {\delta_ {0}}{1 6 w (f ^ {*})} \geq \frac {\mathbb {E} [ \mathrm {c o s t} _ {\mathrm {M}} (y _ {j}) \mid \mathcal {E} _ {j - 1} ]}{3 2 w (f ^ {*})}, \\ \end{array}
$$

where the second last inequality follows from $\delta _ { 0 } = d ( y _ { j } , \bar { F } ( y _ { j } ) ) \leq 2 ^ { \ell }$ and $w ( f ^ { * } ) = 2 ^ { \ell ^ { * } - 1 }$ . Thus,

$$
\begin{array}{l} + \sum_ {t _ {j} \in D} \sum_ {z _ {j} ^ {\mathrm {P}}} \Pr \left[ z _ {j} ^ {\mathrm {M}} = t _ {j}, z _ {j} ^ {\mathrm {P}} \mid \mathcal {E} _ {j - 1} \right] \mathbb {E} \left[ \sum_ {i = j + 1} ^ {\tau_ {\ell}} \left(\operatorname {c o s t} _ {\mathrm {M}} \left(y _ {i}\right) - 1 8 d \left(y _ {i}, f ^ {*}\right)\right) \mathbf {1} \left(i \in \sigma \left(I _ {\ell}\right)\right) \mid \mathcal {E} _ {j - 1}, z _ {j} ^ {\mathrm {M}} = t _ {j}, z _ {j} ^ {\mathrm {P}} \right] \\ \leq \Pr \left[ z _ {j} ^ {\mathrm {M}} \notin D \mid \mathcal {E} _ {j - 1} \right] \cdot 3 2 w \left(f ^ {*}\right) + \Pr \left[ z _ {j} ^ {\mathrm {M}} \in D \mid \mathcal {E} _ {j - 1} \right] \cdot 3 2 w \left(f ^ {*}\right) \\ \leq 3 2 w \left(f ^ {*}\right), \\ \end{array}
$$

where the second step follows by induction hypothesis. In summary, we have

$$
\mathbb {E} \left[ \sum_ {i = 1} ^ {\tau_ {\ell}} \left(\operatorname {c o s t} _ {\mathrm {M}} \left(y _ {i}\right) - 1 8 d \left(y _ {i}, f ^ {*}\right)\right) \mathbf {1} (i \in \sigma \left(I _ {\ell}\right)) \mid \mathcal {E} _ {j - 1} \right] \leq 3 2 w \left(f ^ {*}\right)
$$

Proof of Lemma 3.5. We conclude Lemma 3.5 by combining Lemma 3.6, 3.7, 3.8, and 3.9.

$$
\begin{array}{l} \sum_ {i \in [ m ]} \mathbb {E} \left[ \operatorname {c o s t} _ {\mathrm {M}} \left(x _ {i}\right) \right] = \mathbb {E} \left[ \sum_ {i \in [ \tau - 1 ]} \operatorname {c o s t} _ {\mathrm {M}} \left(x _ {i}\right) \right] + \mathbb {E} \left[ \operatorname {c o s t} _ {\mathrm {M}} \left(x _ {\tau}\right) \right] \\ + \sum_ {\ell = \underline {{\ell}}} ^ {\bar {\ell}} \mathbb {E} \left[ \sum_ {i \in I _ {\ell}} \operatorname {c o s t} _ {\mathrm {M}} \left(x _ {i}\right) \right] + \sum_ {i \in I _ {\leq \underline {{\ell}} - 1}} \mathbb {E} \left[ \operatorname {c o s t} _ {\mathrm {M}} \left(x _ {i}\right) \right] \\ \leq 2 w \left(f ^ {*}\right) + \left(2 w \left(f ^ {*}\right) + 2 \sum_ {i = 1} ^ {m} d \left(x _ {i}, f ^ {*}\right)\right) \\ + \sum_ {\ell = \underline {{\ell}}} ^ {\bar {\ell}} \left(1 8 \sum_ {i \in I _ {\ell}} d \left(x _ {i}, f ^ {*}\right) + 3 2 w \left(f ^ {*}\right)\right) + \frac {2 | X (f ^ {*}) |}{n} \cdot \text {O P T} \\ \leq \left(O (\bar {\ell} - \underline {{\ell}}) + O (1)\right) \cdot \left(\sum_ {i \in [ m ]} d \left(x _ {i}, f ^ {*}\right) + w \left(f ^ {*}\right)\right) + \frac {2 | X \left(f ^ {*}\right) |}{n} \mathrm {O P T} \\ \leq O \left(\max  \left\{1, \log \frac {n \eta_ {\infty}}{\mathrm {O P T}} \right\}\right) \cdot \left(\sum_ {i \in [ m ]} d \left(x _ {i}, f ^ {*}\right) + w \left(f ^ {*}\right)\right) + O \left(\frac {| X (f ^ {*}) |}{n} \mathrm {O P T}\right). \\ \end{array}
$$

![](images/cc88761b172d638aef20b604d0f139470ef3f52a366d70899dbd53ad90b8db19.jpg)

# 3.2 PROOF OF LEMMA 3.5: WHEN $7 \eta _ { \infty } > 4 w ( f ^ { \ast } )$

The proof is mostly the same as in the other case which we prove in Section 3.1, and we only highlight the key differences. We use the same definition for parameters τ , `, and `. It can be verified that Lemma 3.7, 3.8 and 3.9 still holds and their proof in Section 3.1 still works.

However, Lemma 3.6 relies on that $7 \eta _ { \infty } \leq 4 w ( f ^ { \ast } )$ , which does not work in the current case. We provide Lemma 3.11 in replacement of Lemma 3.6, which offers a slightly different bound but it still suffices for Lemma 3.5.

Lemma 3.10. For every $i < \tau$ , $d ( x _ { i } , f ^ { * } ) \geq w ( f ^ { * } ) .$ .

Proof. We prove the statement by contradiction. Suppose $d ( x _ { i } , f ^ { * } ) < w ( f ^ { * } )$ . Let $\delta _ { 0 } , \delta _ { 1 } , \dots , \delta _ { W }$ be defined as in our algorithm. Then, we have

$$
\delta_ {0} = d (x _ {i}, \bar {F} (x _ {i})) \geq d (f ^ {*}, \bar {F} (x _ {i})) - d (f ^ {*}, x _ {i}) \geq d (f ^ {*}, \bar {F} (x _ {i})) - w (f ^ {*}) > 3 w (f ^ {*}),
$$

where the last inequality follows from the definition of $\tau$ that $d ( f ^ { * } , \bar { F } ( x _ { i } ) ) > 4 w ( f ^ { * } )$ . Let $k =$ $\operatorname* { m i n } \{ k \mid \delta _ { k } \leq 3 w ( f ^ { * } ) \}$ . We have $k \geq 1$ . We prove that a facility within a distance of $3 w ( f ^ { * } )$ from $x _ { i }$ must be open at this step.

$$
\begin{array}{l} \geq \min  \left(\sum_ {j \geq k} p _ {k}, 1\right) \geq \min  \left(\sum_ {j = k} ^ {\ell^ {*}} \frac {\delta_ {j - 1} - \delta_ {j}}{2 ^ {j}}, 1\right) \\ \geq \min \left(\frac {\delta_ {k - 1} - \delta_ {\ell^ {*}}}{2 ^ {\ell^ {*}}}, 1\right) \geq \min \left(\frac {3 w (f ^ {*}) - w (f ^ {*})}{2 ^ {\ell^ {*}}}, 1\right) = 1, \\ \end{array}
$$

where the last inequality uses the fact that $\delta _ { \ell ^ { * } } \leq d ( x _ { i } , f ^ { * } ) \leq w ( f ^ { * } )$ . Consequently,

$$
d \left(f ^ {*}, \hat {F} \left(x _ {i + 1}\right)\right) \leq d \left(f ^ {*}, F \left(x _ {i}\right)\right) \leq d \left(x _ {i}, F \left(x _ {i}\right)\right) + d \left(x _ {i}, f ^ {*}\right) <   4 w \left(f ^ {*}\right),
$$

which contradicts the definition of $\tau$

Lemma 3.11. $\begin{array} { r } { \mathbb E \left[ \sum _ { i \in [ \tau ] } \mathrm { c o s t } ( x _ { i } ) \right] \leq 6 \mathbb E [ \sum _ { i \in [ \tau ] } d ( x _ { i } , f ^ { * } ) ] + 4 \cdot w ( f ^ { * } ) . } \end{array}$

Proof. On the arrival of $x _ { i }$ for each $i \in [ \tau ]$ , let $\delta _ { 0 } , \delta _ { 1 } , \dots , \delta _ { L }$ be defined as in our algorithm. We first study the connecting cost of request $x _ { i }$ . We have

$$
d (x _ {i}, F (x _ {i})) \leq \max  \left(d (x _ {i}, F (x _ {j}) \backslash \hat {F} (x _ {i})), d (x _ {i}, \hat {F} (x _ {i}))\right) = \max  \left(\delta_ {0}, d (x _ {i}, \hat {F} (x _ {i}))\right).
$$

We prove this value is at most $d ( x _ { i } , f ^ { * } ) + 2 w ( f ^ { * } )$ . Suppose $\delta _ { 0 } > d ( x _ { i } , f ^ { * } ) + 2 w ( f ^ { * } )$ , as otherwise the statement holds. Recall that $\delta _ { j }$ is non-increasing, let $k = \operatorname* { m i n } \{ k \mid \delta _ { k } \leq d ( x _ { i } , f ^ { * } ) + 2 w ( f ^ { * } ) \}$ . By the definition of $k$ , we have that

$\Pr \left[ d ( x _ { i } , F ( x _ { i } ) ) \leq d ( x _ { i } , f ^ { * } ) + 2 w ( f ^ { * } ) \right] \geq \operatorname* { P r } [ f _ { j }$ is opened for some $j \geq k ]$

$$
\begin{array}{l} \geq \min  \left(\sum_ {j \geq k} p _ {j}, 1\right) \geq \min  \left(\sum_ {j \geq k} ^ {\ell^ {*}} \frac {\delta_ {j - 1} - \delta_ {j}}{2 ^ {j}}, 1\right) \geq \min  \left(\frac {\delta_ {k - 1} - \delta_ {\ell^ {*}}}{2 ^ {\ell^ {*}}}, 1\right) \\ \geq \min  \left(\frac {d (x _ {i} , f ^ {*}) + 2 w (f ^ {*}) - \delta_ {\ell^ {*}}}{2 ^ {\ell^ {*}}}, 1\right) \geq \min  \left(\frac {w (f ^ {*})}{2 ^ {\ell^ {*} - 1}}, 1\right) = 1. \\ \end{array}
$$

To sum up, we have shown that the connecting cost $d ( x _ { i } , F ( x _ { i } ) ) \leq d ( x _ { i } , f ^ { * } ) + 2 w ( f ^ { * } ) ,$ .

Next, we study the expected opening cost $\mathbb { E } [ w ( \hat { F } ( x _ { i } ) ) ]$ . We have

$$
\begin{array}{l} \mathbb {E} [ w (\hat {F} (x _ {i})) ] = \sum_ {j \in [ L ]} \Pr \left[ \hat {F} (x _ {i}) = \{f _ {j} \} \right] \cdot w (f _ {j}) \leq \sum_ {j \in [ L ]} \min  \left(\frac {\delta_ {j - 1} - \delta_ {j}}{2 ^ {j}}, 1\right) \cdot 2 ^ {j - 1} \\ \leq \sum_ {j \leq \ell^ {*}} 2 ^ {j - 1} + \sum_ {j > \ell^ {*}} \frac {\delta_ {j - 1} - \delta_ {j}}{2 ^ {j}} \cdot 2 ^ {j - 1} <   2 ^ {\ell^ {*}} + \frac {1}{2} \delta_ {\ell^ {*}} <   2 w (f ^ {*}) + d (x _ {i}, f ^ {*}). \\ \end{array}
$$

By Lemma 3.10, we conclude the proof of the statement:

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {i \in [ \tau ]} \mathrm {c o s t} (x _ {i}) \right] = \mathbb {E} \left[ \sum_ {i \in [ \tau ]} \left(d (x _ {i}, F (x _ {i})) + w (\hat {F} (x _ {i}))\right) \right] \\ \leq \mathbb {E} \left[ \sum_ {i \in [ \tau ]} \left(2 d \left(x _ {i}, f ^ {*}\right) + 4 w \left(f ^ {*}\right)\right) \right] \\ \leq 6 \mathbb {E} \left[ \sum_ {i <   \tau} d \left(x _ {i}, f ^ {*}\right) + \left(2 d \left(x _ {\tau}, f ^ {*}\right) + 4 w \left(f ^ {*}\right)\right) \right] \\ \leq 6 \mathbb {E} \left[ \sum_ {i \in [ \tau ]} d \left(x _ {i}, f ^ {*}\right) \right] + 4 w \left(f ^ {*}\right). \\ \end{array}
$$

![](images/637f9528c218350b8af8d8f72bde75d19f664295861f43583e8e04a1bc1cedce.jpg)

# 4 LOWER BOUND

In the classical online facility location problem, a tight lower bound of O( log nlog log n ) $O { \big ( } { \frac { \log n } { \log \log n } } { \big ) }$ is established by Fotakis (2008), even for the special case when the facility cost is uniform and the metric space is a binary hierarchically well-separated tree (HST). We extend their construction to the setting with ratio of o( log OPTlog log n ) predictions, proving that when the predictions are not precise (i.e. $O \Big ( \frac { \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } } { \log \log n } \Big )$ is impossible. $\eta _ { \infty } > 0$ ), achieving a competitive

Theorem 4.1. Consider OFL with predictions with a uniform opening cost of 1. For every $\eta _ { \infty } \in$ $( 0 , 1 ]$ , there exists a class of inputs, such that no (randomized) online algorithm is $O \Big ( \frac { \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } } { \log \log n } \Big )$ competitive, even when $\eta _ { 1 } = O ( 1 )$ .

Before we provide the proof of our theorem, we give some implications of our theorem. First of all, we normalize the opening cost to be 1. Consequently, the optimal cost is at least 1. Furthermore, we are only interested in the case when $\eta _ { \infty } \leq 1$ . Indeed, the optimal facility for each demand point must be within a distance of 1, as otherwise, we can reduce the cost by opening a new facility at the demand point. Therefore, we can without loss of generality to study predictions with $\eta _ { \infty } \stackrel { \cdot } { = } O ( 1 )$ . Without the normalization, our theorem implies an impossibility result of $O \Big ( \frac { \log \frac { n \eta _ { \infty } } { \mathrm { O P T } } } { \log \log n } \Big )$ log for online facility location with predictions.

Moreover, recall the definition of total prediction error $\begin{array} { r } { \eta _ { 1 } = \sum _ { i = 1 } ^ { n } d ( f _ { x _ { i } } ^ { \mathrm { p r e d } } , f _ { x _ { i } } ^ { \mathrm { o p t } } ) } \end{array}$ , which is at least η∞ = maxi d(f predxi , f optxi ). O $\eta _ { \infty } = \operatorname* { m a x } _ { i } d ( f _ { x _ { i } } ^ { \mathrm { p r e d } } , f _ { x _ { i } } ^ { \mathrm { o p t } } )$ , J xi ur theorem states that even when the total error $\eta _ { 1 }$ is constant times larger than the opening cost of 1, there is no hope for a good algorithm. Note that our bound above holds for any constant value of $\eta _ { \infty }$ . As an implication, our construction rules out the possibility of an $o \Big ( \frac { \log \frac { \eta _ { 1 } } { \mathrm { O P T } } } { \log \log n } \Big )$ log η1OPT -competitive algorithm.

Proof. By Yao’s principle, the expected cost of a randomized algorithm on the worst-case input is no better than the expected cost for a worst-case probability distribution on the inputs of the deterministic algorithm that performs best against that distribution. For each $\eta _ { \infty }$ , we shall construct a family of randomized instance, so that the expected cost of any deterministic algorithm is at least $\Omega \big ( \frac { \log \frac { \bar { n ^ { \cdot } } \eta _ { \infty } } { \mathrm { O P T } } } { \log \log n } \big )$ g nη∞OPT ) - competitive.

We first import the construction for the classical online facility location problem by Fotakis (2008). Consider a hierarchically well-separated perfect binary tree. Let the distance between root and its children as $D$ . For every vertex $i$ of the tree, the distance between $i$ and its children is $\textstyle { \frac { 1 } { m } }$ times m the distance betwchildren of height $i$ anis parent. That is to say, the d. Let the height of the tree be ance between a height . $i$ vertex and its $i + 1$ $\textstyle { \frac { D } { m ^ { i } } }$ $h$

The demand sequence is consisted of $h + 1$ phases. For each $0 \leq i \leq h$ , in the $( i + 1 )$ -th phase, $m ^ { i }$ demand points arrive consecutively at some vertex of height $i$ . For $i \geq 1$ , the identity of the height $i$ vertex is independently and uniformly chosen between the two children of $i$ -th phase vertex.

Consider the solution that opens only one facility at the leaf node that is the $( h + 1 )$ -th phase demand vertex and assign all demand points to the leaf. The cost of this solution equals

$$
1 + \sum_ {i = 0} ^ {h} m ^ {i} \sum_ {j = i} ^ {h - 1} \frac {D}{m ^ {j}} \leq 1 + \sum_ {i = 0} ^ {h - 1} m ^ {i} \frac {D}{m ^ {i}} \frac {m}{m - 1} = 1 + h D \frac {m}{m - 1}.
$$

This value serves as an upper bound of the optimal cost. I.e., $\begin{array} { r } { \mathrm { O P T } \leq 1 + h D \frac { m } { m - 1 } } \end{array}$ .

Next, we describe the predictions associated with the demand points. Intuitively, we try the best to hide the identity of the leaf node in the last phase. We denote the leaf node as $f ^ { * }$ , which is also the open facility in the above described solution. Our prediction is produced according to the following rule. When the distance between the current demand point and $f ^ { * }$ is less than $\eta _ { \infty }$ , let the prediction be the same as the demand vertex. Otherwise, let the prediction be the vertex on the path from the current demand point to $f ^ { * }$ , whose distance to $f ^ { * }$ equals $\eta _ { \infty }$ .

We prove a lower bound on the expected cost of any deterministic algorithm. We first overlook the first a few phases until the subtree of the current demand vertex has diameter less than $\eta _ { \infty }$ . Let it be the $h ^ { \prime }$ -th phase. We now focus on the cost induced after the $h ^ { \prime }$ -th phase and notice that the predictions are useless as they are just the demand points. When the $m ^ { i }$ demand points at vertex of height $i$ comes, we consider the following two cases:

Case 1: There is no facility opened in the current subtree. Then we either open a new facility bearing an opening cost 1 or assign the current demands to facilities outside the subtree bearing a large connection cost at least $\frac { D } { m ^ { i - 1 } }$ per demand. So the cost of this vertex is $\operatorname* { m i n } \{ 1 , D m \}$ .

Case 2: There has been at least one facility opened in the current subtree. Since the next vertex is uniformly randomly chosen in its two children, the expected number of facility that will not enter the next phase subtree is at least $\frac { 1 } { 2 }$ . We call it abandoned. The expected sum of abandoned facility is $\textstyle { \frac { 1 } { 2 } }$ of the occurrence of case 2. Thus the expected cost in every occurrence of case 2 is $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ .

We carefully choose $D , m , h$ so that 1) $m D = 1 ; 2$ ) the total number of demand points is $n$ , i.e. $\sum _ { i = 0 } ^ { h } m ^ { i } = n$ ; and 3) $\begin{array} { r } { \frac D { m ^ { h ^ { \prime } } } = \eta _ { \infty } } \end{array}$ . These conditions give that $( h + 1 ) \log m = \log n$ , $\left( h ^ { \prime } + 1 \right) \log m =$ $- \log \eta _ { \infty }$ . Consequently, $\begin{array} { r } { h - h ^ { \prime } = \frac { \log n \eta _ { \infty } } { \log m } } \end{array}$ log m .

To sum up, an lower bound of any deterministic algorithm is $\begin{array} { r } { ( h - h ^ { \prime } ) \operatorname* { m i n } \{ \frac { 1 } { 2 } , m D \} = \Omega \left( \frac { \log n \eta _ { \infty } } { \log m } \right) , } \end{array}$ while the optimal cost is at most $\begin{array} { r } { 1 + h D \frac { m } { m - 1 } = O \left( \frac { \log n } { m \log m } \right) } \end{array}$ Setting $\begin{array} { r } { m = \frac { \log n } { \log \log n } } \end{array}$ , the optimal cost is O(1). And we prove the claimed lower bound of Ω( log nη∞log log n ) $O ( 1 )$ $\Omega ( \frac { \log n \eta _ { \infty } } { \log \log n } )$ . Finally, it is straightforward to see that the summation of the prediction error $\eta _ { 1 } \leq \mathrm { O P T } = \mathsf { \breve { O } } ( \mathrm { \breve { 1 } } )$ . □

# 5 EXPERIMENTS

We validate our online algorithms on various datasets of different types. In particular, we consider three Euclidean data sets, a) Twitter (Chan et al.), b) Adult (Dua & Graff, 2017), and c) Non-Uni (Cebecauer & Buzna, 2018) which are datasets consisting of numeric features in $\mathbb { R } ^ { d }$ and the distance is measured by $\ell _ { 2 }$ , and one graph dataset, US-PG (Rossi & Ahmed, 2015) which represents US power grid in a graph, where the points are vertices in a graph and the distance is measured by the shortest path distance. The opening cost is non-uniform in Non-Uni dataset, while it is uniform in all the other three. These datasets have also been used in previous papers that study facility location and related clustering problems (Chierichetti et al., 2017; Chan et al., 2018; Cohen-Addad et al., 2019). A summary of the specification of datasets can be found in Table 1.

Tradeoff between $\eta _ { \infty }$ and empirical competitive ratio. Our first experiment aims to measure the tradeoff between $\eta _ { \infty }$ and the empirical competitive ratio of our algorithm, and we compare against two “extreme” baselines, a) the vanilla Meyerson’s algorithm without using predictions which we call MEYERSON, and b) a naive algorithm that always follows the prediction which we call FOLLOW-PREDICT. Intuitively, a simultaneously consistent and robust online algorithm should

Table 1: Specifications of datasets   

<table><tr><td>dataset</td><td>type</td><td>size</td><td># of dimension/edges</td><td>non-uniform</td></tr><tr><td>Twitter</td><td>Euclidean</td><td>30k</td><td># dimension = 2</td><td>no</td></tr><tr><td>Adult</td><td>Euclidean</td><td>32k</td><td># dimension = 6</td><td>no</td></tr><tr><td>US-PG</td><td>Graph</td><td>4.9k</td><td># edges = 6.6k</td><td>no</td></tr><tr><td>Non-Uni</td><td>Euclidean</td><td>4.8k</td><td># dimension = 2</td><td>yes</td></tr></table>

perform similarly to the FOLLOW-PREDICT baseline when the prediction is nearly perfect, while comparable to MEYERSON when the prediction is of low quality.

Since it is NP-hard to find the optimal solution to facility location problem, we run a simple 3- approximate MP algorithm (Mettu & Plaxton, 2000) to find a near optimal solution $F ^ { \star }$ for every dataset, and we use this solution as the offline optimal solution (which we use as the benchmark for the competitive ratio). Then, for a given $\eta _ { \infty }$ and a demand point $x$ , we pick a random facility $f \in { \mathcal { F } }$ such that $\eta _ { \infty } / 2 \le d ( f , F _ { x } ^ { \star } ) \le \eta _ { \infty }$ as the prediction, where $F _ { x } ^ { \star }$ is the point in $F ^ { \star }$ that is closest to $x$ . The empirical competitive ratio is evaluated for every baselines as well as our algorithm on top of every dataset, subject to various values of $\eta _ { \infty }$ .

All our experiments are conducted on a laptop with Intel Core i7 CPU and 16GB memory. Since the algorithms are randomized, we repeat every run 10 times and take the average cost.

In Figure 1 we plot for every dataset a line-plot for all baselines and our algorithm, whose x-axis is $\eta _ { \infty }$ and y-axis is the empirical competitive ratio. Here, the scale of $\mathbf { X }$ -axis is different since $\eta _ { \infty }$ is dependent on the scale of the dataset. From Figure 1, it can be seen that our algorithm performs consistently better than MEYERSON when $\eta _ { \infty }$ is relatively small, and it has comparable performance to MEYERSON when $\eta _ { \infty }$ is so large that the FOLLOW-PREDICT baseline loses control of the competitive ratio. For instance, in the Twitter dataset, our algorithm performs better than MEYERSON when $\eta _ { \infty } \leq 2$ , and when $\eta _ { \infty }$ becomes larger, our algorithm performs almost the same with MEYERSON while the ratio of FOLLOW-PREDICT baseline increases rapidly.

Furthermore, we observe that our performance lead over MERYERSON is especially significant on dataset Non-Uni whose opening cost is non-uniform. This suggests that our algorithm manages to use the predictions effectively in the non-uniform setting, even provided that the prediction is tricky to use since predictions at “good” locations could have large opening costs. In particular, our algorithm does a good job to find a “nearby” facility that has the correct opening cost and location tradeoff.

A simple predictor and its empirical performance. We also present a simple predictor that can be constructed easily from a training set, and evalute its performance on our datasets. We assume the predictor has access to a training set $T$ . Initially, the predictor runs the 3-approximate MP algorithm on the training set $T$ to obtain a solution $F ^ { * }$ . Then when demand points from the dataset (i.e., test set) $X$ arrives, the predictor periodically reruns the MP algorithm, on the union of $T$ and the already-seen test data $X ^ { \prime } \subseteq X$ , and update $F ^ { * }$ . For a demand point $x$ , the prediction is defined as the nearest facility to $x$ in $F ^ { * }$ .

To evaluate the performance of this simple predictor, we take a random sample of $30 \%$ points from the dataset as the training set $T$ , and take the remaining $70 \%$ as the test set $X$ . We list the accuracy achieved by the predictor in this setup in Table 2. From the table, we observe that the predictor achieves a reasonable accuracy since the ratio of Follow-Predict baseline is comparable to Meyerson’s algorithm. Moveover, when combining with this predictor, our algorithm outperforms both baselines, especially on the Non-Uni dataset where the improvement is almost two times. This not only shows the effectiveness of the simple predictor, but also shows the strength of our algorithm.

Finally, we emphasize that this simple predictor, even without using any advanced machine learning techniques or domain-specific signals/features from the data points (which are however commonly used in designing predictors), already achieves a reasonable performance. Hence, we expect to see an even better result if a carefully engineered predictor is employed.

![](images/84d05567db08318e0568f8c8d9322d8162cc19f49a03f21d5f52f65d07103d30.jpg)  
Twitter dataset

![](images/13b0a14a2b802ce6c8a0a27ba52529db2bba6ab8076699c65b7f3f23bed92f8f.jpg)  
Adult dataset

![](images/80e737ff57196d79463dfbef8a56bbfc9d598415a3f75ebdf34f63f93b0c9af7.jpg)  
US-PG dataset

![](images/f984901781eb70901998d797b390dc600fc63b8f16903134144c5db3cfce7e94.jpg)  
Non-Uni dataset   
Figure 1: The tradeoff between prediction error $\eta _ { \infty }$ and empirical competitive ratio over four datasets: Twitter, Adult, US-GD and Non-Uni.

Table 2: Empirical competitive ratio evaluation for the simple predictor   

<table><tr><td>dataset</td><td>Meyerson</td><td>Follow-Predict</td><td>Ours</td></tr><tr><td>Twitter</td><td>1.70</td><td>1.69</td><td>1.57</td></tr><tr><td>Adult</td><td>1.55</td><td>1.57</td><td>1.49</td></tr><tr><td>US-PG</td><td>1.47</td><td>1.47</td><td>1.43</td></tr><tr><td>Non-Uni</td><td>5.66</td><td>5.7</td><td>2.93</td></tr></table>

# ACKNOWLEDGMENTS

This work is supported by Science and Technology Innovation 2030 — “New Generation of Artificial Intelligence” Major Project No.(2018AAA0100903), Program for Innovative Research Team of Shanghai University of Finance and Economics (IRTSHUFE) and the Fundamental Research Funds for the Central Universities. Shaofeng H.-C. Jiang is supported in part by Ministry of Science and Technology of China No. 2021YFA1000900. Zhihao Gavin Tang is partially supported by Huawei Theory Lab. We thank all anonymous reviewers’ for their insightful comments.

# REFERENCES

Amir Ahmadi-Javid, Pardis Seyedi, and Siddhartha S. Syam. A survey of healthcare facility location. Comput. Oper. Res., 79:223–263, 2017.   
Antonios Antoniadis, Themis Gouleakis, Pieter Kleer, and Pavel Kolev. Secretary and online matching problems with machine learned advice. In NeurIPS, 2020.   
Yossi Azar, Stefano Leonardi, and Noam Touitou. Flow time scheduling with uncertain processing time. In STOC, pp. 1070–1080, 2021.

Etienne Bamas, Andreas Maggiori, Lars Rohwedder, and Ola Svensson. Learning augmented energy ´ minimization via speed scaling. In NeurIPS, 2020a.   
Etienne Bamas, Andreas Maggiori, and Ola Svensson. The primal-dual method for learning aug- ´ mented algorithms. In NeurIPS, 2020b.   
Matej Cebecauer and L’ubos Buzna. Large-scale test data set for location problems. ˇ Data in brief, 17:267–274, 2018.   
T.-H. Hubert Chan, Arnaud Guerqin, and Mauro Sozio. Fully dynamic $k$ -center clustering. In WWW, pp. 579–587. ACM, 2018.   
T. Hubert Chan, A. Guerqin, and M. Sozio. Twitter data set. URL https://github.com/ fe6Bc5R4JvLkFkSeExHM/k-center.   
Flavio Chierichetti, Ravi Kumar, Silvio Lattanzi, and Sergei Vassilvitskii. Fair clustering through fairlets. In NIPS, pp. 5029–5037, 2017.   
Vincent Cohen-Addad, Niklas Hjuler, Nikos Parotsidis, David Saulpic, and Chris Schwiegelshohn. Fully dynamic consistent facility location. In NeurIPS, pp. 3250–3260, 2019.   
Gabriella Diveki and Csan ´ ad Imreh. Online facility location with facility movements. ´ Central Eur. J. Oper. Res., 19(2):191–200, 2011.   
Zvi Drezner and Horst W. Hamacher. Facility location - applications and theory. Springer, 2002.   
Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL https://archive. ics.uci.edu/ml/datasets/adult.   
Paul Dutting, Silvio Lattanzi, Renato Paes Leme, and Sergei Vassilvitskii. Secretaries with advice. ¨ In EC, pp. 409–429, 2021.   
David Eisenstat, Claire Mathieu, and Nicolas Schabanel. Facility location in evolving metrics. In ICALP, volume 8573, pp. 459–470, 2014.   
Dimitris Fotakis. On the competitive ratio for online facility location. Algorithmica, 50(1):1–57, 2008.   
Dimitris Fotakis, Evangelia Gergatsouli, Themis Gouleakis, and Nikolas Patris. Learning augmented online facility location. CoRR, abs/2107.08277, 2021a.   
Dimitris Fotakis, Loukas Kavouras, and Lydia Zakynthinou. Online facility location in evolving metrics. Algorithms, 14(3):73, 2021b.   
Sreenivas Gollapudi and Debmalya Panigrahi. Online algorithms for rent-or-buy with expert advice. In ICML, volume 97 of Proceedings of Machine Learning Research, pp. 2319–2327. PMLR, 2019.   
Zhihao Jiang, Pinyan Lu, Zhihao Gavin Tang, and Yuhao Zhang. Online selection problems against constrained adversary. In ICML, volume 139 of Proceedings of Machine Learning Research, pp. 5002–5012. PMLR, 2021.   
Andreas Klose and Andreas Drexl. Facility location models for distribution system design. Eur. J. Oper. Res., 162(1):4–29, 2005.   
Gilbert Laporte, Stefan Nickel, and Francisco Saldanha da Gama. Location science. Springer, 2019.   
Silvio Lattanzi, Thomas Lavastida, Benjamin Moseley, and Sergei Vassilvitskii. Online scheduling via learned weights. In SODA, pp. 1859–1877, 2020.   
Thodoris Lykouris and Sergei Vassilvitskii. Competitive caching with machine learned advice. J. ACM, 68(4):24:1–24:25, 2021.   
Andres Munoz Medina and Sergei Vassilvitskii. Revenue optimization with approximate bid pre- ˜ dictions. In NIPS, pp. 1858–1866, 2017.

M. Teresa Melo, Stefan Nickel, and Francisco Saldanha-da-Gama. Facility location and supply chain management - A review. Eur. J. Oper. Res., 196(2):401–412, 2009.   
Ramgopal R. Mettu and C. Greg Plaxton. The online median problem. In FOCS, pp. 339–348. IEEE Computer Society, 2000.   
Adam Meyerson. Online facility location. In FOCS, pp. 426–431. IEEE Computer Society, 2001.   
Alessandro Panconesi, Flavio Chierichetti, Giuseppe Re, Matteo Almanza, and Silvio Lattanzi. Online facility location with multiple advice. In NeurIPS 2021, 2021.   
Manish Purohit, Zoya Svitkina, and Ravi Kumar. Improving online algorithms via ML predictions. In NeurIPS, pp. 9684–9693, 2018.   
Dhruv Rohatgi. Near-optimal bounds for online caching with machine learned advice. In SODA, pp. 1834–1845. SIAM, 2020.   
Ryan A. Rossi and Nesreen K. Ahmed. The network data repository with interactive graph analytics and visualization. In AAAI, 2015. URL http://networkrepository.com.   
Alexander Wei and Fred Zhang. Optimal robustness-consistency trade-offs for learning-augmented online algorithms. In NeurIPS, 2020.

# A $t$ -OUTLIER SETTING

Observe that our main error parameter $\eta _ { \infty }$ could be very sensitive to a single outlier prediction that has a large error. To make the error parameter bahave more smoothly, we introduce an integer parameter $1 \leq t \leq n$ , and define $\eta _ { \infty } ^ { ( t ) }$ as the $t$ -th largest prediction error, i.e., the maximum prediction error excluding the $t - 1$ high-error outliers. Define $\eta _ { 1 } ^ { ( \bar { t } ) }$ similarly.

In the following, stated in Theorem A.1, we argue that our algorithm, without any modification (but with an improved analysis), actually has a ratio of $\begin{array} { r } { O ( \log ( 1 + t ) + \log { \frac { n \eta _ { \infty } ^ { ( t ) } } { \mathrm { O P T } } } ) } \end{array}$ that holds for every $t$ . This also means the ratio would be $\begin{array} { r } { \operatorname* { m i n } _ { 1 \le t \le n } O ( \log ( 1 + t ) + \log \frac { n \eta _ { \infty } ^ { ( t ) } } { \mathrm { O P T } } ) } \end{array}$ nη(t) ∞ , and this is clearly a generalization of Theorem 1.1, since $\eta _ { \infty } ^ { ( t ) } = \eta _ { \infty }$ when $t = 1$ .

Moreover, we note that our lower bound, Theorem 1.2, is still valid for ruling out the possibility of replacing $\eta _ { \infty } ^ { ( t ) }$ with $\eta _ { 1 } ^ { ( t ) }$ in the abovementioned ratio, since one can still apply Theorem 1.2 with $t = 1$ . However, it is an interesting open question to explore whether or not an algorithm with a ratio like $\begin{array} { r } { O ( \log t + \frac { \eta _ { 1 } ^ { ( t ) } } { \mathrm { O P T } } ) } \end{array}$ 1OPT ) for some specific t (e.g., t ≥ √n) exists. η(t) $t$ $t \geq \sqrt { n } )$

Theorem A.1. For every integer $\begin{array} { r l r l r l } { { 1 } } & { { } \le { } } & { t } & { { } \le { } } & { n } \end{array}$ , Algorithm 1 is $O ( \log ( t ~ + ~ 1 ) ~ + ~$ $\begin{array} { r l } { \ } & { { } \operatorname* { m i n } \{ \log n , \operatorname* { m a x } \{ 1 , \log \frac { n \eta _ { \infty } ^ { ( t ) } } { \mathrm { O P T } } \} \} \big ) } \end{array}$ -competitive.

Proof sketch. Let $\begin{array} { c c l } { \Gamma } & { = } & { \left( \gamma _ { 1 } , \gamma _ { 2 } , \dots , \gamma _ { t - 1 } \right) } \end{array}$ be indices of the $t \mathrm { ~ - ~ } 1$ largest prediction error $d ( f _ { x _ { \gamma } } ^ { \mathrm { p r e d } } , f _ { x _ { \gamma } } ^ { \mathrm { o p t } } )$ , f x 1 2 t−1 . The high level idea is to break the dataset $X$ −  into a good part $X _ { G } : = \{ x _ { i } : i \in [ n ] \backslash \Gamma \}$ and a bad part $X _ { B } : = \{ x _ { i } : i \in \Gamma \}$ according to whether or not the prediction is within the outlier, and we argue that the expected cost of ALG on $X _ { G }$ is $\begin{array} { r } { O \big ( \frac { n \eta _ { \infty } ^ { ( t ) } } { \mathrm { O P T } } \big ) } \end{array}$ times larger than that in OPT, and show the expected cost on $X _ { B }$ is $O ( \log ( t + 1 ) )$ times.

For the good part $X _ { G }$ , we let $X _ { - } ( f ^ { * } ) : = X ( f ^ { * } ) \backslash \Gamma$ and apply a similar analysis as in the proof of Lemma 3.5 to show that

$$
E \left[ \sum_ {x \in X _ {-} (f ^ {*})} \operatorname {c o s t} (x) \right] \leq O \left(\max  \left\{1, \log \frac {n \eta_ {\infty} ^ {(t)}}{\mathrm {O P T}} \right\}\right) \left(w (f ^ {*}) + \sum_ {x \in X _ {-} (f ^ {*})} d (x, f ^ {*})\right).
$$

For the bad part $X _ { B }$ , we assume all predictions for points in $X _ { B }$ are of infinite error (which could only increase the cost of the algorithm), and let $X _ { + } ( f ^ { * } ) : = X ( f ^ { * } ) \cap \Gamma$ . Then this lies in the case

of Section 3.2, where $7 \eta _ { \infty } > 4 w ( f ^ { \ast } )$ , which essentially means the prediction is almost useless and the whole proof reverts to pure Meyerson’s algorithm. Combine the arguments from Section 3.2 and the analysis for the short distance stage, we have

$$
E \left[ \sum_ {x \in X _ {+} (f ^ {*})} \operatorname {c o s t} (x) \right] \leq O (\log (t + 1)) \left(w (f ^ {*}) + \sum_ {x \in X _ {+} (f ^ {*})} d (x, f ^ {*})\right).
$$

We finish the proof by combining the bound for the two parts.