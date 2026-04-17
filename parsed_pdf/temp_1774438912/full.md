# Towards Rotation Invariance in Object Detection

Agastya Kalra1, Guy Stoppi1, Bradley Brown1, Rishav Agarwal1 and Achuta Kadambi1,2 1Akasha Imaging, Palo Alto CA 2UCLA, Los Angeles CA

{agastya, guy.stoppi, bradley.brown, rishav}@akasha.im, {achuta}@ee.ucla.edu

# Abstract

Rotation augmentations generally improve a model’s invariance/equivariance to rotation - except in object detection. In object detection the shape is not known, therefore rotation creates a label ambiguity. We show that the defacto method for bounding box label rotation, the Largest Box Method, creates very large labels, leading to poor performance and in many cases worse performance than using no rotation at all. We propose a new method of rotation augmentation that can be implemented in a few lines of code. First, we create a differentiable approximation of label accuracy and show that axis-aligning the bounding box around an ellipse is optimal. We then introduce Rotation Uncertainty (RU) Loss, allowing the model to adapt to the uncertainty of the labels. On five different datasets (including COCO, PascalVOC, and Transparent Object Bin Picking), this approach improves the rotational invariance of both one-stage and two-stage architectures when measured with AP, AP50, and AP75. The code is available at https: //github.com/akasha-imaging/ICCV2021.

# 1. Introduction

It is desirable for object detectors to work when scenes are rotated. But there is a problem: methods like Convolutional Neural Networks (CNNs) may be scale and translation invariant but CNNs are not rotation invariant [15]. To overcome this problem, the training dataset can be expanded to include data at new rotation angles. This is known as rotation augmentation. In object detection, rotation augmentation can be abstracted as follows: given an original bounding box and any desired angle of rotation, how should we determine the axis-aligned rotated bounding box label? If the shape of the object is known, this is quite simple: we rotate the object shape and re-compute the bounding box. However in the case of object detection, the shape is unknown.

The prevailing folk wisdom in the community is to select a label large enough to completely overlap the ro-

![](images/71c4894cac7906418d12dcd22582df39cb7ef3f4e0329f3e5337e4e0e28c0bf8.jpg)

![](images/280989d3403f198ff91165ba6694ded8d2c91788214e77dc6124cde1f448dc2a.jpg)

![](images/047e79a432ed987ab9316270ad04db5c92b3b169a7973654ecad15d2305b7436.jpg)  
Figure 1: A method is proposed to properly rotate a bounding box for rotation augmentation. The previous solution of largest box is an overestimate of the perfect bounding box for a rotated scene. See table for how choice of rotation augmentation affects object detection performance.

<table><tr><td>Method</td><td>AP50</td><td>AP75</td></tr><tr><td>No Rotation</td><td>51.35</td><td>35.63</td></tr><tr><td>Largest Box</td><td>51.49</td><td>14.95</td></tr><tr><td>Our Rotation</td><td>55.73</td><td>38.05</td></tr><tr><td>Perfect Rotation</td><td>55.91</td><td>39.61</td></tr></table>

Models tested on COCO validation set with 20° test set rotation.

tated box [45]. In studying this problem, we find that this method may hurt performance, and on COCO [22], we find that every other prior we tried is better. Yet somehow this “Largest Box” method is very prevalent both in academia and in large scale object detection codebases [43, 4, 45, 37, 28, 26, 5, 7, 1, 17, 6, 40]. Indeed, recent analysis has found that largest box is only robust to about $< 3 ^ { \circ }$ [16] of rotation.

In this paper, we propose an advance on the largest box solution that achieves significantly better performance on five object detection datasets; while retaining the simplicity of a few lines of code implementation.

# 1.1. Contributions

In a nutshell, our solution has two aspects. First, we derive an elliptical shape prior from first principles to deter-

mine the rotated box label. We compare it to many other novel priors and show this is optimal. Second, we introduce a novel Rotation Uncertainty (RU) Loss function, which allows the network to adapt the labels at higher rotations using priors from lower rotations based on label certainty. We demonstrate the effectiveness of this solution by both improving performance datasets where rotation is important such as Pascal VOC [14] and Transparent Object Bin Picking [18] and generalizing to novel test time rotations on MS COCO [22] (Figure 1).

# 1.2. Scope

Rotation data augmentation in object detection is not new. This paper is not about finding the best overall way to use a rotation data augmentation. For that - a brute force search or papers like AutoAugment [45] might be better examples. This paper focuses solely on methods to perform a rotation augmentation on axis-aligned bounding boxes. When implemented, these proposed modifications boil down to a few lines of code and leave little reason to use the current Largest Box method.

# 2. Related Work

Data Augmentations are an effective technique to boost the performance of object detection. Data augmentation increases the quantity and improves the diversity of data. Data augmentations are of two types. Photo-metric transforms modify the color channels such that the detector becomes invariant to change in lighting and color. Classical photometric techniques include adding Gaussian blur or adding colour jitters. Modern photo-metric augmentations like Cutout [11] and CutMix [44], randomly remove patches of the image. On the other hand, geometric transforms modify the image’s geometry, making the detector invariant to position and orientation. Geometric modifications require corresponding changes to the labels as well. Geometric transforms are difficult to implement [5] and contribute more towards accuracy improvements [38]. We focus specifically on rotation augmentations and object detection.

Rotation Augmentation in Object Detection is currently done by the largest box method for major repositories (e.g. [5, 7, 1, 17, 6, 40]) and publications (e.g. [43, 4, 45, 37, 28, 26]) that do bounding box rotation for deep learning object detection. The largest box method does a great job guaranteeing containment, but at large angles, it severely overestimates the bounding box’s size. Figure 2 shows an example of these over-sized bounding boxes. For that reason, FastAI [16] recommends rotation of no more than 3 degrees. Some recent work, such as AutoAugment [45, 37], use rotation as part of a complex learned data augmentation scheme. While learning rotation augmentation directly is interesting, it requires extensive computing resources. We

![](images/91aeebf7d9af95b54ddca90bb2a0db0668251efe11ea66d8ec1ab70cee4f427a.jpg)  
Figure 2: Comparison between our method and other methods. We show example predictions from models trained with each rotation augmentation above. Our method has comparable performance to using perfect segmentation labels without requiring the extra shape information.

<table><tr><td>Rotation Method</td><td>AP50 (Coarse)</td><td>AP75 (Fine)</td><td>Shape Label</td></tr><tr><td>No Rotation</td><td>Med</td><td>Med</td><td>No</td></tr><tr><td>Largest Box (e.g. [1, 17, 5])</td><td>Med</td><td>Low</td><td>No</td></tr><tr><td>Ellipse + RU Loss (Ours)</td><td>Very High</td><td>High</td><td>No</td></tr><tr><td>Perfect Box (Gold Std)</td><td>Very High</td><td>Very High</td><td>Yes</td></tr></table>

seek to achieve the simplicity of the largest box, with performance improvements for larger angles.

Oriented Bounding Boxes, a sister of object detection, is the task of predicting non-axis aligned bounding boxes, also known as oriented bounding boxes. Several methods like [12, 8, 25], aim to achieve rotation invariance when predicting rotated bounding boxes. However, these methods already have labelled rotated boxes as input and do not end up with loose boxes when the input image is rotated. As this is a different task, it is out of the scope of this paper. Our paper focuses on axis-aligned bounding boxes only.

Rotational Invariance is an important problem to solve in object detection. Classical computer vision methods achieved rotational invariance by extracting features from images [41, 23, 24, 34]. With the rise of neural networks, newer methods attempt to modify the architecture to achieve rotation invariance [10, 9, 43]. These methods rotate the input images and add special layers that learn the object’s orientation in the image. Our general-purpose method can again aid these methods as well.

# 3. Background

# 3.1. Rotation Augmentation for Object Detection

An image is parameterized by $x$ and $y$ coordinates. Suppose the image contains an object with shape $S$ . Let $S$ denote the shape set that describes all points in an object:

$$
S = \{(x, y) \mid \forall (x, y) \in \text {o b j e c t} \}. \tag {1}
$$

In object detection, a bounding box is defined to be the tightest fitting axis-aligned box around a shape. Therefore a shape determines the coordinates of a bounding box,

${ \bf b } = [ x _ { m i n } , y _ { m i n } , x _ { m a x } , y _ { m a x } ] ^ { T }$ . Each of the four edges of the bounding box intersects at least one element of the shape set. Let the operator $\boldsymbol { B }$ represent the perfect conversion of a shape to a bounding box:

$$
\mathbf {b} \triangleq \mathcal {B} (S). \tag {2}
$$

The operator $\boldsymbol { B }$ extracts the bounding box b (tightest fitting axis aligned box) for $S$ by taking the minimum/maximum $( x , y )$ coordinates of the shape. As $S$ is not unique, the same bounding box can be generated by many shapes. For example, a square of side length $d$ and a circle of diameter $d$ are just two of many unique shapes that generate the same bounding box. More formally, let $V _ { \mathbf { b } }$ denote the set of shapes that could possibly generate a bounding box b, such that

$$
V _ {\mathbf {b}} = \left\{S _ {i} \mid \mathcal {B} \left(S _ {i}\right) = \mathbf {b}, \forall S _ {i} \in \mathcal {P} \right\}. \tag {3}
$$

Where $\mathcal { P }$ is the dataset specific distribution of shapes. Let us consider the problem of rotation augmentation where an image and corresponding box label b is rotated by angle $\theta$ . If the shape of the object is known, then we can rotate the original shape by angle $\theta$ using a rotation operator: $\mathcal { R } ^ { \theta } ( S )$ . In analogy to Equation 2, we can then use the perfect method to obtain an axis-aligned bounding box for the rotated image as:

$$
\mathbf {b} _ {\text {p e r f e c t}} ^ {\theta} \triangleq \mathcal {B} \left(\mathcal {R} ^ {\theta} \left(S _ {\text {p e r f e c t}}\right)\right), \tag {4}
$$

We call this method perfect labels, where $S _ { \mathrm { { p e r f e c t } } }$ is the actual shape of the object for a given bounding box b. However, this requires shape labels, which are not available for object detection. In object detection, humans label boxes by implicitly segmenting the shape. Without knowing the shape labels, any shape $S \in V _ { \mathbf { b } }$ could be $S _ { \mathrm { { p e r f e c t } } }$ , leading to many possible boxes $\mathbf { b } ^ { \theta }$ . This paper seeks a method to estimate the rotated bounding box when we do not know the shape. We are only provided with the original bounding box b, which we will hereafter write as $ { \mathbf { b } } ^ { 0 }$ by making explicit that $\theta = 0$ . The problem statement follows.

Problem Statement: Given only an input bounding box $ { \mathbf { b } } ^ { 0 }$ and an angle $\theta$ by which the image should be rotated, find the axis aligned bounding box $\bar { \widehat { \mathbf { b } } } ^ { \theta }$ that: (1) has high IoU with $\mathbf { b } _ { \mathrm { p e r f e c t } } ^ { \theta }$ b; and (2) improves model performance on rotated versions of vision datasets.

# 3.2. Largest Valid Box Method

Rotation augmentation without shape knowledge is not a new problem statement. The de facto method in the object detection community for determining the bounding

box post-rotation with no shape priors is the largest box method. The largest box method is extremely prevalent (e.g. [28, 26, 5, 7, 1, 17, 6, 40, 20, 19, 27, 33, 36, 39]). Just like our proposed method, the largest box takes only the original bounding box $ { \mathbf { b } } ^ { 0 }$ and $\theta$ as input. From Equation 3 it is clear that several shapes could define $ { \mathbf { b } } ^ { 0 }$ . This creates an ambiguity problem. The largest box method chooses the single largest of these possible shapes in area, $S _ { \mathrm { l a r g e s t } }$ . This shape is simply the box itself (Table 1). Treating this as the object shape, Equation 4 can be adapted to obtain

$$
\mathbf {b} _ {\text {l a r g e s t}} ^ {\theta} \triangleq \mathcal {B} \left(\mathcal {R} ^ {\theta} \left(S _ {\text {l a r g e s t}}\right)\right). \tag {5}
$$

The benefit of this method is that it produces a box that is guaranteed to contain the original object [45], and it is easy to implement. The downside is that the method produces oversized boxes [16, 45, 35, 2, 29, 3], and if used generously, hurts performance more than it helps (Table 9). Surprisingly, to our best knowledge, including personal communication with practitioners and posts on internet forums, no alternatives have been adopted. We hope our method will change that.

# 4. Proposed Solution

We now describe our solution to the problem: given $\mathbf { b ^ { 0 } }$ and desired rotation angle $\theta$ , find $\widehat { \mathbf { b } } ^ { \theta }$ . In a nutshell, our solution estimates a rotated bounded box by assuming the original shape is an ellipse (Table 1, Figure 3) and rotating accordingly (Section 4.1). We then adapt the loss function to account for error in the labels (Section 4.3).

# 4.1. Ellipse Method

In this section, we first derive the ellipse assumption from first principles by trying to find the shape that is most likely to have high overlap with potential ground truth boxes. Then we discuss the implementation and intuition of the ellipse method. Finally, we mention other novel methods we developed.

# 4.1.1 Ellipse from Maximizing Expect IoU

We start with a simple assumption: the optimal method for determining a bounding box post-rotation augmentation should maximize label accuracy, which in the case of object detection is measured in IoU.

We define $\widehat { \mathbf { b } } ^ { \theta }$ as the optimal rotated bounding box. We are provided the input angle $\theta$ and box $ { \mathbf { b } } ^ { 0 }$ . From Equation 3, this box could have been generated from any number of shapes: $V _ { \mathbf { b } ^ { 0 } } = \{ S _ { i } \} _ { i = 1 , 2 , . . . , N }$ . For each shape we can use the “perfect method” from Equation 4 to obtain a potential rotated bounding box. Since multiple shapes can lead to the same rotated box, we obtain $M \leq N$ possible bounding boxes, which we write as the set:

![](images/fa32e1a9317d15cd973f6204a0ab2335cdc52cd3e5fe611aad4c5ff6d18caf36.jpg)  
Original Bounding Box (b°)   
3C

![](images/de37aba737284e3f7192eceaca19840ae274f9f3aea28ff3aca1b81fe4cf934c.jpg)  
Largest Box (Previous Work)

![](images/e8cea2fbb47f05da7614f9615f5a5ba2c24fae32113490c46ad6e551fe4421d5.jpg)  
$b 6}$ ellipse Elliipse Box (Ours)

![](images/be949924ea4ef1d842fa657832c45ec3cd9a5308db65404a2a9f2c2838c7cfc1.jpg)  
bθ6}$ perfect Ground Truth (Extra Labels)

![](images/628fb84eb9c16d70729afbe24beb9f22d85873f64497748f6774d8c3a9ccdaf7.jpg)  
Set of all possible ground truth boxes $( \mathsf { Q } _ { \mathrm { ~ } \mathbf { b } } ^ { \boldsymbol { \mathsf { \theta } } } )$   
Figure 3: Our Ellipse method leads to good initial training labels while the Largest Box overestimates the labels. From left to right: (1) The original bounding box prior to rotation. (2) The oversized Largest Box estimate of the ground truth label post rotation. (3) The tighter Ellipse estimate (Section 4.1). (4) The actual ground truth which we get from segmentation shape labels. (5) The set of all possible ground truth boxes given a rotation and an initial box.

$$
\begin{array}{l} Q _ {\mathbf {b} ^ {0}} ^ {\theta} = \text {u n i q u e} \left\{\mathbf {b} _ {j} ^ {\theta} = \mathcal {B} \left(\mathcal {R} ^ {\theta} \left(S _ {i}\right)\right), \forall S _ {i} \in V _ {\mathbf {b} ^ {0}} \right\} (6) \\ = \left\{\mathbf {b} _ {j} ^ {\theta} \right\} _ {j = 1, \dots , M}. (7) \\ \end{array}
$$

Hereafter, and without loss of generality, the paper will assume that $ { \mathbf { b } } ^ { 0 }$ is the input allowing notation to be simplified:

$$
Q ^ {\theta} \triangleq Q _ {\mathbf {b} ^ {0}} ^ {\theta}. \tag {8}
$$

Now the task becomes to pick the “best” of the $M$ possible bounding boxes in $Q ^ { \theta }$ . Recall that the de facto solution is to choose the largest box in $Q ^ { \theta }$ . This largest box is guaranteed to contain the object. However, optimizing for containment does not seem like a good choice to directly address the metric of AP because AP uses IoU to determine true positives, not containment. A more relevant goal for object detection is to select a box that maximizes:

$$
\widehat {\mathbf {b}} ^ {\theta} = \underset {\widehat {\mathbf {b}} ^ {\theta} \in Q ^ {\theta}} {\operatorname {a r g m a x}} I o U \left(\widehat {\mathbf {b}} ^ {\theta}, \mathbf {b} _ {\text {p e r f e c t}} ^ {\theta}\right). \tag {9}
$$

In which case $\widehat { \mathbf { b } } ^ { \theta } = \mathbf { b } _ { \mathrm { p e r f e c t } } ^ { \theta }$ . Of course, we are not given bθ $\mathbf { b } _ { \mathrm { p e r f e c t } } ^ { \theta }$ bθperfect. So for the moment, let us assume any of the bounding boxes in $Q ^ { \theta }$ has an equal chance of being the perfect box. Then, it would make sense to optimize over:

$$
\widehat {\mathbf {b}} ^ {\theta} = \underset {\widehat {\mathbf {b}} ^ {\theta} \in Q ^ {\theta}} {\operatorname {a r g m a x}} \operatorname {A v g} \left\{\operatorname {I o U} \left(\widehat {\mathbf {b}} ^ {\theta}, \mathbf {b} _ {j} ^ {\theta}\right) \mid \forall \mathbf {b} _ {j} ^ {\theta} \in Q ^ {\theta} \right\}. \tag {10}
$$

Now, let us break the assumption that each candidate box in $Q ^ { \theta }$ is equally likely to be the perfect box. Indeed, we know that many shapes can produce the same box (since $M \leq N _ { \mathrm { \ell } }$ ), so certain boxes are more likely than others. For example, the only shape that can produce the largest box is the original box itself, whereas other rotated boxes can

Table 1: Our method can be compared with the Largest Box in the shape domain. The implementation difference is one line of code.   

<table><tr><td>Method</td><td>Shape Definition</td></tr><tr><td>Slargest</td><td>{(x1,y1),(x2,y1),(x2,y2),(x1,y2)}</td></tr><tr><td>Sellipse (Ours)</td><td>{(x,y)|(x-xc)/(bW/2)2+(y-yc)2/bH/2)=1}</td></tr></table>

be generated by multiple object shapes in the dataset. Denote $p ( \mathbf { b } _ { j } ^ { \theta } )$ as the probability that box $\mathbf { b } _ { j } ^ { \theta } = \mathbf { b } _ { \mathrm { p e r f e c t } } ^ { \theta }$ . Then Equation 10 can be reformulated as:

$$
\widehat {\mathbf {b}} ^ {\theta} = \underset {\widehat {\mathbf {b}} ^ {\theta} \in Q ^ {\theta}} {\operatorname {a r g m a x}} \sum_ {i = 1} ^ {M} p \left(\mathbf {b} _ {j} ^ {\theta}\right) I o U \left(\widehat {\mathbf {b}} ^ {\theta}, \mathbf {b} _ {j} ^ {\theta}\right). \tag {11}
$$

Readers may recognize this equation as being analogous to an expectation. We refer to this in the paper as the Expected IoU. The expected IoU is not directly tractable: we do not know $p ( \mathbf { b } _ { j } ^ { \theta } )$ a priori. However, if we can sample $K$ random shapes from a dataset distribution over shapes $S _ { k } \sim \mathcal { P }$ where $\mathbf { b } _ { k } ^ { \theta } = B ( \mathcal { R } ^ { \theta } ( S _ { k } ) )$ we get the following optimization objective:

$$
\widehat {\mathbf {b}} ^ {\theta} = \underset {\widehat {\mathbf {b}} ^ {\theta} \in Q ^ {\theta}} {\operatorname {a r g m a x}} \frac {1}{K} \sum_ {k = 1} ^ {K} I o U \left(\widehat {\mathbf {b}} ^ {\theta}, \mathbf {b} _ {k} ^ {\theta}\right). \tag {12}
$$

Since all object detection datasets do not have shape labels, we sample $\mathcal { P }$ by generating random shapes that touch each side of $b ^ { 0 }$ once. This way we are not dataset-specific. We analyze using COCO shapes in the supplement and show the performance is extremely similar. The above equation is fully differentiable, and so we can solve with gradient ascent. The problem here is that we would then have to solve this equation for every $\theta$ and every box $b$ , and this is not practical. Therefore we generalize this further to a canonical shape.

![](images/298293d95670d84be59ac11d3bb63b4aca98c9c361eaec3d0f3b76a37f3cd9c9.jpg)  
Figure 4: The optimal shape to maximize expected IoU with potential ground truth boxes converges to an ellipse. Curve showing the progression of gradient ascent starting from the largest box and converging to an elliptical shape. The final converged expected IoU for $\widehat { S }$ matches that of the largest inscribed ellipse $S _ { \mathrm { e l l i p s e } }$ .

<table><tr><td>Label Method</td><td>Slargest</td><td>Sellipse (Ours)</td><td>S(Ours)</td></tr><tr><td>Expected IoU</td><td>60.8</td><td>72.9</td><td>72.9</td></tr></table>

Optimizing Equation 12 via a canonical shape: Instead of solving Equation 12 for every possible combination of $b ^ { 0 }$ and $\theta$ , we attempt to find a shape that is optimal across different input bounding boxes. This way, we could solve for some best shape $\widehat { S } \in V _ { \mathbf { b ^ { 0 } } }$ , and solve for $\widehat { \mathbf { b } } ^ { \theta }$ as follows:

$$
\widehat {\mathbf {b}} ^ {\theta} \triangleq \mathcal {B} (\mathcal {R} ^ {\theta} (\widehat {S})), \tag {13}
$$

To obtain the likely shape, we combine Equations 12 and 13 to optimize the quantity:

$$
\widehat {S} = \underset {S \in \mathcal {V} _ {\mathbf {b} ^ {0}}} {\operatorname {a r g m a x}} \sum_ {\theta} \frac {1}{K} \sum_ {k = 1} ^ {K} \left[ I o U \left(\mathcal {B} \left(\mathcal {R} ^ {\theta} (S)\right), \mathbf {b} _ {k} ^ {\theta}\right) \right]. \tag {14}
$$

Note that we now optimize over all rotation angles and aspect ratios simultaneously. This adds enough constraints to find a unique shape. The goal is to find the shape $\widehat { S }$ that produces an augmented bounding box that has high IoU with likely ground truth boxes. Since $\mathcal { R }$ and $\boldsymbol { B }$ are differentiable operators, Equation 14 can be optimized through gradient ascent to solve for $\widehat { S }$ . We provide details and pseudocode and some analysis in the supplement. The stable solution found by gradient ascent is that of an elliptical shape. We show the progression of gradient ascent in Figure 4 from the largest box shape to a circle. If we change the aspect ratio, it simply converges to the largest inscribed ellipse. Also in Figure 4, we show the Expected IoU for the Largest Box shape is much lower than the Ellipse, and in Figure 5 we show that the resulting AP of the Ellipse labels is much better. The elliptical solution is similar to the optimized shape

![](images/64adb177bd134e34251bbbe0178d04b6639da99c1b769eb21c5241059cfefbc3.jpg)  
Figure 5: Comparing Label AP (assuming uniform confidence) at different IOU thresholds for both methods at a $1 5 ^ { o } - 3 0 ^ { o }$ rotation augmentation. Ours is significantly better at AP50 and AP75.

<table><tr><td rowspan="2"></td><td colspan="4">AP50</td><td colspan="4">AP75</td></tr><tr><td>10°</td><td>20°</td><td>30°</td><td>40°</td><td>10°</td><td>20°</td><td>30°</td><td>40°</td></tr><tr><td>Largest Box</td><td>98.2</td><td>93.79</td><td>86.31</td><td>82.9</td><td>59.2</td><td>25.6</td><td>19.9</td><td>17.3</td></tr><tr><td>Ellipse (Ours)</td><td>99.6</td><td>98.6</td><td>97.0</td><td>96.5</td><td>86.8</td><td>56.2</td><td>47.2</td><td>46.5</td></tr></table>

for various tested distributions of $\mathcal { P }$ , including the random model described in the previous paragraph.

# 4.1.2 The Ellipse Method

When we model the shape as an ellipse, we can find the estimated bounding box as:

$$
\widehat {\mathbf {b}} ^ {\theta} = \mathcal {B} \left(\mathcal {R} ^ {\theta} \left(S _ {\text {e l l i p s e}}\right)\right), \tag {15}
$$

where $S _ { \mathrm { e l l i p s e } }$ is the largest inscribed ellipse inside $ { \mathbf { b } } ^ { 0 }$ , expressed as:

$$
S _ {\text {e l l i p s e}} = \left\{(x, y) \left| \frac {(x - x _ {c}) ^ {2}}{\left(b _ {W} ^ {0} / 2\right) ^ {2}} + \frac {(y - y _ {c}) ^ {2}}{\left(b _ {H} ^ {0} / 2\right) ^ {2}} = 1 \right. \right\}, \tag {16}
$$

where $( x _ { c } , y _ { c } )$ is the location of the center of $b ^ { 0 }$ and $b _ { W } ^ { 0 } , b _ { H } ^ { 0 }$ are the width and height of $b ^ { 0 }$ respectively.

This equation is fast, simple to implement, and highperforming on modern vision datasets. The elliptical approximation can be implemented in the same line of code as the largest box method (cf. Appendix A), yet it greatly improves performance. We see in Figure 5 the ellipse labels are far more accurate than the largest box labels. However, one disadvantage with the proposed elliptical box method is that the elliptical box can underestimate the object size or aspect ratio. This still causes some noise in the labels, especially at large rotations. We mitigate this by allowing the model to adapt labels at higher rotations based on priors from lower rotations in Section 4.3.

$$
C (\theta) = \max  \left(0. 5, 1 + \frac {1 - \cos (4 \theta)}{2 \cos (4 \delta) - 2}\right)
$$

![](images/26445b5d55bf83fb71aa7e30e8fa95a58fbef25a7bf8895409752f76caf40bd6.jpg)  
Figure 6: Rotational certainty used by RU Loss as a function of theta plotted for different hyper-parameters $\delta$ where $\delta$ is the angle at which $C ( \theta ) = 0 . 5$ .

# 4.2. Other Methods

We do not limit our analysis to the ellipse method. To perform a complete study we came up with an additional 4 methods. To conserve space, full details and results of these novel methods are available in the supplement Appendix B, we provide a quick summary for these methods here.

• Scaled Octagon: We use an octagon with a scaling factor (s) to interpolate between the largest box shape and a diamond shape.   
• Random Boxes: We sample random valid boxes and use those as ground truth labels.   
• RotIoU: We select the label that has the maximum IoU with the rotated ground truth box rather than the expected axis-aligned ground truth box.   
• COCO Shape: Rather than using random shapes for the optimization, we use the shapes from the COCO dataset. We keep results from this to the supplement since the performance between this and the ellipse method is negligible and we want this paper to be dataset independent and easy to implement.

# 4.3. Rotation Uncertainty Loss

As shown in Figure 4 the expected IoU with random shapes is 72.9. This means attaining good performance at the higher APs, like AP75, will be very difficult using just these labels. To tackle this problem, we create a custom loss function that adapts the regression loss to account for the uncertainty of the rotation. The idea is simple - if we are uncertain of the label, we turn off the regression loss if the model is close enough. The labels are more uncertain

![](images/b73551ae0b9d21130e251c0f3a51091e069b0b23edf8804bb0c27eeaa43e00a7.jpg)  
Figure 7: The Expected IoU of a rotation method is heavily correlated with performance. Our final Ellipse is optimal for both.

Table 2: The AP at different test rotations on the COCO val2017 set for different methods. (a) The previous method of largest box leads to the worst performance - every other idea we had was better. (b) The Ellipse is the best of all the label generation methods. (c) Our RU Loss with $\delta \ : = \ : 1 0 ^ { \circ }$ leads to the best AP across all rotations and therefore we use this as our final method. Note: We bold within 0.2 of best result.   

<table><tr><td>COCO val2017 Ablations</td><td>0°</td><td>10°</td><td>20°</td><td>30°</td></tr><tr><td colspan="5">(a) Previous method</td></tr><tr><td>Largest Box(e.g. [4, 45, 37])</td><td>35.20</td><td>28.37</td><td>22.34</td><td>18.47</td></tr><tr><td colspan="5">(b) Our Rotation Label Methods (Section 4.2)</td></tr><tr><td>Random</td><td>37.39</td><td>35.59</td><td>32.22</td><td>28.33</td></tr><tr><td>Octagon s = 0.1</td><td>35.82</td><td>31.64</td><td>27.16</td><td>23.54</td></tr><tr><td>Octagon s = 0.2</td><td>36.52</td><td>34.57</td><td>31.65</td><td>28.15</td></tr><tr><td>Octagon s = 0.5 (Diamond)</td><td>38.36</td><td>35.39</td><td>28.76</td><td>22.92</td></tr><tr><td>RotIoU</td><td>38.32</td><td>36.48</td><td>32.68</td><td>28.94</td></tr><tr><td>Ellipse (Section 4.1.2)</td><td>38.21</td><td>36.83</td><td>33.59</td><td>29.95</td></tr><tr><td colspan="5">(c) With Our Loss (Section 4.3)</td></tr><tr><td>Ellipse + RU Loss δ = 45°</td><td>38.54</td><td>37.45</td><td>34.56</td><td>31.26</td></tr><tr><td>Ellipse + RU Loss δ = 30°</td><td>39.09</td><td>37.99</td><td>35.45</td><td>32.25</td></tr><tr><td>Ellipse + RU Loss δ = 15°</td><td>39.14</td><td>38.19</td><td>35.78</td><td>32.50</td></tr><tr><td>Ellipse + RU Loss δ = 10° (Final)</td><td>39.33</td><td>38.31</td><td>36.00</td><td>32.72</td></tr></table>

as the rotation approaches $4 5 ^ { \circ }$ , $1 3 5 ^ { \circ }$ , 225◦, 315◦. and perfectly certain at $0 ^ { \circ }$ , $9 0 °$ , $1 8 0 ^ { \circ }$ and $2 7 0 ^ { \circ }$ . We formalize on the concept of certainty (Figure 6) as a function of $\theta$ :

$$
\alpha = 2 \cos (4 \delta), \tag {17}
$$

$$
C (\theta) = 1 + \frac {1}{\alpha - 2} (1 - \cos (4 \theta)). \tag {18}
$$

This function maps a rotation $\theta$ to an IoU threshold $C ( \theta )$ . We use this IoU threshold to serve as an indicator for applying regression loss. If the predicted box is greater than

![](images/e7b8a93687f0533bee42c2efa1323f6c2a2d927d2e66f4be05fdd23c2b9ce27f.jpg)

![](images/d0641b79b5c7172b3c34e7eb5f5283dd62fe128b28ccee32af6f895a92e339b6.jpg)

![](images/1fc695d2d6862a13b0d3e0c8b11535d639f49e30520e3e0b18bbd066493acf6b.jpg)

![](images/fecb9c5f210fade6ef8066961f1b05e4330a332bc902ae7e9f4b0f7958c4ce01.jpg)

![](images/a132c6c7c8cf4e8e8a0bb7efd3762133ada625894557c75204197d79020aab96.jpg)  
Figure 8: Example bounding box predictions for Largest Box model (top row) and Ellipse model (bottom row) for all 5 datasets. We can see that ours produces tighter bounding boxes overall.

Table 3: Across four separate datasets we show that our method of rotation augmentation leads to an improved performance where the previous method hurts performance. Especially in the case of transparent object bin picking where the largest box is almost $50 \%$ worse and ours is $4 . 5 \%$ better AP75.   

<table><tr><td>Datasets (At 0° Test Rotation)</td><td colspan="3">Pascal VOC [14]</td><td colspan="3">Transparent Bin Picking [18]</td><td colspan="3">Synthetic Fruit Dataset [13]</td><td colspan="3">Oxford Pets [30]</td></tr><tr><td>Methods</td><td>AP</td><td>AP50</td><td>AP75</td><td>AP</td><td>AP50</td><td>AP75</td><td>AP</td><td>AP50</td><td>AP75</td><td>AP</td><td>AP50</td><td>AP75</td></tr><tr><td>No Rotation</td><td>51.94</td><td>80.91</td><td>56.54</td><td>48.53</td><td>79.14</td><td>54.3</td><td>84.3</td><td>95.07</td><td>92.6</td><td>80.70</td><td>92.80</td><td>88.76</td></tr><tr><td rowspan="2">Largest Box(e.g. [4, 45, 37]) 
(relative improvement)</td><td>50.23</td><td>81.31</td><td>54.3</td><td>37.49</td><td>79.09</td><td>28.45</td><td>83.47</td><td>95.05</td><td>92.24</td><td>79.54</td><td>94.20</td><td>90.03</td></tr><tr><td>-3.29%</td><td>0.49%</td><td>-3.96%</td><td>-22.7%</td><td>-0.06%</td><td>-47.6%</td><td>-0.98%</td><td>-0.02%</td><td>-0.39%</td><td>-1.43%</td><td>1.56%</td><td>1.43%</td></tr><tr><td rowspan="2">Ellipse + RU Loss (Ours) 
(relative improvement)</td><td>52.89</td><td>81.57</td><td>57.97</td><td>50.36</td><td>81.78</td><td>56.76</td><td>84.83</td><td>95.83</td><td>93.17</td><td>81.28</td><td>94.37</td><td>91.09</td></tr><tr><td>1.84%</td><td>0.82%</td><td>2.53%</td><td>3.78%</td><td>3.35%</td><td>4.53%</td><td>1.05%</td><td>0.80%</td><td>0.62%</td><td>0.72%</td><td>1.69%</td><td>2.63%</td></tr></table>

$m a x ( 0 . 5 , C ( \theta ) )$ , it uses the regression loss, otherwise, it does not and assumes the model’s prediction is correct. We parameterize $C$ with δ. $\delta$ is the angle at which $C ( \theta ) = 0 . 5$ . We visualize $C$ in Figure 6. We bound it by 0.5 since that is the threshold for anchor-matching in standard object detection architectures [21].

This function allows the model to take the priors it learns at the confident rotations and apply them to the higher rotations, preventing it from overfitting to poor labels. We show in Table 2.

# 5. Results

# 5.1. Setup

Our hardware setup contains only a single P100 GPU for training, and all our code is implemented in Detectron2 [42] with Pytorch [31]. We use the default training pipeline for both Faster-RCNN [32] and RetinaNet [21]. We conduct most of our experiments on the standard COCO benchmark since it contains a variety of objects with many different shapes - making it a challenging test set.

Training Since we have only a single GPU, we can only fit a batch size of 3. To account for this, we increase the training time by around ${ 5 } \mathrm { x }$ from the default configurations. This allows us to match online available pre-trained baselines for RetinaNet [21] and Faster-RCNN [32]. Since most

datasets are right-side-up images, we train with a normal distribution with a mean of 0 and a standard deviation of 15 degrees for all experiments. Since this paper aims to find the optimal rotation augmentation method, not the strategy for applying rotation augmentation, we do not try other combinations. This may be left for future work.

Testing: For all datasets except COCO we do not have complete segmentation labels, so we only test on the standard test set $( 0 ^ { \circ } )$ . For COCO we generate our test set by taking the COCO val2017 set and rotating it from $0 ^ { \circ } \mathrm { - } 4 0 ^ { \circ }$ to simulate out-of-distribution rotations. We then bucket these rotations in intervals of 10 and evaluate using segmentation labels to generate ground truth. We leave COCO results for Faster-RCNN to supplement because they are similar to the results for RetinaNet shown below.

# 5.2. Ablation Studies

In Figure 7 and the accompanying table, we conduct a thorough ablation study on both the method for choosing the label and the impact of the RU Loss function.

Justifying EIoU Optimization: In Section 4.1.1 we assumed that the optimal method for label rotation should maximize label accuracy, which we approximated as Expected IoU (Eq. 14). In Figure 7 we demonstrate a strong correlation between Expected IoU and performance

![](images/12931172d7a3caf21ffdc1b75dd7f19b71fff3254e3025dab14e42a6b63cca3b.jpg)

![](images/f1445e6ed2b8396953b81f28bbe0a069793b93ce73d1471b169fea92775d162a.jpg)

![](images/078e91d0f8324fd56cea238663402cb16696c339bad99cc3169698b7e9ca05eb.jpg)  
Figure 9: Our method of Ellipse $^ +$ RU Loss performs close to perfect labels for AP50 and performs better than No Rotation and Largest Box for AP and AP75 - demonstrating the first reliable rotation augmentation without shape labels.

<table><tr><td rowspan="2">COCO val2017 Results</td><td colspan="4">AP</td><td colspan="4">AP50</td><td colspan="4">AP75</td></tr><tr><td>\( 0^{\circ} \)</td><td>\( 10^{\circ} \)</td><td>\( 20^{\circ} \)</td><td>\( 30^{\circ} \)</td><td>\( 0^{\circ} \)</td><td>\( 10^{\circ} \)</td><td>\( 20^{\circ} \)</td><td>\( 30^{\circ} \)</td><td>\( 0^{\circ} \)</td><td>\( 10^{\circ} \)</td><td>\( 20^{\circ} \)</td><td>\( 30^{\circ} \)</td></tr><tr><td>No Rotation</td><td>39.26</td><td>37.54</td><td>33.68</td><td>29.19</td><td>59.68</td><td>56.88</td><td>51.35</td><td>45.37</td><td>41.69</td><td>40.14</td><td>35.63</td><td>30.39</td></tr><tr><td>Largest Box (e.g. [4, 45, 37])</td><td>35.20</td><td>28.37</td><td>22.34</td><td>18.47</td><td>58.79</td><td>56.31</td><td>51.49</td><td>46.30</td><td>36.00</td><td>25.37</td><td>14.95</td><td>10.91</td></tr><tr><td>(relative improvement)</td><td>-10.3%</td><td>-24.4%</td><td>-33.7%</td><td>-36.7%</td><td>-1.48%</td><td>-1.01%</td><td>0.27%</td><td>2.04%</td><td>-13.6%</td><td>-36.8%</td><td>-58.1%</td><td>-64.1%</td></tr><tr><td>Ellipse + RU Loss (Ours)</td><td>39.33</td><td>38.31</td><td>36.00</td><td>32.72</td><td>60.08</td><td>58.66</td><td>55.73</td><td>51.60</td><td>41.74</td><td>40.71</td><td>38.05</td><td>33.97</td></tr><tr><td>(relative improvement)</td><td>0.17%</td><td>2.05%</td><td>6.88%</td><td>12.1%</td><td>0.67%</td><td>3.12%</td><td>8.54%</td><td>13.7%</td><td>0.13%</td><td>1.42%</td><td>6.79%</td><td>11.8%</td></tr><tr><td>Perfect Labels</td><td>39.66</td><td>39.17</td><td>37.24</td><td>34.08</td><td>60.28</td><td>58.89</td><td>55.91</td><td>51.83</td><td>42.05</td><td>41.73</td><td>39.61</td><td>36.02</td></tr><tr><td>(relative improvement)</td><td>1.02%</td><td>4.36%</td><td>10.6%</td><td>16.7%</td><td>1.00%</td><td>3.53%</td><td>8.88%</td><td>14.2%</td><td>0.88%</td><td>3.94%</td><td>11.2%</td><td>18.5%</td></tr></table>

on COCO at $1 0 ^ { \circ }$ across all methods, proving the effectiveness of our first principles derivation. We see similar correlations at other angles as well.

Justifying the Ellipse: In Section 4.1 we introduced many potential methods for rotating a box label. In the ablation Table 2b we show that the Ellipse leads to the best performance across all rotations except $0 ^ { \circ }$ where it is within a small noise tolerance. It is also important to note that all methods we tried perform significantly better than the Largest Box - showing the importance of fixing this issue.

RU Loss Ablation: In Section 4.3 we introduce a hyperparameter $\delta$ in our final method. We ablate over that in Table 2 and demonstrate that $1 0 ^ { \circ }$ is optimal. We found this to be true on COCO, however, on simpler datasets we use larger values of $\delta$ for optimal performance.

# 5.3. Overall Performance

Our best performing method consists of using both Ellipse-based label rotation and RU Loss. In this section, we show it leads to much better performance across multiple datasets and approximates segmentation-based rotation augmentations on COCO.

# 5.3.1 Object Detection Datasets

In Table 3, we provide four datasets where our method of rotation augmentation improves performance while the previous one (Largest Box) hurts performance. We notice this

to be especially bad at higher APs, such as AP75. The gap is also larger in complex datasets such as transparent object bin picking where the largest box reduces performance by almost $50 \%$ and ours increases it by $4 . 5 \%$ .

# 5.3.2 Generalizing to new Rotation Angles

Our method significantly outperforms the original largest box method and also outperforms not using rotation for AP, AP50, and AP75 across all new angles from $[ 0 ^ { o } - 3 0 ^ { o } ]$ on COCO in Figure 1 and Figure 9. In the case of AP50, we show very similar improvements compared to using segmentation-based labels. This is a huge improvement since the largest box method hurts rotation performance.

# 6. Conclusion

The widespread Largest Box method (e.g. [43, 4, 45, 37, 28, 26]) is based on the folk wisdom of maximizing overlap. Instead, we show that by maximizing Expected IoU and accounting for label certainty in the loss, we can completely match the performance of perfect “segmentation-based” labels at AP50 while also achieving significant gains for AP and AP75. These results represent a step toward achieving rotation invariance for object detection models, while adding only a few lines of complexity to object detection codebases.

Acknowledgements: We thank Yuri Boykov, Tomas Gerlich, Abhijit Ghosh, Olga Veksler and Kartik Venkataraman for their helpful discussions and edits to improve the paper.

# References

[1] Mart´ın Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large-scale machine learning. In 12th $\{ U S E N I X \}$ symposium on operating systems design and implementation $\left( \left\{ O S D I \right\} { } I 6 \right)$ , pages 265–283, 2016.   
[2] Albumentations-Team. Why does the bounding box become so loose after rotation data aug? is there a better way? · issue 746 albumentations-team/albumentations. https://github.com/albumentationsteam/albumentations/issues/746.   
[3] Aleju. Make the bounding box more tighten for the object after image rotation · issue 90 · aleju/imgaug. https://github.com/aleju/ imgaug/issues/90.   
[4] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020.   
[5] Alexander Buslaev, Vladimir I Iglovikov, Eugene Khvedchenya, Alex Parinov, Mikhail Druzhinin, and Alexandr A Kalinin. Albumentations: fast and flexible image augmentations. Information, 11(2):125, 2020.   
[6] Angela Casado-Garc ´ ´ıa, Cesar Dom ´ ´ınguez, Manuel Garc´ıa-Dom´ınguez, Jonathan Heras, Adri ´ an In ´ es, ´ Eloy Mata, and Vico Pascual. Clodsa: a tool for augmentation in classification, localization, detection, semantic segmentation and instance segmentation tasks. BMC bioinformatics, 20(1):323, 2019.   
[7] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019.   
[8] Gong Cheng, Junwei Han, Peicheng Zhou, and Dong Xu. Learning rotation-invariant and fisher discriminative convolutional neural networks for object detection. IEEE Transactions on Image Processing, 28(1):265–278, 2018.   
[9] Gong Cheng, Peicheng Zhou, and Junwei Han. Learning rotation-invariant convolutional neural networks for object detection in vhr optical remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 54(12):7405–7415, 2016.   
[10] Gong Cheng, Peicheng Zhou, and Junwei Han. Rifdcnn: Rotation-invariant and fisher discriminative convolutional neural networks for object detection. In

Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2884–2893, 2016.   
[11] Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017.   
[12] Jian Ding, Nan Xue, Yang Long, Gui-Song Xia, and Qikai Lu. Learning roi transformer for oriented object detection in aerial images. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[13] Brad Dwyer. How to create a synthetic dataset for computer vision. https: //blog.roboflow.com/how-to-createa-synthetic-dataset-for-computervision/, Aug 2020.   
[14] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303–338, 2010.   
[15] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016. http:// www.deeplearningbook.org.   
[16] Jeremy Howard. Lesson 9: Deep learning part 2 2018 - multi-object detection - fast ai. https:// youtu.be/0frKXR-2PBY?t=554.   
[17] Alexander B Jung, K Wada, J Crall, S Tanaka, J Graving, S Yadav, J Banerjee, G Vecsei, A Kraft, J Borovec, et al. imgaug, 2018.   
[18] Agastya Kalra, Vage Taamazyan1, Supreeth Krishna Raol, Kartik Venkataraman, Ramesh Raskar, and Achuta Kadambi. Deep polarization cues for transparent object segmentation. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.   
[19] Ayoosh Kathuria. Data augmentation for bounding boxes: Rethinking image transforms for object detection. https://www.kdnuggets.com/ 2018/09/data-augmentation-boundingboxes-image-transforms.html.   
[20] Ayoosh Kathuria. Data augmentation for object detection: How to rotate bounding boxes. https://blog.paperspace.com/dataaugmentation-for-object-detectionrotation-and-shearing/, Sep 2018.   
[21] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object de-´ tection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.   
[22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and ´

C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.   
[23] Baozhen Liu, Hang Wu, Weihua Su, Wenchang Zhang, and Jinggong Sun. Rotation-invariant object detection using sector-ring hog and boosted random ferns. The Visual Computer, 34(5):707–719, 2018.   
[24] Kun Liu, Henrik Skibbe, Thorsten Schmidt, Thomas Blein, Klaus Palme, Thomas Brox, and Olaf Ronneberger. Rotation-invariant hog descriptors using fourier analysis in polar and spherical coordinates. International Journal of Computer Vision, 106(3):342– 364, 2014.   
[25] Lei Liu, Zongxu Pan, and Bin Lei. Learning a rotation invariant detector with rotatable bounding box. arXiv preprint arXiv:1711.09405, 2017.   
[26] Yang Liu, Lei Huang, Xianglong Liu, and Bo Lang. A novel rotation adaptive object detection method based on pair hough model. Neurocomputing, 194:246–259, 2016.   
[27] Rodrigo Lozuwa. lozuwa/impy. https:// github.com/lozuwa/impy, Mar 2019.   
[28] Daniel Mas Montserrat, Qian Lin, Jan Allebach, and Edward J Delp. Training object detection and recognition cnn models using data augmentation. Electronic Imaging, 2017(10):27–36, 2017.   
[29] Open-Mmlab. Why does the bounding box become so loose after rotation data aug? is there a better way · issue 4070 · open-mmlab/mmdetection. https://github.com/open-mmlab/ mmdetection/issues/4070.   
[30] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.   
[31] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. In Advances in neural information processing systems, pages 8026– 8037, 2019.   
[32] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems, pages 91–99, 2015.   
[33] Pranjal Saxena. Data augmentation for custom object detection: Yolo. https://medium.com/ predict/data-augmentationfor-custom-object-detection-15674966e0c8, Sep 2020.

[34] Uwe Schmidt and Stefan Roth. Learning rotationaware features: From invariant priors to equivariant descriptors. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 2050–2057. IEEE, 2012.   
[35] Dennis Soemers. How data augmentation like rotation affects the quality of detection? https: //ai.stackexchange.com/questions/ 9935/how-data-augmentation-likerotation-affects-the-quality-ofdetection, Mar 2018.   
[36] Jacob Solawetz. Why and how to implement random rotate data augmentation. https://blog.roboflow.com/why-andhow-to-implement-random-rotatedata-augmentation/, Aug 2020.   
[37] Mingxing Tan and Quoc V Le. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.   
[38] Luke Taylor and Geoff Nitschke. Improving deep learning with generic data augmentation. In 2018 IEEE Symposium Series on Computational Intelligence (SSCI), pages 1542–1547. IEEE, 2018.   
[39] Matlab Team. Augment bounding boxes for object detection. https://www.mathworks.com/ help/deeplearning/ug/bounding-boxaugmentation-using-computer-visiontoolbox.html.   
[40] Aleksei Tiulpin. Solt: Streaming over lightweight transformations, July 2019.   
[41] Guoli Wang, Bin Fan, and Chunhong Pan. Ordinal pyramid pooling for rotation invariant object recognition. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1349–1353. IEEE, 2015.   
[42] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2, 2019.   
[43] Yue Xi, Jiangbin Zheng, Xiuxiu Li, Xinying Xu, Jinchang Ren, and Gang Xie. Sr-pod: sample rotation based on principal-axis orientation distribution for data augmentation in deep object detection. Cognitive Systems Research, 52:144–154, 2018.   
[44] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE International Conference on Computer Vision, pages 6023–6032, 2019.   
[45] Barret Zoph, Ekin D Cubuk, Golnaz Ghiasi, Tsung-Yi Lin, Jonathon Shlens, and Quoc V Le. Learning data

augmentation strategies for object detection. arXiv preprint arXiv:1906.11172, 2019.