# Align and Prompt: Video-and-Language Pre-training with Entity Prompts

Dongxu Li1,2, Junnan $\mathrm { L i ^ { 1 } }$ , Hongdong $\mathrm { L i ^ { 2 } }$ , Juan Carlos Niebles1, Steven C.H. Hoi1 1Salesforce Research, 2The Australian National University

dongxuli1005@gmail.com, {junnan.li,jniebles,shoi}@salesforce.com, hongdong.li@anu.edu.au

# Abstract

Video-and-language pre-training has shown promising improvements on various downstream tasks. Most previous methods capture cross-modal interactions with a standard transformer-based multimodal encoder, not fully addressing the misalignment between unimodal video and text features. Besides, learning fine-grained visual-language alignment usually requires off-the-shelf object detectors to provide object information, which is bottlenecked by the detector’s limited vocabulary and expensive computation cost.

In this paper, we propose Align and Prompt: a new video-and-language pre-training framework (ALPRO), which operates on sparsely-sampled video frames and achieves more effective cross-modal alignment without explicit object detectors. First, we introduce a video-text contrastive (VTC) loss to align unimodal video-text features at the instance level, which eases the modeling of cross-modal interactions. Then, we propose a novel visually-grounded pre-training task, prompting entity modeling (PEM), which learns fine-grained alignment between visual region and text entity via an entity prompter module in a self-supervised way. Finally, we pretrain the video-and-language transformer models on large webly-source video-text pairs using the proposed VTC and PEM losses as well as two standard losses of masked language modeling (MLM) and video-text matching (VTM). The resulting pre-trained model achieves state-of-the-art performance on both text-video retrieval and videoQA, outperforming prior work by a substantial margin. Implementation and pre-trained models are available at https://github.com/salesforce/ALPRO.

# 1. Introduction

Video-and-language pre-training aims to jointly learn multimodal representations that transfer effectively to downstream tasks, such as text-video retrieval and videoQAvideo question answering. Compared with images, videos usually contain more redundancy in consecutive frames. This challenges models on both capacity and computation efficiency. Most prior approaches [30, 35, 37, 39, 48, 57] circumvent the expensive computation overhead by using offline-extracted video features. Since the video feature ex-

![](images/48b10b4a93ae463dcfba27b49d3aa013cbcf6bf2b9d5ebda8de3dd3a9cb8632f.jpg)  
Figure 1. Generating supervision for region-entity alignment. Above: previous methods (e.g. ActBERT [57]) rely on object detectors with expensive computation cost and limited object categories, leaving text data unexploited. Below: ALPRO generates soft entity labels with a prompter module, which computes similarities between video crops and textual entity prompts. ALPRO requires no detector while taking advantage of video-text alignment to generate entity labels with a large vocabulary, thus strengthening the cross-modal learning.

tractors are fixed without finetuning, these approaches are suboptimal when transferring to distinct target domains. In contrast, recent emerging approaches [3, 26] sample frames sparsely from videos, which enable end-to-end pre-training and finetuning of video backbones. In this work, we adopt the sparse video-text pre-training paradigm considering their effectiveness on downstream tasks.

Despite their promising performance, current video-text pre-training models have several limitations. (1) The interaction between video and text features is commonly mod-

eled trivially using either dot-product [3,37,39,52] or crossmodal transformer encoders [26, 30, 48, 57]. However, features from individual modalities typically reside in different embedding spaces. Such misalignment makes it less effective to directly model cross-modal interaction. (2) Many visually-grounded pre-training tasks [30, 48] do not explicitly model fine-grained regional visual information (e.g. objects), which proves important for downstream tasks emphasizing on visual reasoning (e.g. videoQA). Although there are attempts which employ object detectors [7, 57] to generate pseudo-labels as supervision, they suffer from imprecise detections and a restricted number of object categories. For example, detectors trained on MSCOCO [31] recognize less than a hundred different categories. (3) The previous sparse pre-training model [26] is trained with image-text pairs using an image encoder, which makes it less effective in modeling temporal information.

In this paper, we tackle these challenges with a new video-and-language pre-training framework: Align and Prompt (ALPRO). Architecture-wise, ALPRO first encodes frames and text independently using a transformer-based video encoder and a text encoder, and then employs a multimodal encoder to capture cross-modal interaction. AL-PRO learns both instance-level video-text alignment and fine-grained region-entity alignment. The instance-level alignment is learned by applying a video-text contrastive loss (VTC) on the unimodal features, which encourages paired video-text instances to have similar representations.

In order to better capture fine-grained visual information and strengthen region-entity alignment, ALPRO introduces a new visually-grounded pre-training task, called prompting entity modeling, where we ask the video-text model to predict entities appearing in randomly-selected video crops using jointly video and text inputs (see Figure 1). To address the unavailability of entity annotations, we design a standalone entity prompter module that generates reliable pseudo-labels. Specifically, the entity prompter consists of two unimodal encoders to extract video and text features, respectively. We first train the entity prompter using only VTC loss and freeze its parameters thereafter. Then during pre-training, we feed video crops and text prompts (e.g. “A video of {Entity}.”) to the prompter, where each Entity is from the frequent nouns appearing in the pretraining corpus. We then compute the normalized similarity between the entity prompts and the video crop as the pseudo-label to supervise the pre-training.

Our key contributions are: (1) We introduce ALPRO, a video-language pre-training method that is the first to learn effective cross-modal representations from sparse video frames and texts. (2) We introduce a video-text contrastive loss to better align instance-level unimodal representations, thus easing the modeling of cross-modal interaction. (3) We propose a novel visually-grounded pre-training task,

prompting entity modeling, that enables the model to capture fine-grained region-entity alignment. (4) We demonstrate the effectiveness of ALPRO on both video-text retrieval and videoQA. ALPRO significantly improves over previous state-of-the-art methods, for example, achieving $3 . 0 \%$ and $5 . 4 \%$ absolute lift in recall scores on the finetuning and zero-shot text-video retrieval task on MSRVTT.

# 2. Related Work

Dense versus Sparse Video Representation. Consecutive frames in videos usually contain visually similar information. Such redundancy opens up a research question on how to learn effective video-and-language representations without excessive computation overhead. Most prior methods on text-video retrieval [14, 32, 43, 53, 55] and videoQA [12, 15, 25, 27] employ pre-trained visual backbones and extract video features for each frame densely yet offline. However, since visual backbones are usually pretrained on image [23] and/or video datasets [21] without access to text, these features are less effective for video-andlanguage tasks. Besides, video feature extractors in these approaches are not finetuned on target task data, preventing features to easily adapt to different domains. In contrast, recent methods ClipBERT [26] and FiT [3] demonstrate more effective results by end-to-end finetuning the visual backbone with only a few sparsely sampled frames. However, ClipBERT is pre-trained with image-text data thus is less effective in aggregating information across frames, while FiT is a retrieval-specific architecture that does not naturally generalize to videoQA task. In this regard, our AL-PRO is the first sparse pre-training architecture that tackles both tasks while in the meantime, demonstrating the benefit of pre-training on video-text pairs.

Video-and-language Pre-training. Apart from the canonical pre-training tasks, such as masked language modeling (MLM) [10, 26, 30, 35, 48, 57] and video-text matching (VTM) [30, 35], several methods [35, 37, 52] apply contrastive learning on offline extracted visual features. Without adapting the visual backbone, their ability to align cross-modal features remain limited. ALPRO jointly learns the unimodal and multimodal encoders, thus mitigating the disconnection in-between. In order to design effective visually-grounded pre-training tasks, VideoBERT [48] predicts centroids of vector quantizations of video features. Such unsupervised quantization is noisy per se while neglecting textual cues, which curtails its capabilities in learning cross-modal interactions. ActBERT [57] uses detectors to acquire object information. In addition to their computational inefficiency, detectors pre-trained on images usually have limited categories and compromised detection results on videos. In contrast, our proposed prompting entity modeling task is detector-free. By exploiting the instancelevel video-text alignment, we can generate reliable entity

![](images/e7e22f60d41400058278e09b866a574df14b40506590e7a11c1b3407661285e3.jpg)  
Figure 2. ALPRO pre-training framework. Left: the video-language pre-training model contains a space-time video encoder, a text encoder, and a multi-modal encoder, all of which are transformer-based. Besides two canonical objectives masked language modeling (MLM) and video-text matching (VTM), we introduce video-text contrastive loss (VTC) to learn instance-level video-text alignment, and prompting entity modeling (PEM) to learn fine-grained region-entity alignment. Right: the prompter which generates soft entity labels as supervision for PEM. The prompter consists of frozen unimodal encoders that are trained with VTC. During pre-training, it produces similarity scores between a randomly-selected video crop and a set of text prompts instantiated with entity names.

pseudo-labels with a large vocabulary, leading to more efficient and effective learning of region-entity alignment.

Zero-shot Visual Recognition with Prompts. There have been longstanding efforts to exploit text descriptions for learning visual recognition models. These include early efforts [4, 13, 24] that use text to learn attributes of images; [40,41,47] that map images to the pretrained text embedding space, and visual n-gram [28] that predicts text ngrams given image inputs. More recently, CLIP [44] instantiates prompt templates with label text of visual categories. It then predicts the category by computing the similarity between each image-prompt pairs. This inspires us to the design of entity prompter. Since the entity prompter is trained with the entire video-text corpus, during pre-training, it can provide additional entity information unavailable in the text description for each individual video-text pair, thus leading to better entity-informed video representations.

# 3. Video-Language Pre-training with ALPRO

In this section, we first introduce important constituent modules of ALPRO in Section 3.1. Then we present the pre-training objectives in Section 3.2, with a particular focus on the proposed video-text contrastive (VTC) loss and the prompting entity modeling (PEM) pre-training task. We introduce pre-training datasets in Section 3.3. Lastly, we describe important implementation details in Section 3.4.

# 3.1. ALPRO Architecture

Figure 2 gives an overview of ALPRO ’s architecture. In particular, ALPRO consists of two main modules, a videolanguage pre-training model and a prompter. The prompter

serves to generate soft entity labels to supervise the pretraining of the video-language model. Both modules contain their own video encoder and text encoder to extract features for video and text inputs, respectively. The pretraining model has an additional multimodal encoder to further capture the interaction between the two modalities. Details for each component are as follows.

Visual Encoder. We use a 12-layer TimeSformer224 [5] to extract video features, with 224 the height and width of input frames. For $N _ { v }$ frames sparsely sampled from each input video, TimeSformer first partitions each frame into $K$ non-overlapping patches, which are flattened and fed to a linear projection layer to produce a sequence of patch tokens. Learnable positional embeddings are also added to the patch tokens. Then the TimeSformer applies self-attention along the temporal and spatial dimensions separately in order, leading to per-frame features $\tilde { \pmb { v } } \in \mathbb { R } ^ { N _ { v } \times K \times d }$ , with $d$ the feature dimension. A temporal fusion layer (i.e. meanpooling) is applied to $\tilde { v }$ along the temporal dimension to aggregate per-frame features into video features. As the output of visual encoder, we obtain a sequence of visual embeddings: $\{ \pmb { v } _ { \mathrm { c l s } } , \pmb { v } _ { 1 } , . . . , \pmb { v } _ { K } \}$ , with $\pmb { v } _ { i } \in \mathbb { R } ^ { d }$ and $v _ { \mathrm { c l s } }$ the embedding of the video [CLS] token.

Text Encoder. We use a 6-layer transformer [49] model to represent text tokens. Given an input text description of $N _ { t }$ tokens, the text encoder outputs an embedding sequence $\{ t _ { \mathrm { c l s } } , t _ { 1 } , . . . , t _ { N _ { t } } \}$ , with $t _ { i } \in \mathbb { R } ^ { d }$ and $t _ { \mathrm { c l s } }$ the embedding of the text [CLS] token. Similar to video encoder, we also add positional embeddings to the text tokens.

Multimodal Encoder. We employ a 6-layer transformer to model the interaction between video and text features

from the two unimodal encoders. Since positional embeddings are already injected in each unimodal encoder, we directly concatenate video and text features to feed the multimodal transformer. The outputs are multimodal embeddings $\{ e _ { \mathrm { c l s } } , e _ { 1 } , . . . , e _ { N _ { v } + N _ { t } } \}$ , with $e _ { i } \in \mathbb { R } ^ { d }$ . For notational convenience, we drop the multimodal embedding for the video [CLS] token as it is not used in pre-training losses.

# 3.2. Pre-training for ALPRO

We pre-train ALPRO with four objectives, including two canonical ones, i.e. masked language modeling (MLM) and video-text matching (VTM) as in [26,30,48,57]. In this section, we focus on presenting the new techniques in ALPRO, i.e. the video-text contrastive (VTC) loss and the prompting entity modeling (PEM) loss, while only briefly outlining MLM and VTM in Section 3.2.3, referring interested readers to [10, 26, 48] for details.

The motivation of both VTC and PEM is to strengthen cross-modal alignment between video and text. While VTC emphasizes on capturing instance-level alignment for video-text pairs, PEM encourages the model to align local video regions with textual entities. In the following, we introduce these two pre-training objectives in order.

# 3.2.1 Contrastive Video-Text Alignment

Existing sparse video-language pre-training models use either dot-product [3, 37, 39, 52] or rely entirely on a transformer encoder [26, 30, 48, 57] to model cross-modal interactions. However, since video and text features reside in different embedding spaces, such methods lead to less satisfactory alignment. To this end, we present a video-text contrastive (VTC) loss to align features from the unimodal encoders before sending them into the multimodal encoder. Specifically, given the embeddings of video and text [CLS] tokens, we optimize a similarity function between video $V$ and text $T$ :

$$
s (V, T) = g _ {v} \left(\boldsymbol {v} _ {\mathrm {c l s}}\right) \cdot g _ {t} \left(\boldsymbol {t} _ {\mathrm {c l s}}\right), \tag {1}
$$

such that paired video and text descriptions have higher similarity scores, where $g _ { v } ( \cdot )$ and $g _ { t } ( \cdot )$ are linear projections that transform the [CLS] embeddings to a common normalized low-dimensional (e.g. 256-d) space.

Following [18, 44], the contrastive loss considers matched pairs as positive and all others pairs that can be formed in a batch as negatives. For each input video-text pair $\langle V _ { i } , T _ { i } \rangle$ , the video-text contrastive loss consists of two symmetric terms, one for video-to-text classification:

$$
\mathcal {L} _ {\mathrm {v 2 t}} = - \log \frac {\exp \left(s \left(V _ {i} , T _ {i}\right) / \tau\right)}{\sum_ {j = 1} ^ {B} \exp \left(s \left(V _ {i} , T _ {j}\right) / \tau\right)} \tag {2}
$$

and the other for text-to-video classification:

$$
\mathcal {L} _ {\mathrm {t 2 v}} = - \log \frac {\exp (s \left(T _ {i} , V _ {i}\right) / \tau)}{\sum_ {j = 1} ^ {B} \exp (s \left(T _ {i} , V _ {j}\right) / \tau)}, \tag {3}
$$

where $\tau$ is a learnable temperature parameter, and $B$ is the batch size. The video-text contrastive loss is then defined as $\begin{array} { r } { \mathcal { L } _ { \mathrm { v t c } } = \frac { 1 } { 2 } ( \mathcal { L } _ { \mathrm { v 2 t } } + \mathcal { L } _ { \mathrm { t 2 v } } ) } \end{array}$ .

# 3.2.2 Prompting Entity Modeling

While masked language modeling has demonstrated its effectiveness on learning token-level text representations [10, 33], it remains a challenge to design its visually-grounded counterpart. As a result, the limited capabilities in visual reasoning adversely impact previous work on downstream tasks, especially those requiring region-level visual information such as objects. This is in particular an issue for existing video-language pre-training models [30, 37, 39, 39, 48], which usually retain only coarse-grained spatial information after pooling thus losing fine-grained visual cues. One exception is ActBERT [57] that attempts to use offthe-shelf object detectors to obtain regional features. Apart from its inefficiency, detectors trained with images tend to produce compromised detection results on video inputs. In addition, detectors are usually trained with restricted object categories (e.g. less than a hundred [31]), given its prohibitive expense to scale up the laborious annotations.

We introduce prompting entity modeling (PEM), a new visually-grounded pre-training task that improves the models’ capabilities in capturing local regional information and strengthening cross-modal alignment between video regions and textual entities. Specifically, PEM requires a prompter module that generates soft pseudo-labels identifying entities that appear in random video crops. The pretraining model is then asked to predict the entity categories in the video crop, given the pseudo-label as the target.

The prompter serves to produce pseudo-labels of entity categories given a video crop, without dense annotations other than webly-sourced video-text pairs with possibly noisy alignment. To this end, we are inspired by CLIP [44] that learns image-text alignment from noisy pairs. Specifically, we first pre-train the prompter, which consists of two unimodal encoders, on video-text pairs with the VTC loss as in Section 3.2.1, and freeze its parameters thereafter.

The prompter maintains a predetermined list of $M$ text prompts. Each text prompt is an instantiation of a template, e.g. “A video of {ENTITY}.”, where ENTITY is a frequent noun in the pre-training corpus, such as dog, grass, sky, etc. After the prompter is pre-trained, it computes the [CLS] embedding for each text prompt as $\{ t _ { \mathrm { c l s } } ^ { 1 } , t _ { \mathrm { c l s } } ^ { 2 } , . . . , t _ { \mathrm { c l s } } ^ { M } \}$ .

To generate entity labels, given one video input, we first obtain a random video crop $\hat { V }$ (e.g. the same spatial region across sampled frames) and its [CLS] embedding $\hat { v } _ { \mathrm { c l s } }$ from the prompter’s video encoder. The prompter then computes an entity pseudo-label $\pmb q _ { \hat { V } } \in \mathbb { R } ^ { M }$ for the video crop as the softmax-normalized similarity between $\hat { v } _ { \mathrm { c l s } }$ and all the

prompt embeddings $\{ \pmb { t } _ { \mathrm { c l s } } ^ { m } \} _ { m = 1 } ^ { M }$ :

$$
q _ {\hat {V}, m} = \frac {\exp (s (\hat {V} , T _ {m}) / \tau)}{\sum_ {m = 1} ^ {M} \exp (s (\hat {V} , T _ {m}) / \tau)} \tag {4}
$$

During pre-training of the video-language model, we apply mean pooling on the embeddings from the multimodal encoder that correspond to the spatial location of the video crop $\hat { V }$ , denoted as $\boldsymbol { e } _ { \hat { V } } \in \mathbb { R } ^ { d }$ . We use a classifier (e.g. MLP) to compute the softmax-normalized entity prediction $\pmb { p } _ { \hat { V } }$ . The prompting entity modeling loss is then defined as the cross-entropy between $\pmb { p } _ { \hat { V } }$ and $\pmb { q } _ { \hat { V } }$ :

$$
\mathcal {L} _ {\mathrm {p e m}} = - \sum_ {m = 1} ^ {M} q _ {\hat {V}, m} \cdot \log p _ {\hat {V}, m} \tag {5}
$$

Prompting entity modeling features a diverge range of entities while requiring no extra human annotations, which yields an efficient and scalable solution to generate visuallygrounded regional supervisions for cross-modal learning.

# 3.2.3 Overall Pre-training Objectives

We also employ the widely-adopted masked language modeling (MLM) loss ${ \mathcal { L } } _ { \mathrm { m l m } }$ and video-text matching ${ \mathcal { L } } _ { \mathrm { v t m } }$ considering their effectiveness. The MLM objective utilizes both video and the contextual text to predict the masked text tokens. We randomly mask input tokens with a probability of $1 5 \%$ and replace them with a special token [MASK]. Video-text matching is a binary classification task which predicts whether a video and a text description are matched with each other. We use the multimodal [CLS] token $e _ { \mathrm { c l s } }$ as the joint representation of the video-text pair, and trains the model with a cross entropy loss. Negative samples are generated from non-parallel video-text pairs from the batch. Following [29], we employ contrastive hard negative mining to find more informative in-batch negatives for VTM. The overall pre-training objective of ALPRO is:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm {v t c}} + \mathcal {L} _ {\mathrm {p e m}} + \mathcal {L} _ {\mathrm {m l m}} + \mathcal {L} _ {\mathrm {v t m}} \tag {6}
$$

# 3.3. Pre-training Datasets

We pre-train our model with the webly-sourced dataset WebVid-2M [3], which contains 2.5M video-text pairs. In addition, as suggested by ClipBERT [26] and FiT [3], pretraining with image-pairs can improve spatial representations of videos, we therefore include CC-3M [46] into our pre-training corpus. During pre-training, we duplicate images from CC-3M to make static videos. This in total amounts to 5.5M video-text pairs, which is an order of magnitude less than the commonly-adopted HowTo100M [30, 37, 57] and of a comparable size to those used in [3, 26].

# 3.4. Implementation Details

We implement ALPRO in PyTorch [42]. In detail, we initialize both the spatial and temporal attention blocks of TimeSformer by reusing ViT-B/16 weights pre-trained on ImageNet-21k [11]. Text encoders are initialized using the first 6-layer of the $\mathbf { B E R T _ { b a s e } }$ model [10], and the multimodal encoder is initialized using the last 6-layers weights of $\mathbf { B E R T _ { b a s e } }$ . We pre-train ALPRO for 100k iterations, roughly equivalent to 10 epochs, using a batch size of 256 on 16 NVIDIA A100 GPUs. We use AdamW [34] optimizer with a weight decay of 0.001. The learning rate is first warmed-up to $1 e ^ { - 4 }$ , then it follows a linear decay schedule. Since videos are usually of different aspect ratios, we first rescale them to $2 2 4 \times 2 2 4$ . For each video, we sample 4 frames randomly as inputs to the visual encoder while preserving their orderings in-between. For PEM, we use POS tagger1 and retain the top 1k most frequent nouns as the entity names. We obtain random video crops occupying $3 0 \% - 5 0 \%$ of the original spatial area as inputs to the prompter. We discard a pseudo-label if the most likely entity has a normalized similarity score smaller than 0.2.

# 4. Experiments

We evaluate the performance of ALPRO on text-video retrieval and video question answering tasks across four commonly-used datasets, introduced in Section 4.1. The purpose of the evaluation is three-fold. First, we demonstrate the effectiveness of major technical contributions (i.e. video-text contrastive loss and prompting entity modeling) in Section 4.2. We then compare the performance of ALPRO with previous methods, including task-specific and pre-training architectures, in Section 4.3 and Section 4.4, on retrieval and question answering tasks, respectively. Finally, we show ablation results on design choices and analyze model behaviors in Section 4.5.

# 4.1. Downstream Tasks and Datasets

Text-Video Retrieval. (i) MSRVTT [53] contains 10K videos with 200K text captions. We follow the common protocol [26, 30, 39, 55, 57] and use 7k videos for training and report results on the 1k test split [55]. (ii) DiDeMo [2] contains 10k videos from Flickr with 40k text descriptions. We follow [26, 32, 35] and evaluate paragraph-to-video retrieval, where sentence descriptions for each video are concatenated together as a single text query. We do not use the ground-truth proposals for temporal localization to ensure fair comparisons with previous work.

Video Question Answering. We focus on the task of openended video question answering. (i) MSVD-QA [51] is built upon videos and text descriptions from MSVD [6].

Table 1. Evaluations of the proposed pre-training objectives on four downstream datasets. MLM: masked language modeling loss. VTM: video-text matching loss. PEM: prompting entity modeling loss. VTC: video-text contrastive loss. $\mathbf { R } \ @ \mathbf { k }$ denotes recall $( \% )$ with k retrieval efforts; MdR denotes median ranking for retrieved videos. We use acc. to denote accuracy.   

<table><tr><td rowspan="2">Pre-training tasks</td><td colspan="4">MSRVTT Retrieval</td><td colspan="4">DiDeMo Retrieval</td><td rowspan="2">MSVD-QA Acc.↑</td><td rowspan="2">MSRVTT-QA Acc.↑</td></tr><tr><td>R1↑</td><td>R5↑</td><td>R10↑</td><td>MdR↓</td><td>R1↑</td><td>R5↑</td><td>R10↑</td><td>MdR↓</td></tr><tr><td>w/o pre-training</td><td>16.5</td><td>42.8</td><td>57.9</td><td>7</td><td>9.5</td><td>29.1</td><td>42.5</td><td>14</td><td>41.5</td><td>39.6</td></tr><tr><td>MLM + VTM</td><td>28.5</td><td>53.0</td><td>66.8</td><td>5</td><td>29.8</td><td>57.7</td><td>69.7</td><td>4</td><td>43.3</td><td>40.9</td></tr><tr><td>MLM + VTM + PEM</td><td>30.3</td><td>56.7</td><td>67.8</td><td>4</td><td>31.0</td><td>61.8</td><td>73.5</td><td>3</td><td>46.3</td><td>41.8</td></tr><tr><td>MLM + VTM + VTC</td><td>32.8</td><td>59.2</td><td>70.3</td><td>3</td><td>36.8</td><td>64.7</td><td>77.4</td><td>2</td><td>45.5</td><td>41.9</td></tr><tr><td>MLM + VTM + PEM + VTC</td><td>33.9</td><td>60.7</td><td>73.2</td><td>3</td><td>35.9</td><td>67.5</td><td>78.8</td><td>3</td><td>45.9</td><td>42.1</td></tr></table>

![](images/5f20e89ecde538b553a9a6ae80e72283ed985295f19c715c675113a08680626b.jpg)  
Young fit female in sportswear doing stretching lunge exercise indoor.

# Top 5 pseudo-labels

1. Yoga (0.69)   
2. Poses (0.11)   
3. Workout (0.08)   
4. Exercises (0.04)   
5. Fitness (0.04)

![](images/b3bb695b42cddd5c179aa82e694e1bee258f4015d117bb4ea28c065532d9027e.jpg)  
A pigeon takes off from the roof of the garage.

# To discard

1. Hat (0.07)   
2. Gun (0.07)   
3. Shoes (0.05)   
4. Poster (0.05)   
5. Cartoon (0.04)

![](images/34935975177c6ac4c62933460ff17f40dfe30a3dbed7bb016f00ad756a0ca50b.jpg)  
Mt. Fuji with chureito pagoda in spring, fujiyoshida, Japan.

# Top 5 pseudo-labels

1. Monastery (0.25)   
2. Architecture (0.18)   
3. Towers (0.16)   
4. Buildings (0.06)   
5. Heritage (0.04)

![](images/7744a0e66d08cb12ea1d16882d6c18e28731510b72a269c80d70808824142a2c.jpg)  
Figure 3. Examples of the pseudo-labels generated by the prompter (scores in bracket). The highlighted areas are fed to the prompter. Our method generates a diverse range of common entity categories that are not usually covered by object detectors, e.g. towers, summit, yoga. Besides, entity labels do not always appear in the text description, serving as a source of corpus-level supervision. Bottom left: a random crop that does not contain entities. The prompter thereby produces pseudo-labels with low similarities. During pre-training, we discard a pseudo-label if its most likely entity has a score less than 0.2. Right: labels generated for different crops from the same video.

Mt. Fuji with chureito pagoda in spring, fujiyoshida, Japan.

# Top 5 pseudo-labels

1. Summit (0.84)   
2. Mountain (0.09)   
3. Peaks (0.02)   
4. Town (0.01)   
5. Pollution (0.01)

The MSVD-QA dataset has in total 1,970 videos and 50k question answer pairs, with 2,423 answer candidates. (ii) MSRVTT-QA [51] is built upon videos and captions from MSRVTT, which contains 10k videos with 243k openended questions and 1.5k answer candidates.

Finetuning Setups. On downstream tasks, ALPRO allows end-to-end finetuning of the video backbone with raw video frames as input. During finetuning, we randomly sample $N$ frames per video, where $N = 8$ for retrieval and $N = 1 6$ for QA, with more ablations present in Section 4.4. Temporal position embeddings in TimeSformer are interpolated to accommodate different number of input frames. During inference, we sample frames uniformly to ensure reproducibility. To keep pre-training and finetuning setups consistent, we resize all the videos to $2 2 4 \times 2 2 4$ before feeding them into the model. Although this does not maintain the original aspect ratios, we observe no significant performance drop as our pre-training dataset contains videos of various aspect ratios. For finetuning on retrieval, we reuse the video-text matching head during pre-training and optimize the sum of both VTC and VTM losses. We obtain similarity scores from the output of VTM head during inference. For QA task, we add a simple MLP on the multimodal [CLS] token for classification and optmize the conventional cross-entropy loss between predictions and ground-truth answer labels. Dur-

ing inference, predictions are obtained as the answer with the highest probability. All the finetuning experiments are performed on 8 NVIDIA A100 GPUs, taking one to five hours to complete depending on the datasets. More training details can be found in the appendix.

# 4.2. Evaluation on the Proposed Methods

We first evaluate the impact of our main technical contributions (i.e. video-text contrastive loss and prompting entity modeling) in Table 1. Compared with pre-training using only MLM and VTM, both PEM and VTC substantially improve the performance across all the datasets. VTC is in particular useful for the retrieval task. The reason is that the VTC loss explicitly maximizes the instance-level similarity between positive video-text pairs, which is wellaligned with the goal of retrieval. We notice that PEM significantly improves the performance of videoQA, especially on MSVD-QA, due to its ability to learn finer-grained regional features. While enabling both PEM and VTC losses has complementary effects for most datasets, we also observe it leads to slightly worse accuracy on MSVD-QA. Our observation is that MSVD-QA contains more questions requiring region-level knowledge, including object categories (e.g. dough, swords), animal species (e.g. hare, eagle) and scenes (e.g. river, cliff), which can be well modeled using

<table><tr><td>Method</td><td>PT datasets</td><td>R1↑</td><td>R5↑</td><td>R10↑</td><td>MdR↓</td></tr><tr><td colspan="6">Finetuning</td></tr><tr><td>JSFusion [55]</td><td>-</td><td>10.2</td><td>31.2</td><td>43.2</td><td>13</td></tr><tr><td>HT100M [39]</td><td>HT (100M)</td><td>14.9</td><td>40.2</td><td>52.8</td><td>9</td></tr><tr><td>ActBERT [57]</td><td>HT (100M)</td><td>16.3</td><td>42.8</td><td>56.9</td><td>10</td></tr><tr><td>NoiseEst. [1]</td><td>HT (100M)</td><td>17.4</td><td>41.6</td><td>53.6</td><td>8</td></tr><tr><td>HERO [12]</td><td>HT (100M)</td><td>16.8</td><td>43.4</td><td>57.7</td><td>-</td></tr><tr><td>ClipBERT [26]</td><td>COCO + VG (5.6M)</td><td>22.0</td><td>46.8</td><td>59.9</td><td>6</td></tr><tr><td>AVLNet [25]</td><td>HT (100M)</td><td>27.1</td><td>55.6</td><td>66.6</td><td>4</td></tr><tr><td>VideoClip [52]</td><td>HT (100M)</td><td>30.9</td><td>55.4</td><td>66.8</td><td>-</td></tr><tr><td>SupportSet [43]</td><td>HT (100M)</td><td>30.1</td><td>58.5</td><td>69.3</td><td>3</td></tr><tr><td>FiT [3]</td><td>Web2M + CC3M (5.5M)</td><td>31.0</td><td>59.5</td><td>70.5</td><td>3</td></tr><tr><td>ALPRO</td><td>Web2M + CC3M (5.5M)</td><td>33.9</td><td>60.7</td><td>73.2</td><td>3</td></tr><tr><td colspan="6">Zero-shot</td></tr><tr><td>HT100M [39]</td><td>HT (100M)</td><td>7.5</td><td>21.2</td><td>29.6</td><td>38</td></tr><tr><td>ActBERT [57]</td><td>HT (100M)</td><td>8.6</td><td>23.4</td><td>33.1</td><td>36</td></tr><tr><td>SupportSet [43]</td><td>HT (100M)</td><td>8.7</td><td>23.0</td><td>31.1</td><td>31</td></tr><tr><td>MIL-NCE [37]</td><td>HT (100M)</td><td>9.9</td><td>24.0</td><td>32.4</td><td>29.5</td></tr><tr><td>VideoCLIP [52]</td><td>HT (100M)</td><td>10.4</td><td>22.2</td><td>30.0</td><td>-</td></tr><tr><td>FiT [3]</td><td>Web2M + CC3M (5.5M)</td><td>18.7</td><td>39.5</td><td>51.6</td><td>10</td></tr><tr><td>ALPRO</td><td>Web2M + CC3M (5.5M)</td><td>24.1</td><td>44.7</td><td>55.4</td><td>8</td></tr></table>

Table 2. Comparisons with existing text-to-video retrieval methods with finetuning and zero-shot setups on MSRVTT. We follow the common partition with 7k training videos. Methods using 9k training videos are greyed out. Both partition protocols share the same 1k testing videos. ${ \bf R } @ { \bf k }$ denotes recall $( \% )$ with k retrieval efforts; MdR denotes median ranking for retrieved videos. The pre-training datasets are HowTo100M (HT) [39], MS-COCO (COCO) [31], Visual Genome (VG) [22], WebVid2M (Web2M) [3] and Conceptual Captions (CC3M) [46].

PEM, rendering the impact of VTC negligible. In contrast, MSRVTT-QA involves more coarse-grained visual information such as activities. As a result, using both PEM and VTC complements with each other on MSRVTT-QA.

Example Pseudo-labels. In Figure 3, we show examples of pseudo-labels generated by the prompter module. Our approach generates a more diverse range of entity categories beyond typical object classes from detection annotations. This is in particular beneficial when downstream tasks require a large vocabulary, such as open-ended videoQA.

# 4.3. Evaluation on Video-Text Retrieval

In Table 2 and Table 3, we compare ALPRO with existing methods using finetuning and zero-shot text-to-video retrieval on MSRVTT and DiDeMo datasets, respectively. ALPRO surpasses previous methods by a significant margin while exploiting orders of magnitude less video-text pairs with no human-written texts required. On both datasets, ALPRO obtains more than $6 \%$ lift in terms of R10 scores. Note that we do not include comparison with the re-

<table><tr><td>Method</td><td>PT datasets</td><td>R1↑</td><td>R5↑</td><td>R10↑</td><td>MdR↓</td></tr><tr><td colspan="6">Finetuning</td></tr><tr><td>S2VT [50]</td><td>-</td><td>11.9</td><td>33.6</td><td>-</td><td>13</td></tr><tr><td>FSE [56]</td><td>-</td><td>13.9</td><td>36.0</td><td>-</td><td>11</td></tr><tr><td>CE [32]</td><td>-</td><td>16.1</td><td>41.1</td><td>-</td><td>8</td></tr><tr><td>MoEE [38]</td><td>-</td><td>16.1</td><td>41.2</td><td>55.2</td><td>8</td></tr><tr><td>ClipBERT [26]</td><td>COCO + VG (5.6M)</td><td>20.4</td><td>48.0</td><td>60.8</td><td>6</td></tr><tr><td>TT-CE [8]</td><td>-</td><td>21.6</td><td>48.6</td><td>62.9</td><td>6</td></tr><tr><td>FiT [3]</td><td>Web2M + CC3M (5.5M)</td><td>31.0</td><td>59.8</td><td>72.4</td><td>3</td></tr><tr><td>ALPRO</td><td>Web2M + CC3M (5.5M)</td><td>35.9</td><td>67.5</td><td>78.8</td><td>3</td></tr><tr><td colspan="6">Zero-shot</td></tr><tr><td>VideoCLIP [52]</td><td>HT (100M)</td><td>16.6</td><td>46.9</td><td>-</td><td>-</td></tr><tr><td>FiT [3]</td><td>Web2M + CC3M (5.5M)</td><td>21.1</td><td>46.0</td><td>56.2</td><td>7</td></tr><tr><td>ALPRO</td><td>Web2M + CC3M (5.5M)</td><td>23.8</td><td>47.3</td><td>57.9</td><td>6</td></tr></table>

Table 3. Comparisons with existing text-to-video retrieval methods with finetuning and zero-shot setups on DiDeMo. ${ \bf R } @ { \bf k }$ denotes recall $( \% )$ with k retrieval efforts; MdR denotes median ranking for retrieved videos.

<table><tr><td>Method</td><td>PT datasets</td><td>MSRVTT</td><td>MSVD</td></tr><tr><td>E-SA [51]</td><td>-</td><td>29.3</td><td>27.6</td></tr><tr><td>ST-TP [17]</td><td>-</td><td>30.9</td><td>31.3</td></tr><tr><td>AMU [51]</td><td>-</td><td>32.5</td><td>32.0</td></tr><tr><td>Co-mem [15]</td><td>-</td><td>32.0</td><td>31.7</td></tr><tr><td>HME [12]</td><td>-</td><td>33.0</td><td>33.7</td></tr><tr><td>LAGCN [16]</td><td>-</td><td>-</td><td>34.3</td></tr><tr><td>HGA [20]</td><td>-</td><td>35.5</td><td>34.7</td></tr><tr><td>QUEST [19]</td><td>-</td><td>34.6</td><td>36.1</td></tr><tr><td>HCRN [25]</td><td>-</td><td>35.6</td><td>36.1</td></tr><tr><td>ClipBERT [26]</td><td>COCO + VG (5.6M)</td><td>37.4</td><td>-</td></tr><tr><td>SSML [1]</td><td>HT (100M)</td><td>35.1</td><td>35.1</td></tr><tr><td>CoMVT [45]</td><td>HT (100M)</td><td>39.5</td><td>42.6</td></tr><tr><td>VQA-T [54]</td><td>HTVQA (69M)</td><td>41.5</td><td>46.3</td></tr><tr><td>ALPRO</td><td>Web2M + CC3M (5.5M)</td><td>42.1</td><td>45.9</td></tr></table>

Table 4. Comparisons with existing methods on MSRVTT-QA and MSVD-QA in top-1 accuracy $( \% )$ . VQA-T [54] uses 69M QA domain-specific data to pre-train their model while ALPRO uses an order of magnitude less video-text pairs from the web.

cent paper [36] which utilizes the powerful encoders from CLIP [44] (pretrained on 400M image-text pairs) and further post-trains them on HowTo100M [39]. Since [44] uses the same objective as VideoClip [52], the improvement over VideoClip validates the advantage of ALPRO pre-training.

Table 5. Effect of ALPRO pre-training with and without prompt ensembling (ens.) on MSVD-QA and MSRVTT text-video retrieval with finetuning (FT) and zero-shot (ZS) setups.   

<table><tr><td rowspan="2"></td><td colspan="3">MSRVTT-FT</td><td colspan="3">MSRVTT-ZS</td><td rowspan="2">MSVD-QA Acc.↑</td></tr><tr><td>R1↑</td><td>R10↑</td><td>MdR↓</td><td>R1↑</td><td>R10↑</td><td>MdR↓</td></tr><tr><td>w/o ens.</td><td>32.7</td><td>73.1</td><td>3</td><td>22.6</td><td>52.3</td><td>9</td><td>45.0</td></tr><tr><td>with ens.</td><td>33.9</td><td>73.2</td><td>3</td><td>24.1</td><td>55.4</td><td>8</td><td>45.9</td></tr></table>

Table 6. Effect of the number of entities for PEM. We report results on MSVD-QA and MSRVTT text-video retrieval with finetuning (FT) and zero-shot (ZS) setups. The first row corresponds to the model trained with MLM+VTM+VTC (i.e w/o PEM).   

<table><tr><td rowspan="2">#ent.</td><td colspan="3">MSRVTT-FT</td><td colspan="3">MSRVTT-ZS</td><td>MSVD-QA</td></tr><tr><td>R1↑</td><td>R10↑</td><td>MdR↓</td><td>R1↑</td><td>R10↑</td><td>MdR↓</td><td>Acc.↑</td></tr><tr><td>∅</td><td>32.8</td><td>70.3</td><td>3</td><td>22.6</td><td>53.0</td><td>9</td><td>45.5</td></tr><tr><td>500</td><td>33.0</td><td>71.9</td><td>3</td><td>22.7</td><td>54.1</td><td>8</td><td>45.6</td></tr><tr><td>1000</td><td>33.9</td><td>73.2</td><td>3</td><td>24.1</td><td>55.4</td><td>8</td><td>45.9</td></tr><tr><td>2000</td><td>34.7</td><td>72.4</td><td>3</td><td>22.4</td><td>52.8</td><td>9</td><td>45.3</td></tr></table>

# 4.4. Evaluation on Video Question Answering

Table 4 compares ALPRO with existing methods on open-ended video question answering datasets MSRVTT-QA and MSVD-QA. Most competitors have QA-specific architectures while that of ALPRO is generic for other videolanguage tasks, such as retrieval. We obtain on-par results with VQA-T [54], which exploits 69M QA-specific domain data for pre-training. In contrast, ALPRO uses only 5.5M video-text pairs from the web without domain knowledge. ALPRO surpasses other methods by a substantial margin, with $2 . 6 \%$ and $3 . 3 \%$ lift in accuracy. This demonstrates the competitive visual reasoning ability of ALPRO.

# 4.5. Ablations and Analysis

Prompt design and ensembling. Similar to [44], we observe that it is important to design and ensemble prompts with multiple templates. Without much engineering effort, we employ a preliminary set of prompt templates, such as “A video of a {ENTITY}”, “A footage of one {ENTITY}” for video inputs; “A photo of a {ENTITY}” and “A picture of the $\{ \mathrm { E N T I T Y } \} ^ { \mathrm { , } }$ for image inputs. In total, we design 12 templates for video and image inputs each. We build the ensemble by averaging over the $t _ { \mathrm { c l s } }$ embeddings of prompts instantiated with the same entity. The effect of prompt ensembling is shown in Table 5. Despite our minimal engineering efforts (we only experimented with a single set of templates), prompt ensembling demonstrates its importance in generating high-quality pseudo-labels. It is our future work to explore more prompt engineering strategies.

Effect of number of entities. We investigate the effect of the number of entities for PEM in Table 6. Compared with the model pre-trained with MLM+VTM+VTC, adding

Table 7. Effect of the number of frames on MSRVTT text-video retrieval and MSVD-QA. More frames generally lead to better performance with 8-16 frames achieve a good trade-off between metrics and computation expense.   

<table><tr><td rowspan="2">#frms</td><td colspan="3">MSRVTT-FT</td><td colspan="3">MSRVTT-ZS</td><td>MSVD-QA</td></tr><tr><td>R1↑</td><td>R10↑</td><td>MdR↓</td><td>R1↑</td><td>R10↑</td><td>MdR↓</td><td>Acc.↑</td></tr><tr><td>2</td><td>25.7</td><td>63.9</td><td>5</td><td>17.3</td><td>48.9</td><td>11</td><td>43.8</td></tr><tr><td>4</td><td>31.0</td><td>69.6</td><td>4</td><td>21.4</td><td>54.4</td><td>8</td><td>44.5</td></tr><tr><td>8</td><td>33.9</td><td>73.2</td><td>3</td><td>24.1</td><td>55.4</td><td>8</td><td>45.4</td></tr><tr><td>16</td><td>34.2</td><td>72.6</td><td>3</td><td>24.7</td><td>55.0</td><td>7</td><td>45.9</td></tr></table>

PEM brings consistent improvement with frequent entities. This suggests that the underlying principle of PEM to learn better region-entity alignment plays the essential role in its effectiveness. However, adding more low-frequency entities introduces noises in generating entity pseudo-labels, thus harming the pre-training.

Effect of number of frames. In Table 7, we show the results on downstream tasks with different numbers of input frames. Generally more frames lead to better performance, while such benefit saturates with more than 8 frames on the retrieval task. By sparsely sampling frames from the video and enabling end-to-end training of the visual backbone, ALPRO learns more effective representations than previous methods that use fixed offline features.

# 5. Conclusion

This paper proposes ALPRO , a new video-language pretraining framework that operates on sparsely-sampled video frames. ALPRO introduces video-text contrastive learning to align instance-level unimodal features, and prompting entity modeling for fine-grained region-entity alignment. We verify the efficiency and efficacy of ALPRO on multiple downstream datasets, where ALPRO achieves substantial performance improvement over existing models.

We believe ALPRO opens up a new direction for visionlanguage research, by exploiting the uptrending technique of prompting to generate semantic pseudo-labels. Here we list two potential ideas that can be further explored to improve ALPRO: (1) better prompt engineering / prompt tuning to improve the quality of entity pseudo-labels; (2) prompt-guided region selection with temporal information taken into consideration, which might improve the current way of random region selection. Last but not least, AL-PRO is not restricted to the video domain, and can be naturally extended to image-text representation learning, or even image representation learning.

Limitations and Broader Impacts. While Section 5 discusses the technical aspect and proposes potential improvements, here we highlight the potential negative societal impact. Our pre-training data is collected from the web, which may contain unsuitable videos, harmful texts, and private information, which could leak into the pre-trained models. Additional model analysis is necessary before deployment.

# References

[1] Elad Amrani, Rami Ben-Ari, Daniel Rotman, and Alex Bronstein. Noise estimation using density estimation for self-supervised multimodal learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 6644–6652, 2021. 7   
[2] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017. 5   
[3] Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisser-¨ man. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021. 1, 2, 4, 5, 7   
[4] Kobus Barnard, Pinar Duygulu, and David Forsyth. Clustering art. In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, volume 2, pages II–II. IEEE, 2001. 3   
[5] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Proceedings of the International Conference on Machine Learning (ICML), July 2021. 3   
[6] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pages 190–200, 2011. 5   
[7] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER: universal image-text representation learning. In ECCV, volume 12375, pages 104–120, 2020. 2   
[8] Ioana Croitoru, Simion-Vlad Bogolin, Marius Leordeanu, Hailin Jin, Andrew Zisserman, Samuel Albanie, and Yang Liu. Teachtext: Crossmodal generalized distillation for textvideo retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11583–11593, 2021. 7   
[9] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 702–703, 2020. 11   
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 2, 4, 5   
[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020. 5   
[12] Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, and Heng Huang. Heterogeneous memory enhanced multimodal attention model for video question answering. In Proceedings of the IEEE/CVF conference on

computer vision and pattern recognition, pages 1999–2007, 2019. 2, 7   
[13] Ali Farhadi, Ian Endres, Derek Hoiem, and David Forsyth. Describing objects by their attributes. In IEEE conference on computer vision and pattern recognition, pages 1778–1785. IEEE, 2009. 3   
[14] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval. In Proceedings of the European Conference on Computer Vision, pages 214–229. Springer, 2020. 2   
[15] Jiyang Gao, Runzhou Ge, Kan Chen, and Ram Nevatia. Motion-appearance co-memory networks for video question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6576–6585, 2018. 2, 7   
[16] Deng Huang, Peihao Chen, Runhao Zeng, Qing Du, Mingkui Tan, and Chuang Gan. Location-aware graph convolutional networks for video question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11021–11028, 2020. 7   
[17] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2758–2766, 2017. 7   
[18] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. arXiv preprint arXiv:2102.05918, 2021. 4   
[19] Jianwen Jiang, Ziqiang Chen, Haojie Lin, Xibin Zhao, and Yue Gao. Divide and conquer: Question-guided spatiotemporal contextual attention for video question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11101–11108, 2020. 7   
[20] Pin Jiang and Yahong Han. Reasoning with heterogeneous graph alignment for video question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11109–11116, 2020. 7   
[21] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017. 2   
[22] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017. 7   
[23] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25:1097–1105, 2012. 2   
[24] Christoph H Lampert, Hannes Nickisch, and Stefan Harmeling. Learning to detect unseen object classes by betweenclass attribute transfer. In IEEE Conference on Computer Vi-

sion and Pattern Recognition, pages 951–958. IEEE, 2009. 3   
[25] Thao Minh Le, Vuong Le, Svetha Venkatesh, and Truyen Tran. Hierarchical conditional relation networks for video question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9972–9981, 2020. 2, 7   
[26] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7331–7341, 2021. 1, 2, 4, 5, 7   
[27] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. Tvqa: Localized, compositional video question answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 1369–1379, 2018. 2   
[28] Ang Li, Allan Jabri, Armand Joulin, and Laurens van der Maaten. Learning visual n-grams from web data. In Proceedings of the IEEE International Conference on Computer Vision, pages 4183–4192, 2017. 3   
[29] Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven Hoi. Align before fuse: Vision and language representation learning with momentum distillation. In Advances in neural information processing systems, 2021. 5   
[30] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. Hero: Hierarchical encoder for video+ language omni-representation pre-training. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2046–2065, 2020. 1, 2, 4, 5   
[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 2, 4, 7   
[32] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts. In British Machine Vision Conference, 2019. 2, 5, 7   
[33] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. 2019. 4   
[34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018. 5   
[35] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon Bharti, and Ming Zhou. Univl: A unified video and language pre-training model for multimodal understanding and generation. arXiv preprint arXiv:2002.06353, 2020. 1, 2, 5   
[36] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval. arXiv preprint arXiv:2104.08860, 2021. 7

[37] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9879– 9889, 2020. 1, 2, 4, 5, 7   
[38] Antoine Miech, Ivan Laptev, and Josef Sivic. Learning a text-video embedding from incomplete and heterogeneous data. arXiv preprint arXiv:1804.02516, 2018. 7   
[39] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2630–2640, 2019. 1, 2, 4, 5, 7   
[40] Mohammad Norouzi, Tomas Mikolov, Samy Bengio, Yoram Singer, Jonathon Shlens, Andrea Frome, Greg Corrado, and Jeffrey Dean. Zero-shot learning by convex combination of semantic embeddings. In ICLR, 2014. 3   
[41] Mark Palatucci, Dean Pomerleau, Geoffrey E Hinton, and Tom M Mitchell. Zero-shot learning with semantic output codes. Advances in Neural Information Processing Systems, 22, 2009. 3   
[42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32:8026– 8037, 2019. 5   
[43] Mandela Patrick, Po-Yao Huang, Yuki Asano, Florian Metze, Alexander G Hauptmann, Joao F. Henriques, and Andrea Vedaldi. Support-set bottlenecks for video-text representation learning. In International Conference on Learning Representations, 2021. 2, 7   
[44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021. 3, 4, 7, 8, 11   
[45] Paul Hongsuck Seo, Arsha Nagrani, and Cordelia Schmid. Look before you speak: Visually contextualized utterances. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16877–16887, 2021.   
[46] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 5, 7   
[47] Richard Socher, Milind Ganjoo, Christopher D Manning, and Andrew Ng. Zero-shot learning through cross-modal transfer. In Advances in Neural Information Processing Systems, pages 935–943, 2013. 3   
[48] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video

and language representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7464–7473, 2019. 1, 2, 4   
[49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017. 3   
[50] Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond J Mooney, and Kate Saenko. Translating videos to natural language using deep recurrent neural networks. In HLT-NAACL, 2015. 7   
[51] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the ACM international conference on Multimedia, pages 1645–1653, 2017. 5, 6, 7   
[52] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 6787–6800, 2021. 2, 4, 7   
[53] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 2, 5   
[54] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1686–1697, 2021. 7, 8   
[55] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In Proceedings of the European Conference on Computer Vision, pages 471–487, 2018. 2, 5, 7   
[56] Bowen Zhang, Hexiang Hu, and Fei Sha. Cross-modal and hierarchical modeling of video and text. In Proceedings of the European Conference on Computer Vision, pages 374– 390, 2018. 7   
[57] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8746–8755, 2020. 1, 2, 4, 5, 7

# A. Implementation Details

# A.1. Prompt Templates

Inspired by [44], we ensemble multiple prompts templates to improve the quality of the generated pseudo-labels. We create the ensemble over the embedding space. Namely, we compute the average embedding of all the prompts for each entity. This allows scalable ensembling with more templates with the same computation cost as using a single prompt. During pre-training, the dataloader switches between video and image inputs. And we employ the following prompt templates for videos and images.

Table 8. Prompt templates used for video and image inputs.   

<table><tr><td>Prompts for video inputs</td><td>Prompts for image inputs</td></tr><tr><td>A footage of a {}.</td><td>A photo of a {}.</td></tr><tr><td>A footage of the {}.</td><td>A photo of the {}.</td></tr><tr><td>A footage of one {}.</td><td>A photo of one {}.</td></tr><tr><td>A video of a {}.</td><td>A picture of a {}.</td></tr><tr><td>A video of the {}.</td><td>A picture of the {}.</td></tr><tr><td>A video of one {}.</td><td>A picture of one {}.</td></tr><tr><td>A portrait of a {}.</td><td>A good photo of a {}.</td></tr><tr><td>A portrait of the {}.</td><td>A good photo of the {}.</td></tr><tr><td>A portrait of one {}.</td><td>A good photo of one {}.</td></tr><tr><td>A video footage of a {}.</td><td>A good picture of a {}.</td></tr><tr><td>A video footage of the {}.</td><td>A good picture of the {}.</td></tr><tr><td>A video footage of one {}.</td><td>A good picture of one {}.</td></tr></table>

# A.2. Finetuning Setups

In this section, we describe implementation details for fine-tuning the pre-trained model. For all the downstream datasets, we resize video frames to $2 2 4 \times 2 2 4$ . During finetuning, we randomly select $N _ { v }$ frames from the video, with parameters $\frac { N _ { v } } { 2 }$ frames from the first and second half of the video, respectively. We use RandomAugment [9]. We apply the same augmentation across frames sampled from one video. The default settings for finetuning on each dataset are in Table 9. During inference, we do not use augmentation and sample uniformly. In all the experiments, we use the same random seed (i.e. 42) to ensure reproducibility.

Table 9. End-to-end finetuning configurations for MSRVTT and DiDeMo text-to-video retrieval.   

<table><tr><td>config</td><td>MSRVTT</td><td>DiDeMo</td></tr><tr><td>optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>base learning rate</td><td>2.5e-5</td><td>4e-5</td></tr><tr><td>weight decay</td><td>1e-3</td><td>1e-3</td></tr><tr><td>optimizer momentum</td><td>β1, β2=0.9, 0.98</td><td>β1, β2=0.9, 0.98</td></tr><tr><td>learning rate schedule</td><td>linear decay</td><td>linear decay</td></tr><tr><td>batch size</td><td>64</td><td>96</td></tr><tr><td>max. text length</td><td>40</td><td>50</td></tr><tr><td>frame number</td><td>8</td><td>8</td></tr><tr><td>warmup ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>training epochs</td><td>5</td><td>10</td></tr><tr><td>augmentation</td><td>RandAug(2, 5)</td><td>RandAug(2, 5)</td></tr><tr><td>gradient accumulate step</td><td>1</td><td>1</td></tr></table>

Table 10. End-to-end finetuning configurations for MSRVTT-QA and MSVD-QA.   

<table><tr><td>config</td><td>MSRVTT</td><td>MSVD</td></tr><tr><td>optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>base learning rate</td><td>5e-5</td><td>5e-5</td></tr><tr><td>weight decay</td><td>1e-3</td><td>1e-3</td></tr><tr><td>optimizer momentum</td><td>β1, β2=0.9, 0.98</td><td>β1, β2=0.9, 0.98</td></tr><tr><td>learning rate schedule</td><td>linear decay</td><td>linear decay</td></tr><tr><td>batch size</td><td>96</td><td>96</td></tr><tr><td>max. text length</td><td>40</td><td>40</td></tr><tr><td>frame number</td><td>16</td><td>16</td></tr><tr><td>warmup ratio</td><td>0.1</td><td>0.1</td></tr><tr><td>training epochs</td><td>10</td><td>15</td></tr><tr><td>augmentation</td><td>RandAug(2, 5)</td><td>RandAug(2, 5)</td></tr><tr><td>gradient accumulate step</td><td>2</td><td>2</td></tr></table>

# B. Additional Examples of Pseudo-labels

![](images/89a1db5383d1b28fa7aa694aead763411132285f2690b0874a9f54ba624e019d.jpg)  
Top 5 pseudo-labels

1. Cruise (0.74)   
2. Ship (0.13)   
3. Ferry (0.10)   
4. Yacht (0.08)

Singapore - 19 august 2017: ferry boat and skyline in downtown core at marina bay.

![](images/09285bbfa4febe106db33278cd08d902a9574211c39cc46f15051cc8dbbc302a.jpg)  
Top 5 pseudo-labels

1. Pot (0.54)   
2. Flowers (0.12)   
  
  
5. Blooming (0.04)

Cat with big orange eyes sitting on dark wood surface near the houseplant...

![](images/422a907773060c71f49c411fef77230a865b234a59e4d51fdfda4dd28018905e.jpg)  
Top 5 pseudo-labels

1. Destination (0.24)   
2. Shore (0.20)   
3. Waters (0.17)   
4. Islands (0.12)   
5. Pool (0.03)

![](images/ce9cbfee04f161537931548605cb16cad140bc731f7387edaa9a70198a20884f.jpg)  
Taibila river landmark - dan.   
Top 5 pseudo-labels

1. Fruits (0.91)   
2. Lemon (0.04)   
3. Juice (0.02)   
4. Ingredients (0.01)   
5. Apples (0.01)

![](images/c9f24e78f5114261788fb58c28f319643d9e5bc048233270caf66ee476073ec7.jpg)  
Cutting apple into pieces on a cutting board.   
Top 5 pseudo-labels

1. Family (0.50)   
2. Friends (0.11)   
3. Couple (0.07)   
4. Parents (0.05)   
5. Horizon (0.03)

![](images/f9dcfcb5230e90282feff2b8458374e93953f50b40d6c186762f22326cf8d31e.jpg)  
A couple stands on a rocky shore as waves roll in and crash.   
Top 5 pseudo-labels

1. Cottage (0.88)   
2. Camping (0.08)   
3. Porch (0.01)   
4. Village (0.01)   
5. House (0.01)

![](images/c1c585606a6470c53f5f205723cf06aeba271ef2689b3a4b8f6422a9866e8b80.jpg)  
A slider shot of the old boat shed and the white gravel shore of dove lake…   
Top 5 pseudo-labels

1. Instruments (0.91)   
2. Plays (0.04)   
3. Play (0.02)   
4. Music (0.01)   
5. Performance (0.01)

![](images/2527fd26b1d21c707cbb81a69c3fdf34315519ebbd8afd141ad6dc532622afdf.jpg)  
Young women play the violin classic music fiddle.   
Top 5 pseudo-labels

1. Grazing (0.75)   
2. Sheep (0.12)   
3. Cattle (0.07)   
4. Animals (0.04)   
5. Cows (0.01)

![](images/154aef1de19466c73eac034daee850273856639a50c7e1cb221daf3ac7b44d30.jpg)  
A beautiful herd of brown goats and sheep grazing grass …   
Top 5 pseudo-labels

1. Skyscrappers (0.29)   
2. Towers (0.19)   
3. Development (0.05)   
4. Buildings (0.05)   
5. Rendering

Singapore - 19 august 2017: ferry boat and skyline in downtown core at marina bay.

![](images/37eac2487304964bbf1e1cadda50f52feb40831e446492e9f47ee8c1cac4d143.jpg)  
Top 5 pseudo-labels

1. Cats (0.95)   
2. Eyes (0.01)   
3. Sits (0.01)   
4. Animal (0.01)   
5. Rabbit (0.01)

Cat with big orange eyes sitting on dark wood surface near the houseplant...

![](images/bb947f28818e30e1cfc6f2e82b5a58231588a26753c4dcf1201f71b91ca5b069.jpg)  
Top 5 pseudo-labels

1. Mountains (0.40)   
2. Mountain (0.27)   
3. Peaks (0.11)   
4. Region (0.06)   
5. Hills (0.04)

![](images/7b3458989002c27ca36c0178876170e811021d51ecc9df5a26c93d8720a390c4.jpg)  
Taibila river landmark - dan.   
Top 5 pseudo-labels

1. Cooking (0.82)   
2. Cuts (0.09)   
3. Slices (0.06)   
4. Knife (0.01)   
5. Fingers (0.01)

![](images/c77eb74d08e915059f4f04132eb20884a0adf2d5dc1c87714aaa5b6d362369d9.jpg)  
Cutting apple into pieces on a cutting board.   
Top 5 pseudo-labels

1. Waves (0.55)   
2. Surf (0.16)   
3. Seashore (0.09)   
4. Sea (0.05)   
5. Horizon (0.04)

![](images/4e06943b0cfb73abcf2ad86de457ef05acc760a8ce8cf0059e6b20937d44f176.jpg)  
A couple stands on a rocky shore as waves roll in and crash.   
Top 5 pseudo-labels

1. Wildlife (0.31)   
2. Animals (0.20)   
3. Zoo (0.11)   
4. Fur (0.06)   
5. Cat (0.04)

![](images/4c1b8f557e07377d8cd8d524c3551e815522e2ad04bfd3575cf62eca54645513.jpg)  
Coati (nasua) walking in booking iguazu falls, argentina.   
Top 5 pseudo-labels

1. Roses (0.97)   
2. Gifts (0.01)   
3. Bridal (0.01)   
4. Card (<0.01)   
5. Petals (<0.01)

Beautiful wedding bouquet.

![](images/82ed3a9f38cf13906dba8eaaf532654c09ee54f0b6fd8408063cff27225b6cd5.jpg)  
… male tennis player hitting forehand and running to the net…   
Figure 4. Examples of the pseudo-labels generated by the prompter (scores in bracket). The highlighted areas are fed to the prompter. Our method generates a diverse range of common entity categories that are not usually covered by object detectors, e.g. towers, summit, yoga. Besides, entity labels do not always appear in the text description, serving as a source of corpus-level supervision.

Top 5 pseudo-labels

1. Tournament (0.83)   
2. Sports (0.07)   
3. Player (0.04)   
4. Athlete (0.02)   
5. Championship (0.02)