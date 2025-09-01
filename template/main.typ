#import "../lib.typ": *
//#import "@preview/nifty-ntnu-thesis:0.1.1": *
#let chapters-on-odd = false
#abbr.make(
  ("3DGS", "3D gaussian splatting"),
  ("TV", "total variation"),
  ("NVS", "novel view synthesis"),
  ("NeRF", "neural radiance field"),
  ("MLP", "multilayer perceptron")
)

#show: nifty-ntnu-thesis.with(
  title: [Optimizing 3DGS Feature-Planes for codec-enhanced compression],
  short-title: [],
  authors: ("Simon Sasse",),
  titlepage: true,
  chapters-on-odd: chapters-on-odd,
  bibliography: bibliography("thesis.bib", style: "council-of-science-editors-author-date"),
  figure-index: (enabled: true, title: "Figures"),
  table-index: (enabled: true, title: "Tables"),
  listing-index: (enabled: true, title: "Code listings"),
  abstract-en: [
    3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, offering real-time rendering capabilities with high visual fidelity. However, the large storage requirements of 3DGS scenes remain a significant barrier to widespread adoption, particularly on mobile and VR devices. This thesis addresses the compression challenges of CodecGS, a state-of-the-art 3DGS compression method that uses K-Planes feature representations encoded with video codecs. The core problem lies in the mismatch between video codecs, which are optimized for natural images with high spatial and temporal correlation, and the noisy, uncorrelated nature of learned feature-planes. To overcome this limitation, this work introduces smoothness modeling through Total Variation (TV) loss terms that encourage both spatial (intra-frame) and temporal (inter-frame) coherence in the feature-planes during training. Additionally, various video codec configurations are systematically evaluated, and optimizations for plane resolution and channel sizing are explored. The effectiveness of the approach is demonstrated through implementation of a complete decoding pipeline in Unity3D, successfully running on a Meta Quest 3 headset. Results show significant compression improvements while maintaining reconstruction quality, with the Unity implementation proving the practical viability of CodecGS for real-time applications on resource-constrained devices. This work advances the state-of-the-art in 3DGS compression and demonstrates a pathway toward deploying high-quality 3D scene representations on mobile platforms.],
  abstract-other: (
    "Zusammenfassung",
    "3D Gaussian Splatting (3DGS) hat sich im Bereich Novel View Synthesesis etabliert und bietet eine hohe Bildwiderholungsrate mit sehr guter visueller Qualität. Die großen Speicheranforderungen von 3DGS-Szenen stellen jedoch eine Herausforderung für die breite Anwendung dar, insbesondere auf mobilen Geräten und VR-Headsets. Diese Arbeit optimiert die Kompressionsrate von CodecGS, einer 3DGS-Komprimierungsmethode, die K-Planes-Feature-Repräsentationen verwendet, welche dann mit Video-Codecs kodiert werden. Die Feature-Planes werden mittels einem erweiterten Verlustterm für die effektive Kompression durch Video-Codecs optimiert. Da Video-Codecs für natürliche Bilder mit hoher räumlicher und zeitlicher Korrelation optimiert sind, werden Methoden entwickelt, um die gelernten Feature-Planes während des Trainings zu glätten. Die Glättungsmodellierung nutzt Total Variation (TV) Verlustterme, um sowohl räumliche (Intra-Frame) als auch zeitliche (Inter-Frame) Kohärenz in den Feature-Planes während des Trainings zu fördern. Zusätzlich werden verschiedene Video-Codec-Konfigurationen systematisch evaluiert und Optimierungen der Dimensionen der Feature-Planes getestet. Die Anwendbarkeit von CodecGS mit den vorgeschlagenen Verbesserungen wird Anhand einer Unity3D implementierung des gesamten Dekodierungsprozesses demonstriert. Die Ergebnisse zeigen signifikante Komprimierungsverbesserungen bei Erhaltung der Rekonstruktionsqualität, wobei die Unity-Implementierung die praktische Realisierbarkeit von CodecGS für Echtzeitanwendungen auf ressourcenbeschränkten Geräten demonstriert. Diese Arbeit erweitert den Stand der Technik in der 3DGS-Komprimierung und zeigt einen Weg zur Bereitstellung hochwertiger 3D-Szenenrepräsentationen auf mobilen Plattformen auf.",
  ),
)


// LATEX CHANGES from Felix
#set page(margin: 1.4in)
#set par(leading: .55em, spacing: 1.5em, first-line-indent: 0em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)

// more space around figures
#show figure.where(kind: image): set block(above: 2.5em, below: 2.5em)
#show figure.where(kind: "algorithm"): set block(above: 1.5em, below: 1.5em)

// Summary of figure caption
// author: laurmaedje
// Put this somewhere in your template or at the start of your document.
// #let in-outline = state("in-outline", false)
// #show outline: it => {
//   in-outline.update(true)
//   it
//   in-outline.update(false)
// }

// #let flex-caption(long, short) = context if in-outline.get() { short } else { long }

// And this is in the document.
// #outline(title: [Figures], target: figure)



= Introduction
<chap:intro>
#abbr.a[3DGS] is a #abbr.a[NVS] method published by #cite( <kerbl3Dgaussians>, form: "prose"). It optimizes the characteristics of many 3D gaussian primitives to explicitly model complex scenes. A #abbr.a[3DGS] scene is naively stored in an unorganized point-by-point manner. Every 3D point has a list of 59 attributes. #abbr.a[3DGS] dramatically increased rendering speed compared to its predecessors, but the resulting file size poses a major challenge.
== 3D scene representation (check again)
<sec:3dscene-representation>

Traditional 2D images capture 3D space by projecting it onto a 2D plane, but they inherently lack depth information. This creates ambiguity in monocular depth perception, as it becomes difficult to distinguish whether an object appears smaller because it is actually smaller or simply located further away from the camera @zeng2024rsaresolvingscaleambiguities. Depth information can be very useful for a variety of applications like obstacle detection, metric measurement, medical imaging and image post-processing @wang20243drepresentationmethodssurvey.

To address these limitations, enriched 2D representations have been developed. These include depth maps that provide explicit distance information, surface normal maps that describe the orientation of surfaces, and stereo image pairs that enable depth estimation through disparity calculation between multiple viewpoints.

For applications requiring complete free viewpoint navigation, full 3D representations are necessary. These representations include voxel grids that discretize space into regular 3D cells, point clouds that represent geometry as collections of 3D points, and meshes that define surfaces through vertices, edges, and faces @wang20243drepresentationmethodssurvey. #abbr.a[3DGS] is a point cloud, where every point is augmented with a learnable 3D gaussian function to accurately model the appearance @kerbl3Dgaussians. 

In contrast to these explicit representations, implicit representations define 3D scenes through mathematical functions. Examples include deepSDF @SDFpaper that encodes the distance to the nearest surface by learning a signed distance function, #abbr.pla[NeRF] @NeRFPaper that use neural networks to represent radiance fields, and triplane representations that project 3D information onto orthogonal 2D planes, popularized by #cite(<triplaneBasePaper>, form: "prose")

The choice of scene representation also influences the rendering approach, with methods typically using either rasterized rendering, which projects 3D geometry onto 2D images, or ray tracing, which simulates light transport by tracing rays through the scene. In ray-casting, for example used in #abbr.a[NeRF], a ray is shot from the desired viewpoint through the scene for every pixel of the output image. On every ray, multiple points are sampled for which a model like #abbr.a[NeRF] or deepSDF is queried and returns color and opacity. These sampled points are then blended to form the final pixel color. This method is computationally intensive but delivers comparably realistic results. Rasterization on the other hand is a faster process that produces less realistic lighting. Objects from the 3D scene are projected onto the image plane. In the case of #abbr.a[3DGS], all 3D gaussian functions are projected or splat onto the image plane and then blended in the inverse order of the distance to the camera @kerbl3Dgaussians
== Novel view synthesis
<sec:nvs>

#figure(
  image("figures/nvsExplanation.png", width: 80%),
  placement: auto,
  caption: [*Novel view synthesis.* #abbr.a[NVS] generates unseen views onto a 3D object or scene from sparse input perspectives.]
)
<fig:nvs-explanation>
// *USE 3dgs introduction for this.*
// - #abbr.pla[NVS] – an overview
//   - Motivation: achieving seamless 3D scene rendering from sparse views
//   - Key challenges: occlusions, lighting variations, and real-time performance
#abbr.a[NVS] is the generation of new viewpoints from a sparse input dataset usually consisting of real world images from different perspectives #cite(<NVSOrigin>). Some methods generate an explicit 3D representation of the scene while others directly generate the image from the queried viewpoint. Key challenges are occlusion of objects or parts of objects, realistic lighting, real-time rendering, training time and storage requirements.
Since the first approaches of warping images and finding point correspondence to fill in new perpectives, new methods have emerged that use machine learning to create more sophisticated reconstructions. An important milestone was the publication of #abbr.add("SfM", "Structure from Motion") #abbr.a[SfM] that introduced the possibility to reconstruct sparse point clouds from images#cite(<sfm2006>). Subsequently, methods where published that use re-projection of the images onto the generated geometry. In #cite(<NeRFPaper>, form: "year") the publication of #abbr.pla[NeRF] made it possible to synthesize novel photorealistic views by using a large #abbr.a[MLP] to predict the pixel color for a location and viewing direction, though rendering speed was far below one #abbr.add("fps", "frame per second", "frames per second") #abbr.a[fps]#cite(<NeRFPaper>)#cite(<kerbl3Dgaussians>).

In #cite(<kerbl3Dgaussians>, form: "year") the publication of #abbr.a[3DGS] introduced a novel method with an explicit 3D representation using many gaussian primitives, featuring rendering speeds with far more than 30 #abbr.a[fps] and competitive quality#cite(<kerbl3Dgaussians>).


// - Transition to Gaussian Splatting
//   - Concept: representing scenes with 3D Gaussian primitives
//   - Benefits: enhanced compression and improved rendering quality
// - From traditional methods to Gaussian Splatting
//   - Limitations of existing view synthesis techniques
//   - How Gaussian Splatting addresses these challenges
// - Chapter roadmap
//   - Review of historical approaches in novel view synthesis
//   - Detailed introduction to Gaussian Splatting and its algorithmic steps

== 3D gaussian splatting
<sec:gaussian-splatting>

// - Overview of Gaussian Splatting
//   - Definition and significance in computer graphics
//   - Comparison with traditional rendering techniques
// - Key components of Gaussian Splatting
//   - Gaussian primitives: definition and properties
//   - Rendering pipeline: from primitives to final image
// - Advantages of Gaussian Splatting
//   - Real-time performance and efficiency
//   - Enhanced visual quality and detail preservation
// - Challenges and limitations
//   - Computational complexity and resource requirements
//   - Addressing artifacts and rendering issues
//   - File size and compression
#abbr.a[3DGS] generates a 3D representation from input images from different perspectives of a static scene.
Splatting was originally published in #cite(<splatting1991>, form:"year") as a volume rendering technique that uses 3D gaussian functions that are projected onto the image plane, blending into the final image instead of relying on computationally intensive ray-casting #cite(<splatting1991>).

In #abbr.a[3DGS], the smallest entity of the scene is a 3D gaussian function. This is a 3D coordinate with 59 attributes: a covariance matrix#footnote[The covariance matrix represents the shape and orientation of the 3D Gaussian primitive giving it freedom to be rotated and scaled in all directions.], opacity and #abbr.add("SH", "Spherical Harmonics") #abbr.a[SH] coefficients #footnote[Spherical Harmonics are functions that are defined on the surface of a sphere. They can be used to model view-dependent effects for every gaussian primitives, like reflections.The first degree is the diffuse RGB color.]#cite(<compressionsurvey3dgs>). These gaussian primitives with their attributes are optimized during training to accurately model the scene. The gaussians are initialized using the sparse point cloud resulting from #abbr.a[SfM] on the input images #cite(<sfm2016>). During training, primitives are redistributed, split or removed to better model high detail areas. This densification is interleaved with optimization of the attributes of each gaussian to accurately model the geometry and the appearance at the same time (see @fig:3dgs-opt).

#figure(
  image("figures/3dgs_optimization.png", width: 125%),
  placement: auto,
  caption: [*3DGS optimization workflow.* Structure from motion is used to derive the camera positions and the original point-cloud to initialize the gaussian centers. The other attributes are initialized randomly. During training, a camera is selected and the 3D gaussians are rendered onto its image plane. The resulting image is compared to the corresponding ground truth image and the gradient calculated. Adaptive density control adjusts the number of gaussians according to the gradient and the gaussian attributes are updated as well.]
)
<fig:3dgs-opt>

This method achieves impressive visual fidelity while also featuring real-time rendering speed. The storage requirement for resulting scenes depend only on the number of gaussians and can be as large as several #abbr.add("GB", "gigabyte") #abbr.pla[GB] for complex scenes. #cite(<kerbl3Dgaussians>, form: "prose") do not suggest a compressed storage format but utilize a #abbr.add("PLY", "Polygon File Format") #abbr.a[PLY] that naively stores the data in a point-wise manner. Subsequently, many compression approaches where developed (see @sec:compression)

== Rate-distortion theory
<sec:rate-distortion>
#figure(
  image("figures/rd_example.png"),
  placement: auto,
  caption: [*Rate-distortion example curve.* Rate-distortion curves show the trade-off between reconstruction quality and bitrate. The closer a measurement is to the top left, the better is the trade-off, because reconstruction quality is higher and bitrate is lower. In the example case, method B shows higher reconstruction quality for very low bitrates, while method A has a better reconstruction quality than method B when higher bitrates can be accepted.]
)
<fig:rd-example>
Rate-Distortion Theory is a concept of information theory, that defines theoretical limits of lossy data compression. In a typical lossy coding system, the input data $s$ is encoded and represented in a compressed state $b$ using the encoder. The decoder yields the reconstruction $s'$. The rate is defined as the average number of bits of $b$ per input symbol of $s$. In lossy compression, the encoding is irreversible and thus the $s'$ is only an approximation of $s$. The difference between $s$ and $s'$ measured by a fitting metric is called distortion. The Rate-Distortion theory gives a lower bound for the minimum average number of bits per input symbol needed to represent $s$ as $b$ while keeping the distortion below a certain threshold. This theoretical limit is not practically computable @videoCodecBasics1 and does not relate to a specific compression method, but the defined trade-off is a crucial measure for comparing compression approaches. This is often done by using Rate-Distortion curves, plotting the rate on the x-axis and the distortion on the y-axis.


== Objective of this thesis
<sec:objective>
The goal of this thesis is to improve the rate-distortion performance of a #abbr.a[3DGS] compression approach named CodecGS (see @sec:codecgs).

In the following #link(<sec:preliminaries>, "chapter"), the concepts and algorithms related to this work are described. Additionally, a detailed recap of CodecGS is given. Video codecs play an important role in CodecGS. Hence, the main principles used in modern video codecs are explained.

The #link(<chap:methods>, "methods chapter") first describes finding optimization possibilities by analyzing CodecGS thoroughly. Then, the main ideas that are introduced in this work to improve the rate-distortion performance of CodecGS are explained. The #link(<chap:results>, "results chapter") describes the outcomes of the analysis and the implemented methods. At last, the results are summarized and discussed in the context of current research and potential application.

= Preliminaries
<sec:preliminaries>
This section describes existing compression methods for #abbr.a[3DGS] and explains the theoretical background about methods that form the basis of this work.

== Existing compression methods for #abbr.lo[3DGS]
<sec:compression>
// Compaction, compression, implicit explicit, decoding etc
// - describe ply format
// - important are \#points and \#features
//   - reduce SH
//   - prune points
//   - reduce accuray to half float or so
// - self organizing gaussians
// - Feature-plane approach
//   - planes are trainable MLP parameters
Although #abbr.a[3DGS] improved rendering speed, quality and training duration, the needed storage still hinders it from being adopted on mobile devices like standalone #abbr.add("HMD", "Head Mounted Display") #abbr.pla[HMD] #cite(<compressionsurvey3dgs>).
#cite(<compressionsurvey3dgs>, form: "prose") identified two categories of tools used to reduce the size of #abbr.a[3DGS] scenes as *compaction* and *compression*. Compaction tools aim at reducing the number of gaussians while compression tools try to minimize the size of the gaussian attributes being the position, rotation, scale, #abbr.a[SH] coefficients and opacity. The latter can be grouped into attribute compression and structured representations. Structured representation use completely different data structures that can be used to accurately restore the gaussian primitives for rendering and require less space. Attribute compression methods attempt to reduce the size of individual attributes, for example by reducing redundancies or applying quantization. Tools from these categories are combined in different publications to optimize the rate-distortion problem for #abbr.a[3DGS].

*Compaction* is mainly achieved by the #abbr.add("ADC","Adaptive Density Control") #abbr.a[ADC] already integrated in the original paper #cite(<kerbl3Dgaussians>). #abbr.a[ADC] optimizes the spatial distribution of gaussians such that it accurately models detailed areas while not wasting gaussians on sparse areas. During training, gaussians can be split, cloned or removed, which enables modeling parts in more detail, filling gaps where no gaussians where present and removing gaussians that are out of place or not contributing to the reconstruction quality (see optimization flow in @fig:3dgs-opt). #cite(<kerbl3Dgaussians>, form: "prose") apply a densification step every 100 iterations and remove any gaussians of which the $alpha$ value is below a certain threshold $epsilon_alpha$. Densification is done in two scenarios: Over-reconstruction, where large gaussians cover areas that are more detailed in the ground truth and under-reconstruction in areas with missing geometry in the gaussian model. In the original #abbr.a[3DGS] these areas are defined as having large view-space gradients. Additionally, all opacities $alpha$ are periodically set to a value close to zero. As optimization continues, only the $alpha$ of more relevant gaussians is increased and others can be removed #cite(<kerbl3Dgaussians>). Subsequently published methods introduce different ways to decide, when to clone, split or remove primitives. With mini-splatting, #cite(<minisplatting2024>, form: "prose") increased reconstruction quality while greatly reducing the number of gaussians #cite(<compressionsurvey3dgs>). They analyzed that the original densification leads to a suboptimal distribution due to overlapping gaussians and underreconstructed areas. Blurry artefacts are identified by finding larger areas, where one gaussian has the strongest influence for each pixel @minisplatting2024. The responsible gaussians are then split. To reduce under-reconstruction, mini-splatting uses depth information to repeatedly reinitialize the gaussian point cloud during optimization. This produces a denser reconstruction, which is then simplified by retaining gaussians with the largest impact on the rendering and a sampling approach that takes into account the geometry @minisplatting2024. Besides the compaction focused variant, the authors show experiments for a quality focused variant, mini-splatting-D, omitting the simplification step. Both variants of mini-splatting will be integrated with the feature-plane based approach of CodecGS to increase reconstruction quality and rate-distortion performance (see @sec:compaction).

Gaussian primitives generally consist of 59 attributes that might not be necessary in all cases. *Attribute compression* aims to reduce redundancy or find more efficient ways to encode the information needed for each gaussian. The most prominent approach is reducing the #abbr.a[SH] coefficients or replacing it by another system to model view-dependent effects, since in #abbr.a[3DGS], every gaussian has 48 #abbr.a[SH] coefficients. If we assume equally sized floats for all attributes, the #abbr.a[SH] coefficients make up for more than 80% of the data. But attributes can also be compressed by quantizing some or all values to lower precision floating point numbers, by using vector quantization paired with importance estimation guiding the quantization strength.

*Structured representations* transform the data into other data structures such that similarities between primitives are exploited, instead of storing all gaussians unordered.
CompGS and HAC use spatial correlations by deploying anchor based approaches paired with highly efficient coding techniques.
SOG uses the simple fact, that the unstructured gaussians can be reorganized in the storage layout to reorder them such that, when stored in 2D planes per attribute, these planes resemble smooth images that increase the rate-distortion performance when encoded with video or image codecs @sog2025.

== K-planes model
<sec:kplanes>
Originally developed as a standalone radiance field#footnote[A radiance field defines defines color and density/opacity for every Point in space for all viewing directions. Time can possibly be added as a fourth dimension.] model, the k-planes model uses a feature-plane representation to encode the attributes of the radiance field in two-dimensional feature-planes @kplanes2023. From all N dimensions, all pairs are chosen to form 2D planes, resulting in _N choose 2_ planes, i.g. for 3 dimension: XY, XZ and YZ (see @fig:kplanes-model). Points sampled from ray-casting are used to extract the values from the planes, using bilinear interpolation and element-wise multiplication. Small #abbr.add("MLP", "multi layer perceptron") #abbr.a[MLP] decoders are used to infer the final view-dependent color and density. This model can be adapted to predicting the gaussian attributes for a given point and thus compressing #abbr.a[3DGS] scenes  as shown in the following section @lee2025compression3dgaussiansplatting.

#figure(
  image("figures/DecodingKPlanesModel.png"),
  placement: auto,
  caption: [*K-planes model decoding pipeline.* To render an image using the k-planes model, rays are cast from the camera for every pixel. Points are sampled along the ray and for every point, the features from the k-planes model's XZ, XY and YZ planes are extracted using bilinear interpolation. A small #abbr.a[MLP] decoder is used to derive the view-dependent color and opacity. To obtain the final pixel color, all points along a ray are blended. During training, the values in the planes and the #abbr.a[MLP] decoder are optimized.]
)
<fig:kplanes-model>

== CodecGS model
<sec:codecgs>

#cite(<lee2025compression3dgaussiansplatting>, form: "prose") proposed CodecGS, a #abbr.a[3DGS] compression method that integrates the k-planes model with traditional video-codecs @kplanes2023. While originally the k-planes model predicts color and density, gaussian primitives have far more attributes to be inferred. To account for this, the complexity of the model is raised by adding multiple channels to each plane and additionally employing four k-planes models for predicting view-dependent color, rotation, scale and opacity, respectively. For every coordinate of the separately stored point cloud, each k-planes model is queried and the outputs passed to a small #abbr.a[MLP] that predicts the gaussian attributes. The planes are treated as monochrome images, concatenated and encoded using #abbr.add("HEVC", "High Efficiency Video Coding") #abbr.a[HEVC]@flynn2016HEVC.

The feature-plane training starts after 15k iterations of the original #abbr.a[3DGS] densification and is structured progressively. First, only the first channel of each plane is optimized and this is step-by-step extended to the full channel size. To optimize the planes for video-codec compression, #cite(<lee2025compression3dgaussiansplatting>, form: "prose") propose entropy modeling, that decreases the entropy towards the end of the training by adding an entropy term to the loss. The entropy is calculated on 8 by 8 blocks of the#abbr.add("DCT", "discrete cosine transform") #abbr.a[DCT] coefficients of the planes. Additionally, $cal(L)_1$ loss#footnote[The $cal(L)_1$ loss penalizes the total sum of all values in a plane.] is applied progressively to reduce  non-zero values in blank regions.

#figure(
  image("figures/codecGS_optimization.png"),
  placement: auto,
  caption: ([*CodecGS optimization workflow.* First, the original #abbr.a[3DGS] densification method is used to initialize and densify the point cloud, after which the feature-plane training starts. During training, the randomly initialized kplanes models are queried for every point using bilinear interpolation. The extracted features are decoded using the #abbr.a[MLP] and the resulting 3D gaussians are rendered using the #abbr.a[3DGS] rendering pipeline. Gradients are passed back to update the #abbr.pla[MLP] and kplanes models. Additionally,  #abbr.a[DCT]-Entropy modeling contributes to the gradients to reduce the entropy in the feature-planes.]),
)

#figure(
  image("figures/DecodingCodecGS.png"),
  placement: auto,
  caption: [*CodecGS decoding pipeline.* To render an image using CodecGS,  the gaussian features need to be decoded first. Four k-planes models are used, for color and SH, scale, rotation and opacity.  All gaussian positions are used to extract the corresponding features from all four K-Planes model's XZ, XY and YZ planes using bilinear interpolation. Four small #abbr.pla[MLP] are used to derive the gaussian attributes. These can then be used for standard #abbr.a[3DGS] rendering.]
)
<fig:decodingCodecGS>

CodecGS yields competitive results regarding compression and quality as shown by #cite(<compressionsurvey3dgs>, form: "prose"). While reaching a better rate-distortion trade off than the state-of-the-art hac++ and contextGS in terms of LPIPS and SSIM, codecGS  is outperformed when using PSNR as a metric. Decompression time has not yet been a measure included in publications in the field of #abbr.a[3DGS] compression, thus at the time of writing there is no representative comparison between different compression approaches. By integrating hardware-accelerated video codecs, CodecGS potentially allows to run the decompression on low-end devices like #abbr.pla[HMD]. This is demonstrated in section @sec:unity-implementation by implementing the decoding pipeline in Unity3D being able to run it on a Meta Quest 3.

// The results shown in this work build upon CodecGS 2, an enhanced yet unpublished version of CodecGS @leeSoonbinCodecGSUnpublished. Advanced densification yields improved reconstruction quality.

== Video Compression
<sec:video-compression>
Uncompressed videos exhibit sizes that prevent any mass usage as we see it in most applications like digital television, social-media, video-communication etc. Storage and especially network bandwidth necessitate high compression. To have an acceptable user-experience, fast decompression is important. Since digital storage devices opened the possibility to store videos digitally, methods where developed to decrease their size. In the following, the tools needed to understand the developed methods in this thesis regarding video-codecs will be introduced.

=== Video Codecs (SOURCES NEEDED)
<sec:codecs>

// possible sources:
// https://iphome.hhi.de/wiegand/assets/pdfs/VBpart1.pdf
// https://www.inf.fu-berlin.de/lehre/SS22/ImageVideoCoding/extra/VBpart2.pdf
A video codec is a pair of software or hardware that is capable of compressing and decompressing a video. The word _codec_ is a portmanteau of _coder_ and _decoder_, respectively responsible for compression and decompression.

Video codecs use two main principles, the reduction of redundancy and the removal of irrelevant information. Natural images typically exhibit strong temporal and spatial correlation. Higher correlation in this case means, that a pixel can be predicted by its surrounding pixels (spatial) as well as by pixels in previous or succeeding frames (temporal) with higher probability. Intraframe and interframe coding tools aim at  reducing the redundancies and remove information irrelevant to the perceiver. E.g. the sky is very likely to be a large almost uniform area in outdoor videos which can be represented by far less bits than a raw video would need. Additionally, humans do not perceive every bit of information equally strong. E.g. humans notice color changes at a lower resolution than brightness changes, leading to the idea of chroma subsampling, where color information is stored at half the resolution of brightness information @videoCodecBasics2.

In a statistical sense, video codecs optimize the rate-distortion-problem (see @sec:rate-distortion), that in the context of videos is used to determine the minimum average storage needed to represent a given raw video without exceeding a given reconstruction error. The reconstruction error is a measured difference between the raw video and the video after being encoded and decoded again.

Intraframe compression relies on transform coding (see @sec:transform-coding) and quantization while interframe compression searches for similar areas in the current and all reference frames and only stores the differences and a motion vector (see @sec:inter-frame-coding).

// - Coder Decoder
// - intra frame
// - inter frame
// - Decorrelating the data using DCT
// - Quantization
// - Rate Distortion Optimization
// - Quality Settings regarding lossiness:
//   - bit depth is bits per color per pixel (8/10/12/16bit)
//   - QP value
//   - Resolution

=== Transform Coding
<sec:transform-coding>
All common video codecs use transform coding for intraframe prediction. Here, the pixel data is transformed into another domain where it is processed further. This transform needs to be reversible in order to reconstruct the video. Following #cite(<videoCodecBasics1>, form: "prose") PAGE 177, the main goal of transform coding is to reduce the correlation of the video data which concentrates the information on a few transform coefficients and enables more effective quantization.
The #abbr.a[DCT] or approximations are used in most video codecs and thus will looked at in more depth.

The #abbr.a[DCT] is a frequency transformation that takes as input a discrete spatial or temporal signal and outputs frequency information @DCTorigin. In video compression it mostly operates on 8x8 pixel blocks that are transformed into an 8x8 coefficient grid for a symmetric matrix of cosine functions, every index doubling the frequency (see @fig:DCT a).
#subfigure(
  align: bottom,
  placement: auto,
  figure(
    gap: 10mm,
    image("figures/dct2_basis.png"),
    caption: [*DCT-2 basis*
  ]),<fig:dct-coeff>,
  figure(
    image("figures/dct2_matrices.png"),
    caption: [*DCT-2 example.*]
    ),<fig:DCT>,
  columns: (1fr, 1.5fr),  
  caption: [*DCT-2 example.* (a) Shows the cosine function forming the basis for a 8x8 #abbr.a[DCT]-2. These 64 functions are weighted using the #abbr.a[DCT]-2 coefficient matrix. (b) Shows two example 8x8 image blocks, the top one containing random values and the second one representing a smooth horizontal gradient. The #abbr.a[DCT]-2 coefficient matrices for both examples show exemplarily that a smooth image block can be represented by dramatically less coefficients than a random one. The coefficients are the weights corresponding to the #abbr.a[DCT]-2 basis in (a).],
)
<fig:dct-intro>

This transform separates high frequency information from low frequency information enabling quantization targeted at higher detail. This targets information that is less important for human perception. Additionally, for natural images it leads to a more concentrated representation by de-corelating the data. The smoother the input data is, the easier it can be described with low frequency cosine functions and thus with less #abbr.a[DCT] coefficients (see @fig:DCT b).

The #abbr.a[DCT] builds on the #abbr.add("DFT", "Discrete Fourier Transform") #abbr.a[DFT] which is a widely used transform to represent data in the frequency domain. It periodically extends the original signal and models it as a set of cosine functions with different importance. Using the inverse-#abbr.a[DFT] the original signal can be reconstructed @videoCodecBasics1.
The mostly used #abbr.a[DCT]-II is constructed by appending the mirrored input signal sequence to the original signal and doubling it. The result is a symmetric repetitive signal of four times the length. Due to the symmetry of the extended input sequence, the imaginary components of the #abbr.a[DFT] cancel out. Additionally, the #abbr.a[DCT]-II is independent of the input sequence @videoCodecBasics1. The #abbr.a[DCT]-II extended to two dimensions is used in many video and image coding standards, although it has been replaced by a faster approximation in the newer standards since H264/AVC @videoCodecBasics1.

=== Temporal redundancy reduction
<sec:inter-frame-coding>
Video sequences typically exhibit strong temporal correlation that is not exploited by transform coding @videoCodecBasics2. The idea of reducing temporal redundancy was first mentioned in a British Patent from #cite(<interpatent>, form: "year"). Combining both inter and intracoding leads to the concept of hybrid video coding, patented in the United States in #cite(<hybridCoding>, form: "year").

 Most modern video coding systems divide the frames into blocks and decide on the best coding mode for each block. This approach makes it possible to flexibly use transform coding for new elements and interframe coding for parts of the picture, that existed in the reference frame. The simplest interframe coding method is the conditional replenishment or skip mode @skipmode. As suggested by the name, in this mode the block is just skipped and taken as is from the already coded reference frame. This works well for static backgrounds but not for slightly moving scenes @videoCodecBasics2. To further improve the coding efficiency, the difference mode was introduced. It takes advantage of the fact, that a block $A$ can be easily reconstructed from another block $B$ and the difference between both $Delta_(A,B) = A-B$:  
 
 $
 A = B - Delta_(A,B)
 $ <form:differenceMode>

In cases with relatively flat object that are slightly shifted between to blocks, the residue block $Delta_(A,B)$ contains more zeros and can then be transmitted via transform coding more efficiently than the actual block $A$.
This only holds up, if the coded region does not move to fast or has to much texture. Thus both, the skip and the difference modes treat special cases of moving pictures. A more generalized approach is needed to cover faster moving regions. Motion-compensated prediction tackles this by allowing to predict block $A$ from a freely selected block $B$ in the reference frame. To be able to reconstruct $A$, a displacement vector is transmitted that points to the reference block $B$ @videoCodecBasics2. The process of finding the best matching reference block is called motion estimation. More coding modes exist, but are not in the scope of this work.

What modes can be used in a frame is defined by the coding structure. Specifically, it defines frames that are entirely intra coded, called I frames, P frames that can use intra and single direction inter coding tools and B frames that can use intra and bi-directional inter coding. Bi-directional coding refers to using a succeeding and a preceding block to predict the current block. Typical coding patterns are IPPP and IBBP, where IPPP is the simplest coding structure encoding all frames in display order using the preceding frame only. IBBP first codes all I frames, then all P frames using predictions from the I frames. Only now, the B frames can be coded, because they rely on bi-directional prediction. This leads to a tree like coding structure. If we have more than two succeeding B frames, we need to specify the hierarchichal order in which they are coded and thus which frames can be taken into account for their prediction @videoCodecBasics2. A complex IBBB coding scheme is depicted in @fig:ibbbCoding.

#figure(
  image("figures/IBBPcoding.png"),
  placement: auto,
  caption:[*Coding structure example.* Showing an IBBB coding structure with four hierarchy levels @videoCodecBasics2. The subscript numbers show the level in the hierarchy whilst the numbers below show the order of coding. Looking at the first group of pictures, the I-frame is coded first only with intra coding tools, followed by the lowest level B-frame, referencing the I-frame. Now the first level B-frame can be coded taking into account the two previously coded frames. Two more levels are coded and then, the next group of pictures is processed similarly.]
)
<fig:ibbbCoding>



=== Compression of non-image data
<sec:non-image-data>

The feature-planes resulting from CodecGS consist of learned parameters that are optimized to best reconstruct the gaussian features (see @sec:codecgs). They are then compressed using #abbr.a[HEVC], which is a video coding standard optimized for natural images. Thus the question arises, if this non-image data is suited for compression with video codecs.
Little to no literature exists about how to best store non-image data in images suitable for compression, thus the basic principles of video codecs and image compression will be used to develop methods to optimize the feature-planes for video-codec integration. As described in @sec:codecs video codecs use inter and intraframe coding to reduce temporal and spatial correlation and reduce irrelevant information. Wether information is irrelevant depends on the purpose of the data, which in this case is not the perception of the decoded video by a human, but the reconstruction of a #abbr.a[3DGS] scene. Information irrelevant to human perceivers may not be irrelevant to the quality of the decoded scene. Therefore, CodecGS relies on near-lossless compression by setting the #abbr.add("QP", "quantization paremeter") #abbr.a[QP] of the video codec to 1. Despite of that, transform coding (see @sec:transform-coding) and slight quantization effectively reduce the size of the feature-planes @lee2025compression3dgaussiansplatting.

To further improve the compression ratio, CodecGS prepares the feature-planes such that video codecs can work more effectively.
This is done using loss terms that aim at improving the video codecs rate-distortion performance. Since the video codec cannot be used in training due to not being differentiable, metrics are needed that approximate the video codecs rate-distortion performance.
// - not the expected properties of natural video
// - intra frame coding main mechanisms (DCT and Quantization) might not work well
//   - result: intra frames are lot larger than data that is not as noisy
// - inter frame coding might not work well due to lack of temporal correlation
//   - result: inter frame P/B frames are same size as pure intra frames -> no temporal prediction possible

// --> find a way to temporally or spatially smooth data

=== Metrics estimating rate-distortion performance of video codecs
<sec:metrics-for-compressibility>
The ratio by which a compression software like a video encoder can compress the input, depends on the configurations of the software, especially the accepted loss, but is also influenced by the input itself. Here, multiple metrics are used, that can estimate the compression ratio given an input file, without actually running the compression software. This can be useful, when running the video codec is not feasible due to time constraints or in scenarios, where differentiability is important. In the context of CodecGS, a measure of how large the plane-videos will be is useful to guide the training process towards smaller compressed planes by introducing a loss term that penalizes hard-to-compress planes.

#cite(<lee2025compression3dgaussiansplatting>, form: "prose") use entropy modeling, where the block-wise entropy $I$ is calculated on the #abbr.a[DCT] coefficients $F$ of a plane $P$, adding uniform noise $U$ to the #abbr.a[DCT] coefficients.

$ I(F(P)) &= EE [ - log p (accent(F(P), ~))] \
accent(F(P), ~) &= F(P) + u,u ~ U(- 1/Q_"step", 1/Q_"step") \
 F(P)_(u,v) &= sum_(x=0)^(N-1) sum_(y=0)^(M-1) P_(x,y) cos((pi(2x + 1)u)/(2N)) cos((pi(2y + 1)v)/(2M)) $
 <eq:entropie>

$I(F(P))$ will be denoted as _#abbr.a[DCT]-entropy_ or as the loss term $cal(L)_"ent"$. Additionally, the _$cal(L)_1$-norm_ that captures the total sum of values in the planes is applied to increase their sparsity and thus reduce unnecessary non-zero values.

$ cal(L)_1(P) = sum_(x,y) P_(x,y) $

Intuitively, an image or video exhibits a higher compression ratio when it has less fine grain detail. To capture this feature, the #abbr.a[TV] can be used, that calculates the sum of absolute differences between all neighboring pixel pairs, horizontally and vertically @totalvariationloss. #abbr.a[TV]-loss is used in plenoxels, tensorRF and kplanes models as a loss term to encourage smoothness in space @plenoxels @tensorrf @kplanes2023.

$ cal(L)_"TV" (P) = sum_(i,j)  sqrt(|P_"i+1,j" - P_"i,j"|^2 + |P_"i,j+1" - P_"i,j"|^2) $
<eq:tv>

In the following chapter, the feature-planes will be analyzed and multiple methods to increase the rate-distortion performance described.

// 
// Another way to measure smoothness is to apply a gaussian filter to the image and calculating the difference between the smoothed and the original image. This will be referred to as _Smoothed-to-Original Difference_ (SOURCE).

// FORMELN UND IDENTIFIER a la $cal(L)_"ent"$ $cal(L)_"smth"$ $cal(L)_"sd"$


= Methods
<chap:methods>
CodecGS uses the k-planes model to store gaussian attributes compactly in feature-planes and incorporates video codecs to encode and decode the concatenated planes @lee2025compression3dgaussiansplatting. Storing non-image data in videos poses problems regarding the efficiency of video codecs, which are highly optimized on natural videos. These typically exhibit high levels of correlation within and between frames, which are exploited by video codecs. In the following, the feature-planes are analyzed regarding video codec suitability and multiple approaches to increase the compression-rate are implemented and evaluated. Furthermore, the decoding pipeline will be implemented and optimized on a standalone #abbr.a[HMD] using Unity3D to demonstrate its applicability @unity6.
// Multiple techniques are implemented and evaluated both indivudually and combined that aim at improving compression performance.
// 1. Using yuv444 instead of 3 grayscale videos
//   - no effect except easier handling
// 2. 8bit vs 16bit
//   - no effect except faster decoding with HW acceleration
// 3. varying plane resolution
// 4. varying channel sizes (\#frames)
// 5. Intra-frame smoothing
//   - enables better compression/quality trade off in high compression areas
// 6. inter-frame smoothing
//   - same as above

== Baseline for all experiments(check)
<sec:method-baseline>
to have a proper baseline we use the exact same configuration for all scenes while in the paper individual configs where used

== Analyzing CodecGS feature-planes
<sec:suitability>
CodecGS as published in #cite(<lee2025compression3dgaussiansplatting>, form: "year") compresses a #abbr.a[3DGS] scene into three videos with each 32 monochrome frames of 512x512 16bit pixels containing the feature planes. Additionally, the positions of the gaussians are stored in a lossless binary file and for the #abbr.pla[MLP] the .pth format is used. The sum of the sizes of all those files makes up the total size of the compressed scene. In the following, the feature-plane videos will be analyzed and optimized for size and decoding time.

CodecGS uses four k-planes models, which respectively predict color, scale, rotation and opacity @lee2025compression3dgaussiansplatting. The values of the corresponding planes of each k-planes model are concatenated, resulting in three videos. The number of channels of the k-planes models can be varied, meaning, that if the k-planes model for color is queried with one position, it returns not only 3 values, corresponding to XY, XZ and YZ, but a multiple of 3. With 8 channels, querying the color model returns $ 3 * 8=24$ values per position. This is referred to as channel-size. Originally, all k-planes models use a channel-size of 8, resulting in 32 frames for each of the three plane video. See @fig:kplanes-to-video for the detailed concatenation process.

#figure(
  placement: auto,
  image("figures/kplanes_channel_frame_explanation.png"),
  caption: [*CodecGS plane video structure.* CodecGS uses four K-Planes models from which Color and SH, Rotation, Scale and Opacity are predicted. Every Plane has multiple levels, called channels, 8 per default. These channels are concatenated to form 3 videos. The first video contains all XY planes, first all 8 channels from the color model's XY plane (yellow), then all 8 levels from the rotation model's XY plane (green) etc. The second video contains all planes of the XZ plane and the third video consists of the YZ planes. Since CodecGS uses 8 levels and fourn K-Plane models, each of the three videos contains 8x4=32 frames. In total CodecGS uses 96 video frames. Each frame has a resolution of 512x512 pixels and uses a precision of 16bit.],
)
<fig:kplanes-to-video>

Due to the non-differentiability of video codecs, they cannot be integrated in the loss function of gradient-descend based training. Thus, other methods that approximate the video codecs compression performance need to be used in the loss function of the training. The better the method predicts the compression ratio, the more effectively the planes are optimized for increasing the rate-distortion performance of the video codec. #cite(<lee2025compression3dgaussiansplatting>, form: "prose") introduce #abbr.a[DCT]-entropy modeling and progressive $cal(L)_1$-loss as a method to prepare the planes for more efficient compression (see @sec:codecgs).
Yet, subjectively, the feature-planes exhibit strong noise and flickering (see @fig:og-bike-planes), even though the overall structure stays similar, pixel values change unpredictably over time and space.
// Yet, #abbr.a[HEVC] reaches only a compression ratio of 6.24 averaged over
// the learned feature-planes of the Mip-NeRF360 Dataset with a #abbr.a[QP] of 1 @barron2022mipnerf360. Meanwhile with the same configuration, #abbr.a[HEVC] reaches a compression ratio of 10.71 on natural videos#footnote[Using parts of the SJTU HDR Video Sequence Dataset with 16bit natural videos scaled down to 512x512 and converted to monochrome @song2016Dataset16bit] (see @fig:compression-ratio-og-planes-hevc).

// #figure(
//   image("figures/compression-ratio-og-planes-hevc.png"),
//   caption: [Multiple codec configurations were tested regarding their compression ratio on the Mip-NeRF360 dataset, all with a QP value of 1. The original configuration is denoted as baseline. SHOULD SHOW ONLY BASELINE BUT ON DIFFERENT DATA (natural images, vs planes)]
// )
// <fig:compression-ratio-og-planes-hevc>
// // RAW 16bit video dataset: https://medialab.sjtu.edu.cn/post/sjtu-hdr-video-sequences/
#figure(
  placement: auto,
  image("figures/feature-planes-bicycle-og.png"),
  caption: [*CodecGS feature-planes of the bike scene.* Displayed are the first three channels of the XY plane of the trained bicycle scene's color k-planes model. The planes similarly structured noise. The enlargements depict co-located smooth areas. In the noisy regions, pixel values do not stay consistent over time (left to right).],
)
<fig:og-bike-planes>

The metrics $cal(L)_1$, #abbr.a[DCT]-Entropy, #abbr.a[TV] and Smoothed-to-original difference are compared on different video sets, one being the plane-videos of all trained scenes of the Mip-NeRF-360 Dataset and the other being a set of natural videos.

@fig:metric-comparison shows that #abbr.a[DCT]-Entropy, #abbr.a[TV] and Smoothed-to-Original Difference are strongly negatively correlated with the compression ratio achieved by #abbr.a[HEVC]. This can be interpreted as these metrics can be used to predict the compression ratio of a video and used as a loss function for training planes that will as a result be more efficiently compressed. Thus, #abbr.a[TV] and Smoothed-to-Original Difference are candidates for a loss term leading to a higher compression ratio.

#figure(
  placement: auto,
  image("figures/scatter_plots_top_metrics.png", width: 100%),
  caption: [*Correlation of metrics and compression ratio.* Correlating DCT-Entropy, Intra-TV and Smoothed-to-Original Difference with the actual compression ratio of a set of 18 16bit grayscale video files.],
)
<fig:metric-comparison>

// _Hypothesis: Entropy Modeling is good for very messy videos, but on already smoothed videos it works less good. That is why entropy modeling does not succeed in Shrinking size much beyond a certain threshold .
// There should be a correlation between smoothness value (whatever measure) and the compression prediction performance of Entropy Modeling. TV should be better, the smoother the videos are._


The achieved compression ratio depends on the features of the input video and the codec with its configuration. The input contains learned parameters that are used as inputs for #abbr.pla[MLP] to predict the gaussian attributes. Altering the values of the feature-planes directly influences the reconstruction quality. Slight changes introduced by lossy compression are tolerable @lee2025compression3dgaussiansplatting. Completely reordering the pixel values, like done by #cite(<sog2025>, form: "prose"), is not feasible. Thus, in the following multiple approaches to improve compression are evaluated. The planes values will be optimized to yield better compression through smoothness modeling and the video codecs configurations will be evaluated as well as the shape of the planes and the plane videos.

// === Optimizing Codec Capacity
// <sec:bit-depth>
// - all following option parameters will be optimized using grid-search ?!
//   - because they all determine the capacity and file size and thus can be optimized jointly

// - *bit depth*
//   - how to cut off values or squeeze them into smaller bits
//   - codec supports 8/10/12/16 bits (HM but not ffmpeg)
//     - ffmpeg libx265 uses max 12 bit internally (so 16 bit values are quantized anyways)
//     - ffmpeg libx265 pix_fmt gray12le, gray10le, gray, or use yuv420 with format=gray
//     - HM-Rext can use 16bit (extended Range)
//     - HM-Rext has input, internal and output bitdepth -> set to 8,10,12,16
//       - using profiles monochrome/-12/-16 from @flynn2016HEVC
//     - try vvc, h264 or other ?!?!
//   - HM Rext @flynn2016HEVC
//   - As experiments show, changing the major settings of the HM video encoder has a relatively small effect on the compression performance. All feature planes of the bonsai, garden, stump and bicycle scene where encoded using the baseline configuration as used by @lee2025compression3dgaussiansplatting. Intra frame coding only, monochrome16, monochrome8, monochrome12, no-random-access and different variations of inter frame coding patterns, all with a QP value of 1, where compared to the baseline configuration. The comppression ratio for all settings is close to 26%. If we compare this to encoding a natural 16bit video sequence with the same settings, the compression ratio is arround XXXHIGHER% (Graphics)
//   - This can be explained by the higher entropy of the planes. Since MLP weights are learned and safed into a 2d grid without any spatial relation, the planes are close to random images. The main mechanisms of a video codec, spatial and temporal prediction, do not work on random data without any correlation. Thus, changing settings of the video codec does not make a noticeable difference.
//   - In order to improve the compression ratio, the data in the planes needs to be rearranged in a way that reduces inter and intra frame entropy.

// - The following ideas wont meaningfully change the entropy so we need to think about this first.
//   - *Plane resolution*
//   - *Tiling* multiple frames into one
//   - *Number of Frames*
//   - *QP Values*
//     - determines loss

// === Minimize Temporal Entropy
// <sec:temporal-entropy>
// - temporal entropy term in loss function
// - reordering?
// - create blocks and sort/map  with a tree or so?

// === Minimize Entropy in the Frequency Domain
// <sec:dct-entropy>
// - Entropy is minimized in the frequency domain because video codecs work mainly in frequency domain
// - current block is 4x4
//   - lets try larger blocks
//   - need to adapt entropy loss term 1e-9 -> 1e-8, 1e-7
// === Transform Planes to reduce Entropy
// <sec:smoothing>
// - planes are really noisy
//   - hard to find spatial or temporal similarities for the codec
//   - "no correlation to be reduced"
// - smooth reversibly
//   - Burrows Wheeler Transform (BWT)
//     - in snake wise order
//     - in Hilbert order
//     - blocks
//     - little extra data
//     - works on images (paper)
//   - Move with Interleaving (MWI)
//     -
// - do we need the order of mlp parameters or are they irrelevant partly due to fully connected layers?
//   - if so, can we sort them and not reversibly
// - Find a way to use SOG technology
//   -  maybe as an optimization term (smoothness?)
//   - problem: They don't care about reversibility
//     - just pierce through all planes at one index and get all data for one gaussian
//     - all planes are sorted in the same way
== Smoothness modeling
<sec:smoothness-modeling>
As an extension to the #abbr.a[DCT]-entropy modeling, smoothness modeling uses $cal(L)_"TV"$ (see @eq:tv) to encourage less noisy planes and stronger temporal consistency. For spatial smoothness, the $cal(L)_"TV"$ calculated for every plane individually is be added to the overall loss function as $cal(L)_"intraTV"$ (see @fig:intra-smooth). Assume a sequence of $n$ plane images $P = p_0, p_1, ... , p_n$ ordered like they are concatenated in the plane video.


$ cal(L)_"IntraTV" (P) = sum_(p in P) cal(L)_"TV" (p) $
<IntraTV>

Additionally, $cal(L)_"TV"$ is adjusted slightly, such that  it captures temporal smoothness. This is done in the order of concatenation of the planes as described in @fig:kplanes-to-video and takes into account the video codecs coding structure. It will be denoted as $cal(L)_"interTV"$. For every pixel position $x,y$, we calculate a one dimensional $cal(L)_"TV"$ on the co-located temporally ordered pixel values (see @fig:inter-smooth). 


$ cal(L)_"interTV" (P) = sum_"i,j" sum_(k in [0..(n-2)]) sqrt(|p(k)_"i,j" - p(k+1)_"i,j"|^2 ) $
<InterTV>

#subfigure(
  figure(
    image("figures/intra-smoothing.png"),
    caption: []
  ),<fig:intra-smooth>,
  figure(
    image("figures/inter-smoothing.png"),
    caption: []
  ),<fig:inter-smooth>,
  columns: (1fr,1fr),
  placement: auto,
  caption: [*Explanation of inter and intraframe smoothing.* #abbr.a[TV]-loss is adopted to penalize spatial and temporal variation by measuring the #abbr.a[TV] (a) along rows and columns of one frame and (b) along the co-located values of all planes in the order they will be concatenated and coded.],
)
<fig:intrainterexplanation>

The channels of the same plane show structural similarities and are therefore grouped and concatenated (see @fig:kplanes-to-video). The 
$cal(L)_"CSInterTV"$ is taking this into account by not applying #abbr.a[TV] across channels stemming from different planes. Thus, it will be calculated as follows, using the channel-sizes $C_"color"$, $C_"scale"$, $C_"rotation"$ and  $C_"opacity"$, specifying the channel-sizes for the k-planes models predicting color, scale, rotation and opacity, respectively. Let $P_"XY"$ be the list of all XY plane's channels for all 4 k-planes models, this contains the planes forming the first plane video:
$ cal(L)_"CSInterTV" (P_"XY") &= cal(L)_"interTV" (P_"XY" (0...(C_"color"-1)) \
&+ cal(L)_"interTV" (P_"XY" (C_"color"...(C_"cs" -1)) \
&+ cal(L)_"interTV" (P_"XY" (C_"cs" ...(C_"csr" -1) ) \ 
&+ cal(L)_"interTV" (P_"XY" (C_"csr"... C_"csro" -1) \
text("where"), \
C_"cs" &= C_"color" + C_"scale" \
C_"csr" &= C_"color" + C_"scale" + C_"rotation" \ 
C_"csro" &= C_"color" + C_"scale" + C_"rotation" + C_"opacity"
$
<eq:>
// The Smoothed-to-Original Difference loss $cal(L)_"SOD"$ applies a gaussian blur $G(y)$ to the input image $y$ and calculates the absolute pixel difference between the blurred and the original image.
// $ cal(L)_"SOD" (Y) = sum_(y in Y) |G(y) - y| $

The $cal(L)_"CSInterTV"$ and the $cal(L)_"IntraTV"$ will be evaluated independently as well as combined. The strength controlled by the respective weight $lambda$ of these additional loss terms is increased progressively every 5000 iterations starting from the iteration 30000 until being fully applied in iteration 50000. The rate-distortion trade-off is compared to #abbr.a[DCT]-entropy modeling and $cal(L)_1$ loss introduce by CodecGS @kerbl3Dgaussians @lee2025compression3dgaussiansplatting.
The complete loss function is based on the original #abbr.a[3DGS] rendering loss $cal(L)_"render"$ and the $cal(L)_"ent"$ and $cal(L)_"1"$:

$ cal(L) = cal(L)_"render" + lambda cal(L)_"ent" + lambda_"1"cal(L)_"1" + lambda_"CSInterTV"cal(L)_"CSInterTV" + lambda_"IntraTV"cal(L)_"IntraTV" $
== Evaluating video-codec configurations/coding structure
<sec:config-evaluation>
As described in @sec:codecs video codecs use intra and interframe coding to reduce redundancy and irrelevant information. These mechanisms yield stronger compression, the stronger the temporal and spatial correlation and the more loss is allowed by the configuration. The baseline configuration used by #cite(<lee2025compression3dgaussiansplatting>, form: "prose") specify a #abbr.a[QP] of 1, a standard coding structure using a #abbr.add("GOP", "Group of Pictures") #abbr.a[GOP] of 16 and random access. All frames are type B and a complex hierarchical frame order is used. The #abbr.a[QP]-offset is set to 0 for all frames. The three plane videos are encoded as individual 16bit monochrome videos. This baseline was compared to a configurations that aligns with the plane concatenation pattern. These configurations were evaluated on the CodecGS plane videos of the Mip-NeRF360 dataset.

As shown in @fig:frameConcatAnalysis the plane videos frames similarity is not evenly distributed but clustered in blocks of 8 frames. This corresponds to the concatenation pattern (see @fig:kplanes-to-video), where all 8 channels of one plane model are grouped together. Based on this, a coding pattern with a #abbr.a[GOP] guided by the channel-size is suggested. This also aligns with the proposed interframe smoothness loss (see @sec:smoothness-modeling).

DESCRIBE CODING PATTERN FROM YAGO (prob gop8, random accessmaybe, maybe Qp offset?)

#figure(
  image("figures/garden_plane_concat_sim_psnr.png", width: 70%),
  placement: auto,
  caption:[*Frame similarity heatmap.* This heatmap shows the similarity in PSNR between all frames of the first plane-video of the codecGS bicycle scene. Clearly visible is a 8 by 8 block structure. This is caused by the concatenation pattern, see @fig:kplanes-to-video.  
]
)
<fig:frameConcatAnalysis>


// #figure(
//   placement: auto,
//   table(
//     stroke: none,
//     columns: 5,
//     inset: 5pt,
//     table.hline(),
//     table.header([], [*Bit-Depth*], [*RA*], [*GOP*], [*Profile*]),
//     table.hline(),
//     [Baseline], table.vline(), [16], [y], [16], [Main-RExt],
//     [monochrome8], [8], [y], [16], [monochrome8],
//     [monochrome10], [10], [y], [16], [monochrome10],
//     [monochrome12], [12], [y], [16], [monochrome12],
//     [monochrome16], [16], [y], [16], [monochrome16],
//     [Intra-Only], [16], [y], [1], [Main-RExt],
//     [No-RA], [16], [n], [16], [Main-RExt],
//     table.hline(),
//   ),
//   caption: [*Video codec configurations.* HEVC configurations comparison showing only the differences from the baseline. RA denotes Random Access setting of the video codec.],
// ) <tab:hevc-config>


== Enable hardware decoding using 8bit or 10bit
<sec:bitdepth>

CodecGS uses three 16bit #abbr.a[HEVC] grayscale videos planes with 32 frames each. 16bit decoders are rarely used and especially not implemented in common hardware. To be able to speed up the decoding time greatly, hardware acceleration is necessary. 
Before compression, the plane values are converted to integers and the range supported by 8bit and 10bit is only 255 and 1023 respectively, compared to 65535 for 16bit. 


== Optimizing plane resolution and channel-size
<sec:yuv444>
Due to time and computational constraints, an exhaustive optimization of resolution and the four channel sizes is not feasible. Instead, channel size was decreased for one channel at a time and only the default plane resolution. Based on the results, different configurations where tested. Additionally, the plane resolution was varied.


// == TBC Compressing Positions
// <sec:methods-gpcc>
// - positions only compressed separately using brotli
// - use point cloud compression g-pcc instead
// - Maybe use quantization and put positions into video frames
//   - norm11
//   - norm16x3

== Feature-plane reordering
<sec:reordering>

As shown in @fig:kplanes-to-video, the feature planes are concatenated to form the three plane videos. The order of the planes is not relevant for the reconstruction quality, but influences the compression performance. The order of the planes can be optimized to reduce intra and inter frame entropy. Originally, the planes for color, scale, rotation and opacity are concatenated in that order. An interleaved order was tested, where the first plane of each kplanes model is concatenated, followed by the second plane of each kplanes model and so on. This interleaved order was compared to the original order regarding compression performance and decoding time.

== Using the color channels
<sec:yuv-monochrome>

CodecGS produces three gray scale videos, each of which has to be coded and decoded. To simplify this process, an extension enabling the usage of one YUV444#footnote[YUV444 is a color encoding format, where color pixels are not encoded as red, green and blue values (RGB) but separated into brightness and color information. 444 defines that the color information is not subsampled, as done in YUV422 @PalYUV1967.] video was implemented and analyzed in terms of file size, reconstruction quality and decoding speed.


== Compaction using mini splatting
<sec:compaction>

The previously described methods all aim at reducing the final size without reducing the number of gaussian primitives. The densification is the same as published by #cite(<kerbl3Dgaussians>, form: "prose"). Mini splatting @minisplatting2024 is used to evaluate the presented methods on a highly compacted point cloud. Mini-splatting is run for 30000 iterations and its point cloud used to initialize the feature-plane training of CodecGS. The feature planes where trained for 40000 iterations, resulting in a training of 70000 iterations in total.

= Results
<chap:results>
- Could improve overall xxxx%
- found that with those noisy highly uncorrelated planes, video codec can barely optimize. Any codec settings result in way larger files than compared to a natural video. (maybe only because of bit depth?)
-
- we need to change the data so that video codec can actually work


== Smoothness Modeling

Experiments suggest a multiplication factor of 1e-9 to 1e-8 to stay within reasonable quality loss of -0.15 dB PSNR (see @tab:variationloss).
The smoothing pattern was aligned to a CODING STRUCTURE OF THE VIDEO CODEC and uses. The intra and inter smoothing is clearly visible as depicted in GRAPHIC ERSTELLEN (show cutouts of different intra smoothing levels and show stripes of consecutive frames with different inter smoothing levels)
#figure(
  placement: auto,
  table(
    stroke: none,
    columns: 5,
    inset: 5pt,
    table.hline(),
    table.header([*TV-Loss Term*], [*Bonsai*], [*Garden*], [*Stump*], [*Bonsai*]),
    table.hline(),
    [1e-7], [110], [110], [110], [110],
    [1e-8], [120], [110], [110], [110],
    [1e-9], [145], [110], [110], [110],
    [1e-10], [120], [110], [110], [110],
    table.hline(),
  ),
  caption: [*NO REAL DATA YET.* PSNR for multiple scenes with different #abbr.a[TV] loss terms.],
) <tab:variationloss>


#figure(
  placement: auto,
  image("figures/InterIntraTVLossZoom.png", width: 100%),
  caption: [*Intra and interframe smoothing.* Comparing Intra and Inter Frame TV-Loss],
)
<fig:smoothnessComparison>

#figure(
  placement: auto,
  image("figures/intra_tiles.png", width: 100%),
  caption: [*Smoothed feature-planes.* Visual comparison of increasing $cal(L)_"IntraTV"$ values for the first channel of the garden scne's first color plane. The color values of the frames are centered around zero, no normalization is applied. The overall structure of the channel is preserved while the noise like structure is reduced in most areas.],
)
<fig:smoothnessComparison>

#figure(
  placement: auto,
  table(
    stroke: none,
    columns: 7,
    inset: 5pt,
    table.hline(),
    table.header([$cal(L)_"IntraTV"$], [$cal(L)_"InterTV"$], [*CS*], [*PS*], [*8Bit*], table.vline(), [*PSNR*], [*Size*(MB)]),
    table.hline(),
    [], [], [], [], [], [26.57], [17.43],
    [$checkmark$], [], [], [], [], [26.47], [14.31],
    [$checkmark$], [$checkmark$], [], [], [], [26.45], [12.03],
    [$checkmark$], [$checkmark$], [$checkmark$], [], [], [26.44], [10.90],
    [$checkmark$], [$checkmark$], [$checkmark$], [$checkmark$], [], [26.47], [10.82],
    [$checkmark$], [$checkmark$], [$checkmark$], [$checkmark$], [$checkmark$], [26.25], [10.92],
    table.hline(),
  ),
  caption: [*Ablation Study of proposed methods.* showing the individual effect of the proposed methods on the Mip-NeRF-360 dataset.],
) <tab:ablation>

== Feature-Plane Characteristics

Resolution and channel-size ... (see @fig:res-channel-result)
#figure(
  placement: auto,
  image("figures/resolution_channel_size_comp.png", width: 100%),
  caption: [*Channel size analysis.* Comparing effect of different channel-size configurations and plane resolutions averaged over the Mip-NeRF-360 Dataset.],
)
<fig:res-channel-result>

YUV444 does not influence size nor quality (see @fig:color-yuv-result) . Decompression TIME??
#figure(
  placement: auto,
  image("figures/color_mode_comparison_qp1.png", width: 100%),
  caption: [*YUV444 and monochrome video.* The effect of using a single YUV444 video instead of three monochrome videos is minimal.],
)
<fig:color-yuv-result>
When encoding the feature planes using video encoders, the resulting file sizes exhibit counter-intuitive behavior with respect to bit depth. In most cases, the 8bit coded files are larger than 10,12 or 16bit. Since the planes show little temporal or spatial similarities and high entropy, they prevent the video codec from exploiting just those traits. When at the same time setting the quantization parameter to 1, so that the codec cannot remove high frequency or noise-like data, the codec approaches the theoretical compression limit given by the Shannon Entropy Limit of the source data. The mechanisms of video codecs are not beneficial for this kind of data and are optimized for natural highly correlated images and videos.

Still the benefit of faster decoding due to the 8-/10-bit decoders being implemented in most modern hardware remains.

== Implementing the CodecGS decoding pipeline in Unity3D
<sec:unity-implementation>
To demonstrate the applicability of CodecGS, the decoding pipeline was re-implemented using Unity3D (see @fig:unityDecoding). It is not highly optimized for speed or memory usage. As a target device the Meta Quest 3 was selected, due to its accessible android based platform and relatively large user base.
The hardware video decoder is used to decode the planes values, which are then further processed on the #abbr.add("GPU", "Graphics Processing Unit") #abbr.a[GPU]. The Unity Sentis package provides a fast inference engine @unity6Sentis. Additionally, it can be used to build simple models with the most common operations provided by common frameworks like Pytorch.

#figure(
  image("figures/UnityOverview.png"),
  placement: auto,
  caption: [*CodecGS decoding in Unity3D.* An overview of the decoding pipeline implemented in Unity3d. The planes and positions are decompressed and processed using the Unity Sentis package, that runs the #abbr.pla[MLP]. Then, the #abbr.a[3DGS] scene is rendered using a customized renderer.],
)
<fig:unityDecoding>
The data needed for the planes of the different attributes color, rotation, scale and opacity is distributed in the different plane videos as shown in @fig:kplanes-to-video. The needed operations are implemented in a Unity Sentis model that runs on the #abbr.a[GPU].

The reconstructed attribute planes are passed to another Unity Sentis Model that combines the kplanes model and attribute #abbr.pla[MLP] into one runnable model. Inside, the planes data are passed to each attribute specific kplanes model and the resulting output is passed to the attribute #abbr.a[MLP]. Finally, the data normalization is reverted.

Now, the data is back in a point-wise .PLY-file-like shape and can be rendered by any #abbr.a[3DGS] renderer. Unity does not provide a gaussian renderer, thus we used a 3rd party renderer implemented by #cite(<unityGaussianSplatting>, form: "prose"). This was adapted to run on an #abbr.a[HMD] rendering two slightly inwards rotated images.

The hardware limitations of the Quest 3 allow only for rendering small objects not occupying the full display area and not exceeding 50,000 - 200,000 guassian primitives. Decoding the Lego scene displayed in @fig:unityDecoding takes rougly 3 seconds.

= Discussion
<chap:discussion>
The presented methods efficiently increase the compression performance of CodecGS while maintaining comparable reconstruction quality.

Additionally, these methods clearly improve the performance in a low bandwidth scenario, by drastically reducing the plane size by an average factor of XX times, while still reaching acceptable reconstruction. This especially interesting for streaming and other bandwidth constrained use cases.

The implementation of the decoding pipeline on a low-end mobile device further demonstrates the practicability of CodecGS as a #abbr.a[3DGS] compression method for low and high quality applications, despite of not being extensively optimized.

COMPARE TO STATE OF THE ART APPROACHES HAC++, ContextGS, MPEG BYTEDANCE APPROACH etc

MAYBE STANDARDIZATION


// = Using the Template
// <chap:usage>
// == Thesis Setup
// <sec:setup>
// The document class is initialized by calling
// ```typst #show: nifty-ntnu-thesis.with()``` at the beginning of your `.typ` file. Currently it only supports english. The `nifty-ntnu-thesis` function has a number of options you can set, most of which will be described in this document. The rest will be documented in this templates repository.

// The titlepage at the beginning of this document is a placeholder to be used when writing
// the thesis. This should be removed before handing in the thesis, by settting `titlepage: false`.
// Instead the official NTNU titlepage for the corresponding thesis type
// should be added as described on Innsida.#footnote[see #link("https://innsida.ntnu.no/wiki/-/wiki/English/Finalizing+the+bachelor+and+master+thesis") for bachelor and master, and #link("https://innsida.ntnu.no/wiki/-/wiki/English/Printing+your+thesis")
// for PhD.]

// == Title, Author, and Date
// <title-author-and-date>
// The title of your thesis should be set by changing the `title` parameter of the template. The title will appear on the titlepage as well as in the running header of the even numbered pages. If the title is too long for the header, you can use `short-title` to set a version for the header.

// The authors should be listed with full names in the `authors` parameter. This is an array, with multiple authors separated by a comma. As with the title, you can use `short-author` to set a version for the header.

// Use `date` to set the date of the document. It will only appear on
// the temporary title page. To keep track of temporary versions, it can be
// a good idea to use `date: datetime.today()` while working on the thesis.

// == Page Layout
// <page-layout>
// The document class is designed to work with twosided printing. This
// means that all chapters start on odd (right hand) pages, and that blank
// pages are inserted where needed to make sure this happens. However,
// since the theses are very often read on displays, the margins are kept
// the same on even and odd pages in order to avoid that the page is
// jumping back and forth upon reading.

// By default this is turned off. You can turn it on by setting
// `chapters-on-odd: false` at the top of the file.

// == Structuring Elements
// <structuring-elements>
// The standard typst headings are supported, and are set using =.

// === This is a level 3 heading
// <this-is-a-subsection>

// ==== This is level 4 heading
// <this-is-a-subsubsection>

// ===== This is a level 5 heading
// <this-is-a-paragraph>

// Headings up to level 3 will be included in the table of
// contents, whereas the lower level structuring elements will not appear
// there. Don’t use too many levels of headings; how many are appropriate,
// will depend on the size of the document. Also, don’t use headings too
// frequently.

// Make sure that the chapter and section headings are correctly
// capitalised depending on the language of the thesis, e.g.,
// '#emph[Correct Capitalisation of Titles in English];' vs. '#emph[Korrekt
// staving av titler på norsk];'.

// Simple paragraphs are the lowest structuring elements and should be used
// the most. They are made by leaving one (or more) blank line(s) in the
// `.typ` file. In the typeset document they will appear indented and with
// no vertical space between them.

// == Lists
// <lists>
// Numbered and unnumbered lists are used just as in regular typst, but are typeset
// somewhat more densely and with other labels. Unnumbered list:

// - first item

// - second item

//   - first subitem

//   - second subitem

//     - first subsubitem

//     - second subsubitem

// - last item

// Numbered list:

// + first item

// + second item

//   + first subitem

//   + second subitem

//     + first subsubitem

//     + second subsubitem

// + last item


// == Figures
// <figures>
// Figures are added using ```typst #figure()```. An example is shown in
// #link(<fig:mapNTNU>)[2.1];. By default figures are placed in the flow, exactly where it was specified. To change this set the ```placement``` option to either `top`, `bottom`, or `auto`. To add an image, use ```typst #image()``` and set the `height` or `width` to include the graphics file. If the caption consists of a single sentence fragment (incomplete sentence), it should not be punctuated.


// #figure(image("figures/kart_student.png", width: 50%),
// caption: [
//     The map shows the three main campuses of NTNU.
//   ]
// )
// <fig:mapNTNU>

// For figures compsed of several sub-figures, the `subpar` module has been used. See #link(<fig:subfig>)[2.4]
// with #link(<sfig:a>)[\[sfig:a\]] for an example.

// #subfigure(
//   figure(image("figures/kart_student.png", width: 100%),
//     caption: [First sub-figure]), <sfig:a>,
//   figure(image("figures/kart_student.png", width: 100%),
//     caption: [Second sub-figure]), <sfig:b>,
//     columns: (1fr, 1fr),
//    caption: [A figure composed of two sub-figures. It has a long caption in order to demonstrate how that is typeset.
//   ],
// <fig:subfig>

// )

// == Tables
// <tables>
// Tables are added using ```typst #table()```, wrapped in a ```typst #figure()``` to allow referencing. An example is given in
// @tab:example1. If the caption consists
// of a single sentence fragment (incomplete sentence), it should not be
// punctuated.


// #figure(
//   table(
//     stroke: none,
//     columns: 2,
//     table.hline(),
//     table.header([*age*], [*IQ*],),
//     table.hline(),
//     [10], [110],
//     [20], [120],
//     [30], [145],
//     [40], [120],
//     [50], [100],
//     table.hline(),
//   ), caption: [A simple, manually formatted example table]
//   ) <tab:example1>
// Tables can also be automatically generated from CSV files #footnote(link("https://typst.app/docs/reference/data-loading/csv/")).

// == Listings
// <listings>
// Code listings are are also wrapped in a ```typst #figure()```. Code listings are defined by using three ``` `backticks` ```. The programming language can also be provided. See the typst documentation for details. The
// code is set with the monospace font, and the font size is reduced to
// allow for code lines up to at least 60 characters without causing line
// breaks. If the caption consists of a single sentence
// fragment (incomplete sentence), it should not be punctuated.

// #figure(caption: "Python code in typst")[
// ```python
// import numpy as np
// import matplotlib.pyplot as plt

// x = np.linspace(0, 1)
// y = np.sin(2 * np.pi * x)

// plt.plot(x, y)
// plt.show()
// ```
// ]<lst:python>
// #figure(caption: "C++ code in typst")[
// ```cpp
// #include <iostream>
// using namespace std;

// int main()
// {
//   cout << "Hello, World!" << endl;
//   return 0;
// }
// ```]<lst:cpp>


// == Equations
// <equations>
// Equations are typeset as normally in typst. It is common to consider
// equations part of the surrounding sentences, and include punctuation in
// the equations accordingly, e.g.,
// $ f (x) = integral_1^x 1 / y thin d y = ln x thin . $ <logarithm>
// For more advanced symbols like, e.g., $grad, pdv(x,y)$, the `physica` module is preloaded.
// As you can see, the simple math syntax makes typst very easy to use.
// == Fonts
// <fonts>
// Charter at 11pt with the has been selected as the main font for the thesis template. For code examples, the monospaced font should be used – for this, a scaled
// version of the DejaVu Sans Mono to match the main font is preselected.

// == Cross References
// <sec:crossref>
// Cross references are inserted using `=` in typst. For examples on usage, see @sec:crossref in @chap:usage, @tab:example1
// @fig:mapNTNU, @logarithm,
// @lst:cpp and #link(<app:additional>)[Appendix A];.



// == Bibliography
// <bibliography>
// The bibliography is typeset as in standard typst. It is added in the initializing function as such: ```typst bibliography: bibliography("thesis.bib")```.
// With this setup, using `@` will give a number
// only~@landes1951scrutiny, and ```typst #cite(, form: "prose") ``` will give author and number like this: #cite(<landes1951scrutiny>, form: "prose");.


// == Appendices
// <appendices>
// Additional material that does not fit in the main thesis but may still
// be relevant to share, e.g., raw data from experiments and surveys, code
// listings, additional plots, pre-project reports, project agreements,
// contracts, logs etc., can be put in appendices. Simply issue the command
// ```typst #show: appendix``` in the main `.typst` file, and then the following chapters become appendices. See #link(<app:additional>)[Appendix A]
// for an example.

// = Thesis Structure
// <thesis-structure>
// The following is lifted more or less directly from the original template.

// The structure of the thesis, i.e., which chapters and other document
// elements that should be included, depends on several factors such as the
// study level (bachelor, master, PhD), the type of project it describes
// (development, research, investigation, consulting), and the diversity
// (narrow, broad). Thus, there are no exact rules for how to do it, so
// whatever follows should be taken as guidelines only.

// A thesis, like any book or report, can typically be divided into three
// parts: front matter, body matter, and back matter. Of these, the body
// matter is by far the most important one, and also the one that varies
// the most between thesis types.

// == Front Matter
// <sec:frontmatter>
// The front matter is everything that comes before the main part of the
// thesis. It is common to use roman page numbers for this part to indicate
// this. The minimum required front matter consists of a title page,
// abstract(s), and a table of contents. A more complete front matter, in a
// typical order, is as follows.

// / Title page\:: #block[
// The title page should, at minimum, include the thesis title, authors and
// a date. A more complete title page would also include the name of the
// study programme, and possibly the thesis supervisor(s). See
// #link(<sec:setup>)[2.1];.
// ]

// / Abstracts\:: #block[
// The abstract should be an extremely condensed version of the thesis.
// Think one sentence with the main message from each of the chapters of
// the body matter as a starting point.
// #cite(<landes1951scrutiny>, form: "prose") have given some very nice
// instructions on how to write a good abstract. A thesis from a Norwegian
// Univeristy should contain abstracts in both Norwegian and English
// irrespectively of the thesis language (typically with the thesis
// language coming first).
// ]

// / Dedication\:: #block[
// If you wish to dedicate the thesis to someone (increasingly common with
// increasing study level), you may add a separate page with a dedication
// here. Since a dedication is a personal statement, no template is given.
// Design it according to your preference.
// ]

// / Acknowledgements\:: #block[
// If there is someone who deserves a 'thank you', you may add
// acknowledgements here. If so, make it an unnumbered chapter.
// ]

// / Table of contents\:: #block[
// A table of contents should always be present in a document at the size
// of a thesis. It is generated automatically using the `outline()`
// command. The one generated by this document class also contains the
// front matter and unnumbered chapters.
// ]

// / List of figures\:: #block[
// If the thesis contains many figures that the reader might want to refer
// back to, a list of figures can be included here. It is generated using
// `outline()`.
// ]

// / List of tables\:: #block[
// If the thesis contains many tables that the reader might want to refer
// back to, a list of tables can be included here. It is generated using
// `outline()`.
// ]

// / List of code listings\:: #block[
// If the thesis contains many code listings that the reader might want to
// refer back to, a list of code listings can be included here. It is
// generated using `outline()`.
// ]

// / Other lists\:: #block[
// If there are other list you would like to include, this would be a good
// place. Examples could be lists of definitions, theorems, nomenclature,
// abbreviations, glossary etc.
// ]

// / Preface or Foreword\:: #block[
// A preface or foreword is a good place to make other personal statements
// that do not fit whithin the body matter. This could be information about
// the circumstances of the thesis, your motivation for choosing it, or
// possibly information about an employer or an external company for which
// it has been written. Add this in the initializing function of this template.
// ]

// == Body Matter
// <body-matter>
// The body matter consists of the main chapters of the thesis. It starts
// the Arabic page numbering with page~1. There is a great diversity in the
// structure chosen for different thesis types. Common to almost all is
// that the first chapter is an introduction, and that the last one is a
// conclusion followed by the bibliography.

// === Development Project
// <sec:development>
// For many bachelor and some master projects in computer science, the main
// task is to develop something, typically a software prototype, for an
// 'employer' (e.g., an external company or a research group). A thesis
// describing such a project is typically structured as a software
// development report whith more or less the following chapters:

// / Introduction\:: #block[
// The introduction of the thesis should take the reader all the way from
// the big picture and context of the project to the concrete task that has
// been solved in the thesis. A nice skeleton for a good introduction was
// given by #cite(<claerbout1991scrutiny>, form: "prose");:
// #emph[review–claim–agenda];. In the review part, the background of the
// project is covered. This leads up to your claim, which is typically that
// some entity (software, device) or knowledge (research questions) is
// missing and sorely needed. The agenda part briefly summarises how your
// thesis contributes.
// ]

// / Requirements\:: #block[
// The requirements chapter should lead up to a concrete description of
// both the functional and non-functional requirements for whatever is to
// be developed at both a high level (use cases) and lower levels (low
// level use cases, requirements). If a classical waterfall development
// process is followed, this chapter is the product of the requirement
// phase. If a more agile model like, e.g., SCRUM is followed, the
// requirements will appear through the project as, e.g., the user stories
// developed in the sprint planning meetings.
// ]

// / Technical design\:: #block[
// The technical design chapter describes the big picture of the chosen
// solution. For a software development project, this would typically
// contain the system arcitechture (client-server, cloud, databases,
// networking, services etc.); both how it was solved, and, more
// importantly, why this architecture was chosen.
// ]

// / Development Process\:: #block[
// In this chapter, you should describe the process that was followed. It
// should cover the process model, why it was chosen, and how it was
// implemented, including tools for project management, documentation etc.
// Depending on how you write the other chapters, there may be good reasons
// to place this chapters somewhere else in the thesis.
// ]

// / Implementation\:: #block[
// Here you should describe the more technical details of the solution.
// Which tools were used (programming languages, libraries, IDEs, APIs,
// frameworks, etc.). It is a good idea to give some code examples. If
// class diagrams, database models etc. were not presented in the technical
// design chapter, they can be included here.
// ]

// / Deployment\:: #block[
// This chapter should describe how your solution can be deployed on the
// employer’s system. It should include technical details on how to set it
// up, as well as discussions on choices made concerning scalability,
// maintenance, etc.
// ]

// / Testing and user feedback\:: #block[
// This chapter should describe how the system was tested during and after
// development. This would cover everything from unit testing to user
// testing; black-box vs. white-box; how it was done, what was learned from
// the testing, and what impact it had on the product and process.
// ]

// / Discussion\:: #block[
// Here you should discuss all aspect of your thesis and project. How did
// the process work? Which choices did you make, and what did you learn
// from it? What were the pros and cons? What would you have done
// differently if you were to undertake the same project over again, both
// in terms of process and product? What are the societal consequences of
// your work?
// ]

// / Conclusion\:: #block[
// The conclusion chapter is usually quite short – a paragraph or two –
// mainly summarising what was achieved in the project. It should answer
// the #emph[claim] part of the introduction. It should also say something
// about what comes next ('future work').
// ]

// / Bibliography\:: #block[
// The bibliography should be a list of quality-assured peer-reviewed
// published material that you have used throughout the work with your
// thesis. All items in the bibliography should be referenced in the text.
// The references should be correctly formatted depending on their type
// (book, journal article, conference publication, thesis etc.). The bibliography should
// not contain links to arbitrary dynamic web pages where the content is
// subject to change at any point of time. Such links, if necessary, should
// rather be included as footnotes throughout the document. The main point
// of the bibliography is to back up your claims with quality-assured
// material that future readers will actually be able to retrieve years
// ahead.
// ]

// === Research Project
// <sec:resesarch>
// For many master and some bachelor projects in computer science, the main
// task is to gain knew knowledge about something. A thesis describing such
// a project is typically structed as an extended form of a scientific
// paper, following the so-called IMRaD (Introduction, Method, Results, and
// Discussion) model:

// / Introduction\:: #block[
// See #link(<sec:development>)[3.2.1];.
// ]

// / Background\:: #block[
// Research projects should always be based on previous research on the
// same and/or related topics. This should be described as a background to
// the thesis with adequate bibliographical references. If the material
// needed is too voluminous to fit nicely in the review part of the
// introduction, it can be presented in a separate background chapter.
// ]

// / Method\:: #block[
// The method chapter should describe in detail which activities you
// undertake to answer the research questions presented in the
// introduction, and why they were chosen. This includes detailed
// descriptions of experiments, surveys, computations, data analysis,
// statistical tests etc.
// ]

// / Results\:: #block[
// The results chapter should simply present the results of applying the
// methods presented in the method chapter without further ado. This
// chapter will typically contain many graphs, tables, etc. Sometimes it is
// natural to discuss the results as they are presented, combining them
// into a 'Results and Discussion' chapter, but more often they are kept
// separate.
// ]

// / Discussion\:: #block[
// See #link(<sec:development>)[3.2.1];.
// ]

// / Conclusion\:: #block[
// See #link(<sec:development>)[3.2.1];.
// ]

// / Bibliography\:: #block[
// See #link(<sec:development>)[3.2.1];.
// ]

// === Monograph PhD Thesis
// <sec:monograph>
// Traditionally, it has been common to structure a PhD thesis as a single
// book – a #emph[monograph];. If the thesis is in the form of one single
// coherent research project, it can be structured along the lines of
// #link(<sec:resesarch>)[3.2.2];. However, for such a big work that a PhD
// thesis constitutes, the tasks undertaken are often more diverse, and
// thus more naturally split into several smaller research projects as
// follows:

// / Introduction\:: #block[
// The introduction would serve the same purpose as for a smaller research
// project described in #link(<sec:development>)[3.2.1];, but would
// normally be somewhat more extensive. The #emph[agenda] part should
// inform the reader about the structure of the rest of the document, since
// this may vary significantly between theses.
// ]

// / Background\:: #block[
// Where as background chapters are not necessarily needed in smaller
// works, they are almost always need in PhD thesis. They may even be split
// into several chapters if there are significantly different topics to
// cover. See #link(<sec:resesarch>)[3.2.2];.
// ]

// / Main chapters\:: #block[
// Each main chapter can be structured more or less like a scientific
// paper. Depending on how much is contained in the introduction and
// background sections, the individual introduction and background sections
// can be significantly reduced or even omitted completely.

// - (Introduction)

// - (Background)

// - Method

// - Results

// - Discussion

// - Conclusion
// ]

// / Discussion\:: #block[
// In addition to the discussions within each of the individual chapters,
// the contribution of the thesis #emph[as a whole] should be thoroughly
// discussed here.
// ]

// / Conclusion\:: #block[
// In addition to the conclusions of each of the individual chapters, the
// overall conclusion of the thesis, and how the different parts contribute
// to it, should be presented here. The conclusion should answer to the
// research questions set out in the main introduction. See also
// #link(<sec:development>)[3.2.1];.
// ]

// / Bibliography\:: #block[
// See #link(<sec:development>)[3.2.1];.
// ]

// === Compiled PhD Thesis
// <sec:compiledphd>
// Instead of writing up the PhD thesis as a monograph, compiled PhD theses
// (also known as stapler theses, sandwich theses, integrated theses, PhD
// by published work) consisting of reproductions of already published
// research papers are becoming increasingly common. At least some of the
// papers should already have been accepted for publication at the time of
// submission of the thesis, and thus have been through a real quality
// control by peer review.

// / Introduction\:: #block[
// See #link(<sec:monograph>)[3.2.3];.
// ]

// / Background\:: #block[
// See #link(<sec:monograph>)[3.2.3];.
// ]

// / Main contributions\:: #block[
// This chapter should sum up #emph[and integrate] the contribution of the
// thesis as a whole. It should not merely be a listing of the abstracts of
// the individual papers – they are already available in the attached
// papers, and, as such, not needed here.
// ]

// / Discussion\:: #block[
// See #link(<sec:monograph>)[3.2.3];.
// ]

// / Conclusion\:: #block[
// See #link(<sec:monograph>)[3.2.3];.
// ]

// / Bibliography\:: #block[
// See #link(<sec:development>)[3.2.1];.
// ]

// / Paper I\:: #block[
// First included paper with main contributions. It can be included
// verbatim as a PDF. The publishers PDF should be used if the copyright
// permits it. This should be checked with the SHERPA/RoMEO
// database#footnote[#link("http://sherpa.ac.uk/romeo/index.php");] or with
// the publisher. Even when it is no general permission by the publisher,
// you may write and ask for one.
// ]

// / Paper II\:: #block[
// etc.
// ]

// == Back Matter
// <back-matter>
// Material that does not fit elsewhere, but that you would still like to
// share with the readers, can be put in appendices. See
// #link(<app:additional>)[5];.

// = Conclusion
// <conclusion>
// You definitely should use the `nifty-ntnu-thesis` typst template for your
// thesis.

// #show: appendix.with(chapters-on-odd: chapters-on-odd)
// = Additional Material
// <app:additional>
// Additional material that does not fit in the main thesis but may still
// be relevant to share, e.g., raw data from experiments and surveys, code
// listings, additional plots, pre-project reports, project agreements,
// contracts, logs etc., can be put in appendices. Simply issue the command
// ```#show: appendix``` in the main `.typ` file, and make one chapter per appendix.
