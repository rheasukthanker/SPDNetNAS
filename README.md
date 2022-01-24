# Neural Architecture Search of SPD Manifold Networks
### Accepted to IJCAI 2021
#### Authors: [Rhea Sukthanker](https://rheasukthanker.github.io/), [Zhiwu Huang](https://zhiwu-huang.github.io/), [Suryansh Kumar](https://suryanshkumar.github.io/), [Erik Goron Endsjo](https://ch.linkedin.com/in/erikgoron), [Yan Wu](https://vision.ee.ethz.ch/people-details.MjUzMDc2.TGlzdC8zMjg5LC0xOTcxNDY1MTc4.html) and [Luc Van Gool](https://scholar.google.ch/citations?hl=en&user=TwMib_QAAAAJ)

Paper: https://www.ijcai.org/proceedings/2021/0413.pdf

![alt text](images/overview.png)

## Abstract
In this paper, we propose a new neural architecture search (NAS) problem of Symmetric Positive Definite (SPD) manifold networks, aiming to automate
the design of SPD neural architectures. To address this problem, we first introduce a geometrically rich and diverse SPD neural architecture search space
for an efficient SPD cell design. Further, we model our new NAS problem with a one-shot training process of a single supernet. Based on the supernet modeling, we exploit a differentiable NAS algorithm on our relaxed continuous search space for SPD neural architecture search. Statistical evaluation of our method on drone, action, and emotion recognition tasks mostly provides better results than the state-of-the-art SPD networks and traditional NAS algorithms. Empirical results show that our algorithm excels in discovering better performing SPD network design and provides models that are more than three times lighter than searched by the state-of-the-art NAS algorithms.

## Overview
1. [Installation & Dependencies](#Dependencies)
2. [Prepration](#Prepration)
    1. [Directories](#Directories)
    2. [Data](#Data)
    3. [Pretrained Weights](#Weights)
3. [Training](#Training)
    1. [Launch the Training](#launch)
    2. [Important Training Parameters](#params)
    3. [Training Stages](#stage)
5. [Evaluation](#Evaluation)
    1. [Metrics](#Metrics)
    2. [Final Evaluation](#Final)
6. [Results](#Results)
7. [Contact](#Contact)
8. [How to Cite](#How-to-Cite)
