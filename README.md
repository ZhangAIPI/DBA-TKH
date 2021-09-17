# DBA-PM:  A Decision-based Black-Box Attack with Pseudo Momentum

Based on [SurFree]([t-maho/SurFree: SurFree: a fast surrogate-free black-box attack (github.com)](https://github.com/t-maho/SurFree)) algorithm, we propose a better decision-based black-box attack algorithm: DBA-PM, which performs better when query time is limited to less than 5000.

The attacked image is regarded as a high-dimensional point. In the selected two-dimensional subspace, $x_o$ is the original image, $x_a$ is the initial adversarial image, the blue curve is the judging boundary (unknown), $\pmb{u}$ and $\pmb{v}$ are a pair of orthogonal unit vectors, and the red curve is a semicircle whose endpoints are $x_o$ and $x_a$. With $x_o$ as the starting point, make a line whose inclination with $x_ox_a$ is $\theta$, and the ray of degree angle intersects the semicircle at $x$. Constantly changing $\theta$(In the algorithm, dichotomy is used until $x$ is adversarial point with max θ, at this time $\theta$ Recorded as $\theta^*$。

![basic principle](https://github.com/TongKangheng/Images/blob/main/DBA-PM_basis.png?raw=true)

In our paper and code, we will compare DBA-PM with some famous algorithms, such as SurFree, HSJA, QEBA, OPT, Sign-OPT and BA. Now we have completed part of work, and we will constantly improve it.

<center>Performance of each algorithm with 1000 queries
</center>

| **DBA-PM**   | **6.2049** | **5.4687** | **5.9860** | **6.7389** | **5.4364** |
| ------------ | ---------- | :--------- | ---------- | ---------- | ---------- |
| **SurFree**  | 10.3708    | 7.8749     | 9.3120     | 9.2793     | 7.6905     |
| **OPT**      | 24.2680    | 22.1033    | 24.4556    | 27.8917    | 23.6855    |
| **Sign-OPT** | 14.6477    | 13.7709    | 15.9180    | 16.9340    | 14.7985    |
| **Win**      | DBA-PM     | DBA-PM     | DBA-PM     | DBA-PM     | DBA-PM     |

![Average1000](https://github.com/TongKangheng/Images/blob/main/Average1000.png?raw=true)

<center>Performance of each algorithm with 20000 queries
</center>

| **DBA-PM**   | we         | 2.2376     | 3.5555     | 3.2538     | 2.2165     |
| ------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| **SurFree**  | **1.7646** | **0.6922** | 1.1104     | 1.2936     | 0.5315     |
| **OPT**      | 8.5750     | 4.5156     | 3.6858     | 5.9713     | 2.1248     |
| **Sign-OPT** | 1.8733     | 0.6955     | **0.8997** | **1.2334** | **0.4818** |
| **Win**      | SurFree    | SurFree    | Sign-OPT   | Sign-OPT   | Sign-OPT   |

![Average20000](https://github.com/TongKangheng/Images/blob/main/Average20000.png?raw=true)