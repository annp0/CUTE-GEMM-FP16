We introduce the notion of layouts and their algebraic structures. This is intented as a quick yet clear introduction. We will handwave proofs. 

**Definition.** A *nested tuple* is a finite tuple

$$
T = (T_1, T_2, \cdots, T_{N-1})
$$

where each component $T_i$ is either a positive integer or another nested tuple.

A nested tuple $T$ can be naturally *flattened* into a tuple of integers, denoted $\mathrm{flat}(T)$. For example, $\mathrm{flat}(2, (3, 4)) = (2, 3, 4).$

**Definition.** A *layout* $L$ consists of two pieces of data: a *shape* $S$, which is a nested tuple, and a *stride* $T$, which has the same structure as $S$.

Given a shape, we define the set of *coordinates* $C$ to be the set of nested tuples of the same structure, where each number is bounded by the corresponding number in the shape. Then the layout defines a function $L : C \to \mathbb{N}$ given by the dot product between the flattened coordinate and the stride.

The constructs above do not necessitate nested structures. However, it is important because of how CuTe defines natural 1D and 2D indexing.

All shapes have 1D indexing. For 1D indexing, it increments from the left. After going through all the elements on the left, we reset the left and increment the right by 1. If the 1D index exceeds the total number of elements, the overflow is recorded in the rightmost coordinate, while all coordinates to the left wrap around periodically. This extends naturally to hierarchical structures, and with it we can view a layout as a map $\mathbb{N}\to\mathbb{N}$.

Formally, given a flat shape $(D_0, \cdots, D_{N-1})$ this map is a bijective map $[0, D]\to [0, D_0]\times\cdots[0, D_{N-1}]$, with
$$
x\mapsto \left(x\,\mathrm{mod}\,D_0, \left\lfloor\dfrac{x}{D_0}\right\rfloor\,\mathrm{mod}\,D_1, \cdots, \left\lfloor\dfrac{x}{D_0\cdots D_{n-1}}\right\rfloor\right)
$$
note that we do not take modulo over the last entry, therefore this function is naturally extended.


2D indexing is available only for shapes with two top-level components. The two coordinates correspond to the respective 1D indexing of each component.

**Example.** Suppose we have a matrix composed of $M_o \times N_o$ tiles, each of size $M_i \times N_i$. Then the natural 2D indexing of the shape $((M_i, M_o), (N_i, N_o))$ aligns with this structure.

Now we move to define _coalescence_. It simplifies the layout computation and does not change the layout function. Given a layout, we first flatten it. Then without loss of generality we can assume $L=(s_0, s_1):(d_0, d_1)$, then we split it into four cases:

- $d_1=s_0d_0$, in this case $L=(s_0s_1:d_0)$, where the equality is over the natural 1D indexing;
- $s_0=1$, in this case $L=(s_1:d_1)$;
- $s_1=1$, in this case $L=(s_0:d_0)$;
- Anything else, for there is nothing we can do.

_By-Mode Coalescence_ is just applying coalescence to each component as we specifiy. We generally assume the layouts are coalesced.

We say a flat layout $L=(s_0,\cdots, s_n):(d_0,\cdots, d_n)$ is _sorted_ if $d_0\le\cdots\le d_n$ and for all $i<j$, if $d_i=d_j$, then $s_i\le s_j$. Note that sorting a flat layout in general will change the function.

**Definition.** Given a layout $L=(s_0,\cdots,s_n):(d_0,\cdots,d_n)$ and a positive integer $M$, if $L$ is not sorted we replace it with its sorted version. We say the pair $(L,M)$ is _admissible for complementation_ if:

- for all $1\le i\le n$, $s_{i-1}d_{i-1}$ divides $d_{i}$;
- $s_n d_n$ divides $M$.

For such pairs $(L,M)$, its _complement_ is defined to be the layout

$$
(L,M)^*=\left(d_0,\dfrac{d_1}{s_0d_0},\cdots, \dfrac{d_n}{s_{n-1}d_{n-1}}, \dfrac{M}{s_nd_n}\right):(1, s_0d_0, \cdots, s_nd_n)
$$

The intuition is that, for each mode, we fill the gaps left by the stride $d_i$. The admissibility condition makes sure that (1) $s_{i-1}d_{i-1}\le d_i$, i.e. their range does not overlap; (2) divisibility gurantees everything is aligned. We start from the smallest stride $d_0$, to fill it we need $(d_0:1)$. And for the second stride $d_1$, we fill it with $\dfrac{d_1}{s_0d_0}$ many blocks of length $s_0d_0$, i.e. $\left(\dfrac{d_1}{s_0d_0}:s_0d_0\right)$. The rest follows by induction.

We note 2 properties that are obvious from the intuition: (1) the size of $(L, M)^*$ is $M$ divided by the size of $L$; (2) $(L,M)^*$ is strictly increasing as a function $\mathbb{N}\to\mathbb{N}$.

The following proposition states our intuition for consturcting complements.

**Proposition.** Given admissible pairs $(A,M)$ and its complement $B=(A,M)^*$, and $C=(A,B)$ their concatenation, then the size of $C$ is $M$, the image of $C$ is $[0, M-1]$, and it is a bijection.

**Proof (Sketch).** First, observe that we can freely permutate modes in $C$ without changing its image and bijectivity. By sorting $C=(A, B)$, we get

$$
C=\left(d_0, s_0, \dfrac{d_1}{s_0d_0}, d_1, \cdots\right):(1, d_0, s_0d_0, d_1,\cdots)
$$

From previous analysis it can be shown that this is indeed a bijection on $[0, M-1]$.

**Definition.** Given $(s_0,\cdots, s_n)$, we say it is _left divisible_ by $d$, if there exists $i$ with

1. There exist an integer $c$ such that $c(s_0 s_1\cdots s_{i-1})=d$ (this condition is empty when $i=0$);
2. If $i<n$, then $c < s_i$ and $c$ is a factor of $s_i$.

It can be proven if such $i$ exists, it is unique.

**Definition.** Given a shape $S=(s_0,\cdots, s_n)$ and a layout $B=s:d$, the pair $(S,B)$ is said to be _admissible for composition_ if:

1. $(s_0,\cdots, s_n)$ is left divisible by $d$ This is called the _stride divisibility condition_.
2. Denote $S'=\left(s_i/c,s_{i+1}\cdots s_n\right)$, $S'$ is left divisible by $s$. This is called the _shape divisibility condition_.

The stride divisibility ensures the starting point of $B$ aligns with $A$’s stride grid, and the shape divisibility ensures the extent of $B$ fits inside a stride period of $A$.

**Definition.** Given a flat layout $A=(s_0,\cdots, s_n):(d_0,\cdots,d_n)$ and a single mode $B=s:d$, then if the shape of $A$ and $B$ is admissible for composition, then we define their _composition_ to be
$$
A\circ B=(s_i/c,s_{i+1},\cdots,s_{i'-1}, c'):(cd_i,\cdots,d_j)
$$

where $i, c$ is from the stride divisibility and $i',c'$ is from the shape divisibility condition. 

We can verify that on the domain of $B$, this definition indeed agrees with their composition as functions.

**Definition.** Given a single node $B=s:d$ and a shape $S=(s_0,\cdots,s_n)$, let $D=s_0\cdots s_n$, the _interval of image_ of $B$ under $S$ is $[d, d(s-1)]\cap[1, D)$.

**Definition.** Let $A=(s_0,\cdots, s_n):(d_0,\cdots, d_n)$ and $B=(s_0',\cdots, s_n'):(d_0',\cdots, d_m')=(B_0,\cdots, B_m)$, then we say $(A, B)$ is _admissible for composition_ if:

1. All modes in $B$ are admissible for the shape of $A$;
2. Each mode in $B$ has disjoint interval of image.

Under those conditions, their _composition_ can be defined as

$$
A\circ B = (A\circ B_0,\cdots, A\circ B_m)
$$

**Proposition.** Our definition of composition agress with their composition as functions.

**Proof (Sketch).** The key here is to show that, for example, given cords $(x_0', x_1')$ in $B$, then we need to show that $\lfloor (x_0'd_0'+x_1'd_1')/d_0\rfloor$ is the same as $\lfloor x_0'd_0'/d_0\rfloor+\lfloor x_1'd_1'/d_0\rfloor$. This is exactly covered by the second condition.

Obviously, we can also define the _by-mode composoition_ by applying composition to each individual component.

**Definition.** Given that $(B,M)$ and $(A,B)$ are admissible for complementation and composition, the _logical division_ of layout $A$ and $B$ is the layout

$$
A/B=A\circ(B, (B,M)^*)
$$

The admissibility of the composition for the second component can be shown from the conditions.

The first component gathers data contained in $A\circ B$, while the second component gives the tiling of $B$ blocks.

In a similar fashion we can define logical products, where $A\times B$ is just a repetition of $A$ against $B$. 