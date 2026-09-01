# Тексты с провалом крупной полосы: до и после починки форматирования

Правка механическая, только пробелы и пунктуация. Поток букв не изменён.

---

## [wiki_csai] −−−  ·  крупная 12.47 → 13.45 (+7.9%)  ·  снято 126 символов  ·  урон 3

*судья: Systematic LaTeX rendering math junk throughout the text severely degrades mathematical content.*

**до:**

```text
In mathematics, the Bernoulli numbers Bn are a sequence of rational numbers which occur frequently in analysis. The Bernoulli numbers appear in (and can be defined by) the Taylor series expansions of the tangent and hyperbolic tangent functions, in Faulhaber's formula for the sum of m-th powers of the first n positive integers, in the Euler–Maclaurin formula, and in expressions for certain values of the Riemann zeta function.
The values of the first 20 Bernoulli numbers are given in the adjacent table. Two conventions are used in the literature, denoted here by 




B

n


−






{\displaystyle B_{n}^{-{}}}

 and 




B

n


+






{\displaystyle B_{n}^{+{}}}

; they differ only for n = 1, where 




B

1


−




=
−
1

/

2


{\displaystyle B_{1}^{-{}}=-1/2}

 and 




B

1


+




=
+
1

/

2


{\displaystyle B_{1}^{+{}}=+1/2}

. For every odd n > 1, Bn = 0. For every even n > 0, Bn is negative if n is divisible by 4 and positive otherwise. The Bernoulli numbers are special values of the Bernoulli polynomials 




B

n


(
x
)


{\displaystyle B_{n}(x)}

, with 




B

n


−




=

B

n


(
0
)


{\displaystyle B_{n}^{-{}}=B_{n}(0)}

 and 




B

n


+


=

B

n


(
1
)


{\displaystyle B_{n}^{+}=B_{n}(1)}

.
The Bernoulli numbers were discovered around the same time by the Swiss mathematician Jacob Bernoulli, after whom they are named, and independently by Japanese mathematician Seki Takakazu.  Seki's discovery was posthumously published in 1712 in his work Katsuyō Sanpō; Bernoulli's, also posthumously, in his Ars Conjectandi of 1713.  Ada Lovelace's note G on the Analytical Engine from 1842 describes an algorithm for generating Bernoulli numbers with Babbage's machine. As a result, the Bernoulli numbers have the distinction of being the subject of the first published complex computer program.
```

**после:**

```text
In mathematics, the Bernoulli numbers Bn are a sequence of rational numbers which occur frequently in analysis. The Bernoulli numbers appear in (and can be defined by) the Taylor series expansions of the tangent and hyperbolic tangent functions, in Faulhaber's formula for the sum of m-th powers of the first n positive integers, in the Euler–Maclaurin formula, and in expressions for certain values of the Riemann zeta function.
The values of the first 20 Bernoulli numbers are given in the adjacent table. Two conventions are used in the literature, denoted here by B n − {\displaystyle B_{n}^{-{}}} and B n + {\displaystyle B_{n}^{+{}}}; they differ only for n = 1, where B 1 − =
−
1 / 2 {\displaystyle B_{1}^{-{}}=-1/2} and B 1 + =
+
1 / 2 {\displaystyle B_{1}^{+{}}=+1/2}. For every odd n > 1, Bn = 0. For every even n > 0. Bn is negative if n is divisible by 4 and positive otherwise. The Bernoulli numbers are special values of the Bernoulli polynomials B n (x) {\displaystyle B_{n}(x)}, with B n − = B n (0) {\displaystyle B_{n}^{-{}}=B_{n}(0)} and B n + = B n (1) {\displaystyle B_{n}^{+}=B_{n}(1)}.
The Bernoulli numbers were discovered around the same time by the Swiss mathematician Jacob Bernoulli, after whom they are named, and independently by Japanese mathematician Seki Takakazu. Seki's discovery was posthumously published in 1712 in his work Katsuyō Sanpō; Bernoulli's, also posthumously, in his Ars Conjectandi of 1713. Ada Lovelace's note G on the Analytical Engine from 1842 describes an algorithm for generating Bernoulli numbers with Babbage's machine. As a result, the Bernoulli numbers have the distinction of being the subject of the first published complex computer program.
```

## [wiki_csai] −−−  ·  крупная 12.09 → 14.50 (+19.9%)  ·  снято 120 символов  ·  урон 3

*судья: LaTeX math formulas systematic rendering junk from Wikipedia mediawiki conversion.*

**до:**

```text
In mathematics, a Markov decision process (MDP) is a discrete-time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. MDPs are useful for studying optimization problems solved via dynamic programming. MDPs were known at least as early as the 1950s; a core body of research on Markov decision processes resulted from Ronald Howard's 1960 book, Dynamic Programming and Markov Processes. They are used in many disciplines, including robotics, automatic control, economics and manufacturing. The name of MDPs comes from the Russian mathematician Andrey Markov as they are an extension of Markov chains.
At each time step, the process is in some state 



s


{\displaystyle s}

, and the decision maker may choose any action 



a


{\displaystyle a}

 that is available in state 



s


{\displaystyle s}

. The process responds at the next time step by randomly moving into a new state 




s
′



{\displaystyle s'}

, and giving the decision maker a corresponding reward 




R

a


(
s
,

s
′

)


{\displaystyle R_{a}(s,s')}

.
The probability that the process moves into its new state 




s
′



{\displaystyle s'}

 is influenced by the chosen action. Specifically, it is given by the state transition function 




P

a


(
s
,

s
′

)


{\displaystyle P_{a}(s,s')}

. Thus, the next state 




s
′



{\displaystyle s'}

 depends on the current state 



s


{\displaystyle s}

 and the decision maker's action 



a


{\displaystyle a}

. But given 



s


{\displaystyle s}

 and 



a


{\displaystyle a}

, it is conditionally independent of all previous states and actions; in other words, the state transitions of an MDP satisfy the Markov property.
Markov decision processes are an extension of Markov chains; the difference is the addition of actions (allowing choice) and rewards (giving motivation). Conversely, if only one action exists for each state (e.g. "wait") and all rewards are the same (e.g. "zero"), a Markov decision process reduces to a Markov chain.
```

**после:**

```text
In mathematics, a Markov decision process (MDP) is a discrete-time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. MDPs are useful for studying optimization problems solved via dynamic programming. MDPs were known at least as early as the 1950s; a core body of research on Markov decision processes resulted from Ronald Howard's 1960 book, Dynamic Programming and Markov Processes. They are used in many disciplines, including robotics, automatic control, economics and manufacturing. The name of MDPs comes from the Russian mathematician Andrey Markov as they are an extension of Markov chains.
At each time step, the process is in some state s {\displaystyle s}, and the decision maker may choose any action a {\displaystyle a} that is available in state s {\displaystyle s}. The process responds at the next time step by randomly moving into a new state s
′ {\displaystyle s'}, and giving the decision maker a corresponding reward R a (s, s
′) {\displaystyle R_{a}(s,s')}.
The probability that the process moves into its new state s
′ {\displaystyle s'} is influenced by the chosen action. Specifically, it is given by the state transition function P a (s, s
′) {\displaystyle P_{a}(s,s')}. Thus, the next state s
′ {\displaystyle s'} depends on the current state s {\displaystyle s} and the decision maker's action a {\displaystyle a}. But given s {\displaystyle s} and a {\displaystyle a}, it is conditionally independent of all previous states and actions; in other words, the state transitions of an MDP satisfy the Markov property.
Markov decision processes are an extension of Markov chains; the difference is the addition of actions (allowing choice) and rewards (giving motivation). Conversely, if only one action exists for each state (e.g. "wait") and all rewards are the same (e.g. "zero"), a Markov decision process reduces to a Markov chain.
```

## [wiki_csai] −−−  ·  крупная 12.22 → 15.18 (+24.2%)  ·  снято 120 символов  ·  урон 3

*судья: LaTeX math formulas systematic rendering junk from Wikipedia mediawiki conversion.*

**до:**

```text
In mathematics, a Markov decision process (MDP) is a discrete-time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. MDPs are useful for studying optimization problems solved via dynamic programming. MDPs were known at least as early as the 1950s; a core body of research on Markov decision processes resulted from Ronald Howard's 1960 book, Dynamic Programming and Markov Processes. They are used in many disciplines, including robotics, automatic control, economics and manufacturing. The name of MDPs comes from the Russian mathematician Andrey Markov as they are an extension of Markov chains.
At each time step, the process is in some state 



s


{\displaystyle s}

, and the decision maker may choose any action 



a


{\displaystyle a}

 that is available in state 



s


{\displaystyle s}

. The process responds at the next time step by randomly moving into a new state 




s
′



{\displaystyle s'}

, and giving the decision maker a corresponding reward 




R

a


(
s
,

s
′

)


{\displaystyle R_{a}(s,s')}

.
The probability that the process moves into its new state 




s
′



{\displaystyle s'}

 is influenced by the chosen action. Specifically, it is given by the state transition function 




P

a


(
s
,

s
′

)


{\displaystyle P_{a}(s,s')}

. Thus, the next state 




s
′



{\displaystyle s'}

 depends on the current state 



s


{\displaystyle s}

 and the decision maker's action 



a


{\displaystyle a}

. But given 



s


{\displaystyle s}

 and 



a


{\displaystyle a}

, it is conditionally independent of all previous states and actions; in other words, the state transitions of an MDP satisfy the Markov property.
Markov decision processes are an extension of Markov chains; the difference is the addition of actions (allowing choice) and rewards (giving motivation). Conversely, if only one action exists for each state (e.g. "wait") and all rewards are the same (e.g. "zero"), a Markov decision process reduces to a Markov chain.
```

**после:**

```text
In mathematics, a Markov decision process (MDP) is a discrete-time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. MDPs are useful for studying optimization problems solved via dynamic programming. MDPs were known at least as early as the 1950s; a core body of research on Markov decision processes resulted from Ronald Howard's 1960 book, Dynamic Programming and Markov Processes. They are used in many disciplines, including robotics, automatic control, economics and manufacturing. The name of MDPs comes from the Russian mathematician Andrey Markov as they are an extension of Markov chains.
At each time step, the process is in some state s {\displaystyle s}, and the decision maker may choose any action a {\displaystyle a} that is available in state s {\displaystyle s}. The process responds at the next time step by randomly moving into a new state s
′ {\displaystyle s'}, and giving the decision maker a corresponding reward R a (s, s
′) {\displaystyle R_{a}(s,s')}.
The probability that the process moves into its new state s
′ {\displaystyle s'} is influenced by the chosen action. Specifically, it is given by the state transition function P a (s, s
′) {\displaystyle P_{a}(s,s')}. Thus, the next state s
′ {\displaystyle s'} depends on the current state s {\displaystyle s} and the decision maker's action a {\displaystyle a}. But given s {\displaystyle s} and a {\displaystyle a}, it is conditionally independent of all previous states and actions; in other words, the state transitions of an MDP satisfy the Markov property.
Markov decision processes are an extension of Markov chains; the difference is the addition of actions (allowing choice) and rewards (giving motivation). Conversely, if only one action exists for each state (e.g. "wait") and all rewards are the same (e.g. "zero"), a Markov decision process reduces to a Markov chain.
```

## [reddit_eli5] −0−  ·  крупная 15.11 → 13.94 (-7.7%)  ·  снято 88 символов  ·  урон 3

*судья: Systematic tokenization artifacts added spaces before punctuation throughout, alongside inline URL placeholders from web scraping.*

**до:**

```text
When graphing parabolas , the x and y coordinates can be expressed with the general formula y = ax^(2)+bx+c , known as a * quadratic equation * . When you plug in specific numbers for a , b , and c , it will affect how the parabola looks . [ Try moving the sliders here , and see how changing each parameter ( a , b , and c are called * parameters * ) affects the final look of the parabola . ] ( URL_1 ) One common question with parabolas is " Where , if any place , does the parabola cross the x axis ? " The location of x axis can be described very simply : y = 0 . Wherever y=0 , you 'll find the x axis there . So , to plot the points on a graph on graph ONLY on the x axis , we replace y with 0 in the formula to get : ax^(2)+bx+c=0 Even when you 're given a specific a , b , and c for that formula , it 's not very user - friendly when trying to find what x is when y=0 . A better approach would be to rearrange the formula so that x is on one side , and all the other variables are on the other side . Then you could just plug in a , b , and c , and find out x right away ! [ Let 's ask Wolfram|Alpha to rearrange ax^(2)+bx+c=0 for us so that x is on one side , and you 'll see that we get the quadratic formula ! ] ( URL_0 ) You might want to click on the * Step - by - step solution * button to see how Wolfram|Alpha got from here to there . Also , note that you only get the quadratic formula if the formula is set equal to 0 . [ If you set the formula to equal , say , 3 , you get something completely different . ] ( URL_2 ) So , the quadratic equation is just the parabola formula rearranged to make it easier to find out where , if any place , the parabola crosses the x - axis . [ When playing around with various values for a , b , and c here ] ( URL_1 ) , you can see not only how they change the look of the parabola , but where the parabola crosses the x axis , if any place . ( It can cross , as you can see , at 2 places , 1 place , or no place . )
```

**после:**

```text
When graphing parabolas, the x and y coordinates can be expressed with the general formula y = ax^(2)+bx+c, known as a * quadratic equation *. When you plug in specific numbers for a, b, and c, it will affect how the parabola looks. [Try moving the sliders here, and see how changing each parameter (a, b, and c are called * parameters *) affects the final look of the parabola.] (URL_1) One common question with parabolas is " Where, if any place, does the parabola cross the x axis? " The location of x axis can be described very simply: y = 0. Wherever y=0, you'll find the x axis there. So, to plot the points on a graph on graph ONLY on the x axis, we replace y with 0 in the formula to get: ax^(2)+bx+c=0 Even when you're given a specific a, b, and c for that formula, it's not very user-friendly when trying to find what x is when y=0. A better approach would be to rearrange the formula so that x is on one side, and all the other variables are on the other side. Then you could just plug in a, b, and c, and find out x right away! [Let's ask Wolfram|Alpha to rearrange ax^(2)+bx+c=0 for us so that x is on one side, and you'll see that we get the quadratic formula!] (URL_0) You might want to click on the * Step-by-step solution * button to see how Wolfram|Alpha got from here to there. Also, note that you only get the quadratic formula if the formula is set equal to 0. [If you set the formula to equal, say, 3, you get something completely different.] (URL_2) So, the quadratic equation is just the parabola formula rearranged to make it easier to find out where, if any place, the parabola crosses the x-axis. [When playing around with various values for a, b, and c here] (URL_1), you can see not only how they change the look of the parabola, but where the parabola crosses the x axis, if any place. (It can cross, as you can see, at 2 places, 1 place, or no place.)
```

## [wiki_csai] −−−  ·  крупная 13.28 → 15.41 (+16.1%)  ·  снято 82 символов  ·  урон 3

*судья: Mathematical formulas have been systematically mangled into raw LaTeX and rendering artifacts.*

**до:**

```text
In computer science, the time complexity is the computational complexity that describes the amount of computer time it takes to run an algorithm. Time complexity is commonly estimated by counting the number of elementary operations performed by the algorithm, supposing that each elementary operation takes a fixed amount of time to perform. Thus, the amount of time taken and the number of elementary operations performed by the algorithm are taken to be related by a constant factor.
Since an algorithm's running time may vary among different inputs of the same size, one commonly considers the worst-case time complexity, which is the maximum amount of time required for inputs of a given size. Less common, and usually specified explicitly, is the average-case complexity, which is the average of the time taken on inputs of a given size (this makes sense because there are only a finite number of possible inputs of a given size). In both cases, the time complexity is generally expressed as a function of the size of the input.: 226  Since this function is generally difficult to compute exactly, and the running time for small inputs is usually not consequential, one commonly focuses on the behavior of the complexity when the input size increases—that is, the asymptotic behavior of the complexity. Therefore, the time complexity is commonly expressed using big O notation, typically 



O
(
n
)


{\displaystyle O(n)}

, 



O
(
n
log
⁡
n
)


{\displaystyle O(n\log n)}

, 



O
(

n

α


)


{\displaystyle O(n^{\alpha })}

, 



O
(

2

n


)


{\displaystyle O(2^{n})}

, etc., where n is the size in units of bits needed to represent the input.
Algorithmic complexities are classified according to the type of function appearing in the big O notation. For example, an algorithm with time complexity 



O
(
n
)


{\displaystyle O(n)}

 is a linear time algorithm and an algorithm with time complexity 



O
(

n

α


)


{\displaystyle O(n^{\alpha })}

 for some constant 



α
>
1


{\displaystyle \alpha >1}

 is a polynomial time algorithm.
```

**после:**

```text
In computer science, the time complexity is the computational complexity that describes the amount of computer time it takes to run an algorithm. Time complexity is commonly estimated by counting the number of elementary operations performed by the algorithm, supposing that each elementary operation takes a fixed amount of time to perform. Thus, the amount of time taken and the number of elementary operations performed by the algorithm are taken to be related by a constant factor.
Since an algorithm's running time may vary among different inputs of the same size, one commonly considers the worst-case time complexity, which is the maximum amount of time required for inputs of a given size. Less common, and usually specified explicitly, is the average-case complexity, which is the average of the time taken on inputs of a given size (this makes sense because there are only a finite number of possible inputs of a given size). In both cases, the time complexity is generally expressed as a function of the size of the input.: 226 Since this function is generally difficult to compute exactly, and the running time for small inputs is usually not consequential, one commonly focuses on the behavior of the complexity when the input size increases—that is, the asymptotic behavior of the complexity. Therefore, the time complexity is commonly expressed using big O notation, typically O
(n) {\displaystyle O(n)}, O
(n
log
⁡
n) {\displaystyle O(n\log n)}, O
(n α) {\displaystyle O(n^{\alpha })}, O
(2 n) {\displaystyle O(2^{n})}, etc. where n is the size in units of bits needed to represent the input.
Algorithmic complexities are classified according to the type of function appearing in the big O notation. For example, an algorithm with time complexity O
(n) {\displaystyle O(n)} is a linear time algorithm and an algorithm with time complexity O
(n α) {\displaystyle O(n^{\alpha })} for some constant α
>
1 {\displaystyle \alpha >1} is a polynomial time algorithm.
```

## [wiki_csai] −−−  ·  крупная 13.42 → 15.65 (+16.7%)  ·  снято 73 символов  ·  урон 3

*судья: LaTeX formula rendering artifacts corrupted the variables in the third paragraph.*

**до:**

```text
Category theory is a general theory of mathematical structures and their relations that was introduced by Samuel Eilenberg and Saunders Mac Lane in the middle of the 20th century in their foundational work on algebraic topology. Nowadays, category theory is used in almost all areas of mathematics, and in some areas of computer science. In particular, many constructions of new mathematical objects from previous ones, that appear similarly in several contexts are conveniently expressed and unified in terms of categories. Examples include quotient spaces, direct products, completion, and duality.
A category is formed by two sorts of objects: the objects of the category, and the morphisms, which relate two objects called the source and the target of the morphism. One often says that a morphism is an arrow that maps its source to its target. Morphisms can be composed if the target of the first morphism equals the source of the second one, and morphism composition has similar properties as function composition (associativity and existence of identity morphisms). Morphisms are often some sort of function, but this is not always the case. For example, a monoid may be viewed as a category with a single object, whose morphisms are the elements of the monoid.
The second fundamental concept of category is the concept of a functor, which plays the role of a morphism between two categories 




C

1




{\displaystyle C_{1}}

 and 




C

2


:


{\displaystyle C_{2}:}

 it maps objects of 




C

1




{\displaystyle C_{1}}

 to objects of 




C

2




{\displaystyle C_{2}}

 and morphisms of 




C

1




{\displaystyle C_{1}}

 to morphisms of 




C

2




{\displaystyle C_{2}}

 in such a way that sources are mapped to sources and targets are mapped to targets (or, in the case of a contravariant functor, sources are mapped to targets and vice-versa). A third fundamental concept is a natural transformation that may be viewed as a morphism of functors.
```

**после:**

```text
Category theory is a general theory of mathematical structures and their relations that was introduced by Samuel Eilenberg and Saunders Mac Lane in the middle of the 20th century in their foundational work on algebraic topology. Nowadays, category theory is used in almost all areas of mathematics, and in some areas of computer science. In particular, many constructions of new mathematical objects from previous ones, that appear similarly in several contexts are conveniently expressed and unified in terms of categories. Examples include quotient spaces, direct products, completion, and duality.
A category is formed by two sorts of objects: the objects of the category, and the morphisms, which relate two objects called the source and the target of the morphism. One often says that a morphism is an arrow that maps its source to its target. Morphisms can be composed if the target of the first morphism equals the source of the second one, and morphism composition has similar properties as function composition (associativity and existence of identity morphisms). Morphisms are often some sort of function, but this is not always the case. For example, a monoid may be viewed as a category with a single object, whose morphisms are the elements of the monoid.
The second fundamental concept of category is the concept of a functor, which plays the role of a morphism between two categories C 1 {\displaystyle C_{1}} and C 2: {\displaystyle C_{2}:} it maps objects of C 1 {\displaystyle C_{1}} to objects of C 2 {\displaystyle C_{2}} and morphisms of C 1 {\displaystyle C_{1}} to morphisms of C 2 {\displaystyle C_{2}} in such a way that sources are mapped to sources and targets are mapped to targets (or, in the case of a contravariant functor, sources are mapped to targets and vice-versa). A third fundamental concept is a natural transformation that may be viewed as a morphism of functors.
```

## [reddit_eli5] −00  ·  крупная 13.09 → 18.58 (+41.9%)  ·  снято 57 символов  ·  урон 3

*судья: The document has systematic tokeniser corruption (detached clitics and spaces before all punctuation marks).*

**до:**

```text
Most of the time , they 'll subpoena ( legal request for information ) a site and get copies of their access logs ( I 'd guess that 90 % of sites on the web have a web - server side access log somewhere , except for ones where it is extremely impractical ) or they will monitor and externally log the site , then send out DMCA notifications to ISPs . Additionally , some of them will add fake nodes or peers ( fake bittorrent users ) to BitTorrent swarms , and record what IPs are in the swarm , or which ones try to connect to the peer for some specific piece of content . That 's why I use an IP Blacklist in my torrent client , to block IPs of the industry snitch peers . Most torrent clients will provide a list , or you can use PeerGuardian or PeerBlock to do this . Then they will report to the ISPs . If either of those conditions are met , you will recieve a letter or have service impacts depending on your ISPs policies . For BitTorrent , always use and keep up to date your IP block lists , and put bittorrent on a really high ( 30000 - 65535 ) port . Prefer TLS / Encrypted ( secure connection that ca n't be sniffed ) on your client , there should be an option for this . In the case of FrostWire / LimeWire , GnuTella ( the protocol behind them ) always works the same , uses the same ports , everything , so it 's relatively easy to just watch what people are doing on a specific port and log what you 're searching and downloading . Since it does n't use the HTTP port set , it 's relatively easy to spear fish for the LimeWire / Gnutella packets . So , really , all they do is use educated guesses to look at what you 're doing , or MPAA / RIAA people planting traps .
```

**после:**

```text
Most of the time, they'll subpoena (legal request for information) a site and get copies of their access logs (I'd guess that 90% of sites on the web have a web-server side access log somewhere, except for ones where it is extremely impractical) or they will monitor and externally log the site, then send out DMCA notifications to ISPs. Additionally, some of them will add fake nodes or peers (fake bittorrent users) to BitTorrent swarms, and record what IPs are in the swarm, or which ones try to connect to the peer for some specific piece of content. That's why I use an IP Blacklist in my torrent client, to block IPs of the industry snitch peers. Most torrent clients will provide a list, or you can use PeerGuardian or PeerBlock to do this. Then they will report to the ISPs. If either of those conditions are met, you will recieve a letter or have service impacts depending on your ISPs policies. For BitTorrent, always use and keep up to date your IP block lists, and put bittorrent on a really high (30000-65535) port. Prefer TLS / Encrypted (secure connection that can't be sniffed) on your client, there should be an option for this. In the case of FrostWire / LimeWire, GnuTella (the protocol behind them) always works the same, uses the same ports, everything, so it's relatively easy to just watch what people are doing on a specific port and log what you're searching and downloading. Since it doesn't use the HTTP port set, it's relatively easy to spear fish for the LimeWire / Gnutella packets. So, really, all they do is use educated guesses to look at what you're doing, or MPAA / RIAA people planting traps.
```

## [reddit_eli5] −−−  ·  крупная 15.55 → 12.00 (-22.8%)  ·  снято 54 символов  ·  урон 3

*судья: Raw Markdown link syntax and systematic space-padded punctuation tokenization run throughout the entire text.*

**до:**

```text
All good answers here , but if you 're interested in a little history I can explain where the words came from . [ * * Cyborg * * ] ( URL_2 ) - short for " Cybernetic Organism " . Officially , cybernetics means " the study of self - regulating systems " , but it has informally come to mean " computer stuff " . Hence , cyborgs are organisms with electronic components . [ * * Android * * ] ( URL_3 ) - Comes from a greek root ἀνδρ , meaning " man " , combined with the suffix -oid , which means " looks like " . Thus , they are human - like creations . Technically , only male - like robots are androids . The female equivalent is " gynoid " . Bonus fact : The word " droid " originally came from a shortened form of " android " [ * * Robot * * ] ( URL_0 ) - comes from the word for " slave labor " in Chezk . The term was created by a writer who wrote one of the earliest stories about automatons , called [ * Rossum ’s Universal Robots * ] ( URL_1 ) . He was originally going to call them * laboři * , but decided he did not like the work and so asked his brother for suggestions . The robots in the story were not very similar to what we would call robots today , but the word was popularized and became the common term for any sort of automaton .
```

**после:**

```text
All good answers here, but if you're interested in a little history I can explain where the words came from. [* * Cyborg * *] (URL_2)-short for " Cybernetic Organism ". Officially, cybernetics means " the study of self-regulating systems ", but it has informally come to mean " computer stuff ". Hence, cyborgs are organisms with electronic components. [* * Android * *] (URL_3)-Comes from a greek root ἀνδρ, meaning " man ", combined with the suffix -oid, which means " looks like ". Thus, they are human-like creations. Technically, only male-like robots are androids. The female equivalent is " gynoid ". Bonus fact: The word " droid " originally came from a shortened form of " android " [* * Robot * *] (URL_0)-comes from the word for " slave labor " in Chezk. The term was created by a writer who wrote one of the earliest stories about automatons, called [* Rossum ’s Universal Robots *] (URL_1). He was originally going to call them * laboři *, but decided he did not like the work and so asked his brother for suggestions. The robots in the story were not very similar to what we would call robots today, but the word was popularized and became the common term for any sort of automaton.
```

## [reddit_eli5] −0−  ·  крупная 12.19 → 15.23 (+24.9%)  ·  снято 51 символов  ·  урон 3

*судья: Tokenizer artifacts (spaces before punctuation, split clitics) occur systematically throughout the document.*

**до:**

```text
Here 's my attempt : Scenario : Your house has a front door that is locked and a burglar is trying to get in . Brute force hacking : Burglar tries a million different keys one - by - one . Vulnerability hacking : Burglar checks to see if the door is locked . If it is , burglar checks all other doors of the house . If all are locked , burglar then checks for open windows . If none , then burglar checks for windows that can be easily opened . And so on and so forth . Rootkit hacking : Burglar pretends to be a locksmith and convinces you to let him upgrade your door lock . He upgrades your lock , gives you the key , but also makes himself a copy of the key . Social engineering hacking : Burglar pretends to be a friendly neighbor and brings a six - pack of beer each weekend to befriend you . Because you think he is your friend , you let him know about the spare key underneath your doormat so that he can come over to water the plants when you 're on vacation . Trojan horse : Burglar pretends to be a phone technician so you unlock the door for him . Key logger : Burglar attaches a stealth device to your doorknob that copies your key when your insert your key into the lock . Black hat hacker : Burglar uses above techniques to break in and steal your stuff . White hat hacker : Person uses above techniques to unlock your door , does n't enter , but afterwards , tells you how he did it . Grey hat hacker : There are many definitions , but one example would be a person that uses above techniques to unlock your door , does n't steal your stuff , but does n't tell you that he 's able to unlock your door .
```

**после:**

```text
Here's my attempt: Scenario: Your house has a front door that is locked and a burglar is trying to get in. Brute force hacking: Burglar tries a million different keys one-by-one. Vulnerability hacking: Burglar checks to see if the door is locked. If it is, burglar checks all other doors of the house. If all are locked, burglar then checks for open windows. If none, then burglar checks for windows that can be easily opened. And so on and so forth. Rootkit hacking: Burglar pretends to be a locksmith and convinces you to let him upgrade your door lock. He upgrades your lock, gives you the key, but also makes himself a copy of the key. Social engineering hacking: Burglar pretends to be a friendly neighbor and brings a six-pack of beer each weekend to befriend you. Because you think he is your friend, you let him know about the spare key underneath your doormat so that he can come over to water the plants when you're on vacation. Trojan horse: Burglar pretends to be a phone technician so you unlock the door for him. Key logger: Burglar attaches a stealth device to your doorknob that copies your key when your insert your key into the lock. Black hat hacker: Burglar uses above techniques to break in and steal your stuff. White hat hacker: Person uses above techniques to unlock your door, doesn't enter, but afterwards, tells you how he did it. Grey hat hacker: There are many definitions, but one example would be a person that uses above techniques to unlock your door, doesn't steal your stuff, but doesn't tell you that he's able to unlock your door.
```

## [reddit_eli5] −0−  ·  крупная 11.99 → 12.48 (+4.1%)  ·  снято 48 символов  ·  урон 2

*судья: Tokenizer artifacts (spaces before punctuation and split clitics) occur systematically throughout the text.*

**до:**

```text
Certain diseases are related to genes . Now for every gene in your body it has a partner . You have two sets of genes , one from mom and one from dad , and need both to live . Some times a defect or change in the gene causes a disease . Now here is the tricky part : if the change is only in one of the two copies of the gene and you get the disease , it is called Dominant . If you need the defect in both copies it is Recessive . Lets make up a gene , called DERP1 . Remember you have two copies of DERP1 , mom and dad versions or DERP1 ( m ) and DERP1 ( f ) . Your DERP1 ( m ) is defective , i.e. it does n't work and can not be used to make proteins , but DERP1 ( p ) is just fine and there is no disease . You are a CARRIER for a disease of DERP1 , but you do not have it , because it is Recessive and you need both copies to be defective to have a disease . Now DERP1 defects are rare , so if you marry some random woman , the odds of her and you both having a DERP1 defect are small , and you wo n't pass along the disease to your kids . But your sister may also be a carrier for the defective DERP1 from mom . So if you mate with here the odds of your off spring getting both defective DERP1s ( one from you and one from her ) is 25 % . As a general rule of thumb , most family trees contain one to two Recessive diseases in their genes , most people are carriers and without genetic testing would never know that a whole gene was broken . But by inbreeding the chances of recessive disease genes linking up are vastly increased . Its funny that incest is probably the only universally reviled practice in human history ( I had a anthropology teacher tell me this ) and that even animals will usually avoid inbreeding . Edit : Fixed mistake with %
```

**после:**

```text
Certain diseases are related to genes. Now for every gene in your body it has a partner. You have two sets of genes, one from mom and one from dad, and need both to live. Some times a defect or change in the gene causes a disease. Now here is the tricky part: if the change is only in one of the two copies of the gene and you get the disease, it is called Dominant. If you need the defect in both copies it is Recessive. Lets make up a gene, called DERP1. Remember you have two copies of DERP1, mom and dad versions or DERP1 (m) and DERP1 (f). Your DERP1 (m) is defective, i.e. it doesn't work and can not be used to make proteins, but DERP1 (p) is just fine and there is no disease. You are a CARRIER for a disease of DERP1, but you do not have it, because it is Recessive and you need both copies to be defective to have a disease. Now DERP1 defects are rare, so if you marry some random woman, the odds of her and you both having a DERP1 defect are small, and you won't pass along the disease to your kids. But your sister may also be a carrier for the defective DERP1 from mom. So if you mate with here the odds of your off spring getting both defective DERP1s (one from you and one from her) is 25%. As a general rule of thumb, most family trees contain one to two Recessive diseases in their genes, most people are carriers and without genetic testing would never know that a whole gene was broken. But by inbreeding the chances of recessive disease genes linking up are vastly increased. Its funny that incest is probably the only universally reviled practice in human history (I had a anthropology teacher tell me this) and that even animals will usually avoid inbreeding. Edit: Fixed mistake with%
```

## [reddit_eli5] −0−  ·  крупная 10.63 → 12.35 (+16.2%)  ·  снято 47 символов  ·  урон 3

*судья: Systematic tokenizer artifact splitting contractions and punctuation (`'ll`, `n't`, `ca n't`, spaces before periods).*

**до:**

```text
I 'll take this from a broad philosophical standpoint because I do n't know a lot about Koko . There are a few questions that * all * animals are capable of answering on a very basic level . " Where . " Where can I find food ? Where is home ? Where is my family ? " What . " What is this ? What are you ? " Who . " Obviously you 've seen dogs react to who you are vs who its owner is . " When " is another one . When do I need to eat ? When do I need to sleep ? When do I need to run ? " How ? " How do I best hunt prey ? How do I escape this lion ? How do I climb this tree ? The one question that separates humans from animal is the question " Why ? " Animals survive based off instincts . They learn , but they do n't ever touch on the concept " why " because they just have n't developed their brain enough to be capable of having thoughts as complex as the " why ? " If we could somehow ask animal " why " , their answer would likely be " well .. because its what I do " and that 's the most we 'd get . Animals can think ... but they ca n't think about what they think . That 's a huge reason humans are so different . It 's why you do n't see vegetarian lions . Therefore , a gorilla wo n't be able to actually focus on it 's own consciousness enough to understand the reasoning behind its choices because ... ya know ... instincts .
```

**после:**

```text
I'll take this from a broad philosophical standpoint because I don't know a lot about Koko. There are a few questions that * all * animals are capable of answering on a very basic level. " Where. " Where can I find food? Where is home? Where is my family? " What. " What is this? What are you? " Who. " Obviously you've seen dogs react to who you are vs who its owner is. " When " is another one. When do I need to eat? When do I need to sleep? When do I need to run? " How? " How do I best hunt prey? How do I escape this lion? How do I climb this tree? The one question that separates humans from animal is the question " Why? " Animals survive based off instincts. They learn, but they don't ever touch on the concept " why " because they just haven't developed their brain enough to be capable of having thoughts as complex as the " why? " If we could somehow ask animal " why ", their answer would likely be " well.. because its what I do " and that's the most we'd get. Animals can think... but they can't think about what they think. That's a huge reason humans are so different. It's why you don't see vegetarian lions. Therefore, a gorilla won't be able to actually focus on it's own consciousness enough to understand the reasoning behind its choices because... ya know... instincts.
```

## [reddit_eli5] −0−  ·  крупная 11.97 → 13.72 (+14.6%)  ·  снято 46 символов  ·  урон 3

*судья: Systematic tokenization artifacts (spaces before commas, periods, and around clitics) occur throughout the text.*

**до:**

```text
It does , but it is a great conductor . Even though it is reflective , being in a 300f environment , it will reach 300f , that is why what 's in the foil still cooks . But aluminum foil is very thin and very conductive , so heat travels quickly through it . If you touch a chunk of aluminum , it feels cold because it is pulling heat out of you and distributing it throughout its mass . Well aluminum foil acts like a radiator . It comes out and has heat , but cooler air touches it and heat always goes hot to less hot , so that little bit of energy it has gets given almost instantly to the air molecules it touches . It 's so thin that it has massive surface area relative to it 's mass , so it just gives the heat quickly to the air , as quickly as it pulls the heat out of your finger . This is why we make radiators and heat sinks and so on out of aluminum , because it gives heat to the air very easily when there is a lot of surface area . So look at it like this , a potato weighs a pound and is 300f , so let 's say it has 500 joules of energy and is an insulator , so it only releases 5 joules per second to the 65f air around it , so it takes 85 seconds for it to cool to 65f(way off of course but it 's just an example ) . The aluminum foil is maybe .001 pounds and 300f right when it comes out so it has 0.5 joules of energy . It 's a conductor and has a lot of surface area so it releases 25 joules per second . So it only takes 0.02 seconds of contact with the 65f air to lose that heat energy and reach 65f . Again , way off in the numbers , but the principle is there . Hope that helps . Edit : There seems to be some confusion on here , aluminum is a good conductor of both heat and electricity . If you Google it wiki confims . I 'm unsure why so many people think it 's a poor conductor of heat .
```

**после:**

```text
It does, but it is a great conductor. Even though it is reflective, being in a 300f environment, it will reach 300f, that is why what's in the foil still cooks. But aluminum foil is very thin and very conductive, so heat travels quickly through it. If you touch a chunk of aluminum, it feels cold because it is pulling heat out of you and distributing it throughout its mass. Well aluminum foil acts like a radiator. It comes out and has heat, but cooler air touches it and heat always goes hot to less hot, so that little bit of energy it has gets given almost instantly to the air molecules it touches. It's so thin that it has massive surface area relative to it's mass, so it just gives the heat quickly to the air, as quickly as it pulls the heat out of your finger. This is why we make radiators and heat sinks and so on out of aluminum, because it gives heat to the air very easily when there is a lot of surface area. So look at it like this, a potato weighs a pound and is 300f, so let's say it has 500 joules of energy and is an insulator, so it only releases 5 joules per second to the 65f air around it, so it takes 85 seconds for it to cool to 65f(way off of course but it's just an example). The aluminum foil is maybe.001 pounds and 300f right when it comes out so it has 0.5 joules of energy. It's a conductor and has a lot of surface area so it releases 25 joules per second. So it only takes 0.02 seconds of contact with the 65f air to lose that heat energy and reach 65f. Again, way off in the numbers, but the principle is there. Hope that helps. Edit: There seems to be some confusion on here, aluminum is a good conductor of both heat and electricity. If you Google it wiki confims. I'm unsure why so many people think it's a poor conductor of heat.
```

## [reddit_eli5] −0−  ·  крупная 11.92 → 12.41 (+4.2%)  ·  снято 44 символов  ·  урон 3

*судья: Systematic tokenizer damage with spaces added before and inside punctuation throughout the text.*

**до:**

```text
Before babies start to use ' language ' they form speech through babble . Basically they play with producing noise through open and closed mouths . These are the easiest utterances for developmental babies to make as they explore sounds . Think of the open and closed mouths like binary code : 0 = the open vowel sound , like ' aaaaaa ' . Then 1 ( or in our case a closed mouth ) - = the closed consonant sound , like ' m ' or ' p ' . Try it . Close your mouth as if you were making the beginning of a ' m ' or ' p ' word , then just open your mouth and push through some air .... You're basically making the ' ma ' and ' pa ' sounds . ' B ' , ' D ' , and ' P ' are pretty much interchangeable as closed consonants . Hence the variations found : baba , abba , papa , dada . etc . So , these are the basic building blocks of vocalisation , so they are the ones babies use . The simplest form for the human mouth . Another interesting question is that these pre - linguistic babies have been the ones to give birth to language in the ' naming ' of mother and father . Why did we follow their lead ? Maybe the simple naming of ' poo ' and ' wee ' in many languages follow a similar explanation . I think grandparents also have similar binary forms in many languages too , like in Hindi . Paternal Grandfather — दादा ( daadaa ) Paternal Grandmother — दादी ( daadii ) Maternal Grandfather — नाना ( naanaa ) Maternal Grandmother — नानी ( naanii )
```

**после:**

```text
Before babies start to use ' language ' they form speech through babble. Basically they play with producing noise through open and closed mouths. These are the easiest utterances for developmental babies to make as they explore sounds. Think of the open and closed mouths like binary code: 0 = the open vowel sound, like ' aaaaaa '. Then 1 (or in our case a closed mouth)-= the closed consonant sound, like ' m ' or ' p '. Try it. Close your mouth as if you were making the beginning of a ' m ' or ' p ' word, then just open your mouth and push through some air.... You're basically making the ' ma ' and ' pa ' sounds. ' B ', ' D ', and ' P ' are pretty much interchangeable as closed consonants. Hence the variations found: baba, abba, papa, dada. etc. So, these are the basic building blocks of vocalisation, so they are the ones babies use. The simplest form for the human mouth. Another interesting question is that these pre-linguistic babies have been the ones to give birth to language in the ' naming ' of mother and father. Why did we follow their lead? Maybe the simple naming of ' poo ' and ' wee ' in many languages follow a similar explanation. I think grandparents also have similar binary forms in many languages too, like in Hindi. Paternal Grandfather — दादा (daadaa) Paternal Grandmother — दादी (daadii) Maternal Grandfather — नाना (naanaa) Maternal Grandmother — नानी (naanii)
```

## [reddit_eli5] −0−  ·  крупная 14.45 → 17.64 (+22.1%)  ·  снято 42 символов  ·  урон 3

*судья: Systematic tokenizer damage with spaces added before punctuation and split clitics throughout.*

**до:**

```text
When you drink , alcohol affects your brain disproportionately in still unclear ways . It seems to have patterns , and is a depressant , but it 's different for everyone . It may make you angry , sad , or happy . It make you euphoric or perhaps lose your inhibitions . It may make you want to call your lost love , or punch your friend in the mouth . It may also affect the process in your brain that stores long term memories . While we do n't know everything , we know it has some sort of genetic and health component . We know that it happens more often in women , and it happens more often to people who drink way too much way too fast . It also happens to alcoholics very often , which is why people who black out should probably dial back their drinking , even if you do n't feel dependent . This is n't part of the question , but it seems that perhaps it 's something that should be addressed . Alcohol has a special place in human hearts . However it 's still a very dangerous drug . It 's not meant to be abused , and many people every year find themselves in very compromised positions because they do n't exactly understand what they are doing . They do n't place alcohol in the same category as other intoxicants . If you find yourself blacking out often , it 's only a matter of time before you find yourself in a position that you are n't going to be happy with . So dial it back , eat more , pace yourself .
```

**после:**

```text
When you drink, alcohol affects your brain disproportionately in still unclear ways. It seems to have patterns, and is a depressant, but it's different for everyone. It may make you angry, sad, or happy. It make you euphoric or perhaps lose your inhibitions. It may make you want to call your lost love, or punch your friend in the mouth. It may also affect the process in your brain that stores long term memories. While we don't know everything, we know it has some sort of genetic and health component. We know that it happens more often in women, and it happens more often to people who drink way too much way too fast. It also happens to alcoholics very often, which is why people who black out should probably dial back their drinking, even if you don't feel dependent. This isn't part of the question, but it seems that perhaps it's something that should be addressed. Alcohol has a special place in human hearts. However it's still a very dangerous drug. It's not meant to be abused, and many people every year find themselves in very compromised positions because they don't exactly understand what they are doing. They don't place alcohol in the same category as other intoxicants. If you find yourself blacking out often, it's only a matter of time before you find yourself in a position that you aren't going to be happy with. So dial it back, eat more, pace yourself.
```

## [reddit_eli5] −00  ·  крупная 14.11 → 18.76 (+32.9%)  ·  снято 41 символов  ·  урон 3

*судья: Systematic tokenizer damage has introduced spaces before punctuation and inside clitics throughout the text.*

**до:**

```text
Every one of them are almost completely disconnected from the others , so trying to think of them as a whole will make it really hard to get it . They all take place in different worlds , with different characters and different mythologies . They do have a few common elements , but they 're usually not very story - relevant so do n't get too stuck on them . Essentially , every Final Fantasy game should be thought of as a stand - alone JRPG . Final Fantasy built its fame on some of its very popular ( and for good reason ) earlier titles . I could be wrong , but I gather that 7 is the game that built the massive fame it now has , although it was still great before 7 came around ; 6 has a pretty die - hard fanbase . The general consensus is that it 's been a very long time since the last " good " final fantasy game was released , many arguing that 10 was the last title in the series that was faithful to the original quality and spirit of the franchise , but there 's some dispute on that . It seems that the general consensus is that Final Fantasy 6 - 10 were the games produced in the ' golden age ' of final fantasy , but there 's quite a lot of dispute even among those titles . TLDR : They 're all different games , and the old ones made them famous enough that there 's a lot of die - hard fans that still talk about them . Mix that with newer fans from the newer games and you get a lot of mixed messages .
```

**после:**

```text
Every one of them are almost completely disconnected from the others, so trying to think of them as a whole will make it really hard to get it. They all take place in different worlds, with different characters and different mythologies. They do have a few common elements, but they're usually not very story-relevant so don't get too stuck on them. Essentially, every Final Fantasy game should be thought of as a stand-alone JRPG. Final Fantasy built its fame on some of its very popular (and for good reason) earlier titles. I could be wrong, but I gather that 7 is the game that built the massive fame it now has, although it was still great before 7 came around; 6 has a pretty die-hard fanbase. The general consensus is that it's been a very long time since the last " good " final fantasy game was released, many arguing that 10 was the last title in the series that was faithful to the original quality and spirit of the franchise, but there's some dispute on that. It seems that the general consensus is that Final Fantasy 6-10 were the games produced in the ' golden age ' of final fantasy, but there's quite a lot of dispute even among those titles. TLDR: They're all different games, and the old ones made them famous enough that there's a lot of die-hard fans that still talk about them. Mix that with newer fans from the newer games and you get a lot of mixed messages.
```

## [wikihow] −0−  ·  крупная 12.41 → 11.71 (-5.6%)  ·  снято 41 символов  ·  урон 1

*судья: Minor punctuation artifact from list scraping/concatenation (stray semicolons and leading commas between paragraphs).*

**до:**

```text
There is not enough scientific evidence to determine how much acai juice makes a helpful, safe dosage, but it is recommended that you start small. Straight acai juice can upset stomachs that are not used to it. Measure 2 tablespoons (30 milliliters) of acai juice into a small glass. Consume 1 ounce of acai juice once a day for three to four days. If you do not notice any negative side effects, increase the amount to 6 tablespoons (90 milliliters) each serving, limiting yourself to one or two servings per day.;
, Diluting the acai juice can make it easier on your taste buds and on your stomach. Mix 2 teaspoons (10 milliliters) of chilled acai juice into 8 ounces (250 milliliters) of cold water. Drink the mixture immediately to prevent it from separating. You can gradually add more acai juice, up to 6 tablespoons (90 milliliters), depending on your taste preferences.

, Acai juice tastes especially good with other berry juices. Mix 2 teaspoons to 6 tablespoons (10 to 90 milliliters) of acai juice into an 8-ounce (250-milliliter) glass of mixed berry juice, blackberry juice, or blueberry juice. Acai also tastes great when mixed into pomegranate juice or mango juice.

, While the acai juice won't make your soda any healthier, you will still receive the benefits of acai juice by mixing the two. Add 2 teaspoons (10 milliliters) of acai juice to a standard cola. Standard colas work better than lemon-lime sodas or other flavored drinks. The strong flavor of the cola can mask the taste of the acai if you do not care for it, and the acai will add a hint of berry taste to the cola to create greater depth of flavor.

, Milk absorbs other flavors remarkably well, and the chocolate undertones of the acai berry make it a healthier substitution for chocolate milk. Adding 2 teaspoons to 2 tablespoons (10 to 30 milliliters) of acai juice can really create a strong flavor. If you cannot drink milk or choose to stay away from dairy for diet or health reasons, you can also add the same amount of acai juice to soy milk or almond milk.

, The potency of acai juice can compete with the strong taste of coffee, so you only need to add about 1 or 2 tablespoons (15 to 30 milliliters) to get a strong effect. The berry flavor creates a unique coffee experience, while the chocolate undertones make the taste just familiar enough to work with coffee. Acai juice is also a healthier choice than standard cream and sugar.

, Just 1 tablespoon (15 milliliters) of acai juice in an 8-ounce (250-milliliter) glass of your favorite tea can completely transform your beverage. You can mix acai juice into black or green tea, much like you would mix in raspberry or lemon. Acai juice goes especially well with white tea and fruity herbal teas, however, especially those that are blueberry, strawberry, or mango flavored.

, Try blending together 1/4 cup (60 milliliters) of acai juice with 1 cup (250 milliliters) of milk, 1 banana, 1/4 cup (56.7 grams) of frozen strawberries, 1 tablespoon (14.15 grams) of butter, and four or five ice cubes. You could also simply add 2 to 4 tablespoons (30 to 60 milliliters) of acai juice to your favorite smoothie recipe, especially if the smoothie has a banana, apple, or pear base and uses other berries like blueberries, raspberries, strawberries, and blackberries.


Alternatively, you can blend 2 tablespoons (30 milliliters) of the juice with vanilla or chocolate ice cream to create a simple acai shake.

, Instead of reaching for the caramel or chocolate syrup, pour 1 teaspoon to 1 tablespoon (milliliters) of acai juice over plain vanilla, chocolate, or strawberry ice cream to add a boost of nutrition an otherwise unhealthy treat.

, Simply pour the acai juice into a popsicle mold and freeze until it hardens to create an easy, healthy treat to enjoy during hot weather. A popsicle made with nothing but acai juice may be a bit potent for some tastes, however, so you can also dilute 1/4 cup (60 milliliters) of juice with equal parts water or mix in another juice, like orange, grape, or mango, before freezing.

, Acai juice is not acidic enough on its own to break down meats, so you may need to mix 1/4 cup (60 milliliters) of acai juice with about 2 tablespoons (30 milliliters) of lime juice or apple cider vinegar, along with 1/4 cup (60 milliliters) of olive oil. You can add other spices and seasonings, or you can leave the marinade as is since the acai juice will thoroughly flavor the meat. Acai-based marinades work best with pork and chicken.

, Ham is one meat that tastes notably good with a glaze. While glazes made of pineapple and orange tend to be the most popular, an acai glaze packs a nutritious punch while bestowing a truly unique flavor to the ham. Mix 1/4 cup (60 milliliters) of acai juice with 1 cup (225 grams) of brown sugar. Spread the mixture over the ham and bake it in the oven as you usually would.

, Create a simple vinaigrette 2/3 cup olive oil (167 milliliters) with 1/3 cup (83 milliliters) red wine vinegar or apple cider vinegar. Add 2 to 3 tablespoons (30 to 45 milliliters) of acai juice and a pinch of salt and pepper to transform your dressing into an acai berry vinaigrette. Drizzle the dressing over a simple loose-leaf salad.

, When preparing the filling for a blueberry pie, strawberry shortcake, or peach cobbler, mix in 2 teaspoons to 1 tablespoon (10 to 15 milliliters) of acai juice. The juice will not affect the consistency much, but it will add a little extra complexity to the flavor as well as added nutrition.
```

**после:**

```text
There is not enough scientific evidence to determine how much acai juice makes a helpful, safe dosage, but it is recommended that you start small. Straight acai juice can upset stomachs that are not used to it. Measure 2 tablespoons (30 milliliters) of acai juice into a small glass. Consume 1 ounce of acai juice once a day for three to four days. If you do not notice any negative side effects, increase the amount to 6 tablespoons (90 milliliters) each serving, limiting yourself to one or two servings per day. Diluting the acai juice can make it easier on your taste buds and on your stomach. Mix 2 teaspoons (10 milliliters) of chilled acai juice into 8 ounces (250 milliliters) of cold water. Drink the mixture immediately to prevent it from separating. You can gradually add more acai juice, up to 6 tablespoons (90 milliliters), depending on your taste preferences. Acai juice tastes especially good with other berry juices. Mix 2 teaspoons to 6 tablespoons (10 to 90 milliliters) of acai juice into an 8-ounce (250-milliliter) glass of mixed berry juice, blackberry juice, or blueberry juice. Acai also tastes great when mixed into pomegranate juice or mango juice. While the acai juice won't make your soda any healthier, you will still receive the benefits of acai juice by mixing the two. Add 2 teaspoons (10 milliliters) of acai juice to a standard cola. Standard colas work better than lemon-lime sodas or other flavored drinks. The strong flavor of the cola can mask the taste of the acai if you do not care for it, and the acai will add a hint of berry taste to the cola to create greater depth of flavor. Milk absorbs other flavors remarkably well, and the chocolate undertones of the acai berry make it a healthier substitution for chocolate milk. Adding 2 teaspoons to 2 tablespoons (10 to 30 milliliters) of acai juice can really create a strong flavor. If you cannot drink milk or choose to stay away from dairy for diet or health reasons, you can also add the same amount of acai juice to soy milk or almond milk. The potency of acai juice can compete with the strong taste of coffee, so you only need to add about 1 or 2 tablespoons (15 to 30 milliliters) to get a strong effect. The berry flavor creates a unique coffee experience, while the chocolate undertones make the taste just familiar enough to work with coffee. Acai juice is also a healthier choice than standard cream and sugar. Just 1 tablespoon (15 milliliters) of acai juice in an 8-ounce (250-milliliter) glass of your favorite tea can completely transform your beverage. You can mix acai juice into black or green tea, much like you would mix in raspberry or lemon. Acai juice goes especially well with white tea and fruity herbal teas, however, especially those that are blueberry, strawberry, or mango flavored. Try blending together 1/4 cup (60 milliliters) of acai juice with 1 cup (250 milliliters) of milk, 1 banana, 1/4 cup (56.7 grams) of frozen strawberries, 1 tablespoon (14.15 grams) of butter, and four or five ice cubes. You could also simply add 2 to 4 tablespoons (30 to 60 milliliters) of acai juice to your favorite smoothie recipe, especially if the smoothie has a banana, apple, or pear base and uses other berries like blueberries, raspberries, strawberries, and blackberries. Alternatively, you can blend 2 tablespoons (30 milliliters) of the juice with vanilla or chocolate ice cream to create a simple acai shake. Instead of reaching for the caramel or chocolate syrup, pour 1 teaspoon to 1 tablespoon (milliliters) of acai juice over plain vanilla, chocolate, or strawberry ice cream to add a boost of nutrition an otherwise unhealthy treat. Simply pour the acai juice into a popsicle mold and freeze until it hardens to create an easy, healthy treat to enjoy during hot weather. A popsicle made with nothing but acai juice may be a bit potent for some tastes, however, so you can also dilute 1/4 cup (60 milliliters) of juice with equal parts water or mix in another juice, like orange, grape, or mango, before freezing. Acai juice is not acidic enough on its own to break down meats, so you may need to mix 1/4 cup (60 milliliters) of acai juice with about 2 tablespoons (30 milliliters) of lime juice or apple cider vinegar, along with 1/4 cup (60 milliliters) of olive oil. You can add other spices and seasonings, or you can leave the marinade as is since the acai juice will thoroughly flavor the meat. Acai-based marinades work best with pork and chicken. Ham is one meat that tastes notably good with a glaze. While glazes made of pineapple and orange tend to be the most popular, an acai glaze packs a nutritious punch while bestowing a truly unique flavor to the ham. Mix 1/4 cup (60 milliliters) of acai juice with 1 cup (225 grams) of brown sugar. Spread the mixture over the ham and bake it in the oven as you usually would. Create a simple vinaigrette 2/3 cup olive oil (167 milliliters) with 1/3 cup (83 milliliters) red wine vinegar or apple cider vinegar. Add 2 to 3 tablespoons (30 to 45 milliliters) of acai juice and a pinch of salt and pepper to transform your dressing into an acai berry vinaigrette. Drizzle the dressing over a simple loose-leaf salad. When preparing the filling for a blueberry pie, strawberry shortcake, or peach cobbler, mix in 2 teaspoons to 1 tablespoon (10 to 15 milliliters) of acai juice. The juice will not affect the consistency much, but it will add a little extra complexity to the flavor as well as added nutrition.
```

## [finance] −−−  ·  крупная 13.67 → 14.98 (+9.6%)  ·  снято 41 символов  ·  урон 2

*судья: Missing bullet points/enumeration under a colon line, but text is otherwise clean and readable.*

**до:**

```text
An alternative to a savings account is a money market account.  Not a bank "Money Market" account which pays effectively the same silly rate as a savings account, but an actual Money Market investment account.  You can even write checks against some Money Market investment accounts. I have several accounts worth about 13,000 each.  Originally, my "emergency fund" was in a CD ladder.  I started experimenting with two different Money market investment accounts recently.  Here's my latest results: August returns on various accounts worth about $13k:  - Discover Bank CD:  $13.22  - Discover Bank CD:  $13.27  - Discover Bank CD:  $13.20  - Discover Savings:  $13.18  - Credit Union "Money Market" Savings account:  $1.80  - Fidelity Money Market Account (SPAXX):  $7.35  - Vanguard Money market Account (VMFXX):  $10.86 The actual account values are approximate.  The Fidelity Money Market Account holds the least value, and the Credit Union account by far the most. The result of the experiment is that as the CDs mature, I'll be moving out of Discover Bank into the Vanguard Money Market account. You can put your money into more traditional equities mutual fund.  The danger with them is the stock market may drop big the day before you want to make your withdrawl... and then you don't have the down payment for your house anymore.  But a well chosen mutual fund will yield better. There are 3 ways a mutual fund increase in value: Here's how three of my mutual funds did in the past month...  adjusted as if the accounts had started off to be worth about $13,000: Those must vary wildly month-to-month. By the way, if you look up the ticker symbols, VASGX is a Vanguard "Fund of Funds" -- it invests not 100% in the stock market, but 80% in the stock market and 20% in bonds.  VSMGX is a 60/40 split.  Interesting that VASGX grew less than VSMGX...but that assumes my spreadsheet is correct.  Most of my mutual funds pay dividends and capital gains once or twice a year.  I don't think any pay in August.
```

**после:**

```text
An alternative to a savings account is a money market account. Not a bank "Money Market" account which pays effectively the same silly rate as a savings account, but an actual Money Market investment account. You can even write checks against some Money Market investment accounts. I have several accounts worth about 13,000 each. Originally, my "emergency fund" was in a CD ladder. I started experimenting with two different Money market investment accounts recently. Here's my latest results: August returns on various accounts worth about $13k:-Discover Bank CD: $13.22-Discover Bank CD: $13.27-Discover Bank CD: $13.20-Discover Savings: $13.18-Credit Union "Money Market" Savings account: $1.80-Fidelity Money Market Account (SPAXX): $7.35-Vanguard Money market Account (VMFXX): $10.86 The actual account values are approximate. The Fidelity Money Market Account holds the least value, and the Credit Union account by far the most. The result of the experiment is that as the CDs mature, I'll be moving out of Discover Bank into the Vanguard Money Market account. You can put your money into more traditional equities mutual fund. The danger with them is the stock market may drop big the day before you want to make your withdrawl... and then you don't have the down payment for your house anymore. But a well chosen mutual fund will yield better. There are 3 ways a mutual fund increase in value: Here's how three of my mutual funds did in the past month... adjusted as if the accounts had started off to be worth about $13,000: Those must vary wildly month-to-month. By the way, if you look up the ticker symbols, VASGX is a Vanguard "Fund of Funds" -- it invests not 100% in the stock market, but 80% in the stock market and 20% in bonds. VSMGX is a 60/40 split. Interesting that VASGX grew less than VSMGX...but that assumes my spreadsheet is correct. Most of my mutual funds pay dividends and capital gains once or twice a year. I don't think any pay in August.
```

## [reddit_eli5] −−−  ·  крупная 15.37 → 15.77 (+2.6%)  ·  снято 41 символов  ·  урон 3

*судья: Systematic whitespace insertion before punctuation marks throughout the document due to tokeniser processing.*

**до:**

```text
Programs are essentially a series of instructions in the form of 1s and 0s . 1 represents high voltage and 0 represents low voltage . A programmable circuit consists of a load of transistors that make up logic gates . Logic gates can change the 1 to a 0 and vice versa . To visualise this , think of passing a message up through a chain of your friends , who are standing in these positions : ( behold my magnificent ASCII art ) D | C / \ A B \ / You * you give two pieces of paper with a 1 on them to friend A , and two pieces of paper with a 0 on them to friend B. * friend A has been told to give a 1 to friend C if he receives two 1s , and a 0 if he receives anything else . Since you have given him two 1s , he passes on a 1 to friend C. * friend B has been told to pass on a 1 if either of the pieces of paper he receives have a 1 on them , otherwise a 0 . Since you gave him two 0s , he passes on a 0 . * friend C will pass on a 1 to friend D if only one of the pieces of paper he has has a 1 on them . This is true , so he gives a 1 to friend D. * friend D has been told to jump if he gets a 1 and crouch if he gets a 0 . He gets a 1 , so he jumps . Some examples of logic gates : * a NOT gate inverts the signal * an OR gate will produce a 1 if any of the inputs are 1 ( friend B ) * an AND gate will produce a 1 if both of the inputs are 1 ( friend A ) * a XOR gate will produce a 1 if only one but not all of the inputs is 1 ( friend C ) These circuits are connected to things called actuators , which produce the output of the circuit . In the above example , friend D is the actuator , since he produces the output ( either a jump or a crouch ) You can begin to see how this comes together in larger circuits ( hopefully ) . To give you a sense of how complicated these circuits can get , the CPU in your computer has millions of transistors . It has been a while since I studied this so some of this may be slightly wrong .
```

**после:**

```text
Programs are essentially a series of instructions in the form of 1s and 0s. 1 represents high voltage and 0 represents low voltage. A programmable circuit consists of a load of transistors that make up logic gates. Logic gates can change the 1 to a 0 and vice versa. To visualise this, think of passing a message up through a chain of your friends, who are standing in these positions: (behold my magnificent ASCII art) D | C / \ A B \ / You * you give two pieces of paper with a 1 on them to friend A, and two pieces of paper with a 0 on them to friend B. * friend A has been told to give a 1 to friend C if he receives two 1s, and a 0 if he receives anything else. Since you have given him two 1s, he passes on a 1 to friend C. * friend B has been told to pass on a 1 if either of the pieces of paper he receives have a 1 on them, otherwise a 0. Since you gave him two 0s, he passes on a 0. * friend C will pass on a 1 to friend D if only one of the pieces of paper he has has a 1 on them. This is true, so he gives a 1 to friend D. * friend D has been told to jump if he gets a 1 and crouch if he gets a 0. He gets a 1, so he jumps. Some examples of logic gates: * a NOT gate inverts the signal * an OR gate will produce a 1 if any of the inputs are 1 (friend B) * an AND gate will produce a 1 if both of the inputs are 1 (friend A) * a XOR gate will produce a 1 if only one but not all of the inputs is 1 (friend C) These circuits are connected to things called actuators, which produce the output of the circuit. In the above example, friend D is the actuator, since he produces the output (either a jump or a crouch) You can begin to see how this comes together in larger circuits (hopefully). To give you a sense of how complicated these circuits can get, the CPU in your computer has millions of transistors. It has been a while since I studied this so some of this may be slightly wrong.
```

## [reddit_eli5] −00  ·  крупная 14.65 → 14.67 (+0.2%)  ·  снято 41 символов  ·  урон 3

*судья: Systematic tokenization artifacts (spaces before punctuation and inside contractions) occur throughout the text.*

**до:**

```text
This comes down to your circadian rhythm , usually . From a young age , most of us get into the habit of sleeping at night and staying awake during the day , or maybe having a short nap in the afternoon . This is usually linked to Zeitgebers ( German for ' time - givers ' ) which are basically external cues . The most common of these are external light . Essentially , we train ourselves into having a proper sleep at night and only a nap during the day . It 's surprisingly easy and simultaneously difficult to fuck up your circadian rhythm . Think about how often you stay up all night to watch a show or play a game or read a book or dance while wasted . Basically anything that throws you off your rhythm can fuck up your cycle but this is usually short term , and your body is pretty good at getting itself back on track if you 're not fighting it , thanks to hormones . Hormones help maintain your circadian rhythm . Melatonin and cortisol both affect your sleep cycle . Example : Melatonin , the ' sleepy ' hormone , is usually produced at night , and causes drowsiness and drops your body temperature , both of which will obviously make you want to sleep more . Melatonin usually starts getting produced between 8 pm and 9 pm , spiking around 3 am and 4 am , finishing about 7 am or 8 am . This helps us sleep through the night . There 's usually a smaller spike around lunch time , but this does n't last as long , thus leading to a simple nap rather than a full on sleep . It 's also why it 's easier to slip back into a ' normal ' rhythm after working nights than it is to start doing nights after working days . Fucking hormones !
```

**после:**

```text
This comes down to your circadian rhythm, usually. From a young age, most of us get into the habit of sleeping at night and staying awake during the day, or maybe having a short nap in the afternoon. This is usually linked to Zeitgebers (German for ' time-givers ') which are basically external cues. The most common of these are external light. Essentially, we train ourselves into having a proper sleep at night and only a nap during the day. It's surprisingly easy and simultaneously difficult to fuck up your circadian rhythm. Think about how often you stay up all night to watch a show or play a game or read a book or dance while wasted. Basically anything that throws you off your rhythm can fuck up your cycle but this is usually short term, and your body is pretty good at getting itself back on track if you're not fighting it, thanks to hormones. Hormones help maintain your circadian rhythm. Melatonin and cortisol both affect your sleep cycle. Example: Melatonin, the ' sleepy ' hormone, is usually produced at night, and causes drowsiness and drops your body temperature, both of which will obviously make you want to sleep more. Melatonin usually starts getting produced between 8 pm and 9 pm, spiking around 3 am and 4 am, finishing about 7 am or 8 am. This helps us sleep through the night. There's usually a smaller spike around lunch time, but this doesn't last as long, thus leading to a simple nap rather than a full on sleep. It's also why it's easier to slip back into a ' normal ' rhythm after working nights than it is to start doing nights after working days. Fucking hormones!
```

## [reddit_eli5] −0−  ·  крупная 10.65 → 13.52 (+26.9%)  ·  снято 37 символов  ·  урон 3

*судья: Systematic tokenizer corruption splits clitics and adds spaces around punctuation throughout the entire text.*

**до:**

```text
Let 's say you buy a house for $ 200,000 . Since you do n't have that much money lying around , you get a loan from a bank caledl a mortgage . You might wind up putting in $ 20,000 of your own money , and borrowing $ 180,000 at 5 % for 30 years . Bank A loans you the money , because they know if you ever ca n't pay , they can sell the house and come out ahead . Fast forward 10 years . You house is now worth $ 250,000 , and you 've paid your mortgage down so you only owe $ 150,000 . The difference between what the house is worth and what you owe is called * equity* ... in your case , you have $ 100,000 in equity . You get a loan against that equity , a * second mortgage * . Bank B is willing to give you a loan , because if you ca n't pay it off , they can sell your house , pay off Bank A , and have enough left over for them . Usually the terms are worse , so Bank B is n't going to give you as good as terms . You might be able to borrow $ 50,000 at 7 % for 10 years . But now you have to pay Bank B about $ 600 a month , on top of already paying Bank A the $ 1000 a month . Usually getting a second mortgage is a bad idea , but if you are desperate and have to raise money quickly , it is an option .
```

**после:**

```text
Let's say you buy a house for $ 200,000. Since you don't have that much money lying around, you get a loan from a bank caledl a mortgage. You might wind up putting in $ 20,000 of your own money, and borrowing $ 180,000 at 5% for 30 years. Bank A loans you the money, because they know if you ever can't pay, they can sell the house and come out ahead. Fast forward 10 years. You house is now worth $ 250,000, and you've paid your mortgage down so you only owe $ 150,000. The difference between what the house is worth and what you owe is called * equity*... in your case, you have $ 100,000 in equity. You get a loan against that equity, a * second mortgage *. Bank B is willing to give you a loan, because if you can't pay it off, they can sell your house, pay off Bank A, and have enough left over for them. Usually the terms are worse, so Bank B isn't going to give you as good as terms. You might be able to borrow $ 50,000 at 7% for 10 years. But now you have to pay Bank B about $ 600 a month, on top of already paying Bank A the $ 1000 a month. Usually getting a second mortgage is a bad idea, but if you are desperate and have to raise money quickly, it is an option.
```

## [reddit_eli5] −00  ·  крупная 14.11 → 14.23 (+0.9%)  ·  снято 36 символов  ·  урон 3

*судья: Systematic tokenizer damage splitting clitics, apostrophes, and punctuation with spaces throughout the text.*

**до:**

```text
They do n't . A woman 's biological cycle is not clockwork . The whole " 28 days " thing is an average . Some women have 26 day cycles . Some 30 . Some women float between 25 and 32 . Some are even more random . And even women who start their period every 28 days like clockwork rarely start it at the same time on that day - noon one time , 8 am the next time , 8 pm the next time .. and so on . If you have two women , one with a 29 day cycle and one with a 26 day cycle , then at any given point in time their cycles will appear to be getting closer together . It 's .. kinda like the turn signals on a car . Ever watched two turn signals not anywhere close to each other slowly start to sync up , then they appear to be blinking at exactly the same time ? But what happens after that ? They start driving apart again , because they are n't actually syncing up with each other , they just happen to be in a part of the cycle where they appear together . Same thing with women , only it takes a month to get a data point . Combine the length of time needed to gather data with confirmation bias , and you 'll have women swearing their cycles are getting closer when they 're actually getting farther apart . Because they 're starting to sync up with the preceding one .
```

**после:**

```text
They don't. A woman's biological cycle is not clockwork. The whole " 28 days " thing is an average. Some women have 26 day cycles. Some 30. Some women float between 25 and 32. Some are even more random. And even women who start their period every 28 days like clockwork rarely start it at the same time on that day-noon one time, 8 am the next time, 8 pm the next time.. and so on. If you have two women, one with a 29 day cycle and one with a 26 day cycle, then at any given point in time their cycles will appear to be getting closer together. It's.. kinda like the turn signals on a car. Ever watched two turn signals not anywhere close to each other slowly start to sync up, then they appear to be blinking at exactly the same time? But what happens after that? They start driving apart again, because they aren't actually syncing up with each other, they just happen to be in a part of the cycle where they appear together. Same thing with women, only it takes a month to get a data point. Combine the length of time needed to gather data with confirmation bias, and you'll have women swearing their cycles are getting closer when they're actually getting farther apart. Because they're starting to sync up with the preceding one.
```

## [reddit_eli5] −−−  ·  крупная 14.32 → 14.13 (-1.4%)  ·  снято 35 символов  ·  урон 3

*судья: Systematic tokenizer damage splits punctuation and clitics with spaces throughout the entire text.*

**до:**

```text
* They * actually experience less time , from * your * point of view . The important thing to remember is that the speed of light ( in a vacuum ) is always the same . Now imagine a simple type of clock , a beam of light bouncing between to equally spaced mirrors . * * * * * | | * * * * * Since the speed of light is the same , the time it takes to go from the top to the bottom is the same each time , one tick of the clock . Now imagine this clock is moving : * * * * * * * * * * * * * * * -- > \ / \ / \ \ / \ / \ * * * * * * * * * * * * * * * Since light * always * moves at the same speed , and the diagonal lines are longer , it takes longer for light to go from the top to the bottom , it has longer ticks . But , what if you are moving at the same speed as the moving mirror ? Then , the light will look like it 's going straight up and down again , and since light * always * moves at the same speed , the ticks will be back to their original size ! So if I am on the ground with my own clock , watching you run with your clock , I will see that it takes longer for each tick on your clock compared to my clock . Since both clocks are working properly , I must conclude that time itself is moving slower for you . Now for the * really * confusing part . From your point of view , light on your clock is moving straight up and down , but the light on my clock has to go backwards diagonally , so you will similarly conclude that time is moving more slowly for me ! And we would both be right ! This is why it is called the theory of * relativity * , because the relative point of view you are talking about makes a * huge * difference .
```

**после:**

```text
* They * actually experience less time, from * your * point of view. The important thing to remember is that the speed of light (in a vacuum) is always the same. Now imagine a simple type of clock, a beam of light bouncing between to equally spaced mirrors. * * * * * | | * * * * * Since the speed of light is the same, the time it takes to go from the top to the bottom is the same each time, one tick of the clock. Now imagine this clock is moving: * * * * * * * * * * * * * * * -- > \ / \ / \ \ / \ / \ * * * * * * * * * * * * * * * Since light * always * moves at the same speed, and the diagonal lines are longer, it takes longer for light to go from the top to the bottom, it has longer ticks. But, what if you are moving at the same speed as the moving mirror? Then, the light will look like it's going straight up and down again, and since light * always * moves at the same speed, the ticks will be back to their original size! So if I am on the ground with my own clock, watching you run with your clock, I will see that it takes longer for each tick on your clock compared to my clock. Since both clocks are working properly, I must conclude that time itself is moving slower for you. Now for the * really * confusing part. From your point of view, light on your clock is moving straight up and down, but the light on my clock has to go backwards diagonally, so you will similarly conclude that time is moving more slowly for me! And we would both be right! This is why it is called the theory of * relativity *, because the relative point of view you are talking about makes a * huge * difference.
```

## [reddit_eli5] −00  ·  крупная 13.07 → 15.73 (+20.3%)  ·  снято 34 символов  ·  урон 3

*судья: Systematic tokenization artifacts (spaces before punctuation and clitics) affect the entire text.*

**до:**

```text
There are many reasons Alexander won . First of all , Alexander was not outnumbered by as much as was first thought . 1)"Modern historians find Arrian 's count of six hundred thousand men highly unlikely . They argue that the logistics of fielding more than 100,000 soldiers in battle was extremely difficult at the time . Hans Delbrück gives an estimate as small as 25,000 , although most ( including Engels and Green ) estimate the total size of Darius ' army to be no larger than 100,000 at Issus,[4 ] including 11,000 cavalry,[2 ] 10,000 Persian Immortals , and 10,000 Greek mercenaries.[5 ] Warry estimates 108,000 in total . " 2 ) The Macedonians were better equipped for battle than the regular Persian troops . They were only matched by the Greek mercenaries fielded by Persia . Despite their being outnumbered much of the Persian army consisted of peltasts . Despite this , there was around an equal number of melee infantry available to both sides . 3 ) Companion cavalry were the best in the world at the time , the Persian horse stood no chance . 4 ) The hammer and anvil is the quintessential shock tactic of warfare . One force fixes a group , whilst another slams into the back of that group and causes mass amounts of destruction and confusion . It 's an extremely effective manuever in both the physical and psychological damage it can inflict . It should also be noted that ( generally ) the most casualties from a battle are when one side runs away . Numbers do not equal discipline , which is the most essential thing an army can have .
```

**после:**

```text
There are many reasons Alexander won. First of all, Alexander was not outnumbered by as much as was first thought. 1)"Modern historians find Arrian's count of six hundred thousand men highly unlikely. They argue that the logistics of fielding more than 100,000 soldiers in battle was extremely difficult at the time. Hans Delbrück gives an estimate as small as 25,000, although most (including Engels and Green) estimate the total size of Darius ' army to be no larger than 100,000 at Issus,[4] including 11,000 cavalry,[2] 10,000 Persian Immortals, and 10,000 Greek mercenaries.[5] Warry estimates 108,000 in total. " 2) The Macedonians were better equipped for battle than the regular Persian troops. They were only matched by the Greek mercenaries fielded by Persia. Despite their being outnumbered much of the Persian army consisted of peltasts. Despite this, there was around an equal number of melee infantry available to both sides. 3) Companion cavalry were the best in the world at the time, the Persian horse stood no chance. 4) The hammer and anvil is the quintessential shock tactic of warfare. One force fixes a group, whilst another slams into the back of that group and causes mass amounts of destruction and confusion. It's an extremely effective manuever in both the physical and psychological damage it can inflict. It should also be noted that (generally) the most casualties from a battle are when one side runs away. Numbers do not equal discipline, which is the most essential thing an army can have.
```

## [wikipedia] −−−  ·  крупная 13.00 → 14.21 (+9.4%)  ·  снято 16 символов  ·  урон 2

*судья: The document is mostly clean text and structured lists, but contains missing list content under headers and leftover wiki interwiki markup.*

**до:**

```text
The Portuguese football leagues are divided into divisions (divisões, singular – divisão). The top teams play in the Primeira Liga, named Liga NOS for sponsorship reasons. In each division, with rare exceptions, a team plays all other teams twice, once at home and once away. One can divide the competitions in professional and non-professional.
The Portuguese league and federation teams compete in Europe under UEFA, most notably in the UEFA Champions League, but also in the UEFA Cup, in the extinct Cup Winners' Cup and sometimes in the UEFA Intertoto Cup, European Super Cup and the extinct UEFA/Conmebol Intercontinental Cup (Toyota Cup). They can also compete in the FIFA Club World Cup, although until today no Portuguese team reached this recent competition. The teams also compete in a domestic cup competition each year, called Cup of Portugal (Taça de Portugal) and the winners play against the champions in the SuperCup Cândido de Oliveira.
Current hierarchical divisional breakdowns
Professional
Primeira Liga - First Division (18 teams)
Segunda Liga - Second Division (18 teams)
Non-professional
Campeonato de Portugal - Third Division (72 teams)
Portuguese District First Levels - Fourth Division
Portuguese District Second Levels - Fifth Division
Portuguese District Third Levels - Sixth Division
Portuguese District Fourth Levels - Seventh Division
Primeira Liga teams (2018–19 season)
Segunda Liga teams (2018–19 season)
Campeonato de Portugal teams (2018–19 season)
Serie A
Serie B
Serie C
Serie D
Notable extinct teams
Riopele
União de Lisboa
See also
Portuguese football competitions
Football in Portugal
Portuguese Footballer of the Year
Portugal
 
clubs
Football clubs
ru:Список футбольных клубов Португалии
```

**после:**

```text
The Portuguese football leagues are divided into divisions (divisões, singular – divisão). The top teams play in the Primeira Liga, named Liga NOS for sponsorship reasons. In each division, with rare exceptions, a team plays all other teams twice, once at home and once away. One can divide the competitions in professional and non-professional.
The Portuguese league and federation teams compete in Europe under UEFA, most notably in the UEFA Champions League, but also in the UEFA Cup, in the extinct Cup Winners' Cup and sometimes in the UEFA Intertoto Cup, European Super Cup and the extinct UEFA/Conmebol Intercontinental Cup (Toyota Cup). They can also compete in the FIFA Club World Cup, although until today no Portuguese team reached this recent competition. The teams also compete in a domestic cup competition each year, called Cup of Portugal (Taça de Portugal) and the winners play against the champions in the SuperCup Cândido de Oliveira.
Current hierarchical divisional breakdowns
Professional
Primeira Liga-First Division (18 teams)
Segunda Liga-Second Division (18 teams)
Non-professional
Campeonato de Portugal-Third Division (72 teams)
Portuguese District First Levels-Fourth Division
Portuguese District Second Levels-Fifth Division
Portuguese District Third Levels-Sixth Division
Portuguese District Fourth Levels-Seventh Division
Primeira Liga teams (2018–19 season)
Segunda Liga teams (2018–19 season)
Campeonato de Portugal teams (2018–19 season)
Serie A
Serie B
Serie C
Serie D
Notable extinct teams
Riopele
União de Lisboa
See also
Portuguese football competitions
Football in Portugal
Portuguese Footballer of the Year
Portugal clubs
Football clubs
ru:Список футбольных клубов Португалии
```

## [wikipedia] −0−  ·  крупная 15.44 → 13.30 (-13.9%)  ·  снято 13 символов  ·  урон 2

*судья: Minor LOSS (empty sections) and WEB category tags, but the main text is intact and readable.*

**до:**

```text
The Bruckner Expressway is a freeway in the borough of the Bronx in New York City. It carries Interstate 278 (I-278) and I-95 (and formerly I-878) from the Triborough Bridge to the south end of the New England Thruway at the Pelham Parkway interchange. The highway follows a mostly northeast–southwest alignment through the southern portion of the borough, loosely paralleling the course of the East River. It connects to several major freeways including the Bronx River Parkway, and at the Bruckner Interchange, it connects to the Cross Bronx Expressway, the Whitestone Expressway, and the Hutchinson River Parkway.

Route description

The expressway begins at the northern approach to the Triborough Bridge, where I-278 meets the southern end of I-87, here known as the Major Deegan Expressway. It heads to the northeast as an elevated highway, carrying the I-278 designation through the South Bronx. After , the Bruckner Expressway meets NY 895 (Sheridan Boulevard) and turns eastward to cross the Bronx River into the Soundview neighborhood. Here, the highway connects to the Bronx River Parkway at an interchange one block north of that road's official southern terminus. The Bruckner Expressway remains I-278 into the adjacent neighborhood of Castle Hill, where I-278 enters the west half of the complicated Bruckner Interchange and I-95 transitions from the Cross Bronx Expressway to the Bruckner.

In the east half of the junction, located on the opposite bank of Westchester Creek in Throggs Neck, the Bruckner Expressway (now designated I-95) intersects the Hutchinson River Parkway, I-295, and I-678. I-295 proceeds southeast from the Bruckner Interchange as the Cross Bronx Expressway Extension, while the Hutchinson Parkway and I-678 head to the north and south, respectively. The Bruckner eventually makes a turn to the north as well, connecting with I-695 in the process. The Hutchinson and the Bruckner follow mostly parallel alignments to Pelham Bay Park, where the Bruckner Expressway ends at an interchange with the Pelham Parkway. I-95 continues north from this point as the New England Thruway.

History
The Bruckner Expressway was a project envisioned by Robert Moses, who steered the Bruckner through the Soundview section of the Bronx, further altering the neighborhood after the 15-year construction of the Cross Bronx Expressway, which was completed in 1963. The Bruckner Expressway itself was completed in 1973, making it one of the last roads of the New York City Expressway system to be built.  It is named in honor of former Bronx Borough President and Congressman, Henry Bruckner (1871–1942), and was built on and over the roadway of Bruckner Boulevard (Originally called Eastern Boulevard). 

Unlike the Cross Bronx Expressway, which cut through the existing street grid, the Bruckner Expressway was built along the Bruckner Boulevard alignment (except at its western end, where the Bruckner Expressway and Major Deegan Expressway meet). Between Sheridan Boulevard and the eastern end of the Bruckner Expressway, the Bruckner Boulevard is the service road, except at the Bruckner Interchange, where Bruckner Boulevard passes underneath the flying junction. West of Sheridan Boulevard, Bruckner Boulevard is underneath the expressway, and extends past the expressway's western terminus for about , ending under the Third Avenue Bridge.

In 2019, the New York State Department of Transportation began a $1.1 billion project to rebuild parts of the Bruckner Expressway and improve traffic flow, which would reduce air and noise pollution. The project would rebuild the interchanges with both Sheridan Boulevard and Hunts Point Avenue, add a third lane in both directions of the Bruckner, and rebuild four bridges.

Exit list

References

Transportation in the Bronx
Expressways in New York City
Robert Moses projects
Interstate 95
Interstate 78
```

**после:**

```text
The Bruckner Expressway is a freeway in the borough of the Bronx in New York City. It carries Interstate 278 (I-278) and I-95 (and formerly I-878) from the Triborough Bridge to the south end of the New England Thruway at the Pelham Parkway interchange. The highway follows a mostly northeast–southwest alignment through the southern portion of the borough, loosely paralleling the course of the East River. It connects to several major freeways including the Bronx River Parkway, and at the Bruckner Interchange, it connects to the Cross Bronx Expressway, the Whitestone Expressway, and the Hutchinson River Parkway. Route description The expressway begins at the northern approach to the Triborough Bridge, where I-278 meets the southern end of I-87, here known as the Major Deegan Expressway. It heads to the northeast as an elevated highway, carrying the I-278 designation through the South Bronx. After, the Bruckner Expressway meets NY 895 (Sheridan Boulevard) and turns eastward to cross the Bronx River into the Soundview neighborhood. Here, the highway connects to the Bronx River Parkway at an interchange one block north of that road's official southern terminus. The Bruckner Expressway remains I-278 into the adjacent neighborhood of Castle Hill, where I-278 enters the west half of the complicated Bruckner Interchange and I-95 transitions from the Cross Bronx Expressway to the Bruckner. In the east half of the junction, located on the opposite bank of Westchester Creek in Throggs Neck, the Bruckner Expressway (now designated I-95) intersects the Hutchinson River Parkway, I-295, and I-678. I-295 proceeds southeast from the Bruckner Interchange as the Cross Bronx Expressway Extension, while the Hutchinson Parkway and I-678 head to the north and south, respectively. The Bruckner eventually makes a turn to the north as well, connecting with I-695 in the process. The Hutchinson and the Bruckner follow mostly parallel alignments to Pelham Bay Park, where the Bruckner Expressway ends at an interchange with the Pelham Parkway. I-95 continues north from this point as the New England Thruway. History
The Bruckner Expressway was a project envisioned by Robert Moses, who steered the Bruckner through the Soundview section of the Bronx, further altering the neighborhood after the 15-year construction of the Cross Bronx Expressway, which was completed in 1963. The Bruckner Expressway itself was completed in 1973, making it one of the last roads of the New York City Expressway system to be built. It is named in honor of former Bronx Borough President and Congressman, Henry Bruckner (1871–1942), and was built on and over the roadway of Bruckner Boulevard (Originally called Eastern Boulevard). Unlike the Cross Bronx Expressway, which cut through the existing street grid, the Bruckner Expressway was built along the Bruckner Boulevard alignment (except at its western end, where the Bruckner Expressway and Major Deegan Expressway meet). Between Sheridan Boulevard and the eastern end of the Bruckner Expressway, the Bruckner Boulevard is the service road, except at the Bruckner Interchange, where Bruckner Boulevard passes underneath the flying junction. West of Sheridan Boulevard, Bruckner Boulevard is underneath the expressway, and extends past the expressway's western terminus for about, ending under the Third Avenue Bridge. In 2019, the New York State Department of Transportation began a $1.1 billion project to rebuild parts of the Bruckner Expressway and improve traffic flow, which would reduce air and noise pollution. The project would rebuild the interchanges with both Sheridan Boulevard and Hunts Point Avenue, add a third lane in both directions of the Bruckner, and rebuild four bridges. Exit list References Transportation in the Bronx
Expressways in New York City
Robert Moses projects
Interstate 95
Interstate 78
```

## [cnn] −0−  ·  крупная 12.31 → 12.54 (+1.9%)  ·  снято 12 символов  ·  урон 1

*судья: Minor image caption residue and minor tokenizer/spacing artifacts around quotation marks.*

**до:**

```text
Jose Mourinho revealed he prepared his Chelsea players to deal with the threat of Manchester United's Marouane Fellaini all week - only to be told by a hotel doorman that the Belgian was not playing. Mourinho planned for Kurt Zouma to man mark Fellaini - with teenager Ruben Loftus-Cheek playing the 6ft 4in midfielder in training. But he feared his tactical master plan was set to be for nothing when the Chelsea boss was told that the Manchester United man was not playing by the doorman at the club's hotel. Marouane Fellaini (right) poses with his twin brother Mansour (right) during his days as an Everton player . Mansour (left) and Marouane (right) look on from the stands during Everton's game Leyton Orient in 2012 . Chelsea manager Jose Mourinho has admitted his side were confused as to whether Fellaini was playing . Mourinho, relaying his conversation with the doorman, said: 'Fellaini doesn't play because he was here to get tickets for the game from Eden Hazard. He was dressed in jeans, this and that'. ' (It) didn't smell well for me. I go to Google and I put 'Fellaini brother'. Images. 'So I go with the pictures to the doorman. I say 'hey, this one or this one?' And he looks 'this one'. 'He's the brother'.' Mourinho said the other Fellaini - his twin brother Mansour - posed for photos as the United player, who had been in the Blues boss' thoughts all week, did start the game at Stamford Bridge which Chelsea won 1-0 to move within two wins of the Premier League title. Indeed, Chelsea could claim the trophy as early as April 29 should they beat Arsenal and Leicester in their next two fixtures. In the end, Fellaini did play at Stamford Bridge, but was dealt with in midfield by Chelsea's Kurt Zouma (left) Zouma (right) had been given a specific tactical brief to stifle the threat of Fellaini (left) during the game .
```

**после:**

```text
Jose Mourinho revealed he prepared his Chelsea players to deal with the threat of Manchester United's Marouane Fellaini all week-only to be told by a hotel doorman that the Belgian was not playing. Mourinho planned for Kurt Zouma to man mark Fellaini-with teenager Ruben Loftus-Cheek playing the 6ft 4in midfielder in training. But he feared his tactical master plan was set to be for nothing when the Chelsea boss was told that the Manchester United man was not playing by the doorman at the club's hotel. Marouane Fellaini (right) poses with his twin brother Mansour (right) during his days as an Everton player. Mansour (left) and Marouane (right) look on from the stands during Everton's game Leyton Orient in 2012. Chelsea manager Jose Mourinho has admitted his side were confused as to whether Fellaini was playing. Mourinho, relaying his conversation with the doorman, said: 'Fellaini doesn't play because he was here to get tickets for the game from Eden Hazard. He was dressed in jeans, this and that'. ' (It) didn't smell well for me. I go to Google and I put 'Fellaini brother'. Images. 'So I go with the pictures to the doorman. I say 'hey, this one or this one?' And he looks 'this one'. 'He's the brother'.' Mourinho said the other Fellaini-his twin brother Mansour-posed for photos as the United player, who had been in the Blues boss' thoughts all week, did start the game at Stamford Bridge which Chelsea won 1-0 to move within two wins of the Premier League title. Indeed, Chelsea could claim the trophy as early as April 29 should they beat Arsenal and Leicester in their next two fixtures. In the end, Fellaini did play at Stamford Bridge, but was dealt with in midfield by Chelsea's Kurt Zouma (left) Zouma (right) had been given a specific tactical brief to stifle the threat of Fellaini (left) during the game.
```

## [wikihow] −00  ·  крупная 12.13 → 11.69 (-3.7%)  ·  снято 11 символов  ·  урон 3

*судья: Stray lead punctuation and missing sentence-boundary spacing from scraped list formatting occur systematically throughout the document.*

**до:**

```text
Do you want the wall inside or outside?
What shapes, features and size do you want your wall to be?
You will also want to begin drawing out the type of wall you want.;
, You must use a structural method of building your wall to support the weight of a climber, and if it is outdoors, to resist wind loads in severe weather.
Post and beam construction. This is the most structurally sound building method. Set 8 inch (20.3 cm) diameter treated poles in the ground about 48 inches (121.9 cm) deep. Fill the holes with ready mixed concrete, or bags of premixed concrete mixed in a wheelbarrow. Bolt 4X6 inch beams horizontally across the posts with 5⁄8 to 3⁄4 inch (1.6 to 1.9 cm) galvanized bolts. Nail or screw 3⁄4 inch (1.9 cm) plywood sheathing over the beams, and finish with paint or textured paint.
Wood framed, wood covered. This consists of lumber framing members, attached to a "mud sill" or framed up from a concrete base. 2X6 inch (nominal) lumber at 16 inch (40.6 cm) centers is suggested, with a 3⁄4 inch (1.9 cm) exterior plywood sheathing attached with 2 inch (5.1 cm) number 8 wood screws 8 inches (20.3 cm) on center. You will need to attach diagonal bracing solidly anchored to the back of the wall and the earth behind the structure to keep it upright.
Wood framed, stucco covered. Frame the wall the same as the above, then attach wire plaster lath with 1 1⁄4 inch (3.2 cm) roofing nails, 8 inches (20.3 cm) on center. Bed the lath with 1/2 inch (5.1 cm) of masonry cement plaster, scoring this "coat" (the scratch coat), so that the finish coat will bond. Apply a masonry cement or synthetic plaster finish coat after the scratchcoat has dried thoroughly.
Concrete block (CMU). Place a concrete footing, a minimum of 8 inches (20.3 cm) thick, 24 inches (61.0 cm) wide, with reinforcing rods, 2 or 3 feet (0.6 or 0.9 m) below grade (ground level). Stick number 5 rebar dowels 16 inches (40.6 cm) on center into the concrete, with 48 inches (121.9 cm) projecting upwards for block cell reinforcement. Lay up your block (8X8X16 inch, typical), filling each cell that has a reinforcing rod with mortar as you go. Add rebar with a 32 inch (81.3 cm) lap so that the rebar is continuous to the top of the wall. Backfill the lower part of the wall, to the normal ground level, and compact. This earth fill at the base will help support your wall from high winds and tipping over.
, Begin to figure your specific cut angles, your framework, joints and other important features of your wall.Decide what kind of effects the elements will have on your wall if it is outside. Will your wall suffer from being rained or snowed on? One way to sustain the life of your wall is to use marine board. This is more expensive but if you live in a rainy area, you will find the board to be well worth it.
, If you enjoy building, this will be the best part. The reality of your wall will slowly come to life through this process.
,, Are you going to have padding such as mattresses or gym pads?
, If the wall is outside, be prepared for your bolts to rust over. If you don’t plan on rearranging holds very often, you will typically not have much to worry about. If you do want to move holds around, it may be difficult unscrew them once they have become rusted. You can also find holds pretty cheap if you hit your local climbing gyms. Many times they replace their old holds and will give you or sell you their old holds at a pretty low price. You may also be able to find holds from manufacturers that are clearing out old styles. Once you get your wall up, you will be excited and can climb whenever you want. There will come a time when you get used to the wall and it’s not as thrilling as it was once before. Keep yourself motivated by adding to the wall and developing new routes. The more challenges you can give yourself, the more apt you are to continue training on your wall.
```

**после:**

```text
Do you want the wall inside or outside?
What shapes, features and size do you want your wall to be?
You will also want to begin drawing out the type of wall you want. You must use a structural method of building your wall to support the weight of a climber, and if it is outdoors, to resist wind loads in severe weather.
Post and beam construction. This is the most structurally sound building method. Set 8 inch (20.3 cm) diameter treated poles in the ground about 48 inches (121.9 cm) deep. Fill the holes with ready mixed concrete, or bags of premixed concrete mixed in a wheelbarrow. Bolt 4X6 inch beams horizontally across the posts with 5⁄8 to 3⁄4 inch (1.6 to 1.9 cm) galvanized bolts. Nail or screw 3⁄4 inch (1.9 cm) plywood sheathing over the beams, and finish with paint or textured paint.
Wood framed, wood covered. This consists of lumber framing members, attached to a "mud sill" or framed up from a concrete base. 2X6 inch (nominal) lumber at 16 inch (40.6 cm) centers is suggested, with a 3⁄4 inch (1.9 cm) exterior plywood sheathing attached with 2 inch (5.1 cm) number 8 wood screws 8 inches (20.3 cm) on center. You will need to attach diagonal bracing solidly anchored to the back of the wall and the earth behind the structure to keep it upright.
Wood framed, stucco covered. Frame the wall the same as the above, then attach wire plaster lath with 1 1⁄4 inch (3.2 cm) roofing nails, 8 inches (20.3 cm) on center. Bed the lath with 1/2 inch (5.1 cm) of masonry cement plaster, scoring this "coat" (the scratch coat), so that the finish coat will bond. Apply a masonry cement or synthetic plaster finish coat after the scratchcoat has dried thoroughly.
Concrete block (CMU). Place a concrete footing, a minimum of 8 inches (20.3 cm) thick, 24 inches (61.0 cm) wide, with reinforcing rods, 2 or 3 feet (0.6 or 0.9 m) below grade (ground level). Stick number 5 rebar dowels 16 inches (40.6 cm) on center into the concrete, with 48 inches (121.9 cm) projecting upwards for block cell reinforcement. Lay up your block (8X8X16 inch, typical), filling each cell that has a reinforcing rod with mortar as you go. Add rebar with a 32 inch (81.3 cm) lap so that the rebar is continuous to the top of the wall. Backfill the lower part of the wall, to the normal ground level, and compact. This earth fill at the base will help support your wall from high winds and tipping over. Begin to figure your specific cut angles, your framework, joints and other important features of your wall. Decide what kind of effects the elements will have on your wall if it is outside. Will your wall suffer from being rained or snowed on? One way to sustain the life of your wall is to use marine board. This is more expensive but if you live in a rainy area, you will find the board to be well worth it. If you enjoy building, this will be the best part. The reality of your wall will slowly come to life through this process. Are you going to have padding such as mattresses or gym pads? If the wall is outside, be prepared for your bolts to rust over. If you don’t plan on rearranging holds very often, you will typically not have much to worry about. If you do want to move holds around, it may be difficult unscrew them once they have become rusted. You can also find holds pretty cheap if you hit your local climbing gyms. Many times they replace their old holds and will give you or sell you their old holds at a pretty low price. You may also be able to find holds from manufacturers that are clearing out old styles. Once you get your wall up, you will be excited and can climb whenever you want. There will come a time when you get used to the wall and it’s not as thrilling as it was once before. Keep yourself motivated by adding to the wall and developing new routes. The more challenges you can give yourself, the more apt you are to continue training on your wall.
```

## [cnn] −00  ·  крупная 15.65 → 15.93 (+1.8%)  ·  снято 5 символов  ·  урон 3

*судья: Systematic tokenizer errors (missing spaces) combined with repeated summary sentences typical of scraped news bullet-points.*

**до:**

```text
A Toronto tower crane operator took pictures of a raccoon that climbed almost 700 feet up a metal ladder before taking a high-altitude poop. Rob MacFarlane captured photos of the raccoon on his craneafter it climbed about213 meters up. MacFarlane shared his first picture of 'Little Mac' on social media on Thursday morning after the crapping critter presumably made the climb over the course of the night. RobMacFarlane shared the picture of the brave raccoon, 'Little Mac', on social media on Thursday morning . The little critter climbed almost 700 feet upMacFarlane's tower crane in Toronto before being photographed . He wrote: 'This critter climbed over 700' to crap on my machine deck.' In the shot, Mac can been see holding on to the ladder for dear life with its eyes staring wide. According to MacFarlane, the animal slowly climbed to the ground without being hurt and 'found somewhere else to sleep all day', theToronto Star reported. MacFarlanehasn't documented prior appearances by raccoons, but he does often post photos from his crane . He said: 'They are the tortoise in the story. 'Not so fast, but efficient. 'It's not unusual. Raccoons seem to like cranes.' MacFarlane has seen raccoons on his crane before and has also found'evidence' of their presence in the past. He hasn't documented their previous appearances, but does post photos from the crane quite often. The crane operator said: 'It's not unusual. Raccoons seem to like cranes.' He has found their 'evidence' before . The crane MacFarlane was working on Thursday stands east of the Rogers Centre at Yonge and the Esplanade . The crane he was working on stands east of the Rogers Centre at Yonge and the Esplanade in downtown Toronto. In addition to pooping on cranes, raccoons also have been getting into the city's garbage bins. The Toronto City Council is considering a31million investment in new compost bins featuring turn-dial locks that raccoons cannot open. Mayor John Tory said: 'We have left no stone unturned in our fight against the Raccoon Nation. 'Defeat is not an option.'
```

**после:**

```text
A Toronto tower crane operator took pictures of a raccoon that climbed almost 700 feet up a metal ladder before taking a high-altitude poop. Rob MacFarlane captured photos of the raccoon on his craneafter it climbed about213 meters up. MacFarlane shared his first picture of 'Little Mac' on social media on Thursday morning after the crapping critter presumably made the climb over the course of the night. RobMacFarlane shared the picture of the brave raccoon, 'Little Mac', on social media on Thursday morning. The little critter climbed almost 700 feet upMacFarlane's tower crane in Toronto before being photographed. He wrote: 'This critter climbed over 700' to crap on my machine deck.' In the shot, Mac can been see holding on to the ladder for dear life with its eyes staring wide. According to MacFarlane, the animal slowly climbed to the ground without being hurt and 'found somewhere else to sleep all day', theToronto Star reported. MacFarlanehasn't documented prior appearances by raccoons, but he does often post photos from his crane. He said: 'They are the tortoise in the story. 'Not so fast, but efficient. 'It's not unusual. Raccoons seem to like cranes.' MacFarlane has seen raccoons on his crane before and has also found'evidence' of their presence in the past. He hasn't documented their previous appearances, but does post photos from the crane quite often. The crane operator said: 'It's not unusual. Raccoons seem to like cranes.' He has found their 'evidence' before. The crane MacFarlane was working on Thursday stands east of the Rogers Centre at Yonge and the Esplanade. The crane he was working on stands east of the Rogers Centre at Yonge and the Esplanade in downtown Toronto. In addition to pooping on cranes, raccoons also have been getting into the city's garbage bins. The Toronto City Council is considering a31million investment in new compost bins featuring turn-dial locks that raccoons cannot open. Mayor John Tory said: 'We have left no stone unturned in our fight against the Raccoon Nation. 'Defeat is not an option.'
```

## [wp] −00  ·  крупная 12.03 → 12.12 (+0.7%)  ·  снято 4 символов  ·  урон 1

*судья: Text is intact dialog prose with only a minor punctuation typo and an unfinished closing line.*

**до:**

```text
Oh, hey! Over here! Honey, you're late! I'm already on my second cosmo!" "Yeah, sorry. Traffic was crazy. Valentine's day, you know. All the panicked husbands rushing out at the last minute to buy up any remaining flowers and chocolate." "You know, hon, you're always so pessimistic. I know you've been single for a while, now, but isn't it time to get back on that horse?" "Look, could we talk about something else? Isn't it kind of a cliche for us to be here, two single gals getting drinks, and to just be talking about men?" "It's still a point, hon." "Yeah, I know. I just haven't met anyone yet." "Ah, but I've got the perfect guy for you! He's a little bit older, and one of his eyes tends to roll around a bit, but he's totally a sweetheart..." "Please, no. Just - okay, have you ever heard of the Bechdel test?" "Isn't that a type of sauce?" "No, that's bechamel." "Ooh, yeah, it's really creamy! I love getting that with pasta, even though I always end up hating myself for it later when I'm on the scale." "Er, sure. No, the Bechdel test. It's a way of analyzing movies." "I usually use IMDB." "Shut it and drink your cosmo. No, the Bechdel test is supposed to check whether a movie is balanced in terms of gender. To pass the test, a movie has to have a conversation between two named female characters that isn't about a male character." "Well, that's easy! I bet most movies have that." "No, you'd be surprised. A lot of them either don't name enough female characters, or all the female conversations are about men, and nothing else." "Okay, hon, I'll believe you. But what's your point?" "My point? I feel like every conversation with you fails the Bechdel test." "I don't "I mean, every single conversation is about men, or dates! Maybe we'll chat a little about our work or something, but we basically just get together and gossip over men." "But you got ta look at the situation, too! Where do we meet up?" "Well, at a bar..." "That's right. And has anything interesting happened at work?" "Well, no..." "Uh huh. And even more than that - what day is today?" "It's Valentine's Day..." "Exactly.
```

**после:**

```text
Oh, hey! Over here! Honey, you're late! I'm already on my second cosmo!" "Yeah, sorry. Traffic was crazy. Valentine's day, you know. All the panicked husbands rushing out at the last minute to buy up any remaining flowers and chocolate." "You know, hon, you're always so pessimistic. I know you've been single for a while, now, but isn't it time to get back on that horse?" "Look, could we talk about something else? Isn't it kind of a cliche for us to be here, two single gals getting drinks, and to just be talking about men?" "It's still a point, hon." "Yeah, I know. I just haven't met anyone yet." "Ah, but I've got the perfect guy for you! He's a little bit older, and one of his eyes tends to roll around a bit, but he's totally a sweetheart..." "Please, no. Just-okay, have you ever heard of the Bechdel test?" "Isn't that a type of sauce?" "No, that's bechamel." "Ooh, yeah, it's really creamy! I love getting that with pasta, even though I always end up hating myself for it later when I'm on the scale." "Er, sure. No, the Bechdel test. It's a way of analyzing movies." "I usually use IMDB." "Shut it and drink your cosmo. No, the Bechdel test is supposed to check whether a movie is balanced in terms of gender. To pass the test, a movie has to have a conversation between two named female characters that isn't about a male character." "Well, that's easy! I bet most movies have that." "No, you'd be surprised. A lot of them either don't name enough female characters, or all the female conversations are about men, and nothing else." "Okay, hon, I'll believe you. But what's your point?" "My point? I feel like every conversation with you fails the Bechdel test." "I don't "I mean, every single conversation is about men, or dates! Maybe we'll chat a little about our work or something, but we basically just get together and gossip over men." "But you got ta look at the situation, too! Where do we meet up?" "Well, at a bar..." "That's right. And has anything interesting happened at work?" "Well, no..." "Uh huh. And even more than that-what day is today?" "It's Valentine's Day..." "Exactly.
```

## [eli5] −00  ·  крупная 12.60 → 12.76 (+1.3%)  ·  снято 4 символов  ·  урон 2

*судья: Multiple forum/comment answers were scraped together with raw SQL/array delimiters left between them.*

**до:**

```text
Becuase not everyone will claim the rebate. So the retailer gets to take in the full price for some of the products','One point that hasnt been mad yet: The rebate is offered by the manufacturer. This way the retail shop doesnt take a hit on their profit margin, and the manufacturer sells more units.', "Short Answer: Not everyone who buys it for the rebate will actually use the rebate. Long Answer: You might have 100 sales before the rebate and see that jump to 200 sales with the rebate. However, of those 200 rebate sales, only 25 people might actually send in the rebate, of which only 20 get it approved under the terms of the rebate. Therefore, you've made 20 additional sales at the rebate price and 80 additional sales at full price - plus your original 100 sales at full price - instead of 200 sales at a reduced price.," "Everyone has mentioned that most people don't redeem a mail in rebate. What I have not seen mentioned is that a rebate is also market research for the manufacture. The manufacture can learn about who exactly is buying their product and where these people are. A short survey is usually required with a rebate and the lets the manufacture know things like the age group, gender and location of people that are buying their products. I may have intended to sell my stuff to teenagers but I find out later that older adults are buying more of them. I can change future advertising to include these people. I could find out that people in one region buy a lot of my things and another region buys very little. I could either refocus advertising in the low sales region or ignore them and focus my related products in the high sales region.," "The truth is that not many people cash rebates in. Usually they require sending something in my snail mail and jumping through hoops. Most people don't bother. So it drives up sales, but ends up not costing that much.," "Totally agree.... I'm sure they don't get many people who actually remember to send in their rebates so they end up making more money. Also, I've sent in rebates I have never gotten back and the company fights with you about it too. So, I feel like it is a crooks way of getting more money.,When I worked at Staples, the manager said 7 was the number of people that actually mailed in the rebates correctly.','They are hoping some people will forget and they have to hand out less of the discounts.
```

**после:**

```text
Becuase not everyone will claim the rebate. So the retailer gets to take in the full price for some of the products','One point that hasnt been mad yet: The rebate is offered by the manufacturer. This way the retail shop doesnt take a hit on their profit margin, and the manufacturer sells more units.', "Short Answer: Not everyone who buys it for the rebate will actually use the rebate. Long Answer: You might have 100 sales before the rebate and see that jump to 200 sales with the rebate. However, of those 200 rebate sales, only 25 people might actually send in the rebate, of which only 20 get it approved under the terms of the rebate. Therefore, you've made 20 additional sales at the rebate price and 80 additional sales at full price-plus your original 100 sales at full price-instead of 200 sales at a reduced price. " "Everyone has mentioned that most people don't redeem a mail in rebate. What I have not seen mentioned is that a rebate is also market research for the manufacture. The manufacture can learn about who exactly is buying their product and where these people are. A short survey is usually required with a rebate and the lets the manufacture know things like the age group, gender and location of people that are buying their products. I may have intended to sell my stuff to teenagers but I find out later that older adults are buying more of them. I can change future advertising to include these people. I could find out that people in one region buy a lot of my things and another region buys very little. I could either refocus advertising in the low sales region or ignore them and focus my related products in the high sales region. " "The truth is that not many people cash rebates in. Usually they require sending something in my snail mail and jumping through hoops. Most people don't bother. So it drives up sales, but ends up not costing that much. " "Totally agree.... I'm sure they don't get many people who actually remember to send in their rebates so they end up making more money. Also, I've sent in rebates I have never gotten back and the company fights with you about it too. So, I feel like it is a crooks way of getting more money. When I worked at Staples, the manager said 7 was the number of people that actually mailed in the rebates correctly.','They are hoping some people will forget and they have to hand out less of the discounts.
```

## [yelp] −00  ·  крупная 12.54 → 12.54 (+0.0%)  ·  снято 0 символов  ·  урон 1

*судья: Minor Web residue at the end and a single missing space after a period.*

**до:**

```text
0 STARS My 16 year old son and myself decided to do a road trip. So from Toronto we left at 6:30 pm arrived in jersey city at 5:00 am.slep fo 6 hours then spent the day in NYC. Then off we went to our reserved room in montreal!! Drove for 6 hours, arrived in montreal at 12:15 am at our hotel. I then found out I booked for the following night. Needless to say I almost cried because it had been a very long day. Well 2 long days. I asked if I could switch my booking. They had rooms available. They said no they wouldn't then told me I could have a room for the full price if I choose. So standing there at midnight with my 16 year old son, they told me they couldn't help with my situation unless I paid full price for the room. Now I totally get the fact that I made a really embarrassing stupid mistake...but really. So no room for my son and I. So obviously I was a day ahead of myself, somehow! I felt like a big disappointment to my son, we were so tired and I didn't have a room for us. So to the Gouveneur Place Dupuis Montreal I have nothing good to say. They let me a very petite 43 year old woman and her 16 year old son with no room in a big city we were very unfamiliar with (never mind the language barrier) they could have helped us out. They didn't. So I didn't see montreal at all because I then drive 4 hours to ottawa. I checked a couple hotels on my way out of montreal but they were booked. I can't see myself recommending that hotel to anyone. Bad experience. My actual check in was August 13. Sent from my iPhone
```

**после:**

```text
0 STARS My 16 year old son and myself decided to do a road trip. So from Toronto we left at 6:30 pm arrived in jersey city at 5:00 am.slep fo 6 hours then spent the day in NYC. Then off we went to our reserved room in montreal!! Drove for 6 hours, arrived in montreal at 12:15 am at our hotel. I then found out I booked for the following night. Needless to say I almost cried because it had been a very long day. Well 2 long days. I asked if I could switch my booking. They had rooms available. They said no they wouldn't then told me I could have a room for the full price if I choose. So standing there at midnight with my 16 year old son, they told me they couldn't help with my situation unless I paid full price for the room. Now I totally get the fact that I made a really embarrassing stupid mistake...but really. So no room for my son and I. So obviously I was a day ahead of myself, somehow! I felt like a big disappointment to my son, we were so tired and I didn't have a room for us. So to the Gouveneur Place Dupuis Montreal I have nothing good to say. They let me a very petite 43 year old woman and her 16 year old son with no room in a big city we were very unfamiliar with (never mind the language barrier) they could have helped us out. They didn't. So I didn't see montreal at all because I then drive 4 hours to ottawa. I checked a couple hotels on my way out of montreal but they were booked. I can't see myself recommending that hotel to anyone. Bad experience. My actual check in was August 13. Sent from my iPhone
```

## [xsum] −00  ·  крупная 14.42 → 14.42 (+0.0%)  ·  снято 0 символов  ·  урон 1

*судья: The document is fully intact Welsh news prose with only a single minor currency symbol rendering error ("PS15" for £15).*

**до:**

```text
Yn ol y corff sydd yn cynrychioli busnesau gofal, mae nifer yn ymladd I gadw fynd ac yn wynebu argyfwng recriwtio wrth I ysbytai ac archfarchnadoedd gystadlu am weithwyr. Mae Cymdeithas Llywodraeth Leol Cymru yn dweud bod yn rhaid I awdurdodau lleol gael fwy o arian I dalu cyflogau gwell I'r gweithwyr. Dadl Llywodraeth Cymru yw eu bod wedi buddsoddi yn y sector, gan gynnwys rhoi mwy o gyllid fel bod cynghorau yn gallu cwrdd a'r cynnydd yn y cyflog byw cenedlaethol sydd yn dod I rym mis Ebrill. Gofynnodd rhaglen BBC Week In Week Out wrth awdurdodau lleol faint o gwmniau oedd wedi tynnu yn ol o ddarparu pecyn gofal ar ol cael cytundeb. Dywedodd 13 o'r 21 cyngor wnaeth ymateb eu bod wedi wynebu'r sefyllfa yma. Mae Colin Angel o Gymdeithas Gofal Cartref Prydain, sef y corff sydd yn cynrychioli cwmniau gofal, yn dweud bod nifer o'u haelodau yn ei chael hi'n anodd I oroesi. "Yr hyn rydyn ni'n clywed yng Nghymru yw bod rhai darparwyr mewn cyfyng gyngor, ac yn ceisio yn galed I ganfod ffordd I barhau fel busnes gyda'r cyfraddau maen nhw'n cael eu talu gan y cynghorau lleol." Mae'n dweud bod peryg y bydd cwmniau yn dod I ben, a hynny yn gyflymach mewn ardaloedd gwledig. Un cwmni wnaeth orfod rhoi'r gorau I ddarparu gofal I gyngor Conwy oedd Cymorth Llaw. Roedd y cyngor wedi cynnig talu PS15 yr awr I'r cwmni, ond penderfynodd y busnes nad oedd hynny'n ddigon o arian. Dywedodd pennaeth Cymorth Llaw, Ken Hogg: "Doedden ni ddim yn meddwl y bydden ni yn gallu darparu'r lefel yma o wasanaeth am y pris roedd Conwy yn gynnig. Ac mi oedden ni yn gyndyn iawn, iawn o adael ond mi oedd yn rhaid I ni adael. "Ar ddiwedd y dydd, busnes ydyn ni. Mae'n rhaid I'n rheolwr banc fod yn hapus. Mae'n rhaid I ni wneud rhyw fath o elw ar ddiwedd y flwyddyn." Mae Cyngor Conwy yn dweud eu bod wedi ymrwymo I gefnogi pobl fregus mewn cymunedau, er yr heriau ariannol. Yn ol Steve Thomas o Gymdeithas Llywodraeth Leol Cymru, mae cynghorau angen mwy o arian er mwyn gallu talu cyflogau gwell I ofalwyr. "Fel mae hi, oni bai ein bod ni'n cael mwy o arian yn dod mewn I'r system mae gyda ni broblem fawr," meddai. "Does'na neb yn ymfalchio yn y cyflog rydyn ni yn talu'r rhai sy'n gweithio yn y sector gofal cymdeithasol." Ym mis Mawrth cyhoeddodd Llywodraeth y DU PS2bn yn ychwanegol ar gyfer gofal cymdeithasol yn Lloegr, a PS200m I Gymru. Ond dyw Llywodraeth Cymru ddim wedi penderfynu eto beth fyddan nhw'n gwneud efo'r arian. Mae'r gweinidog sydd a chyfrifoldeb am y maes, Rebecca Evans AC, yn dweud mai penderfyniad I gabinet Llywodraeth Cymru yw sut I wario'r arian ond eu bod nhw'n gweithredu I ddatrys problemau recriwtio ac y bydd cyflwyno cynllun gofrestru ar gyfer gweithwyr yn 2020 yn helpu. "Bydd hynny yn rhoi'r clod dyledus I'r gweithlu, achos pan mae'n dod at edrych ar ol ein pobl fwyaf bregus, pa swydd sydd yn fwy pwysig na hynny? "Fe fydd hefyd yn rhoi cyfle I bobl yn y maes gael y cyfle I feddwl am ddatblygu eu gyrfa, achos dyw hynny ddim yn digwydd ar hyn o bryd." Dywedodd hefyd eu bod wedi buddsoddi yn y sector I wella amodau gweithwyr.
```

**после:**

```text
Yn ol y corff sydd yn cynrychioli busnesau gofal, mae nifer yn ymladd I gadw fynd ac yn wynebu argyfwng recriwtio wrth I ysbytai ac archfarchnadoedd gystadlu am weithwyr. Mae Cymdeithas Llywodraeth Leol Cymru yn dweud bod yn rhaid I awdurdodau lleol gael fwy o arian I dalu cyflogau gwell I'r gweithwyr. Dadl Llywodraeth Cymru yw eu bod wedi buddsoddi yn y sector, gan gynnwys rhoi mwy o gyllid fel bod cynghorau yn gallu cwrdd a'r cynnydd yn y cyflog byw cenedlaethol sydd yn dod I rym mis Ebrill. Gofynnodd rhaglen BBC Week In Week Out wrth awdurdodau lleol faint o gwmniau oedd wedi tynnu yn ol o ddarparu pecyn gofal ar ol cael cytundeb. Dywedodd 13 o'r 21 cyngor wnaeth ymateb eu bod wedi wynebu'r sefyllfa yma. Mae Colin Angel o Gymdeithas Gofal Cartref Prydain, sef y corff sydd yn cynrychioli cwmniau gofal, yn dweud bod nifer o'u haelodau yn ei chael hi'n anodd I oroesi. "Yr hyn rydyn ni'n clywed yng Nghymru yw bod rhai darparwyr mewn cyfyng gyngor, ac yn ceisio yn galed I ganfod ffordd I barhau fel busnes gyda'r cyfraddau maen nhw'n cael eu talu gan y cynghorau lleol." Mae'n dweud bod peryg y bydd cwmniau yn dod I ben, a hynny yn gyflymach mewn ardaloedd gwledig. Un cwmni wnaeth orfod rhoi'r gorau I ddarparu gofal I gyngor Conwy oedd Cymorth Llaw. Roedd y cyngor wedi cynnig talu PS15 yr awr I'r cwmni, ond penderfynodd y busnes nad oedd hynny'n ddigon o arian. Dywedodd pennaeth Cymorth Llaw, Ken Hogg: "Doedden ni ddim yn meddwl y bydden ni yn gallu darparu'r lefel yma o wasanaeth am y pris roedd Conwy yn gynnig. Ac mi oedden ni yn gyndyn iawn, iawn o adael ond mi oedd yn rhaid I ni adael. "Ar ddiwedd y dydd, busnes ydyn ni. Mae'n rhaid I'n rheolwr banc fod yn hapus. Mae'n rhaid I ni wneud rhyw fath o elw ar ddiwedd y flwyddyn." Mae Cyngor Conwy yn dweud eu bod wedi ymrwymo I gefnogi pobl fregus mewn cymunedau, er yr heriau ariannol. Yn ol Steve Thomas o Gymdeithas Llywodraeth Leol Cymru, mae cynghorau angen mwy o arian er mwyn gallu talu cyflogau gwell I ofalwyr. "Fel mae hi, oni bai ein bod ni'n cael mwy o arian yn dod mewn I'r system mae gyda ni broblem fawr," meddai. "Does'na neb yn ymfalchio yn y cyflog rydyn ni yn talu'r rhai sy'n gweithio yn y sector gofal cymdeithasol." Ym mis Mawrth cyhoeddodd Llywodraeth y DU PS2bn yn ychwanegol ar gyfer gofal cymdeithasol yn Lloegr, a PS200m I Gymru. Ond dyw Llywodraeth Cymru ddim wedi penderfynu eto beth fyddan nhw'n gwneud efo'r arian. Mae'r gweinidog sydd a chyfrifoldeb am y maes, Rebecca Evans AC, yn dweud mai penderfyniad I gabinet Llywodraeth Cymru yw sut I wario'r arian ond eu bod nhw'n gweithredu I ddatrys problemau recriwtio ac y bydd cyflwyno cynllun gofrestru ar gyfer gweithwyr yn 2020 yn helpu. "Bydd hynny yn rhoi'r clod dyledus I'r gweithlu, achos pan mae'n dod at edrych ar ol ein pobl fwyaf bregus, pa swydd sydd yn fwy pwysig na hynny? "Fe fydd hefyd yn rhoi cyfle I bobl yn y maes gael y cyfle I feddwl am ddatblygu eu gyrfa, achos dyw hynny ddim yn digwydd ar hyn o bryd." Dywedodd hefyd eu bod wedi buddsoddi yn y sector I wella amodau gweithwyr.
```

## [wp] −00  ·  крупная 15.18 → 15.18 (+0.0%)  ·  снято 0 символов  ·  урон 1

*судья: A minor leftover forum edit note is present at the very end of the text, but the main story content is intact.*

**до:**

```text
The elf took a seat in the running blood. He started examing his ebony-black plate armor, searching for any dents and knicks but found none. At that point he turned his gaze upward to his handiwork. The goblin merchants lay scattered across the road, their scarlet blood washed away by the rain. Their cart overturned in a ditch beside the road. From his pack the elf retrieved a morsel of wyvern jerky. Although not very tasty, those domesticated pseudo-dragons were definitely nutritious. The elf's eyes wandered around the carnage until they found his partner rifling through the bodies. "Jaspar get o'er here once you got all the coins and trade goods," the elf called out. "Yes m'lord Da'oel," replied the tall and thin Ogre. Jaspar gathered all of the coins and gems, the spices and oils, the ales and wines, and the cloth and metal in a knapsack and dragged it over to Da'oel who still sat cross legged chewing on the stringy wyvern meat with his battleaxe across his lap. Jaspar adjusted his hardened leather and sat across from Da'oel with the sack in between them. "This rain is bugging me," stated Da'oel begrudgingly. "I'll remedy that immediately sir da "Do not, call me sir," spat the elf with poison. "I'm sorry master Da'oel. I'll deal with the rain now." With his Jaspar shifted this position and began incanting in some odd language unknown to Da'oel, and as he did a large leather tent formed itself around the two, sheltering them from the rain. "You and your fackin magics. I swear there ain't sometin' right wit'it." The elf's eyes shifted to the bag in between the two. "Right now, let's get to countin'. May'e we'll have enough to buy ourselves off this fackin is'and and back to my home. And once I reach my warm sands you can leave me and I can stop'earin your bitching about being "evil" and how murdering is bad." Upon deciphering this elf's barbaric dialect and comprehending the message the ogre dumped the contents of the bag and began counting coins and appraising the items. Edit: Typo
```

**после:**

```text
The elf took a seat in the running blood. He started examing his ebony-black plate armor, searching for any dents and knicks but found none. At that point he turned his gaze upward to his handiwork. The goblin merchants lay scattered across the road, their scarlet blood washed away by the rain. Their cart overturned in a ditch beside the road. From his pack the elf retrieved a morsel of wyvern jerky. Although not very tasty, those domesticated pseudo-dragons were definitely nutritious. The elf's eyes wandered around the carnage until they found his partner rifling through the bodies. "Jaspar get o'er here once you got all the coins and trade goods," the elf called out. "Yes m'lord Da'oel," replied the tall and thin Ogre. Jaspar gathered all of the coins and gems, the spices and oils, the ales and wines, and the cloth and metal in a knapsack and dragged it over to Da'oel who still sat cross legged chewing on the stringy wyvern meat with his battleaxe across his lap. Jaspar adjusted his hardened leather and sat across from Da'oel with the sack in between them. "This rain is bugging me," stated Da'oel begrudgingly. "I'll remedy that immediately sir da "Do not, call me sir," spat the elf with poison. "I'm sorry master Da'oel. I'll deal with the rain now." With his Jaspar shifted this position and began incanting in some odd language unknown to Da'oel, and as he did a large leather tent formed itself around the two, sheltering them from the rain. "You and your fackin magics. I swear there ain't sometin' right wit'it." The elf's eyes shifted to the bag in between the two. "Right now, let's get to countin'. May'e we'll have enough to buy ourselves off this fackin is'and and back to my home. And once I reach my warm sands you can leave me and I can stop'earin your bitching about being "evil" and how murdering is bad." Upon deciphering this elf's barbaric dialect and comprehending the message the ogre dumped the contents of the bag and began counting coins and appraising the items. Edit: Typo
```

## [yelp] −00  ·  крупная 13.21 → 13.21 (+0.0%)  ·  снято 0 символов  ·  урон 0

*судья: The document is an undamaged user review arriving intact without pipeline defects.*

**до:**

```text
My boyfriend and I decided to try here since we were looking for a new place to eat on Friday night. When we showed up we were asked to wait, even though the restaurant was half empty, but we assumed maybe some wait staff had called out. when we were seated and we ordered, they brought out the bread. It was nice and warm, but one of the loaves was on the edge of being burnt and they both had sesame seeds on top that reminded me of those bread sticks that you get in cellophane wrappers. We had both ordered the house dressing for our salads, but it turned out to be oil and vinegar, which would have been fine except one salad had all the oil, which was bitter, and one had all the vinegar. Also, the salad was half a handful of romaine, 2 onion rings, 2 small black olives, and a tomato slice. Then our dinner showed up. I had gotten the Manicotti and my boyfriend had gotten the Cannelloni. The red sauce on top tasted like it was tomato paste from a can that was watered down and the sauce on mine tasted like someone had added extra sugar. I could honestly taste the aluminum in the sauce. The noodles were soggy, and my boyfriend said that his tasted reheated. I did discover someone had put slivers of garlic under the melted cheese on top, because they weren't fully cooked. As for the service, our waiter would ask questions and then leave without hearing the answer and they brought us small pitchers so we could refill our drinks, then forgot to bring more when those pitchers ran out.
```

**после:**

```text
My boyfriend and I decided to try here since we were looking for a new place to eat on Friday night. When we showed up we were asked to wait, even though the restaurant was half empty, but we assumed maybe some wait staff had called out. when we were seated and we ordered, they brought out the bread. It was nice and warm, but one of the loaves was on the edge of being burnt and they both had sesame seeds on top that reminded me of those bread sticks that you get in cellophane wrappers. We had both ordered the house dressing for our salads, but it turned out to be oil and vinegar, which would have been fine except one salad had all the oil, which was bitter, and one had all the vinegar. Also, the salad was half a handful of romaine, 2 onion rings, 2 small black olives, and a tomato slice. Then our dinner showed up. I had gotten the Manicotti and my boyfriend had gotten the Cannelloni. The red sauce on top tasted like it was tomato paste from a can that was watered down and the sauce on mine tasted like someone had added extra sugar. I could honestly taste the aluminum in the sauce. The noodles were soggy, and my boyfriend said that his tasted reheated. I did discover someone had put slivers of garlic under the melted cheese on top, because they weren't fully cooked. As for the service, our waiter would ask questions and then leave without hearing the answer and they brought us small pitchers so we could refill our drinks, then forgot to bring more when those pitchers ran out.
```

## [xsum] −0−  ·  крупная 11.38 → 11.38 (+0.0%)  ·  снято 0 символов  ·  урон 2

*судья: Contains a stray news feed line, broken encoding for a pound sign (£28m), empty list headers, and a truncated final sentence.*

**до:**

```text
The AE unit at Telford's Princess Royal Hospital could be downgraded, leaving the Royal Shrewsbury as the only hospital in the county to offer emergency and critical care. Women and children's services would also be moved to Shrewsbury, despite a PS28m centre opening in Telford in 2014. The proposals have been made by the NHS'Future Fit' board. More updates on this and other stories in Shropshire Urgent care centres would be based at both sites, which Shrewsbury and Telford Hospital NHS Trust said would attend to the majority of patients who are currently seen at AE. These "preferred options" will be considered by Telford and Wrekin Clinical Commissioning Group (CCG) and Shropshire CCG, who will formally decide if a public consultation can begin. The board is also recommending three other options be put forward for public consultation. No formal decisions will be made until at least Summer 2017, Telford and Wrekin CCG said. The NHS Future Fit programme was launched in 2014 to review the health system across Shropshire, Telford and Wrekin and mid-Wales and is set to finish in spring 2017. Princess Royal Hospital in Telford: Royal Shrewsbury Hospital: There has been criticism of the plans to move women's and children's services from Telford. Shaun Davies, leader of Telford Wrekin Council, described the announcement as "a total farce that is being railroaded through" and said he opposed the idea of downgrading Telford's AE department. "If there is to be only one, then it makes sense from a financial and clinical perspective to locate it at Princess Royal Hospital; in the area where there is a growing population.
```

**после:**

```text
The AE unit at Telford's Princess Royal Hospital could be downgraded, leaving the Royal Shrewsbury as the only hospital in the county to offer emergency and critical care. Women and children's services would also be moved to Shrewsbury, despite a PS28m centre opening in Telford in 2014. The proposals have been made by the NHS'Future Fit' board. More updates on this and other stories in Shropshire Urgent care centres would be based at both sites, which Shrewsbury and Telford Hospital NHS Trust said would attend to the majority of patients who are currently seen at AE. These "preferred options" will be considered by Telford and Wrekin Clinical Commissioning Group (CCG) and Shropshire CCG, who will formally decide if a public consultation can begin. The board is also recommending three other options be put forward for public consultation. No formal decisions will be made until at least Summer 2017, Telford and Wrekin CCG said. The NHS Future Fit programme was launched in 2014 to review the health system across Shropshire, Telford and Wrekin and mid-Wales and is set to finish in spring 2017. Princess Royal Hospital in Telford: Royal Shrewsbury Hospital: There has been criticism of the plans to move women's and children's services from Telford. Shaun Davies, leader of Telford Wrekin Council, described the announcement as "a total farce that is being railroaded through" and said he opposed the idea of downgrading Telford's AE department. "If there is to be only one, then it makes sense from a financial and clinical perspective to locate it at Princess Royal Hospital; in the area where there is a growing population.
```

## [wp] −0−  ·  крупная 13.02 → 13.02 (+0.0%)  ·  снято 0 символов  ·  урон 1

*судья: Minor Web residue note at the end and slightly botched quote tokenization.*

**до:**

```text
The men look at me, and I look to the men. I sense there fear, and know what I must say. " You can all see what I can see, you can all hear what I can hear, and you all know what I know. Our enemy is innumerable, their battle cries shake the heavens, and there are no reinforcements for us. " So why then do we stay, when there is no hope? " you may ask. The reason is really quite simple; we were always going to die. Maybe not in this battle, or even the one after that, but we were always going to die. " " You see, this is what men do. We live; We laugh; We cry; And we die. And seeing how this war has been going, seeing how far they have made into our homeland, it looks like we will not be able to avoid the former for much longer. So if you feel like you should leave, do it, I will not stop you. " " First however I plead you hear me out. The Enemy can not be stopped, but if we try, we can delay them. We are in a pass, meaning they can not bring their cavalry to bear, and their numerical advantage is almost nullified. That means that if you stay, if you fight, we may be able to put up a defense, long enough for our countrymen to muster their strength. " I see my logic does not fall on deft ears, and so I get up. Standing now I raise my voice " So, I beg of you, fight with me. Fight this demon that has swept across our land. Make it bleed so it knows it knows it can be beaten. You were always going to die! So die here with me! Die here, where they will remember you name! Die Here, where, it could make all the difference! Die here, and show this demon that if he wants our home. He will have to pay to get it! " I raise my fist in the air and scream as loud as I can " Now let them hear it!! " " Live; Laugh; Cry; Die " " Live; Laugh; Cry; Die! " " Live; Laugh; Cry; Die!! " " Live; Laugh; Cry; Die!!! " At first it is just me, but slowly my men join in and it becomes a chant of both sadness and resolve. Hearing this, the enemy gives their famed battle cry; drowning out our chant, in an attempt to intimidate us. But it doesn't matter, because they heard, and they know now that we will not so easily give in. edit: was formatted as code.
```

**после:**

```text
The men look at me, and I look to the men. I sense there fear, and know what I must say. " You can all see what I can see, you can all hear what I can hear, and you all know what I know. Our enemy is innumerable, their battle cries shake the heavens, and there are no reinforcements for us. " So why then do we stay, when there is no hope? " you may ask. The reason is really quite simple; we were always going to die. Maybe not in this battle, or even the one after that, but we were always going to die. " " You see, this is what men do. We live; We laugh; We cry; And we die. And seeing how this war has been going, seeing how far they have made into our homeland, it looks like we will not be able to avoid the former for much longer. So if you feel like you should leave, do it, I will not stop you. " " First however I plead you hear me out. The Enemy can not be stopped, but if we try, we can delay them. We are in a pass, meaning they can not bring their cavalry to bear, and their numerical advantage is almost nullified. That means that if you stay, if you fight, we may be able to put up a defense, long enough for our countrymen to muster their strength. " I see my logic does not fall on deft ears, and so I get up. Standing now I raise my voice " So, I beg of you, fight with me. Fight this demon that has swept across our land. Make it bleed so it knows it knows it can be beaten. You were always going to die! So die here with me! Die here, where they will remember you name! Die Here, where, it could make all the difference! Die here, and show this demon that if he wants our home. He will have to pay to get it! " I raise my fist in the air and scream as loud as I can " Now let them hear it!! " " Live; Laugh; Cry; Die " " Live; Laugh; Cry; Die! " " Live; Laugh; Cry; Die!! " " Live; Laugh; Cry; Die!!! " At first it is just me, but slowly my men join in and it becomes a chant of both sadness and resolve. Hearing this, the enemy gives their famed battle cry; drowning out our chant, in an attempt to intimidate us. But it doesn't matter, because they heard, and they know now that we will not so easily give in. edit: was formatted as code.
```

## [wp] −0−  ·  крупная 19.54 → 19.54 (+0.0%)  ·  снято 0 символов  ·  урон 0

*судья: The document is an intact personal letter with no pipeline damage present.*

**до:**

```text
Dear soandso, I remember when we first met. Around five years ago now. How time flies. We were younger then, and I was so naive to the suffering you had gone through at such a young and formative age. I remember us singing in the choir, me helping you out when you couldn't hit the notes right. Even though you weren't the best singer, it was still beautiful to my ears, it always will be. You were dating someone far older than you were, and I will never understand why. Were you insecure? Were you afraid? You were doing things that a person your age shouldn't have had any business doing. I always detested the group of people you would be with: They had and still don't have a direction, and unfortunately, it rubbed off on you. I graduated high school on time, you a little later. But on my graduation day, you told me your intentions with me. I always knew that you and I would be together, everyone did. Little did I know that life would not be easy. Love wasn't easy. Learning to love you despite your flaws was hard. I craved to be outside and experience the world around us, but you never did. You feared the outside. You feared leaving the hole, physically, emotionally and mentally. I had hope that love and time would help, but it made it worse. You played more PC Games, you met some people online. I hate these people. They took you away from me. You fell for all of them, regardless of their emotional abuse and empty suicidal threats. You believed them over me. You threw away five years and friendship and love, regardless of all that I did. Christmas would've been 2.5 years, and I had intended to give myself more to you. To start my life with you. But your intentions were different. I don't know how you're doing, and vice versa. Maybe it's better that way, that way you won't feel the guilt and pain I have been. But it's alright, I'll be okay, at least I don't live with the paranoia they put me through. I'm meeting people and I'm going to the north soon, like you always wanted to. Maybe you weren't singing the wrong notes; maybe you hadn't found the right harmony. I hope you understand one day. Is brea liom tu. PowerFalcons
```

**после:**

```text
Dear soandso, I remember when we first met. Around five years ago now. How time flies. We were younger then, and I was so naive to the suffering you had gone through at such a young and formative age. I remember us singing in the choir, me helping you out when you couldn't hit the notes right. Even though you weren't the best singer, it was still beautiful to my ears, it always will be. You were dating someone far older than you were, and I will never understand why. Were you insecure? Were you afraid? You were doing things that a person your age shouldn't have had any business doing. I always detested the group of people you would be with: They had and still don't have a direction, and unfortunately, it rubbed off on you. I graduated high school on time, you a little later. But on my graduation day, you told me your intentions with me. I always knew that you and I would be together, everyone did. Little did I know that life would not be easy. Love wasn't easy. Learning to love you despite your flaws was hard. I craved to be outside and experience the world around us, but you never did. You feared the outside. You feared leaving the hole, physically, emotionally and mentally. I had hope that love and time would help, but it made it worse. You played more PC Games, you met some people online. I hate these people. They took you away from me. You fell for all of them, regardless of their emotional abuse and empty suicidal threats. You believed them over me. You threw away five years and friendship and love, regardless of all that I did. Christmas would've been 2.5 years, and I had intended to give myself more to you. To start my life with you. But your intentions were different. I don't know how you're doing, and vice versa. Maybe it's better that way, that way you won't feel the guilt and pain I have been. But it's alright, I'll be okay, at least I don't live with the paranoia they put me through. I'm meeting people and I'm going to the north soon, like you always wanted to. Maybe you weren't singing the wrong notes; maybe you hadn't found the right harmony. I hope you understand one day. Is brea liom tu. PowerFalcons
```
