# Двадцать текстов вслепую

Десять из них размерность считает аномальными для корпуса, десять — обычными. Порядок перемешан, пометок нет.

Читая, отмечай не «нравится / не нравится», а **дефект данных**: обрыв, склейка разнородных кусков, форматный мусор, дублирование внутри текста, следы разметки или навигации, смена языка, обрубленное начало или конец, шаблонность, всё прочее.

## Куда записывать

| текст | дефект есть? | какой | это аутлаер? |
|---|---|---|---|
| T01 |  |  |  |
| T02 |  |  |  |
| T03 |  |  |  |
| T04 |  |  |  |
| T05 |  |  |  |
| T06 |  |  |  |
| T07 |  |  |  |
| T08 |  |  |  |
| T09 |  |  |  |
| T10 |  |  |  |
| T11 |  |  |  |
| T12 |  |  |  |
| T13 |  |  |  |
| T14 |  |  |  |
| T15 |  |  |  |
| T16 |  |  |  |
| T17 |  |  |  |
| T18 |  |  |  |
| T19 |  |  |  |
| T20 |  |  |  |

Ключ лежит в `results/blind_read_key.csv` — не открывай, пока не разметишь.

---

## T01  ·  313 слов

```text
Machine translation, sometimes referred to by the abbreviation MT (not to be confused with computer-aided translation, machine-aided human translation or interactive translation), is a sub-field of computational linguistics that investigates the use of software to translate text or speech from one language to another.
On a basic level, MT performs mechanical substitution of words in one language for words in another, but that alone rarely produces a good translation because recognition of whole phrases and their closest counterparts in the target language is needed. Not all words in one language have equivalent words in another language, and many words have more than one meaning. 
Solving this problem with corpus statistical and neural techniques is a rapidly growing field that is leading to better translations, handling differences in linguistic typology, translation of idioms, and the isolation of anomalies.[failed verification]
Current machine translation software often allows for customization by domain or profession (such as weather reports), improving output by limiting the scope of allowable substitutions. This technique is particularly effective in domains where formal or formulaic language is used. It follows that machine translation of government and legal documents more readily produces usable output than machine translation of conversation or less standardised text.
Improved output quality can also be achieved by human intervention: for example, some systems are able to translate more accurately if the user has unambiguously identified which words in the text are proper names. With the assistance of these techniques, MT has proven useful as a tool to assist human translators and, in a very limited number of cases, can even produce output that can be used as is (e.g., weather reports).
The progress and potential of machine translation have been much debated through its history. Since the 1950s, a number of scholars, first and most notably Yehoshua Bar-Hillel, have questioned the possibility of achieving fully automatic machine translation of high quality.
```

## T02  ·  440 слов

```text
The ' brain ' and body in general is always trying to equalize a ' chemical normal ' which is defined in real - time by your static DNA & dynamic RNA signatures . All incoming stimuli ( physiological & neurological ) to the person is evaluated against long - term memory to match that ' normal ' - in order to generate a ' key ' which is compared to that person 's life experiences of similar events in similar context . Minor variations can also be attributed to your ' waking period ' memory , but as implied , these are only relevant to situations learned in the immediate ' awake ' period since last REM sleep . That key comparison will return a ' positive ' branch if the historical experience has a tendency toward restoring ' normal ' metabolic equilibrium . A ' negative ' if the experience history is deviating from your ' normal ' ... --- or an undefined state - which generates a new ' experience node ' at the junction of those unresolved situations - ready for a new outcome to be acquired and learned into experience . Only the two ( + /- ) outcomes are allowed in order to eliminate fence - sitting deadlock of a situation . This process is modified over time by frequency of ' hits ' and currency of ' most recent ' hit - to develop a scalar response in context with changing reality . * * Depression is seeded * * - when the keyed experience returns a negative - which in turn keys to another ' negative ' node and so on ... the ' depth ' of depression is directly related to the number of negative recursions that occur before a ' positive ' or ' new ' experience node is reached . Where you go after that - depends on the forward branching of subsequent situations , experiences and nodes . Think of it like a huge ' tree ' of life experience . You are born with no experience at ' ground level ' and grow outward toward the sky . New ' situations ' ( raindrops ) may fall onto an existing leaf , or branch -- or fall unhindered through to the ground . How the water drop hits & flows down is entirely dependent on where it lands - or indeed if the leaf / branch exists at all ! There 's another dimension to all this , but the above may help you see the light at the end f the tunnel . good luck !
```

## T03  ·  533 слов

```text
Thank you for your helpful comments.  We just uploaded a revised draft, incorporating the reviewers' suggestions, and hopefully addressing many of their concerns.  Below are the things to note:

- The linear GCCA solution for G and U is included in Appendix A, along with a full gradient derivation: "... the rows of G are the top r (orthonormal) eigenvectors of M, and $U_j = C_{jj}^{−1} Y_j G^T$" (reviewer 2)

- In the last paragraph of the Optimization subsection (page 4), we include big-Oh notation for the gradient update time complexity.  We leverage the GCCA solution presented in [R1] to scale DGCCA to large datasets. (reviewer 2)

- We qualify the pronouncement of being "the first nonlinear multiview learning technique" with the adjective "CCA-style".  Although our work focuses on extending CCA-based multiview methods, we recognize that others have attempted to learn embeddings by merging information from multiple views. (reviewer 5)

- In Section 5, "Other Multiview Work", we include a discussion of a non-CCA-based techniques for nonlinear representation learning from multiple views, and how they differ from DGCCA.  As reviewers 2 and 5 mention, the multiview learning literature is vast, and we are not able to address all models proposed that make learned representations from more than one view.  However, we do hope that this section will clarify how our proposed model differs from other representation learning techniques exploiting multiple views. (reviewers 2 and 5)

- Appendix C includes a short discussion of how the DGCCA objective reconstruction error relates to downstream task performance for Twitter hashtag recommendation.  In short, we found that high reconstruction error is a strong signal for poor downstream performance, but there is significant variation in downstream performance between embeddings learned by models with low reconstruction error. (reviewer 4)

We also trained Bridge Correlational Neural Network embeddings for Twitter hashtag and friend recommendation in a series of preliminary experiments.  We swept over hidden layer width in the same range as the DGCCA experiments, $\lambda \in \{0.0, 0.1, 1.0, 10.0\}$ (the strength of the correlation term in the DGCCA objective), and used either the Twitter user ego text view or their friend network view as the pivot view, since these were the solely most effective views for hashtag and friend recommendtion.  Other learning parameters were left at the defaults and networks were trained for 50 epochs.  However, the performance of these embeddings was much worse than the CCA-style models we compare to (R@1000=0.06 for hashtag recommendation.)  We grant that downstream performance would be improved by tuning learning parameters, but these preliminary experiments underscore the fact that this class of models is not a panacea, and may not be appropriate for these recommendation tasks.  We explicitly note this in the text, since these models assume that all views should be correlated with a pivot view representation.  Entraining all embeddings to a single view is, thus, probably not as effective at hashtag recommendation as learning a CCA-style joint representation for all views. (reviewer 5)

Please let us know if any of these revisions are confusing, or if you have any additional suggestions.

[R1] Pushpendre Rastogi, Benamin Van Durme, and Raman Arora. Multiview LSA: Representation Learning via Generalized CCA. Proceedings of NAACL. 2015.
```

## T04  ·  366 слов

```text
So I will attempt to answer the other half of the question since people have given good feedback on the mortgage costs of your various options. Assumptions:   It is certain that I am off on some (or all) of these assumptions, but they are still useful for drawing a comparison.   If you were to make your mortgage payment, then contribute whatever you have left over to savings, this is where you would be at the end of 30 years. Wait, so the 30 year mortgage has me contributing $40k less to savings over the life of the loan, but comes out with a $20k higher balance?  Yes, because of the way compounding interest works getting more money in there faster plays in your favor, but only as long as your savings venue is earning at a higher rate than the cost of the debt your are contrasting it with. If we were to drop the yield on your savings to 3%, then the 30yr would net you $264593, while the 15yr ends up with $283309 in the bank.  Similarly, if we were to increase the savings yield to 10% (not unheard of for a strong mutual fund), the 30yr nets $993418, while the 15yr comes out at $684448.  Yes in all cases, you pay more to the bank on a 30yr mortgage, but as long as you have a decent investment portfolio, and are making the associated contributions, your end savings come out ahead over the time period.  Which sounds like it is the more important item in your overall picture. However, just to reiterate, the key to making this work is that you have an investment portfolio that out performs the interest on the loan.  Rule of thumb is if the debt is costing you more than the investment will reliably earn, pay the debt off first.  In reality, you need your investments to out perform the interest on your debt + inflation to stay ahead overall.  Personally, I would be looking for at least an 8% annual return on your investments, and go with the 30 year option. DISCLAIMER: All investments involve risk and there is no guarantee of making any given earnings target.
```

## T05  ·  354 слов

```text
Notable only as the acting debut of future big-time Hollywood starlet, Sandra Bullock, this ludicrous action flick is so full of holes that one might easily suspect termite infestation. The storyline is incomprehensible and very poorly thought out. The production values stink of cheese. In fact, a total LACK of production values would have been better...at least the film might have seemed grittier that way. The ADR is laughably bad and omni-present in the film. It's debatable as to whether or not ANY of the dialogue tracks from the actual shoot were used.The performances are, for the most part, horrible, though there are a few exceptions. In those exceptions, however, the performances are undermined by the fact that the director was obviously giving the actors poor direction and making them act completely out of character at times. (i.e. characters going from passive to panicked in the blink of an eye. Bad Direction.) Also, the constant "weapon sound effects" (magazines being loaded, slides being cocked, etc.) are completely overused and, more often than not, totally out of sync with the on-screen actions. Add to this cheesy "Bad Guy" vocal distortion for the lead villain (mainly so that you KNOW he's the villain in this incomprehensible mess of a film), and you have a recipe for disaster.The situations in the film go well beyond standard "suspension of disbelief" and become downright laughable. One lead character spends a good portion of the film tied to a chair before he DECIDES to use the butterfly knife tucked in his sock in order to free himself. So, my questions are...why didn't he do this sooner, and why does he even HAVE the butterfly knife. He wasn't searched? RIGHT. This is one of a hundred examples of completely ludicrous situations which have somehow been crammed into this 90-minute package.In whole, "The Hangmen" plays like an unbearably bad R-rated TV movie from the '80s. If not for the subsequent success of Sandra Bullock, this would have NEVER found its way to DVD. But it has, so my only advice is to steer clear. Watching this film may actually impair your IQ.
```

## T06  ·  1953 слов

```text
Val Caniparoli is an American ballet dancer and international choreographer. His work includes more than 100 productions for ballet, opera, and theater for over 50 companies, and his career as a choreographer progressed globally even as he continued his professional dance career with the San Francisco Ballet. He joined the San Francisco Ballet as a dancer in 1973. He was appointed to the position of principal character dancer with the San Francisco Ballet by Artistic Director Helgi Tomasson in 1987. Caniparoli was born in Renton, Washington, to Francisco Caniparoli, a clothing manufacturer, and Leonora (Marconi) Caniparoli, who worked at Boeing. He attended Washington State University (WSU), where he studied music and theater. When the First Chamber Dance Company was touring Eastern Washington, they did performances at WSU, and offered workshops in ballet. Caniparoli attended one and was told he had talent, and should audition at the San Francisco Ballet School. Thereafter he decided to pursue a career in ballet, and left WSU. He received a scholarship from the Ford Foundation that enabled him to attend the San Francisco Ballet School. Caniparoli performed with San Francisco Opera Ballet, and in 1973, just a year and a half into his studies, he was offered a contract with San Francisco Ballet. In his debut season, he worked under Co-Artistic Directors Lew Christensen and Michael Smuin, and later, under Helgi Tomasson. Caniparoli became interested in choreography when he attended a choreography workshop offered by the Pacific Northwest Ballet. After that work, his choreography career expanded and he was appointed resident choreographer for the San Francisco Ballet in the mid-1980s. In 1984, Caniparoli co-founded a choreographic collective called OMO in San Francisco, and a documentary about OMO's founding was broadcast that year on PBS. In 1994, he created his first full-length ballet entitled Lady of the Camellias, based on a story by Alexandre Dumas, and with a score by Frédéric Chopin. Lady of the Camellias became one of Caniparoli's most popular works, and a part of the repertoire of several ballet companies, including Ballet West, Ballet Florida, Boston Ballet, Cincinnati Ballet, Tulsa Ballet, and Royal Winnipeg Ballet. Caniparoli was Resident Choreographer for Ballet West from 1993 to 1997, and for Tulsa Ballet from 2001 to 2006. He continues to create works for San Francisco Ballet. In 1995, Caniparoli choreographed a new work entitled Lambarena, set to a musical blend of J.S. Bach with Traditional African music composed by Pierre Akendengue and Hughes de Courson. Lambarena has become another of Caniparoli's most popular creations, a blend of classical ballet and African dance. This ballet has been performed more than 20 companies, including Atlanta Ballet, Boston Ballet, Cincinnati Ballet, Singapore Dance Theatre, San Francisco Ballet, and State Ballet of South Africa. In 2002, Caniparoli was invited to choreograph a pas de deux to be performed by Evelyn Hart and Rex Harrington for Queen Elizabeth II to celebrate her Golden Jubilee visit to Canada. In May 2010, San Francisco's American Conservatory Theater (A.C.T.) premiered Tosca Cafe, a theater/dance work co-created and co-directed by Caniparoli and A.C.T.'s Carey Perloff; Caniparoli also did the choreography. "'Tosca Cafe"', which started as The Tosca Project, chronicles a wide cast of characters who inhabit Tosca, a bar in the North Beach section of San Francisco in the same location for decades. Caniparoli and Perloff saw this work as a unique opportunity for collaboration between dancers and actors. Since its 2010 premiere in San Francisco, Tosca Cafe has been performed internationally. While growing up in Renton, Washington, Caniparoli studied music for 13 years. His study included private lessons on alto saxophone, clarinet, and flute. He credits his study of music with nurturing his eclectic interest in world music and composers, and varied genres. He has become well known for his use of widely diverse music as a principal foundation for his choreographic work. He was also influenced by the dancing of film stars Gene Kelly and Fred Astaire. Caniparoli's work has been described as "rooted in classicism but influenced by all forms of movement: modern dance, ethnic dance, social dancing, even ice skating." Jekyll & Hyde, 2020 (Composers: Krzysztof Penderecki, Frédéric Chopin, Henryk Górecki, Wojciech Kilar, Henryk Wieniawski) Finnish National Ballet Foreshadow, 2018 (Composer: Ludovico Einaudi) San Francisco Ballet The Nutcracker, 2018 (Composer: Peter Ilych Tchaikovsky) Royal New Zealand Ballet If I Were A Sushi Roll, 2018 (Composer: Nico Muhly) Smuin Ballet Dances for Lou, 2017 (Composer: Lou Harrison) Ballet West 4 in the Morning, 2016 (Composer: William Walton) Amy Seiwert's Imagery Twisted 2, 2016 (Composer: André Previn, Jacques Offenbach, Arnold Schoenberg) BalletMet Repeat After Me, 2016 (Composer: Johann Paul Von Westhoff) Menlo Ballet Without Borders, 2016 (Composer: Yo-Yo Ma and the Silk Road Ensemble) Texas Ballet Theater Beautiful Dreamer, 2016 (Composer: Stephen Collins Foster) Oakland Ballet Stolen Moments, 2015 (Composer: Jean-Phillippe Rameau) Richmond Ballet Das Ballett, 2015 (Composer: Leopold Mozart) Oakland Ballet The Nutcracker 2014 (Composer: Peter Ilych Tchaikovsky) Grand Rapids Ballet Twisted, 2014 (Composer: Benjamin Britten, Gioachino Rossini, Giacomo Puccini) BalletMet Tutto Eccetto Il Lavandino (Everything But The Kitchen Sink) 2014 (Composer: Antonio Vivaldi) Smuin Ballet Spaghetti Western 2014 (Composer: Ennio Morricone) Louisville Ballet Tears, 2014 (Composer: Steve Reich) San Francisco Ballet In Pieces, 2013 (Composer: Poul Ruders) Colorado Ballet Triptych, 2013 (Composer: John Tavener & Alexander Balanescu), Amy Siewert's Imagery The Lottery, 2012 (Composer: Robert Moran) Premiere: Ballet West Chant, 2012 (Composer: Lou Harrison) Premiere: Singapore Dance Theatre Incantations, 2012 (Composer: Alexandre Rabinovitch-Barakovsky) Premiere: Joffrey Ballet Swipe, 2012 (Composer: Gabriel Prokofiev) Premiere: Richmond Ballet Tears From Above, 2011 (Composer: Elena Kats-Chermin) Premiere: Diablo Ballet Double Stop, 2011 (Composer: Philip Glass) San francisco Ballet Blades of Grass, 2010 (Composer: Tan Dun) Premiere: Milwaukee Ballet Still Life, 2010 (Composer: Elena Kats-Chermin) Premiere: Scottish Ballet Amor Con Fortuna, 2009 (Composer: Jordi Savali, Various) Premiere: Tulsa Ballet The Seasons, 2009 (Composer: Alexander Glazunov) Premiere: Pacific Northwest Ballet The Nutcracker, 2009 (Composer: Pyotr Ilyich Tchaikovsky) Premiere: Louisville Ballet Ebony Concerto, 2009 (Composer: Igor Stravinsky) Premiere: San Francisco Ballet Ibsen's House, 2008 (Composer: Antonín Dvořák) Premiere: San Francisco Ballet Suite, 2007 (Composer: George Frederic Handel) Premiere: American Repertory Ballet Violin, 2006 (Composer: Heinrich Ignaz Franz Biber) Premiere: Richmond Ballet Songs, 2005 (Composer: Chick Corea) Premiere: Central West Ballet Ikon of Eros, 2005 (Composer: John Tavenor) Premiere: Washington Ballet Sonata for Two Pianos and Percussion, 2004 (Composer: Béla Bartók) Premiere: Boston Ballet Val Caniparoli's A Cinderella Story, 2004 (Composer: Richard Rodgers) Premiere: Royal Winnipeg Ballet Gustav's Rooster, 2003 (Composer: Hoven Droven) Premiere: Tulsa Ballet Vivace, 2003 (Composer: Franz Schubert) Premiere: Tulsa Ballet Untitled, 2003 (Composer:: Dmitri Shostakovich) Premiere: Royal Winnipeg Ballet No Other, 2002 (Composer: Richard Rodgers) Premiere: San Francisco Ballet Unspoken, 2002 (Composer: Camille Saint-Saëns) Premiere: Royal Winnipeg Ballet Misa Criolla, 2002 (Composer: Ariel Ramirez) Premiere: Tulsa Ballet Devil's Sonata, 2002 (Composer: Guiseppi Tartini) Premiere: Sacramento Ballet boink! 2002 (Composer: Juan Garcia Esquivel) Premiere: Lawrence Pech Dance Company The Nutcracker, 2001 (Composer: Pyotr Ilyich Tchaikovsky) Premiere: Cincinnati Ballet Torque, 2001 (Composer: Michael Torke) Premiere: Pacific Northwest Ballet Jaybird Lounge, 2001 (Composer Uri Caine) Premiere: Pennsylvania Ballet Death of a Moth, 2001 (Composer: Carlos Surinach) Premiere: San Francisco Ballet Bird's Nest, 2000 (Composer: Charlie Parker) Premiere: Washington Ballet Already Dusk, 2000 (Composer: Johannes Brahms) Premiere: Lawrence Pech Dance Company Fade to Black, 2000 (Composer: Nina Simone) Premiere: L. Feijoo and Y. Possokhov Going for Baroque 1999 (Composer: Antonio Vivaldi) Premiere: Tulsa Ballet Attention Please, 1999 (Composer: J.S. Bach) Premiere: Richmond Ballet Aquilarco, 1999 (Composer: Giovanni Sollima) Premiere: San Francisco Ballet Separations, 1999 (Composer: Dmitri Shostakovich) Premiere: Ballet Florida Open Veins, 1998 (Composer: Robert Moran) Premiere: Atlanta Ballet Aria, 1998 (Composer: George Frederic Handel) Premiere: San Francisco Ballet Book of Alleged Dances, 1998 (Composer: John Adams) Premiere: Ballet West Slow, 1998 (Composer: Graham Fitkin) Premiere: San Francisco Ballet The Bridge, 1998 (Composer: Dmitri Shostakovich) Premiere: Pacific Northwest Ballet Djangology, 1997 (Composer: Django Reinhardt) Premiere: Richmond Ballet Ciao, Marcello, 1997 (Composer: Nino Rota) Premiere: San Francisco Ballet Prawn-watching, 1996 (Composer: Michael Nyman) Premiere: Ballet West Bow Out, 1995 (Composers: David Bedford and Roy Powell) Premiere: Richmond Ballet Lambarena, 1995 (Composer: J.S. Bach and Traditional African) Premiere: San Francisco Ballet La Folia, 1994 (Composer: Gregorio Paniagua) Premiere: Marin Ballet Tangazzo, 1994 (Composer: Amadeo Roldan) Premiere: Marin Ballet Lady of the Camellias, 1994 (Composer: Frédéric Chopin) Premiere: Ballet West Seeing Stars, 1993 (Composer: Erno Dohnanyi) Premiere: San Francisco Ballet Concerto Grosso, 1992 (Composer: Arcangelo Corelli) Premiere: Marin Ballet Pulcinella, 1991 (Composer: Igor Stravinsky) Premiere: San Francisco Ballet Tryst, 1991 (Composer: Wolfgang Amadeus Mozart) Premiere: Pacific Northwest Ballet Gran Partita, 1990 (Composer: Wolfgang Amadeus Mozart) Premiere: Pacific Northwest Ballet In Perpetuum, 1990 (Composer: Arvo Pärt) Premiere: San Francisco Ballet Ritual, 1990 (Composer: Alfred Schnittke) Premiere: Johann Renvall and Stars of American Ballet A Door Is Ajar, 1990 (Composer: Kronos Quartet) Premiere: Ririe Woodbury Between Ourselves, 1989 (Composer: Béla Bartók) Premiere: Pittsburgh Ballet Theatre Kinetic Impressions, 1989 (Composer: Francis Poulenc) Premiere: Ballet West Connotations, 1989 (Composer: Benjamin Britten) Premiere: San Francisco Ballet White Mourning, 1989 (Composers: Franz Schubert, Gustav Mahler) Premiere: Ballet West Ophelia, 1988 (Composer: Bohuslav Martinu) Premiere: Ballet West Narcisse, 1987 (Composer: Claude Debussy) Premiere: San Francisco Ballet Hamlet and Ophelia Pas de Deux, 1985 (Composer: Bohuslav Martinu) Premiere: San Francisco Ballet Aubade, 1985 (Composer: Francis Poulenc) Premiere: Israel Ballet Accidental or Abnormal Chromosomal Events, 1984 (Composer: Al Aguis-Sinerco) Premiere: Bay Area Playwrights Festival Tar Marmalade, 1984 (Composer: Douglas Adams) Premiere: Oakland Ballet Chansons de Scheherazade, 1983 (Composer: Maurice Ravel) Premiere: San Francisco Ballet Windows, 1983 (Composer: Ludwig van Beethoven) Premiere: San Francisco Ballet Loves-Lies-Bleeding, 1982 (Composer: Igor Stravinsky) Premiere: San Francisco Ballet Deranged Dances, 1982 (Composer: Charles Ives) Premiere: Marin Ballet Six-for-Eight, 1981 (Composer: George Frederic Handel) Premiere: Palo Alto Dance Theatre Street Songs, 1980 (Composer: Carl Orff) Premiere: Pacific Northwest Ballet Concertino, 1979 (Composer: Carlo Ricciotti) Premiere: Contemporary Dance Theatre of Tucson A Little Night Music (Music and Lyrics: Stephen Sondheim) American Conservatory Theater (A.C.T.), San Francisco Arcadia (2013) American Conservatory Theater (A.C.T.), San Francisco Tosca Café (2011) Theatre Calgary, Vancouver Playhouse The Tosca Project (2010) American Conservatory Theater (A.C.T.), San Francisco Tis Pity She's a Whore (2008) A.C.T., San Francisco Two Women (2015) (Music by Marco Tutino) (Adapted on the novel La Ciociara by Alberta Moravia) San Francisco Opera Capriccio (1990) San Francisco Opera, Metropolitan Opera, Lyric Opera of Chicago The Metropolitan Opera HD Live (TV series) (1 episode) – R. Strauss: Capriccio (2011)Great Dancers of our Time In der Hauptrolle Vladimir Malakhov, Lucia Lacarra und Kiyoko Kimura (DVD – 2005) -- Choreography for Lady of the Camellias Television In 2015, co-choreographed with Helgi Tomasson, a commercial for the 50th Anniversary Super Bowl with dancers from San Francisco Ballet. Choreography from "Lambarena" featured on Sesame Street with dancers Lorena Feijoo and Lorna Feijoo. Caniparoli appeared on PBS in "The San Francisco Ballet in Cinderella" Dance in America (the Great Performances Series) in the role of Cinderella's father. In addition, he appeared in three television specials: The Creation of OMO (1987) in which he discussed the experimental dance company he co-founded A Song for Dead Warriors (1984)Romeo and Juliet, Michael Smuin's ballet production, which aired on PBS in 1976 Recipient, 10 grants for choreography, National Endowment for the Arts Recipient, (2001) Isadora Duncan Dance Award for Choreography, Death of a Moth, San Francisco Ballet Nominated, (1997) Lambarena nominated for the Prix Benois de la Danse for Best Choreography. Recipient, (1997 & 1994) awards from the Choo-San Goh & H. Robert Magee Foundation Recipient, (1997) Isadora Duncan Award for Sustained Achievement Recipient, (1991–1992) Choreographers Fellowship, National Endowment for the Arts Recipient, (1991) Artist Fellowship, California Arts Council Recipient, (1991) Artist Fellowship, California Arts Council Recipient, (1987) Isadora Duncan Award for Aubade'', Bay Area Dance Coalition Recipient, (1981–1988), Choreographers' Fellowship, National Endowment for the Arts
```

## T07  ·  332 слов

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

## T08  ·  300 слов

```text
I 'll take this from a broad philosophical standpoint because I do n't know a lot about Koko . There are a few questions that * all * animals are capable of answering on a very basic level . " Where . " Where can I find food ? Where is home ? Where is my family ? " What . " What is this ? What are you ? " Who . " Obviously you 've seen dogs react to who you are vs who its owner is . " When " is another one . When do I need to eat ? When do I need to sleep ? When do I need to run ? " How ? " How do I best hunt prey ? How do I escape this lion ? How do I climb this tree ? The one question that separates humans from animal is the question " Why ? " Animals survive based off instincts . They learn , but they do n't ever touch on the concept " why " because they just have n't developed their brain enough to be capable of having thoughts as complex as the " why ? " If we could somehow ask animal " why " , their answer would likely be " well .. because its what I do " and that 's the most we 'd get . Animals can think ... but they ca n't think about what they think . That 's a huge reason humans are so different . It 's why you do n't see vegetarian lions . Therefore , a gorilla wo n't be able to actually focus on it 's own consciousness enough to understand the reasoning behind its choices because ... ya know ... instincts .
```

## T09  ·  378 слов

```text
A Coast Guard cutter arrived in San Diego on Thursday with more than 14 tons of cocaine, part of what authorities has described as a surge of seizures near Central and South America. The cocaine, valued by the Coast Guard at 424 million, was seized by U.S. and Canadian forces in 19 separate incidents in the eastern Pacific Ocean near Central and South America. It included a 10-ton bust from a coastal freighter, the largest maritime drug interdiction in that area since 2009. A crewman from the Coast Guard Cutter Boutwell guards some of more than 28,000 pounds of cocaine, seized at sea and offloaded at Naval Base San Diego on Thursday . The cocaine was seized by U.S. and Canadian forces in 19 separate incidents in the eastern Pacific Ocean near Central and South America . The U.S. Navy and Coast Guard and Royal Canadian Navy ships have seized more than 28 tons of cocaine valued at 848 million in the eastern Pacific region near Central and South America in the last six months, resulting in 101 arrests. That's more than the 12-month period that ended in September, and U.S. authorities have called it the most successful run for drug seizures in the area since 2009. Suspicious vessels in international waters were tracked by military or law enforcement aircraft or vessels. Coast Guard personnel operating from Coast Guard cutters and Royal Canadian Navy vessels boarded the ships to seize the drugs. Thursday's arrival marked the second time that the Cutter Boutwell has returned home with a huge load of drugs. In October, its crew turned over more than 14 tons of cocaine to the U.S. Drug Enforcement Administration. Suspicious vessels in international waters were tracked by military or law enforcement aircraft or vessels. Coast Guard personnel operating from Coast Guard cutters and Royal Canadian Navy vessels boarded the ships to seize the drugs. Thursday's arrival marked the second time that the Cutter Boutwell returned home with a huge load of drugs. In October, its crew turned over more than 14 tons of cocaine to the U.S. Drug Enforcement Administration. U.S. Navy and Coast Guard and Royal Canadian Navy ships have seized more than 28 tons of cocaine valued at 848 million in the last six months .
```

## T10  ·  667 слов

```text
Grimsby striker Bogle saw his curling, left-footed free-kick tipped over the crossbar by Jamie Stephens before he opened the scoring after he burst past two defenders and had his shot deflected off Elliot Johnson and into the net. And he doubled the lead 10 minutes before the interval when a mistake by Harry Taylor allowed Bogle to run free and beat Stephens. The Bees got themselves back into the game in the 50th minute when Akinde burst into the box and was brought down by Josh Gowling. Akinde picked himself up to dispatch the penalty past James McKeown. Akinde then levelled from the spot five minutes later after Danny Collins fouled Curtis Weston. Ryan Watson then squandered two chances to win it for Barnet before the late drama. Barnet were reduced to 10 men in the 90th minute when Ali Sesay, who had only been on the pitch for five minutes, was shown a straight red card for a challenge on Scott Vernon. Grimsby were then given a golden chance to win it two minutes into stoppage time when Michael Nelson brought down Gowling, but Bogle sent his spot-kick over the bar. Report supplied by the Press Association. Match ends, Grimsby Town 2, Barnet 2. Second Half ends, Grimsby Town 2, Barnet 2. Danny Collins (Grimsby Town) is shown the yellow card for a bad foul. Foul by Danny Collins (Grimsby Town). John Akinde (Barnet) wins a free kick in the defensive half. Penalty missed! Bad penalty by Omar Bogle (Grimsby Town) left footed shot is just a bit too high. Omar Bogle should be disappointed. Penalty Grimsby Town. Josh Gowling draws a foul in the penalty area. Penalty conceded by Michael Nelson (Barnet) after a foul in the penalty area. Alie Sesay (Barnet) is shown the red card. Scott Vernon (Grimsby Town) wins a free kick on the right wing. Foul by Alie Sesay (Barnet). Attempt missed. Omar Bogle (Grimsby Town) right footed shot from outside the box is high and wide to the right. Scott Vernon (Grimsby Town) wins a free kick on the right wing. Foul by Bira Dembele (Barnet). Foul by Omar Bogle (Grimsby Town). Alie Sesay (Barnet) wins a free kick in the defensive half. Substitution, Barnet. Alie Sesay replaces Curtis Weston. Attempt missed. Brandon Comley (Grimsby Town) right footed shot from outside the box is too high. Zak Mills (Grimsby Town) wins a free kick in the attacking half. Foul by John Akinde (Barnet). Attempt missed. Craig Disley (Grimsby Town) header from the centre of the box misses to the right. Corner, Grimsby Town. Conceded by Mauro Vilhete. Attempt saved. John Akinde (Barnet) header from the centre of the box is saved in the centre of the goal. Tom Bolarinwa (Grimsby Town) is shown the yellow card. Foul by Tom Bolarinwa (Grimsby Town). Sam Muggleton (Barnet) wins a free kick on the right wing. Substitution, Grimsby Town. Scott Vernon replaces Kayden Jackson. Attempt missed. Zak Mills (Grimsby Town) left footed shot from the centre of the box is close, but misses to the right. Attempt missed. Jean-Louis Akpa-Akpro (Barnet) right footed shot from the centre of the box is close, but misses to the left. Attempt missed. Zak Mills (Grimsby Town) right footed shot from outside the box is close, but misses to the right. Tom Champion (Barnet) is shown the yellow card. Omar Bogle (Grimsby Town) wins a free kick in the defensive half. Foul by Mauro Vilhete (Barnet). Zak Mills (Grimsby Town) wins a free kick on the left wing. Foul by Jamal Campbell-Ryce (Barnet). Attempt missed. Ryan Watson (Barnet) right footed shot from outside the box is too high. Brandon Comley (Grimsby Town) wins a free kick in the attacking half. Foul by Tom Champion (Barnet). Attempt missed. Ryan Watson (Barnet) right footed shot from the centre of the box is close, but misses the top left corner. Attempt saved. Curtis Weston (Barnet) right footed shot from outside the box is saved in the centre of the goal.
```

## T11  ·  296 слов

```text
There are, of course, many possible financial emergencies. They range from large medical expenses to losing your job to being sued to major home or car repairs to who-knows-what. I suppose some people are in a position where the chances that they will face any sort of financial emergency are remote. If you live in a country with national health insurance and there is near-zero chance that you will have any need to go outside this system, you are living with your parents and they are equipped to handle any home repairs, you ride the bus or subway and don't own a car so that's not an issue, etc etc, maybe there just isn't any likely scenario where you'd suddenly need cash. I can think of all sorts of scenarios that might affect me. I'm trying to put my kids through college, so if I lost my job, even if unemployment benefits were adequate to live on, they wouldn't pay for college. I have terrible health insurance so big medical bills could cost me a lot. I have an old car so it could break down any time and need expensive repairs, or even have to be replaced. I might suddenly be charged with a crime that I didn't commit and need a lawyer to defend me. Etc. So in a very real sense, everyone's situation is different. On the other hand, no matter how carefully you think it out, it's always possible that you will get bitten by something that you didn't think of. By definition, you can't make a list of unforeseen problems that might affect you! So no matter how safe you think you are, it's always good to have some emergency fund, just in case. How much is very hard to say.
```

## T12  ·  316 слов

```text
Amazon Alexa, also known simply as Alexa, is a virtual assistant technology largely based on a Polish speech synthesiser named Ivona, bought by Amazon in 2013. It was first used in the Amazon Echo smart speaker and the Echo Dot, Echo Studio and Amazon Tap speakers developed by Amazon Lab126. It is capable of voice interaction, music playback, making to-do lists, setting alarms, streaming podcasts, playing audiobooks, and providing weather, traffic, sports, and other real-time information, such as news. Alexa can also control several smart devices using itself as a home automation system. Users are able to extend the Alexa capabilities by installing "skills" (additional functionality developed by third-party vendors, in other settings more commonly called apps) such as weather programs and audio features. It uses automatic speech recognition, natural language processing, and other forms of weak AI to perform these tasks.
Most devices with Alexa allow users to activate the device using a wake-word (such as Alexa or Amazon); other devices (such as the Amazon mobile app on iOS or Android and Amazon Dash Wand) require the user to click a button to activate Alexa's listening mode, although, some phones also allow a user to say a command, such as "Alexa" or "Alexa wake".
As of November 2018[update], Amazon had more than 10,000 employees working on Alexa and related products. In January 2019, Amazon's devices team announced that they had sold over 100 million Alexa-enabled devices.
In September 2019, Amazon launched many new devices achieving many records while competing with the world's smart home industry. The new Echo Studio became the first smart speaker with 360 sound and Dolby sound. Other new devices included an Echo dot with a clock behind the fabric, a new third-generation Amazon Echo, Echo Show 8, a plug-in Echo device, Echo Flex, Alexa built-in wireless earphones, Echo buds, Alexa built-in spectacles, Echo frames, an Alexa built-in Ring, and Echo Loop.
```

## T13  ·  439 слов

```text
I think the part of your question about not wanting to "mess up more" is the most important element. You say you know someone with good credit who is willing to co-sign for you, but let's be honest -- your credit isn't bad for no reason.  Your credit's bad because you have a history of not paying on your obligations. Putting someone else's credit at risk, even though they may be willing to try and help, could be doing exactly what you said you're trying to avoid -- messing up more.  This person's heart is in the right place, but you really have to ask yourself if you should put them in jeopardy by agreeing to guarantee your debts. So the vehicle you bought is older and has a lot of miles -- you knew that when you bought it.  So you're paying a high interest rate because of your bad credit history -- you knew that when you bought it.  Why you think the vehicle's only going to last another year is what confuses me.  There are many vehicles out there with much higher mileage that are still on the road, and with proper preventative maintenance there's no reason your truck can't do the same. The fact is, you just don't like what you're paying or what you're driving (even though you were good with both when someone was willing to extend you credit), so now you see this other person's willingness to co-sign for you as your ticket out of a situation you no longer want to be in. My suggestion is that you stay with the loan you have, take care of the vehicle to make it last, and prove that you can pay your obligations.  Hopping from loan to loan isn't going to do your credit any favors.  One of the big factors for your credit score is the average age of accounts.  Going and signing a new loan now will only drag that number down and hurt your score, not help it.  And there's no guarantee the next car you buy with your friend's help is going to last the length of that loan either. I would be careful about this "grass is greener on the other side" attitude and just bear through your situation, if only to prove to yourself that you can do it.   There's nothing saying your friend won't still be willing to co-sign for you later on down the line of something does happen to the truck, but you can show them that you're trying to be responsible in the meantime by following through on what you already agreed to.
```

## T14  ·  260 слов

```text
Divergence between populations for a given trait can be driven by natural or
sexual selection, interacting with migration behaviour. Mating preference for
different phenotypes can lead to the emergence and persistence of
differentiated populations. Dominance between alleles encoding for divergent
phenotypes can interfere in such processes. Using a diploid model of trait
determining both mating success and migration rate, we explored differentiation
between two connected populations, assuming either co-dominance or strict
dominance between alleles. The model assumes that individuals prefer mating
with partners displaying the same phenotype and therefore tend to move to the
other population when their own phenotype is rare. We show that the emergence
of differentiated populations in this diploid model is limited as compared to
results obtained with the same model assuming haploidy. When assuming
co-dominance, differentiation arises only when migration is limited as compared
to preference. Such differentiation is less dependent on migration when
assuming strict dominance between haplotypes. Dominant alleles frequently
invade populations because their phenotype is more frequently expressed,
resulting in higher mating success and rapid decrease in migration. However,
depending on the initial distribution of alleles, this advantage associated
with dominance (i.e. Haldane's sieve) may lead to fixation of the dominant
allele throughout both populations. Depending on the initial distribution of
heterozygotes, persistence of polymorphisms within populations can also occur
because heterozygotes displaying the predominant phenotype benefit from mating
preferences. Altogether, our results highlight that heterozygotes' behaviour
has a strong impact on population differentiation and stress out the need of
diploid models of differentiation and speciation driven by natural and sexual
selection.
```

## T15  ·  287 слов

```text
Sensitivity and specificity mathematically describe the accuracy of a test which reports the presence or absence of a condition. Individuals for which the condition is satisfied are considered "positive" and those for which it is not are considered "negative".
If the true condition can not be known, a "gold standard test" is assumed to be correct. In a diagnostic test, sensitivity is a measure of how well a test can identify true positives and specificity is a measure of how well a test can identify true negatives. For all testing, both diagnostic and screening, there is usually a trade-off between sensitivity and specificity, such that higher sensitivities will mean lower specificities and vice versa.
If the goal is to return the ratio at which the test identifies the percentage of people highly likely to be identified as having the condition, the number of true positives should be high and the number of false negatives should be very low, which results in high sensitivity. This is especially important when the consequence of failing to treat the condition is serious and/or the treatment is very effective and has minimal side effects.
If the goal is to return the ratio at which the test identifies the percentage of people highly likely to be identified as not having the condition, the number of true negatives should be high and the number of false positives should be very low, which results in high specificity. That is, people highly likely to be excluded by the test. This is especially important when people who are identified as having a condition may be subjected to more testing, expense, stigma, anxiety, etc.
The terms "sensitivity" and "specificity" were introduced by American biostatistician Jacob Yerushalmy in 1947.
```

## T16  ·  326 слов

```text
People will always downvote something they disagree with. It's just human nature. You see something you don't like and you want to have your say on the matter, without having to go to the effort of leaving a comment. "The downvote button is an essential part of monitoring content": If a comment or post is breaking the rules, report it. "Reported posts have to be reviewed, which will create a huge workload the mods": It says in the reddit FAQ [here] that posts can be reviewed by a program. Also, you could say only a post which has been reported a certain number of times gets reviewed. The number of times could be relative to the amount of traffic on that comment thread, so that small subreddits weren't full of rubbish comments that didn't get enough reports. "People will just start using the report button as a downvote button": Subreddits such as rgonewild and even rchangemyview don't have a downvote button on posts. I would be interested to hear from the mods of those subreddits to see if they have an unusually high number of reports. Otherwise, there's no proof to this point and I still think it would be worth a try. Edit: Hey, thanks for the replies. I will be back to respond soon but haven't had a spare bit of time recently to do this, and I don't want to rush a rubbish response. Hello, users of CMV! This is a footnote from your moderators. We'd just like to remind you of a couple of things. Firstly, please remember to [read through our rules] . If you see a comment that has broken one, it is more effective to report it than downvote it. Speaking of which, [downvotes don't change views] ! If you are thinking about submitting a CMV yourself, please have a look through our [popular topics wiki] first. Any questions or concerns? Feel free to [message us] . Happy CMVing!
```

## T17  ·  451 слов

```text
The paper presents two approaches for generating English poetry. The first
approach combine a neural phonetic encoder predicting the next phoneme with a
phonetic-orthographic HMM decoder computing the most likely word corresponding
to a sequence of phonemes. The second approach combines a character language
model with a weigthed FST to impose rythm constraints on the output of the
language model. For the second approach, the authors also present a heuristic
approach which permit constraining the generated poem according to theme (e.g;,
love) or poetic devices (e.g., alliteration). The generated poems are evaluated
both instrinsically by comparing the rythm of the generated lines with a gold
standard and extrinsically by asking 70 human evaluators to (i) determine
whether the poem was written by a human or a machine and (ii) rate poems wrt to
readability, form and evocation.  The results indicate that the second model
performs best and that human evaluators find it difficult to distinguish
between human written and machine generated poems.

This is an interesting, clearly written article with novel ideas (two different
models for poetry generation, one based on a phonetic language model the other
on a character LM) and convincing results.

 For the evaluation, more precision about the evaluators and the protocol would
be good. Did all evaluators evaluate all poems and if not how many judgments
were collected for each poem for each task ? You mention 9 non English native
speakers. Poems are notoriously hard to read. How fluent were these ? 

In the second model (character based), perhaps I missed it, but do you have a
mechanism to avoid generating non words ? If not, how frequent are non words in
the generated poems ?

In the first model, why use an HMM to transliterate from phonetic to an
orhographic representation rather than a CRF? 

Since overall, you rule out the first model as a good generic model for
generating poetry, it might have been more interesting to spend less space on
that model and more on the evaluation of the second model. In particular, I
would have been interested in a more detailed discussion of the impact of the
heuristic you use to constrain theme or poetic devices. How do these impact
evaluation results ? Could they be combined to jointly constrain theme and
poetic devices ? 

The combination of a neural mode with a WFST is reminiscent of the following
paper which combine character based neural model to generate from dialog acts
with an WFST to avoid generating non words. YOu should relate your work to
theirs and cite them. 

Natural Language Generation through Character-Based RNNs with Finite-State
Prior Knowledge
Goyal, Raghav and Dymetman, Marc and Gaussier, Eric and LIG, Uni
COLING 2016
```

## T18  ·  308 слов

```text
In the new modern global economy, the only thing that really matters is the cost of electricity. The economic gap between first and third world countries is shrinking and wage differences are shrinking as a result. In addition, automation means that labor is increasingly replaced by machines, as is happening in the Foxcon factories. If a country wants to remain economically competitive in the future economy, its main priority should be to drive down its electricity costs, preferably while driving up the electricity cost of foreign countries. How is China doing this? Easy. Subsidize shitty solar panels whose performance drops massively after a few years.1 Western countries will buy these solar panels in an effort to move beyond fossil fuels. At the same time, European and American solar panel producers are driven out of business because their solar panels are not economically competitive. After we adjust our whole economy to solar energy, the result is then that the cost of manufacturing becomes too high in our countries. When the performance of the shitty Chinese solar panels begins to drop, we're stuck with expensive electricity and blackouts. This then places China in an economically dominant position. By the time we figure out what's going on, our remaining industrial base either shuts down or moves operations to China. References 1 - Hello, users of CMV! This is a footnote from your moderators. We'd just like to remind you of a couple of things. Firstly, please remember to [read through our rules] . If you see a comment that has broken one, it is more effective to report it than downvote it. Speaking of which, [downvotes don't change views] ! If you are thinking about submitting a CMV yourself, please have a look through our [popular topics wiki] first. Any questions or concerns? Feel free to [message us] . Happy CMVing!
```

## T19  ·  459 слов

```text
Unlike COD, Real War has rules both sides follow In 1864, sixteen European states adopted the first-ever Geneva Convention to save lives to alleviate the suffering of wounded and sick military personnel, and to protect trained medical personnel as civilians, in the act of rendering aid. Chapter IV, Article 25 of the Geneva Convention states that: "Members of the armed forces specially trained for employment, should the need arise, as hospital orderlies, nurses or auxiliary stretcher-bearers, in the search for or the collection, transport or treatment of the wounded and sick shall likewise be respected and protected if they are carrying out these duties at the time when they come into contact with the enemy or fall into his hands." Article 29 reads: "Members of the personnel designated in Article 25 who have fallen into the hands of the enemy, shall be prisoners of war, but shall be employed on their medical duties insofar as the need arises." According to the Geneva Convention, knowingly firing at a medic wearing clear insignia is a war crime.[2] In modern times, most combat medics carry a personal weapon, to be used to protect themselves and the wounded or sick in their care.[3] When and if they use their arms offensively, they then sacrifice their protection under the Geneva Conventions. These medics are specifically trained., "ELI5 answer- it takes two people to carry a stretcher, 3 in the MASH and many more for administration, triage, etc- none of which carry a rifle or take battle to the enemy. Advanced: A principle strategy of warfare is to prevent as many of your opponent's forces from fighting as possible. Armies use ball ammunition which is more likely to wound rather than kill, as opposed to hollowpoints which cause much more damage. The idea is for every soldier wounded, 2 must carry that soldier, 1 must triage, 3-5 must operate on that soldier, recovery and convalescence will take 6 months or longer. All of this keeps soldiers off the battlefield, which is good for your side.," "It's the rules of war to not target non combatants, such as civilians and medics and chaplains. Also, both sides probably acknowledge that without medics, you have more field deaths meaning you have less troops returning to the battlefield to get shot at again. Edit. Also non combatants have a low tactical and assessment value compared to actual troops who are attackingdefending,The Geneva Conventions (ie. the rules of war) state that noncombatants and medics may not be engaged. However, it also stipulates that medics MUST wear the red cross in order to be recognised as such, otherwise they are fair game.','International agreements prevent soldiers from intentionally targeting medics. Making it obvious who the medic are makes unintentional targeting less likely.
```

## T20  ·  384 слов

```text
For the burning action inside the sun, instead of the hydrogen breaking apart into smaller bits, it's being squashed together until it fuses and forms helium. (Some of this helium breaks apart back into hydrogen but only some of it. Atoms breaking apart into smaller elements is what nuclear reactors do here on earth.) The sun just being where it is, is basically a gigantic ongoing nuclear explosion, cept it's fusing light elements instead of breaking superheavy metals apart like in bombs. The sun's nuclear reactions are also happening much slower. That's how it can last for billions of years.
Eventually the sun will run out of hydrogen, so it will start having to fuse together helium to form lithium and beryllium, and when it runs out of helium it fuses the heavier elements it made, and so on.
Every time it runs out of a lightweight element, less energy is released from the reaction and the next sequence becomes less efficient. As the amount of fuel it has overall reduces, its fuel consumption gets faster and faster. Eventually it doesn't have enough fuel of any kind to complete nuclear reactions, and the tremendous amount of heat and energy it emits is not enough to keep it "pushing outward" like the explosion that's been powering it for eons. Its enormous gravity wins the tug-of-war that the gravity and nuclear explosions have been fighting for billions of years, and the star collapses. (All the momentum of that enormous mass collapsing into a single point then causes a super-hyper-mega explosion called a nova.)
All that's left over is a husk of inert elements that can barely react. They'll still react and decay and emit energy (and lots) but nowhere near as much as the star did when it was much younger. Eventually this stellar remnant (called a white dwarf) will cool down and sit there, like a super-dense ball of rock (called a black dwarf).
What we know about stellar remnants like those is not too much: the universe isn't old enough to have any black dwarfs. What we do know is that black dwarfs would likely be several orders of magnitude more dense than if you had a planet made of the densest element. Not as dense as a neutron star, but still extremely substantial.
```
