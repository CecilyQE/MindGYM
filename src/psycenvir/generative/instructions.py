"""Canonical instruction text for generative Psych-101-style environments."""

BADHAM_INSTRUCTION = (
    "You will be shown several examples of geometric objects.\n"
    "Your task is to learn a rule that allows you to tell whether an object belongs "
    "to one category or another category.\n"
    "For each presented object, you will be asked to make a category judgment by "
    "pressing the corresponding key and then you will receive feedback.\n"
    "You will encounter several different problems with different rules."
)

WU_BANDIT_INSTRUCTION = (
    "You will be presented with a series of 16 different environments to explore.\n"
    "In each trial, you can select an option between numbers 1 and 30 by pressing "
    "the corresponding key.\n"
    "By selecting any of these options, you will earn points associated with each "
    "unique option.\n"
    "Imagine these options 1 through 30 as lying next to each other in an ordered "
    "line; options closer to each other tend to have similar rewards as rewards "
    "tend to cluster together.\n"
    "For each environment, you will be able to make either 5 or 10 choices.\n"
    "When you made all your choices in a given environment, you will start making "
    "choices in the next unexplored environment.\n"
    "The rewards underlying the different options are different in each environment "
    "so you will learn them anew for each environment.\n"
    "Each environment starts with the value of a single option revealed.\n"
    "When you choose the number corresponding to a different option, you will be "
    "told the value of that option and receive those points.\n"
    "Previously revealed options, including the starting option, can also be "
    "reselected, although there may be small changes in the point value.\n"
    "It is your task to gain as many points as possible across all 16 environments."
)

FREY_RISK_INSTRUCTION = (
    "Throughout the task, you will be presented with balloons, one at a time.\n"
    "In each step, you can choose to pump up the balloon by pressing H and you will "
    "accumulate 1 point for each pump.\n"
    "At any point, you can stop pumping up the balloon by pressing W and you will "
    "collect your accumulated points.\n"
    "You will repeat this procedure on multiple different balloons.\n"
    "It is your choice to determine how much to pump up the balloon, but be aware "
    "that at some point the balloon will explode.\n"
    "If the balloon explodes before you collect your accumulated points, then you "
    "move on to the next balloon and the points are lost."
)

PETERSON_INSTRUCTION = (
    "You will repeatedly choose between two options labeled with single letters.\n"
    "Each option delivers points according to the probabilities shown before each "
    "choice.\n"
    "Your goal is to maximize the points you receive."
)

WILSON_BANDIT_INSTRUCTION = (
    "You are participating in multiple games involving two slot machines, labeled C and A.\n"
    "The two slot machines are different across different games.\n"
    "Each time you choose a slot machine, you get some points.\n"
    "You choose a slot machine by pressing the corresponding key.\n"
    "Each slot machine tends to pay out about the same amount of points on average.\n"
    "Your goal is to choose the slot machines that will give you the most points across the experiment.\n"
    "The first 4 trials in each game are instructed trials where you will be told which slot machine to choose.\n"
    "After these instructed trials, you will have the freedom to choose for either 1 or 6 trials."
)

LEFEBVRE_INSTRUCTION = (
    "You are going to visit four different casinos (named 1, 2, 3, and 4) 24 times each.\n"
    "Each casino owns two slot machines that return either 0 or 0.5 points stochastically with different probabilities.\n"
    "You can play one of the machines in order to win points by pressing the corresponding key.\n"
    "Your goal is to maximize the sum of received points within all visits."
)

GERSHMAN_MAPPING_INSTRUCTION = (
    "You are presented with a series of stimuli, each associated with one of three possible responses.\n"
    "Your goal is to learn which response is the correct one for each stimulus.\n"
    "When a stimulus is presented, you can press one of three keys to respond.\n"
    "The three responses available are S, F, and A.\n"
    "After your response, you will receive feedback: 1 point for a correct response, or 0 points for an incorrect response.\n"
    "The correct response for one stimulus does not inform you about the correct response for another stimulus.\n"
    "You will play 13 games, each with a different mapping from stimuli to responses."
)

SPEEKENBRINK_WEATHER_INSTRUCTION = (
    "You will be playing a game in which you pretend to be a weather forecaster.\n"
    "In each trial, you will see between one and three tarot cards.\n"
    "Your task is to decide if the combination of cards presented predicts rainy weather (by pressing E) or fine weather (by pressing J)."
)

KOOL_WHEN_EXP1_INSTRUCTION = (
    "Each day you will either be presented with spaceships G and S or with spaceships T and N.\n"
    "These spaceships will take you to two different planets R and Z.\n"
    "You can take a spaceship by pressing the corresponding key.\n"
    "Each planet has one alien on it and each alien has its own space treasure mine.\n"
    "When you arrive at a planet, you will ask the alien for space treasure from its mine.\n"
    "When you ask the alien, you will find out whether you got space treasure.\n"
    "However, sometimes the alien's mine will dig up antimatter.\n"
    "Antimatter is bad because each piece will destroy a piece of space treasure, reducing the total amount of treasure that you have.\n"
    "The quality of each alien's mine will change during the game.\n"
    "Your goal is to get as much treasure and as little antimatter as possible over the next 125 days."
)

KOOL_WHEN_EXP2_INSTRUCTION = (
    "You will be taking one of the spaceships R or U to one of the planets J or T.\n"
    "The spaceships can fly to either planet, but one will mostly fly to planet J, and the other will mostly fly to planet T.\n"
    "The planet a spaceship goes to most won't change during the game.\n"
    "Planet J has aliens W and K, and planet T has aliens I and G on it.\n"
    "Each alien has its own space treasure mine.\n"
    "When you arrive at each planet, you will ask one of the aliens for space treasure from their mines.\n"
    "The treasure an alien can give will change slowly during the game.\n"
    "You can take a spaceship or ask an alien for space treasure by pressing the corresponding key.\n"
    "Your goal is to get as much treasure as possible over the next 125 days."
)

KOOL_COST_EXP2_INSTRUCTION = (
    "You will be taking one of the spaceships V or W to one of the planets I or R.\n"
    "The spaceships can fly to either planet, but one will mostly fly to planet I, and the other will mostly fly to planet R.\n"
    "The planet a spaceship goes to most won't change during the game.\n"
    "Planet I has aliens Q and G, and planet R has aliens B and X on it.\n"
    "Each alien has its own space treasure mine.\n"
    "When you arrive at each planet, you will ask one of the aliens for space treasure from their mines.\n"
    "The treasure an alien can give will change slowly during the game.\n"
    "Before you choose a spaceship, you will be told whether there is a treasure multiplier.\n"
    "If there is a treasure multiplier and you find treasure, you will receive 5 treasure pieces.\n"
    "If there is no treasure multiplier and you find treasure, you will receive 1 treasure piece.\n"
    "You can take a spaceship or ask an alien for space treasure by pressing the corresponding key.\n"
    "Your goal is to get as much treasure as possible over the course of the next 200 days."
)

KOOL_COST_EXP1_INSTRUCTION = (
    "Each day you will either be presented with spaceships P and F or with spaceships Z and J.\n"
    "These spaceships will take you to two different planets L and Q.\n"
    "You can take a spaceship by pressing the corresponding key.\n"
    "Each planet has one alien on it and each alien has its own space treasure mine.\n"
    "When you arrive at a planet, you will ask the alien for space treasure from its mine.\n"
    "When you ask the alien, you will find out whether you got space treasure.\n"
    "However, sometimes the alien will not bring up any treasure.\n"
    "The quality of each alien's mine will change during the game.\n"
    "Before you choose a spaceship, you will be told whether there is a treasure multiplier.\n"
    "If there is a treasure multiplier, you will receive 5 times the amount of treasure you will find.\n"
    "Your goal is to get as much treasure as possible over the next 200 days."
)

GERSHMAN_DECONSTRUCT_INSTRUCTION = (
    "In this task, you have to repeatedly choose between two slot machines labeled U and P.\n"
    "You can choose a slot machine by pressing its corresponding key.\n"
    "When you select one of the machines, you will win or lose points.\n"
    "Machine U will not always give you the same points when you select it again, but machine P will always give 0 points when you select it.\n"
    "Your goal is to choose the slot machines that will give you the most points.\n"
    "You will receive feedback about the outcome after making a choice.\n"
    "You will play 20 games in total, each with a different pair of slot machines.\n"
    "Each game will consist of 10 trials."
)

BAHRAMI_FOUR_ARM_INSTRUCTION = (
    "You will be asked to repeatedly choose between four different options labeled L, G, O, and U.\n"
    "You select an option by pressing the corresponding key on your keyboard.\n"
    "Each time you select an option, you will get a different number of points.\n"
    "Your goal is to win as many points as possible."
)

HILBIG_PRODUCT_INSTRUCTION = (
    "You are repeatedly presented with two options, labeled A and R.\n"
    "Each option represents a fictitious product and you have to infer which product is superior in terms of quality.\n"
    "You select a product by pressing the corresponding key.\n"
    "For each decision, you are provided with four expert ratings (with 1 representing a positive and 0 representing a negative rating).\n"
    "The four experts differ in their validity.\n"
    "The ratings of experts are given in descending order of their validity (having validities of 90%, 80%, 70%, and 60%)."
)

GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION = (
    "In this task, you have to repeatedly choose between two slot machines labeled K and S.\n"
    "You can choose a slot machine by pressing its corresponding key.\n"
    "When you select one of the machines, you will win or lose points.\n"
    "The machines will not always give you the same points when you select them again, but one slot machine is always better than the other.\n"
    "Your goal is to choose the slot machines that will give you the most points.\n"
    "You will receive feedback about the outcome after making a choice.\n"
    "You will play 20 games in total, each with a different pair of slot machines.\n"
    "Each game will consist of 10 trials."
)

WULFF_SAMPLING_INSTRUCTION = (
    "You can sample from two monetary lotteries by pressing K or D.\n"
    "The lotteries offer different points with different probabilities.\n"
    "Initially, you will not know the outcomes and probabilities of the lotteries, but you can learn about them through sampling.\n"
    "Whenever you sample, a random draw from the selected lottery will be generated, which does not affect your bonus.\n"
    "You can sample from the lotteries in whatever order and for as long as you like.\n"
    "Whenever you feel ready, you can stop sampling by pressing X and then choose one lottery for real by pressing the corresponding key.\n"
    "This choice will then trigger a random draw from the chosen lottery that will be added to your bonus.\n"
    "Your goal is to maximize your bonus.\n"
    "You will be presented with multiple choice problems consisting of different lotteries varying in outcomes and probabilities."
)

PLONSKY_GAMBLE_INSTRUCTION = (
    "You will encounter a series of gambling problems where you have to select between two options.\n"
    "You can select an option by pressing the corresponding key.\n"
    "You will encounter each problem 25 times.\n"
    "In the first five encounters, you will not receive feedback.\n"
    "In the remaining 20 encounters, you will receive feedback about the outcomes of both options.\n"
    "In cases where the probabilities are stated to be unknown, they sum up to one and remain constant within a problem."
)

WULFF_DESCRIPTION_INSTRUCTION = (
    "You will choose from two monetary lotteries by pressing W or H.\n"
    "The lotteries offer different points with different probabilities.\n"
    "Your choice will trigger a random draw from the chosen lottery that will be added to your bonus.\n"
    "Your goal is to maximize your bonus.\n"
    "You will be presented with multiple choice problems consisting of different lotteries varying in outcomes and probabilities."
)

STEINGROEVER_IGT_INSTRUCTION = (
    "You see in front of you four decks of cards labeled H, V, J, and D.\n"
    "You get a loan of 2000$ of play money.\n"
    "You have to select one card at a time, from any of the four decks, for 100 trials.\n"
    "You select a card from a deck by pressing the corresponding key.\n"
    "After turning a card, you win some money, the amount varies with the deck.\n"
    "You sometimes also have to pay a penalty, which also varies with the deck.\n"
    "Your goal is to maximize profit on the loan of the play money."
)

COX_PAIR_INSTRUCTION = (
    "You study a list of word pairs and then judge whether each test pair was on the studied list.\n"
    "Press D if the pair was studied and N if it was not."
)

TOMOV_CASTLE_INSTRUCTION = (
    "You will explore a castle, walking from room to room.\n"
    "In each room, you will find different amounts of resources: wood, stone, and iron.\n"
    "In each room, there are three doors that lead to different rooms.\n"
    "You choose a door by pressing the corresponding key.\n"
    "At the beginning of each round, you will be shown how valuable the resources are.\n"
    "These values are given as market prices for wood, stone, and iron.\n"
    "Multiplying the prices with the amounts of resources and adding them up yields a reward.\n"
    "You want to maximize the cumulative reward.\n"
    "After every round, you will start in room 0 again and see the new market prices."
)

SCHULZ_FINDING_INSTRUCTION = (
    "You will be playing a game for 30 rounds.\n"
    "Each round contains 10 trials.\n"
    "In each trial, you have to select one option that will generate a reward between 0 and 50 points.\n"
    "You can choose between options 1, 2, 3, 4, 5, 6, 7 and 8 by pressing the corresponding key.\n"
    "After each round the options reset and each option can produce different rewards in the following round.\n"
    "Your goal is to maximize your reward."
)

FLESCH_TREE_INSTRUCTION = (
    "You are going to plant trees in two different gardens labeled North and South.\n"
    "The trees look different from each other regarding their leafiness and branchiness.\n"
    "There are 5 levels of leafiness (0, 1, 2, 3, 4) and 5 levels of branchiness (0, 1, 2, 3, 4).\n"
    "In each round, you get presented with a tree.\n"
    "You can accept to plant the tree by pressing T and reject to plant it by pressing N.\n"
    "If you accept to plant the tree and your answer is correct, you will be rewarded with points, otherwise, you will lose some points.\n"
    "If you reject to plant the tree, you will not be rewarded (0 points).\n"
    "Your task is to learn which type of tree grows best in each garden.\n"
    "During the training phase, there will be feedback on every trial about your decisions.\n"
    "During the testing phase, there will be no feedback for your decision."
)

DIGIT_SPAN_INSTRUCTION = (
    "You will view a series of digits and are then asked to recall them in the order you have seen them "
    "by pressing the corresponding keys.\n"
    "After having recalled all digits, please press 'S' to indicate the end of your recalled sequence."
)

GONOGO_INSTRUCTION = (
    "In this task, you need to emit responses to certain stimuli and omit responses to others.\n"
    "You will see one of two colours, colour1 or colour2, on the screen in each trial.\n"
    "You need to press button X when you see colour1 and press nothing when you see colour2.\n"
    "You need to respond as quickly as possible.\n"
    "You will be doing 10 practice trials followed by 350 test trials."
)

STEINGROEVER_IGT_EXP3_INSTRUCTION = (
    "You see in front of you four decks of cards labeled U, F, I, and S.\n"
    "You get a loan of 2000$ of play money.\n"
    "You have to select one card at a time, from any of the four decks, until you are told to stop.\n"
    "You select a card from a deck by pressing the corresponding key.\n"
    "After turning a card, you win some money, the amount varies with the deck.\n"
    "You sometimes also have to pay a penalty, which also varies with the deck.\n"
    "Your goal is to maximize profit on the loan of the play money."
)

FREY_CCT_INSTRUCTION = (
    "You will play a games with 84 rounds.\n"
    "In each round, you will be presented with 32 face-down cards.\n"
    "Every card is either a gain card or a loss card.\n"
    "If you turn over a gain card, the gain amount of that card (between 10 and 600 points) will be added to your current game score.\n"
    "If you turn over a loss card, the loss amount of that card (between 25 and 750 points) will be subtracted from your game score.\n"
    "In different rounds, between 1 and 28 cards are loss cards.\n"
    "Loss and gain amounts also differ between rounds.\n"
    "You may keep turning over cards as long as you keep encountering gain cards.\n"
    "You may also stop the round at any point and claim your current payout.\n"
    "If you encounter a loss card, the round ends immediately.\n"
    "Your gains and losses will be summed up to give you your final score for each round.\n"
    "Press E to turn a card over, or C to stop the round and claim your current payout."
)

ENKAVI_RECENT_PROBE_INSTRUCTION = (
    "You will see a set of letters followed by a probe letter.\n"
    "Press one key if the probe was in the set and another if it was not."
)

COLLSI_JUDGMENT_INSTRUCTION = (
    "Estimate Caldionine concentration from Progladine and Amalydine levels.\n"
    "You receive feedback on your estimate during training; feedback may stop later."
)

GARCIA_EXPERIENTIAL_INSTRUCTION = (
    "Choose between letter-labeled options across three parts.\n"
    "Part 1: learn advantageous options. Part 2: letter and described options.\n"
    "Part 3: test knowledge with the same press interface."
)

KRUEGER_IDENTIFYING_INSTRUCTION = (
    "Choose among six gambles; payoffs depend on a sampled ball color.\n"
    "Each round lists color counts in the jar before you choose a gamble."
)

TOMOV_SUBWAY_INSTRUCTION = (
    "Navigate an unfamiliar subway from a start station to a goal station.\n"
    "Move north, west, south, or east when a neighbor exists; press the goal key at the goal."
)
