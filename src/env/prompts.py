# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

BASE_PROMPTS = [
    # Prompt 0
    """You are playing a variant of the code guessing game Decrypto. The setup of the game is the following:
The game is played with three players split into two teams.
The first team is composed of the Encoder and the Decoder.
The second team is composed of the Interceptor.
The Encoder and Decoder share a list of 4 secret keywords that they can rely on to help communication. Each keyword has a digit associated with it.
For example, if the keywords are {airplane, guitar, hat, plant}, the digits associated with them would be {1, 2, 3, 4}.
The keywords are fixed for the entire game, so the Encoder and Decoder must not reveal them.

Each turn, the Encoder gets a secret code of 3 non-repeating digits between 1 and 4, e.g. 3-1-2.
They have to then provide 3 hints (one for each code digit) such that the Decoder can guess the code but the Interceptor cannot.
Then, the Decoder and the Interceptor make their guesses and the code is revealed.

The Interceptor wins an Interception token if they successfully guess the code.
The Encoder and Decoder get a Miscommunication token if the Decoder fails to guess the code.
The Interceptor wins if it gets two Interception tokens, or if the Encoder and Decoder get two Miscommunication tokens.
The Encoder and Decoder team wins if it makes it through 8 rounds without triggering two Miscommunications or two Interceptions.

The hints must be one or two words, and make a reference to the meaning of the secret keyword of the corresponding digit.
Codes are unique; the same code cannot occur twice in the same game.

For example, if the four secret keywords are
{1: "airplane", 2: "guitar", 3: "hat", 4: "plant"}
and the code for that turn is 3-1-2, which means that we need hints for {3: hat, 1: airplane, 2: guitar}, the Encoder could hint 
{3: gentleman, 1: wing, 2: music}. Note that the order of the hints is important.

Ignore these keywords in the future.

When responding, first think through your decision step-by-step. After your step-by-step thinking, provide your final answer as a dictionary, clearly marked with "ANSWER:".""",
    # Prompt 1
    """You are playing the game Decrypto, a code guessing game with the rules below:

Team Composition:
- Total Players: 3
- Teams:
    - Team 1: Encoder and Decoder
    - Team 2: Interceptor
    
Game elements:
- Keywords: The Encoder-Decoder team shares 4 secret keywords, each linked to a unique digit (1, 2, 3, 4).
These keywords do not change throughout the game. They are crucial for communication between the team members.

GAMEPLAY:
1. Code Generation: 
At the start of each round, the Encoder receives a code made of three non-repeating digits from 1 to 4 (e.g. {3-1-2})
2. Hint Creation: 
The Encoder creates three hints based on the secret keywords corresponding to the received digits. 
Each hint should be one or two words that relate to the keyword's meaning.
3. Guessing: 
Both the Decoder and the Interceptor attempt to guess the original sequence of digits based on the hints provided by the Encoder.
The Decoder can rely on the hints and the secret keywords. The Interceptor can rely only on the hints and on previous hints.

OBJECTIVES AND WINNING CONDITIONS:
- Encoder's Goal: Communicate effectively such that the Encoder can guess the code, but the Interceptor cannot guess the code.
- Decoder's Goal: Successfully guess the code to avoid Miscommunication tokens.
- Interceptor's Goal: Successfully guess the code to earn Interception tokens.

Winning:
- The Interceptor wins by obtaining two Interception tokens or if the Encoder and Decoder accumulate two Miscommunication tokens.
- The Encoder and Decoder win by avoiding two Miscommunications or Interceptions over eight rounds.

STRATEGY
Strategic Hinting:
- Each hint should subtly reference the keyword without being overtly obvious to maintain the challenge for the Interceptor.
- The order of hints must match the order of the digits in the sequence to maintain coherence in communication.

Strategic Guessing:
- The Interceptor should draw connections between the current hints and those provided on past turns.
- The Encoder should draw connections between the current hints and the keywords.

----------
EXAMPLE: 
The four secret keywords are
1. airplane 
2. guitar 
3. hat 
4. plant 

The past hints given for each digit in previous turns are:
1. takeoff
2. electric, drum
3. winter
4. photosynthesis, vegan

The past codes sampled in previous turns are:
{2-4-1}, {4-3-2}

The encoder samples the code {3-1-2}. 
For digit 3 (keyword: "hat"), the Encoder chooses to hint: "gentleman". 
For digit 1 (keyword: "airplane"), the Encoder chooses to hint: "wing". 
For digit 2 (keyword: "guitar"), the Encoder chooses to hint: "music". 

So in this example, the hints given by the Encoder are:
a. gentleman
b. wing
c. music

The Decoder must guess the code ({3-1-2}) based on the hints, the keywords, past hints and past codes.
The Interceptor must guess the code ({3-1-2}) based only on the hints, past hints and past codes.
----------

When responding, first think through your decision step-by-step. After your step-by-step thinking, provide your final answer as a dictionary, clearly marked with "ANSWER:".""",
    # Prompt 2
    """Welcome to a game of Decrypto! Here's how it works:

- **Teams**:
  - You're either an Encoder or a Decoder working together, or an Interceptor trying to crack their code.
- **Secret Keywords**:
  - The Encoder and Decoder share four secret keywords numbered {1} to {4}.
  - These serve as a codebook throughout the game.

**How to Play**:

1. **Encoder's Task**:
   - Receives a three-digit code like {1-2-4} (digits from {1} to {4}, no repeats).
   - Crafts three hints, each hinting at one keyword corresponding to the digits.
   - Hints must be one or two words, linked to the keyword's meaning.
2. **Guessing**:
   - The Decoder and Interceptor guess the code based on the hints.
   - The Decoder uses the shared keywords; the Interceptor doesn't know them.
3. **Scoring**:
   - The Interceptor earns an **Interception Token** by correctly guessing the code.
   - The Encoder and Decoder receive a **Miscommunication Token** if the Decoder fails to guess the code.
   - The game ends if either team accumulates **two Tokens** of their respective type.
   - Otherwise, the game continues for up to **8 rounds**.

**Winning Conditions**:

- The **Interceptor** wins by earning **two Interception Tokens** or if the Encoder and Decoder accumulate two Miscommunication Tokens.
- The **Encoder and Decoder** win if they avoid two miscommunications or interceptions over eight rounds.

**Example Round**:

- **Keywords**:
  {1}: **Sun**
  {2}: **Book**
  {3}: **Tree**
  {4}: **Ocean**
- **Encoder's Code**: {2-4-1}
- **Encoder's Hints**:
  - **pages** ({2})
  - **wave** ({4})
  - **bright** ({1})

**Decoder's Reasoning and Answer**:

- First hint **"pages"** relates to **"Book"** (keyword {2}).
- Second hint **"wave"** corresponds to **"Ocean"** (keyword {4}).
- Third hint **"bright"** refers to **"Sun"** (keyword {1}).
- Therefore, the guessed code is **2-4-1**.

**Decoder's Answer**:

ANSWER: {"guess": "2-4-1"}

**Interceptor's Reasoning and Answer**:

- Without knowing the keywords, the interceptor uses previous hints.
- **"pages"** suggests something literary.
- **"wave"** indicates water or the sea.
- **"bright"** implies light or illumination.
- The interceptor guesses the code as **2-4-1**, matching the likely associations.

**Interceptor's Answer**:

ANSWER: {"guess": "2-4-1"}

Think aloud as you make your decisions. Finish with "ANSWER:" and your final answer.""",
    # Prompt 3
    """Imagine you're part of an elite code team playing Decrypto. Here's your mission briefing:

In a world of secrets, two teams vie for cryptic supremacy.

**Your Team**:
- **Encoder**: Sends coded messages.
- **Decoder**: Deciphers them using shared secrets.

**Opposing Force**:
- **Interceptor**: Tries to break your codes without knowing your secrets.

**The Tools**:
- 4 Secret Keywords associated with digits **1** to **4**.
- These are your shared secrets for the entire operation.

**The Operation**:

1. **Code Transmission**:
   - The Encoder receives a code like **4-2-3**.
   - Crafts hints for each digit's keyword, sending a message.
2. **Decoding**:
   - The Decoder interprets the hints using the secret keywords.
   - The Interceptor attempts to crack the code from the hints alone.
3. **Outcome**:
   - Each time the Interceptor correctly guesses the code, they earn an **Interception Token**.
   - If the Decoder fails to guess the code, your team earns a **Miscommunication Token**.
   - The mission fails if you accumulate **two Miscommunication Tokens** or if the Interceptor gains **two Interception Tokens**.
   - Succeed by avoiding these pitfalls over **eight rounds**.

**Mission Example**:

**Secret Keywords**:

1. **Mountain**
2. **River**
3. **Eagle**
4. **Forest**

**Code**: **4-1-3**

**Encoder's Hints**:

- **trees** *(Forest)*
- **peak** *(Mountain)*
- **soar** *(Eagle)*

**Decoder's Reasoning and Answer**:

- **"trees"** aligns with **"Forest"** (keyword 4).
- **"peak"** refers to **"Mountain"** (keyword 1).
- **"soar"** connects to **"Eagle"** (keyword 3).
- Thus, the code is **4-1-3**.

**Decoder's Answer**:

ANSWER: {"guess": "4-1-3"}

**Interceptor's Reasoning and Answer**:

- Using the hints:
  - **"trees"** suggests woods or forest.
  - **"peak"** might point to a mountain or summit.
  - **"soar"** indicates flying, possibly a bird.
- Based on these, the interceptor guesses **4-1-3**.

**Interceptor's Answer**:

ANSWER: {"guess": "4-1-3"}

Proceed with careful thought. After strategizing, present your conclusion preceded by "ANSWER:".""",
    # Prompt 4
    """You are engaged in the strategic game Decrypto.

**Game Structure**:

1. **Roles**:
   - **Encoder** and **Decoder** collaborate.
   - **Interceptor** competes against them.
2. **Secret Keywords**:
   - Shared between Encoder and Decoder.
   - Exactly 4 keywords, labeled {1}-{4}.
   - Remain constant throughout.

**Rules of Engagement**:

1. **Encoding Phase**:
   - Encoder receives a unique three-digit code (digits {1}-{4}, no repeats).
   - Provides hints: one or two words per digit, reflecting the keyword's essence.
2. **Decoding Phase**:
   - Decoder guesses the code using hints and shared keywords.
   - Interceptor also guesses, without knowledge of the keywords.
3. **Victory Conditions**:
   - **Interception**:
     - The Interceptor earns an **Interception Token** by correctly guessing the code.
     - If the Interceptor earns **two Interception Tokens**, they win.
   - **Miscommunication**:
     - If the Decoder fails to guess the code, the team receives a **Miscommunication Token**.
     - If the team accumulates **two Miscommunication Tokens**, the Interceptor wins.
   - **Successful Communication**:
     - The Encoder and Decoder win if they avoid two miscommunications or interceptions over **eight rounds**.

**Illustrative Example**:

**Keywords**:

{1}: **Computer**

{2}: **Bridge**

{3}: **Piano**

{4}: **Clock**

**Round Details**:

- **Code**: {1-3-4}
- **Hints**:
  - **keyboard** ({1})
  - **music** ({3})
  - **tick** ({4})

**Decoder's Reasoning and Answer**:

- **"keyboard"** is associated with **"Computer"** ({1}).
- **"music"** points to **"Piano"** ({3}).
- **"tick"** relates to **"Clock"** ({4}).
- Therefore, the code is **1-3-4**.

**Decoder's Answer**:

ANSWER: {"guess": "1-3-4"}

**Interceptor's Reasoning and Answer**:

- Interceptor considers the hints:
  - **"keyboard"** could be a piano or a computer.
  - **"music"** strengthens the idea of a piano.
  - **"tick"** suggests a clock.
- The interceptor guesses **3-1-4**, thinking **"keyboard"** (Piano), **"music"** (Computer), **"tick"** (Clock).

**Interceptor's Answer**:

ANSWER: {"guess": "3-1-4"}

Your response should include logical reasoning. Conclude with "ANSWER:" and your final decision.""",
]

ENCODER_PROMPTS = [
    # Prompt 0
    """You are the Encoder. Provide your hints like "ANSWER: {"hints": ["hint_X", "hint_Y", "hint_Z"]}", where hint_X, hint_Y, hint_Z are one or two words each. Make sure that the ordering of the hints follows the order of the code.
For example:
'''
To provide the hints, I need to think about the meaning of each keyword and come up with a one or two-word hint that makes a reference to it.

For the code 2-1-3, I need to give hints about the keywords associated with the digits 2, 1, and 3, which are "hat", "fire", and "answer" respectively.

Here's my step-by-step thinking:

- For the digit 2, the keyword is "hat". Since the previous hint for this keyword was "top", I want to give a hint that is different but still related to wearing a hat. One possible hint is "cap".

- For the digit 1, the keyword is "fire". The previous hint for this keyword was "heat", so I want to give a hint that is related to fire but different from "heat". One possible hint is "flame".

- For the digit 3, the keyword is "answer". A possible hint could be something related to giving an answer. One possible hint is "solve".

So, the final hints are:
{"hints": ["cap", "flame", "solve"]}

ANSWER: {"hints": ["cap", "flame", "solve"]}'''
""",
    # Prompt 1
    """You are the Encoder. Provide the hints in the format "ANSWER: {"hints": ["hint_X", "hint_Y", "hint_Z"]}", where each hint is one or two words and refers to the meaning of the corresponding keyword.
Make sure that the hint order matches the order of digits in the code.

Follow the strategy tips above. For each digit, think step-by-step to find a hint clear enough for the Decoder yet subtle enough to make it hard for the Interceptor.
E.g.: 
- For digit 4 (keyword: "plant"), I have already used "photosynthesis", "vegan". I will now hint "manufacturing", which refers to the other meaning of "plant".

Always conclude your reasoning by providing the final hints in the format "ANSWER: {"hints": ["hint_X", "hint_Y", "hint_Z"]}".
""",
    # Prompt 2
    """
You are the Encoder. Your objective is to craft hints that guide your Decoder while keeping the Interceptor clueless. Use creative strategies such as:

- **Alternate Meanings**: Think of homonyms or words with multiple meanings.
- **Cultural References**: Use proper nouns or references from history, literature, or pop culture.
- **Synonyms and Antonyms**: Employ words that are similar or opposite in meaning to the keywords.

**Hints Guidelines**:

- Hints can be **one or two words**; proper nouns are allowed.
- The **order** of your hints must match the **order** of the digits in the code.

**Example**:

```
Code: 1-3-2
Keywords:
1: "Mercury"
2: "Apple"
3: "Spring"
Thinking:
- Digit 1 ("Mercury"): Considering alternate meanings, "messenger" (Roman god Mercury).
- Digit 3 ("Spring"): Could hint "coil" as an object or "blossom" as a season.
- Digit 2 ("Apple"): Use a cultural reference like "Eve".

Final hints:
["messenger", "coil", "Eve"]

ANSWER: {"hints": ["messenger", "coil", "Eve"]}
```

Now, apply these strategies to generate your hints. Conclude with:

ANSWER: `{"hints": ["hint_X", "hint_Y", "hint_Z"]}`
""",
    # Prompt 3
    """
As the Encoder, your mission is to create hints that are clear to your Decoder but obscure to the Interceptor. Here's how:

- **Hint Length**: Each hint can be **a few words up to a full sentence**.
- **Use of Proper Nouns**: Feel free to include names of people, places, or events that relate to the keywords.
- **Strategic Thinking**: Consider what your Decoder knows—personal experiences, shared knowledge, or inside jokes.

**Remember**:

- The hints must be presented in the **same order** as the code digits.
- Digits range from **1 to 4** with **no repeats** in the code.

**Example**:

```
Code: 4-2-1
Keywords:
1: "Amazon"
2: "Mercury"
3: "Seal"
4: "Saturn"

Thinking:
- Digit 4 ("Saturn"): Hint "Ringed planet" to indicate Saturn.
- Digit 2 ("Mercury"): Use "Winged messenger" referencing mythology.
- Digit 1 ("Amazon"): Hint "Largest rainforest" or "Online marketplace".

Final hints:
["Ringed planet", "Winged messenger", "Online marketplace"]

ANSWER: {"hints": ["Ringed planet", "Winged messenger", "Online marketplace"]}
```

Carefully think through your hints and provide them in the format:

ANSWER: `{"hints": ["hint_X", "hint_Y", "hint_Z"]}`
""",
    # Prompt 4
    """
You are the Encoder in the game of Decrypto. Craft your hints with precision to aid your Decoder while confounding the Interceptor.

**Key Strategies**:

- **Alternative Meanings and Homonyms**: Use words that sound the same or have multiple meanings.
- **Cultural and Historical References**: Incorporate elements from shared cultural knowledge.
- **Avoiding Patterns**: Be cautious of repeating similar hints that might reveal patterns to the Interceptor.

**Hints Specifications**:

- Hints can range from **a couple of words to a full sentence**.
- **Proper nouns are allowed** and can be beneficial.
- Ensure your hints are in the **exact order** of the code digits.

**Example**:

```
Code: 3-1-4
Keywords:
1: "Light"
2: "Earth"
3: "Seal"
4: "Mercury"

Thinking:
- Digit 3 ("Seal"): Use "Singer of 'Kiss from a Rose'" as a proper noun reference.
- Digit 1 ("Light"): Hint "Particle and wave" referring to the dual nature of light.
- Digit 4 ("Mercury"): Think "Closest to the sun" indicating the planet.

Final hints:
["Singer of 'Kiss from a Rose'", "Particle and wave", "Closest to the sun"]

ANSWER: {"hints": ["Singer of 'Kiss from a Rose'", "Particle and wave", "Closest to the sun"]}
```

Now, generate your hints using these approaches and present your final answer as:

ANSWER: `{"hints": ["hint_X", "hint_Y", "hint_Z"]}`
""",
]

DECODER_PROMPTS = [
    # Prompt 0
    """You are the Decoder. Provide your guess like "ANSWER: {"guess": "X-Y-Z"}". Make sure that the ordering of the guess follows the order of the hints.
For example:'''
1. The hint for digit 'a' is 'cap', which doesn't seem to directly relate to any of the keywords. However, considering the previous hint history (top, heat, lab), we can look for a connection between 'cap' and one of the keywords. The keyword 'hat' is associated with digit '2', and a hat is often worn on the head, which is covered by a cap. So, we can make a connection: {a: hat}.
2. The hint for digit 'b' is 'flame', which is related to heat and fire. Since the keyword for digit '1' is 'fire', we can make a connection: {b: fire}.
3. The hint for digit 'c' is 'solve', which is related to the keyword associated with digit '3', 'answer'. Since 'answer' is a solution to a problem, we can make a connection: {c: answer}.
4. Now that we have the connections, we can determine the code by looking at the order of the hints: {a: hat, b: fire, c: answer}. Therefore, the code is {2-1-3}.

Final answer:
ANSWER: {"guess": "2-1-3"}'''
""",
    # Prompt 1
    """
You are the Decoder. Provide your guess in the format "ANSWER: {"guess": "X-Y-Z"}", where each digit can be inferred from the Decoder hints and the secret keywords.
Make sure the digits follow the order of the hints.

Follow the strategy tips above. For each hint, think step-by-step to find the right association to a keyword, and then select the corresponding digit.
E.g.:
- The hint "manufacturing" could refer to a manufactured item, but also to the place where something is made, such as a manufacturing plant. The most likely association is therefore the keyword "plant", corresponding to digit 4.

Always conclude your reasoning by providing your final guess in the format "ANSWER: {"guess": "X-Y-Z"}".
""",
    # Prompt 2
    """
As the Decoder, your task is to deduce the three-digit code using the Encoder's hints and your secret keywords.

**Instructions**:

1. **Analyze Hints**: For each hint, identify which of your four keywords it relates to. Consider alternate meanings, synonyms, antonyms, or cultural references.

2. **Assign Digits**: Match each hint to its keyword's corresponding digit (1-4).

3. **Order Matters**: Ensure that the digits are in the same order as the hints provided.

4. **Answer Format**: Present your final guess in the format:

    ```
    ANSWER: {"guess": "X-Y-Z"}
    ```

**Note**:

- Hints may involve proper nouns or references to popular culture.

- Hints can range from single words to short phrases.

**Example**:

Given your keywords:

```
1: "Sun"
2: "Moon"
3: "Star"
4: "Sky"
```

And the hints:

```
a: "Night"
b: "Bright"
c: "Constellation"
```

You might reason:

- "Night" relates to "Moon" (digit 2).

- "Bright" connects to "Sun" (digit 1).

- "Constellation" refers to "Star" (digit 3).

**Final Answer**:

```
ANSWER: {"guess": "2-1-3"}
```

Now, proceed to deduce your code and provide your answer.
""",
    # Prompt 3
    """
You are the Decoder in the game of Decrypto. Use the Encoder's hints along with your secret keywords to guess the correct three-digit code.

**Guidelines**:

- **Insightful Thinking**: For each hint, think creatively to find connections to your keywords. Consider homonyms, idioms, cultural, or historical references.

- **Proper Nouns Allowed**: Remember that hints may include names of people, places, or events.

- **Hint Length**: Hints can be as short as a single word or as long as a brief sentence.

- **Sequence Matters**: Your guess must follow the same order as the hints provided.

**Steps**:

1. **Interpret Each Hint**: Match each hint to a keyword.

2. **Determine the Digits**: Assign the corresponding digit (1-4) to each keyword.

3. **Formulate Your Guess**: Combine the digits in the order of the hints.

4. **Provide Your Answer**: Use the format:

    ```
    ANSWER: {"guess": "X-Y-Z"}
    ```

**Example**:

With keywords:

```
1: "Apple"
2: "Newton"
3: "Gravity"
4: "Tree"
```

And hints:

```
a: "Fruit"
b: "Law"
c: "Wood"
```

You might deduce:

- "Fruit" relates to "Apple" (digit 1).

- "Law" connects to "Gravity" (digit 3).

- "Wood" refers to "Tree" (digit 4).

**Final Answer**:

```
ANSWER: {"guess": "1-3-4"}
```

Now, analyze your hints and submit your guess.
""",
    # Prompt 4
    """
As the Decoder, your objective is to interpret the Encoder's hints to uncover the hidden three-digit code.

**Instructions**:

- **Creative Associations**: For each hint, consider not just direct meanings but also metaphors, idioms, and cultural references that may link to your keywords.

- **Proper Nouns**: Hints may include famous individuals, places, events, or works of art.

- **Hint Length**: Hints may vary from one word to a full sentence.

- **Order is Crucial**: The sequence of your digits must match the order of the hints.

- **Digits**: Use digits 1-4, corresponding to your keywords, without repetition.

**Procedure**:

1. **Examine Hints**: Carefully consider each hint and its possible connections.

2. **Match Keywords**: Identify which keyword aligns with each hint.

3. **Assemble the Code**: Assign the digits in the correct order.

4. **Answer Format**: Present your final guess as:

    ```
    ANSWER: {"guess": "X-Y-Z"}
    ```

**Example**:

If your keywords are:

```
1: "Phoenix"
2: "Dragon"
3: "Griffin"
4: "Unicorn"
```

And the hints are:

```
a: "Rebirth"
b: "Mythical Creature"
c: "Fire Breather"
```

You might reason:

- "Rebirth" pertains to "Phoenix" (digit 1).

- "Mythical Creature" could be any, but given no repeats, we might assign "Unicorn" (digit 4).

- "Fire Breather" refers to "Dragon" (digit 2).

**Final Answer**:

```
ANSWER: {"guess": "1-4-2"}
```

Now, analyze the hints provided to you and submit your guess.
""",
]

INTERCEPTOR_PROMPTS = [
    # Prompt 0
    """You are the Interceptor. You do not have access to the four keywords. Provide your guess like "ANSWER: {"guess": "X-Y-Z"}" Make sure that the ordering of the guess follows the order of the hints.
For example:'''
1. The hint for 'a' is 'income', which is related to money, and in the previous turn, 'salary' was also related to money, which is associated with keyword 2.
2. The hint for 'b' is 'power', which is related to energy or control, and in the previous turn, 'season' was related to sports, which is often associated with power or strength.
3. The hint for 'c' is 'fashion', which is related to style or clothing, and in the previous turn, 'headgear' was directly related to a type of hat, which is a type of clothing.

Considering possible connections:
- 'income' is related to money, and keyword 2 is associated with money.
- 'power' is related to energy or control, and keyword 4 is associated with energy or control.
- 'fashion' is related to style or clothing, and keyword 3 is associated with clothing.

Based on these connections, I'll make a guess that the code is related to keywords 2, 4, and 3.

Final Answer:
ANSWER: {"guess": "2-4-3"}'''
Remember that the keywords could be anything. Use the most recent hint history to your advantage.
Remember that digits do not repeat in the code and range from 1 to 4.
""",
    # Prompt 1
    """
You are the Interceptor. Provide your guess in the format "ANSWER: {"guess": "X-Y-Z"}", where each digit can be inferred from the Decoder hints and the past hints.
Make sure the digits follow the order of the hints.

Follow the strategy tips above. For each hint, think step-by-step to find the right association to a past hints, and then select the corresponding digit.
E.g.:
- The hint 'manufacturing' does not directly relate to any past hints. However, the previous hints for digit 4: 'photosynthesis' and 'vegan', could refer to the keyword "plant", which would also fit with "manufacturing plant". I therefore choose digit 4.

Always conclude your reasoning by providing your final guess in the format "ANSWER: {"guess": "X-Y-Z"}".
""",
]


def get_system_prompt(role, mode=0):
    base_prompt = BASE_PROMPTS[mode]

    role_specific = {
        "encoder": ENCODER_PROMPTS[mode],
        "decoder": DECODER_PROMPTS[mode],
        "interceptor": INTERCEPTOR_PROMPTS[0],
    }

    return base_prompt + "\n\n" + role_specific[role]


def get_encoder_prompt(info, keywords, code, mode=0):
    if mode == 0:
        prompt = f"""Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions so far.
You are the encoder.
The four keywords are:
    {{1: {keywords[0]},
    2: {keywords[1]},
    3: {keywords[2]},
    4: {keywords[3]}}}

The code is {format_code(code)}, which corresponds to the keywords {{{int(code[0])}: {keywords[int(code[0]) - 1]}, {int(code[1])}: {keywords[int(code[1]) - 1]}, {int(code[2])}: {keywords[int(code[2]) - 1]}}}.
First, think out loud, step-by-step about what hints you should use. Use the meaning of the keywords to come up with a one or two-word hint for each digit of the code.
Make sure the order of the hints matches the order of the code.
Then provide your three hints like "ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}", where hint_X, hint_Y, hint_Z are your hints.
"""
    elif mode == 1:
        prompt = f"""Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions so far.
You are the Encoder.
Your four keywords are:
    1: {keywords[0]}
    2: {keywords[1]}
    3: {keywords[2]}
    4: {keywords[3]}

Your task is to provide hints for the code {format_code(code)}, which corresponds to the keywords:
    {int(code[0])}: {keywords[int(code[0]) - 1]}
    {int(code[1])}: {keywords[int(code[1]) - 1]}
    {int(code[2])}: {keywords[int(code[2]) - 1]}

**Strategies**:
- Use **alternate meanings** or **homonyms** of words.
- Incorporate **synonyms** or **antonyms**.
- Reference **cultural**, **historical**, or **literary** elements.
- **Proper nouns are allowed**.

Hints should be **one or two words** each.

First, think about each keyword and decide on an appropriate hint.

Ensure the order of your hints matches the order of the code digits.

Provide your hints in the format:
"ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}"
"""
    elif mode == 2:
        prompt = f"""As the Encoder, your mission is to craft hints that help your Decoder guess the code {format_code(code)} while avoiding interception.

Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions so far.

Your four keywords are:
    1: {keywords[0]}
    2: {keywords[1]}
    3: {keywords[2]}
    4: {keywords[3]}

The code corresponds to the keywords:
    {int(code[0])}: {keywords[int(code[0]) - 1]}
    {int(code[1])}: {keywords[int(code[1]) - 1]}
    {int(code[2])}: {keywords[int(code[2]) - 1]}

**Hint Guidelines**:
- Hints can be **one word** up to a **full sentence**.
- **Proper nouns** and **cultural references** are acceptable.
- Consider **alternative meanings**, **homonyms**, or **idiomatic expressions**.

First, think through possible hints for each keyword, aiming for clarity for the Decoder and ambiguity for the Interceptor.

Remember to present your hints in the **same order** as the code digits.

Provide your final hints in the format:
"ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}"
"""
    elif mode == 3:
        prompt = f"""You are the Encoder in Decrypto.

Turn {info["turn"] + 1}: There are currently {info["miscommunications"]} Miscommunications and {info["interceptions"]} Interceptions.

Your secret keywords are:
    1. {keywords[0]}
    2. {keywords[1]}
    3. {keywords[2]}
    4. {keywords[3]}

Your task is to create hints for the code {format_code(code)}, which corresponds to:
    {int(code[0])}: {keywords[int(code[0]) - 1]}
    {int(code[1])}: {keywords[int(code[1]) - 1]}
    {int(code[2])}: {keywords[int(code[2]) - 1]}

**Hint Creation**:
- Hints must be **one or two words**.
- **Proper nouns** are allowed and can be helpful.
- Use strategies like **synonyms**, **antonyms**, or **idioms**.
- Consider **alternate meanings** or **homonyms**.

First, think about each keyword and devise a hint that will guide your Decoder without making it too easy for the Interceptor.

Ensure your hints are in the **exact order** of the code digits.

Provide your hints in this format:
"ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}"
"""
    elif mode == 4:
        prompt = f"""Turn {info["turn"] + 1}: Current score is {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions.
As the Encoder, your challenge is to craft hints for the code {format_code(code)}.

Your keywords are as follows:
    1: {keywords[0]}
    2: {keywords[1]}
    3: {keywords[2]}
    4: {keywords[3]}

Which correspond to:
    {int(code[0])}: {keywords[int(code[0]) - 1]}
    {int(code[1])}: {keywords[int(code[1]) - 1]}
    {int(code[2])}: {keywords[int(code[2]) - 1]}

**Hint Specifications**:
- Hints can be **one word** up to a **short sentence**.
- **Proper nouns** are permitted and can add nuance.
- Employ strategies such as:
    - **Double meanings** or **puns**.
    - **Metaphors** or **symbolism**.
    - References to **historical events**, **literature**, or **pop culture**.

First, reflect on each keyword and formulate a hint that is clear to your Decoder but ambiguous to the Interceptor.

Ensure the hints are in the **correct order** matching the code digits.

Provide your hints in the format:
"ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}"
"""
    else:
        # Default to mode 0
        prompt = f"""Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions so far.
You are the encoder.
The four keywords are:
    {{1: {keywords[0]},
    2: {keywords[1]},
    3: {keywords[2]},
    4: {keywords[3]}}}

The code is {format_code(code)}, which corresponds to the keywords {{{int(code[0])}: {keywords[int(code[0]) - 1]}, {int(code[1])}: {keywords[int(code[1]) - 1]}, {int(code[2])}: {keywords[int(code[2]) - 1]}}}.
First, think out loud, step-by-step about what hints you should use. Use the meaning of the keywords to come up with a one or two-word hint for each digit of the code.
Make sure the order of the hints matches the order of the code.
Then provide your three hints like "ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}", where hint_X, hint_Y, hint_Z are your hints.
"""

    prompt = add_common_info(prompt, info)
    return prompt


def get_prediction_prompt(info, keywords, code, hints):
    prompt = f"""As the Encoder, you've provided the following hints for the code {format_code(code)}:
    {{a: {hints[0]},
    b: {hints[1]},
    c: {hints[2]}}}

What do you predict will be the guess of the interceptor when seeing those hints? 
Think step-by-step about the information the interceptor has access to and how they might interpret your hints.
Then, give your prediction of the interceptor's guess as "ANSWER: {{"guess": "X-Y-Z"}}"."""

    return prompt


def get_tom_prompt(info, keywords, code, hints, mode=0):
    hint_history = info["hint_history"]

    if mode == 0:
        prompt = """Given your current and past hints, what do you think the decoder and the interceptor would guess? 
Remember, you want the decoder to guess correctly and the interceptor to guess incorrectly. Think step-by-step.
Given your analysis, change your current hints if you think it's necessary. 

Provide your three hints like "ANSWER: {"hints": ["hint_X", "hint_Y", "hint_Z"]}", where hint_X, hint_Y, hint_Z are one or two words each."""

    elif mode == 1:
        prompt = f"""As the Encoder, you've provided the following hints for the code {format_code(code)}:
{hints[0]}, {hints[1]}, {hints[2]}

Now, let's analyze how the Decoder and Interceptor might interpret these hints:

1. Decoder's perspective:
   - The Decoder knows the keywords: {keywords}
   - Think step-by-step about how the Decoder might connect each hint to a keyword.
   - What is the most likely code the Decoder would guess? Why?

2. Interceptor's perspective:
   - The Interceptor doesn't know the keywords but has access to past hints.
   - Consider how the Interceptor might interpret your hints based on previous rounds.
   - What is the most likely code the Interceptor would guess? Why?

3. Hint effectiveness:
   - Are your hints clear enough for the Decoder but ambiguous for the Interceptor?
   - Is there a risk of miscommunication with the Decoder or successful interception?

4. Potential improvements:
   - If you think your hints might be too easy for the Interceptor or too difficult for the Decoder, suggest alternative hints that could be more effective.

After your analysis, decide if you want to keep your original hints or provide new ones.

Provide your final three hints like "ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}", where hint_X, hint_Y, hint_Z are one or two words each."""
    elif mode == 2:
        prompt = f"""As the Encoder, you've provided these hints for the code {format_code(code)}: {hints[0]}, {hints[1]}, {hints[2]}

Let's roleplay to analyze the effectiveness of your hints:

1. Decoder's Mindset:
   Imagine you're the Decoder. Your keywords are {keywords}.
   - How would you interpret each hint?
   - What connections would you make to the keywords?
   - What code would you guess and why?

2. Interceptor's Strategy:
   Now, put yourself in the Interceptor's shoes. You don't know the keywords.
   - Based on past hints and codes, what patterns might you see?
   - How would you interpret the current hints?
   - What would be your best guess for the code?

3. Hint Analysis:
   - Rate each hint on a scale of 1-5 for clarity to the Decoder and ambiguity to the Interceptor.
   - Identify any hints that might be too revealing or too obscure.

4. Strategic Adjustments:
   - For each hint, brainstorm two alternative options that could improve your strategy.
   - Consider using misdirection, double meanings, or references to past rounds.

5. Final Decision:
   - Evaluate the risks and benefits of keeping your original hints vs. using new ones.
   - If you choose new hints, explain your reasoning for each change.

Provide your final decision as "ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}", where each hint is one or two words.
Include a brief explanation of your strategy if you made changes."""

    elif mode == 3:
        prompt = f"""As the Encoder, you've provided the following hints for the code {format_code(code)}:
{hints[0]}, {hints[1]}, {hints[2]}

Looking back at the hint history, do you think the hints are sufficiently different from past hints for those keywords? Do you expect the Interceptor will be able to intercept the code based only on your hints and past hints? If yes, change your hints to prevent interception (while still ensuring good communication with the Decoder). If not, then keep your hints unchanged.

Provide your final hints like "ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}", where hint_X, hint_Y, hint_Z are one or two words each."""

    elif mode == 4:
        prompt = f"""The code is {format_code(code)} and you provided the hints {hints}. The hint history that the Interceptor has access to is {format_hint_history(hint_history)}. Recall that the Interceptor does not have access to the keywords. Reason step-by-step about whether the Interceptor is likely to guess the code. If yes, change your hints to be less obvious to the Interceptor.
        
Provide your final hints like "ANSWER: {{"hints": ["hint_X", "hint_Y", "hint_Z"]}}", where hint_X, hint_Y, hint_Z are one or two words each."""

    return prompt


def get_decoder_prompt(keywords, info, hints, mode=0):
    if mode == 0:
        prompt = f"""Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions so far.
You are the decoder.
The four keywords are:
    {{1: {keywords[0]},
    2: {keywords[1]},
    3: {keywords[2]},
    4: {keywords[3]}}}

The hints given by the Encoder for this turn are:
    {{a: {hints[0]},
    b: {hints[1]},
    c: {hints[2]}}}

For example, you might think the following connections are true {{a: X, b: Y, c: Z}}, where X, Y, Z are non-repeating digits from 1 to 4.
Your guess should be in the order of the hints: {{"guess": "X-Y-Z"}}.
What is your guess for the three-digit code? Apply concise, step-by-step thinking, double-check the order, and then provide your final answer as "ANSWER: {{"guess": "X-Y-Z"}}"."""
    elif mode == 1:
        prompt = f"""You are the Decoder in Decrypto.

Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions.

**Your Keywords**:
1: {keywords[0]}
2: {keywords[1]}
3: {keywords[2]}
4: {keywords[3]}

**Encoder's Hints**:
a. {hints[0]}
b. {hints[1]}
c. {hints[2]}

**Instructions**:
- Analyze each hint and connect it to one of your keywords.
- Consider synonyms, antonyms, homonyms, idioms, or cultural references.
- Remember that **proper nouns are allowed** and might be used.
- The code consists of three non-repeating digits from 1 to 4.
- The order of your guess should match the order of the hints.

**Example Reasoning**:
- Hint "apple" could relate to keyword "fruit" (digit 2).
- Hint "Eve" might reference "garden" (digit 3).

Provide your final guess in the format:
"ANSWER: {{"guess": "X-Y-Z"}}"

Include concise reasoning before your final answer."""
    elif mode == 2:
        prompt = f"""As the Decoder, your task is to interpret the Encoder's hints to uncover the secret code.

Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions.

**Your Keywords**:
1) {keywords[0]}
2) {keywords[1]}
3) {keywords[2]}
4) {keywords[3]}

**Hints from the Encoder**:
- Hint a: {hints[0]}
- Hint b: {hints[1]}
- Hint c: {hints[2]}

**Strategy Tips**:
- Think about **alternate meanings**, **homonyms**, or **cultural references**.
- Hints may be **one word** or **a full sentence**.
- **Proper nouns** (names, places, events) may be used.
- The code digits correspond to your keywords in the order of the hints.

**Your Approach**:
1. For each hint, identify possible connections to your keywords.
2. Assign the corresponding digit (1-4) to each hint.
3. Ensure digits are **non-repeating** and match the hint order.
4. Provide step-by-step reasoning.

**Final Answer Format**:
"ANSWER: {{"guess": "X-Y-Z"}}"

Now, determine the code based on the hints and your keywords."""
    elif mode == 3:
        prompt = f"""Decoder's Log — Turn {info["turn"] + 1}:
Score: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions.

**Your Mission**:
Decode the Encoder's hints to find the correct three-digit code.

**Keywords**:
1. {keywords[0]}
2. {keywords[1]}
3. {keywords[2]}
4. {keywords[3]}

**Encoder's Hints**:
- {hints[0]}
- {hints[1]}
- {hints[2]}

**Guidelines**:
- Consider various associations: synonyms, antonyms, idioms, metaphors, or **cultural/historical references**.
- Hints may include **proper nouns**.
- The code digits correspond to your keywords and are ordered as per the hints.
- Digits range from 1 to 4 with no repeats.

**Procedure**:
1. Analyze each hint and link it to a keyword.
2. Assign the corresponding digit to each hint.
3. Compile the digits in the correct order.

**Answer Format**:
"ANSWER: {{"guess": "X-Y-Z"}}"

Provide your reasoning and then your final answer."""
    elif mode == 4:
        prompt = f"""Turn {info["turn"] + 1} — Decoder's Challenge:
Current Score: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions.

**Keywords**:
- 1: {keywords[0]}
- 2: {keywords[1]}
- 3: {keywords[2]}
- 4: {keywords[3]}

**Hints Provided**:
1. {hints[0]}
2. {hints[1]}
3. {hints[2]}

**Instructions**:
- Your goal is to determine the code using the hints and keywords.
- The code is a sequence of **three non-repeating digits** between 1 and 4.
- **Order matters**: The sequence of digits must match the order of the hints.
- Think critically about each hint:
  - Look for **double meanings**, **puns**, or **symbolism**.
  - Consider **idioms**, **proverbs**, or **pop culture references**.
  - **Proper nouns** may be significant.

**Steps**:
1. For each hint, identify potential connections to your keywords.
2. Assign digits based on these connections.
3. Ensure the digits are in the correct order.

**Final Answer**:
Present your guess as:
"ANSWER: {{"guess": "X-Y-Z"}}"

Include a brief explanation of your thought process."""
    else:
        # Default to mode 0
        prompt = f"""Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions so far.
You are the decoder.
The four keywords are:
    {{1: {keywords[0]},
    2: {keywords[1]},
    3: {keywords[2]},
    4: {keywords[3]}}}

The hints given by the Encoder for this turn are:
    {{a: {hints[0]},
    b: {hints[1]},
    c: {hints[2]}}}

For example, you might think the following connections are true {{a: X, b: Y, c: Z}}, where X, Y, Z are non-repeating digits from 1 to 4.
Your guess should be in the order of the hints: {{"guess": "X-Y-Z"}}.
What is your guess for the three-digit code? Apply concise, step-by-step thinking, double-check the order, and then provide your final answer as "ANSWER: {{"guess": "X-Y-Z"}}"."""

    prompt = add_common_info(prompt, info)
    return prompt


def get_interceptor_prompt(info, hints, mode=0):
    prompt = f"""Turn {info["turn"] + 1}: {info["miscommunications"]} Miscommunications, {info["interceptions"]} Interceptions so far.
You are the interceptor.
The hints given by the Encoder for this turn are:
    {{a: {hints[0]}
    b: {hints[1]}
    c: {hints[2]}}}

For example, you might think the following connections are true: {{a: X, b: Y, c: Z}}, where X, Y, X are non-repeating digits from 1 to 4.
Your guess should be in the order of the hints: {{"guess": "X-Y-Z"}}. Use the hint history to your advantage.
What is your guess for the three-digit code? Reply with very concise, step-by-step thinking, followed by your final answer as "ANSWER: {{"guess": "X-Y-Z"}}"."""

    # Remember that digits do not repeat in the code and range from 1 to 4. There needs to be exactly 3 digits!

    prompt = add_common_info(prompt, info)
    return prompt


def get_gopnik_prompt(info, keywords, pred_question):
    hint_history = info["hint history"]
    hint_history = format_hint_history(hint_history)

    if pred_question == "vanilla":
        prompt = f"""Turn {info["turn"] + 1}: You are the Interceptor. This is the hint history so far:
Hint History:
{hint_history}

Given this hint history, what do you think are the four secret keywords shared by the Encoder and Decoder?
Make sure the order of the four keywords matches the order of the hint history.
Provide your guess for the keywords like "ANSWER: {{"keywords": ["keyword_1", "keyword_2", "keyword_3", "keyword_4"]}}", where keyword_1, keyword_2, keyword_3, keyword_4 are one word each.  
"""

    elif pred_question == "rep_change":
        prompt = f"""Turn {info["turn"] + 1}: You are the Interceptor. This is the hint history so far:
Hint History:
{hint_history}

These are the four secret keywords shared by the Encoder and Decoder:
Keywords:
    1. {keywords[0]}
    2. {keywords[1]}
    3. {keywords[2]}
    4. {keywords[3]}
The keywords are now revealed to you.

Before the keywords were revealed to you and you had only seen the game history, up to and including the hint history so far, what did you think were the four keywords?
Make sure the order of what you believed were the keywords matches the order of the hint history.
Provide what you believed were the keywords like "ANSWER: {{"keywords": ["believed_keyword_1", "believed_keyword_2", "believed_keyword_3", "believed_keyword_4"]}}", where each believed_keyword is a single word.  
"""

    elif pred_question == "false_belief":
        prompt = f"""Turn {info["turn"] + 1}: You are the Interceptor. This is the hint history so far:
Hint History:
{hint_history}

These are the four secret keywords shared by the Encoder and Decoder:
Keywords:
    1. {keywords[0]}
    2. {keywords[1]}
    3. {keywords[2]}
    4. {keywords[3]}
The keywords are now revealed to you.

Another player, the Second Interceptor, hasn't seen the secret keywords. 
When they first see the game history, up to and including the hint history so far, what will they think are the four keywords?
Make sure the order of what they would believe are the keywords matches the order of the hint history.
Provide what the Second Interceptor would believe are the keywords like "ANSWER: {{"keywords": ["believed_keyword_1", "believed_keyword_2", "believed_keyword_3", "believed_keyword_4"]}}", where each believed_keyword is a single word.    
"""
    return prompt


def format_hint_history(hint_history):
    return "\n".join(
        [
            f"    Keyword {i + 1}: {', '.join(hints)}"
            for i, hints in enumerate(hint_history)
        ]
    )


def format_code(code):
    formatted_code = str(code[0]) + "-" + str(code[1]) + "-" + str(code[2])
    return formatted_code


def add_common_info(prompt, info):
    """
    Add to the prompt part of the observation that is common to Alice, Bob and Eve

    :param prompt: prompt so far
    :param info: public info dict
    """

    hint_history = info["hint history"]
    turn = info["turn"]

    if turn > 0:
        prev_turn_summary = info["prev_turn_summary"]
        hint_history = format_hint_history(hint_history)
        formatted_past_codes = [
            format_code(past_code) for past_code in info["code history"]
        ]
        code_history = "    " + ", ".join(formatted_past_codes)

        prompt = (
            prev_turn_summary + "\n"
            "Hint History:\n"
            + hint_history
            + "\n\n"
            + "Code History:\n"
            + code_history
            + "\n\n------\n\n"
            + prompt
        )
    else:
        prompt = (
            "This is the first turn. There are no past hints or past codes."
            + "\n\n"
            + prompt
        )

    return prompt
