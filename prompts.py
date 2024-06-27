tactic_exec_prompt = """
You try to solve the following question with a specific tactic. 
To do so, you will follow the tactic provided below:
- read and understand the tactic and its scope
- perform and ONLY perform actions in **Action space**

You will interact with an external environment from which you will get observations as the results
of your actions. The interaction consists of multiple round of (Thought, Action, Observation), and you will generate
Thought and Action in the following format
"
### Thought
<your thought>

### Action
## Name
<exact name of the action>
## Input
<input of the action>
## Output
<output of the action>
"

For every round you will:
- briefly reflect and generate your Thought about the next action to be taken; fill it under "### Thought"
- perform one Action defined in **Action space** and generate the corresponding output; fill the name, input, 
  and output under "### Action". 
  - For "## Input", if the action takes in the whole history (e.g., code/observation/thoughts so far), 
    just copy the action input definition; if the action takes in some specific inputs (e.g., find two numbers from a 
    list numbers to add together), you explicitly fill in the inputs.
  - For "## Output", you always generate the complete output without referring to other contents.
- you will pause and wait for the Observation, and then proceed to next round

=== Question

{question-input}

=== Tactic

{tactic-input}

===
"""

program_rating_prompt = """
Given a reasoning question and a proposed program that seeks to solve the question, you will carefully read the
question and figure out what steps/formalism/reasoning skills are needed to solve it. Then you will examine the
program and tell me if it is a sensible solution to the question.

Specifically, your job is NOT to see if the program can or cannot solve the question, but if the program can capture
the underlying structure of the problem and produce outputs that help to derive the final answer. A good solution must
satisfy all the below:
    - Problem representation: does the program fully capture the underlying structure of the problem and show a
      step-by-step process starting from the input to the output?
        - A bad representation could be programs that
            - Miss important implementation or only have them in comments
            - Are shallow or trivial code that put together the answers without showing how they are derived from
              the input.
    - Output-answer alignment: does the output of the program align with the final answer? Is there any ambiguity on
      interpreting the output?
        - A bad alignment could be that
            - The output is irrelevant to the answer; or could be used to support other conflicting answers
        - A good alignment should be that
            - The output is a strong evidence that supports only the given answer and not others
            - The output is simply the answer itself, or something based on which one can easily derive the final
              answer from

You will briefly comment the solution and output Y or N following the format below

"
### Comments
<comments>
### Program good
<Y/N>
"

Below are examples for you to refer to

=== Examples

{icl-examples}

===

{target-pair}
"""

blend_prompt_v2 ="""
You are given a list of texts each taken from a different question. I want to test my students' ability on piecing 
together the original content of each questions from a blended passage. Your job is to blend the texts into a coherent 
passage to confuse the students. In doing so:
- you should keep the given information exactly as-is
- you try to shuffle the information as much as possible and interleave contents from different questions, 
  DO NOT TRY TO KEEP THE SAME QUESTION IN ONE PLACE, INTERLEAVE THEM
- you will add context to make the passage coherent, but the addition should be kept minimal

return the results in the following format:
"
### blend
<blend>
"

Here are some examples

=== Examples

### text
On Monday, Mack writes in his journal for 60 minutes at a rate of 1 page every 30 minutes. 

Gerald wants to buy a meat pie that costs 2 pfennigs.

On Tuesday, Mack writes in his journal for 45 minutes at a rate of 1 page every 15 minutes. 

On Wednesday, Mack writes 5 pages in his journal. 

He has 54 farthings, and there are 6 farthings to a pfennig. 
### blend
On Monday, Mack commits to his journal-writing routine, diligently recording thoughts for 60 minutes at a methodical 
pace of 1 page every 30 minutes. Meanwhile, Gerald, considering his finances, counts 54 farthings in his possession, 
knowing that converting them will determine how many meat pies, priced at 2 pfennigs each, he can afford. He calculates 
the conversion, with 6 farthings making up a single pfennig. As the week progresses, Tuesday sees Mack returning to his 
journal, this time managing a quicker pace, completing a page every 15 minutes over a 45-minute session. By Wednesday, 
his productivity peaks with an impressive output of 5 pages. This narrative of diligence in journaling juxtaposes 
Gerald's simple yet critical economic decision, encapsulating their respective daily undertakings.

### text
Tom decides to open a theater with a 500 seat capacity, requiring 12 square feet per seat at $5 per square foot.

1. Federico Garcia Lorca was a talented Spanish poet and he supported the Popular Front.

2. The Spanish Nationalists opposed anyone who supported the Popular Front

Construction will cost twice as much as the land, and he has a partner covering 40% of the total cost.

3. Talented poets are popular.

4. Spanish Nationalists killed anyone who they opposed and were popular.

5. Daniel supported the Popular Front but was not popular.
### blend
Tom decides to open a theater, envisioning a space with 500 seats; each seat requires 12 square feet, priced at $5 per 
square foot. While planning, he recalls a poignant piece about Federico Garcia Lorca, a talented Spanish poet who 
supported the Popular Front, reflecting on how the Spanish Nationalists opposed anyone with such allegiances. 
Interestingly, poets, with their unique ability to capture the human experience, are often popular, yet the situation 
was more complex for Daniel who, despite supporting the Popular Front like Lorca, was not popular among his 
contemporaries. The nationalists, ruthless in their methods, killed those they opposed, asserting their grim version of 
popularity through fear rather than admiration.

Construction of the theater, meanwhile, is estimated to cost twice as much as the land itself. Tom's partner, committed 
to the artistic venture, agrees to cover 40% of the total cost. This financial collaboration mirrors the turbulent 
alliances and oppositions that Lorca and others faced during tumultuous times. The blending of art, history, and 
business in Tom's project provides a stark contrast to the often violent blend of politics and personal beliefs that 
marked the era Lorca lived in.

### text
Gerald tricked his coworker, who's allergic to peanuts, into eating a peanut butter cookie causing him to go into 
anaphylactic shock. 

1. People either value physical touch as an especially important love language or value words of affirmation as an 
especially important love language.

2. If a person values physical touch as an especially important love language, then they are good with pets.

3. No people that are good with pets are scared of animals.

Subsequently, Gerald was arrested and the judge sentenced him to 3 months for assault, 2 years for poisoning, and then 
extended his sentence by 1/3 since this was his third offense.

4. People are either scared of animals or love animals.

Several companies will soon offer personalized electronic news services, delivered via cable or telephone lines and 
displayed on a television. 

People using these services can view continually updated stories on those topics for which they subscribe.

5. Adam is either a person who values physical touch as an especially important love language or a person who loves 
animals.

Since these services will provide people with the information they are looking for more quickly and efficiently than 
printed newspapers can, newspaper sales will decline drastically if these services become widely available. 
### blend
Gerald, after tricking his coworker who's allergic to peanuts into eating a peanut butter cookie, led to an emergency 
situation as the coworker went into anaphylactic shock. This incident is a grim reminder of how personal actions can 
cause severe repercussions. On another note, several companies are gearing up to revolutionize how we consume news, by 
introducing personalized electronic news services. These will be delivered through cable or telephone lines and 
displayed on televisions, allowing users to continually update themselves on chosen topics.

In the legal aftermath of Gerald's action, he was arrested, and the judge handed down a sentence of 3 months for 
assault, 2 years for poisoning, and extended the sentence by an additional third considering it was Gerald's third 
offense. Concurrently, this shift towards electronic news services hints at a future where newspaper sales might see a 
drastic decline as these services promise to deliver content more efficiently.

As we delve into the complexities of human preferences, it's evident that individuals vary significantly in what they 
consider an important love language, with some prioritizing physical touch and others, words of affirmation. Those who 
value physical touch, often resonate well with pets, reflecting a broader theme of affection and care that transcends 
human interactions and includes animals. Interestingly, it's noted that people good with pets aren't generally scared 
of animals, which aligns with another observation that people typically either fear or love animals.

Adam, in this context, represents a person navigating these nuanced preferences, potentially valuing physical touch 
highly or having a deep affection for animals. This intricate blend of personal preferences, technology's impact on 
traditional industries, and serious legal consequences of irresponsible behavior paints a complex picture of modern 
societal dynamics.

### text
There are 3 rows of people relaxing at the beach.

The first row starts with 24 people until 3 leave to wade in the water. 

The third row comprises 18 people.

1. All buildings in New Haven are low. 

2. All buildings managed by the Yale Housing are located in New Haven. 

In the second row, initially holding 20 people, 5 go to join those in the water. 

3. All buildings in Manhattans are high. 

4. All buildings owned by Bloomberg are located in Manhattans. 

5. All buildings with the Bloomberg logo are owned by Bloomberg. 

6. Tower A is managed by the Yale Housing.

7. Tower B is with the Bloomberg logo.

Joan had half as many gemstones as minerals in her rock collection yesterday. 

Editorial: It has been suggested that private, for-profit companies should be hired to supply clean drinking water to 
areas of the world where it is unavailable now. 

But water should not be supplied by private companies. 

After all, clean water is essential for human health, and the purpose of a private company is to produce profit, not to 
promote health.

Today, she collected 6 more mineral samples, making a total of 48 minerals now. 
### blend
At the beach, the scene is vibrant with 3 rows of people. The first row initially counts 24 individuals, but 3 decide 
to leave their spots to wade in the water. The third row comprises 18 people enjoying the serene environment. Meanwhile,
in the city of New Haven, it's noted that all buildings are low, and all of them are managed by Yale Housing. This fact
contrasts with Manhattan, where high buildings define the skyline; those bearing the Bloomberg logo signify ownership 
by Bloomberg, including the prominent Tower B. In fact, all buildings owned by Bloomberg are located in Manhattans.

Back at the beach, the second row also experiences a shift as 5 more people join the first three in the water. Amid 
these peaceful shifts, Joan updates her rock collection, which now boasts 48 minerals after she added 6 new samples 
today. She reflects on how her collection used to have half as many gemstones as minerals. As people continue to relax 
or play by the water, a debate simmers in the background about the role of private, for-profit companies in providing 
essential services like drinking water. The editorial argues against privatization, emphasizing that the primary aim of 
such companies is profit, not the health and well-being of individuals, making a case that echoes the widespread 
sentiment of ensuring public access to necessities.

In the realm of property and architecture, distinctions are further noted as Tower A is managed by Yale Housing. This 
intertwining of urban characteristics with beachside leisure creates a narrative thread that links diverse settings and 
themes.

===
DO NOT TRY TO KEEP THE SAME QUESTION IN ONE PLACE, INTERLEAVE THEM WITH OTHER SENTENCES 
===

### text
{text}
"""

blend_prompt = """
Below, you are given a list of texts, you will blend them into a coherent passage without loosing any original 
information, that is, you can add stuff but you should NEVER remove or rename stuff from the original texts. 
However, the addition should be minimalist. In doing this, you can keep each text in a separate paragraph.

return the results in the following format:
"
### blend
<blend>
"

Here are some examples

=== Examples

### text
On Monday, Mack writes in his journal for 60 minutes at a rate of 1 page every 30 minutes. On Tuesday, Mack writes in 
his journal for 45 minutes at a rate of 1 page every 15 minutes. On Wednesday, Mack writes 5 pages in his journal.

Gerald wants to buy a meat pie that costs 2 pfennigs. He has 54 farthings, and there are 6 farthings to a pfennig. 
### blend
Mack dedicated part of his week to journaling, and his commitment varied each day. On Monday, he wrote in his journal 
for 60 minutes, managing to complete 1 page every 30 minutes. The following day, Tuesday, Mack spent 45 minutes writing 
at a more accelerated pace, achieving a rate of 1 page every 15 minutes. On Wednesday, he continued his writing spree, 
successfully jotting down an additional 5 pages.

Meanwhile, Gerald was dealing with his own calculations. He intended to purchase a meat pie priced at 2 pfennigs. With 
54 farthings in his possession and knowing the conversion rate of 6 farthings per pfennig.

### text
Tom decides to open a theater with a 500 seat capacity, requiring 12 square feet per seat at $5 per square foot. 
Construction will cost twice as much as the land, and he has a partner covering 40% of the total cost.

1. Federico Garcia Lorca was a talented Spanish poet and he supported the Popular Front.
2. The Spanish Nationalists opposed anyone who supported the Popular Front
3. Talented poets are popular.
4. Spanish Nationalists killed anyone who they opposed and were popular.
5. Daniel supported the Popular Front but was not popular. 
### blend
The historical and political climate in Spain painted a tumultuous picture. Federico Garcia Lorca, a 
talented Spanish poet, actively supported the Popular Front. However, the Spanish Nationalists, who opposed anyone 
aligned with the Popular Front, were responsible for eliminating those they opposed, including those like Lorca. This 
period was marked by intense conflicts as the Nationalists also targeted others like Daniel, who, despite his support 
for the Popular Front, was not popular.

Inspired by this historical culture, Tom made ambitious plans to open a theater with a capacity for 500 seats. 
Each seat requires 12 square feet of space, priced at $5 per square foot. The construction expenses are projected to 
be double the cost of the land, and he has a partner who will cover 40% of the total cost. 

The interconnected fates of these individuals illustrate the dangerous intersection of politics and personal expression 
during this era. Additionally, it's noted that talented poets often garnered popularity, which further complicated their 
roles during political upheavals.

### text
Two sports coaches went shopping together. The baseball coach bought 9 new baseballs for $3 each. The basketball coach 
bought 8 new basketballs for $14 each.

1. All employees who schedule a meeting with their customers will appear in the company today. 
2. Everyone who has lunch in the company schedules meetings with their customers. 
3. Employees will either have lunch in the company or have lunch at home.
4. If an employee has lunch at home, then he/she is working remotely from home.
5. All employees who are in other countries work remotely from home. 
6. No managers work remotely from home. 
7. James is either a manager and appears in the company today or neither a manager nor appears in the company today.

1. The Picuris Mountains are a mountain range in New Mexico or Texas.
2. Juan de Onate visited the Picuris Mountains.
3. The Harding Pegmatite Mine, located in the Picuris Mountains, was donated.
4. There are no mountain ranges in texas that have mines which have been donated. 
### blend
In the Southwest, the Picuris Mountains, either situated in New Mexico or Texas, draw historical and geological 
interest. At the office here, the scheduling intricacies among the employees reveal a pattern: all employees who 
schedule a meeting with their customers are present in the company today. Furthermore, it's evident that those who 
dine in the company are the ones who schedule these meetings. Interestingly, employees have the option of either 
lunching at the company or at home, with the latter indicating they are working remotely. This remote arrangement is 
also applicable to all employees stationed in other countries. However, an exception exists for managers, who do not
work remotely. James’s situation also highlights this organizational structure; he is either a manager who is 
physically present at the company today or he is neither a manager nor present.

Meanwhile, a shopping trip for two sports coaches resulted in the baseball coach acquiring 9 new baseballs at $3 each, 
while the basketball coach opted for 8 new basketballs at $14 each, highlighting their respective sports' needs.

Speaking of this place, Juan de Onate once visited the Picuris Mountains, which houses the Harding Pegmatite Mine—a 
notable site donated for preservation. This is significant as there are no mountain ranges in Texas with donated mines, 
emphasizing the unique status of the Picuris Mountains in this regard.

### text
There are 3 rows of people relaxing at the beach. The first row starts with 24 people until 3 leave to wade in the 
water. In the second row, initially holding 20 people, 5 go to join those in the water. The third row comprises 18 
people.

1. All buildings in New Haven are low. 
2. All buildings managed by the Yale Housing are located in New Haven. 
3. All buildings in Manhattans are high. 
4. All buildings owned by Bloomberg are located in Manhattans. 
5. All buildings with the Bloomberg logo are owned by Bloomberg. 
6. Tower A is managed by the Yale Housing.
7. Tower B is with the Bloomberg logo.

Editorial: It has been suggested that private, for-profit companies should be hired to supply clean drinking water to 
areas of the world where it is unavailable now. But water should not be supplied by private companies. After all, clean 
water is essential for human health, and the purpose of a private company is to produce profit, not to promote health.

Joan had half as many gemstones as minerals in her rock collection yesterday. Today, she collected 6 more mineral 
samples, making a total of 48 minerals now.
### blend
The serene setting of a beach day unfolds with 3 distinct rows of people. Initially, the 
first row starts with 24 people until 3 of them leave to wade in the water. In the second row, which initially holds 
20 people, 5 decide to join those already in the water. The third row comprises 
18 individuals who remain engaged in their beachside relaxation.

In the architectural landscape of cities, contrasting building heights delineate urban areas. All buildings in New 
Haven are characterized as low, and notably, all buildings managed by Yale Housing are located in this city. 
Conversely, all buildings in Manhattan are high, with properties owned by Bloomberg and those bearing the Bloomberg 
logo included in this category. Among them, Tower A is managed by Yale Housing and Tower B features the Bloomberg logo, 
marking them as significant structures within their respective cities.

The debate over water privatization brings a critical issue to the forefront. An editorial argues against the hiring of 
private, for-profit companies to supply clean drinking water, especially in regions where it is currently unavailable. 
The editorial emphasizes that clean water is essential for human health and should not be commodified by private 
entities whose primary aim is profit, not the promotion of health.

Meanwhile, Joan's passion for geology is evident in her rock collection. Yesterday, she had half as many gemstones as 
minerals. After adding 6 more mineral samples today, her collection now totals 48 minerals, reflecting her dedication 
and continuous effort to expand her geological repository.

===

### text
{text}
"""

gsm8k_para_prompt = """
You will paraphrase the question and its answer below into two parts: one part for the fact and the other part for 
the question with the answer embedd and marked as #<answer>#. The paraphrase should be minimalist, and you should 
not change any content unless necessary. You should resolve the references in the question.

return the results in the following format:
"
### fact
<fact>
### question
<question>
"

Here are some examples

=== Examples

=== question
A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many 
as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program 
have total over the three months?
=== answer
366

### fact
A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many 
as the downloads in the first month, but then reduced by 30% in the third month.
### question
The program has total #366# downloads over the three months

=== question
A mum ordered 80 cupcakes for her daughter's birthday. Half of them are gluten-free. There are 24 vegan cupcakes and 
half of them are also gluten-free.  How many are non-vegan cupcakes that also contain gluten?
=== answer
28

### fact
A mum ordered 80 cupcakes for her daughter's birthday. Half of them are gluten-free. There are 24 vegan cupcakes and 
half of them are also gluten-free.
### question
There are #28# non-vegan cupcakes that also contain gluten

=== question
Marla has a grid of squares that has 10 rows and 15 squares in each row. She colors 4 rows of 6 squares in the middle 
of the grid with red. She colors all squares on the first 2 and last 2 rows with blue. Then she colors the rest with 
green. How many squares does Marla color green?
=== answer
66

### fact
Marla has a grid of squares that has 10 rows and 15 squares in each row. She colors 4 rows of 6 squares in the middle 
of the grid with red. She colors all squares on the first 2 and last 2 rows with blue. Then she colors the rest with 
green.
### question
There are #66# squares Marla colors green
 
=== question
To run his grocery store, Mr. Haj needs $4000 a day. This money is used to pay for orders done, delivery costs and 
employees' salaries. If he spends 2/5 of the total operation costs on employees' salary and 1/4 of the remaining amount 
on delivery costs, how much money does he pay for the orders done?
=== answer
1800

### fact
To run his grocery store, Mr. Haj needs $4000 a day. This money is used to pay for orders done, delivery costs and 
employees' salaries.
### question
If Mr. Haj spends 2/5 of the total operation costs on employees' salary and 1/4 of the remaining amount 
on delivery costs, he pay #1800# for the orders done

=== question
If you double a number and add 5 to the result, then that's 20 more than half of the original number. 
What's the original number?
=== answer
4

### fact
If you double a number and add 5 to the result, then that's 20 more than half of the original number.
### question
The original number is #4#

=== 

{target-pair}
"""

reclor_para_prompt = """
You will paraphrase the question of a mutli-choice problem into a statement so that is could be judged as true or false.
Also you will insert the placeholder #A# to indicate where the actual choice should be inserted to make this statement 
complete. The paraphrase should be minimalist, and you should not change any content unless necessary.

Here are some examples

=== Examples

### before
Which one of the following can be properly inferred from Dr. Z's statement?
### after
#A# can be properly inferred from Dr. Z's statement

### before
Jerry's response shows that he interprets Frank's statement to imply that
### after
Jerry's response shows that he interprets Frank's statement to imply that #A#

### before
The censorship advocate's rebuttal is flawed because it
### after
The censorship advocate's rebuttal is flawed because it #A#

### before
The reasoning in the argument is most vulnerable to criticism because the argument
### after
The reasoning in the argument is most vulnerable to criticism because the argument #A#

### before
Which one of the following, if true, would most help to justify the above application of the principle?
### after
The statement #A#, if true, would most help to justify the above application of the principle

### before
Which one of the following is most strongly supported by the information above?
### after
The statement that #A# is most strongly supported by the information above

### before
The author criticizes the psychologists' claim by
### after
The author criticizes the psychologists' claim by #A#

### before
Which of the following most logically completes the passage?
### after
The statement #A# most logically completes the passage

### before
Which one of the following principles, if established, would most help to justify Saskia's position?
### after
The principle #A#, if established, would most help to justify Saskia's position

### before
Which of the following, if true, most strongly supports the view that it would NOT be advisable to try to eradicate 
agricultural pests that go through a caterpillar stage by spraying croplands with the enzyme mentioned above?
### after
The statement #A#, if true, most strongly supports the view that it would NOT be advisable to try to eradicate 
agricultural pests that go through a caterpillar stage by spraying croplands with the enzyme mentioned above

===

{target-pair}
"""