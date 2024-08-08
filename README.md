# family member dataset
As the basic unit of society, the family reflects the microscopic structure of division of labor and cooperation. In addition, cooperation among family members also reflects gender differences. The proportion of cooperation between women and women or men and men is higher than that between men and women. In contrast, the conjugal relationship is the most special and frequent form of cooperation in the family. Therefore, the study of family cooperation contributes to understanding the evolution of social division of labor and gender roles.
Analysis based on diachronic text corpus. Through extensive collection of literature resources and narrative analysis, this study aims to build a family cooperation database and deepen our understanding of the mode of cooperation within the family.
# task classification
The research method is divided into two main parts: cooperation and family members. For cooperation, we first verify the feasibility of explicit, implicit, and non-cooperative, then obtain the data through syntactic dependency analysis and large language models, train the model to classify cooperation and non-cooperation, and finally analyze the key motivations in the data. For family members, the scope of the study was defined, data was obtained using syntactic dependency and coreference parsing tools, large language models were fine-tuned and information about family members was extracted, cooperative characteristics between different family members were analyzed, and finally, family tree maps within three generations were created by combining family information.
![image](https://github.com/user-attachments/assets/043dd868-0062-4275-be8f-03110aa5bfff)
# verify the feasibility
The intra-group similarity is higher than inter-group similarity, which proves that the classification standard has some reliability.
![image](https://github.com/user-attachments/assets/049341a4-fb47-4e11-9ba7-51b87daed648)
# tuning and infering llm
The research method is divided into three steps: extraction, fine-tuning and inference. First, traditional tools were used to analyze sentence structure, initially collect data of family members, and reduce the processing load of subsequent tasks. Next, the language model is fine-tuned using the data set containing the task instructions to improve its ability to follow the instructions. Finally, in view of the unknown information in the preliminary data, the fine-tuned model is used to supplement the data and combine the time information for reasoning, so as to obtain more complete and accurate family member relationship and related information.
![image](https://github.com/user-attachments/assets/f0c76a81-df84-4a80-a06c-b05a33c71350)
# timing analysis
When analyzing the temporal distribution and relationship type changes of family members' cooperation patterns from the 18th century to the 21st century, the data show that the relevant laws remain basically stable. Especially in frequent family relationships such as husband and wife, father and son, and brothers, the frequency of cooperative patterns did not change significantly over time, but remained at a high level.
![image](https://github.com/user-attachments/assets/a4a4e694-b684-46f5-8baa-97458e2522f1)
# requency chart of cooperation among family members
The graph shows the rate of cooperation and friendship between different family members. The horizontal axis represents the cooperation ratio (log10) and the vertical axis represents the friendship ratio (log10). Different colors and symbols represent different family relationships, such as father and son, mother and daughter, etc. By comparing these rates, we can see the frequency of cooperation and friendship between different family members, such as husbands and wives, who have the highest rate of cooperation, while other relationships such as grandparents and grandchildren have the lowest rate.
![image](https://github.com/user-attachments/assets/b2bf478c-c030-432f-96c0-137faadd873f)
# family tree maps
With the Curie family as the core, the family tree is organized and optimized using chord diagrams to show the frequency of cooperation among family members. The inner circle is arranged by seniority, and the outer circle shows specific information about people, with blue representing men, red representing women, and gray representing newly organized family members. The figure shows the rule that the immediate relatives are the most, the secondary relatives are the second, and the collateral relatives are less.
![image](https://github.com/user-attachments/assets/d837cf89-2fe8-41cc-8d35-a8c8684f0820)



