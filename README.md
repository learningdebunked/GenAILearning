# GenAILearning

My baby steps into GenAI


![image](https://github.com/user-attachments/assets/f518e6c2-e1b2-43ab-a17a-58fd76953515)


what is Gen AI ?

Type of Artificial intelligence that uses neural networks and deep learning algorithms to identify patterns within existing data as a basis for generating original content ?

Traditional AI, often referred to as machine learning (ML), has primarily focused on analytic tasks like classification

and prediction

What does Gen AI do ?

Traditional AI uses predictive models to classify data, recognize patterns, and predict outcomes within a specific context or domain, such as analyzing medical images to detect irregularities. 

Gen AI models generate entirely new outputs rather than simply making predictions based on prior experience

LLMs use Deep learning 

Central processing units (CPUs) are designed for general-purpose computing tasks, GPUs, initially developed for graphics rendering, are specialized processors that have proven to be adept at ML tasks due to their unique architecture

What is LLM?
An LLM is a general-purpose model primarily useful for tasks related to unstructured text data
LLMs are sophisticated predictive models that anticipate the next word in a sequence based on the context provided to them as part of a process referred to as a completion. Carefully constructed prompts help these models deliver tailored content, yielding better completions. LLM performance is influenced not only by
the training data but also by the context provided by these user inputs

Types of LLMs:
1) General purpose LLM
     a) Task specific LLM like Meta's Code Llama
     b)  Domain specific LLM like NVIDIA's bioBERT

GPT-3, a general-purpose LLM, was developed by OpenAI based on the Generative Pre-trained Transformer (GPT) series of machine learning (ML) models. ChatGPT isn’t a language model
per se, but rather a user interface tailored around a particular language model such as GPT-3, GPT-3.5, or GPT-4 

Domain-specific LLMs focus on a specific subject area or industry. For example, BioBERT is trained on biomedical text, making it an excellent resource for understanding scientific literature
and extracting information from medical documents. CodeBERT is a cybersecurity solution that has been trained to assist with IT security concerns such as vulnerability detection, code review,
and software security analysis. 

Technology behind LLM: CNN or Complex recurrent structures

Key terms and concepts in LLM:
a) prompt : the text you provide to the language model
b) completion: the result produced by the model
c) inference: process of using model to generate text

  Frameworks and Developer Tools
  Open AI GPT Playground
  Snowflake Cortex
  Hugging face Transformer library 
  //TODO what are others?

Governance: Models need data , instead of moving data to models , to ensure security move models to where data resides to ensure security

**LLM APP Project Lifecycle**
     
     **Define the use case and the scope**
          1) create personalized product descriptions
          2) summarize transcripts
          3) extract answers from documents
          4) create compelling characters for a video game 
          5) to train a computer vision system to recognize particlar objects

     **determine what proprietary data you will use to customize or contextualize the model effectively**
     **Selecting  the right LLM **
          **Hosted LLMs **
               a) BARD
               b) Chat GPT
          **Open source LLMs **
               a) LLama

          The parameters in a language model refer to the trainable variables. More parameters mean more knowledge is part of the model out of the box, but bigger isn’t always better. 
          Smaller LLMs have fewer parameters and thus consume less compute resources and are faster to fine-tune and deploy.

           LLMs  have a higher number of parameters (typically 10 billion or more) and can learn more nuanced language patterns, and provide more
           accurate and contextually relevant outputs for a wider range of scenarios.
           For example, with only 117 million parameters, GPT-2 is a good choice for a narrow set of tasks such as language completion and summarization. With 175 billion parameters, GPT-3 is            better for complex tasks, such as translating text and generating dialogue.


**How can you adapt LLMs to your use cases?**

1) Prompt Engineering
          a) Text prompts to LLMs to generate the responses
          b) Prompt engineering is the practice of crafting inputs to shape the output of a language model and achieve a desired result
          c) Prompt Engineering Techniques
                  1) Zero-shot -- Zero-shot prompting is the default; you simply issue a question and rely on the LLM’s pretrained information to answer it.
                  2) one - shot -- You include an example of the desired output to help the model understand the desired output
                  3) Few - shot -- multiple examples to more clearly teach LLMs the desired output structure and language

3)
4) In context learning AKA ICL -- allows the LLM to dynamically update its understanding during a conversation
             a)  in-context learning (ICL) involves training a language model with a data set that aligns with the desired context or domain.
             b) ICL allows users to give context to the model using private data, enhancing its performance in specific domains
6) Retrieval augmented generation ( RAG ) -- combines retrieval and generation models to surface new relevant data as part  of a prompt

          RAG leverages a pretrained language model in conjunction with a large-scale vector search system to enhance the content-generation process.
          RAG accesses up-to-date information by retrieving relevant data stored as vectors (numerical representation of the data for fast retrieval
   
8) Fine tuning
9) Re-inforcement learning from human feedback ( RLHF) -- Fine tune models in real time by providing feedback from human evaluators
          This is a form of fine tuning
          Creators of LLM systems use this to tune their chat bots to carry realistic conversations


Vectors and Vector DB: These mathematical representations enable efficient storage and searching of data, as well as identification of semantically related text.
//TODO What are the vector db in the market and what are the use cases that they solve?
         


history :

What is classification?
What is prediction ?
What is deep learning ?
What are neural networks ?
Are inspired by the structure and functioning of the human brain. These software systems use interconnected nodes (neurons) to process information
What is the USP of Gen AI which relies on deep learning and neural networks ? What's the gap in these ?
What is NLP ?
What are CNN ? {convolutional neural networks } ? – Computer vision tasks
What are RNN ? {Recurrent neural networks} ? – sequential data processing 
What are LLMs ? {Large language models }
Predictive model ??
What is a large-scale vector search system that is used in RAG ?
What are Multimodal language model ?
  Multimodal, meaning it handles both text and other media such as images.
  
Forbes article :  Transformers Revolutionized AI

Google's : Transformer architecture – a deep learning model that replaced traditional recurrent and convolutional structures with a new type of architecture that’s particularly effective at

understanding and contextualizing language, as well as generating text, images, audio, and computer code.

Example OpenAI's Chatgpt is on Transformer's architecture 

ChatGPT - full form : Chatbot Generative Pre-trained Transformer

CPU vs GPU ? - high level understanding ?

GPUs have a large number of cores that can process multiple

tasks simultaneously. Transformers use GPUs to process multiple

threads of information, leading to faster training of AI models

that effectively handle not just text but also images, audio, and

video content.

Cloud Data Science For Dummies (Wiley) by David Baum for additional information on CPU vs GPU
LLM's for Natural language understanding - focus on comprehending the intricate meaning in human communication
can be used for product reviews , social media posts and customer surveys
can i try one on products page or app store reviews and make some sense ?
Can i use LLM for search ?
What are the companies that are working on LLM's ?
What is Nvidia's  MegaMolBART (part of the NVIDIA BioNeMo service and framework ) ?
What custom AI App can i create ? Is it possible ?
Neural networks
CNN ?
Attention mechanism?

Encode? {the part that understands the input}

Decoder? {the part that generates the output}

Reinforcement learning with human feedback (RLHF)

Prompt Engineering

In context learning {ICL}

Vector Databases and their roles in ML

VSS - Vector similarity Search

Retrieval and Generational model 

AI/ML/GenAI general:
DeepLearning.AI https://www.deeplearning.ai/short-courses/ai-python-for-beginners/ (this site has lots AI related courses)
AWS skillbuilder https://explore.skillbuilder.aws/learn/learning-plans/2217/aws-artificial-intelligence-practitioner-learning-plan
Machine Learning University https://www.youtube.com/@machinelearninguniversity1942
Role based learning general + AWS https://aws.amazon.com/ai/learn/
Prompt eng guide https://www.promptingguide.ai/ 

Vectors {in the context of RAG / AI} & Vector db??


AWS specific:
Generative AI Foundations on AWS https://www.youtube.com/watch?v=oYm66fHqHUM&list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF
workshops
https://catalog.us-east-1.prod.workshops.aws/workshops/63069e26-921c-4ce1-9cc7-dd882ff62575/en-US 
https://catalog.us-east-1.prod.workshops.aws/workshops/972fd252-36e5-4eed-8608-743e84957f8e/en-US
https://catalog.workshops.aws/building-gen-ai-apps/en-US
all others https://workshops.aws/ 


