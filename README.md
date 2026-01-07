
# Awesome Code-Mixing & Code-Switching
<p align="center">
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://www.google.com/search?q=%23contributing">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
  </a>
</p>

> A curated list of papers, datasets, and toolkits for **Code-Switching** & **Code-Mixing** in Natural Language Processing.

## Table of Contents

*Click on any link to jump to the corresponding section on this page.*
* [Survey Papers](#survey-papers)
* [Taxonomy Diagram](#-taxonomy-diagram-of-research-landscape)
* [1. NLP Tasks](#1-nlp-tasks)
  * [1.1. Traditional Tasks](#11-traditional-tasks)
      * [Language Identification (LID)](#language-identification-lid)
      * [Part-of-Speech (POS) Tagging](#part-of-speech-pos-tagging)
      * [Named Entity Recognition (NER)](#named-entity-recognition-ner)
      * [Sentiment & Emotion Analysis](#sentiment--emotion-analysis)
      * [Syntactic Analysis](#syntactic-analysis)
      * [Machine Translation (MT)](#machine-translation-mt)
  * [1.2. Emerging and Contemporary Tasks](#12-emerging-and-contemporary-tasks)
      * [Natural Language Inference (NLI)](#natural-language-inference-nli)
      * [Intent Classification](#intent-classification)
      * [Question Answering (QA)](#question-answering-qa)  
      * [Code-Mixed Text Generation](#code-mixed-text-generation)
      * [Cross-lingual Transfer](#cross-lingual-transfer)
      * [Text Summarization](#text-summarization)
      * [Dialogue Generation](#dialogue-generation)
      * [Transliteration](#transliteration)
  * [1.3. Underexplored and Frontier Tasks](#13-underexplored-and-frontier-tasks)
      * [Conversation & Speech](#conversation-and-speech)
      * [Safety & Multimodal](#safety-and-multimodal)
      * [Reasoning & Abstraction](#reasoning-and-abstraction)
      * [Creative & Code Generation](#creative-and-code-generation)
* [2. Datasets & Resources](#2-datasets--resources)
   * [Datasets](#datasets)
   * [Frameworks & Toolkits](#frameworks--toolkits)
* [3. Model Training & Adaptation](#3-model-training--adaptation)
  * [Pre-training Approaches](#pre-training-approaches)
  * [Fine-tuning Approaches](#fine-tuning-approaches)
  * [Post-training Approaches](#post-training-approaches)
* [4. Evaluation & Benchmarking](#4-evaluation--benchmarking)
  * [Benchmarks](#benchmarks)
  * [Evaluation Metrics](#evaluation-metrics)
* [5. Multi- & Cross-Modal Applications](#5-multi--cross-modal-applications)
  * [Speech Processing](#speech-processing)
  * [Vision-Language & Document Processing](#vision-language--document-processing)
  * [Cross-Modal Integration](#cross-modal-integration)
* [Workshops & Shared Tasks](#workshops--shared-tasks)

-----

### Taxonomy of Code-Mixed Language Analytics and representative works for each direction
<img alt="Taxonomy of Code-Mixed Language Analytics" src="https://github.com/user-attachments/assets/e80ab921-bb6c-41d8-b84c-16675c78a00e" width="80%" />

## Survey Papers

> Comprehensive reviews of the code-switching research landscape. A great place to start.
  * **[A Survey of Current Datasets for Code-Switching Research](https://ieeexplore.ieee.org/document/9074205)** - *Jose, N., et al. (2020)*.
  * **[A Survey of Code-switched Speech and Language Processing](https://arxiv.org/pdf/1904.00784)** - *Sitaram, S., et al. (2020)*.
  * **[A Survey of Code-switching: Linguistic and Social Perspectives for Language Technologies](https://aclanthology.org/2021.acl-long.131/)** - *Doğruöz, A. S., et al. (2021)*.
  * **[The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges](https://aclanthology.org/2023.findings-acl.185/)** - *Winata, G. I., et al. (2023)*.
  * **[A Survey of Code-switched Arabic NLP: Progress, Challenges, and Future Directions](https://aclanthology.org/2025.coling-main.307/)** - *Hamed, I., et al. (2025)*.
  * **[Code-Switching in End-to-End ASR: A Systematic Literature Review](https://arxiv.org/abs/2507.07741)** - *Smitesh Patil, et al. (2025)*.

* **Position Paper** * **[Building Educational Technologies for Code-Switching: Current Practices, Difficulties and Future Directions](https://www.mdpi.com/2226-471X/7/3/220)** - *Li Nguyen, et al. (2022)*.
-----

## 1\. NLP Tasks

## 1.1\. Traditional Tasks

> Core competencies for understanding structure, syntax, and linguistic boundaries.

### Language Identification (LID)

  * **[Word Level Language Identification in English Telugu Code Mixed Data](https://aclanthology.org/Y18-1021/)** - *Gundapu, S. & Mamidi, R. (2018)*. 
  * **[A fast, compact, accurate model for language identification of codemixed text](https://arxiv.org/abs/1810.04142)** - *Zhang, Y., et al. (2018)*. 
  * **[Language identification framework in code-mixed social media text based on quantum LSTM](https://api.semanticscholar.org/CorpusID:214459891)** - *Shekhar, S., et al. (2020)*. 
  * **[A Pre-trained Transformer and CNN model with Joint Language ID and Part-of-Speech Tagging for Code-Mixed Social-Media Text](https://aclanthology.org/2021.ranlp-1.42.pdf)** - *Dowlagar, S., et al. (2021)*.
  * **[Much Gracias: Semi-supervised Code-switch Detection for Spanish-English: How far can we get?](https://aclanthology.org/2021.calcs-1.9/)** - *Iliescu, D.-M., et al. (2021)*. 
  * **[IRNLP\_DAIICT@LT-EDI-EACL2021: Hope Speech detection in Code Mixed text using TF-IDF Char N-grams and MuRIL](https://aclanthology.org/2021.ltedi-1.15/)** - *Dave, B., et al. (2021)*. 
  * **[Transformer-based Model for Word Level Language Identification in Code-mixed Kannada-English Texts](https://aclanthology.org/2022.icon-wlli.4/)** - *Lambebo Tonja, A., et al. (2022)*.
  * **[Transformer-based Model for Word Level Language Identification in Code-mixed Kannada-English Texts](https://aclanthology.org/2022.icon-wlli.4.pdf)** - *Tonja, A. L., et al. (2022)*.
  * **[TongueSwitcher: Fine-Grained Identification of German-English Code-Switching](https://aclanthology.org/2023.calcs-1.1/)** - *Sterner, I., & Teufel, S. (2023)*.
  * **[Representativeness as a Forgotten Lesson for Multilingual and Code-switched Data Collection and Preparation](https://aclanthology.org/2023.findings-emnlp.382.pdf)** - *Doğruöz, A. S., et al. (2023)*. 
  * **[Multilingual Large Language Models Are Not (Yet) Code-Switchers](https://aclanthology.org/2023.emnlp-main.774.pdf)** - *Zhang, R., et al. (2023)*. 
  * **[Code-Switched Language Identification is Harder Than You Think](https://aclanthology.org/2024.eacl-long.38.pdf)** - *Burchell, L., et al. (2024)*. 
  * **[Multilingual Identification of English Code-Switching](https://aclanthology.org/2024.vardial-1.14.pdf)** - *Sterner, I. (2024)*. 
  * **[MaskLID: Code-Switching Language Identification through Iterative Masking](https://aclanthology.org/2024.acl-short.43/)** - *Kargaran, A. H., et al. (2024)*. 
  
* **Offensive Language Identification** * **[MUCS@DravidianLangTech-EACL2021:COOLI-Code-Mixing Offensive Language Identification](https://aclanthology.org/2021.dravidianlangtech-1.47.pdf)** - *Balouchzahi, F., et al. (2021)*.
    * **[SJ AJ@DravidianLangTech-EACL2021: Task-Adaptive Pre-Training of Multilingual BERT models for Offensive Language Identification](https://aclanthology.org/2021.dravidianlangtech-1.44.pdf)** - *Jayanthi, S. M., et al. (2021)*.
    * **[DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text](https://link.springer.com/article/10.1007/s10579-022-09583-7)** - *Chakravarthi, B. R., et al. (2022)*.
    * **[Offensive Content Detection Via Synthetic Code-Switched Text](https://aclanthology.org/2022.coling-1.575.pdf)** - *Salaam, C., et al. (2022)*.
    * **[Offensive Language Identification in Transliterated and Code-Mixed Bangla](https://aclanthology.org/2023.banglalp-1.1.pdf)** - *Raihan, M. N., et al. (2023)*.
    * **[OffMix-3L: A Novel Code-Mixed Test Dataset in Bangla-English-Hindi for Offensive Language Identification](https://aclanthology.org/2023.socialnlp-1.3/)** - *Goswami, D., et al. (2023)*.
    * **[Towards Safer Communities: Detecting Aggression and Offensive Language in Code-Mixed Tweets to Combat Cyberbullying](https://aclanthology.org/2023.woah-1.3.pdf)** - *Nafis, N., et al. (2023)*.
    * **[SetFit: A Robust Approach for Offensive Content Detection in Tamil-English Code-Mixed Conversations Using Sentence Transfer Fine-tuning](https://aclanthology.org/2024.dravidianlangtech-1.6.pdf)** - *Kathiravan Pannerselvam, et al. (2024)*.
    * **[LLMsAgainstHate@NLU of Devanagari Script Languages 2025: Hate Speech Detection and Target Identification in Devanagari Languages via Parameter Efficient Fine-Tuning of LLMs](https://aclanthology.org/2025.chipsal-1.34/)** - *Rushendra Sidibomma, et al. (2025)*.

* **Hope Speech Detection** * **[SJ_AJ@DravidianLangTech-EACL2021: Task-Adaptive Pre-Training of Multilingual BERT models for Offensive Language Identification](https://aclanthology.org/2021.dravidianlangtech-1.44/)** - *Jayanthi, S. M., et al. (2021)*.
    * **[IRNLP_DAIICT@LT-EDI-EACL2021: Hope Speech detection in Code Mixed text using TF-IDF Char N-grams and MuRIL](https://aclanthology.org/2021.ltedi-1.15/)** - *Dave, B., et al. (2021)*.
    * **[Hope Speech Detection in Code-Mixed Text: Exploring Deep Learning Models and Language Effects](https://arxiv.org/abs/2108.04616)** - *Bhat, S., et al. (2021)*.
    * **[Hope Speech Detection in code-mixed Roman Urdu tweets](https://api.semanticscholar.org/CorpusID:280011615)** - *Ahmad, M., et al. (2025)*.

### Part-of-Speech (POS) Tagging
  * **[POS Tagging of English-Hindi Code-Mixed Social Media Content](https://aclanthology.org/D14-1105/)** - *Vyas, Y., et al. (2014)*.
  * **[POS Tagging of Hindi-English Code Mixed Text from Social Media: Some Machine Learning Experiments](https://aclanthology.org/W15-5936/)** - *Sequiera, R., et al. (2015)*.
  * **[Development of POS tagger for English-Bengali Code-Mixed data](https://aclanthology.org/2019.icon-1.17/)** - *Raha, T., et al. (2019)*.
  * **[Creation of Corpus and Analysis in Code-Mixed Kannada-English Social Media Data for POS Tagging](https://aclanthology.org/2020.icon-main.13/)** - *Appidi, A. R., et al. (2020)*.
  * **[A Pre-trained Transformer and CNN Model with Joint Language ID and Part-of-Speech Tagging for Code-Mixed Social-Media Text](https://aclanthology.org/2021.ranlp-1.42/)** - *Dowlagar, S. & Mamidi, R. (2021)*.
  * **[Are Multilingual Models Effective in Code-Switching?](https://aclanthology.org/2021.calcs-1.20.pdf)** - *Winata, G. I., et al. (2021)*.
  * **[On Utilizing Constituent Language Resources to Improve Downstream Tasks in Hinglish](https://aclanthology.org/2022.findings-emnlp.283.pdf)** - *Kumar, V., et al. (2022)*.
  * **[PRO-CS : An Instance-Based Prompt Composition Technique for Code-Switched Tasks](https://aclanthology.org/2022.emnlp-main.698.pdf)** - *Bansal, S., et al. (2022)*.
  * **[PACMAN: PArallel CodeMixed dAta generatioN for POS tagging](https://aclanthology.org/2022.icon-main.29.pdf)** - *Chatterjee, A., et al. (2022)*.
  * **[CoMix: Guide Transformers to Code-Mix using POS structure and Phonetics](https://aclanthology.org/2023.findings-acl.506.pdf)** - *Arora, G., et al. (2023)*.
  * **[Improving Sentiment Analysis for Ukrainian Social Media Code-Switching Data](https://aclanthology.org/2025.unlp-1.18.pdf)** - *Shynkarov, Y., et al. (2025)*.
  * **[Fine-Tuning Cross-Lingual LLMs for POS Tagging in Code-Switched Contexts](https://aclanthology.org/2025.resourceful-1.2/)** - *Absar, S. (2025)*.

### Named Entity Recognition (NER)

  * **[Tackling Code-Switched NER: Participation of CMU](https://aclanthology.org/W18-3217.pdf)** - *Geetha, P., et al. (2018)*.
  * **[Cross Script Hindi English NER Corpus from Wikipedia](https://arxiv.org/abs/1810.03430)** - *Ansari, M. Z., et al. (2019)*.
  * **[Character level neural architectures for boosting named entity recognition in code mixed tweets](https://api.semanticscholar.org/CorpusID:216587955)** - *Narayanan, A., et al. (2020)*.
  * **[CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP](https://www.ijcai.org/proceedings/2020/0533.pdf)** - *Qin, L., et al. (2020)*.
  * **[Contextual Embeddings for Arabic-English Code-Switched Data](https://aclanthology.org/2020.wanlp-1.20.pdf)** - *Sabty, C., et al. (2020)*.
  * **[Named Entity Recognition for Code Mixed Social Media Sentences](https://api.semanticscholar.org/CorpusID:232434202)** - *Sharma, Y., et al. (2021)*.
  * **[Switch Point biased Self-Training: Re-purposing Pretrained Models for Code-Switching](https://aclanthology.org/2021.findings-emnlp.373.pdf)** - *Chopra, P., et al. (2021)*.
  * **[Performance analysis of named entity recognition approaches on code-mixed data](https://api.semanticscholar.org/CorpusID:243100435)** - *Gaddamidi, S. & Prasath, R. R. (2021)*.
  * **[Are Multilingual Models Effective in Code-Switching?](https://aclanthology.org/2021.calcs-1.20.pdf)** - *Winata, G. I., et al. (2021)*.
  * **[CMNEROne at SemEval-2022 Task 11: Code-Mixed Named Entity Recognition by leveraging multilingual data](https://aclanthology.org/2022.semeval-1.214/)** - *Dowlagar, S. & Mamidi, R. (2022)*.
  * **[UM6P-CS at SemEval-2022 Task 11: Enhancing Multilingual and Code-Mixed Complex Named Entity Recognition via Pseudo Labels using Multilingual Transformer](https://aclanthology.org/2022.semeval-1.207.pdf)** - *El Mekki, A., et al. (2022)*.
  * **["Kanglish alli names\!" Named Entity Recognition for Kannada-English Code-Mixed Social Media Data](https://aclanthology.org/2022.wnut-1.17/)** - *S, Sumukh & Shrivastava, M. (2022)*.
  * **[MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER](https://aclanthology.org/2022.acl-long.160.pdf)** - *Zhou, R., et al. (2022)*.
  * **[CMB AI Lab at SemEval-2022 Task 11: A Two-Stage Approach for Complex Named Entity Recognition via Span Boundary Detection and Span Classification](https://aclanthology.org/2022.semeval-1.221.pdf)** - *PU, K., et al. (2022)*.
  * **[On Utilizing Constituent Language Resources to Improve Downstream Tasks in Hinglish](https://aclanthology.org/2022.findings-emnlp.283.pdf)** - *Kumar, V., et al. (2022)*.
  * **[COCOA: An Encoder-Decoder Model for Controllable Code-switched Generation](https://aclanthology.org/2022.emnlp-main.158.pdf)** - *Mondal, S., et al. (2022)*.
  * **[Sebastian, Basti, Wastl?! Recognizing Named Entities in Bavarian Dialectal Data](https://aclanthology.org/2024.lrec-main.1262.pdf)** - *Peng, S., et al. (2024)*.
  * **[GPT-NER: Named Entity Recognition via Large Language Models](https://aclanthology.org/2025.findings-naacl.239/)** - *Wang, S., et al. (2025)*.

### Sentiment & Emotion Analysis

  * **[Code-Mixing in Social Media Text: The Last Language Identification Frontier?](https://aclanthology.org/D18-1346.pdf)** - *Mave, D., et al. (2018)*.
  * **[Sentiment Analysis of Code-Mixed Hinglish](https://aclanthology.org/2020.wnut-1.22.pdf)** - *Saha, R., et al. (2020)*.
  * **[Sentiment Analysis of Code-Mixed Indian Languages: An Overview of SAIL_Code-Mixed Shared Task @ICON-2017](https://aclanthology.org/2020.semeval-1.123.pdf)** - *Patra, B. G., et al. (2020)*.
  * **[Overview of the Mixed Script Identification @ ICON-2020](https://aclanthology.org/2020.semeval-1.124.pdf)** - *Sequiera, R., et al. (2020)*.
  * **[SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets](https://aclanthology.org/2020.semeval-1.100/)** - *Patwa, P., et al. (2020)*.
  * **[Sentiment Analysis for Hinglish Code-mixed Tweets by means of Cross-lingual Word Embeddings](https://aclanthology.org/2020.semeval-1.180.pdf)** - *Tiwari, P., et al. (2020)*.
  * **[Sentiment Analysis in Code-Mixed Telugu-English Text with Multilingual Embeddings](https://aclanthology.org/2020.semeval-1.181.pdf)** - *Yasaswini, K., et al. (2020)*.
  * **[Data Augmentation for Low-Resource Code-Switching Speech Recognition](https://aclanthology.org/2020.semeval-1.165.pdf)** - *Gonen, H., et al. (2020)*.
  * **[CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP](https://www.ijcai.org/proceedings/2020/0533.pdf)** - *Qin, L., et al. (2020)*.
  * **[Evaluating Input Representation for Language Identification in Hindi-English Code Mixed Text](https://aclanthology.org/2020.semeval-1.174.pdf)** - *Singh, K., et al. (2020)*.
  * **[BERT-based Language Identification in Code-Mixed Social Media Text](https://aclanthology.org/2021.calcs-1.8.pdf)** - *Dowlagar, S., et al. (2021)*.
  * **[Multitask Learning for Emotionally Analyzing Code-Mixed Social Media Text](https://aclanthology.org/2021.calcs-1.13.pdf)** - *Dowlagar, S., et al. (2021)*.
  * **[Offensive Language Detection in Code-Mixed Social Media Text](https://aclanthology.org/2021.mrl-1.16.pdf)** - *Suryawanshi, S., et al. (2021)*.
  * **[From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](https://aclanthology.org/2021.emnlp-main.727.pdf)** - *Gautam, S., et al. (2021)*.
  * **[Sentiment Analysis For Code-Mixed Indian Social Media Text With Code-Mix Embedding](https://aclanthology.org/2021.wassa-1.21.pdf)** - *Suryawanshi, S., et al. (2021)*.
  * **[Hope Speech Detection in Code-Mixed Dravidian Languages](https://aclanthology.org/2021.dravidianlangtech-1.8.pdf)** - *Chakravarthi, B. R., et al. (2021)*.
  * **[DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text](https://link.springer.com/article/10.1007/s10579-022-09583-7)** - *Chakravarthi, B. R., et al. (2022)*.
  * **[Code-Switching Patterns in Multilingual Dialogue Systems](https://aclanthology.org/2022.sumeval-1.5/)** - *Sitaram, S., et al. (2022)*.
  * **[Code-Switching Text Generation for Multilingual Dialogue](https://aclanthology.org/2022.inlg-genchal.4.pdf)** - *Sitaram, S., et al. (2022)*.
  * **[PRO-CS : An Instance-Based Prompt Composition Technique for Code-Switched Tasks](https://aclanthology.org/2022.emnlp-main.698.pdf)** - *Bansal, S., et al. (2022)*.
  * **[Multi-Label Emotion Classification on Code-Mixed Text: Data and Methods](https://www.google.com/search?q=https://doi.org/10.1109/ACCESS.2022.3143819)** - *Ameer, I., et al. (2022)*.
  * **[Code-Mixed Sentiment Analysis with Pretrained Language Models](https://aclanthology.org/2022.paclic-1.7.pdf)** - *Sitaram, S., et al. (2022)*.
  * **[Code-Mixed Sentiment Analysis with Data Augmentation](https://aclanthology.org/2022.findings-emnlp.499.pdf)** - *Saha, R., et al. (2022)*.
  * **[Sentiment Analysis in Code-Mixed Low-Resource Dravidian Languages](https://aclanthology.org/2023.sealp-1.6/)** - *Chakravarthi, B. R., et al. (2023)*.
  * **[Multitask Learning for Code-Mixed Sentiment and Emotion Analysis](https://aclanthology.org/2023.calcs-1.6/)** - *Dowlagar, S., et al. (2023)*.
  * **[Sentiment Analysis for Code-Mixed Indian Language Texts](https://aclanthology.org/2023.eacl-main.57.pdf)** - *Sitaram, S., et al. (2023)*.
  * **[Emotion Analysis in Code-Mixed WhatsApp Messages](https://aclanthology.org/2023.wassa-1.32.pdf)** - *Suryawanshi, S., et al. (2023)*.
  * **[Offensive Language Identification in Code-Mixed Dravidian Languages](https://aclanthology.org/2023.dravidianlangtech-1.40.pdf)** - *Chakravarthi, B. R., et al. (2023)*.
  * **[Emotion Detection in Code-Mixed Roman Urdu - English Text](https://aclanthology.org/2023.wassa-1.59.pdf)** - *Suryawanshi, S., et al. (2023)*.
  * **[Sarcasm Detection in Dravidian Code-Mixed Text Using Transformer-Based Models](https://www.google.com/search?q=https://citeseerx.ist.psu.edu/document/10.1.1.1092.4862)** - *Bhaumik, A. B. & Das, M. (2023)*.
  * **[Hate Speech Detection in Code-Mixed Hinglish Text](https://aclanthology.org/2023.wassa-1.61.pdf)** - *Saha, R., et al. (2023)*.
  * **[Findings of the WILDRE Shared Task on Code-mixed Less-resourced Sentiment Analysis for Indo-Aryan Languages](https://aclanthology.org/2024.wildre-1.2.pdf)** - *Mishra, A., et al. (2024)*.
  * **[Findings of the WILDRE Shared Task on Code-mixed Less-resourced Sentiment Analysis for Indo-Aryan Languages](https://aclanthology.org/2024.wildre-1.3/)** - *Mishra, A., et al. (2024)*.
  * **[WILDRE Shared Task: Sentiment Analysis in Code-Mixed Telugu-English Text](https://aclanthology.org/2024.wildre-1.10.pdf)** - *Chakravarthi, B. R., et al. (2024)*.
  * **[Code-Mixed Sentiment Analysis with Multimodal Data](https://aclanthology.org/2024.sicon-1.6.pdf)** - *Sitaram, S., et al. (2024)*.
  * **[SemEval-2024 Task 9: Sentiment Analysis in Code-Mixed Text](https://aclanthology.org/2024.semeval-1.56.pdf)** - *Patra, B. G., et al. (2024)*.
  * **[Emotion Analysis in Code-Mixed Social Media Text](https://aclanthology.org/2024.wassa-1.19.pdf)** - *Suryawanshi, S., et al. (2024)*.
  * **[Explainable Sentiment Analysis in Code-Mixed Text](https://aclanthology.org/2024.lrec-main.1234.pdf)** - *Gupta, A., et al. (2024)*.
  * **[Improving Sentiment Analysis for Ukrainian Social Media Code-Switching Data](https://aclanthology.org/2025.unlp-1.18.pdf)** - *Shynkarov, Y., et al. (2025)*.
  * **[Code-Mixed Sentiment Analysis with Low-Resource Settings](https://aclanthology.org/2025.naacl-short.29/)** - *Sitaram, S., et al. (2025)*.
  * **[Cross-Lingual Transfer for Code-Mixed Sentiment Analysis](https://aclanthology.org/2025.naacl-long.260.pdf)** - *Saha, R., et al. (2025)*.
  * **[Improving Sentiment Analysis for Ukrainian Social Media Code-Switching Data](https://aclanthology.org/2025.unlp-1.18/)** - *Shynkarov, Y., et al. (2025)*.
  

### Syntactic Analysis
  * **[Code-Switching Language Modeling using Syntax-Aware Multi-Task Learning](https://aclanthology.org/W18-3207.pdf)** - *Winata, G. I., et al. (2018)*.
  * **[Language Modeling for Code-Mixing: The Role of Linguistic Theory based Synthetic Data](https://aclanthology.org/P18-1143.pdf)** - *Pratapa, A., et al. (2018)*.
  * **[Dependency Parser for Bengali-English Code-Mixed Data enhanced with a Synthetic Treebank](https://aclanthology.org/W19-7810.pdf)** - *Ghosh, U., et al. (2019)*.
  * **[A Semi-supervised Approach to Generate the Code-Mixed Text using Pre-trained Encoder and Transfer Learning](https://aclanthology.org/2020.findings-emnlp.206.pdf)** - *Gupta, D., et al. (2020)*.
  * **[From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](https://aclanthology.org/2021.acl-long.245.pdf)** - *Tarunesh, I., et al. (2021)*.
  * **[PreCogIIITH at HinglishEval : Leveraging Code-Mixing Metrics & Language Model Embeddings To Estimate Code-Mix Quality](https://aclanthology.org/2022.inlg-genchal.4.pdf)** - *Kodali, P., et al. (2022)*.
  * **[SyMCoM - Syntactic Measure of Code Mixing A Study Of English-Hindi Code-Mixing](https://aclanthology.org/2022.findings-acl.40/)** - *Kodali, P., et al. (2022)*.
  * **[Improving Code-Switching Dependency Parsing with Semi-Supervised Auxiliary Tasks](https://aclanthology.org/2022.findings-naacl.87/)** - *Özateş, Ş. B., et al. (2022)*.
  * **[CoMix: Guide Transformers to Code-Mix using POS structure and Phonetics](https://aclanthology.org/2023.findings-acl.506.pdf)** - *Arora, G., et al. (2023)*.
  * **[CST5: Data Augmentation for Code-Switched Semantic Parsing](https://aclanthology.org/2023.tllm-1.1.pdf)** - *Agarwal, A., et al. (2023)*.
  * **[Sebastian, Basti, Wastl?! Recognizing Named Entities in Bavarian Dialectal Data](https://aclanthology.org/2024.lrec-main.307.pdf)** - *Peng, S., et al. (2024)*.
  * **[Towards Safer Communities: Detecting Aggression and Offensive Language in Code-Mixed Tweets to Combat Cyberbullying](https://aclanthology.org/2024.emnlp-main.942.pdf)** - *Nafis, N., et al. (2024)*.
  * **[Fine-Tuning Cross-Lingual LLMs for POS Tagging in Code-Switched Contexts](https://aclanthology.org/2024.findings-emnlp.916.pdf)** - *Absar, S. (2024)*.
  * **[Representativeness as a Forgotten Lesson for Multilingual and Code-switched Data Collection and Preparation](https://aclanthology.org/2024.lrec-main.698.pdf)** - *Doğruöz, A. S., et al. (2024)*.
  * **[A Survey of Code-switched Arabic NLP: Progress, Challenges, and Future Directions](https://aclanthology.org/2025.americasnlp-1.2.pdf)** - *Hamed, I., et al. (2025)*.
  * **[From Human Judgements to Predictive Models: Unravelling Acceptability in Code-Mixed Sentences](https://dl.acm.org/doi/abs/10.1145/3748312)** - *Kodali, P., et al. (2025)*.


### Machine Translation (MT)

  * **[Code-Switching for Enhancing NMT with Pre-Specified Translation](https://aclanthology.org/N19-1044.pdf)** - *Song, K., et al. (2019)*.
  * **[PhraseOut: A Code Mixed Data Augmentation Method for Multilingual Neural Machine Translation](https://aclanthology.org/2020.icon-main.63.pdf)** - *Jasim, B., et al. (2020)*.
  * **[CoMeT: Towards Code-Mixed Translation Using Parallel Monolingual Sentences](https://aclanthology.org/2021.calcs-1.7.pdf)** - *Gautam, D., et al. (2021)*.
  * **[Training Data Augmentation for Code-Mixed Translation](https://aclanthology.org/2021.naacl-main.459.pdf)** - *Gupta, A., et al. (2021)*.
  * **[Translate and Classify: Improving Sequence Level Classification for English-Hindi Code-Mixed Data](https://aclanthology.org/2021.calcs-1.3.pdf)** - *Gautam, D., et al. (2021)*.
  * **[Gated Convolutional Sequence to Sequence Based Learning for English-Hinglish Code-Switched Machine Translation](https://aclanthology.org/2021.calcs-1.4.pdf)** - *Dowlagar, S., et al. (2021)*.
  * **[IITP-MT at CALCS2021: English to Hinglish Neural Machine Translation using Unsupervised Synthetic Code-Mixed Parallel Corpus](https://aclanthology.org/2021.calcs-1.5.pdf)** - *Appicharla, R., et al. (2021)*.
  * **[Exploring Text-to-Text Transformers for English to Hinglish Machine Translation with Synthetic Code-Mixing](https://aclanthology.org/2021.calcs-1.6.pdf)** - *Jawahar, G., et al. (2021)*.
  * **[Investigating Code-Mixed Modern Standard Arabic-Egyptian to English Machine Translation](https://aclanthology.org/2021.calcs-1.8.pdf)** - *Nagoudi, E. M. B., et al. (2021)*.
  * **[Hinglish to English Machine Translation using Multilingual Transformers](https://aclanthology.org/2021.ranlp-srw.3.pdf)** - *Agarwal, V., et al. (2021)*.
  * **[Neural Machine Translation for Sinhala-English Code-Mixed Text](https://aclanthology.org/2021.ranlp-1.82.pdf)** - *Kugathasan, A., et al. (2021)*.
  * **[From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](https://aclanthology.org/2021.acl-long.245.pdf)** - *Tarunesh, I., et al. (2021)*.
  * **[Adapting Multilingual Models for Code-Mixed Translation](https://aclanthology.org/2022.findings-emnlp.528.pdf)** - *Vavre, A., et al. (2022)*.
  * **[MUCS@MixMT: indicTrans-based Machine Translation for Hinglish Text](https://aclanthology.org/2022.wmt-1.113.pdf)** - *Hegde, A., et al. (2022)*.
  * **[SIT at MixMT 2022: Fluent Translation Built on Giant Pre-trained Models](https://aclanthology.org/2022.wmt-1.114.pdf)** - *Khan, A. R., et al. (2022)*.
  * **[Gui at MixMT 2022 : English-Hinglish : An MT approach for translation of code mixed data](https://aclanthology.org/2022.wmt-1.112.pdf)** - *Gahoi, A., et al. (2022)*.
  * **[CNLP-NITS-PP at MixMT 2022: Hinglish–English Code-Mixed Machine Translation](https://aclanthology.org/2022.wmt-1.116.pdf)** - *Laskar, S. R., et al. (2022)*.
  * **[End-to-End Speech Translation for Code Switched Speech](https://aclanthology.org/2022.findings-acl.113.pdf)** - *Weller, O., et al. (2022)*.
  * **[MALM: Mixing Augmented Language Modeling for Zero-Shot Machine Translation](https://aclanthology.org/2022.nlp4dh-1.8.pdf)** - *Gupta, K. (2022)*.
  * **[Can You Translate for Me? Code-Switched Machine Translation with Large Language Models](https://aclanthology.org/2023.ijcnlp-short.10/)** - *Khatri, J., et al. (2023)*.
  * **[Lost in Translation No More : Fine-tuned transformer-based models for CodeMix to English Machine Translation](https://aclanthology.org/2023.icon-1.25.pdf)** - *Chatterjee, A., et al. (2023)*.
  * **[Enhancing Code-mixed Text Generation Using Synthetic Data Filtering in Neural Machine Translation](https://aclanthology.org/2023.conll-1.15.pdf)** - *Sravani, D., et al. (2023)*.
  * **[Towards Real-World Streaming Speech Translation for Code-Switched Speech](https://aclanthology.org/2023.calcs-1.2.pdf)** - *Alastruey, B., et al. (2023)*.
  * **[Exploring Segmentation Approaches for Neural Machine Translation of Code-Switched Egyptian Arabic-English Text](https://aclanthology.org/2023.eacl-main.256.pdf)** - *Gaser, M., et al. (2023)*.
  * **[Exploring Enhanced Code-Switched Noising for Pretraining in Neural Machine Translation](https://aclanthology.org/2023.findings-eacl.72.pdf)** - *Iyer, V., et al. (2023)*.
  * **[Evaluating Code-Switching Translation with Large Language Models](https://aclanthology.org/2024.lrec-main.565.pdf)** - *Huzaifah, M., et al. (2024)*.
  * **[Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?](https://aclanthology.org/2024.findings-eacl.71.pdf)** - *Hada, R., et al. (2024)*.
  * **[ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval](https://aclanthology.org/2024.naacl-short.17.pdf)** - *Do, J., et al. (2024)*.
  * **[Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation](https://aclanthology.org/2024.lrec-main.1345.pdf)** - *Kartik, et al. (2024)*.
  * **[CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units](https://aclanthology.org/2024.acl-srw.40.pdf)** - *Kang, Y. (2024)*.
  * **[Improving Code-Switched Machine Translation with Large Language Models and Synthetic Data](https://aclanthology.org/2024.findings-acl.128/)** - *Chen, Y., et al. (2024)*.
  * **[MIGRATE: Cross-Lingual Adaptation of Domain-Specific LLMs through Code-Switching and Embedding Transfer](https://aclanthology.org/2025.coling-main.617.pdf)** - *Hong, S., et al. (2025)*.
  * **[Next-Level Cantonese-to-Mandarin Translation: Fine-Tuning and Post-Processing with LLMs](https://aclanthology.org/2025.loreslm-1.32.pdf)** - *Dai, Y., et al. (2025)*.
  * **[Investigating and Scaling up Code-Switching for Multilingual Language Model Pre-Training](https://aclanthology.org/2025.findings-acl.575.pdf)** - *Wang, Z., et al. (2025)*.
  * **[From English to Second Language Mastery: Enhancing LLMs with Cross-Lingual Continued Instruction Tuning](https://aclanthology.org/2025.acl-long.1121.pdf)** - *Wu, L., et al. (2025)*.
  * **[The Impact of Code-switched Synthetic Data Quality is Task Dependent: Insights from MT and ASR](https://aclanthology.org/2025.calcs-1.2.pdf)** - *Hamed, I., et al. (2025)*.
  * **[Tongue-Tied: Breaking LLMs Safety Through New Language Learning](https://aclanthology.org/2025.calcs-1.5.pdf)** - *Upadhayay, B., et al. (2025)*.
  * **[Low-resource Machine Translation for Code-switched Kazakh-Russian Language Pair](https://aclanthology.org/2025.naacl-srw.7.pdf)** - *Borisov, M., et al. (2025)*.

-----

## 1.2\. Emerging and Contemporary Tasks

> Tasks focused on generating fluent and coherent code-mixed text.

### Natural Language Inference (NLI)

  * **[Detecting entailment in code-mixed Hindi-English conversations](https://aclanthology.org/2020.wnut-1.22/)** - *Chakravarthy, S., et al. (2020)*.
  * **[A New Dataset for Natural Language Inference from Code-mixed Conversations](https://aclanthology.org/2020.calcs-1.2/)** - *Khanuja, S., et al. (2020)*.
  * **[CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP](https://www.ijcai.org/proceedings/2020/0533.pdf)** - *Qin, L., et al. (2020)*.
* **[The Effectiveness of Intermediate-Task Training for Code-Switched Natural Language Understanding](https://aclanthology.org/2021.mrl-1.16.pdf)** - *Prasad, A., et al. (2021)*.
* **[On Utilizing Constituent Language Resources to Improve Downstream Tasks in Hinglish](https://aclanthology.org/2022.findings-emnlp.283.pdf)** - *Kumar, V., et al. (2022)*.
* **[Toward the Limitation of Code-Switching in Cross-Lingual Transfer](https://aclanthology.org/2022.emnlp-main.400.pdf)** - *Feng, Y., et al. (2022)*.
* **[Aligning Multilingual Embeddings for Improved Code-switched Natural Language Understanding](https://aclanthology.org/2022.coling-1.375.pdf)** - *Fazili, B., et al. (2022)*.
* **[Incontext Mixing (ICM): Codemixed Prompts for Multilingual LLMs](https://aclanthology.org/2024.acl-long.228.pdf)** - *Shankar, B., et al. (2024)*.
* **[Using Contextually Aligned Online Reviews to Measure LLMs’ Performance Disparities Across Language Varieties](https://aclanthology.org/2025.naacl-short.29/)** - *Tang, Z., et al. (2025)*. 

### Intent Classification

  * **[IIT Gandhinagar at SemEval-2020 Task 9: Code-Mixed Sentiment Classification Using Candidate Sentence Generation and Selection](https://aclanthology.org/2020.semeval-1.168/)** - *Srivastava, V. & Singh, M. (2020)*.
  * **[Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling](https://aclanthology.org/2021.mrl-1.18/)** - *Krishnan, J., et al. (2021)*.
  * **[Regional language code-switching for natural language understanding and intelligent digital assistants](https://doi.org/10.1007/978-981-16-0749-3_71)** - *Rajeshwari, S. & Kallimani, J. S. (2021)*.
  * **[Cost-Performance Optimization for Processing Low-Resource Language Tasks Using Commercial LLMs](https://aclanthology.org/2024.findings-emnlp.920.pdf)** - *Nag, A., et al. (2024)*.

### Question Answering (QA)

  * **[Uncovering Code-Mixed Challenges: A Framework for Linguistically Driven Question Generation and Neural based Question Answering](https://aclanthology.org/K18-1012/)** - *Gupta, D., et al. (2018)*.
  * **[Code-Mixed Question Answering Challenge using Deep Learning Methods](https://www.google.com/search?q=https://doi.org/10.1109/ICCES48766.2020.9137971)** - *Thara, S., et al. (2020)*.
  * **[MLQA: Evaluating Cross-lingual Extractive Question Answering](https://aclanthology.org/2020.acl-main.653/)** - *Lewis, P., et al. (2020)*
  * **[The Effectiveness of Intermediate-Task Training for Code-Switched Natural Language Understanding](https://aclanthology.org/2021.mrl-1.16.pdf)** - *Prasad, A., et al. (2021)*.
  * **[To Ask LLMs about English Grammaticality, Prompt Them in a Different Language](https://aclanthology.org/2024.findings-emnlp.916.pdf)** - *Behzad, S., et al. (2024)*.
  * **[COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning](https://aclanthology.org/2024.findings-naacl.198.pdf)** - *Lee, J., et al. (2024)*.
  * **[MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks](https://aclanthology.org/2024.naacl-long.143/)** - *Ahuja, S., et al. (2024)*.
  * **[Controlling Language Confusion in Multilingual LLMs](https://aclanthology.org/2025.acl-srw.81/)** - *Lee, N., et al. (2025)*.
  * **[Qorǵau: Evaluating Safety in Kazakh-Russian Bilingual Contexts](https://aclanthology.org/2025.findings-acl.507.pdf)** - *Goloburda, M., et al. (2025)*.
  * **[Code-Switching Curriculum Learning for Multilingual Transfer in LLMs](https://aclanthology.org/2025.findings-acl.407/)** - *Yoo, H., et al. (2025)*.
      
### Code-Mixed Text Generation

  * **[A Deep Generative Model for Code Switched Text](https://doi.org/10.24963/ijcai.2019/719)** - *Samanta, B., et al. (2019)*.
  * **[A Semi-supervised Approach to Generate the Code-Mixed Text using Pre-trained Encoder and Transfer Learning](https://aclanthology.org/2020.findings-emnlp.206.pdf)** - *Gupta, D., et al. (2020)*.
  * **[Towards Code-Mixed Hinglish Dialogue Generation](https://aclanthology.org/2021.nlp4convai-1.26.pdf)** - *Agarwal, V., et al. (2021)*.
  * **[HinGE: A Dataset for Generation and Evaluation of Code-Mixed Hinglish Text](https://aclanthology.org/2021.eval4nlp-1.20.pdf)** - *Srivastava, V., et al. (2021)*.
  * **[From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](https://aclanthology.org/2021.acl-long.245.pdf)** - *Tarunesh, I., et al. (2021)*.
  * **[PACMAN:PArallel CodeMixed dAta generatioN for POS tagging](https://aclanthology.org/2022.icon-main.29.pdf)** - *Chatterjee, A., et al. (2022)*.
  * **[MulZDG: Multilingual Code-Switching Framework for Zero-shot Dialogue Generation](https://aclanthology.org/2022.coling-1.54.pdf)** - *Liu, Y., et al. (2022)*.
  * **[Proceedings of the 15th International Conference on Natural Language Generation: Generation Challenges](https://aclanthology.org/2022.inlg-genchal.0.pdf)** - *Shaikh, S., et al. (2022)*.
  * **[CoCoa: An Encoder-Decoder Model for Controllable Code-switched Generation](https://aclanthology.org/2022.emnlp-main.158.pdf)** - *Mondal, S., et al. (2022)*.
  * **[Prompting Multilingual Large Language Models to Generate Code-Mixed Texts: The Case of South East Asian Languages](https://aclanthology.org/2023.calcs-1.5.pdf)** - *Yong, Z. X., et al. (2023)*.
  * **[Enhancing Code-mixed Text Generation Using Synthetic Data Filtering in Neural Machine Translation](https://aclanthology.org/2023.conll-1.15.pdf)** - *Sravani, D., et al. (2023)*.
  * **[Code-Switched Text Synthesis in Unseen Language Pairs](https://aclanthology.org/2023.findings-acl.318.pdf)** - *Hsu, I.-H., et al. (2023)*.
  * **[Linguistics Theory Meets LLM: Code-Switched Text Generation via Equivalence Constrained Large Language Models](https://api.semanticscholar.org/CorpusID:273695372)** - *Kuwanto, G., et al. (2024)*.
  * **[Leveraging Large Language Models for Code-Mixed Data Augmentation in Sentiment Analysis](https://aclanthology.org/2024.sicon-1.6.pdf)** - *Zeng, L. (2024)*.
  * **[Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation](https://aclanthology.org/2024.lrec-main.1345.pdf)** - *Kartik, K., et al. (2024)*.
  * **[LLM-based Code-Switched Text Generation for Grammatical Error Correction](https://aclanthology.org/2024.emnlp-main.942.pdf)** - *Potter, T., et al. (2024)*.
  * **[Understanding and Mitigating Language Confusion in LLMs](https://aclanthology.org/2024.emnlp-main.380.pdf)** - *Marchisio, K., et al. (2024)*.
  * **Pun Generation** * **[Bridging Laughter Across Languages: Generation of Hindi-English Code-mixed Puns](https://aclanthology.org/2025.chum-1.5.pdf)** - *Asapu, L., et al. (2025)*.
    * **[Homophonic Pun Generation in Code Mixed Hindi English](https://aclanthology.org/2025.chum-1.4/)** - *Sarrof, Y. R. (2025)*.

### Cross-lingual Transfer

  * **[XLP at SemEval-2020 Task 9: Cross-lingual Models with Focal Loss for Sentiment Analysis of Code-Mixing Language](https://aclanthology.org/2020.semeval-1.126.pdf)** - *Ma, Y., et al. (2020)*.
  * **[CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP](https://www.ijcai.org/proceedings/2020/0533.pdf)** - *Qin, L., et al. (2020)*.
  * **[Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling](https://aclanthology.org/2021.mrl-1.18.pdf)** - *Krishnan, J., et al. (2021)*.
  * **[Saliency-based Multi-View Mixed Language Training for Zero-shot Cross-lingual Classification](https://aclanthology.org/2021.findings-emnlp.55.pdf)** - *Lai, S., et al. (2021)*.
  * **[Scopa: Soft code-switching and pairwise alignment for zero-shot cross-lingual transfer](https://dl.acm.org/doi/10.1145/3459637.3482176)** - *Lee, D., et al. (2021)*.
  * **[Toward the Limitation of Code-Switching in Cross-Lingual Transfer](https://aclanthology.org/2022.emnlp-main.400.pdf)** - *Feng, Y., et al. (2022)*.
  * **[ENTITYCS: Improving Zero-Shot Cross-lingual Transfer with Entity-Centric Code Switching](https://aclanthology.org/2022.findings-emnlp.499.pdf)** - *Whitehouse, C., et al. (2022)*.
  * **[Improving Zero-Shot Cross-Lingual Transfer via Progressive Code-Switching](https://api.semanticscholar.org/CorpusID:270619569)** - *Li, Z., et al. (2024)*. 
  * **[Test-Time Code-Switching for Cross-lingual Aspect Sentiment Triplet Extraction](https://aclanthology.org/2025.naacl-long.260.pdf)** - *Sheng, D., et al. (2025)*.

### Text Summarization

  * **[GupShup: Summarizing Open-Domain Code-Switched Conversations](https://aclanthology.org/2021.emnlp-main.499/)** - *Mehnaz, L., et al. (2021)*.
  * **[Multilingual Large Language Models Are Not (Yet) Code-Switchers](https://aclanthology.org/2023.emnlp-main.774.pdf)** - *Zhang, R., et al. (2023)*.
  * **[CoMix: Guide Transformers to Code-Mix using POS structure and Phonetics](https://aclanthology.org/2023.findings-acl.506.pdf)** - *Arora, G., et al. (2023)*.
  * **[Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?](https://aclanthology.org/2024.findings-eacl.71.pdf)** - *Hada, R., et al. (2024)*.
  * **[CroCoSum: A Benchmark Dataset for Cross-Lingual Code-Switched Summarization](https://aclanthology.org/2024.lrec-main.367/)** - *Zhang, R. & Eickhoff, C. (2024)*.
  * **[Code-Switching Curriculum Learning for Multilingual Transfer in LLMs](https://aclanthology.org/2025.findings-acl.407.pdf)** - *Yoo, H., et al. (2025)*.
  * **[An Adapted Few-Shot Prompting Technique Using ChatGPT to Advance Low-Resource Languages Understanding](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11016028)** - *Sarrof, Y. R., et al. (2025)*.

### Dialogue Generation

  * **[Detecting Entailment in Code-Mixed Hindi-English Conversations](https://aclanthology.org/2020.wnut-1.22.pdf)** - *Sharanya Chakravarthy, et al. (2020)*.
  * **[A New Dataset for Natural Language Inference from Code-mixed Conversations](https://aclanthology.org/2020.calcs-1.2.pdf)** - *Simran Khanuja, et al. (2020)*.
  * **[Do Multilingual Users Prefer Chat-bots that Code-mix? Let's Nudge and Find Out!](https://dl.acm.org/doi/abs/10.1145/3392846)** - *Anshul Bawa, et al. (2020)*.
  * **[CoSDA-ML: Multi-Lingual Code-Switching Data Augmentation for Zero-Shot Cross-Lingual NLP](https://www.ijcai.org/proceedings/2020/0533.pdf)** - *Libo Qin, et al. (2020)*.
  * **[Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling](https://aclanthology.org/2021.mrl-1.18.pdf)** - *Jitin Krishnan, et al. (2021)*.
  * **[Towards Code-Mixed Hinglish Dialogue Generation](https://aclanthology.org/2021.nlp4convai-1.26.pdf)** - *Vibhav Agarwal, et al. (2021)*.
  * **[GupShup: Summarizing Open-Domain Code-Switched Conversations](https://aclanthology.org/2021.emnlp-main.499.pdf)** - *Laiba Mehnaz, et al. (2021)*.
  * **[Code-switched inspired losses for generic spoken dialog representations](https://aclanthology.org/2021.emnlp-main.656.pdf)** - *Emile Chapuis, et al. (2021)*.
  * **[Towards Code-Mixed Hinglish Dialogue Generation](https://aclanthology.org/2021.nlp4convai-1.26.pdf)** - *Vibhav Agarwal, et al. (2021)*.
  * **[MulZDG: Multilingual Code-Switching Framework for Zero-shot Dialogue Generation](https://aclanthology.org/2022.coling-1.54.pdf)** - *Yongkang Liu, et al. (2022)*.
  * **[X-RiSAWOZ: High-Quality End-to-End Multilingual Dialogue Datasets and Few-shot Agents](https://aclanthology.org/2023.findings-acl.174.pdf)** - *Mehrad Moradshahi, et al. (2023)*.
  * **[CST5: Data Augmentation for Code-Switched Semantic Parsing](https://aclanthology.org/2023.tllm-1.1/)** - *Agarwal, A., et al. (2023)*.
  * **[Does a code-switching dialogue system help users learn conversational fluency in Choctaw?](https://aclanthology.org/2025.americasnlp-1.2.pdf)** - *Jacqueline Brixey, et al. (2025)*.
  * **[Performance Analysis of Effective Retrieval of Kannada Translations in Code-Mixed Sentences using BERT and MPnet](https://etasr.com/index.php/ETASR/article/view/9013/4413)** - *H. P. Rohith, et al. (2025)*.

### Transliteration

  * **[Towards an Efficient Code-Mixed Grapheme-to-Phoneme Conversion in an Agglutinative Language: A Case Study on To-Korean Transliteration](https://aclanthology.org/2020.calcs-1.9.pdf)** - *Won Ik Cho, et al. (2020)*.
  * **[Detecting Entailment in Code-Mixed Hindi-English Conversations](https://aclanthology.org/2020.wnut-1.22.pdf)** - *Sharanya Chakravarthy, et al. (2020)*.
  * **[Normalization and Back-Transliteration for Code-Switched Data](https://api.semanticscholar.org/CorpusID:235097478)** - *Parikh, D. & Solorio, T. (2021)*.
  * **[Abusive content detection in transliterated Bengali-English social media corpus](https://aclanthology.org/2021.calcs-1.16.pdf)** - *Salim Sazzed (2021)*.
  * **[MUCS@MixMT: indicTrans-based Machine Translation for Hinglish Text](https://aclanthology.org/2022.wmt-1.113.pdf)** - *Asha Hegde, et al. (2022)*.
  * **[CodeSwitching and BackTransliteration Using a Bilingual Model](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-eacl.102.pdf)** - *Daniel Weisberg Mitelman, et al. (2024)*.
  * **[Cost-Performance Optimization for Processing Low-Resource Language Tasks Using Commercial LLMs](https://aclanthology.org/2024.findings-emnlp.920.pdf)** - *Arijit Nag, et al. (2024)*.
  * **[Homophonic Pun Generation in Code Mixed Hindi English](https://aclanthology.org/2025.chum-1.4/)** - *Yash Raj Sarrof (2025)*.
-----
## 1.3\. Underexplored and Frontier Tasks

> High-potential research directions where Code-Switching intersects with reasoning, safety, creativity, and multimodal interaction.

### Conversation & Speech
*Challenges in naturalistic mixing, phonetic disambiguation, and user engagement.*

* **[BanglAssist: A Bengali-English Generative AI Chatbot for Code-Switching and Dialect-Handling in Customer Service](https://arxiv.org/abs/2503.22283)** - *Francesco Kruk (2025)*.
* **[Does a code-switching dialogue system help users learn conversational fluency in Choctaw?](https://aclanthology.org/2025.americasnlp-1.2.pdf)** - *Jacqueline Brixey et al. (2025)*.
* **[X-RiSAWOZ: High-Quality End-to-End Multilingual Dialogue Datasets and Few-shot Agents](https://aclanthology.org/2023.findings-acl.174/)** - *Mehrad Moradshahi et al. (2023)*.
* **[Development of a code-switched Hindi-Marathi dataset and transformer-based architecture for enhanced speech recognition](https://www.sciencedirect.com/science/article/abs/pii/S0003682X24005590)** - *P. Hemant et al. (2025)*. 
* **[Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10890030)** - *Yu Xi et al. (2024)*. 
* **[Enhancing ASR Accuracy and Coherence Across Indian Languages with Wav2Vec2 and GPT-2](https://ictactjournals.in/paper/IJDSML_Vol_6_Iss_2_Paper_4_761_764.pdf)** - *R. Geetha Rajakumari et al. (2025)*. 

### Safety & Multimodal
*Addressing vulnerability to jailbreaks and grounding failures in mixed-media.*

* **[Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](https://aclanthology.org/2025.acl-long.657/)** - *Haneul Yoo et al. (2025)*. 
* **[Tongue-Tied: Breaking LLMs Safety Through New Language Learning](https://aclanthology.org/2025.calcs-1.5.pdf)** - *Bibek Upadhayay et al. (2025)*. 
* **[CM_CLIP: Unveiling Code-Mixed Multimodal Learning with Cross-Lingual CLIP Adaptations](https://aclanthology.org/2024.icon-1.36/)** - *Gitanjali Kumari et al. (2024)*. 
* **[ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos](https://aclanthology.org/2024.findings-acl.663/)** - *Krishanu Maity et al. (2024)*. 
* **[Multi-task detection of harmful content in code-mixed meme captions using large language models](https://www.sciencedirect.com/science/article/pii/S1110866525000763)** - *Bharath Kancharla et al. (2025)*. 

### Reasoning & Abstraction
*Testing logical, causal, and metaphorical understanding across language boundaries.*

* **[Lost in the Mix: Evaluating LLM Understanding of Code-Switched Text](https://arxiv.org/abs/2506.14012)** - *Amr Mohamed et al. (2025)*. 
* **[SentMix-3L: A Bangla-English-Hindi Code-Mixed Dataset for Sentiment Analysis](https://aclanthology.org/2023.sealp-1.6/)** - *Md Nishat Raihan et al. (2023)*. 
* **[From Human Judgements to Predictive Models: Unravelling Acceptability in Code-Mixed Sentences](https://dl.acm.org/doi/abs/10.1145/3748312)** - *Prashant Kodali et al. (2025)*. 
* **[GupShup: Summarizing Open-Domain Code-Switched Conversations](https://aclanthology.org/2021.emnlp-main.499/)** - *Laiba Mehnaz et al. (2021)*.
  
### Creative & Code Generation
*Exploring structural mixing in programming and creative writing.*

* **[CodeMixBench: Evaluating Large Language Models on Code Generation with Code-Mixed Prompts](https://arxiv.org/abs/2505.05063)** - *Zhen Yang et al. (2025)*. 
* **[COCOA: An Encoder-Decoder Model for Controllable Code-switched Generation](https://aclanthology.org/2022.emnlp-main.158.pdf)** - *Sneha Mondal et al. (2022)*. 
* **[Can You Translate for Me? Code-Switched Machine Translation with Large Language Models](https://aclanthology.org/2023.ijcnlp-short.10/)** - *Jyotsana Khatri et al. (2023)*. 
## 2\. Datasets & Resources

> Corpora, toolkits, and frameworks to support your research.

### Datasets

| **Name** | **Description** | **Lang Pair** | **Type/Task** | **Link** |
|:---|:---|:---|:---|:---:|
| **AfroCS-xs** | High-quality human-validated synthetic data. | 4 African-En | Machine Translation | [Link](https://aclanthology.org/2025.acl-long.1601/) |
| **ASCEND** | 10.6h spontaneous conversational speech. | Mandarin-En | ASR/Dialogue | [Link](https://arxiv.org/abs/2112.06223) |
| **BanglishRev** | 23K Bangla-English reviews for sentiment. | Bengali-En | Sentiment | [Link](https://arxiv.org/abs/2501.00000) |
| **COMI-LINGUA** | Expert annotated large-scale dataset. | Hindi-En | Multitask NLU | [Link](https://aclanthology.org/2025.findings-emnlp.422/) |
| **HiACC** | Hinglish adult & children code-switched corpus. | Hindi-En | Speech/Text | [Link](https://doi.org/10.1016/j.dib.2025.111886) |
| **MMS-5** | Multi-scenario multimodal hate speech. | Tamil/Kan-En | MM Hate Speech | [Link](https://arxiv.org/pdf/2310.07423) |
| **MultiCoNER** | Large-scale benchmark for complex NER. | 11 Langs | NER | [Link](https://aclanthology.org/2022.coling-1.334/) |
| **My Boli** | Corpora & Pre-trained Models for Marathi-English. | Marathi-En | NLU | [Link](https://aclanthology.org/2023.eacl-main.249) |
| **SwitchLingua** | Massive multi-ethnic code-switching dataset. | 83 Langs | General NLU | [Link](https://arxiv.org/abs/2506.00087) |
| **ToxVidLM** | Framework & dataset for toxicity in code-mixed videos. | Mixed | Video Toxicity | [Link](https://aclanthology.org/2024.findings-acl.663/) |


  * **[Language Modeling for Code-Mixing: The Role of Linguistic Theory based Synthetic Data](https://aclanthology.org/P18-1143.pdf)** - *Adithya Pratapa, et al. (2018)*.
  * **[Uncovering Code-Mixed Challenges: A Framework for Linguistically Driven Question Generation and Neural Based Question Answering](https://aclanthology.org/K18-1012/)** - *Deepak Gupta, et al. (2018)*.
  * **[Dependency Parser for Bengali-English Code-Mixed Data enhanced with a Synthetic Treebank](https://aclanthology.org/W19-7810/)** - *Upendra Kumar, et al. (2019)*.
  * **[Dependency Parsing for English–Malayalam Code-mixed Text](https://aclanthology.org/K19-1026/)** - *Sanket Sonu, et al. (2019)*.
  * **[A New Dataset for Natural Language Inference from Code-mixed Conversations](https://aclanthology.org/2020.calcs-1.2.pdf)** - *Simran Khanuja, et al. (2020)*.
  * **[Detecting Entailment in Code-Mixed Hindi-English Conversations](https://aclanthology.org/2020.wnut-1.22/)** - *Sharanya Chakravarthy, et al. (2020)*.
  * **[GupShup: Summarizing Open-Domain Code-Switched Conversations](https://aclanthology.org/2021.emnlp-main.499.pdf)** - *Laiba Mehnaz, et al. (2021)*.
  * **[CoMeT: Towards Code-Mixed Translation Using Parallel Monolingual Sentences](https://aclanthology.org/2021.calcs-1.7.pdf)** - *Devansh Gautam, et al. (2021)*.
  * **[Exploring Language Identification from Short Multilingual Code-Switched Texts](https://aclanthology.org/2022.paclic-1.7.pdf)** - *Pei-Chi Lo, et al. (2022)*.
  * **[A Comparison of Architectures and Pretraining Methods for Contextualized Multilingual Word Embeddings](https://arxiv.org/abs/2204.08398)** - *Milana Karaica, et al. (2022)*.
  * **[Code-MixPro: A Framework for Code-Mixed Data Augmentation via Prompt Tuning](https://aclanthology.org/2023.ranlp-1.108.pdf)** - *Rohit Kundu, et al. (2023)*.
  * **[OffMix-3L: A Novel Code-Mixed Test Dataset in Bangla-English-Hindi for Offensive Language Identification](https://aclanthology.org/2023.socialnlp-1.3/)** - *Goswami, D., et al. (2023)*.
  * **[My Boli: A Comprehensive Suite of Corpora and Pre-trained Models for Marathi-English Code-Mixing](https://aclanthology.org/2023.eacl-main.249)** - *Joshi, A., et al. (2023)*.
  * **[Sentiment Analysis in Code-Mixed Telugu-English Text with Multi-task Learning](https://aclanthology.org/2024.wassa-1.19.pdf)** - *Siva Sai, et al. (2024)*.
  * **[Multilingual Harmful Meme Detection Using Large Language Models](https://aclanthology.org/2024.woah-1.3.pdf)** - *Sanchit Ahuja, et al. (2024)*.
  * **[Aligning Speech to Languages to Enhance Code-switching Speech Recognition](https://arxiv.org/pdf/2403.05887)** - *Hexin Liu, et al. (2024)*.
  * **[HiACC: Hinglish adult & children code-switched corpus](https://doi.org/10.1016/j.dib.2025.111886)** - *Singh, S., et al. (2025)*.
  * **[AfroCS-xs: Creating a Compact, High-Quality, Human-Validated Code-Switched Dataset for African Languages](https://aclanthology.org/2025.acl-long.1601/)** - *Olaleye, K., et al. (2025)*.
  * **[COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing](https://aclanthology.org/2025.findings-emnlp.422.pdf)** - *Rajvee Sheth, et al. (2025)*.

### Frameworks & Toolkits

  * **[CoSSAT: Code-Switched Speech Annotation Tool](https://aclanthology.org/D19-5907/)** - *Shah, S., et al. (2019)*.
  * **[A Unified Framework for Multilingual and Code-Mixed Visual Question Answering](https://aclanthology.org/2020.aacl-main.90/)** - *Deepak Gupta, et al. (2020)*.
  * **[CodemixedNLP: An Extensible and Open NLP Toolkit for Code-Mixing](https://aclanthology.org/2021.calcs-1.14/)** - *Jayanthi, S. M., et al. (2021)*.
  * **[GCM: A Toolkit for Generating Synthetic Code-mixed Text](https://aclanthology.org/2021.eacl-demos.24/)** - *Rizvi, M. S. Z., et al. (2021)*.
  * **[Commentator: A Code-mixed Multilingual Text Annotation Framework](https://aclanthology.org/2024.emnlp-demo.11)** - *Sheth, R., et al. (2024)*.
  * **[ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos](https://aclanthology.org/2024.findings-acl.663.pdf)** - *Krishanu Maity, et al. (2024)*.
  * **[CHAI for LLMs: Improving Code-Mixed Translation in Large Language Models through Reinforcement Learning with AI Feedback](https://arxiv.org/abs/2411.09073)** - *Wenbo Zhang (2024)*.

-----

## 3\. Model Training & Adaptation

> Techniques for building and adapting models to understand and generate code-mixed language.

### Pre-training Approaches

  * **[Modeling Code-Switch Languages Using Bilingual Parallel Corpus](https://aclanthology.org/2020.acl-main.80.pdf)** - *Grandee Lee, et al. (2020)*.
  * **[SJ AJ@DravidianLangTech-EACL2021: Task-Adaptive Pre-Training of Multilingual BERT models for Offensive Language Identification](https://aclanthology.org/2021.dravidianlangtech-1.44.pdf)** - *Sai Muralidhar Jayanthi, et al. (2021)*.
  * **[Switch Point biased Self-Training: Re-purposing Pretrained Models for Code-Switching](https://aclanthology.org/2021.findings-emnlp.373.pdf)** - *Parul Chopra, et al. (2021)*.
  * **[Unsupervised Self-Training for Sentiment Analysis of Code-Switched Data](https://aclanthology.org/2021.calcs-1.13.pdf)** - *Akshat Gupta, et al. (2021)*.
  * **[Task-Specific Pre-Training and Cross Lingual Transfer for Code-Switched Data](https://aclanthology.org/2021.dravidianlangtech-1.9.pdf)** - *Akshat Gupta, et al. (2021)*.
  * **[BERTologiCoMix: How does Code-Mixing interact with Multilingual BERT?](https://aclanthology.org/2021.adaptnlp-1.12/)** - *Santy, S., et al. (2021)*.
  * **[HingBERT: A Code Mixed Hindi-English Dataset and BERT Language Models](https://aclanthology.org/2022.wildre-1.2/)** - *Nayak, R. & Joshi, R. (2022)*.
  * **[L3Cube-HingCorpus and HingBERT: A Code Mixed Hindi-English Dataset and BERT Model for Language Identification](https://aclanthology.org/anthology-files/anthology-files/pdf/wildre/2022.wildre-1.pdf#page=19)** - *Raviraj Joshi, et al. (2022)*.
  * **[MALM: Mixing Augmented Language Modeling for Zero-Shot Machine Translation](https://aclanthology.org/2022.nlp4dh-1.8.pdf)** - *Kshitij Gupta (2022)*.
  * **[Transfer Learning for Code-Mixed Data: Do Pretraining Languages Matter?](https://aclanthology.org/2023.wassa-1.32.pdf)** - *Kushal Tatariya, et al. (2023)*.
  * **[Improving Pretraining Techniques for Code-Switched NLP](https://aclanthology.org/2023.acl-long.66.pdf)** - *Richeek Das, et al. (2023)*.
  * **[Exploring Enhanced Code-Switched Noising for Pretraining in Neural Machine Translation](https://aclanthology.org/2023.findings-eacl.72.pdf)** - *Vivek Iyer, et al. (2023)*.
  * **[Investigating and Scaling up Code-Switching for Multilingual Language Model Pre-Training](https://aclanthology.org/2025.findings-acl.575.pdf)** - *Zhijun Wang, et al. (2025)*.
  * **[Breaking the Language Barrier: Can One Language Model Understand All Languages?](https://aclanthology.org/2025.unlp-1.1.pdf)** - *Sanchit Ahuja, et al. (2025)*.

### Fine-tuning Approaches

  * **[From English to Code-Switching: Transfer Learning with Strong Morphological Clues](https://aclanthology.org/2020.acl-main.716.pdf)** - *Gustavo Aguilar, et al. (2020)*.
  * **[FiSSA at SemEval-2020 Task 9: Fine-tuned for Feelings](https://aclanthology.org/2020.semeval-1.165/)** - *Bertelt Braaksma, et al. (2020)*.
  * **[A Semi-supervised Approach to Generate the Code-Mixed Text using Pre-trained Encoder and Transfer Learning](https://aclanthology.org/2020.findings-emnlp.206.pdf)** - *Deepak Gupta, et al. (2020)*.
  * **[A Pre-trained Transformer and CNN model with Joint Language ID and Part-of-Speech Tagging for Code-Mixed Social-Media Text](https://aclanthology.org/2021.ranlp-1.42.pdf)** - *Suman Dowlagar, et al. (2021)*.
  * **[The Effectiveness of Intermediate-Task Training for Code-Switched Natural Language Understanding](https://aclanthology.org/2021.mrl-1.16.pdf)** - *Archiki Prasad, et al. (2021)*.
  * **[Saliency-based Multi-View Mixed Language Training for Zero-shot Cross-lingual Classification](https://aclanthology.org/2021.findings-emnlp.55.pdf)** - *Siyu Lai, et al. (2021)*.
  * **[On Utilizing Constituent Language Resources to Improve Downstream Tasks in Hinglish](https://aclanthology.org/2022.findings-emnlp.283.pdf)** - *Vishwajeet Kumar, et al. (2022)*.
  * **[Adapting Multilingual Models for Code-Mixed Translation](https://aclanthology.org/2022.findings-emnlp.528.pdf)** - *Aditya Vavre, et al. (2022)*.
  * **[PRO-CS : An Instance-Based Prompt Composition Technique for Code-Switched Tasks](https://aclanthology.org/2022.emnlp-main.698.pdf)** - *Srijan Bansal, et al. (2022)*.
  * **[Progressive Sentiment Analysis for Code-Switched Text Data](https://aclanthology.org/2022.findings-emnlp.82.pdf)** - *Sudhanshu Ranjan, et al. (2022)*.
  * **[ENTITYCS: Improving Zero-Shot Cross-lingual Transfer with Entity-Centric Code Switching](https://aclanthology.org/2022.findings-emnlp.499.pdf)** - *Chenxi Whitehouse, et al. (2022)*.
  * **[COCOA: An Encoder-Decoder Model for Controllable Code-switched Generation](https://aclanthology.org/2022.emnlp-main.158.pdf)** - *Sneha Mondal, et al. (2022)*.
  * **[Transfer Learning for Code-Mixed Data: Do Pretraining Languages Matter?](https://aclanthology.org/2023.wassa-1.32.pdf)** - *Kushal Tatariya, et al. (2023)*.
  * **[From Translation to Generative LLMs: Classification of Code-Mixed Affective Tasks](https://ieeexplore.ieee.org/abstract/document/10938193)** - *Anjali Yadav, et al. (2024)*.
  * **[SetFit: A Robust Approach for Offensive Content Detection in Tamil-English Code-Mixed Conversations Using Sentence Transfer Fine-tuning](https://aclanthology.org/2024.dravidianlangtech-1.6.pdf)** - *Kathiravan Pannerselvam, et al. (2024)*.
  * **[Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation](https://aclanthology.org/2024.lrec-main.1345.pdf)** - *Kartik, et al. (2024)*.
  * **[COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning](https://aclanthology.org/2024.findings-naacl.198/)** - *Lee, J., et al. (2024)*.
  * **[Demystifying Instruction Mixing for Fine-tuning Large Language Models](https://arxiv.org/abs/2312.10793)** - *Wang, R., et al. (2024)*.
  * **[CHAI for LLMs: Improving Code-Mixed Translation in LLMs through Reinforcement Learning with AI Feedback](https://arxiv.org/abs/2411.09073)** - *Zhang, W., et al. (2025)*.
  * **[LLMsAgainstHate@NLU of Devanagari Script Languages 2025: Hate Speech Detection and Target Identification in Devanagari Languages via Parameter Efficient Fine-Tuning of LLMs](https://aclanthology.org/2025.chipsal-1.34/)** - *Rushendra Sidibomma, et al. (2025)*.
  * **[Controlling Language Confusion in Multilingual LLMs](https://aclanthology.org/2025.acl-srw.81/)** - *Nahyun Lee, et al. (2025)*.
  * **[Fine-Tuning Cross-Lingual LLMs for POS Tagging in Code-Switched Contexts](https://aclanthology.org/2025.resourceful-1.2/)** - *Shayaan Absar (2025)*.
  * **[Code-Switching Curriculum Learning for Multilingual Transfer in LLMs](https://aclanthology.org/2025.findings-acl.407/)** - *Haneul Yoo, et al. (2025)*.
  * **[MIGRATE: Cross-Lingual Adaptation of Domain-Specific LLMs through Code-Switching and Embedding Transfer](https://aclanthology.org/2025.coling-main.617.pdf)** - *Seongtae Hong, et al. (2025)*.
  * **[Next-Level Cantonese-to-Mandarin Translation: Fine-Tuning and Post-Processing with LLMs](https://aclanthology.org/2025.loreslm-1.32.pdf)** - *Yuqian Dai, et al. (2025)*.
  * **[Investigating and Scaling up Code-Switching for Multilingual Language Model Pre-Training](https://aclanthology.org/2025.findings-acl.575.pdf)** - *Zhijun Wang, et al. (2025)*.
  * **[Beyond Monolingual Limits: Fine-Tuning Monolingual ASR for Yoruba-English Code-Switching](https://aclanthology.org/2025.calcs-1.3.pdf)** - *Oreoluwa Babatunde, et al. (2025)*.
  * **[Tongue-Tied: Breaking LLMs Safety Through New Language Learning](https://aclanthology.org/2025.calcs-1.5.pdf)** - *Bibek Upadhayay, et al. (2025)*.
  * **[Identifying Aggression and Offensive Language in Code-Mixed Tweets: A Multi-Task Transfer Learning Approach](https://aclanthology.org/2025.indonlp-1.14.pdf)** - *Bharath Kancharla, et al. (2025)*.
  * **[Multi-task detection of harmful content in code-mixed meme captions using large language models with zero-shot, few-shot, and fine-tuning approaches](https://www.sciencedirect.com/science/article/pii/S1110866525000763)** - *Bharath Kancharla, et al. (2025)*.
  * **[Adapting Multilingual Models to Code-Mixed Tasks via Model Merging](https://aclanthology.org/2025.calcs-1.16.pdf)** - *Sanchit Ahuja, et al. (2025)*.

### Post-training Approaches

  * **[Saliency-based Multi-View Mixed Language Training for Zero-shot Cross-lingual Classification](https://aclanthology.org/2021.findings-emnlp.55.pdf)** - *Siyu Lai, et al. (2021)*.
  * **[Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling](https://aclanthology.org/2021.mrl-1.18.pdf)** - *Jitin Krishnan, et al. (2021)*.
  * **[PRO-CS : An Instance-Based Prompt Composition Technique for Code-Switched Tasks](https://aclanthology.org/2022.emnlp-main.698/)** - *Bansal, S., et al. (2022)*.
  * **[ENTITY CS: Improving Zero-Shot Cross-lingual Transfer with Entity-Centric Code Switching](https://aclanthology.org/2022.findings-emnlp.499.pdf)** - *Chenxi Whitehouse, et al. (2022)*.
  * **[MulZDG: Multilingual Code-Switching Framework for Zero-shot Dialogue Generation](https://aclanthology.org/2022.coling-1.54.pdf)** - *Yongkang Liu, et al. (2022)*.
  * **[MALM: Mixing Augmented Language Modeling for Zero-Shot Machine Translation](https://aclanthology.org/2022.nlp4dh-1.8.pdf)** - *Kshitij Gupta (2022)*.
  * **[Multilingual Large Language Models Are Not (Yet) Code-Switchers](https://aclanthology.org/2023.emnlp-main.774.pdf)** - *Ruochen Zhang, et al. (2023)*.
  * **[Transfer Learning for Code-Mixed Data: Do Pretraining Languages Matter?](https://aclanthology.org/2023.wassa-1.32.pdf)** - *Kushal Tatariya, et al. (2023)*.
  * **[Prompting Multilingual Large Language Models to Generate Code-Mixed Texts: The Case of South East Asian Languages](https://aclanthology.org/2023.calcs-1.5.pdf)** - *Zheng-Xin Yong, et al. (2023)*.
  * **[OffMix-3L: A Novel Code-Mixed Test Dataset in Bangla-English-Hindi for Offensive Language Identification](https://aclanthology.org/2023.socialnlp-1.3/)** - *Dhiman Goswami, et al. (2023)*.
  * **[Leveraging Large Language Models for Code-Mixed Data Augmentation in Sentiment Analysis](https://aclanthology.org/2024.sicon-1.6/)** - *Zeng, L. (2024)*.
  * **[In-context Mixing (ICM): Code-mixed Prompts for Multilingual LLMs](https://aclanthology.org/2024.acl-long.228/)** - *Shankar, B., et al. (2024)*.
  * **[From Translation to Generative LLMs: Classification of Code-Mixed Affective Tasks](https://ieeexplore.ieee.org/abstract/document/10938193)** - * Anjali Yadav, et al. (2024)*.
  * **[COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing](https://arxiv.org/pdf/2503.21670)** - *Rajvee Sheth, et al. (2025)*.
  * **[DweshVaani: An LLM for Detecting Religious Hate Speech in Code-Mixed Hindi-English](https://aclanthology.org/2025.chipsal-1.5.pdf)** - *Varad Srivastava (2025)*.
  * **[Multi-task detection of harmful content in code-mixed meme captions using large language models with zero-shot, few-shot, and fine-tuning approaches](https://www.sciencedirect.com/science/article/pii/S1110866525000763)** - *Bharath Kancharla, et al. (2025)*.
  * **[An Adapted Few-Shot Prompting Technique Using ChatGPT to Advance Low-Resource Languages Understanding](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11016028)** - *Yash Raj Sarrof, et al. (2025)*.

-----

## 4\. Evaluation & Benchmarking

> Resources for evaluating model performance on code-switching tasks.
> 
### 📊 Benchmark Comparison

*A comparison of major evaluation suites for Code-Switching, categorized by data origin and evaluation focus.*

| **Benchmark** | **Task Scope** | **Data Origin** | **Eval Focus** | **Link** |
|:---|:---|:---:|:---|:---:|
| **CodeMixBench**| Code Generation (Python) | 🧑‍💻 **Human** | Syntax & Executability | [🔗](https://arxiv.org/abs/2505.05063) |
| **CS-Sum** | Dialogue, Summarization | 🧑‍💻 **Human** | General LLM Capability | [🔗](https://www.researchgate.net/publication/391911130_CS-Sum_A_Benchmark_for_Code-Switching_Dialogue_Summarization_and_the_Limits_of_Large_Language_Models) |
| **GLUECoS** | QA, NLI, Sentiment | 🧑‍💻 **Human** | NLU Performance | [🔗](https://aclanthology.org/2020.acl-main.329/) |
| **LinCE** | LID, NER, POS, Sentiment | 🧑‍💻 **Human** | Linguistic Accuracy (F1) | [🔗](https://aclanthology.org/2020.lrec-1.223/) |
| **MEGAVERSE** | Multimodal QA | ⚡ **Hybrid** | Factuality & Robustness | [🔗](https://aclanthology.org/2024.naacl-long.143/) |
| **SwitchLingua** | Multitask NLU (83 Langs) | ⚡ **Hybrid** | Scale & Diversity | [🔗](https://arxiv.org/abs/2506.00087) |

*(**Legend:** 🧑‍💻 **Human** = Manually annotated/curated; 🤖 **LLM-Synth** = Generated by Large Language Models; ⚡ **Hybrid** = Mixed sources or Human-filtered Synthetic data.)*

### Benchmarks

* **[LinCE: A centralized benchmark for linguistic code-switching evaluation](https://aclanthology.org/2020.lrec-1.223/)** – Aguilar et al. (2020)
* **[GLUECoS: An Evaluation Benchmark for Code-Switched NLP](https://aclanthology.org/2020.acl-main.329/)** – Khanuja et al. (2020)
* **[PACMAN: Parallel Code-Mixed Data Generation for POS Tagging](https://aclanthology.org/2022.icon-main.29/)** – Chatterjee et al. (2022)
* **[MultiCoNER: A Large-scale Multilingual Dataset for Complex NER](https://aclanthology.org/2022.coling-1.334/)** – Malmasi et al. (2022)
* **[X-RiSAWOZ: High-Quality Multilingual Dialogue Datasets](https://aclanthology.org/2023.findings-acl.174/)** – Moradshahi et al. (2023)
* **[CS-Sum: A Benchmark for Code-Switching Dialogue Summarization and the Limits of Large Language Models](https://www.researchgate.net/publication/391911130_CS-Sum_A_Benchmark_for_Code-Switching_Dialogue_Summarization_and_the_Limits_of_Large_Language_Models)** – Krishnan et al. (2025)
* **[CroCoSum: Cross-Lingual Code-Switched Summarization Benchmark](https://aclanthology.org/2024.lrec-main.367.pdf)** – Zhang et al. (2024)
* **[MEGAVERSE: Benchmarking LLMs Across Languages and Tasks](https://aclanthology.org/2024.naacl-long.143/)** – Ahuja et al. (2024)
* **[COMI-LINGUA: Hindi–English Code-Mixed Multitask Dataset](https://aclanthology.org/2025.findings-emnlp.422.pdf)** – Sheth et al. (2025)
* **[CodeMixBench: Code Generation with Code-Mixed Prompts](https://arxiv.org/abs/2505.05063)** – Sawant (2025)
* **[SwitchLingua: Large-Scale Multilingual Code-Switching Dataset](https://arxiv.org/abs/2506.00087)** – Xie (2025)

  
### Evaluation Metrics

  * **[Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)** - *Papineni, K., et al. (2002)*.
  * **[chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/)** - *Popović, M. (2015)*.
  * **[Code-Mixing in Social Media Text](https://aclanthology.org/2013.tal-3.3/)** - *Amitava Das, et al. (2013)*.
  * **[Comparing the Level of Code-Switching in Corpora](https://aclanthology.org/L16-1292/)** - *Björn Gambäck, et al. (2016)*.
  * **[Automatic Detection of Code-switching Style from Acoustics](https://aclanthology.org/W18-3209.pdf)** - *SaiKrishna Rallabandi, et al. (2018)*.
  * **[Detecting de minimis Code-Switching in Historical German Books](https://aclanthology.org/2020.coling-main.163.pdf)** - *Shijia Liu, et al. (2020)*.
  * **[Challenges and Limitations with the Metrics Measuring the Complexity of Code-Mixed Text](https://aclanthology.org/2021.calcs-1.2/)** - *Vivek Srivastava, et al. (2021)*.
  * **[SyMCoM - Syntactic Measure of Code Mixing A Study Of English-Hindi Code-Mixing](https://aclanthology.org/2022.findings-acl.40/)** - *Prashant Kodali, et al. (2022)*.
  * **[PreCogIIITH at HinglishEval: Leveraging Code-Mixing Metrics & Language Model Embeddings To Estimate Code-Mix Quality](https://aclanthology.org/2022.inlg-genchal.4/)** - *Prashant Kodali, et al. (2022)*.
  * **[Code-Switching Metrics Using Intonation Units](https://aclanthology.org/2023.emnlp-main.1047/)** - *Rebecca Pattichis, et al. (2023)*.
  * **[Minimal Pair-Based Evaluation of Code-Switching](https://aclanthology.org/2025.acl-long.910/)** - *Sterner, I. & Teufel, S. (2025)*.
  * **[PIER: A Novel Metric for Evaluating What Matters in Code-Switching](https://arxiv.org/abs/2501.09512)** - *Ugan, E. Y., et al. (2025)*.
  * **[Code-Mixer Ya Nahi: Novel Approaches to Measuring Multilingual LLMs' Code-Mixing Capabilities](https://arxiv.org/abs/2501.09512)** - *Joshi, R., et al. (2025)*.

-----

## 5\. Multi & Cross-Modal Applications

> Applying code-switching NLP to speech, vision, and other modalities.

### Speech Processing
* **ASR** * **[Dependency Parsing for English–Malayalam Code-mixed Text](https://aclanthology.org/K19-1026.pdf)** - *Sanket Sonu, et al. (2019)*.
  * **[Semi-supervised Acoustic and Language Model Training for English-isiZulu Code-Switched Speech Recognition](https://aclanthology.org/2020.calcs-1.7.pdf)** - *Astik Biswas, et al. (2020)*.
  * **[Improving code-switched ASR with linguistic information](https://aclanthology.org/2022.coling-1.627.pdf)** - *Jie Chi, et al. (2022)*.
  * **[End-to-End Speech Translation for Code Switched Speech](https://aclanthology.org/2022.findings-acl.113.pdf)** - *Orion Weller, et al. (2022)*.
  * **[Representativeness as a Forgotten Lesson for Multilingual and Code-switched Data Collection and Preparation](https://aclanthology.org/2023.findings-emnlp.382.pdf)** - *A. Seza Doğruöz, et al. (2023)*.
  * **[New Datasets and Controllable Iterative Data Augmentation Method for Code-switching ASR Error Correction](https://aclanthology.org/2023.findings-emnlp.543.pdf)** - *Zhaohong Wan, et al. (2023)*.
  * **[Code-Mixed Text Augmentation for Latvian ASR](https://aclanthology.org/2024.lrec-main.308.pdf)** - *Martins Kronis, et al. (2024)*.
  * **[The Impact of Code-switched Synthetic Data Quality is Task Dependent: Insights from MT and ASR](https://aclanthology.org/2025.calcs-1.2.pdf)** - *Injy Hamed, et al. (2025)*.
  * **[Development of a code-switched Hindi-Marathi dataset and transformer-based architecture for enhanced speech recognition using dynamic switching algorithms](https://www.sciencedirect.com/science/article/abs/pii/S0003682X24005590)** - *Palash Jain, et al. (2025)*.
  * **[ENHANCING ASR ACCURACY AND COHERENCE ACROSS INDIAN LANGUAGES WITH WAV2VEC2 AND GPT - 2](https://ictactjournals.in/paper/IJDSML_Vol_6_Iss_2_Paper_4_761_764.pdf)** - *R. Geetha Rajakumari, et al. (2025)*.
  * **[Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10890030)** - *Yu Xi, et al. (2024)*.
  * **[Adapting Whisper for Low-Resource Hindi-English Code-Mix Speech](https://aclanthology.org/2025.calcs-1.12.pdf)** - *Sakshi Koli, et al. (2025)*.
* **Speech Translation**

  * **[Towards Developing a Multilingual and Code-Mixed Visual Question Answering System by Knowledge Distillation](https://aclanthology.org/2021.findings-emnlp.151.pdf)** - *Humair Raj Khan, et al. (2021)*.
  * **[End-to-End Speech Translation for Code Switched Speech](https://aclanthology.org/2022.findings-acl.113/)** - *Weller, O., et al. (2022)*.
  * **[CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units](https://aclanthology.org/2024.acl-srw.40/)** - *Kang, Y. (2024)*.
  * **[Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10890030&tag=1)** - *Yu Xi, et al. (2024)*.
  * **[The Impact of Code-switched Synthetic Data Quality is Task Dependent: Insights from MT and ASR](https://aclanthology.org/2025.calcs-1.2.pdf)** - *Injy Hamed, et al. (2025)*.
  * **[Code-Switching and Syntax: A Large–Scale Experiment](https://aclanthology.org/2025.findings-acl.600.pdf)** - *Igor Sterner, et al. (2025)*.
  * **[Development of a code-switched Hindi-Marathi dataset and transformer-based architecture for enhanced speech recognition using dynamic switching algorithms](https://www.sciencedirect.com/science/article/pii/S0003682X24005590)** - *P. Hemant, et al. (2025)*.
  * **[ENHANCING ASR ACCURACY AND COHERENCE ACROSS INDIAN LANGUAGES WITH WAV2VEC2 AND GPT - 2](https://ictactjournals.in/paper/IJDSML_Vol_6_Iss_2_Paper_4_761_764.pdf)** - *R. Geetha Rajakumari, et al. (2025)*.
     
### Vision-Language & Document Processing

  * **[A Unified Framework for Multilingual and Code-Mixed Visual Question Answering](https://aclanthology.org/2020.aacl-main.90/)** - *Deepak Gupta, et al. (2020)*.
  * **[Towards Developing a Multilingual and Code-Mixed Visual Question Answering System by Knowledge Distillation](https://aclanthology.org/2021.findings-emnlp.151/)** - *Raj Khan, H., et al. (2021)*.
  * **["To Have the 'Million' Readers Yet": Building a Digitally Enhanced Edition of the Bilingual Irish-English Newspaper](https://aclanthology.org/2024.lt4hala-1.9/)** - *Dereza, O., et al. (2024)*.
  * **[MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks](https://aclanthology.org/2024.naacl-long.143.pdf)** - *Sanchit Ahuja, et al. (2024)*.
  * **[ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos](https://aclanthology.org/2024.findings-acl.663.pdf)** - *Krishanu Maity, et al. (2024)*.
  * **[Multi-task detection of harmful content in code-mixed meme captions using large language models with zero-shot, few-shot, and fine-tuning approaches](https://www.sciencedirect.com/science/article/pii/S1110866525000763)** - *Bharath Kancharla, et al. (2025)*.
  * **[BanglAssist: A Bengali-English Generative AI Chatbot for Code-Switching and Dialect-Handling in Customer Service](https://arxiv.org/abs/2503.22283)** - *Francesco Kruk (2025)*.
  * **[Qorǵau: Evaluating Safety in Kazakh-Russian Bilingual Contexts](https://aclanthology.org/2025.findings-acl.507/)** - *Maiya Goloburda, et al. (2025)*.
  * **[Enhancing Participatory Development Research in South Asia through LLM Agents System: An Empirically-Grounded Methodological Initiative from Field Evidence in Sri Lankan](https://aclanthology.org/2025.indonlp-1.13/)** - *Xinjie Zhao, et al. (2025)*.

### Cross-Modal Integration

  * **[Code-Switched Language Models Using Neural Based Synthetic Data from Parallel Sentences](https://aclanthology.org/K19-1026/)** - *Genta Indra Winata, et al. (2019)*.
  * **[Translate and Classify: Improving Sequence Level Classification for English-Hindi Code-Mixed Data](https://aclanthology.org/2021.calcs-1.3/)** - *Devansh Gautam, et al. (2021)*.
  * **[Data Augmentation to Address Out of Vocabulary Problem in Low Resource Sinhala English Neural Machine Translation](https://aclanthology.org/2021.paclic-1.7/)** - *Aloka Fernando, et al. (2021)*.
  * **[CI-AVSR: A Cantonese Audio-Visual Speech Dataset for In-car Command Recognition](https://aclanthology.org/2022.lrec-1.731/)** - *Dai, W., et al. (2022)*.
  * **[Typo-Robust Representation Learning for Dense Retrieval](https://aclanthology.org/2023.acl-short.95/)** - *Panuthep Tasawong, et al. (2023)*.
  * **[Advancing Multi-Criteria Chinese Word Segmentation Through Criterion Classification and Denoising](https://aclanthology.org/2023.acl-long.356/)** - *Tzu Hsuan Chou, et al. (2023)*.
  * **[ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos](https://aclanthology.org/2024.findings-acl.663/)** - *Maity, K., et al. (2024)*.
  * **[Machine Translation and Transliteration for Indo-Aryan Languages: A Systematic Review](https://aclanthology.org/2025.indonlp-1.2/)** - *Sandun Sameera Perera, et al. (2025)*.
-----


## Workshops & Shared Tasks

> A list of academic workshops and community shared tasks dedicated to code-switching.

  * [**CALCS 2018:** Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/volumes/W18-32/).
  * [**CALCS 2020:** Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/2020.calcs-1.0/).
  * [**CALCS 2021:** Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/events/calcs-2021/).
  * [**WILDRE-6 2022:** Workshop within the 13th Language Resources and Evaluation Conference](https://aclanthology.org/volumes/2022.wildre-1/).
  * [**ICON 2022:** 19th International Conference on Natural Language Processing (ICON)](https://aclanthology.org/volumes/2022.icon-wlli/).
  * [**CALCS 2023:** 6th Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/events/calcs-2023/).
  * [**CALCS 2025:** 7th Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/volumes/2025.calcs-1/).

-----

## Contributing

Your contributions are always welcome and make this community resource better\!

If you have a paper, dataset, or tool you'd like to add:

1.  Fork the repository.
2.  Add your resource to the relevant section.
3.  Please try to follow the existing format and include a direct link.
4.  Submit a pull request\!
