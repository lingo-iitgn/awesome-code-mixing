

# Awesome Code-Mixing & Code-Switching

<p align="center">
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://www.google.com/search?q=%23contributing">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
  </a>
</p>

> A curated list of awesome papers, datasets, and toolkits for **Code-Switching** & **Code-Mixing** in Natural Language Processing.

## Table of Contents

*Click on any link to jump to the corresponding section on this page.*
* [Survey Papers](#survey-papers)
* [1. NLP Tasks](#1-nlp-tasks)
  * [1.1. Natural Language Understanding (NLU) Tasks](#11-natural-language-understanding-nlu-tasks)
      * [Language Identification (LID)](#language-identification-lid)
      * [Part-of-Speech (POS) Tagging](#part-of-speech-pos-tagging)
      * [Named Entity Recognition (NER)](#named-entity-recognition-ner)
      * [Sentiment & Emotion Analysis](#sentiment--emotion-analysis)
      * [Syntactic Analysis](#syntactic-analysis)
      * [Intent Classification](#intent-classification)
      * [Question Answering (QA)](#question-answering-qa)
      * [Natural Language Inference (NLI)](#natural-language-inference-nli)
  * [1.2. Natural Language Generation (NLG) Tasks](#12-natural-language-generation-nlg-tasks)
      * [Code-Mixed Text Generation](#code-mixed-text-generation)
      * [Machine Translation (MT)](#machine-translation-mt)
      * [Cross-lingual Transfer](#cross-lingual-transfer)
      * [Text Summarization](#text-summarization)
      * [Dialogue Generation](#dialogue-generation)
      * [Transliteration](#transliteration)
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
  * [Contributing](#contributing)

-----

## Survey Papers

> Comprehensive reviews of the code-switching research landscape. A great place to start.
  * **[The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges](https://aclanthology.org/2023.findings-acl.185/)** - *Winata, G. I., et al. (2023)*.
  * **[Challenges of Computational Processing of Code-Switching](https://aclanthology.org/W16-5801/)** - *Çetinoğlu, Ö., et al. (2016)*.

-----

## 1\. NLP Tasks

## 1.1\. Natural Language Understanding (NLU) Tasks

> Tasks focused on understanding, parsing, and extracting meaning from code-mixed text.

### Language Identification (LID)

  * **[Transformer-based Model for Word Level Language Identification in Code-mixed Kannada-English Texts](https://aclanthology.org/2022.icon-wlli.4/)** - *Lambebo Tonja, A., et al. (2022)*.
  * **[TongueSwitcher: Fine-Grained Identification of German-English Code-Switching](https://aclanthology.org/2023.calcs-1.1/)** - *Sterner, I., & Teufel, S. (2023)*.
  * **[MaskLID: Code-Switching Language Identification through Iterative Masking](https://aclanthology.org/2024.acl-short.43/)** - *Kargaran, A. H., et al. (2024)*.
  * **[Much Gracias: Semi-supervised Code-switch Detection for Spanish-English: How far can we get?](https://aclanthology.org/2021.calcs-1.9/)** - *Iliescu, D.-M., et al. (2021)*.
  * **[Hope Speech Detection in code-mixed Roman Urdu tweets](https://api.semanticscholar.org/CorpusID:280011615)** - *Ahmad, M., et al. (2025)*.
  * **[IRNLP\_DAIICT@LT-EDI-EACL2021: Hope Speech detection in Code Mixed text using TF-IDF Char N-grams and MuRIL](https://aclanthology.org/2021.ltedi-1.15/)** - *Dave, B., et al. (2021)*.
  * **[Word-level Language Identification using CRF: Code-switching Shared Task Report of MSR India System](https://aclanthology.org/W14-3908/)** - *Chittaranjan, G., et al. (2014)*.
  * **[Word Level Language Identification in English Telugu Code Mixed Data](https://aclanthology.org/Y18-1021/)** - *Gundapu, S. & Mamidi, R. (2018)*.
  * **[A fast, compact, accurate model for language identification of codemixed text](https://arxiv.org/abs/1810.04142)** - *Zhang, Y., et al. (2018)*.
  * **[Language identification framework in code-mixed social media text based on quantum LSTM](https://api.semanticscholar.org/CorpusID:214459891)** - *Shekhar, S., et al. (2020)*.

### Part-of-Speech (POS) Tagging

  * **[Fine-Tuning Cross-Lingual LLMs for POS Tagging in Code-Switched Contexts](https://aclanthology.org/2025.resourceful-1.2/)** - *Absar, S. (2025)*.
  * **[A Pre-trained Transformer and CNN Model with Joint Language ID and Part-of-Speech Tagging for Code-Mixed Social-Media Text](https://aclanthology.org/2021.ranlp-1.42/)** - *Dowlagar, S. & Mamidi, R. (2021)*.
  * **[Development of POS tagger for English-Bengali Code-Mixed data](https://aclanthology.org/2019.icon-1.17/)** - *Raha, T., et al. (2019)*.
  * **[POS Tagging of Hindi-English Code Mixed Text from Social Media: Some Machine Learning Experiments](https://aclanthology.org/W15-5936/)** - *Sequiera, R., et al. (2015)*.
  * **[Creation of Corpus and Analysis in Code-Mixed Kannada-English Social Media Data for POS Tagging](https://aclanthology.org/2020.icon-main.13/)** - *Appidi, A. R., et al. (2020)*.
  * **[POS Tagging of English-Hindi Code-Mixed Social Media Content](https://aclanthology.org/D14-1105/)** - *Vyas, Y., et al. (2014)*.

### Named Entity Recognition (NER)

  * **[Cross Script Hindi English NER Corpus from Wikipedia](https://arxiv.org/abs/1810.03430)** - *Ansari, M. Z., et al. (2019)*.
  * **[Named Entity Recognition for Code Mixed Social Media Sentences](https://api.semanticscholar.org/CorpusID:232434202)** - *Sharma, Y., et al. (2021)*.
  * **[Performance analysis of named entity recognition approaches on code-mixed data](https://api.semanticscholar.org/CorpusID:243100435)** - *Gaddamidi, S. & Prasath, R. R. (2021)*.
  * **[Character level neural architectures for boosting named entity recognition in code mixed tweets](https://api.semanticscholar.org/CorpusID:216587955)** - *Narayanan, A., et al. (2020)*.
  * **[CMNEROne at SemEval-2022 Task 11: Code-Mixed Named Entity Recognition by leveraging multilingual data](https://aclanthology.org/2022.semeval-1.214/)** - *Dowlagar, S. & Mamidi, R. (2022)*.
  * **["Kanglish alli names\!" Named Entity Recognition for Kannada-English Code-Mixed Social Media Data](https://aclanthology.org/2022.wnut-1.17/)** - *S, Sumukh & Shrivastava, M. (2022)*.
  * **[GPT-NER: Named Entity Recognition via Large Language Models](https://aclanthology.org/2025.findings-naacl.239/)** - *Wang, S., et al. (2025)*.

### Sentiment & Emotion Analysis

  * **[Multi-Label Emotion Classification on Code-Mixed Text: Data and Methods](https://www.google.com/search?q=https://doi.org/10.1109/ACCESS.2022.3143819)** - *Ameer, I., et al. (2022)*.
  * **[Sarcasm Detection in Dravidian Code-Mixed Text Using Transformer-Based Models](https://www.google.com/search?q=https://citeseerx.ist.psu.edu/document/10.1.1.1092.4862)** - *Bhaumik, A. B. & Das, M. (2023)*.
  * **[SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets](https://aclanthology.org/2020.semeval-1.100/)** - *Patwa, P., et al. (2020)*.
  * **[Towards Sub-Word Level Compositions for Sentiment Analysis of Hindi-English Code Mixed Text](https://aclanthology.org/C16-1234/)** - *Joshi, A., et al. (2016)*.
  * **[Improving Sentiment Analysis for Ukrainian Social Media Code-Switching Data](https://aclanthology.org/2025.unlp-1.18/)** - *Shynkarov, Y., et al. (2025)*.

### Syntactic Analysis

  * **[SyMCoM - Syntactic Measure of Code Mixing A Study Of English-Hindi Code-Mixing](https://aclanthology.org/2022.findings-acl.40/)** - *Kodali, P., et al. (2022)*.
  * **[Dependency Parser for Bengali-English Code-Mixed Data enhanced with a Synthetic Treebank](https://aclanthology.org/W19-7810/)** - *Ghosh, U., et al. (2019)*.
  * **[Improving Code-Switching Dependency Parsing with Semi-Supervised Auxiliary Tasks](https://aclanthology.org/2022.findings-naacl.87/)** - *Özateş, Ş. B., et al. (2022)*.

### Intent Classification

  * **[Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling](https://aclanthology.org/2021.mrl-1.18/)** - *Krishnan, J., et al. (2021)*.
  * **[Regional language code-switching for natural language understanding and intelligent digital assistants](https://doi.org/10.1007/978-981-16-0749-3_71)** - *Rajeshwari, S. & Kallimani, J. S. (2021)*.
  * **[IIT Gandhinagar at SemEval-2020 Task 9: Code-Mixed Sentiment Classification Using Candidate Sentence Generation and Selection](https://aclanthology.org/2020.semeval-1.168/)** - *Srivastava, V. & Singh, M. (2020)*.

### Question Answering (QA)

  * **[Code-Mixed Question Answering Challenge using Deep Learning Methods](https://www.google.com/search?q=https://doi.org/10.1109/ICCES48766.2020.9137971)** - *Thara, S., et al. (2020)*.
  * **[Uncovering Code-Mixed Challenges: A Framework for Linguistically Driven Question Generation and Neural based Question Answering](https://aclanthology.org/K18-1012/)** - *Gupta, D., et al. (2018)*.
  * **[MLQA: Evaluating Cross-lingual Extractive Question Answering](https://aclanthology.org/2020.acl-main.653/)** - *Lewis, P., et al. (2020)*.

### Natural Language Inference (NLI)

  * **[Detecting entailment in code-mixed Hindi-English conversations](https://aclanthology.org/2020.wnut-1.22/)** - *Chakravarthy, S., et al. (2020)*.
  * **[A New Dataset for Natural Language Inference from Code-mixed Conversations](https://aclanthology.org/2020.calcs-1.2/)** - *Khanuja, S., et al. (2020)*.

-----

## 1.2\. Natural Language Generation (NLG) Tasks

> Tasks focused on generating fluent and coherent code-mixed text.

### Code-Mixed Text Generation

  * **[Linguistics Theory Meets LLM: Code-Switched Text Generation via Equivalence Constrained Large Language Models](https://api.semanticscholar.org/CorpusID:273695372)** - *Kuwanto, G., et al. (2024)*.
  * **[A Deep Generative Model for Code Switched Text](https://doi.org/10.24963/ijcai.2019/719)** - *Samanta, B., et al. (2019)*.
  * **[Homophonic Pun Generation in Code Mixed Hindi English](https://aclanthology.org/2025.chum-1.4/)** - *Sarrof, Y. R. (2025)*.

### Machine Translation (MT)

  * **[Towards translating mixed-code comments from social media](https://doi.org/10.1007/978-3-319-77116-8_34)** - *Singh, T. D. & Solorio, T. (2017)*.
  * **[PhraseOut: A Code Mixed Data Augmentation Method for MultilingualNeural Machine Tranlsation](https://aclanthology.org/2020.icon-main.63/)** - *Jasim, B., et al. (2020)*.
  * **[From Machine Translation to Code-Switching: Generating High-Quality Code-Switched Text](https://aclanthology.org/2021.acl-long.245/)** - *Tarunesh, I., et al. (2021)*.
  * **[Neural Machine Translation for Sinhala-English Code-Mixed Text](https://aclanthology.org/2021.ranlp-1.82/)** - *Kugathasan, A. & Sumathipala, S. (2021)*.

### Cross-lingual Transfer

  * **[Improving Zero-Shot Cross-Lingual Transfer via Progressive Code-Switching](https://api.semanticscholar.org/CorpusID:270619569)** - *Li, Z., et al. (2024)*.
  * **[Scopa: Soft code-switching and pairwise alignment for zero-shot cross-lingual transfer](https://dl.acm.org/doi/10.1145/3459637.3482176)** - *Lee, D., et al. (2021)*.
  * **[EntityCS: Improving Zero-Shot Cross-lingual Transfer with Entity-Centric Code Switching](https://aclanthology.org/2022.findings-emnlp.499/)** - *Whitehouse, C., et al. (2022)*.

### Text Summarization

  * **[CroCoSum: A Benchmark Dataset for Cross-Lingual Code-Switched Summarization](https://aclanthology.org/2024.lrec-main.367/)** - *Zhang, R. & Eickhoff, C. (2024)*.
  * **[GupShup: Summarizing Open-Domain Code-Switched Conversations](https://aclanthology.org/2021.emnlp-main.499/)** - *Mehnaz, L., et al. (2021)*.

### Dialogue Generation

  * **[MulZDG: Multilingual Code-Switching Framework for Zero-shot Dialogue Generation](https://aclanthology.org/2022.coling-1.54/)** - *Liu, Y., et al. (2022)*.
  * **[CST5: Data Augmentation for Code-Switched Semantic Parsing](https://aclanthology.org/2023.tllm-1.1/)** - *Agarwal, A., et al. (2023)*.

### Transliteration

  * **[Normalization and Back-Transliteration for Code-Switched Data](https://api.semanticscholar.org/CorpusID:235097478)** - *Parikh, D. & Solorio, T. (2021)*.
  * **[IndicTrans: A Python Library for Indic Language Transliteration](https://aclanthology.org/2022.aacl-demo.9)** - *Anand, T. A. G. & Kumar, J. (2022)*.

-----

## 2\. Datasets & Resources

> Corpora, toolkits, and frameworks to support your research.

### Datasets

  * **[HiACC: Hinglish adult & children code-switched corpus](https://doi.org/10.1016/j.dib.2025.111886)** - *Singh, S., et al. (2025)*.
  * **[AfroCS-xs: Creating a Compact, High-Quality, Human-Validated Code-Switched Dataset for African Languages](https://aclanthology.org/2025.acl-long.1601/)** - *Olaleye, K., et al. (2025)*.
  * **[OffMix-3L: A Novel Code-Mixed Test Dataset in Bangla-English-Hindi for Offensive Language Identification](https://aclanthology.org/2023.socialnlp-1.3/)** - *Goswami, D., et al. (2023)*.
  * **[My Boli: A Comprehensive Suite of Corpora and Pre-trained Models for Marathi-English Code-Mixing](https://aclanthology.org/2023.eacl-main.249)** - *Joshi, A., et al. (2023)*.

### Frameworks & Toolkits

  * **[CoSSAT: Code-Switched Speech Annotation Tool](https://aclanthology.org/D19-5907/)** - *Shah, S., et al. (2019)*.
  * **[Commentator: A Code-mixed Multilingual Text Annotation Framework](https://aclanthology.org/2024.emnlp-demo.11)** - *Sheth, R., et al. (2024)*.
  * **[CodemixedNLP: An Extensible and Open NLP Toolkit for Code-Mixing](https://aclanthology.org/2021.calcs-1.14/)** - *Jayanthi, S. M., et al. (2021)*.
  * **[GCM: A Toolkit for Generating Synthetic Code-mixed Text](https://aclanthology.org/2021.eacl-demos.24/)** - *Rizvi, M. S. Z., et al. (2021)*.

-----

## 3\. Model Training & Adaptation

> Techniques for building and adapting models to understand and generate code-mixed language.

### Pre-training Approaches

  * **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)** - *Devlin, J., et al. (2019)*.
  * **[BERTologiCoMix: How does Code-Mixing interact with Multilingual BERT?](https://aclanthology.org/2021.adaptnlp-1.12/)** - *Santy, S., et al. (2021)*.
  * **[Where Does mBERT Understand Code-Mixing? Layer-Dependent Performance on Semantic Tasks](https://www.google.com/search?q=https://doi.org/10.1109/ACCESS.2025.3594135)** - *Somani, A., et al. (2025)*.
  * **[HingBERT: A Code Mixed Hindi-English Dataset and BERT Language Models](https://aclanthology.org/2022.wildre-1.2/)** - *Nayak, R. & Joshi, R. (2022)*.

### Fine-tuning Approaches

  * **[COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning](https://aclanthology.org/2024.findings-naacl.198/)** - *Lee, J., et al. (2024)*.
  * **[Demystifying Instruction Mixing for Fine-tuning Large Language Models](https://arxiv.org/abs/2312.10793)** - *Wang, R., et al. (2024)*.
  * **[CHAI for LLMs: Improving Code-Mixed Translation in LLMs through Reinforcement Learning with AI Feedback](https://arxiv.org/abs/2411.09073)** - *Zhang, W., et al. (2025)*.

### Post-training Approaches

  * **[In-context Mixing (ICM): Code-mixed Prompts for Multilingual LLMs](https://aclanthology.org/2024.acl-long.228/)** - *Shankar, B., et al. (2024)*.
  * **[Leveraging Large Language Models for Code-Mixed Data Augmentation in Sentiment Analysis](https://aclanthology.org/2024.sicon-1.6/)** - *Zeng, L. (2024)*.
  * **[PRO-CS : An Instance-Based Prompt Composition Technique for Code-Switched Tasks](https://aclanthology.org/2022.emnlp-main.698/)** - *Bansal, S., et al. (2022)*.

-----

## 4\. Evaluation & Benchmarking

> Resources for evaluating model performance on code-switching tasks.

### Benchmarks

  * **[Overview for the First Shared Task on Language Identification in Code-Switched Data](https://aclanthology.org/W14-3907/)** - *Solorio, T., et al. (2014)*.
  * **[Overview for the Second Shared Task on Language Identification in Code-Switched Data](https://aclanthology.org/W16-5805/)** - *Molina, G., et al. (2016)*.
  * **[LinCE: A centralized benchmark for linguistic code-switching evaluation](https://aclanthology.org/2020.lrec-1.223/)** - *Aguilar, G., et al. (2020)*.
  * **[GLUECoS: An Evaluation Benchmark for Code-Switched NLP](https://aclanthology.org/2020.acl-main.329/)** - *Khanuja, S., et al. (2020)*.

### Evaluation Metrics

  * **[Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)** - *Papineni, K., et al. (2002)*.
  * **[chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/)** - *Popović, M. (2015)*.
  * **[Minimal Pair-Based Evaluation of Code-Switching](https://aclanthology.org/2025.acl-long.910/)** - *Sterner, I. & Teufel, S. (2025)*.
  * **[PIER: A Novel Metric for Evaluating What Matters in Code-Switching](https://arxiv.org/abs/2501.09512)** - *Ugan, E. Y., et al. (2025)*.

-----

## 5\. Multi & Cross-Modal Applications

> Applying code-switching NLP to speech, vision, and other modalities.

### Speech Processing

  * **[Bilingual by default: Voice Assistants and the role of code-switching in creating a bilingual user experience](https://doi.org/10.1145/3543829.3544511)** - *Cihan, H., et al. (2022)*.
  * **[End-to-End Speech Translation for Code Switched Speech](https://aclanthology.org/2022.findings-acl.113/)** - *Weller, O., et al. (2022)*.
  * **[CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units](https://aclanthology.org/2024.acl-srw.40/)** - *Kang, Y. (2024)*.

### Vision-Language & Document Processing

  * **[Towards Developing a Multilingual and Code-Mixed Visual Question Answering System by Knowledge Distillation](https://aclanthology.org/2021.findings-emnlp.151/)** - *Raj Khan, H., et al. (2021)*.
  * **["To Have the 'Million' Readers Yet": Building a Digitally Enhanced Edition of the Bilingual Irish-English Newspaper](https://aclanthology.org/2024.lt4hala-1.9/)** - *Dereza, O., et al. (2024)*.

### Cross-Modal Integration

  * **[CI-AVSR: A Cantonese Audio-Visual Speech Dataset for In-car Command Recognition](https://aclanthology.org/2022.lrec-1.731/)** - *Dai, W., et al. (2022)*.
  * **[ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos](https://aclanthology.org/2024.findings-acl.663/)** - *Maity, K., et al. (2024)*.

-----


## Workshops & Shared Tasks

> A list of academic workshops and community shared tasks dedicated to code-switching.

  * [CALCS @ EMNLP 2014, 2016, 2023**: Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/events/emnlp-2014/).
  * [CALCS @ ACL 2018**: Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/volumes/W18-32/).
  * [CALCS @ LREC 2020**: Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/2020.calcs-1.0/).
  * [CALCS @ NAACL 2021*: Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/events/calcs-2021/).
  * [WILDRE-6 2022: Workshop within the 13th Language Resources and Evaluation Conference](https://aclanthology.org/volumes/2022.wildre-1/).
  * [ICON @ ICON 2022*: 19th International Conference on Natural Language Processing (ICON)](https://aclanthology.org/volumes/2022.icon-wlli/).
  * [CALCS @ 2023**: 6th Workshop on Computational Approaches to Linguistic Code-Switching](https://aclanthology.org/events/calcs-2023/).

-----

## Contributing

Your contributions are always welcome and make this community resource better\! Please read the `CONTRIBUTING.md` file for guidelines.

If you have a paper, dataset, or tool you'd like to add:

1.  Fork the repository.
2.  Add your resource to the relevant section.
3.  Please try to follow the existing format and include a direct link.
4.  Submit a pull request\!
