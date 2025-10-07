
# Awesome Code-Mixing & Code-Switching



This collection, inspired by the survey paper **"Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models"**, presents tutorials, workshops, papers, and other resources focused on computational linguistics research in code-switching. The list will be continuously updated, and contributions through pull requests are encouraged.

For direct links to many of the papers listed below, please refer to this community-maintained spreadsheet: [Code-Switching Paper Links](https://docs.google.com/spreadsheets/d/1G1FbMDWZPvb6FLHCcohr9s7woc6uI-dzYGMNL_Px-2s/edit?usp=sharing)

## 📑 Table of Contents

  - [✨ Survey Papers](https://www.google.com/search?q=%23-survey-papers)
  - [1. 📝 Natural Language Understanding (NLU) Tasks](https://www.google.com/search?q=%231--natural-language-understanding-nlu-tasks)
      - [Language Identification (LID)](https://www.google.com/search?q=%23language-identification-lid)
      - [Part-of-Speech (POS) Tagging](https://www.google.com/search?q=%23part-of-speech-pos-tagging)
      - [Named Entity Recognition (NER)](https://www.google.com/search?q=%23named-entity-recognition-ner)
      - [Sentiment & Emotion Analysis](https://www.google.com/search?q=%23sentiment--emotion-analysis)
      - [Syntactic Analysis](https://www.google.com/search?q=%23syntactic-analysis)
      - [Intent Classification](https://www.google.com/search?q=%23intent-classification)
      - [Question Answering (QA)](https://www.google.com/search?q=%23question-answering-qa)
      - [Natural Language Inference (NLI)](https://www.google.com/search?q=%23natural-language-inference-nli)
  - [2. ✍️ Natural Language Generation (NLG) Tasks](https://www.google.com/search?q=%232-%EF%B8%8F-natural-language-generation-nlg-tasks)
      - [Code-Mixed Text Generation](https://www.google.com/search?q=%23code-mixed-text-generation)
      - [Machine Translation (MT)](https://www.google.com/search?q=%23machine-translation-mt)
      - [Cross-lingual Transfer](https://www.google.com/search?q=%23cross-lingual-transfer)
      - [Text Summarization](https://www.google.com/search?q=%23text-summarization)
      - [Dialogue Generation](https://www.google.com/search?q=%23dialogue-generation)
  - [3. 🧠 Model Training & Adaptation](https://www.google.com/search?q=%233--model-training--adaptation)
      - [Pre-training Approaches](https://www.google.com/search?q=%23pre-training-approaches)
      - [Fine-tuning Approaches](https://www.google.com/search?q=%23fine-tuning-approaches)
      - [Post-training Approaches](https://www.google.com/search?q=%23post-training-approaches)
  - [4. 📊 Evaluation & Benchmarking](https://www.google.com/search?q=%234--evaluation--benchmarking)
      - [Benchmarks](https://www.google.com/search?q=%23benchmarks)
      - [Evaluation Metrics](https://www.google.com/search?q=%23evaluation-metrics)
  - [5. 🖼️ Multi- & Cross-Modal Applications](https://www.google.com/search?q=%235-%EF%B8%8F-multi--cross-modal-applications)
      - [Speech Processing](https://www.google.com/search?q=%23speech-processing)
      - [Vision-Language & Document Processing](https://www.google.com/search?q=%23vision-language--document-processing)
      - [Cross-Modal Integration](https://www.google.com/search?q=%23cross-modal-integration)
  - [6. 📚 Datasets & Resources](https://www.google.com/search?q=%236--datasets--resources)
      - [Datasets](https://www.google.com/search?q=%23datasets)
      - [Frameworks & Toolkits](https://www.google.com/search?q=%23frameworks--toolkits)
  - [🏫 Workshops & Shared Tasks](https://www.google.com/search?q=%23-workshops--shared-tasks)
  - [🤝 Contributing](https://www.google.com/search?q=%23-contributing)

## ✨ Survey Papers

  * Anonymous ACL Submission. (2025). Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models.
  * Winata, G. I., Aji, A. F., Yong, Z. X., & Solorio, T. (2023). The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges. *ACL Findings*.

## 1\. 📝 Natural Language Understanding (NLU) Tasks

### Language Identification (LID)

  * Lambebo Tonja, A., et al. (2022). Transformer-based Model for Word Level Language Identification in Code-mixed Kannada-English Texts. *ICON*. [[Paper]](https://aclanthology.org/2022.icon-wlli.4/)
  * Sterner, I., & Teufel, S. (2023). TongueSwitcher: Fine-Grained Identification of German-English Code-Switching. *CALCS, EMNLP*. [[Paper]](https://aclanthology.org/2023.calcs-1.1/)
  * Kargaran, A. H., et al. (2024). MaskLID: Code-Switching Language Identification through Iterative Masking. *ACL*. [[Paper]](https://aclanthology.org/2024.acl-short.43/)
  * Kuwanto, G., et al. (2024). Linguistics Theory Meets LLM: Code-Switched Text Generation via Equivalence Constrained Large Language Models. *ArXiv*. [[Paper]](https://api.semanticscholar.org/CorpusID:273695372)
  * Iliescu, D.-M., et al. (2021). Much Gracias: Semi-supervised Code-switch Detection for Spanish-English: How far can we get?. *CALCS, NAACL*. [[Paper]](https://aclanthology.org/2021.calcs-1.9/)
  * Hossain, E., et al. (2021). NLP-CUET@LT-EDI-EACL2021: Multilingual Code-Mixed Hope Speech Detection using Cross-lingual Representation Learner. *LT-EDI, EACL*. [[Paper]](https://aclanthology.org/2021.ltedi-1.23)
  * Dave, B., et al. (2021). IRNLP\_DAIICT@LT-EDI-EACL2021: Hope Speech detection in Code Mixed text using TF-IDF Char N-grams and MuRIL. *LT-EDI, EACL*. [[Paper]](https://aclanthology.org/2021.ltedi-1.15/)

### Part-of-Speech (POS) Tagging

  * Absar, S. (2025). Fine-tuning cross-lingual LLMs for POS tagging in code-switched contexts. *RESOURCEFUL*. `[Paper]`
  * Aguilar, G., & Solorio, T. (2020). From English to code-switching: Transfer learning with strong morphological clues. *ACL*. `[Paper]` `[Code]`
  * Dowlagar, S., & Mamidi, R. (2021). A pre-trained transformer and CNN model with joint language ID and part-of-speech tagging for code-mixed social-media text. *RANLP*. `[Paper]`
  * Chopra, P., et al. (2021). Switch point biased self-training: Re-purposing pretrained models for code-switching. *EMNLP Findings*. `[Paper]`

### Named Entity Recognition (NER)

  * Wang, C., et al. (2018). Code-switched named entity recognition with embedding attention. *CALCS, ACL*. `[Paper]`
  * Zhou, R., et al. (2022). MELM: Data augmentation with masked entity language modeling for low-resource NER. *ACL*. `[Paper]`
  * Malmasi, S., et al. (2022). MultiCoNER: A large-scale multilingual and code-mixed dataset for complex NER. *NAACL*. `[Paper]`
  * Zaratiana, U., et al. (2024). GLINER: Generalist model for named entity recognition using bidirectional transformer. *NAACL*. `[Paper]`

### Sentiment & Emotion Analysis

  * Angel, J., et al. (2020). NLP-CIC at SemEval-2020 task 9: Analysing sentiment in code-switching language. *SemEval*. `[Paper]`
  * Chakravarthi, B. R., et al. (2022). Dravidiancodemix: Sentiment analysis and offensive language identification dataset for dravidian languages in code-mixed text. *Language Resources and Evaluation*. `[Paper]`
  * Zeng, L. (2024). Leveraging large language models for code-mixed data augmentation in sentiment analysis. *SICon*. `[Paper]`

### Syntactic Analysis

  * Kodali, P., et al. (2022). SyMCOM syntactic measure of code mixing a study of English-Hindi code-mixing. *ACL Findings*. `[Paper]`
  * Ghosh, U., et al. (2019). Dependency parser for Bengali-English code-mixed data enhanced with a synthetic treebank. *TLT, Syntax Fest*. `[Paper]`
  * Arora, G., et al. (2023). CoMix: Guide transformers to code-mix using POS structure and phonetics. *ACL Findings*. `[Paper]`

### Intent Classification

  * Krishnan, J., et al. (2021). Multilingual code-switching for zero-shot cross-lingual intent prediction and slot filling. *MRL, EMNLP*. `[Paper]`
  * Bansal, S., et al. (2022). PRO-CS: An instance-based prompt composition technique for code-switched tasks. *EMNLP*. `[Paper]` `[Code]`

### Question Answering (QA)

  * Gupta, D., et al. (2018). Uncovering code-mixed challenges: A framework for linguistically driven question generation and neural based question answering. *CoNLL*. `[Paper]`
  * Lee, J., et al. (2024). COMMIT: Code-mixing English-centric large language model for multilingual instruction tuning. *NAACL Findings*. `[Paper]` `[Code]`
  * Ahuja, S., et al. (2024). MEGAVERSE: Benchmarking large language models across languages, modalities, models and tasks. *NAACL*. `[Paper]`

### Natural Language Inference (NLI)

  * Khanuja, S., et al. (2020). A new dataset for natural language inference from code-mixed conversations. *CALCS, LREC*. `[Paper]`
  * Qin, L., et al. (2020). Cosda-ml: Multi-lingual code-switching data augmentation for zero-shot cross-lingual nlp. *IJCAI*. `[Paper]` `[Code]`
  * Shankar, B., et al. (2024). In-context mixing (ICM): Code-mixed prompts for multilingual LLMs. *ACL*. `[Paper]`

## 2\. ✍️ Natural Language Generation (NLG) Tasks

### Code-Mixed Text Generation

  * Tarunesh, I., et al. (2021). From machine translation to code-switching: Generating high-quality code-switched text. *ACL*. `[Paper]`
  * Mondal, S., et al. (2022). CoCoa: An encoder-decoder model for controllable code-switched generation. *EMNLP*. `[Paper]` `[Code]`
  * Yong, Z. X., et al. (2023). Prompting multilingual large language models to generate code-mixed texts. *CALCS, EMNLP*. `[Paper]`

### Machine Translation (MT)

  * Gupta, A., et al. (2021). Training data augmentation for code-mixed translation. *NAACL*. `[Paper]`
  * Dowlagar, S., & Mamidi, R. (2021). Gated convolutional sequence to sequence based learning for English-hingilsh code-switched machine translation. *CALCS, NAACL*. `[Paper]`
  * Chatterjee, A., et al. (2023). Lost in translation no more: Fine-tuned transformer-based models for CodeMix to English machine translation. *ICON*. `[Paper]`

### Cross-lingual Transfer

  * Li, Z., et al. (2024). Improving zero-shot cross-lingual transfer via progressive code-switching. *ArXiv*. `[Paper]`
  * Lee, D., et al. (2021). Scopa: Soft code-switching and pairwise alignment for zero-shot cross-lingual transfer. *CIKM*. `[Paper]`
  * Yoo, H., et al. (2025). Code-switching curriculum learning for multilingual transfer in LLMs. *ACL Findings*. `[Paper]`

### Text Summarization

  * Mehnaz, L., et al. (2021). GupShup: Summarizing open-domain code-switched conversations. *EMNLP*. `[Paper]` `[Dataset]`
  * Zhang, R., & Eickhoff, C. (2024). CroCoSum: A benchmark dataset for cross-lingual code-switched summarization. *LREC-COLING*. `[Paper]`
  * Suresh, S. K., et al. (2025). Cs-sum: A benchmark for code-switching dialogue summarization and the limits of large language models. *ArXiv*. `[Paper]`

### Dialogue Generation

  * Liu, Y., et al. (2022). MulZDG: Multilingual code-switching framework for zero-shot dialogue generation. *COLING*. `[Paper]`
  * Agarwal, A., et al. (2023). CST5: Data augmentation for code-switched semantic parsing. *Taming Large Language Models*. `[Paper]`

## 3\. 🧠 Model Training & Adaptation

### Pre-training Approaches

  * **Specialized Code-Mixed Models**:
      * Nayak, R., & Joshi, R. (2022). L3Cube-HingCorpus and HingBERT: A code mixed Hindi-English dataset and BERT language models. *WILDRE, LREC*. `[Paper]` `[Models]`
  * **Task-Adaptive Pre-Training (TAPT)**:
      * Gururangan, S., et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *ACL*. `[Paper]`
  * **Cross-lingual Alignment**:
      * Lample, G., et al. (2018). Word translation without parallel data. *ICLR*. `[Paper]`

### Fine-tuning Approaches

  * **Task-specific fine-tuning**:
      * Yoo, H., et al. (2025). Code-switching curriculum learning for multilingual transfer in LLMs. *ACL Findings*. `[Paper]`
  * **Multi-task fine-tuning**:
      * Aguilar, G., et al. (2020). LinCE: A centralized benchmark for linguistic code-switching evaluation. *LREC*. `[Paper]`
  * **Instruction Tuning**:
      * Lee, J., et al. (2024). COMMIT: Code-mixing English-centric large language model for multilingual instruction tuning. *NAACL Findings*. `[Paper]` `[Code]`
  * **Parameter Efficient Methods (PEFT)**:
      * Srivastava, V. (2025). Dwesh Vaani: An LLM for detecting religious hate speech in code-mixed Hindi-English. *CHIPSAL*. `[Paper]`

### Post-training Approaches

  * **Zero-, One- and Few-shot Learning**:
      * Winata, G. I., et al. (2021). Are Multilingual Models Effective in Code-Switching?. *CALCS, NAACL*. `[Paper]`
  * **Instance-based Prompting**:
      * Bansal, S., et al. (2022). PRO-CS: An instance-based prompt composition technique for code-switched tasks. *EMNLP*. `[Paper]` `[Code]`

## 4\. 📊 Evaluation & Benchmarking

### Benchmarks

  * Khanuja, S., et al. (2020). GLUECOS: An evaluation benchmark for code-switched NLP. *ACL*. `[Paper]` `[Dataset]`
  * Aguilar, G., et al. (2020). LinCE: A centralized benchmark for linguistic code-switching evaluation. *LREC*. `[Paper]`
  * Ahuja, S., et al. (2024). MEGAVERSE: Benchmarking large language models across languages, modalities, models and tasks. *NAACL*. `[Paper]`
  * Sheokand, M., et al. (2025). CodeMixBench: A new benchmark for generating code from code-mixed prompts. *ICON*. `[Paper]`

### Evaluation Metrics

  * **CS-Specific Metrics**:
      * Kodali, P., et al. (2022). SyMCOM syntactic measure of code mixing a study of English-Hindi code-mixing. *ACL Findings*. `[Paper]`
  * **Task-Specific Metrics**:
      * Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. *WMT*. `[Paper]`
  * **Quality Assessment**:
      * Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. *EMNLP*. `[Paper]` `[Code]`

## 5\. 🖼️ Multi- & Cross-Modal Applications

### Speech Processing

  * Weller, O., et al. (2022). End-to-end speech translation for code switched speech. *ACL Findings*. `[Paper]`
  * Dai, W., et al. (2022). CI-AVSR: A Cantonese audio-visual speech dataset- for in-car command recognition. *LREC*. `[Paper]`
  * Kronis, M., et al. (2024). Code-mixed text augmentation for Latvian ASR. *LREC-COLING*. `[Paper]`

### Vision-Language & Document Processing

  * Raj Khan, H., et al. (2021). Towards developing a multilingual and code-mixed visual question answering system by knowledge distillation. *EMNLP Findings*. `[Paper]`
  * Ahuja, S., et al. (2024). MEGAVERSE: Benchmarking large language models across languages, modalities, models and tasks. *NAACL*. `[Paper]`

### Cross-Modal Integration

  * Chi, J., & Bell, P. (2022). Improving code-switched ASR with linguistic information. *COLING*. `[Paper]`
  * Srivastava, V. (2025). Dwesh Vaani: An LLM for detecting religious hate speech in code-mixed Hindi-English. *CHIPSAL*. `[Paper]`

## 6\. 📚 Datasets & Resources

### Datasets

  * **Low-Resource Coverage**:
      * Borisov, M., et al. (2025). Low-resource machine translation for code-switched Kazakh-Russian language pair. *NAACL*. `[Paper]`
      * Samih, Y., & Maier, W. (2016). An Arabic-Moroccan Darija code-switched corpus. *LREC*. `[Paper]`
  * **Multilingual Coverage**:
      * Xie, P., et al. (2025). Switchlingua: The first large-scale multilingual and multi-ethnic code-switching dataset. *ArXiv*. `[Paper]`
  * **Synthetic Data Generation**:
      * Kuwanto, G., et al. (2024). Linguistics theory meets llm: Code-switched text generation via equivalence constrained large language models. *ArXiv*. `[Paper]`

### Frameworks & Toolkits

  * **Annotation Frameworks**:
      * Sheth, R., et al. (2024). Commentator: A code-mixed multilingual text annotation framework. *EMNLP*. `[Paper]`
  * **Synthetic Data Generation Toolkits**:
      * Gautam, D., et al. (2021). CoMeT: Towards code-mixed translation using parallel monolingual sentences. *CALCS, NAACL*. `[Paper]`
      * Potter, T., & Yuan, Z. (2024). LLM-based code-switched text generation for grammatical error correction. *EMNLP*. `[Paper]`

## 🏫 Workshops & Shared Tasks

A list of the code-switching workshop series and related events:

  * [First Workshop on Processing Code-Mixed Social Media Text (LingoMix 2018)](https://lingo.iitgn.ac.in/codemixing/)
  * First Workshop on Computational Approaches to Code-switching, EMNLP 2014
  * Second Workshop on Computational Approaches to Code-switching, EMNLP 2016
  * Third Workshop on Computational Approaches to Linguistic Code-switching, ACL 2018
  * Fourth Workshop on Computational Approaches to Linguistic Code-switching, LREC 2020
  * Fifth Workshop on Computational Approaches to Linguistic Code-switching, NAACL 2021
  * Sixth Workshop on Computational Approaches to Linguistic Code-switching, EMNLP 2023
  * Seventh Workshop on Computational Approaches to Linguistic Code-switching, NAACL 2025

## 🤝 Contributing

Your contributions are always welcome\! Please read the `CONTRIBUTING.md` file first. If you have a paper, dataset, or tool you'd like to add, please send a pull request to update the list and become one of our contributors\! We especially welcome additions of `[Paper]` and `[Code]` links.
