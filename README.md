# Beyond Monolingual Assumptions: Code-Switched NLP in the LLM Era

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This collection presents tutorials, workshops, papers, and other resources focused on computational linguistics research in code-switching. The list will be continuously updated, and contributions through pull requests are encouraged.

## Table of Contents

- [🚀 Highlights](#-highlights)
- [🏫 Workshops & Conferences](#-workshops--conferences)
- [📚 Surveys & Literature Reviews](#-surveys--literature-reviews)
- [🧠 Code-Mixed Language Analytics](#-code-mixed-language-analytics)
  - [Natural Language Understanding](#natural-language-understanding)
  - [Natural Language Generation](#natural-language-generation)
- [📊 Datasets & Resources](#-datasets--resources)
  - [Datasets](#datasets)
  - [Frameworks & Toolkits](#frameworks--toolkits)
- [🛠️ Model Training & Adaptation](#️-model-training--adaptation)
  - [Pre-training Approaches](#pre-training-approaches)
  - [Fine-tuning Approaches](#fine-tuning-approaches)
  - [Post-training Approaches](#post-training-approaches)
- [📏 Evaluation & Benchmarking](#-evaluation--benchmarking)
  - [Benchmarks](#benchmarks)
  - [Evaluation Metrics](#evaluation-metrics)
- [🎥 Multi- & Cross-Modal Applications](#-multi--cross-modal-applications)
  - [Speech Processing](#speech-processing)
  - [Vision-Language Processing](#vision-language-processing)
  - [Cross-Modal Integration](#cross-modal-integration)
- [📖 Books & Theses](#-books--theses)
- [🔧 Tools & Libraries](#-tools--libraries)
- [👥 Contributors](#-contributors)

## 🚀 Highlights

- **The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges** - Comprehensive survey paper on code-switching research evolution
- **NAACL 2025** - Code-switching workshop coming soon! 
- **EMNLP 2023** - Successful code-switching workshop organized
- **CodeMixEval** - Most extensive analytical framework for studying code-mixing in LLMs across 18 languages
- **MEGAVERSE** - Benchmarking large language models across 22 datasets and 83 languages

## 🏫 Workshops & Conferences

### Recent Workshops
- **[CALCS 2025](https://aclanthology.org/2025.calcs-1.pdf)** - 7th Workshop on Computational Approaches to Linguistic Code-Switching
- **[EMNLP 2023](https://github.com/gentaiscool/code-switching-papers)** - Code-switching workshop
- **[FIRE 2020](https://github.com/goru001/nlp-for-hinglish)** - Dravidian-Codemix-HASOC workshop

### Upcoming Events
- **NAACL 2025** - Code-switching workshop (details coming soon)

## 📚 Surveys & Literature Reviews

### Comprehensive Surveys
- **[A Survey of Code-switching: Linguistic and Social Perspectives for Language Technologies](https://arxiv.org/abs/2301.01967)** (2023) - Doğruöz et al.
- **[Code-Switching in End-to-End Automatic Speech Recognition: A Systematic Literature Review](https://arxiv.org/html/2507.07741v1)** (2025) - Agro et al.
- **[Beyond Monolingual Assumptions: A Survey of Code-Switched NLP in the Era of Large Language Models](file:2)** - Anonymous ACL submission covering 304 studies across 5 research areas

### Specialized Reviews
- **[Shared Lexical Items as Triggers of Code Switching](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00613/118711/)** (2023) - Wintner et al.
- **[Analyzing the Role of Part-of-Speech in Code-Switching](https://aclanthology.org/2024.findings-eacl.120.pdf)** (2024) - Chi et al.

## 🧠 Code-Mixed Language Analytics

### Natural Language Understanding

#### Language Identification
- **TongueSwitcher** - Boundary detection enhancement for German-English CSW
- **MaskLID** - Subdominant language pattern identification
- **[COOLI](https://github.com/topics/code-mixed)** - Code-mixing offensive language identification
- **Equivalence Constraint Theory** - Integration with transformers for grammatically valid switch points

#### POS Tagging
- **XLM-RoBERTa** - Fine-tuning with S-index for Hinglish POS tagging
- **[PRO-CS](https://aclanthology.org/2024.findings-naacl.198.pdf)** - Instance-based prompt composition for code-switched tasks
- **CoMix** - POS-guided attention with phonetic signals
- **PACMAN** - Parallel CodeMixed data generation for POS tagging

#### Named Entity Recognition
- **Multi-CoNER** - Multilingual complex named entity recognition
- **[CodemixedNLP](https://github.com/AI4Bharat/indicnlp_catalog)** - Toolkit supporting multilingual and Hinglish NER
- **GLiNER** - Outperforms ChatGPT in zero-shot NER settings
- **CMB Models** - Two-stage approach for Hinglish NER

#### Sentiment & Emotion Analysis
- **[HinglishNLP](https://github.com/TrigonaMinima/HinglishNLP)** - Comprehensive NLP resources for Hinglish
- **EmoMix-3L** - Bangla-English-Hindi emotion classification
- **OffMix-3L** - Trilingual offensive language identification
- **DravidianCodeMix** - Sentiment analysis for Dravidian languages

### Natural Language Generation

#### Machine Translation
- **[PHINC Corpus](https://www.kaggle.com/datasets/mrutyunjaybiswal/phincparallel-hinglish-corpus-machine-translation)** - Parallel Hinglish corpus for machine translation
- **CoVoSwitch** - Prosody-aware synthetic data generation
- **COMET-based** - Synthetic data boost for Indic translation
- **Fine-tuned T5** - Strong CodeMix-to-English results

#### Text Generation
- **[COMMIT](https://aclanthology.org/2024.findings-naacl.198.pdf)** - Code-mixing English-centric LLM for multilingual instruction tuning
- **COCOA** - Strong English-Spanish generation
- **EZSwitch** - Code-mixed text generation framework
- **Dependency Tree Methods** - CSW text production without parallel corpora

#### Dialogue & Conversation
- **GupShup** - First code-mixed dialogue summarization dataset
- **MulZDG** - Multilingual code-switching framework for zero-shot dialogue generation
- **CST5** - Data augmentation for code-switched semantic parsing

## 📊 Datasets & Resources

### Datasets

#### Multilingual Coverage
- **[MEGAVERSE](https://arxiv.org/abs/2410.13394)** - 83 languages across 22 datasets
- **SwitchLingua** - 420k texts, 80+ hours across 12 languages
- **GLUECoS** - Comprehensive benchmark for code-switched tasks
- **X-RiSAWOZ** - 18k+ utterances per language

#### Low-Resource Coverage
- **BnSentMix** - Bengali-English code-mixed sentiment dataset
- **DravidianCodeMix** - Tamil, Kannada, Malayalam-English datasets
- **KRCS** - Kazakh-Russian code-switching corpus
- **AfroCS-xs** - 4 African languages + English agricultural domain

#### Hinglish Specialized
- **[Hinglish Language Corpus](https://data.mendeley.com/datasets/vdtcp2yt9n/3)** - Synthetic and manually written Hinglish sentences
- **[Hinglish Everyday Conversations](https://huggingface.co/datasets/Abhishekcr448/Hinglish-Everyday-Conversations-1M)** - 1M synthetic Hinglish conversations
- **L3Cube-HingCorpus** - Used for HingBERT training
- **IIIT-H en-hi-codemixed-corpus** - Gold standard parallel corpus

#### Speech Datasets
- **ASCEND** - Mandarin-English conversational dataset
- **SEAME** - Mandarin-English-Hokkien three-language mixing
- **CI-AVSR** - Cantonese audio-visual speech dataset
- **English-isiZulu CS** - Low-resource African language dataset

### Frameworks & Toolkits

#### Annotation Frameworks
- **COMMENTATOR** - LLM-integrated robust text annotation
- **CoSSAT** - Speech annotation enablement
- **ToxVidLM** - Multimodal video toxicity detection

#### Synthetic Data Generation
- **GCM** - Open-source code-mixed text generation
- **[CodemixedNLP](https://github.com/sagorbrur/codeswitch)** - Language identification, POS, NER, sentiment analysis toolkit
- **VACS** - Text generation with variational autoencoder
- **SynCS** - Zero-shot gains through synthetic data

## 🛠️ Model Training & Adaptation

### Pre-training Approaches

#### Specialized Code-Mixed Models
- **[HingBERT](https://github.com/l3cube-pune/MarathiNLP)** - Outperforms mBERT on GLUECoS
- **CoMix** - Translation boost by 12.98 BLEU
- **CMLFormer** - Dual-decoder with switching point learning
- **CMCLIP** - 5-10% improvement over baselines

#### Task-Adaptive Pre-Training
- **Boundary-aware MLM** - Improves QA/SA performance
- **Alignment-based pre-training** - +7.32% SA, +0.76% NER, +1.9% QA
- **SynCS** - +10.14 points in Chinese performance

### Fine-tuning Approaches

#### Task-Specific Fine-tuning
- **XLM-RoBERTa** - S-index for switching intensity measurement
- **Transformer models** - State-of-the-art Kannada-English LID
- **mBART/mT5** - Enhanced fluency for Hinglish

#### Multi-task Fine-tuning
- **Intermediate-task mBERT** - 2-3% improvement on GLUECoS
- **Joint training** - Boosts offensive LID and NER
- **AdapterFusion** - Modular approaches for zero-shot transfer

### Post-training Approaches

#### Few-shot & Zero-shot Learning
- **EntityCS** - 10% gain in slot filling with Wikidata
- **CoSDA-ML** - 0.70 score for zero-shot NER
- **ChatGPT prompting** - Various few-shot approaches
- **RAG-based learning** - 63.03 score for hate speech detection

#### Instance-based Prompting
- **PRO-CS** - 10-15% improvement in NER and POS tagging
- **GLOSS** - 55% BLEU/METEOR gains through self-training
- **In-Context Mixing** - 5-8% improvement in intent classification

## 📏 Evaluation & Benchmarking

### Benchmarks

#### Comprehensive Benchmarks
- **[CodeMixEval](https://arxiv.org/pdf/2507.18791.pdf)** - 18 languages from 7 language families
- **[MMLU-ProX](https://mmluprox.github.io)** - 29 languages multilingual benchmark
- **GLUECoS** - Multi-task evaluation framework
- **LinCE** - Centralized benchmark for linguistic code-switching

#### Domain-Specific Benchmarks
- **CodeMixBench** - 5k+ Hinglish, Spanish-English, Chinese Pinyin-English prompts
- **Medical Dialogue Dataset** - Telugu-English with 3k dialogs
- **CroCoSum** - Manually annotated summarization data

### Evaluation Metrics

#### Traditional Metrics
- **F1, Precision, Recall** - Classification tasks
- **BLEU, ROUGE, METEOR** - Generative tasks
- **Accuracy measures** - General performance

#### CS-Specific Metrics
- **Code-Mixing Index (CMI)** - Word-level mixing measurement
- **SyMCoM** - Grammatical well-formedness evaluation
- **I-Index** - Switch-point integration assessment
- **HinglishEval** - Linguistic metrics with embeddings

#### Task-Specific Metrics
- **Semantic-Aware Error Rate (SAER)** - Enhanced ASR evaluation
- **PhoBLEU** - Orthographic variation handling
- **chrF++** - Character n-grams for morphologically rich languages

## 🎥 Multi- & Cross-Modal Applications

### Speech Processing

#### Speech Translation & ASR
- **Whisper-based segmentation** - Korean-English CS data generation
- **Wav2Vec2/GPT-2 fusion** - Enhanced performance across Indian languages
- **MoE with SC-LLMs** - Audio-visual integration for Mandarin-English
- **Transformer-based** - Hindi-Marathi ASR

#### Audio-Visual Recognition
- **CI-AVSR** - Enhanced visual features in Mandarin-English
- **Low-resource enhancement** - Yoruba-English ASR improvements
- **Multimodal fusion** - Audio, visual, textual cue integration

### Vision-Language Processing

#### Multimodal VQA
- **Knowledge distillation** - Hinglish systems for code-mixed queries
- **BanglAssist** - RAG for Bengali-English CSW
- **Cross-modal alignment** - Vision-language integration

#### Document Processing
- **Multilingual OCR** - Robust extraction capabilities
- **Contrastive learning** - Vietnamese-English analysis improvement

### Cross-Modal Integration

#### Phonetic & Multimodal Processing
- **Discriminative language modeling** - Multilingual CSW enhancement
- **Transliteration** - Agglutinative language support
- **Transformer-based phonetic guidance** - CSW phonetic task improvement

## 📖 Books & Theses

### Books
- **Multilingual NLP** - Comprehensive guide to multilingual processing
- **Code-Switching: Theory and Practice** - Linguistic foundations

### Theses
- **Understanding and Modeling Code-Switching** - Edinburgh Research Archive
- **Code-Mixing in Social Media** - Computational approaches
- **Multilingual Neural Networks** - Advanced architectures

## 🔧 Tools & Libraries

### General NLP Libraries
- **[polyglot](https://sunscrapers.com/blog/9-best-python-natural-language-processing-nlp/)** - Multilingual NLP library with 130+ languages
- **[Indic NLP Library](https://github.com/AI4Bharat/indicnlp_catalog)** - Comprehensive Indian language NLP toolkit
- **[iNLTK](https://github.com/AI4Bharat/indicnlp_catalog)** - Out-of-box support for Indic languages

### Code-Switching Specific
- **[CodeSwitch](https://github.com/sagorbrur/codeswitch)** - Language identification, POS tagging, NER, sentiment analysis
- **[HinglishNLP](https://github.com/TrigonaMinima/HinglishNLP)** - Comprehensive Hinglish NLP resources
- **Sanskrit Coders Indic Transliteration** - Script conversion and romanization

### Research Frameworks
- **[ULCA](https://github.com/AI4Bharat/indicnlp_catalog)** - Universal Language Contribution API
- **[AI4Bharat](https://github.com/AI4Bharat/indicnlp_catalog)** - Indian language technologies
- **LTRC** - Language Technologies Research Center tools

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for more information.

### How to Contribute
1. Fork the repository
2. Add your resource in the appropriate section
3. Follow the existing format
4. Submit a pull request
5. Ensure your addition includes proper citations and links

## 👥 Contributors

Thanks to all the researchers and developers who have contributed to this awesome list!


## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

This work is licensed under CC0 1.0 Universal.

---

**Maintained by**: Research Community  
**Last Updated**: January 2025  
**Total Resources**: 200+ papers, datasets, and tools

> "Code-switching is not a deficient form of language use but a sophisticated linguistic skill that reflects the multilingual reality of global communication." - Computational Linguistics Community
