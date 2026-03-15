# transformer-based-English-to-Hindi-translator
Neural Machine Translation for English–Hindi Public Notices in Higher Education Using Transformer Models


The majority of public announcements from Indian higher education institutions are published in English, which poses serious accessibility challenges for stakeholders who do not understand English, such as parents, students, and administrative personnel. In order to translate academic and administrative public notices between Hindi and English in both directions, a bidirectional Transformer-based neural machine translation (NMT) system is presented in this study. Using a curated dataset of 5,000 parallel sentence pairs from institutional announcements, the proposed system uses a unified encoder-decoder Transformer architecture with language-specific tokens and domain-specific fine-tuning. The bidirectional technique facilitates genuinely multilingual institutional communication by enabling smooth translation from Hindi to English and English to Hindi within a single model. In order to provide real-time bidirectional communication, the NMT model is also included into a bilingual chatbot interface.


⚙️ How It Works
You type English text in the browser
            ↓
Flask server (app.py) receives the text
            ↓
MarianTokenizer converts text → token IDs
            ↓
MarianMT Transformer Neural Network
(Encoder understands English → Decoder generates Hindi)
            ↓
Beam Search (k=5) picks the best translation
            ↓
Tokenizer decodes token IDs → Hindi (Devanagari)
            ↓
Browser displays the Hindi translation
