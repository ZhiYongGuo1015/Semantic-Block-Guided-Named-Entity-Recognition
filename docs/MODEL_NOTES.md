# Model Notes

## RoBERTa-SBG-CRF

RoBERTa-SBG-CRF is designed to let geological semantic block information participate in three parts of the NER model:

1. **Representation modulation**: the semantic block label is embedded and injected into token-level contextual representations through a residual gate.
2. **Type-aware emission adjustment**: the semantic block representation modulates entity-type emission scores through a type emission gate.
3. **Block-adaptive CRF decoding**: the CRF transition matrix is adjusted with a low-rank semantic-block residual, allowing BIO transition preferences to vary across geological sections.

The default manuscript configuration is:

```text
category embedding dim = 32
semantic adapter dim = 64
semantic adapter scale = 1.0
type emission gate scale = 0.20
block CRF rank = 8
block CRF scale = 0.2
R-Drop alpha = 0.5
```

## Backbone-Specific Variants

The code also supports targeted semantic-block adaptations for recurrent decoding structures:

- BiLSTM: semantic block state initialization, RoBERTa bridge, type gate, and light block CRF.
- BiGRU: pre-GRU residual input gate, block state initialization, RoBERTa bridge, and decoupled type decoder.
- BiGRU-Attention: semantic block memory in attention plus GRU state initialization and type gate.

These variants are mainly used for supplementary analysis showing that semantic block information needs to be adapted to the internal structure of each backbone.

