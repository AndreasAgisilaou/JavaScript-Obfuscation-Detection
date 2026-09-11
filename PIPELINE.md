# Pipeline

How [detection/](detection/) moves a batch of JavaScript files from labeled training data to a final
**Obfuscated** / **NonObfuscated** call — across three independently trained models, an agreement vote,
and an OpenAI fallback for whatever the vote can't settle.

```mermaid
flowchart TD
    subgraph S1["01 · Train — runs on every /*/train call"]
        direction TB
        TRAIN_IN(["data/train/&lt;dataset&gt;/<br/>{obfuscated, non_obfuscated}"])
        TRAIN_IN --> RF1["RandomForest<br/>features.py: feature extraction"]
        TRAIN_IN --> XG1["XGBoost<br/>features.py: feature extraction"]
        TRAIN_IN --> NN1["Neural Network<br/>own extractor (13 features)"]

        RF1 --> RF2["IsolationForest → StandardScaler"]
        XG1 --> XG2["IsolationForest → StandardScaler"]
        NN1 --> NN2["StandardScaler — no outlier filter"]

        RF2 --> RF3["RandomForestClassifier.fit()"]
        XG2 --> XG3["XGBClassifier.fit()"]
        NN2 --> NN3["Dense 32→16→2 + EarlyStopping"]

        RF3 --> RFM["random_forest/&lt;name&gt;.pkl<br/>{model, scaler}"]
        XG3 --> XGM["xgboost/&lt;name&gt;.pkl<br/>{model, scaler}"]
        NN3 --> NNM["neural_network/&lt;name&gt;.h5<br/>+ &lt;name&gt;_scaler.pkl"]
    end

    subgraph S2["02 · Test — optional, only if test_dataset is given"]
        direction TB
        MODEL(["saved model + scaler<br/>(any of the three, from Train)"])
        TEST_IN(["data/test/&lt;dataset&gt;/<br/>{obfuscated, non_obfuscated}"])
        MODEL --> EVAL["evaluate_model():<br/>same extractor → scaler.transform() → predict() → accuracy_score()"]
        TEST_IN --> EVAL
        EVAL --> ACC["accuracy (0–1)<br/>returned with model_path"]
    end

    subgraph S3["03 · Predict &amp; Sort — POST /sort"]
        direction TB
        UNSORTED(["data/unsorted/&lt;dataset&gt;/<br/>flat, unlabeled .js files"])
        UNSORTED --> PICK["pick 2 or 3 saved models to vote with"]
        PICK --> PRF["RandomForest predict_directory()"]
        PICK --> PXG["XGBoost predict_directory()"]
        PICK --> PNN["Neural Net predict_directory()"]
        PRF --> VOTE{"all agree?"}
        PXG --> VOTE
        PNN --> VOTE
        VOTE -- all → 1 --> OBF1["obfuscated/"]
        VOTE -- all → 0 --> NOB1["non_obfuscated/"]
        VOTE -- split --> UNI1["unidentified/"]
    end

    subgraph S4["04 · Resolve — POST /openai/classify (fallback only)"]
        direction TB
        UNI2(["unidentified/<br/>(from Predict &amp; Sort)"])
        UNI2 --> GPT["resolve_unidentified():<br/>~3000-char chunks → gpt-4o-mini per chunk (temp 0)<br/>→ majority vote across chunks"]
        GPT --> OBF2["obfuscated/"]
        GPT --> NOB2["non_obfuscated/"]
        GPT --> UNI3["unidentified/<br/>every chunk failed"]
    end

    RFM -.-> MODEL
    XGM -.-> MODEL
    NNM -.-> MODEL
    UNI1 -.-> UNI2

    classDef artifact fill:#eef1f8,stroke:#3e4e82,color:#1b2029,stroke-width:1.5px;
    classDef obf fill:#fdf1e4,stroke:#b5661c,color:#5c3410,stroke-width:2px;
    classDef nonobf fill:#e9f7f4,stroke:#1d7a6e,color:#0f4a43,stroke-width:2px;
    classDef pending fill:#f5eef7,stroke:#7a4f86,color:#4a2f53,stroke-width:2px;

    class RFM,XGM,NNM,ACC artifact
    class OBF1,OBF2 obf
    class NOB1,NOB2 nonobf
    class UNI1,UNI2,UNI3 pending
```

## How the stages connect

1. RandomForest and XGBoost share one feature extractor and outlier filter (`detection/features.py`).
   The neural network keeps its own separate 13-feature extractor and skips outlier filtering — it
   trains on every row as given.
2. `features.py`'s `is_only_code_blocks()` check runs ahead of RandomForest and XGBoost's features: a
   file that's structurally just JSON/object data is forced to NonObfuscated — regardless of which
   training folder it came from, or what the model would have guessed.
3. Test and Predict always re-run the exact same feature extractor and the exact same saved scaler used
   at Train time — a model is only ever scored or asked to vote on features built the same way it
   learned them.
4. `/sort` only writes to `obfuscated/` or `non_obfuscated/` when every selected model agrees. Any split
   vote — and only a split vote — lands in `unidentified/`, which is the sole input `/openai/classify`
   ever reads.

See [README.md](README.md) for setup, endpoints, and dataset layout.
