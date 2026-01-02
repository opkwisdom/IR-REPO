MSMARCO_MAP = {
    "msmarco": {
        "passages.tar.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
        "queries.tar.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
        "train_qrels.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
        "dev_qrels.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",

        "triples_train.tar.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz",   # Official MSMARCO triple
        "top1000.dev.tar.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz"              # Official MSMARCO dev top-1000
    }
}

HF_MAP = {
    "nq": {
        "path": "google-research-datasets/nq_open",
        "name": None,
    },
    "triviaqa": {
        "path": "mandarjoshi/trivia_qa",
        "name": "rc"
    },
    "hotpotqa": {
        "path": "hotpotqa/hotpot_qa",
        "name": "fullwiki"
    }
}

BEIR_MAP = {
    "nq": {
        "data": "BeIR/nq",
        "qrel": "BeIR/nq-qrels"
    }
}

# "passages": "natural-questions",
IR_MAP = {
    "nq": {
        "train": "dpr-w100/natural-questions/train",
        "dev": "dpr-w100/natural-questions/dev"
    },
    "triviaqa": {
        "train": "dpr-w100/trivia-qa/train",
        "dev": "dpr-w100/trivia-qa/dev"
    },
    "msmarco": {
        "train": "msmarco-passage/train/judged",
        "dev": "msmarco-passage/dev/small"
    }
}

### =================================== TEMP ========================================= ###
WIKI_DUMP = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"   # DPR, Retro