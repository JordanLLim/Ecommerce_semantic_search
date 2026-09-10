# Evaluation framework

This project separates ranking quality, ANN fidelity, system performance, and real-user outcomes. These are different questions and should not be reported as interchangeable metrics.

## 1. Offline ranking relevance

Amazon ESCI labels are used as offline relevance supervision. Query-product pairs are labeled as Exact, Substitute, Complement, or Irrelevant.

The ranking experiments report:

- NDCG@10
- Precision@1
- Precision@5
- Recall@10
- MRR@10

Canonical validation results on 2,089 queries:

| Model | NDCG@10 | P@1 | P@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.810273 | 0.557683 | 0.511441 | 0.641731 | 0.700712 |
| Frozen MiniLM | 0.833919 | 0.627573 | 0.546865 | 0.670976 | 0.753809 |
| Fine-tuned A, lr=2e-5 | 0.845803 | 0.649593 | 0.566108 | 0.685205 | 0.770350 |
| Fine-tuned B, lr=1e-5 | 0.843344 | 0.649593 | 0.564864 | 0.683229 | 0.767688 |

The best NDCG@10 result is 0.845803, up 0.011884 from the frozen MiniLM baseline. Query-level bootstrap analysis was also used to test whether the change was broadly consistent rather than driven by a few queries.

## 2. ANN retrieval fidelity

FAISS IVF is compared with exact FAISS Flat search using top-k neighbor overlap. This metric measures how faithfully approximate search reproduces exact vector neighbors. It is not a user-relevance metric.

On the 1.3M-product benchmark with 1,000 real Amazon queries and `nlist=4096`:

| Index | nprobe | ANN Recall@10 | Mean FAISS search time |
| --- | ---: | ---: | ---: |
| Flat | - | 1.0000 | 5.129 ms/query |
| IVF | 64 | 0.9379 | 1.375 ms/query |
| IVF | 128 | 0.9592 | 2.615 ms/query |
| IVF | 256 | 0.9753 | 5.097 ms/query |

`nprobe=128` is the selected operating point because it retained 95.92% of exact Flat Top-10 neighbors while reducing the measured batch-normalized FAISS search time by about 2x.

## 3. Serving performance

Serving latency includes more than the FAISS benchmark. The live API performs query encoding, ANN retrieval, optional lexical fusion, metadata handling, and optional CrossEncoder reranking.

Observed local CPU measurements for `gaming keyboard` on the 1.3M catalog:

- Dense retrieval: 51.944 ms backend latency
- Dense + lexical fusion: 86.624 ms backend latency
- Dense + lexical fusion + CrossEncoder before candidate reduction: 19,068.851 ms backend latency

The CrossEncoder path is therefore optional. The serving stack now reranks only a smaller first-stage subset, controlled by `RERANK_CANDIDATES`, instead of blindly reranking the entire candidate pool.

## 4. Real-user satisfaction

Offline relevance scores do not prove that real users are satisfied. A production system would validate ranking changes through online experiments.

Useful online metrics include:

- click-through rate
- add-to-cart rate
- conversion rate
- search abandonment
- query reformulation rate
- latency and error rate

A reasonable A/B test would compare the default dense/lexical retrieval path with an enhanced ranking path while monitoring both business metrics and latency.

## Interview summary

A concise explanation is:

> I validated the system at multiple levels. ESCI labels and NDCG, Precision, Recall and MRR measure offline ranking relevance. FAISS IVF recall against Flat measures ANN fidelity, not user relevance. Latency benchmarks measure system performance. Since I do not have production traffic, I do not claim user satisfaction from offline metrics; in production I would validate that using A/B tests with CTR, conversion, reformulation and abandonment metrics.
