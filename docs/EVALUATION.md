# Evaluation framework

This project separates ranking relevance, ANN fidelity, serving performance, and real-user outcomes. They answer different questions and should not be reported as interchangeable metrics.

## Offline ranking relevance

Amazon ESCI labels provide offline relevance supervision. The experiments report NDCG@10, Precision@1, Precision@5, Recall@10, and MRR@10.

| Model | NDCG@10 | P@1 | P@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.810273 | 0.557683 | 0.511441 | 0.641731 | 0.700712 |
| Frozen MiniLM | 0.833919 | 0.627573 | 0.546865 | 0.670976 | 0.753809 |
| Fine-tuned A, lr=2e-5 | 0.845803 | 0.649593 | 0.566108 | 0.685205 | 0.770350 |
| Fine-tuned B, lr=1e-5 | 0.843344 | 0.649593 | 0.564864 | 0.683229 | 0.767688 |

The best NDCG@10 result is 0.845803, up 0.011884 from the frozen MiniLM baseline. Query-level bootstrap analysis was used to check whether the change was broadly consistent rather than driven by only a few queries.

## ANN retrieval fidelity

FAISS IVF is compared with exact FAISS Flat search using top-k neighbor overlap. This measures approximate-search fidelity, not user relevance.

On the 1.3M-product benchmark with 1,000 real Amazon queries and `nlist=4096`:

| Index | nprobe | ANN Recall@10 | Mean FAISS search time |
| --- | ---: | ---: | ---: |
| Flat | - | 1.0000 | 5.129 ms/query |
| IVF | 64 | 0.9379 | 1.375 ms/query |
| IVF | 128 | 0.9592 | 2.615 ms/query |
| IVF | 256 | 0.9753 | 5.097 ms/query |

`nprobe=128` is the selected operating point: 95.92% overlap with exact Flat Top-10 neighbors at roughly half the measured batch-normalized FAISS search time.

## Serving performance

Observed local CPU measurements for `gaming keyboard` on the 1.3M catalog:

- Dense retrieval: 51.944 ms backend latency
- Dense + lexical fusion: 86.624 ms backend latency
- Dense + lexical fusion + CrossEncoder before candidate reduction: 19,068.851 ms backend latency

The CrossEncoder is therefore optional. The serving stack now reranks only a smaller first-stage subset controlled by `RERANK_CANDIDATES`.

## Real-user satisfaction

Offline relevance scores do not prove user satisfaction. A production system would validate ranking changes through A/B tests using metrics such as CTR, add-to-cart rate, conversion, abandonment, query reformulation, latency, and error rate.

Interview summary:

> ESCI labels and NDCG/Precision/Recall/MRR measure offline ranking relevance. FAISS IVF recall against Flat measures ANN fidelity, not user relevance. Latency measures system performance. Since this project has no production traffic, real user satisfaction is not claimed from offline metrics; that would require online A/B testing.
