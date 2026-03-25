**Language / 语言：** [English](./README.md) | [中文](./README.Chinese.md)

---

# Meng Zhichao's Knowledge Note Repository

## About Me

I am Meng Zhichao, an algorithm engineer.

Starting from mathematics, I entered the data science field through self-study, progressing through power big data, financial risk modeling, NLP, and LLM engineering — gradually forming a working style centered on "model-driven thinking". I have published 200+ technical articles on [CSDN](https://blog.csdn.net/hbkybkzw) with nearly 300,000 cumulative views.

In 2026, I participated in SemEval-2026 Task 7 (Multilingual Everyday Commonsense Evaluation). In Track 1 (Short Answer task) I ranked **1st** with a score of 78.7672, and in Track 2 (Multiple Choice task) I achieved an overall accuracy of 96.35% on approximately 47,000 samples.

---

## Repository Contents

This repository contains notes from my daily learning and engineering practice, covering the following areas:

### 📂 Directory Structure

```
.
├── markdown_notes/     # Technical notes in Markdown format (approx. 2020 onwards)
├── handwritten_notes/  # Handwritten notes (scanned, approx. before 2020)
└── README.md
```

### Knowledge System

The `handwritten_notes` folder contains notes primarily from after university graduation up to around 2020, when handwriting was the main format. The content is math-heavy, covering calculus, linear algebra, statistical learning methods, machine learning, and foundational theory. Probability theory and mathematical statistics were also included, though those notes are no longer available.

The `markdown_notes` folder contains notes primarily in Markdown format (converted to PDF inside the folder), covering learning and accumulated knowledge from work experience after 2020.

All notes were written by me personally. The handwritten PDF scans have original notebooks as source, and the Markdown notes have original `.md` files.

#### 📝 handwritten_notes (Handwritten Notes, before 2020)

| Topic | Content |
| --- | --- |
| **Calculus** | Foundations and advanced topics: limits, derivatives, integrals |
| **Linear Algebra** | Matrix operations, eigenvalues, spatial transformations |
| **Statistical Learning Methods** | Full coverage of Li Hang's *Statistical Learning Methods* 1st edition (Chapters 1–12) |
| **Machine Learning** | Core content from Zhou Zhihua's *Machine Learning* |

#### 💻 markdown_notes (Markdown Notes)

| Topic | Content |
| --- | --- |
| **Large Language Models** | Transformer architecture, LLM training (RLHF), reasoning models, weight quantization, deployment and load testing |
| **LLM Engineering** | Implementing Qwen3 from scratch, MCP protocol, OpenAI API, various platform APIs, Huawei 910B deployment |
| **Deep Learning** | PyTorch custom datasets, TensorFlow 2.0, activation functions, MindSpore framework |
| **Machine Learning** | Sklearn, L2 regularization, hyperparameter tuning (Hyperopt), ensemble learning, time series (Prophet) |
| **Mathematics & Information Theory** | Information entropy, optimization, GPU FLOPS analysis, LaTeX formulas, reading notes on *The Beauty of Mathematics* |
| **Data Engineering** | NumPy, Pandas, PySpark, Elasticsearch, data structures and algorithms |
| **DevOps & Tools** | Docker, Linux common commands, Python async programming, web scraping, Oracle database |

---

## Switching Between Chinese and English README

This repository provides two README files:

- `README.md` — English version (this file)
- `README.Chinese.md` — Chinese version (中文版)

### How to Switch on GitHub

GitHub displays `README.md` by default. To read the Chinese version, click the file [`README.Chinese.md`](./README.Chinese.md) directly in the repository file list.

### Git Usage Guide

#### Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

#### Check current branch

```bash
git branch
```

#### Switch to an existing branch

```bash
git checkout <branch-name>
# or (Git 2.23+)
git switch <branch-name>
```

#### Create and switch to a new branch

```bash
git checkout -b <new-branch-name>
# or (Git 2.23+)
git switch -c <new-branch-name>
```

#### Switch back to the main branch

```bash
git checkout main
# or
git switch main
```

#### View all branches (local + remote)

```bash
git branch -a
```

#### Pull latest changes after switching

```bash
git pull origin <branch-name>
```

> Continuously updated.
