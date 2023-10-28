# GoRec
Here are codes of [[2023 ACM Multimedia] GoRec: A Generative Cold-Start Recommendation Framework](https://dl.acm.org/doi/pdf/10.1145/3581783.3612238)

In this work, we innovatively break the alignment function-based schema and propose a Generative cold Recommendation (GoRec) framework for multimedia-based new item recommendation.

# Datasets
We provide the pre-trained representations and cluster labels used in the paper. You can find the source of the whole dataset from the paper, copy it to the data directory and run gorec.py in the main directory to train the model.

# Hyperparameters

## Baby
uni_coeff=5 ; kl_coeff=10

## Clothing
uni_coeff=1 ; kl_coeff=5000

## Sports
uni_coeff=15 ; kl_coeff=5000

# Cite
If the paper and code are helpful to you, please cite our paper. Also welcome to contact the first author via email for discussion or cooperation.
  @article{Bai2023GoRecAG,
    title={GoRec: A Generative Cold-start Recommendation Framework},
    author={Haoyue Bai and Min Hou and Le Wu and Yonghui Yang and Kun Zhang and Richang Hong and Meng Wang},
    journal={Proceedings of the 31st ACM International Conference on Multimedia},
    year={2023},
    url={https://api.semanticscholar.org/CorpusID:264492017}
  }
