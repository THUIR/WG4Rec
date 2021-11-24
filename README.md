# WG4Rec

This is our implementation for the paper:

*Shaoyun Shi, Weizhi Ma, Zhen Wang, Min Zhang, Kun Fang, Jingfang Xu, Yiqun Liu, and Shaoping Ma. 2021. [WG4Rec: Modeling Textual Content with Word Graph for News Recommendation](https://dl.acm.org/doi/10.1145/3459637.3482401).  In Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM ’21).*



## Refernece

For inquiries contact Shaoyun Shi (shisy13 AT gmail.com).

```
@inproceedings{DBLP:conf/cikm/ShiMWZFXLM21,
  author    = {Shaoyun Shi and
               Weizhi Ma and
               Zhen Wang and
               Min Zhang and
               Kun Fang and
               Jingfang Xu and
               Yiqun Liu and
               Shaoping Ma},
  editor    = {Gianluca Demartini and
               Guido Zuccon and
               J. Shane Culpepper and
               Zi Huang and
               Hanghang Tong},
  title     = {WG4Rec: Modeling Textual Content with Word Graph for News Recommendation},
  booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
               and Knowledge Management, Virtual Event, Queensland, Australia, November
               1 - 5, 2021},
  pages     = {1651--1660},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3459637.3482401},
  doi       = {10.1145/3459637.3482401},
  timestamp = {Mon, 08 Nov 2021 14:07:22 +0100},
  biburl    = {https://dblp.org/rec/conf/cikm/ShiMWZFXLM21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



## Environments

[requirements.txt](https://github.com/THUIR/WG4Rec/blob/master/requirements.txt) - The codes can be successfully run with following packages in an Anaconda environment:

```
pandas==1.3.4
tqdm==4.62.3
pytorch==1.10.0
numpy==1.21.2
scikit-learn==0.23.2
cudatoolkit==10.2.89
python==3.9.7
```

Other settings with `pytorch>=1.3.1` may also work.



## Datasets

The processed datasets can be downloaded from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/a8b40e16594540a38f5d/) 

You should place the datasets in the `./dataset/`. The tree structure of directories should look like:

```
.
├── dataset
│   ├── sogou
│   └── adressa-1w
└── src
    ├── data_readers
    ├── data_processors
    ├── modules
    ├── models
    ├── runners
    └── utils
```



## Example to run the codes

-   Some running commands can be found in [`./command/command.py`](https://github.com/THUIR/WG4Rec/blob/master/command/command.py)
-   For example:

```
# WG4Rec on Sogou dataset
> cd WG4Rec/src/
> python main.py --model_name WG4RecSogou --dataset sogou --metrics ndcg@5,ndcg@10,hit@5,hit@10 --test_sample_n 99 --lr 0.001 --l2 1e-4 --sent_max 10 --sample_pop 100 --regenerate 0 --gpu 0
```

