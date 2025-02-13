# DETRs with Collaborative Hybrid Assignments Training

```shell
python compute_mean_std.py /path/to/data1 /path/to/data2 --batch_size 100
```

```shell
!bash tools/dist_finetune.sh projects/configs/custom/fisheye8k.py 2 output --cfg-options $(cat projects/configs/tuning/fisheye8k.txt)
```