# Incremental Learning in the Mobile Scenario

This is the main repository for the master's thesis [*Incremental Learning in the Mobile Scenario*](https://drive.google.com/file/d/1lnJQM0NVPYmgoMCZB_rPu4YcMiRR4PRB/view?usp=sharing).



## Experiment Results

**Average incremental accuracy**

| New Classes / Step    | 1         | 2         | 5         | 10        |
| --------------------- | --------- | --------- | --------- | --------- |
| **iCaRL-ResNet**\*    | 44.20     | 50.60     | 53.79     | 58.08     |
| **iCaRL-MobileNetV2** | 23.45     | 39.40     | 43.90     | 50.15     |
| **BiC-ResNet**\*      | 47.09     | 48.96     | 53.21     | 56.86     |
| **BiC-MobileNetV2**   | 36.54     | 36.57     | 41.43     | 45.25     |
| **PODNet-ResNet**\*   | 57.98     | 60.72     | 63.19     | 64.83     |
| **PODNet-MobilNetV2** | **60.44** | **63.30** | **66.36** | **68.70** |

\* Results of ResNet-backbone models are reported directly from [[Douillard et al. 2020]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650086.pdf).

## Requirements and Pipeline

If  `conda` is not installed yet, please install `Miniconda` or `Anaconda` first. 

This pipeline was tested and run on `Ubuntu` and `Windows WSL`. 

When you install the requirements on other OS, if you encounter `ResolvePackageNotFound` error, please first comment these packages in `envrionment.yml` and try again.

1. Change the working directory to the root of the project directory.

   ```shell
   cd <your_project_dirctory>
   ```

2. Setup the environment

   ```shell
   conda env create --name il_mobile --file environment.yml
   ```

3. Activate the environment

   ```shell
   conda activate il_mobile
   ```

4. Experiments on CIFAR-100

   * Run MobileNetV2 baseline

     ```shell
     python demo_baseline.py
     ```

   * Run PODNet using ResNet as backbone

     ```shell
     python3 -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
         --initial-increment 50 --increment 1 --fixed-memory \
         --device <GPU_ID> --label podnet_cnn_cifar100_50steps_resnet \
         --data-path <PATH/TO/DATA>
     ```

   * Run PODNet using MobileNetV2 as backbone

     ```shell
     python3 -minclearn --options options/podnet/podnet_cnn_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml \
         --initial-increment 50 --increment 1 --fixed-memory \
         --device <GPU_ID> --label podnet_cnn_cifar100_50steps_mobilenetv2 \
         --data-path <PATH/TO/DATA>
     ```

   * Run iCaRL or BiC, just change to corresponding `--options` and give a new `--label`

5. Deactivate the environment

   ```shell
   conda deactivate
   ```

6. [Optional] Delete the environment if you do not need it any more

   ```shell
   conda env list
   conda env remove --name il_mobile
   ```




## Code

Some of the code are from their original repositories and adapted to our protocol. To get full look of the original repositories please check [incremental_learning.pytorch](https://github.com/arthurdouillard/incremental_learning.pytorch) and  [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
