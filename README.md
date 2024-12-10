## 被Eval数据的存储方式与读取方式
由于数据基本上在s3以对象存储，因此此框架目前设计为所有欲评估的数据存在一个your_dataset.txt文件中，方便数据的读取、划分与并行。此方法对于.json文件同理，若有如下数据存在层次：

- all_dataset_folder/
  - your_dataset_1/
    - dataset_1_split_1.json
    - dataset_1_split_2.json
    - dataset_1_split_3.json
  - your_dataset_2/
    - dataset_2_split_1.json
    - dataset_2_split_2.json
    - dataset_2_split_3.json

则需通过`folder2txt.py`将文件结构分别存成以数据集为单位的.txt文件。以上述的数据层次为例，应存成以下两个.txt文件，以生成对应的数据集日志与打分结果：

`your_dataset_1.txt` 文件包含以下内容：

`all_dataset_folder/your_dataset_1/dataset_1_split_1.json`
`all_dataset_folder/your_dataset_1/dataset_1_split_2.json`
`all_dataset_folder/your_dataset_1/dataset_1_split_3.json`

`your_dataset_2.txt` 文件包含以下内容：

`all_dataset_folder/your_dataset_2/dataset_2_split_1.json`
`all_dataset_folder/your_dataset_2/dataset_2_split_2.json`
`all_dataset_folder/your_dataset_2/dataset_2_split_3.json`

## Ray的运行方法：
1. `sbatch slurm_start_head.sh`，GPU最好设置为8块，这样可以独占机器，不会有端口冲突
2. `export head_node_ip=xx.xxx.xx.xx` ，具体可在起好的head的log内查看ip
3. `sbatch slurm_start_worker.sh`，GPU最好设置为8块，这样可以独占机器，不会有端口冲突，--quotatype=spot可以自动占用空闲的机器，但有可能会被抢占
4. `sbatch slurm_submit_job.sh python script.py --dataset your_dataset --scorer your_scorer`，提交需要执行的任务

## text_eval_example.py中需修改的路径
1. output_dir： 打分结果存储路径
2. log_dir： 日志存储路径
3. file_path： 数据集.txt文件存储路径
4. cfg_path: config.yaml的路径
