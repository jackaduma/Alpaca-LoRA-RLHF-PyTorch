# **Alpaca-RLHF-PyTorch**

---
## **Table of Contents**
- [**Alpaca-RLHF-PyTorch**](#alpaca-rlhf-pytorch)
  - [**Table of Contents**](#table-of-contents)
  - [**Environment Setup**](#environment-setup)
  - [**Run**](#run)
    - [**Supervised Finetune**](#supervised-finetune)
    - [**Merge PEFT adapter into Model**](#merge-peft-adapter-into-model)
    - [**Train Reward Model**](#train-reward-model)
  - [**Topics**](#topics)
  - [**Reference**](#reference)
---

## **Environment Setup**
```
穷人卡：2080Ti 12G
torch2.0.0
cuda11.8
```

---

- [x] SFT: Supervised Finetune
- [x] Merge Adapter into Model
- [ ] RLHF
  - [x] train reward model
  - [ ] tuning with RL

## **Run**
---
### **Supervised Finetune**

```
 check src/peft/utils/save_and_load.py , Only comment the line 52 to # #to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
```

```bash
python supervised_finetune.py --base_model 'decapoda-research/llama-7b-hf' --data_path 'yahma/alpaca-cleaned' --output_dir './lora-alpaca' --num_epochs 1
```


### **Merge PEFT adapter into Model**

```bash
pip uninstall peft -y
pip install peft==0.2.0  # 0.3.0.dev0 raise many errors
python merge_peft_adapter.py
```

### **Train Reward Model**

```
python train_reward_model.py --bf16 False --model_name 'decapoda-research/llama-7b-hf' --gradient_accumulation_steps 32 --per_device_train_batch_size 1 --train_subset 2000 --eval_subset 100 --local_rank 0
```

---

## **Topics**
1. 第一步SFT之前，切记有个注意事项，需要检查下 安装的peft代码， src/peft/utils/save_and_load.py , 如果 line 52 有这行代码  #to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}，需要将其注释掉，否则在finetune完之后，保存不了 adapter model 的参数。切记！
2. PEFT的版本，目前从git上安装的是 0.3.0.dev0 版本，在merge_peft_adapter的时候有问题，需要切换到peft==0.2.0 (0.3.0.dev0 没有 _get_submodules()这个函数)
3. train reward model的时候 会发生另一个问题：

## **Reference**
utils & templates 来自 [alpaca-lora](https://github.com/tloen/
alpaca-lora) 。

requirements 主要是按照 [alpaca-lora](https://github.com/tloen/
alpaca-lora) 来配环境。
* [https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
* [https://github.com/lvwerra/trl](https://github.com/lvwerra/trl)
* [https://github.com/jasonvanf/llama-trl](https://github.com/jasonvanf/llama-trl)